"""S4: kinship-corrected GWAS on the full 1001 Genomes imputed SNP set.

EMMAX-style mixed model. The variance ratio h2 is estimated once per phenotype under the null
(REML, IBS kinship), the phenotype and every SNP are rotated into the whitened space, and each SNP
is tested by OLS with an intercept. Phenotypes are rank-inverse-normal transformed so ratio
outliers cannot drive peaks. SNPs with panel MAF < 5% are skipped.

For each peak (best SNP per 20 kb), the script also reports:
  - the rescue-raising allele,
  - whether Cod-0 carries it,
  - the overlapping and nearby genes.
It then re-tests the easyGWAS candidate genes, using the best SNP within 1 kb of each gene.

Writes results/S4_gwas.json, results/S4_gwas_peaks.csv, _cache/S4_gwas_p1e-4.csv.
"""
import h5py
import numpy as np
import pandas as pd
from scipy import stats

from common import CACHE, RESULTS, save
from cod0_genetics import GENO, FOCAL, ALLIES, load_annotation, annotate, reml_h2

PHENOS = {
    "shoot_rescue": "shoot rescue (W-M_D)/(M_N-M_D)",
    "root_rescue": "root rescue",
    "shoot_interaction": "shoot WCS417 x drought interaction (log)",
    "root_interaction": "root WCS417 x drought interaction (log)",
    "shoot_log_promotion_drought": "shoot log(W/M_D)",
    "shoot_log_promotion_nonstress": "shoot WCS417 effect without drought (log)",
    "shoot_mock_drought_loss": "shoot drought loss, mock (control trait)",
}
MAF = 0.05
CHUNK = 250_000
WINDOW = 20_000
KEEP_P = 1e-4
FLANK = 10_000
EASYGWAS = {  # gene -> trait family it was reported for (easyGWAS output sheet)
    "AT1G51430": "root_rescue", "AT1G62770": "root_rescue", "AT2G06025": "root_rescue", "AT2G06095": "root_rescue",
    "AT3G15115": "root_rescue", "AT5G38950": "root_rescue", "AT5G60600": "root_rescue",
    "AT5G48640": "shoot_rescue", "AT5G48650": "shoot_rescue",
    "AT3G48630": "shoot_log_promotion_drought", "AT1G43720": "shoot_log_promotion_drought",
    "AT1G58007": "shoot_log_promotion_drought", "AT1G77810": "shoot_log_promotion_drought",
    "AT1G77815": "shoot_log_promotion_drought", "AT2G18550": "shoot_log_promotion_drought",
    "AT2G18570": "shoot_log_promotion_drought", "AT2G36490": "shoot_log_promotion_drought",
    "AT3G26020": "shoot_log_promotion_drought", "AT3G26050": "shoot_log_promotion_drought",
    "AT3G30247": "shoot_log_promotion_drought", "AT5G40740": "shoot_log_promotion_drought",
    "AT5G66070": "shoot_log_promotion_drought",
    "AT1G77660": "shoot_mock_drought_loss", "AT2G07213": "shoot_mock_drought_loss", "AT5G23960": "shoot_mock_drought_loss",
}


def inverse_normal(y):
    r = stats.rankdata(y)
    return stats.norm.ppf((r - 0.5) / len(y))


def scan(traits=tuple(PHENOS), exclude=()):
    """Genome-wide EMMAX scan. Returns hits (p < KEEP_P, with HDF5 row), tests, h2, lambda, analysed gids."""
    ph = pd.read_csv(RESULTS / "phenotypes_full.csv").set_index("gid")
    ph = ph[list(traits)].dropna()
    ph = ph[~ph.index.isin(exclude)]
    f = h5py.File(GENO / "imputed_snps_binary.hdf5", "r")
    ids = [int(x) for x in f["accessions"][:]]
    gids = [g for g in ph.index if g in ids]
    ph = ph.loc[gids]
    col = np.array([ids.index(g) for g in gids])
    order = np.argsort(col)                       # h5py fancy indexing needs increasing columns
    col_sorted = col[order]
    inv = np.argsort(order)
    n = len(gids)

    kf = h5py.File(GENO / "kinship_ibs_mac5.hdf5", "r")
    kid = {int(x): i for i, x in enumerate(kf["accessions"][:])}
    K0 = kf["kinship"][:]
    ki = np.array([kid[g] for g in gids])
    K = K0[np.ix_(ki, ki)]
    s, U = np.linalg.eigh(K)

    names = list(traits)
    Y = np.column_stack([inverse_normal(ph[c].to_numpy(float)) for c in names])
    h2 = {c: reml_h2(Y[:, j], K) for j, c in enumerate(names)}
    W = np.column_stack([1 / np.sqrt(h2[c]["h2"] * s + (1 - h2[c]["h2"])) for c in names])   # n x P
    Ut = U.T
    one_t = (Ut @ np.ones(n))[:, None] * W                        # n x P
    y_t = (Ut @ Y) * W
    q = one_t / np.linalg.norm(one_t, axis=0)
    y_r = y_t - q * (q * y_t).sum(0)
    yy = (y_r * y_r).sum(0)

    pos_all = f["positions"][:]
    chr_regions = f["positions"].attrs["chr_regions"]
    snps = f["snps"]
    keep, n_tests, pvals_hist = [], 0, {c: [] for c in names}
    for ci, (a, b) in enumerate(chr_regions):
        chrom = str(ci + 1)
        for s0 in range(a, b, CHUNK):
            s1 = min(b, s0 + CHUNK)
            G = snps[s0:s1, :][:, col_sorted][:, inv].astype(np.float32).T      # n x m
            af = G.mean(0)
            ok = (af >= MAF) & (af <= 1 - MAF)
            if not ok.any():
                continue
            G = G[:, ok]
            pos = pos_all[s0:s1][ok]
            Gr = Ut @ G
            n_tests += G.shape[1]
            for j, c in enumerate(names):
                xt = Gr * W[:, [j]]
                xr = xt - q[:, [j]] * (q[:, [j]] * xt).sum(0)
                xx = (xr * xr).sum(0)
                beta = (xr * y_r[:, [j]]).sum(0) / xx
                rss = yy[j] - beta * beta * xx
                t = beta / np.sqrt(rss / (n - 2) / xx)
                p = 2 * stats.t.sf(np.abs(t), n - 2)
                pvals_hist[c].append(np.quantile(p, np.linspace(0, 1, 201)) if len(p) > 200 else p)
                hit = p < KEEP_P
                for k in np.flatnonzero(hit):
                    keep.append((c, chrom, int(pos[k]), int(s0 + np.flatnonzero(ok)[k]), float(p[k]), float(beta[k]),
                                 float(af[ok][k])))
        print("chr", chrom, "done", n_tests, flush=True)

    H = pd.DataFrame(keep, columns=["trait", "chrom", "pos", "row", "p", "beta", "alt_freq"])
    lam = {}                        # lambda_GC from pooled per-chunk quantile summaries (approximate)
    for c in names:
        allq = np.concatenate(pvals_hist[c])
        lam[c] = float(np.median(stats.chi2.isf(allq, 1)) / stats.chi2.ppf(0.5, 1))
    return H, n_tests, h2, lam, gids


def genotypes_at(rows, gids):
    """0/1 genotypes (rows x accessions) for given HDF5 rows."""
    f = h5py.File(GENO / "imputed_snps_binary.hdf5", "r")
    ids = [int(x) for x in f["accessions"][:]]
    col = np.array([ids.index(g) for g in gids])
    u, back = np.unique(np.asarray(rows), return_inverse=True)     # h5py needs unique increasing rows
    return f["snps"][u, :][:, col][back]


def clump(H, trait, n_peaks):
    taken, out = [], []
    for r in H[H.trait == trait].sort_values("p").itertuples():
        if any(r.chrom == tc and abs(r.pos - tp) < WINDOW for tc, tp in taken):
            continue
        taken.append((r.chrom, r.pos))
        out.append(r)
        if len(out) >= n_peaks:
            break
    return pd.DataFrame(out)


def main():
    H, n_tests, h2, lam, gids = scan()
    names = list(PHENOS)
    H.to_csv(CACHE / "S4_gwas_p1e-4.csv", index=False)
    bonf = 0.05 / n_tests
    H["cod0_allele"] = genotypes_at(H.row, [FOCAL])[:, 0] if len(H) else []

    genes, index = load_annotation()
    gdf = pd.DataFrame([{"gene": g, **v} for g, v in genes.items()])

    def nearby(chrom, pos, flank=FLANK):
        sub = gdf[(gdf.chrom == chrom) & (gdf.end >= pos - flank) & (gdf.start <= pos + flank)]
        return ";".join(f"{r.gene}{'(' + r.symbol + ')' if r.symbol and r.symbol != r.gene else ''}" for r in sub.itertuples())

    # clumped peaks
    peaks = []
    for c in names:
        for r in clump(H, c, 15).itertuples():
            ann = annotate(r.chrom, np.array([r.pos]), index)[0]
            raise_allele = 1 if r.beta > 0 else 0
            peaks.append({"trait": c, "chrom": r.chrom, "pos": r.pos, "p": r.p, "minus_log10_p": -np.log10(r.p),
                          "bonferroni": bool(r.p < bonf), "alt_freq": r.alt_freq, "beta_sd": r.beta,
                          "raising_allele": "non-Col-0" if raise_allele else "Col-0",
                          "cod0_carries_raising": bool(r.cod0_allele == raise_allele),
                          "feature": None if ann is None else ann[0], "gene": None if ann is None else ann[1],
                          "symbol": None if ann is None else genes.get(ann[1], {}).get("symbol", ""),
                          "genes_within_10kb": nearby(r.chrom, r.pos)})
    P = pd.DataFrame(peaks)
    P.to_csv(RESULTS / "S4_gwas_peaks.csv", index=False)

    # Cod-0 at the top peaks: does it carry the raising allele more often than expected by allele frequency?
    cod_test = {}
    for c in names:
        sub = P[P.trait == c].head(10)
        if len(sub):
            exp = np.where(sub.raising_allele == "non-Col-0", sub.alt_freq, 1 - sub.alt_freq)
            cod_test[c] = {"peaks": int(len(sub)), "cod0_carries_raising": int(sub.cod0_carries_raising.sum()),
                           "expected": float(exp.sum())}

    # re-test easyGWAS candidates: best SNP within gene +/- 1 kb, under the trait family it came from
    retest = []
    for g, c in EASYGWAS.items():
        info = genes.get(g)
        if info is None:
            retest.append({"gene": g, "trait": c, "found": False})
            continue
        sub = H[(H.trait == c) & (H.chrom == info["chrom"]) & (H.pos >= info["start"] - 1000) & (H.pos <= info["end"] + 1000)]
        best = sub.sort_values("p").head(1)
        retest.append({"gene": g, "symbol": info["symbol"], "trait": c, "found": True,
                       "best_p": float(best.p.iloc[0]) if len(best) else None,
                       "minus_log10_p": float(-np.log10(best.p.iloc[0])) if len(best) else None,
                       "note": "no SNP with p < 1e-4 within 1 kb" if best.empty else ""})

    out = {"n_snps_tested": int(n_tests), "maf": MAF, "bonferroni_p": bonf,
           "minus_log10_bonferroni": float(-np.log10(bonf)), "h2_null": h2, "lambda_gc": lam,
           "n_accessions": len(gids), "traits": PHENOS, "n_hits_p1e-4": H.trait.value_counts().to_dict(),
           "n_bonferroni": {c: int(((H.trait == c) & (H.p < bonf)).sum()) for c in names},
           "peaks": P.to_dict(orient="records"), "cod0_at_top_peaks": cod_test, "easygwas_retest": retest}
    save("S4_gwas", out)
    return out


if __name__ == "__main__":
    o = main()
    print("SNPs tested", o["n_snps_tested"], "Bonferroni -log10", round(o["minus_log10_bonferroni"], 2))
    for c in o["traits"]:
        print(f"\n== {c}: h2 {o['h2_null'][c]['h2']:.2f}  lambda {o['lambda_gc'][c]:.2f}  bonferroni hits {o['n_bonferroni'][c]}")
        for p in [p for p in o["peaks"] if p["trait"] == c][:8]:
            print(f"  {p['chrom']}:{p['pos']:>9d}  -log10p {p['minus_log10_p']:.2f}  af {p['alt_freq']:.2f}  raise={p['raising_allele']:9s}"
                  f" cod0={'Y' if p['cod0_carries_raising'] else 'n'}  {p['feature']} {p['gene']} {p['symbol']} | {p['genes_within_10kb'][:90]}")
        print("  Cod-0 at top peaks:", o["cod0_at_top_peaks"].get(c))
    print("\neasyGWAS re-test:")
    for r in o["easygwas_retest"]:
        print(" ", r)
