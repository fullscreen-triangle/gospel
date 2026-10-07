"""S2: what is genetically distinctive about the replicated super-rescuer Cod-0?

Inputs: 1001 Genomes imputed SNP matrix (v3.1; 0 = Col-0 reference, 1 = non-reference),
kinship (IBS, MAC>=5), Araport11 annotation (Ensembl Plants TAIR10 GFF3), and the
per-accession rescue table from S1.

Writes results/S2_cod0_genetics.json and results/S2_cod0_rare_variant_genes.csv.
"""
import gzip
import re
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from scipy import stats

from common import CACHE, RESULTS, load, save

GENO = CACHE / "geno" / "1001_SNP_MATRIX"
GFF = CACHE / "geno" / "TAIR10.gff3.gz"
FOCAL = 9836                      # Cod-0
ALLIES = {9813: "BI-4", 9529: "IP-Cap-1", 7071: "Chat-1", 9834: "Cho-0"}
RARE_SPECIES = 0.01               # Cod-0 allele carried by <= 1% of the 1,135 accessions
PROMOTER = 1000
CHUNK = 250_000

CURATED = {
    "WCS417 / ISR / iron-coumarin": {
        "AT1G56160": "MYB72", "AT5G36890": "BGLU42", "AT3G13610": "F6'H1", "AT4G31940": "CYP82C4",
        "AT3G12900": "S8H", "AT3G53480": "ABCG37/PDR9", "AT4G19690": "IRT1", "AT1G01580": "FRO2",
        "AT4G30190": "AHA2", "AT2G28160": "FIT", "AT3G56970": "bHLH38", "AT3G56980": "bHLH39",
        "AT3G47640": "PYE", "AT3G18290": "BTS", "AT3G08040": "FRD3", "AT5G03280": "EIN2",
        "AT1G64280": "NPR1", "AT1G32640": "MYC2", "AT2G46370": "JAR1", "AT3G23150": "ETR2",
    },
    "immunity / microbe perception": {
        "AT5G46330": "FLS2", "AT4G33430": "BAK1", "AT5G20480": "EFR", "AT3G21630": "CERK1",
        "AT2G33580": "LYK5", "AT1G73080": "PEPR1", "AT2G19190": "FRK1", "AT3G52430": "PAD4",
        "AT3G48090": "EDS1", "AT1G74710": "ICS1/SID2",
    },
    "drought / ABA": {
        "AT4G26080": "ABI1", "AT5G57050": "ABI2", "AT3G14440": "NCED3", "AT1G45249": "ABF2/AREB1",
        "AT5G05410": "DREB2A", "AT5G52310": "RD29A", "AT4G33950": "OST1/SnRK2.6", "AT2G26040": "PYL2",
        "AT4G17870": "PYR1", "AT5G25610": "RD22", "AT1G20440": "COR47", "AT3G53420": "PIP2;1",
        "AT2G40170": "EM6", "AT1G52690": "LEA",
    },
    "root development / auxin": {
        "AT3G62980": "TIR1", "AT5G20730": "ARF7", "AT1G19220": "ARF19", "AT2G42430": "LBD16",
        "AT4G14550": "IAA14/SLR", "AT1G73590": "PIN1", "AT2G38120": "AUX1", "AT1G70560": "TAA1",
        "AT5G20960": "AAO1",
    },
}


# ------------------------------------------------------------------ annotation
def load_annotation():
    genes, feats = {}, defaultdict(list)          # feats[chrom] -> (start, end, kind, gene)
    tx_gene = {}
    with gzip.open(GFF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[0] not in "12345":
                continue
            c, kind, s, e, strand, attr = p[0], p[2], int(p[3]), int(p[4]), p[6], p[8]
            a = dict(kv.split("=", 1) for kv in attr.split(";") if "=" in kv)
            if kind in ("gene", "ncRNA_gene"):
                gid = a.get("gene_id", a.get("ID", "").replace("gene:", ""))
                desc = re.sub(r" \[Source:.*", "", a.get("description", "")).replace("%2C", ",").replace("%3B", ";")
                genes[gid] = {"chrom": c, "start": s, "end": e, "strand": strand, "symbol": a.get("Name", ""),
                              "description": desc, "biotype": a.get("biotype", "")}
                ps, pe = (s - PROMOTER, s - 1) if strand == "+" else (e + 1, e + PROMOTER)
                feats[c].append((ps, pe, "promoter", gid))
                feats[c].append((s, e, "gene", gid))
            elif kind in ("mRNA", "transcript", "ncRNA", "lnc_RNA", "miRNA", "tRNA"):
                tx_gene[a.get("ID")] = a.get("Parent", "").replace("gene:", "")
            elif kind in ("CDS", "five_prime_UTR", "three_prime_UTR", "exon"):
                gid = tx_gene.get(a.get("Parent", ""), a.get("Parent", "").replace("transcript:", "").split(".")[0])
                feats[c].append((s, e, {"CDS": "CDS", "five_prime_UTR": "UTR5", "three_prime_UTR": "UTR3", "exon": "exon"}[kind], gid))
    index = {}
    for c, fl in feats.items():
        fl.sort()
        index[c] = (np.array([f[0] for f in fl]), np.array([f[1] for f in fl]), [f[2] for f in fl], [f[3] for f in fl])
    return genes, index


RANK = {"CDS": 0, "UTR5": 1, "UTR3": 1, "exon": 2, "promoter": 3, "gene": 4}


def annotate(chrom, pos, index):
    """Best (most functional) feature class per SNP -> (class, gene) or None."""
    st, en, kind, gene = index[chrom]
    maxlen = int((en - st).max()) + 1
    lo = np.searchsorted(st, pos - maxlen, "left")
    hi = np.searchsorted(st, pos, "right")
    out = []
    for p, a, b in zip(pos, lo, hi):
        best = None
        for j in range(a, b):
            if st[j] <= p <= en[j]:
                k = kind[j]
                k = "intron" if k == "gene" else k
                r = RANK.get(k, 4)
                if best is None or r < best[0]:
                    best = (r, k, gene[j])
        out.append(None if best is None else (best[1], best[2]))
    return out


# ------------------------------------------------------------------ heritability
def reml_h2(y, K):
    """Single-component REML h2 via eigendecomposition (intercept only)."""
    y = (y - y.mean()) / y.std()
    s, U = np.linalg.eigh(K)
    yt = U.T @ y
    one = U.T @ np.ones_like(y)
    best = None
    for h in np.linspace(0.001, 0.999, 999):
        v = h * s + (1 - h)
        v = np.maximum(v, 1e-9)
        w = 1 / v
        beta = np.sum(w * one * yt) / np.sum(w * one * one)
        r = yt - beta * one
        n = len(y)
        sig = np.sum(w * r * r) / (n - 1)
        ll = -0.5 * (np.sum(np.log(v)) + (n - 1) * np.log(sig) + np.log(np.sum(w * one * one)))
        if best is None or ll > best[1]:
            best = (h, ll)
    # likelihood-ratio test vs h2 = 0
    v0 = np.ones_like(s)
    beta0 = np.sum(one * yt) / np.sum(one * one)
    r0 = yt - beta0 * one
    ll0 = -0.5 * ((len(y) - 1) * np.log(np.sum(r0 * r0) / (len(y) - 1)) + np.log(np.sum(one * one)))
    lrt = max(2 * (best[1] - ll0), 0)
    return {"h2": float(best[0]), "lrt": float(lrt), "p": float(0.5 * stats.chi2.sf(lrt, 1))}


def main():
    D = load()
    names = D.accessions["name"].to_dict()
    f = h5py.File(GENO / "imputed_snps_binary.hdf5", "r")
    ids = [int(x) for x in f["accessions"][:]]
    panel = sorted(set(pd.read_csv(CACHE / "plants_all.csv").gid.dropna().astype(int)) & set(ids))
    pidx = np.array([ids.index(g) for g in panel])
    fi = ids.index(FOCAL)
    ally_idx = {ids.index(g): n for g, n in ALLIES.items()}
    pos_all = f["positions"][:]
    chr_regions = f["positions"].attrs["chr_regions"]
    snps = f["snps"]

    genes, index = load_annotation()
    rare = []                       # (chrom, pos, cod_allele, n_species, n_panel, allies_sharing)
    n_species = len(ids)
    cod_alt = 0
    shared_rows = []
    for ci, (a, b) in enumerate(chr_regions):
        chrom = str(ci + 1)
        for s0 in range(a, b, CHUNK):
            s1 = min(b, s0 + CHUNK)
            X = snps[s0:s1, :]
            ac = X.sum(1)
            cod = X[:, fi]
            cod_alt += int(cod.sum())
            n_with = np.where(cod == 1, ac, n_species - ac)                # carriers of Cod-0's allele (species)
            Xp = X[:, pidx]
            n_with_p = np.where(cod == 1, Xp.sum(1), len(pidx) - Xp.sum(1))
            keep = n_with <= RARE_SPECIES * n_species
            if keep.any():
                kk = np.where(keep)[0]
                share = np.zeros(len(kk), int)
                share_names = [[] for _ in kk]
                for ai, nm in ally_idx.items():
                    m = X[kk, ai] == cod[kk]
                    share += m
                    for t in np.where(m)[0]:
                        share_names[t].append(nm)
                pos = pos_all[s0:s1][kk]
                ann = annotate(chrom, pos, index)
                for t in range(len(kk)):
                    rare.append((chrom, int(pos[t]), int(cod[kk[t]]), int(n_with[kk[t]]), int(n_with_p[kk[t]]),
                                 int(share[t]), ",".join(share_names[t]), ann[t]))
            # rare-in-panel alleles shared by Cod-0 and >= 2 replicated high rescuers
            pfreq_ok = n_with_p <= 0.05 * len(pidx)
            if pfreq_ok.any():
                kk = np.where(pfreq_ok)[0]
                sh = sum((X[kk, ai] == cod[kk]).astype(int) for ai in ally_idx)
                good = kk[sh >= 2]
                if len(good):
                    ann = annotate(chrom, pos_all[s0:s1][good], index)
                    for t, g in enumerate(good):
                        if ann[t] is not None:
                            shared_rows.append((chrom, int(pos_all[s0 + g]), ann[t][0], ann[t][1], int(n_with_p[g]),
                                                ",".join(nm for ai, nm in ally_idx.items() if X[g, ai] == cod[g])))
        print("chromosome", chrom, "done; rare so far", len(rare))

    R = pd.DataFrame(rare, columns=["chrom", "pos", "cod_allele", "carriers_species", "carriers_panel",
                                    "allies_sharing", "allies", "ann"])
    R["class"] = R.ann.map(lambda x: x[0] if x else "intergenic")
    R["gene"] = R.ann.map(lambda x: x[1] if x else None)
    R = R.drop(columns="ann")
    private = R[R.carriers_species == 1]

    # gene-level summary
    G = R[R.gene.notna()].groupby("gene").agg(
        n_rare=("pos", "size"), n_cds=("class", lambda c: int((c == "CDS").sum())),
        n_utr=("class", lambda c: int(c.isin(["UTR5", "UTR3"]).sum())),
        n_promoter=("class", lambda c: int((c == "promoter").sum())),
        n_private=("carriers_species", lambda c: int((c == 1).sum())),
        max_allies=("allies_sharing", "max"),
    ).reset_index()
    G["symbol"] = G.gene.map(lambda g: genes.get(g, {}).get("symbol", ""))
    G["description"] = G.gene.map(lambda g: genes.get(g, {}).get("description", ""))
    G["length"] = G.gene.map(lambda g: genes[g]["end"] - genes[g]["start"] + 1 if g in genes else np.nan)
    G["biotype"] = G.gene.map(lambda g: genes.get(g, {}).get("biotype", ""))
    G["functional"] = G.n_cds + G.n_utr + G.n_promoter
    # background: how unusual is this gene's rare-functional count, per kb?
    G["functional_per_kb"] = G.functional / ((G.length + PROMOTER) / 1000)
    G = G.sort_values(["functional", "n_cds"], ascending=False)

    curated_hits = []
    for cat, gl in CURATED.items():
        for g, sym in gl.items():
            sub = R[R.gene == g]
            curated_hits.append({"category": cat, "gene": g, "symbol": sym, "n_rare": int(len(sub)),
                                 "classes": sub["class"].value_counts().to_dict(),
                                 "positions": sub[["pos", "class", "carriers_species"]].to_dict(orient="records")[:20]})

    cand = pd.read_csv(CACHE / "easyGWAS_output.csv")
    gwas_hits = []
    for g in sorted(cand.Gene.unique()):
        sub = R[R.gene == g]
        gwas_hits.append({"gene": g, "symbol": genes.get(g, {}).get("symbol", ""), "n_rare_cod0": int(len(sub)),
                          "classes": sub["class"].value_counts().to_dict()})

    # relatives of Cod-0 and their rescue
    kf = h5py.File(GENO / "kinship_ibs_mac5.hdf5", "r")
    kid = [int(x) for x in kf["accessions"][:]]
    K = kf["kinship"][:]
    S1 = pd.read_csv(RESULTS / "S1_rescue_posterior.csv")
    resc = S1[S1.organ == "Shoot"].set_index("gid").raw_mean_rescue
    rk = K[kid.index(FOCAL)]
    order = [kid[i] for i in np.argsort(-rk) if kid[i] != FOCAL]
    rel_panel = [g for g in order if g in resc.index][:10]
    relatives = [{"name": names.get(g), "gid": g, "kinship": float(rk[kid.index(g)]), "shoot_rescue_mean": float(resc[g])}
                 for g in rel_panel]
    kin_panel = np.array([[K[kid.index(a), kid.index(b)] for b in resc.index] for a in resc.index])
    col0_kin = float(rk[kid.index(6909)])
    pct_rank_rel = float(np.mean([resc[g] for g in rel_panel]))

    # heritability of the phenotypes
    ph = pd.read_csv(RESULTS / "phenotypes_full.csv").set_index("gid")
    common_ids = [g for g in ph.index if g in kid]
    Kc = np.array([[K[kid.index(a), kid.index(b)] for b in common_ids] for a in common_ids])
    h2 = {}
    for col in ["shoot_rescue", "shoot_canonical_rescue", "shoot_log_promotion_drought", "shoot_mock_drought_loss",
                "shoot_log_promotion_nonstress", "root_rescue", "root_mock_drought_loss", "shoot_log_MN"]:
        y = ph.loc[common_ids, col].to_numpy(float)
        ok = np.isfinite(y)
        h2[col] = {**reml_h2(y[ok], Kc[np.ix_(ok, ok)]), "n": int(ok.sum())}

    out = {
        "focal": {"gid": FOCAL, "name": names.get(FOCAL)},
        "n_snps": int(len(pos_all)), "n_species": n_species, "n_panel_genotyped": int(len(pidx)),
        "cod0_nonreference_alleles": cod_alt,
        "rare_threshold_species": RARE_SPECIES,
        "n_rare_cod0_alleles": int(len(R)), "n_private_cod0_alleles": int(len(private)),
        "rare_by_class": R["class"].value_counts().to_dict(),
        "private_by_class": private["class"].value_counts().to_dict(),
        "n_genes_with_rare_functional": int((G.functional > 0).sum()),
        "top_genes": G.head(40).to_dict(orient="records"),
        "curated": curated_hits,
        "gwas_candidates": gwas_hits,
        "shared_with_high_rescuers": {
            "n_sites": len(shared_rows),
            "genes": pd.DataFrame(shared_rows, columns=["chrom", "pos", "class", "gene", "carriers_panel", "allies"])
                       .assign(symbol=lambda d: d.gene.map(lambda g: genes.get(g, {}).get("symbol", "")))
                       .groupby(["gene", "symbol"]).agg(n=("pos", "size"),
                                                         classes=("class", lambda c: ",".join(sorted(set(c)))),
                                                         allies=("allies", lambda a: ",".join(sorted(set(",".join(a).split(",")))))
                                                         ).reset_index().sort_values("n", ascending=False).head(40)
                       .to_dict(orient="records") if shared_rows else []},
        "relatives": relatives, "relatives_mean_shoot_rescue": pct_rank_rel,
        "panel_mean_shoot_rescue": float(resc.mean()), "col0_kinship_to_cod0": col0_kin,
        "heritability_reml": h2,
    }
    save("S2_cod0_genetics", out)
    G.to_csv(RESULTS / "S2_cod0_rare_variant_genes.csv", index=False)
    R.to_csv(CACHE / "S2_cod0_rare_sites.csv", index=False)
    return out


if __name__ == "__main__":
    o = main()
    print({k: v for k, v in o.items() if k in ("n_rare_cod0_alleles", "n_private_cod0_alleles", "rare_by_class",
                                                 "private_by_class", "n_genes_with_rare_functional",
                                                 "relatives_mean_shoot_rescue", "panel_mean_shoot_rescue")})
    print("h2:", {k: (round(v["h2"], 2), f"{v['p']:.1e}") for k, v in o["heritability_reml"].items()})
    print("relatives:", [(r["name"], round(r["kinship"], 3), round(r["shoot_rescue_mean"], 2)) for r in o["relatives"]])
    print("top genes:")
    for g in o["top_genes"][:25]:
        print(f"  {g['gene']} {g['symbol']:12s} cds={g['n_cds']} utr={g['n_utr']} prom={g['n_promoter']} priv={g['n_private']} "
              f"allies={g['max_allies']} | {g['description'][:70]}")
    print("curated with rare variants:")
    for c in o["curated"]:
        if c["n_rare"]:
            print(f"  [{c['category']}] {c['symbol']} {c['gene']}: {c['classes']}")
    print("gwas candidates with Cod-0 rare variants:", [(g["gene"], g["symbol"], g["classes"]) for g in o["gwas_candidates"] if g["n_rare_cod0"]])
    print("shared with high rescuers:", o["shared_with_high_rescuers"]["n_sites"])
    for g in o["shared_with_high_rescuers"]["genes"][:15]:
        print("  ", g)
