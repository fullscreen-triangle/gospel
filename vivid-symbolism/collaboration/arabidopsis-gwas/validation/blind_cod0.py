"""S5: does Cod-0 stack rescue-raising alleles, judged blind?

The GWAS (S4) is re-run with Cod-0 held out, so Cod-0's phenotype cannot pick the alleles. Then, at
the top N clumped peaks of each trait, the rescue-raising allele is counted in every accession.
Every other accession is still in-sample, so its count is biased upward and the comparison is
conservative for Cod-0.

Second test: the allele score is computed from the run-1 GWAS. It is then correlated with rescue
measured in runs 2 and 3, which never entered the GWAS, across the rescreened accessions.

Writes results/S5_blind_polygenic.json.
"""
import ast

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, save
from cod0_genetics import FOCAL, ALLIES, load_annotation, annotate
from gwas_lmm import scan, clump, genotypes_at, FLANK

TRAITS = ("shoot_rescue", "root_rescue", "shoot_interaction", "shoot_log_promotion_nonstress")
NS = (10, 25, 50, 100)
SHOW = {FOCAL: "Cod-0", 6909: "Col-0", **ALLIES}


def poisson_binomial_upper(probs, k):
    """P(X >= k) for X = sum of independent Bernoulli(probs), exact by convolution."""
    dist = np.array([1.0])
    for p in probs:
        dist = np.convolve(dist, [1 - p, p])
    return float(dist[k:].sum())


def main():
    H, n_tests, h2, lam, gids = scan(TRAITS, exclude={FOCAL})
    panel = gids + [FOCAL]
    S1 = pd.read_csv(RESULTS / "S1_rescue_posterior.csv")
    later = {}
    for organ in ("Shoot", "Root"):
        sub = S1[(S1.organ == organ) & (S1.runs >= 2)]
        later[organ] = {int(r.gid): np.mean([v for k, v in ast.literal_eval(r.rescue_by_run).items() if k != "BR1"])
                        for r in sub.itertuples()}
    out = {"held_out": "Cod-0", "n_accessions_in_scan": len(gids), "n_snps_tested": int(n_tests), "lambda_gc": lam,
           "h2_null": h2, "traits": {}}
    for t in TRAITS:
        res = []
        for N in NS:
            P = clump(H, t, N)
            if P.empty or (res and res[-1]["n_peaks"] == len(P)):
                continue
            G = genotypes_at(P.row.to_numpy(), panel).astype(int)        # peaks x accessions
            raise_ = (P.beta.to_numpy() > 0).astype(int)
            carries = (G == raise_[:, None])
            count = carries.sum(0)
            freq_raise = np.where(raise_ == 1, P.alt_freq, 1 - P.alt_freq)
            weighted = (np.abs(P.beta.to_numpy())[:, None] * carries).sum(0)
            ci = panel.index(FOCAL)
            row = {"n_peaks": int(len(P)), "min_minus_log10_p": float(-np.log10(P.p.max())),
                   "cod0_count": int(count[ci]), "expected_count": float(freq_raise.sum()),
                   "p_cod0_by_frequency": poisson_binomial_upper(freq_raise, int(count[ci])),
                   "cod0_percentile_count": float(stats.percentileofscore(count, count[ci], kind="strict") / 100),
                   "cod0_percentile_weighted": float(stats.percentileofscore(weighted, weighted[ci], kind="strict") / 100),
                   "panel_count_mean": float(count.mean()), "panel_count_max": int(count.max()),
                   "named": {SHOW[g]: int(count[panel.index(g)]) for g in SHOW if g in panel}}
            organ = "Root" if t.startswith("root") else "Shoot"
            ids = [g for g in later[organ] if g in panel and g != FOCAL]
            if len(ids) >= 8:
                x = [weighted[panel.index(g)] for g in ids]
                y = [later[organ][g] for g in ids]
                rho, pr = stats.spearmanr(x, y)
                row["runs23_validation"] = {"n": len(ids), "spearman": float(rho), "p": float(pr),
                                            "note": "score from run-1 GWAS, rescue from runs 2-3; Cod-0 excluded"}
            res.append(row)
        out["traits"][t] = res
    out["held_out_rescreen"] = replicate(later)
    save("S5_blind_polygenic", out)
    return out


def replicate(later):
    """Hold out every rescreened accession, scan run 1 on the rest, then test the held-out set."""
    held = set(later["Shoot"]) | set(later["Root"]) | {FOCAL}
    H, n_tests, h2, lam, gids = scan(("shoot_rescue", "root_rescue"), exclude=held)
    ph = pd.read_csv(RESULTS / "phenotypes_full.csv").set_index("gid")
    genes, index = load_annotation()
    rep = {"n_scan": len(gids), "n_held_out": len(held), "lambda_gc": lam}
    for t, organ in (("shoot_rescue", "Shoot"), ("root_rescue", "Root")):
        ids = [g for g in later[organ] if g != FOCAL and g in ph.index and np.isfinite(ph.loc[g, t])]
        y23 = np.array([later[organ][g] for g in ids])
        y1 = ph.loc[ids, t].to_numpy(float)
        base_rho, base_p = stats.spearmanr(y1, y23)
        rows = []
        for N in NS:
            P = clump(H, t, N)
            if P.empty or (rows and rows[-1]["n_peaks"] == len(P)):
                continue
            G = genotypes_at(P.row.to_numpy(), ids).astype(float)
            raise_ = (P.beta.to_numpy() > 0).astype(int)
            score = (np.abs(P.beta.to_numpy())[:, None] * (G == raise_[:, None])).sum(0)
            r1, p1 = stats.spearmanr(score, y1)
            r23, p23 = stats.spearmanr(score, y23)
            rows.append({"n_peaks": int(len(P)), "rho_run1_heldout": float(r1), "p_run1": float(p1),
                         "rho_runs23": float(r23), "p_runs23": float(p23)})
        # per-peak replication in the held-out accessions (direction + one-sided p), top 50 peaks
        P = clump(H, t, 50)
        G = genotypes_at(P.row.to_numpy(), ids).astype(float)
        peaks = []
        for k, r in enumerate(P.itertuples()):
            g = G[k]
            if g.min() == g.max():
                continue
            d1 = y1[g == 1].mean() - y1[g == 0].mean()
            d23 = y23[g == 1].mean() - y23[g == 0].mean()
            same = np.sign(d23) == np.sign(r.beta)
            u = stats.mannwhitneyu(y23[g == 1], y23[g == 0], alternative="greater" if r.beta > 0 else "less")
            ann = annotate(r.chrom, np.array([r.pos]), index)[0]
            near = [f"{gg}({v['symbol']})" if v["symbol"] and v["symbol"] != gg else gg for gg, v in genes.items()
                    if v["chrom"] == r.chrom and v["end"] >= r.pos - FLANK and v["start"] <= r.pos + FLANK]
            peaks.append({"chrom": r.chrom, "pos": int(r.pos), "minus_log10_p_scan": float(-np.log10(r.p)),
                          "alt_freq": float(r.alt_freq), "raising": "non-Col-0" if r.beta > 0 else "Col-0",
                          "n_carriers_heldout": int(g.sum()), "diff_run1_heldout": float(d1), "diff_runs23": float(d23),
                          "same_direction_runs23": bool(same), "p_runs23_one_sided": float(u.pvalue),
                          "feature": None if ann is None else ann[0], "gene": None if ann is None else ann[1],
                          "genes_within_10kb": ";".join(near)})
        sign_n = sum(p["same_direction_runs23"] for p in peaks)
        rep[t] = {"n_heldout_with_runs23": len(ids), "baseline_rho_run1_vs_runs23": float(base_rho),
                  "baseline_p": float(base_p), "scores": rows,
                  "sign_concordance": {"same": int(sign_n), "of": len(peaks),
                                       "p_binomial": float(stats.binomtest(sign_n, len(peaks), 0.5, alternative="greater").pvalue)},
                  "peaks": sorted(peaks, key=lambda p: p["p_runs23_one_sided"])}
    return rep


if __name__ == "__main__":
    o = main()
    for t, rs in o["traits"].items():
        print(f"\n== {t}")
        for r in rs:
            v = r.get("runs23_validation", {})
            print(f"  top {r['n_peaks']:3d}: Cod-0 {r['cod0_count']:3d} vs expected {r['expected_count']:.1f}"
                  f" (p {r['p_cod0_by_frequency']:.2g}); percentile count {r['cod0_percentile_count']:.2f},"
                  f" weighted {r['cod0_percentile_weighted']:.2f}; panel max {r['panel_count_max']} | {r['named']}"
                  f" | runs2-3 rho {v.get('spearman', float('nan')):.2f} p {v.get('p', float('nan')):.2g} n {v.get('n')}")
    R = o["held_out_rescreen"]
    print("\nHELD-OUT RESCREEN: scan n", R["n_scan"], "held out", R["n_held_out"], "lambda", R["lambda_gc"])
    for t in ("shoot_rescue", "root_rescue"):
        r = R[t]
        print(f"\n== {t}: n {r['n_heldout_with_runs23']}; raw run1 vs runs2-3 rho {r['baseline_rho_run1_vs_runs23']:.2f} (p {r['baseline_p']:.2g});"
              f" sign concordance {r['sign_concordance']}")
        for s_ in r["scores"]:
            print(f"  top {s_['n_peaks']:3d}: score vs run1(held-out) rho {s_['rho_run1_heldout']:.2f} p {s_['p_run1']:.2g};"
                  f" vs runs2-3 rho {s_['rho_runs23']:.2f} p {s_['p_runs23']:.2g}")
        for p in r["peaks"][:10]:
            print(f"  {p['chrom']}:{p['pos']:>9d} scan {p['minus_log10_p_scan']:.1f} raise {p['raising']:9s} carriers {p['n_carriers_heldout']:2d}"
                  f" d23 {p['diff_runs23']:+.2f} p23 {p['p_runs23_one_sided']:.3f} {p['feature']} {p['gene']} | {p['genes_within_10kb'][:80]}")
