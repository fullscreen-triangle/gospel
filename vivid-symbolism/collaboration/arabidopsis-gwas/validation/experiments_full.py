"""Second validation series, on NS+DS.xlsx (per-plant data for all four cells).

Run:  python experiments_full.py        (writes results/F*.json and results/phenotypes_full.csv)

F1  audit         new workbook vs the first one; definition of the non-stress gain
F2  noise         measured plant and batch variation in every cell; is batch shared
                  between mock and inoculated plants of a block?
F3  retest        the 50 accessions regrown in BR2/BR3: does weak rescue replicate?
F4  replicates    BR2 vs BR3 reliability of each construction, against the
                  disagreement between constructions within one replicate
F5  lambda        canonical exponent re-estimated with all four cells measured
F6  drought       drought rescue against non-stress growth promotion; the
                  inoculation x water interaction as a phenotype
F7  certify       certification limits under measured noise
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, SEED, f_lambda, load, save
from load_full import extract_full

RNG = np.random.default_rng(SEED)
B = 1000
Z = 1.6448536269514722
ORG = {"Shoot": "shoot", "Root": "root"}
COL0 = 6909

D = load()
P = extract_full()
P = P[P.gid.notna()].copy()
P["gid"] = P["gid"].astype(int)
P["cell"] = P["media"] + "_" + P["treatment"]          # DS_Mock, DS_WCS417, NS_Mock, NS_WCS417
P.loc[P.media == "NS", "br"] = "NS"                     # non-stress was one run
P["block"] = P.groupby(["sheet", "cell"])["sheet_row"].transform(lambda s: s.diff().fillna(1).ne(1).cumsum())
E4 = json.loads((RESULTS / "E4_canonical.json").read_text(encoding="utf-8"))
E5 = json.loads((RESULTS / "E5_four_column.json").read_text(encoding="utf-8"))
E10 = json.loads((RESULTS / "E10_classes.json").read_text(encoding="utf-8"))
results_index = {}


def run(name, fn):
    t = time.time()
    out = fn()
    save(name, out)
    results_index[name] = round(time.time() - t, 2)
    print(f"{name:14s} {time.time() - t:6.1f}s")
    return out


def plants(gid, cell, br, organ):
    """Per-plant weights for one cell of one accession (drought cells by replicate)."""
    s = P[(P.gid == gid) & (P.cell == cell)]
    if cell.startswith("DS"):
        s = s[s.br == br]
    return s[organ].dropna().to_numpy()


def cell_table(organ, br="BR1"):
    """Per accession: plant arrays for W (DS_WCS417), MD (DS_Mock), MN (NS_Mock), WN (NS_WCS417)."""
    out = {}
    for gid in sorted(P.gid.unique()):
        if gid == COL0:
            continue
        c = {"W": plants(gid, "DS_WCS417", br, organ), "MD": plants(gid, "DS_Mock", br, organ),
             "MN": plants(gid, "NS_Mock", None, organ), "WN": plants(gid, "NS_WCS417", None, organ)}
        if all(len(v) >= 3 for v in c.values()):
            out[gid] = c
    return out


def boot_means(arr, rng, B=B):
    idx = rng.integers(0, len(arr), (B, len(arr)))
    return arr[idx].mean(axis=1)


def lognorm(x, cv, rng):
    s = np.sqrt(np.log1p(cv ** 2))
    return x * np.exp(rng.normal(-0.5 * s * s, s, np.shape(x)))


CONS = {
    "gain": lambda W, MD, MN: W - MD,
    "increase": lambda W, MD, MN: W / MD - 1,
    "rescue": lambda W, MD, MN: (W - MD) / (MN - MD),
}


def lam_old(o):
    return float(E4["organs"][o]["lambda_hat_corrected"])


# ===================================================================== F1 audit
def f1():
    old = D.raw.groupby("ID")["shoot"].mean()
    out = {"n_plants": int(len(P)), "accessions": int(P.gid.nunique()),
           "plants_by_cell_and_replicate": {f"{c}|{b}": int(n) for (c, b), n in P.groupby(["cell", "br"]).size().items()},
           "missing_weights": {o: int(P[o].isna().sum()) for o in ["shoot", "root", "total"]},
           "replicated_accessions": sorted(int(g) for g in P[P.br.isin(["BR2", "BR3"])].gid.unique() if g != COL0)}
    w1 = P[(P.cell == "DS_WCS417") & (P.br == "BR1")].groupby("gid")["shoot"].mean()
    j = pd.concat([old, w1], axis=1, keys=["old", "new"]).dropna()
    out["W_BR1_equals_first_workbook"] = {"n": int(len(j)), "max_abs_diff": float((j.old - j.new).abs().max())}
    for key, cell, rec in [("MD", "DS_Mock", "Shoot_MD"), ("MN", "NS_Mock", "Shoot_MN")]:
        s = P[(P.cell == cell) & ((P.br == "BR1") | (P.br == "NS"))].groupby("gid")["shoot"].mean()
        d = (D.cells[rec] - s).abs().dropna()
        out[f"recovered_{key}_vs_measured"] = {"n": int(len(d)), "n_match_1e-2": int((d < 1e-2).sum()),
                                               "mismatch": {str(k): float(v) for k, v in d[d >= 1e-2].items()}}
    mn = P[P.cell == "NS_Mock"].groupby("gid")["shoot"].mean()
    wn = P[P.cell == "NS_WCS417"].groupby("gid")["shoot"].mean()
    g = D.traits["AVG_Gain_in_SFW_Non_Stress"]
    for name, val in [("WN_minus_MN", wn - mn), ("W_minus_WN", D.cells["Shoot_W"] - wn)]:
        r = (val - g).abs().dropna()
        out[f"gain_non_stress_is_{name}"] = {"n": int(len(r)), "n_match_1e-6": int((r < 1e-6).sum())}
    sel = D.traits.loc[D.traits.index.isin(out["replicated_accessions"]), "Shoot_Rescuer"]
    out["replicated_accessions_rescuer_label"] = {str(k): int(v) for k, v in sel.value_counts().items()}
    resc = (D.cells.Shoot_W - D.cells.Shoot_MD) / (D.cells.Shoot_MN - D.cells.Shoot_MD)
    pct = resc.rank(pct=True)
    out["replicated_accessions_BR1_rescue_percentile_median"] = float(pct[pct.index.isin(out["replicated_accessions"])].median())
    return out


# ===================================================================== F2 noise
def block_batch(values_by_block):
    blocks = [b for b in values_by_block if len(b) >= 3]
    means = np.array([b.mean() for b in blocks])
    within = np.median([b.std(ddof=1) / b.mean() for b in blocks])
    n = np.median([len(b) for b in blocks])
    v = means.var(ddof=1) / means.mean() ** 2 - within ** 2 / n
    f, p = stats.f_oneway(*blocks)
    return {"n_blocks": len(blocks), "block_means": means, "batch_cv": float(np.sqrt(max(v, 0))),
            "plant_cv_within_block": float(within), "F": float(f), "p": float(p)}


def f2():
    out = {}
    for O, o in ORG.items():
        oo = {"plant_cv": {}, "col0_batch": {}}
        for cell in ["DS_Mock", "DS_WCS417", "NS_Mock", "NS_WCS417"]:
            sub = P[(P.cell == cell) & (P.gid != COL0) & ((P.br == "BR1") | (P.br == "NS"))]
            cv = sub.groupby("gid")[o].agg(lambda s: s.std(ddof=1) / s.mean() if s.count() >= 3 else np.nan).dropna()
            oo["plant_cv"][cell] = {"median": float(cv.median()), "q25": float(cv.quantile(.25)), "q75": float(cv.quantile(.75))}
            col = P[(P.gid == COL0) & (P.cell == cell)]
            for br, g in col.groupby("br"):
                oo["col0_batch"][f"{cell}|{br}"] = block_batch([b[o].dropna().to_numpy() for _, b in g.groupby("block")])
        # is the batch offset shared by mock and inoculated plants of the same block?
        shared = {}
        for media, brs in [("DS", ["BR1", "BR2", "BR3"]), ("NS", ["NS"])]:
            for br in brs:
                a = oo["col0_batch"].get(f"{media}_Mock|{br}")
                b = oo["col0_batch"].get(f"{media}_WCS417|{br}")
                if a and b and len(a["block_means"]) == len(b["block_means"]) and len(a["block_means"]) >= 3:
                    la, lb = np.log(a["block_means"]), np.log(b["block_means"])
                    r = stats.pearsonr(la, lb)
                    shared[f"{media}|{br}"] = {"n_blocks": len(la), "r_log_block_means": float(r[0]), "p": float(r[1])}
        oo["mock_inoculated_block_correlation"] = shared
        # replicate offsets for Col-0 under drought
        oo["col0_replicate_means"] = {f"{c}|{br}": float(g[o].mean()) for (c, br), g in
                                      P[(P.gid == COL0) & (P.media == "DS")].groupby(["cell", "br"])}
        mn0 = P[(P.gid == COL0) & (P.cell == "NS_Mock")][o].dropna().to_numpy()
        col_gain = {}
        for br in ["BR1", "BR2", "BR3"]:
            W0, M0 = plants(COL0, "DS_WCS417", br, o), plants(COL0, "DS_Mock", br, o)
            gb = boot_means(W0, RNG) - boot_means(M0, RNG)
            col_gain[br] = {"n_W": len(W0), "n_MD": len(M0), "gain": float(W0.mean() - M0.mean()),
                            "lo": float(np.percentile(gb, 2.5)), "hi": float(np.percentile(gb, 97.5)),
                            "rescue": float((W0.mean() - M0.mean()) / (mn0.mean() - M0.mean())),
                            "loss": float(1 - M0.mean() / mn0.mean())}
        oo["col0_gain_by_replicate"] = col_gain
        out[O] = oo
    return out


# ===================================================================== F3 retest
def f3():
    rep = F1["replicated_accessions"]
    names = D.accessions["name"].to_dict()
    out = {"n": len(rep), "organs": {}}
    for O, o in ORG.items():
        rows = []
        for gid in rep:
            r = {"gid": gid, "name": names.get(gid)}
            mn = plants(gid, "NS_Mock", None, o)
            pooled_W, pooled_MD = [], []
            for br in ["BR1", "BR2", "BR3"]:
                W, MD = plants(gid, "DS_WCS417", br, o), plants(gid, "DS_Mock", br, o)
                if len(W) < 3 or len(MD) < 3:
                    continue
                if br != "BR1":
                    pooled_W.append(W); pooled_MD.append(MD)
                gb = boot_means(W, RNG) - boot_means(MD, RNG)
                r[br] = {"gain": float(W.mean() - MD.mean()), "lo": float(np.percentile(gb, 2.5)),
                         "hi": float(np.percentile(gb, 97.5)),
                         "rescue": float((W.mean() - MD.mean()) / (mn.mean() - MD.mean())) if len(mn) >= 3 else None}
            if pooled_W:
                W, MD = np.concatenate(pooled_W), np.concatenate(pooled_MD)
                gb = boot_means(W, RNG) - boot_means(MD, RNG)
                r["retest_pooled"] = {"gain": float(W.mean() - MD.mean()), "lo": float(np.percentile(gb, 2.5)),
                                      "hi": float(np.percentile(gb, 97.5)), "n_replicates": len(pooled_W)}
            rows.append(r)
        g1 = np.array([r["BR1"]["gain"] for r in rows if "BR1" in r and "retest_pooled" in r])
        g2 = np.array([r["retest_pooled"]["gain"] for r in rows if "BR1" in r and "retest_pooled" in r])
        all_g = (D.cells[f"{O}_W"] - D.cells[f"{O}_MD"]).reindex(D.main).dropna()
        zero_br1 = [r["name"] for r in rows if "BR1" in r and r["BR1"]["lo"] <= 0]
        zero_retest = [r["name"] for r in rows if "retest_pooled" in r and r["retest_pooled"]["lo"] <= 0]
        neg_retest = [r["name"] for r in rows if "retest_pooled" in r and r["retest_pooled"]["gain"] < 0]
        out["organs"][O] = {
            "rows": rows,
            "mean_gain_BR1": float(g1.mean()), "mean_gain_retest": float(g2.mean()),
            "mean_gain_all_accessions_BR1": float(all_g.mean()),
            "regression_toward_population_mean": float((g2.mean() - g1.mean()) / (all_g.mean() - g1.mean())),
            "paired_wilcoxon_p": float(stats.wilcoxon(g2 - g1).pvalue),
            "frac_retest_higher": float(np.mean(g2 > g1)),
            "compatible_with_zero_BR1": zero_br1, "compatible_with_zero_retest": zero_retest,
            "negative_point_estimate_retest": neg_retest,
            "consistently_compatible_with_zero": sorted(set(zero_br1) & set(zero_retest)),
        }
    return out


# ===================================================================== F4 replicates
def replicate_values(o, gids, br, lam):
    vals = {}
    for gid in gids:
        W, MD = plants(gid, "DS_WCS417", br, o), plants(gid, "DS_Mock", br, o)
        MN = plants(gid, "NS_Mock", None, o)
        if min(len(W), len(MD), len(MN)) < 3:
            continue
        w, md, mn = W.mean(), MD.mean(), MN.mean()
        vals[gid] = {**{n: f(w, md, mn) for n, f in CONS.items()}, "canonical": f_lambda(w, md, mn, lam),
                     "logW": np.log(w), "logMD": np.log(md), "loss": 1 - md / mn}
    return pd.DataFrame(vals).T


def f4():
    rep = [g for g in F1["replicated_accessions"]]
    out = {}
    for O, o in ORG.items():
        lam = lam_old(O)
        R = {br: replicate_values(o, rep, br, lam) for br in ["BR1", "BR2", "BR3"]}
        both = R["BR2"].index.intersection(R["BR3"].index)
        names = ["gain", "increase", "rescue", "canonical"]
        test_retest = {n: float(stats.spearmanr(R["BR2"].loc[both, n], R["BR3"].loc[both, n])[0]) for n in names}
        components = {n: float(stats.spearmanr(R["BR2"].loc[both, n], R["BR3"].loc[both, n])[0]) for n in ["logW", "logMD", "loss"]}
        # Fisher-z 95% intervals for Spearman rho (n - 3 df, with the 1.06 Spearman correction)
        se_z = np.sqrt(1.06 / (len(both) - 3))
        ci = lambda r: [float(np.tanh(np.arctanh(r) - 1.96 * se_z)), float(np.tanh(np.arctanh(r) + 1.96 * se_z))]
        retest_ci = {n: ci(r) for n, r in {**test_retest, **components}.items()}
        mean_by_br = {br: {n: float(R[br].loc[both.intersection(R[br].index), n].astype(float).mean())
                           for n in ["gain", "rescue", "loss", "logW", "logMD"]} for br in R}
        icc = {}
        for n in names:
            x, y = R["BR2"].loc[both, n].astype(float), R["BR3"].loc[both, n].astype(float)
            m = (x + y) / 2
            msb = 2 * m.var(ddof=1)
            msw = ((x - y) ** 2).sum() / (2 * len(x))
            icc[n] = float((msb - msw) / (msb + msw))
        within = {}
        for br in ["BR2", "BR3"]:
            within[br] = {f"{a}~{b}": float(stats.spearmanr(R[br].loc[both, a], R[br].loc[both, b])[0])
                          for a, b in itertools.combinations(names, 2)}
        all3 = both.intersection(R["BR1"].index)
        br1_vs = {n: float(stats.spearmanr(R["BR1"].loc[all3, n], (R["BR2"].loc[all3, n] + R["BR3"].loc[all3, n]) / 2)[0])
                  for n in names}
        # empirical replicate-level (batch + accession x run) variation of log cell means
        batch = {}
        for key in ["logW", "logMD"]:
            x, y = R["BR2"].loc[both, key].astype(float), R["BR3"].loc[both, key].astype(float)
            d = (x - y)
            plant_var = []
            for gid in both:
                cell = "DS_WCS417" if key == "logW" else "DS_Mock"
                for br in ["BR2", "BR3"]:
                    v = plants(gid, cell, br, o)
                    plant_var.append((v.std(ddof=1) / v.mean()) ** 2 / len(v))
            excess = d.var(ddof=1) / 2 - np.mean(plant_var)
            batch[key] = {"mean_log_offset_BR2_minus_BR3": float(d.mean()),
                          "sd_log_difference": float(d.std(ddof=1)),
                          "replicate_cv_beyond_plants": float(np.sqrt(max(excess, 0))),
                          "mean_plant_sampling_var": float(np.mean(plant_var))}
        # is the replicate offset shared by W and M_D of an accession (cancels in the gain ratio)?
        dW = R["BR2"].loc[both, "logW"] - R["BR3"].loc[both, "logW"]
        dM = R["BR2"].loc[both, "logMD"] - R["BR3"].loc[both, "logMD"]
        rr = stats.pearsonr(dW.astype(float), dM.astype(float))
        full = (D.cells[f"{O}_W"] - D.cells[f"{O}_MD"]).reindex(D.main).dropna()
        out[O] = {"n_pairs": int(len(both)), "test_retest_spearman_BR2_BR3": test_retest, "icc_BR2_BR3": icc,
                  "component_test_retest_spearman": components, "test_retest_ci95": retest_ci,
                  "mean_by_replicate": mean_by_br,
                  "within_replicate_construction_spearman": within, "BR1_vs_retest_mean_spearman": br1_vs,
                  "replicate_variation": batch,
                  "W_MD_offset_correlation": {"r": float(rr[0]), "p": float(rr[1])},
                  "range_restriction_gain_sd_ratio": float(R["BR1"].loc[all3, "gain"].astype(float).std() / full.std()),
                  "n_with_all_three": int(len(all3))}
    return out


# ===================================================================== F5 lambda
def fit_b(W, MD, MN):
    g = W - MD
    ok = (g > 0) & (MN > MD) & (MD > 0)
    X = np.column_stack([np.ones(ok.sum()), np.log(MD[ok]), np.log(MN[ok] - MD[ok])])
    beta, *_ = np.linalg.lstsq(X, np.log(g[ok]), rcond=None)
    return beta, np.log(g[ok]) - X @ beta


def f5():
    out = {}
    for O, o in ORG.items():
        T = cell_table(o)
        gids = sorted(T)
        W = np.array([T[g]["W"].mean() for g in gids]); MD = np.array([T[g]["MD"].mean() for g in gids])
        MN = np.array([T[g]["MN"].mean() for g in gids])
        beta, resid = fit_b(W, MD, MN)
        cvb = F2[O]["col0_batch"]
        cb = {c: cvb.get(f"{c}|BR1", cvb.get(f"{c}|NS"))["batch_cv"] for c in ["DS_WCS417", "DS_Mock", "NS_Mock"]}
        bs = []
        for b in range(B):
            ii = RNG.integers(0, len(gids), len(gids))
            w = np.array([RNG.choice(T[gids[i]]["W"], len(T[gids[i]]["W"])).mean() for i in ii])
            md = np.array([RNG.choice(T[gids[i]]["MD"], len(T[gids[i]]["MD"])).mean() for i in ii])
            mn = np.array([RNG.choice(T[gids[i]]["MN"], len(T[gids[i]]["MN"])).mean() for i in ii])
            w, md, mn = lognorm(w, cb["DS_WCS417"], RNG), lognorm(md, cb["DS_Mock"], RNG), lognorm(mn, cb["NS_Mock"], RNG)
            bs.append(fit_b(w, md, mn)[0])
        bs = np.array(bs)
        # calibration with measured per-cell noise
        pcv = {c: F2[O]["plant_cv"][c]["median"] for c in ["DS_WCS417", "DS_Mock", "NS_Mock"]}
        sd_theta = float(resid.std(ddof=3))
        lams = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
        means = []
        for lam in lams:
            est = []
            for _ in range(200):
                gt = np.exp(RNG.normal(0, sd_theta, len(gids))) * MD ** lam * (MN - MD) ** (1 - lam)
                gt *= np.median(W - MD) / np.median(gt)
                ws = lognorm(lognorm(MD + gt, pcv["DS_WCS417"] / np.sqrt(7), RNG), cb["DS_WCS417"], RNG)
                mds = lognorm(lognorm(MD, pcv["DS_Mock"] / np.sqrt(7), RNG), cb["DS_Mock"], RNG)
                mns = lognorm(lognorm(MN, pcv["NS_Mock"] / np.sqrt(7), RNG), cb["NS_Mock"], RNG)
                est.append(fit_b(ws, mds, mns)[0][1])
            means.append(np.mean(est))
        inv = lambda x: float(np.interp(x, means, lams, left=np.nan, right=np.nan))
        ci = np.percentile(bs[:, 1], [2.5, 97.5])
        out[O] = {"n": len(gids), "b1": float(beta[1]), "b1_plus_b2": float(beta[1] + beta[2]),
                  "b1_ci95": ci, "lambda_hat_corrected": inv(beta[1]), "lambda_ci95_corrected": [inv(ci[0]), inv(ci[1])],
                  "calibration_means": dict(zip(map(str, lams), map(float, means))),
                  "previous_E4": {"lambda_hat": lam_old(O), "ci": E4["organs"][O]["lambda_ci95_corrected"]},
                  "batch_cv_used": cb, "plant_cv_used": pcv}
    return out


# ===================================================================== F6 drought vs non-stress
def f6():
    out = {}
    names = D.accessions["name"].to_dict()
    for O, o in ORG.items():
        T = cell_table(o)
        gids = sorted(T)
        rows = []
        for g in gids:
            c = T[g]
            m = {k: v.mean() for k, v in c.items()}
            se2 = {k: (v.std(ddof=1) / v.mean()) ** 2 / len(v) for k, v in c.items()}   # var of log mean (delta)
            pd_ = np.log(m["W"] / m["MD"]); pn = np.log(m["WN"] / m["MN"])
            rows.append({"gid": g, "name": names.get(g),
                         "log_promotion_drought": pd_, "se_pd": np.sqrt(se2["W"] + se2["MD"]),
                         "log_promotion_nonstress": pn, "se_pn": np.sqrt(se2["WN"] + se2["MN"]),
                         "interaction": pd_ - pn, "se_int": np.sqrt(sum(se2.values())),
                         "rescue": (m["W"] - m["MD"]) / (m["MN"] - m["MD"]),
                         "canonical": f_lambda(m["W"], m["MD"], m["MN"], lam_old(O)),
                         "loss": 1 - m["MD"] / m["MN"], "MN": m["MN"]})
        T_ = pd.DataFrame(rows)
        res = {"n": len(T_)}
        for col, se in [("log_promotion_drought", "se_pd"), ("log_promotion_nonstress", "se_pn"), ("interaction", "se_int")]:
            y, s = T_[col].to_numpy(), T_[se].to_numpy()
            w = 1 / s ** 2
            ybar = np.sum(w * y) / np.sum(w)
            Q = float(np.sum(w * (y - ybar) ** 2))
            tau2 = max(np.var(y, ddof=1) - np.mean(s ** 2), 0)
            res[col] = {"mean": float(y.mean()), "median": float(np.median(y)),
                        "frac_significantly_positive": float(np.mean(y - 1.96 * s > 0)),
                        "frac_significantly_negative": float(np.mean(y + 1.96 * s < 0)),
                        "Q": Q, "df": len(y) - 1, "reliability_plant_noise": float(tau2 / (tau2 + np.mean(s ** 2)))}
        res["spearman"] = {
            "promotion_nonstress~promotion_drought": float(stats.spearmanr(T_.log_promotion_nonstress, T_.log_promotion_drought)[0]),
            "promotion_nonstress~rescue": float(stats.spearmanr(T_.log_promotion_nonstress, T_.rescue)[0]),
            "promotion_nonstress~canonical": float(stats.spearmanr(T_.log_promotion_nonstress, T_.canonical)[0]),
            "interaction~rescue": float(stats.spearmanr(T_.interaction, T_.rescue)[0]),
            "interaction~loss": float(stats.spearmanr(T_.interaction, T_.loss)[0]),
            "interaction~MN": float(stats.spearmanr(T_.interaction, T_.MN)[0]),
            "promotion_nonstress~MN": float(stats.spearmanr(T_.log_promotion_nonstress, T_.MN)[0]),
        }
        res["accessions"] = T_.to_dict(orient="records")
        out[O] = res
    return out


# ===================================================================== F7 certification
def f7():
    out = {}
    for O, o in ORG.items():
        T = cell_table(o)
        gids = sorted(T)
        lam = lam_old(O)
        cvb = F2[O]["col0_batch"]
        cb = {c: cvb.get(f"{c}|BR1", cvb.get(f"{c}|NS"))["batch_cv"] for c in ["DS_WCS417", "DS_Mock", "NS_Mock"]}
        rv = F4[O]["replicate_variation"]
        cr = {"W": rv["logW"]["replicate_cv_beyond_plants"], "MD": rv["logMD"]["replicate_cv_beyond_plants"]}
        v, s_plant, s_full, s_run = [], [], [], []
        for g in gids:
            w, md, mn = (boot_means(T[g][k], RNG) for k in ("W", "MD", "MN"))
            v.append(f_lambda(T[g]["W"].mean(), T[g]["MD"].mean(), T[g]["MN"].mean(), lam))
            s_plant.append(np.nanstd(f_lambda(w, md, mn, lam), ddof=1))
            wb, mdb, mnb = lognorm(w, cb["DS_WCS417"], RNG), lognorm(md, cb["DS_Mock"], RNG), lognorm(mn, cb["NS_Mock"], RNG)
            s_full.append(np.nanstd(f_lambda(wb, mdb, mnb, lam), ddof=1))
            wr, mdr = lognorm(w, cr["W"], RNG), lognorm(md, cr["MD"], RNG)
            s_run.append(np.nanstd(f_lambda(wr, mdr, mnb, lam), ddof=1))
        v, s_plant, s_full, s_run = map(np.array, (v, s_plant, s_full, s_run))
        I, J = np.triu_indices(len(v), 1)
        sd = v.std(ddof=1)
        res = {"n": len(v), "batch_cv_used": cb, "run_cv_used": cr}
        for lab, s in [("measured_plant_noise", s_plant), ("measured_plant_plus_batch", s_full),
                       ("measured_run_to_run", s_run)]:
            dmin = (np.abs(v[I] - v[J]) + Z * np.hypot(s[I], s[J])) / sd
            # resolution depth
            order = np.argsort(v); depth = np.ones(len(v), int)
            for pos, i in enumerate(order):
                lower = order[:pos]
                if len(lower):
                    ok = (v[i] - v[lower]) > Z * np.hypot(s[i], s[lower])
                    if ok.any():
                        depth[i] = 1 + depth[lower[ok]].max()
            tau2 = max(v.var(ddof=1) - np.mean(s ** 2), 0)
            res[lab] = {"median_se_over_sd": float(np.median(s) / sd),
                        "reliability": float(tau2 / (tau2 + np.mean(s ** 2))),
                        "min_certifiable_margin_sd_quantiles": np.percentile(dmin, [5, 25, 50, 75, 95]),
                        "frac_certifiable_below_1sd": float(np.mean(dmin < 1)),
                        "tiers": int(depth.max()), "tier_sizes": np.bincount(depth)[1:]}
        res["previous_modelled"] = {
            "M2_median_margin": E5["organs"][O]["models"]["M2_plant_all_cells"]["response"]["canonical"]["min_certifiable_margin_sd_quantiles"][3],
            "M3_median_margin": E5["organs"][O]["models"]["M3_plant_plus_batch"]["response"]["canonical"]["min_certifiable_margin_sd_quantiles"][3],
            "M2_tiers": E10[O]["M2_plant_all_cells"]["max_depth"]["canonical"],
            "M3_tiers": E10[O]["M3_plant_plus_batch"]["max_depth"]["canonical"]}
        out[O] = res
    return out


def export_phenotypes():
    rows = {}
    for O in ORG:
        for r in F6[O]["accessions"]:
            d = rows.setdefault(r["gid"], {"gid": r["gid"], "name": r["name"]})
            pre = O.lower()
            d[f"{pre}_canonical_rescue"] = r["canonical"]
            d[f"{pre}_rescue"] = r["rescue"]
            d[f"{pre}_log_promotion_drought"] = r["log_promotion_drought"]
            d[f"{pre}_log_promotion_nonstress"] = r["log_promotion_nonstress"]
            d[f"{pre}_interaction"] = r["interaction"]
            d[f"{pre}_mock_drought_loss"] = r["loss"]
            d[f"{pre}_log_MN"] = np.log(r["MN"])
    df = pd.DataFrame(rows.values()).sort_values("gid")
    df.to_csv(RESULTS / "phenotypes_full.csv", index=False)
    return {"file": "phenotypes_full.csv", "n": int(len(df)), "columns": list(df.columns)}


if __name__ == "__main__":
    t0 = time.time()
    F1 = run("F1_audit", f1)
    F2 = run("F2_noise", f2)
    F3 = run("F3_retest", f3)
    F4 = run("F4_replicates", f4)
    F5 = run("F5_lambda", f5)
    F6 = run("F6_drought", f6)
    F7 = run("F7_certify", f7)
    run("F_phenotypes", export_phenotypes)
    save("F_index", {"seed": SEED, "B": B, "experiments": results_index, "total_seconds": round(time.time() - t0, 1)})
    print(f"done in {time.time() - t0:.1f}s")
