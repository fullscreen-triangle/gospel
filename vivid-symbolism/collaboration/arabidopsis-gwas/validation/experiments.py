"""Validation suite for the WCS417 drought-rescue analysis.

Run:  python experiments.py            (writes results/E*.json, ~1-2 min)

E1  audit          workbook identities, cell coverage, labelling
E2  heterogeneity  is the inoculation effect a single value across accessions?
                   could a loss of rescue have been detected?
E3  dependence     do admissible response constructions agree, beyond noise?
E4  canonical      nuisance invariance, the lambda family, data-driven choice
E5  four-column    baseline x response verdicts over all accession pairs
E6  calibration    operating characteristics of the three-valued verdict
E7  batch          Col-0 blocks as an internal batch control
E8  candidates     GWAS candidates by trait family and nuisance loading
E9  ratio          apparent rescue change of a baseline-only mutant
E10 classes        response-equivalence classes and their stability
E11 geography      descriptive association with collection site
"""
from __future__ import annotations

import itertools
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

from common import (CONSTRUCTIONS, SEED, f_lambda, gain, increase, load, rescue, save)

D = load()
RNG = np.random.default_rng(SEED)
B = 1000
N_REP = 7                       # plants per cell in the main set
Z = 1.6448536269514722          # one-sided 5%
MAIN = D.main
ORG = ["Shoot", "Root"]
RAWCOL = {"Shoot": "shoot", "Root": "root"}


def cells(o, ids=MAIN):
    c = D.cells.loc[ids]
    return c[f"{o}_W"].to_numpy(), c[f"{o}_MD"].to_numpy(), c[f"{o}_MN"].to_numpy()


def plant_matrix(o, ids=MAIN):
    """(A, 7) matrix of per-plant fresh weights, inoculated x drought."""
    g = D.raw.groupby("ID")[RAWCOL[o]].apply(lambda s: s.to_numpy())
    out = np.full((len(ids), N_REP), np.nan)
    for k, i in enumerate(ids):
        v = g.loc[i]
        out[k, :len(v)] = v[:N_REP]
    return out


# ---------------------------------------------------------------- noise model
def plant_cv(o):
    P = plant_matrix(o)
    cv = np.nanstd(P, axis=1, ddof=1) / np.nanmean(P, axis=1)
    return float(np.nanmedian(cv))


def col0_blocks(o):
    col = D.raw[D.raw.ID == 6909].copy()
    col["block"] = col["sheet_row"].diff().fillna(1).ne(1).cumsum()
    return col.groupby("block")[RAWCOL[o]].apply(lambda s: s.dropna().to_numpy())


def batch_cv(o):
    blocks = [b for b in col0_blocks(o) if len(b) >= 3]
    means = np.array([b.mean() for b in blocks])
    within = np.median([b.std(ddof=1) / b.mean() for b in blocks])
    n = np.median([len(b) for b in blocks])
    v = means.var(ddof=1) / means.mean() ** 2 - within ** 2 / n
    return float(np.sqrt(max(v, 0.0)))


CV = {o: plant_cv(o) for o in ORG}
CVB = {o: batch_cv(o) for o in ORG}
MODELS = {
    "M1_plant_inoculated_only": dict(mock=False, batch=False, mock_mult=1.0),
    "M2_plant_all_cells": dict(mock=True, batch=False, mock_mult=1.0),
    "M3_plant_plus_batch": dict(mock=True, batch=True, mock_mult=1.0),
    "M4_M3_mock_noise_x1.5": dict(mock=True, batch=True, mock_mult=1.5),
}
REF = "M3_plant_plus_batch"


def lognorm(x, cv, rng, size=None):
    s = np.sqrt(np.log1p(cv ** 2))
    return x * np.exp(rng.normal(-0.5 * s * s, s, size if size is not None else np.shape(x)))


def draw(o, model, rng, B=B, ids=MAIN):
    """Bootstrap/perturbation draws of the three cell means: arrays (B, A)."""
    m = MODELS[model]
    P = plant_matrix(o, ids)
    A = P.shape[0]
    idx = rng.integers(0, N_REP, (B, A, N_REP))
    Pb = np.take_along_axis(np.broadcast_to(P, (B, A, N_REP)), idx, axis=2)
    W = np.nanmean(Pb, axis=2)
    _, MD, MN = cells(o, ids)
    MD = np.broadcast_to(MD, (B, A)).copy()
    MN = np.broadcast_to(MN, (B, A)).copy()
    if m["mock"]:
        se = CV[o] / np.sqrt(N_REP) * m["mock_mult"]
        MD, MN = lognorm(MD, se, rng), lognorm(MN, se, rng)
    if m["batch"]:
        W, MD, MN = (lognorm(X, CVB[o], rng) for X in (W, MD, MN))
    return W, MD, MN


def topk(v, k):
    return set(np.argsort(-v)[:k])


def jac(a, b):
    return len(a & b) / len(a | b)


results_index = {}


def run(name, fn):
    t = time.time()
    out = fn()
    p = save(name, out)
    results_index[name] = {"file": p.name, "seconds": round(time.time() - t, 2)}
    print(f"{name:14s} {time.time() - t:6.1f}s -> {p.name}")
    return out


# ================================================================== E1 audit
def e1():
    tr = D.traits
    out = {"workbook": "GWAS Kundai.xlsx", "n_plants_raw": int(len(D.raw)),
           "raw_cells_present": sorted({f"{a} x {b}" for a, b in zip(D.raw.treatment, D.raw.media)}),
           "n_accessions": int(tr.shape[0]), "repeated_lines": D.repeated,
           "n_main": len(MAIN),
           "plants_per_accession": D.raw.groupby("ID").size().value_counts().to_dict(),
           "root_length_filled": int(D.raw.root_length.notna().sum()),
           "identities": {}}
    for o, a in [("Shoot", "SFW"), ("Root", "RFW"), ("Total", "TFW")]:
        c = D.cells
        W, MD, MN = c[f"{o}_W"], c[f"{o}_MD"], c[f"{o}_MN"]
        inc_col = f"AVG_percent_increase_in_{a}" + ("_" if o == "Shoot" else "")
        r_res = (rescue(W, MD, MN) - tr[f"AVG_percent_Rescue_{o}"]).abs()
        r_inc = (increase(W, MD, MN) - tr[inc_col]).abs()
        rawmean = D.raw.groupby("ID")[o.lower()].mean().reindex(tr.index)
        r_raw = (rawmean - W).abs()
        bad_inc = sorted(int(i) for i in r_inc.index[r_inc > 1e-6])
        bad_res = sorted(int(i) for i in r_res.index[r_res > 1e-6])
        out["identities"][o] = {
            "rescue_eq_(W-MD)/(MN-MD)_n_hold": int((r_res <= 1e-6).sum()),
            "rescue_eq_(W-MD)/(MN-MD)_median_abs_resid": float(r_res.median()),
            "rescue_mismatch_ids": bad_res,
            "rescue_mismatch_resid": [float(r_res[i]) for i in bad_res],
            "increase_eq_W/MD-1_n_hold": int((r_inc <= 1e-6).sum()),
            "increase_mismatch_ids": bad_inc,
            "increase_mismatch_are_repeated_lines": set(bad_inc) <= set(D.repeated),
            "raw_mean_eq_AVG_W_max_abs_resid": float(r_raw.max()),
            "median_MN": float(MN.median()), "median_MD": float(MD.median()),
            "median_W": float(W.median()),
            "median_fractional_loss": float((1 - MD / MN).median()),
        }
    t1 = D.table1.merge(D.cells, left_on="Genome ID", right_index=True)
    out["table1_is_drought"] = {
        "r_mock_vs_MD": float(np.corrcoef(t1["Average Mock SFW (mg)"], t1["Shoot_MD"])[0, 1]),
        "r_wcs_vs_W": float(np.corrcoef(t1["Average WCS417 SFW (mg)"], t1["Shoot_W"])[0, 1]),
        "n": int(len(t1))}
    t2 = D.table2.merge(tr, left_on="Genome ID", right_index=True)
    eq = (t2[" Average Shoot Drought Tolerance"] - t2["AVG_percent_Loss_in_SFW_under_drought"]).abs() < 1e-9
    out["table2_tolerance_is_loss"] = {"n": int(len(t2)), "identical": int(eq.sum())}
    lab = {}
    for c, o in [("Shoot_Rescuer", "Shoot"), ("Root_Rescuer", "Root")]:
        g = D.cells[f"{o}_W"] - D.cells[f"{o}_MD"]
        m = tr[c].notna()
        lab[c] = {"n1": int((tr[c] == 1).sum()), "n0": int((tr[c] == 0).sum()),
                  "gain_range_label1": [float(g[m & (tr[c] == 1)].min()), float(g[m & (tr[c] == 1)].max())],
                  "gain_range_label0": [float(g[m & (tr[c] == 0)].min()), float(g[m & (tr[c] == 0)].max())],
                  "n_negative_gain": int((g < 0).sum())}
    out["rescuer_labels"] = lab
    return out


# ========================================================== E2 heterogeneity
def e2():
    out = {"noise": {"plant_cv": CV, "batch_cv": CVB, "B": B}, "organs": {}}
    for o in ORG:
        W, MD, MN = cells(o)
        oo = {}
        for model in ["M2_plant_all_cells", REF, "M4_M3_mock_noise_x1.5"]:
            Wb, MDb, MNb = draw(o, model, RNG)
            mm = {}
            for name, f in CONSTRUCTIONS.items():
                y = f(W, MD, MN)
                se = np.nanstd(f(Wb, MDb, MNb), axis=0, ddof=1)
                w = 1 / se ** 2
                ybar = np.sum(w * y) / np.sum(w)
                Q = float(np.sum(w * (y - ybar) ** 2))
                df = len(y) - 1
                tau2 = max(np.var(y, ddof=1) - np.mean(se ** 2), 0.0)
                mm[name] = {"Q": Q, "df": df, "p": float(stats.chi2.sf(Q, df)),
                            "log10_p": float(stats.chi2.logsf(Q, df) / np.log(10)),
                            "I2": max(0.0, (Q - df) / Q),
                            "tau_over_mean_se": float(np.sqrt(tau2) / np.mean(se)),
                            "reliability": float(tau2 / (tau2 + np.mean(se ** 2)))}
                if name == "gain":
                    lo_ = np.nanpercentile(f(Wb, MDb, MNb), 2.5, axis=0)
                    zero_ok = np.where(lo_ <= 0)[0]
                    mm[name]["ci_includes_zero"] = [
                        {"gid": int(MAIN[i]), "name": D.accessions["name"].get(MAIN[i]),
                         "gain": float(y[i]), "ci_lo": float(lo_[i])} for i in zero_ok]
                if name == "gain" and model == REF:
                    z = y / se
                    lo = np.nanpercentile(f(Wb, MDb, MNb), 2.5, axis=0)
                    mm[name]["detectability"] = {
                        "z_min": float(z.min()), "z_median": float(np.median(z)),
                        "frac_ci_excludes_zero": float(np.mean(lo > 0)),
                        "median_se_mg": float(np.median(se)),
                        "min_detectable_gain_mg_80pct": float(2.8 * np.median(se)),
                        "min_observed_gain_mg": float(y.min()),
                        "per_accession": {"gain": y, "se": se, "lo": lo,
                                          "hi": np.nanpercentile(f(Wb, MDb, MNb), 97.5, axis=0)},
                    }
            oo[model] = mm
        out["organs"][o] = oo
    return out


# ============================================================ E3 dependence
def e3():
    out = {"k_fracs": [0.1, 0.2, 0.3], "organs": {}}
    for o in ORG:
        W, MD, MN = cells(o)
        vals = {n: f(W, MD, MN) for n, f in CONSTRUCTIONS.items()}
        nuis = {"W_size": W, "MN_vigour": MN, "loss_mock_only": 1 - MD / MN}
        allv = {**vals, **nuis}
        names = list(allv)
        rho = {a: {b: float(stats.spearmanr(allv[a], allv[b])[0]) for b in names} for a in names}
        A = len(W)
        cons = {}
        for kf in out["k_fracs"]:
            k = int(round(kf * A))
            tops = {n: topk(v, k) for n, v in vals.items()}
            pair = {f"{a}~{b}": {"jaccard": jac(tops[a], tops[b]),
                                 "flip_rate": 1 - len(tops[a] & tops[b]) / k}
                    for a, b in itertools.combinations(vals, 2)}
            pair["all_three"] = len(set.intersection(*tops.values()))
            cons[str(kf)] = {"k": k, "pairs": pair}
        noise = {}
        for model in MODELS:
            Wb, MDb, MNb = draw(o, model, RNG, B=300)
            nm = {}
            for n, f in CONSTRUCTIONS.items():
                fb = f(Wb, MDb, MNb)
                rr, jj, ff = [], [], []
                k = int(round(0.2 * A))
                t0 = topk(vals[n], k)
                for b in range(fb.shape[0]):
                    ok = np.isfinite(fb[b])
                    rr.append(stats.spearmanr(fb[b][ok], vals[n][ok])[0])
                    tb = topk(np.where(ok, fb[b], -np.inf), k)
                    jj.append(jac(t0, tb))
                    ff.append(1 - len(t0 & tb) / k)
                nm[n] = {"rho_median": float(np.median(rr)), "rho_p5": float(np.percentile(rr, 5)),
                         "jaccard_top20_median": float(np.median(jj)),
                         "flip_rate_top20_median": float(np.median(ff)),
                         "flip_rate_top20_p95": float(np.percentile(ff, 95))}
            noise[model] = nm
        # excess disagreement at 20% under the reference model
        k20 = cons["0.2"]["pairs"]
        excess = {p: k20[p]["flip_rate"] - max(noise[REF][p.split("~")[0]]["flip_rate_top20_p95"],
                                                noise[REF][p.split("~")[1]]["flip_rate_top20_p95"])
                  for p in k20 if "~" in p}
        out["organs"][o] = {"spearman": rho, "construction_agreement": cons,
                            "noise_agreement": noise, "excess_flip_rate_vs_noise_p95_M3": excess,
                            "rank_scatter": {"increase": stats.rankdata(vals["increase"]) / A,
                                             "rescue": stats.rankdata(vals["rescue"]) / A,
                                             "gain": stats.rankdata(vals["gain"]) / A}}
    return out


# ============================================================= E4 canonical
LAM_GRID = np.round(np.arange(-0.5, 1.5001, 0.05), 3)
LAM_TRUE = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]


def fit_b(W, MD, MN):
    g = W - MD
    ok = (g > 0) & (MN > MD) & (MD > 0)
    X = np.column_stack([np.ones(ok.sum()), np.log(MD[ok]), np.log(MN[ok] - MD[ok])])
    beta, *_ = np.linalg.lstsq(X, np.log(g[ok]), rcond=None)
    resid = np.log(g[ok]) - X @ beta
    return beta, resid


def e4():
    out = {"lambda_grid": LAM_GRID, "organs": {}}
    for o in ORG:
        W, MD, MN = cells(o)
        oo = {}
        # (a) scale invariance, numerically
        inv = {}
        for s in (0.5, 2.0):
            for n, f in {**CONSTRUCTIONS, "f_0.5": lambda a, b, c: f_lambda(a, b, c, 0.5)}.items():
                r = np.max(np.abs(f(s * W, s * MD, s * MN) / f(W, MD, MN) - 1))
                inv.setdefault(n, {})[str(s)] = float(r)
        oo["scale_invariance_max_rel_change"] = inv
        # (b) nuisance loading along the lambda family
        loss = 1 - MD / MN
        oo["loading"] = {
            "rho_vs_loss": [float(stats.spearmanr(f_lambda(W, MD, MN, l), loss)[0]) for l in LAM_GRID],
            "rho_vs_MN": [float(stats.spearmanr(f_lambda(W, MD, MN, l), MN)[0]) for l in LAM_GRID],
            "gain_rho_vs_loss": float(stats.spearmanr(gain(W, MD, MN), loss)[0]),
            "gain_rho_vs_MN": float(stats.spearmanr(gain(W, MD, MN), MN)[0]),
        }
        # (c) the log-linear mechanism fit on observed data
        beta, resid = fit_b(W, MD, MN)
        oo["fit_observed"] = {"b0": beta[0], "b1_lambda": beta[1], "b2": beta[2],
                              "b1_plus_b2": beta[1] + beta[2], "resid_sd": float(resid.std(ddof=3))}
        # (d) uncertainty: case resampling of accessions x noise model M3
        Wb, MDb, MNb = draw(o, REF, RNG, B=1000)
        bs = []
        A = len(W)
        for b in range(1000):
            ii = RNG.integers(0, A, A)
            be, _ = fit_b(Wb[b, ii], MDb[b, ii], MNb[b, ii])
            bs.append(be)
        bs = np.array(bs)
        oo["bootstrap"] = {"b1_ci95": np.percentile(bs[:, 1], [2.5, 97.5]),
                           "b1_plus_b2_ci95": np.percentile(bs[:, 1] + bs[:, 2], [2.5, 97.5]),
                           "b1_samples": bs[:300, 1], "b1b2_samples": (bs[:300, 1] + bs[:300, 2])}
        # (e) simulation recovery: known lambda*, empirical baselines, M3 noise
        rec = {}
        sd_theta = float(resid.std(ddof=3))
        for lam in LAM_TRUE:
            est, sums = [], []
            for r in range(200):
                theta = np.exp(RNG.normal(0, sd_theta, A))
                shape = np.power(MD, lam) * np.power(MN - MD, 1 - lam)
                gt = theta * shape
                gt *= np.median(W - MD) / np.median(gt)
                Wt = MD + gt
                se = CV[o] / np.sqrt(N_REP)
                Ws = lognorm(lognorm(Wt, se, RNG), CVB[o], RNG)
                MDs = lognorm(lognorm(MD, se, RNG), CVB[o], RNG)
                MNs = lognorm(lognorm(MN, se, RNG), CVB[o], RNG)
                be, _ = fit_b(Ws, MDs, MNs)
                est.append(be[1])
                sums.append(be[1] + be[2])
            rec[str(lam)] = {"mean": float(np.mean(est)), "sd": float(np.std(est)),
                             "bias": float(np.mean(est) - lam),
                             "sum_mean": float(np.mean(sums)), "sum_sd": float(np.std(sums))}
        oo["recovery"] = rec
        lams = np.array(LAM_TRUE)
        biases = np.array([rec[str(l)]["bias"] for l in lams])
        # bias-correct by inverting the (monotone) mean-estimate curve
        means = np.array([rec[str(l)]["mean"] for l in lams])
        lam_hat = float(np.interp(beta[1], means, lams, left=np.nan, right=np.nan))
        ci = [float(np.interp(x, means, lams, left=np.nan, right=np.nan)) for x in oo["bootstrap"]["b1_ci95"]]
        oo["lambda_hat_corrected"] = lam_hat
        oo["lambda_ci95_corrected"] = ci
        oo["bias_curve"] = {"lambda_true": lams, "bias": biases}
        # (f) does restricting to the data-admissible lambda range restore agreement?
        lo = np.nan_to_num(ci[0], nan=lams.min()); hi = np.nan_to_num(ci[1], nan=lams.max())
        flo, fhi = f_lambda(W, MD, MN, lo), f_lambda(W, MD, MN, hi)
        k = int(round(0.2 * A))
        oo["within_admissible"] = {"lambda_lo": lo, "lambda_hi": hi,
                                   "spearman": float(stats.spearmanr(flo, fhi)[0]),
                                   "flip_rate_top20": 1 - len(topk(flo, k) & topk(fhi, k)) / k}
        out["organs"][o] = oo
    return out


# ============================================================ E5 four-column
def verdict(d, s, delta):
    """Three-valued comparison. C: equivalent within delta (TOST, 5%);
    D: different by more than delta (one-sided 5%); U: undecided (decline)."""
    ad = np.abs(d)
    v = np.full(d.shape, "U", dtype="<U1")
    v[ad + Z * s < delta] = "C"
    v[ad - Z * s > delta] = "D"
    return v


def pair_idx(A):
    return np.triu_indices(A, 1)


def baseline_verdict(base, I, J, delta_b):
    vs = [verdict(v[I] - v[J], np.hypot(s[I], s[J]), delta_b) for v, s in base.values()]
    out = np.full(vs[0].shape, "U", dtype="<U1")
    out[np.all([v == "C" for v in vs], axis=0)] = "C"
    out[np.any([v == "D" for v in vs], axis=0)] = "D"
    return out


def lam_of(o):
    lam = E4["organs"][o]["lambda_hat_corrected"]
    return 0.0 if lam is None or not np.isfinite(lam) else float(lam)


def column_values_model(o, lam, model):
    W, MD, MN = cells(o)
    Wb, MDb, MNb = draw(o, model, RNG)
    fs = {**CONSTRUCTIONS, "canonical": lambda a, b, c: f_lambda(a, b, c, lam)}
    resp = {n: (f(W, MD, MN), np.nanstd(f(Wb, MDb, MNb), axis=0, ddof=1)) for n, f in fs.items()}
    base = {"logMD": (np.log(MD), np.nanstd(np.log(MDb), axis=0, ddof=1)),
            "logMN": (np.log(MN), np.nanstd(np.log(MNb), axis=0, ddof=1))}
    return resp, base


def e5():
    """Four columns: two baseline (mock) columns and two response columns per pair.

    For every pair and every column we report the three-valued verdict and, more
    informatively, the smallest margin at which the pair could have been
    certified equivalent: delta_min = |d| + z s. Verdicts are reported for two
    noise models (M2 without batch, M3 with batch) and a sweep of margins."""
    margins = [0.25, 0.5, 1.0, 2.0]
    out = {"z": Z, "margins_sd": margins, "baseline_margins_log": [float(np.log(x)) for x in (1.15, 1.3, 1.5)],
           "organs": {}}
    global VERDICTS
    VERDICTS = {}
    names = D.accessions["name"].to_dict()
    for o in ORG:
        lam = lam_of(o)
        A = len(MAIN)
        I, J = pair_idx(A)
        oo = {"lambda_used": lam, "n_pairs": int(len(I)), "models": {}}
        for model in ["M2_plant_all_cells", REF]:
            resp, base = column_values_model(o, lam, model)
            mo = {"baseline": {}, "response": {}}
            # baseline: both mock columns must be equivalent to certify, either divergent to diverge
            for bm in out["baseline_margins_log"]:
                bv = baseline_verdict(base, I, J, bm)
                mo["baseline"][f"{bm:.3f}"] = {k: int((bv == k).sum()) for k in "CDU"}
            bmin = np.maximum.reduce([np.abs(v[I] - v[J]) + Z * np.hypot(s[I], s[J]) for v, s in base.values()])
            mo["baseline_min_certifiable_log_margin_quantiles"] = np.percentile(bmin, [5, 25, 50, 75, 95])
            dmins, verdicts = {}, {}
            for n, (v, s) in resp.items():
                sd = np.std(v, ddof=1)
                d = v[I] - v[J]
                sp = np.hypot(s[I], s[J])
                dmin = (np.abs(d) + Z * sp) / sd
                dmins[n] = dmin
                sweep = {}
                for m in margins:
                    vv = verdict(d, sp, m * sd)
                    sweep[str(m)] = {k: int((vv == k).sum()) for k in "CDU"}
                    if m == 0.5:
                        verdicts[n] = vv
                # certified ordering at the reference margin: sign of a D verdict
                sgn = np.where(verdicts[n] == "D", np.sign(d), 0)
                verdicts[n + "_sign"] = sgn
                mo["response"][n] = {
                    "sd_between": float(sd), "median_pair_se_over_sd": float(np.median(sp) / sd),
                    "min_certifiable_margin_sd_quantiles": np.percentile(dmin, [1, 5, 25, 50, 75, 95]),
                    "frac_certifiable_at_1sd": float(np.mean(dmin < 1.0)),
                    "sweep": sweep}
            agree = {}
            for a, b in itertools.combinations(resp, 2):
                sa, sb = verdicts[a + "_sign"], verdicts[b + "_sign"]
                agree[f"{a}~{b}"] = {
                    "same_verdict_0.5sd": float(np.mean(verdicts[a] == verdicts[b])),
                    "both_certified_order": int(np.sum((sa != 0) & (sb != 0))),
                    "opposite_certified_order": int(np.sum(sa * sb < 0)),
                    "certified_by_one_only": int(np.sum((sa != 0) ^ (sb != 0)))}
            mo["agreement_between_constructions"] = agree
            # four-column classes at the reference margins
            bv = baseline_verdict(base, I, J, float(np.log(1.3)))
            rv = verdicts["canonical"]
            mo["four_column_table"] = {b: {r: int(((bv == b) & (rv == r)).sum()) for r in "CDU"} for b in "CDU"}
            # the most extreme "false friend" and "convergent" candidates by continuous scores
            v, s = resp["canonical"]
            zr = (np.abs(v[I] - v[J]) - 0.5 * np.std(v, ddof=1)) / np.hypot(s[I], s[J])
            bz = bmin  # smallest log margin at which baselines certify as equivalent
            cand_ff = np.where(bz < np.percentile(bz, 5))[0]
            cand_ff = cand_ff[np.argsort(-zr[cand_ff])][:5]
            cand_cv = np.where(dmins["canonical"] < np.percentile(dmins["canonical"], 5))[0]
            bdist = np.hypot(base["logMD"][0][I] - base["logMD"][0][J], base["logMN"][0][I] - base["logMN"][0][J])
            cand_cv = cand_cv[np.argsort(-bdist[cand_cv])][:5]

            def ex(sel):
                return [{"a": int(MAIN[I[t]]), "b": int(MAIN[J[t]]),
                         "a_name": names.get(MAIN[I[t]]), "b_name": names.get(MAIN[J[t]]),
                         "resp_a": float(v[I[t]]), "resp_b": float(v[J[t]]),
                         "baseline_logdist": float(bdist[t]), "baseline_min_margin_log": float(bz[t]),
                         "resp_z_beyond_half_sd": float(zr[t]),
                         "resp_min_margin_sd": float(dmins["canonical"][t])} for t in sel]
            mo["closest_false_friends"] = ex(cand_ff)
            mo["closest_convergent"] = ex(cand_cv)
            oo["models"][model] = mo
            if model == REF:
                VERDICTS[o] = {n: verdicts[n] for n in resp} | {"_sign_" + n: verdicts[n + "_sign"] for n in resp}
        out["organs"][o] = oo
    return out


# ============================================================ E6 calibration
def e6():
    """Operating characteristics of the three-valued verdict, as a function of the
    true difference (in margin units) and of the pair standard error (in margin units).
    The empirical s/delta of this screen is included as one of the levels."""
    ratios = np.round(np.linspace(0, 3, 25), 3)
    emp = []
    for o in ORG:
        resp, _ = column_values_model(o, lam_of(o), REF)
        v, s = resp["canonical"]
        I, J = pair_idx(len(v))
        emp.append(np.hypot(s[I], s[J]) / (0.5 * np.std(v, ddof=1)))
    emp_med = float(np.median(np.concatenate(emp)))
    levels = [0.1, 0.25, 0.5, 1.0, round(emp_med, 3)]
    N = 40000
    curves = {}
    for sl in levels:
        rows = []
        for r in ratios:
            d = RNG.normal(r, sl, N)
            v = verdict(d, np.full(N, sl), 1.0)
            rows.append({"true_diff_over_delta": float(r), "P_C": float(np.mean(v == "C")),
                         "P_D": float(np.mean(v == "D")), "P_U": float(np.mean(v == "U"))})
        curves[str(sl)] = rows
    allrows = [x for rows in curves.values() for x in rows]
    return {"s_over_delta_levels": levels, "empirical_s_over_delta_at_0.5sd": emp_med,
            "curves": curves, "alpha": 0.05,
            "max_false_correspond_at_or_beyond_margin": max(x["P_C"] for x in allrows if x["true_diff_over_delta"] >= 1),
            "max_false_diverge_at_or_within_margin": max(x["P_D"] for x in allrows if x["true_diff_over_delta"] <= 1)}


# ================================================================== E7 batch
def e7():
    out = {}
    for o in ORG:
        blocks = [b for b in col0_blocks(o) if len(b) >= 3]
        means = np.array([b.mean() for b in blocks])
        f, p = stats.f_oneway(*blocks)
        n = np.mean([len(b) for b in blocks])
        grand = np.concatenate(blocks)
        msb = n * means.var(ddof=1)
        msw = np.mean([b.var(ddof=1) for b in blocks])
        icc = (msb - msw) / (msb + (n - 1) * msw)
        acc_means = D.cells.loc[MAIN, f"{o}_W"].to_numpy()
        dup = {}
        for gid in D.repeated:
            if gid == 6909:
                continue
            sub = D.raw[D.raw.ID == gid].copy()
            sub["block"] = sub["sheet_row"].diff().fillna(1).ne(1).cumsum()
            bm = sub.groupby("block")[RAWCOL[o]].mean().to_numpy()
            dup[str(gid)] = bm
        out[o] = {"n_blocks": len(blocks), "block_means": means, "grand_mean": float(grand.mean()),
                  "block_starts_sheet_row": col0_blocks(o).index.tolist(),
                  "F": float(f), "p": float(p), "ICC1": float(icc), "batch_cv": CVB[o],
                  "plant_cv": CV[o],
                  "between_block_sd_over_between_accession_sd": float(means.std(ddof=1) / acc_means.std(ddof=1)),
                  "block_mean_percentile_span": [float(np.mean(acc_means < means.min())),
                                                 float(np.mean(acc_means < means.max()))],
                  "duplicated_lines_block_means": dup}
    return out


# ============================================================= E8 candidates
def family(m):
    m = str(m).lower()
    if "rescue" in m:
        return "rescue"
    if "increase" in m:
        return "increase"
    if "gain" in m and "drought" in m:
        return "gain"
    if "loss" in m:
        return "mock_only_loss"
    if "fw" in m:
        return "inoculated_size"
    return "other"


def metric_col(m):
    s = str(m).strip()
    cols = {c.lower().rstrip("_"): c for c in D.traits.columns}
    key = s.replace(" ", "_").lower().rstrip("_")
    key = key.replace("drought", "drought")
    if key in cols:
        return cols[key]
    alias = {"root_rescue": "Root_Rescuer", "total_rescue": None}
    return alias.get(key)


def e8():
    c = D.candidates.copy()
    c["family"] = c["Metric"].map(family)
    loss = D.traits["AVG_percent_Loss_in_SFW_under_drought"]
    size = D.traits["AVG_Shoot_FW"]
    rows = []
    for _, r in c.iterrows():
        col = metric_col(r["Metric"])
        rl = rs = np.nan
        if col is not None:
            x = D.traits[col]
            ok = x.notna() & loss.notna()
            rl = stats.spearmanr(x[ok], loss[ok])[0]
            rs = stats.spearmanr(x[ok], size[ok])[0]
        rows.append({"gene": r["Gene"], "symbol": r["Gene Symbol"] if isinstance(r["Gene Symbol"], str) else "",
                     "variant": r["Gene Variant"], "metric": r["Metric"], "family": r["family"],
                     "metric_column": col, "rho_metric_vs_mock_loss": rl, "rho_metric_vs_size": rs})
    fam_gene = c.groupby("Gene")["family"].nunique()
    met_gene = c.groupby("Gene")["Metric"].nunique()
    return {"rows": rows, "n_rows": int(len(c)), "n_genes": int(c.Gene.nunique()),
            "genes_by_family": c.groupby("family")["Gene"].nunique().to_dict(),
            "genes_multi_metric": int((met_gene > 1).sum()),
            "genes_multi_family": int((fam_gene > 1).sum()),
            "multi_metric_genes": {g: sorted(c[c.Gene == g]["Metric"].unique().tolist())
                                   for g in met_gene[met_gene > 1].index}}


# ================================================================== E9 ratio
def e9():
    W, MD, MN = cells("Shoot")
    lam = lam_of("Shoot")
    eps = np.round(np.arange(0, 0.51, 0.05), 2)
    rows = []
    for e in eps:
        MDm = (1 - e) * MD
        Wm = MDm + (W - MD)                    # same absolute gain
        r = {"eps": float(e)}
        for n, f in {**CONSTRUCTIONS, "canonical": lambda a, b, c: f_lambda(a, b, c, lam)}.items():
            ratio = f(Wm, MDm, MN) / f(W, MD, MN)
            r[n] = {"median": float(np.median(ratio)), "q25": float(np.percentile(ratio, 25)),
                    "q75": float(np.percentile(ratio, 75))}
        rows.append(r)
    return {"lambda_used": lam, "rows": rows,
            "statement": "baseline-only mutant: mock-drought biomass reduced by eps, absolute gain and non-stress biomass unchanged"}


# ================================================================ E10 classes
def resolution_tiers(v, s):
    """Longest chain in the certified order a > b  <=>  v_a - v_b > z * s_ab.
    Returns each item's depth (1 = bottom tier) and the maximum depth."""
    order = np.argsort(v)
    depth = np.ones(len(v), int)
    for pos, i in enumerate(order):
        lower = order[:pos]
        if len(lower):
            ok = (v[i] - v[lower]) > Z * np.hypot(s[i], s[lower])
            if ok.any():
                depth[i] = 1 + depth[lower[ok]].max()
    return depth


def e10():
    """How many levels of rescue can the screen resolve, and do the levels agree
    across constructions? Uses the certified strict order (margin 0)."""
    out = {}
    for o in ORG:
        lam = lam_of(o)
        oo = {}
        for model in ["M2_plant_all_cells", REF]:
            resp, _ = column_values_model(o, lam, model)
            tiers = {n: resolution_tiers(v, s) for n, (v, s) in resp.items()}
            oo[model] = {
                "max_depth": {n: int(t.max()) for n, t in tiers.items()},
                "tier_sizes_canonical": np.bincount(tiers["canonical"])[1:],
                "tier_spearman": {f"{a}~{b}": float(stats.spearmanr(tiers[a], tiers[b])[0])
                                  for a, b in itertools.combinations(tiers, 2)},
                "tier_ari": {f"{a}~{b}": float(adjusted_rand_score(tiers[a], tiers[b]))
                             for a, b in itertools.combinations(tiers, 2)},
                "labels": {n: t for n, t in tiers.items()}}
        out[o] = oo
    return out


# ================================================================== E12 design
def e12():
    """Replication needed before pairwise equivalence becomes certifiable.
    n = plants per cell per run, r = independent runs (each with its own batch
    effect). Relative SE of a cell mean: sqrt(cv_p^2/(n r) + cv_b^2/r)."""
    grid_n = [7, 14, 21, 28]
    grid_r = [1, 2, 3, 4, 6]
    margins = [0.5, 1.0]
    out = {"grid_n": grid_n, "grid_r": grid_r, "margins_tau": margins, "organs": {}}
    for o in ORG:
        lam = lam_of(o)
        W, MD, MN = cells(o)
        y = f_lambda(W, MD, MN, lam)
        Wb, MDb, MNb = draw(o, REF, RNG)
        se_now = np.nanstd(f_lambda(Wb, MDb, MNb, lam), axis=0, ddof=1)
        tau2 = max(np.var(y, ddof=1) - np.mean(se_now ** 2), 1e-12)
        tau = float(np.sqrt(tau2))
        tab = {}
        for n in grid_n:
            for r in grid_r:
                rel = np.sqrt(CV[o] ** 2 / (n * r) + CVB[o] ** 2 / r)
                Ws, MDs, MNs = (lognorm(np.broadcast_to(X, (400, len(X))), rel, RNG) for X in (W, MD, MN))
                se = float(np.median(np.nanstd(f_lambda(Ws, MDs, MNs, lam), axis=0, ddof=1)))
                sp = np.sqrt(2) * se
                cell = {"se_over_tau": se / tau, "reliability": tau2 / (tau2 + se ** 2)}
                for m in margins:
                    cell[f"P_certify_equal_{m}"] = float(max(0.0, 2 * stats.norm.cdf((m * tau - Z * sp) / sp) - 1))
                    cell[f"P_certify_diff_at_{2*m}tau"] = float(stats.norm.sf((m * tau + Z * sp - 2 * m * tau) / sp))
                tab[f"{n}x{r}"] = cell
        out["organs"][o] = {"tau": tau, "lambda": lam, "table": tab,
                            "current": tab[f"{N_REP}x1"]}
    return out


# ============================================================= E11 geography
def e11():
    acc = D.accessions.reindex(MAIN)
    lat = pd.to_numeric(acc["latitude"], errors="coerce").to_numpy()
    lon = pd.to_numeric(acc["longitude"], errors="coerce").to_numpy()
    out = {"n_with_coords": int(np.isfinite(lat).sum()),
           "n_countries": int(acc["country"].nunique()),
           "countries_top": acc["country"].value_counts().head(8).to_dict(), "organs": {}}
    for o in ORG:
        W, MD, MN = cells(o)
        lam = lam_of(o)
        vars_ = {"canonical": f_lambda(W, MD, MN, lam), "rescue": rescue(W, MD, MN),
                 "increase": increase(W, MD, MN), "loss_mock_only": 1 - MD / MN, "MN": MN}
        ok = np.isfinite(lat) & np.isfinite(lon)
        out["organs"][o] = {n: {"rho_lat": float(stats.spearmanr(v[ok], lat[ok])[0]),
                                "p_lat": float(stats.spearmanr(v[ok], lat[ok])[1]),
                                "rho_lon": float(stats.spearmanr(v[ok], lon[ok])[0]),
                                "p_lon": float(stats.spearmanr(v[ok], lon[ok])[1])} for n, v in vars_.items()}
    return out


# ================================================================ per-accession export
def export_accessions():
    rows = []
    acc = D.accessions
    per = {}
    for o in ORG:
        W, MD, MN = cells(o)
        lam = lam_of(o)
        Wb, MDb, MNb = draw(o, REF, np.random.default_rng(SEED))
        per[o] = {"W": W, "MD": MD, "MN": MN, "gain": gain(W, MD, MN), "increase": increase(W, MD, MN),
                  "rescue": rescue(W, MD, MN), "canonical": f_lambda(W, MD, MN, lam),
                  "gain_se": np.nanstd(gain(Wb, MDb, MNb), axis=0, ddof=1),
                  "canonical_se": np.nanstd(f_lambda(Wb, MDb, MNb, lam), axis=0, ddof=1)}
    for k, gid in enumerate(MAIN):
        a = acc.loc[gid] if gid in acc.index else None
        r = {"gid": gid, "name": None if a is None else a["name"],
             "country": None if a is None else a["country"],
             "lat": None if a is None else pd.to_numeric(a["latitude"], errors="coerce"),
             "lon": None if a is None else pd.to_numeric(a["longitude"], errors="coerce")}
        for o in ORG:
            for key, v in per[o].items():
                r[f"{o.lower()}_{key}"] = float(v[k])
            r[f"{o.lower()}_tier"] = int(E10[o][REF]["labels"]["canonical"][k])
        rows.append(r)
    return {"accessions": rows}


if __name__ == "__main__":
    t0 = time.time()
    E1 = run("E1_audit", e1)
    E2 = run("E2_heterogeneity", e2)
    E3 = run("E3_dependence", e3)
    E4 = run("E4_canonical", e4)
    E5 = run("E5_four_column", e5)
    E6 = run("E6_calibration", e6)
    E7 = run("E7_batch", e7)
    E8 = run("E8_candidates", e8)
    E9 = run("E9_ratio", e9)
    E10 = run("E10_classes", e10)
    E11 = run("E11_geography", e11)
    E12 = run("E12_design", e12)
    run("accessions", export_accessions)
    save("index", {"seed": SEED, "bootstrap_B": B, "noise_models": MODELS, "reference_model": REF,
                   "plant_cv": CV, "batch_cv": CVB, "n_main": len(MAIN), "excluded_repeated": D.repeated,
                   "experiments": results_index, "total_seconds": round(time.time() - t0, 1)})
    print(f"done in {time.time() - t0:.1f}s")
