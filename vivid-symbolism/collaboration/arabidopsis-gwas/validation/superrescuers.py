"""S1: Is there an accession that breaks the ~30-40% rescue ceiling?

A single run's rescue is mostly noise (F4: run-to-run rho ~0.1-0.2), so ranking accessions
by their raw run-1 value would mostly pick lucky runs. Instead, each accession's true rescue
is estimated by empirical-Bayes shrinkage:

  observed rescue in run r  =  true rescue (variance tau^2 across accessions)
                             + run-specific error (variance sigma^2, measured from runs 2 vs 3)

posterior mean = mu + tau^2 / (tau^2 + sigma^2 / k) * (ybar - mu),  k = number of runs.
Accessions grown in three runs (the rescreen) get much tighter estimates.

Writes results/S1_superrescuers.json.
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import SEED, save
from load_full import extract_full

RNG = np.random.default_rng(SEED)
ORG = {"Shoot": "shoot", "Root": "root"}
COL0 = 6909
CEILING = 0.40

P = extract_full()
P = P[P.gid.notna()].copy()
P["gid"] = P["gid"].astype(int)
P["cell"] = P["media"] + "_" + P["treatment"]
P.loc[P.media == "NS", "br"] = "NS"
NAMES = None


def names():
    global NAMES
    if NAMES is None:
        from common import load
        NAMES = load().accessions[["name", "country", "latitude", "longitude"]]
    return NAMES


def per_run(o):
    """Rescue, gain and drought loss per accession and drought run, with bootstrap SE."""
    mn = P[P.cell == "NS_Mock"].groupby("gid")[o].apply(lambda s: s.dropna().to_numpy())
    rows = []
    for (gid, br), g in P[P.media == "DS"].groupby(["gid", "br"]):
        W = g[g.treatment == "WCS417"][o].dropna().to_numpy()
        MD = g[g.treatment == "Mock"][o].dropna().to_numpy()
        MN = mn.get(gid, np.array([]))
        if min(len(W), len(MD), len(MN)) < 3:
            continue
        b = lambda x: x[RNG.integers(0, len(x), (500, len(x)))].mean(1)
        wb, mdb, mnb = b(W), b(MD), b(MN)
        rb = (wb - mdb) / (mnb - mdb)
        rows.append({"gid": gid, "run": br, "rescue": (W.mean() - MD.mean()) / (MN.mean() - MD.mean()),
                     "rescue_se": float(np.nanstd(rb, ddof=1)), "gain": W.mean() - MD.mean(),
                     "loss": 1 - MD.mean() / MN.mean(), "MD": MD.mean(), "MN": MN.mean(), "W": W.mean()})
    return pd.DataFrame(rows)


def analyse(O, o):
    R = per_run(o)
    col0 = R[R.gid == COL0].set_index("run")
    A = R[R.gid != COL0]
    # error variance between runs, from accessions grown in both run 2 and run 3
    w = A.pivot_table(index="gid", columns="run", values="rescue")
    both = w[["BR2", "BR3"]].dropna()
    sigma2 = float(((both.BR2 - both.BR3) ** 2).mean() / 2)
    run1 = A[A.run == "BR1"]
    mu = float(run1.rescue.mean())
    tau2 = max(float(run1.rescue.var(ddof=1)) - sigma2, 1e-6)
    reliability_1run = tau2 / (tau2 + sigma2)
    # Col-0 pooled over runs (147 plants under drought) as the reference
    col0_mean = float(col0.rescue.mean())
    out_rows = []
    for gid, g in A.groupby("gid"):
        k = len(g)
        ybar = float(g.rescue.mean())
        shrink = tau2 / (tau2 + sigma2 / k)
        post = mu + shrink * (ybar - mu)
        post_sd = float(np.sqrt(1 / (1 / tau2 + k / sigma2)))
        nm = names()
        out_rows.append({
            "gid": int(gid), "name": nm["name"].get(gid), "country": nm["country"].get(gid),
            "runs": k, "runs_list": sorted(g.run), "rescue_by_run": {r: float(v) for r, v in zip(g.run, g.rescue)},
            "gain_by_run": {r: float(v) for r, v in zip(g.run, g.gain)},
            "raw_mean_rescue": ybar, "posterior_rescue": post, "posterior_sd": post_sd,
            "P_above_ceiling": float(stats.norm.sf((CEILING - post) / post_sd)),
            "P_above_col0": float(stats.norm.sf((col0_mean - post) / post_sd)),
            "P_above_2x_col0": float(stats.norm.sf((2 * col0_mean - post) / post_sd)),
            "mean_gain": float(g.gain.mean()), "mean_loss": float(g.loss.mean()),
            "MN": float(g.MN.iloc[0]),
        })
    T = pd.DataFrame(out_rows).sort_values("posterior_rescue", ascending=False)
    # ---- discovery (run 1) -> replication (runs 2, 3) among the rescreened accessions
    rs = w[["BR1", "BR2", "BR3"]].dropna()
    n = len(rs)
    up = lambda col: (rs[col].rank(ascending=False, method="max") / n)     # upper-tail rank fraction
    rep = pd.DataFrame({"name": [names()["name"].get(g) for g in rs.index], "BR1": rs.BR1, "BR2": rs.BR2, "BR3": rs.BR3,
                        "q1": up("BR1"), "q2": up("BR2"), "q3": up("BR3")}, index=rs.index)
    rep["p_replication"] = rep.q2 * rep.q3          # P(both independent runs this high | no accession effect)
    disc = rep[rep.q1 <= 0.25].sort_values("p_replication")
    disc["p_bonferroni"] = np.minimum(1, disc.p_replication * len(disc))
    # permutation check of the same statistic (runs shuffled independently across accessions)
    best = disc.p_replication.min()
    perm_hits = 0
    for _ in range(20000):
        q2 = RNG.permutation(rep.q2.to_numpy()); q3 = RNG.permutation(rep.q3.to_numpy())
        if (q2 * q3)[(rep.q1 <= 0.25).to_numpy()].min() <= best:
            perm_hits += 1
    # per-run comparison of each discovery accession with Col-0 (bootstrap CI of the difference)
    def boot_rescue(gid, br):
        g = P[(P.gid == gid) & (P.media == "DS") & (P.br == br)]
        W_ = g[g.treatment == "WCS417"][o].dropna().to_numpy(); M_ = g[g.treatment == "Mock"][o].dropna().to_numpy()
        N_ = P[(P.gid == gid) & (P.cell == "NS_Mock")][o].dropna().to_numpy()
        bb = lambda x: x[RNG.integers(0, len(x), (2000, len(x)))].mean(1)
        return (bb(W_) - bb(M_)) / (bb(N_) - bb(M_))
    vs_col0 = {}
    for gid in disc.index[:6]:
        vs_col0[names()["name"].get(gid)] = {br: {"diff": float(np.median(d := boot_rescue(gid, br) - boot_rescue(COL0, br))),
                                                   "lo": float(np.nanpercentile(d, 2.5)), "hi": float(np.nanpercentile(d, 97.5))}
                                              for br in ["BR1", "BR2", "BR3"]}
    loss_by_run = {names()["name"].get(g): {r: float(v) for r, v in zip(A[A.gid == g].run, A[A.gid == g].loss)}
                   for g in disc.index[:6]}
    gain_by_run = {names()["name"].get(g): {r: float(v) for r, v in zip(A[A.gid == g].run, A[A.gid == g].gain)}
                   for g in disc.index[:6]}
    # variance components from the two unselected runs (covariance = true between-accession variance)
    cov23 = float(np.cov(rs.BR2, rs.BR3)[0, 1])
    boots = [np.cov(*rs.iloc[RNG.integers(0, n, n)][["BR2", "BR3"]].to_numpy().T)[0, 1] for _ in range(2000)]
    # did the rescreen include high run-1 rescuers, and did they stay high?
    rescreen = T[T.runs >= 3].copy()
    r1 = run1.set_index("gid").rescue
    rescreen["run1_percentile"] = rescreen.gid.map(r1.rank(pct=True))
    hi = rescreen[rescreen.run1_percentile >= 0.75]
    hi_rows = [{"name": r["name"], "run1_percentile": float(r.run1_percentile), **{k: r["rescue_by_run"].get(k) for k in ["BR1", "BR2", "BR3"]}}
               for _, r in hi.iterrows()]
    # raw run-1 top 10: how many would survive?
    top_raw = run1.sort_values("rescue", ascending=False).head(10).gid.tolist()
    return T, {
        "organ": O, "n_accessions": int(len(T)), "mu_run1": mu, "tau2": tau2, "sigma2_run_to_run": sigma2,
        "tau_sd": float(np.sqrt(tau2)), "reliability_one_run": reliability_1run,
        "reliability_three_runs": tau2 / (tau2 + sigma2 / 3),
        "col0_by_run": {r: {"rescue": float(v.rescue), "gain": float(v.gain), "loss": float(v.loss)} for r, v in col0.iterrows()},
        "col0_pooled_rescue": col0_mean, "ceiling_used": CEILING,
        "max_raw_run1_rescue": float(run1.rescue.max()),
        "n_raw_run1_above_ceiling": int((run1.rescue > CEILING).sum()),
        "n_posterior_above_ceiling": int((T.posterior_rescue > CEILING).sum()),
        "n_P_above_ceiling_gt_0.5": int((T.P_above_ceiling > 0.5).sum()),
        "n_P_above_2x_col0_gt_0.9": int((T.P_above_2x_col0 > 0.9).sum()),
        "top_by_posterior": T.head(15).to_dict(orient="records"),
        "rescreen_high_run1": hi_rows,
        "raw_top10_run1": [{"name": names()["name"].get(g), "run1": float(r1[g]),
                            "posterior": float(T.set_index("gid").posterior_rescue[g]),
                            "runs": int(T.set_index("gid").runs[g])} for g in top_raw],
        "corr_posterior_vs_loss": float(stats.spearmanr(T.posterior_rescue, T.mean_loss)[0]),
        "replication": {"n_rescreened_with_3_runs": int(n), "n_discovered_top25_run1": int(len(disc)),
                        "discovered": disc.reset_index().rename(columns={"index": "gid"}).to_dict(orient="records"),
                        "permutation_p_best": perm_hits / 20000,
                        "vs_col0_by_run": vs_col0, "loss_by_run": loss_by_run, "gain_by_run": gain_by_run,
                        "all_rescreened": rep.reset_index().rename(columns={"index": "gid"}).to_dict(orient="records")},
        "true_variance_from_runs23": {"cov": cov23, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                      "tau_sd": float(np.sqrt(max(cov23, 0))), "total_var_run2": float(rs.BR2.var()),
                                      "total_var_run3": float(rs.BR3.var())},
    }


if __name__ == "__main__":
    out = {}
    tables = []
    for O, o in ORG.items():
        T, s = analyse(O, o)
        out[O] = s
        tables.append(T.assign(organ=O))
    save("S1_superrescuers", out)
    pd.concat(tables).to_csv(__import__("common").RESULTS / "S1_rescue_posterior.csv", index=False)
    for O in ORG:
        s = out[O]
        print(f"\n== {O}: mu={s['mu_run1']:.3f} tau={s['tau_sd']:.3f} sigma={np.sqrt(s['sigma2_run_to_run']):.3f} "
              f"rel1={s['reliability_one_run']:.2f} rel3={s['reliability_three_runs']:.2f}")
        print("Col-0 by run:", {k: round(v['rescue'], 3) for k, v in s['col0_by_run'].items()}, "pooled", round(s['col0_pooled_rescue'], 3))
        print(f"raw run1 > {CEILING}: {s['n_raw_run1_above_ceiling']} (max {s['max_raw_run1_rescue']:.2f}); posterior > ceiling: "
              f"{s['n_posterior_above_ceiling']}; P>ceiling>0.5: {s['n_P_above_ceiling_gt_0.5']}; P>2xCol0>0.9: {s['n_P_above_2x_col0_gt_0.9']}")
        for r in s["top_by_posterior"][:12]:
            print(f"  {r['name']:14s} runs={r['runs']} raw={r['raw_mean_rescue']:.2f} post={r['posterior_rescue']:.3f}±{r['posterior_sd']:.3f} "
                  f"P>0.4={r['P_above_ceiling']:.2f} P>2xCol0={r['P_above_2x_col0']:.2f} gain={r['mean_gain']:.1f} loss={r['mean_loss']:.2f} "
                  f"byrun={ {k: round(v, 2) for k, v in r['rescue_by_run'].items()} }")
        R_ = s["replication"]
        print(f"rescreened n={R_['n_rescreened_with_3_runs']} discovered(top25% run1)={R_['n_discovered_top25_run1']} permutation p(best)={R_['permutation_p_best']:.4f}")
        for d in R_["discovered"]:
            print(f"   {d['name']:12s} BR1={d['BR1']:.2f}(q{d['q1']:.2f}) BR2={d['BR2']:.2f}(q{d['q2']:.2f}) BR3={d['BR3']:.2f}(q{d['q3']:.2f}) p={d['p_replication']:.4f} bonf={d['p_bonferroni']:.3f}")
        print("   vs Col-0:", {k: {r: (round(x['diff'],2), round(x['lo'],2), round(x['hi'],2)) for r, x in v.items()} for k, v in R_["vs_col0_by_run"].items()})
        print("   loss by run:", {k: {r: round(x,2) for r, x in v.items()} for k, v in R_["loss_by_run"].items()})
        print("   gain by run:", {k: {r: round(x,1) for r, x in v.items()} for k, v in R_["gain_by_run"].items()})
        print("   true var (cov runs 2,3):", {k: (round(v,4) if isinstance(v,float) else [round(x,4) for x in v]) for k, v in s["true_variance_from_runs23"].items()})
        print("raw top10 run1:", [(x['name'], round(x['run1'], 2), round(x['posterior'], 2)) for x in s["raw_top10_run1"]])
