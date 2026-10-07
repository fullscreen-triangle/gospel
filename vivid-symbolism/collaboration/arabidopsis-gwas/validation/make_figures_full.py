"""Panel 7: what the full per-plant workbook (NS+DS.xlsx) shows. Reads results/F*.json."""
import json

import matplotlib.pyplot as plt
import numpy as np

import make_figures as mf  # shared style, colours and helpers
from common import FIGURES, RESULTS


def R(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


F2, F3, F4, F6 = R("F2_noise"), R("F3_retest"), R("F4_replicates"), R("F6_drought")
C, INK, INK2, MUTED, AXIS, RED = mf.C, mf.INK, mf.INK2, mf.MUTED, mf.AXIS, mf.RED


def panel7():
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.9))
    fig.subplots_adjust(hspace=0.5, wspace=0.34)

    # A: Col-0 gain by independent run
    ax = axs[0, 0]; mf.tag(ax, "A")
    for j, (O, col) in enumerate([("Shoot", C["gain"]), ("Root", MUTED)]):
        g = F2[O]["col0_gain_by_replicate"]
        for i, br in enumerate(["BR1", "BR2", "BR3"]):
            d = g[br]
            x = i + (j - 0.5) * 0.22
            ax.errorbar(x, d["gain"], yerr=[[d["gain"] - d["lo"]], [d["hi"] - d["gain"]]], fmt="o", ms=5,
                        color=col, capsize=0, lw=1.4, label=O.lower() if i == 0 else None)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["run 1", "run 2", "run 3"])
    ax.set_ylabel("Col-0 gain W − M_D (mg, 95% CI)"); ax.legend(loc="upper right")
    ax.set_title("the reference line's rescue changes by run", color=INK, loc="left")

    # B: replicate reproducibility of components vs constructions
    ax = axs[0, 1]; mf.tag(ax, "B")
    keys = ["loss", "logMD", "logW", "increase", "gain", "rescue", "canonical"]
    labels = ["mock drought\nloss", "log M_D", "log W", "% increase", "gain", "rescue", "canonical"]
    cols = [MUTED, MUTED, MUTED, C["increase"], C["gain"], C["rescue"], C["canonical"]]
    for j, O in enumerate(["Shoot", "Root"]):
        f = F4[O]
        vals = {**f["test_retest_spearman_BR2_BR3"], **f["component_test_retest_spearman"]}
        for i, k in enumerate(keys):
            y = i + (j - 0.5) * 0.3
            lo, hi = f["test_retest_ci95"][k]
            ax.plot([lo, hi], [y, y], color=cols[i], lw=1.4, alpha=[1, 0.5][j])
            ax.plot(vals[k], y, "o" if j == 0 else "s", color=cols[i], ms=5, alpha=[1, 0.6][j])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(labels, fontsize=6.5); ax.invert_yaxis()
    ax.set_xlim(-0.3, 1); ax.set_xlabel("Spearman ρ, run 2 vs run 3 (48 accessions, 95% CI)")
    ax.plot([], [], "o", color=INK2, label="shoot"); ax.plot([], [], "s", color=INK2, alpha=0.6, label="root")
    ax.legend(loc="lower right", fontsize=6.3)
    ax.set_title("drought sensitivity replicates; rescue does not", color=INK, loc="left")

    # C: regression to the mean of the retested accessions
    ax = axs[1, 0]; mf.tag(ax, "C")
    rows = [r for r in F3["organs"]["Shoot"]["rows"] if "BR1" in r and "retest_pooled" in r]
    g1 = np.array([r["BR1"]["gain"] for r in rows]); g2 = np.array([r["retest_pooled"]["gain"] for r in rows])
    z1 = np.array([r["BR1"]["lo"] <= 0 for r in rows]); z2 = np.array([r["retest_pooled"]["lo"] <= 0 for r in rows])
    ax.scatter(g1[~z1 & ~z2], g2[~z1 & ~z2], s=14, color=C["gain"], lw=0, label="rescued in both")
    ax.scatter(g1[z1], g2[z1], s=26, facecolor="white", edgecolor=RED, lw=1.2, label="CI reaches 0 in run 1")
    ax.scatter(g1[z2], g2[z2], s=26, marker="s", facecolor="white", edgecolor=INK, lw=1.0, label="CI reaches 0 in retest")
    lim = [min(g1.min(), g2.min()) - 1, max(g1.max(), g2.max()) + 1]
    ax.plot(lim, lim, color=AXIS, ls="--", lw=0.8)
    m = F3["organs"]["Shoot"]["mean_gain_all_accessions_BR1"]
    ax.axhline(m, color=MUTED, lw=0.8, ls=":"); ax.text(lim[1], m + 0.3, "all-accession mean", ha="right", fontsize=6.3, color=INK2)
    ax.set_xlabel("shoot gain, run 1 (mg)"); ax.set_ylabel("shoot gain, runs 2–3 pooled (mg)")
    ax.legend(loc="upper left", fontsize=6.3)
    ax.set_title("no weak rescuer stays weak", color=INK, loc="left")

    # D: drought promotion vs non-stress promotion
    ax = axs[1, 1]; mf.tag(ax, "D")
    A = F6["Shoot"]["accessions"]
    pn = np.array([a["log_promotion_nonstress"] for a in A]); pdr = np.array([a["log_promotion_drought"] for a in A])
    ax.scatter(pn, pdr, s=9, color=C["canonical"], lw=0, alpha=0.8)
    ax.axvline(0, color=INK, lw=0.8); ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("log(W_N / M_N): effect without drought"); ax.set_ylabel("log(W / M_D): effect under drought")
    s = F6["Shoot"]
    ax.text(0.98, 0.04, f"ρ = {s['spearman']['promotion_nonstress~promotion_drought']:.2f}\n"
            f"non-stress: {100 * s['log_promotion_nonstress']['frac_significantly_negative']:.0f}% inhibited, "
            f"{100 * s['log_promotion_nonstress']['frac_significantly_positive']:.0f}% promoted",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, color=INK2)
    ax.set_title("shoot: the benefit is drought-specific", color=INK, loc="left")
    mf.save(fig, "panel_7_full_workbook")


if __name__ == "__main__":
    panel7()
