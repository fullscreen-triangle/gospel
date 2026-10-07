"""Panel 8: the super-rescuer and what the genome says. Reads plants_all.csv and results/S4, S5 json."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import make_figures as mf
from common import CACHE, RESULTS

C, INK, INK2, MUTED, AXIS, RED = mf.C, mf.INK, mf.INK2, mf.MUTED, mf.AXIS, mf.RED
S4 = json.loads((RESULTS / "S4_gwas.json").read_text(encoding="utf-8"))
S5 = json.loads((RESULTS / "S5_blind_polygenic.json").read_text(encoding="utf-8"))
NAMED = {9836: ("Cod-0", RED), 6909: ("Col-0", INK), 9813: ("BI-4", C["gain"])}


def folds():
    d = pd.read_csv(CACHE / "plants_all.csv")
    m = d.groupby(["gid", "br", "treatment"]).shoot.mean().unstack("treatment")
    f = (m["WCS417"] / m["Mock"]).unstack("br")
    return f.dropna()                                   # accessions grown in all three runs


def panel8():
    fig, axs = plt.subplots(1, 3, figsize=(10.8, 3.4))
    fig.subplots_adjust(wspace=0.42)

    # A: WCS417 / mock shoot under drought, every run, rescreened accessions
    ax = axs[0]; mf.tag(ax, "A")
    F = folds()
    for g, row in F.iterrows():
        if g not in NAMED:
            ax.plot([0, 1, 2], row.values, color=AXIS, lw=0.6, alpha=0.7, zorder=1)
    for g, (name, col) in NAMED.items():
        if g in F.index:
            ax.plot([0, 1, 2], F.loc[g].values, "-o", color=col, lw=1.8, ms=4, zorder=3, label=name)
    ax.axhline(1, color=INK, lw=0.8)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["run 1", "run 2", "run 3"])
    ax.set_ylabel("shoot FW, WCS417 / mock (drought)")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_title(f"steadiest responder, not the largest (n = {len(F)})", color=INK, loc="left")

    # B: Manhattan, root rescue (SNPs with p < 1e-4 are stored)
    ax = axs[1]; mf.tag(ax, "B")
    H = pd.read_csv(CACHE / "S4_gwas_p1e-4.csv")
    H = H[H.trait == "root_rescue"]
    off, ticks = 0, []
    for c in range(1, 6):
        h = H[H.chrom.astype(str) == str(c)]
        span = {1: 30_427_671, 2: 19_698_289, 3: 23_459_830, 4: 18_585_056, 5: 26_975_502}[c]
        ax.scatter(h.pos + off, -np.log10(h.p), s=5, lw=0, color=C["canonical"] if c % 2 else MUTED)
        ticks.append(off + span / 2)
        off += span
    ax.axhline(S4["minus_log10_bonferroni"], color=RED, lw=0.9, ls="--")
    ax.set_xticks(ticks); ax.set_xticklabels([f"chr{c}" for c in range(1, 6)])
    ax.set_ylim(4, S4["minus_log10_bonferroni"] + 0.6)
    ax.set_ylabel("−log10 p (root rescue)")
    rr = S5["held_out_rescreen"]["root_rescue"]["sign_concordance"]
    ax.text(0.98, 0.96, f"{S4['n_snps_tested'] / 1e6:.2f} M SNPs, λ = {S4['lambda_gc']['root_rescue']:.2f}\n"
            f"top-50 peaks, same sign in rescreen: {rr['same']}/{rr['of']} (p = {rr['p_binomial']:.3f})",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color=INK2)
    ax.set_title("root rescue: diffuse, but it replicates", color=INK, loc="left")

    # C: Cod-0's count of rescue-raising alleles at the top-10 shoot-rescue peaks, in vs held out
    ax = axs[2]; mf.tag(ax, "C")
    inn = [p for p in S4["peaks"] if p["trait"] == "shoot_rescue"][:10]
    n_in = sum(p["cod0_carries_raising"] for p in inn)
    e_in = S4["cod0_at_top_peaks"]["shoot_rescue"]["expected"]
    out = S5["traits"]["shoot_rescue"][0]
    xs = [0, 1]
    ax.bar(xs, [n_in, out["cod0_count"]], color=[MUTED, RED], width=0.55)
    ax.scatter(xs, [e_in, out["expected_count"]], marker="_", s=600, color=INK, zorder=3, label="expected from allele frequency")
    ax.set_xticks(xs); ax.set_xticklabels(["Cod-0 in the GWAS", "Cod-0 held out"])
    ax.set_ylim(0, 11.5); ax.set_ylabel("rescue-raising alleles, top 10 shoot peaks")
    ax.text(1, out["cod0_count"] + 0.3, f"p = {out['p_cod0_by_frequency']:.2f}", ha="center", fontsize=6.5, color=INK2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=6.3, frameon=False)
    ax.set_title("not a stack of common alleles", color=INK, loc="left")
    mf.save(fig, "panel_8_superrescuer")


if __name__ == "__main__":
    panel8()
