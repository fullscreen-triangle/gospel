"""Build the six manuscript panels from results/*.json (and the cached workbook
extract for the two plots that show per-plant or per-accession raw values).

Colour is fixed by role, never by rank:
  gain = blue, increase = orange, rescue = aqua, canonical = violet.
Organs are separated by facets, never by colour.
"""
from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch

from common import FIGURES, RESULTS, load

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#fcfcfb"
C = {"gain": "#2a78d6", "increase": "#eb6834", "rescue": "#1baf7a", "canonical": "#4a3aa7"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RED, GRAY_MID = "#e34948", "#f0efec"
LBL = {"gain": "gain", "increase": "% increase", "rescue": "rescue", "canonical": "canonical"}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 7, "lines.linewidth": 1.6,
    "savefig.dpi": 220, "savefig.bbox": "tight",
})


def R(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def tag(ax, letter):
    ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
            color=INK, va="bottom", ha="left")


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12 = (R(n) for n in [
    "E1_audit", "E2_heterogeneity", "E3_dependence", "E4_canonical", "E5_four_column",
    "E6_calibration", "E7_batch", "E8_candidates", "E9_ratio", "E10_classes",
    "E11_geography", "E12_design"])
ACC = R("accessions")["accessions"]
REF, M2 = "M3_plant_plus_batch", "M2_plant_all_cells"
D = load()


# ======================================================================== panel 1
def panel1():
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.8))
    fig.subplots_adjust(hspace=0.45, wspace=0.32)

    # A: design schematic
    ax = axs[0, 0]; tag(ax, "A")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    cells = [("mock\nnon-stress", "M_N", 1.0, 5.4, False), ("WCS417\nnon-stress", "W_N", 5.4, 5.4, False),
             ("mock\ndrought", "M_D", 1.0, 0.8, False), ("WCS417\ndrought", "W", 5.4, 0.8, True)]
    for label, sym, x, y, raw in cells:
        fc = C["gain"] if raw else "white"
        ec = C["gain"] if raw else AXIS
        ax.add_patch(FancyBboxPatch((x, y), 3.8, 3.6, boxstyle="round,pad=0.02,rounding_size=0.25",
                                    fc=fc, ec=ec, lw=1.2, alpha=0.14 if raw else 1))
        ax.text(x + 1.9, y + 2.35, label, ha="center", va="center", fontsize=7.5, color=INK)
        note = "1,841 plants\n(7 per accession)" if raw else ("not recoverable" if sym == "W_N" else "mean only\n(recovered)")
        ax.text(x + 1.9, y + 0.85, f"${sym.replace('_', '_{')}{'}' if '_' in sym else ''}$\n{note}",
                ha="center", va="center", fontsize=6.5, color=INK2)
    ax.set_title("design: one run, raw data for one cell", color=INK, loc="left")

    # B: recovered cell means, sorted by M_N
    ax = axs[0, 1]; tag(ax, "B")
    mn = np.array([a["shoot_MN"] for a in ACC]); md = np.array([a["shoot_MD"] for a in ACC])
    w = np.array([a["shoot_W"] for a in ACC])
    o = np.argsort(mn); x = np.arange(len(o))
    ax.scatter(x, mn[o], s=5, color=MUTED, label="$M_N$ mock, non-stress", lw=0)
    ax.scatter(x, w[o], s=5, color=C["gain"], label="$W$ WCS417, drought", lw=0)
    ax.scatter(x, md[o], s=5, color=INK, label="$M_D$ mock, drought", lw=0)
    ax.set_yscale("log"); ax.set_xlabel("accession (sorted by $M_N$)"); ax.set_ylabel("shoot fresh weight (mg)")
    ax.legend(loc="upper left", markerscale=2.2, handletextpad=0.2, ncol=1)
    ax.set_title(f"drought removes {100*E1['identities']['Shoot']['median_fractional_loss']:.0f}% of mock shoot mass",
                 color=INK, loc="left")
    ax.set_ylim(1.5, 700)

    # C: identities hold to rounding error
    ax = axs[1, 0]; tag(ax, "C")
    tr = D.traits
    c = D.cells
    for name, col, fn, y0 in [("rescue", "AVG_percent_Rescue_Shoot",
                               lambda W, MD, MN: (W - MD) / (MN - MD), 1),
                              ("% increase", "AVG_percent_increase_in_SFW_",
                               lambda W, MD, MN: W / MD - 1, 0)]:
        r = (fn(c["Shoot_W"], c["Shoot_MD"], c["Shoot_MN"]) - tr[col]).abs()
        r = r.dropna()
        rep = r.index.isin(D.repeated)
        lr = np.log10(np.maximum(r.to_numpy(), 1e-16))
        jit = np.random.default_rng(1).uniform(-0.18, 0.18, len(lr))
        ax.scatter(lr[~rep], y0 + jit[~rep], s=6, color=C["rescue"] if y0 else C["increase"], lw=0)
        ax.scatter(lr[rep], y0 + jit[rep], s=22, facecolor="white", edgecolor=RED, lw=1.2,
                   label="lines grown in >1 block" if y0 else None)
    ax.axvline(-6, color=MUTED, lw=0.8, ls="--")
    ax.text(-6.1, -0.45, "tolerance $10^{-6}$", ha="right", fontsize=6.5, color=INK2)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["$W/M_D-1$", "$(W-M_D)/(M_N-M_D)$"])
    ax.set_ylim(-0.6, 1.8); ax.set_xlabel("log$_{10}$ |workbook value − formula|")
    ax.legend(loc="upper center")
    ax.set_title("derived traits = exact formulas", color=INK, loc="left")

    # D: the rescuer label is a threshold on a positive gain
    ax = axs[1, 1]; tag(ax, "D")
    g = (c["Shoot_W"] - c["Shoot_MD"]).reindex(tr.index)
    lab = tr["Shoot_Rescuer"]
    bins = np.linspace(0, 27, 37)
    ax.hist(g[lab == 0], bins=bins, color=MUTED, label=f"labelled non-rescuer (n={int((lab == 0).sum())})",
            edgecolor="white", lw=0.8)
    ax.hist(g[lab == 1], bins=bins, color=C["gain"], alpha=0.85,
            label=f"labelled rescuer (n={int((lab == 1).sum())})", edgecolor="white", lw=0.8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("gain under drought, $W - M_D$ (mg shoot)"); ax.set_ylabel("accessions")
    ax.legend(loc="upper right", fontsize=6.3)
    ax.set_title("all gains > 0; label = cut at 6–8 mg", color=INK, loc="left")
    save(fig, "panel_1_audit")


# ======================================================================== panel 2
def panel2():
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.8))
    fig.subplots_adjust(hspace=0.48, wspace=0.32)

    # A: Col-0 blocks
    ax = axs[0, 0]; tag(ax, "A")
    col = D.raw[D.raw.ID == 6909].copy()
    col["block"] = col["sheet_row"].diff().fillna(1).ne(1).cumsum()
    accm = np.array([a["shoot_W"] for a in ACC])
    lo, hi = np.percentile(accm, [5, 95])
    ax.axhspan(lo, hi, color=SEQ[0], alpha=0.6, lw=0, label="5–95% of accession means")
    k = 0
    for b, grp in col.groupby("block"):
        v = grp["shoot"].dropna().to_numpy()
        if len(v) < 3:
            continue
        ax.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(v)), v,
                   s=7, color=MUTED, lw=0)
        ax.plot([k - 0.3, k + 0.3], [v.mean()] * 2, color=INK, lw=2)
        k += 1
    ax.set_xlabel("Col-0 block, in sowing order"); ax.set_ylabel("shoot FW, WCS417 drought (mg)")
    s = E7["Shoot"]
    ax.set_title(f"batch: ICC = {s['ICC1']:.2f}, F-test p = {s['p']:.1e}", color=INK, loc="left")
    ax.legend(loc="upper right")

    # B: caterpillar of gain with 95% CI (M3)
    ax = axs[0, 1]; tag(ax, "B")
    d = E2["organs"]["Shoot"][REF]["gain"]["detectability"]["per_accession"]
    g, lo_, hi_ = map(np.array, (d["gain"], d["lo"], d["hi"]))
    o = np.argsort(g); x = np.arange(len(o))
    zero = lo_[o] <= 0
    ax.vlines(x[~zero], lo_[o][~zero], hi_[o][~zero], color=SEQ[2], lw=0.7)
    ax.vlines(x[zero], lo_[o][zero], hi_[o][zero], color=RED, lw=1.0)
    ax.scatter(x, g[o], s=4, color=INK, lw=0, zorder=3)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("accession (sorted by gain)"); ax.set_ylabel("gain, mg shoot (95% interval)")
    n0 = int(zero.sum())
    ax.set_title(f"{n0} of {len(g)} intervals reach zero (red)", color=INK, loc="left")

    # C: reliability by construction and noise model
    ax = axs[1, 0]; tag(ax, "C")
    models = [(M2, "plant noise only"), (REF, "+ batch"), ("M4_M3_mock_noise_x1.5", "+ batch, mock ×1.5")]
    names = ["gain", "increase", "rescue"]
    width = 0.26
    for j, organ in enumerate(["Shoot", "Root"]):
        for i, n in enumerate(names):
            for m, (mk, ml) in enumerate(models):
                v = E2["organs"][organ][mk][n]["reliability"]
                xpos = j * 3.6 + i * 1.1 + (m - 1) * width
                ax.bar(xpos, v, width * 0.92, color=C[n], alpha=[1.0, 0.6, 0.32][m], lw=0)
    ax.set_xticks([i * 1.1 + j * 3.6 for j in range(2) for i in range(3)])
    ax.set_xticklabels(["gain", "% inc.", "rescue"] * 2, rotation=0, fontsize=6.5)
    ax.text(1.1, -0.2, "shoot", ha="center", transform=ax.get_xaxis_transform(), color=INK2)
    ax.text(4.7, -0.2, "root", ha="center", transform=ax.get_xaxis_transform(), color=INK2)
    ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("reliability  τ² / (τ² + se²)")
    ax.legend(handles=[Patch(color=INK2, alpha=[1.0, 0.6, 0.32][m], label=ml) for m, (_, ml) in enumerate(models)],
              loc="upper center", ncol=3, fontsize=6, columnspacing=0.8, handlelength=1.2)
    ax.set_title("rescue is the noisiest of the three constructions", color=INK, loc="left")

    # D: design, reliability vs number of independent runs
    ax = axs[1, 1]; tag(ax, "D")
    for j, organ in enumerate(["Shoot", "Root"]):
        tab = E12["organs"][organ]["table"]
        for n, ls in [(7, "-"), (28, ":")]:
            rs = E12["grid_r"]
            ys = [tab[f"{n}x{r}"]["reliability"] for r in rs]
            ax.plot(rs, ys, ls, color=[C["canonical"], MUTED][j], marker="o", ms=3.5,
                    label=f"{organ.lower()}, {n} plants/cell")
    ax.set_xlabel("independent runs r"); ax.set_ylabel("reliability of canonical rescue")
    ax.set_ylim(0, 1); ax.legend(loc="lower right")
    ax.set_title("more runs, not more plants", color=INK, loc="left")
    save(fig, "panel_2_noise")


# ======================================================================== panel 3
def panel3():
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.9))
    fig.subplots_adjust(hspace=0.5, wspace=0.38)

    # A: Spearman heatmap (shoot)
    ax = axs[0, 0]; tag(ax, "A")
    rho = E3["organs"]["Shoot"]["spearman"]
    keys = ["gain", "increase", "rescue", "W_size", "MN_vigour", "loss_mock_only"]
    lab = ["gain", "% increase", "rescue", "W (size)", "$M_N$ (vigour)", "mock loss"]
    M = np.array([[rho[a][b] for b in keys] for a in keys])
    cmap = mpl.colors.LinearSegmentedColormap.from_list("div", ["#e34948", GRAY_MID, "#2a78d6"])
    ax.imshow(M, cmap=cmap, vmin=-1, vmax=1)
    ax.grid(False)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.3,
                    color=INK if abs(M[i, j]) < 0.7 else "white")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(lab, rotation=40, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(lab, fontsize=6.5)
    ax.set_title("shoot, Spearman ρ across 235 accessions", color=INK, loc="left")

    # B: rank-rank increase vs rescue
    ax = axs[0, 1]; tag(ax, "B")
    rs = E3["organs"]["Shoot"]["rank_scatter"]
    loss = np.array([1 - a["shoot_MD"] / a["shoot_MN"] for a in ACC])
    sc = ax.scatter(rs["rescue"], rs["increase"], c=loss, cmap=mpl.colors.ListedColormap(SEQ), s=9, lw=0)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    ax.axvline(0.8, color=AXIS, lw=0.8); ax.axhline(0.8, color=AXIS, lw=0.8)
    ax.set_xlabel("rank by rescue (quantile)"); ax.set_ylabel("rank by % increase (quantile)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02); cb.set_label("mock drought loss", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("% increase ranks drought-sensitive lines high", color=INK, loc="left")

    # C: top-20% flip rate: construction pairs vs noise
    ax = axs[1, 0]; tag(ax, "C")
    pairs = ["gain~increase", "gain~rescue", "increase~rescue"]
    for j, organ in enumerate(["Shoot", "Root"]):
        oo = E3["organs"][organ]
        for i, p in enumerate(pairs):
            x = j * 4 + i
            fr = oo["construction_agreement"]["0.2"]["pairs"][p]["flip_rate"]
            a, b = p.split("~")
            n1 = max(oo["noise_agreement"][M2][a]["flip_rate_top20_p95"], oo["noise_agreement"][M2][b]["flip_rate_top20_p95"])
            n3 = max(oo["noise_agreement"][REF][a]["flip_rate_top20_p95"], oo["noise_agreement"][REF][b]["flip_rate_top20_p95"])
            ax.bar(x, fr, 0.62, color=INK2, lw=0)
            ax.plot([x - 0.36, x + 0.36], [n1, n1], color=C["gain"], lw=2)
            ax.plot([x - 0.36, x + 0.36], [n3, n3], color=RED, lw=2)
    ax.set_xticks([0, 1, 2, 4, 5, 6])
    ax.set_xticklabels(["g~i", "g~r", "i~r"] * 2)
    ax.text(1, -0.2, "shoot", ha="center", transform=ax.get_xaxis_transform(), color=INK2)
    ax.text(5, -0.2, "root", ha="center", transform=ax.get_xaxis_transform(), color=INK2)
    ax.set_ylim(0, 0.95); ax.set_ylabel("top-20% membership changed")
    ax.legend(handles=[Patch(color=INK2, label="between constructions"),
                       mpl.lines.Line2D([], [], color=C["gain"], lw=2, label="noise 95th pct, plant only"),
                       mpl.lines.Line2D([], [], color=RED, lw=2, label="noise 95th pct, + batch")],
              loc="upper left", fontsize=6.3, ncol=1)
    ax.set_title("construction vs noise, top 20% of 235", color=INK, loc="left")

    # D: opposite certified orderings
    ax = axs[1, 1]; tag(ax, "D")
    pairs = ["gain~increase", "gain~rescue", "increase~rescue", "increase~canonical"]
    short = ["g~i", "g~r", "i~r", "i~c"]
    for j, organ in enumerate(["Shoot", "Root"]):
        for i, p in enumerate(pairs):
            for m, mk in enumerate([M2, REF]):
                v = E5["organs"][organ]["models"][mk]["agreement_between_constructions"][p]["opposite_certified_order"]
                x = j * 5 + i + (m - 0.5) * 0.36
                ax.bar(x, max(v, 0.8), 0.34, color=[C["gain"], RED][m], lw=0)
                ax.text(x, max(v, 0.8) * 1.15, str(v), ha="center", fontsize=5.8, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(0.7, 3000)
    ax.set_xticks([j * 5 + i for j in range(2) for i in range(4)]); ax.set_xticklabels(short * 2)
    ax.text(1.5, -0.2, "shoot", ha="center", transform=ax.get_xaxis_transform(), color=INK2)
    ax.text(6.5, -0.2, "root", ha="center", transform=ax.get_xaxis_transform(), color=INK2)
    ax.legend(handles=[Patch(color=C["gain"], label="plant noise only"), Patch(color=RED, label="+ batch")],
              loc="upper left", fontsize=6.3); ax.set_ylabel("pairs certified in opposite order")
    ax.set_title("decisive contradictions between constructions", color=INK, loc="left")
    save(fig, "panel_3_dependence")


# ======================================================================== panel 4
def panel4():
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.8))
    fig.subplots_adjust(hspace=0.48, wspace=0.32)
    lam = np.array(E4["lambda_grid"])

    # A: nuisance loading along lambda (shoot)
    ax = axs[0, 0]; tag(ax, "A")
    s = E4["organs"]["Shoot"]
    lo, hi = s["lambda_ci95_corrected"]
    ax.axvspan(lo, hi, color=C["canonical"], alpha=0.12, lw=0, label="95% interval of $\\hat\\lambda$")
    ax.axvline(s["lambda_hat_corrected"], color=C["canonical"], lw=1.2)
    ax.plot(lam, s["loading"]["rho_vs_loss"], color=INK, label="ρ with mock drought loss")
    ax.plot(lam, s["loading"]["rho_vs_MN"], color=MUTED, ls="--", label="ρ with $M_N$ (vigour)")
    ax.axhline(0, color=AXIS, lw=0.8)
    for x0, n in [(0, "rescue"), (1, "increase")]:
        ax.scatter([x0], [np.interp(x0, lam, s["loading"]["rho_vs_loss"])], s=40, color=C[n], zorder=4)
        ax.text(x0, 0.95, LBL[n], ha="center", fontsize=7, color=INK2)
    ax.set_xlabel("λ  in  $(W-M_D)\\,/\\,M_D^{\\lambda}(M_N-M_D)^{1-\\lambda}$")
    ax.set_ylabel("Spearman ρ"); ax.set_ylim(-0.6, 1.05); ax.legend(loc="lower right", fontsize=6.3)
    ax.set_title("shoot: nuisance loading vs λ", color=INK, loc="left")

    # B: simulation recovery
    ax = axs[0, 1]; tag(ax, "B")
    for j, organ in enumerate(["Shoot", "Root"]):
        rec = E4["organs"][organ]["recovery"]
        t = np.array([float(k) for k in rec]); m = np.array([rec[k]["mean"] for k in rec])
        sd = np.array([rec[k]["sd"] for k in rec])
        ax.errorbar(t + (j - 0.5) * 0.03, m, yerr=sd, fmt="o", ms=3.5, color=[C["canonical"], MUTED][j],
                    lw=1, capsize=0, label=organ.lower())
    ax.plot([-0.8, 1.05], [-0.8, 1.05], color=AXIS, lw=0.8, ls="--")
    ax.set_xlabel("true λ (simulated, batch-inclusive noise)"); ax.set_ylabel("estimate $b_1$ (mean ± sd)")
    ax.legend(loc="upper left")
    ax.set_title("the estimator is biased by ≈ −0.1; we invert it", color=INK, loc="left")

    # C: bootstrap of b1 and corrected lambda
    ax = axs[1, 0]; tag(ax, "C")
    for j, organ in enumerate(["Shoot", "Root"]):
        oo = E4["organs"][organ]
        b = np.array(oo["bootstrap"]["b1_samples"])
        ax.hist(b, bins=30, color=[C["canonical"], MUTED][j], alpha=0.55, edgecolor="white", lw=0.5,
                label=f"{organ.lower()}: $\\hat\\lambda$ = {oo['lambda_hat_corrected']:.2f} "
                      f"[{oo['lambda_ci95_corrected'][0]:.2f}, {oo['lambda_ci95_corrected'][1]:.2f}]")
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.55); ax.set_xlim(-0.8, 1.15)
    for x0, n in [(0, "rescue"), (1, "increase")]:
        ax.axvline(x0, color=C[n], lw=1.4)
        ax.text(x0 + 0.03, top * 0.55, LBL[n], color=INK2, fontsize=7, rotation=90, va="bottom")
    ax.set_xlabel("$b_1$ (uncorrected), bootstrap over accessions × noise"); ax.set_ylabel("draws")
    ax.legend(loc="upper left", fontsize=6.3)
    ax.set_title("λ = 1 (% increase) is excluded in both organs", color=INK, loc="left")

    # D: agreement within vs across
    ax = axs[1, 1]; tag(ax, "D")
    for j, organ in enumerate(["Shoot", "Root"]):
        oo = E4["organs"][organ]; e3 = E3["organs"][organ]
        across = e3["construction_agreement"]["0.2"]["pairs"]["increase~rescue"]["flip_rate"]
        within = oo["within_admissible"]["flip_rate_top20"]
        noise = e3["noise_agreement"][REF]["rescue"]["flip_rate_top20_p95"]
        x = j * 3
        ax.bar(x, across, 0.8, color=C["increase"], lw=0)
        ax.bar(x + 1, within, 0.8, color=C["canonical"], lw=0)
        ax.plot([x - 0.5, x + 1.5], [noise, noise], color=RED, lw=2)
        ax.text(x, across + 0.02, f"{across:.2f}", ha="center", fontsize=7, color=INK2)
        ax.text(x + 1, within + 0.02, f"{within:.2f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks([0, 1, 3, 4]); ax.set_xticklabels(["increase\n~rescue", "within\nλ̂ interval"] * 2, fontsize=6.5)
    ax.text(0.5, -0.28, "shoot", ha="center", transform=ax.get_xaxis_transform(), color=INK2)
    ax.text(3.5, -0.28, "root", ha="center", transform=ax.get_xaxis_transform(), color=INK2)
    ax.plot([], [], color=RED, lw=2, label="noise 95th pct (+ batch)")
    ax.set_ylim(0, 0.85); ax.set_ylabel("top-20% membership changed"); ax.legend(loc="upper right", fontsize=6.3)
    ax.set_title("inside the data-admissible range, disagreement ≤ noise", color=INK, loc="left")
    save(fig, "panel_4_canonical")


# ======================================================================== panel 5
def panel5():
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.3))
    fig.subplots_adjust(hspace=0.72, wspace=0.34)

    # A: calibration curves
    ax = axs[0, 0]; tag(ax, "A")
    lv = E6["s_over_delta_levels"]
    for k, sl in enumerate(lv):
        rows = E6["curves"][str(sl)]
        t = [r["true_diff_over_delta"] for r in rows]
        col = SEQ[2 + k] if k < 4 else RED
        lab = f"s/δ = {sl:g}" + (" (this screen)" if k == len(lv) - 1 else "")
        ax.plot(t, [r["P_C"] for r in rows], color=col, label=lab)
        ax.plot(t, [r["P_D"] for r in rows], color=col, ls="--")
    ax.axvline(1, color=AXIS, lw=0.8); ax.axhline(0.05, color=AXIS, lw=0.8, ls=":")
    ax.set_xlabel("true difference / margin δ"); ax.set_ylabel("probability of verdict")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=6, columnspacing=1.0,
              title="solid: correspond · dashed: diverge", title_fontsize=6.3)
    ax.set_title("verdict error rates ≤ 5% at all noise levels", color=INK, loc="left")

    # B: min certifiable margin CDF
    ax = axs[0, 1]; tag(ax, "B")
    for organ, ls in [("Shoot", "-"), ("Root", "--")]:
        for mk, col in [(M2, C["gain"]), (REF, RED)]:
            q = E5["organs"][organ]["models"][mk]["response"]["canonical"]["min_certifiable_margin_sd_quantiles"]
            ps = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95]
            ax.plot(q, ps, ls, color=col, marker="o", ms=3,
                    label=f"{organ.lower()}, {'plant noise' if mk == M2 else '+ batch'}")
    ax.axvline(1, color=AXIS, lw=0.8)
    ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.set_xlabel("smallest certifiable margin (× between-accession SD)")
    ax.set_ylabel("fraction of 27,495 pairs"); ax.legend(loc="lower right", fontsize=6.3)
    ax.set_title("equivalence needs margins wider than the population", color=INK, loc="left")

    # C: four-column table (plant-noise model, shoot)
    ax = axs[1, 0]; tag(ax, "C")
    t = E5["organs"]["Shoot"]["models"][M2]["four_column_table"]
    order = ["C", "U", "D"]; names = {"C": "correspond", "U": "decline", "D": "diverge"}
    M = np.array([[t[b][r] for r in order] for b in order], float)
    ax.imshow(np.log10(M + 1), cmap=mpl.colors.ListedColormap(SEQ[:6]), vmin=0, vmax=np.log10(M.max() + 1))
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(M[i, j]):,}", ha="center", va="center", fontsize=7.5,
                    color="white" if np.log10(M[i, j] + 1) > 3.2 else INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([names[k] for k in order])
    ax.set_yticks(range(3)); ax.set_yticklabels([names[k] for k in order])
    ax.set_xlabel("response columns (canonical, margin 0.5 SD)")
    ax.set_ylabel("baseline columns (mock, margin 30%)")
    ax.set_title("shoot, plant noise: four-column table", color=INK, loc="left")

    # D: resolution tiers
    ax = axs[1, 1]; tag(ax, "D")
    tiers = np.array(E10["Shoot"][REF]["labels"]["canonical"])
    v = np.array([a["shoot_canonical"] for a in ACC]); se = np.array([a["shoot_canonical_se"] for a in ACC])
    o = np.argsort(v); x = np.arange(len(o))
    nt = tiers.max()
    cols = [SEQ[1 + min(5, int(round(5 * (k - 1) / max(nt - 1, 1))))] for k in range(1, nt + 1)]
    ax.vlines(x, (v - 1.96 * se)[o], (v + 1.96 * se)[o], color=GRID, lw=0.8)
    ax.scatter(x, v[o], c=[cols[t - 1] for t in tiers[o]], s=9, lw=0, zorder=3)
    for k in range(1, nt + 1):
        ax.scatter([], [], color=cols[k - 1], s=18, label=f"tier {k} (n={int((tiers == k).sum())})")
    ax.set_xlabel("accession (sorted by canonical rescue)"); ax.set_ylabel("canonical rescue, shoot")
    ax.legend(loc="upper left", fontsize=6.3)
    d2 = E10["Shoot"][M2]["max_depth"]["canonical"]
    ax.set_title(f"the screen resolves {nt} tiers (+ batch); {d2} with plant noise", color=INK, loc="left")
    save(fig, "panel_5_four_column")


# ======================================================================== panel 6
def panel6():
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.8))
    fig.subplots_adjust(hspace=0.5, wspace=0.34)

    # A: candidates by family
    ax = axs[0, 0]; tag(ax, "A")
    fams = ["rescue", "increase", "gain", "inoculated_size", "mock_only_loss"]
    labs = ["rescue", "% increase", "gain", "size under\nWCS417", "mock-only\nloss"]
    rows = E8["rows"]
    counts = [E8["genes_by_family"].get(f, 0) for f in fams]
    load_ = []
    for f in fams:
        v = [r["rho_metric_vs_mock_loss"] for r in rows if r["family"] == f and r["rho_metric_vs_mock_loss"] is not None]
        load_.append(np.median(np.abs(v)) if v else np.nan)
    cmap = mpl.colors.ListedColormap(SEQ)
    bars = ax.bar(range(len(fams)), counts, 0.66, color=[cmap(min(0.999, l)) if np.isfinite(l) else GRID for l in load_], lw=0)
    for i, (c_, l) in enumerate(zip(counts, load_)):
        ax.text(i, c_ + 0.3, f"{c_}", ha="center", fontsize=7.5, color=INK)
        ax.text(i, -2.4, f"|ρ|={l:.2f}" if np.isfinite(l) else "", ha="center", fontsize=6, color=INK2)
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(labs, fontsize=6.5)
    ax.set_ylabel("candidate genes"); ax.set_ylim(-3.2, 14)
    ax.set_title(f"{E8['n_genes']} genes; none recurs across families", color=INK, loc="left")
    ax.text(0.99, 0.95, "shade = |ρ| of the trait\nwith mock-only drought loss", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.3, color=INK2)

    # B: ratio artefact of a baseline-only mutant
    ax = axs[0, 1]; tag(ax, "B")
    eps = [r["eps"] for r in E9["rows"]]
    for n in ["increase", "gain", "rescue", "canonical"]:
        med = [r[n]["median"] for r in E9["rows"]]
        ax.plot(eps, med, color=C[n], marker="o", ms=3, label=LBL[n])
        ax.fill_between(eps, [r[n]["q25"] for r in E9["rows"]], [r[n]["q75"] for r in E9["rows"]],
                        color=C[n], alpha=0.12, lw=0)
    ax.axhline(1, color=AXIS, lw=0.8)
    ax.set_xlabel("ε: fractional loss of mock-drought biomass, no change in gain")
    ax.set_ylabel("apparent rescue, mutant / wild type"); ax.legend(loc="upper left")
    ax.set_title("a weaker baseline inflates only % increase", color=INK, loc="left")

    # C: design heatmap (root reliability)
    ax = axs[1, 0]; tag(ax, "C")
    tab = E12["organs"]["Root"]["table"]
    gn, gr = E12["grid_n"], E12["grid_r"]
    M = np.array([[tab[f"{n}x{r}"]["reliability"] for r in gr] for n in gn])
    ax.imshow(M, cmap=mpl.colors.ListedColormap(SEQ), vmin=0.3, vmax=1, origin="lower", aspect="auto")
    ax.grid(False)
    for i in range(len(gn)):
        for j in range(len(gr)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.78 else INK)
    ax.set_xticks(range(len(gr))); ax.set_xticklabels(gr); ax.set_yticks(range(len(gn))); ax.set_yticklabels(gn)
    ax.set_xlabel("independent runs r"); ax.set_ylabel("plants per cell per run n")
    ax.set_title("root: reliability of canonical rescue by design", color=INK, loc="left")

    # D: geography
    ax = axs[1, 1]; tag(ax, "D")
    lat = np.array([a["lat"] if a["lat"] is not None else np.nan for a in ACC], float)
    lon = np.array([a["lon"] if a["lon"] is not None else np.nan for a in ACC], float)
    v = np.array([a["shoot_canonical"] for a in ACC])
    ok = np.isfinite(lat) & np.isfinite(lon)
    q = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
    sc = ax.scatter(lon[ok], lat[ok], c=q, cmap=mpl.colors.ListedColormap(SEQ), s=12, lw=0.3, edgecolor="white")
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02); cb.set_label("rescue quantile (shoot)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_xlabel("longitude (°)"); ax.set_ylabel("latitude (°)")
    g = E11["organs"]["Shoot"]["canonical"]
    ax.set_title(f"rescue vs longitude ρ = {g['rho_lon']:.2f} (p = {g['p_lon']:.1e})", color=INK, loc="left")
    save(fig, "panel_6_consequences")


if __name__ == "__main__":
    for p in (panel1, panel2, panel3, panel4, panel5, panel6):
        p()
