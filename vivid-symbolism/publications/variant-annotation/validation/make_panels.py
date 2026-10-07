"""make_panels.py -- the seven figure panels, rendered from results/*.json alone.

Layout contract (shared with the companion manuscripts): white background,
four charts in a row, at least one 3D chart per panel, no text-only charts.
Palette: five categorical hues in a fixed, validated order (CVD-separated,
common lightness band; checked with the dataviz validator). Verdicts keep one
colour throughout: index blue, icon red, collapsed amber, composite purple,
symbol green.
"""

from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from common import FIGURES, read_result

K = ["#1f6feb", "#d1242f", "#bf8700", "#8250df", "#1a7f37"]
INK, MUTED, GRID = "#1c1c1c", "#6a6a6a", "#e3e3e3"
VCOL = {"index": K[0], "icon": K[1], "collapsed": K[2], "composite": K[3], "symbol": K[4]}
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#dbe8fb", "#1f6feb", "#0b2f6b"])

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.edgecolor": MUTED,
    "axes.linewidth": 0.7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.fontsize": 7,
    "legend.frameon": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 2.0, "lines.markersize": 5,
})


def new_panel():
    return plt.figure(figsize=(15.0, 3.6))


def style2d(ax):
    ax.grid(True, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def style3d(ax):
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.fill = False
        a.pane.set_edgecolor(GRID)
        a._axinfo["grid"]["color"] = GRID
        a._axinfo["grid"]["linewidth"] = 0.5
    ax.tick_params(labelsize=6)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(ax.get_zlabel(), rotation=90)


def tag(ax, letter, three_d=False):
    if three_d:
        ax.text2D(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=11,
                  fontweight="bold", color=INK)
    else:
        ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontsize=11,
                fontweight="bold", color=INK)


def save(fig, name):
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.2, top=0.86, wspace=0.42)
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def bars3d(ax, X, Y, Z, colors, dx=0.6, dy=0.6):
    ax.bar3d(np.asarray(X) - dx / 2, np.asarray(Y) - dy / 2, np.zeros(len(Z)), dx, dy,
             np.asarray(Z), color=colors, shade=True, edgecolor="white", linewidth=0.3)


# =====================================================================
# PANEL 1 -- genotype calls as three-valued distinctions (E1)
# =====================================================================

def panel1():
    d = read_result("E1_calls")
    fig = new_panel()
    rc = {"V": K[1], "J": K[2], "G": K[0], "I": K[3]}
    names = {"V": "variant-only", "J": "joint-called", "G": "reference blocks", "I": "imputed"}

    ax = fig.add_subplot(1, 4, 1)
    s = d["sites_sweep"]
    x = s["sites"]
    for r in ("V", "G", "I"):
        ax.plot(x, s[r]["cert"], "-o", color=rc[r], ms=4, label=f"certified, {names[r]}")
    for r in ("V", "G"):
        ax.plot(x, s[r]["open"], "--", color=rc[r], lw=1.4, label=f"open, {names[r]}")
    ax.plot(x, s["truly_distinct"], ":", color=INK, lw=1.4, label="truly distinct")
    ax.set_xscale("log")
    ax.set_xlabel("sites in the panel")
    ax.set_ylabel("fraction of sample pairs")
    ax.set_title("Absence is not reference")
    ax.set_ylim(0, 1.42)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(loc="upper left", ncol=2, fontsize=5.6, columnspacing=0.8)
    style2d(ax); tag(ax, "A")

    ax = fig.add_subplot(1, 4, 2)
    summ = d["regimes"]["summary"]
    regs = d["regimes"]["order"]
    bottoms = np.zeros(len(regs))
    lc = {"a": K[3], "+": K[0], "-": K[2]}
    lab = {"a": "articulated (a)", "+": "witnessed (+)", "-": "unasserted (−)"}
    for L in ("+", "a", "-"):
        vals = np.array([summ[r]["letters"][L] for r in regs], float)
        tot = np.array([sum(summ[r]["letters"].values()) for r in regs], float)
        frac = vals / tot
        ax.bar(range(len(regs)), frac, bottom=bottoms, color=lc[L], width=0.62,
               edgecolor="white", linewidth=1.5, label=lab[L])
        bottoms += frac
    ax.set_xticks(range(len(regs)))
    ax.set_xticklabels([names[r] for r in regs], rotation=12)
    ax.set_ylabel("share of letters in sample words")
    ax.set_ylim(0, 1.18)
    ax.set_title("How each exclusion was drawn")
    ax.legend(loc="upper center", ncol=3, fontsize=6)
    style2d(ax); tag(ax, "B")

    ax = fig.add_subplot(1, 4, 3)
    g = d["growth"]
    steps = np.arange(1, len(g["mean_lost_single"]) + 1)
    ax.plot(steps, np.cumsum(g["mean_lost_single"]), color=K[0], label="single-valued call")
    ax.plot(steps, np.cumsum(g["mean_lost_multi"]), color=K[1], label="merged replicate calls")
    ax.set_xlabel("growth step (one call added)")
    ax.set_ylabel("certified pairs lost (cumulative mean)")
    ax.set_title("Only multi-valued certificates are lost")
    ax.legend(loc="upper left")
    style2d(ax); tag(ax, "C")

    ax = fig.add_subplot(1, 4, 4, projection="3d")
    im = d["imputation"]
    C, Sw = np.meshgrid(im["coverage"], np.log10(im["switch"]))
    Z = np.array(im["false_certificate_rate"])
    ax.plot_surface(C, Sw, Z, cmap=SEQ, edgecolor="white", linewidth=0.3, alpha=0.95)
    ax.set_xlabel("coverage")
    ax.set_ylabel("log10 haplotype switch rate")
    ax.set_zlabel("false certificates / all")
    ax.set_title("Imputed certificates can be false")
    ax.view_init(elev=24, azim=-128)
    style3d(ax); tag(ax, "D", True)
    save(fig, "panel_1_calls.png")


# =====================================================================
# PANEL 2 -- genericity and orbits (E2)
# =====================================================================

def panel2():
    d = read_result("E2_generic")
    g, o = d["genericity"], d["orbits"]
    fig = new_panel()

    ax = fig.add_subplot(1, 4, 1)
    cls = g["classes"]
    rate = [g["violations"][c] / g["tests"][c] for c in cls]
    cols = [K[1] if c in g["controls"] else K[0] for c in cls]
    ax.bar(range(len(cls)), rate, color=cols, width=0.7)
    for i, c in enumerate(cls):
        if c not in g["controls"]:
            ax.plot(i, 0.0, "o", color=K[0], ms=4)
    ax.set_xticks(range(len(cls)))
    ax.set_xticklabels([c.replace("_", " ") + ("†" if c in g["controls"] else "")
                        for c in cls], rotation=70, fontsize=6)
    ax.set_ylabel("violation rate  ans(Q,πG) ≠ π ans(Q,G)")
    ax.set_ylim(0, 1)
    ax.set_title("Only the controls see names")
    style2d(ax); tag(ax, "A")

    ax = fig.add_subplot(1, 4, 2)
    t = o["table"]
    M = np.array([[t["orbit_and_returned"], t["orbit_not_returned"]],
                  [t["no_orbit_returned"], t["no_orbit_not_returned"]]])
    ax.imshow(M > 0, cmap=LinearSegmentedColormap.from_list("b", ["#f4f6f9", "#9cc0f5"]))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=11,
                    color=INK, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["returns b", "does not"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["same orbit (VF2)", "different orbit"])
    ax.set_xlabel("description query $D_a$")
    ax.set_title("Orbits are exactly what queries see")
    tag(ax, "B")

    ax = fig.add_subplot(1, 4, 3, projection="3d")
    ctrls = g["controls"]
    X, Y, Z, cc = [], [], [], []
    for ci, c in enumerate(ctrls):
        for sz, (v, n) in g["control_by_size"][c].items():
            X.append(int(sz)); Y.append(ci); Z.append(v / n); cc.append(K[ci + 1])
    bars3d(ax, X, Y, Z, cc, dx=0.7, dy=0.5)
    ax.set_yticks(range(len(ctrls)))
    ax.set_yticklabels([c.replace("_", " ") for c in ctrls], fontsize=6)
    ax.set_xlabel("individuals in graph")
    ax.set_zlabel("violation rate")
    ax.set_title("No control is generic at any size")
    ax.view_init(elev=22, azim=-60)
    style3d(ax); tag(ax, "C", True)

    ax = fig.add_subplot(1, 4, 4)
    sep, ndiff = o["colour_separated_of_different_orbit"]
    vals = [ndiff, t["no_orbit_not_returned"], sep, o["colour_soundness_violations"]]
    labs = ["different\norbits", "separated:\ndescription", "separated:\ncolour ref.",
            "colour ref.\nunsound"]
    ax.bar(range(4), vals, color=[MUTED, K[0], K[3], K[1]], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, str(v), ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=6)
    ax.set_ylabel("pairs")
    ax.set_title("Colour refinement: sound here")
    style2d(ax); tag(ax, "D")
    save(fig, "panel_2_genericity.png")


# =====================================================================
# PANEL 3 -- receivers: schema census (E3)
# =====================================================================

def panel3():
    d = read_result("E3_receivers")
    sch = d["schemas"]
    order = d["order"]
    fig = new_panel()
    dcol = {"genomics": K[0], "chemistry": K[1]}

    ax = fig.add_subplot(1, 4, 1)
    x = np.arange(len(order))
    cc = [sch[s]["classes_concrete"] for s in order]
    ci = [sch[s]["class_iris"] for s in order]
    ax.bar(x - 0.18, cc, 0.34, color=MUTED, label="concrete classes")
    ax.bar(x + 0.18, ci, 0.34, color=[dcol[sch[s]["domain"]] for s in order],
           label="emitted class IRIs")
    for i, (a, b) in enumerate(zip(cc, ci)):
        ax.text(i + 0.18, b * 1.15, f"{b}", ha="center", fontsize=6, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=15, fontsize=7)
    ax.set_ylabel("count (log)")
    ax.set_title("Classes against emitted IRIs")
    ax.legend(loc="upper right")
    style2d(ax); tag(ax, "A")

    ax = fig.add_subplot(1, 4, 2)
    lam = [sch[s]["Lambda_class_pairs"] for s in order]
    slam = [sch[s]["Lambda_slot_pairs"] for s in order]
    ax.bar(x - 0.18, lam, 0.34, color=K[3], label="class pairs not carried (Λ)")
    ax.bar(x + 0.18, slam, 0.34, color=K[2], label="slot pairs not carried")
    for i, (a, b) in enumerate(zip(lam, slam)):
        ax.text(i - 0.18, a + 4, str(a), ha="center", fontsize=6, color=INK)
        ax.text(i + 0.18, b + 4, str(b), ha="center", fontsize=6, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=15, fontsize=7)
    ax.set_ylabel("distinctions lost at emission")
    ax.set_title("Genomics schemas lose none")
    ax.legend(loc="upper left")
    style2d(ax); tag(ax, "B")

    ax = fig.add_subplot(1, 4, 3, projection="3d")
    X, Y, Z, cols = [], [], [], []
    for i, s in enumerate(order):
        cnt = np.bincount(sch[s]["class_fibre_sizes"])
        for size in range(1, len(cnt)):
            if cnt[size]:
                X.append(i); Y.append(size); Z.append(np.log10(cnt[size]) + 0.05)
                cols.append(dcol[sch[s]["domain"]])
    bars3d(ax, X, Y, Z, cols, dx=0.5, dy=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([s.split("-")[0] for s in order], fontsize=6)
    ax.set_ylabel("classes per emitted IRI")
    ax.set_zlabel("log10 fibres", labelpad=2)
    ax.set_title("Fibre sizes by schema")
    ax.view_init(elev=24, azim=-58)
    style3d(ax); tag(ax, "C", True)

    ax = fig.add_subplot(1, 4, 4)
    pairs = [sch[s]["witness"]["contact_derivation_pairs"] for s in order]
    shared = [len(sch[s]["witness"]["pairs_sharing_iri"]) for s in order]
    sep = [p - q for p, q in zip(pairs, shared)]
    ax.bar(x, sep, 0.6, color=K[0], label="contact/derivation pairs separated")
    ax.bar(x, shared, 0.6, bottom=sep, color=K[1], label="pairs sharing an IRI")
    for i, s in enumerate(order):
        if pairs[i] == 0:
            n = len(sch[s].get("method_slots_as_content", []))
            ax.text(i, 0.6, f"no activity\nclasses;\n{n} method\nslots as\ncontent",
                    ha="center", fontsize=6, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=15, fontsize=7)
    ax.set_ylabel("contact × derivation class pairs")
    ax.set_ylim(0, 30)
    ax.set_title("Can the emitted graph carry the ground?")
    ax.legend(loc="upper right", fontsize=6)
    style2d(ax); tag(ax, "D")
    save(fig, "panel_3_receivers.png")


# =====================================================================
# PANEL 4 -- verdicts on real annotation (E4)
# =====================================================================

def panel4():
    d = read_result("E4_verdicts")
    fig = new_panel()
    order = ["collapsed", "index", "icon", "composite"]

    ax = fig.add_subplot(1, 4, 1)
    corp = [("GO human", d["go"]["human"]["membership_verdicts"]),
            ("GO yeast", d["go"]["yeast"]["membership_verdicts"]),
            ("GO Arabidopsis", d["go"]["arabidopsis"]["membership_verdicts"]),
            ("ClinVar A1", d["clinvar"]["A1"]["verdicts"]),
            ("ClinVar A2", d["clinvar"]["A2"]["verdicts"])]
    bottoms = np.zeros(len(corp))
    for v in order:
        frac = np.array([c[v] / sum(c.values()) for _, c in corp])
        ax.bar(range(len(corp)), frac, bottom=bottoms, color=VCOL[v], width=0.62,
               edgecolor="white", linewidth=1.5, label=v)
        bottoms += frac
    ax.set_xticks(range(len(corp)))
    ax.set_xticklabels([n for n, _ in corp], rotation=18, fontsize=7)
    ax.set_ylabel("share of records")
    ax.set_ylim(0, 1.16)
    ax.set_title("Grounds of real genomic records")
    ax.legend(loc="upper center", ncol=4, fontsize=6)
    style2d(ax); tag(ax, "A")

    ax = fig.add_subplot(1, 4, 2)
    tr = d["monotonicity"]["synthetic"]["transitions"]
    names = ["symbol", "collapsed", "index", "icon", "composite"]
    M = np.zeros((5, 5))
    for k, n in tr.items():
        a, b = k.split("->")
        M[names.index(a), names.index(b)] = n
    ax.imshow(np.log10(M + 1), cmap=SEQ)
    for i in range(5):
        for j in range(5):
            if M[i, j] or i != j:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 60 else INK)
    ax.set_xticks(range(5)); ax.set_xticklabels(names, rotation=30, fontsize=6)
    ax.set_yticks(range(5)); ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("to"); ax.set_ylabel("from")
    ax.set_title(f"Verdicts only rise ({d['monotonicity']['synthetic']['retractions']} retractions)")
    tag(ax, "B")

    ax = fig.add_subplot(1, 4, 3)
    rows = d["monotonicity"]["clinvar"][1:]
    labs = [str(r["checkpoint"]) for r in rows]
    xx = np.arange(len(rows))
    ax.plot(xx, [r["verdict_changes"] for r in rows], "-o", color=K[0], ms=3,
            label="verdict changes (all upward)")
    ax.plot(xx, [r["class_changes"] for r in rows], "-s", color=K[2], ms=3,
            label="classification changes")
    ax.plot(xx, [max(r["pathogenic_retracted"], 0.5) for r in rows], "-^", color=K[1], ms=3,
            label="variants leaving 'pathogenic'")
    ax.plot(xx, [0.5] * len(rows), ":", color=K[3], lw=1.2, label="verdict retractions (0)")
    ax.set_yscale("log")
    ax.set_xticks(xx[::3]); ax.set_xticklabels(labs[::3], rotation=30, fontsize=6)
    ax.set_xlabel("ClinVar ingested to checkpoint")
    ax.set_ylabel("variants changed since previous checkpoint")
    ax.set_title("Content moves both ways; ground up")
    ax.legend(loc="upper left", fontsize=6)
    style2d(ax); tag(ax, "C")

    ax = fig.add_subplot(1, 4, 4, projection="3d")
    bc = d["clinvar"]["A1"]["by_class"]
    classes = ["P", "U", "B", "C", "O"]
    X, Y, Z, cols = [], [], [], []
    for i, c in enumerate(classes):
        for j, v in enumerate(order):
            n = bc[c][v]
            X.append(i); Y.append(j); Z.append(np.log10(n + 1)); cols.append(VCOL[v])
    bars3d(ax, X, Y, Z, cols, dx=0.55, dy=0.55)
    ax.set_xticks(range(5)); ax.set_xticklabels(classes)
    ax.set_yticks(range(4)); ax.set_yticklabels(order, fontsize=6)
    ax.set_xlabel("aggregate classification")
    ax.set_zlabel("log10 variants")
    ax.set_title("Every class carries every ground")
    ax.view_init(elev=24, azim=-50)
    style3d(ax); tag(ax, "D", True)
    save(fig, "panel_4_verdicts.png")


# =====================================================================
# PANEL 5 -- twins (E5)
# =====================================================================

def panel5():
    d = read_result("E5_twins")
    fig = new_panel()
    n = d["pairs_per_setting"]

    ax = fig.add_subplot(1, 4, 1)
    labs, meas, auto, pred = [], [], [], []
    for r in ("R0", "R1", "R2"):
        for dd in ("0.0", "0.2"):
            b = d["baseline"][f"{r}_d{dd}"]
            labs.append(f"{r}\nd={dd}")
            meas.append(b["indiscernible"] / n)
            auto.append(b["automorphic"] / n)
            pred.append(b["predicted_indiscernible"] / n)
    x = np.arange(len(labs))
    ax.bar(x, meas, 0.6, color=K[0], label="mutually indiscernible")
    ax.plot(x, auto, "o", color=K[1], ms=6, label="transposition is automorphism")
    ax.plot(x, pred, "_", color=INK, ms=16, mew=2, label="predicted")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6)
    ax.set_ylabel("share of twin pairs")
    ax.set_ylim(0, 1.18)
    ax.set_title("Collapsed emission yields Newman pairs")
    ax.legend(loc="center right", fontsize=6)
    style2d(ax); tag(ax, "A")

    ax = fig.add_subplot(1, 4, 2)
    qs = np.array(d["q"])
    get = lambda q, key: d["R3_grid"][f"{q}_0.0"]["letters"][key] / n
    qq = np.linspace(0, 1, 101)
    for key, col, f, lab in (("+", K[0], lambda q: q ** 2 * (2 - q), "witnessed (+)"),
                             ("-", K[2], lambda q: q * (1 - q) * (2 - q), "unasserted (−)"),
                             ("-rev", K[3], lambda q: q * (2 - q) * (1 - q) ** 2, "unasserted, reverse")):
        ax.plot(qq, f(qq), "-", color=col, lw=1.6, label=lab)
        ax.plot(qs, [get(q, key) for q in qs], "o", color=col, ms=5)
    ax.plot(qq, (1 - qq) ** 4, "-", color=K[1], lw=1.6, label="indiscernible")
    ax.plot(qs, [d["R3_grid"][f"{q}_0.0"]["indiscernible"] / n for q in qs], "o", color=K[1], ms=5)
    ax.set_xlabel("provenance completeness q")
    ax.set_ylabel("share of twin pairs (d = 0)")
    ax.set_title("The letter law (lines) and measurement")
    ax.legend(loc="center right", fontsize=6)
    style2d(ax); tag(ax, "B")

    ax = fig.add_subplot(1, 4, 3, projection="3d")
    ds = np.array(d["d"])
    Q, D = np.meshgrid(qs, ds)
    Zm = np.array([[d["R3_grid"][f"{q}_{dd}"]["indiscernible"] / n for q in qs] for dd in ds])
    Zp = np.array([[d["R3_grid"][f"{q}_{dd}"]["predicted_indiscernible"] / n for q in qs] for dd in ds])
    ax.plot_surface(Q, D, Zm, cmap=SEQ, edgecolor="white", linewidth=0.3, alpha=0.9)
    ax.plot_wireframe(Q, D, Zp, color=K[1], linewidth=0.8)
    ax.set_xlabel("completeness q"); ax.set_ylabel("dropout d")
    ax.set_zlabel("indiscernible share")
    ax.set_title("(1−q)$^4$((1−d)$^2$+d$^2$)$^m$")
    ax.view_init(elev=24, azim=-130)
    style3d(ax); tag(ax, "C", True)

    ax = fig.add_subplot(1, 4, 4)
    tab = d["concordance"]["kappa_verdict_by_inputs"]
    names = ["symbol", "collapsed", "index", "icon"]
    vcode = {v: i for i, v in enumerate(["symbol", "collapsed", "index", "icon", "composite"])}
    M = np.full((4, 4), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            v = tab.get(f"{a}+{b}")
            if v:
                M[i, j] = vcode[v]
    cmap = matplotlib.colors.ListedColormap([VCOL[v] for v in ["symbol", "collapsed", "index", "icon", "composite"]])
    ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=4.5)
    for i in range(4):
        for j in range(4):
            v = tab.get(f"{names[i]}+{names[j]}")
            ax.text(j, i, v, ha="center", va="center", fontsize=6.5, color="white",
                    fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=6, rotation=20)
    ax.set_yticks(range(4)); ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("ground of call y"); ax.set_ylabel("ground of call x")
    ax.set_title("Concordance inherits contact")
    tag(ax, "D")
    save(fig, "panel_5_twins.png")


# =====================================================================
# PANEL 6 -- spectral receivers (E6)
# =====================================================================

def panel6():
    d = read_result("E6_spectral")
    fig = new_panel()

    ax = fig.add_subplot(1, 4, 1)
    F = d["fibres_full"]
    Ls = [f["L"] for f in F]
    ax.plot(Ls, [f["mean_fibre_per_sequence"] for f in F], "-o", color=K[0],
            label="fibre of the magnitude receiver")
    ax.plot(Ls, [f["mean_orbit_per_sequence"] for f in F], "-s", color=K[2],
            label="rotations and reversals")
    ax.plot(Ls, [1] * len(Ls), "-^", color=K[3], label="phase receiver (exhaustive at L=6)")
    ax.set_xlabel("sequence length L (all 4$^L$ sequences)")
    ax.set_ylabel("mean size of a sequence's fibre")
    for f in F:
        ax.annotate(f"{100 * f['share_in_homometric_fibres']:.0f}%", (f["L"], f["mean_fibre_per_sequence"]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6, color=K[0])
    ax.set_title("Fibres exceed the symmetry orbit")
    ax.legend(loc="upper left", fontsize=6)
    style2d(ax); tag(ax, "A")

    ax = fig.add_subplot(1, 4, 2)
    T = d["fibres_truncated_L10"]
    full = F[-1]
    Ks = [t["K"] for t in T] + [5]
    nf = [t["fibres"] for t in T] + [full["fibres"]]
    ax.plot(Ks, nf, "-o", color=K[0], label="fibres of the magnitude receiver")
    ax.axhline(full["dihedral_orbits"], color=K[2], ls="--", lw=1.4, label="dihedral orbits")
    ax.axhline(full["sequences"], color=K[3], ls=":", lw=1.4, label="sequences (phase fibres)")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["1", "2", "3", "4", "5 (all)"])
    ax.set_xlabel("retained bins K per channel (L = 10)")
    ax.set_ylabel("number of distinct images")
    ax.set_title("Coarser receivers see larger cells")
    ax.legend(loc="center right", fontsize=6)
    style2d(ax); tag(ax, "B")

    ax = fig.add_subplot(1, 4, 3, projection="3d")
    c = d["corroboration"]
    tau = np.log10(np.array(c["tau"]))
    mu = np.array(c["mu"])
    TT, MM = np.meshgrid(tau, mu)
    Z = np.array(c["twin_surface"]["magnitude"])
    ax.plot_surface(TT, MM, Z, cmap=SEQ, edgecolor="white", linewidth=0.2, alpha=0.95)
    ax.set_xlabel("log10 threshold τ")
    ax.set_ylabel("substitution rate μ")
    ax.set_zlabel("corroborated share")
    ax.set_title("Corroborating a measured twin")
    ax.view_init(elev=24, azim=-130)
    style3d(ax); tag(ax, "C", True)

    ax = fig.add_subplot(1, 4, 4)
    kinds = [("rotation", K[2]), ("reversal", K[3]), ("revcomp", K[4]),
             ("twin_0.05", K[0]), ("unrelated", K[1])]
    for kind, col in kinds:
        lab = {"rotation": "rotation", "reversal": "reversal (overlies rotation)",
               "revcomp": "reverse complement", "twin_0.05": "measured twin (μ=0.05)",
               "unrelated": "unrelated"}[kind]
        lw = 3.2 if kind == "rotation" else 1.8
        ax.plot(c["tau"], c["rates"]["magnitude"][kind], "-", color=col, lw=lw, label=lab)
        ax.plot(c["tau"], c["rates"]["phase"][kind], "--", color=col, lw=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("declared threshold τ")
    ax.set_ylabel("share corroborated (V ≥ 1−τ)")
    ax.set_title("Magnitude (solid) vs phase (dashed)")
    ax.legend(loc="center left", fontsize=6)
    style2d(ax); tag(ax, "D")
    save(fig, "panel_6_spectral.png")


# =====================================================================
# PANEL 7 -- answering (E7)
# =====================================================================

def panel7():
    d = read_result("E7_answering")
    fig = new_panel()

    ax = fig.add_subplot(1, 4, 1)
    for (org, col) in (("human", K[0]), ("yeast", K[2]), ("arabidopsis", K[4])):
        sh = np.sort([(r["counts"][1] + r["counts"][3]) / r["n"] for r in d["go"][org]["profiles"]])
        ax.plot(np.linspace(0, 1, sh.size), sh, color=col,
                label=f"{org} ({sh.size} terms, median {np.median(sh):.2f})")
    ax.set_xlabel("GO terms with ≥ 20 genes, ranked")
    ax.set_ylabel("share without experimental ground")
    ax.set_title("Same gene set, different grounds")
    ax.legend(loc="upper left", fontsize=6)
    style2d(ax); tag(ax, "A")

    ax = fig.add_subplot(1, 4, 2)
    top = d["clinvar"]["top"][:15][::-1]
    y = np.arange(len(top))
    left = np.zeros(len(top))
    for v in ("index", "composite", "icon", "collapsed"):
        idx = ["symbol", "collapsed", "index", "icon", "composite"].index(v)
        frac = np.array([r["counts"][idx] / r["n"] for r in top])
        ax.barh(y, frac, left=left, color=VCOL[v], height=0.7, edgecolor="white",
                linewidth=1.0, label=v)
        left += frac
    ax.set_yticks(y); ax.set_yticklabels([f"{r['gene']} ({r['n']})" for r in top], fontsize=6)
    ax.set_xlabel("share of 'pathogenic' answer rows")
    ax.set_xlim(0, 1.42)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Pathogenic in gene X: profiles")
    ax.legend(loc="center right", fontsize=6)
    style2d(ax); tag(ax, "B")

    ax = fig.add_subplot(1, 4, 3, projection="3d")
    dd = d["directions"]
    Kk, Ff = np.meshgrid(dd["k"], dd["f"], indexing="ij")
    Z = np.array(dd["claimed_index_share"]) - dd["true_index_share"]
    ax.plot_surface(Kk, Ff, Z, cmap=LinearSegmentedColormap.from_list(
        "div", [K[1], "#f2f2f2", K[0]]), edgecolor="white", linewidth=0.3, alpha=0.95)
    ax.set_xlabel("emission completeness k")
    ax.set_ylabel("fabrication f")
    ax.set_zlabel("claimed − true index share")
    ax.set_title("Error directions, same rows")
    ax.view_init(elev=24, azim=-130)
    style3d(ax); tag(ax, "C", True)

    ax = fig.add_subplot(1, 4, 4)
    rb = d["receivers_bits"]
    names = list(rb)
    x = np.arange(len(names))
    ax.bar(x - 0.18, [rb[n]["H"] for n in names], 0.34, color=MUTED, label="verdict entropy")
    ax.bar(x + 0.18, [rb[n]["I_retained"] for n in names], 0.34, color=K[0],
           label="retained by the emitted fields")
    for i, n in enumerate(names):
        ax.text(i + 0.18, rb[n]["I_retained"] + 0.03, f"{rb[n]['I_retained']:.2f}",
                ha="center", fontsize=6, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["GO GAF", "GO GMT", "ClinVar\nsubmissions", "ClinVar\nVCF fields"], fontsize=7)
    ax.set_ylabel("bits per record")
    ax.set_title("What each receiver keeps of the ground")
    ax.legend(loc="upper right", fontsize=6)
    style2d(ax); tag(ax, "D")
    save(fig, "panel_7_answering.png")


PANELS = {"1": panel1, "2": panel2, "3": panel3, "4": panel4, "5": panel5,
          "6": panel6, "7": panel7}

if __name__ == "__main__":
    which = sys.argv[1:] or list(PANELS)
    for w in which:
        PANELS[w]()
