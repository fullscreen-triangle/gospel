"""
generate_panels.py -- eight figure panels for synopsis-genomic-scripting.tex.

Every value plotted is recomputed here from core.py / semantics.py /
lang.py at plot time. Nothing is read from validation/results/*.json.
The panels are therefore an independent second execution of the
framework rather than a redraw of the validation suite's output; where a
panel agrees with the suite, that agreement is evidence.

Layout contract (the paper's figure style):
  * white background
  * four charts in a row
  * at least one 3D chart per panel
  * no conceptual diagrams, no text-only charts, no tables

Palette: five categorical hues in FIXED order, chosen for CVD
separation, a common lightness band, a chroma floor, and contrast
against white. Do not reorder -- adjacent-pair separation was checked in
this order.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "validation"))

import core as C  # noqa: E402
import lang as L  # noqa: E402
import semantics as S  # noqa: E402

SEED = 20260816

# validated categorical order -- do not permute
K = ["#1f6feb", "#d1242f", "#bf8700", "#8250df", "#1a7f37"]
INK = "#1c1c1c"
MUTED = "#6a6a6a"
GRID = "#e3e3e3"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
})


def new_panel():
    return plt.figure(figsize=(15.0, 3.5))


def style2d(ax):
    ax.grid(True, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def style3d(ax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_edgecolor(GRID)
        a._axinfo["grid"]["color"] = GRID
        a._axinfo["grid"]["linewidth"] = 0.5
    ax.tick_params(labelsize=6)
    # matplotlib auto-rotates the z label with the view and can land it
    # upside down; pin it upright.
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(ax.get_zlabel(), rotation=90)


def tag(ax, letter, three_d=False):
    if three_d:
        ax.text2D(-0.08, 1.06, letter, transform=ax.transAxes,
                  fontsize=11, fontweight="bold", color=INK)
    else:
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=INK)


def save(fig, name):
    fig.tight_layout(w_pad=2.4)
    fig.savefig(HERE / name, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


# =====================================================================
# PANEL 1 -- the contact floor and conditional expansion
# =====================================================================

def panel1(rng):
    fig = new_panel()

    # (A) sigma(v)/beta never below 1
    ax = fig.add_subplot(1, 4, 1)
    ratios = []
    for _ in range(220):
        g = C.random_contact_structure(rng, int(rng.integers(3, 7)))
        b = g.floor()
        for v in g.items():
            ratios.append(g.separation_cost(v) / b)
    r = np.array(ratios)
    ax.hist(r, bins=44, color=K[0], edgecolor="white", linewidth=0.4)
    ax.axvline(1.0, color=K[1], lw=2.0, ls="--")
    ax.set_xlabel(r"$\sigma(v)\,/\,\beta$")
    ax.set_ylabel("count")
    ax.set_title("Separation never falls below the floor")
    ax.set_xlim(left=0.0)
    style2d(ax)
    tag(ax, "A")
    print(f"  P1A n={len(r)} min={r.min():.4f} med={np.median(r):.3f} "
          f"max={r.max():.2f}")

    # (B) floor vs size
    ax = fig.add_subplot(1, 4, 2)
    sizes = list(range(3, 10))
    lo, med, hi = [], [], []
    for n in sizes:
        vals = [C.random_contact_structure(rng, n).floor() for _ in range(60)]
        lo.append(np.percentile(vals, 10))
        med.append(np.median(vals))
        hi.append(np.percentile(vals, 90))
    ax.fill_between(sizes, lo, hi, color=K[0], alpha=0.18, linewidth=0)
    ax.plot(sizes, med, color=K[0], marker="o", ms=5)
    ax.axhline(0.0, color=K[1], lw=1.6, ls="--")
    ax.set_xlabel("items in structure")
    ax.set_ylabel(r"floor $\beta$")
    ax.set_title("Floor declines but stays positive")
    ax.set_ylim(bottom=0.0)
    style2d(ax)
    tag(ax, "B")
    print(f"  P1B med {med[0]:.4f} -> {med[-1]:.4f}")

    # (C) 3D separation-cost surface
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    ns = np.arange(3, 9)
    wlos = np.linspace(0.05, 0.8, 8)
    Z = np.zeros((len(wlos), len(ns)))
    for i, wl in enumerate(wlos):
        for j, n in enumerate(ns):
            acc = []
            for _ in range(12):
                g = C.random_contact_structure(rng, int(n), w_lo=float(wl),
                                               w_hi=float(wl) + 1.2)
                acc.append(np.mean([g.separation_cost(v) for v in g.items()]))
            Z[i, j] = np.mean(acc)
    X, Y = np.meshgrid(ns, wlos)
    ax.plot_surface(X, Y, Z, cmap="Blues", edgecolor="white",
                    linewidth=0.25, antialiased=True, alpha=0.95)
    ax.set_xlabel("items")
    ax.set_ylabel("min weight")
    ax.set_zlabel(r"mean $\sigma$")
    ax.set_title("Separation cost surface")
    ax.view_init(24, -128)
    style3d(ax)
    tag(ax, "C", three_d=True)
    print(f"  P1C Z {Z.min():.3f}..{Z.max():.3f}")

    # (D) local floor under three expansion regimes
    ax = fig.add_subplot(1, 4, 4)
    labels = {"1/k": r"$w=1/k$", "1/k2": r"$w=1/k^2$", "stab": "stabilising"}
    finals = {}
    for idx, mode in enumerate(["1/k", "1/k2", "stab"]):
        g = C.ContactStructure()
        g.add_edge("v", "u", 1.0)
        g.add_medium_edges(1.0)
        ys = []
        for k in range(1, 41):
            w = (1.0 / k if mode == "1/k" else
                 1.0 / (k * k) if mode == "1/k2" else 1.0 + 0.01 * k)
            g.add_edge("v", f"d{k}", w)
            ys.append(g.local_floor("v"))
        ax.plot(range(1, 41), ys, color=K[idx], label=labels[mode])
        finals[mode] = ys[-1]
    ax.set_yscale("log")
    ax.set_xlabel("expansion step $k$")
    ax.set_ylabel(r"local floor $\beta_k(v)$")
    ax.set_title("Only stabilising expansions keep identity")
    ax.legend(loc="lower left")
    style2d(ax)
    tag(ax, "D")
    print(f"  P1D finals {finals}")

    save(fig, "panel_1_floor.png")


# =====================================================================
# PANEL 2 -- invariance: labels, medium, skeleton
# =====================================================================

def panel2(rng):
    fig = new_panel()

    # (A) relabelling preserves sigma exactly
    ax = fig.add_subplot(1, 4, 1)
    before, after = [], []
    for _ in range(160):
        g = C.random_contact_structure(rng, int(rng.integers(3, 7)))
        items = sorted(g.items())
        perm = list(items)
        rng.shuffle(perm)
        mapping = {a: f"z{b}" for a, b in zip(items, perm)}
        h = g.relabel(mapping)
        for v in items:
            before.append(g.separation_cost(v))
            after.append(h.separation_cost(mapping[v]))
    before, after = np.array(before), np.array(after)
    ax.scatter(before, after, s=16, color=K[0], alpha=0.55,
               edgecolor="white", linewidth=0.3)
    lim = [0, max(before.max(), after.max()) * 1.05]
    ax.plot(lim, lim, color=K[1], lw=1.6, ls="--")
    ax.set_xlabel(r"$\sigma(v)$ before relabelling")
    ax.set_ylabel(r"$\sigma(\pi v)$ after")
    ax.set_title("Labels carry nothing")
    style2d(ax)
    tag(ax, "A")
    dev = float(np.abs(before - after).max())
    print(f"  P2A n={len(before)} max|dev|={dev:.3e}")

    # (B) moving the medium destroys the invariant
    ax = fig.add_subplot(1, 4, 2)
    devs = []
    for _ in range(140):
        g = C.random_contact_structure(rng, int(rng.integers(4, 7)))
        items = sorted(g.items())
        v = items[0]
        s0 = g.separation_cost(v)
        # swap the medium's role with an item: relabel refuses, so build
        # the swapped structure directly and measure the damage.
        h = C.ContactStructure()
        sw = {C.MEDIUM: items[-1], items[-1]: C.MEDIUM}
        for e, w in g.weights.items():
            a, b = tuple(e)
            h.weights[frozenset((sw.get(a, a), sw.get(b, b)))] = w
        if v not in h.vertices or C.MEDIUM not in h.vertices:
            continue
        devs.append(abs(h.separation_cost(v) - s0) / max(s0, 1e-12))
    devs = np.array(devs)
    ax.hist(devs, bins=36, color=K[3], edgecolor="white", linewidth=0.4)
    ax.axvline(0.0, color=K[1], lw=2.0, ls="--")
    ax.set_xlabel(r"relative change in $\sigma(v)$")
    ax.set_ylabel("count")
    ax.set_title("Swapping the medium is not an isomorphism")
    style2d(ax)
    tag(ax, "B")
    print(f"  P2B n={len(devs)} frac_nonzero="
          f"{float((devs > 1e-9).mean()):.3f} mean={devs.mean():.3f}")

    # (C) 3D: skeleton invariance under perturbation
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    g0 = C.random_contact_structure(rng, 5)
    edges = sorted(g0.weights, key=lambda e: tuple(sorted(e)))
    deltas = np.linspace(0.0, 2.0, 12)
    v0 = sorted(g0.items())[0]
    s_base = g0.separation_cost(v0)
    n_base = len(g0.weights)
    xs, ys, zs, cs = [], [], [], []
    skeleton_fixed = True
    for i, e in enumerate(edges[:10]):
        for d in deltas:
            h = C.perturb(g0, {e: float(d)})
            skeleton_fixed &= (len(h.weights) == n_base)
            xs.append(i)
            ys.append(d)
            zs.append(h.separation_cost(v0))
            cs.append(abs(h.separation_cost(v0) - s_base) / s_base)
    cs = np.array(cs, dtype=float)
    p = ax.scatter(xs, ys, zs, c=cs, cmap="viridis", s=16,
                   edgecolor="white", linewidth=0.2)
    ax.set_xlabel("perturbed edge")
    ax.set_ylabel(r"$\delta$")
    ax.set_zlabel(r"$\sigma(v)$")
    ax.set_title("Costs move, contacts do not")
    ax.view_init(22, -132)
    style3d(ax)
    tag(ax, "C", three_d=True)
    cb = fig.colorbar(p, ax=ax, shrink=0.55, pad=0.11)
    cb.set_label(r"relative change in $\sigma$", size=6)
    cb.ax.tick_params(labelsize=6)
    print(f"  P2C skeleton fixed={skeleton_fixed}; "
          f"rel change 0..{cs.max():.3f} over {n_base} edges")

    # (D) cut weight is the invariant: sigma vs degree, coloured by cut
    ax = fig.add_subplot(1, 4, 4)
    degs, sigs = [], []
    for _ in range(120):
        g = C.random_contact_structure(rng, int(rng.integers(4, 7)))
        for v in sorted(g.items()):
            degs.append(len(g.incident(v)))
            sigs.append(g.separation_cost(v))
    degs, sigs = np.array(degs), np.array(sigs)
    for d in sorted(set(degs.tolist())):
        m = degs == d
        if m.sum() < 3:
            continue
        ax.scatter(np.full(m.sum(), d) + rng.uniform(-0.14, 0.14, m.sum()),
                   sigs[m], s=14, color=K[0], alpha=0.45,
                   edgecolor="white", linewidth=0.25)
        ax.plot([d - 0.3, d + 0.3], [np.median(sigs[m])] * 2,
                color=K[1], lw=1.8)
    ax.set_xlabel("degree of $v$")
    ax.set_ylabel(r"$\sigma(v)$")
    ax.set_title("Cost tracks the boundary, not the name")
    style2d(ax)
    tag(ax, "D")
    print(f"  P2D n={len(degs)}")

    save(fig, "panel_2_invariance.png")


# =====================================================================
# PANEL 3 -- region and receiver
# =====================================================================

def panel3(rng):
    fig = new_panel()

    # (A) regionality: minimiser size distribution
    ax = fig.add_subplot(1, 4, 1)
    sizes = []
    for _ in range(150):
        g = C.random_contact_structure(rng, int(rng.integers(4, 7)))
        for v in sorted(g.items()):
            _, Smin = g.min_cut(v, C.MEDIUM)
            sizes.append(len(Smin))
    sizes = np.array(sizes)
    vals, counts = np.unique(sizes, return_counts=True)
    ax.bar(vals, counts, color=K[0], edgecolor="white", linewidth=0.6)
    ax.set_xlabel("vertices in minimising set")
    ax.set_ylabel("count")
    ax.set_title("Minimisers are regions")
    style2d(ax)
    tag(ax, "A")
    print(f"  P3A frac singleton={float((sizes == 1).mean()):.3f}")

    # (B) region/point crossover in the two-cluster family
    ax = fig.add_subplot(1, 4, 2)
    Ws = np.linspace(0.05, 3.0, 26)
    for idx, r in enumerate([3, 4, 5]):
        ys = []
        for W in Ws:
            g = C.two_cluster_structure(r, float(W), 0.2)
            _, Smin = g.min_cut("a1", C.MEDIUM)
            ys.append(len(Smin))
        ax.plot(Ws, ys, color=K[idx], marker="", label=f"$r={r}$")
    ax.set_xlabel("intra-cluster weight $W$")
    ax.set_ylabel("minimiser size")
    ax.set_title("Where a region collapses to a point")
    ax.legend()
    style2d(ax)
    tag(ax, "B")

    # (C) 3D receiver-relativity surface
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    base = C.random_contact_structure(rng, 5)
    bg = [e for e in sorted(base.weights, key=lambda e: tuple(sorted(e)))]
    u = sorted(base.items())[0]
    bgs = [e for e in bg if u not in e][:9]
    ds = np.linspace(0.0, 2.5, 10)
    Z = np.zeros((len(ds), len(bgs)))
    for j, e in enumerate(bgs):
        for i, d in enumerate(ds):
            Z[i, j] = C.perturb(base, {e: float(d)}).separation_cost(u)
    s0 = base.separation_cost(u)
    # A surface would interpolate across edges that are not adjacent in
    # any meaningful order and would smooth over the fact that most
    # edges do not move sigma at all. Bars keep the discreteness.
    # An edge that does not move sigma has bar height exactly zero and
    # would vanish into the baseline, which is the same picture as
    # "no edge plotted here". Those columns are drawn as a flat marker
    # line at s0 so absence of response is visible as a response.
    moved = 0
    for j in range(len(bgs)):
        does_move = bool(np.ptp(Z[:, j]) > 1e-9)
        moved += int(does_move)
        if does_move:
            ax.bar3d(np.full(len(ds), j) - 0.34, ds - 0.09,
                     np.full(len(ds), s0),
                     0.68, 0.18, Z[:, j] - s0,
                     color=K[1], edgecolor="white", linewidth=0.25,
                     shade=True, alpha=0.92)
        else:
            ax.plot(np.full(len(ds), j), ds, np.full(len(ds), s0),
                    color=K[0], lw=2.2, solid_capstyle="round",
                    alpha=0.95)
    ax.plot([], [], [], color=K[1], lw=4, label=r"moves $\sigma(u)$")
    ax.plot([], [], [], color=K[0], lw=2.2, label="does not")
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    ax.set_xlabel("background edge")
    ax.set_ylabel(r"$\delta$")
    ax.set_zlabel(r"$\sigma_N(u)$")
    ax.set_title("The value depends on the receiver")
    ax.set_xticks(range(len(bgs)))
    ax.view_init(24, -126)
    style3d(ax)
    tag(ax, "C", three_d=True)
    print(f"  P3C edges moving sigma: {moved}/{len(bgs)}")

    # (D) how often, and by how much
    ax = fig.add_subplot(1, 4, 4)
    diffs = []
    n_edges_tested = 0
    n_edges_moved = 0
    for _ in range(200):
        g = C.random_contact_structure(rng, int(rng.integers(4, 6)))
        u = sorted(g.items())[0]
        s0 = g.separation_cost(u)
        for e in sorted(g.weights, key=lambda e: tuple(sorted(e))):
            if u in e:
                continue
            n_edges_tested += 1
            s1 = C.perturb(g, {e: 0.6}).separation_cost(u)
            if abs(s1 - s0) > 1e-9:
                n_edges_moved += 1
                diffs.append(abs(s1 - s0))
    diffs = np.array(diffs)
    ax.hist(diffs, bins=34, color=K[1], edgecolor="white", linewidth=0.4)
    ax.axvline(1e-6, color=K[0], lw=1.8, ls="--")
    ax.set_xlabel(r"$|\Delta\sigma(u)|$ when it moves")
    ax.set_ylabel("count")
    ax.set_title("Rare, but never marginal")
    style2d(ax)
    tag(ax, "D")
    print(f"  P3D moved {n_edges_moved}/{n_edges_tested}="
          f"{n_edges_moved / n_edges_tested:.4f} min={diffs.min():.4f}")

    save(fig, "panel_3_region_receiver.png")


# =====================================================================
# PANEL 4 -- alignment, anchors, residue
# =====================================================================

def panel4(rng):
    fig = new_panel()

    # (A) Delta >= 0 always
    ax = fig.add_subplot(1, 4, 1)
    ds = []
    for _ in range(180):
        g = C.random_contact_structure(rng, int(rng.integers(4, 7)))
        items = sorted(g.items())
        for a, b in itertools.combinations(items, 2):
            ds.append(C.distance_to_floor(g, a, b))
    ds = np.array(ds)
    ax.hist(ds, bins=46, color=K[0], edgecolor="white", linewidth=0.4)
    ax.axvline(0.0, color=K[1], lw=2.0, ls="--")
    ax.set_xlabel(r"$\Delta(u,v)=a(u,v)-\beta/\Omega$")
    ax.set_ylabel("count")
    ax.set_title("Distance to floor is non-negative")
    style2d(ax)
    tag(ax, "A")
    print(f"  P4A n={len(ds)} min={ds.min():.3e} "
          f"frac_zero={float((ds < 1e-12).mean()):.3f}")

    # (B) ablation score distribution: anchors separate from residue
    ax = fig.add_subplot(1, 4, 2)
    anch, resid = [], []
    for _ in range(120):
        g = C.random_contact_structure(rng, 6)
        x = sorted(g.items())[0]
        a, res = C.anchors_and_residue(g, x, 2)
        for u in a:
            anch.append(C.ablation_score(g, x, u))
        for u in res:
            resid.append(C.ablation_score(g, x, u))
    anch, resid = np.array(anch), np.array(resid)
    bins = np.linspace(min(anch.min(), resid.min()),
                       max(anch.max(), resid.max()), 34)
    ax.hist(anch, bins=bins, color=K[1], alpha=0.75, label="anchors",
            edgecolor="white", linewidth=0.3)
    ax.hist(resid, bins=bins, color=K[0], alpha=0.65, label="residue",
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel(r"ablation score $\alpha(x;u)$")
    ax.set_ylabel("count")
    ax.set_title("The split is by measured effect")
    ax.legend()
    style2d(ax)
    tag(ax, "B")
    print(f"  P4B anchor med={np.median(anch):.4f} "
          f"residue med={np.median(resid):.4f}")

    # (C) 3D cross-demand surface over (r, perturbation)
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    # 11 items, not 6: on a small structure the residue is down to one
    # item by r=4 and the surface is computed on almost nothing.
    #
    # The second axis is the FRACTION of edges perturbed, not the
    # magnitude on one edge: one edge moves cross-demand by about 0.05
    # against the budget's 0.73, so a magnitude axis is flat to the eye.
    # Even swept properly the surface stays shallow in this direction --
    # that is the finding, and the title says so rather than hiding it.
    rs = np.arange(1, 7)
    base = C.random_contact_structure(rng, 11)
    edges = sorted(base.weights, key=lambda e: tuple(sorted(e)))
    fracs = np.linspace(0.0, 1.0, 9)
    Z = np.zeros((len(fracs), len(rs)))
    for i, f in enumerate(fracs):
        nk = int(round(f * len(edges)))
        gB = C.perturb(base, {e: 0.8 for e in edges[:nk]})
        for j, r in enumerate(rs):
            x = sorted(base.items())[0]
            _, res = C.anchors_and_residue(gB, x, int(r))
            corr = {i2: i2 for i2 in res}
            others = sorted(gB.items())
            corr = {i2: others[(others.index(i2) + 1) % len(others)]
                    for i2 in res}
            Z[i, j] = C.cross_demand(base, gB, res, corr)
    X, Y = np.meshgrid(rs, fracs)
    ax.plot_surface(X, Y, Z, cmap="cividis", edgecolor="white",
                    linewidth=0.25, alpha=0.95)
    ax.set_xlabel("anchors $r$")
    ax.set_ylabel("fraction of edges perturbed")
    ax.set_zlabel(r"cross-demand $D$")
    ax.set_title("Budget sets the demand;\nperturbation barely moves it")
    ax.view_init(24, -130)
    style3d(ax)
    tag(ax, "C", three_d=True)
    dr = float(np.mean(Z.max(axis=0) - Z.min(axis=0)))
    dd = float(np.mean(Z.max(axis=1) - Z.min(axis=1)))
    print(f"  P4C D {Z.min():.4f}..{Z.max():.4f}; "
          f"span over budget {dd:.4f} vs over perturbation {dr:.4f}")

    # (D) how much separation the anchor budget buys, and where it stops
    ax = fig.add_subplot(1, 4, 4)
    # The structures are drawn large enough that the budget stays inside
    # the available set: at r close to the number of non-medium items the
    # residue empties and the gap is undefined, not zero.
    rs = list(range(1, 7))
    lo, med, hi, captured = [], [], [], []
    for r in rs:
        gaps, caps = [], []
        for _ in range(90):
            g = C.random_contact_structure(rng, 11)
            x = sorted(g.items())[0]
            a, res = C.anchors_and_residue(g, x, r)
            sa = [C.ablation_score(g, x, u) for u in a]
            sr = [C.ablation_score(g, x, u) for u in res]
            if not sa or not sr:
                continue
            gaps.append(min(sa) - max(sr))
            tot = sum(max(0.0, s) for s in sa + sr)
            caps.append(sum(max(0.0, s) for s in sa) / tot if tot > 0 else
                        1.0)
        if not gaps:
            raise RuntimeError(f"P4D: no admissible split at r={r}")
        lo.append(np.percentile(gaps, 25))
        med.append(np.median(gaps))
        hi.append(np.percentile(gaps, 75))
        captured.append(np.median(caps))
    ax.fill_between(rs, lo, hi, color=K[0], alpha=0.18, linewidth=0)
    ax.plot(rs, med, color=K[0], marker="o", ms=5, label="anchor-residue gap")
    ax.axhline(0.0, color=MUTED, lw=1.0, ls=":")
    ax2 = ax.twinx()
    ax2.plot(rs, captured, color=K[4], marker="s", ms=4, ls="--",
             label="fraction of effect captured")
    ax2.set_ylabel("fraction of effect captured", color=K[4])
    ax2.tick_params(axis="y", colors=K[4], labelsize=7)
    ax2.set_ylim(0, 1.05)
    ax2.spines["top"].set_visible(False)
    ax.set_xlabel("anchor budget $r$")
    ax.set_ylabel("score gap at the split", color=K[0])
    ax.tick_params(axis="y", colors=K[0])
    ax.set_title("The budget buys separation, then stops")
    style2d(ax)
    tag(ax, "D")
    print(f"  P4D gap {med[0]:.4f}->{med[-1]:.4f} "
          f"captured {captured[0]:.3f}->{captured[-1]:.3f}")

    save(fig, "panel_4_alignment.png")


# =====================================================================
# PANEL 5 -- relaxation
# =====================================================================

def panel5(rng):
    fig = new_panel()

    # (A) demand trajectories
    ax = fig.add_subplot(1, 4, 1)
    for idx in range(5):
        D0 = float(rng.uniform(1.0, 4.0))
        eta = float(rng.uniform(0.15, 0.5))
        res = C.relax(D0, lambda D, e=eta: max(0.0, D - e * 1.15), eta)
        ax.plot(range(len(res.demands)), res.demands, color=K[idx % 5],
                marker="o", ms=3, lw=1.6)
    ax.axhline(0.0, color=MUTED, lw=1.0, ls=":")
    ax.set_xlabel("step $k$")
    ax.set_ylabel(r"demand $D_k$")
    ax.set_title("Monotone decrease")
    style2d(ax)
    tag(ax, "A")

    # (B) steps vs certified bound
    ax = fig.add_subplot(1, 4, 2)
    steps, bounds = [], []
    for _ in range(260):
        D0 = float(rng.uniform(0.5, 6.0))
        eta = float(rng.uniform(0.05, 0.8))
        res = C.relax(D0, lambda D, e=eta: max(0.0, D - e * float(
            rng.uniform(1.0, 1.6))), eta)
        steps.append(res.steps)
        bounds.append(res.bound)
    steps, bounds = np.array(steps), np.array(bounds)
    ax.scatter(bounds, steps, s=16, color=K[0], alpha=0.5,
               edgecolor="white", linewidth=0.3)
    lim = [0, bounds.max() * 1.05]
    ax.plot(lim, lim, color=K[1], lw=1.8, ls="--")
    ax.set_xlabel(r"certified bound $\lceil D_0/\eta\rceil$")
    ax.set_ylabel("steps taken")
    ax.set_title("The bound is never exceeded")
    style2d(ax)
    tag(ax, "B")
    print(f"  P5B violations={int((steps > bounds).sum())}/{len(steps)}")

    # (C) 3D bound surface over (D0, eta)
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    D0s = np.linspace(0.5, 6.0, 12)
    etas = np.linspace(0.08, 1.0, 12)
    Z = np.zeros((len(etas), len(D0s)))
    for i, e in enumerate(etas):
        for j, d0 in enumerate(D0s):
            res = C.relax(float(d0), lambda D, ee=e: max(0.0, D - ee * 1.05),
                          float(e))
            Z[i, j] = res.steps
    X, Y = np.meshgrid(D0s, etas)
    ax.plot_surface(X, Y, Z, cmap="Blues", edgecolor="white",
                    linewidth=0.25, alpha=0.95)
    ax.set_xlabel(r"$D_0$")
    ax.set_ylabel(r"$\eta$")
    ax.set_zlabel("steps")
    ax.set_title("Termination is a surface\nin the parameters")
    ax.view_init(26, -132)
    style3d(ax)
    tag(ax, "C", three_d=True)

    # (D) the dichotomy: quiescence vs decline
    ax = fig.add_subplot(1, 4, 4)
    ps = np.linspace(0.0, 0.6, 13)
    q_frac, d_frac, med_steps = [], [], []
    for p in ps:
        q = d = 0
        st = []
        for _ in range(60):
            eta = 0.3
            def upd(D, p=p):
                if rng.random() < p:
                    return None
                return max(0.0, D - eta * 1.1)
            res = C.relax(3.0, upd, eta)
            if res.declined:
                d += 1
            else:
                q += 1
                st.append(res.steps)
        q_frac.append(q / 60)
        d_frac.append(d / 60)
        med_steps.append(np.median(st) if st else np.nan)
    ax.plot(ps, q_frac, color=K[4], marker="o", ms=4, label="quiescent")
    ax.plot(ps, d_frac, color=K[1], marker="s", ms=4, label="declined")
    ax.set_xlabel("per-step probability of no effective update")
    ax.set_ylabel("fraction of runs")
    ax.set_title("Two outcomes, and only two")
    ax.set_ylim(-0.05, 1.08)
    ax.legend()
    style2d(ax)
    tag(ax, "D")
    print(f"  P5D q+d always 1: "
          f"{all(abs(a + b - 1) < 1e-12 for a, b in zip(q_frac, d_frac))}")

    save(fig, "panel_5_relaxation.png")


# =====================================================================
# PANEL 6 -- the separation results and the demand plane
# =====================================================================

def panel6(rng):
    fig = new_panel()
    beta, wf = 0.2, 1.0

    # (A) false friends: both-pendant raise
    ax = fig.add_subplot(1, 4, 1)
    cs = np.linspace(0.0, 5.0, 41)
    resp, cent = [], []
    base = C.pendant_pair_structure(beta, n_filler=3, w_filler=wf)
    for c in cs:
        h = C.perturb(base, {frozenset(("u", C.MEDIUM)): float(c),
                             frozenset(("v", C.MEDIUM)): float(c)})
        resp.append(C.distance_to_floor(h, "u", "v"))
        cent.append(C.distance_to_floor(base, "u", "v"))
    # the one-sided attempt, for contrast -- it never moves
    one = [C.distance_to_floor(
        C.perturb(base, {frozenset(("u", C.MEDIUM)): float(c)}), "u", "v")
        for c in cs]
    ax.plot(cs, resp, color=K[1], label="both pendants raised")
    ax.plot(cs, one, color=K[3], ls="--", label="one pendant only")
    ax.plot(cs, cent, color=K[0], label="central demand")
    ax.axvline(wf - beta, color=MUTED, lw=1.0, ls=":")
    ax.set_xlabel("response magnitude $c$")
    ax.set_ylabel(r"$\Delta$")
    ax.set_title("Content pinned, response diverges")
    ax.legend(loc="upper left")
    style2d(ax)
    tag(ax, "A")
    print(f"  P6A both {resp[0]:.4f}->{resp[-1]:.4f} "
          f"one max={max(one):.3e}")

    # (B) converse: bridge vs W
    # Both sweeps are shown against their own range normalised to [0,1],
    # so the two monotonicities can be read off one axis: the bridge
    # rises, the intra-cluster weight falls.
    ax = fig.add_subplot(1, 4, 2)
    dvals = np.linspace(0.0, 2.0, 25)
    bridge = [C.distance_to_floor(
        C.perturb(C.two_cluster_structure(4, 1.0, beta),
                  {frozenset(("a0", "b0")): float(d)}), "a1", "b1")
        for d in dvals]
    Wvals = np.linspace(0.5, 6.0, 25)
    wcurve = [C.distance_to_floor(
        C.two_cluster_structure(4, float(W), beta), "a1", "b1")
        for W in Wvals]
    t = np.linspace(0.0, 1.0, 25)
    ax.plot(t, bridge, color=K[1],
            label=r"bridge $d:0\to2$ (crosses the cut)")
    ax.plot(t, wcurve, color=K[3], ls="--",
            label=r"weight $W:0.5\to6$ (does not)")
    ax.set_xlabel("swept parameter, normalised")
    ax.set_ylabel(r"central $\Delta(a_1,b_1)$")
    ax.set_title("Only the crossed edge drives divergence")
    ax.legend(loc="upper center")
    style2d(ax)
    tag(ax, "B")
    print(f"  P6B bridge {bridge[0]:.4f}->{bridge[-1]:.4f}; "
          f"W {wcurve[0]:.4f}->{wcurve[-1]:.4f}")

    # (C) 3D: the demand plane lifted over the response magnitude
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    # Both families are read on the same two axes: the central demand is
    # measured on the unperturbed structure, the response demand on the
    # structure after that family's response method has been applied.
    #
    # Family A (false friends): the pair is pendant, so the central
    # column is pinned at the floor for every c, while raising both
    # pendants drives the response column up.
    ca = np.linspace(1.5, 5.0, 8)
    fam_a = []
    for c in ca:
        h = C.perturb(base, {frozenset(("u", C.MEDIUM)): float(c),
                             frozenset(("v", C.MEDIUM)): float(c)})
        fam_a.append((C.distance_to_floor(base, "u", "v"),
                      C.distance_to_floor(h, "u", "v"), float(c)))
    # Family B (converse): the bridge drives the central column up, and
    # the response method is the null perturbation -- it finds nothing to
    # raise, so the response column stays where the content put it.
    db = np.linspace(0.25, 2.0, 8)
    fam_b = []
    for d in db:
        g = C.perturb(C.two_cluster_structure(4, 1.0, beta),
                      {frozenset(("a0", "b0")): float(d)})
        gr = C.pendant_pair_structure(beta, n_filler=3, w_filler=wf)
        fam_b.append((C.distance_to_floor(g, "a1", "b1"),
                      C.distance_to_floor(gr, "u", "v"), float(d)))
    A = np.array(fam_a)
    B = np.array(fam_b)
    ax.plot(A[:, 0], A[:, 1], A[:, 2], color=K[1], marker="o", ms=4)
    ax.plot(B[:, 0], B[:, 1], B[:, 2], color=K[0], marker="s", ms=4)
    ax.set_xlabel(r"$D^{\mathrm{central}}$")
    ax.set_ylabel(r"$D^{\mathrm{response}}$")
    ax.set_zlabel("family index")
    ax.set_title("Two families on two axes")
    ax.view_init(20, -126)
    style3d(ax)
    tag(ax, "C", three_d=True)
    print(f"  P6C A central max={A[:, 0].max():.3e}; "
          f"B response max={B[:, 1].max():.3e}")

    # (D) the plane with the four verdict quadrants
    ax = fig.add_subplot(1, 4, 4)
    theta = 0.02
    ax.scatter(A[:, 0], A[:, 1], s=44, color=K[1], marker="o",
               edgecolor="white", linewidth=0.6, label="false friends")
    ax.scatter(B[:, 0], B[:, 1], s=44, color=K[0], marker="s",
               edgecolor="white", linewidth=0.6, label="converse")
    # a sampled population for the interior
    for _ in range(90):
        g = C.random_contact_structure(rng, 5)
        items = sorted(g.items())
        e = sorted(g.weights, key=lambda e: tuple(sorted(e)))[0]
        h = C.perturb(g, {e: float(rng.uniform(0, 1.5))})
        ax.scatter(C.distance_to_floor(g, items[0], items[1]),
                   C.distance_to_floor(h, items[0], items[1]),
                   s=10, color=MUTED, alpha=0.4, linewidth=0)
    ax.axvline(theta, color=K[4], lw=1.4, ls="--")
    ax.axhline(theta, color=K[4], lw=1.4, ls="--")
    ax.set_xlabel(r"$D^{\mathrm{central}}$")
    ax.set_ylabel(r"$D^{\mathrm{response}}$")
    ax.set_title("No single column separates the plane")
    ax.legend(loc="upper right")
    style2d(ax)
    tag(ax, "D")

    save(fig, "panel_6_separation.png")


# =====================================================================
# PANEL 7 -- executable semantics
# =====================================================================

def rand_dna(rng, n):
    return "".join(rng.choice(list("ACGT"), size=n))


def panel7(rng):
    fig = new_panel()

    # (A) FFT and naive cross-correlation agree
    ax = fig.add_subplot(1, 4, 1)
    errs = []
    for _ in range(60):
        t = S.channelise_dna(rand_dna(rng, int(rng.integers(120, 260))))
        q = S.channelise_dna(rand_dna(rng, int(rng.integers(8, 24))))
        a = S.xcorr_naive(q, t)
        b = S.xcorr_fft(q, t)
        if len(a):
            errs.append(float(np.abs(a - b).max()))
    errs = np.array(errs)
    ax.hist(np.log10(np.maximum(errs, 1e-18)), bins=30, color=K[0],
            edgecolor="white", linewidth=0.4)
    ax.set_xlabel(r"$\log_{10}$ max abs difference")
    ax.set_ylabel("count")
    ax.set_title("Two implementations, one denotation")
    style2d(ax)
    tag(ax, "A")
    print(f"  P7A max err={errs.max():.3e}")

    # (B) planted motif recovery
    ax = fig.add_subplot(1, 4, 2)
    motif = rand_dna(rng, 16)
    for idx, nrep in enumerate([1, 2, 3]):
        bg = list(rand_dna(rng, 400))
        pos = [70, 190, 310][:nrep]
        for p in pos:
            bg[p:p + len(motif)] = list(motif)
        t = S.channelise_dna("".join(bg))
        q = S.channelise_dna(motif)
        s = S.xcorr_fft(q, t)
        ax.plot(s, color=K[idx], lw=1.2, alpha=0.85,
                label=f"{nrep} planted")
    ax.set_xlabel("offset")
    ax.set_ylabel("normalised correlation")
    ax.set_title("Planted sites recovered by score")
    ax.legend()
    style2d(ax)
    tag(ax, "B")

    # (C) 3D: what the embedding is, and is not, sensitive to.
    #
    # The embedding does NOT separate base composition: measured below,
    # the spread within a composition family is several times the
    # distance between family means, so a family scatter would be three
    # interleaved clouds. It IS sensitive to progressive corruption of a
    # reference. Both facts are computed here and the ratio is printed
    # into the axis, so the panel reports the null rather than asserting
    # a separation it does not have.
    ax = fig.add_subplot(1, 4, 3, projection="3d")

    # composition families -- measured, then summarised as one number
    fams, cents, spreads = {"AT-rich": "AT", "random": None, "GC-rich": "GC"}, [], []
    for nm, bias in fams.items():
        P = []
        for _ in range(60):
            if bias is None:
                s = rand_dna(rng, 200)
            else:
                pool = list(bias * 4 + "ACGT")
                s = "".join(rng.choice(pool, size=200))
            P.append(S.spectral(S.cardinal(s)[:, 0], 12))
        P = np.array(P)
        cents.append(P.mean(axis=0))
        spreads.append(float(np.mean(np.linalg.norm(P - P.mean(axis=0), axis=1))))
    between = float(np.mean([np.linalg.norm(cents[i] - cents[j])
                             for i in range(3) for j in range(i + 1, 3)]))
    ratio = between / float(np.mean(spreads))

    # Corrupted replicates as a cloud per rate, plotted in the frame of
    # their own reference: the displacement from the uncorrupted
    # embedding. Individual trajectories cross and overlap into a
    # scribble; the quantity that carries the result is how far the
    # cloud spreads from the origin as the rate rises.
    rates_c = np.linspace(0.0, 0.5, 6)
    for rate in rates_c:
        dx, dy = [], []
        for _ in range(26):
            ref_t = rand_dna(rng, 200)
            q_t = S.spectral(S.cardinal(ref_t)[:, 0], 12)
            s = list(ref_t)
            for k in range(len(s)):
                if rng.random() < rate:
                    s[k] = str(rng.choice(list("ACGT")))
            e = S.spectral(S.cardinal("".join(s))[:, 0], 12)
            dx.append(float(e[0] - q_t[0]))
            dy.append(float(e[1] - q_t[1]))
        ax.scatter(dx, dy, np.full(len(dx), rate), s=13, color=K[0],
                   alpha=0.75, edgecolor="white", linewidth=0.25)
    ax.scatter([0], [0], [0], s=46, color=K[1], marker="*",
               edgecolor="white", linewidth=0.4, zorder=6)
    ax.set_xlabel(r"$\Delta$ coeff 1")
    ax.set_ylabel(r"$\Delta$ coeff 2")
    ax.set_zlabel("substitution rate")
    ax.set_title(f"Sensitive to divergence, not to\ncomposition "
                 f"(between/within $={ratio:.2f}$)")
    ax.view_init(20, -132)
    style3d(ax)
    tag(ax, "C", three_d=True)
    print(f"  P7C between/within = {ratio:.3f} (<1 means no separation)")

    # (D) shader distance against divergence.
    #
    # 300 draws per rate over 30 independent references: at 40 draws off
    # one reference the curve is dominated by that reference's own
    # accidents and is not monotone. Stopped at 0.5 -- beyond it the
    # sequence is closer to unrelated than to the reference and the
    # statistic saturates.
    ax = fig.add_subplot(1, 4, 4)
    rates = np.linspace(0.0, 0.5, 11)
    lo, med, hi = [], [], []
    for rate in rates:
        d = []
        for _ in range(30):
            ref = rand_dna(rng, 200)
            q = S.spectral(S.cardinal(ref)[:, 0], 16)
            bank = []
            for _ in range(10):
                s = list(ref)
                for k in range(len(s)):
                    if rng.random() < rate:
                        s[k] = str(rng.choice(list("ACGT")))
                bank.append(S.spectral(S.cardinal("".join(s))[:, 0], 16))
            d.extend(S.shader_distance(np.array(bank), q).tolist())
        d = np.array(d)
        lo.append(np.percentile(d, 25))
        med.append(np.median(d))
        hi.append(np.percentile(d, 75))
    ax.fill_between(rates, lo, hi, color=K[3], alpha=0.18, linewidth=0)
    ax.plot(rates, med, color=K[3], marker="o", ms=4)
    ax.axhline(0.0, color=K[1], lw=1.4, ls="--")
    ax.set_xlabel("per-base substitution rate")
    ax.set_ylabel(r"shader distance $1-Bq$")
    ax.set_title("Distance grows with divergence")
    style2d(ax)
    tag(ax, "D")
    infl = sum(1 for i in range(1, len(med)) if med[i] < med[i - 1])
    print(f"  P7D med {med[0]:.4f} -> {med[-1]:.4f}, inversions {infl}/{len(med)-1}")

    save(fig, "panel_7_semantics.png")


# =====================================================================
# PANEL 8 -- the language and the open problem
# =====================================================================

import experiments as X  # noqa: E402


def panel8(rng):
    fig = new_panel()

    # (A) the checker's response to progressive corruption.
    #
    # A bar per fixed corpus would be five counts -- a table drawn as
    # bars. Instead the valid programs are corrupted by deleting k
    # non-blank lines and the acceptance rate is swept over k. The
    # curve has to start at 0 (the corpus is valid) and saturate, and
    # the composition of the refusals has to shift from semantic to
    # syntactic as more structure is destroyed. Both are measured.
    ax = fig.add_subplot(1, 4, 1)
    ks = list(range(0, 7))
    rej_frac, sem_frac = [], []
    classes = {}
    for k in ks:
        rej = tot = sem = 0
        for _, src in X.POSITIVE:
            for _ in range(25):
                lines = src.split("\n")
                idx = [i for i, ln in enumerate(lines) if ln.strip()]
                kk = min(k, len(idx))
                drop = (set(rng.choice(idx, size=kk, replace=False))
                        if kk else set())
                mod = "\n".join(ln for i, ln in enumerate(lines)
                                if i not in drop)
                tot += 1
                if not L.accepts(mod)[0]:
                    rej += 1
                    try:
                        L.check(mod)
                    except Exception as ex:
                        nm = type(ex).__name__
                        classes[nm] = classes.get(nm, 0) + 1
                        if nm != "ParseError":
                            sem += 1
        rej_frac.append(rej / tot)
        sem_frac.append(sem / rej if rej else 0.0)
    ax.plot(ks, rej_frac, color=K[1], marker="o", ms=5, label="rejected")
    ax.plot(ks, sem_frac, color=K[0], marker="s", ms=4, ls="--",
            label="refused on meaning,\nnot on syntax")
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("lines deleted from a valid program")
    ax.set_ylabel("fraction")
    ax.set_title("Refusals are specific, not blanket")
    ax.legend(fontsize=6, loc="center right", frameon=False)
    style2d(ax)
    tag(ax, "A")
    print(f"  P8A reject {rej_frac[0]:.2f} -> {rej_frac[-1]:.2f}; "
          f"semantic share {sem_frac[1]:.2f} -> {sem_frac[-1]:.2f}")

    # the fixed negative corpus still supplies the error-class breakdown
    neg_rej = neg_right = 0
    for name, src, exp in X.NEGATIVE:
        if not L.accepts(src)[0]:
            neg_rej += 1
            try:
                L.check(src)
            except Exception as ex:
                if isinstance(ex, exp):
                    neg_right += 1
    print(f"  P8A negatives {neg_rej}/{len(X.NEGATIVE)} "
          f"right-class {neg_right}")

    # (B) which error classes fire
    ax = fig.add_subplot(1, 4, 2)
    items = sorted(classes.items(), key=lambda kv: -kv[1])
    ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1],
            color=K[0], edgecolor="white", linewidth=0.6)
    ax.set_xlabel("corrupted programs rejected")
    ax.set_title("Each theorem has its own refusal")
    ax.tick_params(axis="y", labelsize=6.5)
    style2d(ax)
    tag(ax, "B")

    # (C) 3D: response-independence discrepancy surface
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    mags = np.linspace(0.05, 0.6, 8)
    sizes = np.arange(4, 8)
    Z = np.zeros((len(mags), len(sizes)))
    for i, m in enumerate(mags):
        for j, n in enumerate(sizes):
            acc = []
            for _ in range(18):
                g = C.random_contact_structure(rng, int(n))
                for v in sorted(g.items()):
                    inc = [e for e in sorted(g.incident(v),
                                             key=lambda e: tuple(sorted(e)))
                           if C.MEDIUM not in e]
                    if len(inc) < 2:
                        continue
                    others = [u for u in sorted(g.items()) if u != v]
                    if not others:
                        continue
                    w = others[0]
                    d1 = C.distance_to_floor(
                        C.perturb(g, {inc[0]: float(m)}), v, w)
                    d2 = C.distance_to_floor(
                        C.perturb(g, {inc[1]: float(m)}), v, w)
                    acc.append(abs(d1 - d2))
                    break
            Z[i, j] = np.mean(acc) if acc else 0.0
    Xg, Yg = np.meshgrid(sizes, mags)
    ax.plot_surface(Xg, Yg, Z, cmap="Reds", edgecolor="white",
                    linewidth=0.25, alpha=0.95)
    ax.set_xlabel("items")
    ax.set_ylabel("perturbation magnitude")
    ax.set_zlabel("mean discrepancy")
    ax.set_title("Response dependence does not vanish")
    ax.view_init(24, -128)
    style3d(ax)
    tag(ax, "C", three_d=True)
    print(f"  P8C Z {Z.min():.5f}..{Z.max():.5f}")

    # (D) the discrepancy distribution against theta
    ax = fig.add_subplot(1, 4, 4)
    theta = 0.01
    ds = []
    for _ in range(200):
        g = C.random_contact_structure(rng, int(rng.integers(4, 6)))
        for v in sorted(g.items()):
            inc = [e for e in sorted(g.incident(v),
                                     key=lambda e: tuple(sorted(e)))
                   if C.MEDIUM not in e]
            if len(inc) < 2:
                continue
            others = [u for u in sorted(g.items()) if u != v]
            if not others:
                continue
            w = others[0]
            d1 = C.distance_to_floor(C.perturb(g, {inc[0]: 0.10}), v, w)
            d2 = C.distance_to_floor(C.perturb(g, {inc[1]: 0.10}), v, w)
            ds.append(abs(d1 - d2))
    ds = np.array(ds)
    ax.hist(ds, bins=40, color=K[1], edgecolor="white", linewidth=0.4)
    ax.axvline(theta, color=K[0], lw=2.0, ls="--")
    ax.set_xlabel("|demand discrepancy|")
    ax.set_ylabel("count")
    ax.set_title(r"The assumption fails at $\theta=0.01$")
    style2d(ax)
    tag(ax, "D")
    print(f"  P8D n={len(ds)} within={float((ds <= theta).mean()):.4f} "
          f"mean={ds.mean():.5f} max={ds.max():.5f}")

    save(fig, "panel_8_language_open.png")


def main():
    rng = np.random.default_rng(SEED)
    for fn in [panel1, panel2, panel3, panel4, panel5, panel6, panel7,
               panel8]:
        fn(rng)


if __name__ == "__main__":
    main()
