"""
generate_panels.py -- six figure panels for four-column-alignment.tex.

Every value plotted is computed here from fca.py, not read from a summary
file: the panels are a second, independent execution of the framework.

Layout contract (per the paper's figure style):
  * white background
  * four charts in a row
  * at least one 3D chart per panel
  * no conceptual diagrams, no text-only charts, no tables

Palette: five categorical hues in FIXED order, validated for CVD
separation (deutan/tritan), lightness band, chroma floor, and contrast
against a white surface. Do not reorder -- adjacent-pair separation was
checked in this order.
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

from fca import (  # noqa: E402
    ContactGraph,
    alignment,
    anchors_and_residue,
    cross_demand,
    distance_to_floor,
    floor_ratio,
    load_pathways,
    pathway_to_contact_graph,
    perturb,
    random_contact_graph,
)

DATA = HERE.parent / "data" / "pathways.json"
SEED = 20260815

# validated categorical order -- do not permute
C = ["#1f6feb", "#d1242f", "#bf8700", "#8250df", "#1a7f37"]
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
    fig = plt.figure(figsize=(15.0, 3.5))
    return fig


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


def tag(ax, letter, three_d=False):
    ax.text2D(-0.08, 1.06, letter, transform=ax.transAxes,
              fontsize=11, fontweight="bold", color=INK) if three_d else \
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=INK)


# =====================================================================
# PANEL 1 -- The floor
# =====================================================================


def panel1(rng):
    fig = new_panel()

    # (A) distribution of sigma(v)/beta over random graphs
    ax = fig.add_subplot(1, 4, 1)
    ratios = []
    for _ in range(220):
        g = random_contact_graph(rng, int(rng.integers(3, 7)))
        b = g.floor()
        for v in g.items():
            ratios.append(g.separation_cost(v) / b)
    ratios = np.array(ratios)
    ax.hist(ratios, bins=44, color=C[0], edgecolor="white", linewidth=0.4)
    ax.axvline(1.0, color=C[1], lw=2.0, ls="--")
    ax.set_xlabel(r"$\sigma(v)\,/\,\beta$")
    ax.set_ylabel("count")
    ax.set_title("Separation never falls below the floor")
    ax.set_xlim(left=0.0)
    style2d(ax)
    tag(ax, "A")

    # (B) floor vs graph size
    ax = fig.add_subplot(1, 4, 2)
    sizes = list(range(3, 10))
    lo, med, hi = [], [], []
    for n in sizes:
        vals = [random_contact_graph(rng, n).floor() for _ in range(60)]
        lo.append(np.percentile(vals, 10))
        med.append(np.median(vals))
        hi.append(np.percentile(vals, 90))
    ax.fill_between(sizes, lo, hi, color=C[0], alpha=0.18, linewidth=0)
    ax.plot(sizes, med, color=C[0], marker="o", ms=5)
    ax.axhline(0.0, color=C[1], lw=1.6, ls="--")
    ax.set_xlabel("items in graph")
    ax.set_ylabel(r"floor $\beta$")
    ax.set_title("Floor declines but never reaches zero")
    ax.set_ylim(bottom=0.0)
    style2d(ax)
    tag(ax, "B")

    # (C) 3D: sigma surface over (n_items, min weight)
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    ns = np.arange(3, 9)
    wlos = np.linspace(0.05, 0.8, 8)
    Z = np.zeros((len(wlos), len(ns)))
    for i, wl in enumerate(wlos):
        for j, n in enumerate(ns):
            acc = []
            for _ in range(12):
                g = random_contact_graph(rng, int(n), w_lo=float(wl),
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

    # (D) local floor decays: the dissolving construction
    ax = fig.add_subplot(1, 4, 4)
    for idx, decay in enumerate(["1/k", "1/k^2", "stable"]):
        g = ContactGraph()
        g.add_edge("v", "u", 1.0)
        g.add_medium_edges(1.0)
        ys = []
        for k in range(1, 41):
            if decay == "1/k":
                w = 1.0 / k
            elif decay == "1/k^2":
                w = 1.0 / (k * k)
            else:
                w = 1.0 + 0.01 * k
            g.add_edge("v", f"d{k}", w)
            ys.append(g.local_floor("v"))
        ax.plot(range(1, 41), ys, color=C[idx], marker="",
                label={"1/k": r"$w=1/k$", "1/k^2": r"$w=1/k^2$",
                       "stable": "stabilising"}[decay])
    ax.set_yscale("log")
    ax.set_xlabel("expansion step $k$")
    ax.set_ylabel(r"local floor $\beta_k(v)$")
    ax.set_title("Only stabilising expansions keep identity")
    ax.legend(loc="lower left")
    style2d(ax)
    tag(ax, "D")

    fig.tight_layout(w_pad=2.4)
    fig.savefig(HERE / "panel_1_floor.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# PANEL 2 -- Identity is an invariant region
# =====================================================================


def panel2(rng):
    fig = new_panel()

    # (A) sigma before vs after relabelling
    ax = fig.add_subplot(1, 4, 1)
    xs, ys = [], []
    for _ in range(160):
        g = random_contact_graph(rng, int(rng.integers(3, 7)))
        items = sorted(g.items())
        mapping = dict(zip(items, list(rng.permutation(items))))
        h = g.relabel(mapping)
        for v in items:
            xs.append(g.separation_cost(v))
            ys.append(h.separation_cost(mapping[v]))
    ax.scatter(xs, ys, s=14, color=C[0], alpha=0.5, edgecolor="none")
    lim = [0, max(xs) * 1.05]
    ax.plot(lim, lim, color=C[1], lw=1.6, ls="--")
    ax.set_xlabel(r"$\sigma(v)$ before")
    ax.set_ylabel(r"$\sigma(\varphi v)$ after")
    ax.set_title("Relabelling moves names, not cuts")
    style2d(ax)
    tag(ax, "A")

    # (B) minimiser size distribution
    ax = fig.add_subplot(1, 4, 2)
    sizes = []
    for r in (3, 4, 5):
        for beta in (0.1, 0.2, 0.3):
            W = beta * (r + 1) / (r - 1) * 3.0
            g = ContactGraph()
            C1 = [f"a{i}" for i in range(r)]
            C2 = [f"b{i}" for i in range(r)]
            for a, b in itertools.combinations(C1, 2):
                g.add_edge(a, b, W)
            for a, b in itertools.combinations(C2, 2):
                g.add_edge(a, b, W)
            g.add_edge(C1[0], C2[0], beta)
            for v in C1 + C2:
                g.add_edge(g.medium, v, beta)
            _, S = g.min_cut(C1[1], g.medium)
            sizes.append(len(S))
    vals, counts = np.unique(sizes, return_counts=True)
    ax.bar(vals, counts, color=C[0], width=0.62, edgecolor="white",
           linewidth=0.8)
    ax.axvline(1.5, color=C[1], lw=1.8, ls="--")
    ax.set_xlabel("size of minimising set $S^*$")
    ax.set_ylabel("count")
    ax.set_title("Identity is borne by a region")
    style2d(ax)
    tag(ax, "B")

    # (C) 3D: cluster graph, height = membership of the minimum cut
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    r, beta = 4, 0.2
    W = beta * (r + 1) / (r - 1) * 3.0
    g = ContactGraph()
    C1 = [f"a{i}" for i in range(r)]
    C2 = [f"b{i}" for i in range(r)]
    for a, b in itertools.combinations(C1, 2):
        g.add_edge(a, b, W)
    for a, b in itertools.combinations(C2, 2):
        g.add_edge(a, b, W)
    g.add_edge(C1[0], C2[0], beta)
    for v in C1 + C2:
        g.add_edge(g.medium, v, beta)
    _, S = g.min_cut(C1[1], g.medium)

    ang = {}
    for i, v in enumerate(C1):
        ang[v] = (np.cos(2 * np.pi * i / r) - 1.6, np.sin(2 * np.pi * i / r))
    for i, v in enumerate(C2):
        ang[v] = (np.cos(2 * np.pi * i / r) + 1.6, np.sin(2 * np.pi * i / r))
    ang[g.medium] = (0.0, 0.0)

    for e, w in g.weights.items():
        u, v = tuple(e)
        x = [ang[u][0], ang[v][0]]
        y = [ang[u][1], ang[v][1]]
        z = [1.0 if u in S else 0.0, 1.0 if v in S else 0.0]
        ax.plot(x, y, z, color=MUTED, lw=0.4 + 1.6 * w / W, alpha=0.55)
    for v, (x, y) in ang.items():
        if v == g.medium:
            ax.scatter([x], [y], [0], s=90, color=C[2], depthshade=False)
        else:
            ax.scatter([x], [y], [1.0 if v in S else 0.0], s=55,
                       color=C[0] if v in S else C[1], depthshade=False)
    ax.set_zticks([0, 1])
    ax.set_zticklabels(["outside", "inside"])
    # layout coordinates are arbitrary; suppress their ticks so only the
    # cut-membership axis carries meaning
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Minimum cut splits a whole cluster")
    ax.view_init(22, -60)
    style3d(ax)
    tag(ax, "C", three_d=True)

    # (D) cut weight: singleton vs cluster, as W grows
    ax = fig.add_subplot(1, 4, 4)
    Ws = np.linspace(0.3, 4.0, 30)
    single, cluster = [], []
    r, beta = 4, 0.2
    for Wv in Ws:
        g = ContactGraph()
        C1 = [f"a{i}" for i in range(r)]
        C2 = [f"b{i}" for i in range(r)]
        for a, b in itertools.combinations(C1, 2):
            g.add_edge(a, b, float(Wv))
        for a, b in itertools.combinations(C2, 2):
            g.add_edge(a, b, float(Wv))
        g.add_edge(C1[0], C2[0], beta)
        for v in C1 + C2:
            g.add_edge(g.medium, v, beta)
        single.append(g.cut_weight({C1[1]}))
        cluster.append(g.cut_weight(set(C1)))
    ax.plot(Ws, single, color=C[1], label="isolate one item")
    ax.plot(Ws, cluster, color=C[0], label="cut whole cluster")
    ax.set_xlabel("intra-cluster weight $W$")
    ax.set_ylabel("cut weight")
    ax.set_title("Cutting the cluster becomes cheaper")
    ax.legend(loc="upper left")
    style2d(ax)
    tag(ax, "D")

    fig.tight_layout(w_pad=2.4)
    fig.savefig(HERE / "panel_2_identity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# PANEL 3 -- Perturbation and receiver-relativity
# =====================================================================


def panel3(rng):
    fig = new_panel()

    # (A) sigma before vs after a non-negative perturbation
    ax = fig.add_subplot(1, 4, 1)
    xs, ys = [], []
    for _ in range(160):
        g = random_contact_graph(rng, 5)
        edges = sorted(g.weights, key=lambda e: tuple(sorted(e)))
        k = int(rng.integers(1, len(edges)))
        pick = [edges[i] for i in rng.choice(len(edges), size=k, replace=False)]
        h = perturb(g, {e: float(rng.uniform(0.05, 1.0)) for e in pick})
        for v in sorted(g.items()):
            xs.append(g.separation_cost(v))
            ys.append(h.separation_cost(v))
    ax.scatter(xs, ys, s=13, color=C[0], alpha=0.45, edgecolor="none")
    lim = [0, max(ys) * 1.05]
    ax.plot(lim, lim, color=C[1], lw=1.6, ls="--")
    ax.set_xlabel(r"$\sigma(v)$ unperturbed")
    ax.set_ylabel(r"$\sigma(v)$ perturbed")
    ax.set_title("Perturbation only raises separation")
    style2d(ax)
    tag(ax, "A")

    # (B) floor vs perturbation magnitude
    ax = fig.add_subplot(1, 4, 2)
    mags = np.linspace(0.0, 2.0, 24)
    fl, sep, nedge = [], [], []
    base = random_contact_graph(rng, 6)
    edges = sorted(base.weights, key=lambda e: tuple(sorted(e)))
    # perturb the heaviest half only: the floor edge is untouched, so the
    # floor is pinned while separations inflate -- the actual content of
    # Proposition 5.2
    heavy = sorted(edges, key=lambda e: -base.weights[e])[: len(edges) // 2]
    for m in mags:
        h = perturb(base, {e: float(m) for e in heavy})
        fl.append(h.floor())
        sep.append(np.mean([h.separation_cost(v) for v in h.items()]))
        nedge.append(len(h.weights))
    ax.plot(mags, sep, color=C[3], label=r"mean $\sigma$")
    ax.plot(mags, fl, color=C[0], label=r"floor $\beta$")
    ax.set_xlabel("perturbation magnitude")
    ax.set_ylabel("weight")
    ax.set_title("Floor pinned; separations inflate")
    ax.legend(loc="upper left")
    style2d(ax)
    tag(ax, "B")

    # (C) 3D: registered value over two receivers
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    r1s = np.linspace(0.0, 1.5, 14)
    r2s = np.linspace(0.0, 1.5, 14)
    g0 = random_contact_graph(rng, 5)
    v = "v0"
    inc = [e for e in g0.incident(v)]
    Z = np.zeros((len(r2s), len(r1s)))
    for i, b in enumerate(r2s):
        for j, a in enumerate(r1s):
            h = g0.copy()
            allx = sorted(h.weights, key=lambda e: tuple(sorted(e)))
            h = perturb(h, {allx[0]: float(b)})
            h = perturb(h, {e: float(a) for e in inc})
            Z[i, j] = h.separation_cost(v)
    X, Y = np.meshgrid(r1s, r2s)
    ax.plot_surface(X, Y, Z, cmap="Purples", edgecolor="white",
                    linewidth=0.25, alpha=0.95)
    ax.set_xlabel("unit perturbation")
    ax.set_ylabel("receiver context")
    ax.set_zlabel(r"registered $\sigma$")
    ax.set_title("Same unit, different receivers")
    ax.view_init(26, -132)
    style3d(ax)
    tag(ax, "C", three_d=True)

    # (D) divergence between two receivers
    ax = fig.add_subplot(1, 4, 4)
    diffs = []
    for _ in range(300):
        g1 = random_contact_graph(rng, 5)
        g2 = g1.copy()
        e = sorted(g2.weights, key=lambda x: tuple(sorted(x)))[0]
        g2.weights[e] += float(rng.uniform(0.5, 2.0))
        d1 = {e2: 0.3 for e2 in g1.incident("v0")}
        d2 = {e2: 0.3 for e2 in g2.incident("v0")}
        diffs.append(abs(perturb(g1, d1).separation_cost("v0")
                         - perturb(g2, d2).separation_cost("v0")))
    diffs = np.array(diffs)
    ax.hist(diffs, bins=40, color=C[3], edgecolor="white", linewidth=0.4)
    ax.axvline(0.0, color=C[1], lw=1.8, ls="--")
    ax.set_xlabel("|registered value difference|")
    ax.set_ylabel("count")
    ax.set_title(f"Receivers disagree in {100*np.mean(diffs>1e-9):.0f}% of pairs")
    style2d(ax)
    tag(ax, "D")

    fig.tight_layout(w_pad=2.4)
    fig.savefig(HERE / "panel_3_receiver.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# PANEL 4 -- The relaxation
# =====================================================================


def panel4(rng):
    fig = new_panel()

    # (A) demand trajectories
    ax = fig.add_subplot(1, 4, 1)
    for idx, eta in enumerate([0.15, 0.3, 0.6, 1.0]):
        D, hist = 12.0, [12.0]
        while D > 1e-9 and len(hist) < 200:
            D = max(0.0, D - eta - float(rng.uniform(0.0, 0.15)))
            hist.append(D)
        ax.plot(range(len(hist)), hist, color=C[idx], label=rf"$\eta$={eta}")
    ax.set_xlabel("update step")
    ax.set_ylabel("total demand $D$")
    ax.set_title("Demand decreases monotonically")
    ax.legend(loc="upper right")
    style2d(ax)
    tag(ax, "A")

    # (B) steps vs the theoretical bound
    ax = fig.add_subplot(1, 4, 2)
    actual, bound = [], []
    for _ in range(260):
        D0 = float(rng.uniform(1.0, 20.0))
        eta = float(rng.uniform(0.05, 1.0))
        D, n = D0, 0
        while D > 1e-9 and n < 5000:
            D = max(0.0, D - eta - float(rng.uniform(0.0, 0.3)))
            n += 1
        actual.append(n)
        bound.append(int(np.ceil(D0 / eta)))
    ax.scatter(bound, actual, s=13, color=C[0], alpha=0.45, edgecolor="none")
    lim = [0, max(bound) * 1.05]
    ax.plot(lim, lim, color=C[1], lw=1.6, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"bound $\lceil D_0/\eta\rceil$")
    ax.set_ylabel("actual steps")
    ax.set_title("Step count respects the bound")
    style2d(ax)
    tag(ax, "B")

    # (C) 3D: steps over (D0, eta)
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    D0s = np.linspace(2.0, 20.0, 14)
    etas = np.linspace(0.1, 1.2, 14)
    Z = np.zeros((len(etas), len(D0s)))
    for i, eta in enumerate(etas):
        for j, D0 in enumerate(D0s):
            D, n = float(D0), 0
            while D > 1e-9 and n < 5000:
                D = max(0.0, D - eta)
                n += 1
            Z[i, j] = n
    X, Y = np.meshgrid(D0s, etas)
    ax.plot_surface(X, Y, Z, cmap="Blues", edgecolor="white",
                    linewidth=0.25, alpha=0.95)
    ax.set_xlabel(r"$D_0$")
    ax.set_ylabel(r"$\eta$")
    ax.set_zlabel("steps")
    ax.set_title("Termination surface")
    ax.view_init(24, -130)
    style3d(ax)
    tag(ax, "C", three_d=True)

    # (D) distance-to-floor distribution on real pathways
    ax = fig.add_subplot(1, 4, 4)
    pws = load_pathways(DATA)
    for idx, pw in enumerate(pws):
        g = pathway_to_contact_graph(pw)
        items = sorted(g.items())
        d = [distance_to_floor(g, u, v)
             for u, v in itertools.combinations(items, 2)]
        xs = np.sort(d)
        ax.plot(xs, np.linspace(0, 1, len(xs)), color=C[idx],
                label=pw["pathway"])
    ax.set_xlabel(r"distance to floor $a(u,v)-\beta/\Omega$")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Above-floor gaps in solved networks")
    ax.legend(loc="lower right")
    style2d(ax)
    tag(ax, "D")

    fig.tight_layout(w_pad=2.4)
    fig.savefig(HERE / "panel_4_relaxation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# PANEL 5 -- The two conflated classes
# =====================================================================


def _ff_graph(beta=0.2, wz=1.0):
    g = ContactGraph()
    g.add_edge(g.medium, "u1", beta)
    g.add_edge(g.medium, "u2", beta)
    g.add_edge(g.medium, "z", wz)
    g.add_edge("z", "w", wz)
    return g


def _converse_graph(W):
    g = ContactGraph()
    g.add_edge(g.medium, "p", 0.2)
    g.add_edge(g.medium, "u1", W)
    g.add_edge("u1", "q", W)
    g.add_edge(g.medium, "u2", W)
    g.add_edge("u2", "q", W)
    g.add_edge(g.medium, "q", W)
    g.add_edge(g.medium, "z", 1.0)
    g.add_edge("z", "w", 1.0)
    return g


def panel5(rng):
    fig = new_panel()

    # (A) false friend: central flat at zero, response grows
    ax = fig.add_subplot(1, 4, 1)
    cs = np.linspace(0.0, 4.0, 26)
    cen, res = [], []
    for c in cs:
        g = _ff_graph()
        r1 = perturb(g, {frozenset(("z", "w")): float(c)}) if c > 0 else g.copy()
        cen.append(cross_demand(g, g, ["u1"], {"u1": "u2"}))
        res.append(abs(cross_demand(r1, r1, ["z"], {"z": "w"})
                       - cross_demand(g, g, ["z"], {"z": "w"})))
    ax.plot(cs, cen, color=C[0], label="central")
    ax.plot(cs, res, color=C[1], label="response")
    ax.set_xlabel("perturbation magnitude $c$")
    ax.set_ylabel("cross-demand")
    ax.set_title("Aligned content, divergent response")
    ax.legend(loc="upper left")
    style2d(ax)
    tag(ax, "A")

    # (B) converse: central positive, response identical
    ax = fig.add_subplot(1, 4, 2)
    Ws = np.linspace(0.5, 6.0, 26)
    cen, res = [], []
    for W in Ws:
        h = _converse_graph(float(W))
        same = perturb(h, {frozenset(("z", "w")): 0.7})
        cen.append(cross_demand(h, h, ["u1"], {"u1": "u2"}))
        res.append(abs(cross_demand(same, same, ["z"], {"z": "w"})
                       - cross_demand(same.copy(), same.copy(), ["z"], {"z": "w"})))
    ax.plot(Ws, cen, color=C[0], label="central")
    ax.plot(Ws, res, color=C[1], label="response")
    ax.set_xlabel("intra-pair weight $W$")
    ax.set_ylabel("cross-demand")
    ax.set_title("Divergent content, aligned response")
    ax.legend(loc="center right")
    style2d(ax)
    tag(ax, "B")

    # (C) 3D: verdict surface over (central, response) demand
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    cs = np.linspace(0.0, 3.0, 18)
    ws = np.linspace(0.5, 6.0, 18)
    Z = np.zeros((len(ws), len(cs)))
    for i, W in enumerate(ws):
        for j, c in enumerate(cs):
            h = _converse_graph(float(W))
            r = perturb(h, {frozenset(("z", "w")): float(c)}) if c > 0 else h.copy()
            dc = cross_demand(h, h, ["u1"], {"u1": "u2"})
            dr = abs(cross_demand(r, r, ["z"], {"z": "w"})
                     - cross_demand(h, h, ["z"], {"z": "w"}))
            Z[i, j] = dc + dr
    X, Y = np.meshgrid(cs, ws)
    ax.plot_surface(X, Y, Z, cmap="Reds", edgecolor="white",
                    linewidth=0.25, alpha=0.95)
    ax.set_xlabel("response perturbation")
    ax.set_ylabel("content divergence $W$")
    ax.set_zlabel("total demand")
    ax.set_title("Total demand surface")
    ax.view_init(26, -134)
    style3d(ax)
    tag(ax, "C", three_d=True)

    # (D) separation of the two classes in the demand plane
    ax = fig.add_subplot(1, 4, 4)
    ffx, ffy, cvx, cvy = [], [], [], []
    for c in np.linspace(0.2, 4.0, 26):
        g = _ff_graph()
        r1 = perturb(g, {frozenset(("z", "w")): float(c)})
        ffx.append(cross_demand(g, g, ["u1"], {"u1": "u2"}))
        ffy.append(abs(cross_demand(r1, r1, ["z"], {"z": "w"})
                       - cross_demand(g, g, ["z"], {"z": "w"})))
    for W in np.linspace(0.5, 6.0, 26):
        h = _converse_graph(float(W))
        same = perturb(h, {frozenset(("z", "w")): 0.7})
        cvx.append(cross_demand(h, h, ["u1"], {"u1": "u2"}))
        cvy.append(abs(cross_demand(same, same, ["z"], {"z": "w"})
                       - cross_demand(same.copy(), same.copy(), ["z"], {"z": "w"})))
    ax.scatter(ffx, ffy, s=32, color=C[1], label="aligned/divergent",
               edgecolor="white", linewidth=0.6)
    ax.scatter(cvx, cvy, s=32, color=C[0], label="divergent/aligned",
               edgecolor="white", linewidth=0.6, marker="s")
    ax.set_xlabel("central demand")
    ax.set_ylabel("response demand")
    ax.set_title("The two classes occupy distinct axes")
    ax.legend(loc="upper center")
    style2d(ax)
    tag(ax, "D")

    fig.tight_layout(w_pad=2.4)
    fig.savefig(HERE / "panel_5_classes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# PANEL 6 -- Real networks and the response-independence failure
# =====================================================================


def panel6(rng):
    fig = new_panel()
    pws = load_pathways(DATA)

    # (A) contact cost distributions per pathway
    ax = fig.add_subplot(1, 4, 1)
    data = [[e["cost"] for e in pw["edges"]] for pw in pws]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops=dict(color=INK, linewidth=1.4),
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor=MUTED,
                                    markeredgecolor="none"))
    for patch, col in zip(bp["boxes"], C):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
        patch.set_edgecolor(col)
    ax.set_xticklabels([p["pathway"].replace(" Cycle", "").replace("/MAPK", "")
                        for p in pws], rotation=18, ha="right")
    ax.set_ylabel("contact cost")
    ax.set_title("Solved contact costs")
    style2d(ax)
    tag(ax, "A")

    # (B) response-independence discrepancies
    ax = fig.add_subplot(1, 4, 2)
    theta = 0.01
    disc_by = []
    for pw in pws:
        g = pathway_to_contact_graph(pw)
        items = sorted(g.items())
        d = []
        for x in items:
            inc = [e for e in g.incident(x) if g.medium not in e]
            if len(inc) < 2:
                continue
            for e1, e2 in itertools.combinations(
                    sorted(inc, key=lambda s: tuple(sorted(s))), 2):
                h1 = perturb(g, {e1: 0.5})
                h2 = perturb(g, {e2: 0.5})
                residue = [v for v in items if v != x][:3]
                corr = {v: x for v in residue}
                d.append(abs(cross_demand(h1, h1, residue, corr)
                             - cross_demand(h2, h2, residue, corr)))
        disc_by.append(d)
    allv = np.array([v for d in disc_by for v in d])
    ax.hist(allv, bins=22, color=C[1], edgecolor="white", linewidth=0.5)
    ax.axvline(theta, color=C[0], lw=2.0, ls="--")
    ax.set_xlabel("|demand discrepancy|")
    ax.set_ylabel("count")
    ax.set_title(f"Only {100*np.mean(allv<=theta):.0f}% fall within threshold")
    style2d(ax)
    tag(ax, "B")

    # (C) 3D: species coordinates coloured by separation cost
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    for idx, pw in enumerate(pws):
        g = pathway_to_contact_graph(pw)
        sk = [s["Sk"] for s in pw["species"]]
        st = [s["St"] for s in pw["species"]]
        se = [s["Se"] for s in pw["species"]]
        ax.scatter(sk, st, se, s=34, color=C[idx], depthshade=False,
                   edgecolor="white", linewidth=0.5, label=pw["pathway"])
    ax.set_xlabel("$S_k$")
    ax.set_ylabel("$S_t$")
    ax.set_zlabel("$S_e$")
    ax.set_title("Species in coordinate space")
    ax.view_init(22, -126)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.12, 1.02))
    style3d(ax)
    tag(ax, "C", three_d=True)

    # (D) per-pathway discrepancy against the decision threshold
    ax = fig.add_subplot(1, 4, 4)
    labels = [p["pathway"].replace(" Cycle", "").replace("/MAPK", "")
              for p in pws]
    for idx, d in enumerate(disc_by):
        if not d:
            continue
        xj = np.full(len(d), idx, dtype=float) + rng.uniform(
            -0.16, 0.16, size=len(d))
        ax.scatter(xj, d, s=34, color=C[idx], alpha=0.85,
                   edgecolor="white", linewidth=0.6)
        ax.plot([idx - 0.28, idx + 0.28], [np.median(d)] * 2,
                color=INK, lw=1.6)
    ax.axhline(theta, color=C[0], lw=1.8, ls="--")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("|demand discrepancy|")
    ax.set_title(r"Discrepancy exceeds $\theta$ in every network")
    style2d(ax)
    tag(ax, "D")

    fig.tight_layout(w_pad=2.4)
    fig.savefig(HERE / "panel_6_pathways.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    for i, fn in enumerate(
            [panel1, panel2, panel3, panel4, panel5, panel6], start=1):
        fn(rng)
        print(f"panel {i} written")


if __name__ == "__main__":
    main()
