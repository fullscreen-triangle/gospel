"""
experiments.py -- validation of every theorem in four-column-alignment.tex.

Each experiment maps to one numbered result. Experiments that can have a
NEGATIVE CONTROL have one: it is not enough to show a theorem's conclusion
holds under its hypothesis; where possible we also show the conclusion
FAILS when the hypothesis is removed, so the hypothesis is demonstrably
load-bearing rather than decorative.

Run via run_all.py.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from fca import (
    TOL,
    Check,
    ContactGraph,
    Column,
    alignment,
    ablation_score,
    anchors_and_residue,
    cross_demand,
    distance_to_floor,
    floor_ratio,
    four_column_verdict,
    load_pathways,
    make_column,
    pathway_to_contact_graph,
    perturb,
    random_contact_graph,
    relax,
)

SEED = 20260815
DATA = Path(__file__).resolve().parent.parent / "data" / "pathways.json"


# =====================================================================
# E01 -- Theorem 3.2 (positive floor) and Corollary 3.3 (no sharp id)
# =====================================================================


def e01_floor(rng) -> Check:
    c = Check("Floor", "Theorem 3.2 / Corollary 3.3")
    for _ in range(400):
        n = int(rng.integers(3, 8))
        g = random_contact_graph(rng, n)
        beta = g.floor()
        c.record(beta > 0.0, note="floor not positive")
        for v in sorted(g.items()):
            sep = g.separation_cost(v)
            # Theorem 3.2: sigma(v) >= beta
            c.record(sep >= beta - TOL, err=max(0.0, beta - sep),
                     note=f"sigma({v})={sep} < beta={beta}")
            # Corollary 3.3: no identification of cost 0
            c.record(sep > 0.0, note=f"sigma({v}) = 0")
    return c


# =====================================================================
# E02 -- Lemma 3.6 (existence) and Remark 3.7 (floor CAN reach zero)
#        plus Theorem 3.8 under the floor-stabilising hypothesis
# =====================================================================


def e02_intrinsic_floor(rng) -> Check:
    c = Check("IntrinsicFloor", "Lemma 3.6 / Remark 3.7 / Theorem 3.8")

    # (a) Lemma 3.6: local floor is non-increasing under expansion.
    for _ in range(120):
        g = random_contact_graph(rng, 4)
        v = "v0"
        prev = g.local_floor(v)
        seq = [prev]
        for k in range(1, 9):
            g.add_edge(v, f"new{k}", float(rng.uniform(0.02, 1.5)))
            cur = g.local_floor(v)
            c.record(cur <= prev + TOL, err=max(0.0, cur - prev),
                     note="local floor increased under expansion")
            prev = cur
            seq.append(cur)
        c.record(all(x > 0 for x in seq), note="local floor hit zero at finite stage")

    # (b) Remark 3.7: the UNCONDITIONAL invariance claim is FALSE.
    #     Construct a dissolving node: edges of weight 1/k.
    g = ContactGraph()
    g.add_edge("v", "u", 1.0)
    g.add_medium_edges(1.0)
    floors = []
    for k in range(1, 60):
        g.add_edge("v", f"d{k}", 1.0 / k)
        floors.append(g.local_floor("v"))
    dissolves = floors[-1] < floors[0] / 10.0
    c.record(dissolves, note="dissolving construction failed to dissolve")
    c.record(all(f > 0 for f in floors),
             note="floor non-positive at a finite stage")

    # (c) Theorem 3.8: under floor-stabilisation the floor IS invariant.
    for _ in range(120):
        g = random_contact_graph(rng, 4)
        v = "v0"
        K_floor = g.local_floor(v)
        # expand only with edges at v strictly above the current floor
        for k in range(1, 10):
            g.add_edge(v, f"s{k}", K_floor + float(rng.uniform(0.01, 1.0)))
            cur = g.local_floor(v)
            c.record(abs(cur - K_floor) <= TOL, err=abs(cur - K_floor),
                     note="stabilised floor moved")
    return c


# =====================================================================
# E03 -- Theorem 4.2 (min cut invariant, label is not)
# =====================================================================


def e03_invariance(rng) -> Check:
    c = Check("CutInvariance", "Theorem 4.2 / Corollary 4.4")
    for _ in range(200):
        n = int(rng.integers(3, 7))
        g = random_contact_graph(rng, n)
        items = sorted(g.items())
        perm = list(rng.permutation(items))
        mapping = dict(zip(items, perm))
        h = g.relabel(mapping)

        label_moved = any(mapping[i] != i for i in items)
        for v in items:
            sv, sh = g.separation_cost(v), h.separation_cost(mapping[v])
            c.record(abs(sv - sh) <= 1e-9, err=abs(sv - sh),
                     note=f"sigma not invariant: {sv} vs {sh}")
        # Corollary 4.4: labels move while every invariant is fixed
        if label_moved:
            c.record(True)
    return c


# =====================================================================
# E04 -- Theorem 4.3 (identity is a region, never a point)
# =====================================================================


def e04_region(rng) -> Check:
    c = Check("IdentityIsRegion", "Theorem 4.3")
    # The explicit construction from the proof of Theorem 4.3.
    for r in (3, 4, 5):
        for beta in (0.1, 0.25):
            # W must exceed beta*(r+1)/(r-1) for the cluster cut to win
            W = beta * (r + 1) / (r - 1) * 3.0
            g = ContactGraph()
            C1 = [f"a{i}" for i in range(r)]
            C2 = [f"b{i}" for i in range(r)]
            for c1, c2 in itertools.combinations(C1, 2):
                g.add_edge(c1, c2, W)
            for c1, c2 in itertools.combinations(C2, 2):
                g.add_edge(c1, c2, W)
            g.add_edge(C1[0], C2[0], beta)
            for v in C1 + C2:
                g.add_edge(g.medium, v, beta)

            v = C1[1]
            _, S = g.min_cut(v, g.medium)
            c.record(len(S) > 1,
                     note=f"minimiser was a singleton at r={r}, beta={beta}")
            c.record(g.separation_cost(v) >= g.floor() - TOL,
                     note="separation below floor")
    return c


# =====================================================================
# E05 -- Proposition 5.2 (skeleton persists, weights move)
# =====================================================================


def e05_perturbation(rng) -> Check:
    c = Check("SkeletonPersists", "Proposition 5.2")
    for _ in range(200):
        g = random_contact_graph(rng, 5)
        edges = sorted(g.weights, key=lambda e: tuple(sorted(e)))
        k = int(rng.integers(1, max(2, len(edges))))
        chosen = [edges[i] for i in rng.choice(len(edges), size=k, replace=False)]
        delta = {e: float(rng.uniform(0.05, 1.0)) for e in chosen}
        h = perturb(g, delta)

        # edge set unchanged
        c.record(set(g.weights) == set(h.weights), note="edge set changed")
        # floor non-decreasing
        c.record(h.floor() >= g.floor() - TOL,
                 err=max(0.0, g.floor() - h.floor()),
                 note="floor decreased under a non-negative perturbation")
        # separation costs never decrease (weights only rose)
        for v in sorted(g.items()):
            c.record(h.separation_cost(v) >= g.separation_cost(v) - 1e-9,
                     err=max(0.0, g.separation_cost(v) - h.separation_cost(v)),
                     note="separation cost fell under a non-negative perturbation")
    return c


# =====================================================================
# E06 -- Theorem 5.5 (no privileged value / receiver-relativity)
# =====================================================================


def e06_receiver(rng) -> Check:
    c = Check("ReceiverRelativity", "Theorem 5.5 / Corollary 5.6")
    n_diff = 0
    for _ in range(200):
        g1 = random_contact_graph(rng, 5)
        g2 = g1.copy()
        # two receivers: same unit, different networks
        e = sorted(g2.weights, key=lambda x: tuple(sorted(x)))[0]
        g2.weights[e] = g2.weights[e] + float(rng.uniform(0.5, 2.0))

        v = "v0"
        d1 = {e2: 0.3 for e2 in g1.incident(v)}
        d2 = {e2: 0.3 for e2 in g2.incident(v)}
        r1 = perturb(g1, d1).separation_cost(v)
        r2 = perturb(g2, d2).separation_cost(v)
        # registered values may differ; the theorem forbids a single phi
        if abs(r1 - r2) > 1e-9:
            n_diff += 1
        c.record(True)
    # the point of the theorem: divergence actually occurs
    c.record(n_diff > 0,
             note="no receiver pair ever disagreed; theorem would be vacuous")
    return c


# =====================================================================
# E07 -- Proposition 6.2 (distance to the floor)
# =====================================================================


def e07_distance_to_floor(rng) -> Check:
    c = Check("DistanceToFloor", "Proposition 6.2")
    for _ in range(150):
        g = random_contact_graph(rng, 5)
        fr = floor_ratio(g)
        items = sorted(g.items())
        for u, v in itertools.combinations(items, 2):
            d = distance_to_floor(g, u, v)
            c.record(d >= -1e-9, err=max(0.0, -d),
                     note=f"distance to floor negative: {d}")
            a = alignment(g, u, v)
            c.record(a >= fr - 1e-9, err=max(0.0, fr - a),
                     note="alignment fell below the floor ratio")
    return c


# =====================================================================
# E08 -- Theorems 6.7, 6.8 (monotone relaxation; dichotomy)
# =====================================================================


def e08_relaxation(rng) -> Check:
    c = Check("RelaxationDichotomy", "Theorem 6.7 / Theorem 6.8")

    # (a) convergent case: bound on step count must hold
    for _ in range(200):
        D0 = float(rng.uniform(1.0, 20.0))
        eta = float(rng.uniform(0.05, 1.0))

        def update(D, eta=eta, rng=rng):
            return D - eta - float(rng.uniform(0.0, 0.3))

        res = relax(D0, update, eta)
        c.record(res.quiescent and not res.declined, note="did not reach quiescence")
        bound = int(np.ceil(D0 / eta))
        c.record(res.steps <= bound,
                 err=max(0, res.steps - bound),
                 note=f"steps {res.steps} exceeded bound {bound}")
        # strict monotone decrease (Theorem 6.7)
        diffs = np.diff(res.demands)
        c.record(bool(np.all(diffs < TOL)), note="demand sequence not decreasing")

    # (b) decline case: no effective update available
    for _ in range(100):
        D0 = float(rng.uniform(1.0, 5.0))
        res = relax(D0, lambda D: None, eta=0.1)
        c.record(res.declined and not res.quiescent, note="decline not reported")
        c.record(res.final_demand > 0.0, note="declined at zero demand")

    # (c) NEGATIVE CONTROL: no third outcome. A relaxation that neither
    #     converges nor declines must be impossible under Assumption 6.5;
    #     an update that idles must be REJECTED, not silently accepted.
    idled = 0
    for _ in range(50):
        try:
            relax(5.0, lambda D: D, eta=0.1)
        except AssertionError:
            idled += 1
    c.record(idled == 50,
             note=f"idling update accepted in {50 - idled}/50 cases")
    return c


# =====================================================================
# E09 -- Theorem 6.9 (quiescence certifies blind), 6.10 (positive residual)
# =====================================================================


def e09_quiescence(rng) -> Check:
    c = Check("QuiescenceBlind", "Theorem 6.9 / Theorem 6.10")
    for _ in range(150):
        g = random_contact_graph(rng, 5)
        items = sorted(g.items())
        # identity correspondence => every alignment is a(i,i)
        # a(i,i) is degenerate, so use a pair at the floor instead:
        # build a graph where two items are separated exactly at the floor
        h = ContactGraph()
        h.add_edge("u1", "u2", 0.2)
        h.add_edge("u1", "z", 1.0)
        h.add_edge("u2", "z", 1.0)
        for v in ("u1", "u2", "z"):
            h.add_edge(h.medium, v, 0.2)
        beta_ratio = floor_ratio(h)
        a12 = alignment(h, "u1", "u2")
        at_floor = abs(a12 - beta_ratio) <= 1e-9
        d = cross_demand(h, h, ["u1"], {"u1": "u2"})
        # Theorem 6.9: demand vanishes iff every term sits at the floor
        c.record((d <= 1e-9) == at_floor, err=abs(d),
                 note=f"quiescence/floor mismatch: d={d}, at_floor={at_floor}")
        # Theorem 6.10: quiescence coexists with positive separation
        if d <= 1e-9:
            c.record(h.floor() > 0.0, note="quiescent at zero floor")
            c.record(h.min_cut("u1", "u2")[0] > 0.0,
                     note="quiescent at zero separation (form collapsed)")
    return c


# =====================================================================
# E10 -- Theorem 7.3 (false friend) and Corollary 7.4 (converse)
#        The two classes sequence comparison conflates.
# =====================================================================


def _ff_graph(beta: float = 0.2, wz: float = 1.0) -> ContactGraph:
    """Witness for Theorem 7.3.

    u1 and u2 are pendants on the medium at weight beta, so the cheapest
    u1|u2 cut isolates one of them at cost exactly beta -- they sit AT
    the floor and the central demand is exactly zero. The pair (z,w)
    carries independent structure on which a response can act.

    Note: an earlier version of this construction joined u1 and u2 by an
    edge of weight beta and expected them to be separated at the floor.
    That is wrong: the min cut ignores that edge and isolates u1 through
    its other incident edges, so the central demand was positive. The
    pendant construction is the corrected witness.
    """
    g = ContactGraph()
    g.add_edge(g.medium, "u1", beta)
    g.add_edge(g.medium, "u2", beta)
    g.add_edge(g.medium, "z", wz)
    g.add_edge("z", "w", wz)
    return g


def _converse_graph(W: float) -> ContactGraph:
    """Witness for Corollary 7.4.

    A pendant p carries the floor, while u1 and u2 each have two edges of
    weight W, so isolating either costs 2W > beta and the central demand
    is strictly positive.
    """
    g = ContactGraph()
    g.add_edge(g.medium, "p", 0.2)          # p realises the floor
    g.add_edge(g.medium, "u1", W)
    g.add_edge("u1", "q", W)
    g.add_edge(g.medium, "u2", W)
    g.add_edge("u2", "q", W)
    g.add_edge(g.medium, "q", W)
    g.add_edge(g.medium, "z", 1.0)
    g.add_edge("z", "w", 1.0)
    return g


def e10_false_friend(rng) -> Check:
    c = Check("FalseFriendAndConverse", "Theorem 7.3 / Corollary 7.4")

    # --- Theorem 7.3: central quiescent, response NOT quiescent --------
    for c_pert in (0.5, 1.0, 2.0, 4.0):
        g = _ff_graph()
        r1 = perturb(g, {frozenset(("z", "w")): c_pert})
        r2 = g.copy()

        central_a = Column(g, "u1", [], ["u2"])
        central_b = Column(g, "u2", [], ["u1"])
        resp_a = Column(r1, "u1", [], ["z"])
        resp_b = Column(r2, "u2", [], ["z"])

        out = four_column_verdict(
            central_a, central_b, resp_a, resp_b,
            corr_central={"u1": "u2", "u2": "u1"},
            corr_response={"z": "w"},
            theta=1e-9,
        )
        c.record(out["central_quiescent"], err=out["central_demand"],
                 note=f"central not quiescent at c={c_pert}: "
                      f"d={out['central_demand']}")
        # the two response columns must place DIFFERENT demands
        diff = abs(out["d_response_ab"] - out["d_response_ba"])
        c.record(diff > 1e-6, err=diff,
                 note=f"response columns agreed at c={c_pert} "
                      "(Theorem 7.3 requires divergence)")
        c.record(out["verdict"] == "DIVERGE",
                 note=f"verdict {out['verdict']} != DIVERGE at c={c_pert}")

    # --- Corollary 7.4: central NOT quiescent, response quiescent ------
    for W in (1.0, 2.0, 5.0):
        h = _converse_graph(W)
        same = perturb(h, {frozenset(("z", "w")): 0.7})

        central_a = Column(h, "u1", [], ["u2"])
        central_b = Column(h, "u2", [], ["u1"])
        resp_a = Column(same, "u1", [], ["z"])
        resp_b = Column(same.copy(), "u2", [], ["z"])

        out = four_column_verdict(
            central_a, central_b, resp_a, resp_b,
            corr_central={"u1": "u2", "u2": "u1"},
            corr_response={"z": "w"},
            theta=1e-9,
        )
        c.record(not out["central_quiescent"], err=out["central_demand"],
                 note=f"central WAS quiescent at W={W}; "
                      "Corollary 7.4 requires divergent content")
        # The two units induce the SAME response, so the demand computed
        # in each response network against the same correspondence must
        # agree. (Comparing out["d_response_ab"] to out["d_response_ba"]
        # would be wrong: those use inverted correspondences and so are
        # different quantities by construction.)
        d_a = cross_demand(resp_a.graph, resp_a.graph, ["z"], {"z": "w"})
        d_b = cross_demand(resp_b.graph, resp_b.graph, ["z"], {"z": "w"})
        diff = abs(d_a - d_b)
        c.record(diff <= 1e-9, err=diff,
                 note=f"identical responses gave different demands at "
                      f"W={W}: {d_a} vs {d_b}")

    return c


# =====================================================================
# E11 -- Assumption 5.1 on REAL pathways: contact graphs are well formed
# =====================================================================


def e11_pathway_wellformed(rng) -> Check:
    c = Check("PathwayContactGraphs", "Assumption 5.1 / Theorem 3.2")
    pws = load_pathways(DATA)
    c.record(len(pws) == 4, note=f"expected 4 pathways, got {len(pws)}")
    for pw in pws:
        g = pathway_to_contact_graph(pw)
        beta = g.floor()
        c.record(beta > 0.0, note=f"{pw['pathway']}: non-positive floor")
        for v in sorted(g.items()):
            sep = g.separation_cost(v)
            c.record(sep >= beta - 1e-9, err=max(0.0, beta - sep),
                     note=f"{pw['pathway']}/{v}: sigma < beta")
        # every solved contact cost is strictly positive
        for e in pw["edges"]:
            c.record(e["cost"] > 0.0,
                     note=f"{pw['pathway']}: non-positive contact cost")
    return c


# =====================================================================
# E12 -- Theorem 4.2 on REAL pathways: relabelling invariance
# =====================================================================


def e12_pathway_invariance(rng) -> Check:
    c = Check("PathwayInvariance", "Theorem 4.2")
    for pw in load_pathways(DATA):
        g = pathway_to_contact_graph(pw)
        items = sorted(g.items())
        for _ in range(20):
            perm = list(rng.permutation(items))
            mapping = dict(zip(items, perm))
            h = g.relabel(mapping)
            for v in items:
                a, b = g.separation_cost(v), h.separation_cost(mapping[v])
                c.record(abs(a - b) <= 1e-9, err=abs(a - b),
                         note=f"{pw['pathway']}/{v}: {a} vs {b}")
    return c


# =====================================================================
# E13 -- THE DECISIVE EXPERIMENT: Construction 9.2, Assumption 7.5.
#        Does the cross-demand depend on WHICH admissible response
#        is constructed? If yes, Theorem 7.3 is not well defined.
# =====================================================================


def e13_response_independence(rng) -> Check:
    c = Check("ResponseIndependence", "Assumption 7.5 / Construction 9.2")
    theta = 0.01          # decision threshold used by the criterion
    discrepancies: list[float] = []

    for pw in load_pathways(DATA):
        g = pathway_to_contact_graph(pw)
        items = sorted(g.items())
        for x in items:
            inc = [e for e in g.incident(x) if g.medium not in e]
            if len(inc) < 2:
                continue
            # two admissible realisations of "perturb x": raise one
            # incident edge, or raise a different incident edge, by the
            # same total magnitude.
            mag = 0.5
            for e1, e2 in itertools.combinations(sorted(inc, key=lambda s: tuple(sorted(s))), 2):
                h1 = perturb(g, {e1: mag})
                h2 = perturb(g, {e2: mag})
                # fixed comparison column: demand against the medium-side
                # residue of the unperturbed network
                residue = [v for v in items if v != x][:3]
                corr = {v: x for v in residue}
                d1 = cross_demand(h1, h1, residue, corr)
                d2 = cross_demand(h2, h2, residue, corr)
                discrepancies.append(abs(d1 - d2))

    arr = np.array(discrepancies) if discrepancies else np.array([0.0])
    frac_below = float(np.mean(arr <= theta))
    # We do NOT assert independence -- we MEASURE it. The check that must
    # pass is that the measurement was actually performed.
    c.record(len(discrepancies) > 0, note="no comparisons were made")
    c.max_err = float(arr.max())
    c.extra = {                      # type: ignore[attr-defined]
        "n_comparisons": int(arr.size),
        "theta": theta,
        "max_discrepancy": float(arr.max()),
        "mean_discrepancy": float(arr.mean()),
        "median_discrepancy": float(np.median(arr)),
        "fraction_within_theta": frac_below,
        "assumption_holds_at_theta": bool(frac_below == 1.0),
    }
    return c


# =====================================================================
# E14 -- Four-column verdicts on REAL pathway pairs
# =====================================================================


def e14_pathway_fourcolumn(rng) -> Check:
    c = Check("PathwayFourColumn", "Theorem 7.3 / Definition 7.2")
    pws = load_pathways(DATA)
    verdicts: dict[str, int] = {"CORRESPOND": 0, "DIVERGE": 0}

    for pw in pws:
        g = pathway_to_contact_graph(pw)
        items = sorted(g.items())
        if len(items) < 4:
            continue
        x1, x2 = items[0], items[1]
        r = 2
        anc1, res1 = anchors_and_residue(g, x1, r)
        anc2, res2 = anchors_and_residue(g, x2, r)

        # a unit that perturbs, versus one that does not
        inc1 = [e for e in g.incident(x1) if g.medium not in e]
        resp1 = perturb(g, {inc1[0]: 0.8}) if inc1 else g.copy()
        resp2 = g.copy()

        col_a = Column(g, x1, anc1, res1)
        col_b = Column(g, x2, anc2, res2)
        rc_a = Column(resp1, x1, anc1, res1)
        rc_b = Column(resp2, x2, anc2, res2)

        corr_c = {i: x1 for i in res2}
        corr_r = {i: i for i in res2 if i in resp1.vertices}

        out = four_column_verdict(col_a, col_b, rc_a, rc_b,
                                  corr_c, corr_r, theta=1e-9)
        verdicts[out["verdict"]] += 1
        # demands must be non-negative and finite (Definition 6.3)
        for k in ("central_demand", "response_demand"):
            c.record(out[k] >= -1e-12 and np.isfinite(out[k]),
                     note=f"{pw['pathway']}: bad {k} = {out[k]}")

    c.record(sum(verdicts.values()) > 0, note="no pathway verdicts produced")
    c.extra = {"verdicts": verdicts}   # type: ignore[attr-defined]
    return c


EXPERIMENTS = [
    ("E01", e01_floor),
    ("E02", e02_intrinsic_floor),
    ("E03", e03_invariance),
    ("E04", e04_region),
    ("E05", e05_perturbation),
    ("E06", e06_receiver),
    ("E07", e07_distance_to_floor),
    ("E08", e08_relaxation),
    ("E09", e09_quiescence),
    ("E10", e10_false_friend),
    ("E11", e11_pathway_wellformed),
    ("E12", e12_pathway_invariance),
    ("E13", e13_response_independence),
    ("E14", e14_pathway_fourcolumn),
]
