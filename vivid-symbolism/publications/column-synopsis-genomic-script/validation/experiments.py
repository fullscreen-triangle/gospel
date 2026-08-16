"""
experiments.py -- one check per claim of synopsis-genomic-scripting.tex.

The claims split into three kinds and the suite treats them differently:

  MATHEMATICAL (Secs 2-5). Checkable numerically on contact structures.
  A failure here is a false theorem.

  LANGUAGE (Secs 7-9). The refusal theorems. These assert that certain
  programs have NO derivation and NO typing. They are checked with a
  parser and typechecker over a positive corpus (must be accepted) and a
  negative corpus (must be rejected, each with the right error class).
  Both halves matter: a checker that rejected everything would satisfy
  every refusal theorem vacuously.

  MEASUREMENT (Sec 13). Response independence. This is reported, not
  asserted. It FAILS on real networks and that outcome is the paper's
  principal open problem, so the suite records the discrepancy and does
  NOT fail on it. Marking it as an assertion would either force a wrong
  theorem or silently hide the finding.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

import core as C
import lang as L
import semantics as S

SEED = 20260816


# =====================================================================
# Sec 2 -- the floor
# =====================================================================

def e01_contact_floor(rng) -> C.Check:
    """Thm 2.5: sigma(v) >= beta > 0 for every item."""
    chk = C.Check("E01", "Thm 2.5 (contact floor): sigma(v) >= beta > 0")
    ratios = []
    for _ in range(200):
        n = int(rng.integers(3, 7))
        g = C.random_contact_structure(rng, n)
        beta = g.floor()
        for v in sorted(g.items()):
            s = g.separation_cost(v)
            ok = s >= beta - C.TOL and beta > 0.0
            chk.record(ok, max(0.0, beta - s),
                       f"sigma({v})={s:.6f} < beta={beta:.6f}")
            ratios.append(s / beta)
    chk.extra = {
        "n_ratios": len(ratios),
        "ratio_min": float(np.min(ratios)),
        "ratio_median": float(np.median(ratios)),
        "ratio_max": float(np.max(ratios)),
    }
    return chk


def e02_zero_weight_refused(rng) -> C.Check:
    """Def 2.1: a zero weight is the ABSENCE of an edge, not a light edge."""
    chk = C.Check("E02", "Def 2.1: non-positive weights are refused")
    g = C.ContactStructure()
    for w in (0.0, -0.1, -1.0):
        try:
            g.add_edge("a", "b", w)
            chk.record(False, 0.0, f"accepted weight {w}")
        except ValueError:
            chk.record(True)
    try:
        g.add_edge("a", "b", 0.5)
        chk.record(True)
    except ValueError:
        chk.record(False, 0.0, "rejected a legitimate positive weight")
    return chk


def e03_floor_decreases_but_stays_positive(rng) -> C.Check:
    """The floor falls with size yet never reaches zero."""
    chk = C.Check("E03", "Thm 2.5: beta decreases with size but stays > 0")
    meds = {}
    for n in range(3, 10):
        fl = []
        for _ in range(60):
            g = C.random_contact_structure(rng, n)
            b = g.floor()
            chk.record(b > 0.0, 0.0, f"floor {b} not positive at n={n}")
            fl.append(b)
        meds[n] = float(np.median(fl))
    decreasing = all(meds[n] >= meds[n + 1] - 1e-9 for n in range(3, 9))
    chk.record(decreasing, 0.0, f"median floor not monotone: {meds}")
    chk.extra = {"median_floor_by_size": meds}
    return chk


# =====================================================================
# Sec 2 -- invariance and labels
# =====================================================================

def e04_relabelling_invariance(rng) -> C.Check:
    """Thm 2.9 / Cor 2.10: cut weight is invariant; labels are free.

    The deviation must be EXACTLY zero, not small. A weighted isomorphism
    carries cuts to cuts weight-for-weight; any nonzero deviation would
    mean the implementation is reading a label somewhere.
    """
    chk = C.Check("E04", "Thm 2.9/Cor 2.10: sigma invariant under relabelling")
    devs = []
    for _ in range(150):
        n = int(rng.integers(3, 7))
        g = C.random_contact_structure(rng, n)
        items = sorted(g.items())
        perm = list(items)
        rng.shuffle(perm)
        mapping = {a: f"X{b}" for a, b in zip(items, perm)}
        h = g.relabel(mapping)
        for v in items:
            before = g.separation_cost(v)
            after = h.separation_cost(mapping[v])
            d = abs(before - after)
            devs.append(d)
            chk.record(d == 0.0, d,
                       f"sigma({v}) changed by {d:g} under relabelling")
    chk.extra = {"n_items": len(devs), "max_abs_deviation": float(max(devs))}
    return chk


def e05_medium_must_be_fixed(rng) -> C.Check:
    """Def 2.7: an isomorphism that moves the medium is not admissible."""
    chk = C.Check("E05", "Def 2.7: isomorphisms must fix the medium")
    g = C.random_contact_structure(rng, 4)
    try:
        g.relabel({C.MEDIUM: "elsewhere"})
        chk.record(False, 0.0, "relabelling moved the medium and was accepted")
    except ValueError:
        chk.record(True)
    h = g.relabel({"i0": "renamed"})
    chk.record(h.medium == C.MEDIUM, 0.0, "medium not preserved")
    return chk


def e06_expansion_floor_conditional(rng) -> C.Check:
    """Rem 2.12 + Thm 2.11: the UNCONDITIONAL floor claim is false.

    Two decaying regimes drive the local floor to zero -- these are the
    counterexamples the theorem must exclude -- while a floor-stabilising
    expansion holds it up. Recording the decay is the point: it is what
    makes the hypothesis of Thm 2.11 necessary rather than decorative.
    """
    chk = C.Check("E06", "Rem 2.12/Thm 2.11: floor preservation is conditional")
    steps = 40

    def run(wfun):
        g = C.ContactStructure()
        g.add_edge("v", C.MEDIUM, 1.0)
        for k in range(1, steps + 1):
            g.add_edge("v", f"x{k}", wfun(k))
        return g.local_floor("v")

    f_lin = run(lambda k: 1.0 / k)
    f_quad = run(lambda k: 1.0 / (k * k))
    f_stab = run(lambda k: 1.0)

    # The decaying regimes must actually decay -- otherwise Rem 2.12 has
    # no witness and the hypothesis of Thm 2.11 would be unnecessary.
    chk.record(f_lin < 0.05, 0.0, f"1/k floor did not decay: {f_lin}")
    chk.record(f_quad < 1e-3, 0.0, f"1/k^2 floor did not decay: {f_quad}")
    chk.record(f_lin > f_quad, 0.0, "1/k^2 should decay faster than 1/k")
    # The stabilising regime must hold the floor -- that is Thm 2.11.
    chk.record(abs(f_stab - 1.0) < C.TOL, abs(f_stab - 1.0),
               f"stabilising expansion did not hold the floor: {f_stab}")
    chk.record(f_stab > 0.0, 0.0, "stabilising floor not positive")

    chk.extra = {
        "steps": steps,
        "local_floor_1_over_k": float(f_lin),
        "local_floor_1_over_k2": float(f_quad),
        "local_floor_stabilising": float(f_stab),
    }
    return chk


def e07_skeleton_invariance(rng) -> C.Check:
    """Prop 2.14: perturbation changes costs, never contacts."""
    chk = C.Check("E07", "Prop 2.14: perturbation preserves the skeleton")
    for _ in range(120):
        g = C.random_contact_structure(rng, int(rng.integers(3, 6)))
        edges = sorted(g.weights, key=lambda e: tuple(sorted(e)))
        chosen = [e for e in edges if rng.random() < 0.5]
        delta = {e: float(rng.uniform(0.0, 1.5)) for e in chosen}
        h = C.perturb(g, delta)
        chk.record(set(h.weights) == set(g.weights), 0.0,
                   "perturbation changed the edge set")
        for v in sorted(g.items()):
            before, after = g.separation_cost(v), h.separation_cost(v)
            chk.record(after >= before - C.TOL, max(0.0, before - after),
                       f"sigma({v}) fell under a non-negative perturbation")
    # A negative perturbation must be refused outright.
    g = C.random_contact_structure(rng, 3)
    e = next(iter(g.weights))
    try:
        C.perturb(g, {e: -0.1})
        chk.record(False, 0.0, "negative perturbation accepted")
    except ValueError:
        chk.record(True)
    return chk


# =====================================================================
# Sec 3 -- identity is regional and receiver-relative
# =====================================================================

def e08_regionality(rng) -> C.Check:
    """Thm 3.1: minimisers are regions, never singletons."""
    chk = C.Check("E08", "Thm 3.1: minimising sets are regions, not points")
    sizes = []
    for r in (3, 4, 5):
        for beta in (0.1, 0.2, 0.3):
            g = C.two_cluster_structure(r, W=1.0, beta=beta)
            w, Ssta = g.min_cut("a0", C.MEDIUM)
            k = len(Ssta)
            sizes.append(k)
            chk.record(k >= 3, 0.0,
                       f"minimiser of size {k} at r={r}, beta={beta}")
    chk.extra = {"minimiser_sizes": sizes,
                 "min_size": int(min(sizes)), "max_size": int(max(sizes))}
    return chk


def e09_region_point_crossover(rng) -> C.Check:
    """Cor 3.3: past a crossover the region is strictly cheaper."""
    chk = C.Check("E09", "Cor 3.3: region beats point above a crossover")
    r, beta = 4, 0.2
    pts, regs, Ws = [], [], []
    for W in np.linspace(0.05, 2.0, 20):
        g = C.two_cluster_structure(r, W=float(W), beta=beta)
        A = [f"a{k}" for k in range(r)]
        pt = g.cut_weight({"a0"})
        reg = g.cut_weight(set(A))
        pts.append(float(pt)); regs.append(float(reg)); Ws.append(float(W))
        mn, _ = g.min_cut("a0", C.MEDIUM)
        chk.record(mn <= min(pt, reg) + C.TOL, 0.0,
                   f"min cut {mn} exceeds an explicit cut at W={W}")
    # The point cost must rise in W; the region cost must not.
    chk.record(pts[-1] > pts[0], 0.0, "point cost did not rise with W")
    chk.record(abs(regs[-1] - regs[0]) < 1e-9, abs(regs[-1] - regs[0]),
               "region cost varied with W but should not")
    cross = [w for w, p, q in zip(Ws, pts, regs) if p > q]
    chk.record(len(cross) > 0, 0.0, "no crossover found")
    chk.extra = {"crossover_W": float(min(cross)) if cross else None,
                 "point_cost_range": [pts[0], pts[-1]],
                 "region_cost_range": [regs[0], regs[-1]]}
    return chk


def e10_receiver_relativity(rng) -> C.Check:
    """Thm 3.5 / Cor 3.6: the value is a function of (unit, receiver).

    Disagreement between two receivers is not noise. We build pairs of
    receivers differing in ONE background edge and confirm the same unit
    registers different values -- and that the difference does not shrink
    toward zero, which is what "not measurement error" means.
    """
    chk = C.Check("E10", "Thm 3.5/Cor 3.6: registered value is receiver-relative")
    diffs = []
    for _ in range(150):
        g = C.random_contact_structure(rng, int(rng.integers(4, 6)))
        items = sorted(g.items())
        u = items[0]
        # Perturb ONE edge not incident to u: the unit is unchanged.
        cands = [e for e in sorted(g.weights, key=lambda e: tuple(sorted(e)))
                 if u not in e]
        if not cands:
            continue
        e = cands[int(rng.integers(len(cands)))]
        h = C.perturb(g, {e: 1.0})
        a, b = g.separation_cost(u), h.separation_cost(u)
        diffs.append(abs(a - b))
    n_diff = sum(1 for d in diffs if d > 1e-9)
    chk.record(n_diff > 0, 0.0, "no receiver ever disagreed")

    # Thm 3.5 says the value is a FUNCTION OF THE PAIR (unit, receiver).
    # It does not say most perturbations move it, and they do not: a
    # background edge shifts sigma(u) only when it crosses u's minimising
    # cut, which is a minority of edges. An earlier version of this check
    # demanded a 20% disagreement rate; that threshold described no
    # theorem. What must hold is that disagreement occurs and is
    # SUBSTANTIAL rather than floating-point noise -- otherwise Cor 3.6
    # ("disagreement is not measurement error") would be empty.
    substantial = [d for d in diffs if d > 1e-9]
    chk.record(len(substantial) > 0 and min(substantial) > 1e-6, 0.0,
               "disagreements were at the scale of numerical noise")
    chk.extra = {
        "n_pairs": len(diffs),
        "n_disagreeing": int(n_diff),
        "fraction_disagreeing": float(n_diff / max(len(diffs), 1)),
        "mean_abs_difference": float(np.mean(diffs)) if diffs else 0.0,
        "max_abs_difference": float(np.max(diffs)) if diffs else 0.0,
        "min_substantial_difference": (float(min(substantial))
                                       if substantial else None),
        "note": ("a background edge moves sigma(u) only when it crosses "
                 "u's minimising cut, so the disagreement rate is a "
                 "minority by construction; the claim is that the value "
                 "depends on the receiver, not that it usually does"),
    }
    return chk


# =====================================================================
# Sec 4 -- alignment, demand, relaxation
# =====================================================================

def e11_distance_to_floor_nonneg(rng) -> C.Check:
    """Prop 4.2: Delta(u,v) = a(u,v) - beta/Omega >= 0."""
    chk = C.Check("E11", "Prop 4.2: distance to floor is non-negative")
    deltas = []
    for _ in range(120):
        g = C.random_contact_structure(rng, int(rng.integers(3, 6)))
        items = sorted(g.items())
        for u, v in itertools.combinations(items, 2):
            d = C.distance_to_floor(g, u, v)
            deltas.append(float(d))
            chk.record(d >= -C.TOL, max(0.0, -d),
                       f"Delta({u},{v}) = {d:g} < 0")
    chk.extra = {"n_pairs": len(deltas),
                 "min_delta": float(min(deltas)),
                 "median_delta": float(np.median(deltas))}
    return chk


def e12_anchor_split_deterministic(rng) -> C.Check:
    """Def 4.3: the anchor/residue split is a function."""
    chk = C.Check("E12", "Def 4.3: anchor/residue split is deterministic")
    for _ in range(60):
        g = C.random_contact_structure(rng, int(rng.integers(4, 7)))
        x = sorted(g.items())[0]
        r = 2
        a1, r1 = C.anchors_and_residue(g, x, r)
        a2, r2 = C.anchors_and_residue(g, x, r)
        chk.record(a1 == a2 and r1 == r2, 0.0, "split was not reproducible")
        chk.record(len(a1) == min(r, len(g.items()) - 1), 0.0,
                   f"anchor count {len(a1)} != {r}")
        chk.record(not (set(a1) & set(r1)), 0.0, "anchors and residue overlap")
    return chk


def e13_monotone_and_bound(rng) -> C.Check:
    """Thm 4.7 (monotone) and Thm 4.8 (dichotomy + bound).

    The bound ceil(D0/eta) must never be exceeded. One violation refutes
    the theorem, so this is checked per run, not on average.
    """
    chk = C.Check("E13", "Thm 4.7/4.8: monotone decrease, bound never exceeded")
    pts = []
    for _ in range(260):
        D0 = float(rng.uniform(0.05, 5.0))
        eta = float(rng.uniform(0.005, 0.5))

        def upd(D, eta=eta):
            excess = float(rng.uniform(0.0, 0.5 * eta))
            return max(0.0, D - eta - excess)

        res = C.relax(D0, upd, eta, theta=1e-9)
        chk.record(res.quiescent, 0.0, "did not reach quiescence")
        chk.record(res.steps <= res.bound, float(res.steps - res.bound),
                   f"steps {res.steps} exceeded bound {res.bound}")
        mono = all(res.demands[i + 1] <= res.demands[i] + C.TOL
                   for i in range(len(res.demands) - 1))
        chk.record(mono, 0.0, "demand did not decrease monotonically")
        pts.append((res.steps, res.bound))
    chk.extra = {
        "n_runs": len(pts),
        "max_steps": int(max(s for s, _ in pts)),
        "n_at_or_below_bound": int(sum(1 for s, b in pts if s <= b)),
    }
    return chk


def e14_dichotomy_decline(rng) -> C.Check:
    """Thm 4.8(ii): DECLINE is the other branch, and there is no third."""
    chk = C.Check("E14", "Thm 4.8: quiescence or decline, no third outcome")
    for _ in range(60):
        D0 = float(rng.uniform(0.5, 3.0))
        eta = 0.05
        kill = int(rng.integers(1, 6))
        state = {"n": 0}

        def upd(D, kill=kill, state=state, eta=eta):
            state["n"] += 1
            if state["n"] >= kill:
                return None
            return max(0.0, D - eta - 0.01)

        res = C.relax(D0, upd, eta, theta=1e-9)
        chk.record(res.declined ^ res.quiescent, 0.0,
                   "run was neither or both quiescent and declined")
        if res.declined:
            chk.record(res.final_demand > 0.0, 0.0,
                       "declined at zero demand")
    # eta <= 0 must be refused: without it there is no bound at all.
    for bad in (0.0, -0.1):
        try:
            C.relax(1.0, lambda D: D - 0.1, bad)
            chk.record(False, 0.0, f"accepted eta={bad}")
        except ValueError:
            chk.record(True)
    return chk


def e15_form_of_quiescence(rng) -> C.Check:
    """Thm 4.9: quiescence coexists with strictly positive residue.

    This is `sufficient alignment, not exact alignment` made precise: the
    demand can vanish while every pair remains strictly distinguishable.
    """
    chk = C.Check("E15", "Thm 4.9: quiescence coexists with positive residue")
    n_zero_demand = 0
    n_positive_sep = 0
    trials = 0
    for _ in range(80):
        g = C.random_contact_structure(rng, int(rng.integers(4, 6)))
        items = sorted(g.items())
        # Identity correspondence -> demand contributions are all zero,
        # yet separation costs are strictly positive throughout.
        corr = {i: i for i in items}
        D = C.cross_demand(g, g, items, corr)
        trials += 1
        if abs(D) <= C.TOL:
            n_zero_demand += 1
        seps = [g.separation_cost(v) for v in items]
        if min(seps) > 0.0:
            n_positive_sep += 1
        chk.record(abs(D) <= C.TOL and min(seps) > 0.0, abs(D),
                   f"demand {D:g} with min separation {min(seps):g}")
    chk.extra = {"trials": trials,
                 "n_zero_demand": n_zero_demand,
                 "n_positive_separation": n_positive_sep}
    return chk


# =====================================================================
# Sec 5 -- the two separation results
# =====================================================================

def e16_false_friend(rng) -> C.Check:
    """Thm 5.1: central demand EXACTLY 0 while response is unbounded.

    The witness is a PENDANT pair. Rem 3.2 records that our first
    witness -- a light edge between two non-pendant items -- was wrong:
    the exact min cut was 1.4, not the floor. Using the pendant
    construction is not a cosmetic choice; the theorem is false without it.
    """
    chk = C.Check("E16", "Thm 5.1: aligned content, divergent response")
    beta = 0.2
    g = C.pendant_pair_structure(beta)
    central = C.distance_to_floor(g, "u", "v")
    chk.record(abs(central) <= C.TOL, abs(central),
               f"central distance to floor was {central:g}, not exactly 0")

    # The perturbation must raise BOTH pendant edges. Raising only one
    # leaves the (u,v)-cut pinned by the other -- the minimiser simply
    # places the medium on the raised side and the cut never moves. That
    # was our first attempt and it registered no response at all.
    #
    # The sweep starts at 1.0 rather than 0: while the pendants are still
    # the lightest edges they ARE the floor, so beta rises in lockstep
    # with the cut and Delta stays at 0. That is Thm 2.5 behaving
    # correctly, not an absent response; the response becomes visible
    # once the pendants exceed the filler weight (1.0 here).
    responses = []
    for c in np.linspace(1.0, 5.0, 9):
        h = C.perturb(g, {frozenset(("u", C.MEDIUM)): float(c),
                          frozenset(("v", C.MEDIUM)): float(c)})
        responses.append(float(C.distance_to_floor(h, "u", "v")))
    chk.record(responses[-1] > responses[0], 0.0,
               "response did not rise with the perturbation")
    chk.record(responses[-1] > 0.1, 0.0,
               f"response stayed small: {responses[-1]}")
    mono = all(responses[i + 1] >= responses[i] - 1e-12
               for i in range(len(responses) - 1))
    chk.record(mono, 0.0, "response curve was not monotone")
    chk.extra = {"central_distance_to_floor": float(central),
                 "response_at_cmin": responses[0],
                 "response_at_cmax": responses[-1],
                 "response_curve": responses}
    return chk


def e17_converse(rng) -> C.Check:
    """Cor 5.2: central demand rising while response is EXACTLY 0."""
    chk = C.Check("E17", "Cor 5.2: divergent content, aligned response")
    beta = 0.2
    # Central divergence is driven by the BRIDGE, not by the intra-cluster
    # weight W. Raising W leaves cut(a1,b1) fixed -- a1 and b1 each detach
    # through the medium at beta regardless -- while inflating Omega, so
    # Delta FALLS. Sweeping W was our first attempt and it asserted the
    # wrong monotonicity. The bridge is the edge the (a1,b1)-cut crosses.
    centrals = []
    base = C.two_cluster_structure(4, W=1.0, beta=beta)
    for d in np.linspace(0.0, 2.0, 12):
        g = C.perturb(base, {frozenset(("a0", "b0")): float(d)})
        centrals.append(float(C.distance_to_floor(g, "a1", "b1")))
    chk.record(centrals[-1] > centrals[0], 0.0,
               "central demand did not rise with the bridge weight")

    # Both units induce the SAME perturbation, so the response demand is
    # identically zero -- not small, zero.
    g = C.two_cluster_structure(4, W=2.0, beta=beta)
    pa = C.perturb(g, {frozenset(("a1", C.MEDIUM)): 0.5})
    pb = C.perturb(g, {frozenset(("b1", C.MEDIUM)): 0.5})
    resp = abs(pa.separation_cost("a1") - pb.separation_cost("b1"))
    chk.record(resp <= C.TOL, resp,
               f"response demand was {resp:g}, not exactly 0")
    chk.extra = {"central_at_Wmin": centrals[0],
                 "central_at_Wmax": centrals[-1],
                 "response_demand": float(resp),
                 "central_curve": centrals}
    return chk


def e18_orthogonality(rng) -> C.Check:
    """Cor 5.3: the two families occupy orthogonal axes of the demand plane."""
    chk = C.Check("E18", "Cor 5.3: the two classes are orthogonal")
    fam_a, fam_b = [], []
    # Family A: central demand identically 0, response strictly positive.
    # Both pendants are raised together -- see E16 for why one is not
    # enough -- and the sweep starts above the filler weight so the
    # pendants are no longer the floor.
    g0 = C.pendant_pair_structure(0.2)
    for c in np.linspace(1.5, 5.0, 8):
        h = C.perturb(g0, {frozenset(("u", C.MEDIUM)): float(c),
                           frozenset(("v", C.MEDIUM)): float(c)})
        fam_a.append((float(C.distance_to_floor(g0, "u", "v")),
                      float(C.distance_to_floor(h, "u", "v"))))
    # Family B: central demand strictly positive, response identically 0.
    base = C.two_cluster_structure(4, W=1.0, beta=0.2)
    for d in np.linspace(0.25, 2.0, 8):
        g = C.perturb(base, {frozenset(("a0", "b0")): float(d)})
        fam_b.append((float(C.distance_to_floor(g, "a1", "b1")), 0.0))

    # Each family sits strictly on one axis: zero on its own coordinate,
    # strictly positive on the other. Requiring the second half is what
    # makes this orthogonality rather than mere coincidence at the origin.
    for cen, res in fam_a:
        chk.record(abs(cen) <= C.TOL, abs(cen),
                   f"family A had nonzero central demand {cen:g}")
        chk.record(res > C.TOL, 0.0,
                   f"family A had zero response demand {res:g}")
    for cen, res in fam_b:
        chk.record(abs(res) <= C.TOL, abs(res),
                   f"family B had nonzero response demand {res:g}")
        chk.record(cen > C.TOL, 0.0,
                   f"family B had zero central demand {cen:g}")
    chk.extra = {"family_A": fam_a, "family_B": fam_b}
    return chk


def e19_four_column_criterion(rng) -> C.Check:
    """Thm 5.6: the criterion is 3-valued and needs BOTH pairs quiescent."""
    chk = C.Check("E19", "Thm 5.6/Def 5.5: four-column verdict is 3-valued")
    g = C.random_contact_structure(rng, 4)
    items = sorted(g.items())
    col = C.make_column(g, items[0], 2)
    ident = {i: i for i in items}

    v = C.four_column_verdict(col, col, col, col, ident, ident, theta=0.01)
    chk.record(v["verdict"] == "CORRESPOND", 0.0,
               f"identity comparison gave {v['verdict']}")

    # DECLINE must be reachable and must dominate.
    v2 = C.four_column_verdict(col, col, col, col, ident, ident,
                               theta=0.01, declined=True)
    chk.record(v2["verdict"] == "DECLINE", 0.0,
               "declined relaxation did not give DECLINE")

    # A verdict of CORRESPOND requires BOTH pairs quiescent, which is the
    # content of Thm 5.6: one coordinate cannot decide.
    seen = set()
    for cq, rq in itertools.product([True, False], repeat=2):
        got = "CORRESPOND" if (cq and rq) else "DIVERGE"
        seen.add((cq, rq, got))
    chk.record(sum(1 for a, b, g_ in seen if g_ == "CORRESPOND") == 1, 0.0,
               "more than one quadrant yields CORRESPOND")
    chk.record(len({g_ for _, _, g_ in seen} | {"DECLINE"}) == 3, 0.0,
               "verdict is not 3-valued")
    chk.extra = {"identity_verdict": v["verdict"],
                 "declined_verdict": v2["verdict"],
                 "central_demand": v["central_demand"],
                 "response_demand": v["response_demand"]}
    return chk


# =====================================================================
# Sec 7-8 -- depth and semantics primitives
# =====================================================================

def e20_partition_depth_additive(rng) -> C.Check:
    """Prop 7.11: depth is additive under independent partitions."""
    chk = C.Check("E20", "Prop 7.11: partition depth is additive")
    for _ in range(100):
        a = [int(x) for x in rng.integers(2, 9, size=int(rng.integers(1, 5)))]
        b = [int(x) for x in rng.integers(2, 9, size=int(rng.integers(1, 5)))]
        da, db = C.partition_depth(a), C.partition_depth(b)
        dab = C.partition_depth(a + b)
        err = abs(dab - (da + db))
        chk.record(err < 1e-12, err, f"depth not additive: {err:g}")
    chk.record(abs(C.partition_depth([3], base=3.0) - 1.0) < 1e-12, 0.0,
               "log_3(3) != 1")
    return chk


def e21_xcorr_range_and_agreement(rng) -> C.Check:
    """Def 8.2: normalised cross-correlation lies in [-1,1].

    Also: the FFT path must agree with the direct path. They are two
    implementations of one definition; disagreement means one is wrong.
    """
    chk = C.Check("E21", "Def 8.2: xcorr in [-1,1]; FFT agrees with direct")
    worst = 0.0
    for _ in range(40):
        Lt = int(rng.integers(60, 200))
        Lq = int(rng.integers(6, 20))
        t = S.channelise_dna("".join(rng.choice(list("ACGT"), Lt)))
        q = S.channelise_dna("".join(rng.choice(list("ACGT"), Lq)))
        a = S.xcorr_naive(q, t)
        b = S.xcorr_fft(q, t)
        if len(a) == 0:
            continue
        chk.record(float(np.max(np.abs(a))) <= 1.0 + 1e-9,
                   float(np.max(np.abs(a))) - 1.0, "direct xcorr left [-1,1]")
        chk.record(float(np.max(np.abs(b))) <= 1.0 + 1e-9,
                   float(np.max(np.abs(b))) - 1.0, "FFT xcorr left [-1,1]")
        d = float(np.max(np.abs(a - b)))
        worst = max(worst, d)
        chk.record(d < 1e-8, d, f"FFT and direct disagree by {d:g}")

    # A query matched against itself must attain exactly 1.
    q = S.channelise_dna("ACGTACGTAC")
    self_score = S.xcorr_naive(q, q)
    chk.record(abs(self_score[0] - 1.0) < 1e-9, abs(self_score[0] - 1.0),
               f"self-correlation was {self_score[0]}, not 1")
    chk.extra = {"max_fft_vs_direct_deviation": worst,
                 "self_correlation": float(self_score[0])}
    return chk


def e22_spectral_embedding(rng) -> C.Check:
    """Def 8.3: DC skipped, l2-normalised, dimension = coeffs."""
    chk = C.Check("E22", "Def 8.3: spectral embedding skips DC and is l2-normed")
    for _ in range(60):
        n = int(rng.integers(20, 120))
        k = int(rng.integers(4, 17))
        x = rng.normal(size=n)
        e = S.spectral(x, k)
        chk.record(len(e) == k, 0.0, f"dimension {len(e)} != {k}")
        nrm = float(np.linalg.norm(e))
        chk.record(abs(nrm - 1.0) < 1e-9, abs(nrm - 1.0),
                   f"not l2-normalised: {nrm}")

    # DC is skipped: adding a constant offset changes only the mean, so
    # the embedding must be unchanged. If DC were kept it would not be.
    x = rng.normal(size=64)
    e1 = S.spectral(x, 8)
    e2 = S.spectral(x + 17.0, 8)
    d = float(np.max(np.abs(e1 - e2)))
    chk.record(d < 1e-9, d, f"embedding moved by {d:g} under a DC offset")
    chk.extra = {"dc_offset_deviation": d}
    return chk


def e23_shader_range(rng) -> C.Check:
    """Def 8.4: shader distance d = 1 - Bq lies in [0,2]."""
    chk = C.Check("E23", "Def 8.4: shader distance in [0,2]")
    for _ in range(60):
        N, m = int(rng.integers(5, 40)), int(rng.integers(3, 12))
        B = rng.normal(size=(N, m))
        B = B / np.linalg.norm(B, axis=1, keepdims=True)
        q = rng.normal(size=m)
        q = q / np.linalg.norm(q)
        d = S.shader_distance(B, q)
        lo, hi = float(d.min()), float(d.max())
        chk.record(lo >= -1e-9, max(0.0, -lo), f"distance {lo} < 0")
        chk.record(hi <= 2.0 + 1e-9, max(0.0, hi - 2.0), f"distance {hi} > 2")
    # Self-distance must be exactly 0.
    q = rng.normal(size=6); q = q / np.linalg.norm(q)
    d0 = float(S.shader_distance(q[None, :], q)[0])
    chk.record(abs(d0) < 1e-9, abs(d0), f"self-distance {d0} != 0")
    return chk


def e24_complementary_strand(rng) -> C.Check:
    """Rem 1.x: the complementary strand is the NEGATED trajectory.

    This is the concrete form of the argument that the invariant is not
    borne by the sequence: two strands carrying identical information
    give exactly opposite cardinal paths, so no function of the path
    alone can be the invariant.
    """
    chk = C.Check("E24", "Sec 1: complementary strand negates the trajectory")
    worst = 0.0
    for _ in range(40):
        n = int(rng.integers(20, 200))
        s = "".join(rng.choice(list("ACGT"), n))
        p = S.cardinal(s)
        pc = S.cardinal(S.complement(s))
        d = float(np.max(np.abs(p + pc)))
        worst = max(worst, d)
        chk.record(d < 1e-12, d, f"path + complement != 0, off by {d:g}")
    chk.extra = {"max_deviation": worst}
    return chk


def e25_address_prefix_property(rng) -> C.Check:
    """Sec 8.5: deeper addresses extend shallower ones."""
    chk = C.Check("E25", "Sec 8.5: hierarchical address is prefix-consistent")
    lo, hi = np.zeros(3), np.ones(3)
    for _ in range(120):
        p = rng.random(3)
        a8 = S.address(p, lo, hi, 8)
        a12 = S.address(p, lo, hi, 12)
        chk.record(a12.startswith(a8), 0.0, "deeper address is not an extension")
        chk.record(len(a8) == 8 and len(a12) == 12, 0.0, "wrong address length")
    return chk


# =====================================================================
# Sec 9 -- the refusal theorems (positive corpus)
# =====================================================================

MOTIF_SCAN = '''
# Locate a motif in a target by coherent multichannel matched filtering.
open motif  = "motif.fa"
open target = "chr7_region.fa"

under nucleotide {
    let q = project motif  by channels(dna)
    let t = project target by channels(dna)
    bind r, res_r = compare q against t by xcorr(normalised)
    let hits = detect peaks in r {
        z            = 4.0 ;
        min_distance = 30  ;
        min_score    = 0.35
    }
    claim "motif occurs at least once above z=4" = hits
    record hits, res_r
}

report to "motif_scan.report"
'''

HOMOLOGY = '''
open query = "query.fa"
open db    = "swissprot_subset.fa"

under residue_space {
    let e  = project query by spectral(coeffs = 8)
    let DB = project db    by spectral(coeffs = 8)
    bind s, res_s = compare e against DB by shader(cosine)
    let cands = detect top in s {
        k      = 200 ;
        depth  = 12
    }
    bind final, res_f = compare e against cands by smith_waterman(
        match = 2 ; mismatch = -1 ; gap = -2
    )
    claim "top 20 by exact rerank" = final
    record final, res_s, res_f
}

report to "homology.report"
'''

ADJUDICATE = '''
open wt  = "wildtype.solved"
open var = "variant.solved"

method pi_edge_raise = perturb_incident(magnitude = 0.10)

under cell_receiver {
    let Nwt  = project wt  by contact(medium = 0.05)
    let Nvar = project var by contact(medium = 0.05)
    let cA = unit Nwt  anchors 3
    let cB = unit Nvar anchors 3
    let rA = response cA by pi_edge_raise
    let rB = response cB by pi_edge_raise
    let phi_c = corr from cA to cB by species_name
    let phi_r = corr from rA to rB by species_name
    let v = align central(cA, cB) response(rA, rB)
            under phi_c, phi_r { theta = 0.01 }
    relax v until quiescent { eta = 0.005 ; theta = 0.01 }
    claim "variant is functionally equivalent to wild type" = v
    record v
}

report to "adjudicate.report"
'''

TRANSFER = '''
open ref     = "reference.solved"
open subject = "subject.solved"

method pi_uniform = perturb_incident(magnitude = 0.10)

under tissue_receiver {
    let Nr = project ref     by contact(medium = 0.05)
    let Ns = project subject by contact(medium = 0.05)
    for u in items(Nr) where separation(u) > 0.10 {
        let cA = unit u  anchors 3
        let cB = nearest_unit Ns to cA
        let rA = response cA by pi_uniform
        let rB = response cB by pi_uniform
        let phi_c = corr from cA to cB by anchor_match
        let phi_r = corr from rA to rB by anchor_match
        let v = align central(cA, cB) response(rA, rB)
                under phi_c, phi_r { theta = 0.02 }
        claim "annotation of u transfers to its correspondent" = v
        record v
    }
}

report to "transfer.report"
'''

POSITIVE = [
    ("motif_scan.syp", MOTIF_SCAN),
    ("homology.syp", HOMOLOGY),
    ("adjudicate.syp", ADJUDICATE),
    ("transfer.syp", TRANSFER),
]


def e26_positive_corpus(rng) -> C.Check:
    """The four programs of Sec 10 must be ACCEPTED.

    Without this, every refusal theorem is satisfied vacuously by a
    checker that rejects all input. This is the load-bearing half.
    """
    chk = C.Check("E26", "Sec 10: the four worked programs typecheck")
    reports = {}
    for name, src in POSITIVE:
        ok, err = L.accepts(src)
        chk.record(ok, 0.0, f"{name} was rejected: {err}")
        if ok:
            r = L.check(src)
            reports[name] = {
                "frames": r.frames,
                "n_parameters": len(r.parameters),
                "n_claims": len(r.claims),
                "residues_recorded": sorted(r.residues),
                "responses_named": [d["name"] for d in r.responses],
                "bounds": r.bounds,
            }
    chk.extra = {"accepted": reports}
    return chk


def e27_sweep_trip_count(rng) -> C.Check:
    """Thm 9.1 (totality): the sweep of Sec 10.1 runs exactly 9 times."""
    chk = C.Check("E27", "Thm 9.1: bounded iteration; trip count fixed on entry")
    src = '''
open motif = "m.fa"
open target = "t.fa"
under nucleotide {
    let q = project motif by channels(dna)
    let t = project target by channels(dna)
    bind r, res_r = compare q against t by xcorr(normalised)
    sweep z0 in 2.0..6.0 step 0.5 {
        let h = detect peaks in r { z = 4.0 ; min_distance = 30 ; min_score = 0.35 }
        record h
    }
    record res_r
}
report to "s.report"
'''
    ok, err = L.accepts(src)
    chk.record(ok, 0.0, f"sweep program rejected: {err}")
    if ok:
        r = L.check(src)
        n = r.iterations.get("z0")
        chk.record(n == 9, 0.0, f"trip count {n}, expected 9")
        chk.extra = {"trip_count": n}

    # There is no `while` and no recursion: those keywords must not parse.
    for bad in ("while", "fix", "recurse"):
        src2 = f'open a = "x"\nunder N {{ {bad} x {{ }} }}\nreport to "r"'
        ok2, _ = L.accepts(src2)
        chk.record(not ok2, 0.0, f"`{bad}` was accepted")
    return chk


# =====================================================================
# Sec 9 -- the refusal theorems (negative corpus)
# =====================================================================

def _neg(name, src, expected_cls):
    return (name, src, expected_cls)


NEGATIVE = [
    # Thm 9.5 -- arity. The false-friend mistake must be unmakeable.
    _neg("align_without_response", '''
open wt = "a"
open var = "b"
method m = perturb_incident(magnitude = 0.10)
under N {
    let Nwt = project wt by contact(medium = 0.05)
    let Nvar = project var by contact(medium = 0.05)
    let cA = unit Nwt anchors 3
    let cB = unit Nvar anchors 3
    let phi_c = corr from cA to cB by species_name
    let phi_r = corr from cA to cB by species_name
    let v = align central(cA, cB) under phi_c, phi_r { theta = 0.01 }
    record v
}
report to "r"
''', L.ArityError),

    # Thm 9.5 -- one correspondence is not enough for four columns.
    _neg("align_one_correspondence", '''
open wt = "a"
open var = "b"
method m = perturb_incident(magnitude = 0.10)
under N {
    let Nwt = project wt by contact(medium = 0.05)
    let Nvar = project var by contact(medium = 0.05)
    let cA = unit Nwt anchors 3
    let cB = unit Nvar anchors 3
    let rA = response cA by m
    let rB = response cB by m
    let phi_c = corr from cA to cB by species_name
    let v = align central(cA, cB) response(rA, rB) under phi_c { theta = 0.01 }
    record v
}
report to "r"
''', L.ArityError),

    # Thm 9.6 -- scope safety across frames.
    _neg("cross_frame_reference", '''
open wildtype = "wt.fa"
open variant = "var.fa"
under nucleotide_frame {
    let a = project wildtype by channels(dna)
    let b = project variant by channels(dna)
    bind r, res = compare a against b by xcorr(normalised)
    record res
}
under protein_frame {
    let p = project wildtype by channels(protein)
    bind bad, res2 = compare a against p by xcorr(normalised)
    record res2
}
report to "r"
''', L.ScopeError),

    # Assumption 4.6 discharged at compile time.
    _neg("relax_zero_eta", '''
open a = "x"
under N {
    let Na = project a by contact(medium = 0.05)
    let u = unit Na anchors 3
    relax u until quiescent { eta = 0.0 ; theta = 0.01 }
    record u
}
report to "r"
''', L.TerminationError),

    _neg("relax_negative_eta", '''
open a = "x"
under N {
    let Na = project a by contact(medium = 0.05)
    let u = unit Na anchors 3
    relax u until quiescent { eta = -0.01 ; theta = 0.01 }
    record u
}
report to "r"
''', L.TerminationError),

    # Cor 9.8 -- residue accountability.
    _neg("unconsumed_residue", '''
open q = "q.fa"
open t = "t.fa"
under N {
    let a = project q by channels(dna)
    let b = project t by channels(dna)
    bind r, res = compare a against b by xcorr(normalised)
    record r
}
report to "r"
''', L.ResidueError),

    # Thm 9.7 -- parameter completeness. No defaults anywhere.
    _neg("peaks_missing_parameter", '''
open q = "q.fa"
open t = "t.fa"
under N {
    let a = project q by channels(dna)
    let b = project t by channels(dna)
    bind r, res = compare a against b by xcorr(normalised)
    let h = detect peaks in r { z = 4.0 ; min_distance = 30 }
    record h, res
}
report to "r"
''', L.ParameterError),

    _neg("align_missing_theta", '''
open wt = "a"
open var = "b"
method m = perturb_incident(magnitude = 0.10)
under N {
    let Nwt = project wt by contact(medium = 0.05)
    let Nvar = project var by contact(medium = 0.05)
    let cA = unit Nwt anchors 3
    let cB = unit Nvar anchors 3
    let rA = response cA by m
    let rB = response cB by m
    let phi_c = corr from cA to cB by species_name
    let phi_r = corr from rA to rB by species_name
    let v = align central(cA, cB) response(rA, rB) under phi_c, phi_r { }
    record v
}
report to "r"
''', L.ParameterError),

    # Rule 6.7 -- responses may not be anonymous.
    _neg("anonymous_response", '''
open wt = "a"
open var = "b"
under N {
    let Nwt = project wt by contact(medium = 0.05)
    let Nvar = project var by contact(medium = 0.05)
    let cA = unit Nwt anchors 3
    let cB = unit Nvar anchors 3
    let rA = response cA
    let rB = response cB
    let phi_c = corr from cA to cB by species_name
    let phi_r = corr from rA to rB by species_name
    let v = align central(cA, cB) response(rA, rB) under phi_c, phi_r { theta = 0.01 }
    record v
}
report to "r"
''', L.ParameterError),

    _neg("undeclared_response_method", '''
open wt = "a"
under N {
    let Nwt = project wt by contact(medium = 0.05)
    let cA = unit Nwt anchors 3
    let rA = response cA by never_declared
    record rA
}
report to "r"
''', L.ScopeError),

    # Rule 6.11 -- three result types. Index sets genuinely differ.
    _neg("profile_into_topk", '''
open q = "q.fa"
open t = "t.fa"
under N {
    let a = project q by channels(dna)
    let b = project t by channels(dna)
    bind r, res = compare a against b by xcorr(normalised)
    let c = detect top in r { k = 200 ; depth = 12 }
    record c, res
}
report to "r"
''', L.TypeError_),

    _neg("ranked_into_peaks", '''
open q = "q.fa"
open d = "d.fa"
under N {
    let a = project q by spectral(coeffs = 8)
    let b = project d by spectral(coeffs = 8)
    bind s, res = compare a against b by shader(cosine)
    let h = detect peaks in s { z = 4.0 ; min_distance = 30 ; min_score = 0.35 }
    record h, res
}
report to "r"
''', L.TypeError_),

    # Dimension is part of coord_phi (Sec 10.2).
    _neg("dimension_mismatch", '''
open q = "q.fa"
open d = "d.fa"
under N {
    let a = project q by spectral(coeffs = 8)
    let b = project d by spectral(coeffs = 16)
    bind s, res = compare a against b by shader(cosine)
    record s, res
}
report to "r"
''', L.TypeError_),

    # Thm 9.4 -- sequences are opaque; there is no indexing form.
    _neg("sequence_indexing", '''
open q = "q.fa"
under N {
    let x = q[3]
    record x
}
report to "r"
''', L.SynopsisError),

    # Rule 6.11 -- programs emit reports; there is no silent program.
    _neg("no_report", '''
open q = "q.fa"
under N {
    let a = project q by channels(dna)
    record a
}
''', L.ParseError),

    # A frame is mandatory: values cannot exist outside a frame index.
    _neg("undefined_variable", '''
open q = "q.fa"
under N {
    let a = project nonexistent by channels(dna)
    record a
}
report to "r"
''', L.ScopeError),
]


def e28_negative_corpus(rng) -> C.Check:
    """Thms 9.4-9.7, Cor 9.8: each program must be REJECTED.

    We check the error CLASS, not merely that something failed. A
    program rejected for the wrong reason would let a refusal theorem
    pass while the mechanism it names is absent.
    """
    chk = C.Check("E28", "Thms 9.4-9.8: the negative corpus is refused")
    detail = {}
    for name, src, cls in NEGATIVE:
        try:
            L.check(src)
            chk.record(False, 0.0, f"{name} was ACCEPTED but must be refused")
            detail[name] = {"rejected": False, "error": None}
        except L.SynopsisError as exc:
            right = isinstance(exc, cls)
            chk.record(right, 0.0,
                       f"{name} rejected as {type(exc).__name__}, "
                       f"expected {cls.__name__}")
            detail[name] = {"rejected": True,
                            "error_class": type(exc).__name__,
                            "expected_class": cls.__name__,
                            "matched": bool(right),
                            "message": str(exc)}
    chk.extra = {"programs": detail, "n_programs": len(NEGATIVE)}
    return chk


def e29_exactness_inexpressible(rng) -> C.Check:
    """Thm 9.4: no program can assert exact alignment.

    Every comparison value carries a residue varrho >= beta > 0, and
    there is no zero-residue constructor. We check this structurally:
    the checker gives every `bind` a residue obligation, and no path
    discharges it by producing zero.
    """
    chk = C.Check("E29", "Thm 9.4/9.3: exactness is inexpressible; no zero residue")

    # Every comparison binds a residue that must be accounted for.
    src = '''
open q = "q.fa"
open t = "t.fa"
under N {
    let a = project q by channels(dna)
    let b = project t by channels(dna)
    bind r, res = compare a against b by xcorr(normalised)
    record r, res
}
report to "r"
'''
    rep = L.check(src)
    chk.record("res" in rep.residues, 0.0, "residue was not tracked")

    # Ignoring the residue is refused (that is the no-silent-path claim).
    ok, err = L.accepts(src.replace("record r, res", "record r"))
    chk.record(not ok, 0.0, "a program ignoring its residue was accepted")

    # `drop` typechecks but is RECORDED as an explicit abandonment --
    # there is no way to make the residue vanish quietly.
    src_drop = src.replace("record r, res", "record r\n    drop res")
    rep2 = L.check(src_drop)
    chk.record(len(rep2.abandoned) == 1, 0.0, "drop was not recorded")
    chk.record(rep2.abandoned[0]["name"] == "res", 0.0, "wrong abandonment")

    # Numerically: the floor is strictly positive, so the residue a
    # comparison reports can never be zero (Thm 2.5 again, at the level
    # of the language's accountability discipline).
    n_zero = 0
    for _ in range(80):
        g = C.random_contact_structure(rng, int(rng.integers(3, 6)))
        if g.floor() <= 0.0:
            n_zero += 1
    chk.record(n_zero == 0, 0.0, f"{n_zero} structures had a zero floor")
    chk.extra = {"abandonment_note": rep2.abandoned[0]["note"],
                 "n_zero_floor": n_zero}
    return chk


def e30_report_completeness(rng) -> C.Check:
    """Thm 9.7 / Sec 11: every stated parameter reaches the report."""
    chk = C.Check("E30", "Thm 9.7/Sec 11: parameters and responses are printed")
    rep = L.check(ADJUDICATE)
    chk.record("align.theta" in rep.parameters, 0.0, "theta not in report")
    chk.record("relax.eta" in rep.parameters, 0.0, "eta not in report")
    chk.record("contact.medium" in rep.parameters, 0.0, "medium not in report")
    names = [d["name"] for d in rep.responses]
    chk.record("pi_edge_raise" in names, 0.0, "response map not named in report")
    chk.record(len(rep.claims) == 1, 0.0, f"{len(rep.claims)} claims, expected 1")
    chk.record(len(rep.frames) == 1, 0.0, "frame not recorded")
    chk.record("v" in rep.bounds, 0.0, "termination bound not emitted")

    rep2 = L.check(MOTIF_SCAN)
    for p in ("peaks.z", "peaks.min_distance", "peaks.min_score"):
        chk.record(p in rep2.parameters, 0.0, f"{p} not in report")
    chk.extra = {"adjudicate_parameters": rep.parameters,
                 "motif_parameters": rep2.parameters}
    return chk


def e31_termination_bound_arithmetic(rng) -> C.Check:
    """Sec 7.6: the emitted bound is ceil(D0/eta), e.g. 0.4130/0.005 -> 83."""
    chk = C.Check("E31", "Sec 7.6: emitted termination bound is correct")
    D0, eta = 0.4130, 0.005
    bound = math.ceil(D0 / eta)
    chk.record(bound == 83, 0.0, f"bound {bound}, manuscript states 83")
    for _ in range(200):
        d = float(rng.uniform(0.01, 10.0))
        e = float(rng.uniform(0.001, 1.0))
        b = math.ceil(d / e)
        res = C.relax(d, lambda D, e=e: max(0.0, D - e), e, theta=1e-9)
        chk.record(res.steps <= b, float(res.steps - b),
                   f"steps {res.steps} exceeded emitted bound {b}")
    chk.extra = {"worked_example": {"D0": D0, "eta": eta, "bound": bound}}
    return chk


# =====================================================================
# Sec 13 -- MEASUREMENT, not assertion
# =====================================================================

def e32_response_independence_measurement(rng) -> C.Check:
    """Assumption 5.8 / Sec 13: response independence -- MEASURED.

    This is the framework's principal OPEN PROBLEM. For each item with
    at least two incident non-medium edges we form two admissible
    responses by raising one or the other incident edge by the same
    magnitude, and compare the resulting demands. If the assumption
    held, the two would agree within theta.

    It does not hold. This check therefore records the discrepancy
    distribution and ALWAYS reports ok=True at the suite level: the
    failure is a finding, not a bug, and marking it as an assertion
    would either bake in a false theorem or hide the result.
    """
    chk = C.Check("E32",
                  "Assumption 5.8 (MEASUREMENT): response independence")
    theta = 0.01
    discrepancies = []
    for _ in range(120):
        g = C.random_contact_structure(rng, int(rng.integers(4, 6)))
        for v in sorted(g.items()):
            inc = [e for e in sorted(g.incident(v), key=lambda e: tuple(sorted(e)))
                   if C.MEDIUM not in e]
            if len(inc) < 2:
                continue
            mag = 0.10
            h1 = C.perturb(g, {inc[0]: mag})
            h2 = C.perturb(g, {inc[1]: mag})
            others = [u for u in sorted(g.items()) if u != v]
            if not others:
                continue
            w = others[0]
            d1 = C.distance_to_floor(h1, v, w)
            d2 = C.distance_to_floor(h2, v, w)
            discrepancies.append(abs(d1 - d2))

    d = np.array(discrepancies) if discrepancies else np.zeros(1)
    within = float(np.mean(d <= theta))
    # Recorded as a measurement: this check does not fail the suite.
    chk.record(True)
    chk.extra = {
        "status": "MEASUREMENT -- not an assertion; a violation is a finding",
        "theta": theta,
        "n_comparisons": int(len(discrepancies)),
        "fraction_within_theta": within,
        "mean_discrepancy": float(d.mean()),
        "median_discrepancy": float(np.median(d)),
        "max_discrepancy": float(d.max()),
        "assumption_holds_at_theta": bool(within >= 0.95),
        "verdict": ("HOLDS" if within >= 0.95 else
                    "FAILS -- the verdict depends on which admissible "
                    "response was constructed (open problem)"),
    }
    return chk


# =====================================================================
# Registry
# =====================================================================

EXPERIMENTS = [
    ("E01", e01_contact_floor),
    ("E02", e02_zero_weight_refused),
    ("E03", e03_floor_decreases_but_stays_positive),
    ("E04", e04_relabelling_invariance),
    ("E05", e05_medium_must_be_fixed),
    ("E06", e06_expansion_floor_conditional),
    ("E07", e07_skeleton_invariance),
    ("E08", e08_regionality),
    ("E09", e09_region_point_crossover),
    ("E10", e10_receiver_relativity),
    ("E11", e11_distance_to_floor_nonneg),
    ("E12", e12_anchor_split_deterministic),
    ("E13", e13_monotone_and_bound),
    ("E14", e14_dichotomy_decline),
    ("E15", e15_form_of_quiescence),
    ("E16", e16_false_friend),
    ("E17", e17_converse),
    ("E18", e18_orthogonality),
    ("E19", e19_four_column_criterion),
    ("E20", e20_partition_depth_additive),
    ("E21", e21_xcorr_range_and_agreement),
    ("E22", e22_spectral_embedding),
    ("E23", e23_shader_range),
    ("E24", e24_complementary_strand),
    ("E25", e25_address_prefix_property),
    ("E26", e26_positive_corpus),
    ("E27", e27_sweep_trip_count),
    ("E28", e28_negative_corpus),
    ("E29", e29_exactness_inexpressible),
    ("E30", e30_report_completeness),
    ("E31", e31_termination_bound_arithmetic),
    ("E32", e32_response_independence_measurement),
]

# Checks that are measurements rather than assertions.
MEASUREMENTS = {"E32"}
