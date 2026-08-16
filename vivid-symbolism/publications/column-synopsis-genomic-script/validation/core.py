"""
core.py -- the mathematical substrate of the synopsis manuscript.

This module implements Definitions 2.1-2.13, 3.x and 4.x of
synopsis-genomic-scripting.tex directly, with no shortcuts that could
make a validation pass for the wrong reason.

Two deliberate implementation choices:

  * Minimum cuts are computed by EXHAUSTIVE ENUMERATION over separating
    vertex subsets, not by max-flow. The theory (Def 2.2) defines the cut
    as a minimum over separating sets, and an approximate max-flow could
    agree with an approximate expectation and hide a real disagreement.
    This is exponential and is affordable only on small structures, which
    is exactly the regime the validation uses. Remark in Sec 8.4 of the
    manuscript commits to this.

  * Weights are strictly positive by construction (Def 2.1). add_edge
    REFUSES a non-positive weight rather than clamping it, because the
    contact floor theorem (Thm 2.5) is false without strict positivity
    and a silent clamp would manufacture the theorem it is meant to test.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

TOL = 1e-12
MEDIUM = "__medium__"


# =====================================================================
# Definition 2.1 / 2.3 -- finite weighted graph, contact structure
# =====================================================================

@dataclass
class ContactStructure:
    """A finite weighted graph with a distinguished medium vertex.

    Weights live in R_{>0}. An edge of weight zero is not a light edge;
    it is the absence of an edge (Def 2.1), and we enforce that.
    """

    weights: dict[frozenset[str], float] = field(default_factory=dict)
    medium: str = MEDIUM

    # -- construction ------------------------------------------------
    def add_edge(self, u: str, v: str, w: float) -> None:
        if u == v:
            raise ValueError(f"no self-loops: {u}")
        if w <= 0.0:
            raise ValueError(
                f"weights must be strictly positive (Def 2.1); got {w} on "
                f"{{{u},{v}}}. A zero weight is the ABSENCE of an edge."
            )
        self.weights[frozenset((u, v))] = float(w)

    def add_medium_edges(self, w: float) -> None:
        """Attach the medium to every item that is not already attached.

        Def 2.3 requires the medium adjacent to EVERY other vertex; this
        is what makes Thm 2.5 go through.
        """
        for v in sorted(self.items()):
            e = frozenset((v, self.medium))
            if e not in self.weights:
                self.add_edge(v, self.medium, w)

    # -- accessors ---------------------------------------------------
    @property
    def vertices(self) -> set[str]:
        out: set[str] = set()
        for e in self.weights:
            out |= set(e)
        return out

    def items(self) -> set[str]:
        return self.vertices - {self.medium}

    def weight(self, u: str, v: str) -> float:
        return self.weights.get(frozenset((u, v)), 0.0)

    def incident(self, v: str) -> list[frozenset[str]]:
        return [e for e in self.weights if v in e]

    def floor(self) -> float:
        """beta = min over edges of w(e).  Def 2.3."""
        if not self.weights:
            raise ValueError("floor undefined on an edgeless structure")
        return min(self.weights.values())

    def local_floor(self, v: str) -> float:
        """beta_k(v) = min weight incident to v.  Def 2.6."""
        inc = self.incident(v)
        if not inc:
            raise ValueError(f"vertex {v} has no incident edges")
        return min(self.weights[e] for e in inc)

    def total_weight(self) -> float:
        """Omega.  Def 2.3."""
        return sum(self.weights.values())

    # -- cuts --------------------------------------------------------
    def cut_weight(self, S: Iterable[str]) -> float:
        """cut(S) = sum of weights of edges with exactly one end in S."""
        Sset = set(S)
        return sum(w for e, w in self.weights.items() if len(e & Sset) == 1)

    def min_cut(self, source: str, sink: str) -> tuple[float, frozenset[str]]:
        """Exact minimum (source, sink)-cut by subset enumeration.

        Returns (weight, minimising set). Deterministic tie-breaking:
        among minimisers, the lexicographically smallest sorted tuple
        wins, so the function is a function and results reproduce.
        """
        if source == sink:
            raise ValueError("source and sink must differ")
        verts = sorted(self.vertices)
        if source not in verts or sink not in verts:
            raise ValueError(f"{source} or {sink} not in structure")
        free = [v for v in verts if v not in (source, sink)]

        best_w = math.inf
        best_S: frozenset[str] = frozenset()
        for r in range(len(free) + 1):
            for combo in itertools.combinations(free, r):
                S = frozenset((source,) + combo)
                w = self.cut_weight(S)
                if w < best_w - TOL:
                    best_w, best_S = w, S
                elif abs(w - best_w) <= TOL:
                    if tuple(sorted(S)) < tuple(sorted(best_S)):
                        best_S = S
        return best_w, best_S

    def separation_cost(self, v: str) -> float:
        """sigma(v) = sigma(v | medium).  Def 2.3."""
        return self.min_cut(v, self.medium)[0]

    # -- structural ops ----------------------------------------------
    def copy(self) -> "ContactStructure":
        g = ContactStructure(medium=self.medium)
        g.weights = dict(self.weights)
        return g

    def relabel(self, mapping: dict[str, str]) -> "ContactStructure":
        """Apply a weighted isomorphism (Def 2.7). Must fix the medium."""
        if self.medium in mapping and mapping[self.medium] != self.medium:
            raise ValueError("a weighted isomorphism must fix the medium")
        g = ContactStructure(medium=self.medium)
        for e, w in self.weights.items():
            u, v = tuple(e)
            g.add_edge(mapping.get(u, u), mapping.get(v, v), w)
        return g

    def delete_item(self, u: str) -> "ContactStructure":
        g = ContactStructure(medium=self.medium)
        for e, w in self.weights.items():
            if u not in e:
                g.weights[e] = w
        return g


# =====================================================================
# Definition 2.13 -- perturbation
# =====================================================================

def perturb(g: ContactStructure, delta: dict[frozenset[str], float]) -> ContactStructure:
    """w -> w + delta, delta >= 0 (Def 2.13).

    Non-negativity is enforced: a negative perturbation would create
    separation for free and Prop 2.14 would be false.
    """
    out = g.copy()
    for e, d in delta.items():
        if d < 0.0:
            raise ValueError(f"perturbations are non-negative (Def 2.13); got {d}")
        if e not in out.weights:
            raise ValueError(f"perturbation on a non-edge {set(e)}")
        out.weights[e] = out.weights[e] + d
    return out


# =====================================================================
# Definitions 4.1 / 4.3 -- alignment, anchors, residue
# =====================================================================

def alignment(g: ContactStructure, u: str, v: str) -> float:
    """a(u,v) = sigma(u|v) / Omega.  Def 4.1."""
    return g.min_cut(u, v)[0] / g.total_weight()


def floor_ratio(g: ContactStructure) -> float:
    return g.floor() / g.total_weight()


def distance_to_floor(g: ContactStructure, u: str, v: str) -> float:
    """Delta(u,v) = a(u,v) - beta/Omega.  Def 4.1."""
    return alignment(g, u, v) - floor_ratio(g)


def ablation_score(g: ContactStructure, x: str, u: str) -> float:
    """alpha(x;u) = sigma_G(x) - sigma_{G\\u}(x).  Def 4.3."""
    if u == x or u == g.medium:
        return -math.inf
    before = g.separation_cost(x)
    h = g.delete_item(u)
    if x not in h.vertices or h.medium not in h.vertices:
        return -math.inf
    return before - h.separation_cost(x)


def anchors_and_residue(g: ContactStructure, x: str, r: int
                        ) -> tuple[list[str], list[str]]:
    """Split items into r anchors and the residue.  Def 4.3.

    Ties are broken by a FIXED total order on names. Without this the
    split is not a function and the language could not be reproducible.
    """
    cands = sorted(g.items() - {x})
    scored = sorted(cands, key=lambda u: (-ablation_score(g, x, u), u))
    return scored[:r], scored[r:]


# =====================================================================
# Definition 4.4 -- cross-demand
# =====================================================================

def cross_demand(g_from: ContactStructure,
                 g_to: ContactStructure,
                 residue_to: list[str],
                 correspondence: dict[str, str]) -> float:
    """D_{A->B}, Eq (4.1).

    Sum over residue items of B of max(0, a_B(i, phi(i)) - beta_B/Omega_B).
    Reads no labels beyond the correspondence itself: the quantity is
    computed from cuts. That is what makes it blind.
    """
    beta_ratio = floor_ratio(g_to)
    total = 0.0
    for i in residue_to:
        j = correspondence.get(i)
        if j is None or j not in g_to.vertices or i not in g_to.vertices:
            continue
        if i == j:
            continue
        total += max(0.0, alignment(g_to, i, j) - beta_ratio)
    return total


# =====================================================================
# Definition 4.5 / Assumption 4.6 -- relaxation
# =====================================================================

@dataclass
class RelaxationResult:
    quiescent: bool
    declined: bool
    steps: int
    demands: list[float]
    final_demand: float
    bound: int


def relax(initial_demand: float,
          update: Callable[[float], float | None],
          eta: float,
          theta: float = TOL,
          max_steps: int = 10_000) -> RelaxationResult:
    """Run a relaxation under Assumption 4.6.

    `update(D)` returns the next demand, or None to DECLINE (dichotomy
    case (ii), Thm 4.7).

    The effectiveness assumption is ENFORCED, not assumed: if an update
    returns a value that does not decrease the demand by at least eta,
    we raise. A silently-weak update would let the loop run long and make
    Thm 4.7 look false for the wrong reason.
    """
    if eta <= 0.0:
        raise ValueError("relax requires eta > 0 (Assumption 4.6)")

    bound = math.ceil(initial_demand / eta) if initial_demand > 0 else 0
    D = float(initial_demand)
    demands = [D]
    steps = 0

    while D > theta:
        if steps >= max_steps:
            raise RuntimeError("max_steps exceeded; bound violated")
        nxt = update(D)
        if nxt is None:
            return RelaxationResult(False, True, steps, demands, D, bound)
        # Assumption 4.6 requires U(D) <= D - eta OR quiescence. The
        # second disjunct matters: when D < eta the update must be
        # allowed to land on zero, which is not <= a negative number.
        # Demanding the first disjunct unconditionally would make the
        # final step of every short run look like a violation.
        if nxt > D - eta + TOL and nxt > theta:
            raise AssertionError(
                f"Assumption 4.6 violated: update gave {nxt} from {D} "
                f"with eta={eta} (needed <= {D - eta} or <= theta={theta})"
            )
        D = nxt
        demands.append(D)
        steps += 1

    return RelaxationResult(True, False, steps, demands, D, bound)


# =====================================================================
# Definition 5.5 -- four columns
# =====================================================================

@dataclass
class Column:
    graph: ContactStructure
    unit: str
    anchors: list[str]
    residue: list[str]


def make_column(g: ContactStructure, unit: str, r: int) -> Column:
    a, res = anchors_and_residue(g, unit, r)
    return Column(graph=g, unit=unit, anchors=a, residue=res)


def four_column_verdict(central_a: Column,
                        central_b: Column,
                        resp_a: Column,
                        resp_b: Column,
                        corr_central: dict[str, str],
                        corr_response: dict[str, str],
                        theta: float,
                        declined: bool = False) -> dict:
    """The three-valued verdict of Def 5.5.

    DECLINE is a first-class outcome sourced from Thm 4.7(ii), not a
    hedge -- see Remark 5.7. It is passed in because whether a relaxation
    declined is a property of the relaxation, not of the four columns.
    """
    inv_c = {v: k for k, v in corr_central.items()}
    inv_r = {v: k for k, v in corr_response.items()}

    d_ab = cross_demand(central_a.graph, central_b.graph,
                        central_b.residue, corr_central)
    d_ba = cross_demand(central_b.graph, central_a.graph,
                        central_a.residue, inv_c)
    d_ra = cross_demand(resp_a.graph, resp_b.graph,
                        resp_b.residue, corr_response)
    d_rb = cross_demand(resp_b.graph, resp_a.graph,
                        resp_a.residue, inv_r)

    central = d_ab + d_ba
    response = d_ra + d_rb

    central_q = central <= theta
    response_q = response <= theta

    if declined:
        verdict = "DECLINE"
    elif central_q and response_q:
        verdict = "CORRESPOND"
    else:
        verdict = "DIVERGE"

    return {
        "verdict": verdict,
        "central_demand": central,
        "response_demand": response,
        "d_ab": d_ab, "d_ba": d_ba, "d_ra": d_ra, "d_rb": d_rb,
        "central_quiescent": bool(central_q),
        "response_quiescent": bool(response_q),
        "theta": theta,
    }


# =====================================================================
# Definition 7.10 -- partition depth
# =====================================================================

def partition_depth(parts: Iterable[int], base: float = 3.0) -> float:
    """depth = sum_j log_b(k_j).  Def 7.10."""
    return sum(math.log(k, base) for k in parts)


# =====================================================================
# Random instance generators
# =====================================================================

def random_contact_structure(rng: np.random.Generator,
                             n_items: int,
                             p_edge: float = 0.55,
                             w_lo: float = 0.05,
                             w_hi: float = 2.0) -> ContactStructure:
    """A random contact structure with strictly positive weights."""
    g = ContactStructure()
    names = [f"i{k}" for k in range(n_items)]
    for a, b in itertools.combinations(names, 2):
        if rng.random() < p_edge:
            g.add_edge(a, b, float(rng.uniform(w_lo, w_hi)))
    for v in names:
        if not g.incident(v):
            g.add_edge(v, MEDIUM, float(rng.uniform(w_lo, w_hi)))
    for v in names:
        e = frozenset((v, MEDIUM))
        if e not in g.weights:
            g.add_edge(v, MEDIUM, float(rng.uniform(w_lo, w_hi)))
    return g


def two_cluster_structure(r: int, W: float, beta: float) -> ContactStructure:
    """The witness of Thm 3.1 / Cor 5.2.

    Two r-cliques at intra-weight W, joined by ONE bridge of weight beta,
    every item attached to the medium at beta.
    """
    g = ContactStructure()
    A = [f"a{k}" for k in range(r)]
    B = [f"b{k}" for k in range(r)]
    for x, y in itertools.combinations(A, 2):
        g.add_edge(x, y, W)
    for x, y in itertools.combinations(B, 2):
        g.add_edge(x, y, W)
    g.add_edge("a0", "b0", beta)
    for v in A + B:
        g.add_edge(v, MEDIUM, beta)
    return g


def pendant_pair_structure(beta: float, n_filler: int = 3,
                           w_filler: float = 1.0) -> ContactStructure:
    """The witness of Thm 5.1.

    Two PENDANT items u, v attached only to the medium at weight beta,
    so sigma(u|v) = beta exactly and Delta(u,v) = 0. Remark 3.2 records
    that a light edge between NON-pendant items does not suffice -- that
    was our first, wrong, witness.
    """
    g = ContactStructure()
    g.add_edge("u", MEDIUM, beta)
    g.add_edge("v", MEDIUM, beta)
    filler = [f"f{k}" for k in range(n_filler)]
    for x, y in itertools.combinations(filler, 2):
        g.add_edge(x, y, w_filler)
    for x in filler:
        g.add_edge(x, MEDIUM, w_filler)
    return g


# =====================================================================
# Check bookkeeping
# =====================================================================

@dataclass
class Check:
    name: str
    claim: str
    n: int = 0
    passed: int = 0
    max_err: float = 0.0
    failures: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def record(self, ok: bool, err: float = 0.0, msg: str = "") -> None:
        self.n += 1
        if ok:
            self.passed += 1
        else:
            if len(self.failures) < 8:
                self.failures.append(msg)
        self.max_err = max(self.max_err, float(err))

    @property
    def ok(self) -> bool:
        return self.passed == self.n and self.n > 0

    def summary(self) -> dict:
        return {
            "name": self.name,
            "claim": self.claim,
            "n": int(self.n),
            "passed": int(self.passed),
            "ok": bool(self.ok),
            "max_err": float(self.max_err),
            "failures": list(self.failures),
        }
