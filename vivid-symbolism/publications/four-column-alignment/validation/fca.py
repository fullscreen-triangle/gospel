"""
fca.py -- reference implementation of the four-column alignment framework.

Everything here follows the paper directly. Section/theorem numbers in
docstrings refer to four-column-alignment.tex.

Design notes:

  * Minimum cuts are computed EXACTLY, by brute-force enumeration over all
    vertex subsets separating the two terminals. This is exponential and
    deliberately so: the validation must not be able to pass because an
    approximate max-flow implementation happened to agree with an
    approximate expectation. Graphs in the suite are kept small (n <= 14)
    so exact enumeration is affordable.

  * No randomness anywhere except through an explicitly seeded numpy
    Generator passed in by the caller.

  * Floats are compared with an absolute tolerance; the tolerance is a
    module constant so the suite reports it rather than hiding it.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TOL = 1e-12

MEDIUM = "__medium__"


# =====================================================================
# Contact graphs (Definitions 3.1 - 3.4)
# =====================================================================


@dataclass
class ContactGraph:
    """A finite weighted graph with a distinguished medium vertex.

    Definition 3.3: the medium is adjacent to every item. We do not
    enforce that on construction -- `add_medium_edges` does it -- because
    several tests build a graph incrementally and check the floor before
    and after.
    """

    weights: dict[frozenset[str], float] = field(default_factory=dict)
    medium: str = MEDIUM

    # -- construction ------------------------------------------------

    def add_edge(self, u: str, v: str, w: float) -> None:
        if u == v:
            raise ValueError("no self-loops")
        if w <= 0.0:
            raise ValueError(f"Definition 3.1 requires w > 0; got {w}")
        self.weights[frozenset((u, v))] = float(w)

    def add_medium_edges(self, w: float) -> None:
        """Make every item adjacent to the medium at weight `w`."""
        for v in list(self.items()):
            e = frozenset((self.medium, v))
            if e not in self.weights:
                self.weights[e] = float(w)

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

    # -- Theorem 3.2: the contact floor ------------------------------

    def floor(self) -> float:
        """beta = min edge weight. Theorem 3.2."""
        if not self.weights:
            raise ValueError("floor undefined on an edgeless graph")
        return min(self.weights.values())

    def local_floor(self, v: str) -> float:
        """beta_k(v), Definition 3.5: min weight incident to v."""
        inc = self.incident(v)
        if not inc:
            raise ValueError(f"{v} has no incident edges")
        return min(self.weights[e] for e in inc)

    def total_weight(self) -> float:
        """Omega, used to normalise alignment (Definition 6.1)."""
        return sum(self.weights.values())

    # -- cuts (Definition 3.2) ---------------------------------------

    def cut_weight(self, S: set[str]) -> float:
        rest = self.vertices - S
        return sum(
            w for e, w in self.weights.items()
            if len(e & S) == 1 and len(e & rest) == 1
        )

    def min_cut(self, source: str, sink: str) -> tuple[float, frozenset[str]]:
        """Exact min cut separating `source` from `sink`.

        Brute force over all subsets containing source and excluding sink.
        Returns (weight, minimising set). Deterministic tie-breaking: among
        minimisers, the lexicographically smallest sorted tuple wins, so
        the result does not depend on iteration order.
        """
        verts = sorted(self.vertices)
        if source not in verts or sink not in verts:
            raise ValueError("terminals must be vertices")
        free = [v for v in verts if v not in (source, sink)]

        best_w = float("inf")
        best_S: frozenset[str] | None = None
        for r in range(len(free) + 1):
            for combo in itertools.combinations(free, r):
                S = {source, *combo}
                w = self.cut_weight(S)
                key_better = w < best_w - TOL
                key_tie = abs(w - best_w) <= TOL
                if key_better or (
                    key_tie
                    and best_S is not None
                    and tuple(sorted(S)) < tuple(sorted(best_S))
                ):
                    best_w, best_S = w, frozenset(S)
        assert best_S is not None
        return best_w, best_S

    def separation_cost(self, v: str) -> float:
        """sigma(v), Definition 3.4: min cut from v to the medium."""
        return self.min_cut(v, self.medium)[0]

    # -- misc --------------------------------------------------------

    def copy(self) -> "ContactGraph":
        g = ContactGraph(medium=self.medium)
        g.weights = dict(self.weights)
        return g

    def relabel(self, mapping: dict[str, str]) -> "ContactGraph":
        """Apply a weighted isomorphism (Definition 4.1).

        `mapping` must fix the medium and be a bijection on items.
        """
        if mapping.get(self.medium, self.medium) != self.medium:
            raise ValueError("a weighted isomorphism must fix the medium")
        g = ContactGraph(medium=self.medium)
        for e, w in self.weights.items():
            u, v = tuple(e)
            g.weights[frozenset((mapping.get(u, u), mapping.get(v, v)))] = w
        return g


# =====================================================================
# Perturbations (Definition 5.2)
# =====================================================================


def perturb(g: ContactGraph, delta: dict[frozenset[str], float]) -> ContactGraph:
    """Apply a non-negative perturbation: w -> w + delta.

    Definition 5.2 requires delta >= 0 (perturbations raise the cost of
    distinctions; they never create separation for free).
    """
    out = g.copy()
    for e, d in delta.items():
        if d < 0.0:
            raise ValueError("Definition 5.2 requires delta >= 0")
        if e not in out.weights:
            raise ValueError("perturbation must act on existing edges")
        out.weights[e] = out.weights[e] + d
    return out


# =====================================================================
# Alignment and cross-demand (Definitions 6.1, 6.3)
# =====================================================================


def alignment(g: ContactGraph, u: str, v: str) -> float:
    """a(u,v) = sigma(u|v) / Omega, Definition 6.1."""
    return g.min_cut(u, v)[0] / g.total_weight()


def floor_ratio(g: ContactGraph) -> float:
    """beta / Omega -- the value alignment takes at the floor."""
    return g.floor() / g.total_weight()


def distance_to_floor(g: ContactGraph, u: str, v: str) -> float:
    """a(u,v) - beta/Omega >= 0, Definition 6.1 / Proposition 6.2."""
    return alignment(g, u, v) - floor_ratio(g)


def ablation_score(g: ContactGraph, x: str, u: str) -> float:
    """sigma(x) - sigma_{G \\ u}(x), used for anchoring (Definition 6.4)."""
    if u == x or u == g.medium:
        return float("-inf")
    base = g.separation_cost(x)
    h = ContactGraph(medium=g.medium)
    h.weights = {e: w for e, w in g.weights.items() if u not in e}
    if x not in h.vertices or g.medium not in h.vertices:
        return float("-inf")
    return base - h.separation_cost(x)


def anchors_and_residue(
    g: ContactGraph, x: str, r: int
) -> tuple[list[str], list[str]]:
    """Definition 6.4. Ties broken by sorted item name, so deterministic."""
    cands = sorted(g.items() - {x})
    scored = sorted(cands, key=lambda u: (-ablation_score(g, x, u), u))
    anc = scored[:r]
    res = [u for u in cands if u not in set(anc)]
    return anc, res


def cross_demand(
    g_from: ContactGraph,
    g_to: ContactGraph,
    residue_to: list[str],
    correspondence: dict[str, str],
) -> float:
    """d_{A->B}, Equation (6.1).

    Sum over residue items of B of the above-floor alignment gap between
    the item and its correspondent. Terms are clamped at 0 -- by
    Proposition 6.2 they are non-negative, and the clamp only guards
    float noise.
    """
    beta_ratio = g_to.floor() / g_to.total_weight()
    total = 0.0
    for i in residue_to:
        j = correspondence.get(i)
        if j is None or j not in g_to.vertices:
            continue
        gap = alignment(g_to, i, j) - beta_ratio
        total += max(0.0, gap)
    return total


# =====================================================================
# Relaxation (Definition 6.6, Assumption 6.5, Theorems 6.7-6.8)
# =====================================================================


@dataclass
class RelaxationResult:
    quiescent: bool
    declined: bool
    steps: int
    demands: list[float]
    final_demand: float


def relax(
    initial_demand: float,
    update,
    eta: float,
    theta: float = TOL,
    max_steps: int = 10_000,
) -> RelaxationResult:
    """Drive a relaxation to quiescence or decline.

    `update(D) -> float | None` returns the new total demand, or None if
    no effective update exists (Theorem 6.8 case (ii)).

    Assumption 6.5 requires each update to decrease D by at least eta.
    We ENFORCE it: an update returning a decrease below eta is treated as
    a violation and raises, so a test cannot silently pass while breaking
    the assumption the theorem needs.
    """
    D = float(initial_demand)
    hist = [D]
    steps = 0
    while D > theta:
        if steps >= max_steps:
            raise RuntimeError("exceeded max_steps; Theorem 6.8 bound violated")
        nxt = update(D)
        if nxt is None:
            return RelaxationResult(False, True, steps, hist, D)
        nxt = float(nxt)
        if nxt > D - eta + TOL:
            raise AssertionError(
                f"Assumption 6.5 violated: {D} -> {nxt} is not a decrease of >= {eta}"
            )
        D = max(0.0, nxt)
        hist.append(D)
        steps += 1
    return RelaxationResult(True, False, steps, hist, D)


# =====================================================================
# Four-column construction (Construction 7.1, Definition 7.2)
# =====================================================================


@dataclass
class Column:
    graph: ContactGraph
    unit: str
    anchors: list[str]
    residue: list[str]


def make_column(g: ContactGraph, unit: str, r: int) -> Column:
    anc, res = anchors_and_residue(g, unit, r)
    return Column(graph=g, unit=unit, anchors=anc, residue=res)


def four_column_verdict(
    central_a: Column,
    central_b: Column,
    resp_a: Column,
    resp_b: Column,
    corr_central: dict[str, str],
    corr_response: dict[str, str],
    theta: float,
) -> dict:
    """Definition 7.2 / Theorem 7.3.

    Returns the four demands and the verdict. The verdict is
    CORRESPOND only when BOTH pairs are quiescent.
    """
    d_ab = cross_demand(central_a.graph, central_b.graph,
                        central_b.residue, corr_central)
    d_ba = cross_demand(central_b.graph, central_a.graph,
                        central_a.residue,
                        {v: k for k, v in corr_central.items()})
    d_ra = cross_demand(resp_a.graph, resp_b.graph,
                        resp_b.residue, corr_response)
    d_rb = cross_demand(resp_b.graph, resp_a.graph,
                        resp_a.residue,
                        {v: k for k, v in corr_response.items()})

    central = d_ab + d_ba
    response = d_ra + d_rb
    central_q = central <= theta
    response_q = response <= theta

    if central_q and response_q:
        verdict = "CORRESPOND"
    else:
        verdict = "DIVERGE"

    return {
        "d_central_ab": d_ab,
        "d_central_ba": d_ba,
        "d_response_ab": d_ra,
        "d_response_ba": d_rb,
        "central_demand": central,
        "response_demand": response,
        "central_quiescent": central_q,
        "response_quiescent": response_q,
        "verdict": verdict,
    }


# =====================================================================
# Pathway loading (Assumption 5.1)
# =====================================================================


def load_pathways(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["pathways"]


def pathway_to_contact_graph(
    pw: dict, medium_weight: str | float = "min_edge"
) -> ContactGraph:
    """Build a contact graph from a solved pathway.

    Edge weights are the solved contact costs. The medium is attached to
    every species; its weight is by default the minimum observed contact
    cost, which makes the medium the cheapest available distinction and
    so realises the floor without inventing a scale.
    """
    g = ContactGraph()
    for e in pw["edges"]:
        g.add_edge(e["source"], e["target"], e["cost"])
    if medium_weight == "min_edge":
        w = min(e["cost"] for e in pw["edges"])
    else:
        w = float(medium_weight)
    g.add_medium_edges(w)
    return g


# =====================================================================
# Reporting helper
# =====================================================================


class Check:
    """Accumulates pass/fail with a recorded max error."""

    def __init__(self, name: str, theorem: str):
        self.name = name
        self.theorem = theorem
        self.n = 0
        self.passed = 0
        self.max_err = 0.0
        self.failures: list[str] = []

    def record(self, ok: bool, err: float = 0.0, note: str = "") -> None:
        self.n += 1
        if ok:
            self.passed += 1
        elif len(self.failures) < 8:
            self.failures.append(note)
        self.max_err = max(self.max_err, float(err))

    @property
    def ok(self) -> bool:
        return self.n > 0 and self.passed == self.n

    def summary(self) -> dict:
        return {
            "check": self.name,
            "theorem": self.theorem,
            "n_checks": self.n,
            "n_passed": self.passed,
            "passed": self.ok,
            "max_error": self.max_err,
            "failures": self.failures,
        }


def random_contact_graph(
    rng: np.random.Generator,
    n_items: int,
    p_edge: float = 0.55,
    w_lo: float = 0.05,
    w_hi: float = 2.0,
) -> ContactGraph:
    """A random contact graph with strictly positive weights."""
    g = ContactGraph()
    items = [f"v{i}" for i in range(n_items)]
    for i, j in itertools.combinations(range(n_items), 2):
        if rng.random() < p_edge:
            g.add_edge(items[i], items[j], float(rng.uniform(w_lo, w_hi)))
    # guarantee every item is present even if the coin flips isolated it
    for v in items:
        if v not in g.vertices:
            g.weights[frozenset((g.medium, v))] = float(rng.uniform(w_lo, w_hi))
    g.add_medium_edges(float(rng.uniform(w_lo, w_hi)))
    return g
