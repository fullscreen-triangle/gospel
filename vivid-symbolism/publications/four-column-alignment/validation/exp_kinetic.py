"""
exp_kinetic.py -- Experiment 4: does kinetic constraint recover response
independence?

THE QUESTION
------------
E13 measures Assumption 7.5 and finds it violated: only 28.57% of
comparisons fall within theta = 0.01. The open question is whether that
is a fundamental failure of the four-column criterion or an artefact of
how the perturbations were chosen.

E13 chooses them combinatorially. For each item x it takes EVERY pair of
incident edges and raises each by the same flat magnitude (mag = 0.5).
That treats all pairs as equally admissible realisations of "perturb x".
Physically they are not. A perturbation of a metabolic network is
realised by an enzyme, and an enzyme cannot raise an arbitrary edge by an
arbitrary amount: the reachable perturbation is bounded by the kinetics
of the reaction it catalyses.

So the hypothesis under test is:

    Assumption 7.5 fails on the UNCONSTRAINED perturbation set because
    that set contains realisations no enzyme could produce. Restricted to
    the kinetically reachable set, the discrepancy should shrink.

WHY NOT BRENDA
--------------
The obvious move is to pull k_cat/K_M from BRENDA. I deliberately do not,
for a reason that is about the experiment rather than about convenience.
The claim being tested is that *kinetic constraint of any kind* removes
the artefact. If the constraint arrives as an external table keyed by
enzyme name, a negative result is unreadable: it could mean the
hypothesis is wrong, or that the join was wrong, or that the vendored
pathway species names did not match BRENDA's. The pathway data already
carries `conductance` and `flux` -- solved steady-state quantities that
ARE the kinetic realisability signal, in the same units as the costs they
must constrain, with no join to get wrong. That makes the result
attributable to the hypothesis and reproducible with no network access.

`conductance` is the natural constraint. In the solved contact map an
edge's conductance is how readily flux moves across it; an edge with high
conductance is one the network can actually act on, and a near-zero
conductance edge is kinetically inert -- perturbing it is a mathematical
operation with no enzymatic realisation.

WHAT IS MEASURED
----------------
Exactly what E13 measures -- |d1 - d2| over pairs of admissible
realisations -- with one variable changed: which pairs count as
admissible, and by how much each is raised. Everything else (theta, the
residue rule, the correspondence, the cross_demand call) is copied from
E13 unchanged, so the two numbers are comparable.

Four arms, sharing all other machinery:

  unconstrained   E13 exactly. The baseline; must reproduce 0.2857.
  gated           Only edges with conductance >= the pathway median are
                  admissible. Flat magnitude, as E13. Isolates the effect
                  of RESTRICTING THE SET.
  scaled          All pairs admissible, but each edge is raised by an
                  amount proportional to its conductance rather than a
                  flat 0.5. Isolates the effect of SCALING THE MAGNITUDE.
  kinetic         Both: gated set, scaled magnitude. The full hypothesis.

The two middle arms exist so that a positive result in `kinetic` can be
attributed. Without them, "constraining helps" would not distinguish
"only reachable edges count" from "reachable edges move less", and those
are different claims about why the assumption failed.

NEGATIVE CONTROL
----------------
`scaled` and `kinetic` both shrink the perturbation magnitude, and a
smaller perturbation trivially produces a smaller discrepancy. That
confound would let the experiment claim success for a reason that has
nothing to do with kinetics. So there is a fourth comparison:

  shuffled        The conductance values are permuted across edges within
                  each pathway, then the `kinetic` arm is run on the
                  shuffled assignment.

Shuffling preserves the multiset of magnitudes exactly and destroys only
the correspondence between an edge and its own conductance. If `kinetic`
beats `shuffled`, the improvement is carried by kinetic STRUCTURE. If
they match, the improvement is a magnitude artefact and the hypothesis is
not supported -- and that is a real result about the framework's
boundary, which is why the arm is here.

The control is run as a PERMUTATION TEST over N_PERM draws, not as a
single shuffle. This is not fastidiousness: the gated arms retain only
~8 comparisons, and at that size one draw is worthless. Measured here, a
single shuffle returned 0.2222 -- comfortably below the kinetic arm --
while the distribution over 200 draws has mean 0.4745 and gives
P(shuffled >= kinetic) = 0.49. A one-draw control would have reported a
positive result that the permutation distribution does not support.

Run:  python exp_kinetic.py
"""

from __future__ import annotations

import itertools
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fca import (
    ContactGraph,
    cross_demand,
    load_pathways,
    perturb,
)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "pathways.json"
RESULTS = HERE / "results"

SEED = 20260815
THETA = 0.01          # identical to E13
BASE_MAG = 0.5        # identical to E13
N_PERM = 200          # permutation draws for the negative control
TOL_ZERO = 1e-12      # what counts as an exactly-zero discrepancy


# =====================================================================
# Graph construction, carrying kinetics alongside the cost
# =====================================================================


def graph_and_kinetics(
    pw: dict, conductance_override: dict[frozenset[str], float] | None = None
) -> tuple[ContactGraph, dict[frozenset[str], float]]:
    """Build the contact graph AND the per-edge conductance map.

    The graph itself is built exactly as `pathway_to_contact_graph` does
    -- same costs, same medium rule -- so the baseline arm reproduces E13
    rather than merely resembling it. The conductance map is returned
    separately because it constrains WHICH perturbations are admissible;
    it is not part of the graph the criterion sees.

    Medium edges carry no conductance: the medium is not a reaction, so
    there is no enzyme that could realise a perturbation of it. They are
    absent from the map, and the admissibility test below treats absence
    as inert -- which matches E13, whose `inc` filter already excludes
    medium edges.
    """
    g = ContactGraph()
    kin: dict[frozenset[str], float] = {}
    for e in pw["edges"]:
        g.add_edge(e["source"], e["target"], e["cost"])
        key = frozenset((e["source"], e["target"]))
        kin[key] = float(e["conductance"])
    g.add_medium_edges(min(e["cost"] for e in pw["edges"]))

    if conductance_override is not None:
        kin = dict(conductance_override)
    return g, kin


# =====================================================================
# The four arms differ only in these two functions
# =====================================================================


def admissible_edges(
    inc: list[frozenset[str]],
    kin: dict[frozenset[str], float],
    gate: float | None,
) -> list[frozenset[str]]:
    """Which incident edges count as realisable perturbation sites.

    `gate is None` reproduces E13: every non-medium incident edge is
    admissible. Otherwise an edge must carry at least `gate` conductance
    -- it must be an edge the network can actually act on.
    """
    if gate is None:
        return inc
    return [e for e in inc if kin.get(e, 0.0) >= gate]


def magnitude(
    e: frozenset[str],
    kin: dict[frozenset[str], float],
    scale: bool,
    norm: float,
) -> float:
    """How far this edge is raised.

    Flat (E13) or proportional to conductance. The proportional case is
    normalised by the pathway's max conductance so that the LARGEST
    admissible perturbation in a pathway equals BASE_MAG. Without that
    normalisation the scaled arms would simply be uniformly smaller than
    the baseline and the comparison would measure magnitude, not
    structure.
    """
    if not scale:
        return BASE_MAG
    if norm <= 0.0:
        return BASE_MAG
    return BASE_MAG * (kin.get(e, 0.0) / norm)


# =====================================================================
# One arm
# =====================================================================


def run_arm(
    name: str,
    gate_quantile: float | None,
    scale: bool,
    rng: np.random.Generator | None = None,
    shuffle: bool = False,
) -> dict:
    """Measure |d1 - d2| over admissible pairs under one constraint regime.

    The body below is E13's loop with `inc` and `mag` made variable.
    Nothing else is touched: same theta, same `residue = first three other
    items`, same `corr = {v: x}`, same `cross_demand(h, h, ...)` call.
    """
    discrepancies: list[float] = []
    n_pairs_dropped = 0

    for pw in load_pathways(DATA):
        override = None
        if shuffle:
            # Permute conductances across edges, preserving the multiset.
            base_keys = [frozenset((e["source"], e["target"])) for e in pw["edges"]]
            vals = [float(e["conductance"]) for e in pw["edges"]]
            assert rng is not None, "shuffled arm needs a seeded generator"
            override = dict(zip(base_keys, rng.permutation(vals)))

        g, kin = graph_and_kinetics(pw, conductance_override=override)
        items = sorted(g.items())

        real = [v for k, v in kin.items() if len(k) == 2]
        gate = float(np.quantile(real, gate_quantile)) if gate_quantile is not None else None
        norm = max(real) if real else 0.0

        for x in items:
            inc = [e for e in g.incident(x) if g.medium not in e]
            if len(inc) < 2:
                continue
            adm = admissible_edges(inc, kin, gate)
            n_pairs_dropped += (
                len(list(itertools.combinations(inc, 2)))
                - len(list(itertools.combinations(adm, 2)))
            )
            if len(adm) < 2:
                continue

            ordered = sorted(adm, key=lambda s: tuple(sorted(s)))
            for e1, e2 in itertools.combinations(ordered, 2):
                h1 = perturb(g, {e1: magnitude(e1, kin, scale, norm)})
                h2 = perturb(g, {e2: magnitude(e2, kin, scale, norm)})
                residue = [v for v in items if v != x][:3]
                corr = {v: x for v in residue}
                d1 = cross_demand(h1, h1, residue, corr)
                d2 = cross_demand(h2, h2, residue, corr)
                discrepancies.append(abs(d1 - d2))

    arr = np.array(discrepancies) if discrepancies else np.array([0.0])
    return {
        "arm": name,
        "gate_quantile": gate_quantile,
        "magnitude_scaled": scale,
        "shuffled": shuffle,
        "n_comparisons": int(arr.size),
        "n_pairs_excluded_by_gate": int(n_pairs_dropped),
        "theta": THETA,
        "max_discrepancy": float(arr.max()),
        "mean_discrepancy": float(arr.mean()),
        "median_discrepancy": float(np.median(arr)),
        "fraction_within_theta": float(np.mean(arr <= THETA)),
        "assumption_holds_at_theta": bool(float(np.mean(arr <= THETA)) == 1.0),
        # Retained so a reader can check the magnitude confound directly
        # rather than taking the shuffled control on trust.
        "discrepancies": [float(v) for v in arr],
    }


# =====================================================================


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    RESULTS.mkdir(exist_ok=True)

    arms = [
        run_arm("unconstrained", gate_quantile=None, scale=False),
        run_arm("gated", gate_quantile=0.5, scale=False),
        run_arm("scaled", gate_quantile=None, scale=True),
        run_arm("kinetic", gate_quantile=0.5, scale=True),
    ]

    by = {a["arm"]: a for a in arms}
    base = by["unconstrained"]
    kinetic = by["kinetic"]

    # ---- permutation control -----------------------------------------
    # One shuffle is not a control at n=8. The null is the distribution
    # of the statistic over conductance assignments that preserve the
    # magnitude multiset, so the statistic must be compared against that
    # distribution rather than against one draw from it.
    null = [
        run_arm("shuffled", gate_quantile=0.5, scale=True,
                rng=rng, shuffle=True)["fraction_within_theta"]
        for _ in range(N_PERM)
    ]
    null_arr = np.array(null)
    p_value = float(np.mean(null_arr >= kinetic["fraction_within_theta"]))

    # ---- the exact-zero diagnostic ------------------------------------
    # The baseline's 0.2857 is carried entirely by comparisons whose
    # discrepancy is EXACTLY zero -- pairs that are already perfectly
    # independent. Whether the gate keeps or destroys those pairs is the
    # difference between refining the admissible set and merely
    # discarding the evidence, so it is measured rather than assumed.
    base_zeros = int(np.sum(np.array(base["discrepancies"]) <= TOL_ZERO))
    kin_zeros = int(np.sum(np.array(kinetic["discrepancies"]) <= TOL_ZERO))

    beats_baseline = (
        kinetic["fraction_within_theta"] > base["fraction_within_theta"]
    )
    significant = p_value < 0.05

    if beats_baseline and significant:
        verdict = "SUPPORTED"
        reading = (
            "Kinetic constraint improves response independence beyond what "
            "the same magnitudes achieve when detached from their edges. "
            "The E13 violation is, at least in part, an artefact of "
            "admitting perturbations no enzyme could realise."
        )
    elif beats_baseline and not significant:
        verdict = "NOT SUPPORTED (confounded)"
        reading = (
            f"The kinetic arm scores above the baseline, but permuting the "
            f"conductances across edges reproduces that gain "
            f"(p = {p_value:.3f}). The improvement is carried by "
            f"perturbation MAGNITUDE, not by kinetic structure. It is also "
            f"not a refinement of the admissible set: the baseline's score "
            f"rests on {base_zeros} comparisons with exactly zero "
            f"discrepancy, and the gate destroys all but {kin_zeros} of "
            f"them. Constraining the perturbation set does not rescue "
            f"Assumption 7.5; the criterion still needs a canonical "
            f"response."
        )
    else:
        verdict = "NOT SUPPORTED"
        reading = (
            "Kinetic constraint does not improve response independence. "
            "The violation measured by E13 is not explained by excess "
            "degrees of freedom in the perturbation set."
        )

    out = {
        "experiment": "E4 -- response independence under kinetic constraint",
        "question": (
            "Is the E13 violation of Assumption 7.5 a fundamental failure, "
            "or an artefact of unconstrained perturbation choice?"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "theta": THETA,
        "base_magnitude": BASE_MAG,
        "data": str(DATA.relative_to(DATA.parent.parent.parent)),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "arms": arms,
        "comparison": {
            "baseline_fraction": base["fraction_within_theta"],
            "kinetic_fraction": kinetic["fraction_within_theta"],
            "kinetic_beats_baseline": bool(beats_baseline),
        },
        "permutation_control": {
            "n_permutations": N_PERM,
            "statistic": "fraction_within_theta",
            "observed_kinetic": kinetic["fraction_within_theta"],
            "null_mean": float(null_arr.mean()),
            "null_median": float(np.median(null_arr)),
            "null_min": float(null_arr.min()),
            "null_max": float(null_arr.max()),
            "p_value": p_value,
            "significant_at_0.05": bool(significant),
            "null_distribution": [float(v) for v in null_arr],
        },
        "exact_zero_diagnostic": {
            "note": (
                "The baseline score is carried by comparisons with exactly "
                "zero discrepancy. If the gate removes them it is "
                "discarding evidence of independence, not refining the "
                "admissible set."
            ),
            "baseline_exact_zeros": base_zeros,
            "baseline_n": base["n_comparisons"],
            "kinetic_exact_zeros": kin_zeros,
            "kinetic_n": kinetic["n_comparisons"],
        },
        "verdict": verdict,
        "reading": reading,
        "elapsed_seconds": round(time.time() - t0, 3),
    }

    path = RESULTS / "E4_kinetic_independence.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    # ---- report ----------------------------------------------------
    print()
    print("=" * 72)
    print("E4 -- Response independence under kinetic constraint")
    print("=" * 72)
    # ASCII only: the Windows console encodes cp1252 and would die on a
    # theta here, after the measurement had already succeeded.
    print(f"{'arm':<16}{'n':>6}{'frac<=th':>10}{'mean':>11}{'median':>11}{'max':>11}")
    print("-" * 72)
    for a in arms:
        print(
            f"{a['arm']:<16}{a['n_comparisons']:>6}"
            f"{a['fraction_within_theta']:>10.4f}"
            f"{a['mean_discrepancy']:>11.6f}"
            f"{a['median_discrepancy']:>11.6f}"
            f"{a['max_discrepancy']:>11.6f}"
        )
    print("-" * 72)
    print(f"  baseline (E13 reproduction) : {base['fraction_within_theta']:.4f}"
          f"   ({base_zeros}/{base['n_comparisons']} exactly zero)")
    print(f"  kinetic                     : {kinetic['fraction_within_theta']:.4f}"
          f"   ({kin_zeros}/{kinetic['n_comparisons']} exactly zero)")
    print()
    print(f"  permutation control ({N_PERM} draws, magnitudes preserved):")
    print(f"    null mean   : {null_arr.mean():.4f}")
    print(f"    null median : {np.median(null_arr):.4f}")
    print(f"    null range  : [{null_arr.min():.4f}, {null_arr.max():.4f}]")
    print(f"    p-value     : {p_value:.4f}")
    print()
    print(f"  kinetic > baseline : {beats_baseline}")
    print(f"  p < 0.05           : {significant}")
    print(f"  VERDICT            : {verdict}")
    print()
    for line in _wrap(reading, 68):
        print(f"  {line}")
    print("=" * 72)
    print(f"written to {path}")
    return 0


def _wrap(s: str, w: int) -> list[str]:
    words, out, cur = s.split(), [], ""
    for word in words:
        if len(cur) + len(word) + 1 > w:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


if __name__ == "__main__":
    sys.exit(main())
