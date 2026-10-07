"""E2 -- identifier-blind queries over genomic provenance graphs.

Genericity: ans(Q, pi G) == pi ans(Q, G) for eleven query classes of the
fragment, against three controls that each lift one exclusion (lexical
inspection of an identifier, ordering with truncation, a constant naming a
call). Orbits: on graphs with a planted symmetry, the description query returns
b from a exactly when VF2 finds an automorphism mapping a to b. Colour
refinement is checked for soundness.
"""

from __future__ import annotations

import time

import numpy as np

from common import rng_for, write_result
from gsv import rdfgraphs as R


def genericity(rng, n_graphs=40, renamings=6):
    classes = list(R.FRAGMENT) + list(R.CONTROLS)
    viol = {c: 0 for c in classes}
    tests = {c: 0 for c in classes}
    by_size = {c: {} for c in R.CONTROLS}
    sizes = []
    for _ in range(n_graphs):
        g = R.random_graph(rng)
        nind = len(R.individuals(g))
        sizes.append(nind)
        for _ in range(renamings):
            pi = R.random_renaming(g, rng)
            for c in classes:
                q = R.FRAGMENT.get(c) or R.CONTROLS.get(c) or R.constant_query(g)
                ok = R.generic(g, q, pi)
                tests[c] += 1
                viol[c] += int(not ok)
                if c in R.CONTROLS:
                    d = by_size[c].setdefault(nind, [0, 0])
                    d[0] += int(not ok); d[1] += 1
    return {"classes": classes, "fragment": list(R.FRAGMENT), "controls": list(R.CONTROLS),
            "violations": viol, "tests": tests,
            "control_by_size": {c: {str(k): v for k, v in sorted(d.items())}
                                for c, d in by_size.items()},
            "individuals_min": int(min(sizes)), "individuals_max": int(max(sizes))}


def orbits(rng, n_graphs=40):
    rows = []
    times = []
    for i in range(n_graphs):
        g, names, ren = R.symmetric_graph(rng, break_symmetry=bool(i % 2))
        inds = R.individuals(g)
        others = [x for x in inds if "z_" in str(x)]
        picks = []
        for a in names[:3]:
            picks.append((a, ren[a]))
        picks.append((names[0], others[0]))
        col = R.colour_refinement(g)
        cache = {}
        for a, b in picks:
            orbit = R.same_orbit(g, a, b)
            if a not in cache:
                t = time.time()
                cache[a] = {row[0] for row in R.answer(g, R.description_query(g, a))}
                times.append(time.time() - t)
            returned = b in cache[a]
            rows.append({"orbit": orbit, "description_returns_b": returned,
                         "colour_same": col[a] == col[b],
                         "broken": bool(i % 2), "individuals": len(inds)})
    tab = {"orbit_and_returned": 0, "orbit_not_returned": 0,
           "no_orbit_returned": 0, "no_orbit_not_returned": 0}
    for r in rows:
        if r["orbit"] and r["description_returns_b"]:
            tab["orbit_and_returned"] += 1
        elif r["orbit"]:
            tab["orbit_not_returned"] += 1
        elif r["description_returns_b"]:
            tab["no_orbit_returned"] += 1
        else:
            tab["no_orbit_not_returned"] += 1
    cr_sound_viol = sum(1 for r in rows if r["orbit"] and not r["colour_same"])
    cr_separated = sum(1 for r in rows if not r["orbit"] and not r["colour_same"])
    n_diff = sum(1 for r in rows if not r["orbit"])
    return {"pairs": len(rows), "table": tab, "colour_soundness_violations": cr_sound_viol,
            "colour_separated_of_different_orbit": [cr_separated, n_diff],
            "description_query_median_s": float(np.median(times)),
            "description_query_max_s": float(np.max(times))}


def main():
    t0 = time.time()
    rng = rng_for(2)
    write_result("E2_generic", {"genericity": genericity(rng), "orbits": orbits(rng)}, t0)


if __name__ == "__main__":
    main()
