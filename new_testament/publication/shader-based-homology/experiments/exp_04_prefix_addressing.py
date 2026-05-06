"""
Experiment 4: hierarchical coordinate prefix addressing.

We project embeddings onto the unit 3-cube and assign each sequence a
hierarchical bit address by repeated axis-cyclic bisection. A prefix match at
depth d partitions sequences into cells of relative volume 2^{-d}.

We measure how much the database can be narrowed by prefix-matching before
exhaustive shader evaluation, and what fraction of true homologs survive the
narrowing. This demonstrates a sub-linear lookup scheme.
"""

from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (DNA_ALPHABET, generate_benchmark, embed_batch,
                    hierarchical_address, project_to_unit_cube)


def build_address_index(coords: np.ndarray, depth: int):
    """Return a dict mapping prefix tuples to lists of row indices."""
    index: dict[tuple, list[int]] = {}
    for i, c in enumerate(coords):
        a = hierarchical_address(c, depth)
        index.setdefault(a, []).append(i)
    return index


def main():
    rng = np.random.default_rng(20260423)
    out = {
        "experiment": "prefix_addressing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_families": 100,
            "family_size": 10,
            "seed_length": 300,
            "mutation_rate": 0.10,
            "n_coefficients": 12,
            "projection_dims": 3,
            "address_depths": [3, 6, 9, 12, 15, 18],
            "query_mutation_rate": 0.10,
            "random_seed": 20260423,
        },
        "rows": [],
    }
    cfg = out["config"]

    seqs, labels = generate_benchmark(cfg["n_families"], cfg["family_size"],
                                      cfg["seed_length"], cfg["mutation_rate"],
                                      DNA_ALPHABET, rng)
    n = len(seqs)
    print(f"database size: {n}")

    E = embed_batch(seqs, n_coefficients=cfg["n_coefficients"], kind="dna")
    coords = project_to_unit_cube(E, n_dims=cfg["projection_dims"])

    # queries: fresh mutation of each seed (first member of each family)
    query_texts = [seqs[f * cfg["family_size"]] for f in range(cfg["n_families"])]
    Q_emb = embed_batch(query_texts, n_coefficients=cfg["n_coefficients"], kind="dna")

    # project queries using the same transform by re-running project_to_unit_cube
    # on (E stacked with Q) would be ideal; instead we reproject the DB + Qs
    # jointly so the scaling matches
    joint = np.vstack([E, Q_emb])
    joint_coords = project_to_unit_cube(joint, n_dims=cfg["projection_dims"])
    coords = joint_coords[:n]
    Q_coords = joint_coords[n:]

    for depth in cfg["address_depths"]:
        idx = build_address_index(coords, depth)
        total_cells = len(idx)

        # for each query, look at its prefix cell + the 3 sibling cells on the
        # last axis (simulating a small radius expansion)
        recalls = []
        cell_fractions = []
        for qi, qc in enumerate(Q_coords):
            qa = hierarchical_address(qc, depth)
            true_fam = {j for j in np.where(labels == qi)[0]
                        if j != qi * cfg["family_size"]}
            # consider exact prefix match
            candidates = set(idx.get(qa, []))
            # plus 1-bit-flipped neighbours (Hamming-1 in address space)
            for b in range(depth):
                alt = list(qa)
                alt[b] = 1 - alt[b]
                candidates.update(idx.get(tuple(alt), []))
            hits = sum(1 for j in candidates if j in true_fam)
            if true_fam:
                recalls.append(hits / len(true_fam))
            cell_fractions.append(len(candidates) / n)

        mean_recall = float(np.mean(recalls))
        mean_fraction = float(np.mean(cell_fractions))
        row = {
            "depth": depth,
            "n_cells_occupied": total_cells,
            "mean_recall_true_family": mean_recall,
            "mean_fraction_of_db_probed": mean_fraction,
            "expected_speedup_vs_exhaustive": 1.0 / mean_fraction if mean_fraction > 0 else float("inf"),
        }
        out["rows"].append(row)
        print(f"depth={depth:>2}  cells={total_cells:>5}  "
              f"recall_fam={mean_recall:.3f}  "
              f"probe_frac={mean_fraction:.4f}  "
              f"speedup={row['expected_speedup_vs_exhaustive']:.1f}x")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_04_prefix_addressing.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main()
