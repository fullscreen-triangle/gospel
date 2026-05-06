"""
Experiment 1: position recovery accuracy under controlled mutation.

For each combination of query length and substitution rate, plant a single
mutated copy of a known motif at a random position in a long random
background and measure whether the matched-filter argmax recovers the
planted position. The metric is the median absolute position error in bp,
averaged over many trials.
"""

from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (DNA_ALPHABET, random_sequence, mutate, plant_motif,
                    dna_channels, normalised_xcorr)


def trial(query_len: int, mutation_rate: int, target_len: int,
          rng: np.random.Generator):
    motif = random_sequence(query_len, DNA_ALPHABET, rng)
    target = random_sequence(target_len, DNA_ALPHABET, rng)
    pos = int(rng.integers(0, target_len - query_len))
    planted = mutate(motif, mutation_rate, DNA_ALPHABET, rng) if mutation_rate > 0 else motif
    target = plant_motif(target, planted, pos)

    Q = dna_channels(motif)
    T = dna_channels(target)
    scores = normalised_xcorr(Q, T)
    arg = int(np.argmax(scores))
    return {
        "planted_position": pos,
        "argmax_position": arg,
        "abs_error_bp": abs(arg - pos),
        "peak_score": float(scores[arg]),
        "score_at_planted": float(scores[pos]),
    }


def main():
    rng = np.random.default_rng(20260425)
    cfg = {
        "target_length": 100_000,
        "query_lengths": [50, 100, 200, 500],
        "mutation_rates": [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
        "trials_per_cell": 60,
        "random_seed": 20260425,
    }
    out = {
        "experiment": "position_recovery",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "rows": [],
    }

    for Lq in cfg["query_lengths"]:
        for mr in cfg["mutation_rates"]:
            errs, peaks, scores_at_planted, exact_hits = [], [], [], 0
            t0 = time.perf_counter()
            for _ in range(cfg["trials_per_cell"]):
                r = trial(Lq, mr, cfg["target_length"], rng)
                errs.append(r["abs_error_bp"])
                peaks.append(r["peak_score"])
                scores_at_planted.append(r["score_at_planted"])
                if r["abs_error_bp"] == 0:
                    exact_hits += 1
            dt = time.perf_counter() - t0

            errs = np.asarray(errs)
            row = {
                "query_length": Lq,
                "mutation_rate": mr,
                "n_trials": cfg["trials_per_cell"],
                "exact_hits": exact_hits,
                "exact_hit_rate": exact_hits / cfg["trials_per_cell"],
                "median_abs_error_bp": float(np.median(errs)),
                "mean_abs_error_bp": float(np.mean(errs)),
                "p95_abs_error_bp": float(np.percentile(errs, 95)),
                "max_abs_error_bp": int(errs.max()),
                "mean_peak_score": float(np.mean(peaks)),
                "mean_score_at_planted": float(np.mean(scores_at_planted)),
                "wall_clock_s": dt,
            }
            out["rows"].append(row)
            print(f"Lq={Lq:>4}  mr={mr:.2f}  exact={exact_hits}/{cfg['trials_per_cell']}  "
                  f"median_err={row['median_abs_error_bp']:.1f}  "
                  f"peak={row['mean_peak_score']:.3f}  ({dt:.1f}s)")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_01_position_recovery.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main()
