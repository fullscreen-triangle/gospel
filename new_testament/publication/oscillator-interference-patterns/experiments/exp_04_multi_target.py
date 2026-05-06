"""
Experiment 4: multi-target detection.

Plant N copies of a motif at random positions in a long target, each with
its own substitution rate, then run the matched filter once and pick all
peaks above a significance threshold via greedy maximum-suppression.
Report precision and recall over the planted set.
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
                    dna_channels, normalised_xcorr, find_peaks,
                    background_zscore)


def random_positions(n: int, target_len: int, query_len: int,
                     min_gap: int, rng: np.random.Generator) -> list[int]:
    out: list[int] = []
    for _ in range(2000):
        cand = int(rng.integers(0, target_len - query_len))
        if all(abs(cand - p) >= min_gap for p in out):
            out.append(cand)
            if len(out) == n:
                break
    return out


def main():
    rng = np.random.default_rng(20260425)
    cfg = {
        "target_length": 200_000,
        "query_length": 100,
        "n_planted": 8,
        "min_planted_gap": 5_000,
        "mutation_rates": [0.0, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.3],
        "trials": 25,
        "z_thresholds": [3.0, 4.0, 5.0, 6.0, 8.0],
        "min_peak_separation": 200,
        "random_seed": 20260425,
    }
    assert len(cfg["mutation_rates"]) == cfg["n_planted"]

    out = {
        "experiment": "multi_target",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "rows": [],
    }

    for z_thresh in cfg["z_thresholds"]:
        recalls, precisions, f1s, n_pred = [], [], [], []
        peak_pos_errors = []
        t0 = time.perf_counter()
        for _ in range(cfg["trials"]):
            motif = random_sequence(cfg["query_length"], DNA_ALPHABET, rng)
            target = random_sequence(cfg["target_length"], DNA_ALPHABET, rng)
            positions = random_positions(cfg["n_planted"], cfg["target_length"],
                                         cfg["query_length"], cfg["min_planted_gap"],
                                         rng)
            for pos, mr in zip(positions, cfg["mutation_rates"]):
                planted = mutate(motif, mr, DNA_ALPHABET, rng) if mr > 0 else motif
                target = plant_motif(target, planted, pos)

            Q = dna_channels(motif)
            T = dna_channels(target)
            scores = normalised_xcorr(Q, T)
            mu, sigma = background_zscore(scores)
            zs = (scores - mu) / sigma if sigma > 0 else scores
            min_score = z_thresh
            peaks = find_peaks(zs, cfg["min_peak_separation"], min_score)
            n_pred.append(len(peaks))

            matched = set()
            for ppos, _ in peaks:
                # nearest planted that hasn't been matched yet
                best_i, best_d = -1, cfg["min_peak_separation"]
                for i, gpos in enumerate(positions):
                    if i in matched:
                        continue
                    d = abs(ppos - gpos)
                    if d < best_d:
                        best_d, best_i = d, i
                if best_i >= 0:
                    matched.add(best_i)
                    peak_pos_errors.append(best_d)
            tp = len(matched)
            fn = cfg["n_planted"] - tp
            fp = len(peaks) - tp
            recall = tp / cfg["n_planted"]
            precision = tp / max(len(peaks), 1)
            f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
        dt = time.perf_counter() - t0

        row = {
            "z_threshold": z_thresh,
            "mean_recall": float(np.mean(recalls)),
            "mean_precision": float(np.mean(precisions)),
            "mean_f1": float(np.mean(f1s)),
            "mean_predictions_per_trial": float(np.mean(n_pred)),
            "median_position_error_bp": float(np.median(peak_pos_errors)) if peak_pos_errors else None,
            "p95_position_error_bp": float(np.percentile(peak_pos_errors, 95)) if peak_pos_errors else None,
            "trials": cfg["trials"],
            "wall_clock_s": dt,
        }
        out["rows"].append(row)
        print(f"z={z_thresh:>4}  recall={row['mean_recall']:.3f}  "
              f"precision={row['mean_precision']:.3f}  "
              f"F1={row['mean_f1']:.3f}  preds/trial={row['mean_predictions_per_trial']:.1f}")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "exp_04_multi_target.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved exp_04")


if __name__ == "__main__":
    main()
