"""
Experiment 2: detection sensitivity vs. mutation rate.

We treat each trial as a binary hypothesis test:
  H0: no motif planted (pure random target)
  H1: motif planted at some unknown position

Under H1 the score is the global maximum of the matched-filter scan with a
motif present. Under H0 the score is the global maximum on a target that
contains no planted motif (random max from an Lt - Lq + 1 noise vector).

We sweep substitution rates and report:
  - peak z-score = (peak - background_mean) / background_std
  - mean signal-to-noise contrast peak_H1 / peak_H0
  - ROC AUC over many H0 / H1 trials
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
                    dna_channels, normalised_xcorr, background_zscore)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    P = tp[-1]
    N = fp[-1]
    if P == 0 or N == 0:
        return float("nan")
    tpr = tp / P
    fpr = fp / N
    fpr = np.concatenate(([0.0], fpr))
    tpr = np.concatenate(([0.0], tpr))
    return float(np.trapezoid(tpr, fpr))


def main():
    rng = np.random.default_rng(20260425)
    cfg = {
        "target_length": 50_000,
        "query_length": 100,
        "mutation_rates": [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        "trials_per_rate": 80,
        "h0_trials": 80,
        "random_seed": 20260425,
        "exclude_window_for_background_bp": 200,
    }
    out = {
        "experiment": "detection_sensitivity",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "rows": [],
    }

    # Pre-compute a pool of H0 (no planted motif) peak scores -- one per trial.
    print("computing H0 baseline...")
    motif_for_query = random_sequence(cfg["query_length"], DNA_ALPHABET, rng)
    Q = dna_channels(motif_for_query)
    h0_peaks = []
    for _ in range(cfg["h0_trials"]):
        target = random_sequence(cfg["target_length"], DNA_ALPHABET, rng)
        T = dna_channels(target)
        scores = normalised_xcorr(Q, T)
        h0_peaks.append(float(scores.max()))
    h0_peaks = np.asarray(h0_peaks)
    print(f"  H0 peak: mean={h0_peaks.mean():.4f} std={h0_peaks.std():.4f} "
          f"max={h0_peaks.max():.4f}")

    for mr in cfg["mutation_rates"]:
        h1_peaks = []
        z_scores = []
        recovered_position = []
        t0 = time.perf_counter()
        for _ in range(cfg["trials_per_rate"]):
            motif = motif_for_query
            target = random_sequence(cfg["target_length"], DNA_ALPHABET, rng)
            pos = int(rng.integers(0, cfg["target_length"] - cfg["query_length"]))
            planted = mutate(motif, mr, DNA_ALPHABET, rng) if mr > 0 else motif
            target = plant_motif(target, planted, pos)
            T = dna_channels(target)
            scores = normalised_xcorr(Q, T)
            peak = float(scores.max())
            argmax = int(np.argmax(scores))
            half = cfg["exclude_window_for_background_bp"]
            mu, sigma = background_zscore(
                scores, exclude=[(argmax - half, argmax + half)]
            )
            z = (peak - mu) / sigma if sigma > 0 else float("inf")
            h1_peaks.append(peak)
            z_scores.append(z)
            recovered_position.append(int(abs(argmax - pos) <= 5))
        h1_peaks = np.asarray(h1_peaks)
        z_scores = np.asarray(z_scores)
        dt = time.perf_counter() - t0

        labels = np.concatenate([np.zeros(len(h0_peaks)), np.ones(len(h1_peaks))])
        all_scores = np.concatenate([h0_peaks, h1_peaks])
        auc = roc_auc(all_scores, labels)

        row = {
            "mutation_rate": mr,
            "n_h1_trials": cfg["trials_per_rate"],
            "n_h0_trials": cfg["h0_trials"],
            "h1_peak_mean": float(h1_peaks.mean()),
            "h1_peak_std": float(h1_peaks.std()),
            "h0_peak_mean": float(h0_peaks.mean()),
            "h0_peak_std": float(h0_peaks.std()),
            "h1_z_mean": float(z_scores.mean()),
            "h1_z_std": float(z_scores.std()),
            "h1_position_recovered_within_5bp": float(np.mean(recovered_position)),
            "roc_auc": auc,
            "wall_clock_s": dt,
        }
        out["rows"].append(row)
        print(f"mr={mr:.2f}  H1_peak={row['h1_peak_mean']:.3f}  "
              f"H0_peak={row['h0_peak_mean']:.3f}  AUC={auc:.4f}  "
              f"z={row['h1_z_mean']:.2f}  rec={row['h1_position_recovered_within_5bp']:.2f}  "
              f"({dt:.1f}s)")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "exp_02_detection_sensitivity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved exp_02")


if __name__ == "__main__":
    main()
