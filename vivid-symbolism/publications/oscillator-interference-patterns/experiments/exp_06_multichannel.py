"""
Experiment 6: multi-channel matched filter vs single channel.

A DNA sequence has four one-hot channels (A, C, G, T). A single-channel
matched filter operates only on (e.g.) the A indicator and discards the
other three; a multi-channel filter sums the per-channel cross-correlations,
giving a coherent combiner across all four channels.

We compare these two on the same planted-motif task and report the
detection AUC as a function of mutation rate, plus the SNR boost.
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
                    dna_channels, cross_correlate_fft, normalised_xcorr,
                    background_zscore)


def single_channel_normalised(query_ch_one: np.ndarray,
                               target_ch_one: np.ndarray) -> np.ndarray:
    """Normalised single-channel cross-correlation."""
    Lq = query_ch_one.size
    Lt = target_ch_one.size
    raw = cross_correlate_fft(query_ch_one, target_ch_one)
    qe = float(np.sqrt(np.dot(query_ch_one, query_ch_one)))
    if qe <= 0:
        return np.zeros_like(raw)
    sq = target_ch_one * target_ch_one
    cum = np.zeros(Lt + 1)
    cum[1:] = np.cumsum(sq)
    we = np.sqrt(np.maximum(cum[Lq:] - cum[:Lt - Lq + 1], 0.0))
    out = np.zeros_like(raw)
    nz = we > 0
    out[nz] = raw[nz] / (qe * we[nz])
    return out


def main():
    rng = np.random.default_rng(20260425)
    cfg = {
        "target_length": 50_000,
        "query_length": 100,
        "mutation_rates": [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
        "trials_per_rate": 60,
        "h0_trials": 60,
        "single_channel_index": 0,    # use channel A
        "random_seed": 20260425,
    }
    out = {
        "experiment": "multichannel",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "rows": [],
    }

    # H0 baselines (no planted motif), one for single-channel and one for multi.
    h0_single = []
    h0_multi = []
    base_motif = random_sequence(cfg["query_length"], DNA_ALPHABET, rng)
    Q = dna_channels(base_motif)
    for _ in range(cfg["h0_trials"]):
        target = random_sequence(cfg["target_length"], DNA_ALPHABET, rng)
        T = dna_channels(target)
        s = single_channel_normalised(Q[cfg["single_channel_index"]],
                                       T[cfg["single_channel_index"]])
        m = normalised_xcorr(Q, T)
        h0_single.append(float(s.max()))
        h0_multi.append(float(m.max()))
    h0_single = np.asarray(h0_single)
    h0_multi = np.asarray(h0_multi)

    print(f"H0 single-channel peak: mean={h0_single.mean():.4f} std={h0_single.std():.4f}")
    print(f"H0 multi-channel  peak: mean={h0_multi.mean():.4f}  std={h0_multi.std():.4f}")

    for mr in cfg["mutation_rates"]:
        h1_single, h1_multi = [], []
        z_single, z_multi = [], []
        rec_single, rec_multi = [], []
        t0 = time.perf_counter()
        for _ in range(cfg["trials_per_rate"]):
            target = random_sequence(cfg["target_length"], DNA_ALPHABET, rng)
            pos = int(rng.integers(0, cfg["target_length"] - cfg["query_length"]))
            planted = mutate(base_motif, mr, DNA_ALPHABET, rng) if mr > 0 else base_motif
            target = plant_motif(target, planted, pos)
            T = dna_channels(target)

            ssingle = single_channel_normalised(Q[cfg["single_channel_index"]],
                                                  T[cfg["single_channel_index"]])
            smulti = normalised_xcorr(Q, T)

            h1_single.append(float(ssingle.max()))
            h1_multi.append(float(smulti.max()))

            mu_s, sg_s = background_zscore(ssingle, exclude=[(int(np.argmax(ssingle)) - 200,
                                                              int(np.argmax(ssingle)) + 200)])
            mu_m, sg_m = background_zscore(smulti, exclude=[(int(np.argmax(smulti)) - 200,
                                                              int(np.argmax(smulti)) + 200)])
            z_single.append((float(ssingle.max()) - mu_s) / sg_s if sg_s > 0 else float("inf"))
            z_multi.append((float(smulti.max()) - mu_m) / sg_m if sg_m > 0 else float("inf"))
            rec_single.append(int(abs(int(np.argmax(ssingle)) - pos) <= 5))
            rec_multi.append(int(abs(int(np.argmax(smulti)) - pos) <= 5))
        dt = time.perf_counter() - t0

        # AUC over H0+H1 peak heights
        def auc(h0_arr, h1_arr):
            scores = np.concatenate([h0_arr, h1_arr])
            labels = np.concatenate([np.zeros(len(h0_arr)), np.ones(len(h1_arr))])
            order = np.argsort(-scores, kind="stable")
            y = labels[order]
            tp = np.cumsum(y == 1)
            fp = np.cumsum(y == 0)
            P, N = tp[-1], fp[-1]
            if P == 0 or N == 0:
                return float("nan")
            tpr = tp / P
            fpr = np.concatenate(([0], fp / N))
            tpr = np.concatenate(([0], tpr))
            return float(np.trapezoid(tpr, fpr))

        row = {
            "mutation_rate": mr,
            "single_channel_peak_mean": float(np.mean(h1_single)),
            "multi_channel_peak_mean": float(np.mean(h1_multi)),
            "single_channel_z_mean": float(np.mean(z_single)),
            "multi_channel_z_mean": float(np.mean(z_multi)),
            "single_channel_auc": auc(h0_single, np.asarray(h1_single)),
            "multi_channel_auc": auc(h0_multi, np.asarray(h1_multi)),
            "single_channel_recovery_rate": float(np.mean(rec_single)),
            "multi_channel_recovery_rate": float(np.mean(rec_multi)),
            "trials": cfg["trials_per_rate"],
            "wall_clock_s": dt,
        }
        out["rows"].append(row)
        print(f"mr={mr:.2f}  single AUC={row['single_channel_auc']:.4f} z={row['single_channel_z_mean']:.2f}  "
              f"multi AUC={row['multi_channel_auc']:.4f} z={row['multi_channel_z_mean']:.2f}  ({dt:.1f}s)")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "exp_06_multichannel.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved exp_06")


if __name__ == "__main__":
    main()
