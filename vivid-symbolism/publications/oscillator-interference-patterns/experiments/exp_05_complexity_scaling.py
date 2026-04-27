"""
Experiment 5: complexity scaling.

Compare wall-clock time of three implementations of the same matched filter
output (over Lt - Lq + 1 lags):

  - naive sliding inner product   O(Lq * (Lt - Lq + 1))
  - FFT-based linear cross-correlation  O((Lq + Lt) log(Lq + Lt))
  - O(Lq * Lt) is the relevant comparison since FFT cost is invariant
    to query size

We sweep Lt from 10^3 to 10^6 with Lq fixed; the naive method is run only
up to a budget so the curves remain finite.
"""

from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (DNA_ALPHABET, random_sequence, dna_channels,
                    cross_correlate_fft, cross_correlate_naive,
                    cross_correlate_multichannel)


def main():
    rng = np.random.default_rng(20260425)
    cfg = {
        "query_length": 100,
        "target_lengths": [1_000, 3_162, 10_000, 31_622, 100_000, 316_227, 1_000_000],
        "trials_per_size": 3,
        "naive_max_target": 31_622,
        "random_seed": 20260425,
    }

    out = {
        "experiment": "complexity_scaling",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "rows": [],
    }

    motif = random_sequence(cfg["query_length"], DNA_ALPHABET, rng)
    Q = dna_channels(motif)

    for Lt in cfg["target_lengths"]:
        target = random_sequence(Lt, DNA_ALPHABET, rng)
        T = dna_channels(target)

        # FFT (multichannel)
        fft_times = []
        for _ in range(cfg["trials_per_size"]):
            t0 = time.perf_counter()
            _ = cross_correlate_multichannel(Q, T)
            fft_times.append(time.perf_counter() - t0)
        fft_mean = float(np.mean(fft_times))

        # Naive (single channel A only) - extrapolate to four channels by 4x
        naive_extrapolated = None
        naive_measured = None
        if Lt <= cfg["naive_max_target"]:
            t0 = time.perf_counter()
            _ = cross_correlate_naive(Q[0], T[0])  # one channel
            single_naive = time.perf_counter() - t0
            naive_measured = single_naive * 4.0  # multiply by 4 channels
            naive_extrapolated = naive_measured
        else:
            # Extrapolate from a probe at this size by running a smaller naive
            probe_Lq = cfg["query_length"]
            probe_Lt = 5_000
            sub_target = target[:probe_Lt]
            sub_T = dna_channels(sub_target)
            t0 = time.perf_counter()
            _ = cross_correlate_naive(Q[0], sub_T[0])
            probe_t = time.perf_counter() - t0
            scale = (Lt - probe_Lq + 1) / (probe_Lt - probe_Lq + 1)
            naive_extrapolated = probe_t * scale * 4.0

        row = {
            "target_length": int(Lt),
            "query_length": cfg["query_length"],
            "fft_seconds_mean": fft_mean,
            "fft_seconds_std": float(np.std(fft_times)),
            "naive_seconds_measured": naive_measured,
            "naive_seconds_extrapolated": naive_extrapolated,
            "speedup_fft_over_naive": (naive_extrapolated or 0.0) / fft_mean if fft_mean > 0 else float("inf"),
        }
        out["rows"].append(row)
        print(f"Lt={Lt:>8}  fft={fft_mean*1e3:>9.3f} ms  "
              f"naive={'-' if naive_extrapolated is None else f'{naive_extrapolated*1e3:>10.1f} ms'}  "
              f"speedup={row['speedup_fft_over_naive']:>8.1f}x"
              f"{' (extrap.)' if naive_measured is None else ''}")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "exp_05_complexity_scaling.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved exp_05")


if __name__ == "__main__":
    main()
