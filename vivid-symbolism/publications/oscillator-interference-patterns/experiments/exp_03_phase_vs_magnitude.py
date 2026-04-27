"""
Experiment 3: phase information is what localises a hit.

The matched filter computes the cross-correlation directly in the time
domain (or via the FFT, which is the same). It is sensitive to phase: the
peak appears at the lag that makes the query and target signals coincide.

If we instead compare *only the magnitude spectra* of the query and a
sliding window of the target -- a windowed analogue of the prior paper's
spectral embedding -- we lose phase information and therefore lose the
ability to localise. We construct both scans on the same data and show:
the matched filter recovers position at sub-bp resolution; the magnitude-
spectrum sliding window cannot localise to better than the window
boundary.
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


def magnitude_spectrum(signal: np.ndarray, n_coeffs: int) -> np.ndarray:
    L = signal.shape[-1]
    spec = np.abs(np.fft.rfft(signal, axis=-1))
    if spec.shape[-1] - 1 < n_coeffs:
        pad = n_coeffs - (spec.shape[-1] - 1)
        return np.concatenate(
            [spec[..., 1:1 + n_coeffs], np.zeros(spec.shape[:-1] + (pad,))], axis=-1
        ) / max(L, 1)
    return spec[..., 1:1 + n_coeffs] / max(L, 1)


def magnitude_window_scan(query_ch: np.ndarray, target_ch: np.ndarray,
                           n_coeffs: int) -> np.ndarray:
    """
    For every starting position k in target, compute the magnitude-spectrum
    of the channel-stacked window of length Lq, then return cosine
    similarity to the magnitude-spectrum of the query. This is a sliding
    version of the embedding used in the prior shader paper.
    """
    c, Lq = query_ch.shape
    Lt = target_ch.shape[1]
    qspec = magnitude_spectrum(query_ch, n_coeffs).reshape(-1)
    qnorm = np.linalg.norm(qspec) or 1.0
    qhat = qspec / qnorm

    out = np.zeros(Lt - Lq + 1)
    for k in range(Lt - Lq + 1):
        win = target_ch[:, k:k + Lq]
        ws = magnitude_spectrum(win, n_coeffs).reshape(-1)
        wnorm = np.linalg.norm(ws) or 1.0
        out[k] = float(np.dot(qhat, ws / wnorm))
    return out


def peak_sharpness(scores: np.ndarray, peak_idx: int, drop: float = 0.5,
                   max_radius: int = 5000) -> int:
    """
    Returns the half-width-at-(drop)-of-peak in lag steps. Used to compare
    how sharp each scan's peak is.
    """
    peak = scores[peak_idx]
    if not np.isfinite(peak):
        return -1
    threshold = peak - drop * (peak - np.min(scores))
    left = peak_idx
    while left > 0 and scores[left] >= threshold:
        left -= 1
        if peak_idx - left >= max_radius:
            break
    right = peak_idx
    while right < scores.size - 1 and scores[right] >= threshold:
        right += 1
        if right - peak_idx >= max_radius:
            break
    return right - left


def main():
    rng = np.random.default_rng(20260425)
    cfg = {
        "target_length": 5_000,
        "query_length": 100,
        "n_coefficients_for_magnitude": 12,
        "mutation_rates": [0.00, 0.10, 0.20],
        "trials": 30,
        "random_seed": 20260425,
    }
    out = {
        "experiment": "phase_vs_magnitude",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "rows": [],
        "example_traces": None,
    }

    saved_example = False
    for mr in cfg["mutation_rates"]:
        xcorr_errors, mag_errors = [], []
        xcorr_widths, mag_widths = [], []
        t0 = time.perf_counter()
        for _ in range(cfg["trials"]):
            motif = random_sequence(cfg["query_length"], DNA_ALPHABET, rng)
            target = random_sequence(cfg["target_length"], DNA_ALPHABET, rng)
            pos = int(rng.integers(0, cfg["target_length"] - cfg["query_length"]))
            planted = mutate(motif, mr, DNA_ALPHABET, rng) if mr > 0 else motif
            target = plant_motif(target, planted, pos)

            Q = dna_channels(motif)
            T = dna_channels(target)
            xc = normalised_xcorr(Q, T)
            ms = magnitude_window_scan(Q, T, cfg["n_coefficients_for_magnitude"])

            x_arg = int(np.argmax(xc))
            m_arg = int(np.argmax(ms))
            xcorr_errors.append(abs(x_arg - pos))
            mag_errors.append(abs(m_arg - pos))
            xcorr_widths.append(peak_sharpness(xc, x_arg))
            mag_widths.append(peak_sharpness(ms, m_arg))

            if not saved_example and mr == 0.0:
                out["example_traces"] = {
                    "mutation_rate": mr,
                    "planted_position": pos,
                    "matched_filter": xc.tolist(),
                    "magnitude_window": ms.tolist(),
                    "matched_filter_argmax": x_arg,
                    "magnitude_window_argmax": m_arg,
                }
                saved_example = True

        dt = time.perf_counter() - t0
        row = {
            "mutation_rate": mr,
            "trials": cfg["trials"],
            "matched_filter_median_error_bp": float(np.median(xcorr_errors)),
            "matched_filter_mean_error_bp": float(np.mean(xcorr_errors)),
            "matched_filter_max_error_bp": int(np.max(xcorr_errors)),
            "magnitude_window_median_error_bp": float(np.median(mag_errors)),
            "magnitude_window_mean_error_bp": float(np.mean(mag_errors)),
            "magnitude_window_max_error_bp": int(np.max(mag_errors)),
            "matched_filter_peak_width_bp": float(np.mean(xcorr_widths)),
            "magnitude_window_peak_width_bp": float(np.mean(mag_widths)),
            "wall_clock_s": dt,
        }
        out["rows"].append(row)
        print(f"mr={mr:.2f}  xcorr_med_err={row['matched_filter_median_error_bp']:.1f} "
              f"mag_med_err={row['magnitude_window_median_error_bp']:.1f}  "
              f"xcorr_w={row['matched_filter_peak_width_bp']:.1f} "
              f"mag_w={row['magnitude_window_peak_width_bp']:.1f}  ({dt:.1f}s)")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "exp_03_phase_vs_magnitude.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved exp_03")


if __name__ == "__main__":
    main()
