"""Generate paper figures from the JSON result files."""

from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def _load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def fig_01():
    d = _load("exp_01_position_recovery.json")
    rows = d["rows"]
    Lqs = sorted({r["query_length"] for r in rows})
    mrs = sorted({r["mutation_rate"] for r in rows})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    cm = plt.get_cmap("viridis")
    for i, Lq in enumerate(Lqs):
        sub = [r for r in rows if r["query_length"] == Lq]
        sub.sort(key=lambda r: r["mutation_rate"])
        col = cm(i / max(1, len(Lqs) - 1))
        ax1.plot([r["mutation_rate"] for r in sub],
                 [r["exact_hit_rate"] for r in sub],
                 "o-", color=col, label=f"Lq = {Lq}")
        ax2.plot([r["mutation_rate"] for r in sub],
                 [r["mean_peak_score"] for r in sub],
                 "o-", color=col, label=f"Lq = {Lq}")
    ax1.set_xlabel("substitution rate"); ax1.set_ylabel("exact-position hit rate")
    ax1.set_title("(a) sub-bp position recovery"); ax1.set_ylim(-0.02, 1.05)
    ax1.legend(frameon=False, fontsize=8)
    ax2.set_xlabel("substitution rate"); ax2.set_ylabel("mean normalised peak score")
    ax2.set_title("(b) peak height vs. divergence")
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_01_position_recovery.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "fig_01_position_recovery.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_02():
    d = _load("exp_02_detection_sensitivity.json")
    rows = d["rows"]; rows.sort(key=lambda r: r["mutation_rate"])
    mrs = [r["mutation_rate"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax1.plot(mrs, [r["h1_peak_mean"] for r in rows], "o-", color="#2a6f97", label="signal (H1) peak")
    ax1.plot(mrs, [r["h0_peak_mean"] for r in rows], "s--", color="#c9423a", label="noise (H0) peak")
    ax1.fill_between(mrs,
                     [r["h0_peak_mean"] - r["h0_peak_std"] for r in rows],
                     [r["h0_peak_mean"] + r["h0_peak_std"] for r in rows],
                     color="#c9423a", alpha=0.2)
    ax1.set_xlabel("substitution rate"); ax1.set_ylabel("global-max normalised xcorr")
    ax1.set_title("(a) signal vs. noise peak heights"); ax1.legend(frameon=False)

    ax2.plot(mrs, [r["h1_z_mean"] for r in rows], "o-", color="#2a6f97")
    ax2.set_xlabel("substitution rate"); ax2.set_ylabel("peak z-score")
    ax2.set_title("(b) detection z-score")
    ax2.axhline(5, color="grey", linestyle=":", linewidth=1)
    ax2.text(0.42, 5.6, "z = 5", color="grey", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_02_detection.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "fig_02_detection.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_03():
    d = _load("exp_03_phase_vs_magnitude.json")
    rows = d["rows"]
    mrs = [r["mutation_rate"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax1.plot(mrs, [r["matched_filter_peak_width_bp"] for r in rows], "o-",
             color="#2a6f97", label="matched filter")
    ax1.plot(mrs, [r["magnitude_window_peak_width_bp"] for r in rows], "s-",
             color="#c9423a", label="magnitude window cosine")
    ax1.set_xlabel("substitution rate"); ax1.set_ylabel("peak half-width [bp]")
    ax1.set_title("(a) peak sharpness"); ax1.set_yscale("log"); ax1.legend(frameon=False)

    # Plot example traces if present
    ex = d.get("example_traces")
    if ex:
        x = np.arange(len(ex["matched_filter"]))
        ax2.plot(x, ex["matched_filter"], color="#2a6f97", linewidth=0.7,
                 label="matched filter (norm. xcorr)")
        ax2.plot(x, ex["magnitude_window"], color="#c9423a", linewidth=0.7,
                 alpha=0.85, label="magnitude window cosine")
        ax2.axvline(ex["planted_position"], color="black", linestyle=":",
                    linewidth=1, label="planted position")
        ax2.set_xlabel("lag [bp]"); ax2.set_ylabel("score")
        ax2.set_title("(b) example traces"); ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_03_phase_vs_magnitude.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "fig_03_phase_vs_magnitude.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_04():
    d = _load("exp_04_multi_target.json")
    rows = d["rows"]
    zs = [r["z_threshold"] for r in rows]
    rec = [r["mean_recall"] for r in rows]
    pre = [r["mean_precision"] for r in rows]
    f1 = [r["mean_f1"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax1.plot(zs, rec, "o-", color="#2a6f97", label="recall")
    ax1.plot(zs, pre, "s-", color="#c9423a", label="precision")
    ax1.plot(zs, f1, "^-", color="#6b8f3a", label="F1")
    ax1.set_xlabel("z-score threshold"); ax1.set_ylabel("score")
    ax1.set_title("(a) precision/recall vs. threshold")
    ax1.set_ylim(-0.02, 1.05); ax1.legend(frameon=False)

    err = [(r["median_position_error_bp"] or 0) for r in rows]
    ax2.bar(range(len(zs)), err, color="#2a6f97")
    ax2.set_xticks(range(len(zs)))
    ax2.set_xticklabels([f"{z}" for z in zs])
    ax2.set_xlabel("z-score threshold"); ax2.set_ylabel("median position error [bp]")
    ax2.set_title("(b) localisation accuracy of recovered hits")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_04_multi_target.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "fig_04_multi_target.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_05():
    d = _load("exp_05_complexity_scaling.json")
    rows = d["rows"]
    Lt = np.array([r["target_length"] for r in rows])
    fft = np.array([r["fft_seconds_mean"] for r in rows])
    naive = np.array([r["naive_seconds_extrapolated"] for r in rows])
    speed = np.array([r["speedup_fft_over_naive"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax1.loglog(Lt, fft * 1e3, "o-", color="#2a6f97", label="FFT-based")
    ax1.loglog(Lt, naive * 1e3, "s-", color="#c9423a", label="naive sliding")
    ax1.set_xlabel("target length $L_t$ [bp]"); ax1.set_ylabel("time [ms]")
    ax1.set_title("(a) wall-clock per scan"); ax1.legend(frameon=False)

    ax2.loglog(Lt, speed, "o-", color="#6b8f3a")
    ax2.set_xlabel("target length $L_t$ [bp]"); ax2.set_ylabel("speedup factor")
    ax2.set_title("(b) FFT vs. naive speedup")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_05_scaling.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "fig_05_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_06():
    d = _load("exp_06_multichannel.json")
    rows = d["rows"]
    mrs = [r["mutation_rate"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax1.plot(mrs, [r["multi_channel_z_mean"] for r in rows], "o-", color="#2a6f97",
             label="4-channel")
    ax1.plot(mrs, [r["single_channel_z_mean"] for r in rows], "s-", color="#c9423a",
             label="1-channel (A)")
    ax1.set_xlabel("substitution rate"); ax1.set_ylabel("peak z-score")
    ax1.set_title("(a) detection z-score"); ax1.legend(frameon=False)

    ax2.plot(mrs, [r["multi_channel_auc"] for r in rows], "o-", color="#2a6f97",
             label="4-channel")
    ax2.plot(mrs, [r["single_channel_auc"] for r in rows], "s-", color="#c9423a",
             label="1-channel (A)")
    ax2.set_xlabel("substitution rate"); ax2.set_ylabel("ROC-AUC")
    ax2.set_title("(b) discrimination of signal vs. random"); ax2.set_ylim(0.45, 1.02)
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_06_multichannel.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "fig_06_multichannel.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    for name, fn in [("fig_01", fig_01), ("fig_02", fig_02), ("fig_03", fig_03),
                     ("fig_04", fig_04), ("fig_05", fig_05), ("fig_06", fig_06)]:
        try:
            fn()
            print(f"ok  {name}")
        except FileNotFoundError as e:
            print(f"skip {name}: {e.filename}")


if __name__ == "__main__":
    main()
