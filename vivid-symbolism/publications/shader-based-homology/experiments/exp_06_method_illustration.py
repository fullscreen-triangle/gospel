"""
Produce a method-illustration figure (fig_06) with four panels:
  (a) channelised representation of a DNA sequence and its mutated homolog
  (b) DFT magnitude spectra of the two sequences overlaid; the low-freq
      window used for the embedding is highlighted
  (c) 2D PCA of embeddings for a 40x8 benchmark, coloured by family
  (d) cosine-similarity heatmap for a sorted subset of that benchmark
"""

from __future__ import annotations
import os
import sys
import json
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import (DNA_ALPHABET, generate_benchmark, spectral_embedding,
                    dna_channels, embed_batch)

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIGDIR, exist_ok=True)


def pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:2].T).astype(np.float32)


def main():
    rng = np.random.default_rng(20260423)
    L = 240
    K = 12

    # panel a: two sequences, original and 10% mutated
    seqs0 = generate_benchmark(n_families=1, family_size=2,
                               seed_length=L, mutation_rate=0.10,
                               alphabet=DNA_ALPHABET, rng=rng)[0]
    s_a, s_b = seqs0
    X_a = dna_channels(s_a)
    X_b = dna_channels(s_b)

    # panel b: DFT magnitudes
    spec_a = np.abs(np.fft.rfft(X_a, axis=1))
    spec_b = np.abs(np.fft.rfft(X_b, axis=1))

    # panels c and d: benchmark embeddings and similarity matrix
    rng2 = np.random.default_rng(20260423)
    seqs, labels = generate_benchmark(n_families=10, family_size=8,
                                      seed_length=L, mutation_rate=0.10,
                                      alphabet=DNA_ALPHABET, rng=rng2)
    E = embed_batch(seqs, n_coefficients=K, kind="dna")
    pca = pca_2d(E)

    # sort sequences by family for a block-diagonal similarity view
    order = np.argsort(labels)
    sim = E[order] @ E[order].T

    fig = plt.figure(figsize=(11.5, 8.8))

    # panel a
    ax_a = fig.add_subplot(2, 2, 1)
    colours = ["#2a6f97", "#5a99c9", "#c9423a", "#6b8f3a"]
    t = np.arange(L)
    for ch_idx, ch_name, col in zip(range(4), "ACGT", colours):
        ax_a.plot(t, X_a[ch_idx] + ch_idx * 1.3, color=col,
                  linewidth=0.8, label=f"query  {ch_name}")
        ax_a.plot(t, X_b[ch_idx] + ch_idx * 1.3, color=col,
                  linewidth=0.8, linestyle="--", alpha=0.7)
    ax_a.set_xlabel("sequence position")
    ax_a.set_yticks([0, 1.3, 2.6, 3.9])
    ax_a.set_yticklabels(["A", "C", "G", "T"])
    ax_a.set_title("(a) channelised signals (solid: seed, dashed: $10\\%$-mutated homolog)")

    # panel b
    ax_b = fig.add_subplot(2, 2, 2)
    k_axis = np.arange(spec_a.shape[1])
    for ch_idx, ch_name, col in zip(range(4), "ACGT", colours):
        ax_b.plot(k_axis, spec_a[ch_idx], color=col, linewidth=0.8,
                  label=f"{ch_name}", alpha=0.9)
        ax_b.plot(k_axis, spec_b[ch_idx], color=col, linewidth=0.8,
                  linestyle="--", alpha=0.7)
    ax_b.axvspan(1, 1 + K, color="gold", alpha=0.28, label=f"embedding window\n(K = {K})")
    ax_b.set_xlabel("Fourier bin $k$")
    ax_b.set_ylabel("$|\\hat{X}_{c,k}|$")
    ax_b.set_title("(b) DFT magnitudes; gold band = kept coefficients")
    ax_b.set_xlim(0, min(80, spec_a.shape[1] - 1))
    ax_b.legend(frameon=False, fontsize=7, loc="upper right")

    # panel c
    ax_c = fig.add_subplot(2, 2, 3)
    cmap = plt.get_cmap("tab10")
    for f in range(10):
        idx = np.where(labels == f)[0]
        ax_c.scatter(pca[idx, 0], pca[idx, 1], color=cmap(f % 10),
                     s=28, label=f"family {f}", edgecolor="black",
                     linewidth=0.3)
    ax_c.set_xlabel("PC 1")
    ax_c.set_ylabel("PC 2")
    ax_c.set_title("(c) 2D PCA of embeddings, 10 families $\\times$ 8 members")
    ax_c.legend(frameon=False, fontsize=6, loc="upper left", ncol=2)

    # panel d
    ax_d = fig.add_subplot(2, 2, 4)
    im = ax_d.imshow(sim, aspect="auto", cmap="viridis", vmin=0.6, vmax=1.0)
    # draw family separators
    n_per_fam = 8
    for f in range(1, 10):
        ax_d.axhline(f * n_per_fam - 0.5, color="white", linewidth=0.6)
        ax_d.axvline(f * n_per_fam - 0.5, color="white", linewidth=0.6)
    ax_d.set_title("(d) pairwise cosine similarity, sequences sorted by family")
    ax_d.set_xlabel("sequence index")
    ax_d.set_ylabel("sequence index")
    plt.colorbar(im, ax=ax_d, label="cosine similarity", shrink=0.75)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_06_method.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "fig_06_method.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # also dump the cluster-level metric for completeness
    out = {
        "experiment": "method_illustration",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "within_family_cos_mean": float(
            np.mean([sim[f * n_per_fam:(f + 1) * n_per_fam,
                          f * n_per_fam:(f + 1) * n_per_fam].mean()
                     for f in range(10)])),
        "between_family_cos_mean": float(
            (sim.sum() - np.trace(sim) -
             sum(sim[f * n_per_fam:(f + 1) * n_per_fam,
                     f * n_per_fam:(f + 1) * n_per_fam].sum() -
                 np.trace(sim[f * n_per_fam:(f + 1) * n_per_fam,
                              f * n_per_fam:(f + 1) * n_per_fam])
                 for f in range(10))) / (sim.size - 80**2)),
        "family_sizes": [n_per_fam] * 10,
        "n_coefficients": K,
        "seq_length": L,
    }
    with open(os.path.join(RESULTS, "exp_06_method_illustration.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved figure + results")


if __name__ == "__main__":
    main()
