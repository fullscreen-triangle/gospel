"""
Experiment 1: embedding separates homologs from non-homologs.

We generate N synthetic families of sequences sharing a common seed (homologs
at a controlled substitution rate) and measure whether the spectral embedding's
cosine-similarity distribution for within-family pairs is separable from the
between-family distribution. We sweep the substitution rate to characterise
where the method starts to degrade.
"""

from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (DNA_ALPHABET, PROTEIN_ALPHABET, generate_benchmark,
                    embed_batch)


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
    # trapezoidal area, prepend (0, 0)
    fpr = np.concatenate(([0.0], fpr))
    tpr = np.concatenate(([0.0], tpr))
    return float(np.trapezoid(tpr, fpr))


def evaluate(n_families: int, family_size: int, seed_length: int,
             mutation_rate: float, kind: str, n_coeff: int, seed: int):
    rng = np.random.default_rng(seed)
    alphabet = DNA_ALPHABET if kind == "dna" else PROTEIN_ALPHABET
    seqs, labels = generate_benchmark(n_families, family_size, seed_length,
                                      mutation_rate, alphabet, rng)
    t0 = time.perf_counter()
    E = embed_batch(seqs, n_coefficients=n_coeff, kind=kind)
    t_embed = time.perf_counter() - t0

    sims = E @ E.T
    n = len(seqs)
    iu = np.triu_indices(n, k=1)
    pair_sim = sims[iu]
    pair_same = (labels[iu[0]] == labels[iu[1]]).astype(np.int32)

    within_mean = float(pair_sim[pair_same == 1].mean())
    within_std = float(pair_sim[pair_same == 1].std())
    between_mean = float(pair_sim[pair_same == 0].mean())
    between_std = float(pair_sim[pair_same == 0].std())
    separation = within_mean - between_mean
    # Cohen's d (pooled std)
    pooled = np.sqrt(0.5 * (within_std**2 + between_std**2))
    cohen_d = separation / pooled if pooled > 0 else float("inf")

    auc = roc_auc(pair_sim, pair_same)

    return {
        "within_mean": within_mean,
        "within_std": within_std,
        "between_mean": between_mean,
        "between_std": between_std,
        "separation": separation,
        "cohen_d": float(cohen_d),
        "auc": auc,
        "n_within_pairs": int(pair_same.sum()),
        "n_between_pairs": int(n * (n - 1) // 2 - pair_same.sum()),
        "embed_time_s": t_embed,
        "embed_dim": int(E.shape[1]),
    }


def main():
    out = {
        "experiment": "embedding_separation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_families": 40,
            "family_size": 8,
            "seed_length_dna": 300,
            "seed_length_protein": 200,
            "n_coefficients": 12,
            "mutation_rates": [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
            "random_seed": 20260423,
        },
        "dna": [],
        "protein": [],
    }
    cfg = out["config"]

    for mr in cfg["mutation_rates"]:
        row = evaluate(cfg["n_families"], cfg["family_size"],
                       cfg["seed_length_dna"], mr, "dna",
                       cfg["n_coefficients"], cfg["random_seed"])
        row["mutation_rate"] = mr
        out["dna"].append(row)
        row2 = evaluate(cfg["n_families"], cfg["family_size"],
                        cfg["seed_length_protein"], mr, "protein",
                        cfg["n_coefficients"], cfg["random_seed"])
        row2["mutation_rate"] = mr
        out["protein"].append(row2)
        print(f"mr={mr:.2f}  dna auc={row['auc']:.4f}  d={row['cohen_d']:.2f}   "
              f"protein auc={row2['auc']:.4f}  d={row2['cohen_d']:.2f}")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_01_embedding_separation.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main()
