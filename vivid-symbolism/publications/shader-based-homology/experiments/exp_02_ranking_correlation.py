"""
Experiment 2: ranking correlation with Smith-Waterman and k-mer Jaccard.

For a small database where exhaustive Smith-Waterman is feasible, we check
that the spectral-embedding ranking of database entries by cosine similarity
to a query agrees with the ranking induced by true local-alignment score and
by k-mer Jaccard similarity.

We report Spearman rank correlation and recall@K for several K.
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
                    embed_batch, kmer_jaccard, smith_waterman,
                    shader_kernel_distances)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x):
        order = np.argsort(x, kind="stable")
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(x))
        # average ties
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        if counts.max() > 1:
            # simple tied-rank average
            r_avg = np.empty_like(r)
            for v in range(len(counts)):
                idx = np.where(inv == v)[0]
                r_avg[idx] = r[idx].mean()
            return r_avg
        return r
    ra = rank(a)
    rb = rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def recall_at_k(pred_rank_order: np.ndarray, true_top_set: set[int], k: int) -> float:
    top_pred = set(pred_rank_order[:k].tolist())
    if not true_top_set:
        return float("nan")
    return len(top_pred & true_top_set) / len(true_top_set)


def evaluate_kind(kind: str, seed_length: int, n_families: int, family_size: int,
                  mutation_rate: float, n_coeff: int, kmer_k: int,
                  top_k_values: list[int], n_queries: int, rng: np.random.Generator):
    alphabet = DNA_ALPHABET if kind == "dna" else PROTEIN_ALPHABET
    seqs, labels = generate_benchmark(n_families, family_size, seed_length,
                                      mutation_rate, alphabet, rng)
    n = len(seqs)

    t0 = time.perf_counter()
    E = embed_batch(seqs, n_coefficients=n_coeff, kind=kind)
    t_embed = time.perf_counter() - t0

    # Build query set: first member of each family (homolog target is the rest
    # of the family)
    query_idx = [f * family_size for f in range(min(n_queries, n_families))]

    sw_spearman = []
    jaccard_spearman = []
    recalls_sw = {k: [] for k in top_k_values}
    recalls_jac = {k: [] for k in top_k_values}

    t_sw_total = 0.0
    t_jac_total = 0.0
    t_emb_total = 0.0

    for q in query_idx:
        # embedding distances
        t0 = time.perf_counter()
        d_emb = shader_kernel_distances(E[q], E)
        t_emb_total += time.perf_counter() - t0
        d_emb[q] = np.inf  # exclude self
        order_emb = np.argsort(d_emb)

        # Smith-Waterman (dense; small n only)
        t0 = time.perf_counter()
        sw = np.zeros(n, dtype=np.int64)
        for j in range(n):
            if j == q:
                continue
            sw[j] = smith_waterman(seqs[q], seqs[j])
        t_sw_total += time.perf_counter() - t0
        order_sw = np.argsort(-sw)  # descending (higher score = more similar)

        # k-mer Jaccard
        t0 = time.perf_counter()
        jac = np.zeros(n, dtype=np.float64)
        for j in range(n):
            if j == q:
                continue
            jac[j] = kmer_jaccard(seqs[q], seqs[j], kmer_k)
        t_jac_total += time.perf_counter() - t0
        order_jac = np.argsort(-jac)

        # Spearman correlations (exclude self)
        mask = np.arange(n) != q
        sw_spearman.append(spearman(-d_emb[mask], sw[mask].astype(float)))
        jaccard_spearman.append(spearman(-d_emb[mask], jac[mask]))

        # recall@k relative to true family membership
        true_fam = set(idx for idx in np.where(labels == labels[q])[0] if idx != q)
        for k in top_k_values:
            recalls_sw[k].append(recall_at_k(order_emb[order_emb != q][:k], true_fam, k))
        for k in top_k_values:
            recalls_jac[k].append(recall_at_k(order_jac[order_jac != q][:k], true_fam, k))

    return {
        "kind": kind,
        "n_sequences": n,
        "n_queries": len(query_idx),
        "mutation_rate": mutation_rate,
        "seed_length": seed_length,
        "n_coefficients": n_coeff,
        "kmer_k": kmer_k,
        "embed_time_s_total": t_embed,
        "per_query_embed_s_mean": t_emb_total / len(query_idx),
        "per_query_sw_s_mean": t_sw_total / len(query_idx),
        "per_query_jaccard_s_mean": t_jac_total / len(query_idx),
        "spearman_vs_sw_mean": float(np.nanmean(sw_spearman)),
        "spearman_vs_sw_std": float(np.nanstd(sw_spearman)),
        "spearman_vs_jaccard_mean": float(np.nanmean(jaccard_spearman)),
        "spearman_vs_jaccard_std": float(np.nanstd(jaccard_spearman)),
        "recall_embedding_vs_true_family": {
            str(k): float(np.mean(recalls_sw[k])) for k in top_k_values
        },
        "recall_jaccard_vs_true_family": {
            str(k): float(np.mean(recalls_jac[k])) for k in top_k_values
        },
    }


def main():
    rng = np.random.default_rng(20260423)
    out = {
        "experiment": "ranking_correlation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_families": 12,
            "family_size": 6,
            "seed_length_dna": 200,
            "seed_length_protein": 140,
            "mutation_rates": [0.05, 0.10, 0.20],
            "n_coefficients": 12,
            "kmer_k_dna": 6,
            "kmer_k_protein": 3,
            "top_k_values": [1, 5, 10],
            "n_queries": 12,
            "random_seed": 20260423,
        },
        "runs": [],
    }
    cfg = out["config"]

    for mr in cfg["mutation_rates"]:
        dna = evaluate_kind("dna", cfg["seed_length_dna"], cfg["n_families"],
                            cfg["family_size"], mr, cfg["n_coefficients"],
                            cfg["kmer_k_dna"], cfg["top_k_values"],
                            cfg["n_queries"], rng)
        out["runs"].append(dna)
        print(f"dna  mr={mr:.2f}  rho_SW={dna['spearman_vs_sw_mean']:.3f}  "
              f"rho_J={dna['spearman_vs_jaccard_mean']:.3f}  "
              f"R@5={dna['recall_embedding_vs_true_family']['5']:.2f}  "
              f"(J R@5={dna['recall_jaccard_vs_true_family']['5']:.2f})")

        prot = evaluate_kind("protein", cfg["seed_length_protein"], cfg["n_families"],
                             cfg["family_size"], mr, cfg["n_coefficients"],
                             cfg["kmer_k_protein"], cfg["top_k_values"],
                             cfg["n_queries"], rng)
        out["runs"].append(prot)
        print(f"prot mr={mr:.2f}  rho_SW={prot['spearman_vs_sw_mean']:.3f}  "
              f"rho_J={prot['spearman_vs_jaccard_mean']:.3f}  "
              f"R@5={prot['recall_embedding_vs_true_family']['5']:.2f}  "
              f"(J R@5={prot['recall_jaccard_vs_true_family']['5']:.2f})")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_02_ranking_correlation.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main()
