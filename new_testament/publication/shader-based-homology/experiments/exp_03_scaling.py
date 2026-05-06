"""
Experiment 3: scaling behaviour of the shader-kernel similarity engine.

We compare wall-clock time per query as a function of database size for:
  - exhaustive Smith-Waterman (reference upper bound on cost)
  - k-mer Jaccard
  - spectral-embedding cosine distance computed via the vectorised kernel
    (the CPU analogue of the fragment-shader evaluation)

All three compute the same quantity conceptually (per-database-entry
similarity score for one query) but at very different costs.
"""

from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (DNA_ALPHABET, generate_benchmark, embed_batch,
                    kmer_jaccard, smith_waterman, shader_kernel_distances)


def time_query_sw(query: str, db: list[str], budget: int) -> tuple[float, int]:
    """Time Smith-Waterman over up to `budget` database entries, then extrapolate."""
    probe_n = min(len(db), budget)
    t0 = time.perf_counter()
    for j in range(probe_n):
        smith_waterman(query, db[j])
    dt = time.perf_counter() - t0
    return dt, probe_n


def time_query_jaccard(query: str, db: list[str], k: int) -> float:
    t0 = time.perf_counter()
    for j in range(len(db)):
        kmer_jaccard(query, db[j], k)
    return time.perf_counter() - t0


def time_query_shader(query_vec: np.ndarray, db_mat: np.ndarray) -> float:
    t0 = time.perf_counter()
    _ = shader_kernel_distances(query_vec, db_mat)
    return time.perf_counter() - t0


def main():
    rng = np.random.default_rng(20260423)
    out = {
        "experiment": "scaling",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seed_length": 300,
            "family_size": 1,          # all "singletons" — random sequences
            "mutation_rate": 0.0,
            "n_coefficients": 12,
            "kmer_k": 6,
            "sw_probe_budget": 200,    # cap SW probing to keep runtime bounded
            "db_sizes": [100, 316, 1000, 3162, 10000, 31622, 100000],
            "n_queries_timed": 5,
            "random_seed": 20260423,
        },
        "rows": [],
    }
    cfg = out["config"]

    # build the biggest database once; subsample for smaller sizes
    max_n = max(cfg["db_sizes"])
    print("generating", max_n, "sequences...")
    seqs, _ = generate_benchmark(max_n, cfg["family_size"],
                                  cfg["seed_length"], cfg["mutation_rate"],
                                  DNA_ALPHABET, rng)

    print("embedding full database...")
    t0 = time.perf_counter()
    E_full = embed_batch(seqs, n_coefficients=cfg["n_coefficients"], kind="dna")
    build_time_total = time.perf_counter() - t0
    print(f"  build time: {build_time_total:.3f} s ({build_time_total / max_n * 1e6:.2f} us/seq)")

    for N in cfg["db_sizes"]:
        db = seqs[:N]
        E = E_full[:N]
        # queries outside the db (pick random external queries with same length)
        queries = [seqs[i % max_n] for i in rng.integers(0, max_n, size=cfg["n_queries_timed"])]
        Q = embed_batch(queries, n_coefficients=cfg["n_coefficients"], kind="dna")

        # shader kernel timings
        shader_times = [time_query_shader(Q[i], E) for i in range(len(queries))]
        shader_mean = float(np.mean(shader_times))
        shader_std = float(np.std(shader_times))

        # k-mer Jaccard
        jac_times = [time_query_jaccard(queries[i], db, cfg["kmer_k"])
                     for i in range(len(queries))]
        jac_mean = float(np.mean(jac_times))
        jac_std = float(np.std(jac_times))

        # Smith-Waterman: probe up to budget, extrapolate linearly
        sw_est_times = []
        sw_probe_n = 0
        for i in range(len(queries)):
            t_probe, pn = time_query_sw(queries[i], db, cfg["sw_probe_budget"])
            sw_probe_n = pn
            if pn == 0:
                sw_est_times.append(0.0)
            else:
                sw_est_times.append(t_probe * N / pn)
        sw_mean = float(np.mean(sw_est_times))
        sw_std = float(np.std(sw_est_times))

        row = {
            "db_size": int(N),
            "shader_kernel_s_mean": shader_mean,
            "shader_kernel_s_std": shader_std,
            "kmer_jaccard_s_mean": jac_mean,
            "kmer_jaccard_s_std": jac_std,
            "smith_waterman_s_mean_extrapolated": sw_mean,
            "smith_waterman_s_std_extrapolated": sw_std,
            "sw_probe_n_per_query": int(sw_probe_n),
            "shader_speedup_over_jaccard": jac_mean / shader_mean if shader_mean > 0 else float("inf"),
            "shader_speedup_over_sw_extrapolated": sw_mean / shader_mean if shader_mean > 0 else float("inf"),
        }
        out["rows"].append(row)
        print(f"N={N:>6}  shader={shader_mean*1e3:.3f} ms  "
              f"jaccard={jac_mean*1e3:.3f} ms  sw~{sw_mean*1e3:.3f} ms  "
              f"(speedup vs J: {row['shader_speedup_over_jaccard']:.0f}x, "
              f"vs SW: {row['shader_speedup_over_sw_extrapolated']:.0f}x)")

    out["build_time_total_s"] = build_time_total
    out["build_us_per_sequence"] = build_time_total / max_n * 1e6

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_03_scaling.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main()
