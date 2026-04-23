"""
Experiment 5: end-to-end retrieval pipeline.

We construct a database of synthetic protein families and evaluate a
two-stage pipeline:
  Stage A: shader-kernel cosine distance over all database entries.
  Stage B: top-K candidates re-ranked by Smith-Waterman local alignment.

We also evaluate a prefix-prefiltered variant (Stage 0: prefix match at
depth d, radius r; Stage A on the survivors; Stage B on the shader-top-K).

Metrics: retrieval wall-clock and recall of known family members at several
ranks K. We compare against full k-mer Jaccard over the whole database as
the alignment-free baseline.
"""

from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime, timezone
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (PROTEIN_ALPHABET, generate_benchmark, embed_batch,
                    kmer_jaccard, smith_waterman, shader_kernel_distances,
                    hierarchical_address, project_to_unit_cube)


def hamming_neighbours(addr: tuple, radius: int):
    """Yield all addresses within Hamming distance `radius`."""
    yield addr
    for r in range(1, radius + 1):
        for idx in combinations(range(len(addr)), r):
            alt = list(addr)
            for i in idx:
                alt[i] = 1 - alt[i]
            yield tuple(alt)


def run_pipeline(E: np.ndarray, seqs: list[str], coords: np.ndarray,
                 labels: np.ndarray, queries_idx: list[int],
                 depth: int, radius: int, top_k_shader: int, top_k_final: int,
                 kmer_k: int, use_prefilter: bool):
    n = E.shape[0]
    results = {"recall_at": {str(top_k_final): [], str(top_k_shader): []},
               "shader_time_s": [], "sw_time_s": [], "prefilter_time_s": [],
               "prefilter_fraction": []}

    # Build prefix index
    index: dict[tuple, list[int]] = {}
    for i, c in enumerate(coords):
        a = hierarchical_address(c, depth)
        index.setdefault(a, []).append(i)

    for q in queries_idx:
        true_fam = set(np.where(labels == labels[q])[0].tolist()) - {q}

        t0 = time.perf_counter()
        if use_prefilter:
            qa = hierarchical_address(coords[q], depth)
            cand = set()
            for a in hamming_neighbours(qa, radius):
                cand.update(index.get(a, []))
            cand.discard(q)
            candidates = np.array(sorted(cand), dtype=np.int64)
            if len(candidates) == 0:
                candidates = np.arange(n)
        else:
            candidates = np.arange(n)
        t_pref = time.perf_counter() - t0
        results["prefilter_time_s"].append(t_pref)
        results["prefilter_fraction"].append(len(candidates) / n)

        t0 = time.perf_counter()
        sub_d = shader_kernel_distances(E[q], E[candidates])
        t_shader = time.perf_counter() - t0
        results["shader_time_s"].append(t_shader)

        # top-K shader
        if len(sub_d) > top_k_shader:
            order_part = np.argpartition(sub_d, top_k_shader)[:top_k_shader]
            order = order_part[np.argsort(sub_d[order_part])]
        else:
            order = np.argsort(sub_d)
        top_shader_idx = candidates[order]

        # Re-rank with Smith-Waterman
        t0 = time.perf_counter()
        sw_scores = np.array([smith_waterman(seqs[q], seqs[j])
                              for j in top_shader_idx], dtype=np.int64)
        t_sw = time.perf_counter() - t0
        results["sw_time_s"].append(t_sw)
        final_order = top_shader_idx[np.argsort(-sw_scores)]

        # recall at shader-top
        hits_shader = sum(1 for j in top_shader_idx[:top_k_shader] if j in true_fam)
        hits_final = sum(1 for j in final_order[:top_k_final] if j in true_fam)
        denom = min(len(true_fam), top_k_shader) or 1
        results["recall_at"][str(top_k_shader)].append(hits_shader / denom)
        denom_f = min(len(true_fam), top_k_final) or 1
        results["recall_at"][str(top_k_final)].append(hits_final / denom_f)

    return {k: ([float(np.mean(v)) for v in [vv]] if isinstance(vv, list) else vv)
            for k, vv in results.items() if not isinstance(vv, dict)} | {
        "recall_mean": {k: float(np.mean(v)) for k, v in results["recall_at"].items()},
        "prefilter_mean_fraction": float(np.mean(results["prefilter_fraction"])),
        "shader_time_s_mean": float(np.mean(results["shader_time_s"])),
        "shader_time_s_std": float(np.std(results["shader_time_s"])),
        "prefilter_time_s_mean": float(np.mean(results["prefilter_time_s"])),
        "sw_time_s_mean": float(np.mean(results["sw_time_s"])),
    }


def run_jaccard_baseline(seqs: list[str], labels: np.ndarray,
                         queries_idx: list[int], top_k: int, kmer_k: int):
    n = len(seqs)
    recalls = []
    times = []
    for q in queries_idx:
        true_fam = set(np.where(labels == labels[q])[0].tolist()) - {q}
        t0 = time.perf_counter()
        jac = np.zeros(n, dtype=np.float64)
        for j in range(n):
            if j == q:
                continue
            jac[j] = kmer_jaccard(seqs[q], seqs[j], kmer_k)
        times.append(time.perf_counter() - t0)
        order = np.argsort(-jac)
        order = order[order != q][:top_k]
        denom = min(len(true_fam), top_k) or 1
        recalls.append(sum(1 for j in order if j in true_fam) / denom)
    return {
        "recall_at_k": float(np.mean(recalls)),
        "time_s_mean": float(np.mean(times)),
        "time_s_std": float(np.std(times)),
        "top_k": top_k,
        "kmer_k": kmer_k,
    }


def main():
    rng = np.random.default_rng(20260423)
    cfg = {
        "n_families": 200,
        "family_size": 5,
        "seed_length": 180,
        "mutation_rate": 0.12,
        "n_coefficients": 12,
        "n_queries": 40,
        "top_k_shader": 20,
        "top_k_final": 10,
        "kmer_k": 3,
        "random_seed": 20260423,
        "prefilter_depths": [3, 6, 9],
        "prefilter_radii": [0, 1, 2],
    }

    seqs, labels = generate_benchmark(cfg["n_families"], cfg["family_size"],
                                      cfg["seed_length"], cfg["mutation_rate"],
                                      PROTEIN_ALPHABET, rng)
    n = len(seqs)
    print(f"database size: {n} (protein)")

    t0 = time.perf_counter()
    E = embed_batch(seqs, n_coefficients=cfg["n_coefficients"], kind="protein")
    build_emb = time.perf_counter() - t0
    print(f"embedding build: {build_emb:.3f} s")

    joint_coords = project_to_unit_cube(E, n_dims=3)
    coords = joint_coords

    queries_idx = [int(rng.integers(0, n)) for _ in range(cfg["n_queries"])]

    out = {
        "experiment": "end_to_end",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "database_size": n,
        "embedding_build_s": build_emb,
        "pipelines": [],
        "jaccard_baseline": None,
    }

    # Full shader pipeline (no prefilter) + SW rerank
    print("\npipeline: shader-only + SW rerank")
    r = run_pipeline(E, seqs, coords, labels, queries_idx, depth=3, radius=3,
                     top_k_shader=cfg["top_k_shader"], top_k_final=cfg["top_k_final"],
                     kmer_k=cfg["kmer_k"], use_prefilter=False)
    r["name"] = "shader+SW rerank (no prefilter)"
    out["pipelines"].append(r)
    print(f"  recall@{cfg['top_k_final']}={r['recall_mean'][str(cfg['top_k_final'])]:.3f}  "
          f"recall@{cfg['top_k_shader']}={r['recall_mean'][str(cfg['top_k_shader'])]:.3f}  "
          f"shader={r['shader_time_s_mean']*1e3:.2f} ms  "
          f"SW_rerank={r['sw_time_s_mean']*1e3:.2f} ms")

    # With prefilter sweeps
    for d in cfg["prefilter_depths"]:
        for rad in cfg["prefilter_radii"]:
            print(f"\npipeline: prefilter(d={d}, r={rad}) + shader + SW rerank")
            r = run_pipeline(E, seqs, coords, labels, queries_idx,
                             depth=d, radius=rad,
                             top_k_shader=cfg["top_k_shader"],
                             top_k_final=cfg["top_k_final"],
                             kmer_k=cfg["kmer_k"], use_prefilter=True)
            r["name"] = f"prefilter(d={d},r={rad}) + shader + SW"
            r["depth"] = d
            r["radius"] = rad
            out["pipelines"].append(r)
            print(f"  recall@{cfg['top_k_final']}={r['recall_mean'][str(cfg['top_k_final'])]:.3f}  "
                  f"probe_frac={r['prefilter_mean_fraction']:.3f}  "
                  f"prefilter={r['prefilter_time_s_mean']*1e6:.1f} us  "
                  f"shader={r['shader_time_s_mean']*1e3:.2f} ms  "
                  f"SW={r['sw_time_s_mean']*1e3:.2f} ms")

    print("\nbaseline: k-mer Jaccard full DB (protein, k=3)")
    out["jaccard_baseline"] = run_jaccard_baseline(
        seqs, labels, queries_idx, cfg["top_k_final"], cfg["kmer_k"])
    print(f"  recall@{cfg['top_k_final']}={out['jaccard_baseline']['recall_at_k']:.3f}  "
          f"time={out['jaccard_baseline']['time_s_mean']*1e3:.2f} ms")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_05_end_to_end.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main()
