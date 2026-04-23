"""Generate figures from the JSON results of each experiment."""

from __future__ import annotations
import json
import os
import sys

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


def fig_exp01():
    d = _load("exp_01_embedding_separation.json")
    mrs = d["config"]["mutation_rates"]
    dna_auc = [r["auc"] for r in d["dna"]]
    prot_auc = [r["auc"] for r in d["protein"]]
    dna_d = [r["cohen_d"] for r in d["dna"]]
    prot_d = [r["cohen_d"] for r in d["protein"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(mrs, dna_auc, "o-", label="DNA", color="#2a6f97")
    ax1.plot(mrs, prot_auc, "s-", label="protein", color="#c9423a")
    ax1.axhline(0.5, linestyle=":", color="grey", linewidth=1)
    ax1.set_xlabel("substitution rate")
    ax1.set_ylabel("ROC-AUC, within-vs-between family")
    ax1.set_ylim(0.45, 1.02)
    ax1.legend(frameon=False)
    ax1.set_title("(a) homolog–non-homolog separation")

    ax2.plot(mrs, dna_d, "o-", label="DNA", color="#2a6f97")
    ax2.plot(mrs, prot_d, "s-", label="protein", color="#c9423a")
    ax2.set_xlabel("substitution rate")
    ax2.set_ylabel("Cohen's d (within vs. between)")
    ax2.set_title("(b) effect size")
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_01_separation.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "fig_01_separation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_exp02():
    d = _load("exp_02_ranking_correlation.json")
    runs = d["runs"]
    mrs = sorted(set(r["mutation_rate"] for r in runs))
    kinds = ["dna", "protein"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, kind in zip(axes, kinds):
        sub = [r for r in runs if r["kind"] == kind]
        sub.sort(key=lambda r: r["mutation_rate"])
        mrsk = [r["mutation_rate"] for r in sub]
        r1 = [r["recall_embedding_vs_true_family"]["1"] for r in sub]
        r5 = [r["recall_embedding_vs_true_family"]["5"] for r in sub]
        r10 = [r["recall_embedding_vs_true_family"]["10"] for r in sub]
        jr5 = [r["recall_jaccard_vs_true_family"]["5"] for r in sub]
        ax.plot(mrsk, r1, "o-", label="shader R@1", color="#2a6f97")
        ax.plot(mrsk, r5, "s-", label="shader R@5", color="#c9423a")
        ax.plot(mrsk, r10, "^-", label="shader R@10", color="#6b8f3a")
        ax.plot(mrsk, jr5, "x--", label="k-mer Jaccard R@5", color="grey")
        ax.set_xlabel("substitution rate")
        ax.set_ylabel("recall@k for true family")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"({'a' if kind=='dna' else 'b'}) {kind}")
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_02_recall.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "fig_02_recall.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_exp03():
    d = _load("exp_03_scaling.json")
    rows = d["rows"]
    N = np.array([r["db_size"] for r in rows], dtype=float)
    t_shader = np.array([r["shader_kernel_s_mean"] for r in rows])
    t_jac = np.array([r["kmer_jaccard_s_mean"] for r in rows])
    t_sw = np.array([r["smith_waterman_s_mean_extrapolated"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.loglog(N, t_shader * 1e3, "o-", label="shader kernel (numpy vec.)", color="#2a6f97")
    ax1.loglog(N, t_jac * 1e3, "s-", label="k-mer Jaccard (Python)", color="#c9423a")
    ax1.loglog(N, t_sw * 1e3, "^-", label="Smith–Waterman (extrap.)", color="#6b8f3a")
    ax1.set_xlabel("database size $N$")
    ax1.set_ylabel("per-query time [ms]")
    ax1.set_title("(a) per-query cost vs. database size")
    ax1.legend(frameon=False, fontsize=8)

    speedup_j = t_jac / t_shader
    speedup_sw = t_sw / t_shader
    ax2.loglog(N, speedup_j, "s-", label="vs. k-mer Jaccard", color="#c9423a")
    ax2.loglog(N, speedup_sw, "^-", label="vs. Smith–Waterman", color="#6b8f3a")
    ax2.set_xlabel("database size $N$")
    ax2.set_ylabel("speedup factor")
    ax2.set_title("(b) speedup factor")
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_03_scaling.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "fig_03_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_exp04():
    d = _load("exp_04_prefix_addressing.json")
    rows = d["rows"]
    depth = [r["depth"] for r in rows]
    recall = [r["mean_recall_true_family"] for r in rows]
    frac = [r["mean_fraction_of_db_probed"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(5.2, 3.6))
    ax1.plot(depth, recall, "o-", color="#2a6f97", label="family recall")
    ax1.set_xlabel("prefix depth")
    ax1.set_ylabel("family recall (radius-1 expansion)", color="#2a6f97")
    ax1.tick_params(axis="y", labelcolor="#2a6f97")
    ax1.set_ylim(-0.02, 1.02)
    ax2 = ax1.twinx()
    ax2.semilogy(depth, frac, "s--", color="#c9423a", label="db fraction probed")
    ax2.set_ylabel("fraction of database probed (log)", color="#c9423a")
    ax2.tick_params(axis="y", labelcolor="#c9423a")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_04_prefix.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "fig_04_prefix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_exp05():
    d = _load("exp_05_end_to_end.json")
    pipes = d["pipelines"]
    jac = d["jaccard_baseline"]
    cfg = d["config"]
    k_final = cfg["top_k_final"]

    labels = [p["name"] for p in pipes]
    recall = [p["recall_mean"][str(k_final)] for p in pipes]
    total_time = [(p["prefilter_time_s_mean"] + p["shader_time_s_mean"] +
                    p["sw_time_s_mean"]) for p in pipes]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.scatter([t * 1e3 for t in total_time], recall, s=60, color="#2a6f97")
    for x, y, lbl in zip(total_time, recall, labels):
        ax.annotate(lbl, (x * 1e3, y), fontsize=7, textcoords="offset points",
                    xytext=(4, 3))
    ax.scatter([jac["time_s_mean"] * 1e3], [jac["recall_at_k"]],
               s=60, color="#c9423a", marker="s", label="k-mer Jaccard baseline")
    ax.annotate("Jaccard (full DB)",
                (jac["time_s_mean"] * 1e3, jac["recall_at_k"]),
                fontsize=7, textcoords="offset points", xytext=(4, 3))
    ax.set_xscale("log")
    ax.set_xlabel("total per-query wall time [ms]")
    ax.set_ylabel(f"recall@{k_final}")
    ax.set_title("Speed–recall Pareto: shader pipeline vs. k-mer Jaccard")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_05_pareto.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, "fig_05_pareto.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    for name, fn in [
        ("fig_01", fig_exp01),
        ("fig_02", fig_exp02),
        ("fig_03", fig_exp03),
        ("fig_04", fig_exp04),
        ("fig_05", fig_exp05),
    ]:
        try:
            fn()
            print(f"ok  {name}")
        except FileNotFoundError as e:
            print(f"skip {name}: missing {e.filename}")


if __name__ == "__main__":
    main()
