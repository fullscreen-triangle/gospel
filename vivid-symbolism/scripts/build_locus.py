"""
Build a synthetic but biologically-structured 1 Mb DNA locus with planted
features, slide a window across it, and precompute the spectral embedding
of every window. The web browser tool then computes cosine similarity to a
user-supplied query in real time and paints one pixel per window.

Output files (in `public/data/`):
  locus.meta.json        - metadata + planted feature annotations
  locus.fasta            - the raw DNA sequence
  locus.embeddings.bin   - Float32 little-endian, shape (N_windows, embed_dim)

Planted features:
  - CpG island (high GC content)
  - AT-rich heterochromatin
  - Tandem repeat
  - Two copies of a 200 bp motif (the second 10% mutated): the canonical
    use case for the browser is to paste this motif and see both hits.
"""

from __future__ import annotations
import json
import os
import sys
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "publications", "shader-based-homology",
                                "experiments"))
from common import DNA_ALPHABET, embed_batch, mutate, random_sequence


LOCUS_LENGTH = 1_000_000  # 1 Mb
WINDOW_SIZE = 200
WINDOW_STRIDE = 100
N_COEFFICIENTS = 12

CpG_START, CpG_END = 100_000, 105_000
AT_START, AT_END = 200_000, 220_000
REPEAT_START, REPEAT_END = 350_000, 360_000
MOTIF_LENGTH = 200
MOTIF_A_POS = 500_000
MOTIF_B_POS = 800_000


def random_uniform(length: int, rng: np.random.Generator) -> str:
    idx = rng.integers(0, 4, size=length)
    return "".join(DNA_ALPHABET[i] for i in idx)


def biased_dna(length: int, gc_fraction: float, rng: np.random.Generator) -> str:
    """Generate DNA with a tunable GC content."""
    p = np.array([(1 - gc_fraction) / 2, gc_fraction / 2,
                  gc_fraction / 2, (1 - gc_fraction) / 2])  # A, C, G, T
    idx = rng.choice(4, size=length, p=p)
    return "".join(DNA_ALPHABET[i] for i in idx)


def cpg_island(length: int, rng: np.random.Generator) -> str:
    """High-GC sequence with elevated CpG dinucleotide frequency."""
    base = biased_dna(length, 0.70, rng)
    # boost CpG: scan and sometimes flip GpC -> CpG
    bases = list(base)
    for i in range(len(bases) - 1):
        if bases[i] == "C" and bases[i + 1] != "G" and rng.random() < 0.35:
            bases[i + 1] = "G"
    return "".join(bases)


def tandem_repeat(length: int, motif: str = "ATATCCG") -> str:
    repeats = (length // len(motif)) + 1
    return (motif * repeats)[:length]


def build_locus(rng: np.random.Generator) -> tuple[str, list[dict]]:
    parts: list[str] = []
    annotations: list[dict] = []
    cursor = 0

    def emit(seq: str, label: str | None = None):
        nonlocal cursor
        parts.append(seq)
        if label:
            annotations.append({"start": cursor, "end": cursor + len(seq), "label": label})
        cursor += len(seq)

    # 0 - 100k: random
    emit(random_uniform(CpG_START, rng))

    # CpG island
    emit(cpg_island(CpG_END - CpG_START, rng), "CpG island (GC ~ 0.70)")

    # ... -> AT-rich
    emit(random_uniform(AT_START - cursor, rng))
    emit(biased_dna(AT_END - AT_START, 0.20, rng), "AT-rich heterochromatin (GC ~ 0.20)")

    # ... -> tandem repeat
    emit(random_uniform(REPEAT_START - cursor, rng))
    emit(tandem_repeat(REPEAT_END - REPEAT_START), "Tandem repeat (ATATCCG)")

    # ... -> planted motif A at MOTIF_A_POS
    emit(random_uniform(MOTIF_A_POS - cursor, rng))
    motif_seed = random_sequence(MOTIF_LENGTH, DNA_ALPHABET, rng)
    emit(motif_seed, "Planted motif A (canonical)")

    # ... -> planted motif B (10% mutated copy of motif A) at MOTIF_B_POS
    emit(random_uniform(MOTIF_B_POS - cursor, rng))
    motif_b = mutate(motif_seed, 0.10, DNA_ALPHABET, rng)
    emit(motif_b, "Planted motif A (10% mutated)")

    # tail to LOCUS_LENGTH
    emit(random_uniform(LOCUS_LENGTH - cursor, rng))

    full = "".join(parts)
    assert len(full) == LOCUS_LENGTH, (len(full), LOCUS_LENGTH)
    return full, annotations, motif_seed


def slide_windows(seq: str, win: int, stride: int) -> tuple[list[str], list[int]]:
    starts = list(range(0, len(seq) - win + 1, stride))
    windows = [seq[s:s + win] for s in starts]
    return windows, starts


def main():
    rng = np.random.default_rng(20260425)

    print(f"building {LOCUS_LENGTH:,} bp locus...")
    locus, annotations, motif_seed = build_locus(rng)

    print(f"sliding window: {WINDOW_SIZE} bp, stride {WINDOW_STRIDE} bp")
    windows, starts = slide_windows(locus, WINDOW_SIZE, WINDOW_STRIDE)
    print(f"  {len(windows):,} windows")

    print("computing spectral embeddings...")
    E = embed_batch(windows, n_coefficients=N_COEFFICIENTS, kind="dna").astype(np.float32)
    print(f"  embedding matrix: {E.shape}, {E.nbytes / 1024**2:.2f} MiB")

    out_dir = os.path.join(HERE, "..", "public", "data")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Sequence as a plain FASTA so we can also slice it on the client.
    fasta_path = os.path.join(out_dir, "locus.fasta")
    with open(fasta_path, "w") as f:
        f.write(">synthetic_locus_1Mb\n")
        for i in range(0, len(locus), 80):
            f.write(locus[i:i + 80] + "\n")

    # 2. Embeddings as a tightly-packed Float32 binary (little-endian).
    bin_path = os.path.join(out_dir, "locus.embeddings.bin")
    E.astype("<f4").tofile(bin_path)

    # 3. Metadata + planted features + canonical query motif.
    meta = {
        "name": "synthetic_locus_1Mb",
        "kind": "dna",
        "length": LOCUS_LENGTH,
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "n_windows": int(E.shape[0]),
        "embed_dim": int(E.shape[1]),
        "n_coefficients": N_COEFFICIENTS,
        "starts_first": int(starts[0]),
        "starts_last": int(starts[-1]),
        "annotations": annotations,
        "demo_query": motif_seed,
        "build_seed": 20260425,
    }
    meta_path = os.path.join(out_dir, "locus.meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote\n  {fasta_path} ({os.path.getsize(fasta_path) / 1024:.1f} KiB)")
    print(f"  {bin_path} ({os.path.getsize(bin_path) / 1024**2:.2f} MiB)")
    print(f"  {meta_path} ({os.path.getsize(meta_path) / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
