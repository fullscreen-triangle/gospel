"""
Common utilities for the shader-based homology search experiments.

Implements:
- Synthetic sequence family generation (seed + point mutation)
- Spectral sequence embedding (discrete Fourier transform of one-hot / physicochemical channels)
- Shader-kernel simulation (vectorised cosine distance over a database coordinate array)
- Reference baselines: k-mer Jaccard similarity and Smith-Waterman local alignment
- Hierarchical bisection addressing for coordinate-space prefix search

All random generation uses explicit numpy Generators for reproducibility.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable


DNA_ALPHABET = "ACGT"
PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# Sequence simulation
# ---------------------------------------------------------------------------

def random_sequence(length: int, alphabet: str, rng: np.random.Generator) -> str:
    idx = rng.integers(0, len(alphabet), size=length)
    return "".join(alphabet[i] for i in idx)


def mutate(seq: str, rate: float, alphabet: str, rng: np.random.Generator) -> str:
    out = list(seq)
    n = len(seq)
    # substitutions
    mask = rng.random(n) < rate
    for i in np.nonzero(mask)[0]:
        out[i] = alphabet[rng.integers(0, len(alphabet))]
    return "".join(out)


def generate_family(seed_length: int, family_size: int, mutation_rate: float,
                    alphabet: str, rng: np.random.Generator) -> list[str]:
    seed = random_sequence(seed_length, alphabet, rng)
    members = [seed]
    for _ in range(family_size - 1):
        members.append(mutate(seed, mutation_rate, alphabet, rng))
    return members


def generate_benchmark(n_families: int, family_size: int, seed_length: int,
                       mutation_rate: float, alphabet: str,
                       rng: np.random.Generator) -> tuple[list[str], np.ndarray]:
    """Return a flat list of sequences and a label array marking family membership."""
    seqs: list[str] = []
    labels: list[int] = []
    for fam in range(n_families):
        members = generate_family(seed_length, family_size, mutation_rate, alphabet, rng)
        seqs.extend(members)
        labels.extend([fam] * family_size)
    return seqs, np.asarray(labels)


# ---------------------------------------------------------------------------
# Spectral embedding
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydrophobicity, normalised to [0, 1]
_KD = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
_KD_MIN, _KD_MAX = min(_KD.values()), max(_KD.values())
HYDROPATHY = {aa: (v - _KD_MIN) / (_KD_MAX - _KD_MIN) for aa, v in _KD.items()}

# van der Waals volumes (Å^3), normalised
_VDW = {
    "G": 48.0, "A": 67.0, "S": 73.0, "C": 86.0, "D": 91.0, "P": 90.0,
    "N": 96.0, "T": 93.0, "E": 109.0, "V": 105.0, "Q": 114.0, "H": 118.0,
    "M": 124.0, "I": 124.0, "L": 124.0, "K": 135.0, "R": 148.0, "F": 135.0,
    "Y": 141.0, "W": 163.0,
}
_V_MIN, _V_MAX = min(_VDW.values()), max(_VDW.values())
VOLUME = {aa: (v - _V_MIN) / (_V_MAX - _V_MIN) for aa, v in _VDW.items()}

# Net charge at neutral pH, scaled to [0, 1]
_CHARGE = {
    "K": 1.0, "R": 1.0, "H": 0.5,
    "D": -1.0, "E": -1.0,
    "A": 0.0, "C": 0.0, "F": 0.0, "G": 0.0, "I": 0.0, "L": 0.0,
    "M": 0.0, "N": 0.0, "P": 0.0, "Q": 0.0, "S": 0.0, "T": 0.0,
    "V": 0.0, "W": 0.0, "Y": 0.0,
}
CHARGE = {aa: (v + 1.0) / 2.0 for aa, v in _CHARGE.items()}


def dna_channels(seq: str) -> np.ndarray:
    """4 x L one-hot representation of a DNA sequence."""
    idx = np.fromiter((DNA_ALPHABET.find(b) for b in seq), dtype=np.int32, count=len(seq))
    valid = idx >= 0
    out = np.zeros((4, len(seq)), dtype=np.float32)
    out[idx[valid], np.where(valid)[0]] = 1.0
    # Remove global mean per channel (so the DC component does not dominate the spectrum)
    out -= out.mean(axis=1, keepdims=True)
    return out


def protein_channels(seq: str) -> np.ndarray:
    """3 x L physicochemical encoding: hydropathy, volume, charge."""
    out = np.zeros((3, len(seq)), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in HYDROPATHY:
            out[0, i] = HYDROPATHY[aa]
            out[1, i] = VOLUME[aa]
            out[2, i] = CHARGE[aa]
    out -= out.mean(axis=1, keepdims=True)
    return out


def spectral_embedding(seq: str, n_coefficients: int, kind: str = "dna") -> np.ndarray:
    """
    Map a sequence to a fixed-dimensional vector by taking the first
    `n_coefficients` magnitudes of the DFT of each channel. Length-normalised.
    """
    if kind == "dna":
        channels = dna_channels(seq)
    elif kind == "protein":
        channels = protein_channels(seq)
    else:
        raise ValueError(kind)

    c, L = channels.shape
    if L == 0:
        return np.zeros(c * n_coefficients, dtype=np.float32)

    # rFFT magnitudes at the lowest `n_coefficients` non-DC frequencies
    spec = np.abs(np.fft.rfft(channels, axis=1))
    k = min(n_coefficients, spec.shape[1] - 1)
    out = spec[:, 1:1 + k]                     # skip DC
    if k < n_coefficients:
        pad = np.zeros((c, n_coefficients - k), dtype=spec.dtype)
        out = np.concatenate([out, pad], axis=1)
    # length normalisation so short/long variants share the same scale
    out = out / max(L, 1)
    return out.astype(np.float32).ravel()


def embed_batch(seqs: Iterable[str], n_coefficients: int, kind: str) -> np.ndarray:
    vecs = [spectral_embedding(s, n_coefficients, kind) for s in seqs]
    M = np.stack(vecs, axis=0)
    # L2-normalise so cosine similarity = dot product
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms


# ---------------------------------------------------------------------------
# Shader-kernel simulation
# ---------------------------------------------------------------------------

def shader_kernel_distances(query_vec: np.ndarray, db_matrix: np.ndarray) -> np.ndarray:
    """
    Vectorised analogue of the fragment-shader similarity evaluation.

    Each pixel/fragment conceptually corresponds to one database row; the
    computation performed here (per-row dot product with a query uniform) is
    the same arithmetic that a fragment shader executes in parallel across all
    pixels of the database texture. We use numpy as a CPU reference, but the
    operation maps trivially onto the GPU rendering pipeline.

    Returns cosine distances in [0, 2]; lower is more similar.
    """
    sims = db_matrix @ query_vec
    return 1.0 - sims


# ---------------------------------------------------------------------------
# Reference baselines
# ---------------------------------------------------------------------------

def kmer_set(seq: str, k: int) -> set[str]:
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def kmer_jaccard(a: str, b: str, k: int) -> float:
    A = kmer_set(a, k)
    B = kmer_set(b, k)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def smith_waterman(a: str, b: str, match: int = 2, mismatch: int = -1,
                   gap: int = -2) -> int:
    """Standard Smith-Waterman local alignment score (no traceback)."""
    la, lb = len(a), len(b)
    H = np.zeros((la + 1, lb + 1), dtype=np.int32)
    best = 0
    for i in range(1, la + 1):
        ai = a[i - 1]
        row_prev = H[i - 1]
        row_cur = H[i]
        for j in range(1, lb + 1):
            s = match if ai == b[j - 1] else mismatch
            diag = row_prev[j - 1] + s
            up = row_prev[j] + gap
            left = row_cur[j - 1] + gap
            v = max(0, diag, up, left)
            row_cur[j] = v
            if v > best:
                best = v
    return int(best)


# ---------------------------------------------------------------------------
# Hierarchical address
# ---------------------------------------------------------------------------

def hierarchical_address(coord: np.ndarray, depth: int, value_range: tuple[float, float]
                         = (0.0, 1.0)) -> tuple[int, ...]:
    """
    Given a (already-bounded) coordinate vector, produce a hierarchical bit
    address by dimension-rotating bisection. At each step, choose the next
    dimension cyclically and record whether the point is in the upper or lower
    half of the active interval along that dimension, then halve the interval.
    """
    lo = np.full_like(coord, value_range[0], dtype=np.float64)
    hi = np.full_like(coord, value_range[1], dtype=np.float64)
    d = len(coord)
    bits: list[int] = []
    for step in range(depth):
        axis = step % d
        mid = 0.5 * (lo[axis] + hi[axis])
        if coord[axis] >= mid:
            bits.append(1)
            lo[axis] = mid
        else:
            bits.append(0)
            hi[axis] = mid
    return tuple(bits)


def project_to_unit_cube(embeddings: np.ndarray, n_dims: int = 3) -> np.ndarray:
    """
    Reduce the spectral embedding to an n_dims-dimensional cube address via
    deterministic random projection followed by min-max scaling. We use a
    fixed seed so the projection is reproducible across runs/datasets.
    """
    rng = np.random.default_rng(17)
    D = embeddings.shape[1]
    R = rng.standard_normal((D, n_dims)).astype(np.float32)
    R /= np.linalg.norm(R, axis=0, keepdims=True)
    proj = embeddings @ R
    mn = proj.min(axis=0, keepdims=True)
    mx = proj.max(axis=0, keepdims=True)
    scale = np.where(mx - mn > 0, mx - mn, 1.0)
    return ((proj - mn) / scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

@dataclass
class Timing:
    seconds: float
    operations: int = 0

    def throughput(self) -> float:
        return self.operations / self.seconds if self.seconds > 0 else float("inf")
