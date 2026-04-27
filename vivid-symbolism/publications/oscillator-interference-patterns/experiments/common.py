"""
Shared utilities for the oscillator-interference-pattern experiments.

The fundamental object is the linear cross-correlation of two real-valued
signals computed via the FFT, plus a multi-channel matched-filter built on
top of it.  All other experiments instantiate combinations of:

  - synthetic sequence simulation with planted motifs at known positions
  - DNA / protein channelisation (one-hot for DNA; physicochemical for protein)
  - linear cross-correlation via numpy.fft, with sliding-window energy
    normalisation
  - naive O(N*M) cross-correlation for testing
  - peak detection: argmax for the global peak, prominence-based for multiple
"""

from __future__ import annotations

import numpy as np
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
    mask = rng.random(n) < rate
    for i in np.nonzero(mask)[0]:
        out[i] = alphabet[rng.integers(0, len(alphabet))]
    return "".join(out)


def plant_motif(target: str, motif: str, position: int) -> str:
    """Replace target[position : position + len(motif)] with the motif."""
    if position < 0 or position + len(motif) > len(target):
        raise ValueError("motif does not fit at requested position")
    return target[:position] + motif + target[position + len(motif):]


def random_target_with_motifs(length: int, motif: str, positions: list[int],
                              mutation_rates: list[float], alphabet: str,
                              rng: np.random.Generator) -> tuple[str, list[str]]:
    target = random_sequence(length, alphabet, rng)
    planted: list[str] = []
    for pos, mr in zip(positions, mutation_rates):
        m = mutate(motif, mr, alphabet, rng) if mr > 0 else motif
        target = plant_motif(target, m, pos)
        planted.append(m)
    return target, planted


# ---------------------------------------------------------------------------
# Channelisation
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydropathy, normalised to [0, 1]
_KD = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
_kdmin, _kdmax = min(_KD.values()), max(_KD.values())
HYDROPATHY = {a: (v - _kdmin) / (_kdmax - _kdmin) for a, v in _KD.items()}

_VDW = {
    "G": 48.0, "A": 67.0, "S": 73.0, "C": 86.0, "D": 91.0, "P": 90.0,
    "N": 96.0, "T": 93.0, "E": 109.0, "V": 105.0, "Q": 114.0, "H": 118.0,
    "M": 124.0, "I": 124.0, "L": 124.0, "K": 135.0, "R": 148.0, "F": 135.0,
    "Y": 141.0, "W": 163.0,
}
_vmin, _vmax = min(_VDW.values()), max(_VDW.values())
VOLUME = {a: (v - _vmin) / (_vmax - _vmin) for a, v in _VDW.items()}

_CHARGE = {
    "K": 1.0, "R": 1.0, "H": 0.5, "D": -1.0, "E": -1.0,
    "A": 0.0, "C": 0.0, "F": 0.0, "G": 0.0, "I": 0.0, "L": 0.0,
    "M": 0.0, "N": 0.0, "P": 0.0, "Q": 0.0, "S": 0.0, "T": 0.0,
    "V": 0.0, "W": 0.0, "Y": 0.0,
}
CHARGE = {a: (v + 1.0) / 2.0 for a, v in _CHARGE.items()}


def dna_channels(seq: str, demean: bool = True) -> np.ndarray:
    """4 x L one-hot encoding, optionally per-channel mean-subtracted."""
    L = len(seq)
    out = np.zeros((4, L), dtype=np.float64)
    for i, b in enumerate(seq):
        idx = DNA_ALPHABET.find(b)
        if idx >= 0:
            out[idx, i] = 1.0
    if demean:
        out -= out.mean(axis=1, keepdims=True)
    return out


def protein_channels(seq: str, demean: bool = True) -> np.ndarray:
    """3 x L physicochemical encoding."""
    L = len(seq)
    out = np.zeros((3, L), dtype=np.float64)
    for i, aa in enumerate(seq):
        if aa in HYDROPATHY:
            out[0, i] = HYDROPATHY[aa]
            out[1, i] = VOLUME[aa]
            out[2, i] = CHARGE[aa]
    if demean:
        out -= out.mean(axis=1, keepdims=True)
    return out


# ---------------------------------------------------------------------------
# Cross-correlation kernels
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    return 1 << (max(1, n - 1)).bit_length()


def cross_correlate_fft(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Linear cross-correlation between a real query signal q[0..Lq-1] and a
    real target signal t[0..Lt-1]. Returns the array

        r[k] = sum_{n=0..Lq-1} q[n] * t[n + k],   k = 0..Lt - Lq.

    Computed via the FFT in O((Lq + Lt) log(Lq + Lt)) time.
    """
    Lq = len(query)
    Lt = len(target)
    if Lq == 0 or Lt == 0 or Lq > Lt:
        return np.zeros(max(0, Lt - Lq + 1), dtype=np.float64)

    L = Lq + Lt - 1
    n = _next_pow2(L)
    Q = np.fft.rfft(query, n=n)
    T = np.fft.rfft(target, n=n)
    full = np.fft.irfft(np.conj(Q) * T, n=n)
    # r[k] for k = 0..Lt-Lq lies in full[0..Lt-Lq] after the conjugate
    # convention above: cross-correlation = inverse-FFT of conj(Q) * T.
    return full[: Lt - Lq + 1]


def cross_correlate_naive(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    """O(Lq * (Lt - Lq + 1)) sliding-window inner product. Reference only."""
    Lq, Lt = len(query), len(target)
    out = np.zeros(Lt - Lq + 1, dtype=np.float64)
    for k in range(Lt - Lq + 1):
        out[k] = np.dot(query, target[k : k + Lq])
    return out


def cross_correlate_multichannel(query_ch: np.ndarray, target_ch: np.ndarray
                                 ) -> np.ndarray:
    """
    Sum of per-channel linear cross-correlations.

    query_ch  shape (c, Lq), target_ch shape (c, Lt). Output length Lt - Lq + 1.
    """
    c = query_ch.shape[0]
    accum = None
    for j in range(c):
        r = cross_correlate_fft(query_ch[j], target_ch[j])
        accum = r if accum is None else accum + r
    return accum


def normalised_xcorr(query_ch: np.ndarray, target_ch: np.ndarray
                     ) -> np.ndarray:
    """
    Normalised matched-filter score:

        score[k] = sum_c <q_c, t_c[k:k+Lq]> / (||Q|| * ||T_k||)

    where ||Q|| = sqrt(sum_c sum_n q_c[n]^2) and ||T_k|| is the Frobenius
    norm of the c x Lq target window starting at k. Returns values in
    approximately [-1, 1].
    """
    c, Lq = query_ch.shape
    Lt = target_ch.shape[1]
    raw = cross_correlate_multichannel(query_ch, target_ch)

    q_energy = np.sqrt(np.sum(query_ch ** 2))
    if q_energy <= 0:
        return np.zeros_like(raw)

    sq = target_ch ** 2
    cum = np.zeros(Lt + 1, dtype=np.float64)
    for j in range(c):
        cum[1:] += np.cumsum(sq[j])
    window_energy = np.sqrt(np.maximum(cum[Lq:] - cum[:Lt - Lq + 1], 0.0))

    out = np.zeros_like(raw)
    nz = window_energy > 0
    out[nz] = raw[nz] / (q_energy * window_energy[nz])
    return out


# ---------------------------------------------------------------------------
# Significance / peak picking
# ---------------------------------------------------------------------------

def background_zscore(scores: np.ndarray, exclude: list[tuple[int, int]] | None = None
                      ) -> tuple[float, float]:
    """
    Robust background mean/std.  If `exclude` ranges are supplied (each
    is a half-open [a, b) of indices to ignore), the statistics are computed
    over the complement.
    """
    if exclude is None:
        return float(scores.mean()), float(scores.std())
    mask = np.ones(scores.size, dtype=bool)
    for a, b in exclude:
        a = max(0, a)
        b = min(scores.size, b)
        if b > a:
            mask[a:b] = False
    sub = scores[mask]
    if sub.size == 0:
        return float(scores.mean()), float(scores.std())
    return float(sub.mean()), float(sub.std())


def find_peaks(scores: np.ndarray, min_separation: int, min_score: float
               ) -> list[tuple[int, float]]:
    """
    Greedy maximum-suppression: repeatedly pick the global argmax that has
    score >= min_score and is at least `min_separation` away from any
    previously-picked peak.
    """
    out: list[tuple[int, float]] = []
    masked = scores.copy()
    while True:
        idx = int(np.argmax(masked))
        score = float(masked[idx])
        if score < min_score:
            break
        out.append((idx, score))
        a = max(0, idx - min_separation)
        b = min(masked.size, idx + min_separation + 1)
        masked[a:b] = -np.inf
        if not np.isfinite(masked.max()):
            break
    return out
