"""spectral.py -- three receivers of a nucleotide sequence.

  magnitude  psi(s): magnitudes of the non-DC DFT coefficients 1..K of each
             mean-centred one-hot channel (A, C, G, T), divided by L, then
             L2-normalised. K = None keeps every bin up to floor(L/2).
  strand     psi_sym(s): psi(s) + P psi(s), L2-normalised, where P exchanges
             the A<->T and C<->G channel blocks.
  phase      rho0(s, t): cosine of the mean-centred one-hot matrices at zero
             lag (the matched filter evaluated at full overlap).
Visibility of two sequences under a receiver is the cosine of their images.
"""

from __future__ import annotations

import numpy as np

ALPH = "ACGT"
COMP = np.array([3, 2, 1, 0])        # A<->T, C<->G on channel indices


def onehot(s: np.ndarray) -> np.ndarray:
    """s: (..., L) integer array in 0..3 -> (..., 4, L) float."""
    return (s[..., None, :] == np.arange(4)[:, None]).astype(float)


def centred(s: np.ndarray) -> np.ndarray:
    X = onehot(s)
    return X - X.mean(axis=-1, keepdims=True)


def magnitude(s: np.ndarray, K: int | None = None) -> np.ndarray:
    L = s.shape[-1]
    F = np.abs(np.fft.rfft(centred(s), axis=-1))[..., 1:]
    if K is not None:
        F = F[..., :K]
    E = (F / L).reshape(*s.shape[:-1], -1)
    n = np.linalg.norm(E, axis=-1, keepdims=True)
    return np.divide(E, n, out=np.zeros_like(E), where=n > 0)


def strand(s: np.ndarray, K: int | None = None) -> np.ndarray:
    L = s.shape[-1]
    F = np.abs(np.fft.rfft(centred(s), axis=-1))[..., 1:]
    if K is not None:
        F = F[..., :K]
    F = F + F[..., COMP, :]
    E = F.reshape(*s.shape[:-1], -1)
    n = np.linalg.norm(E, axis=-1, keepdims=True)
    return np.divide(E, n, out=np.zeros_like(E), where=n > 0)


def phase_image(s: np.ndarray) -> np.ndarray:
    E = centred(s).reshape(*s.shape[:-1], -1)
    n = np.linalg.norm(E, axis=-1, keepdims=True)
    return np.divide(E, n, out=np.zeros_like(E), where=n > 0)


def visibility(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1)


def rotate(s, k):
    return np.roll(s, k, axis=-1)


def reverse(s):
    return s[..., ::-1]


def revcomp(s):
    return COMP[s[..., ::-1]]


def mutate(rng, s, mu):
    t = s.copy()
    hit = rng.random(s.shape) < mu
    t[hit] = (t[hit] + rng.integers(1, 4, size=hit.sum())) % 4
    return t


def all_sequences(L: int) -> np.ndarray:
    idx = np.arange(4 ** L)
    return np.stack([(idx // 4 ** (L - 1 - j)) % 4 for j in range(L)], axis=1).astype(np.int8)


def encode(s: np.ndarray) -> np.ndarray:
    L = s.shape[-1]
    return (s.astype(np.int64) * (4 ** np.arange(L - 1, -1, -1))).sum(-1)


def dihedral_codes(s: np.ndarray) -> np.ndarray:
    """Codes of every rotation and every rotated reversal of each row of s."""
    L = s.shape[-1]
    imgs = [rotate(s, k) for k in range(L)] + [rotate(reverse(s), k) for k in range(L)]
    return np.stack([encode(x) for x in imgs], axis=1)


def keys(E: np.ndarray, decimals: int = 9) -> np.ndarray:
    """Hashable fibre keys for rows of an embedding."""
    R = np.round(E, decimals) + 0.0
    return np.array([hash(r.tobytes()) for r in R])
