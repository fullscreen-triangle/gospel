"""
semantics.py -- the executable primitives of Sec 8.

These are the denotations the typing rules point at. Each one has a
stated range or normalisation in the manuscript, and each of those is a
checkable claim.
"""

from __future__ import annotations

import numpy as np

# Kyte-Doolittle hydropathy, van der Waals volume (A^3), net charge.
AA = "ACDEFGHIKLMNPQRSTVWY"
KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
    "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
    "W": -0.9, "Y": -1.3,
}
VOL = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9, "G": 60.1,
    "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7, "M": 162.9, "N": 114.1,
    "P": 112.7, "Q": 143.8, "R": 173.4, "S": 89.0, "T": 116.1, "V": 140.0,
    "W": 227.8, "Y": 193.6,
}
CHG = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5}


# =====================================================================
# Definition 8.1 -- channelisation
# =====================================================================

def channelise_dna(seq: str) -> np.ndarray:
    """4-channel one-hot, then CENTRED per channel.

    Centring is what makes the correlation of Def 8.2 a correlation
    rather than a raw overlap; without it a run of A's correlates with
    everything.
    """
    seq = seq.upper()
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    X = np.zeros((len(seq), 4))
    for i, ch in enumerate(seq):
        j = idx.get(ch)
        if j is not None:
            X[i, j] = 1.0
    return X - X.mean(axis=0, keepdims=True)


def channelise_protein(seq: str) -> np.ndarray:
    """3-channel physicochemical, normalised to unit scale then centred."""
    seq = seq.upper()
    X = np.zeros((len(seq), 3))
    for i, ch in enumerate(seq):
        X[i, 0] = KD.get(ch, 0.0)
        X[i, 1] = VOL.get(ch, 0.0)
        X[i, 2] = CHG.get(ch, 0.0)
    scale = np.array([4.5, 227.8, 1.0])
    X = X / scale
    return X - X.mean(axis=0, keepdims=True)


def cardinal(seq: str) -> np.ndarray:
    """2-vector trajectory. The complementary strand is the NEGATED path.

    This is the concrete form of the argument that the invariant is not
    borne by the sequence: two strands carrying identical information
    give exactly opposite trajectories.
    """
    step = {"A": (0.0, 1.0), "T": (0.0, -1.0), "G": (1.0, 0.0), "C": (-1.0, 0.0)}
    out = np.zeros((len(seq), 2))
    p = np.zeros(2)
    for i, ch in enumerate(seq.upper()):
        d = step.get(ch, (0.0, 0.0))
        p = p + np.array(d)
        out[i] = p
    return out


def complement(seq: str) -> str:
    return seq.upper().translate(str.maketrans("ACGT", "TGCA"))


# =====================================================================
# Definition 8.2 -- normalised cross-correlation
# =====================================================================

def xcorr_naive(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Direct O(L_t L_q C) reference implementation."""
    Lq, C = q.shape
    Lt = t.shape[0]
    n = Lt - Lq + 1
    if n <= 0:
        return np.zeros(0)
    qn = np.linalg.norm(q)
    out = np.zeros(n)
    for k in range(n):
        w = t[k:k + Lq]
        wn = np.linalg.norm(w)
        d = qn * wn
        out[k] = 0.0 if d <= 1e-15 else float((q * w).sum() / d)
    return out


def xcorr_fft(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """FFT implementation with coherent channel summation.

    Channels are summed IN THE NUMERATOR before normalising -- summing
    per-channel correlations instead would let one channel's agreement
    cancel another's disagreement.
    """
    Lq, C = q.shape
    Lt = t.shape[0]
    n = Lt - Lq + 1
    if n <= 0:
        return np.zeros(0)
    nfft = 1
    while nfft < Lt + Lq:
        nfft *= 2

    num = np.zeros(nfft)
    for c in range(C):
        Q = np.fft.rfft(q[::-1, c], nfft)
        T = np.fft.rfft(t[:, c], nfft)
        num += np.fft.irfft(Q * T, nfft)
    num = num[Lq - 1:Lq - 1 + n]

    # Rolling Frobenius norm of the window.
    sq = (t ** 2).sum(axis=1)
    cs = np.concatenate([[0.0], np.cumsum(sq)])
    wn = np.sqrt(np.maximum(cs[Lq:Lq + n] - cs[:n], 0.0))
    qn = np.linalg.norm(q)

    d = qn * wn
    out = np.zeros(n)
    nz = d > 1e-15
    out[nz] = num[nz] / d[nz]
    return out


# =====================================================================
# Definition 8.3 -- spectral embedding
# =====================================================================

def spectral(x: np.ndarray, coeffs: int) -> np.ndarray:
    """rFFT magnitudes SKIPPING DC, zero-padded, length-scaled, l2-normed.

    The DC term is skipped because it is the mean, which is composition,
    not structure; keeping it would make two sequences of equal base
    composition look similar regardless of arrangement.
    """
    v = x if x.ndim == 1 else x.mean(axis=1)
    L = len(v)
    mag = np.abs(np.fft.rfft(v))[1:]          # skip DC
    e = np.zeros(coeffs)
    m = min(coeffs, len(mag))
    e[:m] = mag[:m]
    e = e / max(L, 1)
    nrm = np.linalg.norm(e)
    return e if nrm <= 1e-15 else e / nrm


# =====================================================================
# Definition 8.4 -- shader distance
# =====================================================================

def shader_distance(B: np.ndarray, q: np.ndarray) -> np.ndarray:
    """d = 1 - Bq, in [0,2]^N for l2-normalised rows and query."""
    return 1.0 - B @ q


# =====================================================================
# Hierarchical address (Sec 8.5) -- axis-cyclic bisection
# =====================================================================

def address(point: np.ndarray, lo: np.ndarray, hi: np.ndarray,
            depth: int) -> str:
    """Bisect cyclically over axes; return the bit string."""
    lo = lo.astype(float).copy()
    hi = hi.astype(float).copy()
    bits = []
    d = len(point)
    for j in range(depth):
        ax = j % d
        mid = 0.5 * (lo[ax] + hi[ax])
        if point[ax] <= mid:
            bits.append("0")
            hi[ax] = mid
        else:
            bits.append("1")
            lo[ax] = mid
    return "".join(bits)
