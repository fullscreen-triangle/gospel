"""calculus.py -- three-valued facets, pair statuses, cells and canonical nests.

A carving over n entities and f facets is an integer matrix M of shape (n, f).
M[x, F] is a bit set: bit v is on iff entity x is witnessed to carry value v
of facet F. M[x, F] == 0 means x is undetermined on F (the value "bottom").
A genotype call is single-valued: exactly one bit, 1 << g, for g in {0,1,2}.

Pair statuses (Definition "pair status" of the manuscript):
  CERT  some facet has both sets non-empty and different;
  OPEN  not certified, but some facet has different sets (one of them empty);
  IND   every facet has equal sets.
"""

from __future__ import annotations

import numpy as np

IND, OPEN, CERT = 0, 1, 2
LETTER_A, LETTER_P, LETTER_M = "a", "+", "-"


def status_matrix(M: np.ndarray) -> np.ndarray:
    """n x n matrix of pair statuses; the diagonal is -1."""
    M = np.asarray(M)
    n = M.shape[0]
    cert = np.zeros((n, n), dtype=bool)
    diff = np.zeros((n, n), dtype=bool)
    for F in range(M.shape[1]):
        a = M[:, F]
        det = a != 0
        d = a[:, None] != a[None, :]
        diff |= d
        cert |= d & det[:, None] & det[None, :]
    S = np.where(cert, CERT, np.where(diff, OPEN, IND)).astype(np.int8)
    np.fill_diagonal(S, -1)
    return S


def status_counts(S: np.ndarray) -> dict:
    iu = np.triu_indices(S.shape[0], 1)
    v = S[iu]
    return {"cert": int((v == CERT).sum()), "open": int((v == OPEN).sum()),
            "ind": int((v == IND).sum()), "pairs": int(v.size)}


def status_of_pair(M: np.ndarray, x: int, y: int) -> int:
    a, b = M[x], M[y]
    d = a != b
    if np.any(d & (a != 0) & (b != 0)):
        return CERT
    if np.any(d):
        return OPEN
    return IND


def certifying_facets(M: np.ndarray, x: int, y: int) -> np.ndarray:
    """Indices of facets that certify the pair {x, y}."""
    a, b = M[x], M[y]
    return np.flatnonzero((a != b) & (a != 0) & (b != 0))


def cells(M: np.ndarray) -> np.ndarray:
    """Cell label of each entity: equal rows <=> same cell."""
    _, lab = np.unique(np.asarray(M), axis=0, return_inverse=True)
    return lab.ravel()


def descriptor(M: np.ndarray, s: int) -> list[tuple[int, int]]:
    """The (facet, value) pairs entity s is witnessed to carry, in tie order."""
    out = []
    for F in range(M.shape[1]):
        m = int(M[s, F])
        v = 0
        while m:
            if m & 1:
                out.append((F, v))
            m >>= 1
            v += 1
    return out


def canonical_nest(M: np.ndarray, s: int, domain: np.ndarray | None = None):
    """Canonical nest of entity s (most general strict refinement first, ties by
    the lexicographic order of (facet, value)).

    Returns (word, cell, n_repetitions), where word is a string over {a,+,-},
    cell is the boolean mask of the final block, and repetitions are the pairs
    of the descriptor that never refined the block strictly.
    """
    M = np.asarray(M)
    n = M.shape[0]
    B = np.ones(n, dtype=bool) if domain is None else domain.copy()
    D = descriptor(M, s)
    if not D:
        return "", B, 0
    facets = np.array([F for F, _ in D])
    bits = np.array([1 << v for _, v in D], dtype=np.int64)
    Ext = (M[:, facets].T & bits[:, None]) != 0          # |D| x n
    remaining = np.ones(len(D), dtype=bool)
    word = []
    while True:
        inter = (Ext & B[None, :]).sum(axis=1)
        size = B.sum()
        strict = remaining & (inter < size)
        if not strict.any():
            break
        cand = np.flatnonzero(strict)
        best = cand[np.argmax(inter[cand])]               # argmax keeps the first: tie order
        F = facets[best]
        R = B & ~Ext[best]
        wit = R & (M[:, F] != 0)
        unw = R & (M[:, F] == 0)
        word.append(LETTER_A if (wit.any() and unw.any()) else
                    (LETTER_P if wit.any() else LETTER_M))
        B = B & Ext[best]
        remaining[best] = False
    return "".join(word), B, int(remaining.sum())


def tolerance_failures(S: np.ndarray) -> tuple[int, int, int]:
    """Count chains x T y, y T z with {x,z} certified (T = 'not certified').

    Returns (chains, failures, failures_without_open_link). The trichotomy
    theorem says the last number is always zero.
    """
    n = S.shape[0]
    T = (S != CERT)
    np.fill_diagonal(T, False)
    chains = fails = bad = 0
    for y in range(n):
        xs = np.flatnonzero(T[y])
        for i in xs:
            for k in xs:
                if i >= k:
                    continue
                chains += 1
                if S[i, k] == CERT:
                    fails += 1
                    if not (S[i, y] == OPEN or S[y, k] == OPEN):
                        bad += 1
    return chains, fails, bad
