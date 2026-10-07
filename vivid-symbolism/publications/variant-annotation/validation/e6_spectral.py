"""E6 -- identity at a declared floor: three receivers of a sequence.

(a) Exhaustive fibres of the magnitude receiver for L = 4..10 (every sequence
    over ACGT except the four homopolymers): every fibre contains the dihedral
    orbit of its members (rotations and rotated reversals); the measured
    question is how often it contains more (homometric sequences). Truncating
    to K < floor(L/2) bins coarsens the fibres further.
(b) Exhaustive pairs at L = 6: the phase receiver has visibility 1 only on the
    diagonal (identity) -- its fibres are singletons.
(c) Strand receiver: invariant under reverse complement; magnitude receiver
    is not.
(d) Corroboration at L = 300, K = 12 (the shader embedding): measured twins
    (substitution rate mu), rotations, reversals, reverse complements and
    unrelated sequences, against a declared threshold tau, under the three
    receivers.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict

import numpy as np

from common import rng_for, write_result
from gsv import spectral as Sp


def fibres(L: int, K=None):
    S = Sp.all_sequences(L)
    nonconst = ~np.all(S == S[:, :1], axis=1)
    S = S[nonconst]
    E = Sp.magnitude(S, K)
    key = Sp.keys(E)
    fib = defaultdict(list)
    for i, k in enumerate(key):
        fib[k].append(i)
    codes = Sp.encode(S)
    pos = np.full(4 ** L, -1, np.int64)
    pos[codes] = np.arange(len(S))
    D = Sp.dihedral_codes(S)                     # (n, 2L) codes of dihedral images
    Ds = np.sort(D, axis=1)
    orbit_size = 1 + (np.diff(Ds, axis=1) != 0).sum(axis=1)
    orbits = np.unique(Ds[:, 0]).size            # least code identifies the orbit
    # invariance: every dihedral image lies in the fibre of the sequence
    viol = int((key[pos[D]] != key[:, None]).sum()) if K is None else 0
    sizes = np.array([len(fib[k]) for k in key])
    strict = sizes > orbit_size                  # fibre strictly larger than the orbit
    fibre_sizes = Counter(len(v) for v in fib.values())
    return {"L": L, "K": K, "sequences": int(len(S)), "fibres": len(fib),
            "dihedral_orbits": int(orbits),
            "invariance_violations": int(viol),
            "share_in_homometric_fibres": float(strict.mean()),
            "max_fibre": int(sizes.max()), "mean_fibre_per_sequence": float(sizes.mean()),
            "mean_orbit_per_sequence": float(orbit_size.mean()),
            "fibre_size_histogram": {str(k): v for k, v in sorted(fibre_sizes.items())}}


def phase_exhaustive(L=6):
    S = Sp.all_sequences(L)
    S = S[~np.all(S == S[:, :1], axis=1)]
    P = Sp.phase_image(S)
    G = P @ P.T
    off = ~np.eye(len(S), dtype=bool)
    return {"L": L, "sequences": int(len(S)), "pairs": int(off.sum() // 2),
            "offdiagonal_at_one": int((np.abs(G[off] - 1) < 1e-12).sum() // 2),
            "max_offdiagonal": float(G[off].max())}


def corroboration(rng, L=300, K=12, n=400):
    taus = np.logspace(-4, 0, 25)
    mus = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    kinds = ["twin_0.05", "rotation", "reversal", "revcomp", "unrelated"]
    vis = {r: {k: [] for k in kinds} for r in ("magnitude", "strand", "phase")}
    twin_mu = {mu: {"magnitude": [], "phase": []} for mu in mus}
    for _ in range(n):
        s = rng.integers(0, 4, L)
        imgs = {"twin_0.05": Sp.mutate(rng, s, 0.05),
                "rotation": Sp.rotate(s, int(rng.integers(1, L))),
                "reversal": Sp.reverse(s), "revcomp": Sp.revcomp(s),
                "unrelated": rng.integers(0, 4, L)}
        e = {"magnitude": Sp.magnitude(s, K), "strand": Sp.strand(s, K), "phase": Sp.phase_image(s)}
        for k, t in imgs.items():
            f = {"magnitude": Sp.magnitude(t, K), "strand": Sp.strand(t, K),
                 "phase": Sp.phase_image(t)}
            for r in vis:
                vis[r][k].append(float(Sp.visibility(e[r], f[r])))
        for mu in mus:
            t = Sp.mutate(rng, s, mu)
            twin_mu[mu]["magnitude"].append(float(Sp.visibility(e["magnitude"], Sp.magnitude(t, K))))
            twin_mu[mu]["phase"].append(float(Sp.visibility(e["phase"], Sp.phase_image(t))))
    rate = {r: {k: [float(np.mean(np.array(v) >= 1 - tau)) for tau in taus]
                for k, v in vis[r].items()} for r in vis}
    surface = {r: [[float(np.mean(np.array(twin_mu[mu][r]) >= 1 - tau)) for tau in taus]
                   for mu in mus] for r in ("magnitude", "phase")}
    summary = {r: {k: {"min": float(np.min(v)), "median": float(np.median(v)),
                       "max": float(np.max(v))} for k, v in vis[r].items()} for r in vis}
    return {"L": L, "K": K, "n": n, "tau": taus, "mu": mus, "rates": rate,
            "twin_surface": surface, "summary": summary,
            "visibility_samples": {r: {k: v[:400] for k, v in vis[r].items()} for r in vis}}


def main():
    t0 = time.time()
    rng = rng_for(6)
    exact = [fibres(L) for L in range(4, 11)]
    trunc = [fibres(10, K) for K in (1, 2, 3, 4)]
    payload = {"fibres_full": exact, "fibres_truncated_L10": trunc,
               "phase_exhaustive": phase_exhaustive(6),
               "corroboration": corroboration(rng)}
    write_result("E6_spectral", payload, t0)


if __name__ == "__main__":
    main()
