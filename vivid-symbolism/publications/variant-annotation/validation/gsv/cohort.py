"""cohort.py -- a synthetic diploid cohort with known truth, and four emissions.

Truth. K founder haplotypes over m biallelic sites, each site's alternative
allele drawn with a site frequency p ~ U(0.05, 0.5). Each sample carries two
haplotypes, each a mosaic of founders that switches founder at every site with
probability `switch`. The true genotype is the alternative-allele count, 0/1/2.

Coverage. Each (sample, site) is covered independently with probability c. A
covered call is reported correctly (the calculus is tested on truthful
emission; imputation is the only source of error).

Emissions (bit-set matrices, see calculus.py):
  V  variant-only: merged single-sample VCFs. A call is emitted iff covered
     and non-reference. Reference and no-coverage are both absent.
  J  joint-called: a site is emitted iff some covered sample is non-reference;
     at an emitted site every covered sample receives its call, 0/0 included,
     and every uncovered sample receives ./. (absent).
  G  reference blocks (gVCF): every covered call is emitted.
  I  G, with every absent call imputed from the most concordant sample that is
     covered at the site.
"""

from __future__ import annotations

import numpy as np

REGIMES = ("V", "J", "G", "I")


def simulate(rng, n=40, m=200, K=8, switch=0.02):
    p = rng.uniform(0.05, 0.5, size=m)
    founders = (rng.random((K, m)) < p[None, :]).astype(np.int8)

    def mosaic():
        out = np.empty(m, dtype=np.int8)
        f = rng.integers(K)
        for j in range(m):
            if rng.random() < switch:
                f = rng.integers(K)
            out[j] = founders[f, j]
        return out

    G = np.array([mosaic() + mosaic() for _ in range(n)], dtype=np.int8)
    return G


def bits(g: np.ndarray) -> np.ndarray:
    return (np.int64(1) << g.astype(np.int64))


def emit(G: np.ndarray, covered: np.ndarray, regime: str, rng=None):
    """Return (M, imputed_mask). imputed_mask marks calls that are derived."""
    n, m = G.shape
    B = bits(G)
    imputed = np.zeros((n, m), dtype=bool)
    if regime == "V":
        M = np.where(covered & (G > 0), B, 0)
    elif regime == "J":
        site = (covered & (G > 0)).any(axis=0)
        M = np.where(covered & site[None, :], B, 0)
    elif regime in ("G", "I"):
        M = np.where(covered, B, 0)
        if regime == "I":
            M, imputed = impute(G, covered)
    else:
        raise ValueError(regime)
    return M.astype(np.int64), imputed


def impute(G: np.ndarray, covered: np.ndarray):
    """Nearest-neighbour imputation: fill each uncovered call with the call of
    the sample most concordant with it over jointly covered sites, among the
    samples covered at that site. The imputed value is a derivation: it is
    computed from other samples' calls, not read from this sample's reads."""
    n, m = G.shape
    obs = np.where(covered, G, -1)
    conc = np.zeros((n, n))
    for x in range(n):
        both = covered[x][None, :] & covered
        agree = (obs[x][None, :] == obs) & both
        conc[x] = agree.sum(1) / np.maximum(both.sum(1), 1)
    np.fill_diagonal(conc, -1.0)
    order = np.argsort(-conc, axis=1, kind="stable")
    M = np.where(covered, bits(G), 0).astype(np.int64)
    imputed = np.zeros((n, m), dtype=bool)
    for x in range(n):
        for j in np.flatnonzero(~covered[x]):
            for y in order[x]:
                if covered[y, j]:
                    M[x, j] = np.int64(1) << np.int64(G[y, j])
                    imputed[x, j] = True
                    break
    return M, imputed
