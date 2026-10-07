"""E1 -- genotype calls as three-valued distinctions.

Tests: the trichotomy and its tolerance law (exhaustive); the equivalence of the
two characterisations of the status (exhaustive); the inclusion of certified
sets along the emission chain V <= J <= G <= I; permanence of certification on
single-valued facets under growth and its failure on multi-valued ones; the
letters of sample words under each emission; witnessed against undetermined
absence of a non-reference call; non-locality of words; and the false
certificates that imputation introduces.
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from common import rng_for, write_result
from gsv import calculus as C
from gsv import cohort as H


# ---------------------------------------------------------------- exhaustive

def status_by_distinctions(M, x, y):
    """Status read off the value distinctions chi_{F,v} (Lemma 'statuses read
    off the distinctions'): certified iff some distinction is + on one entity
    and - on the other; open iff no such distinction but some differs."""
    any_diff = False
    for F in range(M.shape[1]):
        a, b = int(M[x, F]), int(M[y, F])
        for v in range(3):
            def chi(s):
                if s == 0:
                    return "bot"
                return "+" if (s >> v) & 1 else "-"
            ca, cb = chi(a), chi(b)
            if {ca, cb} == {"+", "-"}:
                return C.CERT
            if ca != cb:
                any_diff = True
    return C.OPEN if any_diff else C.IND


def exhaustive():
    configs = [(3, 1), (3, 2), (4, 1), (4, 2)]
    out = []
    for n, f in configs:
        tot = {"cert": 0, "open": 0, "ind": 0}
        chains = fails = bad = 0
        disagree = 0
        values = [0, 1, 2, 4]          # bottom, 0/0, 0/1, 1/1 as bit sets
        for cfg in itertools.product(values, repeat=n * f):
            M = np.array(cfg, dtype=np.int64).reshape(n, f)
            S = C.status_matrix(M)
            cnt = C.status_counts(S)
            for k in tot:
                tot[k] += cnt[k]
            for x, y in itertools.combinations(range(n), 2):
                if status_by_distinctions(M, x, y) != S[x, y]:
                    disagree += 1
            ch, fa, b = C.tolerance_failures(S)
            chains += ch; fails += fa; bad += b
        out.append({"n": n, "f": f, "carvings": 4 ** (n * f), **tot,
                    "characterisation_disagreements": disagree,
                    "tolerance_chains": chains, "tolerance_failures": fails,
                    "failures_without_open_link": bad})
    return out


# ------------------------------------------------------------ regimes

def regime_run(rng, n, m, c, switch=0.02):
    G = H.simulate(rng, n=n, m=m, switch=switch)
    cov = rng.random(G.shape) < c
    truth = C.status_matrix(H.bits(G).astype(np.int64))
    res = {}
    cert_sets = {}
    for r in H.REGIMES:
        M, imp = H.emit(G, cov, r)
        S = C.status_matrix(M)
        cnt = C.status_counts(S)
        iu = np.triu_indices(n, 1)
        cert = S[iu] == C.CERT
        cert_sets[r] = cert
        # certificates: (pair, site) with both values present and different
        false_cert = 0
        total_cert = 0
        icon_only = 0
        icon_cert = 0
        for x, y in zip(*iu):
            fac = C.certifying_facets(M, x, y)
            total_cert += fac.size
            false_cert += int((G[x, fac] == G[y, fac]).sum())
            derived = imp[x, fac] | imp[y, fac]
            icon_cert += int(derived.sum())
            if fac.size and np.all(derived):
                icon_only += 1
        truly_distinct = truth[iu] == C.CERT
        false_ind = int(((S[iu] == C.IND) & truly_distinct).sum())
        # letters of every sample's word
        letters = {"a": 0, "+": 0, "-": 0}
        depth = []
        for s in range(n):
            w, _, _ = C.canonical_nest(M, s)
            depth.append(len(w))
            for ch in w:
                letters[ch] += 1
        # absence of a non-reference call: witnessed reference vs undetermined
        no_alt = (M & 6) == 0            # neither 0/1 (bit 2) nor 1/1 (bit 4)
        witnessed_ref = int((no_alt & (M == 1)).sum())
        undetermined = int((no_alt & (M == 0)).sum())
        res[r] = {**cnt, "certificates": total_cert, "false_certificates": false_cert,
                  "icon_certificates": icon_cert,
                  "icon_only_certified_pairs": icon_only,
                  "false_indiscernible": false_ind,
                  "truly_distinct": int(truly_distinct.sum()),
                  "letters": letters, "mean_depth": float(np.mean(depth)),
                  "absent_alt_witnessed_ref": witnessed_ref,
                  "absent_alt_undetermined": undetermined}
    viol = 0
    chain = list(H.REGIMES)
    for a, b in zip(chain, chain[1:]):
        viol += int((cert_sets[a] & ~cert_sets[b]).sum())
    return res, viol


# ------------------------------------------------------------ panel size

def sites_sweep(rng, n=40, c=0.8, reps=8):
    sizes = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    out = {r: {"cert": [], "open": [], "ind": []} for r in H.REGIMES}
    out["truly_distinct"] = []
    for m in sizes:
        acc = {r: {"cert": [], "open": [], "ind": []} for r in H.REGIMES}
        td = []
        for _ in range(reps):
            res, _ = regime_run(rng, n, m, c)
            for r in H.REGIMES:
                for k in ("cert", "open", "ind"):
                    acc[r][k].append(res[r][k] / res[r]["pairs"])
            td.append(res["G"]["truly_distinct"] / res["G"]["pairs"])
        for r in H.REGIMES:
            for k in ("cert", "open", "ind"):
                out[r][k].append(float(np.mean(acc[r][k])))
        out["truly_distinct"].append(float(np.mean(td)))
    out["sites"] = sizes
    return out


# ------------------------------------------------------------ growth

def growth(rng, n=20, m=8, steps=160, multi=False, err=0.2):
    G = H.simulate(rng, n=n, m=m, switch=0.05)
    M = np.zeros((n, m), dtype=np.int64)
    lost = []
    prev = C.status_matrix(M) == C.CERT
    for _ in range(steps):
        x, j = rng.integers(n), rng.integers(m)
        if multi:
            g = G[x, j] if rng.random() > err else rng.choice([k for k in range(3) if k != G[x, j]])
            M[x, j] |= np.int64(1) << np.int64(g)
        else:
            if M[x, j] == 0:
                M[x, j] = np.int64(1) << np.int64(G[x, j])
        cur = C.status_matrix(M) == C.CERT
        lost.append(int((prev & ~cur).sum()) // 2)
        prev = cur
    return lost


# ------------------------------------------------------------ non-locality

def nonlocality(rng, trials=600, n=12, m=24, c=0.6):
    changed_word = changed_ind = tested = 0
    for _ in range(trials):
        G = H.simulate(rng, n=n, m=m, switch=0.05)
        cov = rng.random(G.shape) < c
        M, _ = H.emit(G, cov, "G")
        x = rng.integers(n)
        cand = [(y, j) for y in range(n) for j in range(m) if y != x and M[y, j] == 0]
        if not cand:
            continue
        y, j = cand[rng.integers(len(cand))]
        w0, _, _ = C.canonical_nest(M, x)
        ind0 = np.all(C.status_matrix(M)[x, np.arange(n) != x] == C.CERT)
        M2 = M.copy()
        M2[y, j] = np.int64(1) << np.int64(G[y, j])
        w1, _, _ = C.canonical_nest(M2, x)
        ind1 = np.all(C.status_matrix(M2)[x, np.arange(n) != x] == C.CERT)
        tested += 1
        changed_word += int(w0 != w1)
        changed_ind += int(ind0 != ind1)
    return {"tested": tested, "word_changed": changed_word,
            "individuation_changed": changed_ind,
            "note": "x's own calls are identical before and after"}


# ------------------------------------------------------------ imputation sweep

def imputation_sweep(rng, n=30, m=60, reps=3):
    covs = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    switches = [0.005, 0.01, 0.02, 0.05, 0.1]
    false_rate = np.zeros((len(switches), len(covs)))
    icon_share = np.zeros_like(false_rate)
    icon_cert_share = np.zeros_like(false_rate)
    open_G = np.zeros_like(false_rate)
    for a, sw in enumerate(switches):
        for b, c in enumerate(covs):
            fr, ic, og, icc = [], [], [], []
            for _ in range(reps):
                r, _ = regime_run(rng, n, m, c, switch=sw)
                I = r["I"]
                fr.append(I["false_certificates"] / max(I["certificates"], 1))
                ic.append(I["icon_only_certified_pairs"] / max(I["cert"], 1))
                og.append(r["G"]["open"] / r["G"]["pairs"])
                icc.append(I["icon_certificates"] / max(I["certificates"], 1))
            false_rate[a, b] = np.mean(fr)
            icon_share[a, b] = np.mean(ic)
            open_G[a, b] = np.mean(og)
            icon_cert_share[a, b] = np.mean(icc)
    return {"coverage": covs, "switch": switches, "false_certificate_rate": false_rate,
            "icon_only_share_of_certified": icon_share,
            "icon_certificate_share": icon_cert_share, "open_fraction_G": open_G}


def main():
    t0 = time.time()
    rng = rng_for(1)
    ex = exhaustive()
    reps = 10
    n, m, c = 40, 60, 0.8
    per = {r: [] for r in H.REGIMES}
    viols = 0
    for _ in range(reps):
        res, v = regime_run(rng, n, m, c)
        viols += v
        for r in H.REGIMES:
            per[r].append(res[r])
    agg = {}
    for r in H.REGIMES:
        rows = per[r]
        agg[r] = {k: float(np.mean([row[k] for row in rows]))
                  for k in rows[0] if not isinstance(rows[0][k], dict)}
        agg[r]["letters"] = {L: int(sum(row["letters"][L] for row in rows)) for L in "a+-"}
        agg[r]["false_certificates_total"] = int(sum(row["false_certificates"] for row in rows))
    gseq = 200
    lost_single = np.array([growth(rng, multi=False) for _ in range(gseq)])
    lost_multi = np.array([growth(rng, multi=True) for _ in range(gseq)])
    payload = {
        "exhaustive": ex,
        "regimes": {"n": n, "m": m, "coverage": c, "replicates": reps,
                    "order": list(H.REGIMES), "summary": agg,
                    "inclusion_violations": viols},
        "growth": {"sequences": gseq, "n": 20, "m": 8,
                   "mean_lost_single": lost_single.mean(0),
                   "mean_lost_multi": lost_multi.mean(0),
                   "total_lost_single": int(lost_single.sum()),
                   "total_lost_multi": int(lost_multi.sum())},
        "sites_sweep": sites_sweep(rng),
        "nonlocality": nonlocality(rng),
        "imputation": imputation_sweep(rng),
    }
    write_result("E1_calls", payload, t0)


if __name__ == "__main__":
    main()
