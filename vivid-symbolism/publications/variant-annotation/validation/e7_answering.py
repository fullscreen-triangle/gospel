"""E7 -- answers and their grounding profiles, on real annotation.

(a) GO: for every term with at least 20 member genes, the grounding profile of
    the answer to "genes annotated to the term" (direct annotations, positive
    lines). A gene-set file (GMT) returns the same rows with every ground
    erased. NOT-qualified lines are the only witnessed exclusions.
(b) ClinVar: profiles of "variants classified pathogenic in gene X" for the
    30 genes with the most such variants.
(c) Profile freedom, checked with SPARQL on a ClinVar subsample: for random
    verdict assignments a graph with identical content is constructed, content
    queries are evaluated on both, and the verdicts read back.
(d) Directions of error on ClinVar: emission completeness k (each SCV's
    collection method emitted with probability k) and fabrication f (a
    contact-typed SCV added to a variant without contact with probability f):
    claimed against true index share of the pathogenic answer rows.
(e) Verdict information retained by four receivers.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
from rdflib import Graph, Literal
from rdflib.namespace import RDF

from common import rng_for, write_result
from gsv import corpora as K
from gsv import verdicts as V
from gsv.rdfgraphs import EX, PREFIXES, PROV, ind


def entropy(counts):
    c = np.asarray(counts, float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def go_profiles(min_genes=20):
    out = {}
    for org in K.GO_FILES:
        mem, nots, _, _ = K.load_gaf(org)
        by_term = {}
        for (gene, term), evs in mem.items():
            by_term.setdefault(term, []).append(K.verdict(*K.go_ground(evs)))
        rows = []
        for term, vs in by_term.items():
            if len(vs) < min_genes:
                continue
            c = np.bincount(vs, minlength=5)
            rows.append({"term": term, "n": len(vs), "counts": c.tolist()})
        all_v = np.bincount([v for vs in by_term.values() for v in vs], minlength=5)
        genes = {g for g, _ in mem}
        out[org] = {
            "terms": len(rows),
            "profiles": rows,
            "H_membership_verdict_bits": entropy(all_v),
            "membership_verdicts": all_v.tolist(),
            "not_lines": len(nots),
            "absent_memberships": len(genes) * len(by_term) - len(mem),
        }
    return out


def clinvar_gene_profiles(a, genes, top=30):
    sel = (a["agg"] == 0) & (a["gene"] >= 0)
    g = a["gene"][sel]
    vd = a["verdict"][sel]
    cnt = Counter(g.tolist())
    rows = []
    for gi, n in cnt.most_common(top):
        c = np.bincount(vd[g == gi], minlength=5)
        rows.append({"gene": genes[gi], "n": int(n), "counts": c.tolist()})
    all_c = np.bincount(vd, minlength=5)
    return {"pathogenic_variants_with_gene": int(sel.sum()), "overall": all_c.tolist(),
            "top": rows}


def profile_freedom(rng, a, genes, n_var=400, assignments=40):
    """Content: (v ex:classification C), (v ex:inGene G) for real variants.
    Ground: private gadgets realising an arbitrary assignment of verdicts."""
    pick = rng.choice(np.flatnonzero(a["gene"] >= 0), n_var, replace=False)
    content = Graph()
    recs = []
    for i, k in enumerate(pick):
        r = ind(f"v{i}")
        recs.append(r)
        content.add((r, RDF.type, EX.VariantInterpretation))
        content.add((r, EX.classification, Literal(K.AGG_NAMES[int(a["agg"][k])])))
        content.add((r, EX.inGene, EX[f"gene{int(a['gene'][k])}"]))
    queries = {
        "pathogenic": 'SELECT ?v WHERE { ?v ex:classification "P" }',
        "by_gene_count": "SELECT ?g (COUNT(?v) AS ?n) WHERE { ?v ex:inGene ?g } GROUP BY ?g",
        "not_benign": 'SELECT ?v WHERE { ?v a ex:VariantInterpretation FILTER NOT EXISTS { ?v ex:classification "B" } }',
        "conflict_or_vus": 'SELECT ?v WHERE { { ?v ex:classification "C" } UNION { ?v ex:classification "U" } }',
    }
    base = {q: Counter(tuple(r) for r in content.query(PREFIXES + s)) for q, s in queries.items()}
    W = V.Witness(A_c=[EX.ClinicalTesting], A_d=[EX.Curation])
    same = 0
    realised = 0
    tv = []
    for t in range(assignments):
        target = rng.integers(0, 5, n_var)
        g = Graph()
        for tr in content:
            g.add(tr)
        for i, r in enumerate(recs):
            v = int(target[i])
            if v == K.SYMBOL:
                continue
            act = ind(f"t{t}_a{i}")
            g.add((r, V.GEN, act))
            g.add((act, RDF.type, PROV.Activity))
            if v in (K.INDEX, K.COMPOSITE):
                g.add((act, RDF.type, EX.ClinicalTesting))
            if v in (K.ICON, K.COMPOSITE):
                g.add((act, RDF.type, EX.Curation))
        ans = {q: Counter(tuple(r) for r in g.query(PREFIXES + s)) for q, s in queries.items()}
        same += int(ans == base)
        gr = V.ground_sparql(g, recs, W)
        got = np.array([K.verdict(*gr[r]) for r in recs])
        realised += int(np.all(got == target))
        p_rows = [i for i, r in enumerate(recs) if (r,) in base["pathogenic"]]
        prof = np.bincount(got[p_rows], minlength=5) / max(len(p_rows), 1)
        tv.append(prof)
    tvd = [0.5 * np.abs(tv[i] - tv[j]).sum() for i in range(len(tv)) for j in range(i)]
    return {"variants": n_var, "assignments": assignments, "answers_identical": same,
            "assignments_realised": realised, "max_profile_tv": float(np.max(tvd)),
            "median_profile_tv": float(np.median(tvd))}


def directions(rng, z):
    """Simulated emission over the real ClinVar ground of pathogenic variants."""
    a = K.aggregate(z["var"], z["cls"], z["meth"], z["rev"], K.DERIV_MASK_A1)
    order = np.argsort(z["var"], kind="stable")
    v = z["var"][order]
    uniq, start = np.unique(v, return_index=True)
    n_contact = np.add.reduceat(((z["meth"][order] & K.CONTACT_MASK) != 0).astype(np.int64), start)
    P = a["agg"] == 0
    nc = n_contact[P]
    true_c = (nc > 0).astype(float)
    ks = np.round(np.linspace(0, 1, 11), 2)
    fs = np.round(np.linspace(0, 1, 11), 2)
    claimed = np.zeros((len(ks), len(fs)))
    for i, k in enumerate(ks):
        for j, f in enumerate(fs):
            emitted = rng.random(nc.size) < 1 - (1 - k) ** nc
            fabricated = (~emitted) & (true_c == 0) & (rng.random(nc.size) < f)
            claimed[i, j] = float(np.mean(emitted | fabricated))
    return {"pathogenic_variants": int(P.sum()), "true_index_share": float(true_c.mean()),
            "k": ks, "f": fs, "claimed_index_share": claimed}


def main():
    t0 = time.time()
    rng = rng_for(7)
    go = go_profiles()
    z, genes = K.load_clinvar()
    gmiss = np.where(np.array([g == "-" for g in genes])[z["gene"]], -1, z["gene"])
    a = K.aggregate(z["var"], z["cls"], z["meth"], z["rev"], K.DERIV_MASK_A1, gmiss)
    cv = clinvar_gene_profiles(a, genes)
    # verdict information retained by receivers
    vd = a["verdict"]
    key = a["agg"].astype(np.int64) * 16 + (a["review"].astype(np.int64) + 4)
    Hc = 0.0
    for k in np.unique(key):
        sel = key == k
        Hc += sel.mean() * entropy(np.bincount(vd[sel], minlength=5))
    Hcv = entropy(np.bincount(vd, minlength=5))
    receivers = {
        "GO GAF (evidence codes)": [go["human"]["H_membership_verdict_bits"], go["human"]["H_membership_verdict_bits"]],
        "GO gene-set file (GMT)": [go["human"]["H_membership_verdict_bits"], 0.0],
        "ClinVar SCV (collection method)": [Hcv, Hcv],
        "ClinVar VCF fields (CLNSIG, CLNREVSTAT)": [Hcv, Hcv - Hc],
    }
    payload = {"go": go, "clinvar": cv,
               "profile_freedom": profile_freedom(rng, a, genes),
               "directions": directions(rng, z),
               "receivers_bits": {k: {"H": v[0], "I_retained": v[1]} for k, v in receivers.items()}}
    write_result("E7_answering", payload, t0)


if __name__ == "__main__":
    main()
