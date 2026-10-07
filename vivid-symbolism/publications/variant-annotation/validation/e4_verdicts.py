"""E4 -- semiotic verdicts on real genomic annotation.

(a) Census of verdicts over GO gene--term memberships (human, yeast,
    Arabidopsis) and ClinVar variants, under the declared witness classes.
(b) Definability: the fixed-point ground equals the ground read by three
    SPARQL property-path queries, on synthetic provenance graphs with chains
    and on RDF emissions of real GO and ClinVar records.
(c) Monotonicity: verdict transitions while synthetic graphs grow triple by
    triple, and while ClinVar is ingested in order of last evaluation; content
    (the aggregate classification) is followed through the same ingestion.
(d) Content independence on real data: for variants sharing gene and
    aggregate classification, which ordered verdict pairs are realised.
(e) A VCF-style receiver for ClinVar: does (classification, review status)
    determine the verdict?
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
from rdflib import Graph
from rdflib.namespace import RDF

from common import rng_for, write_result
from gsv import corpora as K
from gsv import verdicts as V
from gsv.rdfgraphs import EX, ind

GEN = V.GEN


# ------------------------------------------------------------ (a) census

def go_census():
    out = {}
    mems = {}
    for org in K.GO_FILES:
        mem, nots, evc, header = K.load_gaf(org)
        verd = Counter(K.verdict(*K.go_ground(evs)) for evs in mem.values())
        line_v = Counter()
        for ev, n in evc.items():
            line_v[K.verdict(*K.go_ground({ev}))] += n
        out[org] = {"memberships": len(mem), "annotation_lines": int(sum(evc.values())),
                    "not_lines": len(nots),
                    "membership_verdicts": {K.VERDICTS[k]: verd.get(k, 0) for k in range(5)},
                    "line_verdicts": {K.VERDICTS[k]: line_v.get(k, 0) for k in range(5)},
                    "evidence_counts": evc, "header": header,
                    "genes": len({g for g, _ in mem}), "terms": len({t for _, t in mem})}
        mems[org] = mem
    return out, mems


def clinvar_census(z, genes):
    gmiss = np.where(np.array([g == "-" for g in genes])[z["gene"]], -1, z["gene"])
    res = {}
    aggs = {}
    for name, dm in (("A1", K.DERIV_MASK_A1), ("A2", K.DERIV_MASK_A2)):
        a = K.aggregate(z["var"], z["cls"], z["meth"], z["rev"], dm, gmiss)
        aggs[name] = a
        vc = Counter(a["verdict"].tolist())
        cross = {}
        for code, nm in K.AGG_NAMES.items():
            sel = a["agg"] == code
            c = Counter(a["verdict"][sel].tolist())
            cross[nm] = {K.VERDICTS[k]: c.get(k, 0) for k in range(5)}
        res[name] = {"variants": int(a["var"].size),
                     "verdicts": {K.VERDICTS[k]: vc.get(k, 0) for k in range(5)},
                     "by_class": cross}
    meth = Counter()
    for m, bit in K.CV_BIT.items():
        meth[m] = int(((z["meth"] & bit) != 0).sum())
    res["scv"] = int(z["var"].size)
    res["method_counts"] = dict(meth)
    return res, aggs


# ------------------------------------------------------------ (b) definability

GO_W = V.Witness(A_c=[EX[f"ev_{e}"] for e in K.GO_CONTACT],
                 A_d=[EX[f"ev_{e}"] for e in K.GO_DERIV])
CV_W = V.Witness(A_c=[EX["cm_" + m.replace(" ", "_")] for m in K.CV_CONTACT],
                 A_d=[EX["cm_" + m.replace(" ", "_")] for m in K.CV_DERIV_A1])


def emit_go(mem: dict, keys):
    g = Graph()
    recs = []
    for i, key in enumerate(keys):
        r = ind(f"m{i}")
        recs.append(r)
        g.add((r, RDF.type, EX.GeneSetMembership))
        for j, ev in enumerate(sorted(mem[key])):
            a = ind(f"m{i}_a{j}")
            g.add((r, GEN, a))
            g.add((a, RDF.type, EX[f"ev_{ev}"]))
    return g, recs


def emit_clinvar(z, rows_by_var, vids):
    g = Graph()
    recs = []
    for i, v in enumerate(vids):
        r = ind(f"v{i}")
        recs.append(r)
        g.add((r, RDF.type, EX.VariantInterpretation))
        for j, row in enumerate(rows_by_var[v]):
            a = ind(f"v{i}_s{j}")
            g.add((r, GEN, a))
            for m, bit in K.CV_BIT.items():
                if z["meth"][row] & bit:
                    g.add((a, RDF.type, EX["cm_" + m.replace(" ", "_")]))
    return g, recs


def definability(rng, mems, z):
    out = {}
    # synthetic, with chains
    agree = total = 0
    dist = Counter()
    for _ in range(300):
        recs, T = V.random_provenance(rng)
        g = V.graph_of(T)
        fp = V.ground_fixed_point(g, recs, V.SYN)
        sp = V.ground_sparql(g, recs, V.SYN)
        for r in recs:
            total += 1
            agree += int(fp[r] == sp[r])
            dist[K.verdict(*fp[r])] += 1
    out["synthetic"] = {"records": total, "agree": agree,
                        "verdicts": {K.VERDICTS[k]: dist.get(k, 0) for k in range(5)}}
    # real GO (human) sample
    keys = list(mems["human"])
    pick = [keys[i] for i in rng.choice(len(keys), 1500, replace=False)]
    g, recs = emit_go(mems["human"], pick)
    fp, sp = V.ground_fixed_point(g, recs, GO_W), V.ground_sparql(g, recs, GO_W)
    direct = [K.go_ground(mems["human"][k]) for k in pick]
    out["go_human"] = {"records": len(recs),
                       "agree_sparql": sum(fp[r] == sp[r] for r in recs),
                       "agree_direct": sum(fp[r] == d for r, d in zip(recs, direct))}
    # real ClinVar sample
    uv = np.unique(z["var"])
    vids = set(rng.choice(uv, 1500, replace=False).tolist())
    rows = {}
    for i in np.flatnonzero(np.isin(z["var"], list(vids))):
        rows.setdefault(int(z["var"][i]), []).append(i)
    vids = sorted(rows)
    g, recs = emit_clinvar(z, rows, vids)
    fp, sp = V.ground_fixed_point(g, recs, CV_W), V.ground_sparql(g, recs, CV_W)
    direct = []
    for v in vids:
        mm = np.bitwise_or.reduce(z["meth"][rows[v]])
        direct.append((1, int(bool(mm & K.CONTACT_MASK)), int(bool(mm & K.DERIV_MASK_A1))))
    out["clinvar"] = {"records": len(recs),
                      "agree_sparql": sum(fp[r] == sp[r] for r in recs),
                      "agree_direct": sum(fp[r] == d for r, d in zip(recs, direct))}
    return out


# ------------------------------------------------------------ (c) monotonicity

def upward(old, new):
    return new[0] >= old[0] and new[1] >= old[1] and new[2] >= old[2]


def synthetic_growth(rng, n_graphs=300):
    trans = Counter()
    retract = 0
    steps = 0
    for _ in range(n_graphs):
        recs, T = V.random_provenance(rng)
        order = rng.permutation(len(T))
        g = Graph()
        prev = {r: (0, 0, 0) for r in recs}
        for k in order:
            g.add(T[k])
            cur = V.ground_fixed_point(g, recs, V.SYN)
            steps += 1
            for r in recs:
                if cur[r] != prev[r]:
                    trans[(K.verdict(*prev[r]), K.verdict(*cur[r]))] += 1
                    retract += int(not upward(prev[r], cur[r]))
            prev = cur
    return {"graphs": n_graphs, "additions": steps, "retractions": retract,
            "transitions": {f"{K.VERDICTS[a]}->{K.VERDICTS[b]}": n for (a, b), n in trans.items()}}


def clinvar_ingestion(z):
    """Ingest SCVs in order of DateLastEvaluated (undated last). At yearly
    checkpoints compare each variant's ground and aggregate class with the
    previous checkpoint."""
    years = list(range(2005, 2027))
    date = z["date"]
    checkpoints = [(y, y * 10000 + 1231) for y in years] + [("all", None)]
    prev = None
    rows = []
    for label, cut in checkpoints:
        sel = np.ones(date.size, bool) if cut is None else ((date >= 0) & (date <= cut))
        a = K.aggregate(z["var"][sel], z["cls"][sel], z["meth"][sel], z["rev"][sel],
                        K.DERIV_MASK_A1)
        row = {"checkpoint": label, "variants": int(a["var"].size), "scv": int(sel.sum())}
        if prev is not None:
            idx = np.searchsorted(a["var"], prev["var"])
            # every earlier variant persists: SCVs only accumulate along the ingestion
            c0, d0, c1, d1 = prev["c"], prev["d"], a["c"][idx], a["d"][idx]
            changed = (c0 != c1) | (d0 != d1)
            row["verdict_changes"] = int(changed.sum())
            row["verdict_retractions"] = int(((c1 < c0) | (d1 < d0)).sum())
            g0, g1 = prev["agg"], a["agg"][idx]
            row["class_changes"] = int((g0 != g1).sum())
            row["pathogenic_retracted"] = int(((g0 == 0) & (g1 != 0)).sum())
            row["class_transitions"] = {f"{K.AGG_NAMES[int(x)]}->{K.AGG_NAMES[int(y)]}": int(n)
                                        for (x, y), n in Counter(zip(g0[g0 != g1].tolist(),
                                                                     g1[g0 != g1].tolist())).items()}
        rows.append(row)
        prev = {"var": a["var"], "c": a["c"], "d": a["d"], "agg": a["agg"]}
    return rows


# ------------------------------------------------------------ (d), (e)

def content_independence(a):
    """Twins = variants with equal (gene, aggregate class). For each ordered
    pair of verdicts, the number of twin groups that contain both."""
    ok = a["gene"] >= 0
    key = a["gene"][ok].astype(np.int64) * 8 + a["agg"][ok]
    vd = a["verdict"][ok]
    order = np.argsort(key, kind="stable")
    key, vd = key[order], vd[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    present = np.zeros((starts.size, 5), bool)
    seg = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, key.size]))
    present[seg, vd] = True
    mat = np.zeros((5, 5), int)
    for i in range(1, 5):
        for j in range(1, 5):
            mat[i, j] = int((present[:, i] & present[:, j]).sum()) if i != j else int(present[:, i].sum())
    return {"groups": int(starts.size), "pair_matrix": mat[1:, 1:].tolist(),
            "order": K.VERDICTS[1:]}


def vcf_receiver(a):
    """Emit (aggregate class, review rank) only, as a VCF INFO field does, and
    ask how much of the verdict survives."""
    key = a["agg"].astype(np.int64) * 16 + (a["review"].astype(np.int64) + 4)
    vd = a["verdict"]
    classes = {}
    for k in np.unique(key):
        c = np.bincount(vd[key == k], minlength=5)
        classes[int(k)] = c
    tot = np.bincount(vd, minlength=5)
    p = tot / tot.sum()
    H = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    Hc = 0.0
    mixed = 0
    for c in classes.values():
        q = c / c.sum()
        Hc += c.sum() / vd.size * -np.sum(q[q > 0] * np.log2(q[q > 0]))
        if (c > 0).sum() > 1:
            mixed += c.sum()
    return {"emitted_classes": len(classes), "H_verdict_bits": float(H),
            "H_verdict_given_emitted_bits": float(Hc),
            "share_variants_in_mixed_classes": mixed / vd.size,
            "classes": {f"{K.AGG_NAMES[k // 16]}|rev{k % 16 - 4}": c.tolist()
                        for k, c in classes.items()}}


def main():
    t0 = time.time()
    rng = rng_for(4)
    go, mems = go_census()
    z, genes = K.load_clinvar()
    cv, aggs = clinvar_census(z, genes)
    payload = {
        "go": go, "clinvar": cv,
        "definability": definability(rng, mems, z),
        "monotonicity": {"synthetic": synthetic_growth(rng), "clinvar": clinvar_ingestion(z)},
        "content_independence": content_independence(aggs["A1"]),
        "vcf_receiver": vcf_receiver(aggs["A1"]),
        "declarations": {"go_contact": sorted(K.GO_CONTACT), "go_derivation": sorted(K.GO_DERIV),
                         "go_neither": sorted(K.GO_NEITHER),
                         "clinvar_contact": sorted(K.CV_CONTACT),
                         "clinvar_derivation_A1": sorted(K.CV_DERIV_A1),
                         "clinvar_derivation_A2": sorted(K.CV_DERIV_A2)},
    }
    write_result("E4_verdicts", payload, t0)


if __name__ == "__main__":
    main()
