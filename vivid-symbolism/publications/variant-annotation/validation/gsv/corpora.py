"""corpora.py -- real genomic annotation corpora and their witness declarations.

GO annotations (GAF 2.2; human, budding yeast, Arabidopsis). A record is a
gene--term membership (gene, GO term) asserted by at least one positive
annotation line; each annotation line is a generating activity typed by its
evidence code. NOT-qualified lines are witnessed exclusions, kept separately.

ClinVar submissions (submission_summary.txt.gz). A record is a variant
(VariationID); each submitted record (SCV) is a generating activity typed by its
collection method(s). Content is the classification.

The witness declarations below are inputs of the method, not facts derived
from the data; E4 reports a second declaration for ClinVar to show the effect.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict

import numpy as np

from common import DERIVED, RAW

# ------------------------------------------------------------------ GO

GO_CONTACT = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"}
GO_DERIV = {"IEA", "ISS", "ISO", "ISA", "ISM", "IGC", "IBA", "IBD", "IKR", "IRD", "RCA"}
GO_NEITHER = {"TAS", "NAS", "IC", "ND"}
GO_FILES = {"human": "goa_human.gaf.gz", "yeast": "sgd.gaf.gz", "arabidopsis": "tair.gaf.gz"}

# verdict codes; ground (g, c, d) -> name
SYMBOL, COLLAPSED, INDEX, ICON, COMPOSITE = 0, 1, 2, 3, 4
VERDICTS = ["symbol", "collapsed", "index", "icon", "composite"]


def verdict(g: int, c: int, d: int) -> int:
    if not g:
        return SYMBOL
    return {(0, 0): COLLAPSED, (1, 0): INDEX, (0, 1): ICON, (1, 1): COMPOSITE}[(c, d)]


def load_gaf(name: str):
    """Return (memberships, not_lines, line_evidence_counts).
    memberships: dict (gene, term) -> set of evidence codes (positive lines)."""
    mem = defaultdict(set)
    nots = set()
    ev_count = defaultdict(int)
    header = []
    with gzip.open(RAW / GO_FILES[name], "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!"):
                if line.startswith("!date-generated"):
                    header.append(line.strip())
                continue
            c = line.rstrip("\n").split("\t")
            gene = c[0] + ":" + c[1]
            term = c[4]
            ev = c[6]
            if "NOT" in c[3].split("|"):
                nots.add((gene, term))
                continue
            mem[(gene, term)].add(ev)
            ev_count[ev] += 1
    return mem, nots, dict(ev_count), header


def go_ground(evs: set) -> tuple[int, int, int]:
    return 1, int(bool(evs & GO_CONTACT)), int(bool(evs & GO_DERIV))


# ------------------------------------------------------------------ ClinVar

CV_METHODS = ["clinical testing", "research", "literature only", "curation", "not provided",
              "phenotyping only", "reference population", "provider interpretation",
              "case-control", "in vitro", "in vivo", "other"]
CV_BIT = {m: 1 << i for i, m in enumerate(CV_METHODS)}
CV_CONTACT = {"clinical testing", "research", "phenotyping only", "reference population",
              "case-control", "in vitro", "in vivo"}
CV_DERIV_A1 = {"curation", "literature only", "provider interpretation"}
CV_DERIV_A2 = {"curation", "provider interpretation"}      # literature only -> neither

CLASSES = ["P", "U", "B", "O", "N"]         # pathogenic, uncertain, benign, other, none
SIG_MAP = {
    "Pathogenic": "P", "Likely pathogenic": "P", "Pathogenic/Likely pathogenic": "P",
    "Pathogenic, low penetrance": "P", "Likely pathogenic, low penetrance": "P",
    "Benign": "B", "Likely benign": "B", "Benign/Likely benign": "B",
    "Uncertain significance": "U", "VUS-high": "U", "VUS-mid": "U", "VUS-low": "U",
    "not provided": "N", "-": "N",
}
REVIEW_RANK = {"practice guideline": 4, "reviewed by expert panel": 3,
               "criteria provided, single submitter": 1, "no assertion criteria provided": 0,
               "no classification provided": -1, "flagged submission": -2}
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                                      "Sep", "Oct", "Nov", "Dec"], 1)}


def mask(methods: set) -> int:
    return sum(CV_BIT[m] for m in methods)


CONTACT_MASK = mask(CV_CONTACT)
DERIV_MASK_A1 = mask(CV_DERIV_A1)
DERIV_MASK_A2 = mask(CV_DERIV_A2)


def build_clinvar_cache():
    """Parse submission_summary once into compact arrays (data/derived)."""
    out = DERIVED / "clinvar_scv.npz"
    if out.exists():
        return out
    var, cls, date, meth, rev, gene = [], [], [], [], [], []
    genes = {}
    with gzip.open(RAW / "submission_summary.txt.gz", "rt", encoding="utf-8",
                   errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            var.append(int(c[0]))
            cls.append(CLASSES.index(SIG_MAP.get(c[1], "O")))
            d = c[2]
            if d == "-" or len(d) < 8:
                date.append(-1)
            else:
                mon, rest = d.split(" ", 1)
                day, year = rest.split(", ")
                date.append(int(year) * 10000 + MONTHS.get(mon, 1) * 100 + int(day))
            atoms = {a.strip() for a in c[7].split(";")}
            meth.append(sum(CV_BIT.get(a, CV_BIT["other"]) for a in atoms))
            rev.append(REVIEW_RANK.get(c[6], -3))
            gs = c[11]
            gene.append(genes.setdefault(gs, len(genes)))
    np.savez_compressed(out, var=np.array(var, np.int64), cls=np.array(cls, np.int8),
                        date=np.array(date, np.int32), meth=np.array(meth, np.int16),
                        rev=np.array(rev, np.int8), gene=np.array(gene, np.int32))
    (DERIVED / "clinvar_genes.json").write_text(json.dumps(list(genes)), encoding="utf-8")
    return out


def load_clinvar():
    build_clinvar_cache()
    z = np.load(DERIVED / "clinvar_scv.npz")
    genes = json.loads((DERIVED / "clinvar_genes.json").read_text(encoding="utf-8"))
    return {k: z[k] for k in z.files}, genes


def aggregate(var, cls, meth, rev, deriv_mask, gene=None):
    """Per-variant ground bits, aggregate class and emitted review rank.

    Aggregate class (ours, stated in the manuscript): the set of classes in
    {P, U, B} among the variant's SCVs; one class -> that class; more than one
    -> C (conflicting); none -> O."""
    order = np.argsort(var, kind="stable")
    v = var[order]
    uniq, start = np.unique(v, return_index=True)
    c_bit = np.bitwise_or.reduceat(((meth[order] & CONTACT_MASK) != 0).astype(np.int8), start)
    d_bit = np.bitwise_or.reduceat(((meth[order] & deriv_mask) != 0).astype(np.int8), start)
    clsbits = np.left_shift(1, cls[order].astype(np.int64))
    agg_bits = np.bitwise_or.reduceat(clsbits, start)
    pub = agg_bits & 0b111
    n = np.array([0, 1, 1, 2, 1, 2, 2, 3])[pub]
    agg = np.where(n == 0, 5, np.where(n > 1, 6, np.log2(np.maximum(pub, 1)).astype(int)))
    # 0=P 1=U 2=B 5=O(none of P,U,B) 6=C(conflicting)
    review = np.maximum.reduceat(rev[order], start)
    res = {"var": uniq, "c": c_bit, "d": d_bit, "agg": agg, "review": review,
           "verdict": 1 + c_bit.astype(np.int64) + 2 * d_bit.astype(np.int64)}
    if gene is not None:
        # gene: largest non-missing symbol index among the variant's SCVs
        # (caller maps missing symbols to -1); -1 if every SCV lacks one
        res["gene"] = np.maximum.reduceat(gene[order], start)
    return res


AGG_NAMES = {0: "P", 1: "U", 2: "B", 5: "O", 6: "C"}
