"""S3: predicted protein effects of Cod-0's rare variants.

Single-variant annotation. For every rare site from S2 where Cod-0 carries the non-Col-0
allele and that falls in a canonical CDS or a canonical splice site, Cod-0's base is read from
its 1001 Genomes v3.1 pseudogenome. The pseudogenome contains indels, so its coordinates drift
from TAIR10; the site is located by matching the 25-bp TAIR10 flank within +/-4 kb. The single
substitution is then applied to the TAIR10 canonical CDS and classified.

Writes results/S3_cod0_effects.json, S3_cod0_damaging_genes.csv, S3_cod0_coding_variants.csv.
"""
import gzip
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from common import CACHE, RESULTS, save
from cod0_genetics import CURATED

GENO = CACHE / "geno"
CODON = {}
_b = "TCAG"
_aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
for i, a in enumerate(_b):
    for j, b in enumerate(_b):
        for k, c in enumerate(_b):
            CODON[a + b + c] = _aa[16 * i + 4 * j + k]
COMP = str.maketrans("ACGTN", "TGCAN")
GROUPS = [set("AGPST"), set("DENQ"), set("HKR"), set("ILMV"), set("FWY"), set("C")]
LOF = ["stop_gained", "start_lost", "stop_lost", "splice_site"]


def read_fasta(path, rename):
    seqs, name, buf = {}, None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                name, buf = rename(line[1:].strip()), []
            else:
                buf.append(line.strip().upper())
    if name:
        seqs[name] = "".join(buf)
    return seqs


def canonical_cds():
    tx_gene, canon, cds, symbol, desc, first_tx = {}, set(), defaultdict(list), {}, {}, {}
    with gzip.open(GENO / "TAIR10.gff3.gz", "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[0] not in "12345":
                continue
            a = dict(kv.split("=", 1) for kv in p[8].split(";") if "=" in kv)
            if p[2] == "gene":
                g = a.get("gene_id")
                symbol[g] = a.get("Name", "")
                desc[g] = re.sub(r" \[Source:.*", "", a.get("description", "")).replace("%2C", ",").replace("%3B", ";")
            elif p[2] == "mRNA":
                g = a.get("Parent", "").replace("gene:", "")
                tx_gene[a["ID"]] = g
                first_tx.setdefault(g, a["ID"])
                if "Ensembl_canonical" in a.get("tag", ""):
                    canon.add(a["ID"])
            elif p[2] == "CDS":
                cds[a.get("Parent")].append((p[0], p[6], int(p[3]), int(p[4])))
    canon_by_gene = {tx_gene[t]: t for t in canon if t in tx_gene}
    out = {}
    for g, t in first_tx.items():
        parts = cds.get(canon_by_gene.get(g, t))
        if parts:
            out[g] = (parts[0][0], parts[0][1], sorted((s, e) for _, _, s, e in parts))
    return out, symbol, desc


def build(seq, strand, segs):
    s = "".join(seq[a - 1:b] for a, b in segs)
    return s if strand == "+" else s.translate(COMP)[::-1]


def radical(a, b):
    return not any(a in g and b in g for g in GROUPS)


def alt_base(ref_c, cod_c, p, flank=25, win=4000):
    """Cod-0 base at TAIR10 position p, located by unique flank match in the pseudogenome."""
    w0 = max(0, p - 1 - win)
    window = cod_c[w0: p - 1 + win]
    left, right = ref_c[p - 1 - flank: p - 1], ref_c[p: p + flank]
    hits = []
    i = window.find(left)
    if i >= 0 and window.find(left, i + 1) < 0 and i + flank < len(window):
        hits.append(window[i + flank])
    j = window.find(right)
    if j >= 1 and window.find(right, j + 1) < 0:
        hits.append(window[j - 1])
    if not hits or (len(hits) == 2 and hits[0] != hits[1]):
        return None
    return hits[0]


def main():
    ref = read_fasta(GENO / "TAIR10.fa.gz", lambda h: h.split()[0])
    cod = read_fasta(GENO / "pseudo9836.fasta.gz", lambda h: h.split("|")[4].replace("Chr", ""))
    genes, symbol, desc = canonical_cds()
    rare = pd.read_csv(CACHE / "S2_cod0_rare_sites.csv")
    rare["chrom"] = rare.chrom.astype(str)
    n_ref_allele = int((rare.cod_allele == 0).sum())
    alt = rare[rare.cod_allele == 1]
    by_chrom = {c: dict(zip(g.pos, g.carriers_species)) for c, g in alt.groupby("chrom")}

    rows, unresolved, disagree, resolved = [], 0, 0, 0
    for g, (c, strand, segs) in genes.items():
        sites = by_chrom.get(c, {})
        lo, hi = segs[0][0] - 2, segs[-1][1] + 2
        cand = [p for p in sites if lo <= p <= hi]
        if not cand:
            continue
        rcds = build(ref[c], strand, segs)
        if len(rcds) % 3:
            continue
        coords = np.concatenate([np.arange(a, b + 1) for a, b in segs])
        if strand == "-":
            coords = coords[::-1]
        where = {int(x): i for i, x in enumerate(coords)}
        splice = set()
        for (a1, b1), (a2, b2) in zip(segs[:-1], segs[1:]):
            splice.update((b1 + 1, b1 + 2, a2 - 2, a2 - 1))
        for p in cand:
            in_cds, in_splice = p in where, p in splice
            if not (in_cds or in_splice):
                continue
            b = alt_base(ref[c], cod[c], p)
            if b is None or b == "N":
                unresolved += 1
                continue
            if b == ref[c][p - 1]:
                disagree += 1                       # imputed matrix says non-reference, pseudogenome says reference
                continue
            resolved += 1
            base = {"gene": g, "symbol": symbol.get(g, ""), "chrom": c, "pos": int(p),
                    "carriers_species": int(sites[p]), "ref": ref[c][p - 1], "alt": b,
                    "protein_length": len(rcds) // 3}
            if in_splice and not in_cds:
                rows.append({**base, "effect": "splice_site", "codon": np.nan, "ref_aa": "", "cod0_aa": "", "radical": False})
                continue
            i = where[p]
            k = i - i % 3
            a_ = b if strand == "+" else b.translate(COMP)
            rc = rcds[k:k + 3]
            qc = rc[:i % 3] + a_ + rc[i % 3 + 1:]
            ra, qa = CODON.get(rc, "X"), CODON.get(qc, "X")
            if ra == qa:
                eff = "synonymous"
            elif qa == "*":
                eff = "stop_gained"
            elif ra == "*":
                eff = "stop_lost"
            elif k == 0 and ra == "M":
                eff = "start_lost"
            else:
                eff = "missense"
            rows.append({**base, "effect": eff, "codon": k // 3 + 1, "ref_aa": ra, "cod0_aa": qa,
                         "radical": bool(eff == "missense" and radical(ra, qa))})
    E = pd.DataFrame(rows)
    E["description"] = E.gene.map(desc)
    E["frac_position"] = E.codon / E.protein_length
    damaging = E[E.effect.isin(LOF)].copy()
    gene_tab = E.groupby(["gene", "symbol"]).agg(
        n_missense=("effect", lambda e: int((e == "missense").sum())),
        n_radical=("radical", lambda r: int(r.astype(bool).sum())),
        n_lof=("effect", lambda e: int(e.isin(LOF).sum())),
        n_synonymous=("effect", lambda e: int((e == "synonymous").sum())),
        min_carriers=("carriers_species", "min")).reset_index()
    gene_tab["description"] = gene_tab.gene.map(desc)
    gene_tab = gene_tab.sort_values(["n_lof", "n_radical", "n_missense"], ascending=False)

    n_coding = len(genes)
    curated_ids = {g: (cat, sym) for cat, gl in CURATED.items() for g, sym in gl.items()}
    hit = set(gene_tab[(gene_tab.n_missense + gene_tab.n_lof) > 0].gene)
    cur_in = [g for g in curated_ids if g in genes]
    k = sum(g in hit for g in cur_in)
    enrich = {"genes_tested": n_coding, "genes_with_rare_nonsyn": len(hit), "curated_tested": len(cur_in),
              "curated_with_rare_nonsyn": k, "expected": len(cur_in) * len(hit) / n_coding,
              "p_hypergeom": float(stats.hypergeom.sf(k - 1, n_coding, len(hit), len(cur_in)))}
    curated = []
    for g in cur_in:
        sub = E[E.gene == g]
        if len(sub):
            cat, sym = curated_ids[g]
            curated.append({"category": cat, "gene": g, "symbol": sym,
                            "variants": sub[["pos", "codon", "protein_length", "ref_aa", "cod0_aa", "effect", "radical",
                                             "carriers_species"]].to_dict(orient="records")})
    out = {"method": __doc__.strip().splitlines()[2].strip(),
           "rare_sites_where_cod0_has_col0_allele": n_ref_allele,
           "coding_or_splice_sites_resolved": resolved, "unresolved": unresolved,
           "matrix_vs_pseudogenome_disagree": disagree,
           "rare_effects": E.effect.value_counts().to_dict(),
           "rare_radical_missense": int(E.radical.astype(bool).sum()),
           "damaging": damaging.to_dict(orient="records"),
           "top_genes": gene_tab.head(60).to_dict(orient="records"),
           "curated": curated, "curated_enrichment": enrich}
    save("S3_cod0_effects", out)
    gene_tab.to_csv(RESULTS / "S3_cod0_damaging_genes.csv", index=False)
    E.to_csv(RESULTS / "S3_cod0_coding_variants.csv", index=False)
    return out, damaging


if __name__ == "__main__":
    o, dmg = main()
    print("resolved", o["coding_or_splice_sites_resolved"], "unresolved", o["unresolved"],
          "disagree", o["matrix_vs_pseudogenome_disagree"], "| Cod-0 carries Col-0 allele at", o["rare_sites_where_cod0_has_col0_allele"], "rare sites")
    print("effects:", o["rare_effects"], "radical missense:", o["rare_radical_missense"])
    print("curated enrichment:", o["curated_enrichment"])
    print("\nLoss-of-function-type rare variants (sorted by rarity):")
    for _, r in dmg.sort_values(["carriers_species", "effect"]).iterrows():
        fp = "" if pd.isna(r.frac_position) else f"{r.frac_position:.0%}"
        print(f"  {r.gene} {r.symbol:12s} {r.effect:12s} {fp:>5s} carriers={r.carriers_species:2d} | {str(r.description)[:70]}")
    print("\nCurated genes:")
    for c in o["curated"]:
        print(f"  [{c['category']}] {c['symbol']}: " + "; ".join(
            f"{v['ref_aa']}{'' if pd.isna(v['codon']) else int(v['codon'])}{v['cod0_aa']} ({v['effect']}"
            f"{', radical' if v['radical'] else ''}; carriers {v['carriers_species']})" for v in c["variants"]))
