"""fetch_data.py -- download the public inputs into data/raw/ and record digests.

GO annotations are fetched from current.geneontology.org, ClinVar submissions
from the NCBI FTP site, and the LinkML schemas from their repositories at the
commits recorded in data/raw/schemas/snapshot_shas.json. The digests of the
files actually used are written to data/raw/manifest.json; a rerun against
newer releases will change the numbers, and the manifest says which release a
result belongs to.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import yaml

from common import RAW, sha256

GO = "https://current.geneontology.org/annotations/{}.gaf.gz"
CLINVAR = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/submission_summary.txt.gz"
SCHEMAS = {
    "mixs": ("GenomicsStandardsConsortium/mixs", "src/mixs/schema/", ["mixs"]),
    "ghga": ("ghga-de/ghga-metadata-schema", "src/schema/", ["submission"]),
    "nmdc": ("microbiomedata/nmdc-schema", "src/schema/", ["nmdc"]),
    "chemdcat": ("nfdi-de/chem-dcat-ap", "src/chem_dcat_ap/schema/",
                 ["chem_dcat_ap", "chemical_entities_ap", "chemical_reaction_ap",
                  "material_entities_ap"]),
    "dcatplus": ("nfdi-de/dcat-ap-plus", "src/dcat_ap_plus/schema/",
                 ["dcat_ap_linkml", "dcat_ap_plus"]),
}


def get(url: str, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("fetch", url)
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)


def main():
    for org in ("goa_human", "sgd", "tair"):
        get(GO.format(org), RAW / f"{org}.gaf.gz")
    get(CLINVAR, RAW / "submission_summary.txt.gz")
    shas_path = RAW / "schemas" / "snapshot_shas.json"
    shas = json.loads(shas_path.read_text()) if shas_path.exists() else {}
    for key, (repo, base, roots) in SCHEMAS.items():
        ref = shas.get(repo, {}).get("sha", "main")
        url = f"https://raw.githubusercontent.com/{repo}/{ref}/{base}"
        todo, seen = list(roots), set()
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            dest = RAW / "schemas" / key / f"{n}.yaml"
            get(url + f"{n}.yaml", dest)
            if key == "nmdc":      # follow local imports
                d = yaml.safe_load(dest.read_text(encoding="utf-8"))
                todo += [i for i in d.get("imports") or [] if not i.startswith("linkml:")]
    manifest = {str(p.relative_to(RAW)).replace("\\", "/"): sha256(p)
                for p in sorted(RAW.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {len(manifest)} files")


if __name__ == "__main__":
    main()
