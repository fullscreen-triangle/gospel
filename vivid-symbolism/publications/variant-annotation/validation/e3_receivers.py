"""E3 -- receivers: what the emitted graph of a schema can carry.

Census of three genomics LinkML schemas (MIxS, GHGA submission, NMDC) against
two research-data profiles from chemistry (Chem-DCAT-AP and its DCAT-AP-PLUS
core), all read from snapshots pinned in data/raw/schemas/snapshot_shas.json.

For each schema: concrete classes, emitted class IRIs, the number Lambda of
class pairs whose distinction the type triple cannot carry, slots and emitted
slot IRIs, and the verdict capacity under a declared witness assignment: which
classes count as contact (measurement) and which as derivation (computation),
and whether every contact class is separated from every derivation class by
its emitted IRI.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from common import RAW, sha256, write_result
from gsv import receivers as Rc

S = RAW / "schemas"
IMPORT_MAP = {"dcatapplus:latest/schema/dcat_ap_plus": str(S / "dcatplus" / "dcat_ap_plus.yaml")}

SCHEMAS = {
    "MIxS": S / "mixs" / "mixs.yaml",
    "GHGA": S / "ghga" / "submission.yaml",
    "NMDC": S / "nmdc" / "nmdc.yaml",
    "Chem-DCAT-AP": S / "chemdcat" / "chem_dcat_ap.yaml",
    "DCAT-AP-PLUS": S / "dcatplus" / "dcat_ap_plus.yaml",
}
DOMAIN = {"MIxS": "genomics", "GHGA": "genomics", "NMDC": "genomics",
          "Chem-DCAT-AP": "chemistry", "DCAT-AP-PLUS": "chemistry"}


def declared_witness(name: str, classes: dict):
    """The witness declaration: an input of the method, stated per schema.
    Contact = a class whose instances are acts of measurement on material;
    derivation = a class whose instances are computations on data."""
    def concrete_desc(root):
        return sorted(k for k, v in classes.items()
                      if not v["abstract"] and not v["mixin"]
                      and (k == root or root in Rc.ancestors(classes, k)))
    if name == "GHGA":
        return ["Experiment"], ["Analysis"]
    if name == "NMDC":
        return concrete_desc("DataGeneration"), concrete_desc("WorkflowExecution")
    if name in ("Chem-DCAT-AP", "DCAT-AP-PLUS"):
        contact = [k for k in ("DataGeneratingActivity", "SubstanceSampleCharacterization",
                               "ReactionMonitoring") if k in classes]
        return contact, [k for k in ("DataAnalysis",) if k in classes]
    return [], []          # MIxS declares no activity classes at all


def main():
    t0 = time.time()
    out = {}
    for name, root in SCHEMAS.items():
        c = Rc.census(root, IMPORT_MAP)
        classes = c.pop("_classes")
        contact, deriv = declared_witness(name, classes)
        iri = {k: classes[k]["iri"] for k in classes}
        shared = [(a, b) for a in contact for b in deriv if iri[a] == iri[b]]
        c["witness"] = {
            "contact": contact, "derivation": deriv,
            "contact_iris": sorted({iri[k] for k in contact}),
            "derivation_iris": sorted({iri[k] for k in deriv}),
            "contact_derivation_pairs": len(contact) * len(deriv),
            "pairs_sharing_iri": shared,
            "verdict_capable": bool(contact) and bool(deriv) and not shared,
        }
        c["domain"] = DOMAIN[name]
        c["sha256_root"] = sha256(root)
        out[name] = c
    # MIxS writes method and software as content of the sample record
    import yaml
    mixs = yaml.safe_load(SCHEMAS["MIxS"].read_text(encoding="utf-8"))["slots"]
    pat = re.compile(r"(meth|software|pipeline|platform|seq_meth|assembly|annot|"
                     r"binning|_pred|compl_|contam_|sop|protocol)")
    method_slots = sorted(k for k in mixs if pat.search(k) and k != "methane")
    out["MIxS"]["method_slots_as_content"] = method_slots
    snap = json.loads((S / "snapshot_shas.json").read_text())
    write_result("E3_receivers", {"schemas": out, "snapshots": snap,
                                  "order": list(SCHEMAS)}, t0)


if __name__ == "__main__":
    main()
