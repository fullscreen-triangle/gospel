"""receivers.py -- the schema-to-RDF emission of a LinkML schema, read as a
receiver.

For every class the emitted class IRI is its declared class_uri, expanded
through the prefixes of the imports closure, or default_prefix:ClassName of the
file that defines it. For every slot (top-level slots and class attributes) the
emitted predicate is slot_uri, or default_prefix:slot_name. A fibre is the set
of schema classes (slots) sharing one emitted IRI.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

STANDARD = {
    "linkml": "https://w3id.org/linkml/",
    "prov": "http://www.w3.org/ns/prov#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "schema": "http://schema.org/",
    "dcterms": "http://purl.org/dc/terms/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}


def load_closure(root: Path, import_map: dict | None = None):
    """Load a schema and every non-linkml import beside it. import_map resolves
    remote imports (e.g. 'dcatapplus:latest/schema/dcat_ap_plus') to snapshots."""
    import_map = import_map or {}
    files, seen, todo = [], set(), [root]
    while todo:
        p = todo.pop()
        if p in seen or not p.exists():
            continue
        seen.add(p)
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        files.append((p, d))
        for imp in d.get("imports") or []:
            if imp in import_map:
                todo.append(Path(import_map[imp]))
            elif not imp.startswith("linkml:"):
                todo.append(p.parent / f"{imp}.yaml")
    return files


def _prefix_map(files):
    pm = dict(STANDARD)
    for _, d in files:
        for k, v in (d.get("prefixes") or {}).items():
            pm[k] = v["prefix_reference"] if isinstance(v, dict) else v
    return pm


def _expand(curie: str, pm: dict) -> str:
    if "://" in curie:
        return curie
    if ":" in curie:
        pre, local = curie.split(":", 1)
        if pre in pm:
            return pm[pre] + local
    return curie


def census(root: Path, import_map: dict | None = None) -> dict:
    files = load_closure(root, import_map)
    pm = _prefix_map(files)
    classes = {}
    slots = {}
    for p, d in files:
        dp = d.get("default_prefix") or ""
        for name, c in (d.get("classes") or {}).items():
            c = c or {}
            uri = c.get("class_uri") or f"{dp}:{name}"
            classes[name] = {
                "iri": _expand(uri, pm),
                "declared": c.get("class_uri") is not None,
                "abstract": bool(c.get("abstract")),
                "mixin": bool(c.get("mixin")),
                "is_a": c.get("is_a"),
                "file": p.name,
            }
            for an, a in (c.get("attributes") or {}).items():
                a = a or {}
                suri = a.get("slot_uri") or f"{dp}:{an}"
                slots.setdefault(an, _expand(suri, pm))
        for name, s in (d.get("slots") or {}).items():
            s = s or {}
            suri = s.get("slot_uri") or f"{dp}:{name}"
            slots[name] = _expand(suri, pm)
    concrete = {k: v for k, v in classes.items() if not v["abstract"] and not v["mixin"]}
    fib = defaultdict(list)
    for k, v in concrete.items():
        fib[v["iri"]].append(k)
    sfib = defaultdict(list)
    for k, v in slots.items():
        sfib[v].append(k)
    lam = sum(len(v) * (len(v) - 1) // 2 for v in fib.values())
    slam = sum(len(v) * (len(v) - 1) // 2 for v in sfib.values())
    return {
        "files": [p.name for p, _ in files],
        "classes_all": len(classes),
        "classes_concrete": len(concrete),
        "class_iris": len(fib),
        "class_uri_declared": sum(v["declared"] for v in concrete.values()),
        "Lambda_class_pairs": lam,
        "class_fibre_sizes": sorted((len(v) for v in fib.values()), reverse=True),
        "nontrivial_class_fibres": {k: sorted(v) for k, v in fib.items() if len(v) > 1},
        "slots": len(slots),
        "slot_iris": len(sfib),
        "Lambda_slot_pairs": slam,
        "slot_fibre_sizes": sorted((len(v) for v in sfib.values()), reverse=True),
        "nontrivial_slot_fibres": {k: sorted(v) for k, v in sfib.items() if len(v) > 1},
        "_classes": classes,
    }


def ancestors(classes: dict, name: str) -> list[str]:
    out = []
    cur = classes.get(name, {}).get("is_a")
    while cur and cur not in out:
        out.append(cur)
        cur = classes.get(cur, {}).get("is_a")
    return out
