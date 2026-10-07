"""common.py -- seed, paths and result I/O shared by every experiment.

Every experiment draws its randomness from SEED (plus a fixed per-experiment
offset), writes exactly one JSON file to results/, and never reads another
experiment's output. Panels are rendered from results/ alone.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

SEED = 20261003

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
DERIVED = HERE / "data" / "derived"
RESULTS = HERE / "results"
FIGURES = HERE.parent / "figures"

for _d in (DERIVED, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)


def rng_for(offset: int) -> np.random.Generator:
    return np.random.default_rng(SEED + offset)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(type(o))


def write_result(name: str, payload: dict, t0: float | None = None) -> Path:
    payload = dict(payload)
    payload["_meta"] = {
        "experiment": name,
        "seed": SEED,
        "seconds": None if t0 is None else round(time.time() - t0, 2),
    }
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=_default), encoding="utf-8")
    print(f"wrote {path.name}")
    return path


def read_result(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
