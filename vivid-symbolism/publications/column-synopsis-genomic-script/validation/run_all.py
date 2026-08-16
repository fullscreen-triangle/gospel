"""
run_all.py -- run every validation check and write JSON results.

Usage:
    python run_all.py

Writes one JSON file per check into results/, plus results/master.json.
Exits non-zero if any ASSERTION fails.

E32 (response independence) is a MEASUREMENT, not an assertion: it
reports the discrepancy distribution and whether the assumption holds at
the stated threshold, and does not fail the suite if the assumption is
violated -- that outcome is the paper's principal open problem, a
finding, not a bug.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import numpy as np

import experiments as X

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def jsonable(o):
    """Cast numpy scalars/arrays to native types.

    numpy bools serialise as the STRING "True", which silently defeats
    any later `result.get("ok", False)` test. This has bitten a sibling
    suite; casting here rather than trusting the encoder.
    """
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return [jsonable(x) for x in o.tolist()]
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(x) for x in o]
    return o


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rows = []
    failed = 0
    total_checks = 0
    total_passed = 0

    for tag, fn in X.EXPERIMENTS:
        rng = np.random.default_rng(X.SEED)
        start = time.time()
        is_measurement = tag in X.MEASUREMENTS
        try:
            chk = fn(rng)
            rec = chk.summary()
            rec["extra"] = jsonable(chk.extra)
            rec["kind"] = "measurement" if is_measurement else "assertion"
            rec["elapsed_seconds"] = round(time.time() - start, 3)
            ok = bool(rec["ok"])
        except Exception:
            rec = {
                "name": tag,
                "claim": getattr(fn, "__doc__", "").strip().split("\n")[0],
                "n": 0, "passed": 0, "ok": False, "max_err": 0.0,
                "failures": ["EXCEPTION"],
                "traceback": traceback.format_exc(),
                "kind": "measurement" if is_measurement else "assertion",
                "elapsed_seconds": round(time.time() - start, 3),
                "extra": {},
            }
            ok = False

        with open(os.path.join(OUT, f"{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

        total_checks += rec["n"]
        total_passed += rec["passed"]
        counts_against_suite = ok or is_measurement
        if not counts_against_suite:
            failed += 1

        status = "MEASURED" if is_measurement else ("PASS" if ok else "FAIL")
        print(f"[{tag}] {rec['passed']}/{rec['n']} {status:8s} "
              f"({rec['elapsed_seconds']}s)  {rec['claim']}")
        if not ok and not is_measurement:
            for m in rec["failures"][:3]:
                print(f"        ! {m}")
            if "traceback" in rec:
                print("        ! " + rec["traceback"].strip().splitlines()[-1])

        rows.append({
            "tag": tag, "claim": rec["claim"], "kind": rec["kind"],
            "n": rec["n"], "passed": rec["passed"], "ok": ok,
            "max_err": rec["max_err"],
            "elapsed_seconds": rec["elapsed_seconds"],
        })

    elapsed = time.time() - t0
    master = {
        "suite": "synopsis-genomic-scripting",
        "seed": X.SEED,
        "n_categories": len(X.EXPERIMENTS),
        "n_assertions": len(X.EXPERIMENTS) - len(X.MEASUREMENTS),
        "n_measurements": len(X.MEASUREMENTS),
        "n_categories_failed": failed,
        "total_checks": total_checks,
        "total_passed": total_passed,
        "all_passed": failed == 0,
        "elapsed_seconds": round(elapsed, 3),
        "categories": jsonable(rows),
    }
    with open(os.path.join(OUT, "master.json"), "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)

    print()
    print(f"{total_passed}/{total_checks} individual checks passed across "
          f"{len(X.EXPERIMENTS)} categories in {elapsed:.1f}s")
    print(f"assertion categories failed: {failed}")
    print(f"results written to {OUT}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
