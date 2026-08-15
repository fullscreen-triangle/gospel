"""
run_all.py -- execute the four-column alignment validation suite.

Exits non-zero if any theorem check fails, so the suite is a single
pass/fail gate. E13 (response independence) is a MEASUREMENT, not an
assertion: it reports the discrepancy distribution and whether
Assumption 7.5 holds at the stated threshold, and does not fail the suite
if the assumption is violated -- that outcome is a finding, not a bug.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import EXPERIMENTS, SEED

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("=" * 72)
    print("Four-Column Alignment -- validation suite")
    print(f"seed = {SEED}; exact min-cut backend (brute force)")
    print("=" * 72)
    print()

    rows = []
    total_checks = total_passed = 0
    failed_categories = 0
    t0 = time.time()

    for tag, fn in EXPERIMENTS:
        t1 = time.time()
        chk = fn(rng)
        dt = time.time() - t1

        summ = chk.summary()
        summ["tag"] = tag
        summ["seconds"] = round(dt, 3)
        if hasattr(chk, "extra"):
            summ["measurements"] = chk.extra  # type: ignore[attr-defined]

        rows.append(summ)
        total_checks += chk.n
        total_passed += chk.passed
        if not chk.ok:
            failed_categories += 1

        status = "PASS" if chk.ok else "FAIL"
        print(f"[{tag}] {chk.theorem:<44} {chk.passed:>6}/{chk.n:<6} "
              f"{status}  ({dt:.2f}s)")
        if not chk.ok:
            for f in chk.failures:
                print(f"        ! {f}")

        with open(RESULTS / f"{tag}.json", "w", encoding="utf-8") as fh:
            json.dump(summ, fh, indent=1)

    elapsed = time.time() - t0

    # ---- E13 is reported separately: it measures, it does not assert --
    e13 = next((r for r in rows if r["tag"] == "E13"), None)
    print()
    print("-" * 72)
    if e13 and "measurements" in e13:
        m = e13["measurements"]
        print("Assumption 7.5 (response independence) -- MEASUREMENT")
        print(f"  comparisons          : {m['n_comparisons']}")
        print(f"  threshold theta      : {m['theta']}")
        print(f"  max  discrepancy     : {m['max_discrepancy']:.6f}")
        print(f"  mean discrepancy     : {m['mean_discrepancy']:.6f}")
        print(f"  median discrepancy   : {m['median_discrepancy']:.6f}")
        print(f"  fraction within theta: {m['fraction_within_theta']:.4f}")
        verdict = ("HOLDS" if m["assumption_holds_at_theta"]
                   else "VIOLATED -- criterion needs a canonical response")
        print(f"  verdict              : {verdict}")
    print("-" * 72)

    e14 = next((r for r in rows if r["tag"] == "E14"), None)
    if e14 and "measurements" in e14:
        print(f"Pathway four-column verdicts: {e14['measurements']['verdicts']}")
        print("-" * 72)

    print()
    print(f"{len(EXPERIMENTS) - failed_categories}/{len(EXPERIMENTS)} categories, "
          f"{total_passed}/{total_checks} checks passed "
          f"({100.0 * total_passed / max(1, total_checks):.1f}%) in {elapsed:.1f}s")

    master = {
        "suite": "four-column-alignment",
        "seed": SEED,
        "n_categories": len(EXPERIMENTS),
        "n_categories_passed": len(EXPERIMENTS) - failed_categories,
        "total_checks": total_checks,
        "total_passed": total_passed,
        "all_passed": failed_categories == 0,
        "elapsed_seconds": round(elapsed, 3),
        "categories": rows,
    }
    with open(RESULTS / "master.json", "w", encoding="utf-8") as fh:
        json.dump(master, fh, indent=1)

    print(f"results written to {RESULTS}")
    return 0 if failed_categories == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
