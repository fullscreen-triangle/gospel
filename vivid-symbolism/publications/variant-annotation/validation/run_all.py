"""run_all.py -- fetch inputs, run E1--E7, render the panels.

    python run_all.py            # everything
    python run_all.py e4 e7      # selected experiments
    python run_all.py --panels   # re-render from results/ only
"""

from __future__ import annotations

import importlib
import sys

EXPERIMENTS = ["e1_calls", "e2_generic", "e3_receivers", "e4_verdicts",
               "e5_twins", "e6_spectral", "e7_answering"]


def main(argv):
    if "--panels" not in argv:
        importlib.import_module("fetch_data").main()
        chosen = [e for e in EXPERIMENTS if not argv or e.split("_")[0] in argv]
        for e in chosen:
            print(f"== {e}")
            importlib.import_module(e).main()
    mp = importlib.import_module("make_panels")
    for f in mp.PANELS.values():
        f()


if __name__ == "__main__":
    main(sys.argv[1:])
