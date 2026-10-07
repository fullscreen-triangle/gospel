"""Copy results, figures, manuscript and bibliography to the site's public folder.

The /rescue page reads only these files, so every number on the page is the
number in results/*.json.
"""
import shutil

from common import FIGURES, RESULTS, ROOT, SITE

DEST = SITE / "public" / "rescue"


def main():
    for sub in ("results", "figures"):
        (DEST / sub).mkdir(parents=True, exist_ok=True)
    for p in RESULTS.glob("*.json"):
        shutil.copy2(p, DEST / "results" / p.name)
    for p in FIGURES.glob("*.png"):
        shutil.copy2(p, DEST / "figures" / p.name)
    for name in ("arabdopsis-drought-gwas-rescue.pdf", "references.bib", "presentation/rescue-presentation.pdf"):
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, DEST / src.name)
    print("exported to", DEST)


if __name__ == "__main__":
    main()
