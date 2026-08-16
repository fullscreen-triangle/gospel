#!/usr/bin/env python3
"""Check that both compilers' keyword sets match the extracted corpus.

The keyword set is small enough to hand-write and easy enough to get
subtly wrong -- the set literal in lang.py spans seven lines. Rather than
generate it (which would make the source unreadable for a 39-element
set), it is written by hand in each compiler and checked here.

Run:  python synopsis/corpus/check_keywords.py
Exit: 0 if all three agree, 1 otherwise.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

TS = REPO / "synopsis" / "ts" / "src" / "tokens.ts"
RS = REPO / "synopsis" / "rs" / "src" / "tokens.rs"


def corpus_keywords() -> set[str]:
    doc = json.loads((HERE / "corpus.json").read_text(encoding="utf-8"))
    return set(doc["keywords"])


def extract_between(path: Path, start: str, end: str) -> str | None:
    """The text between two markers, or None if the file has no such block."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    i = text.find(start)
    if i < 0:
        return None
    j = text.find(end, i + len(start))
    if j < 0:
        return None
    return text[i + len(start):j]


def ts_keywords() -> set[str] | None:
    block = extract_between(TS, "export const KEYWORDS", ";")
    if block is None:
        return None
    return set(re.findall(r'"([a-z_]+)"', block))


def rs_keywords() -> set[str] | None:
    block = extract_between(RS, "pub const KEYWORDS", "];")
    if block is None:
        return None
    return set(re.findall(r'"([a-z_]+)"', block))


def main() -> int:
    want = corpus_keywords()
    problems: list[str] = []
    checked = 0

    for label, got in (("TypeScript", ts_keywords()), ("Rust", rs_keywords())):
        if got is None:
            print(f"  {label:11s} not present yet -- skipped")
            continue
        checked += 1
        missing = sorted(want - got)
        extra = sorted(got - want)
        if missing:
            problems.append(f"{label}: missing {missing}")
        if extra:
            problems.append(f"{label}: has keywords the reference does not: {extra}")
        if not missing and not extra:
            print(f"  {label:11s} OK ({len(got)} keywords)")

    if problems:
        print("keyword check FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print(f"corpus keywords: {len(want)}; implementations checked: {checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
