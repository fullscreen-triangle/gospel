# Plan: README update + `/ide` page

Two deliverables. The README change is small and factual; the IDE page is the
larger piece and is planned in detail below.

---

## Part 1 — README

### What is true today (and only this goes in)

| Component | State |
|---|---|
| Grammar, tokeniser, parser (TS) | complete, 24/24 conformance |
| Grammar, tokeniser, parser (Rust) | complete, 24/24 conformance, 17/17 unit |
| Conformance corpus + oracle | 20 programs, 4 must-not-parse forms |
| Both runners mutation-tested | yes |
| Type system (Stage B) | **not started** |
| Refusal rules (C), report (D), evaluator (E), bridge (F) | **not started** |

The README must not describe B–F as if they exist. The honest framing is
"the language and its two front-ends are implemented and cross-verified;
the checker follows." A README that overstates is a defect, since the whole
point of the two-implementation design is that claims are checkable.

### Section to add: `## 8. The synopsis language`

Placed after §7 (Implementation), before the references. Written in the
register of the surrounding document — declarative, no marketing, no
emoji (the existing §1–§7 headings use them, but the numbered academic
sections do not; I match the numbered sections).

Subsections:

- **8.1 Motivation.** Genomic analysis is scripted in general-purpose
  languages, where a pipeline that runs to completion is not thereby
  correct. `synopsis` is a total language: the properties are established
  by what the grammar *admits*, not by a linter run afterwards.
- **8.2 Design: one language, two implementations.** TS for the web
  (immediate, no install); Rust for the CLI (local data, reproducible).
  State the invariant plainly: *they are not two programs that resemble
  each other; a disagreement between them is a failed theorem.* Explain
  the corpus as the device that enforces it.
- **8.3 Guarantees carried by the grammar.** The four absent
  productions — no `while`, no `if`, no recursion, no indexing — and the
  two properties they yield (termination; no exact-match escape hatch).
  Note that these are refused at parse time because nothing in the
  grammar produces them.
- **8.4 The refusal hierarchy.** The eight classes as a tree, and the
  conformance rule: a compiler may report a strict subclass but never a
  superclass. Table of the 16 refusal programs by class.
- **8.5 Status.** The table above, unvarnished.
- **8.6 Usage.** `synopsis parse`, the corpus runners, and a pointer to
  `/ide`.

No citation of the user's papers anywhere (standing constraint). The
section must stand alone.

---

## Part 2 — the `/ide` page

### 2.1 Where it goes

`vivid-symbolism` is Next.js 13, **pages router**, JS (not TS), Tailwind,
with `@/components/Layout` + `TransitionEffect` as the page shell. So:

- `src/pages/ide.js` — the page
- `src/lib/synopsis/` — parser, vendored from `synopsis/ts/src/`
- `src/lib/synopsis/tutorial.js` — the script library
- `src/components/Navbar.js` — add `IDE` link (desktop + mobile lists)

The parser is **vendored, not rewritten.** `synopsis/ts/src/*.ts` is the
implementation that passes 24/24. Rewriting it for the browser would
create a third implementation with no conformance runner, which is
exactly the failure the corpus exists to prevent. Next 13 compiles TS
imports natively, so the files can be copied essentially as-is; the sync
is one directional copy, documented at the top of the vendored directory.

### 2.2 Layout — three columns, VS Code idiom

```
┌──────────────────────────────────────────────────────────────────┐
│  title bar:  synopsis — <active file>            [Parse] [Reset] │
├────────────┬───────────────────────────────┬─────────────────────┤
│ EXPLORER   │  ● 01-first-program.syp   ×   │  OUTPUT             │
│            ├───────────────────────────────┤  ┌───────────────┐  │
│ ▾ tutorial │ 1                             │  │ Problems      │  │
│   01 …     │ 2  open motif = "motif.fa"    │  │ AST           │  │
│   02 …     │ 3                             │  │ Tokens        │  │
│   …        │ 4  under nucleotide {         │  └───────────────┘  │
│ ▾ refusals │ 5      let q = project …      │                     │
│   …        │                               │  ✓ parsed           │
│            │  [gutter | code | caret]      │  3 decls, 1 frame   │
├────────────┴───────────────────────────────┴─────────────────────┤
│  status bar:  Ln 5, Col 12   synopsis   ✓ parsed   TS front-end  │
└──────────────────────────────────────────────────────────────────┘
```

Three columns as specified: **files | editor | output.** Details:

- **Explorer** — two groups, `tutorial/` (12 lessons) and `refusals/`
  (6 programs that must fail). Collapsible. Active file highlighted with
  a left accent bar, VS Code style.
- **Editor** — line-number gutter, monospace, editable `<textarea>`
  overlaid for real editing. Syntax highlighting via the **tokeniser we
  already have**: it returns typed tokens with positions, so highlighting
  is a `map` over the token stream rather than a second, divergent regex
  grammar. When the program does not tokenise, we fall back to plain
  text rather than guessing.
- **Output** — three tabs:
  - *Problems* — the refusal: class, line, message. On success, the
    declaration/frame counts. Clicking a problem jumps the caret.
  - *AST* — the same JSON the conformance runner compares, collapsible.
    This is worth showing precisely because it is the artefact the two
    compilers are held to.
  - *Tokens* — the token stream. Makes the "grammar refuses it" claim
    concrete for the refusal lessons.

Parse is **debounced live** (~300 ms) and also on demand, so the page
demonstrates the language rather than requiring a button press.

Dark by default (`bg-[#1e1e1e]`, `#252526` panels, `#007acc` accent) —
the VS Code palette — but honouring the site's existing `dark:` toggle so
it doesn't fight the rest of the site.

### 2.3 The tutorial — progressive, and every script real

**Constraint: every lesson must parse (or fail) exactly as claimed.**
I verify each one against the Rust binary before shipping, so the page
cannot ship a script that the compilers disagree with.

The progression is built from the four corpus programs, decomposed. The
corpus programs are the *destination*; the early lessons are the pieces.

**Part I — the shape of a program (1–3)**

| # | File | Introduces | Ends at |
|---|---|---|---|
| 1 | `01-first-program.syp` | `open`, `under`, `record`, `report to` | smallest legal program |
| 2 | `02-projection.syp` | `let`, `project … by channels(dna)` | a value with a type |
| 3 | `03-comparison.syp` | `compare … against … by`, `bind v, res` | why binding has two names |

Lesson 3 is where the residue idea lands: `bind r, res_r = …` names both
the result *and* what the comparison did not explain. That second name is
not decoration — a later stage requires it to be consumed.

**Part II — detection and claims (4–6)**

| # | File | Introduces | Ends at |
|---|---|---|---|
| 4 | `04-detection.syp` | `detect peaks in … { … }`, parameter blocks | explicit parameters |
| 5 | `05-claims.syp` | `claim "…" = v` | a program that states what it asserts |
| 6 | `06-motif-scan.syp` | — | **the complete corpus program** |

Lesson 6 is `motif_scan.syp` verbatim. The reveal: the reader has now
written it line by line.

**Part III — a second frame of reference (7–9)**

| # | File | Introduces | Ends at |
|---|---|---|---|
| 7 | `07-spectral.syp` | `spectral(coeffs = 8)`, dimension | types carry a dimension |
| 8 | `08-reranking.syp` | `detect top`, chained comparison | profile → ranked → refined |
| 9 | `09-homology.syp` | — | **`homology.syp` verbatim** |

**Part IV — correspondence and alignment (10–12)**

| # | File | Introduces | Ends at |
|---|---|---|---|
| 10 | `10-units-and-response.syp` | `method`, `unit … anchors`, `response … by` | declared perturbations |
| 11 | `11-alignment.syp` | `corr from … to … by`, `align central(…) response(…)`, `relax … until quiescent` | sufficient, not exact, alignment |
| 12 | `12-transfer.syp` | `for … in items(…) where …` | **`transfer.syp` verbatim** |

Lesson 11 carries the language's central idea and the etymology: the
synoptic gospels are the ones that *align*; the goal is **sufficient
alignment, not exact alignment**. `align` requires **two** correspondences
and a `response` clause — hence two of the `ArityError` refusals.

**Part V — refusals (6 programs, separate `refusals/` group)**

Each is a corpus negative, shown with the class it must raise. This is
the pedagogically important half: the language is defined as much by what
it refuses.

| File | Class | Teaches |
|---|---|---|
| `no-report.syp` | `ParseError` | a program must say where its report goes |
| `undefined-variable.syp` | `ScopeError` | — |
| `cross-frame-reference.syp` | `ScopeError` | a value from another frame is a *different type* |
| `unconsumed-residue.syp` | `ResidueError` | residue must be consumed before the frame closes |
| `peaks-missing-parameter.syp` | `ParameterError` | no defaults, anywhere |
| `sequence-indexing.syp` | `SynopsisError` (reported as `ParseError`) | there is no indexing production — no exact-match escape |

On the last row: the corpus records the reference raising the base
`SynopsisError`, but our parsers refuse `q[3]` at parse time, so they
report `ParseError`. That is not a divergence — `ParseError` is a strict
subclass, and the conformance rule permits a compiler to report a
subclass but never a superclass. It is the grammar catching the program
*earlier* than the reference's requirement, which is the intended design:
indexing is absent from the grammar rather than rejected by a later pass.

**Honesty constraint.** Stage A implements the *parser*. Of the six,
only `no-report.syp` and `sequence-indexing.syp` are refused today; the
other four are refused by the checker (Stage B/C). The IDE must not fake
them. Each refusal lesson therefore carries a badge:

- `refused now` — the parser rejects it, live, in this page
- `refused at Stage B` — parses today; the class is stated, and the
  panel says plainly that the check is not yet implemented

This turns a limitation into the accurate statement of where the work is,
and it means the page never claims a diagnostic it cannot produce.

### 2.4 Lesson chrome

Above the editor, a collapsible prose strip per lesson: 2–4 sentences,
the *why*, plus "what to try" (an edit that makes it fail, with the class
to expect). Keeps the page a tutorial, not a file dump.

### 2.5 What I am deliberately not building

- **No Monaco.** ~5 MB of editor to render ~25 lines, and it would bring
  its own tokenizer that would drift from ours. A textarea plus our own
  token stream is smaller, and it is *the same tokeniser the compiler
  uses*.
- **No `synopsis serve` bridge.** That is Stage F. The page will note
  that the CLI connects by token, but the button is not wired to
  something that does not exist yet.
- **No evaluation.** Stage E. The IDE parses and (later) checks; it does
  not produce results, and it must not appear to.

### 2.6 Order of work

1. `src/lib/synopsis/` — vendor `tokens.ts`, `errors.ts`, `ast.ts`,
   `parser.ts`; add `README` recording the copy direction.
2. `src/lib/synopsis/tutorial.js` — 18 scripts, each verified against
   the Rust binary first.
3. `src/components/ide/` — `Explorer`, `Editor`, `OutputPanel`.
4. `src/pages/ide.js` — compose, wire debounced parse.
5. `Navbar.js` — add the link, both lists.
6. Verify: every tutorial script's TS parse result matches the Rust
   binary's. A tutorial that disagrees with the compiler is worse than
   no tutorial.
7. README §8.

Step 6 is the one that must not be skipped: the page is a third consumer
of the parser, and the corpus discipline says a consumer that is not
checked against the oracle will drift.
