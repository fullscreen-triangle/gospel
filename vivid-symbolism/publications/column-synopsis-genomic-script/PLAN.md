# Synopsis DSL — implementation plan

Two compilers for one language: **TypeScript** (rapid prototyping, runs in
the browser) and **Rust** (a CLI the user downloads and runs locally, which
connects to the web tool). The Python `validation/lang.py` is neither — it is
the *reference semantics*, already written and already validated, and it stays
the arbiter of what the language means.

---

## 0. The governing constraint: one language, two implementations

The manuscript proves properties **of the language**, not of a program that
implements it: `thm:scope`, `thm:arity`, `thm:params`, `thm:nozero`,
`thm:form`, `thm:total`, `thm:noexact`. If the TS and Rust compilers disagree
about whether a program is well-formed, at least one of them fails a theorem,
and the manuscript's claims stop being about anything runnable.

So the build order is **conformance first**. Before either compiler grows a
feature, there is a shared corpus and a runner that both must satisfy.

The corpus already exists, in `validation/experiments.py`:

- **4 positive programs** (`POSITIVE`) — `motif_scan.syp`, `homology.syp`,
  `adjudicate.syp`, `transfer.syp`. Every one must be **accepted**. This is
  the load-bearing half: without it, every refusal theorem is satisfied
  vacuously by a checker that rejects all input.
- **16 negative programs** (`NEGATIVE`), each with the error class it must
  raise, spanning 8 classes:

  | Class | n | Theorem / rule |
  |---|---|---|
  | `ArityError` | 2 | `thm:arity`, `rule:arity` |
  | `ParameterError` | 3 | `thm:params`, `rule:nodefault` |
  | `ScopeError` | 3 | `thm:scope`, `rule:frames` |
  | `TypeError` | 3 | `thm:form`, `rule:three` |
  | `TerminationError` | 2 | `thm:nozero`, `rule:floored` |
  | `ResidueError` | 1 | `cor:resacc`, `rule:residue` |
  | `ParseError` | 1 | `rule:report` (`no_report`) |
  | `SynopsisError` | 1 | `thm:noexact`, `rule:opaque` (`sequence_indexing`) |

  Rejection alone is not enough — the *class* must match, or Panel 8B is a
  chart of 16 programs failing for the wrong reasons.

### Step 0.1 — extract the corpus to language-neutral JSON

`conformance/corpus.json`, generated **from** `experiments.py` by a small
script (not hand-copied, so it cannot drift):

```jsonc
{ "positive": [ { "name": "motif_scan.syp", "src": "..." } ],
  "negative": [ { "name": "align_without_response", "src": "...",
                  "expect": "ArityError" } ] }
```

Plus `conformance/report_schema.json` — the fixed report schema of
`sec:report`, since both compilers must emit the same document.

### Step 0.2 — the runner

`conformance/run.mjs` and `conformance/run.rs` execute the same JSON and emit
the same verdict table. **A compiler is "done" for a stage when it passes the
corpus, not when it looks finished.**

---

## 1. Where the code lives

```
synopsis/
  corpus/           # language-neutral conformance data (step 0)
  ts/               # @synopsis/compiler — parser, checker, evaluator
  rs/               # synopsis-cli — the downloadable binary
  bridge/           # the wire protocol shared by both
```

Kept **out of** `vivid-symbolism/` and out of the two existing Rust crates.
Reasons: `gospel-core` and `gospel-rust` are a different framework
(consciousness-mimetic / S-entropy) with heavy deps (`rust-htslib`, `tokio`,
`nalgebra`); synopsis shares no types with them and should not inherit their
build. `vivid-symbolism` is a Next.js **JS** app (`jsconfig.json`, not
`tsconfig.json`) — the TS compiler ships as a package it *imports*, so the
compiler stays testable headlessly and the web app stays a consumer.

The paper directory keeps `validation/` untouched as the reference.

---

## 2. Stages

Each stage ends with the corpus passing. No stage begins before the previous
one's corpus subset is green in **both** compilers.

### Stage A — tokenizer + AST + parser (TS, then Rust)

Ported from `lang.py`'s `TOKEN_RE` and `Parser`, which is already exact:
**39 keywords** (the extracted list in `corpus.json` is authoritative — the
set literal in `lang.py` spans seven lines and is easy to miscount by eye),
`#` comments, `..` range, string/num/ident/punct.

The AST is **22 node kinds** (`lang.py` has 23 `Node` dataclasses, but one of
them is the abstract base `Node` itself). Rather than write it twice and let
it drift, the node set is declared once in `corpus/ast.json` and the TS
interfaces and Rust enums are **generated** from it by `corpus/gen_ast.py`.
Handwriting both is how the two compilers start disagreeing about optional
fields.

The 22 split into four groups: 10 expressions (`Expr`), 8 statements (`Stmt`),
2 declarations (`Decl`), and 2 standalone (`FrameBlock`, `Program`).

Corpus subset: the 1 `ParseError` program, plus `while`/`fix`/`recurse`
which must **not** parse (E27 checks this explicitly).

### Stage B — the type system

The load-bearing device is in `lang.py:138`:

```python
@dataclass(frozen=True)
class Ty:
    name: str
    frame: str | None = None   # types with different frame indices are DIFFERENT TYPES
    dim: int | None = None     # comparing embeddings of different dim is a type error
```

`frame` is what gives `thm:scope` — cross-frame reference is a *type* error,
not a scope lookup that happens to fail. `dim` gives the dimension-mismatch
rejection. The three result types (`profile`, `ranked`, `verdict`) are
deliberately **not** unified (`rule:three`), so `profile_into_topk` and
`ranked_into_peaks` are rejected.

In Rust: `Ty { name, frame: Option<String>, dim: Option<usize> }` deriving
`PartialEq` — structural equality *is* the typing rule, so this must not be
hand-implemented.

Corpus subset: 3 `TypeError`, 3 `ScopeError`.

### Stage C — the refusal rules

`REQUIRED_PARAMS` (`lang.py:670`) is a table, so it ports as data, not code:

```
peaks          → z, min_distance, min_score
top            → k, depth
align          → theta
relax          → eta, theta
smith_waterman → match, mismatch, gap
```

There are **no defaults anywhere** (`rule:nodefault`, `thm:params`); omission
is an error, never a silent fill-in. Also here: `align` arity (central pair +
response pair + two correspondences), `eta > 0` (`thm:nozero` — with the exact
message from `sec:lang`), residue consumption before frame close
(`cor:resacc`), anonymous responses (`rule:namedresponse`), and the
`report to` terminal form.

Corpus subset: the remaining 10 negatives. **At the end of Stage C all 16
negatives and — critically — all 4 positives must pass in both compilers.**

### Stage D — the report

`sec:report`'s schema is fixed and has 7 sections: Provenance, Frames,
Parameters, Claims, Residues, Certificates, Assumptions. The
`Report` dataclass (`lang.py:679`) already has the fields.

Two things that are easy to get wrong and are the point of the design:
`require` assertions are recorded **verbatim and marked unverified** (nothing
silently promotes them), and `relax` emits its certificate — `D_0`, `eta`, the
bound `ceil(D_0/eta)`, **and** the realised step count — so `thm:dichotomy`
is checkable from the report alone.

### Stage E — evaluation

Only now does anything compute. The numeric core is `validation/core.py` and
`validation/semantics.py`, and it is the part where a second implementation
can silently disagree:

- min-cut is **exact enumeration over separating subsets**, never max-flow
  (`def:contact` defines it as a minimum over separating sets, and the
  validation suite and every panel use enumeration). Rust may add a max-flow
  fast path **only** behind an equality test against enumeration on small
  instances.
- `relax` needs the *disjunctive* effective-update assumption
  (`ass:effective`) — `U(D) <= D - eta` **or** `U(D) <= theta`. Enforcing the
  first unconditionally produces a false failure on the last step; the Python
  had this bug and `rem:lastdisjunct` records it.
- `xcorr` has two implementations that must agree (E-panel 7A measured
  3.7e-15); Rust gets naive + FFT and the same equality test.

**Known latent defect, not to be silently fixed:** `semantics.spectral`
collapses multichannel input with `x.mean(axis=1)`, and `channelise_dna` is
centred one-hot, so that mean is identically zero and the embedding
degenerates. The validation suite only passes 1-D arrays so it never trips.
The new implementations must **not** quietly reproduce or quietly repair this
— it needs a decision, because changing it changes validated numbers.

Cross-check: the TS and Rust evaluators run the panel computations and must
reproduce `validation/results/*.json` within tolerance.

### Stage F — the bridge (CLI ↔ web tool)

This is the piece the request is really about, so it gets stated concretely.

**The user-facing flow.** Download the CLI, run it, get a token, paste the
token into the web tool, and the two are connected:

```
$ synopsis serve
  synopsis 0.1.0 listening on http://127.0.0.1:7373
  pairing token:  SYN-4K2P-9WQX-7ZTM      (expires in 30 min)

  Paste this token into the web tool to connect.
```

**Which side listens.** Underneath that flow the **CLI is the server** and the
browser dials out to it, not the reverse. Two facts force this. The web tool
is a static Next.js deployment on Vercel with no backend — no database, no
session store, one trivial API route — so there is nowhere on the server for a
CLI to register a token *to*. And the browser cannot read the user's genome
files; uploading them is exactly what a local tool exists to avoid. So the
token is minted locally and the page presents it back to `127.0.0.1:7373`.
Data never leaves the machine — only verdicts, reports, and the small numeric
summaries the user chooses to display.

**What the token is for.** It is not a login. Any page the user happens to
visit can attempt requests to a localhost port, so without a secret the
`open` form would let an arbitrary website read their filesystem. The server
therefore binds `127.0.0.1` only, requires the token as a bearer header, and
checks `Origin` against `--allow` (default: the deployed web tool). The token
is per-session and expires. This is the default posture, not a hardening
flag — `--no-token` does not exist.

**Protocol** (`bridge/protocol.json`, shared types generated for both sides):

| Endpoint | Purpose |
|---|---|
| `GET /health` | version, protocol version, capabilities |
| `POST /check` | source → diagnostics (no execution) |
| `POST /run` | source → run id; streams progress |
| `GET /run/:id/report` | the fixed-schema report |
| `GET /files` | scoped listing of the `--root` the user opened |

**Degradation is a feature, not an error path.** With no CLI paired, the web
tool still parses, typechecks, and reports — that is the whole reason the TS
compiler exists. It refuses only `open` against local paths, and says so.
Protocol version is checked at `/health` so a stale CLI is reported as a
version mismatch rather than failing mysteriously mid-run.

---

## 3. What I am deliberately not doing

- **No OS integration.** No buhera coupling, no causal-graph nodes, no
  S-values. A user runs genomics experiments without knowing that OS exists.
- **Not touching `validation/`.** It is the reference and it is cited by the
  manuscript; the compilers conform to it, not the reverse.
- **Not reusing `gospel-core`/`gospel-rust`.** Different framework, no shared
  types, heavy deps.
- **No exit codes as verdicts.** `DECLINE` is a *result* (`rem:decline`), not
  a failure — the process exits 0 and the report says what happened. A
  non-zero exit means the tool broke, never that the science came back
  negative.

## 4. Order of work

1. Stage 0 — corpus extraction + runner (both languages)
2. Stage A — TS parser → Rust parser
3. Stage B — TS types → Rust types
4. Stage C — TS rules → Rust rules  ← **all 20 corpus programs green here**
5. Stage D — report, both
6. Stage E — evaluator, both, cross-checked against `results/*.json`
7. Stage F — bridge + `serve`, web tool integration

TS leads each stage (it is faster to iterate), Rust follows within the same
stage. They are never more than one stage apart, so divergence is caught in
hours rather than at the end.
