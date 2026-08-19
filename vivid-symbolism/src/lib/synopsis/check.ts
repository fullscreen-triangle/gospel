// VENDORED from synopsis/ts/src -- do not edit here.
// Re-sync with: node src/lib/synopsis/sync.mjs
// The synopsis typechecker -- Stage B.
//
// A direct port of validation/lang.py:692-967 (`Checker`), in the same
// spirit as parser.ts is a port of the parser. The reference is the
// oracle: where a choice here looks arbitrary it is almost always
// because the reference made it, and the conformance corpus records the
// error class the reference raises.
//
// The division of labour between this file and parser.ts is deliberate
// and is not an implementation detail. The parser refuses what cannot be
// WRITTEN; this file refuses what cannot be MEANT. Two theorems depend
// on that split:
//
//   thm:total    -- there is no `while`, so termination is a property of
//                   the grammar. What is left for the checker is only the
//                   step floor (`eta > 0`) and the sweep step, because
//                   those are numbers, and numbers are not grammar.
//   thm:arity    -- `align` without a response clause PARSES, so that it
//                   can be refused here with a diagnostic naming the
//                   missing column. Dying in the parser would produce a
//                   syntax error that says nothing about four columns.
//
// The type `Ty` carries a frame index. Types with different indices are
// different types, and that single device is what gives thm:noleak: a
// value built under one `under` block cannot be referenced from another,
// not by convention but because it does not typecheck.

import {
  ArityError, ParameterError, ResidueError, ScopeError, TerminationError,
  TypeErr, pyList,
} from "./errors";
import type {
  Align, Bind, Claim, Compare, CorrExpr, Detect, Drop, Expr, For, Let,
  NearestUnit, Num, Params, Program, Project, Record as RecordStmt, Relax,
  ResponseExpr, Stmt, Sweep, UnitExpr, Var,
} from "./ast";

// =====================================================================
// Types (Sec 7.2)
// =====================================================================

/**
 * A synopsis type.
 *
 * `frame` is the frame index phi of Sec 7.4. `dim` carries the
 * coefficient count of coord_phi, so that comparing embeddings of
 * different dimension is a type error (Sec 10.2).
 *
 * The reference is a frozen dataclass, so `==` there is structural. JS
 * has no such thing for objects, which is why every comparison in this
 * file goes through `sameTy` or reads `.name` explicitly -- an `===`
 * would silently become identity and the dimension check would stop
 * firing.
 */
export interface Ty {
  readonly name: string;
  readonly frame: string | null;
  readonly dim: number | null;
}

export function ty(
  name: string,
  frame: string | null = null,
  dim: number | null = null,
): Ty {
  return { name, frame, dim };
}

export function tyToString(t: Ty): string {
  let s = t.name;
  if (t.frame) s += `_${t.frame}`;
  if (t.dim !== null) s += `<${t.dim}>`;
  return s;
}

function sameTy(a: Ty, b: Ty): boolean {
  return a.name === b.name && a.frame === b.frame && a.dim === b.dim;
}

const SEQ: Ty = ty("seq");
const REAL: Ty = ty("real");

/** The three result types of Rule 6.11 -- deliberately not unified. */
export const RESULT_TYPES: ReadonlySet<string> = new Set([
  "profile", "ranked", "verdict",
]);

// =====================================================================
// Required parameters -- Thm 9.7
// =====================================================================

/**
 * There are no defaults anywhere; omission is an error, never a silent
 * fill-in. A default threshold is a threshold the report cannot print,
 * and a method whose thresholds are not printed is not recoverable from
 * the script -- which is the whole point of the language.
 *
 * Mirrors `corpus.json:required_params` and lang.py:670-676.
 */
export const REQUIRED_PARAMS: Readonly<Record<string, readonly string[]>> = {
  peaks: ["z", "min_distance", "min_score"],
  top: ["k", "depth"],
  align: ["theta"],
  relax: ["eta", "theta"],
  smith_waterman: ["match", "mismatch", "gap"],
};

// =====================================================================
// The report (Sec 11)
// =====================================================================

/**
 * What a checked program emits. This is Stage D's input, not merely a
 * by-product of checking: the checker is the only stage that knows both
 * the parameter values and the types they applied to, so it is the only
 * stage that can record them.
 */
export interface Report {
  frames: string[];
  parameters: Record<string, number | string>;
  residues: Record<string, string>;
  abandoned: { name: string; line: number; note: string }[];
  claims: { text: string; type: string }[];
  bounds: Record<string, { eta: number; theta: number }>;
  iterations: Record<string, number | string>;
  responses: { name: string; spec: string; params: Params }[];
}

function emptyReport(): Report {
  return {
    frames: [], parameters: {}, residues: {}, abandoned: [], claims: [],
    bounds: {}, iterations: {}, responses: [],
  };
}

// =====================================================================
// The checker
// =====================================================================

class Checker {
  private env = new Map<string, Ty>();
  private methods = new Map<string, Params>();
  private pendingResidue = new Map<string, number>();
  private report: Report = emptyReport();
  private frame: string | null = null;

  // -- entry ---------------------------------------------------------

  check(p: Program): Report {
    for (const d of p.decls) {
      if (d.node === "Open") {
        this.env.set(d.name, SEQ);
      } else {
        this.methods.set(d.name, d.params);
        this.report.responses.push({
          name: d.name, spec: d.spec, params: d.params,
        });
      }
    }

    for (const fb of p.frames) {
      this.frame = fb.name;
      this.report.frames.push(fb.name);
      // Reset per frame, not per program: a residue bound in one frame
      // cannot be discharged by a `record` in the next, because the
      // value it belongs to is not in scope there either.
      this.pendingResidue = new Map();
      for (const s of fb.body) this.stmt(s);

      // Cor 9.8: residue must be consumed before the frame closes.
      // Sorted so the diagnostic is deterministic across engines --
      // Map preserves insertion order, the reference's dict does too,
      // but the reference sorts, so this sorts.
      if (this.pendingResidue.size > 0) {
        const [nm, ln] = [...this.pendingResidue.entries()]
          .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))[0]!;
        throw new ResidueError(
          `residue \`${nm}\` is bound but never recorded or dropped; ` +
            `every comparison value carries (beta, varrho) and the ` +
            `residue must be accounted for`,
          ln,
        );
      }
      this.frame = null;
    }
    return this.report;
  }

  // -- statements ----------------------------------------------------

  private stmt(s: Stmt): void {
    switch (s.node) {
      case "Let": {
        const t = this.expr((s as Let).expr);
        // `ann` is always null -- ':' does not tokenise -- but the
        // branch is kept so the two implementations have the same
        // shape. See the Let note in corpus/ast.json.
        if (s.ann && s.ann !== t.name) {
          throw new TypeErr(
            `\`${s.name}\` annotated ${s.ann} but has type ${tyToString(t)}`,
            s.line,
          );
        }
        this.env.set(s.name, t);
        return;
      }

      case "Bind": {
        const b = s as Bind;
        const t = this.expr(b.expr);
        this.env.set(b.value, t);
        this.env.set(b.residue, REAL);
        this.pendingResidue.set(b.residue, b.line);
        return;
      }

      case "Record": {
        const r = s as RecordStmt;
        for (const n of r.names) {
          if (!this.env.has(n)) {
            throw new ScopeError(`\`${n}\` is not in scope`, r.line);
          }
          this.pendingResidue.delete(n);
          this.report.residues[n] = tyToString(this.env.get(n)!);
        }
        return;
      }

      case "Drop": {
        const d = s as Drop;
        if (!this.env.has(d.name)) {
          throw new ScopeError(`\`${d.name}\` is not in scope`, d.line);
        }
        this.pendingResidue.delete(d.name);
        // `drop` is not `record`'s quiet sibling: abandoning a residue
        // is recorded, so the report shows what was discarded.
        this.report.abandoned.push({
          name: d.name, line: d.line,
          note: `residue abandoned at line ${d.line}`,
        });
        return;
      }

      case "Claim": {
        const c = s as Claim;
        const t = this.expr(c.expr);
        this.report.claims.push({ text: c.text, type: tyToString(t) });
        return;
      }

      case "Relax": {
        const rx = s as Relax;
        this.checkParams("relax", rx.params, rx.line);
        const eta = rx.params.get("eta") as number;
        if (eta <= 0.0) {
          // The exact message the manuscript prints (Sec 7.6).
          throw new TerminationError(
            "`relax` requires a step floor bounded below (Assumption: " +
              `effective update).\n  eta = ${fmt(eta)} is not positive; no ` +
              "termination bound can be emitted.",
            rx.line,
          );
        }
        if (!this.env.has(rx.target)) {
          throw new ScopeError(`\`${rx.target}\` is not in scope`, rx.line);
        }
        this.report.bounds[rx.target] = {
          eta, theta: rx.params.get("theta") as number,
        };
        return;
      }

      case "Sweep": {
        const sw = s as Sweep;
        // Thm 9.1: the trip count is fixed on entry. Both forms are
        // bounded by literals, so no expression in the body can change
        // how many times the body runs.
        let count: number;
        if (sw.values !== null) {
          count = sw.values.length;
        } else {
          if (sw.step <= 0.0) {
            throw new TerminationError("`sweep` requires a positive step", sw.line);
          }
          count = Math.floor((sw.hi - sw.lo) / sw.step + 1e-9) + 1;
        }
        this.report.iterations[sw.var] = count;
        for (const b of sw.body) this.stmt(b);
        return;
      }

      case "For": {
        const f = s as For;
        this.expr(f.src);
        this.env.set(f.var, ty("unit", this.frame));
        this.report.iterations[f.var] = "finite: items(N)";
        if (f.guard) this.report.parameters[`guard:${f.var}`] = f.guard;
        for (const b of f.body) this.stmt(b);
        return;
      }

      default: {
        const bad = s as { node: string; line: number };
        throw new TypeErr(`unknown statement ${bad.node}`, bad.line ?? 0);
      }
    }
  }

  // -- parameters ----------------------------------------------------

  private checkParams(what: string, got: Params, line: number): void {
    const need = REQUIRED_PARAMS[what] ?? [];
    const missing = need.filter((k) => !got.has(k)).sort();
    if (missing.length > 0) {
      throw new ParameterError(
        `\`${what}\` requires ${pyList([...need].sort())}; missing ` +
          `${pyList(missing)}. There are no defaults (Rule: no ` +
          `default thresholds) -- state the value or the report cannot ` +
          `print it.`,
        line,
      );
    }
    for (const [k, v] of got) this.report.parameters[`${what}.${k}`] = v;
  }

  // -- expressions ---------------------------------------------------

  private expr(e: Expr): Ty {
    switch (e.node) {
      case "Var": {
        const v = e as Var;
        const t = this.env.get(v.name);
        if (t === undefined) {
          throw new ScopeError(`\`${v.name}\` is not in scope`, v.line);
        }
        // Thm 9.6: a value made in another frame is a different type.
        // Note this fires even though the name IS in `env` -- scope
        // alone does not license the reference.
        if (t.frame !== null && t.frame !== this.frame) {
          throw new ScopeError(
            `\`${v.name}\` has type ${tyToString(t)} but the enclosing frame ` +
              `is \`${this.frame}\`; ${t.name}_${t.frame} and ` +
              `${t.name}_${this.frame} are distinct types`,
            v.line,
          );
        }
        return t;
      }

      case "Num":
        return REAL;

      case "Project": {
        const p = e as Project;
        const src = this.expr(p.src);
        if (!sameTy(src, SEQ) && src.name !== "seq" && src.name !== "net") {
          throw new TypeErr(
            `\`project\` takes a seq, got ${tyToString(src)}`, p.line,
          );
        }
        switch (p.projector) {
          case "channels":
            return ty("frame", this.frame);
          case "spectral": {
            if (!p.args.has("coeffs")) {
              throw new ParameterError("spectral(...) requires `coeffs`", p.line);
            }
            return ty("coord", this.frame, Math.trunc(p.args.get("coeffs") as number));
          }
          case "contact": {
            if (!p.args.has("medium")) {
              throw new ParameterError("contact(...) requires `medium`", p.line);
            }
            this.report.parameters["contact.medium"] = p.args.get("medium")!;
            return ty("net", this.frame);
          }
          case "cardinal":
            return ty("frame", this.frame);
          default:
            throw new TypeErr(`unknown projector \`${p.projector}\``, p.line);
        }
      }

      case "Compare": {
        const c = e as Compare;
        const lt = this.expr(c.left);
        const rt = this.expr(c.right);
        // Rule 6.11: one keyword, three result types. The index sets
        // genuinely differ -- lag, entry, and none -- so a single
        // return type would erase what makes each family correct.
        switch (c.method) {
          case "xcorr": {
            if (lt.name !== "frame" || rt.name !== "frame") {
              throw new TypeErr(
                `xcorr compares frame against frame, got ${tyToString(lt)} ` +
                  `and ${tyToString(rt)}`,
                c.line,
              );
            }
            return ty("profile", this.frame);
          }
          case "shader": {
            if (lt.name !== "coord" || rt.name !== "coord") {
              throw new TypeErr(
                `shader compares coord against coord, got ${tyToString(lt)} ` +
                  `and ${tyToString(rt)}`,
                c.line,
              );
            }
            if (lt.dim !== rt.dim) {
              throw new TypeErr(
                `embedding dimensions differ: ${lt.dim} vs ${rt.dim}; ` +
                  `the dimension is part of coord_phi`,
                c.line,
              );
            }
            return ty("ranked", this.frame);
          }
          case "smith_waterman":
            this.checkParams("smith_waterman", c.params, c.line);
            return ty("ranked", this.frame);
          case "jaccard":
            return ty("ranked", this.frame);
          case "demand":
            return ty("verdict", this.frame);
          default:
            throw new TypeErr(`unknown method \`${c.method}\``, c.line);
        }
      }

      case "Detect": {
        const d = e as Detect;
        const st = this.expr(d.src);
        this.checkParams(d.kind, d.params, d.line);
        if (d.kind === "peaks") {
          if (st.name !== "profile") {
            throw new TypeErr(
              `\`detect peaks\` consumes profile (lag-indexed), got ` +
                `${tyToString(st)}`,
              d.line,
            );
          }
          return ty("peaks", this.frame);
        }
        if (d.kind === "top") {
          if (st.name !== "ranked") {
            throw new TypeErr(
              `\`detect top\` consumes ranked (entry-indexed), got ` +
                `${tyToString(st)}`,
              d.line,
            );
          }
          // The dimension survives the detector: `top` selects entries,
          // it does not re-embed them.
          return ty("coord", this.frame, st.dim);
        }
        throw new TypeErr(`unknown detector \`${d.kind}\``, d.line);
      }

      case "UnitExpr": {
        const u = e as UnitExpr;
        const st = this.expr(u.src);
        if (st.name !== "net" && st.name !== "unit") {
          throw new TypeErr(`\`unit\` takes a net, got ${tyToString(st)}`, u.line);
        }
        return ty("unit", this.frame);
      }

      case "ResponseExpr": {
        const r = e as ResponseExpr;
        this.expr(r.src);
        if (r.method === null) {
          // Rule 6.7 -- responses may not be anonymous, because the
          // verdict's independence of this choice is OPEN (Sec 13).
          throw new ParameterError(
            "a response must name its perturbation map " +
              "(`response X by <method>`); anonymous responses are " +
              "rejected because the verdict's independence of this " +
              "choice is an open problem",
            r.line,
          );
        }
        if (!this.methods.has(r.method)) {
          throw new ScopeError(
            `response method \`${r.method}\` is not declared`, r.line,
          );
        }
        return ty("response", this.frame);
      }

      case "NearestUnit": {
        const n = e as NearestUnit;
        this.expr(n.net);
        this.expr(n.to);
        return ty("unit", this.frame);
      }

      case "CorrExpr": {
        const c = e as CorrExpr;
        const a = this.expr(c.src);
        const b = this.expr(c.dst);
        if (a.name !== b.name) {
          throw new TypeErr(
            `\`corr\` relates like to like, got ${tyToString(a)} and ` +
              `${tyToString(b)}`,
            c.line,
          );
        }
        return ty("corr", this.frame);
      }

      case "Align": {
        const a = e as Align;
        // Thm 9.5: align is FOUR-ARY. There is no two-argument form.
        // This is checked BEFORE the columns are typed, so the
        // diagnostic names the missing clause rather than complaining
        // about whatever the first column happened to be.
        if (!a.has_response_clause) {
          throw new ArityError(
            "`align` requires four columns: central(a,b) AND " +
              "response(a,b). There is no two-column form, so a verdict " +
              "cannot be reached from content alone (see the " +
              "false-friend separation).",
            a.line,
          );
        }
        for (const x of a.central) {
          const t = this.expr(x);
          if (t.name !== "unit") {
            throw new TypeErr(
              `central columns take units, got ${tyToString(t)}`, a.line,
            );
          }
        }
        // Safe: `has_response_clause` is true exactly when the parser
        // filled `resp`.
        for (const x of a.resp!) {
          const t = this.expr(x);
          if (t.name !== "response") {
            throw new TypeErr(
              `response columns take responses, got ${tyToString(t)}`, a.line,
            );
          }
        }
        if (a.corrs.length !== 2) {
          throw new ArityError(
            `\`align\` needs two correspondences (central and response), ` +
              `got ${a.corrs.length}`,
            a.line,
          );
        }
        for (const c of a.corrs) {
          if (!this.env.has(c)) {
            throw new ScopeError(`\`${c}\` is not in scope`, a.line);
          }
        }
        this.checkParams("align", a.params, a.line);
        return ty("verdict", this.frame);
      }

      default: {
        const bad = e as { node: string; line: number };
        throw new TypeErr(`cannot type ${bad.node}`, bad.line ?? 0);
      }
    }
  }
}

/**
 * Render a number the way Python's `str(float)` does, so diagnostics are
 * byte-identical across implementations: Python prints `0.0`, JS prints
 * `0`. Same reason as `formatNum` in parser.ts.
 */
function fmt(x: number): string {
  return Number.isInteger(x) ? `${x}.0` : `${x}`;
}

/** Typecheck a parsed program. Throws SynopsisError on refusal. */
export function checkProgram(p: Program): Report {
  return new Checker().check(p);
}
