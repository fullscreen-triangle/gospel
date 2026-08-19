// VENDORED from synopsis/ts/src -- do not edit here.
// Re-sync with: node src/lib/synopsis/sync.mjs
// The synopsis parser.
//
// A direct port of validation/lang.py:333-661. Recursive descent, no
// backtracking, no operator precedence -- every form is prefix-keyword
// led, which is why the grammar can be read off the theorems.
//
// The shape of this grammar is itself load-bearing. There is no `while`
// production, no `if`, and no recursion, so `thm:total` (every program
// terminates) is a property of what can be *written*, not of a check
// that runs afterwards. Likewise there is no indexing production, which
// is what makes `thm:noexact` true of the language rather than of the
// checker: a program cannot reach into a sequence position even in
// principle.

import { ParseError, pyRepr } from "./errors";
import { tokenise, type Token } from "./tokens";
import type {
  Align, Bind, Claim, Compare, CorrExpr, Decl, Detect, Drop, Expr,
  FrameBlock, Let, MethodDecl, NearestUnit, Num, Open, Params, Program,
  Project, Record as RecordStmt, Relax, ResponseExpr, Stmt, Sweep, For,
  UnitExpr, Var,
} from "./ast";

class Parser {
  private readonly toks: Token[];
  private i = 0;

  constructor(toks: Token[]) {
    this.toks = toks;
  }

  // -- helpers -------------------------------------------------------

  private get cur(): Token {
    return this.toks[this.i]!;
  }

  /**
   * Note the kind restriction: an identifier that happens to spell a
   * keyword-like word never matches here, and -- more importantly --
   * `at(":")` is always false, because ':' is not in the tokenizer's
   * punctuation class at all. See the `Let` note in corpus/ast.json.
   */
  private at(text: string): boolean {
    const t = this.cur;
    return (
      t.text === text &&
      (t.kind === "kw" || t.kind === "punct" || t.kind === "range")
    );
  }

  private eat(text: string): Token {
    if (!this.at(text)) {
      throw new ParseError(
        `expected ${JSON.stringify(text)}, found ${JSON.stringify(this.cur.text)}`,
        this.cur.line,
      );
    }
    const t = this.cur;
    this.i += 1;
    return t;
  }

  private ident(): string {
    if (this.cur.kind !== "ident") {
      throw new ParseError(
        `expected identifier, found ${JSON.stringify(this.cur.text)}`,
        this.cur.line,
      );
    }
    const t = this.cur;
    this.i += 1;
    return t.text;
  }

  private number(): number {
    if (this.cur.kind !== "num") {
      throw new ParseError(
        `expected number, found ${JSON.stringify(this.cur.text)}`,
        this.cur.line,
      );
    }
    const t = this.cur;
    this.i += 1;
    return t.value as number;
  }

  private string(): string {
    if (this.cur.kind !== "string") {
      throw new ParseError(
        `expected string, found ${JSON.stringify(this.cur.text)}`,
        this.cur.line,
      );
    }
    const t = this.cur;
    this.i += 1;
    return t.value as string;
  }

  // -- params --------------------------------------------------------

  /**
   * `param { (";" | ",") param }` where a param is `ident = num`.
   *
   * A Map, not an object: the report prints parameters in source order,
   * and JS object key order is only insertion-ordered for non-numeric
   * keys. Relying on that would be a latent reordering bug.
   */
  private params(): Params {
    const out: Params = new Map();
    while (this.cur.kind === "ident") {
      const k = this.ident();
      this.eat("=");
      out.set(k, this.number());
      if (this.at(";") || this.at(",")) {
        this.i += 1;
      } else {
        break;
      }
    }
    return out;
  }

  // -- program -------------------------------------------------------

  program(): Program {
    const decls: Decl[] = [];
    const frames: FrameBlock[] = [];

    while (this.at("open") || this.at("method") || this.at("require")) {
      decls.push(this.decl());
    }
    while (this.at("under")) {
      frames.push(this.frameBlock());
    }

    let report: string;
    if (this.at("report")) {
      this.eat("report");
      this.eat("to");
      report = this.string();
    } else {
      // Rule 6.11: there is no silent program. A run that produces no
      // report is not a run whose result was negative -- it is a
      // program that was never well-formed.
      throw new ParseError(
        "every program must end with `report to <file>` (Rule 6.11: " +
          "programs emit reports)",
        this.cur.line,
      );
    }

    if (this.cur.kind !== "eof") {
      throw new ParseError(
        `trailing input ${JSON.stringify(this.cur.text)}`,
        this.cur.line,
      );
    }

    return { node: "Program", line: 1, decls, frames, report };
  }

  private decl(): Decl {
    if (this.at("open")) {
      const line = this.eat("open").line;
      const name = this.ident();
      this.eat("=");
      return { node: "Open", line, name, path: this.string() } as Open;
    }
    if (this.at("method")) {
      const line = this.eat("method").line;
      const name = this.ident();
      this.eat("=");
      const spec = this.ident();
      this.eat("(");
      const params = this.params();
      this.eat(")");
      return { node: "MethodDecl", line, name, spec, params } as MethodDecl;
    }
    // `require` is accepted by the loop in `program()` but has no decl
    // production, exactly as in the reference: reaching here is a parse
    // error, not a silently ignored declaration.
    throw new ParseError(
      `unexpected declaration ${JSON.stringify(this.cur.text)}`,
      this.cur.line,
    );
  }

  private frameBlock(): FrameBlock {
    const line = this.eat("under").line;
    const name = this.ident();
    this.eat("{");
    const body: Stmt[] = [];
    while (!this.at("}")) {
      if (this.cur.kind === "eof") {
        throw new ParseError("unclosed `under` block", line);
      }
      body.push(this.stmt());
    }
    this.eat("}");
    return { node: "FrameBlock", line, name, body };
  }

  // -- statements ----------------------------------------------------

  private stmt(): Stmt {
    const t = this.cur;

    if (this.at("let")) {
      const line = this.eat("let").line;
      const name = this.ident();
      // The reference has an annotation branch here (`: ident`). It is
      // unreachable -- ':' does not tokenize -- so `ann` is always null.
      // Kept for structural parity; see corpus/ast.json.
      const ann: string | null = null;
      this.eat("=");
      return { node: "Let", line, name, ann, expr: this.expr() } as Let;
    }

    if (this.at("bind")) {
      const line = this.eat("bind").line;
      const value = this.ident();
      this.eat(",");
      const residue = this.ident();
      this.eat("=");
      return { node: "Bind", line, value, residue, expr: this.expr() } as Bind;
    }

    if (this.at("relax")) {
      const line = this.eat("relax").line;
      const target = this.ident();
      this.eat("until");
      this.eat("quiescent");
      this.eat("{");
      const params = this.params();
      this.eat("}");
      return { node: "Relax", line, target, params } as Relax;
    }

    if (this.at("sweep")) return this.sweepStmt();
    if (this.at("for")) return this.forStmt();

    if (this.at("claim")) {
      const line = this.eat("claim").line;
      const text = this.string();
      this.eat("=");
      return { node: "Claim", line, text, expr: this.expr() } as Claim;
    }

    if (this.at("record")) {
      const line = this.eat("record").line;
      const names = [this.ident()];
      while (this.at(",")) {
        this.i += 1;
        names.push(this.ident());
      }
      return { node: "Record", line, names } as RecordStmt;
    }

    if (this.at("drop")) {
      const line = this.eat("drop").line;
      return { node: "Drop", line, name: this.ident() } as Drop;
    }

    throw new ParseError(
      `unexpected statement ${pyRepr(t.text)}`,
      t.line,
    );
  }

  /**
   * `sweep x in lo..hi step s { }` or `sweep x in [a, b, c] { }`.
   *
   * Both forms fix the trip count at entry (`thm:total`): the bounds are
   * numeric literals, so no expression evaluated inside the body can
   * change how many times the body runs.
   */
  private sweepStmt(): Sweep {
    const line = this.eat("sweep").line;
    const varName = this.ident();
    this.eat("in");

    let lo = 0;
    let hi = 0;
    let step = 0;
    let values: number[] | null = null;

    if (this.at("[")) {
      this.i += 1;
      values = [this.number()];
      while (this.at(",")) {
        this.i += 1;
        values.push(this.number());
      }
      this.eat("]");
    } else {
      lo = this.number();
      this.eat("..");
      hi = this.number();
      this.eat("step");
      step = this.number();
    }

    this.eat("{");
    const body: Stmt[] = [];
    while (!this.at("}")) {
      if (this.cur.kind === "eof") {
        throw new ParseError("unclosed `sweep` block", line);
      }
      body.push(this.stmt());
    }
    this.eat("}");

    return { node: "Sweep", line, var: varName, lo, hi, step, values, body };
  }

  /** `for u in items(<src>) [where separation(v) <op> <num>] { }`. */
  private forStmt(): For {
    const line = this.eat("for").line;
    const varName = this.ident();
    this.eat("in");
    this.eat("items");
    this.eat("(");
    const src = this.expr();
    this.eat(")");

    let guard: string | null = null;
    if (this.at("where")) {
      this.i += 1;
      // The guard grammar is exactly `separation(ident) <op> <num>`.
      // It is stored as text because nothing consumes it structurally;
      // the report prints it verbatim so a reader can see what was
      // filtered out.
      this.eat("separation");
      this.eat("(");
      const gv = this.ident();
      this.eat(")");
      const op = this.cur.text;
      this.i += 1;
      const thr = this.number();
      guard = `separation(${gv}) ${op} ${formatNum(thr)}`;
    }

    this.eat("{");
    const body: Stmt[] = [];
    while (!this.at("}")) {
      if (this.cur.kind === "eof") {
        throw new ParseError("unclosed `for` block", line);
      }
      body.push(this.stmt());
    }
    this.eat("}");

    return { node: "For", line, var: varName, src, guard, body };
  }

  // -- expressions ---------------------------------------------------

  expr(): Expr {
    const t = this.cur;
    const line = t.line;

    if (this.at("project")) {
      this.i += 1;
      const src = this.expr();
      this.eat("by");
      const projector = this.cur.text;
      this.i += 1;
      this.eat("(");
      let args: Params = new Map();
      if (projector === "channels") {
        // `channels` takes an encoding NAME, not a numeric parameter --
        // the one place an argument is an identifier.
        args = new Map([["enc", this.ident()]]);
      } else if (!this.at(")")) {
        args = this.params();
      }
      this.eat(")");
      return { node: "Project", line, src, projector, args } as Project;
    }

    if (this.at("compare")) {
      this.i += 1;
      const left = this.expr();
      this.eat("against");
      const right = this.expr();
      this.eat("by");
      const method = this.cur.text;
      this.i += 1;
      let params: Params = new Map();
      if (this.at("(")) {
        this.i += 1;
        // One token of lookahead distinguishes `by m(global)` -- a mode
        // name -- from `by m(gap = 2)`, a parameter list.
        if (this.cur.kind === "ident" && this.toks[this.i + 1]?.text !== "=") {
          params = new Map([["mode", this.ident()]]);
        } else if (!this.at(")")) {
          params = this.params();
        }
        this.eat(")");
      }
      return { node: "Compare", line, left, right, method, params } as Compare;
    }

    if (this.at("detect")) {
      this.i += 1;
      const detKind = this.cur.text;
      this.i += 1;
      this.eat("in");
      const src = this.expr();
      this.eat("{");
      const params = this.params();
      this.eat("}");
      return { node: "Detect", line, kind: detKind, src, params } as Detect;
    }

    if (this.at("align")) {
      this.i += 1;
      this.eat("central");
      this.eat("(");
      const ca = this.expr();
      this.eat(",");
      const cb = this.expr();
      this.eat(")");

      // The response clause is OPTIONAL in the grammar and REQUIRED by
      // the checker (`thm:arity`). That split is deliberate: omitting it
      // is the false-friend mistake, so it must parse in order to be
      // refused with a diagnostic that names what is missing, rather
      // than dying as a syntax error that says nothing.
      const hasResp = this.at("response");
      let resp: [Expr, Expr] | null = null;
      if (hasResp) {
        this.i += 1;
        this.eat("(");
        const ra = this.expr();
        this.eat(",");
        const rb = this.expr();
        this.eat(")");
        resp = [ra, rb];
      }

      this.eat("under");
      const corrs = [this.ident()];
      while (this.at(",")) {
        this.i += 1;
        corrs.push(this.ident());
      }
      this.eat("{");
      const params = this.params();
      this.eat("}");

      return {
        node: "Align", line, central: [ca, cb], resp, corrs, params,
        has_response_clause: hasResp,
      } as Align;
    }

    if (this.at("unit")) {
      this.i += 1;
      const src = this.expr();
      this.eat("anchors");
      return {
        node: "UnitExpr", line, src, anchors: Math.trunc(this.number()),
      } as UnitExpr;
    }

    if (this.at("response")) {
      this.i += 1;
      const src = this.expr();
      let method: string | null = null;
      if (this.at("by")) {
        this.i += 1;
        method = this.ident();
      }
      // method === null means ANONYMOUS, which Rule 6.7 refuses later.
      return { node: "ResponseExpr", line, src, method } as ResponseExpr;
    }

    if (this.at("nearest_unit")) {
      this.i += 1;
      const net = this.expr();
      this.eat("to");
      return { node: "NearestUnit", line, net, to: this.expr() } as NearestUnit;
    }

    if (this.at("corr")) {
      this.i += 1;
      this.eat("from");
      const src = this.expr();
      this.eat("to");
      const dst = this.expr();
      this.eat("by");
      return { node: "CorrExpr", line, src, dst, by: this.ident() } as CorrExpr;
    }

    if (t.kind === "ident") {
      this.i += 1;
      return { node: "Var", line, name: t.text } as Var;
    }

    if (t.kind === "num") {
      this.i += 1;
      return { node: "Num", line, value: t.value as number } as Num;
    }

    throw new ParseError(
      `cannot parse expression at ${JSON.stringify(t.text)}`,
      line,
    );
  }
}

/**
 * Render a number the way Python's `f"{float}"` does, so guard strings
 * are byte-identical across implementations. Python prints `0.5` for
 * 0.5 and `2.0` for 2.0; JavaScript prints `0.5` and `2`.
 */
function formatNum(x: number): string {
  return Number.isInteger(x) ? `${x}.0` : `${x}`;
}

export function parse(src: string): Program {
  return new Parser(tokenise(src)).program();
}
