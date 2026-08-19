// The synopsis error hierarchy.
//
// Mirrors validation/lang.py:30-70. The hierarchy is load-bearing, not
// decorative: the conformance corpus records the most specific class the
// reference raises, and a compiler is allowed to report a strict subclass
// of the declared expectation but never a superclass. So `ArityError
// extends TypeErr` is a claim the runner checks.
//
// `TypeError` is spelled `TypeErr` here only because `TypeError` is a
// JavaScript builtin. The language's name for it -- the string in
// `.className`, which is what the corpus compares against -- is
// "TypeError".

/** Base of every refusal the language can make. */
export class SynopsisError extends Error {
  /** Human-facing category, as printed in diagnostics. */
  static readonly kind: string = "error";

  /** The name the conformance corpus compares against. */
  readonly className: string = "SynopsisError";

  /** 1-based source line, or null when the error is not positional. */
  readonly line: number | null;

  constructor(message: string, line: number | null = null) {
    super(message);
    this.line = line;
    // Required for `instanceof` to survive the ES5 target's class
    // downlevelling; harmless otherwise.
    Object.setPrototypeOf(this, new.target.prototype);
    this.name = new.target.name;
  }

  /**
   * The rendered diagnostic, byte-identical to the reference's
   * `SynopsisError.__str__` (validation/lang.py:41-42). It leads with
   * `kind` -- the human-facing category, "arity error" -- and not with
   * `className`, which is the machine-facing name the conformance
   * corpus compares against. The two are deliberately different
   * strings: a reader wants to be told what kind of refusal this is,
   * a runner wants to check the class it was declared as.
   *
   * `kind` is static, so it is read off the constructor rather than
   * off `this`; the cast is needed because TypeScript types
   * `.constructor` as the bare `Function`.
   */
  override toString(): string {
    const kind = (this.constructor as typeof SynopsisError).kind;
    const where = this.line ? ` (line ${this.line})` : "";
    return `${kind}: ${this.message}${where}`;
  }
}

/** The source is not a well-formed program. */
export class ParseError extends SynopsisError {
  static override readonly kind: string = "parse error";
  override readonly className: string = "ParseError";
}

/** The program parses but does not typecheck. */
export class TypeErr extends SynopsisError {
  static override readonly kind: string = "type error";
  override readonly className: string = "TypeError";
}

/** A form was given the wrong number of operands. */
export class ArityError extends TypeErr {
  static override readonly kind: string = "arity error";
  override readonly className: string = "ArityError";
}

/** A name was used outside the frame that binds it. */
export class ScopeError extends TypeErr {
  static override readonly kind: string = "scope error";
  override readonly className: string = "ScopeError";
}

/** A residue was left unconsumed when its frame closed. */
export class ResidueError extends TypeErr {
  static override readonly kind: string = "residue error";
  override readonly className: string = "ResidueError";
}

/** A required parameter was omitted. There are no defaults. */
export class ParameterError extends TypeErr {
  static override readonly kind: string = "parameter error";
  override readonly className: string = "ParameterError";
}

/** A loop was written that is not guaranteed to terminate. */
export class TerminationError extends TypeErr {
  static override readonly kind: string = "termination error";
  override readonly className: string = "TerminationError";
}

/**
 * The class hierarchy as data, so the conformance runner can check
 * subclassing without depending on `instanceof`. Maps a class name to
 * its ancestors, nearest first. Matches `corpus.json:error_hierarchy`.
 */
export const ERROR_HIERARCHY: Readonly<Record<string, readonly string[]>> = {
  ParseError: ["SynopsisError"],
  TypeError: ["SynopsisError"],
  ArityError: ["TypeError", "SynopsisError"],
  ScopeError: ["TypeError", "SynopsisError"],
  ResidueError: ["TypeError", "SynopsisError"],
  ParameterError: ["TypeError", "SynopsisError"],
  TerminationError: ["TypeError", "SynopsisError"],
};

/**
 * Render a string the way Python's `repr` does, and a list of strings
 * the way Python renders a `list`.
 *
 * These exist for one reason: diagnostics must be byte-identical across
 * the two implementations, and the reference builds two of its messages
 * with `{t.text!r}` and `{sorted(need)}`. `JSON.stringify` is the
 * obvious substitute and it is wrong in both places -- it emits double
 * quotes where Python emits single ones, so `unexpected statement '['`
 * would silently become `unexpected statement "["`.
 *
 * A divergence like that is invisible to the conformance corpus, which
 * compares error classes and not message text. It would surface only in
 * Stage D, as a report that differs from the reference's for no reason
 * a reader could see.
 *
 * Python prefers single quotes but switches to double when the string
 * contains a single quote and no double quote; that case is reproduced
 * here because the language's identifiers are unconstrained.
 */
export function pyRepr(s: string): string {
  if (s.includes("'") && !s.includes('"')) {
    return `"${s}"`;
  }
  return `'${s.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;
}

/** Render a string list as Python's `list.__repr__` would. */
export function pyList(xs: readonly string[]): string {
  return `[${xs.map(pyRepr).join(", ")}]`;
}

/** True when `got` is `want` or a descendant of it. */
export function isSubclassOf(got: string, want: string): boolean {
  if (got === want) return true;
  return (ERROR_HIERARCHY[got] ?? []).includes(want);
}
