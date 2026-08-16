// VENDORED from synopsis/ts/src -- do not edit here.
// Re-sync with: node src/lib/synopsis/sync.mjs
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

  toString(): string {
    const where = this.line === null ? "" : ` (line ${this.line})`;
    return `${this.className}${where}: ${this.message}`;
  }
}

/** The source is not a well-formed program. */
export class ParseError extends SynopsisError {
  static readonly kind = "parse error";
  override readonly className = "ParseError";
}

/** The program parses but does not typecheck. */
export class TypeErr extends SynopsisError {
  static readonly kind = "type error";
  override readonly className = "TypeError";
}

/** A form was given the wrong number of operands. */
export class ArityError extends TypeErr {
  static readonly kind = "arity error";
  override readonly className = "ArityError";
}

/** A name was used outside the frame that binds it. */
export class ScopeError extends TypeErr {
  static readonly kind = "scope error";
  override readonly className = "ScopeError";
}

/** A residue was left unconsumed when its frame closed. */
export class ResidueError extends TypeErr {
  static readonly kind = "residue error";
  override readonly className = "ResidueError";
}

/** A required parameter was omitted. There are no defaults. */
export class ParameterError extends TypeErr {
  static readonly kind = "parameter error";
  override readonly className = "ParameterError";
}

/** A loop was written that is not guaranteed to terminate. */
export class TerminationError extends TypeErr {
  static readonly kind = "termination error";
  override readonly className = "TerminationError";
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

/** True when `got` is `want` or a descendant of it. */
export function isSubclassOf(got: string, want: string): boolean {
  if (got === want) return true;
  return (ERROR_HIERARCHY[got] ?? []).includes(want);
}
