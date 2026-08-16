// VENDORED from synopsis/ts/src -- do not edit here.
// Re-sync with: node src/lib/synopsis/sync.mjs
// The synopsis tokenizer.
//
// A direct port of validation/lang.py:77-131. Three details of that
// implementation are deliberate and are reproduced exactly, because the
// conformance corpus depends on them:
//
//  1. `..` is matched BEFORE a number, so `1..5` is [1, .., 5] and not
//     the number `1.` followed by `.5`.
//  2. A number may carry a leading `-`, but `-` is NOT in the punctuation
//     class. So `1..-5` tokenizes and `a - b` does not: there is no
//     subtraction operator in the language. Arithmetic on results is not
//     something a synopsis program does.
//  3. A string's value is the raw slice between the quotes. Escapes are
//     matched by the pattern but NOT decoded, so `"a\"b"` has the value
//     `a\"b`. Paths are the only strings the language has, and decoding
//     would silently change one.

import { ParseError } from "./errors";

export type TokenKind =
  | "kw"
  | "ident"
  | "num"
  | "string"
  | "punct"
  | "range"
  | "eof";

export interface Token {
  readonly kind: TokenKind;
  readonly text: string;
  readonly line: number;
  /** Decoded payload: number for `num`, unquoted text for `string`. */
  readonly value?: number | string;
}

/**
 * The 39 reserved words.
 *
 * Kept in sync with corpus.json:keywords by
 * `synopsis/corpus/check_keywords.py`, which fails if this list and the
 * extracted one disagree. Notably absent, and absent on purpose:
 * `while`, `fix`, `recurse`, `if`, `else`. Unbounded iteration and
 * conditionals are excluded by the grammar's shape rather than by a
 * later check -- that is what makes `thm:total` a property of the
 * language and not of the checker.
 */
export const KEYWORDS: ReadonlySet<string> = new Set([
  "against", "align", "anchors", "and", "by", "central", "claim",
  "compare", "corr", "detect", "drop", "for", "from", "in", "items",
  "join", "let", "method", "nearest_unit", "open", "peaks", "perturb",
  "project", "quiescent", "record", "relax", "report", "require",
  "response", "separation", "step", "sweep", "to", "top", "under",
  "unit", "until", "where", "bind",
]);

// Order matters: see note 1 above. The `y` flag makes each `exec` a
// match attempt anchored at `lastIndex`, which is how the Python
// `TOKEN_RE.match(src, pos)` loop behaves.
const TOKEN_RE =
  /([ \t\r\n]+)|(#[^\n]*)|("(?:[^"\\]|\\.)*")|(\.\.)|(-?\d+\.\d+|-?\d+)|([A-Za-z_][A-Za-z_0-9]*)|([{}()[\],;=<>+*/])/y;

const WS = 1;
const COMMENT = 2;
const STRING = 3;
const RANGE = 4;
const NUM = 5;
const IDENT = 6;
const PUNCT = 7;

export function tokenise(src: string): Token[] {
  const toks: Token[] = [];
  let pos = 0;
  let line = 1;
  const n = src.length;

  while (pos < n) {
    TOKEN_RE.lastIndex = pos;
    const m = TOKEN_RE.exec(src);
    if (m === null) {
      throw new ParseError(
        `unexpected character ${JSON.stringify(src[pos])}`,
        line,
      );
    }
    const text = m[0];

    if (m[WS] !== undefined) {
      for (const ch of text) if (ch === "\n") line += 1;
    } else if (m[COMMENT] !== undefined) {
      // dropped
    } else if (m[STRING] !== undefined) {
      toks.push({
        kind: "string",
        text,
        line,
        value: text.slice(1, -1), // raw; see note 3
      });
    } else if (m[NUM] !== undefined) {
      toks.push({ kind: "num", text, line, value: Number(text) });
    } else if (m[IDENT] !== undefined) {
      toks.push({
        kind: KEYWORDS.has(text) ? "kw" : "ident",
        text,
        line,
      });
    } else if (m[RANGE] !== undefined) {
      toks.push({ kind: "range", text, line });
    } else if (m[PUNCT] !== undefined) {
      toks.push({ kind: "punct", text, line });
    }

    pos += text.length;
  }

  toks.push({ kind: "eof", text: "", line });
  return toks;
}
