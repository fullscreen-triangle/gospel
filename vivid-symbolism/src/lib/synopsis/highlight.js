// Syntax highlighting for the editor.
//
// This deliberately reuses the compiler's KEYWORDS set rather than
// re-declaring one. A highlighter with its own idea of what a keyword
// is drifts from the language the moment either changes, and then the
// editor colours a word that the parser does not recognise -- which
// teaches the reader something false.
//
// The tokeniser proper discards whitespace and comments (the parser has
// no use for them) and does not record columns, so highlighting cannot
// simply consume its output. Instead we re-scan with the same regex in
// the same order, which keeps the two in step: any token the compiler
// recognises is recognised identically here, and the SPANS are what we
// add. If a scan fails -- an unterminated string, a stray character --
// we stop and hand back the remainder as plain text rather than
// guessing, because a highlighter that guesses past a lexical error
// will paint the rest of the file wrongly.

import { KEYWORDS } from "./tokens";

// Same alternation and same order as tokens.ts. Whitespace and comments
// are captured here (groups 1 and 2) because they must be rendered.
const SCAN =
  /([ \t\r\n]+)|(#[^\n]*)|("(?:[^"\\]|\\.)*")|(\.\.)|(-?\d+\.\d+|-?\d+)|([A-Za-z_][A-Za-z_0-9]*)|([{}()[\],;=<>+*/])/y;

/**
 * The name that follows these keywords is a user-chosen identifier
 * being DEFINED, not a reference. Colouring definitions differently is
 * how an editor shows, at a glance, where a name enters scope.
 */
const BINDERS = new Set(["open", "let", "method", "under", "for", "bind"]);

/**
 * Split source into {text, cls} spans.
 *
 * Classes: kw, str, num, comment, def, ident, punct, plain.
 */
export function highlight(src) {
  const out = [];
  let pos = 0;
  let prevKw = null;

  const push = (text, cls) => {
    if (!text) return;
    const last = out[out.length - 1];
    if (last && last.cls === cls) last.text += text;
    else out.push({ text, cls });
  };

  while (pos < src.length) {
    SCAN.lastIndex = pos;
    const m = SCAN.exec(src);
    if (!m) {
      // Lexically invalid from here on. Emit the rest unstyled; the
      // Problems panel is what explains why, and it does so using the
      // real parser rather than this scan.
      push(src.slice(pos), "plain");
      break;
    }

    if (m[1] !== undefined) {
      push(m[1], "plain");
    } else if (m[2] !== undefined) {
      push(m[2], "comment");
      prevKw = null;
    } else if (m[3] !== undefined) {
      push(m[3], "str");
      prevKw = null;
    } else if (m[4] !== undefined) {
      push(m[4], "punct");
    } else if (m[5] !== undefined) {
      push(m[5], "num");
      prevKw = null;
    } else if (m[6] !== undefined) {
      const word = m[6];
      if (KEYWORDS.has(word)) {
        push(word, "kw");
        prevKw = word;
      } else {
        // A name directly after a binder is a definition.
        push(word, BINDERS.has(prevKw) ? "def" : "ident");
        prevKw = null;
      }
    } else {
      push(m[7], "punct");
      // `bind r, res` defines both names, so a comma does not end the
      // binding position the way another token would.
      if (m[7] !== ",") prevKw = null;
    }

    pos = SCAN.lastIndex;
  }

  return out;
}

/** Spans grouped per line, so the editor can render a gutter. */
export function highlightLines(src) {
  const lines = [[]];
  for (const span of highlight(src)) {
    const parts = span.text.split("\n");
    parts.forEach((part, i) => {
      if (i > 0) lines.push([]);
      if (part) lines[lines.length - 1].push({ text: part, cls: span.cls });
    });
  }
  return lines;
}
