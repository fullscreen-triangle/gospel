// The one place the page talks to the compiler.
//
// Everything the IDE displays comes from here, and nothing here
// re-implements anything: `tokenise` and `parse` are the vendored
// front-end, unmodified. The page is a third consumer of the parser
// after the CLI and the conformance runner, and the value of that is
// entirely conditional on it being the SAME parser -- a page that
// approximated the grammar in order to render something pretty would
// be showing the reader a language that does not exist.

import { tokenise } from "./tokens";
import { parse } from "./parser";
import { SynopsisError } from "./errors";

/**
 * Statements nest, so counting them means walking. `for` bodies are the
 * only nesting the grammar admits today, but recursing on any `body`
 * means the count stays right if that changes.
 */
function countStmts(body) {
  let n = 0;
  for (const s of body) {
    n += 1;
    if (Array.isArray(s.body)) n += countStmts(s.body);
  }
  return n;
}

/**
 * Analyse one source.
 *
 * Returns, on success:
 *   { ok: true, ast, counts: {decls, frames, stmts, report} }
 * and on refusal:
 *   { ok: false, error, message, line }
 *
 * `error` is the corpus class name (`className`), not the JS
 * constructor name, because the class name is what the conformance
 * corpus compares and what the error hierarchy is stated in terms of.
 */
export function analyse(src) {
  try {
    const ast = parse(src);
    return {
      ok: true,
      ast,
      counts: {
        decls: ast.decls.length,
        frames: ast.frames.length,
        stmts: ast.frames.reduce((n, f) => n + countStmts(f.body), 0),
        report: ast.report,
      },
    };
  } catch (e) {
    if (e instanceof SynopsisError) {
      return {
        ok: false,
        error: e.className,
        message: e.message,
        line: e.line,
      };
    }
    // Not a refusal the language defines -- a defect in this page.
    // Surfacing it as such is better than dressing it up as a
    // diagnostic, which would blame the reader's program for our bug.
    return {
      ok: false,
      error: "InternalError",
      message: `The editor failed to analyse this source: ${e && e.message}`,
      line: null,
    };
  }
}

/**
 * The token stream, or null if the source does not even tokenise.
 * Tokenising is attempted separately from parsing so that the Tokens
 * tab still has something to show for a program that parses no further
 * than its first statement.
 */
export function tokensOf(src) {
  try {
    return tokenise(src);
  } catch {
    return null;
  }
}
