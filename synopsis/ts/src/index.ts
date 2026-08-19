// The synopsis compiler's public surface.
//
// `package.json` has pointed `exports` here since before there was
// anything to export; this file exists now because Stage B gave it
// something to name. What it deliberately does NOT export is an
// evaluator, because there isn't one -- see IDE-PLAN.md. A caller can
// parse a program and typecheck it. It cannot run one.

export { SynopsisError, ParseError, TypeErr, ArityError, ScopeError,
  ResidueError, ParameterError, TerminationError, ERROR_HIERARCHY,
  isSubclassOf, pyRepr, pyList } from "./errors.ts";
export { tokenise, type Token } from "./tokens.ts";
export { parse } from "./parser.ts";
export type * from "./ast.ts";
export {
  checkProgram, type Report, type Ty, ty, tyToString, RESULT_TYPES,
  REQUIRED_PARAMS,
} from "./check.ts";

import { SynopsisError } from "./errors.ts";
import { parse } from "./parser.ts";
import { checkProgram, type Report } from "./check.ts";

/** Parse and typecheck. Throws SynopsisError on refusal. */
export function check(src: string): Report {
  return checkProgram(parse(src));
}

/**
 * Non-throwing form, mirroring lang.py:970. The message is the
 * exception's rendered form -- `ClassName (line N): message` -- because
 * the conformance runner compares against exactly that.
 */
export function accepts(src: string): { ok: boolean; message: string } {
  try {
    check(src);
    return { ok: true, message: "" };
  } catch (exc) {
    if (exc instanceof SynopsisError) {
      return { ok: false, message: exc.toString() };
    }
    throw exc;
  }
}
