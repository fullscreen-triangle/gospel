// Checks the /ide page's tutorial against the TypeScript front-end.
//
// `run_tutorial.mjs` already checks the same eighteen scripts against
// the Rust binary. This is the other half: the page renders trees and
// diagnostics produced by the TS parser, so if the two implementations
// disagreed about a lesson, the page would teach a language the CLI
// does not accept. Running both is what closes that gap.
//
// The analyse() below mirrors vivid-symbolism/src/lib/synopsis/analyse.js
// rather than importing it. That is a deliberate, and slightly
// uncomfortable, duplication: the vendored copies drop the `.ts` import
// extensions so webpack can resolve them, which makes them unloadable
// by Node. Mirroring the twenty lines here keeps the check runnable
// without weakening the vendored bundle. If analyse.js grows past this,
// it should move behind a shared module instead.
//
// Run: node --experimental-strip-types check_ide.mts

import { ALL_FILES } from "../../vivid-symbolism/src/lib/synopsis/tutorial.js";
import { tokenise } from "../ts/src/tokens.ts";
import { parse } from "../ts/src/parser.ts";
import { SynopsisError } from "../ts/src/errors.ts";

// Mirrors src/lib/synopsis/analyse.js exactly. Held here because the
// originals keep their .ts import extensions and so resolve under Node;
// the vendored copies are byte-identical modulo that rewrite.
function countStmts(body: any[]): number {
  let n = 0;
  for (const s of body) { n += 1; if (Array.isArray(s.body)) n += countStmts(s.body); }
  return n;
}
function analyse(src: string): any {
  try {
    const ast: any = parse(src);
    return { ok: true, ast, counts: {
      decls: ast.decls.length, frames: ast.frames.length,
      stmts: ast.frames.reduce((n: number, f: any) => n + countStmts(f.body), 0),
      report: ast.report } };
  } catch (e: any) {
    if (e instanceof SynopsisError) return { ok: false, error: (e as any).className, message: e.message, line: (e as any).line };
    return { ok: false, error: "InternalError", message: String(e && e.message), line: null };
  }
}

// Pinned shape of every accepted program: decls, frames, stmts.
// Displaying a count proves nothing; asserting it is what makes a
// change to the tree walk or to a lesson show up as a failure.
const EXPECT: Record<string, [number, number, number]> = {
  "01-first-program": [1,1,1],   "02-projection": [1,1,2],
  "03-comparison": [2,1,4],      "04-detection": [2,1,5],
  "05-claims": [2,1,6],          "06-motif-scan": [2,1,6],
  "07-spectral": [1,1,2],        "08-reranking": [2,1,5],
  "09-homology": [2,1,7],        "10-units-and-response": [2,1,4],
  "11-alignment": [3,1,12],      "12-transfer": [3,1,12],
  "undefined-variable": [1,1,2], "cross-frame-reference": [2,2,7],
  "unconsumed-residue": [2,1,4], "peaks-missing-parameter": [2,1,5],
};

let bad = 0;
for (const f of ALL_FILES as any[]) {
  const r = analyse(f.src);
  let t: any = null; try { t = tokenise(f.src); } catch { t = null; }
  const want = f.group === "tutorial" ? "ok" : (f.reported || f.expect);
  const got = r.ok ? "ok" : r.error;
  if (f.group === "refusals" && f.stage === "B") {
    if (!r.ok) { bad++; console.log("FAIL", f.id, "labelled stage B but the parser refused it:", r.error); continue; }
  } else if (got !== want) { bad++; console.log("FAIL", f.id, "got", got, "want", want); continue; }
  if (r.ok) {
    const want3 = EXPECT[f.id];
    if (!want3) { bad++; console.log("FAIL", f.id, "no pinned counts"); continue; }
    const got3 = [r.counts.decls, r.counts.frames, r.counts.stmts];
    if (String(got3) !== String(want3)) {
      bad++; console.log("FAIL", f.id, `counts ${got3} but pinned ${want3}`); continue;
    }
    if (r.counts.report === undefined) { bad++; console.log("FAIL", f.id, "no report"); continue; }
  }
  console.log("ok  ", f.id.padEnd(24),
    r.ok ? `decls=${r.counts.decls} frames=${r.counts.frames} stmts=${String(r.counts.stmts).padStart(2)} report=${r.counts.report}`
         : `${r.error} line ${r.line}`,
    `| tok=${t ? t.length : "null"}`);
}
console.log(bad ? `\n${bad} FAILURE(S)` : "\nall green -- TS front-end agrees with the tutorial's labels");
