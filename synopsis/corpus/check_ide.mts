// Checks the /ide page's tutorial against the TypeScript front-end.
//
// `run_tutorial.mjs` already checks the same eighteen scripts against
// the Rust binary. This is the other half: the page renders trees and
// diagnostics produced by the TS parser, so if the two implementations
// disagreed about a lesson, the page would teach a language the CLI
// does not accept. Running both is what closes that gap.
//
// This used to mirror analyse.js here rather than importing it, because
// the vendored copies drop their `.ts` import extensions for webpack and
// are therefore unloadable by Node. That duplication has been removed:
// `resolve_ext.mjs` teaches the loader to re-add the extension, so the
// page's ACTUAL analyse() is what runs below. The mirrored copy had
// already gone stale -- it stopped at the parser after the checker
// landed -- which is the failure mode a second implementation always
// has, and the reason it is gone.
//
// Run: node --experimental-strip-types --import ./resolve_ext.mjs check_ide.mts

import { ALL_FILES } from "../../vivid-symbolism/src/lib/synopsis/tutorial.js";
import { analyse } from "../../vivid-symbolism/src/lib/synopsis/analyse.js";
import { tokenise } from "../ts/src/tokens.ts";

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
  if (got !== want) { bad++; console.log("FAIL", f.id, "got", got, "want", want); continue; }

  // `stage` is a claim about WHICH pass refuses the program, and now
  // that both passes exist it can be checked rather than trusted. A
  // lesson labelled "B" that dies in the parser is teaching the wrong
  // rule: the reader is told the grammar accepts the program and the
  // meaning is what fails, and the tree they are shown would be the
  // evidence for it -- except there would be no tree.
  if (f.group === "refusals" && r.stage !== f.stage) {
    bad++;
    console.log("FAIL", f.id, `labelled stage ${f.stage} but refused at stage ${r.stage}`);
    continue;
  }
  if (f.group === "refusals" && f.stage === "B" && !r.ast) {
    bad++; console.log("FAIL", f.id, "stage-B refusal carries no tree"); continue;
  }
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
