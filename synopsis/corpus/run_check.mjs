// Stage B conformance: does the TypeScript checker refuse exactly what
// the reference refuses, and accept exactly what it accepts?
//
// The corpus records, for each negative, the most specific class the
// reference raises. A compiler may report a strict SUBCLASS of that
// expectation but never a superclass -- narrowing a refusal is a
// sharper diagnostic, widening one is a lost distinction. That
// asymmetry is checked here via `isSubclassOf` rather than by string
// equality, which is why the error hierarchy is data.
//
// The positives matter as much as the negatives. A checker that refuses
// everything passes all sixteen refusal tests, so the four worked
// programs are what stop this file from rewarding a checker that has
// simply become hostile.
//
// Run:  node synopsis/corpus/run_check.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));

// pathToFileURL, not a bare path: on Windows an absolute path starts
// `c:\` and the ESM loader reads `c:` as a URL scheme.
const mod = (...p) => pathToFileURL(join(HERE, "..", "ts", "src", ...p)).href;

const { check } = await import(mod("index.ts"));
const { SynopsisError, isSubclassOf } = await import(mod("errors.ts"));

const corpus = JSON.parse(readFileSync(join(HERE, "corpus.json"), "utf8"));

let acceptPass = 0;
let refusePass = 0;
const failures = [];

// ---- the four worked programs must typecheck ------------------------
for (const p of corpus.positive) {
  try {
    const report = check(p.src);
    // A report that names no frame means the program had no `under`
    // block, which none of the positives is. Catching it here stops a
    // checker that silently returns an empty report from passing.
    if (report.frames.length === 0) {
      failures.push([p.name, "typechecked, but the report names no frames"]);
    } else {
      acceptPass += 1;
    }
  } catch (e) {
    const where = e instanceof SynopsisError ? e.toString() : String(e);
    failures.push([p.name, `must typecheck, but was refused: ${where}`]);
  }
}

// ---- the sixteen refusals -------------------------------------------
for (const n of corpus.negative) {
  let raised = null;
  try {
    check(n.src);
  } catch (e) {
    if (!(e instanceof SynopsisError)) {
      failures.push([n.name, `threw a non-Synopsis error: ${e}`]);
      continue;
    }
    raised = e;
  }

  if (raised === null) {
    failures.push([n.name, `expected ${n.expect}, but the program was accepted`]);
    continue;
  }
  if (!isSubclassOf(raised.className, n.expect)) {
    failures.push([
      n.name,
      `raised ${raised.className}, want ${n.expect} (or a subclass of it)`,
    ]);
    continue;
  }
  // A refusal with no line is a refusal a reader cannot act on. The
  // two that die in the parser carry one, so all sixteen must.
  if (raised.line === null || raised.line === 0) {
    failures.push([n.name, `raised ${raised.className} with no source line`]);
    continue;
  }
  refusePass += 1;
}

const pass = acceptPass + refusePass;
const total = corpus.positive.length + corpus.negative.length;

console.log("stage B -- checker conformance (TypeScript)");
console.log(`  worked programs accepted : ${acceptPass}/${corpus.positive.length}`);
console.log(`  refusals, correct class  : ${refusePass}/${corpus.negative.length}`);

if (failures.length) {
  console.log(`\n  ${failures.length} FAILURE(S):`);
  for (const [name, why] of failures) console.log(`    ${name}\n      ${why}`);
  process.exit(1);
}
console.log(`\n  all ${pass}/${total} green`);
