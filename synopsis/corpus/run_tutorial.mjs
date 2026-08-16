// Does the IDE's tutorial agree with the compilers?
//
// The /ide page is a third consumer of the parser, and the corpus
// discipline says a consumer that is not checked against the oracle
// will drift. A tutorial that ships a script the compiler rejects --
// or that labels a refusal with the wrong class -- teaches the language
// wrongly, which is worse than teaching nothing.
//
// Two things are checked per script:
//   1. does it parse exactly as the lesson claims?
//   2. for refusals, does the RUST binary agree about which ones are
//      refused at Stage A, and with which class?
//
// The `stage` field on each refusal is a claim about where the refusal
// comes from, and this runner is what keeps that claim honest: a
// refusal marked "A" must actually be refused by the parser today, and
// one marked "B" must actually still parse.
//
// Run:  node synopsis/corpus/run_tutorial.mjs

import { existsSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const RS = join(HERE, "..", "rs");
const TUT = join(HERE, "..", "..", "vivid-symbolism", "src", "lib", "synopsis", "tutorial.js");

const exe = process.platform === "win32" ? "synopsis.exe" : "synopsis";
const BIN = [
  join(RS, "target", "debug", exe),
  join(RS, "target", "release", exe),
].find(existsSync);

if (!BIN) {
  console.error("synopsis binary not found. Build it first:\n" +
    "  cargo build --manifest-path synopsis/rs/Cargo.toml");
  process.exit(2);
}

const { TUTORIAL, REFUSALS } = await import(pathToFileURL(TUT).href);

/** Run the Rust parser. Returns {ok} or {ok:false, error, line}. */
function runParse(src) {
  let stdout;
  try {
    stdout = execFileSync(BIN, ["parse", "--json", "-"], {
      input: src, encoding: "utf8", stdio: ["pipe", "pipe", "pipe"],
    });
  } catch (e) {
    if (!e.stdout) throw new Error(`binary failed: ${e.stderr || e.message}`);
    stdout = e.stdout;
  }
  const j = JSON.parse(stdout);
  return j.error === undefined ? { ok: true } : { ok: false, ...j };
}

const failures = [];
let pass = 0;

// --- the twelve lessons must all parse -----------------------------
for (const lesson of TUTORIAL) {
  const got = runParse(lesson.src);
  if (got.ok) {
    pass += 1;
  } else {
    failures.push([lesson.id, `must parse, but was refused: ${got.error} (line ${got.line}): ${got.message}`]);
  }
}

// --- the refusals must be refused, or not, exactly as labelled -----
for (const r of REFUSALS) {
  const got = runParse(r.src);
  if (r.stage === "A") {
    // Claimed: the grammar refuses this today.
    if (got.ok) {
      failures.push([r.id, `labelled stage A (refused now) but it PARSED`]);
      continue;
    }
    // `reported` records where the parser's class is a strict subclass
    // of what the reference raises; absent it, they must match.
    const want = r.reported || r.expect;
    if (got.error !== want) {
      failures.push([r.id, `raised ${got.error}, but the lesson says ${want}`]);
      continue;
    }
    pass += 1;
  } else {
    // Claimed: parses today, refused by the checker in a later stage.
    if (!got.ok) {
      failures.push([r.id, `labelled stage ${r.stage} (parses today) but the parser refused it: ${got.error}`]);
      continue;
    }
    pass += 1;
  }
}

const total = TUTORIAL.length + REFUSALS.length;
console.log("IDE tutorial -- agreement with the Rust compiler");
console.log(`  lessons that must parse : ${TUTORIAL.length}`);
console.log(`  refusals, stage-labelled: ${REFUSALS.length}`);

if (failures.length) {
  console.log(`\n  ${failures.length} FAILURE(S):`);
  for (const [name, why] of failures) console.log(`    ${name}\n      ${why}`);
  process.exit(1);
}
console.log(`\n  all ${pass}/${total} green`);
