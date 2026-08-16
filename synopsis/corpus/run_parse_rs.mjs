// Stage A conformance: does the RUST parser build the same trees as the
// reference?
//
// The same oracle and the same comparison as run_parse.mjs, driving the
// compiled binary through `synopsis parse --json`. Written in JS rather
// than as a Rust integration test on purpose: a compiler that grades its
// own homework can drift with its own bugs, and the whole reason there
// are two implementations is that neither gets to be the authority. Both
// runners read corpus/ref_ast.json, which came from the reference.
//
// Run:  cargo build --manifest-path synopsis/rs/Cargo.toml
//       node synopsis/corpus/run_parse_rs.mjs

import { readFileSync, existsSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const RS = join(HERE, "..", "rs");

const exe = process.platform === "win32" ? "synopsis.exe" : "synopsis";
const candidates = [
  join(RS, "target", "debug", exe),
  join(RS, "target", "release", exe),
];
const BIN = candidates.find(existsSync);
if (!BIN) {
  console.error(
    "synopsis binary not found. Build it first:\n" +
      "  cargo build --manifest-path synopsis/rs/Cargo.toml",
  );
  process.exit(2);
}

const corpus = JSON.parse(readFileSync(join(HERE, "corpus.json"), "utf8"));
const oracle = JSON.parse(readFileSync(join(HERE, "ref_ast.json"), "utf8"));

const srcOf = new Map();
for (const p of corpus.positive) srcOf.set(p.name, p.src);
for (const n of corpus.negative) srcOf.set(n.name, n.src);

/**
 * Run `synopsis parse --json` on a source string.
 *
 * Returns {ok, tree} or {ok: false, error, message, line}. A non-zero
 * exit with parseable JSON on stdout is a REFUSAL, which is a normal
 * outcome; anything else is the binary failing, which is not.
 */
function runParse(src) {
  let stdout;
  try {
    stdout = execFileSync(BIN, ["parse", "--json", "-"], {
      input: src,
      encoding: "utf8",
      stdio: ["pipe", "pipe", "pipe"],
    });
  } catch (e) {
    if (e.stdout === undefined || e.stdout === "") {
      throw new Error(`binary failed with no JSON: ${e.stderr || e.message}`);
    }
    stdout = e.stdout;
  }
  const j = JSON.parse(stdout);
  return j.error === undefined ? { ok: true, tree: j } : { ok: false, ...j };
}

/** Stable stringify: key ORDER is part of what we are checking. */
const show = (v) => JSON.stringify(v, null, 1);

let pass = 0;
const failures = [];

for (const entry of oracle.entries) {
  const src = srcOf.get(entry.name);
  if (src === undefined) {
    failures.push([entry.name, "not present in corpus.json"]);
    continue;
  }

  let got;
  try {
    got = runParse(src);
  } catch (e) {
    failures.push([entry.name, String(e.message)]);
    continue;
  }

  if (entry.parse_error) {
    if (got.ok) {
      failures.push([entry.name, `expected ${entry.parse_error}, but it parsed`]);
    } else if (got.error !== entry.parse_error) {
      failures.push([entry.name, `raised ${got.error}, want ${entry.parse_error}`]);
    } else {
      pass += 1;
    }
    continue;
  }

  if (!got.ok) {
    failures.push([entry.name, `parse refused: ${got.error}: ${got.message}`]);
    continue;
  }
  if (show(got.tree) === show(entry.ast)) {
    pass += 1;
  } else {
    failures.push([entry.name, firstDiff(entry.ast, got.tree)]);
  }
}

// The forms whose absence from the grammar is what makes thm:total and
// thm:noexact properties of the LANGUAGE rather than of a later check.
let forbidPass = 0;
for (const f of oracle.must_not_parse) {
  let got;
  try {
    got = runParse(f.src);
  } catch (e) {
    failures.push([`must-not-parse:${f.name}`, String(e.message)]);
    continue;
  }
  if (got.ok) {
    failures.push([`must-not-parse:${f.name}`, "PARSED -- the grammar admits it"]);
  } else {
    forbidPass += 1;
  }
}

/** Walk two trees together and name the first place they differ. */
function firstDiff(want, got, path = "") {
  if (show(want) === show(got)) return null;
  const tw = Array.isArray(want) ? "array" : want === null ? "null" : typeof want;
  const tg = Array.isArray(got) ? "array" : got === null ? "null" : typeof got;
  if (tw !== tg) return `${path || "<root>"}: want ${tw}, got ${tg}`;
  if (tw === "array") {
    if (want.length !== got.length) {
      return `${path}: want ${want.length} items, got ${got.length}`;
    }
    for (let i = 0; i < want.length; i++) {
      const d = firstDiff(want[i], got[i], `${path}[${i}]`);
      if (d) return d;
    }
    return `${path}: arrays differ`;
  }
  if (tw === "object") {
    const kw = Object.keys(want);
    const kg = Object.keys(got);
    const missing = kw.filter((k) => !kg.includes(k));
    const extra = kg.filter((k) => !kw.includes(k));
    if (missing.length) return `${path}: missing field(s) ${missing.join(", ")}`;
    if (extra.length) return `${path}: extra field(s) ${extra.join(", ")}`;
    if (kw.join() !== kg.join()) {
      return `${path}: field ORDER differs -- want [${kw}], got [${kg}]`;
    }
    for (const k of kw) {
      const d = firstDiff(want[k], got[k], path ? `${path}.${k}` : k);
      if (d) return d;
    }
  }
  return `${path}: want ${JSON.stringify(want)}, got ${JSON.stringify(got)}`;
}

const total = oracle.entries.length + oracle.must_not_parse.length;
const ok = pass + forbidPass;

console.log(`stage A -- parser conformance (Rust)`);
console.log(`  trees + parse errors : ${pass}/${oracle.entries.length}`);
console.log(`  must-not-parse forms : ${forbidPass}/${oracle.must_not_parse.length}`);

if (failures.length) {
  console.log(`\n  ${failures.length} FAILURE(S):`);
  for (const [name, why] of failures) console.log(`    ${name}\n      ${why}`);
  process.exit(1);
}
console.log(`\n  all ${ok}/${total} green`);
