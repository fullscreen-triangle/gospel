// Stage A conformance: does the TypeScript parser build the same trees
// as the reference?
//
// Not "does it parse without crashing" -- byte-identical trees. A field
// that differs here is a divergence the checker would later turn into a
// wrong verdict, and the whole point of two implementations of one
// language is that this cannot happen quietly.
//
// Run:  node synopsis/corpus/run_parse.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));

// pathToFileURL, not a bare path: on Windows an absolute path starts
// `c:\` and the ESM loader reads `c:` as a URL scheme.
const mod = (...p) => pathToFileURL(join(HERE, "..", "ts", "src", ...p)).href;

const { parse } = await import(mod("parser.ts"));
const { SynopsisError } = await import(mod("errors.ts"));

const corpus = JSON.parse(readFileSync(join(HERE, "corpus.json"), "utf8"));
const oracle = JSON.parse(readFileSync(join(HERE, "ref_ast.json"), "utf8"));

const srcOf = new Map();
for (const p of corpus.positive) srcOf.set(p.name, p.src);
for (const n of corpus.negative) srcOf.set(n.name, n.src);

/**
 * Canonicalise a TS AST into the reference's JSON shape.
 *
 * Params are Maps here and ordered pair-lists there; integral numbers
 * are ints there. Both are deliberate -- see dump_ref_ast.py.
 */
function canon(x) {
  if (x === null || x === undefined) return null;
  if (x instanceof Map) {
    return [...x.entries()].map(([k, v]) => [k, canon(v)]);
  }
  if (Array.isArray(x)) return x.map(canon);
  if (typeof x === "number") return Number.isInteger(x) ? x : x;
  if (typeof x === "object") {
    const out = {};
    // `node` first, then declaration order -- matching the reference,
    // whose `line` is declared on the base class.
    out.node = x.node;
    out.line = x.line;
    for (const [k, v] of Object.entries(x)) {
      if (k === "node" || k === "line") continue;
      out[k] = canon(v);
    }
    return out;
  }
  return x;
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

  if (entry.parse_error) {
    // Must fail to parse, with the same class.
    try {
      parse(src);
      failures.push([entry.name, `expected ${entry.parse_error}, but it parsed`]);
    } catch (e) {
      if (!(e instanceof SynopsisError)) {
        failures.push([entry.name, `threw a non-Synopsis error: ${e}`]);
      } else if (e.className !== entry.parse_error) {
        failures.push([entry.name, `raised ${e.className}, want ${entry.parse_error}`]);
      } else {
        pass += 1;
      }
    }
    continue;
  }

  // Must parse, to exactly the reference tree.
  let got;
  try {
    got = canon(parse(src));
  } catch (e) {
    failures.push([entry.name, `parse threw: ${e}`]);
    continue;
  }
  const want = entry.ast;
  if (show(got) === show(want)) {
    pass += 1;
  } else {
    failures.push([entry.name, firstDiff(want, got)]);
  }
}

// The forms whose absence from the grammar is what makes thm:total and
// thm:noexact properties of the LANGUAGE rather than of a later check.
let forbidPass = 0;
for (const f of oracle.must_not_parse) {
  try {
    parse(f.src);
    failures.push([`must-not-parse:${f.name}`, "PARSED -- the grammar admits it"]);
  } catch (e) {
    if (e instanceof SynopsisError) forbidPass += 1;
    else failures.push([`must-not-parse:${f.name}`, `non-Synopsis error: ${e}`]);
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

console.log(`stage A -- parser conformance (TypeScript)`);
console.log(`  trees + parse errors : ${pass}/${oracle.entries.length}`);
console.log(`  must-not-parse forms : ${forbidPass}/${oracle.must_not_parse.length}`);

if (failures.length) {
  console.log(`\n  ${failures.length} FAILURE(S):`);
  for (const [name, why] of failures) console.log(`    ${name}\n      ${why}`);
  process.exit(1);
}
console.log(`\n  all ${ok}/${total} green`);
