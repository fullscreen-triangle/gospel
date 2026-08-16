// Re-vendor the synopsis TypeScript front-end into the web tool.
//
//   node src/lib/synopsis/sync.mjs
//
// The parser here is a COPY, and the copy direction is one-way:
// synopsis/ts/src -> here. Never edit these files in place. The
// originals are the implementation that passes 24/24 against the
// conformance oracle in synopsis/corpus; a divergent edit on this side
// would create a third implementation of the language with no runner
// holding it to the reference, which is exactly what the corpus exists
// to prevent.
//
// The one transformation applied is mechanical: the originals import
// with explicit `.ts` extensions, because they run under Node's native
// type stripping where the extension is required. webpack resolves
// extensionless specifiers instead, so the extension is removed on the
// way in. Nothing else changes.

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(HERE, "..", "..", "..", "..", "synopsis", "ts", "src");

const FILES = ["errors.ts", "tokens.ts", "ast.ts", "parser.ts"];

if (!existsSync(SRC)) {
  console.error(`source not found: ${SRC}`);
  process.exit(2);
}

const BANNER =
  "// VENDORED from synopsis/ts/src -- do not edit here.\n" +
  "// Re-sync with: node src/lib/synopsis/sync.mjs\n";

for (const f of FILES) {
  const text = readFileSync(join(SRC, f), "utf8");
  // Drop the `.ts` from relative import specifiers only.
  const rewritten = text.replace(/(from\s+"\.\/[A-Za-z0-9_-]+)\.ts"/g, '$1"');
  writeFileSync(join(HERE, f), BANNER + rewritten);
  const n = (text.match(/from\s+"\.\/[A-Za-z0-9_-]+\.ts"/g) || []).length;
  console.log(`  ${f.padEnd(12)} ${n} import(s) rewritten`);
}
console.log(`\nvendored ${FILES.length} files from ${SRC}`);
