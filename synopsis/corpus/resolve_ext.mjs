// A loader hook that lets Node import the VENDORED front-end.
//
// `sync.mjs` strips the `.ts` from relative import specifiers on its way
// into vivid-symbolism/src/lib/synopsis, because webpack resolves
// extensionless specifiers and Node does not. That rewrite is correct
// for the bundle and fatal for anything that wants to run those files
// directly -- which is why `check_ide.mts` used to carry its own copy of
// analyse() instead of importing the page's.
//
// The copy went stale, as copies do. This hook is the alternative: it
// re-adds the extension at resolution time, so the conformance runner
// exercises the SAME analyse() the page ships rather than a lookalike.
// Nothing is written and no vendored file changes; the rewrite is
// undone only for the duration of the check.
//
// Used as:  node --experimental-strip-types --import ./resolve_ext.mjs <script>

import { register } from "node:module";
import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

/**
 * Re-add `.ts` to a relative specifier that has no extension, but only
 * when the file is actually there. The guard matters: `./tutorial.js`
 * and `./analyse.js` are real JavaScript and must resolve as written,
 * so this must not turn every bare specifier into a `.ts` lookup.
 */
export function resolve(spec, ctx, next) {
  if (spec.startsWith(".") && !/\.[a-z]+$/i.test(spec)) {
    const candidate = new URL(`${spec}.ts`, ctx.parentURL);
    if (existsSync(fileURLToPath(candidate))) return next(`${spec}.ts`, ctx);
  }
  return next(spec, ctx);
}

// Self-registering: passing this file to --import both runs it and
// installs it, so the caller needs one flag rather than two files.
if (!process.env["SYNOPSIS_RESOLVE_EXT"]) {
  process.env["SYNOPSIS_RESOLVE_EXT"] = "1";
  register(import.meta.url, pathToFileURL("./"));
}
