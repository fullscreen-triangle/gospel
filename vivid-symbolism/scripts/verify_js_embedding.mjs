// Smoke-test: the JavaScript spectralEmbedding must produce coordinates
// numerically close to the Python reference stored in the sample database.
// We pick the first ten sequences from the protein sample DB, re-embed them
// with the JS implementation, and report the maximum per-component deviation.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

// Dynamic import of the JS library (pure ESM). On Windows we must
// convert the absolute path into a file:// URL.
const libPath = path.join(here, "..", "src", "lib", "embedding.js");
const { spectralEmbedding } = await import(pathToFileURL(libPath).href);

function load(filename) {
  const p = path.join(here, "..", "public", "data", filename);
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function compare(db, limit = 10) {
  const d = db.embed_dim;
  let maxAbs = 0;
  let maxRel = 0;
  let worstId = null;
  for (let i = 0; i < Math.min(limit, db.sequences.length); i += 1) {
    const seq = db.sequences[i];
    const py = new Float32Array(db.embeddings_flat.slice(i * d, (i + 1) * d));
    const js = spectralEmbedding(seq.text, db.n_coefficients, db.kind);
    let localAbs = 0;
    for (let k = 0; k < d; k += 1) {
      const diff = Math.abs(py[k] - js[k]);
      if (diff > localAbs) localAbs = diff;
    }
    const scale = Math.max(1e-6, ...py.map(Math.abs));
    const rel = localAbs / scale;
    if (localAbs > maxAbs) {
      maxAbs = localAbs;
      maxRel = rel;
      worstId = seq.id;
    }
  }
  return { maxAbs, maxRel, worstId };
}

for (const name of ["sample_db_protein.json", "sample_db_dna.json"]) {
  const db = load(name);
  const { maxAbs, maxRel, worstId } = compare(db);
  console.log(
    `${name}: embed_dim=${db.embed_dim}, ` +
      `max |JS - Py| = ${maxAbs.toExponential(3)}, ` +
      `rel = ${maxRel.toExponential(3)} (worst: ${worstId})`
  );
}
