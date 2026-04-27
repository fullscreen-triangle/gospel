// Quick end-to-end sanity check: load the locus binary, embed the canonical
// demo motif, scan, and confirm that the top hits land on the two planted
// motif positions (500000 and 800000 bp).

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const libPath = path.join(here, "..", "src", "lib", "embedding.js");
const { spectralEmbedding, shaderKernelScan } = await import(pathToFileURL(libPath).href);

const meta = JSON.parse(
  fs.readFileSync(path.join(here, "..", "public", "data", "locus.meta.json"), "utf8")
);
const bin = fs.readFileSync(path.join(here, "..", "public", "data", "locus.embeddings.bin"));
const flat = new Float32Array(bin.buffer, bin.byteOffset, bin.byteLength / 4);

console.log("locus:", meta.length, "bp,", meta.n_windows, "windows");
console.log("query length:", meta.demo_query.length);

const t0 = Date.now();
const q = spectralEmbedding(meta.demo_query, meta.n_coefficients, "dna");
const tEmbed = Date.now() - t0;

const t1 = Date.now();
const { topK } = shaderKernelScan(flat, meta.embed_dim, q, 10);
const tScan = Date.now() - t1;

console.log(`embedding: ${tEmbed} ms, scan: ${tScan} ms over ${flat.length / meta.embed_dim} windows`);
console.log("\ntop 10 hits:");
for (let i = 0; i < topK.length; i += 1) {
  const h = topK[i];
  const bp = h.index * meta.window_stride;
  console.log(
    `  #${i + 1}  window ${String(h.index).padStart(5)}  ` +
      `position ${bp.toLocaleString().padStart(9)} bp  ` +
      `cosine ${h.score.toFixed(4)}`
  );
}

const PLANTED = [500_000, 800_000];
const matched = PLANTED.map((pos) => {
  const idx = Math.round(pos / meta.window_stride);
  const rank = topK.findIndex((h) => Math.abs(h.index - idx) <= 2);
  return { pos, expectedIdx: idx, rank };
});
console.log("\nplanted-feature recovery:");
for (const m of matched) {
  console.log(
    `  ${m.pos.toLocaleString()} bp  expected window ~${m.expectedIdx}  ` +
      `found at rank ${m.rank >= 0 ? m.rank + 1 : "MISS"}`
  );
}
