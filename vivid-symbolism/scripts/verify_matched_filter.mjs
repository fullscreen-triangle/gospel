// Sanity-check the JS matched filter against the Python reference. We
// generate a small DNA target with two planted motifs (one mutated) by
// reading the bundled locus, run the JS filter, and confirm:
//   - the global argmax is at the canonical motif position (500,000 bp)
//   - the second peak is at the 10%-mutated copy (800,000 bp)
//   - the FFT result agrees with a naive sliding inner product to ~1e-10
//
// This is the same workflow used in `scripts/verify_locus.mjs` for the
// browser tool, adapted for the matched-filter library.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const mfPath = path.join(here, "..", "src", "lib", "matched_filter.js");
const { dnaChannels, prepareTarget, matchedFilterScan, backgroundStats, findPeaks }
  = await import(pathToFileURL(mfPath).href);

const meta = JSON.parse(
  fs.readFileSync(path.join(here, "..", "public", "data", "locus.meta.json"), "utf8")
);
const fasta = fs.readFileSync(
  path.join(here, "..", "public", "data", "locus.fasta"),
  "utf8"
);
const target = fasta.split(/\r?\n/).filter((l) => l && !l.startsWith(">")).join("");
console.log("locus length:", target.length, "bp; demo query length:", meta.demo_query.length);

const t0 = Date.now();
const ctx = prepareTarget(target, 4096);
console.log(`prepareTarget: ${Date.now() - t0} ms (FFT length ${ctx.N})`);

const t1 = Date.now();
const result = matchedFilterScan(meta.demo_query, ctx);
console.log(`matchedFilterScan: ${Date.now() - t1} ms`);
console.log("  fft", result.timings.fftMs.toFixed(1), "ms");
console.log("  product", result.timings.prodMs.toFixed(1), "ms");
console.log("  ifft", result.timings.ifftMs.toFixed(1), "ms");
console.log("  norm", result.timings.normMs.toFixed(1), "ms");
console.log("  total", result.timings.totalMs.toFixed(1), "ms");

const { mean, std } = backgroundStats(result.scores);
console.log(`background: mean=${mean.toExponential(3)} std=${std.toExponential(3)}`);

// Convert raw rho to z-scores
const z = new Float64Array(result.scores.length);
for (let i = 0; i < z.length; i += 1) z[i] = (result.scores[i] - mean) / std;

const peaks = findPeaks(z, 5.0, 200);
console.log(`\ntop ${peaks.length} peaks (z >= 5):`);
for (let i = 0; i < Math.min(10, peaks.length); i += 1) {
  const p = peaks[i];
  console.log(
    `  #${i + 1}  position ${p.index.toLocaleString().padStart(9)} bp  ` +
      `rho=${result.scores[p.index].toFixed(4)}  z=${p.score.toFixed(2)}`
  );
}

// Compare top two peaks against the planted positions.
const planted = [500_000, 800_000];
console.log("\nplanted-position recovery:");
for (const pos of planted) {
  const matched = peaks.findIndex((p) => Math.abs(p.index - pos) <= 5);
  console.log(
    `  ${pos.toLocaleString()} bp  -> ` +
      (matched >= 0
        ? `rank ${matched + 1} at offset ${peaks[matched].index - pos} bp, ` +
          `rho=${result.scores[peaks[matched].index].toFixed(4)}`
        : "MISS")
  );
}

// Cross-check FFT vs naive on a small slice.
const naiveTarget = target.slice(0, 5000);
const naiveQuery = meta.demo_query;
const naiveCtx = prepareTarget(naiveTarget, 4096);
const naiveResult = matchedFilterScan(naiveQuery, naiveCtx);

// Naive single-channel xcorr for sanity (just first channel A)
const Q = dnaChannels(naiveQuery);
const T = dnaChannels(naiveTarget);
const Lq = Q[0].length;
const Lt = T[0].length;
const corrLen = Lt - Lq + 1;
const naiveSummed = new Float64Array(corrLen);
for (let c = 0; c < 4; c += 1) {
  for (let k = 0; k < corrLen; k += 1) {
    let s = 0;
    for (let n = 0; n < Lq; n += 1) s += Q[c][n] * T[c][k + n];
    naiveSummed[k] += s;
  }
}
// Normalise by the same factors used inside matchedFilterScan
let qe = 0;
for (let c = 0; c < 4; c += 1) for (let i = 0; i < Lq; i += 1) qe += Q[c][i] * Q[c][i];
const qn = Math.sqrt(qe);
const cum = naiveCtx.targetSquaredCum;
const naiveRho = new Float64Array(corrLen);
for (let k = 0; k < corrLen; k += 1) {
  const wn = Math.sqrt(Math.max(cum[k + Lq] - cum[k], 0));
  if (qn > 0 && wn > 0) naiveRho[k] = naiveSummed[k] / (qn * wn);
}

let maxAbs = 0;
for (let k = 0; k < corrLen; k += 1) {
  const d = Math.abs(naiveRho[k] - naiveResult.scores[k]);
  if (d > maxAbs) maxAbs = d;
}
console.log(`\nFFT vs naive on Lt=5,000 slice: max |diff| = ${maxAbs.toExponential(3)}`);
