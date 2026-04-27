// Multi-channel matched filter for DNA sequence matching.
//
// This is the in-browser realisation of the algorithm described in the
// oscillator-interference-patterns paper. We channelise both the query and
// the target into four mean-centred one-hot indicator signals and compute
// the per-channel linear cross-correlation via the FFT, summing across
// channels to form a coherent multi-channel matched filter; then we
// normalise by the joint Frobenius norm of the query and the matching
// target window so the score lies in [-1, 1].
//
// The target FFTs and the rolling window energy are precomputed once per
// loaded sequence; per-query work is four forward FFTs of the query, four
// elementwise conjugate-multiplies, four inverse FFTs, and a sum.

import { fftRealToComplex, ifftToReal, conjMultiply, nextPow2 } from "./fft.js";

const DNA_ALPHABET = "ACGT";

/**
 * 4 x L mean-centred one-hot representation of a DNA sequence.
 *
 * @param {string} seq
 * @returns {Float64Array[]}
 */
export function dnaChannels(seq) {
  const s = seq.toUpperCase().replace(/[^ACGT]/g, "");
  const L = s.length;
  const ch = [
    new Float64Array(L),
    new Float64Array(L),
    new Float64Array(L),
    new Float64Array(L),
  ];
  for (let i = 0; i < L; i += 1) {
    const idx = DNA_ALPHABET.indexOf(s[i]);
    if (idx >= 0) ch[idx][i] = 1.0;
  }
  for (let c = 0; c < 4; c += 1) {
    if (L === 0) continue;
    let m = 0;
    for (let i = 0; i < L; i += 1) m += ch[c][i];
    m /= L;
    for (let i = 0; i < L; i += 1) ch[c][i] -= m;
  }
  return ch;
}

/**
 * Precomputed target context. Compute this once per loaded sequence.
 *
 * @typedef {object} TargetContext
 * @property {number} N            FFT length (power of two)
 * @property {number} Lt           target length in bp
 * @property {number} maxQueryLen  max query length supported (Lt + maxLq - 1 <= N)
 * @property {Float64Array[]} targetFFT          length-2N complex buffers, one per channel
 * @property {Float64Array}   rollingEnergyMax    rolling sqrt(sum_a sum_n T_a^2[k:k+maxLq]) (precomputed at maxLq)
 * @property {Float64Array}   targetSquaredCum    Lt+1 cumulative sums of sum_a T_a^2 (for any window length)
 */

/**
 * Precompute the per-channel FFT of the target sequence and the cumulative
 * sum of its squared channelised values.
 *
 * @param {string} targetSeq
 * @param {number} maxQueryLen  the longest query the user is allowed to enter
 * @returns {TargetContext}
 */
export function prepareTarget(targetSeq, maxQueryLen) {
  const T = dnaChannels(targetSeq);
  const Lt = T[0].length;
  const N = nextPow2(Lt + maxQueryLen - 1);

  const targetFFT = T.map((channel) => fftRealToComplex(channel, N));

  // Cumulative sum of sum_c T_c[i]^2, used to compute the per-window energy
  // for any query length up to maxQueryLen via two array lookups.
  const cum = new Float64Array(Lt + 1);
  for (let i = 0; i < Lt; i += 1) {
    let s = 0;
    for (let c = 0; c < 4; c += 1) {
      const v = T[c][i];
      s += v * v;
    }
    cum[i + 1] = cum[i] + s;
  }

  return { N, Lt, maxQueryLen, targetFFT, targetSquaredCum: cum };
}

/**
 * Run the matched filter for a query against a precomputed target context.
 * Returns the normalised cross-correlation $\rho(k)$ for $k = 0$ to
 * $L_t - L_q$, plus timing breakdown.
 *
 * @param {string} querySeq
 * @param {TargetContext} ctx
 * @returns {{ scores: Float64Array, lagCount: number, timings: object }}
 */
export function matchedFilterScan(querySeq, ctx) {
  const Q = dnaChannels(querySeq);
  const Lq = Q[0].length;
  const Lt = ctx.Lt;
  const lagCount = Lt - Lq + 1;
  if (lagCount <= 0) {
    return {
      scores: new Float64Array(0),
      lagCount: 0,
      timings: { embedMs: 0, fftMs: 0, prodMs: 0, ifftMs: 0, normMs: 0, totalMs: 0 },
    };
  }
  if (Lq > ctx.maxQueryLen) {
    throw new Error(`query of length ${Lq} exceeds maxQueryLen ${ctx.maxQueryLen}`);
  }

  const t0 = performance.now();

  // 1. Forward FFT of each query channel, padded to N.
  const tFft0 = performance.now();
  const queryFFT = Q.map((channel) => fftRealToComplex(channel, ctx.N));
  const fftMs = performance.now() - tFft0;

  // 2. Per-channel conj(Q) * T, elementwise.
  const tProd0 = performance.now();
  const product = queryFFT.map((qf, c) => {
    const out = new Float64Array(2 * ctx.N);
    return conjMultiply(qf, ctx.targetFFT[c], out);
  });
  const prodMs = performance.now() - tProd0;

  // 3. Inverse FFT each, take real part, sum across channels.
  const tIfft0 = performance.now();
  const summed = new Float64Array(lagCount);
  for (let c = 0; c < 4; c += 1) {
    const r = ifftToReal(product[c]);
    for (let k = 0; k < lagCount; k += 1) summed[k] += r[k];
  }
  const ifftMs = performance.now() - tIfft0;

  // 4. Normalise by ||Q||_F * ||T_window||_F.
  const tNorm0 = performance.now();
  let qEnergy = 0;
  for (let c = 0; c < 4; c += 1) {
    for (let i = 0; i < Lq; i += 1) qEnergy += Q[c][i] * Q[c][i];
  }
  const qNorm = Math.sqrt(qEnergy);
  const cum = ctx.targetSquaredCum;
  const out = new Float64Array(lagCount);
  if (qNorm > 0) {
    for (let k = 0; k < lagCount; k += 1) {
      const winSq = cum[k + Lq] - cum[k];
      const wNorm = Math.sqrt(Math.max(winSq, 0));
      if (wNorm > 0) out[k] = summed[k] / (qNorm * wNorm);
    }
  }
  const normMs = performance.now() - tNorm0;

  return {
    scores: out,
    lagCount,
    timings: {
      fftMs,
      prodMs,
      ifftMs,
      normMs,
      totalMs: performance.now() - t0,
    },
  };
}

/**
 * Robust mean and standard deviation of `scores`, optionally excluding a
 * window of indices.
 *
 * @param {Float64Array} scores
 * @param {Array<[number, number]>=} exclude  half-open ranges [a, b)
 * @returns {{mean: number, std: number}}
 */
export function backgroundStats(scores, exclude) {
  let sum = 0;
  let sum2 = 0;
  let count = 0;
  if (!exclude || exclude.length === 0) {
    for (let i = 0; i < scores.length; i += 1) {
      sum += scores[i];
      sum2 += scores[i] * scores[i];
      count += 1;
    }
  } else {
    const mask = new Uint8Array(scores.length);
    mask.fill(1);
    for (const [a, b] of exclude) {
      const lo = Math.max(0, a);
      const hi = Math.min(scores.length, b);
      for (let i = lo; i < hi; i += 1) mask[i] = 0;
    }
    for (let i = 0; i < scores.length; i += 1) {
      if (mask[i]) {
        sum += scores[i];
        sum2 += scores[i] * scores[i];
        count += 1;
      }
    }
  }
  if (count === 0) return { mean: 0, std: 1 };
  const mean = sum / count;
  const variance = Math.max(0, sum2 / count - mean * mean);
  return { mean, std: Math.sqrt(variance) || 1e-12 };
}

/**
 * Greedy maximum-suppression peak picking on a score (or z-score) array.
 *
 * @param {Float64Array} scores
 * @param {number} minScore        threshold (inclusive)
 * @param {number} minSeparation   minimum gap between accepted peaks (in lags)
 * @param {number} [maxPeaks=200]  cap to keep the loop bounded
 */
export function findPeaks(scores, minScore, minSeparation, maxPeaks = 200) {
  const masked = new Float64Array(scores);
  const peaks = [];
  for (let iter = 0; iter < maxPeaks; iter += 1) {
    let bestIdx = -1;
    let bestScore = -Infinity;
    for (let i = 0; i < masked.length; i += 1) {
      const s = masked[i];
      if (s > bestScore) {
        bestScore = s;
        bestIdx = i;
      }
    }
    if (bestIdx < 0 || bestScore < minScore) break;
    peaks.push({ index: bestIdx, score: bestScore });
    const lo = Math.max(0, bestIdx - minSeparation);
    const hi = Math.min(masked.length, bestIdx + minSeparation + 1);
    for (let i = lo; i < hi; i += 1) masked[i] = -Infinity;
  }
  return peaks;
}
