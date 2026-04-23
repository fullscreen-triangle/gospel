// Spectral sequence embedding, direct port of the Python reference in
// publications/shader-based-homology/experiments/common.py.
//
// We compute only the first K low-frequency DFT magnitudes per channel
// via a direct DFT, which is O(K * L) and plenty fast for interactive
// queries of protein sequences up to a few thousand residues.

import { DNA_ALPHABET, HYDROPATHY, VOLUME, CHARGE } from "./alphabets.js";

/**
 * Produce the channelised signal matrix for a sequence, mean-centred per channel.
 *
 * @param {string} seq
 * @param {"dna"|"protein"} kind
 * @returns {{ channels: Float32Array[], length: number }}
 */
export function channelise(seq, kind) {
  const s = seq.toUpperCase().replace(/[^A-Z]/g, "");
  const L = s.length;
  let c;
  let channels;

  if (kind === "dna") {
    c = 4;
    channels = [
      new Float32Array(L),
      new Float32Array(L),
      new Float32Array(L),
      new Float32Array(L),
    ];
    for (let i = 0; i < L; i += 1) {
      const idx = DNA_ALPHABET.indexOf(s[i]);
      if (idx >= 0) channels[idx][i] = 1.0;
    }
  } else if (kind === "protein") {
    c = 3;
    channels = [new Float32Array(L), new Float32Array(L), new Float32Array(L)];
    for (let i = 0; i < L; i += 1) {
      const aa = s[i];
      if (HYDROPATHY[aa] !== undefined) {
        channels[0][i] = HYDROPATHY[aa];
        channels[1][i] = VOLUME[aa];
        channels[2][i] = CHARGE[aa];
      }
    }
  } else {
    throw new Error(`unknown kind: ${kind}`);
  }

  // Subtract per-channel mean so the DC Fourier bin is empty and carries no info.
  for (let ch = 0; ch < c; ch += 1) {
    if (L === 0) continue;
    let mean = 0;
    const x = channels[ch];
    for (let i = 0; i < L; i += 1) mean += x[i];
    mean /= L;
    for (let i = 0; i < L; i += 1) x[i] -= mean;
  }

  return { channels, length: L };
}

/**
 * Direct DFT magnitude at bin k for a real-valued signal of length L,
 * using the same convention as numpy.fft.rfft.
 *
 * @param {Float32Array} x
 * @param {number} k
 * @returns {number}
 */
function dftMagnitude(x, k) {
  const L = x.length;
  if (L === 0) return 0;
  const phase = (-2 * Math.PI * k) / L;
  let re = 0;
  let im = 0;
  for (let n = 0; n < L; n += 1) {
    const v = x[n];
    const theta = phase * n;
    re += v * Math.cos(theta);
    im += v * Math.sin(theta);
  }
  return Math.hypot(re, im);
}

/**
 * Compute the L2-normalised spectral embedding vector for a sequence.
 *
 * @param {string} seq
 * @param {number} K - number of non-DC coefficients per channel to keep
 * @param {"dna"|"protein"} kind
 * @returns {Float32Array}
 */
export function spectralEmbedding(seq, K, kind) {
  const { channels, length } = channelise(seq, kind);
  const c = channels.length;
  const out = new Float32Array(c * K);

  const Lmax = Math.floor(length / 2);
  for (let ch = 0; ch < c; ch += 1) {
    const x = channels[ch];
    for (let k = 0; k < K; k += 1) {
      const bin = k + 1; // skip DC
      if (bin > Lmax) {
        out[ch * K + k] = 0;
        continue;
      }
      out[ch * K + k] = dftMagnitude(x, bin) / Math.max(length, 1);
    }
  }

  // L2-normalise so cosine similarity equals dot product.
  let norm = 0;
  for (let i = 0; i < out.length; i += 1) norm += out[i] * out[i];
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < out.length; i += 1) out[i] /= norm;
  }
  return out;
}

/**
 * Scan a packed database (flat Float32Array of shape N x d) by computing
 * cosine similarity to a query and returning the top-K most similar indices.
 *
 * @param {Float32Array} dbFlat
 * @param {number} dim
 * @param {Float32Array} query
 * @param {number} topK
 * @returns {Array<{ index: number, score: number }>}
 */
export function shaderKernelTopK(dbFlat, dim, query, topK) {
  const N = dbFlat.length / dim;
  const scores = new Float32Array(N);
  for (let i = 0; i < N; i += 1) {
    let s = 0;
    const base = i * dim;
    for (let j = 0; j < dim; j += 1) s += dbFlat[base + j] * query[j];
    scores[i] = s;
  }

  // Simple partial selection sort for top-K (fine for N up to ~10^5 on demo).
  const k = Math.min(topK, N);
  const best = new Array(k).fill(null).map(() => ({ index: -1, score: -Infinity }));
  for (let i = 0; i < N; i += 1) {
    const sc = scores[i];
    if (sc > best[k - 1].score) {
      let pos = k - 1;
      while (pos > 0 && best[pos - 1].score < sc) {
        best[pos] = best[pos - 1];
        pos -= 1;
      }
      best[pos] = { index: i, score: sc };
    }
  }
  return best.filter((e) => e.index >= 0);
}
