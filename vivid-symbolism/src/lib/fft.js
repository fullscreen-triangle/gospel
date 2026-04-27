// Radix-2 in-place Cooley-Tukey FFT.
//
// All operations work on a Float64Array of length 2N: complex numbers are
// stored interleaved [re_0, im_0, re_1, im_1, ...]. Forward direction is the
// usual X[k] = sum_n x[n] exp(-2 pi i k n / N); inverse divides by N.
//
// The code is written once for clarity; everything else in this directory
// (the matched filter, the rolling-window normaliser) builds on it.

/**
 * In-place complex FFT.
 *
 * @param {Float64Array} buf  interleaved complex array, length 2N (N a power of 2)
 * @param {number} sign       -1 for forward, +1 for inverse
 */
export function fftInPlace(buf, sign) {
  const N = buf.length >>> 1;
  if (N === 0) return buf;
  if ((N & (N - 1)) !== 0) {
    throw new Error(`fft length must be a power of two, got ${N}`);
  }

  // bit-reversal permutation
  for (let i = 1, j = 0; i < N; i += 1) {
    let bit = N >>> 1;
    while (j & bit) {
      j ^= bit;
      bit >>>= 1;
    }
    j ^= bit;
    if (i < j) {
      const ii = i << 1;
      const jj = j << 1;
      let t = buf[ii];     buf[ii] = buf[jj];     buf[jj] = t;
      t = buf[ii + 1];     buf[ii + 1] = buf[jj + 1]; buf[jj + 1] = t;
    }
  }

  // butterflies
  for (let len = 2; len <= N; len <<= 1) {
    const half = len >>> 1;
    const ang = (sign * 2 * Math.PI) / len;
    const wRe0 = Math.cos(ang);
    const wIm0 = Math.sin(ang);
    for (let i = 0; i < N; i += len) {
      let pRe = 1.0;
      let pIm = 0.0;
      for (let k = 0; k < half; k += 1) {
        const a = (i + k) << 1;
        const b = (i + k + half) << 1;
        const bRe = buf[b];
        const bIm = buf[b + 1];
        const tRe = pRe * bRe - pIm * bIm;
        const tIm = pRe * bIm + pIm * bRe;
        buf[b]     = buf[a]     - tRe;
        buf[b + 1] = buf[a + 1] - tIm;
        buf[a]     = buf[a]     + tRe;
        buf[a + 1] = buf[a + 1] + tIm;
        const nRe = pRe * wRe0 - pIm * wIm0;
        const nIm = pRe * wIm0 + pIm * wRe0;
        pRe = nRe;
        pIm = nIm;
      }
    }
  }

  if (sign === 1) {
    const inv = 1 / N;
    for (let i = 0; i < buf.length; i += 1) buf[i] *= inv;
  }
  return buf;
}

/**
 * Forward FFT of a real signal of length L into a fresh complex buffer of
 * length 2N (with N >= L), zero-padded.
 */
export function fftRealToComplex(real, N) {
  if ((N & (N - 1)) !== 0) {
    throw new Error("N must be a power of two");
  }
  const buf = new Float64Array(2 * N);
  const L = Math.min(real.length, N);
  for (let i = 0; i < L; i += 1) buf[2 * i] = real[i];
  fftInPlace(buf, -1);
  return buf;
}

/**
 * Inverse FFT, returns the real part as a freshly-allocated Float64Array of
 * length N. The complex buffer `buf` is mutated.
 */
export function ifftToReal(buf) {
  fftInPlace(buf, +1);
  const N = buf.length >>> 1;
  const out = new Float64Array(N);
  for (let i = 0; i < N; i += 1) out[i] = buf[2 * i];
  return out;
}

/**
 * conj(A) * B element-wise, written into `out`. All three are length-2N
 * complex buffers in interleaved layout. `out` may alias `a` or `b`.
 */
export function conjMultiply(a, b, out) {
  const N = a.length >>> 1;
  for (let i = 0; i < N; i += 1) {
    const aRe = a[2 * i];
    const aIm = a[2 * i + 1];
    const bRe = b[2 * i];
    const bIm = b[2 * i + 1];
    out[2 * i]     = aRe * bRe + aIm * bIm;
    out[2 * i + 1] = aRe * bIm - aIm * bRe;
  }
  return out;
}

/** Smallest power of two greater than or equal to n. */
export function nextPow2(n) {
  if (n <= 1) return 1;
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}
