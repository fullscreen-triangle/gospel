// Loader for the bundled synthetic locus.
//
// The browser tool needs three things to display a chromosome track:
//   1. The metadata (length, window size/stride, planted feature annotations)
//   2. The matrix of windowed spectral embeddings, packed as a Float32 binary
//   3. The raw FASTA sequence so we can show the local sequence at any cursor

const META_URL = "/data/locus.meta.json";
const EMBED_URL = "/data/locus.embeddings.bin";
const FASTA_URL = "/data/locus.fasta";

let cache = null;

function parseFasta(text) {
  const lines = text.split(/\r?\n/);
  const out = [];
  for (let i = 0; i < lines.length; i += 1) {
    const ln = lines[i];
    if (!ln || ln.startsWith(">")) continue;
    out.push(ln.trim());
  }
  return out.join("");
}

/**
 * Fetch all three locus assets in parallel and return them as a single
 * structured object. Cached after the first call.
 */
export async function loadLocus() {
  if (cache) return cache;

  const [meta, embedBuf, fastaText] = await Promise.all([
    fetch(META_URL).then((r) => {
      if (!r.ok) throw new Error(`failed to load locus meta: ${r.status}`);
      return r.json();
    }),
    fetch(EMBED_URL).then((r) => {
      if (!r.ok) throw new Error(`failed to load locus embeddings: ${r.status}`);
      return r.arrayBuffer();
    }),
    fetch(FASTA_URL).then((r) => {
      if (!r.ok) throw new Error(`failed to load locus fasta: ${r.status}`);
      return r.text();
    }),
  ]);

  const flat = new Float32Array(embedBuf);
  const expected = meta.n_windows * meta.embed_dim;
  if (flat.length !== expected) {
    throw new Error(
      `locus embedding length mismatch: got ${flat.length}, expected ${expected}`
    );
  }

  const sequence = parseFasta(fastaText);
  if (sequence.length !== meta.length) {
    console.warn(
      `locus sequence length ${sequence.length} != meta.length ${meta.length}`
    );
  }

  cache = {
    meta,
    flat,
    sequence,
    nWindows: meta.n_windows,
    dim: meta.embed_dim,
    windowSize: meta.window_size,
    windowStride: meta.window_stride,
  };
  return cache;
}

/** Map a window index to genomic position. */
export function windowStart(meta, idx) {
  return idx * meta.window_stride;
}

export function windowCenter(meta, idx) {
  return idx * meta.window_stride + meta.window_size / 2;
}
