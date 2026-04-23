import Head from "next/head";
import { useEffect, useMemo, useRef, useState } from "react";

import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import { spectralEmbedding, shaderKernelTopK } from "@/lib/embedding";

const DEMO_DNA = `ATGCGTACCTGATCGTACGTAGCTAGCTACGATCGATCGATCGTACGTAGCTAGCTACGA
TCGATCGATCGTACGTAGCTAGCTACGATCGATCGATCGTACGTAGCTAGCTACGATCGA
TCGATCGTACGTAGCTAGCTACGATCGATCGATCGTACGTAGCTAGCTACGATCGATCGA`;

const DEMO_PROTEIN = `MKWVTFISLLLLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLI
AFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKL
CTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVM`;

const ALLOWED = {
  dna: /[^ACGTN]/gi,
  protein: /[^ACDEFGHIKLMNPQRSTVWYBXZJU]/gi,
};

function cleanSequence(raw, kind) {
  const upper = raw.toUpperCase().replace(/\s+/g, "");
  return upper.replace(ALLOWED[kind], "");
}

function downloadText(name, content) {
  const blob = new Blob([content], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

export default function Search() {
  const [kind, setKind] = useState("protein");
  const [query, setQuery] = useState(DEMO_PROTEIN);
  const [topK, setTopK] = useState(10);
  const [db, setDb] = useState(null);
  const [dbLoading, setDbLoading] = useState(true);
  const [dbError, setDbError] = useState(null);
  const [results, setResults] = useState(null);
  const [lastTiming, setLastTiming] = useState(null);
  const dbCache = useRef({});

  useEffect(() => {
    let cancelled = false;
    setDbLoading(true);
    setDbError(null);
    const cached = dbCache.current[kind];
    if (cached) {
      setDb(cached);
      setDbLoading(false);
      return () => {
        cancelled = true;
      };
    }

    const url = kind === "dna" ? "/data/sample_db_dna.json" : "/data/sample_db_protein.json";
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`failed to load database: ${r.status}`);
        return r.json();
      })
      .then((data) => {
        if (cancelled) return;
        const flat = new Float32Array(data.embeddings_flat);
        const prepared = {
          kind: data.kind,
          nCoefficients: data.n_coefficients,
          dim: data.embed_dim,
          sequences: data.sequences,
          flat,
        };
        dbCache.current[kind] = prepared;
        setDb(prepared);
        setDbLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setDbError(err.message);
        setDbLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [kind]);

  const cleanedQuery = useMemo(() => cleanSequence(query, kind), [query, kind]);

  const run = () => {
    if (!db || !cleanedQuery || cleanedQuery.length < 10) {
      setResults(null);
      return;
    }
    const t0 = performance.now();
    const qv = spectralEmbedding(cleanedQuery, db.nCoefficients, kind);
    const tEmbed = performance.now() - t0;

    const t1 = performance.now();
    const hits = shaderKernelTopK(db.flat, db.dim, qv, topK);
    const tShader = performance.now() - t1;

    const enriched = hits.map((h) => {
      const s = db.sequences[h.index];
      return {
        ...h,
        id: s.id,
        family: s.family,
        length: s.length,
        text: s.text,
        distance: Math.max(0, 1 - h.score),
      };
    });
    setResults(enriched);
    setLastTiming({
      embedMs: tEmbed,
      shaderMs: tShader,
      totalMs: tEmbed + tShader,
      dim: qv.length,
      querySeqLen: cleanedQuery.length,
      dbSize: db.sequences.length,
    });
  };

  const pasteDemo = () => {
    setQuery(kind === "dna" ? DEMO_DNA : DEMO_PROTEIN);
  };

  const clear = () => {
    setQuery("");
    setResults(null);
    setLastTiming(null);
  };

  return (
    <>
      <Head>
        <title>Shader Homology Search &middot; Gospel</title>
        <meta
          name="description"
          content="Client-side shader-based homology search over a spectral coordinate embedding. Paste a DNA or protein sequence and scan a sample database in milliseconds."
        />
      </Head>

      <TransitionEffect />
      <main className="mb-16 flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Homology Search"
            className="mb-12 !text-7xl !leading-tight lg:!text-6xl sm:mb-6 sm:!text-5xl xs:!text-3xl"
          />

          <p className="mb-8 max-w-3xl text-base font-medium dark:text-light md:text-sm">
            Paste a DNA or protein sequence, then scan a bundled sample
            database through the spectral-embedding kernel. The entire search
            runs in your browser. No data leaves your device.
          </p>

          <section className="grid grid-cols-12 gap-8 lg:gap-6">
            <div className="col-span-8 lg:col-span-12">
              <div className="mb-4 flex items-center gap-4 sm:flex-wrap">
                <label className="flex items-center gap-2 text-base font-semibold">
                  Kind
                  <select
                    value={kind}
                    onChange={(e) => setKind(e.target.value)}
                    className="rounded-md border-2 border-dark bg-light px-3 py-1 text-base font-medium
                      dark:border-light dark:bg-dark dark:text-light"
                  >
                    <option value="protein">protein</option>
                    <option value="dna">DNA</option>
                  </select>
                </label>

                <label className="flex items-center gap-2 text-base font-semibold">
                  Top
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value, 10) || 10)}
                    className="w-20 rounded-md border-2 border-dark bg-light px-3 py-1 text-base font-medium
                      dark:border-light dark:bg-dark dark:text-light"
                  />
                </label>

                <button
                  type="button"
                  onClick={pasteDemo}
                  className="rounded-md border-2 border-dark px-3 py-1 text-base font-semibold
                    hover:bg-dark hover:text-light dark:border-light dark:hover:bg-light dark:hover:text-dark"
                >
                  Load demo
                </button>
                <button
                  type="button"
                  onClick={clear}
                  className="rounded-md border-2 border-dark px-3 py-1 text-base font-semibold
                    hover:bg-dark hover:text-light dark:border-light dark:hover:bg-light dark:hover:text-dark"
                >
                  Clear
                </button>
              </div>

              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                spellCheck={false}
                rows={10}
                className="w-full rounded-xl border-2 border-dark bg-light p-4 font-mono text-sm leading-relaxed
                  focus:outline-none focus:ring-2 focus:ring-primary
                  dark:border-light dark:bg-dark dark:text-light"
                placeholder={kind === "dna" ? "paste a DNA sequence (ACGT)" : "paste a protein sequence (single-letter amino acids)"}
              />

              <div className="mt-2 flex items-center justify-between text-sm font-medium text-dark/80 dark:text-light/70">
                <span>
                  sequence length: <strong>{cleanedQuery.length}</strong>
                  {cleanedQuery.length < 10 ? " (minimum 10)" : ""}
                </span>
                <span>
                  {dbLoading
                    ? "loading database..."
                    : dbError
                    ? `database error: ${dbError}`
                    : `database: ${db.sequences.length} ${db.kind} sequences, ${db.dim}-dim embedding`}
                </span>
              </div>

              <div className="mt-4 flex gap-3">
                <button
                  type="button"
                  onClick={run}
                  disabled={dbLoading || !db || cleanedQuery.length < 10}
                  className="rounded-lg bg-dark px-6 py-2 text-lg font-semibold text-light
                    hover:bg-primary disabled:cursor-not-allowed disabled:opacity-40
                    dark:bg-light dark:text-dark dark:hover:bg-primaryDark"
                >
                  Run search
                </button>
              </div>
            </div>

            <aside className="col-span-4 flex flex-col gap-4 rounded-xl border-2 border-dark p-5
              dark:border-light lg:col-span-12">
              <h3 className="text-xl font-bold">Pipeline</h3>
              <ol className="list-decimal space-y-2 pl-5 text-sm font-medium">
                <li>
                  channelise the sequence into {kind === "dna" ? "4 one-hot" : "3 physicochemical"} signals
                </li>
                <li>
                  compute the first {db ? db.nCoefficients : 12} non-DC DFT magnitudes per channel
                </li>
                <li>
                  length-normalise and L2-normalise the concatenated coefficients
                </li>
                <li>
                  evaluate cosine similarity against every precomputed database embedding
                </li>
                <li>
                  return the top-{topK} most-similar entries
                </li>
              </ol>
              {lastTiming && (
                <div className="mt-2 rounded-md bg-dark/5 p-3 text-xs font-mono dark:bg-light/10">
                  <div>embedding: {lastTiming.embedMs.toFixed(2)} ms</div>
                  <div>kernel scan: {lastTiming.shaderMs.toFixed(2)} ms</div>
                  <div>total: {lastTiming.totalMs.toFixed(2)} ms</div>
                  <div>
                    {lastTiming.dbSize} db sequences &middot;&nbsp;
                    {lastTiming.dim}-dim embedding
                  </div>
                </div>
              )}
            </aside>
          </section>

          <section className="mt-12">
            <h2 className="text-3xl font-bold">Results</h2>
            {!results && (
              <p className="mt-2 text-base font-medium dark:text-light">
                Run the search to see top-{topK} candidates ranked by cosine similarity.
              </p>
            )}
            {results && (
              <div className="mt-6 overflow-hidden rounded-xl border-2 border-dark dark:border-light">
                <table className="w-full border-collapse text-left font-mono text-sm">
                  <thead className="bg-dark text-light dark:bg-light dark:text-dark">
                    <tr>
                      <th className="p-3">rank</th>
                      <th className="p-3">id</th>
                      <th className="p-3">family</th>
                      <th className="p-3">length</th>
                      <th className="p-3">cosine sim.</th>
                      <th className="p-3">distance</th>
                      <th className="p-3">preview</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r, i) => (
                      <tr
                        key={r.id}
                        className={`border-t border-dark/40 dark:border-light/40 ${
                          i % 2 === 0 ? "bg-light/60 dark:bg-dark/60" : ""
                        }`}
                      >
                        <td className="p-3">{i + 1}</td>
                        <td className="p-3">{r.id}</td>
                        <td className="p-3">{r.family}</td>
                        <td className="p-3">{r.length}</td>
                        <td className="p-3">{r.score.toFixed(4)}</td>
                        <td className="p-3">{r.distance.toFixed(4)}</td>
                        <td className="p-3">{r.text.slice(0, 28)}&hellip;</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="flex justify-end border-t border-dark/30 p-3 dark:border-light/30">
                  <button
                    type="button"
                    onClick={() => {
                      const tsv = [
                        "rank\tid\tfamily\tlength\tcosine\tdistance",
                        ...results.map((r, i) =>
                          [i + 1, r.id, r.family, r.length, r.score.toFixed(4), r.distance.toFixed(4)].join("\t")
                        ),
                      ].join("\n");
                      downloadText(`homology_results_${Date.now()}.tsv`, tsv);
                    }}
                    className="rounded-md border-2 border-dark px-3 py-1 text-sm font-semibold
                      hover:bg-dark hover:text-light dark:border-light dark:hover:bg-light dark:hover:text-dark"
                  >
                    Download TSV
                  </button>
                </div>
              </div>
            )}
          </section>
        </Layout>
      </main>
    </>
  );
}
