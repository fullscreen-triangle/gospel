import Head from "next/head";
import { useEffect, useMemo, useRef, useState } from "react";

import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import {
  spectralEmbedding,
  spectralEmbeddingDetail,
  shaderKernelScan,
  build2DProjection,
} from "@/lib/embedding";

import SimilarityHistogram from "@/components/charts/SimilarityHistogram";
import FamilyDistribution from "@/components/charts/FamilyDistribution";
import RankProfile from "@/components/charts/RankProfile";
import EmbeddingMap from "@/components/charts/EmbeddingMap";
import QuerySpectrum from "@/components/charts/QuerySpectrum";

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

function StatTile({ label, value, sub }) {
  return (
    <div className="rounded-lg border-2 border-dark p-3 dark:border-light">
      <div className="text-xs font-semibold uppercase tracking-wide text-primary dark:text-primaryDark">
        {label}
      </div>
      <div className="mt-1 font-mono text-xl font-bold dark:text-light">{value}</div>
      {sub ? (
        <div className="mt-0.5 text-[11px] font-medium text-dark/70 dark:text-light/70">{sub}</div>
      ) : null}
    </div>
  );
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
  const [queryDetail, setQueryDetail] = useState(null);
  const [scoreVec, setScoreVec] = useState(null);
  const [queryProj, setQueryProj] = useState(null);
  const dbCache = useRef({});

  useEffect(() => {
    let cancelled = false;
    setDbLoading(true);
    setDbError(null);
    const cached = dbCache.current[kind];
    if (cached) {
      setDb(cached);
      setDbLoading(false);
      return () => { cancelled = true; };
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
        const projector = build2DProjection(flat, data.embed_dim);
        const prepared = {
          kind: data.kind,
          nCoefficients: data.n_coefficients,
          dim: data.embed_dim,
          sequences: data.sequences,
          flat,
          projection: projector.coords,
          project: projector.project,
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
    return () => { cancelled = true; };
  }, [kind]);

  const cleanedQuery = useMemo(() => cleanSequence(query, kind), [query, kind]);

  const run = () => {
    if (!db || !cleanedQuery || cleanedQuery.length < 10) {
      setResults(null);
      setScoreVec(null);
      setQueryDetail(null);
      setQueryProj(null);
      return;
    }
    const t0 = performance.now();
    const detail = spectralEmbeddingDetail(cleanedQuery, db.nCoefficients, kind);
    const tEmbed = performance.now() - t0;

    const t1 = performance.now();
    const scan = shaderKernelScan(db.flat, db.dim, detail.vector, topK);
    const tShader = performance.now() - t1;

    const enriched = scan.topK.map((h) => {
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
    const projected = db.project(detail.vector);

    setResults(enriched);
    setScoreVec(scan.scores);
    setQueryDetail(detail);
    setQueryProj(projected);
    setLastTiming({
      embedMs: tEmbed,
      shaderMs: tShader,
      totalMs: tEmbed + tShader,
      dim: detail.vector.length,
      querySeqLen: cleanedQuery.length,
      dbSize: db.sequences.length,
    });
  };

  const pasteDemo = () => setQuery(kind === "dna" ? DEMO_DNA : DEMO_PROTEIN);
  const clearAll = () => {
    setQuery("");
    setResults(null);
    setScoreVec(null);
    setQueryDetail(null);
    setQueryProj(null);
    setLastTiming(null);
  };

  const summaryStats = useMemo(() => {
    if (!results || !scoreVec) return null;
    const arr = Array.from(scoreVec);
    arr.sort((a, b) => a - b);
    const med = arr[Math.floor(arr.length / 2)];
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const std = Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length);
    const top1 = results[0]?.score ?? 0;
    const z = std > 0 ? (top1 - mean) / std : 0;
    const dominant = (() => {
      const counts = new Map();
      for (const r of results) counts.set(r.family, (counts.get(r.family) || 0) + 1);
      const best = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
      return best ? { family: best[0], hits: best[1] } : null;
    })();
    return { mean, std, med, top1, z, dominant };
  }, [results, scoreVec]);

  return (
    <>
      <Head>
        <title>Search &middot; Gospel Homology</title>
      </Head>

      <TransitionEffect />
      <main className="mb-16 flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-12">
          <h1 className="text-4xl font-bold sm:text-3xl">Homology search</h1>
          <p className="mt-2 max-w-3xl text-sm font-medium dark:text-light/80">
            Paste a sequence, scan a bundled database in your browser, see the
            full distribution of matches.
          </p>

          <section className="mt-8 grid grid-cols-12 gap-6">
            <div className="col-span-12 lg:col-span-12">
              <div className="mb-3 flex flex-wrap items-center gap-3">
                <label className="flex items-center gap-2 text-sm font-semibold">
                  Kind
                  <select
                    value={kind}
                    onChange={(e) => setKind(e.target.value)}
                    className="rounded-md border-2 border-dark bg-light px-2 py-1 font-medium
                      dark:border-light dark:bg-dark dark:text-light"
                  >
                    <option value="protein">protein</option>
                    <option value="dna">DNA</option>
                  </select>
                </label>
                <label className="flex items-center gap-2 text-sm font-semibold">
                  Top
                  <input
                    type="number" min={1} max={50} value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value, 10) || 10)}
                    className="w-16 rounded-md border-2 border-dark bg-light px-2 py-1 font-medium
                      dark:border-light dark:bg-dark dark:text-light"
                  />
                </label>
                <button type="button" onClick={pasteDemo}
                  className="rounded-md border-2 border-dark px-3 py-1 text-sm font-semibold
                    hover:bg-dark hover:text-light dark:border-light dark:hover:bg-light dark:hover:text-dark">
                  Demo
                </button>
                <button type="button" onClick={clearAll}
                  className="rounded-md border-2 border-dark px-3 py-1 text-sm font-semibold
                    hover:bg-dark hover:text-light dark:border-light dark:hover:bg-light dark:hover:text-dark">
                  Clear
                </button>
                <span className="ml-auto text-xs font-medium text-dark/70 dark:text-light/70">
                  {dbLoading
                    ? "loading database..."
                    : dbError
                    ? `error: ${dbError}`
                    : `db: ${db.sequences.length} ${db.kind}, ${db.dim}-dim`}
                </span>
              </div>

              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                spellCheck={false}
                rows={6}
                className="w-full rounded-xl border-2 border-dark bg-light p-3 font-mono text-sm leading-relaxed
                  focus:outline-none focus:ring-2 focus:ring-primary
                  dark:border-light dark:bg-dark dark:text-light"
                placeholder={kind === "dna" ? "paste a DNA sequence (ACGT)" : "paste a protein sequence"}
              />
              <div className="mt-2 flex items-center justify-between text-xs font-medium text-dark/70 dark:text-light/70">
                <span>length: <strong>{cleanedQuery.length}</strong>{cleanedQuery.length < 10 ? " (min 10)" : ""}</span>
                <button type="button" onClick={run}
                  disabled={dbLoading || !db || cleanedQuery.length < 10}
                  className="rounded-lg bg-dark px-5 py-2 text-base font-semibold text-light
                    hover:bg-primary disabled:cursor-not-allowed disabled:opacity-40
                    dark:bg-light dark:text-dark dark:hover:bg-primaryDark">
                  Run search
                </button>
              </div>
            </div>
          </section>

          {results ? (
            <>
              <section className="mt-8 grid grid-cols-4 gap-4 sm:grid-cols-2">
                <StatTile label="top-1 cosine" value={results[0].score.toFixed(4)}
                  sub={`distance ${results[0].distance.toFixed(4)}`} />
                <StatTile label="top-1 z-score" value={summaryStats ? summaryStats.z.toFixed(2) : "-"}
                  sub="standard deviations above the database mean" />
                <StatTile label="dominant family"
                  value={summaryStats?.dominant ? `fam ${summaryStats.dominant.family}` : "-"}
                  sub={summaryStats?.dominant ? `${summaryStats.dominant.hits} of top-${results.length}` : ""} />
                <StatTile label="kernel time"
                  value={lastTiming ? `${lastTiming.shaderMs.toFixed(2)} ms` : "-"}
                  sub={lastTiming ? `${lastTiming.dbSize} db × ${lastTiming.dim}-dim` : ""} />
              </section>

              <section className="mt-8 grid grid-cols-2 gap-6 lg:grid-cols-1">
                <div className="rounded-xl border-2 border-dark p-4 dark:border-light">
                  <SimilarityHistogram scores={scoreVec} topK={results} />
                </div>
                <div className="rounded-xl border-2 border-dark p-4 dark:border-light">
                  <RankProfile topK={results} />
                </div>
                <div className="rounded-xl border-2 border-dark p-4 dark:border-light">
                  <FamilyDistribution topK={results} />
                </div>
                <div className="rounded-xl border-2 border-dark p-4 dark:border-light">
                  <QuerySpectrum
                    channelSpectra={queryDetail?.channelSpectra}
                    channelLabels={queryDetail?.channelLabels}
                  />
                </div>
                <div className="col-span-2 rounded-xl border-2 border-dark p-4 dark:border-light lg:col-span-1">
                  <EmbeddingMap
                    coords={db.projection}
                    topK={results}
                    queryPosition={queryProj}
                  />
                </div>
              </section>

              <section className="mt-8">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-bold">Top hits</h2>
                  <button type="button"
                    onClick={() => {
                      const tsv = [
                        "rank\tid\tfamily\tlength\tcosine\tdistance",
                        ...results.map((r, i) =>
                          [i + 1, r.id, r.family, r.length, r.score.toFixed(4), r.distance.toFixed(4)].join("\t")),
                      ].join("\n");
                      downloadText(`homology_results_${Date.now()}.tsv`, tsv);
                    }}
                    className="rounded-md border-2 border-dark px-3 py-1 text-sm font-semibold
                      hover:bg-dark hover:text-light dark:border-light dark:hover:bg-light dark:hover:text-dark">
                    Export TSV
                  </button>
                </div>
                <div className="mt-3 overflow-hidden rounded-xl border-2 border-dark dark:border-light">
                  <table className="w-full border-collapse text-left font-mono text-xs">
                    <thead className="bg-dark text-light dark:bg-light dark:text-dark">
                      <tr>
                        <th className="p-2">rank</th>
                        <th className="p-2">id</th>
                        <th className="p-2">family</th>
                        <th className="p-2">len</th>
                        <th className="p-2">cosine</th>
                        <th className="p-2">distance</th>
                        <th className="p-2">preview</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.map((r, i) => (
                        <tr key={r.id} className={`border-t border-dark/40 dark:border-light/40 ${i % 2 === 0 ? "bg-light/60 dark:bg-dark/60" : ""}`}>
                          <td className="p-2">{i + 1}</td>
                          <td className="p-2">{r.id}</td>
                          <td className="p-2">{r.family}</td>
                          <td className="p-2">{r.length}</td>
                          <td className="p-2">{r.score.toFixed(4)}</td>
                          <td className="p-2">{r.distance.toFixed(4)}</td>
                          <td className="p-2 truncate">{r.text.slice(0, 36)}&hellip;</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          ) : (
            <p className="mt-10 text-sm font-medium text-dark/70 dark:text-light/70">
              Run a search to populate the dashboard. The whole computation runs
              client-side: nothing leaves your browser.
            </p>
          )}
        </Layout>
      </main>
    </>
  );
}
