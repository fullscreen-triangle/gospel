import Head from "next/head";
import { useEffect, useMemo, useRef, useState } from "react";

import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import LocusTrack from "@/components/charts/LocusTrack";
import LocusDetail from "@/components/charts/LocusDetail";
import {
  spectralEmbedding,
  shaderKernelScan,
} from "@/lib/embedding";
import { loadLocus } from "@/lib/locus";

const ALLOWED_DNA = /[^ACGTN]/gi;

function cleanDNA(raw) {
  return raw.toUpperCase().replace(/\s+/g, "").replace(ALLOWED_DNA, "");
}

function StatTile({ label, value, sub }) {
  return (
    <div className="rounded-lg border-2 border-dark p-3 dark:border-light">
      <div className="text-xs font-semibold uppercase tracking-wide text-primary dark:text-primaryDark">
        {label}
      </div>
      <div className="mt-1 font-mono text-xl font-bold dark:text-light">{value}</div>
      {sub ? (
        <div className="mt-0.5 text-[11px] font-medium text-dark/70 dark:text-light/70">
          {sub}
        </div>
      ) : null}
    </div>
  );
}

function HitsTable({ hits, meta, onJump }) {
  if (!hits || hits.length === 0) return null;
  return (
    <div className="overflow-hidden rounded-xl border-2 border-dark dark:border-light">
      <table className="w-full border-collapse text-left font-mono text-xs">
        <thead className="bg-dark text-light dark:bg-light dark:text-dark">
          <tr>
            <th className="p-2">rank</th>
            <th className="p-2">window</th>
            <th className="p-2">position (bp)</th>
            <th className="p-2">cosine</th>
            <th className="p-2">distance</th>
            <th className="p-2"></th>
          </tr>
        </thead>
        <tbody>
          {hits.map((h, i) => (
            <tr key={`${h.index}-${i}`} className="border-t border-dark/40 dark:border-light/40">
              <td className="p-2">{i + 1}</td>
              <td className="p-2">{h.index}</td>
              <td className="p-2">
                {(h.index * meta.window_stride).toLocaleString()} -{" "}
                {(h.index * meta.window_stride + meta.window_size).toLocaleString()}
              </td>
              <td className="p-2">{h.score.toFixed(4)}</td>
              <td className="p-2">{Math.max(0, 1 - h.score).toFixed(4)}</td>
              <td className="p-2">
                <button
                  type="button"
                  className="rounded border border-dark px-2 py-0.5 text-[11px] font-semibold
                    hover:bg-dark hover:text-light dark:border-light dark:hover:bg-light dark:hover:text-dark"
                  onClick={() => onJump(h.index)}
                >
                  zoom
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Browser() {
  const [locus, setLocus] = useState(null);
  const [loadingState, setLoadingState] = useState("loading");
  const [loadError, setLoadError] = useState(null);
  const [query, setQuery] = useState("");
  const [scores, setScores] = useState(null);
  const [topHits, setTopHits] = useState([]);
  const [topK, setTopK] = useState(10);
  const [selection, setSelection] = useState(null);
  const [hoverInfo, setHoverInfo] = useState(null);
  const [lastTiming, setLastTiming] = useState(null);
  const initialised = useRef(false);

  useEffect(() => {
    let cancelled = false;
    setLoadingState("loading");
    loadLocus()
      .then((l) => {
        if (cancelled) return;
        setLocus(l);
        setLoadingState("ready");
        if (!initialised.current) {
          setQuery(l.meta.demo_query || "");
          initialised.current = true;
        }
      })
      .catch((e) => {
        if (cancelled) return;
        setLoadError(e.message);
        setLoadingState("error");
      });
    return () => { cancelled = true; };
  }, []);

  const cleaned = useMemo(() => cleanDNA(query), [query]);

  const run = () => {
    if (!locus || cleaned.length < 30) {
      setScores(null);
      setTopHits([]);
      return;
    }
    const t0 = performance.now();
    const qv = spectralEmbedding(cleaned, locus.meta.n_coefficients, "dna");
    const tEmbed = performance.now() - t0;

    const t1 = performance.now();
    const scan = shaderKernelScan(locus.flat, locus.dim, qv, topK);
    const tScan = performance.now() - t1;

    setScores(scan.scores);
    setTopHits(scan.topK);
    setLastTiming({
      embedMs: tEmbed,
      scanMs: tScan,
      totalMs: tEmbed + tScan,
      windows: locus.nWindows,
      dim: locus.dim,
    });
    setSelection(null);
  };

  const summary = useMemo(() => {
    if (!scores || !topHits.length) return null;
    let mean = 0;
    for (let i = 0; i < scores.length; i += 1) mean += scores[i];
    mean /= scores.length;
    let varSum = 0;
    for (let i = 0; i < scores.length; i += 1) varSum += (scores[i] - mean) ** 2;
    const std = Math.sqrt(varSum / scores.length);
    const top1 = topHits[0].score;
    const z = std > 0 ? (top1 - mean) / std : 0;
    return { mean, std, top1, z };
  }, [scores, topHits]);

  const jumpToWindow = (idx) => {
    if (!locus) return;
    const half = 50;
    const a = Math.max(0, idx - half);
    const b = Math.min(locus.nWindows - 1, idx + half);
    setSelection({ startIdx: a, endIdx: b });
  };

  return (
    <>
      <Head>
        <title>Browser &middot; Gospel Homology</title>
      </Head>

      <TransitionEffect />
      <main className="mb-16 flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-12">
          <h1 className="text-4xl font-bold sm:text-3xl">Locus browser</h1>
          <p className="mt-2 max-w-3xl text-sm font-medium dark:text-light/80">
            One pixel per genomic window. The colour is the cosine similarity
            between that window&apos;s spectral embedding and the query you paste
            below. The picture is the analysis.
          </p>

          {loadingState === "error" && (
            <div className="mt-4 rounded-lg border-2 border-dark bg-light px-4 py-3 text-sm font-medium
              dark:border-light dark:bg-dark">
              failed to load locus: {loadError}
            </div>
          )}

          {locus && (
            <>
              <section className="mt-6 grid grid-cols-1 gap-4">
                <div className="flex flex-wrap items-center gap-3">
                  <label className="flex items-center gap-2 text-sm font-semibold">
                    Top
                    <input
                      type="number" min={1} max={50} value={topK}
                      onChange={(e) => setTopK(parseInt(e.target.value, 10) || 10)}
                      className="w-16 rounded-md border-2 border-dark bg-light px-2 py-1 font-medium
                        dark:border-light dark:bg-dark dark:text-light"
                    />
                  </label>
                  <button type="button"
                    onClick={() => setQuery(locus.meta.demo_query || "")}
                    className="rounded-md border-2 border-dark px-3 py-1 text-sm font-semibold
                      hover:bg-dark hover:text-light dark:border-light dark:hover:bg-light dark:hover:text-dark">
                    Demo motif
                  </button>
                  <button type="button"
                    onClick={() => { setQuery(""); setScores(null); setTopHits([]); setSelection(null); setLastTiming(null); }}
                    className="rounded-md border-2 border-dark px-3 py-1 text-sm font-semibold
                      hover:bg-dark hover:text-light dark:border-light dark:hover:bg-light dark:hover:text-dark">
                    Clear
                  </button>
                  <span className="ml-auto text-xs font-medium text-dark/70 dark:text-light/70">
                    locus: {locus.meta.length.toLocaleString()} bp,{" "}
                    {locus.nWindows.toLocaleString()} windows of{" "}
                    {locus.windowSize} bp (stride {locus.windowStride})
                  </span>
                </div>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  spellCheck={false}
                  rows={5}
                  className="w-full rounded-xl border-2 border-dark bg-light p-3 font-mono text-sm leading-relaxed
                    focus:outline-none focus:ring-2 focus:ring-primary
                    dark:border-light dark:bg-dark dark:text-light"
                  placeholder="paste a DNA query (>= 30 bp). Try the demo motif to find its two planted occurrences."
                />
                <div className="flex items-center justify-between text-xs font-medium text-dark/70 dark:text-light/70">
                  <span>
                    cleaned length: <strong>{cleaned.length}</strong>
                    {cleaned.length < 30 ? " (minimum 30)" : ""}
                  </span>
                  <button type="button"
                    onClick={run}
                    disabled={cleaned.length < 30 || loadingState !== "ready"}
                    className="rounded-lg bg-dark px-5 py-2 text-base font-semibold text-light
                      hover:bg-primary disabled:cursor-not-allowed disabled:opacity-40
                      dark:bg-light dark:text-dark dark:hover:bg-primaryDark">
                    Paint locus
                  </button>
                </div>
              </section>

              {scores && (
                <>
                  <section className="mt-6 grid grid-cols-4 gap-4 sm:grid-cols-2">
                    <StatTile label="top-1 cosine"
                      value={topHits[0].score.toFixed(4)}
                      sub={`window ${topHits[0].index}, ${(topHits[0].index * locus.windowStride).toLocaleString()} bp`} />
                    <StatTile label="top-1 z-score"
                      value={summary ? summary.z.toFixed(2) : "-"}
                      sub="standard deviations above the locus mean" />
                    <StatTile label="background mean"
                      value={summary ? summary.mean.toFixed(4) : "-"}
                      sub={summary ? `σ = ${summary.std.toFixed(4)}` : ""} />
                    <StatTile label="kernel scan"
                      value={lastTiming ? `${lastTiming.scanMs.toFixed(2)} ms` : "-"}
                      sub={lastTiming ? `${lastTiming.windows.toLocaleString()} windows` : ""} />
                  </section>

                  <section className="mt-8 rounded-xl border-2 border-dark p-4 dark:border-light">
                    <LocusTrack
                      scores={scores}
                      meta={locus.meta}
                      topHits={topHits}
                      selection={selection}
                      onSelect={setSelection}
                      onHover={setHoverInfo}
                    />
                    <div className="mt-2 flex items-center justify-between text-xs font-medium text-dark/70 dark:text-light/70">
                      <span>drag along the strip to zoom; click a row in the hits table to centre on a hit</span>
                      <span className="font-mono">
                        {hoverInfo
                          ? `cursor @ ${hoverInfo.position.toLocaleString()} bp (window ${hoverInfo.idx})`
                          : ""}
                      </span>
                    </div>
                  </section>

                  {selection && (
                    <section className="mt-6 rounded-xl border-2 border-dark p-4 dark:border-light">
                      <LocusDetail
                        scores={scores}
                        meta={locus.meta}
                        sequence={locus.sequence}
                        selection={selection}
                      />
                    </section>
                  )}

                  <section className="mt-8">
                    <h2 className="mb-3 text-2xl font-bold">Top hits</h2>
                    <HitsTable hits={topHits} meta={locus.meta} onJump={jumpToWindow} />
                  </section>
                </>
              )}

              {!scores && (
                <p className="mt-8 text-sm font-medium text-dark/70 dark:text-light/70">
                  {loadingState === "ready"
                    ? "Paint the locus to start. The strip below will repaint every time you change the query."
                    : "loading locus..."}
                </p>
              )}
            </>
          )}
        </Layout>
      </main>
    </>
  );
}
