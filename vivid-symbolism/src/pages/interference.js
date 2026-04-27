import Head from "next/head";
import { useEffect, useMemo, useRef, useState } from "react";

import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import InterferenceTrace from "@/components/charts/InterferenceTrace";
import LocusDetail from "@/components/charts/LocusDetail";
import {
  prepareTarget,
  matchedFilterScan,
  backgroundStats,
  findPeaks,
} from "@/lib/matched_filter";
import { loadLocus } from "@/lib/locus";

const ALLOWED_DNA = /[^ACGTN]/gi;
const MAX_QUERY_LEN = 4096;
const PEAK_SEPARATION_BP = 200;

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
        <div className="mt-0.5 text-[11px] font-medium text-dark/70 dark:text-light/70">{sub}</div>
      ) : null}
    </div>
  );
}

function PeaksTable({ peaks, scores, onJump }) {
  if (!peaks || peaks.length === 0) return null;
  return (
    <div className="overflow-hidden rounded-xl border-2 border-dark dark:border-light">
      <table className="w-full border-collapse text-left font-mono text-xs">
        <thead className="bg-dark text-light dark:bg-light dark:text-dark">
          <tr>
            <th className="p-2">rank</th>
            <th className="p-2">position (bp)</th>
            <th className="p-2">cosine ρ</th>
            <th className="p-2">z</th>
            <th className="p-2"></th>
          </tr>
        </thead>
        <tbody>
          {peaks.map((p, i) => (
            <tr key={`${p.index}-${i}`} className="border-t border-dark/40 dark:border-light/40">
              <td className="p-2">{i + 1}</td>
              <td className="p-2">{p.index.toLocaleString()}</td>
              <td className="p-2">{scores[p.index].toFixed(4)}</td>
              <td className="p-2">{p.score.toFixed(2)}</td>
              <td className="p-2">
                <button
                  type="button"
                  className="rounded border border-dark px-2 py-0.5 text-[11px] font-semibold
                    hover:bg-dark hover:text-light dark:border-light dark:hover:bg-light dark:hover:text-dark"
                  onClick={() => onJump(p.index)}
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

export default function Interference() {
  const [locus, setLocus] = useState(null);
  const [ctx, setCtx] = useState(null);
  const [loadingState, setLoadingState] = useState("loading");
  const [loadError, setLoadError] = useState(null);
  const [prepMs, setPrepMs] = useState(null);

  const [query, setQuery] = useState("");
  const [zThreshold, setZThreshold] = useState(5.0);
  const [scoreData, setScoreData] = useState(null);
  const [peaks, setPeaks] = useState([]);
  const [selection, setSelection] = useState(null);
  const [hoverInfo, setHoverInfo] = useState(null);
  const [running, setRunning] = useState(false);
  const initialised = useRef(false);

  // Load and prepare the target once.
  useEffect(() => {
    let cancelled = false;
    setLoadingState("loading");
    loadLocus()
      .then(async (l) => {
        if (cancelled) return;
        setLocus(l);
        // Yield to the browser before kicking off the FFT so the UI paints.
        await new Promise((r) => setTimeout(r, 30));
        const t0 = performance.now();
        const c = prepareTarget(l.sequence, MAX_QUERY_LEN);
        const dt = performance.now() - t0;
        if (cancelled) return;
        setCtx(c);
        setPrepMs(dt);
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

  const cleanedQuery = useMemo(() => cleanDNA(query), [query]);

  const run = async () => {
    if (!ctx || cleanedQuery.length < 30 || cleanedQuery.length > MAX_QUERY_LEN) {
      return;
    }
    setRunning(true);
    setSelection(null);
    // Yield once so the spinner state actually paints.
    await new Promise((r) => setTimeout(r, 16));
    try {
      const res = matchedFilterScan(cleanedQuery, ctx);
      const { mean, std } = backgroundStats(res.scores);
      // Compute z-scores in place.
      const z = new Float64Array(res.scores.length);
      for (let i = 0; i < z.length; i += 1) z[i] = (res.scores[i] - mean) / std;
      const found = findPeaks(z, zThreshold, PEAK_SEPARATION_BP, 100);
      setPeaks(found);
      setScoreData({
        scores: res.scores,
        zScores: z,
        mean,
        std,
        timings: res.timings,
        queryLen: cleanedQuery.length,
        lagCount: res.lagCount,
      });
    } finally {
      setRunning(false);
    }
  };

  const reapplyThreshold = () => {
    if (!scoreData) return;
    const found = findPeaks(scoreData.zScores, zThreshold, PEAK_SEPARATION_BP, 100);
    setPeaks(found);
  };

  const summary = useMemo(() => {
    if (!scoreData || !peaks.length) return null;
    const top = peaks[0];
    return {
      top1Rho: scoreData.scores[top.index],
      top1Z: top.score,
      bgMean: scoreData.mean,
      bgStd: scoreData.std,
      nPeaks: peaks.length,
    };
  }, [scoreData, peaks]);

  const jumpToLag = (idx) => {
    if (!ctx) return;
    const half = 400;
    setSelection({
      startIdx: Math.max(0, idx - half),
      endIdx: Math.min(scoreData.scores.length - 1, idx + half),
    });
  };

  return (
    <>
      <Head>
        <title>Interference &middot; Gospel Homology</title>
      </Head>

      <TransitionEffect />
      <main className="mb-16 flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-12">
          <h1 className="text-4xl font-bold sm:text-3xl">Matched-filter interference</h1>
          <p className="mt-2 max-w-3xl text-sm font-medium dark:text-light/80">
            Cross-correlate a query DNA sequence against the entire locus via
            FFT. The trace below is the position-resolved interference
            pattern: peaks are positions where the query and target signals
            phase-coherently reconstruct.
          </p>

          {loadingState === "error" && (
            <div className="mt-4 rounded-lg border-2 border-dark bg-light px-4 py-3 text-sm
              dark:border-light dark:bg-dark">
              failed to load locus: {loadError}
            </div>
          )}

          {locus && (
            <>
              <section className="mt-6 grid grid-cols-1 gap-4">
                <div className="flex flex-wrap items-center gap-3">
                  <label className="flex items-center gap-2 text-sm font-semibold">
                    z threshold
                    <input
                      type="number" min={2} max={20} step={0.5} value={zThreshold}
                      onChange={(e) => setZThreshold(parseFloat(e.target.value) || 5)}
                      className="w-20 rounded-md border-2 border-dark bg-light px-2 py-1 font-medium
                        dark:border-light dark:bg-dark dark:text-light"
                    />
                  </label>
                  <button type="button" onClick={reapplyThreshold} disabled={!scoreData}
                    className="rounded-md border-2 border-dark px-3 py-1 text-sm font-semibold
                      hover:bg-dark hover:text-light disabled:opacity-40
                      dark:border-light dark:hover:bg-light dark:hover:text-dark">
                    Reapply threshold
                  </button>
                  <button type="button" onClick={() => setQuery(locus.meta.demo_query || "")}
                    className="rounded-md border-2 border-dark px-3 py-1 text-sm font-semibold
                      hover:bg-dark hover:text-light dark:border-light dark:hover:bg-light dark:hover:text-dark">
                    Demo motif
                  </button>
                  <button type="button"
                    onClick={() => { setQuery(""); setScoreData(null); setPeaks([]); setSelection(null); }}
                    className="rounded-md border-2 border-dark px-3 py-1 text-sm font-semibold
                      hover:bg-dark hover:text-light dark:border-light dark:hover:bg-light dark:hover:text-dark">
                    Clear
                  </button>
                  <span className="ml-auto text-xs font-medium text-dark/70 dark:text-light/70">
                    locus: {locus.meta.length.toLocaleString()} bp &middot;{" "}
                    {loadingState === "ready"
                      ? `target FFT ready (${prepMs?.toFixed(0)} ms)`
                      : "preparing FFT..."}
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
                    cleaned length: <strong>{cleanedQuery.length}</strong>
                    {cleanedQuery.length < 30 ? " (minimum 30)" :
                      cleanedQuery.length > MAX_QUERY_LEN ? ` (max ${MAX_QUERY_LEN})` : ""}
                  </span>
                  <button type="button"
                    onClick={run}
                    disabled={loadingState !== "ready" || running ||
                              cleanedQuery.length < 30 || cleanedQuery.length > MAX_QUERY_LEN}
                    className="rounded-lg bg-dark px-5 py-2 text-base font-semibold text-light
                      hover:bg-primary disabled:cursor-not-allowed disabled:opacity-40
                      dark:bg-light dark:text-dark dark:hover:bg-primaryDark">
                    {running ? "scanning..." : "Run matched filter"}
                  </button>
                </div>
              </section>

              {scoreData && summary && (
                <>
                  <section className="mt-6 grid grid-cols-4 gap-4 sm:grid-cols-2">
                    <StatTile label="top-1 ρ" value={summary.top1Rho.toFixed(4)}
                      sub={`z = ${summary.top1Z.toFixed(2)}`} />
                    <StatTile label="background"
                      value={`μ = ${summary.bgMean.toExponential(2)}`}
                      sub={`σ = ${summary.bgStd.toFixed(4)}`} />
                    <StatTile label="peaks above z"
                      value={summary.nPeaks.toString()}
                      sub={`threshold z >= ${zThreshold}`} />
                    <StatTile label="scan time"
                      value={`${scoreData.timings.totalMs.toFixed(0)} ms`}
                      sub={`fft ${scoreData.timings.fftMs.toFixed(0)} + ifft ${scoreData.timings.ifftMs.toFixed(0)} ms`} />
                  </section>

                  <section className="mt-8 rounded-xl border-2 border-dark p-4 dark:border-light">
                    <InterferenceTrace
                      scores={scoreData.scores}
                      zScores={scoreData.zScores}
                      meta={locus.meta}
                      peaks={peaks}
                      thresholdRho={summary.bgMean + zThreshold * summary.bgStd}
                      selection={selection}
                      onSelect={setSelection}
                      onHover={setHoverInfo}
                    />
                    <div className="mt-2 flex items-center justify-between text-xs font-medium text-dark/70 dark:text-light/70">
                      <span>drag to zoom; use the zoom button in the peaks table to centre on a hit</span>
                      <span className="font-mono">
                        {hoverInfo
                          ? `cursor @ ${hoverInfo.idx.toLocaleString()} bp, ρ = ${hoverInfo.score.toFixed(4)}`
                          : ""}
                      </span>
                    </div>
                  </section>

                  {selection && (
                    <section className="mt-6 rounded-xl border-2 border-dark p-4 dark:border-light">
                      <LocusDetail
                        scores={scoreData.scores}
                        meta={{
                          ...locus.meta,
                          // The detail view interprets index*window_stride as bp.
                          // For matched-filter output, lag = bp directly, so spoof a stride of 1.
                          window_stride: 1,
                          window_size: 1,
                          n_windows: scoreData.scores.length,
                        }}
                        sequence={locus.sequence}
                        selection={selection}
                        title="Selection detail (interference)"
                      />
                    </section>
                  )}

                  <section className="mt-8">
                    <h2 className="mb-3 text-2xl font-bold">Detected peaks</h2>
                    <PeaksTable peaks={peaks} scores={scoreData.scores} onJump={jumpToLag} />
                    {peaks.length === 0 && (
                      <p className="text-sm font-medium text-dark/70 dark:text-light/70">
                        no peaks above z = {zThreshold}. Lower the threshold to see weaker signals.
                      </p>
                    )}
                  </section>
                </>
              )}

              {!scoreData && (
                <p className="mt-8 text-sm font-medium text-dark/70 dark:text-light/70">
                  {running
                    ? "scanning the locus..."
                    : loadingState === "ready"
                    ? "Run the matched filter to see the interference pattern. The first scan takes ~1 s on a single CPU thread; subsequent scans reuse the precomputed target FFT."
                    : "loading and preparing locus FFT..."}
                </p>
              )}
            </>
          )}
        </Layout>
      </main>
    </>
  );
}
