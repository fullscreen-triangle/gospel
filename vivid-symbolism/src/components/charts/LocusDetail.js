// Zoom-detail panel for a selected genomic range. Renders a higher-resolution
// similarity strip over the selection and shows the local DNA sequence with
// per-base similarity colour underneath.

import { useEffect, useMemo, useRef } from "react";
import * as d3 from "d3";

import { useChartTheme } from "./useChartTheme";

export default function LocusDetail({
  scores,
  meta,
  sequence,
  selection,            // { startIdx, endIdx }
  height = 110,
  title = "Selection detail",
}) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const axisRef = useRef(null);
  const theme = useChartTheme();

  const range = useMemo(() => {
    if (!selection || !meta) return null;
    let a = Math.min(selection.startIdx, selection.endIdx);
    let b = Math.max(selection.startIdx, selection.endIdx);
    if (a === b) {
      a = Math.max(0, a - 25);
      b = Math.min(meta.n_windows - 1, b + 25);
    }
    return { a, b };
  }, [selection, meta]);

  useEffect(() => {
    if (!scores || !meta || !range || !canvasRef.current) return;
    const wrap = wrapRef.current;
    const W = wrap ? wrap.clientWidth : 800;
    const H = height - 30;
    const dpr = window.devicePixelRatio || 1;
    const canvas = canvasRef.current;

    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    canvas.width = Math.max(1, Math.floor(W * dpr));
    canvas.height = Math.max(1, Math.floor(H * dpr));

    const ctx = canvas.getContext("2d", { alpha: false });
    ctx.scale(dpr, dpr);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const N = range.b - range.a + 1;
    let mn = Infinity, mx = -Infinity;
    for (let i = range.a; i <= range.b; i += 1) {
      const v = scores[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    const span = Math.max(mx - mn, 1e-9);

    // One pixel per window, scaled across the full canvas width.
    const colW = W / N;
    for (let i = 0; i < N; i += 1) {
      const v = scores[range.a + i];
      const t = (v - mn) / span;
      ctx.fillStyle = d3.interpolateViridis(Math.max(0, Math.min(1, t)));
      ctx.fillRect(i * colW, 0, Math.max(1, colW + 1), H);
    }
  }, [scores, meta, range, theme, height]);

  useEffect(() => {
    if (!meta || !range || !axisRef.current) return;
    const wrap = wrapRef.current;
    const W = wrap ? wrap.clientWidth : 800;
    const H = 30;
    const svg = d3.select(axisRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);

    const startBp = range.a * meta.window_stride;
    const endBp = range.b * meta.window_stride + meta.window_size;
    const x = d3.scaleLinear().domain([startBp, endBp]).range([0, W]);
    svg.append("g")
      .attr("transform", "translate(0, 4)")
      .call(d3.axisBottom(x).ticks(8).tickFormat((d) => {
        if (d >= 1e6) return `${(d / 1e6).toFixed(2)} Mb`;
        if (d >= 1e3) return `${(d / 1e3).toFixed(1)} kb`;
        return `${d}`;
      }))
      .call((sel) => sel.selectAll("text").attr("fill", theme.fg))
      .call((sel) => sel.selectAll("path,line").attr("stroke", theme.fgMuted));
  }, [meta, range, theme]);

  const seqText = useMemo(() => {
    if (!sequence || !meta || !range) return null;
    const startBp = range.a * meta.window_stride;
    const endBp = Math.min(sequence.length, range.b * meta.window_stride + meta.window_size);
    return { text: sequence.slice(startBp, endBp), start: startBp, end: endBp };
  }, [sequence, meta, range]);

  return (
    <div ref={wrapRef} className="w-full">
      <div className="mb-1 flex items-baseline justify-between">
        <div className="text-sm font-semibold uppercase tracking-wide text-primary dark:text-primaryDark">
          {title}
        </div>
        {range && meta && (
          <div className="font-mono text-xs text-dark/70 dark:text-light/70">
            {(range.a * meta.window_stride).toLocaleString()} &ndash;{" "}
            {(range.b * meta.window_stride + meta.window_size).toLocaleString()} bp
            {" "}({(range.b - range.a + 1).toLocaleString()} windows)
          </div>
        )}
      </div>
      <div className="relative" style={{ height: height - 30 }}>
        <canvas ref={canvasRef} className="absolute inset-0 rounded" />
      </div>
      <svg ref={axisRef} className="block w-full" style={{ height: 30 }} />
      {seqText && (
        <details className="mt-2">
          <summary className="cursor-pointer text-sm font-semibold dark:text-light">
            sequence in selection ({seqText.text.length.toLocaleString()} bp)
          </summary>
          <pre className="mt-2 max-h-48 overflow-y-auto whitespace-pre-wrap break-all rounded-md
            border-2 border-dark bg-light p-3 font-mono text-[11px] leading-relaxed
            dark:border-light dark:bg-dark dark:text-light">
{seqText.text}
          </pre>
        </details>
      )}
    </div>
  );
}
