// Position-resolved matched-filter trace.
// Each pixel column on the canvas plots the maximum normalised cross-
// correlation across the lags it covers, the lower envelope (the bulk of
// the noise band), and a horizontal threshold line. Detected peaks are
// labelled above the strip; selecting an x-range opens the detail view.

import { useEffect, useMemo, useRef } from "react";
import * as d3 from "d3";

import { useChartTheme } from "./useChartTheme";

export default function InterferenceTrace({
  scores,                  // Float64Array of length nLags
  zScores,                 // optional Float64Array same length, used for threshold display
  meta,                    // { length, window_stride is irrelevant here; we use lag = bp }
  height = 220,
  thresholdRho,            // optional ρ threshold to draw
  thresholdZ,              // optional z threshold to draw on right axis
  peaks = [],              // [{ index, score, zScore, rho }]
  selection,               // { startIdx, endIdx } in lag indices
  onSelect,                // (range) => void
  onHover,                 // (info) => void
  title = "Interference pattern (matched-filter score per lag)",
}) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const theme = useChartTheme();

  const stats = useMemo(() => {
    if (!scores || scores.length === 0) return null;
    let mn = Infinity, mx = -Infinity, sum = 0;
    for (let i = 0; i < scores.length; i += 1) {
      const v = scores[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      sum += v;
    }
    return { mn, mx, mean: sum / scores.length };
  }, [scores]);

  // Paint the trace into the canvas.
  useEffect(() => {
    if (!scores || !meta || !canvasRef.current || !stats) return;
    const wrap = wrapRef.current;
    const W = Math.max(280, wrap ? wrap.clientWidth : 800);
    const H = height - 24;
    const dpr = window.devicePixelRatio || 1;
    const canvas = canvasRef.current;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    canvas.width = Math.floor(W * dpr);
    canvas.height = Math.floor(H * dpr);

    const ctx = canvas.getContext("2d", { alpha: false });
    ctx.scale(dpr, dpr);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const N = scores.length;
    const yLo = Math.min(0, stats.mn);
    const yHi = Math.max(stats.mx, 1.0);
    const yPad = 0.04 * (yHi - yLo);
    const dom = [yLo - yPad, yHi + yPad];
    const yScale = (v) => H - ((v - dom[0]) / (dom[1] - dom[0])) * H;

    // baseline
    ctx.strokeStyle = theme.fgMuted;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, yScale(0));
    ctx.lineTo(W, yScale(0));
    ctx.stroke();

    // threshold line (rho)
    if (typeof thresholdRho === "number") {
      ctx.strokeStyle = theme.accent;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(0, yScale(thresholdRho));
      ctx.lineTo(W, yScale(thresholdRho));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // For each pixel column, take min and max across the lag range it spans.
    // The two filled traces give an at-a-glance impression of both the
    // baseline noise floor (min) and the peaks (max).
    const colMax = new Float32Array(W).fill(-Infinity);
    const colMin = new Float32Array(W).fill(Infinity);
    for (let i = 0; i < N; i += 1) {
      const x = Math.floor((i / Math.max(1, N - 1)) * (W - 1));
      const v = scores[i];
      if (v > colMax[x]) colMax[x] = v;
      if (v < colMin[x]) colMin[x] = v;
    }
    let lastMax = stats.mn;
    let lastMin = stats.mn;
    for (let x = 0; x < W; x += 1) {
      if (!Number.isFinite(colMax[x])) colMax[x] = lastMax;
      else lastMax = colMax[x];
      if (!Number.isFinite(colMin[x])) colMin[x] = lastMin;
      else lastMin = colMin[x];
    }

    // noise band
    ctx.fillStyle = theme.fgMuted;
    ctx.globalAlpha = 0.25;
    ctx.beginPath();
    ctx.moveTo(0, yScale(colMin[0]));
    for (let x = 1; x < W; x += 1) ctx.lineTo(x, yScale(colMin[x]));
    for (let x = W - 1; x >= 0; x -= 1) ctx.lineTo(x, yScale(colMax[x]));
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1.0;

    // peaks line (max envelope)
    ctx.strokeStyle = theme.accent;
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    ctx.moveTo(0, yScale(colMax[0]));
    for (let x = 1; x < W; x += 1) ctx.lineTo(x, yScale(colMax[x]));
    ctx.stroke();
  }, [scores, meta, stats, theme, height, thresholdRho]);

  // Overlay: axes, peak markers, selection rectangle.
  useEffect(() => {
    if (!meta || !overlayRef.current) return;
    const wrap = wrapRef.current;
    const W = Math.max(280, wrap ? wrap.clientWidth : 800);
    const H = height + 32;
    const svg = d3.select(overlayRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);

    const x = d3.scaleLinear().domain([0, scores ? scores.length : meta.length]).range([0, W]);

    svg.append("g")
      .attr("transform", `translate(0, ${height - 24})`)
      .call(d3.axisBottom(x).ticks(8).tickFormat((d) => {
        if (d >= 1e6) return `${(d / 1e6).toFixed(1)} Mb`;
        if (d >= 1e3) return `${(d / 1e3).toFixed(0)} kb`;
        return `${d}`;
      }))
      .call((sel) => sel.selectAll("text").attr("fill", theme.fg))
      .call((sel) => sel.selectAll("path,line").attr("stroke", theme.fgMuted));

    // Peak markers as ticks above the strip + text labels for the top few.
    if (peaks && peaks.length > 0) {
      const top = peaks.slice(0, 6);
      const markers = svg.append("g");
      peaks.forEach((p) => {
        const cx = x(p.index);
        markers.append("polygon")
          .attr("points", `${cx - 4},2 ${cx + 4},2 ${cx},10`)
          .attr("fill", theme.accent);
      });
      top.forEach((p, i) => {
        const cx = x(p.index);
        const offset = i % 2 === 0 ? -10 : -22;
        svg.append("text")
          .attr("x", cx)
          .attr("y", 12 + offset)
          .attr("text-anchor", "middle")
          .attr("font-size", 10)
          .attr("font-family", "ui-monospace, monospace")
          .attr("fill", theme.fg)
          .text(`${(p.index / 1000).toFixed(1)} kb`);
      });
    }

    if (selection && selection.startIdx !== selection.endIdx) {
      const a = Math.min(selection.startIdx, selection.endIdx);
      const b = Math.max(selection.startIdx, selection.endIdx);
      svg.append("rect")
        .attr("x", x(a)).attr("y", 0)
        .attr("width", Math.max(1, x(b) - x(a)))
        .attr("height", height - 24)
        .attr("fill", "none")
        .attr("stroke", theme.fg)
        .attr("stroke-width", 1.4)
        .attr("stroke-dasharray", "4 3");
    }
  }, [meta, scores, peaks, theme, height, selection]);

  // Brushing.
  useEffect(() => {
    if (!meta || !canvasRef.current || !scores) return;
    const canvas = canvasRef.current;
    let dragging = false;
    let startIdx = 0;

    const indexFromEvent = (e) => {
      const r = canvas.getBoundingClientRect();
      const px = Math.max(0, Math.min(r.width - 1, e.clientX - r.left));
      const idx = Math.round((px / r.width) * (scores.length - 1));
      return Math.max(0, Math.min(scores.length - 1, idx));
    };
    const down = (e) => {
      dragging = true;
      startIdx = indexFromEvent(e);
      if (onSelect) onSelect({ startIdx, endIdx: startIdx });
    };
    const move = (e) => {
      const idx = indexFromEvent(e);
      if (onHover) onHover({ idx, score: scores[idx] });
      if (dragging && onSelect) onSelect({ startIdx, endIdx: idx });
    };
    const up = () => { dragging = false; };
    const leave = () => { if (onHover) onHover(null); };

    canvas.addEventListener("mousedown", down);
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    canvas.addEventListener("mouseleave", leave);
    return () => {
      canvas.removeEventListener("mousedown", down);
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      canvas.removeEventListener("mouseleave", leave);
    };
  }, [meta, scores, onSelect, onHover]);

  return (
    <div ref={wrapRef} className="w-full">
      <div className="mb-1 flex items-baseline justify-between">
        <div className="text-sm font-semibold uppercase tracking-wide text-primary dark:text-primaryDark">
          {title}
        </div>
        {stats && (
          <div className="font-mono text-xs text-dark/70 dark:text-light/70">
            min {stats.mn.toFixed(3)} &middot; mean {stats.mean.toExponential(2)} &middot; max {stats.mx.toFixed(3)}
          </div>
        )}
      </div>
      <div className="relative" style={{ height }}>
        <canvas ref={canvasRef} className="absolute inset-x-0 top-0 cursor-crosshair rounded" />
        <svg ref={overlayRef} className="pointer-events-none absolute inset-x-0 top-0"
             style={{ height: height + 32 }} />
      </div>
    </div>
  );
}
