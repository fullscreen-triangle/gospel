// Per-pixel similarity track. Each x-coordinate corresponds to one
// genomic window; the colour is the cosine similarity between that
// window's precomputed embedding and the user's query embedding.
//
// The canvas is the analysis: every pixel literally holds the result of
// one similarity evaluation. There is no separate "data" and "view".

import { useEffect, useMemo, useRef } from "react";
import * as d3 from "d3";

import { useChartTheme } from "./useChartTheme";

function viridis(t) {
  // D3's interpolateViridis with manual clamping
  return d3.interpolateViridis(Math.max(0, Math.min(1, t)));
}

function magma(t) {
  return d3.interpolateMagma(Math.max(0, Math.min(1, t)));
}

const COLORMAPS = { viridis, magma };

export default function LocusTrack({
  scores,                  // Float32Array of length n_windows
  meta,
  height = 80,
  selection,               // { startIdx, endIdx } in window indices
  onSelect,                // (range) => void
  onHover,                 // (info) => void  | null
  topHits = [],            // [{ index, score }]
  colormap = "viridis",
  title = "Locus similarity track",
  showAnnotations = true,
}) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const theme = useChartTheme();

  const cmap = COLORMAPS[colormap] || viridis;

  const stats = useMemo(() => {
    if (!scores || scores.length === 0) return null;
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < scores.length; i += 1) {
      const v = scores[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    return { mn, mx };
  }, [scores]);

  // Paint the pixel canvas whenever scores or theme change.
  useEffect(() => {
    if (!scores || !meta || !canvasRef.current || !stats) return;
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    const W = wrap ? wrap.clientWidth : 800;
    const H = height - 20;
    const dpr = window.devicePixelRatio || 1;

    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    canvas.width = Math.max(1, Math.floor(W * dpr));
    canvas.height = Math.max(1, Math.floor(H * dpr));

    const ctx = canvas.getContext("2d", { alpha: false });
    ctx.imageSmoothingEnabled = false;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    const N = scores.length;
    const span = Math.max(stats.mx - stats.mn, 1e-9);

    // Each canvas pixel column samples one or several windows.
    // We compute the maximum (most-similar) window in each column so faint hits
    // are not lost by simple decimation.
    const colMax = new Float32Array(W);
    colMax.fill(-Infinity);
    for (let i = 0; i < N; i += 1) {
      const x = Math.floor((i / (N - 1)) * (W - 1));
      const v = scores[i];
      if (v > colMax[x]) colMax[x] = v;
    }
    // Forward-fill empty columns from the previous one.
    let last = stats.mn;
    for (let x = 0; x < W; x += 1) {
      if (!Number.isFinite(colMax[x])) colMax[x] = last;
      last = colMax[x];
    }

    for (let x = 0; x < W; x += 1) {
      const t = (colMax[x] - stats.mn) / span;
      ctx.fillStyle = cmap(t);
      ctx.fillRect(x, 0, 1, H);
    }

    // Draw planted-feature annotation ticks below the strip.
    if (showAnnotations && Array.isArray(meta.annotations)) {
      ctx.fillStyle = theme.fg;
      meta.annotations.forEach((a) => {
        const x0 = Math.floor((a.start / meta.length) * W);
        const x1 = Math.max(x0 + 1, Math.floor((a.end / meta.length) * W));
        ctx.fillRect(x0, H - 3, x1 - x0, 3);
      });
    }
  }, [scores, meta, stats, theme, height, cmap, showAnnotations]);

  // Overlay SVG for axes, top-hit markers, current selection, and annotation labels.
  useEffect(() => {
    if (!meta || !overlayRef.current) return;
    const wrap = wrapRef.current;
    const W = wrap ? wrap.clientWidth : 800;
    const H = height + 32;
    const svg = d3.select(overlayRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);

    const x = d3.scaleLinear().domain([0, meta.length]).range([0, W]);

    const axis = d3.axisBottom(x)
      .ticks(8)
      .tickFormat((d) => {
        if (d >= 1e6) return `${(d / 1e6).toFixed(1)} Mb`;
        if (d >= 1e3) return `${(d / 1e3).toFixed(0)} kb`;
        return `${d}`;
      });

    svg.append("g")
      .attr("transform", `translate(0, ${height - 20})`)
      .call(axis)
      .call((sel) => sel.selectAll("text").attr("fill", theme.fg))
      .call((sel) => sel.selectAll("path,line").attr("stroke", theme.fgMuted));

    // Planted-feature labels.
    if (showAnnotations && Array.isArray(meta.annotations)) {
      const lab = svg.append("g");
      meta.annotations.forEach((a, i) => {
        const cx = x((a.start + a.end) / 2);
        lab.append("text")
          .attr("x", cx)
          .attr("y", height + 18 + (i % 2 === 0 ? 0 : 12))
          .attr("text-anchor", "middle")
          .attr("font-size", 10)
          .attr("font-family", "ui-monospace, monospace")
          .attr("fill", theme.fgMuted)
          .text(a.label);
      });
    }

    // Top-hit markers above the strip.
    const markers = svg.append("g");
    topHits.forEach((h) => {
      const cx = x(h.index * meta.window_stride + meta.window_size / 2);
      markers.append("polygon")
        .attr("points", `${cx - 4},2 ${cx + 4},2 ${cx},10`)
        .attr("fill", theme.accent);
    });

    // Current selection rectangle.
    if (selection && selection.startIdx !== selection.endIdx) {
      const a = Math.min(selection.startIdx, selection.endIdx);
      const b = Math.max(selection.startIdx, selection.endIdx);
      const x0 = x(a * meta.window_stride);
      const x1 = x(b * meta.window_stride + meta.window_size);
      svg.append("rect")
        .attr("x", x0)
        .attr("y", 0)
        .attr("width", Math.max(1, x1 - x0))
        .attr("height", height - 20)
        .attr("fill", "none")
        .attr("stroke", theme.fg)
        .attr("stroke-width", 1.4)
        .attr("stroke-dasharray", "4 3");
    }
  }, [meta, theme, height, topHits, selection, showAnnotations]);

  // Brushing: drag along the canvas to define a zoom range.
  useEffect(() => {
    if (!meta || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!wrap) return;

    let dragging = false;
    let startIdx = 0;

    const indexFromEvent = (e) => {
      const r = canvas.getBoundingClientRect();
      const px = Math.max(0, Math.min(r.width - 1, e.clientX - r.left));
      const frac = px / r.width;
      const idx = Math.round(frac * (meta.n_windows - 1));
      return Math.max(0, Math.min(meta.n_windows - 1, idx));
    };

    const down = (e) => {
      dragging = true;
      startIdx = indexFromEvent(e);
      if (onSelect) onSelect({ startIdx, endIdx: startIdx });
    };
    const move = (e) => {
      const idx = indexFromEvent(e);
      if (onHover) {
        onHover({ idx, position: idx * meta.window_stride });
      }
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
  }, [meta, onSelect, onHover]);

  return (
    <div ref={wrapRef} className="w-full">
      <div className="mb-1 flex items-baseline justify-between">
        <div className="text-sm font-semibold uppercase tracking-wide text-primary dark:text-primaryDark">
          {title}
        </div>
        {stats && (
          <div className="font-mono text-xs text-dark/70 dark:text-light/70">
            min {stats.mn.toFixed(3)} &middot; max {stats.mx.toFixed(3)}
          </div>
        )}
      </div>
      <div className="relative" style={{ height }}>
        <canvas
          ref={canvasRef}
          className="absolute inset-x-0 top-0 cursor-crosshair rounded"
        />
        <svg
          ref={overlayRef}
          className="pointer-events-none absolute inset-x-0 top-0"
          style={{ height: height + 32 }}
        />
      </div>
    </div>
  );
}
