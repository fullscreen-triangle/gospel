// Shared pieces for the /rescue charts: theme-aware role colours (the same
// roles as the manuscript figures), a container-width hook, a tooltip, axis
// styling and the rank statistics the interactive charts recompute live.

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

import { useChartTheme } from "@/components/charts/useChartTheme";

const ROLE = {
  light: { gain: "#2a78d6", increase: "#eb6834", rescue: "#1baf7a", canonical: "#4a3aa7", alert: "#e34948" },
  dark: { gain: "#3987e5", increase: "#d95926", rescue: "#199e70", canonical: "#9085e9", alert: "#e66767" },
};
export const SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
export const LABEL = { gain: "gain", increase: "% increase", rescue: "rescue", canonical: "canonical" };

export function useRescueTheme() {
  const base = useChartTheme();
  const dark = base.bg === "#1b1b1b";
  return { ...base, dark, role: dark ? ROLE.dark : ROLE.light, muted: dark ? "#898781" : "#898781" };
}

export function useWidth(min = 300) {
  const ref = useRef(null);
  const [w, setW] = useState(640);
  useEffect(() => {
    if (!ref.current) return undefined;
    const ro = new ResizeObserver((entries) => {
      const cw = entries[0].contentRect.width;
      if (cw > 0) setW(Math.max(min, Math.floor(cw)));
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [min]);
  return [ref, w];
}

// Tooltip: an absolutely positioned div inside the chart wrapper.
export function useTooltip() {
  const tipRef = useRef(null);
  const show = (html, event, wrap) => {
    const tip = tipRef.current;
    if (!tip || !wrap) return;
    const box = wrap.getBoundingClientRect();
    tip.innerHTML = html;
    tip.style.opacity = "1";
    const x = event.clientX - box.left + 14;
    const y = event.clientY - box.top + 10;
    const maxX = box.width - tip.offsetWidth - 4;
    tip.style.left = `${Math.min(x, Math.max(4, maxX))}px`;
    tip.style.top = `${y}px`;
  };
  const hide = () => {
    if (tipRef.current) tipRef.current.style.opacity = "0";
  };
  return { tipRef, show, hide };
}

export function Tooltip({ tipRef }) {
  return (
    <div
      ref={tipRef}
      className="pointer-events-none absolute z-20 rounded-md border border-dark/20 bg-light px-2 py-1.5
        font-mono text-[11px] leading-snug text-dark opacity-0 shadow-lg transition-opacity duration-100
        dark:border-light/20 dark:bg-dark dark:text-light"
      style={{ left: 0, top: 0, maxWidth: 260 }}
    />
  );
}

export function styleAxis(sel, theme) {
  sel.selectAll("text").attr("fill", theme.fg).attr("font-size", 11);
  sel.selectAll("path,line").attr("stroke", theme.fgMuted);
  return sel;
}

export function gridY(g, y, width, left, theme, ticks = 5) {
  g.append("g")
    .attr("transform", `translate(${left},0)`)
    .call(d3.axisLeft(y).ticks(ticks).tickSize(-width).tickFormat(""))
    .call((s) => s.select(".domain").remove())
    .call((s) => s.selectAll("line").attr("stroke", theme.grid).attr("stroke-width", 0.6));
}

export function axisLabel(svg, text, x, y, theme, rotate = false) {
  svg.append("text")
    .attr("x", x).attr("y", y)
    .attr("transform", rotate ? `rotate(-90,${x},${y})` : null)
    .attr("text-anchor", "middle").attr("fill", theme.fgMuted)
    .attr("font-size", 11).text(text);
}

// ---------------------------------------------------------------- statistics
export function ranks(v) {
  const idx = v.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]);
  const r = new Array(v.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j += 1;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k += 1) r[idx[k][1]] = avg;
    i = j + 1;
  }
  return r;
}

export function spearman(a, b) {
  const ra = ranks(a);
  const rb = ranks(b);
  const n = ra.length;
  const ma = d3.mean(ra);
  const mb = d3.mean(rb);
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i += 1) {
    num += (ra[i] - ma) * (rb[i] - mb);
    da += (ra[i] - ma) ** 2;
    db += (rb[i] - mb) ** 2;
  }
  return num / Math.sqrt(da * db);
}

export function topSet(v, k) {
  return new Set(v.map((x, i) => [x, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map((d) => d[1]));
}

export function jaccard(a, b) {
  let inter = 0;
  a.forEach((x) => { if (b.has(x)) inter += 1; });
  return inter / (a.size + b.size - inter);
}

export function fLambda(W, MD, MN, lam) {
  return (W - MD) / (Math.pow(MD, lam) * Math.pow(MN - MD, 1 - lam));
}

export const Z = 1.6448536269514722;

export function ChartFrame({ title, subtitle, children, controls, wrapRef }) {
  return (
    <figure className="my-8 rounded-xl border-2 border-dark p-5 dark:border-light md:p-3">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <div className="text-sm font-semibold uppercase tracking-wide text-primary dark:text-primaryDark">
            {title}
          </div>
          {subtitle ? <div className="mt-0.5 text-xs font-medium text-dark/70 dark:text-light/70">{subtitle}</div> : null}
        </div>
        {controls ? <div className="flex flex-wrap items-center gap-3 text-xs">{controls}</div> : null}
      </div>
      <div ref={wrapRef} className="relative w-full">{children}</div>
    </figure>
  );
}

export function Toggle({ options, value, onChange, label }) {
  return (
    <div className="flex items-center gap-1" role="group" aria-label={label}>
      {label ? <span className="mr-1 font-semibold text-dark/70 dark:text-light/70">{label}</span> : null}
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={`rounded border px-2 py-0.5 font-semibold transition
            ${value === o.value
              ? "border-dark bg-dark text-light dark:border-light dark:bg-light dark:text-dark"
              : "border-dark/40 text-dark hover:border-dark dark:border-light/40 dark:text-light dark:hover:border-light"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Slider({ label, min, max, step, value, onChange, format = (v) => v }) {
  return (
    <label className="flex items-center gap-2 font-semibold text-dark/80 dark:text-light/80">
      <span>{label}</span>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-36 accent-primary dark:accent-primaryDark"
      />
      <span className="w-14 font-mono">{format(value)}</span>
    </label>
  );
}
