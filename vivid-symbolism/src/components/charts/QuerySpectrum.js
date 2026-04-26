// Query DFT magnitude spectrum across channels. Shows the actual
// per-channel low-frequency content used by the embedding.

import { useEffect, useRef } from "react";
import * as d3 from "d3";

import { useChartTheme } from "./useChartTheme";

export default function QuerySpectrum({
  channelSpectra,        // number[][] of shape c x K
  channelLabels = [],
  height = 220,
  title = "Query spectrum (per channel)",
}) {
  const ref = useRef(null);
  const wrapRef = useRef(null);
  const theme = useChartTheme();

  useEffect(() => {
    if (!channelSpectra || channelSpectra.length === 0 || !ref.current) return;
    const wrap = wrapRef.current;
    const width = wrap ? wrap.clientWidth : 480;
    const margin = { top: 18, right: 16, bottom: 36, left: 44 };
    const W = Math.max(280, width);
    const H = height;

    const K = channelSpectra[0].length;
    const allValues = channelSpectra.flat();
    const x = d3.scaleLinear().domain([1, K]).range([margin.left, W - margin.right]);
    const y = d3.scaleLinear()
      .domain([0, d3.max(allValues) || 1])
      .nice()
      .range([H - margin.bottom, margin.top]);

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);

    const g = svg.append("g");
    g.append("g")
      .attr("transform", `translate(0,${H - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(Math.min(K, 8)).tickFormat(d3.format("d")).tickSizeOuter(0))
      .call((sel) => sel.selectAll("text").attr("fill", theme.fg))
      .call((sel) => sel.selectAll("path,line").attr("stroke", theme.fgMuted));
    g.append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(4).tickSizeOuter(0))
      .call((sel) => sel.selectAll("text").attr("fill", theme.fg))
      .call((sel) => sel.selectAll("path,line").attr("stroke", theme.fgMuted));

    const palette = ["#2a6f97", "#c9423a", "#6b8f3a", "#b58900"];
    const line = d3.line()
      .x((_, i) => x(i + 1))
      .y((d) => y(d))
      .curve(d3.curveMonotoneX);

    channelSpectra.forEach((spec, ch) => {
      const colour = palette[ch % palette.length];
      g.append("path")
        .datum(spec)
        .attr("fill", "none")
        .attr("stroke", colour)
        .attr("stroke-width", 1.6)
        .attr("opacity", 0.9)
        .attr("d", line);
      g.append("g")
        .selectAll("circle")
        .data(spec)
        .join("circle")
        .attr("cx", (_, i) => x(i + 1))
        .attr("cy", (d) => y(d))
        .attr("r", 2.4)
        .attr("fill", colour);
    });

    const legend = svg.append("g").attr("transform", `translate(${margin.left + 8},${margin.top - 4})`);
    channelSpectra.forEach((_, ch) => {
      const colour = palette[ch % palette.length];
      const lbl = channelLabels[ch] ?? `ch ${ch}`;
      const item = legend.append("g").attr("transform", `translate(${ch * 90},0)`);
      item.append("circle").attr("r", 4).attr("cx", 0).attr("cy", 0).attr("fill", colour);
      item.append("text")
        .attr("x", 8)
        .attr("y", 4)
        .attr("font-size", 11)
        .attr("font-family", "ui-monospace, monospace")
        .attr("fill", theme.fg)
        .text(lbl);
    });

    g.append("text")
      .attr("x", W - margin.right)
      .attr("y", H - 4)
      .attr("text-anchor", "end")
      .attr("font-size", 10)
      .attr("fill", theme.fgMuted)
      .text("Fourier bin k");
  }, [channelSpectra, channelLabels, theme, height]);

  return (
    <div ref={wrapRef} className="w-full">
      <div className="mb-1 text-sm font-semibold uppercase tracking-wide text-primary dark:text-primaryDark">
        {title}
      </div>
      <svg ref={ref} role="img" aria-label={title} />
    </div>
  );
}
