// Histogram of cosine similarities across the entire database, with the
// top-K cutoff line and the query's actual top-1 score highlighted.

import { useEffect, useRef } from "react";
import * as d3 from "d3";

import { useChartTheme } from "./useChartTheme";

export default function SimilarityHistogram({
  scores,
  topK = [],
  height = 220,
  binCount = 40,
  title = "Similarity distribution",
}) {
  const ref = useRef(null);
  const wrapRef = useRef(null);
  const theme = useChartTheme();

  useEffect(() => {
    if (!scores || scores.length === 0 || !ref.current) return;
    const wrap = wrapRef.current;
    const width = wrap ? wrap.clientWidth : 480;
    const margin = { top: 18, right: 16, bottom: 36, left: 44 };
    const W = Math.max(280, width);
    const H = height;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);

    const arr = Array.from(scores);
    const x = d3.scaleLinear()
      .domain([d3.min(arr), d3.max(arr)])
      .nice()
      .range([margin.left, W - margin.right]);

    const bins = d3.bin().domain(x.domain()).thresholds(binCount)(arr);
    const y = d3.scaleLinear()
      .domain([0, d3.max(bins, (b) => b.length)])
      .nice()
      .range([H - margin.bottom, margin.top]);

    const g = svg.append("g");
    g.append("g")
      .attr("transform", `translate(0,${H - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(6).tickSizeOuter(0))
      .call((sel) => sel.selectAll("text").attr("fill", theme.fg))
      .call((sel) => sel.selectAll("path,line").attr("stroke", theme.fgMuted));
    g.append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(4).tickSizeOuter(0))
      .call((sel) => sel.selectAll("text").attr("fill", theme.fg))
      .call((sel) => sel.selectAll("path,line").attr("stroke", theme.fgMuted));

    g.append("g")
      .selectAll("rect")
      .data(bins)
      .join("rect")
      .attr("x", (d) => x(d.x0) + 0.5)
      .attr("y", (d) => y(d.length))
      .attr("width", (d) => Math.max(0, x(d.x1) - x(d.x0) - 1))
      .attr("height", (d) => y(0) - y(d.length))
      .attr("fill", theme.accentSoft)
      .attr("stroke", theme.accent)
      .attr("stroke-width", 0.6);

    if (topK.length > 0) {
      const cutoff = topK[topK.length - 1].score;
      g.append("line")
        .attr("x1", x(cutoff))
        .attr("x2", x(cutoff))
        .attr("y1", margin.top)
        .attr("y2", H - margin.bottom)
        .attr("stroke", theme.accent)
        .attr("stroke-dasharray", "3 3")
        .attr("stroke-width", 1);
      g.append("text")
        .attr("x", x(cutoff) + 4)
        .attr("y", margin.top + 12)
        .attr("fill", theme.fg)
        .attr("font-size", 11)
        .attr("font-family", "ui-monospace, monospace")
        .text(`top-${topK.length} cutoff = ${cutoff.toFixed(3)}`);

      const best = topK[0].score;
      g.append("circle")
        .attr("cx", x(best))
        .attr("cy", H - margin.bottom)
        .attr("r", 5)
        .attr("fill", theme.accent);
      g.append("text")
        .attr("x", x(best))
        .attr("y", H - margin.bottom + 22)
        .attr("text-anchor", "middle")
        .attr("fill", theme.fg)
        .attr("font-size", 10)
        .attr("font-family", "ui-monospace, monospace")
        .text(`top-1 = ${best.toFixed(3)}`);
    }
  }, [scores, topK, theme, height, binCount]);

  return (
    <div ref={wrapRef} className="w-full">
      <div className="mb-1 text-sm font-semibold uppercase tracking-wide text-primary dark:text-primaryDark">
        {title}
      </div>
      <svg ref={ref} role="img" aria-label={title} />
    </div>
  );
}
