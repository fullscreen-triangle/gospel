// Cosine similarity vs. rank curve. A steep drop in the first few ranks
// indicates a clean homology hit; a slow flat decay indicates a query in
// the noise floor of the database.

import { useEffect, useRef } from "react";
import * as d3 from "d3";

import { useChartTheme } from "./useChartTheme";

export default function RankProfile({
  topK = [],
  height = 220,
  title = "Score vs. rank",
}) {
  const ref = useRef(null);
  const wrapRef = useRef(null);
  const theme = useChartTheme();

  useEffect(() => {
    if (!topK.length || !ref.current) return;
    const wrap = wrapRef.current;
    const width = wrap ? wrap.clientWidth : 480;
    const margin = { top: 18, right: 18, bottom: 36, left: 44 };
    const W = Math.max(280, width);
    const H = height;

    const data = topK.map((d, i) => ({ rank: i + 1, score: d.score }));
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);

    const x = d3.scaleLinear().domain([1, data.length]).range([margin.left, W - margin.right]);
    const y = d3.scaleLinear()
      .domain([d3.min(data, (d) => d.score), d3.max(data, (d) => d.score)])
      .nice()
      .range([H - margin.bottom, margin.top]);

    const g = svg.append("g");
    g.append("g")
      .attr("transform", `translate(0,${H - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(Math.min(data.length, 8)).tickFormat(d3.format("d")).tickSizeOuter(0))
      .call((sel) => sel.selectAll("text").attr("fill", theme.fg))
      .call((sel) => sel.selectAll("path,line").attr("stroke", theme.fgMuted));
    g.append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(4).tickSizeOuter(0))
      .call((sel) => sel.selectAll("text").attr("fill", theme.fg))
      .call((sel) => sel.selectAll("path,line").attr("stroke", theme.fgMuted));

    const line = d3.line()
      .x((d) => x(d.rank))
      .y((d) => y(d.score))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", theme.accent)
      .attr("stroke-width", 1.6)
      .attr("d", line);
    g.append("g")
      .selectAll("circle")
      .data(data)
      .join("circle")
      .attr("cx", (d) => x(d.rank))
      .attr("cy", (d) => y(d.score))
      .attr("r", 3)
      .attr("fill", theme.accent);

    g.append("text")
      .attr("x", W - margin.right)
      .attr("y", H - 4)
      .attr("text-anchor", "end")
      .attr("font-size", 10)
      .attr("fill", theme.fgMuted)
      .text("rank");
    g.append("text")
      .attr("x", 6)
      .attr("y", margin.top - 4)
      .attr("font-size", 10)
      .attr("fill", theme.fgMuted)
      .text("cosine similarity");
  }, [topK, theme, height]);

  return (
    <div ref={wrapRef} className="w-full">
      <div className="mb-1 text-sm font-semibold uppercase tracking-wide text-primary dark:text-primaryDark">
        {title}
      </div>
      <svg ref={ref} role="img" aria-label={title} />
    </div>
  );
}
