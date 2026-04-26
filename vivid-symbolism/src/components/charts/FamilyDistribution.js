// Bar chart of how many of the top-K hits fall into each family.
// Indicates whether a query has a clean dominant family or is mixed.

import { useEffect, useRef } from "react";
import * as d3 from "d3";

import { useChartTheme } from "./useChartTheme";

export default function FamilyDistribution({
  topK = [],
  height = 220,
  title = "Top-K hit families",
}) {
  const ref = useRef(null);
  const wrapRef = useRef(null);
  const theme = useChartTheme();

  useEffect(() => {
    if (!topK.length || !ref.current) return;
    const wrap = wrapRef.current;
    const width = wrap ? wrap.clientWidth : 480;
    const margin = { top: 18, right: 16, bottom: 38, left: 44 };
    const W = Math.max(280, width);
    const H = height;

    const counts = d3.rollups(
      topK,
      (v) => v.length,
      (d) => d.family
    ).map(([fam, count]) => ({ fam, count }))
      .sort((a, b) => d3.descending(a.count, b.count))
      .slice(0, 12);

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);

    const x = d3.scaleBand()
      .domain(counts.map((d) => `fam ${String(d.fam).padStart(3, "0")}`))
      .range([margin.left, W - margin.right])
      .padding(0.18);
    const y = d3.scaleLinear()
      .domain([0, d3.max(counts, (d) => d.count) || 1])
      .nice()
      .range([H - margin.bottom, margin.top]);

    const g = svg.append("g");
    g.append("g")
      .attr("transform", `translate(0,${H - margin.bottom})`)
      .call(d3.axisBottom(x).tickSizeOuter(0))
      .call((sel) =>
        sel.selectAll("text")
          .attr("fill", theme.fg)
          .attr("transform", "rotate(-30)")
          .attr("text-anchor", "end")
          .attr("dx", "-0.4em")
          .attr("dy", "0.2em")
          .style("font-size", "10px")
      )
      .call((sel) => sel.selectAll("path,line").attr("stroke", theme.fgMuted));
    g.append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(4).tickSizeOuter(0))
      .call((sel) => sel.selectAll("text").attr("fill", theme.fg))
      .call((sel) => sel.selectAll("path,line").attr("stroke", theme.fgMuted));

    g.append("g")
      .selectAll("rect")
      .data(counts)
      .join("rect")
      .attr("x", (d) => x(`fam ${String(d.fam).padStart(3, "0")}`))
      .attr("y", (d) => y(d.count))
      .attr("width", x.bandwidth())
      .attr("height", (d) => y(0) - y(d.count))
      .attr("fill", theme.accent)
      .attr("opacity", 0.8);

    g.selectAll("text.value")
      .data(counts)
      .join("text")
      .attr("class", "value")
      .attr("x", (d) => x(`fam ${String(d.fam).padStart(3, "0")}`) + x.bandwidth() / 2)
      .attr("y", (d) => y(d.count) - 4)
      .attr("text-anchor", "middle")
      .attr("fill", theme.fg)
      .attr("font-size", 10)
      .attr("font-family", "ui-monospace, monospace")
      .text((d) => d.count);
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
