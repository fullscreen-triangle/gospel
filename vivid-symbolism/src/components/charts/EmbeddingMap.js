// 2D scatter of the database in random-projected coordinate space, with
// the query position and the top-K hits highlighted.

import { useEffect, useRef } from "react";
import * as d3 from "d3";

import { useChartTheme } from "./useChartTheme";

export default function EmbeddingMap({
  coords,           // Float32Array of length N*2, interleaved x,y
  topK = [],
  queryPosition,    // [x, y] from the same projection
  height = 320,
  title = "Database in projected coordinate space",
}) {
  const ref = useRef(null);
  const wrapRef = useRef(null);
  const theme = useChartTheme();

  useEffect(() => {
    if (!coords || coords.length === 0 || !ref.current) return;
    const wrap = wrapRef.current;
    const width = wrap ? wrap.clientWidth : 480;
    const margin = { top: 14, right: 14, bottom: 14, left: 14 };
    const W = Math.max(280, width);
    const H = height;

    const N = coords.length / 2;
    const xs = new Array(N);
    const ys = new Array(N);
    for (let i = 0; i < N; i += 1) { xs[i] = coords[i * 2]; ys[i] = coords[i * 2 + 1]; }

    let xExtent = d3.extent(xs);
    let yExtent = d3.extent(ys);
    if (queryPosition) {
      xExtent = [Math.min(xExtent[0], queryPosition[0]), Math.max(xExtent[1], queryPosition[0])];
      yExtent = [Math.min(yExtent[0], queryPosition[1]), Math.max(yExtent[1], queryPosition[1])];
    }
    const padX = (xExtent[1] - xExtent[0]) * 0.08 || 1;
    const padY = (yExtent[1] - yExtent[0]) * 0.08 || 1;
    const x = d3.scaleLinear()
      .domain([xExtent[0] - padX, xExtent[1] + padX])
      .range([margin.left, W - margin.right]);
    const y = d3.scaleLinear()
      .domain([yExtent[0] - padY, yExtent[1] + padY])
      .range([H - margin.bottom, margin.top]);

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);

    const topSet = new Set(topK.map((d) => d.index));

    const g = svg.append("g");
    g.append("g")
      .selectAll("circle")
      .data(d3.range(N))
      .join("circle")
      .attr("cx", (i) => x(xs[i]))
      .attr("cy", (i) => y(ys[i]))
      .attr("r", (i) => (topSet.has(i) ? 3.5 : 1.4))
      .attr("fill", (i) => (topSet.has(i) ? theme.accent : theme.fgMuted))
      .attr("opacity", (i) => (topSet.has(i) ? 0.95 : 0.4));

    if (queryPosition) {
      g.append("circle")
        .attr("cx", x(queryPosition[0]))
        .attr("cy", y(queryPosition[1]))
        .attr("r", 9)
        .attr("fill", "none")
        .attr("stroke", theme.fg)
        .attr("stroke-width", 1.6);
      g.append("circle")
        .attr("cx", x(queryPosition[0]))
        .attr("cy", y(queryPosition[1]))
        .attr("r", 4)
        .attr("fill", theme.fg);
      g.append("text")
        .attr("x", x(queryPosition[0]) + 12)
        .attr("y", y(queryPosition[1]) + 4)
        .attr("font-size", 11)
        .attr("font-family", "ui-monospace, monospace")
        .attr("fill", theme.fg)
        .text("query");
    }
  }, [coords, topK, queryPosition, theme, height]);

  return (
    <div ref={wrapRef} className="w-full">
      <div className="mb-1 text-sm font-semibold uppercase tracking-wide text-primary dark:text-primaryDark">
        {title}
      </div>
      <svg ref={ref} role="img" aria-label={title} />
    </div>
  );
}
