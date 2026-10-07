// Charts for the full per-plant workbook (F-series): the reference line across
// independent runs, run-2-vs-run-3 reproducibility, the retest of the weak
// rescuers, and the inoculation effect with and without drought.

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

import {
  ChartFrame, LABEL, Toggle, Tooltip, axisLabel, gridY, styleAxis,
  useRescueTheme, useTooltip, useWidth,
} from "./kit";

const ORG = [{ value: "Shoot", label: "shoot" }, { value: "Root", label: "root" }];
const fmt2 = d3.format(".2f");
const fmt1 = d3.format(".1f");

// ------------------------------------------------------------- Col-0 by run
export function Col0Runs({ f2 }) {
  const [organ, setOrgan] = useState("Shoot");
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const { tipRef, show, hide } = useTooltip();
  const svgRef = useRef(null);

  useEffect(() => {
    if (!f2 || !svgRef.current) return;
    const g = f2[organ].col0_gain_by_replicate;
    const rows = ["BR1", "BR2", "BR3"].map((br, i) => ({ br, i, ...g[br] }));
    const W = Math.min(width, 640), H = 260, m = { top: 16, right: 16, bottom: 40, left: 56 };
    const svg = d3.select(svgRef.current).attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const x = d3.scaleBand().domain(rows.map((r) => r.br)).range([m.left, W - m.right]).padding(0.5);
    const y = d3.scaleLinear().domain([Math.min(0, d3.min(rows, (r) => r.lo)) - 0.5, d3.max(rows, (r) => r.hi) + 0.5])
      .nice().range([H - m.bottom, m.top]);
    gridY(svg, y, W - m.left - m.right, m.left, theme, 5);
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5)).call((s) => styleAxis(s, theme));
    svg.append("g").attr("transform", `translate(0,${H - m.bottom})`)
      .call(d3.axisBottom(x).tickFormat((b) => `run ${b.slice(2)}`)).call((s) => styleAxis(s, theme));
    svg.append("line").attr("x1", m.left).attr("x2", W - m.right).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", theme.fg);
    const cx = (r) => x(r.br) + x.bandwidth() / 2;
    svg.append("g").selectAll("line").data(rows).join("line")
      .attr("x1", cx).attr("x2", cx).attr("y1", (r) => y(r.lo)).attr("y2", (r) => y(r.hi))
      .attr("stroke", (r) => (r.lo <= 0 ? theme.role.alert : theme.role.gain)).attr("stroke-width", 3).attr("stroke-linecap", "round");
    svg.append("g").selectAll("circle").data(rows).join("circle")
      .attr("cx", cx).attr("cy", (r) => y(r.gain)).attr("r", 7)
      .attr("fill", (r) => (r.lo <= 0 ? theme.role.alert : theme.role.gain)).attr("stroke", theme.bg).attr("stroke-width", 2)
      .on("mousemove", (ev, r) => show(
        `<b>Col-0, run ${r.br.slice(2)}</b><br/>gain ${fmt2(r.gain)} mg [${fmt2(r.lo)}, ${fmt2(r.hi)}]<br/>rescue ${fmt2(r.rescue)}<br/>mock drought loss ${d3.format(".0%")(r.loss)}<br/>${r.n_W} inoculated, ${r.n_MD} mock plants`,
        ev, wrapRef.current))
      .on("mouseleave", hide);
    svg.append("g").selectAll("text.lbl").data(rows).join("text")
      .attr("x", (r) => cx(r) + 12).attr("y", (r) => y(r.gain) + 4).attr("fill", theme.fg).attr("font-size", 12)
      .attr("font-family", "ui-monospace, monospace").text((r) => `${fmt1(r.gain)} mg`);
    axisLabel(svg, `Col-0 gain W − M_D (mg ${organ.toLowerCase()}, 95% CI)`, 14, (H - m.bottom + m.top) / 2, theme, true);
  }, [f2, organ, width, theme, show, hide, wrapRef]);

  return (
    <ChartFrame
      title="The reference line across three independent runs"
      subtitle="Same genotype, same protocol. Red: the 95% interval reaches zero — in that run Col-0 itself would have been called a non-rescuer."
      controls={<Toggle options={ORG} value={organ} onChange={setOrgan} label="organ" />}
      wrapRef={wrapRef}
    >
      <svg ref={svgRef} role="img" aria-label="Col-0 gain by run" />
      <Tooltip tipRef={tipRef} />
    </ChartFrame>
  );
}

// ------------------------------------------------------- reproducibility forest
const ITEMS = [
  ["loss", "mock drought loss", "comp"], ["logMD", "log M_D", "comp"], ["logW", "log W", "comp"],
  ["increase", LABEL.increase, "cons"], ["gain", LABEL.gain, "cons"],
  ["rescue", LABEL.rescue, "cons"], ["canonical", LABEL.canonical, "cons"],
];

export function RetestForest({ f4 }) {
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const { tipRef, show, hide } = useTooltip();
  const svgRef = useRef(null);

  useEffect(() => {
    if (!f4 || !svgRef.current) return;
    const W = width, H = 330, m = { top: 26, right: 24, bottom: 42, left: 150 };
    const svg = d3.select(svgRef.current).attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const x = d3.scaleLinear().domain([-0.3, 1]).range([m.left, W - m.right]);
    const y = d3.scaleBand().domain(ITEMS.map((d) => d[0])).range([m.top, H - m.bottom]).padding(0.25);
    d3.range(-0.2, 1.01, 0.2).forEach((t) => svg.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", m.top).attr("y2", H - m.bottom).attr("stroke", theme.grid));
    svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", m.top).attr("y2", H - m.bottom).attr("stroke", theme.fg);
    svg.append("g").attr("transform", `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).ticks(7)).call((s) => styleAxis(s, theme));
    const colorOf = (k, kind) => (kind === "comp" ? theme.muted : theme.role[k]);
    ITEMS.forEach(([k, label, kind]) => {
      svg.append("text").attr("x", m.left - 10).attr("y", y(k) + y.bandwidth() / 2 + 4).attr("text-anchor", "end")
        .attr("fill", theme.fg).attr("font-size", 12).text(label);
      ["Shoot", "Root"].forEach((O, j) => {
        const f = f4[O];
        const r = (f.test_retest_spearman_BR2_BR3[k] ?? f.component_test_retest_spearman[k]);
        const [lo, hi] = f.test_retest_ci95[k];
        const yy = y(k) + (j === 0 ? 0.3 : 0.75) * y.bandwidth();
        const col = colorOf(k, kind);
        svg.append("line").attr("x1", x(lo)).attr("x2", x(hi)).attr("y1", yy).attr("y2", yy)
          .attr("stroke", col).attr("stroke-width", 2.5).attr("opacity", j === 0 ? 1 : 0.5);
        const mark = j === 0
          ? svg.append("circle").attr("cx", x(r)).attr("cy", yy).attr("r", 6)
          : svg.append("rect").attr("x", x(r) - 5).attr("y", yy - 5).attr("width", 10).attr("height", 10);
        mark.attr("fill", col).attr("opacity", j === 0 ? 1 : 0.7).attr("stroke", theme.bg).attr("stroke-width", 1.5)
          .on("mousemove", (ev) => show(`<b>${label}</b> · ${O.toLowerCase()}<br/>Spearman ρ run 2 vs run 3: ${fmt2(r)}<br/>95% CI [${fmt2(lo)}, ${fmt2(hi)}]<br/>${f.n_pairs} accessions`, ev, wrapRef.current))
          .on("mouseleave", hide);
      });
    });
    const lg = svg.append("g").attr("transform", `translate(${m.left},12)`);
    lg.append("circle").attr("cx", 4).attr("cy", 0).attr("r", 5).attr("fill", theme.fg);
    lg.append("text").attr("x", 14).attr("y", 4).attr("fill", theme.fg).attr("font-size", 11).text("shoot");
    lg.append("rect").attr("x", 70).attr("y", -5).attr("width", 10).attr("height", 10).attr("fill", theme.fg).attr("opacity", 0.7);
    lg.append("text").attr("x", 86).attr("y", 4).attr("fill", theme.fg).attr("font-size", 11).text("root");
    axisLabel(svg, "Spearman ρ between two independent runs (95% CI)", (m.left + W - m.right) / 2, H - 6, theme);
  }, [f4, width, theme, show, hide, wrapRef]);

  return (
    <ChartFrame
      title="What survives an independent rerun?"
      subtitle="Run 2 against run 3 for the same 48 accessions — no modelling. Grey: components of the phenotype; coloured: the rescue readings."
      wrapRef={wrapRef}
    >
      <svg ref={svgRef} role="img" aria-label="test-retest reproducibility" />
      <Tooltip tipRef={tipRef} />
    </ChartFrame>
  );
}

// ------------------------------------------------------------ retest scatter
export function RetestScatter({ f3 }) {
  const [organ, setOrgan] = useState("Shoot");
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const { tipRef, show, hide } = useTooltip();
  const svgRef = useRef(null);

  useEffect(() => {
    if (!f3 || !svgRef.current) return;
    const o = f3.organs[organ];
    const rows = o.rows.filter((r) => r.BR1 && r.retest_pooled).map((r) => ({
      name: r.name, g1: r.BR1.gain, g2: r.retest_pooled.gain, z1: r.BR1.lo <= 0, z2: r.retest_pooled.lo <= 0,
      lo2: r.retest_pooled.lo, hi2: r.retest_pooled.hi, lo1: r.BR1.lo, hi1: r.BR1.hi,
    }));
    const W = Math.min(width, 720), H = 380, m = { top: 16, right: 16, bottom: 44, left: 56 };
    const svg = d3.select(svgRef.current).attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const ext = d3.extent(rows.flatMap((r) => [r.g1, r.g2]));
    const dom = [Math.min(0, ext[0]) - 0.5, ext[1] + 1];
    const x = d3.scaleLinear().domain(dom).range([m.left, W - m.right]);
    const y = d3.scaleLinear().domain(dom).range([H - m.bottom, m.top]);
    gridY(svg, y, W - m.left - m.right, m.left, theme, 6);
    svg.append("g").attr("transform", `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).ticks(6)).call((s) => styleAxis(s, theme));
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(6)).call((s) => styleAxis(s, theme));
    svg.append("line").attr("x1", x(dom[0])).attr("y1", y(dom[0])).attr("x2", x(dom[1])).attr("y2", y(dom[1]))
      .attr("stroke", theme.fgMuted).attr("stroke-dasharray", "4 3");
    const mean = o.mean_gain_all_accessions_BR1;
    svg.append("line").attr("x1", m.left).attr("x2", W - m.right).attr("y1", y(mean)).attr("y2", y(mean))
      .attr("stroke", theme.fgMuted).attr("stroke-dasharray", "2 3");
    svg.append("text").attr("x", W - m.right - 4).attr("y", y(mean) - 5).attr("text-anchor", "end")
      .attr("fill", theme.fgMuted).attr("font-size", 11).text("mean of all 235 accessions, run 1");
    const fill = (r) => (r.z1 ? theme.bg : r.z2 ? theme.bg : theme.role.gain);
    const stroke = (r) => (r.z1 ? theme.role.alert : r.z2 ? theme.fg : theme.bg);
    svg.append("g").selectAll("circle").data(rows).join("circle")
      .attr("cx", (r) => x(r.g1)).attr("cy", (r) => y(r.g2)).attr("r", (r) => (r.z1 || r.z2 ? 6 : 4.5))
      .attr("fill", fill).attr("stroke", stroke).attr("stroke-width", (r) => (r.z1 || r.z2 ? 2 : 1))
      .on("mousemove", (ev, r) => show(
        `<b>${r.name}</b><br/>run 1: ${fmt2(r.g1)} mg [${fmt2(r.lo1)}, ${fmt2(r.hi1)}]<br/>runs 2–3: ${fmt2(r.g2)} mg [${fmt2(r.lo2)}, ${fmt2(r.hi2)}]${r.z1 ? "<br/>compatible with zero in run 1" : ""}${r.z2 ? "<br/>compatible with zero in the retest" : ""}`,
        ev, wrapRef.current))
      .on("mouseleave", hide);
    const lg = svg.append("g").attr("transform", `translate(${m.left + 10},${m.top + 8})`);
    [["rescued in both", theme.role.gain, theme.bg], ["CI reaches 0 in run 1", theme.bg, theme.role.alert],
      ["CI reaches 0 in retest", theme.bg, theme.fg]].forEach(([t, f, s], i) => {
      lg.append("circle").attr("cx", 0).attr("cy", i * 17).attr("r", 5).attr("fill", f).attr("stroke", s).attr("stroke-width", 2);
      lg.append("text").attr("x", 11).attr("y", i * 17 + 4).attr("fill", theme.fg).attr("font-size", 11).text(t);
    });
    axisLabel(svg, `${organ.toLowerCase()} gain in run 1 (mg)`, (m.left + W - m.right) / 2, H - 8, theme);
    axisLabel(svg, `${organ.toLowerCase()} gain, runs 2–3 pooled (mg)`, 14, (H - m.bottom + m.top) / 2, theme, true);
  }, [f3, organ, width, theme, show, hide, wrapRef]);

  const o = f3 ? f3.organs[organ] : null;
  return (
    <ChartFrame
      title="The weak rescuers, regrown"
      subtitle={o ? `Retest mean ${fmt2(o.mean_gain_retest)} mg vs ${fmt2(o.mean_gain_BR1)} mg in run 1: ${Math.round(100 * o.regression_toward_population_mean)}% of the way back to the population mean. Accessions compatible with zero in both: ${o.consistently_compatible_with_zero.length}.` : ""}
      controls={<Toggle options={ORG} value={organ} onChange={setOrgan} label="organ" />}
      wrapRef={wrapRef}
    >
      <svg ref={svgRef} role="img" aria-label="retest of weak rescuers" />
      <Tooltip tipRef={tipRef} />
    </ChartFrame>
  );
}

// -------------------------------------------------- drought vs non-stress
export function DroughtVsNonStress({ f6 }) {
  const [organ, setOrgan] = useState("Shoot");
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const { tipRef, show, hide } = useTooltip();
  const svgRef = useRef(null);

  useEffect(() => {
    if (!f6 || !svgRef.current) return;
    const A = f6[organ].accessions;
    const W = Math.min(width, 720), H = 380, m = { top: 16, right: 16, bottom: 44, left: 56 };
    const svg = d3.select(svgRef.current).attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const xe = d3.extent(A, (a) => a.log_promotion_nonstress), ye = d3.extent(A, (a) => a.log_promotion_drought);
    const x = d3.scaleLinear().domain([Math.min(xe[0], -0.2) - 0.05, Math.max(xe[1], 0.2) + 0.05]).range([m.left, W - m.right]);
    const y = d3.scaleLinear().domain([Math.min(0, ye[0]) - 0.05, ye[1] + 0.05]).range([H - m.bottom, m.top]);
    gridY(svg, y, W - m.left - m.right, m.left, theme, 6);
    svg.append("g").attr("transform", `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).ticks(6)).call((s) => styleAxis(s, theme));
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(6)).call((s) => styleAxis(s, theme));
    svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", m.top).attr("y2", H - m.bottom).attr("stroke", theme.fg);
    svg.append("line").attr("x1", m.left).attr("x2", W - m.right).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", theme.fg);
    svg.append("g").selectAll("circle").data(A).join("circle")
      .attr("cx", (a) => x(a.log_promotion_nonstress)).attr("cy", (a) => y(a.log_promotion_drought)).attr("r", 3.6)
      .attr("fill", (a) => (a.log_promotion_nonstress + 1.96 * a.se_pn < 0 ? theme.role.alert : theme.role.canonical))
      .attr("opacity", 0.8)
      .on("mousemove", (ev, a) => show(
        `<b>${a.name ?? a.gid}</b><br/>without drought: ${d3.format("+.0%")(Math.exp(a.log_promotion_nonstress) - 1)}<br/>under drought: ${d3.format("+.0%")(Math.exp(a.log_promotion_drought) - 1)}<br/>rescue ${fmt2(a.rescue)}`,
        ev, wrapRef.current))
      .on("mouseleave", hide);
    svg.append("text").attr("x", x(xe[0]) + 4).attr("y", m.top + 12).attr("fill", theme.role.alert).attr("font-size", 11)
      .text("● significantly inhibited without drought");
    axisLabel(svg, "log(W_N / M_N): WCS417 effect without drought", (m.left + W - m.right) / 2, H - 8, theme);
    axisLabel(svg, "log(W / M_D): WCS417 effect under drought", 14, (H - m.bottom + m.top) / 2, theme, true);
  }, [f6, organ, width, theme, show, hide, wrapRef]);

  const s = f6 ? f6[organ] : null;
  return (
    <ChartFrame
      title="The benefit exists only under drought"
      subtitle={s ? `Without drought: ${Math.round(100 * s.log_promotion_nonstress.frac_significantly_negative)}% of accessions significantly inhibited, ${Math.round(100 * s.log_promotion_nonstress.frac_significantly_positive)}% promoted. Under drought: ${Math.round(100 * s.log_promotion_drought.frac_significantly_positive)}% promoted. Correlation between the two: ρ = ${fmt2(s.spearman["promotion_nonstress~promotion_drought"])}.` : ""}
      controls={<Toggle options={ORG} value={organ} onChange={setOrgan} label="organ" />}
      wrapRef={wrapRef}
    >
      <svg ref={svgRef} role="img" aria-label="drought vs non-stress inoculation effect" />
      <Tooltip tipRef={tipRef} />
    </ChartFrame>
  );
}
