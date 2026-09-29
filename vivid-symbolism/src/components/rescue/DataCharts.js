// Charts that describe the screen itself: recovered cell means, gain with
// intervals, the Col-0 batch control, and reliability by construction.

import { useEffect, useMemo, useState } from "react";
import * as d3 from "d3";

import {
  ChartFrame, LABEL, SEQ, Toggle, Tooltip, axisLabel, gridY, styleAxis,
  useRescueTheme, useTooltip, useWidth,
} from "./kit";

const ORGANS = [{ value: "shoot", label: "shoot" }, { value: "root", label: "root" }];
const fmt = d3.format(".2f");

// ------------------------------------------------------------------ cell means
export function CellMeansChart({ accessions }) {
  const [organ, setOrgan] = useState("shoot");
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const { tipRef, show, hide } = useTooltip();
  const svgRef = useMemo(() => ({ current: null }), []);

  useEffect(() => {
    if (!accessions || !svgRef.current) return;
    const H = 360, m = { top: 34, right: 16, bottom: 40, left: 56 };
    const rows = accessions.map((a) => ({
      a, MN: a[`${organ}_MN`], MD: a[`${organ}_MD`], W: a[`${organ}_W`],
    })).sort((p, q) => p.MN - q.MN);
    const svg = d3.select(svgRef.current).attr("viewBox", `0 0 ${width} ${H}`).attr("width", width).attr("height", H);
    svg.selectAll("*").remove();
    const x = d3.scaleLinear().domain([0, rows.length - 1]).range([m.left, width - m.right]);
    const y = d3.scaleLog().domain([d3.min(rows, (r) => r.MD) * 0.8, d3.max(rows, (r) => r.MN) * 1.2]).range([H - m.bottom, m.top]);
    gridY(svg, y, width - m.left - m.right, m.left, theme, 5);
    svg.append("g").attr("transform", `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).ticks(6)).call((s) => styleAxis(s, theme));
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5, "~s")).call((s) => styleAxis(s, theme));
    axisLabel(svg, `accession, sorted by mock non-stress ${organ} mass`, (m.left + width - m.right) / 2, H - 6, theme);
    axisLabel(svg, "fresh weight (mg, log)", 14, (H - m.bottom + m.top) / 2, theme, true);
    const series = [
      { key: "MN", color: theme.muted, name: "mock, non-stress (M_N)" },
      { key: "W", color: theme.role.gain, name: "WCS417, drought (W)" },
      { key: "MD", color: theme.fg, name: "mock, drought (M_D)" },
    ];
    series.forEach((s) => {
      svg.append("g").selectAll("circle").data(rows).join("circle")
        .attr("cx", (_, i) => x(i)).attr("cy", (r) => y(r[s.key])).attr("r", 2.4)
        .attr("fill", s.color);
    });
    // hover column per accession
    const band = (width - m.left - m.right) / rows.length;
    svg.append("g").selectAll("rect").data(rows).join("rect")
      .attr("x", (_, i) => x(i) - band / 2).attr("width", Math.max(band, 2))
      .attr("y", m.top).attr("height", H - m.top - m.bottom).attr("fill", "transparent")
      .on("mousemove", (ev, r) => show(
        `<b>${r.a.name ?? r.a.gid}</b> (${r.a.country ?? "?"})<br/>M_N ${fmt(r.MN)} mg<br/>W ${fmt(r.W)} mg<br/>M_D ${fmt(r.MD)} mg<br/>loss ${d3.format(".0%")(1 - r.MD / r.MN)}`,
        ev, wrapRef.current))
      .on("mouseleave", hide);
    const lg = svg.append("g").attr("transform", `translate(${m.left + 4},12)`);
    series.forEach((s, i) => {
      lg.append("circle").attr("cx", i * 190).attr("cy", 0).attr("r", 4).attr("fill", s.color);
      lg.append("text").attr("x", i * 190 + 10).attr("y", 4).attr("fill", theme.fg).attr("font-size", 11).text(s.name);
    });
  }, [accessions, organ, width, theme, show, hide, svgRef, wrapRef]);

  return (
    <ChartFrame
      title="Recovered cell means, 235 accessions"
      subtitle="Only W was measured plant by plant. M_D and M_N are recovered exactly from the derived traits. Hover an accession."
      controls={<Toggle options={ORGANS} value={organ} onChange={setOrgan} label="organ" />}
      wrapRef={wrapRef}
    >
      <svg ref={(el) => { svgRef.current = el; }} role="img" aria-label="cell means per accession" />
      <Tooltip tipRef={tipRef} />
    </ChartFrame>
  );
}

// ------------------------------------------------------------- gain caterpillar
export function GainCaterpillar({ e2, accessions }) {
  const [organ, setOrgan] = useState("Shoot");
  const [model, setModel] = useState("M3_plant_plus_batch");
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const { tipRef, show, hide } = useTooltip();
  const svgRef = useMemo(() => ({ current: null }), []);
  const [nZero, setNZero] = useState(0);

  useEffect(() => {
    if (!e2 || !accessions || !svgRef.current) return;
    const d = e2.organs[organ][model].gain;
    const zeroSet = new Set((d.ci_includes_zero || []).map((z) => z.gid));
    const ref = e2.organs[organ].M3_plant_plus_batch.gain.detectability.per_accession;
    const rows = accessions.map((a, i) => {
      // intervals are always drawn under M3; the model toggle only changes which are flagged
      const g = ref.gain[i];
      return { a, g, lo: ref.lo[i], hi: ref.hi[i], zero: zeroSet.has(a.gid) };
    }).sort((p, q) => p.g - q.g);
    setNZero(zeroSet.size);
    const H = 300, m = { top: 12, right: 16, bottom: 40, left: 56 };
    const svg = d3.select(svgRef.current).attr("viewBox", `0 0 ${width} ${H}`).attr("width", width).attr("height", H);
    svg.selectAll("*").remove();
    const x = d3.scaleLinear().domain([0, rows.length - 1]).range([m.left, width - m.right]);
    const y = d3.scaleLinear().domain([Math.min(0, d3.min(rows, (r) => r.lo)), d3.max(rows, (r) => r.hi)]).nice().range([H - m.bottom, m.top]);
    gridY(svg, y, width - m.left - m.right, m.left, theme, 5);
    svg.append("g").attr("transform", `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).ticks(6)).call((s) => styleAxis(s, theme));
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5)).call((s) => styleAxis(s, theme));
    svg.append("line").attr("x1", m.left).attr("x2", width - m.right).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", theme.fg);
    svg.append("g").selectAll("line").data(rows).join("line")
      .attr("x1", (_, i) => x(i)).attr("x2", (_, i) => x(i))
      .attr("y1", (r) => y(r.lo)).attr("y2", (r) => y(r.hi))
      .attr("stroke", (r) => (r.zero ? theme.role.alert : theme.dark ? SEQ[4] : SEQ[1]))
      .attr("stroke-width", (r) => (r.zero ? 1.6 : 1));
    svg.append("g").selectAll("circle").data(rows).join("circle")
      .attr("cx", (_, i) => x(i)).attr("cy", (r) => y(r.g)).attr("r", 2).attr("fill", theme.fg);
    const band = (width - m.left - m.right) / rows.length;
    svg.append("g").selectAll("rect").data(rows).join("rect")
      .attr("x", (_, i) => x(i) - band / 2).attr("width", Math.max(band, 2)).attr("y", m.top)
      .attr("height", H - m.top - m.bottom).attr("fill", "transparent")
      .on("mousemove", (ev, r) => show(
        `<b>${r.a.name ?? r.a.gid}</b><br/>gain ${fmt(r.g)} mg<br/>95% interval [${fmt(r.lo)}, ${fmt(r.hi)}]${r.zero ? "<br/><b>compatible with zero under this model</b>" : ""}`,
        ev, wrapRef.current))
      .on("mouseleave", hide);
    axisLabel(svg, "accession, sorted by gain", (m.left + width - m.right) / 2, H - 6, theme);
    axisLabel(svg, `gain W − M_D (mg ${organ.toLowerCase()})`, 14, (H - m.bottom + m.top) / 2, theme, true);
  }, [e2, accessions, organ, model, width, theme, show, hide, svgRef, wrapRef]);

  return (
    <ChartFrame
      title="Could a loss of rescue have been seen?"
      subtitle={`Gain with 95% intervals (batch-inclusive noise). Red: gain compatible with zero under the selected noise model — ${nZero} of 235 accessions.`}
      controls={<>
        <Toggle options={[{ value: "Shoot", label: "shoot" }, { value: "Root", label: "root" }]} value={organ} onChange={setOrgan} label="organ" />
        <Toggle options={[{ value: "M2_plant_all_cells", label: "plant noise" }, { value: "M3_plant_plus_batch", label: "+ batch" }]} value={model} onChange={setModel} label="flag under" />
      </>}
      wrapRef={wrapRef}
    >
      <svg ref={(el) => { svgRef.current = el; }} role="img" aria-label="gain with intervals" />
      <Tooltip tipRef={tipRef} />
    </ChartFrame>
  );
}

// ------------------------------------------------------------------ Col-0 blocks
export function BatchBlocks({ e7, accessions }) {
  const [organ, setOrgan] = useState("Shoot");
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const { tipRef, show, hide } = useTooltip();
  const svgRef = useMemo(() => ({ current: null }), []);

  useEffect(() => {
    if (!e7 || !accessions || !svgRef.current) return;
    const b = e7[organ];
    const acc = accessions.map((a) => a[`${organ.toLowerCase()}_W`]).sort(d3.ascending);
    const H = 280, m = { top: 14, right: 16, bottom: 40, left: 56 };
    const svg = d3.select(svgRef.current).attr("viewBox", `0 0 ${width} ${H}`).attr("width", width).attr("height", H);
    svg.selectAll("*").remove();
    const x = d3.scaleBand().domain(d3.range(b.block_means.length)).range([m.left, width - m.right]).padding(0.35);
    const lo = d3.quantile(acc, 0.05), hi = d3.quantile(acc, 0.95);
    const y = d3.scaleLinear().domain([Math.min(lo, d3.min(b.block_means)) * 0.92, Math.max(hi, d3.max(b.block_means)) * 1.05]).range([H - m.bottom, m.top]);
    gridY(svg, y, width - m.left - m.right, m.left, theme, 5);
    svg.append("rect").attr("x", m.left).attr("width", width - m.left - m.right)
      .attr("y", y(hi)).attr("height", y(lo) - y(hi)).attr("fill", theme.accentSoft).attr("opacity", 0.5);
    svg.append("text").attr("x", width - m.right - 4).attr("y", y(hi) - 4).attr("text-anchor", "end")
      .attr("fill", theme.fgMuted).attr("font-size", 11).text("5–95% of accession means");
    svg.append("line").attr("x1", m.left).attr("x2", width - m.right).attr("y1", y(b.grand_mean)).attr("y2", y(b.grand_mean))
      .attr("stroke", theme.fgMuted).attr("stroke-dasharray", "4 3");
    svg.append("g").attr("transform", `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).tickFormat((i) => i + 1)).call((s) => styleAxis(s, theme));
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5)).call((s) => styleAxis(s, theme));
    svg.append("g").selectAll("rect.bar").data(b.block_means).join("rect")
      .attr("x", (_, i) => x(i)).attr("width", x.bandwidth())
      .attr("y", (v) => y(v) - 2).attr("height", 4).attr("rx", 2).attr("fill", theme.fg)
      .on("mousemove", (ev, v) => {
        const pct = d3.bisectLeft(acc, v) / acc.length;
        show(`Col-0 block mean ${fmt(v)} mg<br/>would rank at the ${d3.format(".0%")(pct)} percentile of accessions`, ev, wrapRef.current);
      })
      .on("mouseleave", hide);
    axisLabel(svg, "Col-0 block, in sowing order", (m.left + width - m.right) / 2, H - 6, theme);
    axisLabel(svg, `${organ.toLowerCase()} FW, WCS417 drought (mg)`, 14, (H - m.bottom + m.top) / 2, theme, true);
  }, [e7, accessions, organ, width, theme, show, hide, svgRef, wrapRef]);

  const s = e7 ? e7[organ] : null;
  return (
    <ChartFrame
      title="The internal batch control"
      subtitle={s ? `Same genotype, 13 blocks: F = ${s.F.toFixed(2)}, p = ${s.p.toExponential(1)}, ICC = ${s.ICC1.toFixed(2)}, batch CV = ${s.batch_cv.toFixed(3)}. Hover a block.` : ""}
      controls={<Toggle options={[{ value: "Shoot", label: "shoot" }, { value: "Root", label: "root" }]} value={organ} onChange={setOrgan} label="organ" />}
      wrapRef={wrapRef}
    >
      <svg ref={(el) => { svgRef.current = el; }} role="img" aria-label="Col-0 block means" />
      <Tooltip tipRef={tipRef} />
    </ChartFrame>
  );
}

// --------------------------------------------------------------- reliability
export function ReliabilityBars({ e2 }) {
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const { tipRef, show, hide } = useTooltip();
  const svgRef = useMemo(() => ({ current: null }), []);
  const models = [
    ["M2_plant_all_cells", "plant noise"],
    ["M3_plant_plus_batch", "+ batch"],
    ["M4_M3_mock_noise_x1.5", "+ batch, mock ×1.5"],
  ];
  useEffect(() => {
    if (!e2 || !svgRef.current) return;
    const data = [];
    ["Shoot", "Root"].forEach((o) => ["gain", "increase", "rescue"].forEach((c) => models.forEach(([mk, ml], mi) => {
      const r = e2.organs[o][mk][c];
      data.push({ o, c, mk, ml, mi, v: r.reliability, Q: r.Q, I2: r.I2 });
    })));
    const H = 280, m = { top: 26, right: 16, bottom: 48, left: 52 };
    const svg = d3.select(svgRef.current).attr("viewBox", `0 0 ${width} ${H}`).attr("width", width).attr("height", H);
    svg.selectAll("*").remove();
    const x0 = d3.scaleBand().domain(["Shoot", "Root"]).range([m.left, width - m.right]).paddingInner(0.12);
    const x1 = d3.scaleBand().domain(["gain", "increase", "rescue"]).range([0, x0.bandwidth()]).paddingInner(0.18);
    const x2 = d3.scaleBand().domain([0, 1, 2]).range([0, x1.bandwidth()]).paddingInner(0.1);
    const y = d3.scaleLinear().domain([0, 1]).range([H - m.bottom, m.top]);
    gridY(svg, y, width - m.left - m.right, m.left, theme, 5);
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5)).call((s) => styleAxis(s, theme));
    svg.append("g").selectAll("rect").data(data).join("rect")
      .attr("x", (d) => x0(d.o) + x1(d.c) + x2(d.mi)).attr("width", x2.bandwidth())
      .attr("y", (d) => y(d.v)).attr("height", (d) => y(0) - y(d.v))
      .attr("rx", 2).attr("fill", (d) => theme.role[d.c]).attr("opacity", (d) => [1, 0.62, 0.34][d.mi])
      .on("mousemove", (ev, d) => show(`<b>${d.o.toLowerCase()} · ${LABEL[d.c]}</b><br/>${d.ml}<br/>reliability ${fmt(d.v)}<br/>Q = ${Math.round(d.Q)} (df 234), I² = ${fmt(d.I2)}`, ev, wrapRef.current))
      .on("mouseleave", hide);
    ["Shoot", "Root"].forEach((o) => {
      ["gain", "increase", "rescue"].forEach((c) => {
        svg.append("text").attr("x", x0(o) + x1(c) + x1.bandwidth() / 2).attr("y", H - m.bottom + 16)
          .attr("text-anchor", "middle").attr("fill", theme.fg).attr("font-size", 11).text(LABEL[c]);
      });
      svg.append("text").attr("x", x0(o) + x0.bandwidth() / 2).attr("y", H - 8).attr("text-anchor", "middle")
        .attr("fill", theme.fgMuted).attr("font-size", 12).text(o.toLowerCase());
    });
    const lg = svg.append("g").attr("transform", `translate(${m.left + 6},10)`);
    models.forEach(([, ml], i) => {
      lg.append("rect").attr("x", i * 140).attr("y", -6).attr("width", 12).attr("height", 10).attr("rx", 2)
        .attr("fill", theme.fg).attr("opacity", [1, 0.62, 0.34][i]);
      lg.append("text").attr("x", i * 140 + 16).attr("y", 3).attr("fill", theme.fg).attr("font-size", 11).text(ml);
    });
    axisLabel(svg, "reliability τ²/(τ²+se²)", 14, (H - m.bottom + m.top) / 2, theme, true);
  }, [e2, width, theme, show, hide, svgRef, wrapRef]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <ChartFrame
      title="How much of each phenotype is between accessions?"
      subtitle="Reliability is an upper bound on the variance any locus could explain. Hover a bar for Cochran's Q."
      wrapRef={wrapRef}
    >
      <svg ref={(el) => { svgRef.current = el; }} role="img" aria-label="reliability by construction and noise model" />
      <Tooltip tipRef={tipRef} />
    </ChartFrame>
  );
}
