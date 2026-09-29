// Charts about what the screen can certify: verdict calibration, a live
// four-column verdict explorer over all 27,495 pairs, resolution tiers,
// the replication design, and collection sites.

import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";

import {
  ChartFrame, SEQ, Slider, Toggle, Tooltip, Z, axisLabel, gridY, styleAxis,
  useRescueTheme, useTooltip, useWidth,
} from "./kit";

const fmt2 = d3.format(".2f");
const pct = d3.format(".1%");
const ORG = [{ value: "shoot", label: "shoot" }, { value: "root", label: "root" }];
const cap = (o) => o[0].toUpperCase() + o.slice(1);

// --------------------------------------------------------------- calibration
// JSON keys are Python float reprs ("1.0"); JS stringifies 1 as "1", so match numerically.
const curveFor = (e6, lv) => e6.curves[Object.keys(e6.curves).find((k) => Math.abs(parseFloat(k) - lv) < 1e-9)];

export function CalibrationCurves({ e6 }) {
  const levels = useMemo(() => (e6 ? e6.s_over_delta_levels : []), [e6]);
  const [sel, setSel] = useState(null);
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const svgRef = useRef(null);
  const active = sel ?? (levels.length ? levels[levels.length - 1] : null);

  useEffect(() => {
    if (!e6 || !svgRef.current) return;
    const W = width, H = 280, m = { top: 16, right: 16, bottom: 42, left: 52 };
    const svg = d3.select(svgRef.current).attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const x = d3.scaleLinear().domain([0, 3]).range([m.left, W - m.right]);
    const y = d3.scaleLinear().domain([0, 1]).range([H - m.bottom, m.top]);
    gridY(svg, y, W - m.left - m.right, m.left, theme, 5);
    svg.append("g").attr("transform", `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).ticks(6)).call((s) => styleAxis(s, theme));
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5)).call((s) => styleAxis(s, theme));
    svg.append("rect").attr("x", x(0)).attr("width", x(1) - x(0)).attr("y", m.top).attr("height", H - m.top - m.bottom)
      .attr("fill", theme.accentSoft).attr("opacity", 0.35);
    svg.append("text").attr("x", x(0.5)).attr("y", m.top + 14).attr("text-anchor", "middle").attr("fill", theme.fgMuted)
      .attr("font-size", 11).text("truly within the margin");
    svg.append("line").attr("x1", m.left).attr("x2", W - m.right).attr("y1", y(0.05)).attr("y2", y(0.05))
      .attr("stroke", theme.fgMuted).attr("stroke-dasharray", "2 3");
    levels.forEach((lv) => {
      const rows = curveFor(e6, lv);
      const on = lv === active;
      [["P_C", null], ["P_D", "5 4"]].forEach(([k, dash]) => {
        svg.append("path").datum(rows)
          .attr("d", d3.line().x((r) => x(r.true_diff_over_delta)).y((r) => y(r[k])))
          .attr("fill", "none").attr("stroke", on ? theme.role.canonical : theme.fgMuted)
          .attr("stroke-width", on ? 2.6 : 1).attr("opacity", on ? 1 : 0.35).attr("stroke-dasharray", dash);
      });
    });
    const rows = curveFor(e6, active);
    svg.append("text").attr("x", W - m.right).attr("y", y(rows[rows.length - 1].P_D) - 6).attr("text-anchor", "end")
      .attr("fill", theme.fg).attr("font-size", 11).text("P(diverge)");
    svg.append("text").attr("x", x(0.05)).attr("y", y(rows[0].P_C) + (rows[0].P_C > 0.5 ? 16 : -6)).attr("fill", theme.fg)
      .attr("font-size", 11).text("P(correspond)");
    axisLabel(svg, "true difference / margin δ", (m.left + W - m.right) / 2, H - 8, theme);
    axisLabel(svg, "probability of verdict", 12, (H - m.bottom + m.top) / 2, theme, true);
  }, [e6, width, theme, levels, active]);

  return (
    <ChartFrame
      title="The verdict is calibrated at every noise level"
      subtitle={e6 ? `Pair standard error s relative to the margin δ. This screen sits at s/δ = ${e6["empirical_s_over_delta_at_0.5sd"].toFixed(2)} (margin 0.5 SD): correspondence is unreachable. Worst false-correspond ${e6.max_false_correspond_at_or_beyond_margin.toFixed(3)}, worst false-diverge ${e6.max_false_diverge_at_or_within_margin.toFixed(3)}.` : ""}
      controls={<Toggle options={levels.map((l) => ({ value: l, label: `s/δ ${l}` }))} value={active} onChange={setSel} />}
      wrapRef={wrapRef}
    >
      <svg ref={svgRef} role="img" aria-label="verdict calibration curves" />
    </ChartFrame>
  );
}

// -------------------------------------------------------- four-column explorer
export function VerdictExplorer({ accessions, index, e5 }) {
  const [organ, setOrgan] = useState("shoot");
  const [mResp, setMResp] = useState(0.5);
  const [mBase, setMBase] = useState(30);
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const { tipRef, show, hide } = useTooltip();
  const cdfRef = useRef(null);
  const tabRef = useRef(null);

  const pairs = useMemo(() => {
    if (!accessions || !index) return null;
    const v = accessions.map((a) => a[`${organ}_canonical`]);
    const s = accessions.map((a) => a[`${organ}_canonical_se`]);
    const lmd = accessions.map((a) => Math.log(a[`${organ}_MD`]));
    const lmn = accessions.map((a) => Math.log(a[`${organ}_MN`]));
    const O = cap(organ);
    const cp = index.plant_cv[O], cb = index.batch_cv[O];
    const sb = Math.sqrt(Math.log1p(cp * cp / 7) + Math.log1p(cb * cb));
    const sd = d3.deviation(v);
    const n = v.length;
    const P = n * (n - 1) / 2;
    const d = new Float64Array(P), sp = new Float64Array(P), bmd = new Float64Array(P), bmn = new Float64Array(P);
    const ii = new Int32Array(P), jj = new Int32Array(P);
    let k = 0;
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        d[k] = v[i] - v[j];
        sp[k] = Math.hypot(s[i], s[j]);
        bmd[k] = Math.abs(lmd[i] - lmd[j]);
        bmn[k] = Math.abs(lmn[i] - lmn[j]);
        ii[k] = i; jj[k] = j; k += 1;
      }
    }
    const dmin = Array.from(d, (x, t) => (Math.abs(x) + Z * sp[t]) / sd).sort(d3.ascending);
    return { d, sp, bmd, bmn, sb: Math.SQRT2 * sb, sd, dmin, P, ii, jj };
  }, [accessions, index, organ]);

  const table = useMemo(() => {
    if (!pairs) return null;
    const delta = mResp * pairs.sd;
    const db = Math.log(1 + mBase / 100);
    const T = { C: { C: 0, D: 0, U: 0 }, D: { C: 0, D: 0, U: 0 }, U: { C: 0, D: 0, U: 0 } };
    const ex = { CD: null, DC: null };
    let bestFF = -Infinity, bestCV = -Infinity;
    for (let t = 0; t < pairs.P; t += 1) {
      const ad = Math.abs(pairs.d[t]);
      const r = ad + Z * pairs.sp[t] < delta ? "C" : ad - Z * pairs.sp[t] > delta ? "D" : "U";
      const cMD = pairs.bmd[t] + Z * pairs.sb < db, cMN = pairs.bmn[t] + Z * pairs.sb < db;
      const dMD = pairs.bmd[t] - Z * pairs.sb > db, dMN = pairs.bmn[t] - Z * pairs.sb > db;
      const b = cMD && cMN ? "C" : dMD || dMN ? "D" : "U";
      T[b][r] += 1;
      if (b === "C" && r === "D" && ad > bestFF) { bestFF = ad; ex.CD = t; }
      if (b === "D" && r === "C" && pairs.bmd[t] + pairs.bmn[t] > bestCV) { bestCV = pairs.bmd[t] + pairs.bmn[t]; ex.DC = t; }
    }
    return { T, ex };
  }, [pairs, mResp, mBase]);

  useEffect(() => {
    if (!pairs || !cdfRef.current) return;
    const W = width > 760 ? Math.floor((width - 20) / 2) : width, H = 260, m = { top: 16, right: 16, bottom: 42, left: 52 };
    const svg = d3.select(cdfRef.current).attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const x = d3.scaleLog().domain([0.3, 8]).range([m.left, W - m.right]);
    const y = d3.scaleLinear().domain([0, 1]).range([H - m.bottom, m.top]);
    gridY(svg, y, W - m.left - m.right, m.left, theme, 5);
    svg.append("g").attr("transform", `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).tickValues([0.5, 1, 2, 4, 8]).tickFormat((v) => `${v}`)).call((s) => styleAxis(s, theme));
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%"))).call((s) => styleAxis(s, theme));
    const step = Math.max(1, Math.floor(pairs.dmin.length / 400));
    const pts = pairs.dmin.filter((_, i) => i % step === 0).map((v, i) => ({ v, p: (i * step) / pairs.dmin.length }));
    svg.append("path").datum(pts).attr("d", d3.line().x((p) => x(Math.max(0.3, Math.min(8, p.v)))).y((p) => y(p.p)))
      .attr("fill", "none").attr("stroke", theme.role.alert).attr("stroke-width", 2.4);
    if (e5) {
      const q = e5.organs[cap(organ)].models.M2_plant_all_cells.response.canonical.min_certifiable_margin_sd_quantiles;
      const ps = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95];
      svg.append("path").datum(q.map((v, i) => ({ v, p: ps[i] })))
        .attr("d", d3.line().x((p) => x(p.v)).y((p) => y(p.p))).attr("fill", "none")
        .attr("stroke", theme.role.gain).attr("stroke-width", 2).attr("stroke-dasharray", "5 4");
    }
    const frac = d3.bisectLeft(pairs.dmin, mResp) / pairs.dmin.length;
    svg.append("line").attr("x1", x(mResp)).attr("x2", x(mResp)).attr("y1", m.top).attr("y2", H - m.bottom)
      .attr("stroke", theme.fg).attr("stroke-dasharray", "3 3");
    svg.append("text").attr("x", x(mResp) + 5).attr("y", m.top + 12).attr("fill", theme.fg).attr("font-size", 11)
      .text(`${pct(frac)} certifiable at ${mResp} SD`);
    svg.append("text").attr("x", W - m.right).attr("y", H - m.bottom - 24).attr("text-anchor", "end").attr("fill", theme.role.alert).attr("font-size", 11).text("— + batch (live)");
    svg.append("text").attr("x", W - m.right).attr("y", H - m.bottom - 8).attr("text-anchor", "end").attr("fill", theme.role.gain).attr("font-size", 11).text("- - plant noise (from E5)");
    axisLabel(svg, "smallest certifiable margin (× between-accession SD)", (m.left + W - m.right) / 2, H - 8, theme);
    axisLabel(svg, "fraction of pairs", 12, (H - m.bottom + m.top) / 2, theme, true);
  }, [pairs, width, theme, mResp, e5, organ]);

  useEffect(() => {
    if (!table || !tabRef.current) return;
    const W = width > 760 ? Math.floor((width - 20) / 2) : width, H = 260, m = { top: 22, right: 12, bottom: 40, left: 96 };
    const svg = d3.select(tabRef.current).attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const keys = ["C", "U", "D"];
    const names = { C: "correspond", U: "decline", D: "diverge" };
    const x = d3.scaleBand().domain(keys).range([m.left, W - m.right]).padding(0.06);
    const y = d3.scaleBand().domain(keys).range([m.top, H - m.bottom]).padding(0.06);
    const max = d3.max(keys.flatMap((b) => keys.map((r) => table.T[b][r])));
    const col = d3.scaleSequentialLog().domain([1, Math.max(2, max)]).range([SEQ[0], SEQ[6]]);
    keys.forEach((b) => keys.forEach((r) => {
      const v = table.T[b][r];
      const cell = svg.append("g");
      cell.append("rect").attr("x", x(r)).attr("y", y(b)).attr("width", x.bandwidth()).attr("height", y.bandwidth())
        .attr("rx", 4).attr("fill", v === 0 ? theme.grid : col(v))
        .attr("stroke", (b === "C" && r === "D") || (b === "D" && r === "C") ? theme.role.canonical : "none").attr("stroke-width", 2.5)
        .on("mousemove", (ev) => show(`baseline <b>${names[b]}</b>, response <b>${names[r]}</b><br/>${v.toLocaleString()} pairs${b === "C" && r === "D" ? "<br/>false friends: same mock phenotype, different rescue" : ""}${b === "D" && r === "C" ? "<br/>convergent: different mock phenotype, same rescue" : ""}`, ev, wrapRef.current))
        .on("mouseleave", hide);
      cell.append("text").attr("x", x(r) + x.bandwidth() / 2).attr("y", y(b) + y.bandwidth() / 2 + 5).attr("text-anchor", "middle")
        .attr("font-size", 13).attr("font-weight", 600).attr("fill", v > max / 6 ? "#ffffff" : "#0b0b0b").attr("pointer-events", "none")
        .text(v.toLocaleString());
    }));
    keys.forEach((k) => {
      svg.append("text").attr("x", x(k) + x.bandwidth() / 2).attr("y", H - m.bottom + 16).attr("text-anchor", "middle").attr("fill", theme.fg).attr("font-size", 11).text(names[k]);
      svg.append("text").attr("x", m.left - 8).attr("y", y(k) + y.bandwidth() / 2 + 4).attr("text-anchor", "end").attr("fill", theme.fg).attr("font-size", 11).text(names[k]);
    });
    axisLabel(svg, "response columns (canonical rescue)", (m.left + W - m.right) / 2, H - 6, theme);
    svg.append("text").attr("x", m.left - 8).attr("y", 12).attr("text-anchor", "end").attr("fill", theme.fgMuted).attr("font-size", 11).text("baseline ↓");
  }, [table, width, theme, show, hide, wrapRef]);

  const exName = (t) => (t === null || !pairs ? "—" : `${accessions[pairs.ii[t]].name} vs ${accessions[pairs.jj[t]].name}`);
  return (
    <ChartFrame
      title="Four columns, three verdicts, 27,495 pairs"
      subtitle="Every pair of accessions is compared on its two mock baselines and its two responses. Recomputed live (batch-inclusive standard errors)."
      controls={<>
        <Toggle options={ORG} value={organ} onChange={setOrgan} label="organ" />
        <Slider label="response margin" min={0.25} max={3} step={0.05} value={mResp} onChange={setMResp} format={(v) => `${v.toFixed(2)} SD`} />
        <Slider label="baseline margin" min={10} max={80} step={5} value={mBase} onChange={setMBase} format={(v) => `${v}%`} />
      </>}
      wrapRef={wrapRef}
    >
      <div className={`grid gap-4 ${width > 760 ? "grid-cols-2" : "grid-cols-1"}`}>
        <svg ref={cdfRef} role="img" aria-label="minimum certifiable margin distribution" />
        <svg ref={tabRef} role="img" aria-label="four-column verdict table" />
      </div>
      {table ? (
        <div className="mt-2 grid grid-cols-2 gap-3 text-xs font-medium text-dark/80 dark:text-light/80 md:grid-cols-1">
          <div>Most divergent false friend at these margins: <b className="font-mono">{exName(table.ex.CD)}</b></div>
          <div>Most distant convergent pair: <b className="font-mono">{exName(table.ex.DC)}</b></div>
        </div>
      ) : null}
      <Tooltip tipRef={tipRef} />
    </ChartFrame>
  );
}

// ---------------------------------------------------------------------- tiers
export function TierChart({ accessions, e10 }) {
  const [organ, setOrgan] = useState("shoot");
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const { tipRef, show, hide } = useTooltip();
  const svgRef = useRef(null);

  useEffect(() => {
    if (!accessions || !svgRef.current) return;
    const rows = accessions.map((a) => ({ a, v: a[`${organ}_canonical`], s: a[`${organ}_canonical_se`], t: a[`${organ}_tier`] }))
      .sort((p, q) => p.v - q.v);
    const nt = d3.max(rows, (r) => r.t);
    const col = (t) => SEQ[Math.min(6, 1 + Math.round((5 * (t - 1)) / Math.max(nt - 1, 1)))];
    const W = width, H = 280, m = { top: 16, right: 16, bottom: 40, left: 56 };
    const svg = d3.select(svgRef.current).attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const x = d3.scaleLinear().domain([0, rows.length - 1]).range([m.left, W - m.right]);
    const y = d3.scaleLinear().domain([d3.min(rows, (r) => r.v - 1.96 * r.s), d3.max(rows, (r) => r.v + 1.96 * r.s)]).nice().range([H - m.bottom, m.top]);
    gridY(svg, y, W - m.left - m.right, m.left, theme, 5);
    svg.append("g").attr("transform", `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).ticks(6)).call((s) => styleAxis(s, theme));
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5)).call((s) => styleAxis(s, theme));
    svg.append("g").selectAll("line").data(rows).join("line")
      .attr("x1", (_, i) => x(i)).attr("x2", (_, i) => x(i))
      .attr("y1", (r) => y(r.v - 1.96 * r.s)).attr("y2", (r) => y(r.v + 1.96 * r.s)).attr("stroke", theme.grid);
    svg.append("g").selectAll("circle").data(rows).join("circle")
      .attr("cx", (_, i) => x(i)).attr("cy", (r) => y(r.v)).attr("r", 3.2).attr("fill", (r) => col(r.t))
      .attr("stroke", theme.bg).attr("stroke-width", 0.6)
      .on("mousemove", (ev, r) => show(`<b>${r.a.name ?? r.a.gid}</b><br/>canonical rescue ${fmt2(r.v)} ± ${fmt2(1.96 * r.s)}<br/>tier ${r.t} of ${nt}`, ev, wrapRef.current))
      .on("mouseleave", hide);
    const lg = svg.append("g").attr("transform", `translate(${m.left + 10},${m.top + 6})`);
    d3.range(1, nt + 1).forEach((t, i) => {
      lg.append("circle").attr("cx", 0).attr("cy", i * 15).attr("r", 4.5).attr("fill", col(t));
      lg.append("text").attr("x", 10).attr("y", i * 15 + 4).attr("fill", theme.fg).attr("font-size", 11)
        .text(`tier ${t} (n=${rows.filter((r) => r.t === t).length})`);
    });
    axisLabel(svg, "accession, sorted by canonical rescue", (m.left + W - m.right) / 2, H - 6, theme);
    axisLabel(svg, "canonical rescue (95% interval)", 14, (H - m.bottom + m.top) / 2, theme, true);
  }, [accessions, organ, width, theme, show, hide, wrapRef]);

  const d = e10 ? e10[cap(organ)] : null;
  return (
    <ChartFrame
      title="How many levels of rescue can the screen tell apart?"
      subtitle={d ? `Longest chain of certified differences: ${d.M3_plant_plus_batch.max_depth.canonical} tiers with batch noise, ${d.M2_plant_all_cells.max_depth.canonical} with plant noise only.` : ""}
      controls={<Toggle options={ORG} value={organ} onChange={setOrgan} label="organ" />}
      wrapRef={wrapRef}
    >
      <svg ref={svgRef} role="img" aria-label="resolution tiers" />
      <Tooltip tipRef={tipRef} />
    </ChartFrame>
  );
}

// ---------------------------------------------------------------------- design
export function DesignHeatmap({ e12 }) {
  const [organ, setOrgan] = useState("Root");
  const [metric, setMetric] = useState("reliability");
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const { tipRef, show, hide } = useTooltip();
  const svgRef = useRef(null);

  useEffect(() => {
    if (!e12 || !svgRef.current) return;
    const tab = e12.organs[organ].table;
    const gn = e12.grid_n, gr = e12.grid_r;
    const W = Math.min(width, 620), H = 250, m = { top: 12, right: 12, bottom: 42, left: 70 };
    const svg = d3.select(svgRef.current).attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const x = d3.scaleBand().domain(gr).range([m.left, W - m.right]).padding(0.05);
    const y = d3.scaleBand().domain(gn.slice().reverse()).range([m.top, H - m.bottom]).padding(0.05);
    const col = d3.scaleQuantize().domain([0, 1]).range(SEQ);
    gn.forEach((n) => gr.forEach((r) => {
      const c = tab[`${n}x${r}`];
      const v = c[metric];
      svg.append("rect").attr("x", x(r)).attr("y", y(n)).attr("width", x.bandwidth()).attr("height", y.bandwidth())
        .attr("rx", 3).attr("fill", col(v))
        .attr("stroke", n === 7 && r === 1 ? theme.role.alert : "none").attr("stroke-width", 2.5)
        .on("mousemove", (ev) => show(`<b>${n} plants × ${r} run${r > 1 ? "s" : ""}</b>${n === 7 && r === 1 ? " (current design)" : ""}<br/>reliability ${fmt2(c.reliability)}<br/>se / τ ${fmt2(c.se_over_tau)}<br/>P(certify equal at 1τ) ${fmt2(c["P_certify_equal_1.0"])}<br/>P(certify diff at 2τ) ${fmt2(c["P_certify_diff_at_2.0tau"])}`, ev, wrapRef.current))
        .on("mouseleave", hide);
      svg.append("text").attr("x", x(r) + x.bandwidth() / 2).attr("y", y(n) + y.bandwidth() / 2 + 4).attr("text-anchor", "middle")
        .attr("font-size", 12).attr("font-family", "ui-monospace, monospace").attr("pointer-events", "none")
        .attr("fill", v > 0.42 ? "#ffffff" : "#0b0b0b").text(fmt2(v));
    }));
    svg.append("g").attr("transform", `translate(0,${H - m.bottom})`).call(d3.axisBottom(x)).call((s) => styleAxis(s, theme));
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y)).call((s) => styleAxis(s, theme));
    axisLabel(svg, "independent runs r", (m.left + W - m.right) / 2, H - 6, theme);
    axisLabel(svg, "plants / cell / run", 16, (H - m.bottom + m.top) / 2, theme, true);
  }, [e12, organ, metric, width, theme, show, hide, wrapRef]);

  return (
    <ChartFrame
      title="Replication: runs, not plants"
      subtitle="Canonical rescue under a design of n plants per cell in r independent runs, each run with its own batch effect. Red outline: the current design."
      controls={<>
        <Toggle options={[{ value: "Shoot", label: "shoot" }, { value: "Root", label: "root" }]} value={organ} onChange={setOrgan} label="organ" />
        <Toggle options={[{ value: "reliability", label: "reliability" }, { value: "P_certify_diff_at_2.0tau", label: "P(certify 2τ diff)" }, { value: "P_certify_equal_1.0", label: "P(certify equal, 1τ)" }]} value={metric} onChange={setMetric} />
      </>}
      wrapRef={wrapRef}
    >
      <svg ref={svgRef} role="img" aria-label="design heatmap" />
      <Tooltip tipRef={tipRef} />
    </ChartFrame>
  );
}

// ---------------------------------------------------------------------- geography
export function GeoMap({ accessions, e11 }) {
  const [metric, setMetric] = useState("shoot_canonical");
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const { tipRef, show, hide } = useTooltip();
  const svgRef = useRef(null);

  useEffect(() => {
    if (!accessions || !svgRef.current) return;
    const rows = accessions.filter((a) => a.lat !== null && a.lon !== null).map((a) => ({
      a, v: metric === "shoot_loss" ? 1 - a.shoot_MD / a.shoot_MN : a[metric],
    }));
    const sorted = rows.map((r) => r.v).sort(d3.ascending);
    const q = (v) => d3.bisectLeft(sorted, v) / (sorted.length - 1);
    const W = width, H = Math.round(width * 0.42), m = { top: 12, right: 12, bottom: 36, left: 48 };
    const svg = d3.select(svgRef.current).attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const x = d3.scaleLinear().domain([-130, 145]).range([m.left, W - m.right]);
    const y = d3.scaleLinear().domain([10, 68]).range([H - m.bottom, m.top]);
    d3.range(-120, 150, 30).forEach((lon) => svg.append("line").attr("x1", x(lon)).attr("x2", x(lon)).attr("y1", m.top).attr("y2", H - m.bottom).attr("stroke", theme.grid));
    d3.range(20, 70, 10).forEach((lat) => svg.append("line").attr("x1", m.left).attr("x2", W - m.right).attr("y1", y(lat)).attr("y2", y(lat)).attr("stroke", theme.grid));
    svg.append("g").attr("transform", `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).ticks(8).tickFormat((v) => `${v}°`)).call((s) => styleAxis(s, theme));
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat((v) => `${v}°`)).call((s) => styleAxis(s, theme));
    const col = d3.scaleQuantize().domain([0, 1]).range(SEQ);
    svg.append("g").selectAll("circle").data(rows).join("circle")
      .attr("cx", (r) => x(r.a.lon)).attr("cy", (r) => y(r.a.lat)).attr("r", 4.2)
      .attr("fill", (r) => col(q(r.v))).attr("stroke", theme.bg).attr("stroke-width", 0.8)
      .on("mousemove", (ev, r) => show(`<b>${r.a.name}</b> (${r.a.country})<br/>${r.a.lat.toFixed(2)}°, ${r.a.lon.toFixed(2)}°<br/>value ${fmt2(r.v)} (quantile ${d3.format(".0%")(q(r.v))})`, ev, wrapRef.current))
      .on("mouseleave", hide);
  }, [accessions, metric, width, theme, show, hide, wrapRef]);

  const g = e11 ? e11.organs.Shoot : null;
  const stat = g ? (metric === "shoot_canonical" ? g.canonical : metric === "shoot_increase" ? g.increase : g.loss_mock_only) : null;
  return (
    <ChartFrame
      title="Where the accessions come from"
      subtitle={stat ? `Colour = quantile of the selected shoot value. Spearman with longitude ρ = ${stat.rho_lon.toFixed(2)} (p = ${stat.p_lon.toExponential(1)}), with latitude ρ = ${stat.rho_lat.toFixed(2)} (p = ${stat.p_lat.toExponential(1)}).` : ""}
      controls={<Toggle options={[{ value: "shoot_canonical", label: "canonical rescue" }, { value: "shoot_increase", label: "% increase" }, { value: "shoot_loss", label: "mock drought loss" }]} value={metric} onChange={setMetric} />}
      wrapRef={wrapRef}
    >
      <svg ref={svgRef} role="img" aria-label="collection sites" />
      <Tooltip tipRef={tipRef} />
    </ChartFrame>
  );
}
