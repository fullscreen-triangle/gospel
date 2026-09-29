// Charts about the choice of phenotype: the lambda family (recomputed live in
// the browser from the recovered cell means), top-k overlap between the
// constructions in use, the baseline-only mutant artefact, and candidate triage.

import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";

import {
  ChartFrame, LABEL, SEQ, Slider, Toggle, Tooltip, axisLabel, fLambda, gridY, jaccard,
  ranks, spearman, styleAxis, topSet, useRescueTheme, useTooltip, useWidth,
} from "./kit";

const fmt2 = d3.format(".2f");
const ORG = [{ value: "shoot", label: "shoot" }, { value: "root", label: "root" }];
const cap = (o) => o[0].toUpperCase() + o.slice(1);

function cellsOf(accessions, organ) {
  return accessions.map((a) => ({ a, W: a[`${organ}_W`], MD: a[`${organ}_MD`], MN: a[`${organ}_MN`] }));
}

// -------------------------------------------------------------- lambda explorer
export function LambdaExplorer({ accessions, e4 }) {
  const [organ, setOrgan] = useState("shoot");
  const [lam, setLam] = useState(0);
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const { tipRef, show, hide } = useTooltip();
  const leftRef = useRef(null);
  const rightRef = useRef(null);

  const cells = useMemo(() => (accessions ? cellsOf(accessions, organ) : null), [accessions, organ]);
  const curves = useMemo(() => {
    if (!cells) return null;
    const loss = cells.map((c) => 1 - c.MD / c.MN);
    const MN = cells.map((c) => c.MN);
    const grid = d3.range(-0.75, 1.5001, 0.05).map((l) => +l.toFixed(2));
    return grid.map((l) => {
      const f = cells.map((c) => fLambda(c.W, c.MD, c.MN, l));
      return { l, loss: spearman(f, loss), MN: spearman(f, MN) };
    });
  }, [cells]);

  const readout = useMemo(() => {
    if (!cells) return null;
    const f = cells.map((c) => fLambda(c.W, c.MD, c.MN, lam));
    const res = cells.map((c) => fLambda(c.W, c.MD, c.MN, 0));
    const inc = cells.map((c) => fLambda(c.W, c.MD, c.MN, 1));
    const k = Math.round(0.2 * cells.length);
    return {
      f,
      loss: spearman(f, cells.map((c) => 1 - c.MD / c.MN)),
      MN: spearman(f, cells.map((c) => c.MN)),
      vsRescue: jaccard(topSet(f, k), topSet(res, k)),
      vsIncrease: jaccard(topSet(f, k), topSet(inc, k)),
    };
  }, [cells, lam]);

  const est = e4 ? e4.organs[cap(organ)] : null;
  const half = width > 760 ? Math.floor((width - 20) / 2) : width;

  useEffect(() => {
    if (!curves || !leftRef.current || !est) return;
    const W = half, H = 280, m = { top: 16, right: 14, bottom: 42, left: 48 };
    const svg = d3.select(leftRef.current).attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const x = d3.scaleLinear().domain([-0.75, 1.5]).range([m.left, W - m.right]);
    const y = d3.scaleLinear().domain([-0.7, 1]).range([H - m.bottom, m.top]);
    gridY(svg, y, W - m.left - m.right, m.left, theme, 6);
    const [lo, hi] = est.lambda_ci95_corrected;
    svg.append("rect").attr("x", x(lo ?? -0.75)).attr("width", x(hi ?? 1.5) - x(lo ?? -0.75))
      .attr("y", m.top).attr("height", H - m.top - m.bottom).attr("fill", theme.role.canonical).attr("opacity", 0.14);
    svg.append("text").attr("x", x(est.lambda_hat_corrected)).attr("y", m.top + 12).attr("text-anchor", "middle")
      .attr("fill", theme.role.canonical).attr("font-size", 11).text(`λ̂ = ${fmt2(est.lambda_hat_corrected)}`);
    svg.append("line").attr("x1", m.left).attr("x2", W - m.right).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", theme.fgMuted);
    svg.append("g").attr("transform", `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).ticks(6)).call((s) => styleAxis(s, theme));
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(6)).call((s) => styleAxis(s, theme));
    const line = (key) => d3.line().x((d) => x(d.l)).y((d) => y(d[key]));
    svg.append("path").datum(curves).attr("d", line("loss")).attr("fill", "none").attr("stroke", theme.fg).attr("stroke-width", 2);
    svg.append("path").datum(curves).attr("d", line("MN")).attr("fill", "none").attr("stroke", theme.fgMuted)
      .attr("stroke-width", 2).attr("stroke-dasharray", "5 4");
    [[0, "rescue"], [1, "increase"]].forEach(([l, n]) => {
      const c = curves.find((d) => Math.abs(d.l - l) < 1e-9);
      svg.append("circle").attr("cx", x(l)).attr("cy", y(c.loss)).attr("r", 5).attr("fill", theme.role[n]);
      svg.append("text").attr("x", x(l)).attr("y", y(c.loss) - 9).attr("text-anchor", "middle")
        .attr("fill", theme.fg).attr("font-size", 11).text(LABEL[n]);
    });
    svg.append("line").attr("x1", x(lam)).attr("x2", x(lam)).attr("y1", m.top).attr("y2", H - m.bottom)
      .attr("stroke", theme.role.canonical).attr("stroke-width", 2);
    svg.append("text").attr("x", W - m.right).attr("y", H - m.bottom - 26).attr("text-anchor", "end").attr("fill", theme.fg).attr("font-size", 11).text("— ρ with mock drought loss");
    svg.append("text").attr("x", W - m.right).attr("y", H - m.bottom - 10).attr("text-anchor", "end").attr("fill", theme.fgMuted).attr("font-size", 11).text("- - ρ with vigour M_N");
    axisLabel(svg, "λ", (m.left + W - m.right) / 2, H - 8, theme);
    axisLabel(svg, "Spearman ρ", 12, (H - m.bottom + m.top) / 2, theme, true);
  }, [curves, lam, half, theme, est]);

  useEffect(() => {
    if (!readout || !rightRef.current) return;
    const W = half, H = 280, m = { top: 16, right: 14, bottom: 42, left: 48 };
    const svg = d3.select(rightRef.current).attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const n = cells.length;
    const loss = cells.map((c) => 1 - c.MD / c.MN);
    const rl = ranks(loss).map((r) => r / n);
    const rf = ranks(readout.f).map((r) => r / n);
    const x = d3.scaleLinear().domain([0, 1]).range([m.left, W - m.right]);
    const y = d3.scaleLinear().domain([0, 1]).range([H - m.bottom, m.top]);
    gridY(svg, y, W - m.left - m.right, m.left, theme, 5);
    svg.append("g").attr("transform", `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).ticks(5)).call((s) => styleAxis(s, theme));
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5)).call((s) => styleAxis(s, theme));
    svg.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(1)).attr("y2", y(1)).attr("stroke", theme.fgMuted).attr("stroke-dasharray", "4 3");
    svg.append("g").selectAll("circle").data(cells).join("circle")
      .attr("cx", (_, i) => x(rl[i])).attr("cy", (_, i) => y(rf[i])).attr("r", 3.2)
      .attr("fill", theme.role.canonical).attr("opacity", 0.75)
      .on("mousemove", (ev, c) => {
        const i = cells.indexOf(c);
        show(`<b>${c.a.name ?? c.a.gid}</b><br/>f_λ ${fmt2(readout.f[i])}<br/>drought loss ${d3.format(".0%")(loss[i])}<br/>rank f_λ ${d3.format(".0%")(rf[i])}`, ev, wrapRef.current);
      })
      .on("mouseleave", hide);
    axisLabel(svg, "rank by mock drought loss (quantile)", (m.left + W - m.right) / 2, H - 8, theme);
    axisLabel(svg, "rank by f_λ (quantile)", 12, (H - m.bottom + m.top) / 2, theme, true);
  }, [readout, cells, half, theme, show, hide, wrapRef]);

  return (
    <ChartFrame
      title="Choosing a phenotype is choosing λ"
      subtitle="f_λ = (W − M_D) / (M_D^λ (M_N − M_D)^(1−λ)). λ = 1 is % increase, λ = 0 is rescue. Everything here is recomputed in your browser from the 235 recovered cell-mean triples."
      controls={<>
        <Toggle options={ORG} value={organ} onChange={setOrgan} label="organ" />
        <Slider label="λ" min={-0.75} max={1.5} step={0.05} value={lam} onChange={setLam} format={(v) => v.toFixed(2)} />
      </>}
      wrapRef={wrapRef}
    >
      <div className={`grid gap-4 ${width > 760 ? "grid-cols-2" : "grid-cols-1"}`}>
        <svg ref={leftRef} role="img" aria-label="nuisance loading along lambda" />
        <svg ref={rightRef} role="img" aria-label="rank of f_lambda against drought loss" />
      </div>
      {readout ? (
        <div className="mt-3 grid grid-cols-4 gap-3 font-mono text-xs md:grid-cols-2">
          <div><span className="text-dark/60 dark:text-light/60">ρ vs drought loss</span><br /><b className="text-base">{fmt2(readout.loss)}</b></div>
          <div><span className="text-dark/60 dark:text-light/60">ρ vs vigour M_N</span><br /><b className="text-base">{fmt2(readout.MN)}</b></div>
          <div><span className="text-dark/60 dark:text-light/60">top-20% Jaccard vs rescue</span><br /><b className="text-base">{fmt2(readout.vsRescue)}</b></div>
          <div><span className="text-dark/60 dark:text-light/60">top-20% Jaccard vs % increase</span><br /><b className="text-base">{fmt2(readout.vsIncrease)}</b></div>
        </div>
      ) : null}
      <Tooltip tipRef={tipRef} />
    </ChartFrame>
  );
}

// ------------------------------------------------------------------ top-k overlap
export function TopKExplorer({ accessions, e3 }) {
  const [organ, setOrgan] = useState("shoot");
  const [kPct, setKPct] = useState(20);
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const { tipRef, show, hide } = useTooltip();
  const barRef = useRef(null);
  const stripRef = useRef(null);

  const data = useMemo(() => {
    if (!accessions) return null;
    const vals = {
      gain: accessions.map((a) => a[`${organ}_gain`]),
      increase: accessions.map((a) => a[`${organ}_increase`]),
      rescue: accessions.map((a) => a[`${organ}_rescue`]),
    };
    const k = Math.max(1, Math.round((kPct / 100) * accessions.length));
    const tops = Object.fromEntries(Object.entries(vals).map(([n, v]) => [n, topSet(v, k)]));
    const pairs = [["gain", "increase"], ["gain", "rescue"], ["increase", "rescue"]].map(([a, b]) => {
      let inter = 0;
      tops[a].forEach((i) => { if (tops[b].has(i)) inter += 1; });
      return { a, b, flip: 1 - inter / k, jac: jaccard(tops[a], tops[b]) };
    });
    let all = 0;
    tops.gain.forEach((i) => { if (tops.increase.has(i) && tops.rescue.has(i)) all += 1; });
    return { vals, tops, pairs, k, all };
  }, [accessions, organ, kPct]);

  useEffect(() => {
    if (!data || !barRef.current || !e3) return;
    const W = width, H = 220, m = { top: 18, right: 16, bottom: 36, left: 52 };
    const svg = d3.select(barRef.current).attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const noise = e3.organs[cap(organ)].noise_agreement;
    const x = d3.scaleBand().domain(data.pairs.map((p) => `${p.a}~${p.b}`)).range([m.left, W - m.right]).padding(0.45);
    const y = d3.scaleLinear().domain([0, 1]).range([H - m.bottom, m.top]);
    gridY(svg, y, W - m.left - m.right, m.left, theme, 5);
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%"))).call((s) => styleAxis(s, theme));
    svg.append("g").selectAll("rect").data(data.pairs).join("rect")
      .attr("x", (p) => x(`${p.a}~${p.b}`)).attr("width", x.bandwidth())
      .attr("y", (p) => y(p.flip)).attr("height", (p) => y(0) - y(p.flip)).attr("rx", 3).attr("fill", theme.fg)
      .on("mousemove", (ev, p) => show(`<b>${LABEL[p.a]} vs ${LABEL[p.b]}</b><br/>top-${kPct}% membership changed: ${d3.format(".0%")(p.flip)}<br/>Jaccard ${fmt2(p.jac)}`, ev, wrapRef.current))
      .on("mouseleave", hide);
    data.pairs.forEach((p) => {
      const xx = x(`${p.a}~${p.b}`);
      [["M2_plant_all_cells", theme.role.gain], ["M3_plant_plus_batch", theme.role.alert]].forEach(([mk, col]) => {
        const c = Math.max(noise[mk][p.a].flip_rate_top20_p95, noise[mk][p.b].flip_rate_top20_p95);
        svg.append("line").attr("x1", xx - 6).attr("x2", xx + x.bandwidth() + 6).attr("y1", y(c)).attr("y2", y(c))
          .attr("stroke", col).attr("stroke-width", 3).attr("opacity", kPct === 20 ? 1 : 0.35);
      });
      svg.append("text").attr("x", xx + x.bandwidth() / 2).attr("y", H - m.bottom + 16).attr("text-anchor", "middle")
        .attr("fill", theme.fg).attr("font-size", 11).text(`${LABEL[p.a]} vs ${LABEL[p.b]}`);
      svg.append("text").attr("x", xx + x.bandwidth() / 2).attr("y", y(p.flip) - 5).attr("text-anchor", "middle")
        .attr("fill", theme.fg).attr("font-size", 11).attr("font-family", "ui-monospace, monospace").text(d3.format(".0%")(p.flip));
    });
    axisLabel(svg, "membership changed", 12, (H - m.bottom + m.top) / 2, theme, true);
  }, [data, width, theme, e3, organ, kPct, show, hide, wrapRef]);

  useEffect(() => {
    if (!data || !stripRef.current) return;
    const W = width, H = 96, m = { top: 8, right: 16, bottom: 8, left: 90 };
    const svg = d3.select(stripRef.current).attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const order = d3.range(accessions.length).sort((i, j) => data.vals.rescue[j] - data.vals.rescue[i]);
    const x = d3.scaleBand().domain(order).range([m.left, W - m.right]);
    const rows = ["rescue", "increase", "gain"];
    const y = d3.scaleBand().domain(rows).range([m.top, H - m.bottom]).padding(0.25);
    rows.forEach((r) => {
      svg.append("text").attr("x", m.left - 8).attr("y", y(r) + y.bandwidth() / 2 + 4).attr("text-anchor", "end")
        .attr("fill", theme.fg).attr("font-size", 11).text(LABEL[r]);
      svg.append("g").selectAll("rect").data(order.filter((i) => data.tops[r].has(i))).join("rect")
        .attr("x", (i) => x(i)).attr("width", Math.max(1.2, x.bandwidth())).attr("y", y(r)).attr("height", y.bandwidth())
        .attr("fill", theme.role[r])
        .on("mousemove", (ev, i) => show(`<b>${accessions[i].name ?? accessions[i].gid}</b><br/>in the top ${kPct}% by ${LABEL[r]}`, ev, wrapRef.current))
        .on("mouseleave", hide);
    });
  }, [data, width, theme, accessions, kPct, show, hide, wrapRef]);

  return (
    <ChartFrame
      title="Do the constructions agree on who rescues best?"
      subtitle={data ? `Top ${kPct}% = ${data.k} accessions; ${data.all} are in the top set under all three constructions. Coloured ticks: 95th-percentile noise ceiling at 20% (blue: plant noise, red: + batch); faded at other k.` : ""}
      controls={<>
        <Toggle options={ORG} value={organ} onChange={setOrgan} label="organ" />
        <Slider label="top k" min={5} max={50} step={1} value={kPct} onChange={setKPct} format={(v) => `${v}%`} />
      </>}
      wrapRef={wrapRef}
    >
      <svg ref={barRef} role="img" aria-label="top-k disagreement between constructions" />
      <div className="mt-1 text-xs font-medium text-dark/70 dark:text-light/70">Top-set membership, accessions ordered by rescue (left = highest):</div>
      <svg ref={stripRef} role="img" aria-label="top-set membership strips" />
      <Tooltip tipRef={tipRef} />
    </ChartFrame>
  );
}

// ------------------------------------------------------------- ratio artefact
export function RatioArtefact({ e9 }) {
  const [eps, setEps] = useState(0.3);
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const svgRef = useRef(null);
  const row = e9 ? e9.rows.reduce((best, r) => (Math.abs(r.eps - eps) < Math.abs(best.eps - eps) ? r : best), e9.rows[0]) : null;

  useEffect(() => {
    if (!e9 || !svgRef.current) return;
    const W = width, H = 280, m = { top: 16, right: 110, bottom: 42, left: 52 };
    const svg = d3.select(svgRef.current).attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const x = d3.scaleLinear().domain([0, 0.5]).range([m.left, W - m.right]);
    const y = d3.scaleLinear().domain([0.6, 2.05]).range([H - m.bottom, m.top]);
    gridY(svg, y, W - m.left - m.right, m.left, theme, 6);
    svg.append("g").attr("transform", `translate(0,${H - m.bottom})`).call(d3.axisBottom(x).ticks(5)).call((s) => styleAxis(s, theme));
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(6).tickFormat((v) => `${v}×`)).call((s) => styleAxis(s, theme));
    svg.append("line").attr("x1", m.left).attr("x2", W - m.right).attr("y1", y(1)).attr("y2", y(1)).attr("stroke", theme.fgMuted);
    ["increase", "gain", "rescue", "canonical"].forEach((n) => {
      const pts = e9.rows.map((r) => ({ e: r.eps, v: r[n].median, lo: r[n].q25, hi: r[n].q75 }));
      svg.append("path").datum(pts).attr("d", d3.area().x((d) => x(d.e)).y0((d) => y(d.lo)).y1((d) => y(d.hi)))
        .attr("fill", theme.role[n]).attr("opacity", 0.14);
      svg.append("path").datum(pts).attr("d", d3.line().x((d) => x(d.e)).y((d) => y(d.v)))
        .attr("fill", "none").attr("stroke", theme.role[n]).attr("stroke-width", 2.2);
      const last = pts[pts.length - 1];
      svg.append("text").attr("x", x(last.e) + 6).attr("y", y(last.v) + 4).attr("fill", theme.fg).attr("font-size", 11)
        .text(`${LABEL[n]} ${last.v.toFixed(2)}×`);
    });
    svg.append("line").attr("x1", x(eps)).attr("x2", x(eps)).attr("y1", m.top).attr("y2", H - m.bottom)
      .attr("stroke", theme.fg).attr("stroke-dasharray", "3 3");
    axisLabel(svg, "ε: mutant's mock-drought biomass reduced by this fraction; absolute gain unchanged", (m.left + W - m.right) / 2, H - 8, theme);
    axisLabel(svg, "apparent rescue, mutant / wild type", 12, (H - m.bottom + m.top) / 2, theme, true);
  }, [e9, eps, width, theme]);

  return (
    <ChartFrame
      title="A weaker baseline masquerading as stronger rescue"
      subtitle={row ? `At ε = ${row.eps.toFixed(2)}: % increase reports ${row.increase.median.toFixed(2)}×, gain ${row.gain.median.toFixed(2)}×, rescue ${row.rescue.median.toFixed(2)}×, canonical ${row.canonical.median.toFixed(2)}× (median over accessions, shoot).` : ""}
      controls={<Slider label="ε" min={0} max={0.5} step={0.05} value={eps} onChange={setEps} format={(v) => v.toFixed(2)} />}
      wrapRef={wrapRef}
    >
      <svg ref={svgRef} role="img" aria-label="ratio artefact" />
    </ChartFrame>
  );
}

// ---------------------------------------------------------------- candidates
const FAMILIES = [
  ["rescue", "rescue"], ["increase", "% increase"], ["gain", "gain"],
  ["inoculated_size", "size under WCS417"], ["mock_only_loss", "mock-only loss"],
];

export function CandidateChart({ e8 }) {
  const [wrapRef, width] = useWidth();
  const theme = useRescueTheme();
  const { tipRef, show, hide } = useTooltip();
  const svgRef = useRef(null);

  useEffect(() => {
    if (!e8 || !svgRef.current) return;
    const byFam = FAMILIES.map(([f, label]) => {
      const rows = e8.rows.filter((r) => r.family === f);
      const genes = Array.from(new Map(rows.map((r) => [r.gene, r])).values());
      const rho = rows.map((r) => r.rho_metric_vs_mock_loss).filter((v) => v !== null && v !== undefined);
      return { f, label, genes, n: genes.length, load: rho.length ? d3.median(rho.map(Math.abs)) : null };
    });
    const W = width, H = 260, m = { top: 20, right: 16, bottom: 58, left: 44 };
    const svg = d3.select(svgRef.current).attr("viewBox", `0 0 ${W} ${H}`).attr("width", W).attr("height", H);
    svg.selectAll("*").remove();
    const x = d3.scaleBand().domain(byFam.map((d) => d.f)).range([m.left, W - m.right]).padding(0.35);
    const y = d3.scaleLinear().domain([0, 14]).range([H - m.bottom, m.top]);
    const col = d3.scaleQuantize().domain([0, 1]).range(SEQ);
    gridY(svg, y, W - m.left - m.right, m.left, theme, 7);
    svg.append("g").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(7)).call((s) => styleAxis(s, theme));
    svg.append("g").selectAll("rect").data(byFam).join("rect")
      .attr("x", (d) => x(d.f)).attr("width", x.bandwidth()).attr("y", (d) => y(d.n)).attr("height", (d) => y(0) - y(d.n))
      .attr("rx", 3).attr("fill", (d) => (d.load === null ? theme.grid : col(d.load)))
      .attr("stroke", theme.fgMuted).attr("stroke-width", 0.5)
      .on("mousemove", (ev, d) => show(
        `<b>${d.label}</b> — ${d.n} genes<br/>median |ρ| with mock loss: ${d.load === null ? "n/a" : fmt2(d.load)}<br/>${d.genes.map((g) => `${g.gene}${g.symbol ? ` ${g.symbol}` : ""}`).join("<br/>")}`,
        ev, wrapRef.current))
      .on("mouseleave", hide);
    byFam.forEach((d) => {
      svg.append("text").attr("x", x(d.f) + x.bandwidth() / 2).attr("y", y(d.n) - 6).attr("text-anchor", "middle")
        .attr("fill", theme.fg).attr("font-size", 12).attr("font-weight", 600).text(d.n);
      svg.append("text").attr("x", x(d.f) + x.bandwidth() / 2).attr("y", H - m.bottom + 16).attr("text-anchor", "middle")
        .attr("fill", theme.fg).attr("font-size", 11).text(d.label);
      svg.append("text").attr("x", x(d.f) + x.bandwidth() / 2).attr("y", H - m.bottom + 32).attr("text-anchor", "middle")
        .attr("fill", theme.fgMuted).attr("font-size", 10).attr("font-family", "ui-monospace, monospace")
        .text(d.load === null ? "" : `|ρ| ${fmt2(d.load)}`);
    });
    axisLabel(svg, "candidate genes", 12, (H - m.bottom + m.top) / 2, theme, true);
  }, [e8, width, theme, show, hide, wrapRef]);

  return (
    <ChartFrame
      title="Which phenotype each candidate is a locus for"
      subtitle={e8 ? `${e8.n_genes} genes, ${e8.n_rows} rows; ${e8.genes_multi_family} genes recur across trait families. Shade = median |ρ| between the family's traits and mock-only drought loss. Hover for gene lists.` : ""}
      wrapRef={wrapRef}
    >
      <svg ref={svgRef} role="img" aria-label="candidate genes by trait family" />
      <Tooltip tipRef={tipRef} />
    </ChartFrame>
  );
}
