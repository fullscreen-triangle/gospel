import { useEffect, useRef } from "react";
import * as d3 from "d3";
import {
  astTree,
  residueLedger,
  frameMap,
  dataflow,
  parameterSurface,
  alignmentArity,
} from "@/lib/synopsis-shape";

/**
 * The D3 views in the third column.
 *
 * Everything drawn here is derived from the syntax tree the parser just
 * produced, and from nothing else. That constraint is worth stating
 * plainly because the obvious alternative was to port the matplotlib
 * panels: those are computed by the Python validation modules at plot
 * time, so reproducing them here would mean either a third
 * implementation of the framework with no oracle holding it to the
 * other two, or reading cached numbers and presenting them as though
 * this program had produced them. The evaluator is not implemented, so
 * this editor has no numeric results to draw -- and drawing someone
 * else's numbers next to a real diagnostic would be the same dishonesty
 * the refusal notice exists to avoid.
 *
 * What the front-end DOES know is what the parser derived, and that is
 * more than a shape. Three of these views are structural: the tree, the
 * frames its values live in, and what it did with the part of each
 * comparison the method could not explain. Three are quantitative, and
 * their axes are quantities the parser actually produces -- the number
 * of derivations between a file and a value, the magnitude of every
 * parameter the program states, the count of columns an alignment
 * carries. Those move when the source moves, which is the property
 * that separates a chart from a decoration, and none of them requires
 * an evaluator that does not exist.
 */

const INK = "#d4d4d4";
const MUTED = "#8a8a8a";
const LINE = "#3a3a3a";

// Node fills by syntactic category. Keyed to the editor's own token
// colours so a `bind` is the same blue in both panes.
const FILL = {
  Program: "#569cd6",
  Open: "#4ec9b0",
  MethodDecl: "#4ec9b0",
  FrameBlock: "#c586c0",
  Let: "#9cdcfe",
  Bind: "#569cd6",
  Claim: "#dcdcaa",
  Record: "#1a7f37",
  Drop: "#d1242f",
  Relax: "#bf8700",
  For: "#c586c0",
  Sweep: "#c586c0",
};
const fillOf = (k) => FILL[k] || MUTED;

/* ------------------------------------------------------------------ */
/* Syntax tree                                                         */

function TreeView({ ast, onJump }) {
  const ref = useRef(null);

  useEffect(() => {
    const host = ref.current;
    if (!host) return;
    d3.select(host).selectAll("*").remove();

    const data = astTree(ast);
    if (!data) return;

    const root = d3.hierarchy(data);

    // A horizontal dendrogram, not a top-down tree: labels are words
    // like "bind r, res_r" and reading them sideways in a narrow column
    // is the difference between a diagram and a decoration.
    const dx = 20;
    const dy = 150;
    d3.tree().nodeSize([dx, dy])(root);

    let x0 = Infinity;
    let x1 = -Infinity;
    root.each((d) => {
      if (d.x > x1) x1 = d.x;
      if (d.x < x0) x0 = d.x;
    });

    const depth = 1 + (d3.max(root.descendants(), (d) => d.depth) ?? 0);
    const width = depth * dy + 220;
    const height = x1 - x0 + dx * 2;

    const svg = d3
      .select(host)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", [-40, x0 - dx, width, height])
      .attr("font-family", "ui-monospace, monospace")
      .attr("font-size", 10.5);

    svg
      .append("g")
      .attr("fill", "none")
      .attr("stroke", LINE)
      .attr("stroke-width", 1)
      .selectAll("path")
      .data(root.links())
      .join("path")
      .attr(
        "d",
        d3
          .linkHorizontal()
          .x((d) => d.y)
          .y((d) => d.x)
      );

    const g = svg
      .append("g")
      .selectAll("g")
      .data(root.descendants())
      .join("g")
      .attr("transform", (d) => `translate(${d.y},${d.x})`)
      .attr("cursor", (d) => (d.data.line ? "pointer" : "default"))
      .on("click", (_, d) => {
        if (d.data.line && onJump) onJump(d.data.line);
      });

    g.append("circle")
      .attr("r", 3.5)
      .attr("fill", (d) => fillOf(d.data.kind))
      .append("title")
      .text((d) => `${d.data.kind}${d.data.line ? ` — line ${d.data.line}` : ""}`);

    // The role a child plays in its parent ("left", "in", "response"),
    // set above the edge. Without it a two-child comparison is just two
    // circles and the reader has to guess which side is which.
    g.filter((d) => d.data.role)
      .append("text")
      .attr("dy", -5)
      .attr("x", -6)
      .attr("text-anchor", "end")
      .attr("fill", MUTED)
      .attr("font-size", 8.5)
      .text((d) => d.data.role);

    const label = g.append("text").attr("dy", 3).attr("x", 8).attr("fill", INK);
    label.text((d) => d.data.label);

    // Parameters, appended after the label in a dimmer colour. These
    // are the values that the raw-JSON view lost entirely: `Params` is
    // a Map and JSON.stringify renders a Map as `{}`. Showing them is
    // not decoration -- "no parameter is supplied silently" is the
    // language's own claim, and a view that hides them contradicts it.
    g.filter((d) => d.data.params)
      .append("text")
      .attr("dy", 12)
      .attr("x", 8)
      .attr("fill", "#bf8700")
      .attr("font-size", 9)
      .text((d) => d.data.params);
  }, [ast, onJump]);

  return <div ref={ref} className="overflow-auto p-2" />;
}

/* ------------------------------------------------------------------ */
/* Frames                                                              */

function FrameView({ ast, onJump }) {
  const ref = useRef(null);

  useEffect(() => {
    const host = ref.current;
    if (!host) return;
    d3.select(host).selectAll("*").remove();

    const fm = frameMap(ast);
    if (!fm.frames.length) return;

    const W = 300;
    const PAD = 10;
    const ROW = 15;
    const HEAD = 26;

    // Height is computed from content rather than fixed: a frame with
    // eleven bindings and one with two should not be drawn the same
    // size, because the point of the diagram is that frames are
    // separate containers with distinct contents.
    const boxes = fm.frames.map((f) => ({
      ...f,
      h: HEAD + Math.max(1, f.bound.length + f.foreign.length) * ROW + PAD,
    }));

    let y = PAD;
    for (const b of boxes) {
      b.y = y;
      y += b.h + 14;
    }

    const svg = d3
      .select(host)
      .append("svg")
      .attr("width", W)
      .attr("height", y)
      .attr("font-family", "ui-monospace, monospace")
      .attr("font-size", 10);

    const g = svg
      .selectAll("g")
      .data(boxes)
      .join("g")
      .attr("transform", (d) => `translate(${PAD},${d.y})`);

    g.append("rect")
      .attr("width", W - PAD * 2)
      .attr("height", (d) => d.h)
      .attr("rx", 4)
      .attr("fill", "#252526")
      .attr("stroke", (d) => (d.foreign.length ? "#f48771" : LINE));

    g.append("text")
      .attr("x", 8)
      .attr("y", 15)
      .attr("fill", "#c586c0")
      .attr("cursor", "pointer")
      .text((d) => `under ${d.name}`)
      .on("click", (_, d) => onJump && onJump(d.line));

    // The frame index, shown because it is part of the type. Two values
    // projected in different frames are of different types with no
    // coercion between them, so the index is not bookkeeping -- it is
    // the thing that makes a cross-frame reference a type error rather
    // than a scope slip.
    g.append("text")
      .attr("x", W - PAD * 2 - 8)
      .attr("y", 15)
      .attr("text-anchor", "end")
      .attr("fill", MUTED)
      .attr("font-size", 8.5)
      .text((d) => `frame ${d.index}`);

    g.each(function (d) {
      const sel = d3.select(this);
      d.bound.forEach((b, i) => {
        const row = sel
          .append("g")
          .attr("transform", `translate(8,${HEAD + i * ROW})`)
          .attr("cursor", "pointer")
          .on("click", () => onJump && onJump(b.line));
        row
          .append("rect")
          .attr("x", -2)
          .attr("y", -8)
          .attr("width", 3)
          .attr("height", 10)
          .attr("fill", b.kind === "residue" ? "#bf8700" : fillOf("Let"));
        row.append("text").attr("x", 8).attr("fill", INK).text(b.name);
        row
          .append("text")
          .attr("x", W - PAD * 2 - 16)
          .attr("text-anchor", "end")
          .attr("fill", MUTED)
          .attr("font-size", 8.5)
          .text(b.kind);
      });

      d.foreign.forEach((n, i) => {
        sel
          .append("text")
          .attr("x", 8)
          .attr("y", HEAD + (d.bound.length + i) * ROW)
          .attr("fill", "#f48771")
          .text(`${n}  — not bound in this frame`);
      });
    });
  }, [ast, onJump]);

  return <div ref={ref} className="overflow-auto" />;
}

/* ------------------------------------------------------------------ */
/* Residue                                                             */

function ResidueView({ ast, onJump }) {
  const ref = useRef(null);

  useEffect(() => {
    const host = ref.current;
    if (!host) return;
    d3.select(host).selectAll("*").remove();

    const rows = residueLedger(ast);
    if (!rows.length) return;

    const W = 300;
    const ROW = 30;
    const PAD = 10;

    const svg = d3
      .select(host)
      .append("svg")
      .attr("width", W)
      .attr("height", rows.length * ROW + PAD * 2)
      .attr("font-family", "ui-monospace, monospace")
      .attr("font-size", 10);

    const g = svg
      .selectAll("g")
      .data(rows)
      .join("g")
      .attr("transform", (_, i) => `translate(${PAD},${PAD + i * ROW})`)
      .attr("cursor", "pointer")
      .on("click", (_, d) => onJump && onJump(d.line));

    // Two bars per comparison, drawn adjacent: the value the method
    // explained, and the residue it did not. They are the same width
    // because the ledger knows nothing about their magnitudes -- only
    // the evaluator would, and it does not exist. Drawing them to
    // scale would be inventing a number.
    g.append("rect")
      .attr("width", 90)
      .attr("height", 12)
      .attr("rx", 2)
      .attr("fill", fillOf("Bind"))
      .attr("opacity", 0.85);

    g.append("rect")
      .attr("x", 92)
      .attr("width", 90)
      .attr("height", 12)
      .attr("rx", 2)
      .attr("fill", (d) => (d.consumed ? "#1a7f37" : "#d1242f"))
      .attr("opacity", 0.85);

    g.append("text")
      .attr("x", 4)
      .attr("y", 9)
      .attr("fill", "#1e1e1e")
      .attr("font-size", 8.5)
      .text((d) => d.value);

    g.append("text")
      .attr("x", 96)
      .attr("y", 9)
      .attr("fill", "#1e1e1e")
      .attr("font-size", 8.5)
      .text((d) => d.name);

    g.append("text")
      .attr("x", 190)
      .attr("y", 9)
      .attr("fill", (d) => (d.consumed ? MUTED : "#f48771"))
      .attr("font-size", 8.5)
      .text((d) => (d.consumed ? "consumed" : "NOT consumed"));

    g.append("text")
      .attr("y", 23)
      .attr("fill", MUTED)
      .attr("font-size", 8)
      .text((d) => `line ${d.line} · under ${d.frame}`);
  }, [ast, onJump]);

  return <div ref={ref} className="overflow-auto" />;
}

/* ------------------------------------------------------------------ */
/* Dataflow                                                            */

/**
 * Names as nodes, uses as edges, laid out in columns by depth.
 *
 * A deliberate departure from the tree: the tree nests by grammar, so
 * two statements that depend on each other are drawn as siblings and
 * the dependency between them is invisible. Here the x axis IS the
 * dependency -- column d holds the values that take d derivations to
 * reach from a file on disk. A force layout would have looked livelier
 * and said less; the column is the measurement.
 */
function DataflowView({ ast, onJump }) {
  const ref = useRef(null);

  useEffect(() => {
    const host = ref.current;
    if (!host) return;
    d3.select(host).selectAll("*").remove();

    const { nodes, links } = dataflow(ast);
    if (!nodes.length) return;

    const COL = 118;
    const ROW = 26;
    const PADX = 54;
    const PADY = 22;

    const maxDepth = d3.max(nodes, (n) => n.depth) ?? 0;
    const byDepth = d3.groups(nodes, (n) => n.depth).sort((a, b) => a[0] - b[0]);

    const pos = new Map();
    for (const [d, group] of byDepth)
      group.forEach((n, i) => pos.set(n, { x: PADX + d * COL, y: PADY + i * ROW }));

    const tallest = d3.max(byDepth, ([, g]) => g.length) ?? 1;
    const width = PADX * 2 + maxDepth * COL + 40;
    const height = PADY * 2 + tallest * ROW;

    const svg = d3
      .select(host)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("font-family", "ui-monospace, monospace")
      .attr("font-size", 10);

    // Column rules and depth labels. Without them a reader sees a
    // scatter of names; with them the axis is legible as an axis.
    const axis = svg.append("g");
    for (let d = 0; d <= maxDepth; d++) {
      axis
        .append("line")
        .attr("x1", PADX + d * COL - 10)
        .attr("x2", PADX + d * COL - 10)
        .attr("y1", 4)
        .attr("y2", height - 12)
        .attr("stroke", LINE)
        .attr("stroke-dasharray", "2 3")
        .attr("opacity", 0.5);
      axis
        .append("text")
        .attr("x", PADX + d * COL - 10)
        .attr("y", height - 3)
        .attr("text-anchor", "middle")
        .attr("fill", MUTED)
        .attr("font-size", 8)
        .text(d === 0 ? "input" : "d" + d);
    }

    // Latest node per name, so an edge lands on the binding that was
    // actually in scope where it was used, not on a shadowed one.
    const latest = new Map();
    for (const n of nodes) latest.set(n.name, n);

    svg
      .append("g")
      .attr("fill", "none")
      .selectAll("path")
      .data(links)
      .join("path")
      .attr("stroke", (l) => (l.resolved ? LINE : "#f48771"))
      .attr("stroke-width", (l) => (l.resolved ? 1 : 1.6))
      .attr("stroke-dasharray", (l) => (l.resolved ? null : "3 2"))
      .attr("d", (l) => {
        const t = pos.get(latest.get(l.target));
        if (!t) return null;
        const sN = latest.get(l.source);
        const sP = sN && pos.get(sN);
        // An unresolved source has no node to start from. It is drawn
        // as a stub reaching in from the left of the target rather than
        // omitted, because the absent end is the whole diagnostic.
        const from = l.resolved && sP ? sP : { x: t.x - COL * 0.55, y: t.y };
        const mx = (from.x + t.x) / 2;
        return "M" + from.x + "," + from.y + "C" + mx + "," + from.y + " " + mx + "," + t.y + " " + t.x + "," + t.y;
      });

    const g = svg
      .append("g")
      .selectAll("g")
      .data(nodes)
      .join("g")
      .attr("transform", (n) => {
        const p = pos.get(n);
        return "translate(" + p.x + "," + p.y + ")";
      })
      .attr("cursor", "pointer")
      .on("click", (_, n) => n.line && onJump && onJump(n.line));

    g.append("circle")
      .attr("r", 3.5)
      .attr("fill", (n) =>
        n.kind === "residue"
          ? "#bf8700"
          : n.kind === "open" || n.kind === "method"
          ? "#4ec9b0"
          : fillOf("Let")
      )
      .append("title")
      .text(
        (n) =>
          n.name +
          " — " +
          n.kind +
          (n.frame ? ", under " + n.frame : "") +
          (n.line ? ", line " + n.line : "")
      );

    g.append("text").attr("x", 7).attr("dy", 3).attr("fill", INK).text((n) => n.name);

    // Names the program uses that this frame never bound. They are
    // labelled at the point of use, since that is the line a reader
    // has to change.
    svg
      .append("g")
      .selectAll("text")
      .data(links.filter((l) => !l.resolved))
      .join("text")
      .attr("x", (l) => {
        const t = pos.get(latest.get(l.target));
        return t ? t.x - COL * 0.55 : 0;
      })
      .attr("y", (l) => {
        const t = pos.get(latest.get(l.target));
        return t ? t.y - 6 : 0;
      })
      .attr("text-anchor", "middle")
      .attr("fill", "#f48771")
      .attr("font-size", 8.5)
      .text((l) => l.source + "?");
  }, [ast, onJump]);

  return <div ref={ref} className="overflow-auto p-2" />;
}

/* ------------------------------------------------------------------ */
/* Parameters                                                          */

/**
 * Every parameter the program states, grouped by whatever owns it.
 *
 * The language has no defaults anywhere, and this is the view where
 * that stops being a slogan: it is the complete set of numbers the
 * result rests on, in one place, which is what a reader would need in
 * order to reproduce it.
 *
 * The bars are scaled per OWNER, never globally. A global scale would
 * put theta = 0.01 and min_distance = 30 on one axis and render theta
 * as nothing, implying the small number matters less. Nothing in the
 * language says that -- these are unrelated quantities in unrelated
 * units, and comparing them across owners would be an artefact of the
 * drawing rather than a property of the program.
 */
function ParameterView({ ast, onJump }) {
  const ref = useRef(null);

  useEffect(() => {
    const host = ref.current;
    if (!host) return;
    d3.select(host).selectAll("*").remove();

    const rows = parameterSurface(ast);
    if (!rows.length) return;

    const groups = d3.groups(rows, (r) => r.owner + "#" + r.line);

    const W = 300;
    const PAD = 10;
    const ROW = 16;
    const HEAD = 18;
    const BARX = 132;
    const BARW = 112;

    let y = PAD;
    const laid = groups.map(([, rs]) => {
      const o = { rows: rs, y, h: HEAD + rs.length * ROW };
      y += o.h + 10;
      return o;
    });

    const svg = d3
      .select(host)
      .append("svg")
      .attr("width", W)
      .attr("height", y)
      .attr("font-family", "ui-monospace, monospace")
      .attr("font-size", 10);

    for (const grp of laid) {
      const head = grp.rows[0];
      const g = svg.append("g").attr("transform", "translate(" + PAD + "," + grp.y + ")");

      g.append("text")
        .attr("fill", "#dcdcaa")
        .attr("cursor", "pointer")
        .text(head.owner)
        .on("click", () => onJump && onJump(head.line));

      g.append("text")
        .attr("x", W - PAD * 2)
        .attr("text-anchor", "end")
        .attr("fill", MUTED)
        .attr("font-size", 8.5)
        .text(head.ownerKind + " · line " + head.line);

      const nums = grp.rows.filter((r) => r.numeric).map((r) => Math.abs(r.value));
      const scale = d3
        .scaleLinear()
        .domain([0, d3.max(nums) || 1])
        .range([0, BARW]);

      grp.rows.forEach((r, i) => {
        const row = g
          .append("g")
          .attr("transform", "translate(0," + (HEAD + i * ROW) + ")")
          .attr("cursor", "pointer")
          .on("click", () => onJump && onJump(r.line));

        row.append("text").attr("fill", INK).attr("font-size", 9.5).text(r.key);

        if (r.numeric) {
          row
            .append("rect")
            .attr("x", BARX)
            .attr("y", -7)
            .attr("width", Math.max(1, scale(Math.abs(r.value))))
            .attr("height", 8)
            .attr("rx", 1.5)
            .attr("fill", r.value < 0 ? "#d1242f" : "#569cd6")
            .attr("opacity", 0.8);
          row
            .append("text")
            .attr("x", BARX - 6)
            .attr("text-anchor", "end")
            .attr("fill", "#b5cea8")
            .attr("font-size", 9)
            .text(r.value);
        } else {
          // A symbolic argument -- `normalised`, `dna`, `species_name`.
          // It is a stated choice like any other and belongs in the
          // set, but it has no magnitude, so it gets no bar.
          row
            .append("text")
            .attr("x", BARX - 6)
            .attr("text-anchor", "end")
            .attr("fill", "#ce9178")
            .attr("font-size", 9)
            .text(String(r.value));
        }
      });
    }
  }, [ast, onJump]);

  return <div ref={ref} className="overflow-auto" />;
}

/* ------------------------------------------------------------------ */
/* Alignment arity                                                     */

/**
 * The four columns of each alignment, drawn as four columns.
 *
 * An alignment carries a central pair, a response pair, and the
 * correspondences relating them, and the rule is that all of it is
 * required: agreement on structure alone, or on behaviour alone, is
 * not sufficient evidence to carry a conclusion across. An alignment
 * written without its response clause is the mistake the arity rule
 * exists to refuse. Drawing the slots as a fixed grid makes a missing
 * one read as a hole in the row -- which is what the checker is about
 * to say in words.
 */
function ArityView({ ast, onJump }) {
  const ref = useRef(null);

  useEffect(() => {
    const host = ref.current;
    if (!host) return;
    d3.select(host).selectAll("*").remove();

    const rows = alignmentArity(ast);
    if (!rows.length) return;

    const W = 300;
    const PAD = 12;
    const BOX = 20;
    const GAP = 14;
    const BLOCK = 70;

    const svg = d3
      .select(host)
      .append("svg")
      .attr("width", W)
      .attr("height", PAD + rows.length * BLOCK)
      .attr("font-family", "ui-monospace, monospace")
      .attr("font-size", 10);

    const COLS = [
      { key: "central", n: 2, label: "central", fill: "#569cd6" },
      { key: "resp", n: 2, label: "response", fill: "#c586c0" },
    ];

    rows.forEach((r, i) => {
      const g = svg
        .append("g")
        .attr("transform", "translate(" + PAD + "," + (PAD + i * BLOCK) + ")");

      g.append("text")
        .attr("fill", "#dcdcaa")
        .attr("cursor", "pointer")
        .text("align " + r.name)
        .on("click", () => onJump && onJump(r.line));

      g.append("text")
        .attr("x", W - PAD * 2)
        .attr("text-anchor", "end")
        .attr("fill", MUTED)
        .attr("font-size", 8.5)
        .text(
          r.theta === null
            ? "line " + r.line
            : "θ = " + r.theta + " · line " + r.line
        );

      let x = 0;
      for (const col of COLS) {
        const x0 = x;
        for (let k = 0; k < col.n; k++) {
          const filled = r[col.key] > k;
          g.append("rect")
            .attr("x", x)
            .attr("y", 8)
            .attr("width", BOX)
            .attr("height", BOX)
            .attr("rx", 3)
            .attr("fill", filled ? col.fill : "none")
            .attr("opacity", filled ? 0.85 : 1)
            .attr("stroke", filled ? "none" : "#f48771")
            .attr("stroke-dasharray", filled ? null : "3 2");
          x += BOX + 2;
        }
        g.append("text")
          .attr("x", (x0 + x - 2) / 2)
          .attr("y", 8 + BOX + 11)
          .attr("text-anchor", "middle")
          .attr("fill", r[col.key] === col.n ? MUTED : "#f48771")
          .attr("font-size", 8)
          .text(col.label);
        x += GAP;
      }

      // The correspondences are named rather than counted: phi_c and
      // phi_r are the functions under which the two sides are compared,
      // and which one is missing matters more than how many there are.
      g.append("text")
        .attr("x", x)
        .attr("y", 22)
        .attr("fill", r.corrs.length ? "#4ec9b0" : "#f48771")
        .attr("font-size", 9)
        .text(r.corrs.length ? "under " + r.corrs.join(", ") : "no correspondence");

      if (!r.hasResponseClause)
        g.append("text")
          .attr("y", 8 + BOX + 24)
          .attr("fill", "#f48771")
          .attr("font-size", 8.5)
          .text("no response clause — structure alone is not sufficient");
    });
  }, [ast, onJump]);

  return <div ref={ref} className="overflow-auto" />;
}

/* ------------------------------------------------------------------ */

const EMPTY = (msg) => (
  <div className="p-4 text-[11px] leading-relaxed text-[#8a8a8a]">{msg}</div>
);

export default function Views({ view, result, file, onJump }) {
  // A program refused by the PARSER has no tree, and saying so is the
  // honest thing to draw; an empty chart would read as "nothing to
  // report", which is the opposite of what happened. A program refused
  // by the CHECKER does have one -- it parsed -- and that tree is worth
  // drawing, because the refusal is usually a relation between two
  // nodes a reader can see in it.
  if (!result.ok && !result.ast)
    return EMPTY(
      `No diagram: the program was refused (${result.error}). ` +
        `The views draw a syntax tree, and a refused program does not have one.`
    );

  const refused = !result.ok;
  const ast = result.ast;

  return (
    <div className="h-full overflow-auto bg-[#1e1e1e]">
      {refused && (
        <div className="border-b border-[#3a3a3a] bg-[#3a2d00] px-3 py-2 text-[10.5px] leading-relaxed text-[#e2c08d]">
          This program parses, so a tree is drawn — but the checker
          refuses it ({result.error}
          {result.line ? `, line ${result.line}` : ""}). The diagram below
          shows a program the language does not accept.
        </div>
      )}
      {view === "Tree" && <TreeView ast={ast} onJump={onJump} />}
      {view === "Frames" && <FrameView ast={ast} onJump={onJump} />}
      {view === "Residue" &&
        (residueLedger(ast).length ? (
          <ResidueView ast={ast} onJump={onJump} />
        ) : (
          EMPTY(
            "This program binds no residue. Only `bind` produces one, and " +
              "this script has no comparison."
          )
        ))}
      {view === "Dataflow" &&
        (dataflow(ast).nodes.length ? (
          <DataflowView ast={ast} onJump={onJump} />
        ) : (
          EMPTY("This program binds no names, so there is no dataflow to draw.")
        ))}
      {view === "Params" &&
        (parameterSurface(ast).length ? (
          <ParameterView ast={ast} onJump={onJump} />
        ) : (
          EMPTY(
            "This program states no parameters. That is not a program " +
              "relying on defaults — the language has none; this one " +
              "simply calls nothing that takes a parameter."
          )
        ))}
      {view === "Arity" &&
        (alignmentArity(ast).length ? (
          <ArityView ast={ast} onJump={onJump} />
        ) : (
          EMPTY(
            "This program contains no `align`. The four-column rule is a " +
              "rule about alignments, and there is none here to check."
          )
        ))}
    </div>
  );
}
