import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { astTree, residueLedger, frameMap } from "@/lib/synopsis-shape";

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
 * the stage-B notice exists to avoid.
 *
 * What the parser DOES know is structural, and structure is what these
 * three views show: the shape of the program, where its values live,
 * and what it did with the part of each comparison the method could not
 * explain.
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

const EMPTY = (msg) => (
  <div className="p-4 text-[11px] leading-relaxed text-[#8a8a8a]">{msg}</div>
);

export default function Views({ view, result, file, onJump }) {
  // A refused program has no tree. Saying so is the honest thing to
  // draw; an empty chart would read as "nothing to report", which is
  // the opposite of what happened.
  if (!result.ok)
    return EMPTY(
      `No diagram: the program was refused (${result.error}). ` +
        `The views draw a syntax tree, and a refused program does not have one.`
    );

  // Stage B is specified but not implemented, so a program the checker
  // would refuse still parses. The Problems tab already says this; the
  // diagram must not quietly contradict it by drawing a clean picture.
  const pending = file?.group === "refusals" && file.stage === "B" && result.ok;

  const ast = result.ast;

  return (
    <div className="h-full overflow-auto bg-[#1e1e1e]">
      {pending && (
        <div className="border-b border-[#3a3a3a] bg-[#3a2d00] px-3 py-2 text-[10.5px] leading-relaxed text-[#e2c08d]">
          This program parses, so a tree is drawn. The rule that refuses it
          belongs to the type checker, which is not implemented — so the
          diagram below shows a program the language does not accept.
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
    </div>
  );
}
