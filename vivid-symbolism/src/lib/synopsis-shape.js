/**
 * Adapters from the syntax tree to the shapes the D3 views draw.
 *
 * These are pure functions over the AST and nothing else. That is the
 * whole design constraint: the panel must never display a number the
 * compiler did not produce, so there is no data source here other than
 * the tree the parser just built. When the evaluator lands (stage E)
 * its results become a second, clearly separate input -- not a
 * retrofit of these functions.
 *
 * One thing worth knowing before reading further: `Params` in the AST
 * is a real `Map`, not a plain object. `JSON.stringify` renders a Map
 * as `{}`, which is why the raw-JSON view silently showed every
 * parameter block as empty. Parameters are the language's central
 * claim -- no defaults, everything stated -- so hiding them was the
 * worst possible field to lose. `paramList` below is the fix, and it is
 * the reason these adapters exist rather than the views reading the
 * tree directly.
 */

/** A Map, a plain object, or null -> [[key, value], ...]. Never throws. */
export function paramList(params) {
  if (!params) return [];
  if (params instanceof Map) return [...params.entries()];
  if (typeof params === "object") return Object.entries(params);
  return [];
}

/** Short human label for a parameter block, e.g. "z=4, min_distance=30". */
function paramSummary(params) {
  const ps = paramList(params);
  if (ps.length === 0) return "";
  return ps.map(([k, v]) => `${k}=${v}`).join(", ");
}

/**
 * A one-line description of an expression, used as a tree node label.
 * Deliberately terse: the tree is read at 11px in a 22rem column.
 */
function exprLabel(e) {
  if (!e || !e.node) return "?";
  switch (e.node) {
    case "Var": return e.name;
    case "Num": return String(e.value);
    case "Project": return `project by ${e.projector}`;
    case "Compare": return `compare by ${e.method}`;
    case "Detect": return `detect ${e.kind}`;
    case "Align": return "align";
    case "UnitExpr": return `unit anchors ${e.anchors}`;
    case "ResponseExpr": return `response by ${e.method ?? "(anonymous)"}`;
    case "NearestUnit": return "nearest_unit";
    case "CorrExpr": return `corr by ${e.by}`;
    default: return e.node;
  }
}

/** Child expressions of an expression, in source order, with role names. */
function exprKids(e) {
  if (!e || !e.node) return [];
  switch (e.node) {
    case "Project": return [["src", e.src]];
    case "Compare": return [["left", e.left], ["right", e.right]];
    case "Detect": return [["in", e.src]];
    case "UnitExpr": return [["src", e.src]];
    case "ResponseExpr": return [["src", e.src]];
    case "NearestUnit": return [["net", e.net], ["to", e.to]];
    case "CorrExpr": return [["from", e.src], ["to", e.dst]];
    case "Align": {
      const out = [["central", e.central[0]], ["central", e.central[1]]];
      // resp is null for the false-friend form the arity rule refuses.
      // Showing its absence is more informative than omitting the row.
      if (e.resp) out.push(["response", e.resp[0]], ["response", e.resp[1]]);
      return out;
    }
    default: return [];
  }
}

/** Parameters carried by an expression, if it has any. */
function exprParams(e) {
  if (!e) return null;
  if (e.node === "Project") return e.args;
  if (e.node === "Compare" || e.node === "Detect" || e.node === "Align") return e.params;
  return null;
}

let uid = 0;
const node = (o) => ({ id: `n${uid++}`, children: [], ...o });

function fromExpr(e, role) {
  const n = node({
    label: exprLabel(e),
    role,
    line: e?.line ?? null,
    kind: e?.node ?? "?",
    params: paramSummary(exprParams(e)),
  });
  n.children = exprKids(e).map(([r, kid]) => fromExpr(kid, r));
  return n;
}

function fromStmt(s) {
  const base = { line: s.line, kind: s.node };
  switch (s.node) {
    case "Let":
      return node({ ...base, label: `let ${s.name}`, children: [fromExpr(s.expr, "=")] });
    case "Bind":
      // Both names are shown because binding two is the point: the
      // residue is not a by-product, it is half the result.
      return node({
        ...base,
        label: `bind ${s.value}, ${s.residue}`,
        residue: s.residue,
        children: [fromExpr(s.expr, "=")],
      });
    case "Relax":
      return node({ ...base, label: `relax ${s.target}`, params: paramSummary(s.params) });
    case "Claim":
      return node({ ...base, label: `claim "${s.text}"`, children: [fromExpr(s.expr, "=")] });
    case "Record":
      return node({ ...base, label: `record ${s.names.join(", ")}` });
    case "Drop":
      return node({ ...base, label: `drop ${s.name}` });
    case "For":
      return node({
        ...base,
        label: `for ${s.var} in items(...)${s.guard ? ` where ${s.guard}` : ""}`,
        children: s.body.map(fromStmt),
      });
    case "Sweep":
      return node({
        ...base,
        label: s.values
          ? `sweep ${s.var} in [${s.values.join(", ")}]`
          : `sweep ${s.var} in ${s.lo}..${s.hi} step ${s.step}`,
        children: s.body.map(fromStmt),
      });
    default:
      return node({ ...base, label: s.node });
  }
}

function fromDecl(d) {
  if (d.node === "Open")
    return node({ label: `open ${d.name}`, line: d.line, kind: "Open", params: d.path });
  return node({
    label: `method ${d.name}`,
    line: d.line,
    kind: "MethodDecl",
    params: `${d.spec}(${paramSummary(d.params)})`,
  });
}

/**
 * The whole program as one d3.hierarchy-ready tree.
 * Returns null for a program that did not parse -- there is no tree to
 * draw, and inventing a partial one would misrepresent what happened.
 */
export function astTree(ast) {
  if (!ast) return null;
  uid = 0;
  return node({
    label: "program",
    kind: "Program",
    line: 1,
    params: ast.report ? `report to "${ast.report}"` : "",
    children: [
      ...ast.decls.map(fromDecl),
      ...ast.frames.map((f) =>
        node({
          label: `under ${f.name}`,
          line: f.line,
          kind: "FrameBlock",
          children: f.body.map(fromStmt),
        })
      ),
    ],
  });
}

/* ------------------------------------------------------------------ */

/** Walk every statement in a frame, including nested for/sweep bodies. */
function walk(body, fn) {
  for (const s of body) {
    fn(s);
    if (Array.isArray(s.body)) walk(s.body, fn);
  }
}

/** Names mentioned anywhere in an expression. */
function mentions(e, out = []) {
  if (!e || !e.node) return out;
  if (e.node === "Var") out.push(e.name);
  for (const [, kid] of exprKids(e)) mentions(kid, out);
  return out;
}

/**
 * The residue ledger: every `bind`, and whether its residue is consumed
 * before the frame closes.
 *
 * This is stage-A information only -- it reports what the source says,
 * not what a checker decided, because the checker does not exist yet.
 * A residue counts as consumed if it is recorded, dropped, or mentioned
 * by a later expression. That is the rule as specified; when stage C
 * implements it, this view and the checker must agree, and any
 * disagreement is a defect in one of them.
 */
export function residueLedger(ast) {
  if (!ast) return [];
  const rows = [];
  for (const f of ast.frames) {
    const binds = [];
    const used = new Set();
    walk(f.body, (s) => {
      if (s.node === "Bind") binds.push({ name: s.residue, value: s.value, line: s.line });
      if (s.node === "Record") s.names.forEach((n) => used.add(n));
      if (s.node === "Drop") used.add(s.name);
      if (s.expr) mentions(s.expr).forEach((n) => used.add(n));
    });
    for (const b of binds)
      rows.push({ ...b, frame: f.name, consumed: used.has(b.name) });
  }
  return rows;
}

/**
 * Frames, and what is bound inside each.
 *
 * The point of drawing this is Thm 9.6: a frame index is part of a
 * value's type, so a name bound in one frame is not the same type as a
 * name bound in another. Showing the frames as disjoint boxes makes
 * that visible rather than merely stated -- and makes the
 * cross-frame-reference refusal legible, since the offending name is
 * visibly in the wrong box.
 */
export function frameMap(ast) {
  if (!ast) return { opens: [], methods: [], frames: [] };
  return {
    opens: ast.decls.filter((d) => d.node === "Open").map((d) => d.name),
    methods: ast.decls.filter((d) => d.node === "MethodDecl").map((d) => d.name),
    frames: ast.frames.map((f, i) => {
      const bound = [];
      const usedHere = new Set();
      walk(f.body, (s) => {
        if (s.node === "Let") bound.push({ name: s.name, line: s.line, kind: "let" });
        if (s.node === "Bind") {
          bound.push({ name: s.value, line: s.line, kind: "bind" });
          bound.push({ name: s.residue, line: s.line, kind: "residue" });
        }
        if (s.node === "For") bound.push({ name: s.var, line: s.line, kind: "for" });
        if (s.node === "Sweep") bound.push({ name: s.var, line: s.line, kind: "sweep" });
        if (s.expr) mentions(s.expr).forEach((n) => usedHere.add(n));
        if (s.node === "Record") s.names.forEach((n) => usedHere.add(n));
      });
      const names = new Set(bound.map((b) => b.name));
      return {
        name: f.name,
        index: i,
        line: f.line,
        bound,
        // Names used here that this frame never bound and no `open`
        // provided. Under the frame rule these are the interesting
        // ones: a value from a sibling frame is a DIFFERENT type, not
        // merely out of scope.
        foreign: [...usedHere].filter(
          (n) =>
            !names.has(n) &&
            !ast.decls.some((d) => d.name === n)
        ),
      };
    }),
  };
}

/* ------------------------------------------------------------------ */
/* Adapters for the quantitative views.                                */
/*                                                                     */
/* The three above answer "what shape is this program". The three      */
/* below answer questions with numeric answers: how far each value     */
/* stands from its inputs, what the whole parameter surface is, and    */
/* how many correspondences each alignment carries. They are still     */
/* measured from the tree and nothing else -- the evaluator does not   */
/* exist, so a chart whose axis was a SCORE would be a chart of a      */
/* number no compiler produced. Depth, count and arity are produced,   */
/* by the parser, and they move when the program moves, which is the   */
/* only property that makes a chart worth drawing rather than          */
/* decorating.                                                         */

/**
 * The dataflow graph: one node per bound name, one edge per use.
 *
 * This is the view the tree cannot give. A syntax tree nests by
 * grammar, so `let a` and `let b` are siblings however tightly the
 * second depends on the first -- the dependency between them travels
 * in a name, not in the nesting. Drawing names as nodes and uses as
 * edges puts the real order of derivation on screen, and a node's
 * depth (its distance from the nearest `open`) measures how much
 * machinery stands between a file on disk and a claim about it.
 */
export function dataflow(ast) {
  if (!ast) return { nodes: [], links: [] };

  const nodes = [];
  const links = [];
  const byName = new Map();

  const add = (n) => {
    // A later binding of the same name shadows the earlier one for
    // subsequent uses, and overwriting in `byName` reproduces that.
    // The earlier node stays in `nodes`: it was still derived, and
    // dropping it would erase work the program actually did.
    nodes.push(n);
    byName.set(n.name, n);
    return n;
  };

  for (const d of ast.decls)
    add({
      name: d.name,
      kind: d.node === "Open" ? "open" : "method",
      line: d.line,
      frame: null,
      depth: 0,
    });

  for (const f of ast.frames) {
    // Visibility is per-frame, not global. A name bound in a sibling
    // frame is NOT in scope here -- under the frame rule it is a value
    // of a different type, with no coercion to this one -- so it must
    // not silently resolve. Declarations are the exception: `open` and
    // `method` sit outside every frame and are visible in all of them.
    const visible = new Set(ast.decls.map((d) => d.name));

    walk(f.body, (s) => {
      const from = s.expr ? mentions(s.expr) : [];

      // A response names its method as a bare string, not a Var, so it
      // never turns up in `mentions`. Adding it explicitly is what
      // makes "this declared method is used here" an edge instead of
      // leaving the method unconnected at the left margin.
      if (s.expr && s.expr.node === "ResponseExpr" && s.expr.method)
        from.push(s.expr.method);

      const targets = [];
      if (s.node === "Let") targets.push({ name: s.name, kind: "let" });
      if (s.node === "Bind") {
        targets.push({ name: s.value, kind: "bind" });
        targets.push({ name: s.residue, kind: "residue" });
      }
      if (s.node === "For") targets.push({ name: s.var, kind: "for" });
      if (s.node === "Sweep") targets.push({ name: s.var, kind: "sweep" });

      for (const t of targets) {
        const depth =
          from.length === 0
            ? 1
            : 1 +
              Math.max(
                0,
                ...from.map((n) => (visible.has(n) ? byName.get(n)?.depth ?? 0 : 0))
              );
        const n = add({ ...t, line: s.line, frame: f.name, depth });
        visible.add(n.name);
        for (const src of from)
          // An unresolved source is exactly the undefined-variable and
          // cross-frame cases. It is kept as a dangling edge rather
          // than dropped, because the missing end IS the diagnostic --
          // and this view is drawn for stage-B refusals.
          links.push({ source: src, target: n.name, resolved: visible.has(src) });
      }
    });
  }

  return { nodes, links };
}

/**
 * Every parameter written anywhere in the program, with its owner.
 *
 * "There are no defaults" is the language's sharpest claim, and no
 * view showed its consequence: the complete set of numbers a result
 * rests on, in one place. That set is what a reader would need in
 * order to reproduce the result, and its size is a real property of
 * the program -- it grows as the program commits to more.
 */
export function parameterSurface(ast) {
  if (!ast) return [];
  const rows = [];

  const push = (owner, ownerKind, line, frame, params) => {
    for (const [k, v] of paramList(params))
      rows.push({
        owner,
        ownerKind,
        key: k,
        value: v,
        numeric: typeof v === "number",
        line,
        frame,
      });
  };

  for (const d of ast.decls)
    if (d.node === "MethodDecl")
      push(d.name, "method", d.line, null, d.params);

  for (const f of ast.frames)
    walk(f.body, (s) => {
      if (s.node === "Relax")
        push(`relax ${s.target}`, "relax", s.line, f.name, s.params);
      const e = s.expr;
      if (!e) return;
      if (e.node === "Project") push(e.projector, "projector", s.line, f.name, e.args);
      if (e.node === "Compare") push(e.method, "method", s.line, f.name, e.params);
      if (e.node === "Detect") push(`detect ${e.kind}`, "detect", s.line, f.name, e.params);
      if (e.node === "Align") push("align", "align", s.line, f.name, e.params);
    });

  return rows;
}

/**
 * Alignments, and the correspondences each one carries.
 *
 * The four-column arity is the rule that an alignment needs BOTH a
 * central pair and a response pair: agreement on structure alone, or
 * on behaviour alone, is not sufficient evidence. That rule is about
 * a count, so a count is the honest thing to draw. A row whose
 * response column is empty is exactly the shape the checker refuses,
 * and it reads as a missing bar rather than as prose.
 *
 * `has_response_clause` is carried rather than inferred from `resp`,
 * because the two differ in the case the rule exists for: the parser
 * records a written-but-empty clause distinctly from an absent one.
 */
export function alignmentArity(ast) {
  if (!ast) return [];
  const rows = [];
  for (const f of ast.frames)
    walk(f.body, (s) => {
      const e = s.expr;
      if (!e || e.node !== "Align") return;
      rows.push({
        name: s.name ?? s.value ?? "align",
        line: s.line,
        frame: f.name,
        central: e.central ? e.central.length : 0,
        resp: e.resp ? e.resp.length : 0,
        hasResponseClause: !!e.has_response_clause,
        corrs: e.corrs ? e.corrs.slice() : [],
        theta: paramList(e.params).find(([k]) => k === "theta")?.[1] ?? null,
      });
    });
  return rows;
}
