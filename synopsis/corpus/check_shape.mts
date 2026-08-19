// Checks the IDE's view adapters against the TypeScript front-end.
//
// `check_ide.mts` verifies that the page's PARSER agrees with the CLI.
// This verifies the layer above it: the adapters in
// vivid-symbolism/src/lib/synopsis-shape.js, which turn a syntax tree
// into the shapes the D3 views draw.
//
// The reason this file exists rather than trusting the views to look
// right: a view that silently drops a field looks fine. That is exactly
// how the raw-JSON tab came to display every parameter block as `{}`
// for as long as it did -- `Params` is a Map, `JSON.stringify` renders
// a Map as `{}`, nothing threw, and nothing looked wrong. So the
// assertions below are written against the fields most likely to
// vanish quietly, and the Map case is checked first and by name.
//
// Run: node --experimental-strip-types check_shape.mts

import { ALL_FILES } from "../../vivid-symbolism/src/lib/synopsis/tutorial.js";
import { parse } from "../ts/src/parser.ts";
import {
  astTree,
  residueLedger,
  frameMap,
  paramList,
  dataflow,
  parameterSurface,
  alignmentArity,
} from "../../vivid-symbolism/src/lib/synopsis-shape.js";

let fails = 0;
const ok = (cond: boolean, what: string) => {
  if (!cond) { fails++; console.log(`  FAIL  ${what}`); }
  return cond;
};

/**
 * The label the adapter is expected to give an expression. Only the
 * cases that appear as comparison operands are needed; anything else
 * returns null, which fails the comparison loudly rather than
 * accidentally matching.
 */
function exprName(e: any): string | null {
  if (e?.node === "Var") return e.name;
  if (e?.node === "Num") return String(e.value);
  if (e?.node === "Project") return `project by ${e.projector}`;
  return null;
}

/** Every node in an adapter tree, flattened. */
function flat(n: any, out: any[] = []): any[] {
  out.push(n);
  for (const k of n.children ?? []) flat(k, out);
  return out;
}

// --------------------------------------------------------------- 1
// The Map bug, checked directly. This is the regression that motivated
// the whole module, so it is asserted on its own rather than left to
// fall out of a broader check.

console.log("params are read from a Map, not stringified");
{
  const src = ALL_FILES.find((f: any) => f.id === "06-motif-scan")!.src;
  const ast: any = parse(src);
  const detect = ast.frames[0].body.find((s: any) => s.expr?.node === "Detect");
  ok(detect !== undefined, "the motif scan contains a detect block");
  ok(detect.expr.params instanceof Map, "its params really are a Map");
  ok(detect.expr.params.size === 3, "the Map holds three parameters");

  // The old behaviour, asserted so the regression is documented in the
  // suite rather than only in a commit message.
  ok(
    JSON.stringify(detect.expr.params) === "{}",
    "JSON.stringify still renders that Map as {} (why the raw view was wrong)",
  );

  const ps = paramList(detect.expr.params);
  ok(ps.length === 3, "paramList recovers all three");
  const keys = ps.map(([k]: any) => k).sort().join(",");
  ok(keys === "min_distance,min_score,z", `keys recovered: got ${keys}`);

  // And end to end: the values must reach the drawn tree.
  const tree = astTree(ast);
  const withParams = flat(tree).filter((n: any) => n.params);
  ok(
    withParams.some((n: any) => n.params.includes("min_score")),
    "min_score reaches a tree node label",
  );
}

// --------------------------------------------------------------- 2
// Structural agreement with the parser, over every tutorial script.
//
// The tree must contain one node per statement (nested bodies
// included) plus one per declaration plus one per frame. Comparing
// against a recount of the AST rather than against a pinned number
// means the check follows the grammar if the grammar grows.

console.log("\ntree covers every declaration, frame and statement");
{
  const countStmts = (body: any[]): number =>
    body.reduce(
      (n, s) => n + 1 + (Array.isArray(s.body) ? countStmts(s.body) : 0),
      0,
    );

  for (const f of ALL_FILES as any[]) {
    let ast: any;
    try { ast = parse(f.src); } catch { continue; } // refusals: no tree
    const tree = astTree(ast);
    const nodes = flat(tree);

    const want =
      ast.decls.length +
      ast.frames.length +
      ast.frames.reduce((n: number, fr: any) => n + countStmts(fr.body), 0);

    // Every statement/decl/frame node, and no fewer. Expression nodes
    // add to the total, so the tree is larger -- but never smaller.
    const structural = nodes.filter((n: any) =>
      ["Open", "MethodDecl", "FrameBlock", "Let", "Bind", "Relax",
       "Claim", "Record", "Drop", "For", "Sweep"].includes(n.kind),
    ).length;

    ok(structural === want, `${f.id}: ${structural} structural nodes, want ${want}`);

    // Operand ORDER, not just operand count. Swapping the two sides of
    // a comparison leaves every count identical, so the checks above
    // pass a tree that says `compare t against q` for a program that
    // says `compare q against t`. The parser suite catches this class
    // at the AST level; the adapter needs its own guard because it
    // re-derives the child order itself.
    for (const fr of ast.frames as any[]) {
      const stack = [...fr.body];
      while (stack.length) {
        const s: any = stack.pop();
        if (Array.isArray(s.body)) stack.push(...s.body);
        if (s.expr?.node !== "Compare") continue;
        const drawn = flat(tree).find(
          (n: any) => n.kind === "Compare" && n.line === s.expr.line,
        );
        ok(drawn !== undefined, `${f.id}: the comparison is drawn`);
        const kids = (drawn?.children ?? []).map((k: any) => k.label);
        ok(kids[0] === exprName(s.expr.left) && kids[1] === exprName(s.expr.right),
           `${f.id}: comparison operands in source order, got [${kids.join(", ")}]`);
      }
    }
    ok(nodes.every((n: any) => n.label && n.label.length > 0),
       `${f.id}: every node carries a label`);
    ok(new Set(nodes.map((n: any) => n.id)).size === nodes.length,
       `${f.id}: node ids are unique`);
  }
}

// --------------------------------------------------------------- 3
// The residue ledger must agree with the refusal the corpus specifies.
//
// `unconsumed-residue` is in the corpus BECAUSE its residue is never
// consumed. If the ledger reported it as consumed, the view would be
// contradicting the language spec, and it would do so silently.

console.log("\nresidue ledger agrees with the corpus refusals");
{
  // Nesting first. `12-transfer` is the only lesson that puts a
  // `record` inside a `for` body, and it is therefore the only script
  // that can tell a walker which descends from one which does not.
  // Without this, dropping the recursion in `walk` passes the whole
  // suite -- verified by mutation, which is why it is asserted by name
  // rather than left implicit in the aggregate checks below.
  const nested = (ALL_FILES as any[]).find((f) => f.id === "12-transfer")!;
  const nast: any = parse(nested.src);
  const inner = nast.frames
    .flatMap((fr: any) => fr.body)
    .find((s: any) => Array.isArray(s.body));
  ok(inner !== undefined, "12-transfer has a nested body");
  const innerNames = inner.body
    .filter((s: any) => s.node === "Record")
    .flatMap((s: any) => s.names);
  ok(innerNames.length > 0, "and records something inside it");

  const seen = new Set(
    frameMap(nast).frames.flatMap((fr: any) => fr.bound.map((b: any) => b.name)),
  );
  // NOT the loop variable: `for u in ...` is itself a top-level
  // statement, so `u` is visited before any recursion and proves
  // nothing. The names bound by the `let`s INSIDE the body are the
  // ones a non-descending walker cannot reach.
  const innerBound = inner.body
    .filter((s: any) => s.node === "Let")
    .map((s: any) => s.name);
  ok(innerBound.length > 0, "the nested body binds names of its own");
  const missed = innerBound.filter((n: string) => !seen.has(n));
  ok(missed.length === 0,
     `names bound inside the loop are reached: missing [${missed.join(", ")}]`);

  const bad = (ALL_FILES as any[]).find((f) => f.id === "unconsumed-residue")!;
  const ledger = residueLedger(parse(bad.src));
  ok(ledger.length > 0, "the offending program does bind a residue");
  ok(ledger.some((r: any) => !r.consumed),
     "and the ledger reports at least one unconsumed");

  // The converse, which is the half that catches a ledger stuck on
  // "everything is unconsumed": the accepted lessons must be clean.
  for (const f of (ALL_FILES as any[]).filter((x) => x.group !== "refusals")) {
    const rows = residueLedger(parse(f.src));
    const leaked = rows.filter((r: any) => !r.consumed).map((r: any) => r.name);
    ok(leaked.length === 0,
       `${f.id}: no unconsumed residue, got [${leaked.join(", ")}]`);
  }
}

// --------------------------------------------------------------- 4
// The frame map must locate the cross-frame reference.

console.log("\nframe map locates the cross-frame reference");
{
  const bad = (ALL_FILES as any[]).find((f) => f.id === "cross-frame-reference")!;
  const fm = frameMap(parse(bad.src));
  ok(fm.frames.length === 2, `two frames, got ${fm.frames.length}`);
  ok(fm.frames.some((fr: any) => fr.foreign.length > 0),
     "one frame uses a name it did not bind");

  // Accepted lessons must show no foreign names at all, or the view
  // would flag correct programs.
  for (const f of (ALL_FILES as any[]).filter((x) => x.group !== "refusals")) {
    const m = frameMap(parse(f.src));
    const foreign = m.frames.flatMap((fr: any) => fr.foreign);
    ok(foreign.length === 0,
       `${f.id}: no foreign names, got [${foreign.join(", ")}]`);
  }
}

// --------------------------------------------------------------- 5
// Dataflow resolves names per frame, not globally.
//
// This is the assertion that caught a real defect. The first version
// of `dataflow` kept one global name table, so in
// `cross-frame-reference` the name `a` -- bound in `nucleotide_frame`
// and used illegally in `protein_frame` -- resolved cleanly and the
// graph reported nothing dangling. Under the frame rule that edge is
// precisely the refusal: `a` in the second frame is not an
// out-of-scope name, it is a value of a different type with no
// coercion to this one. A graph that quietly resolves it draws a
// program the checker refuses.

console.log("\ndataflow resolves names per frame");
{
  const bad = (ALL_FILES as any[]).find((f) => f.id === "cross-frame-reference")!;
  const g = dataflow(parse(bad.src));
  const dangling = g.links.filter((l: any) => !l.resolved);
  ok(dangling.length > 0, "the cross-frame use leaves an unresolved edge");
  ok(dangling.every((l: any) => l.source === "a"),
     `and it is 'a' that fails to resolve, got [${dangling.map((l: any) => l.source).join(", ")}]`);

  // The converse. An adapter that marked everything unresolved would
  // satisfy the check above and be useless, so the accepted lessons
  // must come back clean.
  for (const f of (ALL_FILES as any[]).filter((x) => x.group !== "refusals")) {
    const d = dataflow(parse(f.src));
    const loose = d.links.filter((l: any) => !l.resolved).map((l: any) => l.source);
    ok(loose.length === 0, `${f.id}: every name resolves, got [${loose.join(", ")}]`);

    // Depth is the x axis of the view, so a collapsed depth would
    // render every value in one column and say nothing. A program
    // that projects and then compares must reach at least depth 2.
    const maxd = Math.max(0, ...d.nodes.map((n: any) => n.depth));
    const compares = parse(f.src).frames.some((fr: any) =>
      fr.body.some((s: any) => s.expr?.node === "Compare"));
    if (compares) ok(maxd >= 2, `${f.id}: derivation depth ${maxd}, want >= 2`);
  }

  // `undefined-variable` is the other half of the same property: the
  // name was never bound ANYWHERE, so it must dangle too.
  const undef = (ALL_FILES as any[]).find((f) => f.id === "undefined-variable")!;
  const u = dataflow(parse(undef.src));
  ok(u.links.some((l: any) => !l.resolved && l.source === "nonexistent"),
     "the unbound name dangles");
}

// --------------------------------------------------------------- 6
// The parameter surface is complete.
//
// The language has no defaults, so this view is the whole set of
// numbers a result rests on. A missing row is therefore not a cosmetic
// loss: it is a parameter the reader would not know had been chosen.
// The count is checked against a recount of the AST rather than a
// pinned number, so it follows the grammar if the grammar grows.

console.log("\nparameter surface covers every stated parameter");
{
  for (const f of (ALL_FILES as any[]).filter((x) => x.group !== "refusals")) {
    const ast: any = parse(f.src);
    const rows = parameterSurface(ast);

    // Recount from the tree: every Map-valued params field, plus the
    // arguments carried by a projector or a method declaration.
    let want = 0;
    for (const d of ast.decls)
      if (d.node === "MethodDecl") want += d.params?.size ?? 0;
    const visit = (body: any[]) => {
      for (const s of body) {
        const e = s.expr;
        if (e?.params instanceof Map) want += e.params.size;
        if (e?.args instanceof Map) want += e.args.size;
        if (s.params instanceof Map) want += s.params.size;
        if (Array.isArray(s.body)) visit(s.body);
      }
    };
    for (const fr of ast.frames) visit(fr.body);

    ok(rows.length === want,
       `${f.id}: ${rows.length} parameter rows, want ${want}`);
    ok(rows.every((r: any) => r.owner && r.key && r.line > 0),
       `${f.id}: every row names an owner, a key and a line`);
  }

  // The numeric flag drives whether a bar is drawn at all, so a
  // number misflagged as symbolic would silently lose its magnitude.
  const scan: any = parse(
    (ALL_FILES as any[]).find((f) => f.id === "06-motif-scan")!.src);
  const rows = parameterSurface(scan);
  const z = rows.find((r: any) => r.key === "z");
  ok(z !== undefined && z.numeric === true && z.value === 4.0,
     `z is numeric 4.0, got ${JSON.stringify(z)}`);
  const norm = rows.find((r: any) => String(r.value) === "normalised");
  ok(norm !== undefined && norm.numeric === false,
     "a symbolic argument is not flagged numeric");
}

// --------------------------------------------------------------- 7
// Alignment arity counts all four columns.
//
// The rule the view exists to show is that an alignment needs a
// central pair AND a response pair AND its correspondences: agreement
// on structure alone is not sufficient evidence. `has_response_clause`
// is carried rather than inferred, because the parser distinguishes a
// clause written empty from one that was never written, and that
// distinction is the whole point.

console.log("\nalignment arity carries all four columns");
{
  for (const id of ["11-alignment", "12-transfer"]) {
    const ast: any = parse((ALL_FILES as any[]).find((f) => f.id === id)!.src);
    const rows = alignmentArity(ast);
    ok(rows.length === 1, `${id}: one alignment, got ${rows.length}`);
    const a = rows[0];
    ok(a.central === 2, `${id}: central pair, got ${a.central}`);
    ok(a.resp === 2, `${id}: response pair, got ${a.resp}`);
    ok(a.hasResponseClause === true, `${id}: the response clause is written`);
    ok(a.corrs.length === 2,
       `${id}: two correspondences, got [${a.corrs.join(", ")}]`);
    ok(typeof a.theta === "number" && a.theta > 0,
       `${id}: theta is a stated positive number, got ${a.theta}`);
  }

  // A program with no alignment must yield no rows, not a row of
  // zeroes -- which would draw four empty slots and read as a defect.
  const scan: any = parse(
    (ALL_FILES as any[]).find((f) => f.id === "06-motif-scan")!.src);
  ok(alignmentArity(scan).length === 0,
     "a program without an alignment yields no arity rows");
}

console.log(
  fails === 0 ? "\nall green" : `\n${fails} failing assertion(s)`,
);
process.exit(fails === 0 ? 0 : 1);
