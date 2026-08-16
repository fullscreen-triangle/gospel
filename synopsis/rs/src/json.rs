//! AST -> the oracle's JSON shape.
//!
//! Written by hand rather than derived. The shape is a specification,
//! not a serialisation convenience: `synopsis/corpus/dump_ref_ast.py`
//! fixed it when it dumped the reference parser's trees, and the
//! TypeScript runner canonicalises to the same thing. Three rules carry
//! the weight, and a derive would get all three wrong:
//!
//!  1. `node` comes first, then `line`, then fields in DECLARATION
//!     order. The runners compare stable-stringified text, so field
//!     order is part of what conformance means.
//!  2. Params serialise as an ordered list of `[key, value]` pairs. A
//!     JSON object does not promise order, and the report prints
//!     parameters in source order.
//!  3. Integral floats serialise as integers, so Python's `float` and
//!     JavaScript's `Number` agree on the JSON text.

use crate::ast::*;
use crate::params::{Params, Value};
use serde_json::{json, Map, Value as J};

/// Rule 3: `2.0` becomes `2`, `0.5` stays `0.5`.
fn num(x: f64) -> J {
    if x.fract() == 0.0 && x.is_finite() {
        json!(x as i64)
    } else {
        json!(x)
    }
}

/// Rule 2.
fn params(p: &Params) -> J {
    J::Array(
        p.iter()
            .map(|(k, v)| {
                let jv = match v {
                    Value::Num(x) => num(*x),
                    Value::Name(s) => json!(s),
                };
                J::Array(vec![json!(k), jv])
            })
            .collect(),
    )
}

/// Rule 1: build a node with `node` and `line` first.
fn node(name: &str, line: usize, fields: &[(&str, J)]) -> J {
    let mut m = Map::new();
    m.insert("node".into(), json!(name));
    m.insert("line".into(), json!(line));
    for (k, v) in fields {
        m.insert((*k).into(), v.clone());
    }
    J::Object(m)
}

fn pair(p: &(Box<Expr>, Box<Expr>)) -> J {
    J::Array(vec![expr(&p.0), expr(&p.1)])
}

fn opt_str(s: &Option<String>) -> J {
    match s {
        Some(v) => json!(v),
        None => J::Null,
    }
}

pub fn expr(e: &Expr) -> J {
    match e {
        Expr::Project { line, src, projector, args } => node(
            "Project",
            *line,
            &[
                ("src", expr(src)),
                ("projector", json!(projector)),
                ("args", params(args)),
            ],
        ),
        Expr::Compare { line, left, right, method, params: p } => node(
            "Compare",
            *line,
            &[
                ("left", expr(left)),
                ("right", expr(right)),
                ("method", json!(method)),
                ("params", params(p)),
            ],
        ),
        Expr::Detect { line, kind, src, params: p } => node(
            "Detect",
            *line,
            &[
                ("kind", json!(kind)),
                ("src", expr(src)),
                ("params", params(p)),
            ],
        ),
        Expr::Align { line, central, resp, corrs, params: p, has_response_clause } => node(
            "Align",
            *line,
            &[
                ("central", pair(central)),
                ("resp", resp.as_ref().map_or(J::Null, pair)),
                ("corrs", json!(corrs)),
                ("params", params(p)),
                ("has_response_clause", json!(has_response_clause)),
            ],
        ),
        Expr::UnitExpr { line, src, anchors } => node(
            "UnitExpr",
            *line,
            &[("src", expr(src)), ("anchors", json!(anchors))],
        ),
        Expr::ResponseExpr { line, src, method } => node(
            "ResponseExpr",
            *line,
            &[("src", expr(src)), ("method", opt_str(method))],
        ),
        Expr::NearestUnit { line, net, to } => node(
            "NearestUnit",
            *line,
            &[("net", expr(net)), ("to", expr(to))],
        ),
        Expr::CorrExpr { line, src, dst, by } => node(
            "CorrExpr",
            *line,
            &[("src", expr(src)), ("dst", expr(dst)), ("by", json!(by))],
        ),
        Expr::Var { line, name } => node("Var", *line, &[("name", json!(name))]),
        Expr::Num { line, value } => node("Num", *line, &[("value", num(*value))]),
    }
}

pub fn stmt(s: &Stmt) -> J {
    match s {
        Stmt::Let { line, name, ann, expr: e } => node(
            "Let",
            *line,
            &[
                ("name", json!(name)),
                ("ann", opt_str(ann)),
                ("expr", expr(e)),
            ],
        ),
        Stmt::Bind { line, value, residue, expr: e } => node(
            "Bind",
            *line,
            &[
                ("value", json!(value)),
                ("residue", json!(residue)),
                ("expr", expr(e)),
            ],
        ),
        Stmt::Relax { line, target, params: p } => node(
            "Relax",
            *line,
            &[("target", json!(target)), ("params", params(p))],
        ),
        Stmt::Sweep { line, var, lo, hi, step, values, body } => node(
            "Sweep",
            *line,
            &[
                ("var", json!(var)),
                ("lo", num(*lo)),
                ("hi", num(*hi)),
                ("step", num(*step)),
                (
                    "values",
                    match values {
                        Some(v) => J::Array(v.iter().map(|x| num(*x)).collect()),
                        None => J::Null,
                    },
                ),
                ("body", J::Array(body.iter().map(stmt).collect())),
            ],
        ),
        Stmt::For { line, var, src, guard, body } => node(
            "For",
            *line,
            &[
                ("var", json!(var)),
                ("src", expr(src)),
                ("guard", opt_str(guard)),
                ("body", J::Array(body.iter().map(stmt).collect())),
            ],
        ),
        Stmt::Claim { line, text, expr: e } => node(
            "Claim",
            *line,
            &[("text", json!(text)), ("expr", expr(e))],
        ),
        Stmt::Record { line, names } => node("Record", *line, &[("names", json!(names))]),
        Stmt::Drop { line, name } => node("Drop", *line, &[("name", json!(name))]),
    }
}

pub fn decl(d: &Decl) -> J {
    match d {
        Decl::Open { line, name, path } => node(
            "Open",
            *line,
            &[("name", json!(name)), ("path", json!(path))],
        ),
        Decl::MethodDecl { line, name, spec, params: p } => node(
            "MethodDecl",
            *line,
            &[
                ("name", json!(name)),
                ("spec", json!(spec)),
                ("params", params(p)),
            ],
        ),
    }
}

pub fn frame(f: &FrameBlock) -> J {
    node(
        "FrameBlock",
        f.line,
        &[
            ("name", json!(f.name)),
            ("body", J::Array(f.body.iter().map(stmt).collect())),
        ],
    )
}

pub fn program(p: &Program) -> J {
    node(
        "Program",
        p.line,
        &[
            ("decls", J::Array(p.decls.iter().map(decl).collect())),
            ("frames", J::Array(p.frames.iter().map(frame).collect())),
            ("report", opt_str(&p.report)),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn node_and_line_come_first() {
        // This fails without serde_json's `preserve_order` feature: the
        // default Map is a BTreeMap, so it would sort the keys and emit
        // `decls` first. Field order is part of conformance, so that
        // feature is a requirement and this test is what enforces it.
        let p = parse("open a = \"x\"\nunder f { record a }\nreport to \"r\"").unwrap();
        let j = program(&p);
        let keys: Vec<&String> = j.as_object().unwrap().keys().collect();
        assert_eq!(keys[0], "node");
        assert_eq!(keys[1], "line");
        assert_eq!(keys[2], "decls");
    }

    #[test]
    fn params_keep_source_order_as_pair_lists() {
        let src = "open a = \"x\"\nunder f {\n  let z = detect peaks in a \
                   { z = 3 ; min_distance = 10 }\n}\nreport to \"r\"";
        let j = program(&parse(src).unwrap());
        let det = &j["frames"][0]["body"][0]["expr"];
        assert_eq!(det["params"][0][0], json!("z"));
        assert_eq!(det["params"][1][0], json!("min_distance"));
    }

    #[test]
    fn integral_floats_serialise_as_integers() {
        // So the JSON text matches Python's and JavaScript's.
        assert_eq!(num(3.0), json!(3));
        assert_eq!(num(0.5), json!(0.5));
    }
}
