//! The synopsis parser.
//!
//! A direct port of `validation/lang.py:333-661` and of
//! `synopsis/ts/src/parser.ts`. Recursive descent, no backtracking, no
//! operator precedence -- every form is prefix-keyword led, which is why
//! the grammar can be read off the theorems.
//!
//! The shape of this grammar is itself load-bearing. There is no `while`
//! production, no `if`, and no recursion, so `thm:total` (every program
//! terminates) is a property of what can be *written*, not of a check
//! that runs afterwards. Likewise there is no indexing production, which
//! is what makes `thm:noexact` true of the language rather than of the
//! checker: a program cannot reach into a sequence position even in
//! principle.

use crate::ast::*;
use crate::errors::{Result, SynopsisError};
use crate::params::{Params, Value};
use crate::tokens::{tokenise, Token, TokenKind};

pub fn parse(src: &str) -> Result<Program> {
    let toks = tokenise(src)?;
    Parser { toks, i: 0 }.program()
}

struct Parser {
    toks: Vec<Token>,
    i: usize,
}

impl Parser {
    // -- helpers -------------------------------------------------------

    fn cur(&self) -> &Token {
        // The tokenizer always appends Eof, so this never runs off the
        // end for a parser that stops at Eof.
        &self.toks[self.i]
    }

    fn line(&self) -> usize {
        self.cur().line
    }

    /// Note the kind restriction: an identifier that happens to spell a
    /// keyword-like word never matches here, and -- more importantly --
    /// `at(":")` is always false, because ':' is not in the tokenizer's
    /// punctuation class at all. See the `Let` note in corpus/ast.json.
    fn at(&self, text: &str) -> bool {
        let t = self.cur();
        t.text == text
            && matches!(t.kind, TokenKind::Kw | TokenKind::Punct | TokenKind::Range)
    }

    fn eat(&mut self, text: &str) -> Result<&Token> {
        if !self.at(text) {
            return Err(SynopsisError::parse(
                format!("expected {:?}, found {:?}", text, self.cur().text),
                self.cur().line,
            ));
        }
        let i = self.i;
        self.i += 1;
        Ok(&self.toks[i])
    }

    /// `eat` when only the line is wanted, which is the common case.
    fn eat_line(&mut self, text: &str) -> Result<usize> {
        Ok(self.eat(text)?.line)
    }

    fn ident(&mut self) -> Result<String> {
        if self.cur().kind != TokenKind::Ident {
            return Err(SynopsisError::parse(
                format!("expected identifier, found {:?}", self.cur().text),
                self.cur().line,
            ));
        }
        let s = self.cur().text.clone();
        self.i += 1;
        Ok(s)
    }

    fn number(&mut self) -> Result<f64> {
        if self.cur().kind != TokenKind::Num {
            return Err(SynopsisError::parse(
                format!("expected number, found {:?}", self.cur().text),
                self.cur().line,
            ));
        }
        let v = self.cur().num.expect("Num token carries a value");
        self.i += 1;
        Ok(v)
    }

    fn string(&mut self) -> Result<String> {
        if self.cur().kind != TokenKind::Str {
            return Err(SynopsisError::parse(
                format!("expected string, found {:?}", self.cur().text),
                self.cur().line,
            ));
        }
        let s = self.cur().value_str.clone().expect("Str token carries a value");
        self.i += 1;
        Ok(s)
    }

    /// Consume the current token and return its text. Used where the
    /// grammar accepts any word in a slot (projector, method, detector
    /// kind) and the checker -- not the parser -- decides which words
    /// are legal.
    fn take_text(&mut self) -> String {
        let s = self.cur().text.clone();
        self.i += 1;
        s
    }

    // -- params --------------------------------------------------------

    /// `param { (";" | ",") param }` where a param is `ident = num`.
    fn params(&mut self) -> Result<Params> {
        let mut out = Params::new();
        while self.cur().kind == TokenKind::Ident {
            let k = self.ident()?;
            self.eat("=")?;
            let v = self.number()?;
            out.insert(k, Value::Num(v));
            if self.at(";") || self.at(",") {
                self.i += 1;
            } else {
                break;
            }
        }
        Ok(out)
    }

    // -- program -------------------------------------------------------

    fn program(&mut self) -> Result<Program> {
        let mut decls: Vec<Decl> = Vec::new();
        let mut frames: Vec<FrameBlock> = Vec::new();

        while self.at("open") || self.at("method") || self.at("require") {
            decls.push(self.decl()?);
        }
        while self.at("under") {
            frames.push(self.frame_block()?);
        }

        let report = if self.at("report") {
            self.eat("report")?;
            self.eat("to")?;
            self.string()?
        } else {
            // Rule 6.11: there is no silent program. A run that produces
            // no report is not a run whose result was negative -- it is
            // a program that was never well-formed.
            return Err(SynopsisError::parse(
                "every program must end with `report to <file>` (Rule 6.11: \
                 programs emit reports)",
                self.line(),
            ));
        };

        if self.cur().kind != TokenKind::Eof {
            return Err(SynopsisError::parse(
                format!("trailing input {:?}", self.cur().text),
                self.line(),
            ));
        }

        Ok(Program { line: 1, decls, frames, report: Some(report) })
    }

    fn decl(&mut self) -> Result<Decl> {
        if self.at("open") {
            let line = self.eat_line("open")?;
            let name = self.ident()?;
            self.eat("=")?;
            let path = self.string()?;
            return Ok(Decl::Open { line, name, path });
        }
        if self.at("method") {
            let line = self.eat_line("method")?;
            let name = self.ident()?;
            self.eat("=")?;
            let spec = self.ident()?;
            self.eat("(")?;
            let params = self.params()?;
            self.eat(")")?;
            return Ok(Decl::MethodDecl { line, name, spec, params });
        }
        // `require` is accepted by the loop in `program()` but has no
        // decl production, exactly as in the reference: reaching here is
        // a parse error, not a silently ignored declaration.
        Err(SynopsisError::parse(
            format!("unexpected declaration {:?}", self.cur().text),
            self.line(),
        ))
    }

    fn frame_block(&mut self) -> Result<FrameBlock> {
        let line = self.eat_line("under")?;
        let name = self.ident()?;
        self.eat("{")?;
        let body = self.block_body(line, "under")?;
        Ok(FrameBlock { line, name, body })
    }

    /// Statements up to the closing brace, which this consumes.
    fn block_body(&mut self, open_line: usize, what: &str) -> Result<Vec<Stmt>> {
        let mut body = Vec::new();
        while !self.at("}") {
            if self.cur().kind == TokenKind::Eof {
                return Err(SynopsisError::parse(
                    format!("unclosed `{what}` block"),
                    open_line,
                ));
            }
            body.push(self.stmt()?);
        }
        self.eat("}")?;
        Ok(body)
    }

    // -- statements ----------------------------------------------------

    fn stmt(&mut self) -> Result<Stmt> {
        if self.at("let") {
            let line = self.eat_line("let")?;
            let name = self.ident()?;
            // The reference has an annotation branch here (`: ident`).
            // It is unreachable -- ':' does not tokenize -- so `ann` is
            // always None. Kept for structural parity; see
            // corpus/ast.json.
            let ann: Option<String> = None;
            self.eat("=")?;
            let expr = self.expr()?;
            return Ok(Stmt::Let { line, name, ann, expr: Box::new(expr) });
        }

        if self.at("bind") {
            let line = self.eat_line("bind")?;
            let value = self.ident()?;
            self.eat(",")?;
            let residue = self.ident()?;
            self.eat("=")?;
            let expr = self.expr()?;
            return Ok(Stmt::Bind { line, value, residue, expr: Box::new(expr) });
        }

        if self.at("relax") {
            let line = self.eat_line("relax")?;
            let target = self.ident()?;
            self.eat("until")?;
            self.eat("quiescent")?;
            self.eat("{")?;
            let params = self.params()?;
            self.eat("}")?;
            return Ok(Stmt::Relax { line, target, params });
        }

        if self.at("sweep") {
            return self.sweep_stmt();
        }
        if self.at("for") {
            return self.for_stmt();
        }

        if self.at("claim") {
            let line = self.eat_line("claim")?;
            let text = self.string()?;
            self.eat("=")?;
            let expr = self.expr()?;
            return Ok(Stmt::Claim { line, text, expr: Box::new(expr) });
        }

        if self.at("record") {
            let line = self.eat_line("record")?;
            let mut names = vec![self.ident()?];
            while self.at(",") {
                self.i += 1;
                names.push(self.ident()?);
            }
            return Ok(Stmt::Record { line, names });
        }

        if self.at("drop") {
            let line = self.eat_line("drop")?;
            let name = self.ident()?;
            return Ok(Stmt::Drop { line, name });
        }

        Err(SynopsisError::parse(
            format!("unexpected statement {:?}", self.cur().text),
            self.line(),
        ))
    }

    /// `sweep x in lo..hi step s { }` or `sweep x in [a, b, c] { }`.
    ///
    /// Both forms fix the trip count at entry (`thm:total`): the bounds
    /// are numeric literals, so no expression evaluated inside the body
    /// can change how many times the body runs.
    fn sweep_stmt(&mut self) -> Result<Stmt> {
        let line = self.eat_line("sweep")?;
        let var = self.ident()?;
        self.eat("in")?;

        let mut lo = 0.0;
        let mut hi = 0.0;
        let mut step = 0.0;
        let mut values: Option<Vec<f64>> = None;

        if self.at("[") {
            self.i += 1;
            let mut vs = vec![self.number()?];
            while self.at(",") {
                self.i += 1;
                vs.push(self.number()?);
            }
            self.eat("]")?;
            values = Some(vs);
        } else {
            lo = self.number()?;
            self.eat("..")?;
            hi = self.number()?;
            self.eat("step")?;
            step = self.number()?;
        }

        self.eat("{")?;
        let body = self.block_body(line, "sweep")?;

        Ok(Stmt::Sweep { line, var, lo, hi, step, values, body })
    }

    /// `for u in items(<src>) [where separation(v) <op> <num>] { }`.
    fn for_stmt(&mut self) -> Result<Stmt> {
        let line = self.eat_line("for")?;
        let var = self.ident()?;
        self.eat("in")?;
        self.eat("items")?;
        self.eat("(")?;
        let src = self.expr()?;
        self.eat(")")?;

        let mut guard: Option<String> = None;
        if self.at("where") {
            self.i += 1;
            // The guard grammar is exactly `separation(ident) <op>
            // <num>`. It is stored as text because nothing consumes it
            // structurally; the report prints it verbatim so a reader
            // can see what was filtered out.
            self.eat("separation")?;
            self.eat("(")?;
            let gv = self.ident()?;
            self.eat(")")?;
            let op = self.take_text();
            let thr = self.number()?;
            guard = Some(format!("separation({gv}) {op} {}", format_num(thr)));
        }

        self.eat("{")?;
        let body = self.block_body(line, "for")?;

        Ok(Stmt::For { line, var, src: Box::new(src), guard, body })
    }

    // -- expressions ---------------------------------------------------

    fn expr(&mut self) -> Result<Expr> {
        let line = self.line();

        if self.at("project") {
            self.i += 1;
            let src = self.expr()?;
            self.eat("by")?;
            let projector = self.take_text();
            self.eat("(")?;
            let args = if projector == "channels" {
                // `channels` takes an encoding NAME, not a numeric
                // parameter -- the one place an argument is an
                // identifier.
                let mut p = Params::new();
                p.insert("enc", Value::Name(self.ident()?));
                p
            } else if !self.at(")") {
                self.params()?
            } else {
                Params::new()
            };
            self.eat(")")?;
            return Ok(Expr::Project {
                line,
                src: Box::new(src),
                projector,
                args,
            });
        }

        if self.at("compare") {
            self.i += 1;
            let left = self.expr()?;
            self.eat("against")?;
            let right = self.expr()?;
            self.eat("by")?;
            let method = self.take_text();
            let mut params = Params::new();
            if self.at("(") {
                self.i += 1;
                // One token of lookahead distinguishes `by m(global)` --
                // a mode name -- from `by m(gap = 2)`, a parameter list.
                let next_is_eq = self
                    .toks
                    .get(self.i + 1)
                    .is_some_and(|t| t.text == "=");
                if self.cur().kind == TokenKind::Ident && !next_is_eq {
                    params.insert("mode", Value::Name(self.ident()?));
                } else if !self.at(")") {
                    params = self.params()?;
                }
                self.eat(")")?;
            }
            return Ok(Expr::Compare {
                line,
                left: Box::new(left),
                right: Box::new(right),
                method,
                params,
            });
        }

        if self.at("detect") {
            self.i += 1;
            let kind = self.take_text();
            self.eat("in")?;
            let src = self.expr()?;
            self.eat("{")?;
            let params = self.params()?;
            self.eat("}")?;
            return Ok(Expr::Detect {
                line,
                kind,
                src: Box::new(src),
                params,
            });
        }

        if self.at("align") {
            self.i += 1;
            self.eat("central")?;
            self.eat("(")?;
            let ca = self.expr()?;
            self.eat(",")?;
            let cb = self.expr()?;
            self.eat(")")?;

            // The response clause is OPTIONAL in the grammar and
            // REQUIRED by the checker (`thm:arity`). That split is
            // deliberate: omitting it is the false-friend mistake, so it
            // must parse in order to be refused with a diagnostic that
            // names what is missing, rather than dying as a syntax error
            // that says nothing.
            let has_response_clause = self.at("response");
            let resp = if has_response_clause {
                self.i += 1;
                self.eat("(")?;
                let ra = self.expr()?;
                self.eat(",")?;
                let rb = self.expr()?;
                self.eat(")")?;
                Some((Box::new(ra), Box::new(rb)))
            } else {
                None
            };

            self.eat("under")?;
            let mut corrs = vec![self.ident()?];
            while self.at(",") {
                self.i += 1;
                corrs.push(self.ident()?);
            }
            self.eat("{")?;
            let params = self.params()?;
            self.eat("}")?;

            return Ok(Expr::Align {
                line,
                central: (Box::new(ca), Box::new(cb)),
                resp,
                corrs,
                params,
                has_response_clause,
            });
        }

        if self.at("unit") {
            self.i += 1;
            let src = self.expr()?;
            self.eat("anchors")?;
            let anchors = self.number()?.trunc() as usize;
            return Ok(Expr::UnitExpr { line, src: Box::new(src), anchors });
        }

        if self.at("response") {
            self.i += 1;
            let src = self.expr()?;
            let method = if self.at("by") {
                self.i += 1;
                Some(self.ident()?)
            } else {
                // None means ANONYMOUS, which Rule 6.7 refuses later.
                None
            };
            return Ok(Expr::ResponseExpr {
                line,
                src: Box::new(src),
                method,
            });
        }

        if self.at("nearest_unit") {
            self.i += 1;
            let net = self.expr()?;
            self.eat("to")?;
            let to = self.expr()?;
            return Ok(Expr::NearestUnit {
                line,
                net: Box::new(net),
                to: Box::new(to),
            });
        }

        if self.at("corr") {
            self.i += 1;
            self.eat("from")?;
            let src = self.expr()?;
            self.eat("to")?;
            let dst = self.expr()?;
            self.eat("by")?;
            let by = self.ident()?;
            return Ok(Expr::CorrExpr {
                line,
                src: Box::new(src),
                dst: Box::new(dst),
                by,
            });
        }

        if self.cur().kind == TokenKind::Ident {
            let name = self.take_text();
            return Ok(Expr::Var { line, name });
        }

        if self.cur().kind == TokenKind::Num {
            let value = self.number()?;
            return Ok(Expr::Num { line, value });
        }

        Err(SynopsisError::parse(
            format!("cannot parse expression at {:?}", self.cur().text),
            line,
        ))
    }
}

/// Render a number the way Python's `f"{float}"` does, so guard strings
/// are byte-identical across implementations. Python prints `0.5` for
/// 0.5 and `2.0` for 2.0; a bare Rust `{}` prints `0.5` and `2`.
fn format_num(x: f64) -> String {
    if x.fract() == 0.0 && x.is_finite() {
        format!("{x:.1}")
    } else {
        format!("{x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = "open a = \"x\"\nunder f { record a }\nreport to \"r\"";

    #[test]
    fn a_minimal_program_parses() {
        let p = parse(MINIMAL).unwrap();
        assert_eq!(p.decls.len(), 1);
        assert_eq!(p.frames.len(), 1);
        assert_eq!(p.report.as_deref(), Some("r"));
    }

    #[test]
    fn a_program_without_a_report_is_refused() {
        // Rule 6.11 -- and it must be a ParseError, because the corpus
        // records `no_report` as one.
        let e = parse("open a = \"x\"\nunder f { record a }").unwrap_err();
        assert_eq!(e.class, crate::errors::ErrorClass::Parse);
        assert!(e.message.contains("report to"));
    }

    #[test]
    fn the_forbidden_forms_do_not_parse() {
        // Not a check that runs afterwards: there is no production for
        // any of these, which is what makes thm:total and thm:noexact
        // properties of the language.
        for src in [
            "open a = \"x\"\nunder f { while x { } }\nreport to \"r\"",
            "open a = \"x\"\nunder f { fix x = y }\nreport to \"r\"",
            "open a = \"x\"\nunder f { recurse f }\nreport to \"r\"",
            "open a = \"x\"\nunder f { let y = a[3] }\nreport to \"r\"",
        ] {
            assert!(parse(src).is_err(), "PARSED a forbidden form: {src}");
        }
    }

    #[test]
    fn align_without_a_response_clause_still_parses() {
        // It must parse in order for the checker to refuse it by name.
        let src = "open a = \"x\"\nunder f {\n  let z = align central(a, a) \
                   under c1 { k = 1 }\n}\nreport to \"r\"";
        let p = parse(src).unwrap();
        let Stmt::Let { expr, .. } = &p.frames[0].body[0] else {
            panic!("expected a let")
        };
        let Expr::Align { has_response_clause, resp, .. } = expr.as_ref() else {
            panic!("expected an align")
        };
        assert!(!has_response_clause);
        assert!(resp.is_none());
    }

    #[test]
    fn guard_text_matches_python_float_formatting() {
        assert_eq!(format_num(2.0), "2.0");
        assert_eq!(format_num(0.5), "0.5");
    }
}
