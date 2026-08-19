//! The `synopsis` command-line tool.
//!
//! Subcommands grow with the stages. At Stage A there are two, and
//! `parse --json` exists for a specific reason: it is how the Rust
//! compiler is held to the same trees as the reference and the
//! TypeScript compiler. Conformance is not a test the compiler runs on
//! itself; it is an external comparison against an oracle, and the
//! compiler's job is to expose its parse in the oracle's shape.

use std::io::Read;
use std::process::ExitCode;

use synopsis::{json, parse};

const USAGE: &str = "\
synopsis -- a language for genomic experiments

USAGE:
    synopsis parse [--json] [FILE]     parse a program; - or no FILE reads stdin
    synopsis --version
    synopsis --help

`parse` exits 0 if the program is well-formed as far as the grammar is
concerned, and 1 with a diagnostic on stderr if it is not.

A program can parse and still be refused. The rules that give the
language its guarantees -- frame separation, the residue rule, required
parameters, the four-column arity -- are checked after parsing, by
`synopsis check`. That subcommand is NOT in this binary yet: the checker
exists in the TypeScript front-end and is held to the same corpus, but
it has no Rust port. Until it does, a program this binary accepts has
been shown to be well-formed, not to be meaningful.
";

fn read_source(path: Option<&str>) -> std::io::Result<String> {
    match path {
        None | Some("-") => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            Ok(s)
        }
        Some(p) => std::fs::read_to_string(p),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();

    match argv.as_slice() {
        [] | ["--help"] | ["-h"] | ["help"] => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        ["--version"] | ["-V"] => {
            println!("synopsis {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        ["parse", rest @ ..] => cmd_parse(rest),
        [other, ..] => {
            eprintln!("synopsis: unknown subcommand {other:?}\n");
            eprint!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_parse(args: &[&str]) -> ExitCode {
    let want_json = args.contains(&"--json");
    let path = args.iter().find(|a| !a.starts_with("--")).copied();

    let src = match read_source(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("synopsis: cannot read {}: {e}", path.unwrap_or("<stdin>"));
            return ExitCode::FAILURE;
        }
    };

    match parse(&src) {
        Ok(program) => {
            if want_json {
                // to_string, not to_string_pretty: the runner compares
                // its own stable stringification, so whitespace here is
                // just bytes on a pipe.
                println!("{}", json::program(&program));
            } else {
                println!(
                    "ok: {} declaration(s), {} frame(s)",
                    program.decls.len(),
                    program.frames.len()
                );
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            if want_json {
                // Machine-readable, so the conformance runner can check
                // the error CLASS and not just that something failed.
                println!(
                    "{}",
                    serde_json::json!({
                        "error": e.class.name(),
                        "message": e.message,
                        "line": e.line,
                    })
                );
            } else {
                eprintln!("{e}");
            }
            ExitCode::FAILURE
        }
    }
}
