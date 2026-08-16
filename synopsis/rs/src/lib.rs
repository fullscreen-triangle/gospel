//! synopsis -- a language for genomic experiments.
//!
//! This crate is one of two implementations of a single language. The
//! other is `synopsis/ts`, in TypeScript, for the web tool. They are not
//! independent programs that happen to look alike: the language's
//! properties are proved of the LANGUAGE, so if the two compilers
//! disagree about whether a program is well-formed, at least one of them
//! has stopped implementing it. `synopsis/corpus` is what holds them
//! together -- both are checked against the same trees and the same
//! refusals, extracted from the reference semantics.
//!
//! Reading order: `tokens` -> `parser` -> `ast`. The type checker,
//! evaluator and report follow in later stages.

pub mod ast;
pub mod errors;
pub mod json;
pub mod params;
pub mod parser;
pub mod tokens;

pub use errors::{ErrorClass, Result, SynopsisError};
pub use parser::parse;
