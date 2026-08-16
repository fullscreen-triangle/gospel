//! The synopsis error hierarchy.
//!
//! Mirrors `validation/lang.py:30-70` and `synopsis/ts/src/errors.ts`.
//! Rust has no inheritance, and the hierarchy is load-bearing -- the
//! conformance corpus records the most specific class the reference
//! raises, and a compiler may report a strict subclass but never a
//! superclass. So the relation is encoded as data (`parents`) rather
//! than as types, and `is_subclass_of` is the check the runner uses.

use std::fmt;

/// Every refusal the language can make, tagged with its class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorClass {
    Parse,
    Type,
    Arity,
    Scope,
    Residue,
    Parameter,
    Termination,
    /// The base class, raised directly for refusals that belong to no
    /// narrower category (e.g. `sequence_indexing`, `thm:noexact`).
    Synopsis,
}

impl ErrorClass {
    /// The name the conformance corpus compares against.
    pub const fn name(self) -> &'static str {
        match self {
            ErrorClass::Parse => "ParseError",
            ErrorClass::Type => "TypeError",
            ErrorClass::Arity => "ArityError",
            ErrorClass::Scope => "ScopeError",
            ErrorClass::Residue => "ResidueError",
            ErrorClass::Parameter => "ParameterError",
            ErrorClass::Termination => "TerminationError",
            ErrorClass::Synopsis => "SynopsisError",
        }
    }

    /// Human-facing category, as printed in diagnostics.
    pub const fn kind(self) -> &'static str {
        match self {
            ErrorClass::Parse => "parse error",
            ErrorClass::Type => "type error",
            ErrorClass::Arity => "arity error",
            ErrorClass::Scope => "scope error",
            ErrorClass::Residue => "residue error",
            ErrorClass::Parameter => "parameter error",
            ErrorClass::Termination => "termination error",
            ErrorClass::Synopsis => "error",
        }
    }

    /// Ancestors, nearest first. Matches `corpus.json:error_hierarchy`.
    pub const fn parents(self) -> &'static [ErrorClass] {
        match self {
            ErrorClass::Synopsis => &[],
            ErrorClass::Parse | ErrorClass::Type => &[ErrorClass::Synopsis],
            ErrorClass::Arity
            | ErrorClass::Scope
            | ErrorClass::Residue
            | ErrorClass::Parameter
            | ErrorClass::Termination => &[ErrorClass::Type, ErrorClass::Synopsis],
        }
    }

    /// True when `self` is `other` or a descendant of it.
    pub fn is_subclass_of(self, other: ErrorClass) -> bool {
        self == other || self.parents().contains(&other)
    }

    pub fn from_name(s: &str) -> Option<ErrorClass> {
        Some(match s {
            "ParseError" => ErrorClass::Parse,
            "TypeError" => ErrorClass::Type,
            "ArityError" => ErrorClass::Arity,
            "ScopeError" => ErrorClass::Scope,
            "ResidueError" => ErrorClass::Residue,
            "ParameterError" => ErrorClass::Parameter,
            "TerminationError" => ErrorClass::Termination,
            "SynopsisError" => ErrorClass::Synopsis,
            _ => return None,
        })
    }
}

/// A refusal, with the source position that caused it where one exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynopsisError {
    pub class: ErrorClass,
    pub message: String,
    /// 1-based source line; `None` when the error is not positional.
    pub line: Option<usize>,
}

impl SynopsisError {
    pub fn new(class: ErrorClass, message: impl Into<String>, line: Option<usize>) -> Self {
        Self { class, message: message.into(), line }
    }

    pub fn parse(message: impl Into<String>, line: usize) -> Self {
        Self::new(ErrorClass::Parse, message, Some(line))
    }
}

impl fmt::Display for SynopsisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.line {
            Some(l) => write!(f, "{} (line {}): {}", self.class.name(), l, self.message),
            None => write!(f, "{}: {}", self.class.name(), self.message),
        }
    }
}

impl std::error::Error for SynopsisError {}

pub type Result<T> = std::result::Result<T, SynopsisError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subclassing_matches_the_corpus() {
        assert!(ErrorClass::Arity.is_subclass_of(ErrorClass::Type));
        assert!(ErrorClass::Arity.is_subclass_of(ErrorClass::Synopsis));
        assert!(ErrorClass::Arity.is_subclass_of(ErrorClass::Arity));
        // A ParseError is NOT a TypeError; the corpus distinguishes
        // `no_report` (parse) from the ten type-level refusals.
        assert!(!ErrorClass::Parse.is_subclass_of(ErrorClass::Type));
        // And the relation does not run upward.
        assert!(!ErrorClass::Type.is_subclass_of(ErrorClass::Arity));
    }
}
