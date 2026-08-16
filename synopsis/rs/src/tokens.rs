//! The synopsis tokenizer.
//!
//! A direct port of `validation/lang.py:77-131` and of
//! `synopsis/ts/src/tokens.ts`. Three details of the reference are
//! deliberate and reproduced exactly, because the conformance corpus
//! depends on them:
//!
//!  1. `..` is matched BEFORE a number, so `1..5` is `[1, .., 5]` and
//!     not the number `1.` followed by `.5`.
//!  2. A number may carry a leading `-`, but `-` is NOT in the
//!     punctuation class. So `1..-5` tokenizes and `a - b` does not:
//!     there is no subtraction operator in the language. Arithmetic on
//!     results is not something a synopsis program does.
//!  3. A string's value is the raw slice between the quotes. Escapes are
//!     matched but NOT decoded, so `"a\"b"` has the value `a\"b`. Paths
//!     are the only strings the language has, and decoding would
//!     silently change one.
//!
//! Written by hand rather than with a regex crate. The reference is a
//! single alternation whose ORDER is the specification; a hand-written
//! scanner makes that order visible in the control flow, and it keeps
//! the dependency list at two crates.

use crate::errors::{Result, SynopsisError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Kw,
    Ident,
    Num,
    Str,
    Punct,
    Range,
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    /// The exact source slice.
    pub text: String,
    pub line: usize,
    /// Decoded payload for `Num`; `None` otherwise. Strings carry their
    /// unquoted text in `value_str`.
    pub num: Option<f64>,
    /// Unquoted (but undecoded) text for `Str`; `None` otherwise.
    pub value_str: Option<String>,
}

impl Token {
    fn plain(kind: TokenKind, text: &str, line: usize) -> Self {
        Token { kind, text: text.to_string(), line, num: None, value_str: None }
    }
}

/// The 39 reserved words.
///
/// Kept in sync with `corpus.json:keywords` by
/// `synopsis/corpus/check_keywords.py`, which fails if this list and the
/// extracted one disagree. Notably absent, and absent on purpose:
/// `while`, `fix`, `recurse`, `if`, `else`. Unbounded iteration and
/// conditionals are excluded by the grammar's shape rather than by a
/// later check -- that is what makes `thm:total` a property of the
/// language and not of the checker.
// SYNOPSIS-KEYWORDS-BEGIN
pub const KEYWORDS: [&str; 39] = [
    "against", "align", "anchors", "and", "by", "central", "claim",
    "compare", "corr", "detect", "drop", "for", "from", "in", "items",
    "join", "let", "method", "nearest_unit", "open", "peaks", "perturb",
    "project", "quiescent", "record", "relax", "report", "require",
    "response", "separation", "step", "sweep", "to", "top", "under",
    "unit", "until", "where", "bind",
];
// SYNOPSIS-KEYWORDS-END

pub fn is_keyword(s: &str) -> bool {
    KEYWORDS.contains(&s)
}

/// The punctuation class, exactly `[{}()\[\],;=<>+*/]` from the
/// reference. `-` is absent; see note 2.
const PUNCT: [char; 14] =
    ['{', '}', '(', ')', '[', ']', ',', ';', '=', '<', '>', '+', '*', '/'];

fn is_punct(c: char) -> bool {
    PUNCT.contains(&c)
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

pub fn tokenise(src: &str) -> Result<Vec<Token>> {
    let b: Vec<char> = src.chars().collect();
    let n = b.len();
    let mut toks: Vec<Token> = Vec::new();
    let mut pos = 0usize;
    let mut line = 1usize;

    while pos < n {
        let c = b[pos];

        // 1. whitespace
        if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
            if c == '\n' {
                line += 1;
            }
            pos += 1;
            continue;
        }

        // 2. comment to end of line (the newline itself is left for the
        //    whitespace arm, so the line counter stays correct)
        if c == '#' {
            while pos < n && b[pos] != '\n' {
                pos += 1;
            }
            continue;
        }

        // 3. string
        if c == '"' {
            let start = pos;
            let mut i = pos + 1;
            let mut closed = false;
            while i < n {
                match b[i] {
                    '\\' if i + 1 < n => i += 2, // matched, not decoded
                    '"' => {
                        closed = true;
                        i += 1;
                        break;
                    }
                    _ => i += 1,
                }
            }
            if !closed {
                // The reference's alternation simply fails to match an
                // unterminated string, and the loop then reports the
                // offending character -- which is the opening quote.
                return Err(SynopsisError::parse(
                    format!("unexpected character {:?}", '"'),
                    line,
                ));
            }
            let text: String = b[start..i].iter().collect();
            let value: String = b[start + 1..i - 1].iter().collect();
            toks.push(Token {
                kind: TokenKind::Str,
                text,
                line,
                num: None,
                value_str: Some(value),
            });
            pos = i;
            continue;
        }

        // 4. `..` -- BEFORE the number arm; see note 1.
        if c == '.' && pos + 1 < n && b[pos + 1] == '.' {
            toks.push(Token::plain(TokenKind::Range, "..", line));
            pos += 2;
            continue;
        }

        // 5. number, optionally signed. A lone `-` is not a token, so a
        //    `-` that is not followed by a digit falls through to the
        //    error at the bottom of the loop.
        if c.is_ascii_digit() || (c == '-' && pos + 1 < n && b[pos + 1].is_ascii_digit()) {
            let start = pos;
            if c == '-' {
                pos += 1;
            }
            while pos < n && b[pos].is_ascii_digit() {
                pos += 1;
            }
            // `\d+\.\d+` only: a trailing `.` with no digit after it is
            // not part of the number, which is what lets `1..5` split.
            if pos + 1 < n && b[pos] == '.' && b[pos + 1].is_ascii_digit() {
                pos += 1;
                while pos < n && b[pos].is_ascii_digit() {
                    pos += 1;
                }
            }
            let text: String = b[start..pos].iter().collect();
            let value: f64 = text.parse().map_err(|_| {
                SynopsisError::parse(format!("malformed number {text:?}"), line)
            })?;
            toks.push(Token {
                kind: TokenKind::Num,
                text,
                line,
                num: Some(value),
                value_str: None,
            });
            continue;
        }

        // 6. identifier or keyword
        if is_ident_start(c) {
            let start = pos;
            while pos < n && is_ident_continue(b[pos]) {
                pos += 1;
            }
            let text: String = b[start..pos].iter().collect();
            let kind = if is_keyword(&text) { TokenKind::Kw } else { TokenKind::Ident };
            toks.push(Token::plain(kind, &text, line));
            continue;
        }

        // 7. punctuation
        if is_punct(c) {
            toks.push(Token::plain(TokenKind::Punct, &c.to_string(), line));
            pos += 1;
            continue;
        }

        return Err(SynopsisError::parse(
            format!("unexpected character {c:?}"),
            line,
        ));
    }

    toks.push(Token::plain(TokenKind::Eof, "", line));
    Ok(toks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<(TokenKind, String)> {
        tokenise(src)
            .unwrap()
            .into_iter()
            .filter(|t| t.kind != TokenKind::Eof)
            .map(|t| (t.kind, t.text))
            .collect()
    }

    #[test]
    fn range_wins_over_number() {
        // Note 1: `1..5` must be three tokens, not `1.` and `.5`.
        assert_eq!(
            kinds("1..5"),
            vec![
                (TokenKind::Num, "1".into()),
                (TokenKind::Range, "..".into()),
                (TokenKind::Num, "5".into()),
            ]
        );
    }

    #[test]
    fn negative_bound_tokenises_but_minus_is_not_an_operator() {
        // Note 2: `1..-5` is fine ...
        assert_eq!(kinds("1..-5").len(), 3);
        // ... and `a - b` is not, because `-` alone is not a token.
        let e = tokenise("a - b").unwrap_err();
        assert!(e.message.contains("unexpected character"));
    }

    #[test]
    fn string_value_is_raw() {
        // Note 3: escapes are matched but not decoded.
        let t = tokenise(r#""a\"b""#).unwrap();
        assert_eq!(t[0].value_str.as_deref(), Some(r#"a\"b"#));
    }

    #[test]
    fn comments_are_dropped_and_lines_still_count() {
        let t = tokenise("open # a comment\nunder").unwrap();
        assert_eq!(t[0].text, "open");
        assert_eq!(t[0].line, 1);
        assert_eq!(t[1].text, "under");
        assert_eq!(t[1].line, 2);
    }

    #[test]
    fn keywords_are_distinguished_from_identifiers() {
        let t = tokenise("open a").unwrap();
        assert_eq!(t[0].kind, TokenKind::Kw);
        assert_eq!(t[1].kind, TokenKind::Ident);
    }

    #[test]
    fn forbidden_words_are_plain_identifiers_not_keywords() {
        // The grammar excludes iteration by not having the words; the
        // tokenizer must therefore treat them as ordinary names, and it
        // is the parser that has nowhere to put them.
        for w in ["while", "fix", "recurse", "if", "else"] {
            assert!(!is_keyword(w), "{w} must not be a keyword");
        }
    }
}
