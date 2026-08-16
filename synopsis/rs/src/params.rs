//! Parameter blocks.
//!
//! An ordered map, and ordered on purpose: the report prints parameters
//! in source order so that a reader can check them against the program
//! they came from. A `HashMap` would make that ordering depend on the
//! hasher, which would make two runs of the same program produce
//! different reports -- and the report is the deliverable.
//!
//! Small enough that linear scan beats hashing: the largest required
//! set in the language is three entries (`peaks`, `smith_waterman`).

use std::fmt;

/// A parameter value. Numbers are the common case; the two identifier
/// arguments in the grammar (`project ... by channels(dna)` and
/// `compare ... by m(global)`) carry a name instead.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Num(f64),
    Name(String),
}

impl Value {
    pub fn as_num(&self) -> Option<f64> {
        match self {
            Value::Num(x) => Some(*x),
            Value::Name(_) => None,
        }
    }

    pub fn as_name(&self) -> Option<&str> {
        match self {
            Value::Name(s) => Some(s),
            Value::Num(_) => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Rendered the way Python renders a float, so parameter
            // echoes in the report are byte-identical across the three
            // implementations: 2.0 prints as "2.0", not "2".
            Value::Num(x) if x.fract() == 0.0 && x.is_finite() => write!(f, "{x:.1}"),
            Value::Num(x) => write!(f, "{x}"),
            Value::Name(s) => write!(f, "{s}"),
        }
    }
}

/// An insertion-ordered `String -> Value` map.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Params {
    entries: Vec<(String, Value)>,
}

impl Params {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Insert, replacing in place if the key is already present so that
    /// position is stable.
    pub fn insert(&mut self, key: impl Into<String>, value: Value) {
        let key = key.into();
        if let Some(slot) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            self.entries.push((key, value));
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn num(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(Value::as_num)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(k, _)| k.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl FromIterator<(String, Value)> for Params {
    fn from_iter<T: IntoIterator<Item = (String, Value)>>(iter: T) -> Self {
        let mut p = Params::new();
        for (k, v) in iter {
            p.insert(k, v);
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insertion_order_is_preserved_and_stable_under_replace() {
        let mut p = Params::new();
        p.insert("z", Value::Num(3.0));
        p.insert("min_distance", Value::Num(10.0));
        p.insert("min_score", Value::Num(0.5));
        assert_eq!(p.keys().collect::<Vec<_>>(), ["z", "min_distance", "min_score"]);

        // Replacing must not move the key to the end, or the report
        // would reorder itself depending on how the source was written.
        p.insert("z", Value::Num(4.0));
        assert_eq!(p.keys().collect::<Vec<_>>(), ["z", "min_distance", "min_score"]);
        assert_eq!(p.num("z"), Some(4.0));
    }

    #[test]
    fn integral_numbers_render_like_python_floats() {
        assert_eq!(Value::Num(2.0).to_string(), "2.0");
        assert_eq!(Value::Num(0.5).to_string(), "0.5");
    }
}
