//! Streaming JSON Lines (NDJSON) reader with a pure-Rust JSON parser.
//!
//! This module provides two complementary facilities:
//!
//! 1. **[`JsonLinesReader`]** — lazy, line-by-line reader for NDJSON files.
//!    Each call to `next()` parses exactly one line, keeping memory usage
//!    proportional to the largest individual record.
//!
//! 2. **Custom [`JsonValue`] parser** — a recursive-descent parser for JSON
//!    text that produces a tree of [`JsonValue`] nodes without any external
//!    crate dependency.  Useful when you need to process JSON without pulling
//!    in `serde_json`.
//!
//! Additionally, two utility functions are provided:
//!
//! * [`extract_field`] — dot-path navigation into a [`JsonValue`] tree.
//! * [`flatten_json`] — flatten a nested object/array into a
//!   `HashMap<String, String>` suitable for tabular export.
//!
//! # Examples
//!
//! ```rust,no_run
//! use scirs2_io::streaming_json::{JsonLinesReader, JsonValue, parse_json, extract_field, flatten_json};
//!
//! // Parse a single JSON document.
//! let val = parse_json(r#"{"name":"Alice","scores":[9,8,10]}"#).unwrap();
//!
//! // Navigate with dot-path syntax.
//! let name = extract_field(&val, "name");
//! assert!(matches!(name, Some(JsonValue::String(s)) if s == "Alice"));
//!
//! // Flatten to a string map.
//! let flat = flatten_json(&val, "");
//! assert_eq!(flat["name"], "Alice");
//! assert_eq!(flat["scores.0"], "9");
//!
//! // Stream an NDJSON file.
//! let mut reader = JsonLinesReader::open("data.ndjson").unwrap();
//! for result in &mut reader {
//!     let record = result.unwrap();
//!     println!("{:?}", record);
//! }
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::{IoError, Result};

// ─────────────────────────────── JsonValue ───────────────────────────────────

/// A dynamically-typed JSON value produced by the pure-Rust parser.
///
/// Mirrors the standard JSON specification (RFC 8259):
/// - `null`        → [`JsonValue::Null`]
/// - `true`/`false`→ [`JsonValue::Bool`]
/// - numbers       → [`JsonValue::Number`] (always stored as `f64`)
/// - strings       → [`JsonValue::String`]
/// - arrays        → [`JsonValue::Array`]
/// - objects       → [`JsonValue::Object`] (key order is insertion order)
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    /// JSON `null`
    Null,
    /// JSON boolean (`true` / `false`)
    Bool(bool),
    /// JSON number stored as a 64-bit float.
    Number(f64),
    /// JSON string (unescaped, owned).
    String(String),
    /// JSON array.
    Array(Vec<JsonValue>),
    /// JSON object. Keys are `String`s; insertion order is preserved by using
    /// a `Vec<(String, JsonValue)>` rather than a hash map so that serialised
    /// output is deterministic.
    Object(Vec<(std::string::String, JsonValue)>),
}

impl JsonValue {
    /// Returns the string value if this is a [`JsonValue::String`].
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns the numeric value if this is a [`JsonValue::Number`].
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Returns the boolean value if this is a [`JsonValue::Bool`].
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns `true` if this is a [`JsonValue::Null`].
    pub fn is_null(&self) -> bool {
        matches!(self, JsonValue::Null)
    }

    /// Returns a reference to the array contents, or `None`.
    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Returns a reference to the object key-value pairs, or `None`.
    pub fn as_object(&self) -> Option<&[(std::string::String, JsonValue)]> {
        match self {
            JsonValue::Object(pairs) => Some(pairs.as_slice()),
            _ => None,
        }
    }

    /// Look up a key in an object. Returns `None` for non-objects.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

// ─────────────────────────────── Parser ──────────────────────────────────────

/// Internal parser state holding the source text and the current read position.
struct Parser<'src> {
    src: &'src [u8],
    pos: usize,
}

impl<'src> Parser<'src> {
    fn new(src: &'src str) -> Self {
        Self {
            src: src.as_bytes(),
            pos: 0,
        }
    }

    // ── Primitive helpers ─────────────────────────────────────────────────────

    fn remaining(&self) -> usize {
        self.src.len().saturating_sub(self.pos)
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn consume(&mut self) -> Option<u8> {
        let ch = self.src.get(self.pos).copied()?;
        self.pos += 1;
        Some(ch)
    }

    fn expect_byte(&mut self, expected: u8) -> Result<()> {
        match self.peek() {
            Some(b) if b == expected => {
                self.pos += 1;
                Ok(())
            }
            got => Err(IoError::ParseError(format!(
                "expected '{}' at offset {} but got {:?}",
                expected as char,
                self.pos,
                got.map(|b| b as char)
            ))),
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' || b == b'\r' || b == b'\n' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    // ── Top-level entry point ─────────────────────────────────────────────────

    fn parse_value(&mut self) -> Result<JsonValue> {
        self.skip_whitespace();
        match self
            .peek()
            .ok_or_else(|| IoError::ParseError("unexpected end of input".to_string()))?
        {
            b'n' => self.parse_null(),
            b't' | b'f' => self.parse_bool(),
            b'"' => self.parse_string().map(JsonValue::String),
            b'[' => self.parse_array(),
            b'{' => self.parse_object(),
            b'-' | b'0'..=b'9' => self.parse_number(),
            other => Err(IoError::ParseError(format!(
                "unexpected character '{}' at offset {}",
                other as char, self.pos
            ))),
        }
    }

    // ── Null ──────────────────────────────────────────────────────────────────

    fn parse_null(&mut self) -> Result<JsonValue> {
        self.expect_literal(b"null")?;
        Ok(JsonValue::Null)
    }

    // ── Bool ──────────────────────────────────────────────────────────────────

    fn parse_bool(&mut self) -> Result<JsonValue> {
        match self.peek() {
            Some(b't') => {
                self.expect_literal(b"true")?;
                Ok(JsonValue::Bool(true))
            }
            Some(b'f') => {
                self.expect_literal(b"false")?;
                Ok(JsonValue::Bool(false))
            }
            _ => Err(IoError::ParseError(format!(
                "expected 'true' or 'false' at offset {}",
                self.pos
            ))),
        }
    }

    fn expect_literal(&mut self, lit: &[u8]) -> Result<()> {
        if self.remaining() < lit.len() {
            return Err(IoError::ParseError(format!(
                "expected '{}' at offset {} but input is too short",
                std::str::from_utf8(lit).unwrap_or("<invalid>"),
                self.pos
            )));
        }
        if &self.src[self.pos..self.pos + lit.len()] != lit {
            return Err(IoError::ParseError(format!(
                "expected '{}' at offset {}",
                std::str::from_utf8(lit).unwrap_or("<invalid>"),
                self.pos
            )));
        }
        self.pos += lit.len();
        Ok(())
    }

    // ── Number ────────────────────────────────────────────────────────────────

    fn parse_number(&mut self) -> Result<JsonValue> {
        let start = self.pos;
        // Optional minus sign
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        // Integer part
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
            }
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => {
                return Err(IoError::ParseError(format!(
                    "invalid number at offset {start}"
                )))
            }
        }
        // Optional fractional part
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(IoError::ParseError(format!(
                    "expected digit after '.' at offset {}",
                    self.pos
                )));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        // Optional exponent
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(IoError::ParseError(format!(
                    "expected digit in exponent at offset {}",
                    self.pos
                )));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let num_str = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|e| IoError::ParseError(format!("number UTF-8 error: {e}")))?;
        let n: f64 = num_str
            .parse()
            .map_err(|e| IoError::ParseError(format!("number parse error '{num_str}': {e}")))?;
        Ok(JsonValue::Number(n))
    }

    // ── String ────────────────────────────────────────────────────────────────

    fn parse_string(&mut self) -> Result<std::string::String> {
        self.expect_byte(b'"')?;
        let mut s = std::string::String::new();
        loop {
            let ch = self
                .consume()
                .ok_or_else(|| IoError::ParseError("unterminated string".to_string()))?;
            match ch {
                b'"' => break,
                b'\\' => {
                    let esc = self.consume().ok_or_else(|| {
                        IoError::ParseError("unterminated escape sequence".to_string())
                    })?;
                    match esc {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'b' => s.push('\x08'),
                        b'f' => s.push('\x0C'),
                        b'n' => s.push('\n'),
                        b'r' => s.push('\r'),
                        b't' => s.push('\t'),
                        b'u' => {
                            let code_point = self.parse_unicode_escape()?;
                            s.push(code_point);
                        }
                        other => {
                            return Err(IoError::ParseError(format!(
                                "unknown escape '\\{}' at offset {}",
                                other as char, self.pos
                            )))
                        }
                    }
                }
                b => {
                    // Regular UTF-8 byte — collect multi-byte sequences.
                    let mut buf = vec![b];
                    // Determine how many continuation bytes to expect.
                    let extra = if b & 0b1111_1000 == 0b1111_0000 {
                        3
                    } else if b & 0b1111_0000 == 0b1110_0000 {
                        2
                    } else if b & 0b1110_0000 == 0b1100_0000 {
                        1
                    } else {
                        0
                    };
                    for _ in 0..extra {
                        let cont = self.consume().ok_or_else(|| {
                            IoError::ParseError("truncated UTF-8 sequence in string".to_string())
                        })?;
                        buf.push(cont);
                    }
                    let decoded = std::str::from_utf8(&buf).map_err(|e| {
                        IoError::ParseError(format!("invalid UTF-8 in string: {e}"))
                    })?;
                    s.push_str(decoded);
                }
            }
        }
        Ok(s)
    }

    /// Parse a `\uXXXX` escape sequence and return the char.
    fn parse_unicode_escape(&mut self) -> Result<char> {
        if self.remaining() < 4 {
            return Err(IoError::ParseError(
                "\\u escape requires 4 hex digits".to_string(),
            ));
        }
        let hex_bytes = &self.src[self.pos..self.pos + 4];
        let hex_str = std::str::from_utf8(hex_bytes)
            .map_err(|e| IoError::ParseError(format!("non-UTF8 in \\u escape: {e}")))?;
        let code = u32::from_str_radix(hex_str, 16)
            .map_err(|e| IoError::ParseError(format!("invalid \\u{hex_str}: {e}")))?;
        self.pos += 4;
        char::from_u32(code).ok_or_else(|| {
            IoError::ParseError(format!("code point U+{code:04X} is not a valid char"))
        })
    }

    // ── Array ─────────────────────────────────────────────────────────────────

    fn parse_array(&mut self) -> Result<JsonValue> {
        self.expect_byte(b'[')?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            let val = self.parse_value()?;
            items.push(val);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                other => {
                    return Err(IoError::ParseError(format!(
                        "expected ',' or ']' in array at offset {}, got {:?}",
                        self.pos,
                        other.map(|b| b as char)
                    )))
                }
            }
        }
        Ok(JsonValue::Array(items))
    }

    // ── Object ────────────────────────────────────────────────────────────────

    fn parse_object(&mut self) -> Result<JsonValue> {
        self.expect_byte(b'{')?;
        let mut pairs: Vec<(std::string::String, JsonValue)> = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(pairs));
        }
        loop {
            self.skip_whitespace();
            let key = self.parse_string()?;
            self.skip_whitespace();
            self.expect_byte(b':')?;
            let val = self.parse_value()?;
            pairs.push((key, val));
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                other => {
                    return Err(IoError::ParseError(format!(
                        "expected ',' or '}}' in object at offset {}, got {:?}",
                        self.pos,
                        other.map(|b| b as char)
                    )))
                }
            }
        }
        Ok(JsonValue::Object(pairs))
    }

    // ── Trailer check ─────────────────────────────────────────────────────────

    fn ensure_consumed(&mut self) -> Result<()> {
        self.skip_whitespace();
        if self.pos != self.src.len() {
            Err(IoError::ParseError(format!(
                "trailing garbage at offset {} (len {})",
                self.pos,
                self.src.len()
            )))
        } else {
            Ok(())
        }
    }
}

// ─────────────────────────────── Public parse API ────────────────────────────

/// Parse a JSON text string into a [`JsonValue`] tree.
///
/// Uses a hand-written recursive descent parser — no external crates required.
///
/// # Errors
///
/// Returns [`IoError::ParseError`] for any malformed JSON input.
///
/// # Example
///
/// ```
/// use scirs2_io::streaming_json::{parse_json, JsonValue};
///
/// let v = parse_json(r#"{"x": 1, "y": [2, 3]}"#).unwrap();
/// assert!(matches!(v, JsonValue::Object(_)));
/// ```
pub fn parse_json(s: &str) -> Result<JsonValue> {
    let mut p = Parser::new(s);
    let val = p.parse_value()?;
    p.ensure_consumed()?;
    Ok(val)
}

// ─────────────────────────────── Field extraction ────────────────────────────

/// Navigate a [`JsonValue`] tree using a dot-separated path.
///
/// Path segments are separated by `.`.  Integer segments are used as
/// zero-based array indices when the current node is an array.
///
/// # Examples
///
/// ```
/// use scirs2_io::streaming_json::{parse_json, extract_field, JsonValue};
///
/// let v = parse_json(r#"{"a":{"b":[10,20,30]}}"#).unwrap();
/// let item = extract_field(&v, "a.b.1").unwrap();
/// assert!(matches!(item, JsonValue::Number(n) if (*n - 20.0).abs() < 1e-10));
/// ```
pub fn extract_field<'a>(value: &'a JsonValue, path: &str) -> Option<&'a JsonValue> {
    if path.is_empty() {
        return Some(value);
    }
    let (segment, rest) = match path.find('.') {
        Some(dot) => (&path[..dot], &path[dot + 1..]),
        None => (path, ""),
    };
    let child = match value {
        JsonValue::Object(pairs) => pairs.iter().find(|(k, _)| k == segment).map(|(_, v)| v)?,
        JsonValue::Array(items) => {
            let idx: usize = segment.parse().ok()?;
            items.get(idx)?
        }
        _ => return None,
    };
    if rest.is_empty() {
        Some(child)
    } else {
        extract_field(child, rest)
    }
}

// ─────────────────────────────── Flattening ──────────────────────────────────

/// Flatten a nested [`JsonValue`] into a `HashMap<String, String>`.
///
/// Nested objects and arrays are represented with dotted key paths.
/// Leaf values are converted to their string representation:
/// - `null` → `"null"`
/// - `Bool(b)` → `"true"` / `"false"`
/// - `Number(n)` → formatted with up to 15 significant digits
/// - `String(s)` → `s` (raw, without JSON quotes)
///
/// The `prefix` argument is prepended to every key (use `""` for the root).
///
/// # Example
///
/// ```
/// use scirs2_io::streaming_json::{parse_json, flatten_json};
///
/// let v = parse_json(r#"{"user":{"name":"Alice","age":30},"tags":["a","b"]}"#).unwrap();
/// let flat = flatten_json(&v, "");
/// assert_eq!(flat["user.name"], "Alice");
/// assert_eq!(flat["user.age"],  "30");
/// assert_eq!(flat["tags.0"],    "a");
/// assert_eq!(flat["tags.1"],    "b");
/// ```
pub fn flatten_json(
    value: &JsonValue,
    prefix: &str,
) -> HashMap<std::string::String, std::string::String> {
    let mut map = HashMap::new();
    flatten_json_into(value, prefix, &mut map);
    map
}

fn flatten_json_into(
    value: &JsonValue,
    prefix: &str,
    map: &mut HashMap<std::string::String, std::string::String>,
) {
    match value {
        JsonValue::Null => {
            map.insert(prefix.to_string(), "null".to_string());
        }
        JsonValue::Bool(b) => {
            map.insert(prefix.to_string(), b.to_string());
        }
        JsonValue::Number(n) => {
            // Format with up to 15 significant digits; strip unnecessary trailing zeros.
            let s = format!("{n:.15}");
            let trimmed = s.trim_end_matches('0').trim_end_matches('.');
            map.insert(prefix.to_string(), trimmed.to_string());
        }
        JsonValue::String(s) => {
            map.insert(prefix.to_string(), s.clone());
        }
        JsonValue::Array(items) => {
            for (idx, item) in items.iter().enumerate() {
                let child_key = if prefix.is_empty() {
                    idx.to_string()
                } else {
                    format!("{prefix}.{idx}")
                };
                flatten_json_into(item, &child_key, map);
            }
        }
        JsonValue::Object(pairs) => {
            for (key, child) in pairs {
                let child_key = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten_json_into(child, &child_key, map);
            }
        }
    }
}

// ─────────────────────────────── JsonLinesReader ─────────────────────────────

/// Streaming reader for JSON Lines (NDJSON) files.
///
/// Each line in the file is parsed as a complete JSON document using the
/// pure-Rust [`parse_json`] parser.  Blank lines and lines starting with
/// `#` are silently skipped.
///
/// Implements [`Iterator`]`<Item = `[`Result`]`<`[`JsonValue`]`>>` so it can
/// be used directly in `for` loops and iterator chains.
pub struct JsonLinesReader {
    inner: BufReader<File>,
    line_number: u64,
    finished: bool,
}

impl JsonLinesReader {
    /// Open `path` as a streaming JSON Lines reader.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::FileNotFound`] if the file does not exist.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| IoError::FileNotFound(format!("{}: {e}", path.display())))?;
        Ok(Self {
            inner: BufReader::new(file),
            line_number: 0,
            finished: false,
        })
    }

    /// Parse the next non-blank, non-comment line.
    ///
    /// Returns `Ok(None)` at end-of-file.
    pub fn next_record(&mut self) -> Result<Option<JsonValue>> {
        if self.finished {
            return Ok(None);
        }
        let mut line = std::string::String::new();
        loop {
            line.clear();
            let n = self
                .inner
                .read_line(&mut line)
                .map_err(|e| IoError::FileError(format!("line {}: {e}", self.line_number + 1)))?;
            if n == 0 {
                self.finished = true;
                return Ok(None);
            }
            self.line_number += 1;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let val = parse_json(trimmed)
                .map_err(|e| IoError::ParseError(format!("line {}: {e}", self.line_number)))?;
            return Ok(Some(val));
        }
    }

    /// Collect all remaining records into a `Vec`.
    pub fn collect_all(&mut self) -> Result<Vec<JsonValue>> {
        let mut out = Vec::new();
        while let Some(v) = self.next_record()? {
            out.push(v);
        }
        Ok(out)
    }

    /// Return the 1-based number of lines consumed (including blank/comment lines).
    pub fn line_number(&self) -> u64 {
        self.line_number
    }
}

impl Iterator for JsonLinesReader {
    type Item = Result<JsonValue>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_record() {
            Ok(Some(v)) => Some(Ok(v)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(name: &str, content: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("scirs2_streaming_json_tests");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let p = dir.join(name);
        let mut f = File::create(&p).expect("create");
        f.write_all(content.as_bytes()).expect("write");
        p
    }

    // ── parse_json: primitives ────────────────────────────────────────────────

    #[test]
    fn test_parse_null() {
        assert_eq!(parse_json("null").expect("parse"), JsonValue::Null);
    }

    #[test]
    fn test_parse_bool_true() {
        assert_eq!(parse_json("true").expect("parse"), JsonValue::Bool(true));
    }

    #[test]
    fn test_parse_bool_false() {
        assert_eq!(parse_json("false").expect("parse"), JsonValue::Bool(false));
    }

    #[test]
    fn test_parse_integer() {
        let v = parse_json("42").expect("parse");
        assert!(matches!(v, JsonValue::Number(n) if (n - 42.0).abs() < 1e-10));
    }

    #[test]
    fn test_parse_negative_number() {
        let v = parse_json("-7").expect("parse");
        assert!(matches!(v, JsonValue::Number(n) if (n - (-7.0)).abs() < 1e-10));
    }

    #[test]
    fn test_parse_float() {
        let v = parse_json("1.23456").expect("parse");
        assert!(matches!(v, JsonValue::Number(n) if (n - 1.23456).abs() < 1e-5));
    }

    #[test]
    fn test_parse_exponent() {
        let v = parse_json("1.5e3").expect("parse");
        assert!(matches!(v, JsonValue::Number(n) if (n - 1500.0).abs() < 1e-10));
    }

    #[test]
    fn test_parse_string_simple() {
        let v = parse_json(r#""hello""#).expect("parse");
        assert!(matches!(v, JsonValue::String(ref s) if s == "hello"));
    }

    #[test]
    fn test_parse_string_escape_sequences() {
        let v = parse_json(r#""line1\nline2\ttab""#).expect("parse");
        assert!(matches!(v, JsonValue::String(ref s) if s == "line1\nline2\ttab"));
    }

    #[test]
    fn test_parse_string_unicode_escape() {
        let v = parse_json(r#""\u0041""#).expect("parse"); // 'A'
        assert!(matches!(v, JsonValue::String(ref s) if s == "A"));
    }

    #[test]
    fn test_parse_empty_string() {
        let v = parse_json(r#""""#).expect("parse");
        assert!(matches!(v, JsonValue::String(ref s) if s.is_empty()));
    }

    // ── parse_json: arrays ────────────────────────────────────────────────────

    #[test]
    fn test_parse_empty_array() {
        let v = parse_json("[]").expect("parse");
        assert!(matches!(v, JsonValue::Array(ref a) if a.is_empty()));
    }

    #[test]
    fn test_parse_array_of_numbers() {
        let v = parse_json("[1, 2, 3]").expect("parse");
        if let JsonValue::Array(a) = v {
            assert_eq!(a.len(), 3);
            assert!(matches!(a[0], JsonValue::Number(n) if (n - 1.0).abs() < 1e-10));
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn test_parse_nested_array() {
        let v = parse_json("[[1, 2], [3, 4]]").expect("parse");
        if let JsonValue::Array(outer) = v {
            assert_eq!(outer.len(), 2);
            assert!(matches!(&outer[0], JsonValue::Array(inner) if inner.len() == 2));
        } else {
            panic!("expected array");
        }
    }

    // ── parse_json: objects ───────────────────────────────────────────────────

    #[test]
    fn test_parse_empty_object() {
        let v = parse_json("{}").expect("parse");
        assert!(matches!(v, JsonValue::Object(ref p) if p.is_empty()));
    }

    #[test]
    fn test_parse_simple_object() {
        let v = parse_json(r#"{"name":"Alice","age":30}"#).expect("parse");
        let name = v.get("name").expect("name");
        assert!(matches!(name, JsonValue::String(s) if s == "Alice"));
        let age = v.get("age").expect("age");
        assert!(matches!(age, JsonValue::Number(n) if (n - 30.0).abs() < 1e-10));
    }

    #[test]
    fn test_parse_nested_object() {
        let v = parse_json(r#"{"a":{"b":{"c":99}}}"#).expect("parse");
        let leaf = v.get("a").and_then(|a| a.get("b")).and_then(|b| b.get("c"));
        assert!(matches!(leaf, Some(JsonValue::Number(n)) if (n - 99.0).abs() < 1e-10));
    }

    #[test]
    fn test_parse_object_with_array_value() {
        let v = parse_json(r#"{"items":[1,2,3]}"#).expect("parse");
        let items = v.get("items").expect("items");
        assert!(matches!(items, JsonValue::Array(a) if a.len() == 3));
    }

    // ── parse_json: error cases ───────────────────────────────────────────────

    #[test]
    fn test_parse_trailing_garbage_is_error() {
        assert!(parse_json("42 abc").is_err());
    }

    #[test]
    fn test_parse_unterminated_string_is_error() {
        assert!(parse_json(r#""hello"#).is_err());
    }

    #[test]
    fn test_parse_missing_comma_in_array_is_error() {
        assert!(parse_json("[1 2]").is_err());
    }

    #[test]
    fn test_parse_empty_input_is_error() {
        assert!(parse_json("").is_err());
    }

    // ── extract_field ─────────────────────────────────────────────────────────

    #[test]
    fn test_extract_top_level_field() {
        let v = parse_json(r#"{"x":10}"#).expect("parse");
        let f = extract_field(&v, "x").expect("field");
        assert!(matches!(f, JsonValue::Number(n) if (n - 10.0).abs() < 1e-10));
    }

    #[test]
    fn test_extract_nested_field() {
        let v = parse_json(r#"{"a":{"b":{"c":"deep"}}}"#).expect("parse");
        let f = extract_field(&v, "a.b.c").expect("field");
        assert!(matches!(f, JsonValue::String(s) if s == "deep"));
    }

    #[test]
    fn test_extract_array_index() {
        let v = parse_json(r#"{"arr":[10,20,30]}"#).expect("parse");
        let f = extract_field(&v, "arr.2").expect("field");
        assert!(matches!(f, JsonValue::Number(n) if (n - 30.0).abs() < 1e-10));
    }

    #[test]
    fn test_extract_missing_field_returns_none() {
        let v = parse_json(r#"{"x":1}"#).expect("parse");
        assert!(extract_field(&v, "y").is_none());
    }

    #[test]
    fn test_extract_empty_path_returns_self() {
        let v = parse_json("42").expect("parse");
        let f = extract_field(&v, "").expect("self");
        assert!(matches!(f, JsonValue::Number(n) if (n - 42.0).abs() < 1e-10));
    }

    // ── flatten_json ──────────────────────────────────────────────────────────

    #[test]
    fn test_flatten_flat_object() {
        let v = parse_json(r#"{"a":"x","b":1}"#).expect("parse");
        let flat = flatten_json(&v, "");
        assert_eq!(flat["a"], "x");
        assert_eq!(flat["b"], "1");
    }

    #[test]
    fn test_flatten_nested_object() {
        let v = parse_json(r#"{"user":{"name":"Bob","age":25}}"#).expect("parse");
        let flat = flatten_json(&v, "");
        assert_eq!(flat["user.name"], "Bob");
        assert_eq!(flat["user.age"], "25");
    }

    #[test]
    fn test_flatten_array() {
        let v = parse_json(r#"{"tags":["a","b","c"]}"#).expect("parse");
        let flat = flatten_json(&v, "");
        assert_eq!(flat["tags.0"], "a");
        assert_eq!(flat["tags.1"], "b");
        assert_eq!(flat["tags.2"], "c");
    }

    #[test]
    fn test_flatten_with_prefix() {
        let v = parse_json(r#"{"x":1}"#).expect("parse");
        let flat = flatten_json(&v, "root");
        assert_eq!(flat["root.x"], "1");
    }

    #[test]
    fn test_flatten_null_and_bool() {
        let v = parse_json(r#"{"active":true,"data":null}"#).expect("parse");
        let flat = flatten_json(&v, "");
        assert_eq!(flat["active"], "true");
        assert_eq!(flat["data"], "null");
    }

    // ── JsonLinesReader ───────────────────────────────────────────────────────

    #[test]
    fn test_jsonlines_reader_basic() {
        let content = "{\"id\":1}\n{\"id\":2}\n{\"id\":3}\n";
        let path = write_temp("basic.ndjson", content);
        let mut r = JsonLinesReader::open(&path).expect("open");
        let all = r.collect_all().expect("collect");
        assert_eq!(all.len(), 3);
        assert!(matches!(all[2].get("id"), Some(JsonValue::Number(n)) if (n - 3.0).abs() < 1e-10));
    }

    #[test]
    fn test_jsonlines_reader_skips_blank_and_comment() {
        let content = "\n# comment\n{\"v\":1}\n\n{\"v\":2}\n";
        let path = write_temp("comments.ndjson", content);
        let mut r = JsonLinesReader::open(&path).expect("open");
        let all = r.collect_all().expect("collect");
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_jsonlines_reader_iterator_interface() {
        let content = "{\"x\":10}\n{\"x\":20}\n";
        let path = write_temp("iter.ndjson", content);
        let r = JsonLinesReader::open(&path).expect("open");
        let vals: Vec<_> = r.map(|res| res.expect("ok")).collect();
        assert_eq!(vals.len(), 2);
        assert!(matches!(vals[0].get("x"), Some(JsonValue::Number(n)) if (n - 10.0).abs() < 1e-10));
    }

    #[test]
    fn test_jsonlines_reader_empty_file() {
        let path = write_temp("empty.ndjson", "");
        let mut r = JsonLinesReader::open(&path).expect("open");
        assert!(r.next_record().expect("ok").is_none());
    }

    #[test]
    fn test_jsonlines_reader_line_number_tracking() {
        let content = "\n{\"a\":1}\n{\"a\":2}\n";
        let path = write_temp("lineno.ndjson", content);
        let mut r = JsonLinesReader::open(&path).expect("open");
        r.next_record().expect("ok").expect("some");
        // Line 1 was blank, line 2 was the first record.
        assert!(r.line_number() >= 2);
    }

    #[test]
    fn test_jsonlines_reader_parse_error_propagates() {
        let content = "{\"ok\":1}\n{bad json}\n";
        let path = write_temp("bad.ndjson", content);
        let mut r = JsonLinesReader::open(&path).expect("open");
        let first = r.next_record().expect("first ok").expect("some");
        assert!(first.get("ok").is_some());
        assert!(r.next_record().is_err());
    }

    // ── JsonValue helper methods ──────────────────────────────────────────────

    #[test]
    fn test_jsonvalue_accessors() {
        assert_eq!(JsonValue::Null.as_str(), None);
        assert_eq!(JsonValue::String("hi".to_string()).as_str(), Some("hi"));
        assert!((JsonValue::Number(2.5).as_f64().expect("f64") - 2.5).abs() < 1e-10);
        assert_eq!(JsonValue::Bool(true).as_bool(), Some(true));
        assert!(JsonValue::Null.is_null());
        let arr = JsonValue::Array(vec![JsonValue::Null]);
        assert_eq!(arr.as_array().expect("arr").len(), 1);
    }
}
