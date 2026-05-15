//! Streaming CSV reader with schema inference and typed access.
//!
//! Provides memory-efficient, lazy iteration over CSV files along with
//! schema inference, typed row parsing, and batch (chunk) reading.
//! The reader never loads the entire file into memory; each row is decoded
//! on demand.
//!
//! # Design overview
//!
//! * [`CsvStreamReader`] — core iterator that yields `Result<Vec<String>>`.
//! * [`ColumnType`] — enum representing an inferred column type.
//! * [`TypedRow`] — a parsed row where every field has been coerced to
//!   its inferred type.
//! * [`TypedValue`] — the per-field variant produced by typed parsing.
//! * [`infer_schema`] — scans the first `n_rows` data rows and returns a
//!   `Vec<ColumnType>` describing the file.
//!
//! # Examples
//!
//! ```rust,no_run
//! use scirs2_io::streaming_csv::{CsvStreamReader, infer_schema, ColumnType};
//!
//! // Lazy iterator — only one row is in memory at a time.
//! let mut reader = CsvStreamReader::new("data.csv", b',', true).unwrap();
//! for result in &mut reader {
//!     let row = result.unwrap();
//!     println!("{:?}", row);
//! }
//!
//! // Schema inference
//! let schema = infer_schema("data.csv", b',', 100).unwrap();
//! for (i, col) in schema.iter().enumerate() {
//!     println!("column {i}: {col:?}");
//! }
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::{IoError, Result};

// ─────────────────────────────── ColumnType ──────────────────────────────────

/// The inferred type of a CSV column.
///
/// Type inference uses the following priority order during schema detection:
/// - If every sampled value parses as `i64` → `Integer`
/// - Else if every value parses as `f64` → `Float`
/// - Else if every value is a recognised boolean literal → `Boolean`
/// - Otherwise → `Text`
///
/// Empty / null cells are skipped when determining the dominant type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnType {
    /// 64-bit signed integer column.
    Integer,
    /// 64-bit floating-point column.
    Float,
    /// Boolean column (`true`/`false`/`yes`/`no`/`1`/`0`).
    Boolean,
    /// Free-form text column (fallback).
    Text,
}

// ─────────────────────────────── TypedValue ──────────────────────────────────

/// A single parsed cell value produced by [`read_typed_row`].
#[derive(Debug, Clone, PartialEq)]
pub enum TypedValue {
    /// Parsed 64-bit integer.
    Integer(i64),
    /// Parsed 64-bit float.
    Float(f64),
    /// Parsed boolean.
    Boolean(bool),
    /// Raw text that could not (or should not) be parsed further.
    Text(String),
    /// Empty or explicit null field.
    Null,
}

// ─────────────────────────────── TypedRow ────────────────────────────────────

/// A fully typed row — one [`TypedValue`] per column.
pub type TypedRow = Vec<TypedValue>;

// ─────────────────────────────── parse helpers ───────────────────────────────

/// Parse a quoted CSV row, respecting `""` escape sequences.
///
/// Trailing/leading whitespace is **not** stripped inside quoted fields
/// (RFC 4180 compliance) but is stripped for bare fields.
fn parse_csv_row_quoted(line: &str, delimiter: u8) -> Vec<String> {
    let sep = delimiter as char;
    let mut fields: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '"' {
            if in_quotes {
                if chars.peek() == Some(&'"') {
                    // Escaped double quote
                    chars.next();
                    current.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                in_quotes = true;
            }
        } else if ch == sep && !in_quotes {
            fields.push(current.trim().to_string());
            current.clear();
        } else {
            current.push(ch);
        }
    }
    fields.push(current.trim().to_string());
    fields
}

/// Infer the [`TypedValue`] for a single string cell given a [`ColumnType`] hint.
fn coerce_cell(cell: &str, col_type: &ColumnType) -> TypedValue {
    let trimmed = cell.trim();
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("null")
        || trimmed.eq_ignore_ascii_case("na")
        || trimmed.eq_ignore_ascii_case("n/a")
        || trimmed.eq_ignore_ascii_case("nan")
    {
        return TypedValue::Null;
    }
    match col_type {
        ColumnType::Integer => trimmed
            .parse::<i64>()
            .map(TypedValue::Integer)
            .unwrap_or_else(|_| TypedValue::Text(trimmed.to_string())),
        ColumnType::Float => trimmed
            .parse::<f64>()
            .map(TypedValue::Float)
            .unwrap_or_else(|_| TypedValue::Text(trimmed.to_string())),
        ColumnType::Boolean => match trimmed.to_lowercase().as_str() {
            "true" | "yes" | "1" => TypedValue::Boolean(true),
            "false" | "no" | "0" => TypedValue::Boolean(false),
            _ => TypedValue::Text(trimmed.to_string()),
        },
        ColumnType::Text => TypedValue::Text(trimmed.to_string()),
    }
}

// ─────────────────────────────── CsvStreamReader ─────────────────────────────

/// Streaming CSV reader backed by a file on disk.
///
/// Implements [`Iterator`]`<Item = Result<Vec<String>>>` for lazy, one-row-at-a-time
/// processing.  The reader owns a [`BufReader<File>`] so only a small I/O buffer
/// is kept in memory regardless of file size.
///
/// # Behaviour
///
/// - If `has_header` is `true` the first non-blank line is consumed during
///   construction and exposed via `headers`.
/// - Blank lines inside the data region are silently skipped.
/// - Quoted fields (`"..."`) with internal commas or escaped `""` double-quotes
///   are handled correctly.
pub struct CsvStreamReader {
    inner: BufReader<File>,
    delimiter: u8,
    headers: Option<Vec<String>>,
    finished: bool,
    rows_yielded: u64,
}

impl CsvStreamReader {
    /// Open `path` as a streaming CSV reader.
    ///
    /// # Arguments
    ///
    /// * `path`        — path to the CSV file.
    /// * `delimiter`   — field separator byte (e.g. `b','` or `b'\t'`).
    /// * `has_header`  — if `true` the first row is treated as a header row.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::FileNotFound`] if the path does not exist, or
    /// [`IoError::FileError`] on any I/O failure while reading the header.
    pub fn new<P: AsRef<Path>>(path: P, delimiter: u8, has_header: bool) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| IoError::FileNotFound(format!("{}: {e}", path.display())))?;
        let mut inner = BufReader::new(file);

        let headers = if has_header {
            let mut line = String::new();
            loop {
                line.clear();
                let n = inner
                    .read_line(&mut line)
                    .map_err(|e| IoError::FileError(format!("header read error: {e}")))?;
                if n == 0 {
                    // Empty file — no header line found.
                    break None;
                }
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    let hdrs = parse_csv_row_quoted(trimmed, delimiter);
                    break Some(hdrs);
                }
            }
        } else {
            None
        };

        Ok(Self {
            inner,
            delimiter,
            headers,
            finished: false,
            rows_yielded: 0,
        })
    }

    /// Return the header row if `has_header` was `true`.
    pub fn headers(&self) -> Option<&[String]> {
        self.headers.as_deref()
    }

    /// Total data rows yielded so far (not counting the header).
    pub fn rows_yielded(&self) -> u64 {
        self.rows_yielded
    }

    /// Read up to `n_rows` data rows as a batch.
    ///
    /// Returns an empty `Vec` when the file is exhausted.  Any error encountered
    /// while reading a row causes the entire batch call to fail.
    pub fn read_chunk(&mut self, n_rows: usize) -> Result<Vec<Vec<String>>> {
        let mut batch = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            match self.next_row_inner()? {
                Some(row) => batch.push(row),
                None => break,
            }
        }
        Ok(batch)
    }

    /// Read the next raw string row. Returns `Ok(None)` at EOF.
    fn next_row_inner(&mut self) -> Result<Option<Vec<String>>> {
        if self.finished {
            return Ok(None);
        }
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.inner.read_line(&mut line).map_err(|e| {
                IoError::FileError(format!("read error at row {}: {e}", self.rows_yielded + 1))
            })?;
            if n == 0 {
                self.finished = true;
                return Ok(None);
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            self.rows_yielded += 1;
            return Ok(Some(parse_csv_row_quoted(trimmed, self.delimiter)));
        }
    }
}

impl Iterator for CsvStreamReader {
    type Item = Result<Vec<String>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_row_inner() {
            Ok(Some(row)) => Some(Ok(row)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

// ─────────────────────────────── Schema inference ────────────────────────────

/// Infer the column schema by scanning up to `n_rows` data rows of a CSV file.
///
/// The function opens the file fresh (separate from any existing reader), reads
/// up to `n_rows` rows (after the header) and applies the following heuristic
/// per column:
///
/// 1. All non-null sampled values parse as `i64`  → [`ColumnType::Integer`]
/// 2. All non-null sampled values parse as `f64`  → [`ColumnType::Float`]
/// 3. All non-null values are recognised booleans → [`ColumnType::Boolean`]
/// 4. Otherwise                                   → [`ColumnType::Text`]
///
/// If no non-null values are seen for a column, it defaults to `Text`.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or read.
pub fn infer_schema<P: AsRef<Path>>(
    path: P,
    delimiter: u8,
    n_rows: usize,
) -> Result<Vec<ColumnType>> {
    let path = path.as_ref();
    let mut reader = CsvStreamReader::new(path, delimiter, true)?;

    // Collect n_rows sample rows (or however many exist).
    let sample = reader.read_chunk(n_rows)?;

    if sample.is_empty() {
        return Ok(Vec::new());
    }

    let n_cols = sample.iter().map(|r| r.len()).max().unwrap_or(0);
    if n_cols == 0 {
        return Ok(Vec::new());
    }

    // Per-column tracking flags.
    // We start optimistic (all types possible) and rule out as we see values.
    #[derive(Clone)]
    struct ColFlags {
        can_int: bool,
        can_float: bool,
        can_bool: bool,
        seen_non_null: bool,
    }

    let mut flags = vec![
        ColFlags {
            can_int: true,
            can_float: true,
            can_bool: true,
            seen_non_null: false,
        };
        n_cols
    ];

    let null_sentinels: &[&str] = &["", "null", "na", "n/a", "nan"];

    for row in &sample {
        for (col_idx, cell) in row.iter().enumerate() {
            if col_idx >= n_cols {
                break;
            }
            let trimmed = cell.trim();
            let is_null = null_sentinels
                .iter()
                .any(|s| trimmed.eq_ignore_ascii_case(s));
            if is_null {
                continue;
            }

            let f = &mut flags[col_idx];
            f.seen_non_null = true;

            // Test integer parsability.
            if f.can_int && trimmed.parse::<i64>().is_err() {
                f.can_int = false;
            }
            // Test float parsability (integers are valid floats too).
            if f.can_float && trimmed.parse::<f64>().is_err() {
                f.can_float = false;
            }
            // Test boolean parsability.
            if f.can_bool {
                let lower = trimmed.to_lowercase();
                match lower.as_str() {
                    "true" | "false" | "yes" | "no" | "1" | "0" => {}
                    _ => f.can_bool = false,
                }
            }
        }
    }

    let schema = flags
        .into_iter()
        .map(|f| {
            if !f.seen_non_null {
                return ColumnType::Text;
            }
            if f.can_int {
                ColumnType::Integer
            } else if f.can_float {
                ColumnType::Float
            } else if f.can_bool {
                ColumnType::Boolean
            } else {
                ColumnType::Text
            }
        })
        .collect();

    Ok(schema)
}

// ─────────────────────────────── Typed row parsing ───────────────────────────

/// Parse a raw string row into a [`TypedRow`] by applying the given schema.
///
/// If `row.len() < schema.len()` the trailing columns receive [`TypedValue::Null`].
/// Extra columns beyond the schema length are returned as [`TypedValue::Text`].
///
/// # Errors
///
/// This function is currently infallible (it degrades gracefully) but returns
/// `Result` to allow future validation hooks.
pub fn read_typed_row(row: &[String], schema: &[ColumnType]) -> Result<TypedRow> {
    let len = schema.len().max(row.len());
    let mut typed = Vec::with_capacity(len);
    for col_idx in 0..len {
        let cell = row.get(col_idx).map(String::as_str).unwrap_or("");
        let col_type = schema.get(col_idx).unwrap_or(&ColumnType::Text);
        typed.push(coerce_cell(cell, col_type));
    }
    Ok(typed)
}

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_csv(name: &str, content: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("scirs2_streaming_csv_tests");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join(name);
        let mut f = File::create(&path).expect("create");
        f.write_all(content.as_bytes()).expect("write");
        path
    }

    // ── Iterator interface ────────────────────────────────────────────────────

    #[test]
    fn test_iterator_with_header() {
        let path = write_temp_csv(
            "iter_header.csv",
            "name,age,score\nAlice,30,9.5\nBob,25,8.1\n",
        );
        let mut r = CsvStreamReader::new(&path, b',', true).expect("open");
        assert_eq!(
            r.headers(),
            Some(vec!["name".to_string(), "age".to_string(), "score".to_string()].as_slice())
        );

        let rows: Vec<_> = r.by_ref().map(|x| x.expect("row ok")).collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec!["Alice", "30", "9.5"]);
        assert_eq!(rows[1], vec!["Bob", "25", "8.1"]);
    }

    #[test]
    fn test_iterator_no_header() {
        let path = write_temp_csv("iter_no_header.csv", "1,2,3\n4,5,6\n");
        let mut r = CsvStreamReader::new(&path, b',', false).expect("open");
        assert!(r.headers().is_none());
        let rows: Vec<_> = r.by_ref().map(|x| x.expect("ok")).collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec!["1", "2", "3"]);
    }

    // ── read_chunk ────────────────────────────────────────────────────────────

    #[test]
    fn test_read_chunk_basic() {
        let content = "a,b\n1,2\n3,4\n5,6\n7,8\n9,10\n";
        let path = write_temp_csv("chunk_basic.csv", content);
        let mut r = CsvStreamReader::new(&path, b',', true).expect("open");

        let chunk1 = r.read_chunk(2).expect("chunk1");
        assert_eq!(chunk1.len(), 2);
        assert_eq!(chunk1[0], vec!["1", "2"]);

        let chunk2 = r.read_chunk(2).expect("chunk2");
        assert_eq!(chunk2.len(), 2);
        assert_eq!(chunk2[0], vec!["5", "6"]);

        let chunk3 = r.read_chunk(10).expect("chunk3"); // fewer than requested
        assert_eq!(chunk3.len(), 1);
        assert_eq!(chunk3[0], vec!["9", "10"]);

        let empty = r.read_chunk(5).expect("empty");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_read_chunk_larger_than_file() {
        let path = write_temp_csv("chunk_large.csv", "x\n1\n2\n");
        let mut r = CsvStreamReader::new(&path, b',', true).expect("open");
        let all = r.read_chunk(9999).expect("all");
        assert_eq!(all.len(), 2);
    }

    // ── rows_yielded counter ──────────────────────────────────────────────────

    #[test]
    fn test_rows_yielded_tracks_count() {
        let path = write_temp_csv("rows_yielded.csv", "h\n1\n2\n3\n");
        let mut r = CsvStreamReader::new(&path, b',', true).expect("open");
        assert_eq!(r.rows_yielded(), 0);
        r.next().expect("some").expect("ok");
        assert_eq!(r.rows_yielded(), 1);
        r.read_chunk(5).expect("rest");
        assert_eq!(r.rows_yielded(), 3);
    }

    // ── quoted fields ─────────────────────────────────────────────────────────

    #[test]
    fn test_quoted_fields_with_commas() {
        let content = "name,address\nAlice,\"New York, NY\"\nBob,\"Los Angeles, CA\"\n";
        let path = write_temp_csv("quoted.csv", content);
        let mut r = CsvStreamReader::new(&path, b',', true).expect("open");
        let row1 = r.next().expect("some").expect("ok");
        assert_eq!(row1[1], "New York, NY");
    }

    #[test]
    fn test_escaped_double_quote_inside_field() {
        let content = "id,note\n1,\"He said \"\"hello\"\"\"\n";
        let path = write_temp_csv("escaped_quote.csv", content);
        let mut r = CsvStreamReader::new(&path, b',', true).expect("open");
        let row = r.next().expect("some").expect("ok");
        assert_eq!(row[1], "He said \"hello\"");
    }

    // ── tab delimiter ─────────────────────────────────────────────────────────

    #[test]
    fn test_tab_delimiter() {
        let path = write_temp_csv("tab.csv", "a\tb\tc\n10\t20\t30\n");
        let mut r = CsvStreamReader::new(&path, b'\t', true).expect("open");
        assert_eq!(
            r.headers(),
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()].as_slice())
        );
        let row = r.next().expect("some").expect("ok");
        assert_eq!(row, vec!["10", "20", "30"]);
    }

    // ── blank lines are skipped ───────────────────────────────────────────────

    #[test]
    fn test_blank_lines_skipped() {
        let path = write_temp_csv("blanks.csv", "x\n1\n\n\n2\n");
        let mut r = CsvStreamReader::new(&path, b',', true).expect("open");
        let rows: Vec<_> = r.by_ref().map(|x| x.expect("ok")).collect();
        assert_eq!(rows.len(), 2);
    }

    // ── schema inference ──────────────────────────────────────────────────────

    #[test]
    fn test_infer_schema_mixed_types() {
        let content = "id,value,active,label\n1,3.14,true,hello\n2,2.71,false,world\n";
        let path = write_temp_csv("schema_mixed.csv", content);
        let schema = infer_schema(&path, b',', 50).expect("infer");
        assert_eq!(schema.len(), 4);
        assert_eq!(schema[0], ColumnType::Integer);
        assert_eq!(schema[1], ColumnType::Float);
        assert_eq!(schema[2], ColumnType::Boolean);
        assert_eq!(schema[3], ColumnType::Text);
    }

    #[test]
    fn test_infer_schema_all_integer() {
        let path = write_temp_csv("schema_int.csv", "n\n1\n2\n3\n");
        let schema = infer_schema(&path, b',', 10).expect("infer");
        assert_eq!(schema[0], ColumnType::Integer);
    }

    #[test]
    fn test_infer_schema_float_beats_integer_when_mixed() {
        let path = write_temp_csv("schema_float.csv", "n\n1\n2.5\n3\n");
        let schema = infer_schema(&path, b',', 10).expect("infer");
        assert_eq!(schema[0], ColumnType::Float);
    }

    #[test]
    fn test_infer_schema_with_nulls() {
        // Column with only nulls should default to Text.
        let path = write_temp_csv("schema_null.csv", "a,b\n1,\n2,NA\n");
        let schema = infer_schema(&path, b',', 10).expect("infer");
        assert_eq!(schema[0], ColumnType::Integer);
        assert_eq!(schema[1], ColumnType::Text);
    }

    // ── typed row parsing ─────────────────────────────────────────────────────

    #[test]
    fn test_read_typed_row_all_types() {
        let schema = vec![
            ColumnType::Integer,
            ColumnType::Float,
            ColumnType::Boolean,
            ColumnType::Text,
        ];
        let raw: Vec<String> = vec!["42", "2.5", "true", "hello"]
            .into_iter()
            .map(String::from)
            .collect();
        let typed = read_typed_row(&raw, &schema).expect("parse");
        assert!(matches!(typed[0], TypedValue::Integer(42)));
        assert!(matches!(typed[1], TypedValue::Float(f) if (f - 2.5).abs() < 1e-10));
        assert!(matches!(typed[2], TypedValue::Boolean(true)));
        assert!(matches!(typed[3], TypedValue::Text(ref s) if s == "hello"));
    }

    #[test]
    fn test_read_typed_row_null_cells() {
        let schema = vec![ColumnType::Integer, ColumnType::Float];
        let raw: Vec<String> = vec!["", "NA"].into_iter().map(String::from).collect();
        let typed = read_typed_row(&raw, &schema).expect("parse");
        assert!(matches!(typed[0], TypedValue::Null));
        assert!(matches!(typed[1], TypedValue::Null));
    }

    #[test]
    fn test_read_typed_row_short_row_padded_with_null() {
        let schema = vec![ColumnType::Integer, ColumnType::Float, ColumnType::Boolean];
        let raw: Vec<String> = vec!["1"].into_iter().map(String::from).collect();
        let typed = read_typed_row(&raw, &schema).expect("parse");
        assert_eq!(typed.len(), 3);
        assert!(matches!(typed[0], TypedValue::Integer(1)));
        assert!(matches!(typed[1], TypedValue::Null));
        assert!(matches!(typed[2], TypedValue::Null));
    }

    #[test]
    fn test_read_typed_row_extra_columns_text() {
        let schema = vec![ColumnType::Integer];
        let raw: Vec<String> = vec!["1", "extra"].into_iter().map(String::from).collect();
        let typed = read_typed_row(&raw, &schema).expect("parse");
        assert_eq!(typed.len(), 2);
        assert!(matches!(typed[1], TypedValue::Text(_)));
    }

    // ── large file simulation (many rows, streaming) ──────────────────────────

    #[test]
    fn test_large_file_lazy_iteration() {
        // 10 000 rows — only one row in memory at a time.
        let n = 10_000_usize;
        let mut content = String::from("i,v\n");
        for i in 0..n {
            content.push_str(&format!("{},{}\n", i, i as f64 * 1.1));
        }
        let path = write_temp_csv("large.csv", &content);
        let mut r = CsvStreamReader::new(&path, b',', true).expect("open");
        let mut count = 0usize;
        for item in &mut r {
            let row = item.expect("row ok");
            let _ = row[0].parse::<usize>().expect("int");
            count += 1;
        }
        assert_eq!(count, n);
    }
}
