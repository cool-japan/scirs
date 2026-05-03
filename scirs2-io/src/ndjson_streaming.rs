//! NDJSON / CSV streaming types and TSV utilities
//!
//! Provides memory-efficient streaming interfaces for NDJSON (Newline-Delimited JSON),
//! CSV, and TSV formats. Each type reads or writes one record at a time, making them
//! suitable for datasets that cannot fit in memory.
//!
//! # Examples
//!
//! ```rust,no_run
//! use scirs2_io::ndjson_streaming::{NdjsonReader, NdjsonWriter};
//! use std::io::BufReader;
//! use std::fs::File;
//!
//! // Write records
//! let file = File::create("/tmp/records.ndjson").unwrap();
//! let mut writer = NdjsonWriter::new(file);
//! let record = serde_json::json!({"id": 1, "value": 3.14});
//! writer.write_record(&record).unwrap();
//! writer.flush().unwrap();
//!
//! // Read records back
//! let file = File::open("/tmp/records.ndjson").unwrap();
//! let mut reader = NdjsonReader::new(BufReader::new(file));
//! while let Some(rec) = reader.next_record().unwrap() {
//!     println!("{:?}", rec);
//! }
//! ```

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::{IoError, Result};

// ─────────────────────────────── NdjsonReader ────────────────────────────────

/// NDJSON (Newline-Delimited JSON) streaming reader.
///
/// Reads one JSON value per line, skipping blank lines and `#`-prefixed comments.
/// Memory usage is proportional to the largest single record, not the whole file.
pub struct NdjsonReader<R: BufRead> {
    reader: R,
    line_number: usize,
}

impl<R: BufRead> NdjsonReader<R> {
    /// Create a new reader wrapping any `BufRead` source.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line_number: 0,
        }
    }

    /// Read and parse the next JSON record.
    ///
    /// Returns `Ok(None)` at end-of-file.  Blank lines and `#` comments are skipped.
    pub fn next_record(&mut self) -> Result<Option<serde_json::Value>> {
        let mut line = String::new();
        loop {
            line.clear();
            let n = self
                .reader
                .read_line(&mut line)
                .map_err(|e| IoError::FileError(format!("line {}: {e}", self.line_number + 1)))?;
            if n == 0 {
                return Ok(None);
            }
            self.line_number += 1;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let val = serde_json::from_str(trimmed)
                .map_err(|e| IoError::ParseError(format!("line {}: {e}", self.line_number)))?;
            return Ok(Some(val));
        }
    }

    /// Collect every remaining record into a `Vec`.
    pub fn collect_all(&mut self) -> Result<Vec<serde_json::Value>> {
        let mut out = Vec::new();
        while let Some(v) = self.next_record()? {
            out.push(v);
        }
        Ok(out)
    }

    /// Count remaining records without storing them.
    pub fn count_records(&mut self) -> Result<usize> {
        let mut count = 0usize;
        while self.next_record()?.is_some() {
            count += 1;
        }
        Ok(count)
    }

    /// Current 1-based line number (lines consumed so far).
    pub fn line_number(&self) -> usize {
        self.line_number
    }
}

// ─────────────────────────────── NdjsonWriter ────────────────────────────────

/// NDJSON (Newline-Delimited JSON) streaming writer.
///
/// Serialises each [`serde_json::Value`] as a single line and appends `\n`.
pub struct NdjsonWriter<W: Write> {
    writer: W,
}

impl<W: Write> NdjsonWriter<W> {
    /// Create a new writer wrapping any `Write` sink.
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    /// Serialise `record` and write it as one line.
    pub fn write_record(&mut self, record: &serde_json::Value) -> Result<()> {
        let json = serde_json::to_string(record)
            .map_err(|e| IoError::SerializationError(format!("JSON serialization failed: {e}")))?;
        self.writer
            .write_all(json.as_bytes())
            .map_err(|e| IoError::FileError(format!("write failed: {e}")))?;
        self.writer
            .write_all(b"\n")
            .map_err(|e| IoError::FileError(format!("write newline failed: {e}")))?;
        Ok(())
    }

    /// Flush the underlying writer.
    pub fn flush(&mut self) -> Result<()> {
        self.writer
            .flush()
            .map_err(|e| IoError::FileError(format!("flush failed: {e}")))
    }
}

// ─────────────────────────────── CsvValue ────────────────────────────────────

/// A single type-inferred cell from a CSV / TSV row.
#[derive(Debug, Clone, PartialEq)]
pub enum CsvValue {
    /// Parsed as a 64-bit integer.
    Integer(i64),
    /// Parsed as a 64-bit float.
    Float(f64),
    /// Parsed as a boolean (`true`/`false`/`yes`/`no`/`1`/`0`).
    Boolean(bool),
    /// Raw text that could not be parsed as any structured type.
    Text(String),
    /// Empty field or explicit null sentinel.
    Null,
}

impl CsvValue {
    fn infer(s: &str) -> Self {
        let trimmed = s.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("null") || trimmed == "NA" {
            return CsvValue::Null;
        }
        // Integer check before Boolean to avoid treating "1"/"0" as booleans.
        if let Ok(i) = trimmed.parse::<i64>() {
            return CsvValue::Integer(i);
        }
        // Boolean (word forms only; numeric "1"/"0" handled above as integers)
        match trimmed.to_lowercase().as_str() {
            "true" | "yes" => return CsvValue::Boolean(true),
            "false" | "no" => return CsvValue::Boolean(false),
            _ => {}
        }
        // Float
        if let Ok(f) = trimmed.parse::<f64>() {
            return CsvValue::Float(f);
        }
        CsvValue::Text(trimmed.to_string())
    }
}

// ─────────────────────────────── CsvStreamReader ─────────────────────────────

/// Streaming CSV reader with per-row type inference.
///
/// Reads one row at a time; does not load the entire file into memory.
/// Handles quoted fields and configurable delimiters.
pub struct CsvStreamReader<R: BufRead> {
    reader: R,
    delimiter: u8,
    headers: Option<Vec<String>>,
    has_header: bool,
    line_number: usize,
    finished: bool,
}

impl<R: BufRead> CsvStreamReader<R> {
    /// Create a new streaming CSV reader.
    ///
    /// If `has_header` is `true` the first non-blank line is consumed immediately
    /// and stored; subsequent calls to [`next_row`] return data rows only.
    pub fn new(mut reader: R, has_header: bool, delimiter: u8) -> Result<Self> {
        let headers = if has_header {
            let mut line = String::new();
            loop {
                line.clear();
                let n = reader
                    .read_line(&mut line)
                    .map_err(|e| IoError::FileError(format!("header read error: {e}")))?;
                if n == 0 {
                    break None;
                }
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    let hdrs = parse_csv_row(trimmed, delimiter);
                    break Some(hdrs);
                }
            }
        } else {
            None
        };

        Ok(Self {
            reader,
            delimiter,
            headers,
            has_header,
            line_number: if has_header { 1 } else { 0 },
            finished: false,
        })
    }

    /// Return the parsed header row, or `None` if `has_header` was `false`.
    pub fn headers(&self) -> Option<&[String]> {
        self.headers.as_deref()
    }

    /// Read the next raw (string) row.  Returns `Ok(None)` at end-of-file.
    pub fn next_row(&mut self) -> Result<Option<Vec<String>>> {
        if self.finished {
            return Ok(None);
        }
        let mut line = String::new();
        loop {
            line.clear();
            let n = self
                .reader
                .read_line(&mut line)
                .map_err(|e| IoError::FileError(format!("line {}: {e}", self.line_number + 1)))?;
            if n == 0 {
                self.finished = true;
                return Ok(None);
            }
            self.line_number += 1;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            return Ok(Some(parse_csv_row(trimmed, self.delimiter)));
        }
    }

    /// Read the next row with automatic type inference applied to each field.
    pub fn next_typed_row(&mut self) -> Result<Option<Vec<CsvValue>>> {
        match self.next_row()? {
            None => Ok(None),
            Some(fields) => Ok(Some(fields.iter().map(|s| CsvValue::infer(s)).collect())),
        }
    }
}

/// Parse a single CSV/TSV line respecting double-quoted fields.
fn parse_csv_row(line: &str, delimiter: u8) -> Vec<String> {
    let sep = delimiter as char;
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '"' {
            if in_quotes {
                // Check for escaped quote (`""`)
                if chars.peek() == Some(&'"') {
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

// ─────────────────────────────── TSV helpers ─────────────────────────────────

/// Read an entire TSV file into `(headers, rows)`.
///
/// The first row is treated as the header.  All subsequent rows are data.
pub fn read_tsv(path: &Path) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let file =
        File::open(path).map_err(|e| IoError::FileError(format!("cannot open {:?}: {e}", path)))?;
    let mut reader = CsvStreamReader::new(BufReader::new(file), true, b'\t')?;

    let headers = reader
        .headers()
        .ok_or_else(|| IoError::FormatError("TSV file appears empty".to_string()))?
        .to_vec();

    let mut rows = Vec::new();
    while let Some(row) = reader.next_row()? {
        rows.push(row);
    }
    Ok((headers, rows))
}

/// Write headers and rows to a TSV file.
pub fn write_tsv(path: &Path, headers: &[String], data: &[Vec<String>]) -> Result<()> {
    let file = File::create(path)
        .map_err(|e| IoError::FileError(format!("cannot create {:?}: {e}", path)))?;
    let mut writer = BufWriter::new(file);

    writer
        .write_all(headers.join("\t").as_bytes())
        .map_err(|e| IoError::FileError(format!("write header failed: {e}")))?;
    writer
        .write_all(b"\n")
        .map_err(|e| IoError::FileError(format!("write newline failed: {e}")))?;

    for row in data {
        writer
            .write_all(row.join("\t").as_bytes())
            .map_err(|e| IoError::FileError(format!("write row failed: {e}")))?;
        writer
            .write_all(b"\n")
            .map_err(|e| IoError::FileError(format!("write newline failed: {e}")))?;
    }
    writer
        .flush()
        .map_err(|e| IoError::FileError(format!("flush failed: {e}")))
}

// ─────────────────────────────── File-level helpers ──────────────────────────

/// Open an NDJSON file and return a reader over it.
pub fn open_ndjson_file(path: &Path) -> Result<NdjsonReader<BufReader<File>>> {
    let file =
        File::open(path).map_err(|e| IoError::FileError(format!("cannot open {:?}: {e}", path)))?;
    Ok(NdjsonReader::new(BufReader::new(file)))
}

/// Create / overwrite an NDJSON file and return a buffered writer over it.
pub fn create_ndjson_file(path: &Path) -> Result<NdjsonWriter<BufWriter<File>>> {
    let file = File::create(path)
        .map_err(|e| IoError::FileError(format!("cannot create {:?}: {e}", path)))?;
    Ok(NdjsonWriter::new(BufWriter::new(file)))
}

/// Append to an existing NDJSON file (creates it if absent).
pub fn append_ndjson_file(path: &Path) -> Result<NdjsonWriter<BufWriter<File>>> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| IoError::FileError(format!("cannot open {:?} for append: {e}", path)))?;
    Ok(NdjsonWriter::new(BufWriter::new(file)))
}

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    fn ndjson_bytes(lines: &[&str]) -> Vec<u8> {
        lines.join("\n").into_bytes()
    }

    // ── NdjsonReader ──────────────────────────────────────────────────────────

    #[test]
    fn test_ndjson_reader_single_record() {
        let src = ndjson_bytes(&[r#"{"id":1,"v":2.5}"#]);
        let mut r = NdjsonReader::new(BufReader::new(src.as_slice()));
        let rec = r
            .next_record()
            .expect("should parse")
            .expect("should have record");
        assert_eq!(rec["id"], 1);
        assert!((rec["v"].as_f64().expect("float") - 2.5).abs() < 1e-10);
        assert!(r.next_record().expect("no error").is_none());
    }

    #[test]
    fn test_ndjson_reader_multi_record() {
        let src = ndjson_bytes(&[r#"{"a":1}"#, r#"{"a":2}"#, r#"{"a":3}"#]);
        let mut r = NdjsonReader::new(BufReader::new(src.as_slice()));
        let all = r.collect_all().expect("collect");
        assert_eq!(all.len(), 3);
        assert_eq!(all[2]["a"], 3);
    }

    #[test]
    fn test_ndjson_reader_skips_blank_and_comment_lines() {
        let src = ndjson_bytes(&["", "# comment", r#"{"x":42}"#, "", r#"{"x":99}"#]);
        let mut r = NdjsonReader::new(BufReader::new(src.as_slice()));
        assert_eq!(r.count_records().expect("count"), 2);
    }

    #[test]
    fn test_ndjson_reader_empty_source() {
        let src: &[u8] = b"";
        let mut r = NdjsonReader::new(BufReader::new(src));
        assert!(r.next_record().expect("no error").is_none());
    }

    // ── NdjsonWriter ─────────────────────────────────────────────────────────

    #[test]
    fn test_ndjson_writer_produces_newline_delimited_json() {
        let mut buf: Vec<u8> = Vec::new();
        let mut w = NdjsonWriter::new(&mut buf);
        w.write_record(&serde_json::json!({"k": "v1"}))
            .expect("write");
        w.write_record(&serde_json::json!({"k": "v2"}))
            .expect("write");
        w.flush().expect("flush");

        let text = String::from_utf8(buf).expect("utf8");
        let lines: Vec<_> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        let v: serde_json::Value = serde_json::from_str(lines[1]).expect("parse");
        assert_eq!(v["k"], "v2");
    }

    // ── NdjsonWriter + NdjsonReader round-trip ────────────────────────────────

    #[test]
    fn test_ndjson_roundtrip_via_temp_file() {
        let dir = std::env::temp_dir().join("scirs2_io_ndjson_rt_test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("roundtrip.ndjson");

        let records = vec![
            serde_json::json!({"id": 1, "name": "alpha", "score": 9.5}),
            serde_json::json!({"id": 2, "name": "beta",  "score": 7.2}),
            serde_json::json!({"id": 3, "name": "gamma", "score": 8.8}),
        ];

        {
            let mut w = create_ndjson_file(&path).expect("create");
            for rec in &records {
                w.write_record(rec).expect("write");
            }
            w.flush().expect("flush");
        }

        let mut r = open_ndjson_file(&path).expect("open");
        let loaded = r.collect_all().expect("collect");

        assert_eq!(loaded.len(), 3);
        for (orig, loaded_rec) in records.iter().zip(loaded.iter()) {
            assert_eq!(orig["id"], loaded_rec["id"]);
            assert_eq!(orig["name"], loaded_rec["name"]);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── CsvStreamReader ───────────────────────────────────────────────────────

    #[test]
    fn test_csv_stream_reader_headers_and_rows() {
        let csv = b"name,age,city\nAlice,30,London\nBob,25,Paris\n";
        let mut r =
            CsvStreamReader::new(BufReader::new(csv.as_ref()), true, b',').expect("new reader");

        let hdrs = r.headers().expect("headers").to_vec();
        assert_eq!(hdrs, vec!["name", "age", "city"]);

        let row1 = r.next_row().expect("row1 err").expect("row1 some");
        assert_eq!(row1, vec!["Alice", "30", "London"]);

        let row2 = r.next_row().expect("row2 err").expect("row2 some");
        assert_eq!(row2, vec!["Bob", "25", "Paris"]);

        assert!(r.next_row().expect("eof err").is_none());
    }

    #[test]
    fn test_csv_stream_reader_no_header() {
        let csv = b"1,2,3\n4,5,6\n";
        let mut r =
            CsvStreamReader::new(BufReader::new(csv.as_ref()), false, b',').expect("new reader");
        assert!(r.headers().is_none());
        let row = r.next_row().expect("row").expect("some");
        assert_eq!(row, vec!["1", "2", "3"]);
    }

    #[test]
    fn test_csv_stream_reader_typed_row() {
        let csv = b"id,active,value,label\n1,true,3.14,hello\n2,false,,NA\n";
        let mut r =
            CsvStreamReader::new(BufReader::new(csv.as_ref()), true, b',').expect("new reader");

        let row = r.next_typed_row().expect("row").expect("some");
        assert!(matches!(row[0], CsvValue::Integer(1)));
        assert!(matches!(row[1], CsvValue::Boolean(true)));
        assert!(matches!(row[2], CsvValue::Float(_)));
        assert!(matches!(row[3], CsvValue::Text(_)));

        let row2 = r.next_typed_row().expect("row2").expect("some2");
        assert!(matches!(row2[2], CsvValue::Null));
        assert!(matches!(row2[3], CsvValue::Null));
    }

    #[test]
    fn test_csv_stream_reader_tsv_delimiter() {
        let tsv = b"a\tb\tc\n10\t20\t30\n";
        let mut r =
            CsvStreamReader::new(BufReader::new(tsv.as_ref()), true, b'\t').expect("new reader");
        let hdrs = r.headers().expect("hdrs").to_vec();
        assert_eq!(hdrs, vec!["a", "b", "c"]);
        let row = r.next_row().expect("row").expect("some");
        assert_eq!(row, vec!["10", "20", "30"]);
    }

    // ── TSV read/write round-trip ─────────────────────────────────────────────

    #[test]
    fn test_tsv_roundtrip() {
        let dir = std::env::temp_dir().join("scirs2_io_tsv_rt_test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("data.tsv");

        let headers = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let data = vec![
            vec!["1".to_string(), "2".to_string(), "3".to_string()],
            vec!["4".to_string(), "5".to_string(), "6".to_string()],
        ];

        write_tsv(&path, &headers, &data).expect("write tsv");
        let (read_hdrs, read_data) = read_tsv(&path).expect("read tsv");

        assert_eq!(read_hdrs, headers);
        assert_eq!(read_data, data);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── CsvValue::infer edge-cases ────────────────────────────────────────────

    #[test]
    fn test_csv_value_infer() {
        assert!(matches!(CsvValue::infer(""), CsvValue::Null));
        assert!(matches!(CsvValue::infer("null"), CsvValue::Null));
        assert!(matches!(CsvValue::infer("NA"), CsvValue::Null));
        assert!(matches!(CsvValue::infer("true"), CsvValue::Boolean(true)));
        assert!(matches!(CsvValue::infer("False"), CsvValue::Boolean(false)));
        assert!(matches!(CsvValue::infer("42"), CsvValue::Integer(42)));
        assert!(matches!(CsvValue::infer("3.14"), CsvValue::Float(_)));
        assert!(matches!(CsvValue::infer("hello"), CsvValue::Text(_)));
    }
}
