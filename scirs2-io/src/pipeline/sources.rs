//! Data source abstractions for the streaming pipeline.
//!
//! Every source implements the [`DataSource`] trait which yields batches of
//! `serde_json::Value` records.  Concrete implementations:
//!
//! | Source            | Description                                     |
//! |-------------------|-------------------------------------------------|
//! | [`FileSource`]    | Read CSV / JSON / JSONL files in batches        |
//! | [`DatabaseSource`]| SQLite query-based loading                      |
//! | [`StreamSource`]  | Kafka-like channel-based message consumption    |
//! | [`GeneratorSource`]| Produce data from a user closure               |
//!
//! `DatabaseSource` uses `oxisql-sqlite-compat` (Pure Rust SQLite via Limbo engine)
//! with a sync/async bridge via `tokio::task::block_in_place` / a fresh `Runtime`.

#![allow(missing_docs)]

use crate::error::{IoError, Result};
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

// ─────────────────────────────────────────────────────────────────────────────
// Core trait
// ─────────────────────────────────────────────────────────────────────────────

/// A pull-based data source that yields record batches.
pub trait DataSource: Send {
    /// Return the next batch of records, or `Ok(None)` when exhausted.
    fn next_batch(&mut self, batch_size: usize) -> Result<Option<Vec<Value>>>;

    /// Reset the source to the beginning, if supported.
    fn reset(&mut self) -> Result<()> {
        Err(IoError::Other(
            "reset not supported by this source".to_string(),
        ))
    }

    /// Human-readable name for diagnostics.
    fn name(&self) -> &str {
        "unknown"
    }

    /// Estimated total number of records, if knowable.
    fn estimated_len(&self) -> Option<usize> {
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FileSource
// ─────────────────────────────────────────────────────────────────────────────

/// Supported file formats for [`FileSource`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileSourceFormat {
    /// Comma-separated values
    Csv,
    /// Newline-delimited JSON (one JSON object per line)
    JsonLines,
    /// A single JSON array `[…]` or object
    Json,
    /// Auto-detect from file extension
    Auto,
}

/// Read structured data from a CSV / JSON / JSONL file in batches.
pub struct FileSource {
    path: PathBuf,
    format: FileSourceFormat,
    reader: Option<FileSourceReader>,
    exhausted: bool,
    /// CSV header row (populated lazily on first read).
    csv_headers: Option<Vec<String>>,
    /// Pre-loaded JSON records (for non-streaming JSON).
    json_records: Option<std::collections::VecDeque<Value>>,
}

enum FileSourceReader {
    Lines(BufReader<File>),
}

impl FileSource {
    /// Create a new `FileSource` with automatic format detection.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self::with_format(path, FileSourceFormat::Auto)
    }

    /// Create a new `FileSource` with an explicit format.
    pub fn with_format(path: impl AsRef<Path>, format: FileSourceFormat) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            format,
            reader: None,
            exhausted: false,
            csv_headers: None,
            json_records: None,
        }
    }

    fn resolve_format(&self) -> FileSourceFormat {
        if self.format != FileSourceFormat::Auto {
            return self.format;
        }
        match self
            .path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref()
        {
            Some("csv") => FileSourceFormat::Csv,
            Some("jsonl") | Some("ndjson") => FileSourceFormat::JsonLines,
            Some("json") => FileSourceFormat::Json,
            _ => FileSourceFormat::JsonLines,
        }
    }

    fn open(&mut self) -> Result<()> {
        let fmt = self.resolve_format();
        match fmt {
            FileSourceFormat::Json => {
                // Load the entire JSON file up front.
                let file = File::open(&self.path).map_err(IoError::Io)?;
                let value: Value = serde_json::from_reader(BufReader::new(file))
                    .map_err(|e| IoError::ParseError(e.to_string()))?;
                let records: Vec<Value> = match value {
                    Value::Array(arr) => arr,
                    other => vec![other],
                };
                self.json_records = Some(records.into());
            }
            _ => {
                // Line-oriented formats (CSV, JSONL).
                let file = File::open(&self.path).map_err(IoError::Io)?;
                let reader = BufReader::new(file);
                self.reader = Some(FileSourceReader::Lines(reader));
            }
        }
        Ok(())
    }

    fn next_batch_csv(&mut self, batch_size: usize) -> Result<Option<Vec<Value>>> {
        let reader = match &mut self.reader {
            Some(FileSourceReader::Lines(r)) => r,
            None => return Ok(None),
        };

        // Read header on first call.
        if self.csv_headers.is_none() {
            let mut line = String::new();
            if reader.read_line(&mut line).map_err(IoError::Io)? == 0 {
                return Ok(None);
            }
            let headers: Vec<String> = line
                .trim_end_matches(['\n', '\r'])
                .split(',')
                .map(|h| h.trim().trim_matches('"').to_string())
                .collect();
            self.csv_headers = Some(headers);
        }

        let headers = self.csv_headers.clone().unwrap_or_default();
        let mut records = Vec::with_capacity(batch_size);
        let mut line = String::new();

        while records.len() < batch_size {
            line.clear();
            let n = reader.read_line(&mut line).map_err(IoError::Io)?;
            if n == 0 {
                break;
            }
            let row = line.trim_end_matches(['\n', '\r']);
            if row.is_empty() {
                continue;
            }
            let fields: Vec<&str> = row.split(',').collect();
            let obj: serde_json::Map<String, Value> = headers
                .iter()
                .enumerate()
                .map(|(i, h)| {
                    let raw = fields.get(i).copied().unwrap_or("").trim();
                    let val = raw.trim_matches('"');
                    // Try to parse as number, then bool, then leave as string.
                    let json_val: Value = if let Ok(n) = val.parse::<i64>() {
                        Value::Number(n.into())
                    } else if let Ok(f) = val.parse::<f64>() {
                        Value::Number(
                            serde_json::Number::from_f64(f).unwrap_or(serde_json::Number::from(0)),
                        )
                    } else if val == "true" {
                        Value::Bool(true)
                    } else if val == "false" {
                        Value::Bool(false)
                    } else {
                        Value::String(val.to_string())
                    };
                    (h.clone(), json_val)
                })
                .collect();
            records.push(Value::Object(obj));
        }

        if records.is_empty() {
            Ok(None)
        } else {
            Ok(Some(records))
        }
    }

    fn next_batch_jsonl(&mut self, batch_size: usize) -> Result<Option<Vec<Value>>> {
        let reader = match &mut self.reader {
            Some(FileSourceReader::Lines(r)) => r,
            None => return Ok(None),
        };

        let mut records = Vec::with_capacity(batch_size);
        let mut line = String::new();

        while records.len() < batch_size {
            line.clear();
            let n = reader.read_line(&mut line).map_err(IoError::Io)?;
            if n == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let val: Value = serde_json::from_str(trimmed)
                .map_err(|e| IoError::ParseError(format!("JSONL parse: {e}")))?;
            records.push(val);
        }

        if records.is_empty() {
            Ok(None)
        } else {
            Ok(Some(records))
        }
    }
}

impl DataSource for FileSource {
    fn next_batch(&mut self, batch_size: usize) -> Result<Option<Vec<Value>>> {
        if self.exhausted {
            return Ok(None);
        }

        if self.reader.is_none() && self.json_records.is_none() {
            self.open()?;
        }

        let result = if let Some(queue) = &mut self.json_records {
            // Pre-loaded JSON array.
            if queue.is_empty() {
                Ok(None)
            } else {
                let batch: Vec<Value> = queue.drain(..batch_size.min(queue.len())).collect();
                Ok(Some(batch))
            }
        } else {
            match self.resolve_format() {
                FileSourceFormat::Csv => self.next_batch_csv(batch_size),
                _ => self.next_batch_jsonl(batch_size),
            }
        };

        if matches!(result, Ok(None)) {
            self.exhausted = true;
        }
        result
    }

    fn reset(&mut self) -> Result<()> {
        self.reader = None;
        self.json_records = None;
        self.exhausted = false;
        self.csv_headers = None;
        Ok(())
    }

    fn name(&self) -> &str {
        "file_source"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DatabaseSource  (SQLite-backed)
// ─────────────────────────────────────────────────────────────────────────────

/// Query-based data source backed by an in-process SQLite database.
///
/// Requires the `sqlite` feature.  When the feature is absent the struct still
/// compiles but `next_batch` always returns an "unsupported" error.
pub struct DatabaseSource {
    /// SQLite connection string (path or `:memory:`).
    connection_string: String,
    /// SQL query to execute.
    query: String,
    /// Pre-fetched rows (populated lazily on first read).
    rows: Option<std::collections::VecDeque<Value>>,
    exhausted: bool,
}

impl DatabaseSource {
    /// Create a new `DatabaseSource`.
    ///
    /// - `connection_string`: path to the `.sqlite` / `.db` file, or `:memory:`.
    /// - `query`: full SQL `SELECT` statement.
    pub fn new(connection_string: impl Into<String>, query: impl Into<String>) -> Self {
        Self {
            connection_string: connection_string.into(),
            query: query.into(),
            rows: None,
            exhausted: false,
        }
    }

    #[cfg(feature = "sqlite")]
    fn load_rows(&mut self) -> Result<()> {
        use oxisql_core::{Connection as OxiConnection, Value as OxiValue};
        use oxisql_sqlite_compat::SqliteConnection;
        use std::future::Future;

        /// Run an async future to completion, bridging from sync context.
        fn run_sync_local<F, T, E>(fut: F) -> std::result::Result<T, E>
        where
            F: Future<Output = std::result::Result<T, E>>,
        {
            match tokio::runtime::Handle::try_current() {
                Ok(handle) => tokio::task::block_in_place(|| handle.block_on(fut)),
                Err(_) => tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("tokio runtime for sqlite source")
                    .block_on(fut),
            }
        }

        let conn = run_sync_local(SqliteConnection::open(&self.connection_string))
            .map_err(|e| IoError::DatabaseError(e.to_string()))?;

        let oxi_rows = run_sync_local(conn.query(&self.query, &[]))
            .map_err(|e| IoError::DatabaseError(e.to_string()))?;

        let mut rows_vec = Vec::with_capacity(oxi_rows.len());

        for row in &oxi_rows {
            let mut obj = serde_json::Map::new();
            let col_count = row.column_count();
            let col_names = row.columns().to_vec();

            for i in 0..col_count {
                let col_name = col_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("column_{i}"));
                let json_val = match row.get_by_index(i) {
                    Some(OxiValue::Null) | None => Value::Null,
                    Some(OxiValue::Bool(b)) => Value::Bool(*b),
                    Some(OxiValue::I64(n)) => Value::Number((*n).into()),
                    Some(OxiValue::F64(f)) => Value::Number(
                        serde_json::Number::from_f64(*f).unwrap_or(serde_json::Number::from(0)),
                    ),
                    Some(OxiValue::Text(t)) => Value::String(t.clone()),
                    Some(OxiValue::Blob(b)) => Value::String(format!("<blob {} bytes>", b.len())),
                    Some(OxiValue::Json(j)) => {
                        serde_json::from_str(j).unwrap_or_else(|_| Value::String(j.clone()))
                    }
                    Some(OxiValue::Decimal(d)) => Value::String(d.clone()),
                    Some(OxiValue::Timestamp(ts)) => Value::Number((*ts).into()),
                    Some(OxiValue::Date(d)) => Value::Number((*d).into()),
                    Some(OxiValue::Time(t)) => Value::Number((*t).into()),
                    Some(OxiValue::Uuid(_)) => {
                        // Format UUID using Display impl on OxiValue.
                        Value::String(format!(
                            "{}",
                            row.get_by_index(i)
                                .expect("index checked above in outer match")
                        ))
                    }
                    Some(OxiValue::Array(_)) => Value::String(format!("{:?}", row.get_by_index(i))),
                    // Typed arrays carry the same element data as a plain array
                    // plus a nominal element type; represent them the same way.
                    Some(OxiValue::TypedArray { .. }) => {
                        Value::String(format!("{:?}", row.get_by_index(i)))
                    }
                };
                obj.insert(col_name, json_val);
            }
            rows_vec.push(Value::Object(obj));
        }

        self.rows = Some(rows_vec.into());
        Ok(())
    }

    #[cfg(not(feature = "sqlite"))]
    fn load_rows(&mut self) -> Result<()> {
        Err(IoError::Other(
            "DatabaseSource requires the `sqlite` feature".to_string(),
        ))
    }
}

impl DataSource for DatabaseSource {
    fn next_batch(&mut self, batch_size: usize) -> Result<Option<Vec<Value>>> {
        if self.exhausted {
            return Ok(None);
        }
        if self.rows.is_none() {
            self.load_rows()?;
        }
        let queue = self
            .rows
            .as_mut()
            .expect("rows must be populated after load_rows");
        if queue.is_empty() {
            self.exhausted = true;
            return Ok(None);
        }
        let batch: Vec<Value> = queue.drain(..batch_size.min(queue.len())).collect();
        Ok(Some(batch))
    }

    fn reset(&mut self) -> Result<()> {
        self.rows = None;
        self.exhausted = false;
        Ok(())
    }

    fn name(&self) -> &str {
        "database_source"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamSource  (Kafka-like channel-based source)
// ─────────────────────────────────────────────────────────────────────────────

/// A streaming, channel-based message source that mimics a Kafka consumer.
///
/// Messages are enqueued via `StreamSource::sender` from any thread and
/// consumed in batches via [`DataSource::next_batch`].
pub struct StreamSource {
    receiver: crossbeam_channel::Receiver<Value>,
    timeout: std::time::Duration,
    label: String,
}

/// Handle for sending messages into a [`StreamSource`].
pub type StreamSender = crossbeam_channel::Sender<Value>;

impl StreamSource {
    /// Create an unbounded channel-backed source.
    ///
    /// Returns `(source, sender)`.  Messages sent via `sender` become
    /// available through `source.next_batch`.
    pub fn new_unbounded() -> (Self, StreamSender) {
        let (tx, rx) = crossbeam_channel::unbounded();
        let source = Self {
            receiver: rx,
            timeout: std::time::Duration::from_millis(50),
            label: "stream_source".to_string(),
        };
        (source, tx)
    }

    /// Create a bounded channel-backed source.
    ///
    /// Senders will block once the channel is full (backpressure).
    pub fn new_bounded(capacity: usize) -> (Self, StreamSender) {
        let (tx, rx) = crossbeam_channel::bounded(capacity);
        let source = Self {
            receiver: rx,
            timeout: std::time::Duration::from_millis(50),
            label: "stream_source_bounded".to_string(),
        };
        (source, tx)
    }

    /// Set the receive timeout used when the channel is empty.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set a human-readable label.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.label = name.into();
        self
    }
}

impl DataSource for StreamSource {
    /// Collect up to `batch_size` messages.  Returns `None` only when the
    /// channel is both empty **and** all senders have been dropped.
    fn next_batch(&mut self, batch_size: usize) -> Result<Option<Vec<Value>>> {
        use crossbeam_channel::RecvTimeoutError;

        let mut batch = Vec::with_capacity(batch_size);

        // Block for the first message (or detect channel closure).
        match self.receiver.recv_timeout(self.timeout) {
            Ok(msg) => batch.push(msg),
            Err(RecvTimeoutError::Disconnected) => return Ok(None),
            Err(RecvTimeoutError::Timeout) => return Ok(Some(batch)), // empty batch OK
        }

        // Drain non-blockingly up to batch_size - 1 more items.
        while batch.len() < batch_size {
            match self.receiver.try_recv() {
                Ok(msg) => batch.push(msg),
                Err(_) => break,
            }
        }

        Ok(Some(batch))
    }

    fn name(&self) -> &str {
        &self.label
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GeneratorSource
// ─────────────────────────────────────────────────────────────────────────────

/// Produce data from a user-supplied closure.
///
/// The generator closure is called repeatedly; when it returns `None` the
/// source is exhausted.
pub struct GeneratorSource<F>
where
    F: FnMut() -> Option<Value> + Send,
{
    generator: F,
    label: String,
    exhausted: bool,
}

impl<F> GeneratorSource<F>
where
    F: FnMut() -> Option<Value> + Send,
{
    /// Create a new `GeneratorSource` with the given closure.
    pub fn new(generator: F) -> Self {
        Self {
            generator,
            label: "generator_source".to_string(),
            exhausted: false,
        }
    }

    /// Attach a human-readable label.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.label = name.into();
        self
    }
}

impl<F> DataSource for GeneratorSource<F>
where
    F: FnMut() -> Option<Value> + Send,
{
    fn next_batch(&mut self, batch_size: usize) -> Result<Option<Vec<Value>>> {
        if self.exhausted {
            return Ok(None);
        }
        let mut batch = Vec::with_capacity(batch_size);
        while batch.len() < batch_size {
            match (self.generator)() {
                Some(v) => batch.push(v),
                None => {
                    self.exhausted = true;
                    break;
                }
            }
        }
        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(batch))
        }
    }

    fn reset(&mut self) -> Result<()> {
        // Cannot reset a closure-based generator by default.
        Err(IoError::Other(
            "GeneratorSource does not support reset".to_string(),
        ))
    }

    fn name(&self) -> &str {
        &self.label
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SourceChain – iterate a source to completion
// ─────────────────────────────────────────────────────────────────────────────

/// Convenience: drain an entire [`DataSource`] into a `Vec<Value>`.
pub fn drain_source(source: &mut dyn DataSource, batch_size: usize) -> Result<Vec<Value>> {
    let mut all = Vec::new();
    while let Some(batch) = source.next_batch(batch_size)? {
        all.extend(batch);
    }
    Ok(all)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;

    fn temp_path(suffix: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("scirs2_io_test_{suffix}_{}", std::process::id()));
        p
    }

    #[test]
    fn test_generator_source() {
        let mut counter = 0u64;
        let mut src = GeneratorSource::new(move || {
            if counter < 5 {
                let v = json!(counter);
                counter += 1;
                Some(v)
            } else {
                None
            }
        });

        let batch = src.next_batch(3).unwrap().unwrap();
        assert_eq!(batch.len(), 3);
        let rest = drain_source(&mut src, 10).unwrap();
        assert_eq!(rest.len(), 2);
        assert!(src.next_batch(10).unwrap().is_none());
    }

    #[test]
    fn test_stream_source_basic() {
        let (mut src, tx) = StreamSource::new_unbounded();
        tx.send(json!({"id": 1})).unwrap();
        tx.send(json!({"id": 2})).unwrap();
        drop(tx); // signal EOF

        let all = drain_source(&mut src, 10).unwrap();
        // We expect at least 2 records (timing-dependent for the empty-batch path)
        let total: usize = all.len() + drain_source(&mut src, 10).unwrap_or_default().len();
        assert!(total >= 2 || all.len() >= 2);
    }

    #[test]
    fn test_file_source_jsonl() {
        let path = temp_path("jsonl");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, r#"{{"a":1,"b":"x"}}"#).unwrap();
            writeln!(f, r#"{{"a":2,"b":"y"}}"#).unwrap();
            writeln!(f, r#"{{"a":3,"b":"z"}}"#).unwrap();
        }
        let mut src = FileSource::with_format(&path, FileSourceFormat::JsonLines);
        let all = drain_source(&mut src, 10).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0]["a"], json!(1));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_file_source_csv() {
        let path = temp_path("csv");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "name,score").unwrap();
            writeln!(f, "alice,90").unwrap();
            writeln!(f, "bob,85").unwrap();
        }
        let mut src = FileSource::with_format(&path, FileSourceFormat::Csv);
        let all = drain_source(&mut src, 10).unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0]["name"], json!("alice"));
        assert_eq!(all[0]["score"], json!(90));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_file_source_json_array() {
        let path = temp_path("json");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            write!(f, r#"[{{"x":1}},{{"x":2}},{{"x":3}}]"#).unwrap();
        }
        let mut src = FileSource::with_format(&path, FileSourceFormat::Json);
        // Read in two batches of 2.
        let b1 = src.next_batch(2).unwrap().unwrap();
        assert_eq!(b1.len(), 2);
        let b2 = src.next_batch(2).unwrap().unwrap();
        assert_eq!(b2.len(), 1);
        assert!(src.next_batch(2).unwrap().is_none());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_file_source_batching() {
        let path = temp_path("jsonl2");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            for i in 0..10 {
                writeln!(f, r#"{{"i":{i}}}"#).unwrap();
            }
        }
        let mut src = FileSource::with_format(&path, FileSourceFormat::JsonLines);
        let b1 = src.next_batch(4).unwrap().unwrap();
        assert_eq!(b1.len(), 4);
        let b2 = src.next_batch(4).unwrap().unwrap();
        assert_eq!(b2.len(), 4);
        let b3 = src.next_batch(4).unwrap().unwrap();
        assert_eq!(b3.len(), 2);
        assert!(src.next_batch(4).unwrap().is_none());
        std::fs::remove_file(&path).ok();
    }
}
