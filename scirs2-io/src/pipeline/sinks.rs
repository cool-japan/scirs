//! Data sink abstractions for the streaming pipeline.
//!
//! Every sink implements the [`DataSink`] trait which accepts record batches
//! of `serde_json::Value`.  Concrete implementations:
//!
//! | Sink                | Description                                        |
//! |---------------------|----------------------------------------------------|
//! | [`FileSink`]        | Write records to CSV / JSON / JSONL                |
//! | [`MemorySink`]      | Accumulate records in memory                       |
//! | [`AggregationSink`] | Running statistics (count, sum, mean, min, max)    |

#![allow(missing_docs)]

use crate::error::{IoError, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

// ─────────────────────────────────────────────────────────────────────────────
// Core trait
// ─────────────────────────────────────────────────────────────────────────────

/// A push-based data sink that consumes record batches.
pub trait DataSink: Send {
    /// Receive a batch of records.
    fn write_batch(&mut self, records: Vec<Value>) -> Result<()>;

    /// Flush internal buffers and close the sink.
    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    /// Human-readable name for diagnostics.
    fn name(&self) -> &str {
        "unknown"
    }

    /// Total number of records written so far.
    fn records_written(&self) -> usize {
        0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FileSink
// ─────────────────────────────────────────────────────────────────────────────

/// Output format for [`FileSink`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileSinkFormat {
    /// Comma-separated values (flat objects only; nested values are JSON-encoded)
    Csv,
    /// Newline-delimited JSON (one record per line)
    JsonLines,
    /// Single JSON array written on `close()`
    Json,
    /// Auto-detect from file extension
    Auto,
}

/// Write records to a file.
pub struct FileSink {
    path: PathBuf,
    format: FileSinkFormat,
    writer: Option<BufWriter<File>>,
    records: usize,
    /// JSON-array buffer, used when `format == Json`.
    json_buf: Vec<Value>,
    /// CSV header; written lazily on first batch.
    csv_header_written: bool,
}

impl FileSink {
    /// Create a `FileSink` with automatic format detection.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self::with_format(path, FileSinkFormat::Auto)
    }

    /// Create a `FileSink` with an explicit format.
    pub fn with_format(path: impl AsRef<Path>, format: FileSinkFormat) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            format,
            writer: None,
            records: 0,
            json_buf: Vec::new(),
            csv_header_written: false,
        }
    }

    fn resolve_format(&self) -> FileSinkFormat {
        if self.format != FileSinkFormat::Auto {
            return self.format;
        }
        match self
            .path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref()
        {
            Some("csv") => FileSinkFormat::Csv,
            Some("jsonl") | Some("ndjson") => FileSinkFormat::JsonLines,
            Some("json") => FileSinkFormat::Json,
            _ => FileSinkFormat::JsonLines,
        }
    }

    fn ensure_open(&mut self) -> Result<()> {
        if self.writer.is_some() {
            return Ok(());
        }
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)
            .map_err(IoError::Io)?;
        self.writer = Some(BufWriter::new(file));
        Ok(())
    }

    fn write_csv_batch(&mut self, records: Vec<Value>) -> Result<()> {
        self.ensure_open()?;
        let writer = self.writer.as_mut().expect("writer must be open");

        for record in &records {
            if let Value::Object(map) = record {
                // Write header on first record.
                if !self.csv_header_written {
                    let keys: Vec<&str> = map.keys().map(|k| k.as_str()).collect();
                    writeln!(writer, "{}", keys.join(",")).map_err(IoError::Io)?;
                    self.csv_header_written = true;
                }
                let values: Vec<String> = map
                    .values()
                    .map(|v| match v {
                        Value::String(s) => {
                            if s.contains(',') || s.contains('"') || s.contains('\n') {
                                format!("\"{}\"", s.replace('"', "\"\""))
                            } else {
                                s.clone()
                            }
                        }
                        Value::Null => String::new(),
                        other => other.to_string(),
                    })
                    .collect();
                writeln!(writer, "{}", values.join(",")).map_err(IoError::Io)?;
                self.records += 1;
            }
        }
        Ok(())
    }

    fn write_jsonl_batch(&mut self, records: Vec<Value>) -> Result<()> {
        self.ensure_open()?;
        let writer = self.writer.as_mut().expect("writer must be open");
        for record in records {
            serde_json::to_writer(&mut *writer, &record)
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
            writeln!(writer).map_err(IoError::Io)?;
            self.records += 1;
        }
        Ok(())
    }
}

impl DataSink for FileSink {
    fn write_batch(&mut self, records: Vec<Value>) -> Result<()> {
        match self.resolve_format() {
            FileSinkFormat::Csv => self.write_csv_batch(records),
            FileSinkFormat::Json => {
                self.records += records.len();
                self.json_buf.extend(records);
                Ok(())
            }
            _ => self.write_jsonl_batch(records),
        }
    }

    fn close(&mut self) -> Result<()> {
        if self.resolve_format() == FileSinkFormat::Json {
            self.ensure_open()?;
            let writer = self.writer.as_mut().expect("writer must be open");
            serde_json::to_writer_pretty(&mut *writer, &Value::Array(self.json_buf.clone()))
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
        if let Some(mut w) = self.writer.take() {
            w.flush().map_err(IoError::Io)?;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "file_sink"
    }

    fn records_written(&self) -> usize {
        self.records
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MemorySink
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulate all records in memory for later inspection or further processing.
#[derive(Default)]
pub struct MemorySink {
    records: Vec<Value>,
}

impl MemorySink {
    /// Create an empty `MemorySink`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a reference to all accumulated records.
    pub fn records(&self) -> &[Value] {
        &self.records
    }

    /// Consume the sink and return the accumulated records.
    pub fn into_records(self) -> Vec<Value> {
        self.records
    }

    /// Clear all accumulated records.
    pub fn clear(&mut self) {
        self.records.clear();
    }
}

impl DataSink for MemorySink {
    fn write_batch(&mut self, records: Vec<Value>) -> Result<()> {
        self.records.extend(records);
        Ok(())
    }

    fn name(&self) -> &str {
        "memory_sink"
    }

    fn records_written(&self) -> usize {
        self.records.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AggregationSink – running statistics per numeric field
// ─────────────────────────────────────────────────────────────────────────────

/// Running statistics for a single numeric field.
#[derive(Debug, Clone)]
pub struct FieldStats {
    /// Number of non-null observations.
    pub count: u64,
    /// Sum of all values.
    pub sum: f64,
    /// Sum of squared values (for variance computation).
    pub sum_sq: f64,
    /// Minimum value seen.
    pub min: f64,
    /// Maximum value seen.
    pub max: f64,
}

impl FieldStats {
    fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn update(&mut self, v: f64) {
        self.count += 1;
        self.sum += v;
        self.sum_sq += v * v;
        if v < self.min {
            self.min = v;
        }
        if v > self.max {
            self.max = v;
        }
    }

    /// Arithmetic mean, or `None` if no values have been observed.
    pub fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.sum / self.count as f64)
        }
    }

    /// Population variance, or `None` if fewer than 1 value observed.
    pub fn variance(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let mean = self.sum / self.count as f64;
        Some(self.sum_sq / self.count as f64 - mean * mean)
    }

    /// Population standard deviation, or `None` if fewer than 1 value.
    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }
}

/// A sink that computes running statistics over all numeric fields without
/// storing the raw records.
pub struct AggregationSink {
    /// Per-field statistics.
    stats: HashMap<String, FieldStats>,
    /// Total number of records seen (including records with no numeric fields).
    total_records: usize,
    label: String,
}

impl AggregationSink {
    /// Create a new `AggregationSink`.
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            total_records: 0,
            label: "aggregation_sink".to_string(),
        }
    }

    /// Attach a human-readable label.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.label = name.into();
        self
    }

    /// Return statistics for a specific field, if any data has been seen.
    pub fn field_stats(&self, field: &str) -> Option<&FieldStats> {
        self.stats.get(field)
    }

    /// Return all per-field statistics.
    pub fn all_stats(&self) -> &HashMap<String, FieldStats> {
        &self.stats
    }

    /// Total number of records processed.
    pub fn total_records(&self) -> usize {
        self.total_records
    }

    /// Produce a human-readable summary.
    pub fn summary(&self) -> String {
        let mut lines = vec![format!("Total records: {}", self.total_records)];
        let mut fields: Vec<&String> = self.stats.keys().collect();
        fields.sort();
        for field in fields {
            let s = &self.stats[field];
            lines.push(format!(
                "  {}: count={} sum={:.3} mean={:.3} min={:.3} max={:.3} stddev={:.3}",
                field,
                s.count,
                s.sum,
                s.mean().unwrap_or(f64::NAN),
                s.min,
                s.max,
                s.std_dev().unwrap_or(f64::NAN),
            ));
        }
        lines.join("\n")
    }
}

impl Default for AggregationSink {
    fn default() -> Self {
        Self::new()
    }
}

impl DataSink for AggregationSink {
    fn write_batch(&mut self, records: Vec<Value>) -> Result<()> {
        for record in &records {
            self.total_records += 1;
            if let Value::Object(map) = record {
                for (key, val) in map {
                    let numeric = match val {
                        Value::Number(n) => n.as_f64(),
                        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
                        _ => None,
                    };
                    if let Some(v) = numeric {
                        self.stats
                            .entry(key.clone())
                            .or_insert_with(FieldStats::new)
                            .update(v);
                    }
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        &self.label
    }

    fn records_written(&self) -> usize {
        self.total_records
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-sink: fan-out to several sinks simultaneously
// ─────────────────────────────────────────────────────────────────────────────

/// Fan records out to multiple sinks simultaneously.
pub struct MultiSink {
    sinks: Vec<Box<dyn DataSink>>,
}

impl MultiSink {
    /// Create a `MultiSink` with an initial list of sinks.
    pub fn new(sinks: Vec<Box<dyn DataSink>>) -> Self {
        Self { sinks }
    }

    /// Add another sink.
    pub fn add_sink(&mut self, sink: Box<dyn DataSink>) {
        self.sinks.push(sink);
    }
}

impl DataSink for MultiSink {
    fn write_batch(&mut self, records: Vec<Value>) -> Result<()> {
        for sink in &mut self.sinks {
            sink.write_batch(records.clone())?;
        }
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        for sink in &mut self.sinks {
            sink.close()?;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "multi_sink"
    }

    fn records_written(&self) -> usize {
        self.sinks.first().map(|s| s.records_written()).unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    fn temp_path(suffix: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("scirs2_sink_test_{suffix}_{}", std::process::id()));
        p
    }

    #[test]
    fn test_memory_sink() {
        let mut sink = MemorySink::new();
        sink.write_batch(vec![json!({"a": 1}), json!({"a": 2})])
            .unwrap();
        sink.write_batch(vec![json!({"a": 3})]).unwrap();
        assert_eq!(sink.records_written(), 3);
        assert_eq!(sink.records()[1]["a"], json!(2));
    }

    #[test]
    fn test_aggregation_sink_basic() {
        let mut sink = AggregationSink::new();
        sink.write_batch(vec![
            json!({"x": 1.0, "y": 10.0}),
            json!({"x": 2.0, "y": 20.0}),
            json!({"x": 3.0, "y": 30.0}),
        ])
        .unwrap();
        let xs = sink.field_stats("x").unwrap();
        assert_eq!(xs.count, 3);
        assert!((xs.sum - 6.0).abs() < 1e-9);
        assert!((xs.mean().unwrap() - 2.0).abs() < 1e-9);
        assert!((xs.min - 1.0).abs() < 1e-9);
        assert!((xs.max - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_aggregation_sink_streaming() {
        let mut sink = AggregationSink::new();
        for i in 1..=100 {
            sink.write_batch(vec![json!({"v": i})]).unwrap();
        }
        let stats = sink.field_stats("v").unwrap();
        assert_eq!(stats.count, 100);
        assert!((stats.mean().unwrap() - 50.5).abs() < 1e-9);
    }

    #[test]
    fn test_file_sink_jsonl() {
        let path = temp_path("jsonl");
        let mut sink = FileSink::with_format(&path, FileSinkFormat::JsonLines);
        sink.write_batch(vec![json!({"k": 1}), json!({"k": 2})])
            .unwrap();
        sink.close().unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_file_sink_csv() {
        let path = temp_path("csv");
        let mut sink = FileSink::with_format(&path, FileSinkFormat::Csv);
        sink.write_batch(vec![
            json!({"name": "alice", "score": 90}),
            json!({"name": "bob",   "score": 85}),
        ])
        .unwrap();
        sink.close().unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        // header + 2 data lines
        assert_eq!(lines.len(), 3);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_file_sink_json_array() {
        let path = temp_path("json");
        let mut sink = FileSink::with_format(&path, FileSinkFormat::Json);
        sink.write_batch(vec![json!(1), json!(2), json!(3)])
            .unwrap();
        sink.close().unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let arr: Vec<serde_json::Value> = serde_json::from_str(&content).unwrap();
        assert_eq!(arr.len(), 3);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_multi_sink() {
        let mem1 = Box::new(MemorySink::new()) as Box<dyn DataSink>;
        let mem2 = Box::new(MemorySink::new()) as Box<dyn DataSink>;
        let mut multi = MultiSink::new(vec![mem1, mem2]);
        multi.write_batch(vec![json!({"x": 42})]).unwrap();
        // Both sinks receive the record – verify via records_written.
        assert_eq!(multi.records_written(), 1);
    }
}
