//! Universal data reader with auto-format detection
//!
//! Provides a unified interface for reading data files regardless of format.
//! The reader auto-detects the file format using [`crate::format_detect`] and
//! dispatches to the appropriate format-specific reader.
//!
//! All data is returned as a [`DataTable`], a format-agnostic columnar representation
//! with metadata support.
//!
//! # Supported Formats
//!
//! - CSV (with configurable delimiter/header)
//! - Arrow IPC (columnar binary)
//! - NetCDF Classic (via `netcdf_lite`)
//! - HDF5 lite (pure Rust reader)
//! - NPY/NPZ (NumPy binary)
//! - Matrix Market (sparse/dense)
//! - JSON (array of objects)
//! - WAV (audio data as numeric columns)
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_io::universal_reader::{read_data, ReadOptions, DataTable};
//!
//! // Auto-detect and read
//! let table = read_data("measurements.csv", None)?;
//! println!("Columns: {:?}", table.column_names());
//! println!("Rows: {}", table.num_rows());
//!
//! // Access column data
//! if let Some(col) = table.column("temperature") {
//!     let values = col.as_f64();
//!     println!("Temperature values: {} entries", values.len());
//! }
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```

use crate::error::{IoError, Result};
use crate::format_detect::{detect_format, DataFormat};
use std::collections::HashMap;
use std::path::Path;

// =====================================================================
// Core data types
// =====================================================================

/// A column of data in the unified table representation
#[derive(Debug, Clone)]
pub enum DataColumn {
    /// 32-bit integer column
    Int32(Vec<i32>),
    /// 64-bit integer column
    Int64(Vec<i64>),
    /// 32-bit float column
    Float32(Vec<f32>),
    /// 64-bit float column
    Float64(Vec<f64>),
    /// String column
    String(Vec<String>),
    /// Boolean column
    Boolean(Vec<bool>),
    /// Raw bytes column (for opaque data)
    Bytes(Vec<Vec<u8>>),
}

impl DataColumn {
    /// Number of elements
    pub fn len(&self) -> usize {
        match self {
            Self::Int32(v) => v.len(),
            Self::Int64(v) => v.len(),
            Self::Float32(v) => v.len(),
            Self::Float64(v) => v.len(),
            Self::String(v) => v.len(),
            Self::Boolean(v) => v.len(),
            Self::Bytes(v) => v.len(),
        }
    }

    /// Whether the column is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert to f64 values (non-numeric types return empty vec)
    pub fn as_f64(&self) -> Vec<f64> {
        match self {
            Self::Int32(v) => v.iter().map(|x| *x as f64).collect(),
            Self::Int64(v) => v.iter().map(|x| *x as f64).collect(),
            Self::Float32(v) => v.iter().map(|x| *x as f64).collect(),
            Self::Float64(v) => v.clone(),
            Self::Boolean(v) => v.iter().map(|x| if *x { 1.0 } else { 0.0 }).collect(),
            _ => Vec::new(),
        }
    }

    /// Convert to string values
    pub fn as_strings(&self) -> Vec<String> {
        match self {
            Self::Int32(v) => v.iter().map(|x| x.to_string()).collect(),
            Self::Int64(v) => v.iter().map(|x| x.to_string()).collect(),
            Self::Float32(v) => v.iter().map(|x| x.to_string()).collect(),
            Self::Float64(v) => v.iter().map(|x| x.to_string()).collect(),
            Self::String(v) => v.clone(),
            Self::Boolean(v) => v.iter().map(|x| x.to_string()).collect(),
            Self::Bytes(v) => v.iter().map(|b| format!("<{} bytes>", b.len())).collect(),
        }
    }

    /// Type name for display
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Int32(_) => "int32",
            Self::Int64(_) => "int64",
            Self::Float32(_) => "float32",
            Self::Float64(_) => "float64",
            Self::String(_) => "string",
            Self::Boolean(_) => "bool",
            Self::Bytes(_) => "bytes",
        }
    }
}

/// A unified data table with named columns and metadata
#[derive(Debug, Clone)]
pub struct DataTable {
    /// Column names in order
    column_names: Vec<String>,
    /// Column data keyed by name
    columns: HashMap<String, DataColumn>,
    /// File-level metadata
    metadata: HashMap<String, String>,
    /// Source format that was read
    source_format: DataFormat,
}

impl DataTable {
    /// Create a new empty data table
    pub fn new(source_format: DataFormat) -> Self {
        Self {
            column_names: Vec::new(),
            columns: HashMap::new(),
            metadata: HashMap::new(),
            source_format,
        }
    }

    /// Add a column to the table
    pub fn add_column(&mut self, name: &str, data: DataColumn) {
        if !self.columns.contains_key(name) {
            self.column_names.push(name.to_string());
        }
        self.columns.insert(name.to_string(), data);
    }

    /// Add metadata
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get column names in order
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Get a column by name
    pub fn column(&self, name: &str) -> Option<&DataColumn> {
        self.columns.get(name)
    }

    /// Get a column by index
    pub fn column_by_index(&self, index: usize) -> Option<&DataColumn> {
        self.column_names
            .get(index)
            .and_then(|name| self.columns.get(name))
    }

    /// Number of columns
    pub fn num_columns(&self) -> usize {
        self.column_names.len()
    }

    /// Number of rows (from first column, or 0 if empty)
    pub fn num_rows(&self) -> usize {
        self.column_names
            .first()
            .and_then(|name| self.columns.get(name))
            .map_or(0, |col| col.len())
    }

    /// Get metadata value
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Get all metadata
    pub fn all_metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Source format
    pub fn source_format(&self) -> DataFormat {
        self.source_format
    }

    /// Whether the table is empty
    pub fn is_empty(&self) -> bool {
        self.num_rows() == 0
    }

    /// Get a summary of the table for display
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "DataTable: {} columns x {} rows (from {})",
            self.num_columns(),
            self.num_rows(),
            self.source_format
        ));
        for name in &self.column_names {
            if let Some(col) = self.columns.get(name) {
                lines.push(format!(
                    "  {}: {} ({} values)",
                    name,
                    col.type_name(),
                    col.len()
                ));
            }
        }
        lines.join("\n")
    }

    /// Select a subset of columns
    pub fn select_columns(&self, names: &[&str]) -> Result<DataTable> {
        let mut table = DataTable::new(self.source_format);
        for &name in names {
            let col = self
                .column(name)
                .ok_or_else(|| IoError::NotFound(format!("Column '{name}' not found")))?;
            table.add_column(name, col.clone());
        }
        table.metadata = self.metadata.clone();
        Ok(table)
    }

    /// Slice rows
    pub fn slice_rows(&self, start: usize, end: usize) -> DataTable {
        let mut table = DataTable::new(self.source_format);
        for name in &self.column_names {
            if let Some(col) = self.columns.get(name) {
                let sliced = slice_column(col, start, end);
                table.add_column(name, sliced);
            }
        }
        table.metadata = self.metadata.clone();
        table
    }
}

/// Options for reading data
#[derive(Debug, Clone, Default)]
pub struct ReadOptions {
    /// Override auto-detected format
    pub format: Option<DataFormat>,
    /// Maximum number of rows to read (None = all)
    pub max_rows: Option<usize>,
    /// Specific columns to read (None = all)
    pub columns: Option<Vec<String>>,
    /// CSV delimiter override
    pub csv_delimiter: Option<char>,
    /// Whether CSV has header row
    pub csv_has_header: Option<bool>,
    /// HDF5 dataset path to read
    pub hdf5_dataset: Option<String>,
    /// NetCDF variable name to read
    pub netcdf_variable: Option<String>,
}

// =====================================================================
// Main read functions
// =====================================================================

/// Read a data file, auto-detecting the format.
///
/// Returns a [`DataTable`] with columns and metadata regardless of the
/// source format. The detected format is stored in `table.source_format()`.
pub fn read_data<P: AsRef<Path>>(path: P, options: Option<ReadOptions>) -> Result<DataTable> {
    let path = path.as_ref();
    let opts = options.unwrap_or_default();

    let format = if let Some(fmt) = opts.format {
        fmt
    } else {
        detect_format(path)?
    };

    match format {
        DataFormat::Csv => read_csv_to_table(path, &opts),
        DataFormat::ArrowIpc => read_arrow_to_table(path, &opts),
        DataFormat::Json => read_json_to_table(path, &opts),
        DataFormat::Wav => read_wav_to_table(path, &opts),
        DataFormat::Npy => read_npy_to_table(path, &opts),
        DataFormat::MatrixMarket => read_mtx_to_table(path, &opts),
        DataFormat::Hdf5 => read_hdf5_to_table(path, &opts),
        DataFormat::NetCdf => read_netcdf_to_table(path, &opts),
        _ => Err(IoError::UnsupportedFormat(format!(
            "Universal reader does not support {} format yet",
            format.name()
        ))),
    }
}

// =====================================================================
// Streaming reader interface
// =====================================================================

/// A streaming reader that yields chunks of data
pub struct StreamingReader {
    /// Chunks of data already read
    chunks: Vec<DataTable>,
    /// Current chunk index
    current: usize,
}

impl StreamingReader {
    /// Create a streaming reader for a file
    pub fn open<P: AsRef<Path>>(
        path: P,
        chunk_size: usize,
        options: Option<ReadOptions>,
    ) -> Result<Self> {
        // For now, read the full table and split into chunks
        let full = read_data(path, options)?;
        let total_rows = full.num_rows();

        let mut chunks = Vec::new();
        let mut offset = 0;
        while offset < total_rows {
            let end = (offset + chunk_size).min(total_rows);
            chunks.push(full.slice_rows(offset, end));
            offset = end;
        }

        if chunks.is_empty() {
            chunks.push(full);
        }

        Ok(Self { chunks, current: 0 })
    }

    /// Read the next chunk (returns None when exhausted)
    pub fn next_chunk(&mut self) -> Option<&DataTable> {
        if self.current < self.chunks.len() {
            let chunk = &self.chunks[self.current];
            self.current += 1;
            Some(chunk)
        } else {
            None
        }
    }

    /// Reset to the beginning
    pub fn reset(&mut self) {
        self.current = 0;
    }

    /// Total number of chunks
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Total rows across all chunks
    pub fn total_rows(&self) -> usize {
        self.chunks.iter().map(|c| c.num_rows()).sum()
    }
}

// =====================================================================
// Format-specific readers
// =====================================================================

/// Read CSV into DataTable
fn read_csv_to_table(path: &Path, opts: &ReadOptions) -> Result<DataTable> {
    let config = crate::csv::CsvReaderConfig {
        has_header: opts.csv_has_header.unwrap_or(true),
        delimiter: opts.csv_delimiter.unwrap_or(','),
        ..Default::default()
    };

    let (headers, data) = crate::csv::read_csv(path, Some(config))?;

    let mut table = DataTable::new(DataFormat::Csv);
    table.set_metadata("source_file", &path.display().to_string());

    if headers.is_empty() {
        return Ok(table);
    }

    let nrows = data.nrows();
    let ncols = data.ncols();

    // Determine column types by trying to parse values
    for (col_idx, header) in headers.iter().enumerate() {
        if col_idx >= ncols {
            break;
        }

        let col_name = if header.is_empty() {
            format!("column_{col_idx}")
        } else {
            header.clone()
        };

        // Check if requested columns filter applies
        if let Some(ref wanted) = opts.columns {
            if !wanted.iter().any(|w| w == &col_name) {
                continue;
            }
        }

        let max = opts.max_rows.unwrap_or(usize::MAX);
        let row_count = nrows.min(max);

        let values: Vec<&str> = (0..row_count)
            .map(|r| data[[r, col_idx]].as_str())
            .collect();

        // Try int64 first
        let as_int: std::result::Result<Vec<i64>, _> = values
            .iter()
            .map(|s: &&str| s.trim().parse::<i64>())
            .collect();

        if let Ok(ints) = as_int {
            table.add_column(&col_name, DataColumn::Int64(ints));
            continue;
        }

        // Try f64
        let as_float: std::result::Result<Vec<f64>, _> = values
            .iter()
            .map(|s: &&str| s.trim().parse::<f64>())
            .collect();

        if let Ok(floats) = as_float {
            table.add_column(&col_name, DataColumn::Float64(floats));
            continue;
        }

        // Try boolean
        let as_bool: std::result::Result<Vec<bool>, _> = values
            .iter()
            .map(|s: &&str| match s.trim().to_lowercase().as_str() {
                "true" | "1" | "yes" => Ok(true),
                "false" | "0" | "no" => Ok(false),
                _ => Err(()),
            })
            .collect();

        if let Ok(bools) = as_bool {
            table.add_column(&col_name, DataColumn::Boolean(bools));
            continue;
        }

        // Default to string
        let strings: Vec<String> = values.iter().map(|s: &&str| s.to_string()).collect();
        table.add_column(&col_name, DataColumn::String(strings));
    }

    Ok(table)
}

/// Read Arrow IPC into DataTable
fn read_arrow_to_table(path: &Path, opts: &ReadOptions) -> Result<DataTable> {
    let (schema, batches) = crate::arrow_ipc::read_arrow_ipc_file(path)?;

    let mut table = DataTable::new(DataFormat::ArrowIpc);
    table.set_metadata("source_file", &path.display().to_string());

    for (key, val) in &schema.metadata {
        table.set_metadata(key, val);
    }

    // Merge all batches into single columns
    for (idx, field) in schema.fields.iter().enumerate() {
        if let Some(ref wanted) = opts.columns {
            if !wanted.iter().any(|w| w == &field.name) {
                continue;
            }
        }

        let col = merge_arrow_columns(&batches, idx, &field.dtype, opts.max_rows)?;
        table.add_column(&field.name, col);
    }

    Ok(table)
}

/// Merge Arrow columns across batches
fn merge_arrow_columns(
    batches: &[crate::arrow_ipc::RecordBatch],
    col_idx: usize,
    dtype: &crate::arrow_ipc::ArrowDataType,
    max_rows: Option<usize>,
) -> Result<DataColumn> {
    let max = max_rows.unwrap_or(usize::MAX);

    match dtype {
        crate::arrow_ipc::ArrowDataType::Int32 => {
            let mut values = Vec::new();
            for batch in batches {
                if values.len() >= max {
                    break;
                }
                if let Some(crate::arrow_ipc::ArrowColumn::Int32(v)) = batch.column(col_idx) {
                    let remaining = max - values.len();
                    values.extend_from_slice(&v[..v.len().min(remaining)]);
                }
            }
            Ok(DataColumn::Int32(values))
        }
        crate::arrow_ipc::ArrowDataType::Int64 => {
            let mut values = Vec::new();
            for batch in batches {
                if values.len() >= max {
                    break;
                }
                if let Some(crate::arrow_ipc::ArrowColumn::Int64(v)) = batch.column(col_idx) {
                    let remaining = max - values.len();
                    values.extend_from_slice(&v[..v.len().min(remaining)]);
                }
            }
            Ok(DataColumn::Int64(values))
        }
        crate::arrow_ipc::ArrowDataType::Float32 => {
            let mut values = Vec::new();
            for batch in batches {
                if values.len() >= max {
                    break;
                }
                if let Some(crate::arrow_ipc::ArrowColumn::Float32(v)) = batch.column(col_idx) {
                    let remaining = max - values.len();
                    values.extend_from_slice(&v[..v.len().min(remaining)]);
                }
            }
            Ok(DataColumn::Float32(values))
        }
        crate::arrow_ipc::ArrowDataType::Float64 => {
            let mut values = Vec::new();
            for batch in batches {
                if values.len() >= max {
                    break;
                }
                if let Some(crate::arrow_ipc::ArrowColumn::Float64(v)) = batch.column(col_idx) {
                    let remaining = max - values.len();
                    values.extend_from_slice(&v[..v.len().min(remaining)]);
                }
            }
            Ok(DataColumn::Float64(values))
        }
        crate::arrow_ipc::ArrowDataType::Utf8 => {
            let mut values = Vec::new();
            for batch in batches {
                if values.len() >= max {
                    break;
                }
                if let Some(crate::arrow_ipc::ArrowColumn::Utf8(v)) = batch.column(col_idx) {
                    let remaining = max - values.len();
                    values.extend_from_slice(&v[..v.len().min(remaining)]);
                }
            }
            Ok(DataColumn::String(values))
        }
        crate::arrow_ipc::ArrowDataType::Boolean => {
            let mut values = Vec::new();
            for batch in batches {
                if values.len() >= max {
                    break;
                }
                if let Some(crate::arrow_ipc::ArrowColumn::Boolean(v)) = batch.column(col_idx) {
                    let remaining = max - values.len();
                    values.extend_from_slice(&v[..v.len().min(remaining)]);
                }
            }
            Ok(DataColumn::Boolean(values))
        }
    }
}

/// Read JSON (array of objects) into DataTable
fn read_json_to_table(path: &Path, opts: &ReadOptions) -> Result<DataTable> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| IoError::FileError(format!("Cannot read '{}': {e}", path.display())))?;

    let parsed: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| IoError::FormatError(format!("JSON parse error: {e}")))?;

    let mut table = DataTable::new(DataFormat::Json);
    table.set_metadata("source_file", &path.display().to_string());

    let arr = match &parsed {
        serde_json::Value::Array(a) => a,
        serde_json::Value::Object(_) => {
            // Single object: wrap in array
            return read_json_single_object(&parsed, &mut table, opts);
        }
        _ => {
            return Err(IoError::FormatError(
                "Expected JSON array or object".to_string(),
            ));
        }
    };

    if arr.is_empty() {
        return Ok(table);
    }

    // Collect all field names from first object
    let field_names: Vec<String> = if let Some(serde_json::Value::Object(obj)) = arr.first() {
        obj.keys().cloned().collect()
    } else {
        // Array of scalars: single column
        let values: Vec<f64> = arr
            .iter()
            .take(opts.max_rows.unwrap_or(usize::MAX))
            .filter_map(|v| v.as_f64())
            .collect();
        table.add_column("value", DataColumn::Float64(values));
        return Ok(table);
    };

    let max = opts.max_rows.unwrap_or(usize::MAX);

    for field_name in &field_names {
        if let Some(ref wanted) = opts.columns {
            if !wanted.iter().any(|w| w == field_name) {
                continue;
            }
        }

        // Collect values for this field
        let values: Vec<&serde_json::Value> = arr
            .iter()
            .take(max)
            .filter_map(|item| item.as_object().and_then(|obj| obj.get(field_name)))
            .collect();

        // Determine type from values
        let all_int = values
            .iter()
            .all(|v| v.is_i64() || v.is_u64() || v.is_null());
        let all_float = values.iter().all(|v| v.is_number() || v.is_null());
        let all_bool = values.iter().all(|v| v.is_boolean() || v.is_null());

        if all_bool && !values.is_empty() {
            let bools: Vec<bool> = values
                .iter()
                .map(|v| v.as_bool().unwrap_or(false))
                .collect();
            table.add_column(field_name, DataColumn::Boolean(bools));
        } else if all_int && !values.is_empty() {
            let ints: Vec<i64> = values.iter().map(|v| v.as_i64().unwrap_or(0)).collect();
            table.add_column(field_name, DataColumn::Int64(ints));
        } else if all_float && !values.is_empty() {
            let floats: Vec<f64> = values.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
            table.add_column(field_name, DataColumn::Float64(floats));
        } else {
            let strings: Vec<String> = values
                .iter()
                .map(|v| match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .collect();
            table.add_column(field_name, DataColumn::String(strings));
        }
    }

    Ok(table)
}

/// Read a single JSON object as a one-row table
fn read_json_single_object(
    obj: &serde_json::Value,
    table: &mut DataTable,
    _opts: &ReadOptions,
) -> Result<DataTable> {
    if let serde_json::Value::Object(map) = obj {
        for (key, value) in map {
            match value {
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        table.add_column(key, DataColumn::Int64(vec![i]));
                    } else if let Some(f) = n.as_f64() {
                        table.add_column(key, DataColumn::Float64(vec![f]));
                    }
                }
                serde_json::Value::Bool(b) => {
                    table.add_column(key, DataColumn::Boolean(vec![*b]));
                }
                serde_json::Value::String(s) => {
                    table.add_column(key, DataColumn::String(vec![s.clone()]));
                }
                _ => {
                    table.add_column(key, DataColumn::String(vec![value.to_string()]));
                }
            }
        }
    }
    Ok(table.clone())
}

/// Read WAV into DataTable
fn read_wav_to_table(path: &Path, opts: &ReadOptions) -> Result<DataTable> {
    let (header, data) = crate::wavfile::read_wav(path)?;

    let mut table = DataTable::new(DataFormat::Wav);
    table.set_metadata("source_file", &path.display().to_string());
    table.set_metadata("sample_rate", &header.sample_rate.to_string());
    table.set_metadata("channels", &header.channels.to_string());
    table.set_metadata("bits_per_sample", &header.bits_per_sample.to_string());

    let max = opts.max_rows.unwrap_or(usize::MAX);

    // Flatten ndarray to f64 column
    let flat: Vec<f64> = data.iter().take(max).map(|&s| s as f64).collect();
    table.add_column("samples", DataColumn::Float64(flat));

    Ok(table)
}

/// Read NPY into DataTable
fn read_npy_to_table(path: &Path, opts: &ReadOptions) -> Result<DataTable> {
    let arr = crate::npy::read_npy(path)?;

    let mut table = DataTable::new(DataFormat::Npy);
    table.set_metadata("source_file", &path.display().to_string());
    table.set_metadata("dtype", &format!("{:?}", arr.dtype()));
    table.set_metadata("shape", &format!("{:?}", arr.shape()));

    let max = opts.max_rows.unwrap_or(usize::MAX);

    // Convert NpyArray enum to DataColumn
    match &arr {
        crate::npy::NpyArray::Float64 { data, .. } => {
            let values: Vec<f64> = data.iter().take(max).copied().collect();
            table.add_column("data", DataColumn::Float64(values));
        }
        crate::npy::NpyArray::Float32 { data, .. } => {
            let values: Vec<f32> = data.iter().take(max).copied().collect();
            table.add_column("data", DataColumn::Float32(values));
        }
        crate::npy::NpyArray::Int32 { data, .. } => {
            let values: Vec<i32> = data.iter().take(max).copied().collect();
            table.add_column("data", DataColumn::Int32(values));
        }
        crate::npy::NpyArray::Int64 { data, .. } => {
            let values: Vec<i64> = data.iter().take(max).copied().collect();
            table.add_column("data", DataColumn::Int64(values));
        }
    }

    Ok(table)
}

/// Read Matrix Market into DataTable
fn read_mtx_to_table(path: &Path, opts: &ReadOptions) -> Result<DataTable> {
    let mm = crate::matrix_market::read_sparse_matrix(path)?;

    let mut table = DataTable::new(DataFormat::MatrixMarket);
    table.set_metadata("source_file", &path.display().to_string());
    table.set_metadata("matrix_rows", &mm.rows.to_string());
    table.set_metadata("matrix_cols", &mm.cols.to_string());
    table.set_metadata("nnz", &mm.entries.len().to_string());

    let max = opts.max_rows.unwrap_or(usize::MAX);

    // Store as coordinate format columns
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut vals = Vec::new();

    for (i, entry) in mm.entries.iter().enumerate() {
        if i >= max {
            break;
        }
        row_indices.push(entry.row as i64);
        col_indices.push(entry.col as i64);
        vals.push(entry.value);
    }

    table.add_column("row", DataColumn::Int64(row_indices));
    table.add_column("col", DataColumn::Int64(col_indices));
    table.add_column("value", DataColumn::Float64(vals));

    Ok(table)
}

/// Read HDF5 (lite) into DataTable
fn read_hdf5_to_table(path: &Path, opts: &ReadOptions) -> Result<DataTable> {
    let reader = crate::hdf5_lite::Hdf5Reader::open(path)?;

    let mut table = DataTable::new(DataFormat::Hdf5);
    table.set_metadata("source_file", &path.display().to_string());
    table.set_metadata(
        "superblock_version",
        &reader.superblock().version.to_string(),
    );

    // If a specific dataset is requested, read just that
    if let Some(ref ds_path) = opts.hdf5_dataset {
        let dataset = reader.read_dataset(ds_path)?;
        let col = hdf5_value_to_column(&dataset.data);
        table.add_column(&dataset.name, col);
        table.set_metadata("dataset_path", ds_path);
        table.set_metadata("shape", &format!("{:?}", dataset.shape));

        for (key, attr) in &dataset.attributes {
            let attr_str = match &attr.value {
                crate::hdf5_lite::Hdf5Value::Strings(s) => s.join(", "),
                v => format!("{:?}", v),
            };
            table.set_metadata(&format!("attr_{key}"), &attr_str);
        }
        return Ok(table);
    }

    // Otherwise, list all datasets and read each
    let nodes = reader.list_all()?;
    for node in &nodes {
        if node.node_type == crate::hdf5_lite::Hdf5NodeType::Dataset {
            if let Ok(dataset) = reader.read_dataset(&node.path) {
                let col = hdf5_value_to_column(&dataset.data);
                table.add_column(&node.path, col);
            }
        }
    }

    Ok(table)
}

/// Convert HDF5 value to DataColumn
fn hdf5_value_to_column(value: &crate::hdf5_lite::Hdf5Value) -> DataColumn {
    match value {
        crate::hdf5_lite::Hdf5Value::Int8(v) => {
            DataColumn::Int32(v.iter().map(|x| *x as i32).collect())
        }
        crate::hdf5_lite::Hdf5Value::Int16(v) => {
            DataColumn::Int32(v.iter().map(|x| *x as i32).collect())
        }
        crate::hdf5_lite::Hdf5Value::Int32(v) => DataColumn::Int32(v.clone()),
        crate::hdf5_lite::Hdf5Value::Int64(v) => DataColumn::Int64(v.clone()),
        crate::hdf5_lite::Hdf5Value::UInt8(v) => {
            DataColumn::Int32(v.iter().map(|x| *x as i32).collect())
        }
        crate::hdf5_lite::Hdf5Value::UInt16(v) => {
            DataColumn::Int32(v.iter().map(|x| *x as i32).collect())
        }
        crate::hdf5_lite::Hdf5Value::UInt32(v) => {
            DataColumn::Int64(v.iter().map(|x| *x as i64).collect())
        }
        crate::hdf5_lite::Hdf5Value::UInt64(v) => {
            DataColumn::Int64(v.iter().map(|x| *x as i64).collect())
        }
        crate::hdf5_lite::Hdf5Value::Float32(v) => DataColumn::Float32(v.clone()),
        crate::hdf5_lite::Hdf5Value::Float64(v) => DataColumn::Float64(v.clone()),
        crate::hdf5_lite::Hdf5Value::Strings(v) => DataColumn::String(v.clone()),
        crate::hdf5_lite::Hdf5Value::Raw(v) => DataColumn::Bytes(vec![v.clone()]),
    }
}

/// Read NetCDF into DataTable
fn read_netcdf_to_table(path: &Path, opts: &ReadOptions) -> Result<DataTable> {
    let nc = crate::netcdf_lite::NcFile::read_from_file(path)?;

    let mut table = DataTable::new(DataFormat::NetCdf);
    table.set_metadata("source_file", &path.display().to_string());

    let dims = nc.dimensions();
    let num_records = nc.num_records();

    // If a specific variable is requested
    if let Some(ref var_name) = opts.netcdf_variable {
        let var = nc.variable(var_name)?;
        let values = var.as_f64(dims, num_records)?;
        let max = opts.max_rows.unwrap_or(usize::MAX);
        let truncated: Vec<f64> = values.into_iter().take(max).collect();
        table.add_column(var_name, DataColumn::Float64(truncated));
        return Ok(table);
    }

    // Read all variables
    let var_names: Vec<String> = nc.variable_names().iter().map(|s| s.to_string()).collect();
    for var_name in &var_names {
        if let Some(ref wanted) = opts.columns {
            if !wanted.iter().any(|w| w == var_name) {
                continue;
            }
        }

        if let Ok(var) = nc.variable(var_name) {
            if let Ok(values) = var.as_f64(dims, num_records) {
                let max = opts.max_rows.unwrap_or(usize::MAX);
                let truncated: Vec<f64> = values.into_iter().take(max).collect();
                table.add_column(var_name, DataColumn::Float64(truncated));
            }
        }
    }

    Ok(table)
}

// =====================================================================
// Utility functions
// =====================================================================

/// Slice a column from start to end index
fn slice_column(col: &DataColumn, start: usize, end: usize) -> DataColumn {
    let s = start;
    let e = end;
    match col {
        DataColumn::Int32(v) => DataColumn::Int32(v[s.min(v.len())..e.min(v.len())].to_vec()),
        DataColumn::Int64(v) => DataColumn::Int64(v[s.min(v.len())..e.min(v.len())].to_vec()),
        DataColumn::Float32(v) => DataColumn::Float32(v[s.min(v.len())..e.min(v.len())].to_vec()),
        DataColumn::Float64(v) => DataColumn::Float64(v[s.min(v.len())..e.min(v.len())].to_vec()),
        DataColumn::String(v) => DataColumn::String(v[s.min(v.len())..e.min(v.len())].to_vec()),
        DataColumn::Boolean(v) => DataColumn::Boolean(v[s.min(v.len())..e.min(v.len())].to_vec()),
        DataColumn::Bytes(v) => DataColumn::Bytes(v[s.min(v.len())..e.min(v.len())].to_vec()),
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_column_len() {
        assert_eq!(DataColumn::Int32(vec![1, 2, 3]).len(), 3);
        assert_eq!(DataColumn::Float64(vec![]).len(), 0);
        assert!(DataColumn::String(vec![]).is_empty());
    }

    #[test]
    fn test_data_column_as_f64() {
        let col = DataColumn::Int32(vec![1, 2, 3]);
        let f = col.as_f64();
        assert_eq!(f, vec![1.0, 2.0, 3.0]);

        let col = DataColumn::Boolean(vec![true, false]);
        let f = col.as_f64();
        assert_eq!(f, vec![1.0, 0.0]);

        let col = DataColumn::String(vec!["a".to_string()]);
        assert!(col.as_f64().is_empty());
    }

    #[test]
    fn test_data_column_as_strings() {
        let col = DataColumn::Int32(vec![42]);
        assert_eq!(col.as_strings(), vec!["42"]);

        let col = DataColumn::Boolean(vec![true]);
        assert_eq!(col.as_strings(), vec!["true"]);
    }

    #[test]
    fn test_data_column_type_name() {
        assert_eq!(DataColumn::Int32(vec![]).type_name(), "int32");
        assert_eq!(DataColumn::Float64(vec![]).type_name(), "float64");
        assert_eq!(DataColumn::String(vec![]).type_name(), "string");
        assert_eq!(DataColumn::Boolean(vec![]).type_name(), "bool");
    }

    #[test]
    fn test_data_table_basic() {
        let mut table = DataTable::new(DataFormat::Csv);
        table.add_column("x", DataColumn::Float64(vec![1.0, 2.0, 3.0]));
        table.add_column("y", DataColumn::Int32(vec![10, 20, 30]));

        assert_eq!(table.num_columns(), 2);
        assert_eq!(table.num_rows(), 3);
        assert!(!table.is_empty());
        assert_eq!(table.source_format(), DataFormat::Csv);
        assert_eq!(table.column_names(), &["x", "y"]);
    }

    #[test]
    fn test_data_table_column_access() {
        let mut table = DataTable::new(DataFormat::Json);
        table.add_column("a", DataColumn::Int64(vec![1, 2]));
        table.add_column(
            "b",
            DataColumn::String(vec!["x".to_string(), "y".to_string()]),
        );

        assert!(table.column("a").is_some());
        assert!(table.column("c").is_none());
        assert!(table.column_by_index(0).is_some());
        assert!(table.column_by_index(5).is_none());
    }

    #[test]
    fn test_data_table_metadata() {
        let mut table = DataTable::new(DataFormat::Hdf5);
        table.set_metadata("key1", "value1");
        table.set_metadata("key2", "value2");

        assert_eq!(table.metadata("key1"), Some("value1"));
        assert_eq!(table.metadata("key2"), Some("value2"));
        assert_eq!(table.metadata("key3"), None);
        assert_eq!(table.all_metadata().len(), 2);
    }

    #[test]
    fn test_data_table_select_columns() {
        let mut table = DataTable::new(DataFormat::Csv);
        table.add_column("a", DataColumn::Int32(vec![1]));
        table.add_column("b", DataColumn::Int32(vec![2]));
        table.add_column("c", DataColumn::Int32(vec![3]));

        let selected = table.select_columns(&["a", "c"]).expect("select");
        assert_eq!(selected.num_columns(), 2);
        assert_eq!(selected.column_names(), &["a", "c"]);

        let err = table.select_columns(&["nonexistent"]);
        assert!(err.is_err());
    }

    #[test]
    fn test_data_table_slice_rows() {
        let mut table = DataTable::new(DataFormat::Csv);
        table.add_column("x", DataColumn::Float64(vec![1.0, 2.0, 3.0, 4.0, 5.0]));

        let sliced = table.slice_rows(1, 4);
        assert_eq!(sliced.num_rows(), 3);
        if let Some(DataColumn::Float64(v)) = sliced.column("x") {
            assert_eq!(v, &[2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_data_table_summary() {
        let mut table = DataTable::new(DataFormat::ArrowIpc);
        table.add_column("temp", DataColumn::Float64(vec![20.5, 21.0]));
        let summary = table.summary();
        assert!(summary.contains("1 columns"));
        assert!(summary.contains("2 rows"));
        assert!(summary.contains("Arrow IPC"));
    }

    #[test]
    fn test_data_table_empty() {
        let table = DataTable::new(DataFormat::Unknown);
        assert!(table.is_empty());
        assert_eq!(table.num_rows(), 0);
        assert_eq!(table.num_columns(), 0);
    }

    #[test]
    fn test_read_options_default() {
        let opts = ReadOptions::default();
        assert!(opts.format.is_none());
        assert!(opts.max_rows.is_none());
        assert!(opts.columns.is_none());
        assert!(opts.csv_delimiter.is_none());
    }

    #[test]
    fn test_slice_column_bounds() {
        let col = DataColumn::Int32(vec![1, 2, 3]);
        let sliced = slice_column(&col, 0, 100);
        if let DataColumn::Int32(v) = sliced {
            assert_eq!(v, vec![1, 2, 3]);
        }
    }

    #[test]
    fn test_csv_roundtrip_via_universal() {
        let dir = std::env::temp_dir();
        let path = dir.join("universal_test.csv");

        // Write a simple CSV
        std::fs::write(&path, "x,y,name\n1,1.1,a\n2,2.2,b\n3,3.3,c\n").expect("write csv");

        let table = read_data(&path, None).expect("read");
        assert_eq!(table.source_format(), DataFormat::Csv);
        assert_eq!(table.num_rows(), 3);
        assert!(table.column("x").is_some());
        assert!(table.column("y").is_some());
        assert!(table.column("name").is_some());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_json_roundtrip_via_universal() {
        let dir = std::env::temp_dir();
        let path = dir.join("universal_test.json");

        std::fs::write(
            &path,
            r#"[{"id": 1, "val": 3.14, "ok": true}, {"id": 2, "val": 2.72, "ok": false}]"#,
        )
        .expect("write json");

        let table = read_data(&path, None).expect("read");
        assert_eq!(table.source_format(), DataFormat::Json);
        assert_eq!(table.num_rows(), 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_arrow_roundtrip_via_universal() {
        let dir = std::env::temp_dir();
        let path = dir.join("universal_test.arrow");

        let schema = crate::arrow_ipc::ArrowSchema::new(vec![crate::arrow_ipc::ArrowField::new(
            "x",
            crate::arrow_ipc::ArrowDataType::Float64,
        )]);
        let batch = crate::arrow_ipc::RecordBatch::new(
            schema.clone(),
            vec![crate::arrow_ipc::ArrowColumn::Float64(vec![1.0, 2.0])],
        )
        .expect("batch");

        crate::arrow_ipc::write_arrow_ipc_file(&path, &schema, &[batch]).expect("write");

        let table = read_data(&path, None).expect("read");
        assert_eq!(table.source_format(), DataFormat::ArrowIpc);
        assert_eq!(table.num_rows(), 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_with_max_rows() {
        let dir = std::env::temp_dir();
        let path = dir.join("universal_maxrows.csv");

        std::fs::write(&path, "x\n1\n2\n3\n4\n5\n").expect("write");

        let opts = ReadOptions {
            max_rows: Some(2),
            ..Default::default()
        };
        let table = read_data(&path, Some(opts)).expect("read");
        assert_eq!(table.num_rows(), 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_unsupported_format() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_file.unsupported_ext");

        std::fs::write(&path, [0x00, 0x01, 0x02]).expect("write");

        let opts = ReadOptions {
            format: Some(DataFormat::Fits),
            ..Default::default()
        };
        let result = read_data(&path, Some(opts));
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_streaming_reader() {
        let dir = std::env::temp_dir();
        let path = dir.join("streaming_test.csv");

        std::fs::write(&path, "x\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n").expect("write");

        let mut reader = StreamingReader::open(&path, 3, None).expect("open");
        assert_eq!(reader.total_rows(), 10);
        assert!(reader.num_chunks() >= 3);

        let first_row_count = {
            let first = reader.next_chunk().expect("chunk1");
            assert!(first.num_rows() <= 3);
            first.num_rows()
        };

        reader.reset();
        let first_again = reader.next_chunk().expect("chunk1 again");
        assert_eq!(first_again.num_rows(), first_row_count);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_data_column_bytes() {
        let col = DataColumn::Bytes(vec![vec![1, 2, 3], vec![4, 5]]);
        assert_eq!(col.len(), 2);
        assert_eq!(col.type_name(), "bytes");
        assert!(col.as_f64().is_empty());
        let s = col.as_strings();
        assert!(s[0].contains("3 bytes"));
    }
}
