//! Lightweight pure-Rust columnar format compatible with basic Parquet reading.
//!
//! This module implements a *simplified* columnar binary format that preserves
//! the structural ideas of Apache Parquet (magic bytes, row-group framing, column
//! chunks, and a trailing footer) without requiring any C/Fortran dependency or the
//! full `parquet` crate.
//!
//! ## Wire format
//!
//! ```text
//! [4 bytes magic "PLTE"]
//! [4 bytes version = 1u32 LE]
//! [4 bytes num_columns u32 LE]
//! [4 bytes num_rows u64 LE … stored as u64 little-endian = 8 bytes]
//! For each column:
//!   [2 bytes name_len u16 LE]
//!   [name_len bytes UTF-8 column name]
//!   [1 byte column type tag]
//!   [8 bytes data_len u64 LE]
//!   [data_len bytes raw column payload]
//! [4 bytes magic "PLTE" footer]
//! ```
//!
//! Column type tags:
//!
//! | Tag | Type      | Payload per element |
//! |-----|-----------|---------------------|
//! | 0   | Float64   | 8 bytes LE IEEE 754 |
//! | 1   | Float32   | 4 bytes LE          |
//! | 2   | Int64     | 8 bytes LE          |
//! | 3   | Int32     | 4 bytes LE          |
//! | 4   | Boolean   | 1 byte (0/1)        |
//! | 5   | Utf8      | length-prefixed      |
//!
//! # Examples
//!
//! ```rust
//! use scirs2_io::parquet_lite::{
//!     ParquetSchema, ColumnType, ParquetWriter, ParquetReader,
//! };
//!
//! let schema = ParquetSchema::new(vec![
//!     ("x".to_string(), ColumnType::Float64),
//!     ("y".to_string(), ColumnType::Float64),
//! ]);
//!
//! let cols: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
//! let bytes = ParquetWriter::write_batch(&schema, &cols).unwrap();
//! let (out_schema, out_cols) = ParquetReader::read_columns(&bytes).unwrap();
//! assert_eq!(out_cols[0], cols[0]);
//! assert_eq!(out_schema.columns[0].0, "x");
//! ```

use std::convert::TryInto;
use std::io::{Cursor, Read, Write};

use crate::error::IoError;

/// Result alias used throughout this module.
pub type ParquetLiteResult<T> = Result<T, IoError>;

// ──────────────────────────── Magic + version ────────────────────────────────

const MAGIC: &[u8; 4] = b"PLTE";
const FORMAT_VERSION: u32 = 1;

// ──────────────────────────── Column types ───────────────────────────────────

/// Type of a single column in a [`ParquetSchema`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnType {
    /// 64-bit IEEE 754 floating-point.
    Float64,
    /// 32-bit IEEE 754 floating-point.
    Float32,
    /// Signed 64-bit integer.
    Int64,
    /// Signed 32-bit integer.
    Int32,
    /// Boolean (stored as one byte per value).
    Boolean,
    /// Variable-length UTF-8 string.
    Utf8,
}

impl ColumnType {
    fn to_tag(&self) -> u8 {
        match self {
            ColumnType::Float64 => 0,
            ColumnType::Float32 => 1,
            ColumnType::Int64 => 2,
            ColumnType::Int32 => 3,
            ColumnType::Boolean => 4,
            ColumnType::Utf8 => 5,
        }
    }

    fn from_tag(tag: u8) -> ParquetLiteResult<Self> {
        match tag {
            0 => Ok(ColumnType::Float64),
            1 => Ok(ColumnType::Float32),
            2 => Ok(ColumnType::Int64),
            3 => Ok(ColumnType::Int32),
            4 => Ok(ColumnType::Boolean),
            5 => Ok(ColumnType::Utf8),
            _ => Err(IoError::FormatError(format!(
                "unknown column type tag: {tag}"
            ))),
        }
    }
}

// ──────────────────────────── Schema ─────────────────────────────────────────

/// Schema describing the columns in a [`ParquetWriter`] / [`ParquetReader`] dataset.
///
/// Each entry is `(column_name, column_type)`.
#[derive(Debug, Clone)]
pub struct ParquetSchema {
    /// Ordered list of `(name, type)` pairs.
    pub columns: Vec<(String, ColumnType)>,
}

impl ParquetSchema {
    /// Construct a schema from a list of `(name, type)` pairs.
    pub fn new(columns: Vec<(String, ColumnType)>) -> Self {
        Self { columns }
    }

    /// Number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Look up a column index by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|(n, _)| n == name)
    }
}

// ──────────────────────────── Columnar data ──────────────────────────────────

/// Strongly-typed column data that can be stored inside a `ParquetLiteFile`.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnData {
    /// 64-bit floats.
    Float64(Vec<f64>),
    /// 32-bit floats.
    Float32(Vec<f32>),
    /// 64-bit signed integers.
    Int64(Vec<i64>),
    /// 32-bit signed integers.
    Int32(Vec<i32>),
    /// Booleans.
    Boolean(Vec<bool>),
    /// UTF-8 strings.
    Utf8(Vec<String>),
}

impl ColumnData {
    /// Number of rows in this column.
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Float64(v) => v.len(),
            ColumnData::Float32(v) => v.len(),
            ColumnData::Int64(v) => v.len(),
            ColumnData::Int32(v) => v.len(),
            ColumnData::Boolean(v) => v.len(),
            ColumnData::Utf8(v) => v.len(),
        }
    }

    /// Returns true if the column contains no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn type_tag(&self) -> u8 {
        match self {
            ColumnData::Float64(_) => 0,
            ColumnData::Float32(_) => 1,
            ColumnData::Int64(_) => 2,
            ColumnData::Int32(_) => 3,
            ColumnData::Boolean(_) => 4,
            ColumnData::Utf8(_) => 5,
        }
    }

    /// Try to extract the inner `Vec<f64>`.
    pub fn as_f64(&self) -> Option<&Vec<f64>> {
        if let ColumnData::Float64(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Try to extract the inner `Vec<i64>`.
    pub fn as_i64(&self) -> Option<&Vec<i64>> {
        if let ColumnData::Int64(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Try to extract the inner `Vec<String>`.
    pub fn as_utf8(&self) -> Option<&Vec<String>> {
        if let ColumnData::Utf8(v) = self {
            Some(v)
        } else {
            None
        }
    }

    fn encode(&self) -> Vec<u8> {
        match self {
            ColumnData::Float64(vals) => {
                let mut buf = Vec::with_capacity(vals.len() * 8);
                for &v in vals {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                buf
            }
            ColumnData::Float32(vals) => {
                let mut buf = Vec::with_capacity(vals.len() * 4);
                for &v in vals {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                buf
            }
            ColumnData::Int64(vals) => {
                let mut buf = Vec::with_capacity(vals.len() * 8);
                for &v in vals {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                buf
            }
            ColumnData::Int32(vals) => {
                let mut buf = Vec::with_capacity(vals.len() * 4);
                for &v in vals {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                buf
            }
            ColumnData::Boolean(vals) => vals.iter().map(|&b| b as u8).collect(),
            ColumnData::Utf8(vals) => {
                let mut buf = Vec::new();
                for s in vals {
                    let bytes = s.as_bytes();
                    let len = bytes.len() as u32;
                    buf.extend_from_slice(&len.to_le_bytes());
                    buf.extend_from_slice(bytes);
                }
                buf
            }
        }
    }

    fn decode(tag: u8, data: &[u8], num_rows: usize) -> ParquetLiteResult<Self> {
        match tag {
            0 => {
                // Float64
                if data.len() != num_rows * 8 {
                    return Err(IoError::FormatError(format!(
                        "Float64 column: expected {} bytes, got {}",
                        num_rows * 8,
                        data.len()
                    )));
                }
                let mut vals = Vec::with_capacity(num_rows);
                for chunk in data.chunks_exact(8) {
                    vals.push(f64::from_le_bytes(chunk.try_into().map_err(|_| {
                        IoError::FormatError("Float64 decode failed".to_string())
                    })?));
                }
                Ok(ColumnData::Float64(vals))
            }
            1 => {
                // Float32
                if data.len() != num_rows * 4 {
                    return Err(IoError::FormatError(format!(
                        "Float32 column: expected {} bytes, got {}",
                        num_rows * 4,
                        data.len()
                    )));
                }
                let mut vals = Vec::with_capacity(num_rows);
                for chunk in data.chunks_exact(4) {
                    vals.push(f32::from_le_bytes(chunk.try_into().map_err(|_| {
                        IoError::FormatError("Float32 decode failed".to_string())
                    })?));
                }
                Ok(ColumnData::Float32(vals))
            }
            2 => {
                // Int64
                if data.len() != num_rows * 8 {
                    return Err(IoError::FormatError(format!(
                        "Int64 column: expected {} bytes, got {}",
                        num_rows * 8,
                        data.len()
                    )));
                }
                let mut vals = Vec::with_capacity(num_rows);
                for chunk in data.chunks_exact(8) {
                    vals.push(i64::from_le_bytes(chunk.try_into().map_err(|_| {
                        IoError::FormatError("Int64 decode failed".to_string())
                    })?));
                }
                Ok(ColumnData::Int64(vals))
            }
            3 => {
                // Int32
                if data.len() != num_rows * 4 {
                    return Err(IoError::FormatError(format!(
                        "Int32 column: expected {} bytes, got {}",
                        num_rows * 4,
                        data.len()
                    )));
                }
                let mut vals = Vec::with_capacity(num_rows);
                for chunk in data.chunks_exact(4) {
                    vals.push(i32::from_le_bytes(chunk.try_into().map_err(|_| {
                        IoError::FormatError("Int32 decode failed".to_string())
                    })?));
                }
                Ok(ColumnData::Int32(vals))
            }
            4 => {
                // Boolean
                if data.len() != num_rows {
                    return Err(IoError::FormatError(format!(
                        "Boolean column: expected {} bytes, got {}",
                        num_rows,
                        data.len()
                    )));
                }
                Ok(ColumnData::Boolean(data.iter().map(|&b| b != 0).collect()))
            }
            5 => {
                // Utf8
                let mut vals = Vec::with_capacity(num_rows);
                let mut pos = 0usize;
                for _ in 0..num_rows {
                    if pos + 4 > data.len() {
                        return Err(IoError::FormatError(
                            "Utf8 column: unexpected end of data".to_string(),
                        ));
                    }
                    let len = u32::from_le_bytes(
                        data[pos..pos + 4]
                            .try_into()
                            .map_err(|_| IoError::FormatError("Utf8 len decode".to_string()))?,
                    ) as usize;
                    pos += 4;
                    if pos + len > data.len() {
                        return Err(IoError::FormatError(
                            "Utf8 column: string data truncated".to_string(),
                        ));
                    }
                    let s = std::str::from_utf8(&data[pos..pos + len])
                        .map_err(|e| IoError::FormatError(format!("Utf8 decode: {e}")))?
                        .to_string();
                    vals.push(s);
                    pos += len;
                }
                Ok(ColumnData::Utf8(vals))
            }
            _ => Err(IoError::FormatError(format!(
                "unknown column type tag: {tag}"
            ))),
        }
    }
}

// ──────────────────────────── Writer ─────────────────────────────────────────

/// Encoder for the lightweight Parquet-like columnar format.
pub struct ParquetWriter;

impl ParquetWriter {
    /// Encode a batch of `f64` columns using the supplied schema.
    ///
    /// All columns must have the same length.  Returns the raw bytes of the
    /// encoded file.
    pub fn write_batch(schema: &ParquetSchema, columns: &[Vec<f64>]) -> ParquetLiteResult<Vec<u8>> {
        if schema.num_columns() != columns.len() {
            return Err(IoError::FormatError(format!(
                "schema has {} columns but {} data columns supplied",
                schema.num_columns(),
                columns.len()
            )));
        }

        let num_rows = if columns.is_empty() {
            0usize
        } else {
            let first_len = columns[0].len();
            for (i, col) in columns.iter().enumerate() {
                if col.len() != first_len {
                    return Err(IoError::FormatError(format!(
                        "column {i} has {} rows but column 0 has {first_rows}",
                        col.len(),
                        first_rows = first_len
                    )));
                }
            }
            first_len
        };

        // Build typed ColumnData (all Float64 for this entry point)
        let typed: Vec<ColumnData> = columns
            .iter()
            .map(|c| ColumnData::Float64(c.clone()))
            .collect();

        Self::write_typed(schema, &typed, num_rows)
    }

    /// Encode a batch of typed columns.
    ///
    /// All columns must have the same number of rows.
    pub fn write_typed(
        schema: &ParquetSchema,
        columns: &[ColumnData],
        num_rows: usize,
    ) -> ParquetLiteResult<Vec<u8>> {
        if schema.num_columns() != columns.len() {
            return Err(IoError::FormatError(format!(
                "schema has {} columns but {} data columns supplied",
                schema.num_columns(),
                columns.len()
            )));
        }

        let mut buf: Vec<u8> = Vec::new();

        // Header
        buf.write_all(MAGIC).map_err(io_err)?;
        buf.write_all(&FORMAT_VERSION.to_le_bytes())
            .map_err(io_err)?;
        buf.write_all(&(schema.num_columns() as u32).to_le_bytes())
            .map_err(io_err)?;
        buf.write_all(&(num_rows as u64).to_le_bytes())
            .map_err(io_err)?;

        // Column chunks
        for (idx, (col_data, (col_name, _col_type))) in
            columns.iter().zip(schema.columns.iter()).enumerate()
        {
            if col_data.len() != num_rows {
                return Err(IoError::FormatError(format!(
                    "column {idx} has {} rows but expected {num_rows}",
                    col_data.len()
                )));
            }
            let name_bytes = col_name.as_bytes();
            if name_bytes.len() > u16::MAX as usize {
                return Err(IoError::FormatError(format!(
                    "column name too long: {} bytes",
                    name_bytes.len()
                )));
            }
            buf.write_all(&(name_bytes.len() as u16).to_le_bytes())
                .map_err(io_err)?;
            buf.write_all(name_bytes).map_err(io_err)?;
            buf.write_all(&[col_data.type_tag()]).map_err(io_err)?;

            let payload = col_data.encode();
            buf.write_all(&(payload.len() as u64).to_le_bytes())
                .map_err(io_err)?;
            buf.write_all(&payload).map_err(io_err)?;
        }

        // Footer magic
        buf.write_all(MAGIC).map_err(io_err)?;

        Ok(buf)
    }
}

fn io_err(e: std::io::Error) -> IoError {
    IoError::FileError(e.to_string())
}

// ──────────────────────────── Reader ─────────────────────────────────────────

/// Decoder for the lightweight Parquet-like columnar format.
pub struct ParquetReader;

impl ParquetReader {
    /// Decode a byte slice previously produced by [`ParquetWriter::write_batch`].
    ///
    /// Returns `(schema, columns)` where each element of `columns` is a `Vec<f64>`.
    /// Non-Float64 columns will be lossily converted where possible; use
    /// [`ParquetReader::read_typed`] to preserve the original types.
    pub fn read_columns(data: &[u8]) -> ParquetLiteResult<(ParquetSchema, Vec<Vec<f64>>)> {
        let (schema, typed) = Self::read_typed(data)?;
        let f64_cols: Vec<Vec<f64>> = typed
            .into_iter()
            .map(|col| match col {
                ColumnData::Float64(v) => v,
                ColumnData::Float32(v) => v.into_iter().map(|x| x as f64).collect(),
                ColumnData::Int64(v) => v.into_iter().map(|x| x as f64).collect(),
                ColumnData::Int32(v) => v.into_iter().map(|x| x as f64).collect(),
                ColumnData::Boolean(v) => {
                    v.into_iter().map(|b| if b { 1.0 } else { 0.0 }).collect()
                }
                ColumnData::Utf8(v) => v
                    .iter()
                    .map(|s| s.parse::<f64>().unwrap_or(f64::NAN))
                    .collect(),
            })
            .collect();
        Ok((schema, f64_cols))
    }

    /// Decode a byte slice, preserving the original [`ColumnData`] types.
    pub fn read_typed(data: &[u8]) -> ParquetLiteResult<(ParquetSchema, Vec<ColumnData>)> {
        let mut cursor = Cursor::new(data);

        // Header magic
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic).map_err(io_err)?;
        if &magic != MAGIC {
            return Err(IoError::FormatError(format!(
                "invalid magic bytes: {:?}",
                magic
            )));
        }

        // Version
        let mut ver_buf = [0u8; 4];
        cursor.read_exact(&mut ver_buf).map_err(io_err)?;
        let version = u32::from_le_bytes(ver_buf);
        if version != FORMAT_VERSION {
            return Err(IoError::FormatError(format!(
                "unsupported format version: {version}"
            )));
        }

        // num_columns
        let mut nc_buf = [0u8; 4];
        cursor.read_exact(&mut nc_buf).map_err(io_err)?;
        let num_columns = u32::from_le_bytes(nc_buf) as usize;

        // num_rows
        let mut nr_buf = [0u8; 8];
        cursor.read_exact(&mut nr_buf).map_err(io_err)?;
        let num_rows = u64::from_le_bytes(nr_buf) as usize;

        let mut schema_cols: Vec<(String, ColumnType)> = Vec::with_capacity(num_columns);
        let mut col_data: Vec<ColumnData> = Vec::with_capacity(num_columns);

        for _ in 0..num_columns {
            // name length
            let mut nl_buf = [0u8; 2];
            cursor.read_exact(&mut nl_buf).map_err(io_err)?;
            let name_len = u16::from_le_bytes(nl_buf) as usize;

            // name
            let mut name_bytes = vec![0u8; name_len];
            cursor.read_exact(&mut name_bytes).map_err(io_err)?;
            let name = String::from_utf8(name_bytes)
                .map_err(|e| IoError::FormatError(format!("column name UTF-8: {e}")))?;

            // type tag
            let mut tag_buf = [0u8; 1];
            cursor.read_exact(&mut tag_buf).map_err(io_err)?;
            let col_type = ColumnType::from_tag(tag_buf[0])?;

            // payload length
            let mut dl_buf = [0u8; 8];
            cursor.read_exact(&mut dl_buf).map_err(io_err)?;
            let data_len = u64::from_le_bytes(dl_buf) as usize;

            // payload
            let mut payload = vec![0u8; data_len];
            cursor.read_exact(&mut payload).map_err(io_err)?;

            let decoded = ColumnData::decode(tag_buf[0], &payload, num_rows)?;
            schema_cols.push((name, col_type));
            col_data.push(decoded);
        }

        // Footer magic
        let mut footer = [0u8; 4];
        cursor.read_exact(&mut footer).map_err(io_err)?;
        if &footer != MAGIC {
            return Err(IoError::FormatError(
                "missing footer magic bytes".to_string(),
            ));
        }

        let schema = ParquetSchema::new(schema_cols);
        Ok((schema, col_data))
    }
}

// ──────────────────────────── Tests ──────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn float_schema(names: &[&str]) -> ParquetSchema {
        ParquetSchema::new(
            names
                .iter()
                .map(|n| (n.to_string(), ColumnType::Float64))
                .collect(),
        )
    }

    #[test]
    fn test_roundtrip_f64() {
        let schema = float_schema(&["x", "y", "z"]);
        let cols = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let bytes = ParquetWriter::write_batch(&schema, &cols).expect("write failed");
        let (out_schema, out_cols) = ParquetReader::read_columns(&bytes).expect("read failed");
        assert_eq!(out_schema.num_columns(), 3);
        assert_eq!(out_schema.columns[0].0, "x");
        assert_eq!(out_schema.columns[2].0, "z");
        assert_eq!(out_cols[0], cols[0]);
        assert_eq!(out_cols[1], cols[1]);
        assert_eq!(out_cols[2], cols[2]);
    }

    #[test]
    fn test_roundtrip_empty() {
        let schema = float_schema(&["a"]);
        let cols = vec![vec![]];
        let bytes = ParquetWriter::write_batch(&schema, &cols).expect("write empty");
        let (_s, out_cols) = ParquetReader::read_columns(&bytes).expect("read empty");
        assert_eq!(out_cols[0].len(), 0);
    }

    #[test]
    fn test_roundtrip_typed_int32_and_utf8() {
        let schema = ParquetSchema::new(vec![
            ("id".to_string(), ColumnType::Int32),
            ("label".to_string(), ColumnType::Utf8),
        ]);
        let col_id = ColumnData::Int32(vec![10, 20, 30]);
        let col_label = ColumnData::Utf8(vec![
            "foo".to_string(),
            "bar".to_string(),
            "baz".to_string(),
        ]);
        let bytes = ParquetWriter::write_typed(&schema, &[col_id.clone(), col_label.clone()], 3)
            .expect("write typed");
        let (_s, cols) = ParquetReader::read_typed(&bytes).expect("read typed");
        assert_eq!(cols[0], col_id);
        assert_eq!(cols[1], col_label);
    }

    #[test]
    fn test_roundtrip_boolean() {
        let schema = ParquetSchema::new(vec![("flags".to_string(), ColumnType::Boolean)]);
        let flags = ColumnData::Boolean(vec![true, false, true, true, false]);
        let bytes = ParquetWriter::write_typed(&schema, std::slice::from_ref(&flags), 5)
            .expect("write bool");
        let (_s, cols) = ParquetReader::read_typed(&bytes).expect("read bool");
        assert_eq!(cols[0], flags);
    }

    #[test]
    fn test_corrupt_magic_returns_error() {
        let schema = float_schema(&["v"]);
        let cols = vec![vec![1.0, 2.0]];
        let mut bytes = ParquetWriter::write_batch(&schema, &cols).expect("write");
        bytes[0] = b'X'; // corrupt magic
        let result = ParquetReader::read_columns(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_column_mismatch_returns_error() {
        let schema = float_schema(&["a", "b"]);
        let cols = vec![vec![1.0, 2.0]]; // only one column for two-column schema
        let result = ParquetWriter::write_batch(&schema, &cols);
        assert!(result.is_err());
    }

    #[test]
    fn test_schema_column_index_lookup() {
        let schema = float_schema(&["alpha", "beta", "gamma"]);
        assert_eq!(schema.column_index("beta"), Some(1));
        assert_eq!(schema.column_index("missing"), None);
    }

    #[test]
    fn test_roundtrip_large_dataset() {
        let n = 50_000;
        let schema = float_schema(&["time", "value"]);
        let time: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();
        let value: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let cols = vec![time.clone(), value.clone()];
        let bytes = ParquetWriter::write_batch(&schema, &cols).expect("write large");
        let (_s, out) = ParquetReader::read_columns(&bytes).expect("read large");
        assert_eq!(out[0].len(), n);
        assert!((out[1][1000] - value[1000]).abs() < 1e-15);
    }
}
