//! Binary format utilities
//!
//! Provides three high-level binary serialisation facilities:
//!
//! * **MessagePack** — encode/decode [`serde_json::Value`] trees using `rmp-serde`.
//! * **[`BinaryArrayFile`]** — store typed numeric arrays with shape metadata in a
//!   compact binary file (magic bytes + header + raw data).
//! * **[`ColumnarFile`]** — store a named collection of typed columns
//!   (`f64`, `i64`, `bool`, `String`) in a single file.
//!
//! ## File formats
//!
//! ### BinaryArrayFile (`*.baf`)
//!
//! ```text
//! [magic: 8 bytes "SCIRSARR"]
//! [dtype: 1 byte  (0=f64, 1=i32)]
//! [ndim: u32 LE]
//! [dim_0 .. dim_ndim-1: each u64 LE]
//! [data: ndim*sizeof(T) bytes, little-endian]
//! ```
//!
//! ### ColumnarFile (`*.scircolf`)
//!
//! ```text
//! [magic: 8 bytes "SCIRCOLF"]
//! [ncols: u32 LE]
//! per column:
//!   [name_len: u32 LE][name: UTF-8]
//!   [type_tag: u8  (0=f64, 1=i64, 2=bool, 3=text)]
//!   [nrows: u64 LE]
//!   [data: raw bytes (f64/i64: 8 bytes each LE; bool: 1 byte; text: u32 len + UTF-8)]
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::error::{IoError, Result};

// ─────────────────────────────── MessagePack ─────────────────────────────────

/// Encode a [`serde_json::Value`] as MessagePack bytes.
pub fn msgpack_encode(value: &serde_json::Value) -> Result<Vec<u8>> {
    // Convert serde_json::Value to a MessagePack-serialisable form via rmp-serde
    rmp_serde::to_vec(value)
        .map_err(|e| IoError::SerializationError(format!("msgpack encode failed: {e}")))
}

/// Decode MessagePack bytes into a [`serde_json::Value`].
pub fn msgpack_decode(bytes: &[u8]) -> Result<serde_json::Value> {
    rmp_serde::from_slice(bytes)
        .map_err(|e| IoError::DeserializationError(format!("msgpack decode failed: {e}")))
}

// ─────────────────────────────── BinaryArrayFile ─────────────────────────────

const BAF_MAGIC: &[u8; 8] = b"SCIRSARR";
const BAF_DTYPE_F64: u8 = 0;
const BAF_DTYPE_I32: u8 = 1;

/// Binary array file I/O with shape metadata.
///
/// All numeric values are stored in little-endian byte order.
pub struct BinaryArrayFile;

impl BinaryArrayFile {
    // ── f64 ──────────────────────────────────────────────────────────────────

    /// Write a flat `f64` slice with its `shape` to `path`.
    ///
    /// The caller is responsible for ensuring `data.len() == shape.iter().product()`.
    pub fn write_f64(path: &Path, data: &[f64], shape: &[usize]) -> Result<()> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(IoError::ValidationError(format!(
                "data length {} does not match shape product {expected}",
                data.len()
            )));
        }
        let file = File::create(path)
            .map_err(|e| IoError::FileError(format!("cannot create {:?}: {e}", path)))?;
        let mut w = BufWriter::new(file);
        baf_write_header(&mut w, BAF_DTYPE_F64, shape)?;
        for &v in data {
            w.write_f64::<LittleEndian>(v)
                .map_err(|e| IoError::FileError(format!("write f64 failed: {e}")))?;
        }
        w.flush()
            .map_err(|e| IoError::FileError(format!("flush failed: {e}")))
    }

    /// Read an `f64` array from `path`, returning `(flat_data, shape)`.
    pub fn read_f64(path: &Path) -> Result<(Vec<f64>, Vec<usize>)> {
        let mut r = open_for_read(path)?;
        let (dtype, shape) = baf_read_header(&mut r)?;
        if dtype != BAF_DTYPE_F64 {
            return Err(IoError::FormatError(format!(
                "expected dtype f64 (0), got {dtype}"
            )));
        }
        let n: usize = shape.iter().product();
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(
                r.read_f64::<LittleEndian>()
                    .map_err(|e| IoError::FileError(format!("read f64 failed: {e}")))?,
            );
        }
        Ok((data, shape))
    }

    // ── i32 ──────────────────────────────────────────────────────────────────

    /// Write a flat `i32` slice with its `shape` to `path`.
    pub fn write_i32(path: &Path, data: &[i32], shape: &[usize]) -> Result<()> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(IoError::ValidationError(format!(
                "data length {} does not match shape product {expected}",
                data.len()
            )));
        }
        let file = File::create(path)
            .map_err(|e| IoError::FileError(format!("cannot create {:?}: {e}", path)))?;
        let mut w = BufWriter::new(file);
        baf_write_header(&mut w, BAF_DTYPE_I32, shape)?;
        for &v in data {
            w.write_i32::<LittleEndian>(v)
                .map_err(|e| IoError::FileError(format!("write i32 failed: {e}")))?;
        }
        w.flush()
            .map_err(|e| IoError::FileError(format!("flush failed: {e}")))
    }

    /// Read an `i32` array from `path`, returning `(flat_data, shape)`.
    pub fn read_i32(path: &Path) -> Result<(Vec<i32>, Vec<usize>)> {
        let mut r = open_for_read(path)?;
        let (dtype, shape) = baf_read_header(&mut r)?;
        if dtype != BAF_DTYPE_I32 {
            return Err(IoError::FormatError(format!(
                "expected dtype i32 (1), got {dtype}"
            )));
        }
        let n: usize = shape.iter().product();
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(
                r.read_i32::<LittleEndian>()
                    .map_err(|e| IoError::FileError(format!("read i32 failed: {e}")))?,
            );
        }
        Ok((data, shape))
    }
}

// Header helpers

fn baf_write_header<W: Write>(w: &mut W, dtype: u8, shape: &[usize]) -> Result<()> {
    w.write_all(BAF_MAGIC)
        .map_err(|e| IoError::FileError(format!("write magic failed: {e}")))?;
    w.write_u8(dtype)
        .map_err(|e| IoError::FileError(format!("write dtype failed: {e}")))?;
    w.write_u32::<LittleEndian>(shape.len() as u32)
        .map_err(|e| IoError::FileError(format!("write ndim failed: {e}")))?;
    for &dim in shape {
        w.write_u64::<LittleEndian>(dim as u64)
            .map_err(|e| IoError::FileError(format!("write dim failed: {e}")))?;
    }
    Ok(())
}

fn baf_read_header<R: Read>(r: &mut R) -> Result<(u8, Vec<usize>)> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)
        .map_err(|e| IoError::FileError(format!("read magic failed: {e}")))?;
    if &magic != BAF_MAGIC {
        return Err(IoError::FormatError(
            "not a BinaryArrayFile (bad magic bytes)".to_string(),
        ));
    }
    let dtype = r
        .read_u8()
        .map_err(|e| IoError::FileError(format!("read dtype failed: {e}")))?;
    let ndim = r
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FileError(format!("read ndim failed: {e}")))?;
    let mut shape = Vec::with_capacity(ndim as usize);
    for _ in 0..ndim {
        let d = r
            .read_u64::<LittleEndian>()
            .map_err(|e| IoError::FileError(format!("read dim failed: {e}")))?;
        shape.push(d as usize);
    }
    Ok((dtype, shape))
}

fn open_for_read(path: &Path) -> Result<std::io::BufReader<File>> {
    let file =
        File::open(path).map_err(|e| IoError::FileError(format!("cannot open {:?}: {e}", path)))?;
    Ok(std::io::BufReader::new(file))
}

// ─────────────────────────────── ColumnarFile ────────────────────────────────

const COLF_MAGIC: &[u8; 8] = b"SCIRCOLF";

const COLF_TAG_F64: u8 = 0;
const COLF_TAG_I64: u8 = 1;
const COLF_TAG_BOOL: u8 = 2;
const COLF_TAG_TEXT: u8 = 3;

/// A single column in a [`ColumnarFile`].
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnData {
    /// 64-bit floating-point column.
    F64(Vec<f64>),
    /// 64-bit signed integer column.
    I64(Vec<i64>),
    /// Boolean column.
    Bool(Vec<bool>),
    /// UTF-8 text column.
    Text(Vec<String>),
}

impl ColumnData {
    /// Number of rows in this column.
    pub fn len(&self) -> usize {
        match self {
            ColumnData::F64(v) => v.len(),
            ColumnData::I64(v) => v.len(),
            ColumnData::Bool(v) => v.len(),
            ColumnData::Text(v) => v.len(),
        }
    }

    /// `true` if the column has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Simple columnar binary file writer / reader.
///
/// Columns are stored in insertion order.  Each column is written
/// independently; the file can be read back selectively by name.
pub struct ColumnarFile;

impl ColumnarFile {
    /// Write all columns in `columns` to `path`.
    pub fn write(path: &Path, columns: &HashMap<String, ColumnData>) -> Result<()> {
        // Determine a stable column order (sorted by name for reproducibility)
        let mut names: Vec<&String> = columns.keys().collect();
        names.sort();

        let file = File::create(path)
            .map_err(|e| IoError::FileError(format!("cannot create {:?}: {e}", path)))?;
        let mut w = BufWriter::new(file);

        // Magic
        w.write_all(COLF_MAGIC)
            .map_err(|e| IoError::FileError(format!("write magic failed: {e}")))?;
        // Number of columns
        w.write_u32::<LittleEndian>(names.len() as u32)
            .map_err(|e| IoError::FileError(format!("write ncols failed: {e}")))?;

        for name in &names {
            let col = &columns[*name];
            colf_write_column(&mut w, name, col)?;
        }

        w.flush()
            .map_err(|e| IoError::FileError(format!("flush failed: {e}")))
    }

    /// Read all columns from `path`.
    pub fn read(path: &Path) -> Result<HashMap<String, ColumnData>> {
        let mut r = open_for_read(path)?;
        colf_read_header(&mut r)?;
        let ncols = r
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FileError(format!("read ncols failed: {e}")))?;
        let mut map = HashMap::with_capacity(ncols as usize);
        for _ in 0..ncols {
            let (name, col) = colf_read_column(&mut r)?;
            map.insert(name, col);
        }
        Ok(map)
    }

    /// Read a single named column from `path`, scanning the file serially.
    ///
    /// Returns `Err` if the column is not present.
    pub fn read_column(path: &Path, col_name: &str) -> Result<ColumnData> {
        let mut r = open_for_read(path)?;
        colf_read_header(&mut r)?;
        let ncols = r
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FileError(format!("read ncols failed: {e}")))?;
        for _ in 0..ncols {
            let (name, col) = colf_read_column(&mut r)?;
            if name == col_name {
                return Ok(col);
            }
        }
        Err(IoError::NotFound(format!(
            "column '{col_name}' not found in {:?}",
            path
        )))
    }
}

// Column serialisation helpers

fn colf_write_column<W: Write>(w: &mut W, name: &str, col: &ColumnData) -> Result<()> {
    // Name
    let name_bytes = name.as_bytes();
    w.write_u32::<LittleEndian>(name_bytes.len() as u32)
        .map_err(|e| IoError::FileError(format!("write name_len failed: {e}")))?;
    w.write_all(name_bytes)
        .map_err(|e| IoError::FileError(format!("write name failed: {e}")))?;

    // Type tag + nrows
    let (tag, nrows) = match col {
        ColumnData::F64(v) => (COLF_TAG_F64, v.len()),
        ColumnData::I64(v) => (COLF_TAG_I64, v.len()),
        ColumnData::Bool(v) => (COLF_TAG_BOOL, v.len()),
        ColumnData::Text(v) => (COLF_TAG_TEXT, v.len()),
    };
    w.write_u8(tag)
        .map_err(|e| IoError::FileError(format!("write tag failed: {e}")))?;
    w.write_u64::<LittleEndian>(nrows as u64)
        .map_err(|e| IoError::FileError(format!("write nrows failed: {e}")))?;

    // Data
    match col {
        ColumnData::F64(v) => {
            for &x in v {
                w.write_f64::<LittleEndian>(x)
                    .map_err(|e| IoError::FileError(format!("write f64 datum failed: {e}")))?;
            }
        }
        ColumnData::I64(v) => {
            for &x in v {
                w.write_i64::<LittleEndian>(x)
                    .map_err(|e| IoError::FileError(format!("write i64 datum failed: {e}")))?;
            }
        }
        ColumnData::Bool(v) => {
            for &x in v {
                w.write_u8(if x { 1 } else { 0 })
                    .map_err(|e| IoError::FileError(format!("write bool datum failed: {e}")))?;
            }
        }
        ColumnData::Text(v) => {
            for s in v {
                let bytes = s.as_bytes();
                w.write_u32::<LittleEndian>(bytes.len() as u32)
                    .map_err(|e| IoError::FileError(format!("write text len failed: {e}")))?;
                w.write_all(bytes)
                    .map_err(|e| IoError::FileError(format!("write text bytes failed: {e}")))?;
            }
        }
    }
    Ok(())
}

fn colf_read_header<R: Read>(r: &mut R) -> Result<()> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)
        .map_err(|e| IoError::FileError(format!("read magic failed: {e}")))?;
    if &magic != COLF_MAGIC {
        return Err(IoError::FormatError(
            "not a ColumnarFile (bad magic bytes)".to_string(),
        ));
    }
    Ok(())
}

fn colf_read_column<R: Read>(r: &mut R) -> Result<(String, ColumnData)> {
    // Name
    let name_len = r
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FileError(format!("read name_len failed: {e}")))?;
    let mut name_bytes = vec![0u8; name_len as usize];
    r.read_exact(&mut name_bytes)
        .map_err(|e| IoError::FileError(format!("read name bytes failed: {e}")))?;
    let name = String::from_utf8(name_bytes)
        .map_err(|e| IoError::FormatError(format!("column name is not valid UTF-8: {e}")))?;

    // Tag + nrows
    let tag = r
        .read_u8()
        .map_err(|e| IoError::FileError(format!("read tag failed: {e}")))?;
    let nrows = r
        .read_u64::<LittleEndian>()
        .map_err(|e| IoError::FileError(format!("read nrows failed: {e}")))?;
    let nrows = nrows as usize;

    // Data
    let col = match tag {
        COLF_TAG_F64 => {
            let mut v = Vec::with_capacity(nrows);
            for _ in 0..nrows {
                v.push(
                    r.read_f64::<LittleEndian>()
                        .map_err(|e| IoError::FileError(format!("read f64 datum: {e}")))?,
                );
            }
            ColumnData::F64(v)
        }
        COLF_TAG_I64 => {
            let mut v = Vec::with_capacity(nrows);
            for _ in 0..nrows {
                v.push(
                    r.read_i64::<LittleEndian>()
                        .map_err(|e| IoError::FileError(format!("read i64 datum: {e}")))?,
                );
            }
            ColumnData::I64(v)
        }
        COLF_TAG_BOOL => {
            let mut v = Vec::with_capacity(nrows);
            for _ in 0..nrows {
                let b = r
                    .read_u8()
                    .map_err(|e| IoError::FileError(format!("read bool datum: {e}")))?;
                v.push(b != 0);
            }
            ColumnData::Bool(v)
        }
        COLF_TAG_TEXT => {
            let mut v = Vec::with_capacity(nrows);
            for _ in 0..nrows {
                let len = r
                    .read_u32::<LittleEndian>()
                    .map_err(|e| IoError::FileError(format!("read text len: {e}")))?;
                let mut buf = vec![0u8; len as usize];
                r.read_exact(&mut buf)
                    .map_err(|e| IoError::FileError(format!("read text bytes: {e}")))?;
                let s = String::from_utf8(buf).map_err(|e| {
                    IoError::FormatError(format!("text column contains invalid UTF-8: {e}"))
                })?;
                v.push(s);
            }
            ColumnData::Text(v)
        }
        other => {
            return Err(IoError::FormatError(format!(
                "unknown column type tag {other}"
            )))
        }
    };

    Ok((name, col))
}

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MessagePack ───────────────────────────────────────────────────────────

    #[test]
    fn test_msgpack_encode_decode_object() {
        let val = serde_json::json!({"name": "Alice", "score": 99, "active": true});
        let encoded = msgpack_encode(&val).expect("encode");
        let decoded = msgpack_decode(&encoded).expect("decode");
        assert_eq!(decoded["name"], "Alice");
        assert_eq!(decoded["score"], 99);
        assert_eq!(decoded["active"], true);
    }

    #[test]
    fn test_msgpack_encode_decode_array() {
        let val = serde_json::json!([1, 2, 3, 4, 5]);
        let encoded = msgpack_encode(&val).expect("encode");
        let decoded = msgpack_decode(&encoded).expect("decode");
        assert_eq!(decoded[2], 3);
    }

    #[test]
    fn test_msgpack_encode_decode_null() {
        let val = serde_json::Value::Null;
        let encoded = msgpack_encode(&val).expect("encode");
        let decoded = msgpack_decode(&encoded).expect("decode");
        assert!(decoded.is_null());
    }

    #[test]
    fn test_msgpack_decode_invalid_bytes_errors() {
        let bad = vec![0xc1u8]; // reserved / invalid msgpack byte
        assert!(msgpack_decode(&bad).is_err());
    }

    // ── BinaryArrayFile f64 ───────────────────────────────────────────────────

    #[test]
    fn test_binary_array_file_f64_roundtrip() {
        let dir = std::env::temp_dir().join("scirs2_io_baf_f64_test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("array.baf");

        let data: Vec<f64> = (0..12).map(|i| i as f64 * 0.5).collect();
        let shape = vec![3usize, 4];

        BinaryArrayFile::write_f64(&path, &data, &shape).expect("write f64");
        let (loaded, loaded_shape) = BinaryArrayFile::read_f64(&path).expect("read f64");

        assert_eq!(loaded_shape, shape);
        assert_eq!(loaded.len(), 12);
        for (orig, got) in data.iter().zip(loaded.iter()) {
            assert!((orig - got).abs() < 1e-12, "mismatch: {orig} vs {got}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_binary_array_file_f64_1d() {
        let dir = std::env::temp_dir().join("scirs2_io_baf_f64_1d_test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("vec.baf");

        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        BinaryArrayFile::write_f64(&path, &data, &[5]).expect("write");
        let (loaded, shape) = BinaryArrayFile::read_f64(&path).expect("read");
        assert_eq!(shape, vec![5]);
        assert_eq!(loaded, data);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── BinaryArrayFile i32 ───────────────────────────────────────────────────

    #[test]
    fn test_binary_array_file_i32_roundtrip() {
        let dir = std::env::temp_dir().join("scirs2_io_baf_i32_test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("labels.baf");

        let data: Vec<i32> = (0..20).collect();
        let shape = vec![4usize, 5];

        BinaryArrayFile::write_i32(&path, &data, &shape).expect("write i32");
        let (loaded, loaded_shape) = BinaryArrayFile::read_i32(&path).expect("read i32");

        assert_eq!(loaded_shape, shape);
        assert_eq!(loaded, data);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_binary_array_file_wrong_dtype_error() {
        let dir = std::env::temp_dir().join("scirs2_io_baf_dtype_err_test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("typed.baf");

        // Write as f64, try to read as i32
        BinaryArrayFile::write_f64(&path, &[1.0, 2.0], &[2]).expect("write f64");
        assert!(BinaryArrayFile::read_i32(&path).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── ColumnarFile ──────────────────────────────────────────────────────────

    #[test]
    fn test_columnar_file_all_types_roundtrip() {
        let dir = std::env::temp_dir().join("scirs2_io_colf_alltype_test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("cols.scircolf");

        let mut cols: HashMap<String, ColumnData> = HashMap::new();
        cols.insert(
            "temps".to_string(),
            ColumnData::F64(vec![20.5, 21.1, 19.8, 22.3]),
        );
        cols.insert(
            "counts".to_string(),
            ColumnData::I64(vec![100i64, 200, 300, 400]),
        );
        cols.insert(
            "active".to_string(),
            ColumnData::Bool(vec![true, false, true, true]),
        );
        cols.insert(
            "labels".to_string(),
            ColumnData::Text(vec![
                "alpha".to_string(),
                "beta".to_string(),
                "gamma".to_string(),
                "delta".to_string(),
            ]),
        );

        ColumnarFile::write(&path, &cols).expect("write columnar");
        let loaded = ColumnarFile::read(&path).expect("read columnar");

        assert_eq!(loaded.len(), 4);

        match &loaded["temps"] {
            ColumnData::F64(v) => {
                assert!((v[0] - 20.5).abs() < 1e-10);
                assert!((v[2] - 19.8).abs() < 1e-10);
            }
            _ => panic!("expected F64"),
        }
        match &loaded["counts"] {
            ColumnData::I64(v) => assert_eq!(v[1], 200),
            _ => panic!("expected I64"),
        }
        match &loaded["active"] {
            ColumnData::Bool(v) => {
                assert!(v[0]);
                assert!(!v[1]);
            }
            _ => panic!("expected Bool"),
        }
        match &loaded["labels"] {
            ColumnData::Text(v) => assert_eq!(v[2], "gamma"),
            _ => panic!("expected Text"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_columnar_file_read_single_column() {
        let dir = std::env::temp_dir().join("scirs2_io_colf_single_col_test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("single.scircolf");

        let mut cols: HashMap<String, ColumnData> = HashMap::new();
        cols.insert("x".to_string(), ColumnData::F64(vec![1.1, 2.2, 3.3]));
        cols.insert("y".to_string(), ColumnData::I64(vec![10, 20, 30]));

        ColumnarFile::write(&path, &cols).expect("write");
        let y_col = ColumnarFile::read_column(&path, "y").expect("read column y");

        match y_col {
            ColumnData::I64(v) => assert_eq!(v, vec![10i64, 20, 30]),
            _ => panic!("expected I64"),
        }

        // Missing column should error
        assert!(ColumnarFile::read_column(&path, "z").is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_columnar_file_empty_columns() {
        let dir = std::env::temp_dir().join("scirs2_io_colf_empty_test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("empty.scircolf");

        let mut cols: HashMap<String, ColumnData> = HashMap::new();
        cols.insert("e".to_string(), ColumnData::F64(vec![]));

        ColumnarFile::write(&path, &cols).expect("write");
        let loaded = ColumnarFile::read(&path).expect("read");
        assert!(loaded["e"].is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
