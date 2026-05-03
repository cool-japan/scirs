//! Custom binary serialisation format for scientific data.
//!
//! This module provides three layers of binary I/O:
//!
//! 1. **[`BinaryWriter`] / [`BinaryReader`]** — low-level type-safe primitives
//!    (all integers, both float sizes, byte slices, length-prefixed strings, and
//!    prefixed `f64` arrays).  All multi-byte values are stored in **little-endian**
//!    byte order for maximum portability.
//!
//! 2. **[`ScirsDataFile`]** — a structured scientific container format built on top
//!    of the primitives:
//!    - Fixed 8-byte magic header (`SCIRS2DF`)
//!    - `u8` version number
//!    - `u32` record count
//!    - Sequence of named, typed [`DataRecord`] entries
//!
//! 3. **[`DataRecord`]** — the payload variant that a record may hold:
//!    [`Scalar`](DataRecord::Scalar), [`Vector`](DataRecord::Vector),
//!    [`Matrix`](DataRecord::Matrix), and [`Text`](DataRecord::Text).
//!
//! ## File layout
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │ Magic    : [u8; 8]  = b"SCIRS2DF"                       │
//! │ Version  : u8       = 1                                  │
//! │ N records: u32 LE                                        │
//! ├─────────────────────────────────────────────────────────┤
//! │ Record₁                                                  │
//! │   Name   : u32 LE length-prefix + UTF-8 bytes           │
//! │   Tag    : u8   (0=Scalar, 1=Vector, 2=Matrix, 3=Text)  │
//! │   Data   :                                               │
//! │     Scalar → f64 LE                                      │
//! │     Vector → u64 LE count + count×f64 LE                │
//! │     Matrix → u64 rows + u64 cols + rows×cols×f64 LE     │
//! │     Text   → u32 LE length + UTF-8 bytes                 │
//! ├─────────────────────────────────────────────────────────┤
//! │ Record₂ …                                                │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Examples
//!
//! ```rust,no_run
//! use scirs2_io::binary_format::{write_scirs, read_scirs, DataRecord};
//!
//! let records = vec![
//!     DataRecord::named("pi",   DataRecord::Scalar(std::f64::consts::PI)),
//!     DataRecord::named("data", DataRecord::Vector(vec![1.0, 2.0, 3.0])),
//! ];
//! // write_scirs / read_scirs take &[(name, DataRecord)] and return Vec<(name, DataRecord)>
//! write_scirs("out.scirs2", &records).unwrap();
//! let loaded = read_scirs("out.scirs2").unwrap();
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::error::{IoError, Result};

// ─────────────────────────────── Magic / version ─────────────────────────────

const MAGIC: &[u8; 8] = b"SCIRS2DF";
const FORMAT_VERSION: u8 = 1;

// ─────────────────────────────── DataRecord ──────────────────────────────────

/// The payload carried by a single named record in a [`ScirsDataFile`].
///
/// # Tags (on-disk)
///
/// | Variant  | Tag byte |
/// |----------|----------|
/// | Scalar   | `0`      |
/// | Vector   | `1`      |
/// | Matrix   | `2`      |
/// | Text     | `3`      |
#[derive(Debug, Clone, PartialEq)]
pub enum DataRecord {
    /// A single 64-bit float.
    Scalar(f64),
    /// A one-dimensional array of 64-bit floats.
    Vector(Vec<f64>),
    /// A two-dimensional matrix stored in row-major order.
    Matrix(Vec<Vec<f64>>),
    /// A UTF-8 text string.
    Text(String),
}

impl DataRecord {
    /// Convenience constructor to bundle a name with a `DataRecord` for use
    /// with [`write_scirs`] / [`read_scirs`].
    ///
    /// ```
    /// use scirs2_io::binary_format::DataRecord;
    /// let entry = DataRecord::named("x", DataRecord::Scalar(1.0));
    /// assert_eq!(entry.0, "x");
    /// ```
    pub fn named(name: impl Into<String>, record: DataRecord) -> (String, DataRecord) {
        (name.into(), record)
    }

    fn tag(&self) -> u8 {
        match self {
            DataRecord::Scalar(_) => 0,
            DataRecord::Vector(_) => 1,
            DataRecord::Matrix(_) => 2,
            DataRecord::Text(_) => 3,
        }
    }
}

// ─────────────────────────────── BinaryWriter ────────────────────────────────

/// Type-safe binary writer backed by a [`BufWriter<File>`].
///
/// All multi-byte integer and float values are serialised in **little-endian**
/// byte order.
///
/// # Examples
///
/// ```rust,no_run
/// use scirs2_io::binary_format::BinaryWriter;
///
/// let mut w = BinaryWriter::create("output.bin").unwrap();
/// w.write_u32(42).unwrap();
/// w.write_f64(3.14).unwrap();
/// w.write_string("hello").unwrap();
/// w.flush().unwrap();
/// ```
pub struct BinaryWriter {
    inner: BufWriter<File>,
}

impl BinaryWriter {
    /// Create (or overwrite) the file at `path` and return a buffered writer.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::create(path)
            .map_err(|e| IoError::FileError(format!("cannot create {}: {e}", path.display())))?;
        Ok(Self {
            inner: BufWriter::new(file),
        })
    }

    // ── Integer writers ───────────────────────────────────────────────────────

    /// Write a `u8` value.
    pub fn write_u8(&mut self, val: u8) -> Result<()> {
        self.inner
            .write_u8(val)
            .map_err(|e| IoError::FileError(format!("write_u8: {e}")))
    }

    /// Write a `u16` in little-endian byte order.
    pub fn write_u16(&mut self, val: u16) -> Result<()> {
        self.inner
            .write_u16::<LittleEndian>(val)
            .map_err(|e| IoError::FileError(format!("write_u16: {e}")))
    }

    /// Write a `u32` in little-endian byte order.
    pub fn write_u32(&mut self, val: u32) -> Result<()> {
        self.inner
            .write_u32::<LittleEndian>(val)
            .map_err(|e| IoError::FileError(format!("write_u32: {e}")))
    }

    /// Write a `u64` in little-endian byte order.
    pub fn write_u64(&mut self, val: u64) -> Result<()> {
        self.inner
            .write_u64::<LittleEndian>(val)
            .map_err(|e| IoError::FileError(format!("write_u64: {e}")))
    }

    /// Write an `i8` value.
    pub fn write_i8(&mut self, val: i8) -> Result<()> {
        self.inner
            .write_i8(val)
            .map_err(|e| IoError::FileError(format!("write_i8: {e}")))
    }

    /// Write an `i16` in little-endian byte order.
    pub fn write_i16(&mut self, val: i16) -> Result<()> {
        self.inner
            .write_i16::<LittleEndian>(val)
            .map_err(|e| IoError::FileError(format!("write_i16: {e}")))
    }

    /// Write an `i32` in little-endian byte order.
    pub fn write_i32(&mut self, val: i32) -> Result<()> {
        self.inner
            .write_i32::<LittleEndian>(val)
            .map_err(|e| IoError::FileError(format!("write_i32: {e}")))
    }

    /// Write an `i64` in little-endian byte order.
    pub fn write_i64(&mut self, val: i64) -> Result<()> {
        self.inner
            .write_i64::<LittleEndian>(val)
            .map_err(|e| IoError::FileError(format!("write_i64: {e}")))
    }

    // ── Float writers ─────────────────────────────────────────────────────────

    /// Write a `f32` in little-endian byte order.
    pub fn write_f32(&mut self, val: f32) -> Result<()> {
        self.inner
            .write_f32::<LittleEndian>(val)
            .map_err(|e| IoError::FileError(format!("write_f32: {e}")))
    }

    /// Write a `f64` in little-endian byte order.
    pub fn write_f64(&mut self, val: f64) -> Result<()> {
        self.inner
            .write_f64::<LittleEndian>(val)
            .map_err(|e| IoError::FileError(format!("write_f64: {e}")))
    }

    // ── Compound writers ──────────────────────────────────────────────────────

    /// Write a raw byte slice verbatim.
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.inner
            .write_all(bytes)
            .map_err(|e| IoError::FileError(format!("write_bytes: {e}")))
    }

    /// Write a length-prefixed UTF-8 string (`u32` LE length + raw bytes).
    pub fn write_string(&mut self, s: &str) -> Result<()> {
        let bytes = s.as_bytes();
        let len = bytes.len();
        if len > u32::MAX as usize {
            return Err(IoError::SerializationError(format!(
                "string too long ({len} bytes); maximum is {}",
                u32::MAX
            )));
        }
        self.write_u32(len as u32)?;
        self.write_bytes(bytes)
    }

    /// Write a `u64` element count followed by the raw `f64` values in
    /// little-endian byte order.
    pub fn write_array_f64(&mut self, arr: &[f64]) -> Result<()> {
        self.write_u64(arr.len() as u64)?;
        for &v in arr {
            self.write_f64(v)?;
        }
        Ok(())
    }

    /// Flush the underlying [`BufWriter`].
    pub fn flush(&mut self) -> Result<()> {
        self.inner
            .flush()
            .map_err(|e| IoError::FileError(format!("flush: {e}")))
    }
}

// ─────────────────────────────── BinaryReader ────────────────────────────────

/// Type-safe binary reader backed by a [`BufReader<File>`].
///
/// Mirror image of [`BinaryWriter`]: reads the same little-endian
/// encoding that the writer produces.
///
/// # Examples
///
/// ```rust,no_run
/// use scirs2_io::binary_format::BinaryReader;
///
/// let mut r = BinaryReader::open("output.bin").unwrap();
/// let n   = r.read_u32().unwrap();
/// let f   = r.read_f64().unwrap();
/// let s   = r.read_string().unwrap();
/// ```
pub struct BinaryReader {
    inner: BufReader<File>,
}

impl BinaryReader {
    /// Open `path` for buffered reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| IoError::FileNotFound(format!("{}: {e}", path.display())))?;
        Ok(Self {
            inner: BufReader::new(file),
        })
    }

    // ── Integer readers ───────────────────────────────────────────────────────

    /// Read a single `u8`.
    pub fn read_u8(&mut self) -> Result<u8> {
        self.inner
            .read_u8()
            .map_err(|e| IoError::FileError(format!("read_u8: {e}")))
    }

    /// Read a `u16` (little-endian).
    pub fn read_u16(&mut self) -> Result<u16> {
        self.inner
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FileError(format!("read_u16: {e}")))
    }

    /// Read a `u32` (little-endian).
    pub fn read_u32(&mut self) -> Result<u32> {
        self.inner
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FileError(format!("read_u32: {e}")))
    }

    /// Read a `u64` (little-endian).
    pub fn read_u64(&mut self) -> Result<u64> {
        self.inner
            .read_u64::<LittleEndian>()
            .map_err(|e| IoError::FileError(format!("read_u64: {e}")))
    }

    /// Read a single `i8`.
    pub fn read_i8(&mut self) -> Result<i8> {
        self.inner
            .read_i8()
            .map_err(|e| IoError::FileError(format!("read_i8: {e}")))
    }

    /// Read an `i16` (little-endian).
    pub fn read_i16(&mut self) -> Result<i16> {
        self.inner
            .read_i16::<LittleEndian>()
            .map_err(|e| IoError::FileError(format!("read_i16: {e}")))
    }

    /// Read an `i32` (little-endian).
    pub fn read_i32(&mut self) -> Result<i32> {
        self.inner
            .read_i32::<LittleEndian>()
            .map_err(|e| IoError::FileError(format!("read_i32: {e}")))
    }

    /// Read an `i64` (little-endian).
    pub fn read_i64(&mut self) -> Result<i64> {
        self.inner
            .read_i64::<LittleEndian>()
            .map_err(|e| IoError::FileError(format!("read_i64: {e}")))
    }

    // ── Float readers ─────────────────────────────────────────────────────────

    /// Read a `f32` (little-endian).
    pub fn read_f32(&mut self) -> Result<f32> {
        self.inner
            .read_f32::<LittleEndian>()
            .map_err(|e| IoError::FileError(format!("read_f32: {e}")))
    }

    /// Read a `f64` (little-endian).
    pub fn read_f64(&mut self) -> Result<f64> {
        self.inner
            .read_f64::<LittleEndian>()
            .map_err(|e| IoError::FileError(format!("read_f64: {e}")))
    }

    // ── Compound readers ──────────────────────────────────────────────────────

    /// Read exactly `n` bytes into a new `Vec<u8>`.
    pub fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; n];
        self.inner
            .read_exact(&mut buf)
            .map_err(|e| IoError::FileError(format!("read_bytes({n}): {e}")))?;
        Ok(buf)
    }

    /// Read a length-prefixed UTF-8 string (written by [`BinaryWriter::write_string`]).
    pub fn read_string(&mut self) -> Result<String> {
        let len = self.read_u32()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes)
            .map_err(|e| IoError::ParseError(format!("string UTF-8 error: {e}")))
    }

    /// Read a count-prefixed `f64` array (written by [`BinaryWriter::write_array_f64`]).
    pub fn read_array_f64(&mut self) -> Result<Vec<f64>> {
        let count = self.read_u64()? as usize;
        let mut arr = Vec::with_capacity(count);
        for _ in 0..count {
            arr.push(self.read_f64()?);
        }
        Ok(arr)
    }
}

// ─────────────────────────────── ScirsDataFile ───────────────────────────────

/// Structured scientific data file (`SCIRS2DF` format).
///
/// Wraps [`BinaryWriter`] / [`BinaryReader`] with a header and typed record
/// framing.  Use the standalone [`write_scirs`] and [`read_scirs`] functions
/// rather than constructing this type directly.
pub struct ScirsDataFile;

impl ScirsDataFile {
    /// Write `records` to `path` in the `SCIRS2DF` format.
    ///
    /// Equivalent to calling [`write_scirs`].
    pub fn write<P: AsRef<Path>>(path: P, records: &[(String, DataRecord)]) -> Result<()> {
        write_scirs(path, records)
    }

    /// Read all records from a `SCIRS2DF` file at `path`.
    ///
    /// Equivalent to calling [`read_scirs`].
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Vec<(String, DataRecord)>> {
        read_scirs(path)
    }
}

// ─────────────────────────────── write_scirs ─────────────────────────────────

/// Write a sequence of named [`DataRecord`]s to a `SCIRS2DF` binary file.
///
/// The function creates (or truncates) the file at `path`, writes the
/// 10-byte header, then serialises each record in order.
///
/// # Errors
///
/// - [`IoError::FileError`] on any I/O failure.
/// - [`IoError::SerializationError`] if a name or text value is too long.
pub fn write_scirs<P: AsRef<Path>>(path: P, records: &[(String, DataRecord)]) -> Result<()> {
    let mut w = BinaryWriter::create(path)?;

    // Header
    w.write_bytes(MAGIC)?;
    w.write_u8(FORMAT_VERSION)?;

    let n = records.len();
    if n > u32::MAX as usize {
        return Err(IoError::SerializationError(format!(
            "too many records ({n}); max is {}",
            u32::MAX
        )));
    }
    w.write_u32(n as u32)?;

    // Records
    for (name, record) in records {
        w.write_string(name)?;
        w.write_u8(record.tag())?;

        match record {
            DataRecord::Scalar(v) => {
                w.write_f64(*v)?;
            }
            DataRecord::Vector(arr) => {
                w.write_array_f64(arr)?;
            }
            DataRecord::Matrix(rows) => {
                let n_rows = rows.len() as u64;
                let n_cols = rows.first().map(|r| r.len()).unwrap_or(0) as u64;
                w.write_u64(n_rows)?;
                w.write_u64(n_cols)?;
                for row in rows {
                    // Validate row width.
                    if row.len() as u64 != n_cols {
                        return Err(IoError::SerializationError(format!(
                            "jagged matrix: expected {n_cols} columns per row, got {}",
                            row.len()
                        )));
                    }
                    for &v in row {
                        w.write_f64(v)?;
                    }
                }
            }
            DataRecord::Text(s) => {
                w.write_string(s)?;
            }
        }
    }

    w.flush()
}

// ─────────────────────────────── read_scirs ──────────────────────────────────

/// Read all named [`DataRecord`]s from a `SCIRS2DF` binary file.
///
/// # Errors
///
/// - [`IoError::FileNotFound`] if `path` does not exist.
/// - [`IoError::FormatError`] if the magic bytes or version are wrong.
/// - [`IoError::FileError`] / [`IoError::ParseError`] on any read failure.
pub fn read_scirs<P: AsRef<Path>>(path: P) -> Result<Vec<(String, DataRecord)>> {
    let mut r = BinaryReader::open(path)?;

    // Validate magic bytes
    let magic_bytes = r.read_bytes(8)?;
    if magic_bytes != MAGIC {
        return Err(IoError::FormatError(format!(
            "bad magic: expected {:?}, got {:?}",
            MAGIC, magic_bytes
        )));
    }

    // Validate version
    let version = r.read_u8()?;
    if version != FORMAT_VERSION {
        return Err(IoError::FormatError(format!(
            "unsupported SCIRS2DF version {version}; this reader supports only version {FORMAT_VERSION}"
        )));
    }

    let n_records = r.read_u32()? as usize;
    let mut records = Vec::with_capacity(n_records);

    for rec_idx in 0..n_records {
        let name = r
            .read_string()
            .map_err(|e| IoError::ParseError(format!("record {rec_idx}: name read error: {e}")))?;
        let tag = r.read_u8().map_err(|e| {
            IoError::ParseError(format!("record {rec_idx} '{name}': tag read error: {e}"))
        })?;

        let record = match tag {
            0 => {
                let v = r.read_f64().map_err(|e| {
                    IoError::ParseError(format!(
                        "record {rec_idx} '{name}': Scalar read error: {e}"
                    ))
                })?;
                DataRecord::Scalar(v)
            }
            1 => {
                let arr = r.read_array_f64().map_err(|e| {
                    IoError::ParseError(format!(
                        "record {rec_idx} '{name}': Vector read error: {e}"
                    ))
                })?;
                DataRecord::Vector(arr)
            }
            2 => {
                let n_rows = r.read_u64().map_err(|e| {
                    IoError::ParseError(format!(
                        "record {rec_idx} '{name}': Matrix rows count error: {e}"
                    ))
                })? as usize;
                let n_cols = r.read_u64().map_err(|e| {
                    IoError::ParseError(format!(
                        "record {rec_idx} '{name}': Matrix cols count error: {e}"
                    ))
                })? as usize;
                let mut matrix = Vec::with_capacity(n_rows);
                for row_idx in 0..n_rows {
                    let mut row = Vec::with_capacity(n_cols);
                    for col_idx in 0..n_cols {
                        let v = r.read_f64().map_err(|e| {
                            IoError::ParseError(format!(
                                "record {rec_idx} '{name}': Matrix[{row_idx}][{col_idx}] error: {e}"
                            ))
                        })?;
                        row.push(v);
                    }
                    matrix.push(row);
                }
                DataRecord::Matrix(matrix)
            }
            3 => {
                let s = r.read_string().map_err(|e| {
                    IoError::ParseError(format!("record {rec_idx} '{name}': Text read error: {e}"))
                })?;
                DataRecord::Text(s)
            }
            other => {
                return Err(IoError::FormatError(format!(
                    "record {rec_idx} '{name}': unknown type tag {other}"
                )))
            }
        };

        records.push((name, record));
    }

    Ok(records)
}

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("scirs2_binary_format_tests");
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir.join(name)
    }

    // ── BinaryWriter / BinaryReader round-trips ───────────────────────────────

    #[test]
    fn test_u8_roundtrip() {
        let path = temp_path("u8.bin");
        let mut w = BinaryWriter::create(&path).expect("create");
        w.write_u8(0).expect("write 0");
        w.write_u8(255).expect("write 255");
        w.flush().expect("flush");

        let mut r = BinaryReader::open(&path).expect("open");
        assert_eq!(r.read_u8().expect("r0"), 0);
        assert_eq!(r.read_u8().expect("r255"), 255);
    }

    #[test]
    fn test_i8_roundtrip() {
        let path = temp_path("i8.bin");
        let mut w = BinaryWriter::create(&path).expect("create");
        w.write_i8(-128).expect("write");
        w.write_i8(127).expect("write");
        w.flush().expect("flush");

        let mut r = BinaryReader::open(&path).expect("open");
        assert_eq!(r.read_i8().expect("r-128"), -128);
        assert_eq!(r.read_i8().expect("r127"), 127);
    }

    #[test]
    fn test_u16_roundtrip() {
        let path = temp_path("u16.bin");
        let mut w = BinaryWriter::create(&path).expect("create");
        w.write_u16(0x1234).expect("write");
        w.flush().expect("flush");
        let mut r = BinaryReader::open(&path).expect("open");
        assert_eq!(r.read_u16().expect("read"), 0x1234);
    }

    #[test]
    fn test_u32_roundtrip() {
        let path = temp_path("u32.bin");
        let mut w = BinaryWriter::create(&path).expect("create");
        w.write_u32(0xDEAD_BEEF).expect("write");
        w.flush().expect("flush");
        let mut r = BinaryReader::open(&path).expect("open");
        assert_eq!(r.read_u32().expect("read"), 0xDEAD_BEEF);
    }

    #[test]
    fn test_u64_roundtrip() {
        let path = temp_path("u64.bin");
        let mut w = BinaryWriter::create(&path).expect("create");
        w.write_u64(u64::MAX).expect("write");
        w.flush().expect("flush");
        let mut r = BinaryReader::open(&path).expect("open");
        assert_eq!(r.read_u64().expect("read"), u64::MAX);
    }

    #[test]
    fn test_i16_roundtrip() {
        let path = temp_path("i16.bin");
        let mut w = BinaryWriter::create(&path).expect("create");
        w.write_i16(-32000).expect("write");
        w.flush().expect("flush");
        let mut r = BinaryReader::open(&path).expect("open");
        assert_eq!(r.read_i16().expect("read"), -32000);
    }

    #[test]
    fn test_i32_roundtrip() {
        let path = temp_path("i32.bin");
        let mut w = BinaryWriter::create(&path).expect("create");
        w.write_i32(-1_000_000).expect("write");
        w.flush().expect("flush");
        let mut r = BinaryReader::open(&path).expect("open");
        assert_eq!(r.read_i32().expect("read"), -1_000_000);
    }

    #[test]
    fn test_i64_roundtrip() {
        let path = temp_path("i64.bin");
        let mut w = BinaryWriter::create(&path).expect("create");
        w.write_i64(i64::MIN).expect("write");
        w.flush().expect("flush");
        let mut r = BinaryReader::open(&path).expect("open");
        assert_eq!(r.read_i64().expect("read"), i64::MIN);
    }

    #[test]
    fn test_f32_roundtrip() {
        let path = temp_path("f32.bin");
        let mut w = BinaryWriter::create(&path).expect("create");
        w.write_f32(std::f32::consts::E).expect("write");
        w.flush().expect("flush");
        let mut r = BinaryReader::open(&path).expect("open");
        let v = r.read_f32().expect("read");
        assert!((v - std::f32::consts::E).abs() < 1e-5);
    }

    #[test]
    fn test_f64_roundtrip() {
        let path = temp_path("f64.bin");
        let mut w = BinaryWriter::create(&path).expect("create");
        w.write_f64(std::f64::consts::PI).expect("write");
        w.flush().expect("flush");
        let mut r = BinaryReader::open(&path).expect("open");
        let v = r.read_f64().expect("read");
        assert!((v - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn test_bytes_roundtrip() {
        let path = temp_path("bytes.bin");
        let data: Vec<u8> = (0u8..=255).collect();
        let mut w = BinaryWriter::create(&path).expect("create");
        w.write_bytes(&data).expect("write");
        w.flush().expect("flush");
        let mut r = BinaryReader::open(&path).expect("open");
        let read_back = r.read_bytes(256).expect("read");
        assert_eq!(read_back, data);
    }

    #[test]
    fn test_string_roundtrip() {
        let path = temp_path("string.bin");
        let orig = "Hello, SCIRS2 科学 🔬";
        let mut w = BinaryWriter::create(&path).expect("create");
        w.write_string(orig).expect("write");
        w.flush().expect("flush");
        let mut r = BinaryReader::open(&path).expect("open");
        let s = r.read_string().expect("read");
        assert_eq!(s, orig);
    }

    #[test]
    fn test_empty_string_roundtrip() {
        let path = temp_path("empty_str.bin");
        let mut w = BinaryWriter::create(&path).expect("create");
        w.write_string("").expect("write");
        w.flush().expect("flush");
        let mut r = BinaryReader::open(&path).expect("open");
        assert_eq!(r.read_string().expect("read"), "");
    }

    #[test]
    fn test_array_f64_roundtrip() {
        let path = temp_path("arr_f64.bin");
        let arr = vec![1.1, 2.2, 3.3, 4.4, 5.5];
        let mut w = BinaryWriter::create(&path).expect("create");
        w.write_array_f64(&arr).expect("write");
        w.flush().expect("flush");
        let mut r = BinaryReader::open(&path).expect("open");
        let read_back = r.read_array_f64().expect("read");
        assert_eq!(read_back.len(), arr.len());
        for (a, b) in arr.iter().zip(read_back.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_empty_array_f64_roundtrip() {
        let path = temp_path("empty_arr.bin");
        let arr: Vec<f64> = vec![];
        let mut w = BinaryWriter::create(&path).expect("create");
        w.write_array_f64(&arr).expect("write");
        w.flush().expect("flush");
        let mut r = BinaryReader::open(&path).expect("open");
        let read_back = r.read_array_f64().expect("read");
        assert!(read_back.is_empty());
    }

    // ── write_scirs / read_scirs round-trips ──────────────────────────────────

    #[test]
    fn test_scirs_scalar_roundtrip() {
        let path = temp_path("scalar.scirs2");
        let records = vec![DataRecord::named(
            "pi",
            DataRecord::Scalar(std::f64::consts::PI),
        )];
        write_scirs(&path, &records).expect("write");
        let loaded = read_scirs(&path).expect("read");
        assert_eq!(loaded.len(), 1);
        let (name, rec) = &loaded[0];
        assert_eq!(name, "pi");
        assert!(matches!(rec, DataRecord::Scalar(v) if (v - std::f64::consts::PI).abs() < 1e-15));
    }

    #[test]
    fn test_scirs_vector_roundtrip() {
        let path = temp_path("vector.scirs2");
        let arr = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let records = vec![DataRecord::named("vec", DataRecord::Vector(arr.clone()))];
        write_scirs(&path, &records).expect("write");
        let loaded = read_scirs(&path).expect("read");
        let (name, rec) = &loaded[0];
        assert_eq!(name, "vec");
        if let DataRecord::Vector(v) = rec {
            assert_eq!(v, &arr);
        } else {
            panic!("expected Vector");
        }
    }

    #[test]
    fn test_scirs_matrix_roundtrip() {
        let path = temp_path("matrix.scirs2");
        let matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let records = vec![DataRecord::named("mat", DataRecord::Matrix(matrix.clone()))];
        write_scirs(&path, &records).expect("write");
        let loaded = read_scirs(&path).expect("read");
        let (name, rec) = &loaded[0];
        assert_eq!(name, "mat");
        if let DataRecord::Matrix(m) = rec {
            assert_eq!(m, &matrix);
        } else {
            panic!("expected Matrix");
        }
    }

    #[test]
    fn test_scirs_text_roundtrip() {
        let path = temp_path("text.scirs2");
        let text = "SciRS2 binary format test — 科学 🧪".to_string();
        let records = vec![DataRecord::named("desc", DataRecord::Text(text.clone()))];
        write_scirs(&path, &records).expect("write");
        let loaded = read_scirs(&path).expect("read");
        let (name, rec) = &loaded[0];
        assert_eq!(name, "desc");
        if let DataRecord::Text(s) = rec {
            assert_eq!(s, &text);
        } else {
            panic!("expected Text");
        }
    }

    #[test]
    fn test_scirs_multiple_records_roundtrip() {
        let path = temp_path("multi.scirs2");
        let records = vec![
            DataRecord::named("alpha", DataRecord::Scalar(1.0)),
            DataRecord::named("beta", DataRecord::Vector(vec![10.0, 20.0, 30.0])),
            DataRecord::named(
                "gamma",
                DataRecord::Matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
            ),
            DataRecord::named("delta", DataRecord::Text("delta record".to_string())),
        ];
        write_scirs(&path, &records).expect("write");
        let loaded = read_scirs(&path).expect("read");
        assert_eq!(loaded.len(), 4);
        assert!(matches!(&loaded[0].1, DataRecord::Scalar(v) if (v - 1.0).abs() < 1e-15));
        assert!(matches!(&loaded[1].1, DataRecord::Vector(v) if v.len() == 3));
        assert!(matches!(&loaded[2].1, DataRecord::Matrix(m) if m.len() == 2));
        assert!(matches!(&loaded[3].1, DataRecord::Text(s) if s == "delta record"));
    }

    #[test]
    fn test_scirs_empty_file() {
        let path = temp_path("empty.scirs2");
        write_scirs(&path, &[]).expect("write");
        let loaded = read_scirs(&path).expect("read");
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_scirs_wrong_magic_is_error() {
        use std::io::Write;
        let path = temp_path("bad_magic.scirs2");
        let mut f = File::create(&path).expect("create");
        f.write_all(b"BADMAGIC\x01\x00\x00\x00\x00").expect("write");
        assert!(read_scirs(&path).is_err());
    }

    #[test]
    fn test_scirs_wrong_version_is_error() {
        use std::io::Write;
        let path = temp_path("bad_version.scirs2");
        let mut f = File::create(&path).expect("create");
        // Correct magic, wrong version (99)
        f.write_all(b"SCIRS2DF").expect("magic");
        f.write_all(&[99u8]).expect("version");
        f.write_all(&[0u8; 4]).expect("count");
        assert!(read_scirs(&path).is_err());
    }

    #[test]
    fn test_scirs_jagged_matrix_is_error() {
        let path = temp_path("jagged.scirs2");
        let jagged_matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0], // shorter row — should be rejected
        ];
        let records = vec![DataRecord::named("bad", DataRecord::Matrix(jagged_matrix))];
        assert!(write_scirs(&path, &records).is_err());
    }

    #[test]
    fn test_scirs_empty_vector_roundtrip() {
        let path = temp_path("empty_vec.scirs2");
        let records = vec![DataRecord::named("empty", DataRecord::Vector(vec![]))];
        write_scirs(&path, &records).expect("write");
        let loaded = read_scirs(&path).expect("read");
        assert!(matches!(&loaded[0].1, DataRecord::Vector(v) if v.is_empty()));
    }

    #[test]
    fn test_scirs_data_file_struct_api() {
        let path = temp_path("struct_api.scirs2");
        let records = vec![DataRecord::named("x", DataRecord::Scalar(2.0))];
        ScirsDataFile::write(&path, &records).expect("write");
        let loaded = ScirsDataFile::read(&path).expect("read");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "x");
    }

    #[test]
    fn test_data_record_named_helper() {
        let (name, rec) = DataRecord::named("foo", DataRecord::Scalar(42.0));
        assert_eq!(name, "foo");
        assert!(matches!(rec, DataRecord::Scalar(v) if (v - 42.0).abs() < 1e-15));
    }
}
