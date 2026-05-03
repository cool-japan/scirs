//! ORC-lite columnar format
//!
//! A simplified implementation of the Apache ORC columnar storage format,
//! supporting stripes, a file footer, and the ORC integer RLE v2 codec.
//!
//! Format layout:
//! ```text
//! [MAGIC "ORCLITE\0"] [stripe_0] ... [stripe_N] [FOOTER] [footer_length: u32 LE]
//! ```
//!
//! Each stripe:
//! ```text
//! [STRIPE_MAGIC "STRP"] [num_rows: u32 LE] [num_cols: u32 LE]
//!   [col_0_header] ... [col_N_header]
//!   [col_0_data]   ... [col_N_data]
//! ```
//!
//! Column header: `[name_len: u16 LE][name bytes][encoding: u8][data_len: u32 LE]`

use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::{IoError, Result};

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

const FILE_MAGIC: &[u8; 8] = b"ORCLITE\0";
const STRIPE_MAGIC: &[u8; 4] = b"STRP";
const FOOTER_MAGIC: &[u8; 4] = b"FOOT";
const FILE_MAGIC_LEN: usize = 8;

// ──────────────────────────────────────────────────────────────────────────────
// Column encoding
// ──────────────────────────────────────────────────────────────────────────────

/// Column data encoding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ColumnEncoding {
    /// Raw values stored directly (8 bytes per element for f64, 8 for i64).
    Direct = 0,
    /// Dictionary encoding: values replaced by u16 indices into a dictionary.
    Dictionary = 1,
    /// ORC integer RLE v2 (delta + base encoding for i64 columns).
    RleV2 = 2,
    /// UTF-8 length-prefixed strings (variable width).
    DirectString = 3,
}

impl ColumnEncoding {
    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(ColumnEncoding::Direct),
            1 => Ok(ColumnEncoding::Dictionary),
            2 => Ok(ColumnEncoding::RleV2),
            3 => Ok(ColumnEncoding::DirectString),
            other => Err(IoError::FormatError(format!(
                "Unknown ORC-lite column encoding byte: {}",
                other
            ))),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Column data variants
// ──────────────────────────────────────────────────────────────────────────────

/// In-memory column data.
#[derive(Debug, Clone, PartialEq)]
pub enum OrcColumnData {
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    Boolean(Vec<bool>),
    String(Vec<String>),
    /// Nullable integer (None = null).
    NullableInt64(Vec<Option<i64>>),
}

impl OrcColumnData {
    /// Number of values in the column.
    pub fn len(&self) -> usize {
        match self {
            OrcColumnData::Int64(v) => v.len(),
            OrcColumnData::Float64(v) => v.len(),
            OrcColumnData::Boolean(v) => v.len(),
            OrcColumnData::String(v) => v.len(),
            OrcColumnData::NullableInt64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn type_tag(&self) -> u8 {
        match self {
            OrcColumnData::Int64(_) => 0,
            OrcColumnData::Float64(_) => 1,
            OrcColumnData::Boolean(_) => 2,
            OrcColumnData::String(_) => 3,
            OrcColumnData::NullableInt64(_) => 4,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ORC Integer RLE v2
// ──────────────────────────────────────────────────────────────────────────────

/// ORC integer RLE v2 encoder/decoder.
///
/// This is a simplified subset focusing on DELTA sub-encoding:
/// 1. Compute deltas from the base value.
/// 2. Encode with variable-length zigzag encoding (LEB-128 signed).
pub struct IntRleV2;

impl IntRleV2 {
    /// Encode a slice of `i64` values using delta + zigzag LEB-128.
    ///
    /// Output format:
    /// ```text
    /// [count: u32 LE][base: i64 zigzag LEB128][delta_0 ... delta_N-1 zigzag LEB128]
    /// ```
    pub fn encode(values: &[i64]) -> Vec<u8> {
        let mut out = Vec::new();

        // count
        let count = values.len() as u32;
        out.extend_from_slice(&count.to_le_bytes());

        if values.is_empty() {
            return out;
        }

        // base value
        encode_zigzag_leb128(values[0], &mut out);

        // deltas
        for i in 1..values.len() {
            let delta = values[i].wrapping_sub(values[i - 1]);
            encode_zigzag_leb128(delta, &mut out);
        }

        out
    }

    /// Decode bytes produced by `encode` back into `Vec<i64>`.
    pub fn decode(data: &[u8]) -> Result<Vec<i64>> {
        if data.len() < 4 {
            return Err(IoError::FormatError(
                "RLE v2: too short to read count".into(),
            ));
        }
        let count = u32::from_le_bytes(
            data[0..4]
                .try_into()
                .map_err(|_| IoError::FormatError("RLE v2: count bytes".into()))?,
        ) as usize;

        if count == 0 {
            return Ok(Vec::new());
        }

        let mut pos = 4;
        let (base, consumed) = decode_zigzag_leb128(&data[pos..])?;
        pos += consumed;

        let mut values = Vec::with_capacity(count);
        values.push(base);

        for _ in 1..count {
            let (delta, consumed) = decode_zigzag_leb128(&data[pos..])?;
            pos += consumed;
            let prev = *values.last().expect("values non-empty");
            values.push(prev.wrapping_add(delta));
        }

        Ok(values)
    }
}

/// Encode a signed `i64` with zigzag mapping and then LEB-128.
fn encode_zigzag_leb128(value: i64, out: &mut Vec<u8>) {
    // Zigzag: 0->0, -1->1, 1->2, -2->3, 2->4 ...
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    let mut v = zigzag;
    loop {
        let low7 = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(low7);
            break;
        } else {
            out.push(low7 | 0x80);
        }
    }
}

/// Decode a zigzag LEB-128 value; returns `(value, bytes_consumed)`.
fn decode_zigzag_leb128(data: &[u8]) -> Result<(i64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    for (i, &byte) in data.iter().enumerate() {
        result |= ((byte & 0x7F) as u64) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            // Undo zigzag
            let signed = ((result >> 1) as i64) ^ -((result & 1) as i64);
            return Ok((signed, i + 1));
        }
        if shift >= 64 {
            return Err(IoError::FormatError(
                "zigzag LEB-128: overflow (shift >= 64)".into(),
            ));
        }
    }
    Err(IoError::FormatError(
        "zigzag LEB-128: truncated encoding".into(),
    ))
}

// ──────────────────────────────────────────────────────────────────────────────
// Stripe metadata
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct StripeInfo {
    /// Byte offset of stripe start in the file.
    offset: u64,
    /// Number of rows in the stripe.
    num_rows: u32,
    /// Column names in order.
    column_names: Vec<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// OrcLiteWriter
// ──────────────────────────────────────────────────────────────────────────────

/// Writer for the ORC-lite columnar format.
///
/// Data is written in stripes. Each stripe is self-contained.
/// After all stripes are written, `finalize()` writes the footer.
///
/// # Example
/// ```no_run
/// use scirs2_io::formats::orc_lite::{OrcLiteWriter, OrcColumnData};
/// use std::fs::File;
///
/// let file = File::create("/tmp/data.orc").unwrap();
/// let mut writer = OrcLiteWriter::new(file).unwrap();
/// let cols = vec![
///     ("id".to_string(), OrcColumnData::Int64(vec![1, 2, 3])),
///     ("value".to_string(), OrcColumnData::Float64(vec![1.1, 2.2, 3.3])),
/// ];
/// writer.write_stripe(&cols).unwrap();
/// writer.finalize().unwrap();
/// ```
pub struct OrcLiteWriter<W: Write + Seek> {
    inner: BufWriter<W>,
    stripes: Vec<StripeInfo>,
    current_offset: u64,
}

impl<W: Write + Seek> OrcLiteWriter<W> {
    /// Create a new writer, writing the file magic header.
    pub fn new(writer: W) -> Result<Self> {
        let mut bw = BufWriter::new(writer);
        bw.write_all(FILE_MAGIC).map_err(IoError::Io)?;
        Ok(OrcLiteWriter {
            inner: bw,
            stripes: Vec::new(),
            current_offset: FILE_MAGIC_LEN as u64,
        })
    }

    /// Write one stripe (a batch of columns of equal length).
    ///
    /// All columns must have the same number of rows.
    pub fn write_stripe(&mut self, columns: &[(String, OrcColumnData)]) -> Result<()> {
        if columns.is_empty() {
            return Ok(());
        }

        // Validate all columns have equal length
        let num_rows = columns[0].1.len();
        for (name, col) in columns {
            if col.len() != num_rows {
                return Err(IoError::ValidationError(format!(
                    "column '{}' has {} rows, expected {}",
                    name,
                    col.len(),
                    num_rows
                )));
            }
        }

        let stripe_offset = self.current_offset;

        // STRIPE_MAGIC
        self.inner.write_all(STRIPE_MAGIC).map_err(IoError::Io)?;
        self.current_offset += 4;

        // num_rows, num_cols
        let num_rows_u32 = num_rows as u32;
        let num_cols_u32 = columns.len() as u32;
        self.inner
            .write_all(&num_rows_u32.to_le_bytes())
            .map_err(IoError::Io)?;
        self.inner
            .write_all(&num_cols_u32.to_le_bytes())
            .map_err(IoError::Io)?;
        self.current_offset += 8;

        // Encode each column
        let mut encoded: Vec<(String, ColumnEncoding, u8, Vec<u8>)> = Vec::new();
        for (name, col) in columns {
            let (enc, type_tag, bytes) = encode_column(col)?;
            encoded.push((name.clone(), enc, type_tag, bytes));
        }

        // Write column headers
        for (name, enc, type_tag, bytes) in &encoded {
            let name_bytes = name.as_bytes();
            let name_len = name_bytes.len() as u16;
            self.inner
                .write_all(&name_len.to_le_bytes())
                .map_err(IoError::Io)?;
            self.inner.write_all(name_bytes).map_err(IoError::Io)?;
            self.inner.write_all(&[*enc as u8]).map_err(IoError::Io)?;
            self.inner.write_all(&[*type_tag]).map_err(IoError::Io)?;
            let data_len = bytes.len() as u32;
            self.inner
                .write_all(&data_len.to_le_bytes())
                .map_err(IoError::Io)?;
            self.current_offset += 2 + name_bytes.len() as u64 + 1 + 1 + 4;
        }

        // Write column data
        for (_, _, _, bytes) in &encoded {
            self.inner.write_all(bytes).map_err(IoError::Io)?;
            self.current_offset += bytes.len() as u64;
        }

        self.stripes.push(StripeInfo {
            offset: stripe_offset,
            num_rows: num_rows_u32,
            column_names: columns.iter().map(|(n, _)| n.clone()).collect(),
        });

        Ok(())
    }

    /// Finalise the file by writing the footer and flushing.
    pub fn finalize(mut self) -> Result<()> {
        let footer_offset = self.current_offset;

        // FOOTER_MAGIC
        self.inner.write_all(FOOTER_MAGIC).map_err(IoError::Io)?;

        // Number of stripes
        let n_stripes = self.stripes.len() as u32;
        self.inner
            .write_all(&n_stripes.to_le_bytes())
            .map_err(IoError::Io)?;

        // Per-stripe info
        for stripe in &self.stripes {
            self.inner
                .write_all(&stripe.offset.to_le_bytes())
                .map_err(IoError::Io)?;
            self.inner
                .write_all(&stripe.num_rows.to_le_bytes())
                .map_err(IoError::Io)?;
            let n_cols = stripe.column_names.len() as u32;
            self.inner
                .write_all(&n_cols.to_le_bytes())
                .map_err(IoError::Io)?;
            for col_name in &stripe.column_names {
                let bytes = col_name.as_bytes();
                let len = bytes.len() as u16;
                self.inner
                    .write_all(&len.to_le_bytes())
                    .map_err(IoError::Io)?;
                self.inner.write_all(bytes).map_err(IoError::Io)?;
            }
        }

        // Footer length (from FOOT_MAGIC to here, not including the length field itself)
        let footer_length =
            (self.inner.stream_position().map_err(IoError::Io)? - footer_offset) as u32;
        self.inner
            .write_all(&footer_length.to_le_bytes())
            .map_err(IoError::Io)?;

        self.inner.flush().map_err(IoError::Io)?;
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// OrcLiteReader
// ──────────────────────────────────────────────────────────────────────────────

/// Metadata about a stripe.
#[derive(Debug, Clone)]
pub struct StripeMetadata {
    /// Byte offset of stripe in the file.
    pub offset: u64,
    /// Number of rows.
    pub num_rows: u32,
    /// Column names.
    pub column_names: Vec<String>,
}

/// Reader for the ORC-lite columnar format.
///
/// First call `open()` to read and validate the file header and footer.
/// Then iterate over stripes via `read_stripe()`.
pub struct OrcLiteReader<R: Read + Seek> {
    inner: BufReader<R>,
    pub stripes: Vec<StripeMetadata>,
}

impl<R: Read + Seek> OrcLiteReader<R> {
    /// Open an ORC-lite file and read its footer/stripe index.
    pub fn open(reader: R) -> Result<Self> {
        let mut br = BufReader::new(reader);

        // Validate magic
        let mut magic = [0u8; FILE_MAGIC_LEN];
        br.read_exact(&mut magic).map_err(IoError::Io)?;
        if &magic != FILE_MAGIC {
            return Err(IoError::FormatError(
                "Not an ORC-lite file (bad magic)".into(),
            ));
        }

        // Seek to the last 4 bytes to read footer_length
        let file_size = br.seek(SeekFrom::End(0)).map_err(IoError::Io)?;
        if file_size < FILE_MAGIC_LEN as u64 + 4 {
            return Err(IoError::FormatError("ORC-lite file too small".into()));
        }
        br.seek(SeekFrom::End(-4)).map_err(IoError::Io)?;
        let mut fl_bytes = [0u8; 4];
        br.read_exact(&mut fl_bytes).map_err(IoError::Io)?;
        let footer_length = u32::from_le_bytes(fl_bytes) as u64;

        // Seek to footer start
        let footer_start = file_size - 4 - footer_length;
        br.seek(SeekFrom::Start(footer_start))
            .map_err(IoError::Io)?;

        // Validate footer magic
        let mut fmagic = [0u8; 4];
        br.read_exact(&mut fmagic).map_err(IoError::Io)?;
        if &fmagic != FOOTER_MAGIC {
            return Err(IoError::FormatError("ORC-lite: bad footer magic".into()));
        }

        // Read number of stripes
        let mut n_stripes_bytes = [0u8; 4];
        br.read_exact(&mut n_stripes_bytes).map_err(IoError::Io)?;
        let n_stripes = u32::from_le_bytes(n_stripes_bytes) as usize;

        let mut stripes = Vec::with_capacity(n_stripes);
        for _ in 0..n_stripes {
            let mut buf8 = [0u8; 8];
            br.read_exact(&mut buf8).map_err(IoError::Io)?;
            let offset = u64::from_le_bytes(buf8);

            let mut buf4 = [0u8; 4];
            br.read_exact(&mut buf4).map_err(IoError::Io)?;
            let num_rows = u32::from_le_bytes(buf4);

            br.read_exact(&mut buf4).map_err(IoError::Io)?;
            let n_cols = u32::from_le_bytes(buf4) as usize;

            let mut column_names = Vec::with_capacity(n_cols);
            for _ in 0..n_cols {
                let mut len_bytes = [0u8; 2];
                br.read_exact(&mut len_bytes).map_err(IoError::Io)?;
                let name_len = u16::from_le_bytes(len_bytes) as usize;
                let mut name_bytes = vec![0u8; name_len];
                br.read_exact(&mut name_bytes).map_err(IoError::Io)?;
                let name = String::from_utf8(name_bytes).map_err(|e| {
                    IoError::FormatError(format!("ORC-lite: invalid UTF-8 column name: {}", e))
                })?;
                column_names.push(name);
            }

            stripes.push(StripeMetadata {
                offset,
                num_rows,
                column_names,
            });
        }

        Ok(OrcLiteReader { inner: br, stripes })
    }

    /// Read all columns from stripe `stripe_idx`.
    pub fn read_stripe(&mut self, stripe_idx: usize) -> Result<HashMap<String, OrcColumnData>> {
        let stripe = self.stripes.get(stripe_idx).ok_or_else(|| {
            IoError::NotFound(format!(
                "ORC-lite: stripe index {} out of range",
                stripe_idx
            ))
        })?;
        let stripe_offset = stripe.offset;
        let num_rows = stripe.num_rows as usize;
        let num_cols = stripe.column_names.len();

        self.inner
            .seek(SeekFrom::Start(stripe_offset))
            .map_err(IoError::Io)?;

        // Validate stripe magic
        let mut smagic = [0u8; 4];
        self.inner.read_exact(&mut smagic).map_err(IoError::Io)?;
        if &smagic != STRIPE_MAGIC {
            return Err(IoError::FormatError("ORC-lite: bad stripe magic".into()));
        }

        // Skip num_rows and num_cols (already known from footer)
        let mut skip8 = [0u8; 8];
        self.inner.read_exact(&mut skip8).map_err(IoError::Io)?;

        // Read column headers
        let mut col_headers: Vec<(String, ColumnEncoding, u8, u32)> = Vec::with_capacity(num_cols);
        for _ in 0..num_cols {
            let mut len_bytes = [0u8; 2];
            self.inner.read_exact(&mut len_bytes).map_err(IoError::Io)?;
            let name_len = u16::from_le_bytes(len_bytes) as usize;
            let mut name_bytes = vec![0u8; name_len];
            self.inner
                .read_exact(&mut name_bytes)
                .map_err(IoError::Io)?;
            let name = String::from_utf8(name_bytes).map_err(|e| {
                IoError::FormatError(format!("ORC-lite stripe: invalid UTF-8 column name: {}", e))
            })?;

            let mut enc_byte = [0u8; 1];
            self.inner.read_exact(&mut enc_byte).map_err(IoError::Io)?;
            let enc = ColumnEncoding::from_u8(enc_byte[0])?;

            let mut type_tag_byte = [0u8; 1];
            self.inner
                .read_exact(&mut type_tag_byte)
                .map_err(IoError::Io)?;
            let type_tag = type_tag_byte[0];

            let mut data_len_bytes = [0u8; 4];
            self.inner
                .read_exact(&mut data_len_bytes)
                .map_err(IoError::Io)?;
            let data_len = u32::from_le_bytes(data_len_bytes);

            col_headers.push((name, enc, type_tag, data_len));
        }

        // Read column data
        let mut result = HashMap::new();
        for (name, enc, type_tag, data_len) in col_headers {
            let mut data_bytes = vec![0u8; data_len as usize];
            self.inner
                .read_exact(&mut data_bytes)
                .map_err(IoError::Io)?;
            let col_data = decode_column(&data_bytes, enc, type_tag, num_rows)?;
            result.insert(name, col_data);
        }

        Ok(result)
    }

    /// Read all stripes and concatenate their columns.
    pub fn read_all(&mut self) -> Result<HashMap<String, OrcColumnData>> {
        let n = self.stripes.len();
        if n == 0 {
            return Ok(HashMap::new());
        }

        let mut combined: HashMap<String, OrcColumnData> = HashMap::new();

        for i in 0..n {
            let stripe_data = self.read_stripe(i)?;
            for (name, col) in stripe_data {
                let entry = combined.entry(name).or_insert_with(|| match &col {
                    OrcColumnData::Int64(_) => OrcColumnData::Int64(Vec::new()),
                    OrcColumnData::Float64(_) => OrcColumnData::Float64(Vec::new()),
                    OrcColumnData::Boolean(_) => OrcColumnData::Boolean(Vec::new()),
                    OrcColumnData::String(_) => OrcColumnData::String(Vec::new()),
                    OrcColumnData::NullableInt64(_) => OrcColumnData::NullableInt64(Vec::new()),
                });
                concat_columns(entry, col)?;
            }
        }

        Ok(combined)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Encode / decode helpers
// ──────────────────────────────────────────────────────────────────────────────

fn encode_column(col: &OrcColumnData) -> Result<(ColumnEncoding, u8, Vec<u8>)> {
    let type_tag = col.type_tag();
    match col {
        OrcColumnData::Int64(v) => {
            // Use RLE v2 for integer columns
            let encoded = IntRleV2::encode(v);
            Ok((ColumnEncoding::RleV2, type_tag, encoded))
        }
        OrcColumnData::Float64(v) => {
            // Direct f64 LE encoding
            let mut bytes = Vec::with_capacity(v.len() * 8);
            for &x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            Ok((ColumnEncoding::Direct, type_tag, bytes))
        }
        OrcColumnData::Boolean(v) => {
            // Bit-packed: 8 booleans per byte
            let mut bytes = Vec::with_capacity((v.len() + 7) / 8 + 4);
            let count = v.len() as u32;
            bytes.extend_from_slice(&count.to_le_bytes());
            let mut byte = 0u8;
            for (i, &b) in v.iter().enumerate() {
                if b {
                    byte |= 1 << (i % 8);
                }
                if i % 8 == 7 {
                    bytes.push(byte);
                    byte = 0;
                }
            }
            if v.len() % 8 != 0 {
                bytes.push(byte);
            }
            Ok((ColumnEncoding::Direct, type_tag, bytes))
        }
        OrcColumnData::String(v) => {
            // Length-prefixed UTF-8 strings: u32 count + [u32 len + bytes]...
            let mut bytes = Vec::new();
            let count = v.len() as u32;
            bytes.extend_from_slice(&count.to_le_bytes());
            for s in v {
                let sb = s.as_bytes();
                let slen = sb.len() as u32;
                bytes.extend_from_slice(&slen.to_le_bytes());
                bytes.extend_from_slice(sb);
            }
            Ok((ColumnEncoding::DirectString, type_tag, bytes))
        }
        OrcColumnData::NullableInt64(v) => {
            // Validity bitmap + values (nulls encoded as 0)
            let count = v.len() as u32;
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&count.to_le_bytes());
            // Validity bitmap
            let mut bitmap_byte = 0u8;
            for (i, opt) in v.iter().enumerate() {
                if opt.is_some() {
                    bitmap_byte |= 1 << (i % 8);
                }
                if i % 8 == 7 {
                    bytes.push(bitmap_byte);
                    bitmap_byte = 0;
                }
            }
            if v.len() % 8 != 0 {
                bytes.push(bitmap_byte);
            }
            // Values (0 for null)
            let vals: Vec<i64> = v.iter().map(|o| o.unwrap_or(0)).collect();
            let encoded = IntRleV2::encode(&vals);
            bytes.extend_from_slice(&encoded);
            Ok((ColumnEncoding::RleV2, type_tag, bytes))
        }
    }
}

fn decode_column(
    data: &[u8],
    enc: ColumnEncoding,
    type_tag: u8,
    _num_rows: usize,
) -> Result<OrcColumnData> {
    match type_tag {
        0 => {
            // Int64
            let values = IntRleV2::decode(data)?;
            Ok(OrcColumnData::Int64(values))
        }
        1 => {
            // Float64
            if !data.len().is_multiple_of(8) {
                return Err(IoError::FormatError(
                    "ORC-lite: f64 column data not multiple of 8 bytes".into(),
                ));
            }
            let values: Vec<f64> = data
                .chunks_exact(8)
                .map(|chunk| {
                    let arr: [u8; 8] = chunk.try_into().expect("chunk is exactly 8 bytes");
                    f64::from_le_bytes(arr)
                })
                .collect();
            Ok(OrcColumnData::Float64(values))
        }
        2 => {
            // Boolean
            if data.len() < 4 {
                return Err(IoError::FormatError(
                    "ORC-lite: boolean column too short".into(),
                ));
            }
            let count = u32::from_le_bytes(
                data[0..4]
                    .try_into()
                    .map_err(|_| IoError::FormatError("boolean count bytes".into()))?,
            ) as usize;
            let mut values = Vec::with_capacity(count);
            for i in 0..count {
                let byte_idx = 4 + i / 8;
                let bit_idx = i % 8;
                let byte = data.get(byte_idx).ok_or_else(|| {
                    IoError::FormatError("ORC-lite: boolean bitmap truncated".into())
                })?;
                values.push((byte >> bit_idx) & 1 == 1);
            }
            Ok(OrcColumnData::Boolean(values))
        }
        3 => {
            // String
            if data.len() < 4 {
                return Err(IoError::FormatError(
                    "ORC-lite: string column too short".into(),
                ));
            }
            let count = u32::from_le_bytes(
                data[0..4]
                    .try_into()
                    .map_err(|_| IoError::FormatError("string count bytes".into()))?,
            ) as usize;
            let mut pos = 4;
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                if pos + 4 > data.len() {
                    return Err(IoError::FormatError(
                        "ORC-lite: string length truncated".into(),
                    ));
                }
                let slen = u32::from_le_bytes(
                    data[pos..pos + 4]
                        .try_into()
                        .map_err(|_| IoError::FormatError("string len bytes".into()))?,
                ) as usize;
                pos += 4;
                if pos + slen > data.len() {
                    return Err(IoError::FormatError(
                        "ORC-lite: string data truncated".into(),
                    ));
                }
                let s = String::from_utf8(data[pos..pos + slen].to_vec()).map_err(|e| {
                    IoError::FormatError(format!("ORC-lite: invalid UTF-8 string: {}", e))
                })?;
                values.push(s);
                pos += slen;
            }
            Ok(OrcColumnData::String(values))
        }
        4 => {
            // NullableInt64
            if data.len() < 4 {
                return Err(IoError::FormatError(
                    "ORC-lite: nullable int64 column too short".into(),
                ));
            }
            let count = u32::from_le_bytes(
                data[0..4]
                    .try_into()
                    .map_err(|_| IoError::FormatError("nullable count bytes".into()))?,
            ) as usize;

            let bitmap_bytes = (count + 7) / 8;
            let bitmap_end = 4 + bitmap_bytes;
            if bitmap_end > data.len() {
                return Err(IoError::FormatError(
                    "ORC-lite: nullable bitmap truncated".into(),
                ));
            }

            let values_raw = IntRleV2::decode(&data[bitmap_end..])?;
            let mut values = Vec::with_capacity(count);
            for i in 0..count {
                let byte_idx = 4 + i / 8;
                let bit_idx = i % 8;
                let byte = data
                    .get(byte_idx)
                    .ok_or_else(|| IoError::FormatError("ORC-lite: nullable bitmap read".into()))?;
                let is_valid = (byte >> bit_idx) & 1 == 1;
                values.push(if is_valid {
                    values_raw.get(i).copied().map(Some).unwrap_or(None)
                } else {
                    None
                });
            }
            Ok(OrcColumnData::NullableInt64(values))
        }
        other => Err(IoError::FormatError(format!(
            "ORC-lite: unknown type tag {}",
            other
        ))),
    }
}

/// Append `src` into `dst` (must be same variant).
fn concat_columns(dst: &mut OrcColumnData, src: OrcColumnData) -> Result<()> {
    match (dst, src) {
        (OrcColumnData::Int64(a), OrcColumnData::Int64(b)) => a.extend(b),
        (OrcColumnData::Float64(a), OrcColumnData::Float64(b)) => a.extend(b),
        (OrcColumnData::Boolean(a), OrcColumnData::Boolean(b)) => a.extend(b),
        (OrcColumnData::String(a), OrcColumnData::String(b)) => a.extend(b),
        (OrcColumnData::NullableInt64(a), OrcColumnData::NullableInt64(b)) => a.extend(b),
        _ => {
            return Err(IoError::FormatError(
                "ORC-lite: column type mismatch across stripes".into(),
            ))
        }
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Convenience free functions
// ──────────────────────────────────────────────────────────────────────────────

/// Write columnar data to an ORC-lite file on disk.
pub fn write_orc_lite<P: AsRef<Path>>(path: P, columns: &[(String, OrcColumnData)]) -> Result<()> {
    let file = std::fs::File::create(path.as_ref()).map_err(IoError::Io)?;
    let mut writer = OrcLiteWriter::new(file)?;
    writer.write_stripe(columns)?;
    writer.finalize()
}

/// Read all data from an ORC-lite file on disk.
pub fn read_orc_lite<P: AsRef<Path>>(path: P) -> Result<HashMap<String, OrcColumnData>> {
    let file = std::fs::File::open(path.as_ref()).map_err(IoError::Io)?;
    let mut reader = OrcLiteReader::open(file)?;
    reader.read_all()
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // Helper: write to a Vec<u8> and read back
    fn roundtrip(cols: &[(String, OrcColumnData)]) -> HashMap<String, OrcColumnData> {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = Cursor::new(&mut buf);
            let mut writer = OrcLiteWriter::new(cursor).expect("writer create");
            writer.write_stripe(cols).expect("write stripe");
            writer.finalize().expect("finalize");
        }
        let cursor = Cursor::new(&buf[..]);
        let mut reader = OrcLiteReader::open(cursor).expect("reader open");
        reader.read_all().expect("read all")
    }

    #[test]
    fn test_int64_roundtrip() {
        let vals: Vec<i64> = vec![1, 2, 3, 100, -5, 0, 1000];
        let cols = vec![("ids".to_string(), OrcColumnData::Int64(vals.clone()))];
        let result = roundtrip(&cols);
        assert_eq!(result["ids"], OrcColumnData::Int64(vals));
    }

    #[test]
    fn test_float64_roundtrip() {
        let vals: Vec<f64> = vec![1.1, 2.2, -3.3, 0.0, 1e10];
        let cols = vec![("values".to_string(), OrcColumnData::Float64(vals.clone()))];
        let result = roundtrip(&cols);
        assert_eq!(result["values"], OrcColumnData::Float64(vals));
    }

    #[test]
    fn test_boolean_roundtrip() {
        let vals: Vec<bool> = vec![true, false, true, true, false];
        let cols = vec![("flags".to_string(), OrcColumnData::Boolean(vals.clone()))];
        let result = roundtrip(&cols);
        assert_eq!(result["flags"], OrcColumnData::Boolean(vals));
    }

    #[test]
    fn test_string_roundtrip() {
        let vals: Vec<String> = vec!["hello".into(), "world".into(), "ORC-lite".into(), "".into()];
        let cols = vec![("names".to_string(), OrcColumnData::String(vals.clone()))];
        let result = roundtrip(&cols);
        assert_eq!(result["names"], OrcColumnData::String(vals));
    }

    #[test]
    fn test_nullable_int64_roundtrip() {
        let vals: Vec<Option<i64>> = vec![Some(1), None, Some(3), None, Some(99)];
        let cols = vec![(
            "opt_ids".to_string(),
            OrcColumnData::NullableInt64(vals.clone()),
        )];
        let result = roundtrip(&cols);
        assert_eq!(result["opt_ids"], OrcColumnData::NullableInt64(vals));
    }

    #[test]
    fn test_multi_column_roundtrip() {
        let n = 100;
        let ids: Vec<i64> = (0..n).collect();
        let vals: Vec<f64> = (0..n).map(|i| i as f64 * 1.5).collect();
        let labels: Vec<String> = (0..n).map(|i| format!("item_{}", i)).collect();

        let cols = vec![
            ("id".to_string(), OrcColumnData::Int64(ids.clone())),
            ("val".to_string(), OrcColumnData::Float64(vals.clone())),
            ("label".to_string(), OrcColumnData::String(labels.clone())),
        ];
        let result = roundtrip(&cols);
        assert_eq!(result["id"], OrcColumnData::Int64(ids));
        assert_eq!(result["val"], OrcColumnData::Float64(vals));
        assert_eq!(result["label"], OrcColumnData::String(labels));
    }

    #[test]
    fn test_multi_stripe_roundtrip() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = Cursor::new(&mut buf);
            let mut writer = OrcLiteWriter::new(cursor).expect("writer");
            for stripe_id in 0..3 {
                let start = stripe_id * 10i64;
                let ids: Vec<i64> = (start..start + 10).collect();
                let cols = vec![("id".to_string(), OrcColumnData::Int64(ids))];
                writer.write_stripe(&cols).expect("write stripe");
            }
            writer.finalize().expect("finalize");
        }

        let cursor = Cursor::new(&buf[..]);
        let mut reader = OrcLiteReader::open(cursor).expect("reader");
        assert_eq!(reader.stripes.len(), 3);
        let all = reader.read_all().expect("read all");
        if let OrcColumnData::Int64(ids) = &all["id"] {
            assert_eq!(ids.len(), 30);
            assert_eq!(ids[0], 0);
            assert_eq!(ids[29], 29);
        } else {
            panic!("expected Int64 column");
        }
    }

    #[test]
    fn test_int_rle_v2_roundtrip() {
        let values: Vec<i64> = vec![0, 1, 1, 2, 3, 5, 8, 13, -7, 100, -100, i64::MAX / 2];
        let encoded = IntRleV2::encode(&values);
        let decoded = IntRleV2::decode(&encoded).expect("decode");
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_int_rle_v2_empty() {
        let encoded = IntRleV2::encode(&[]);
        let decoded = IntRleV2::decode(&encoded).expect("decode empty");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_bad_magic() {
        let cursor = Cursor::new(b"NOTMAGIC!!");
        let result = OrcLiteReader::open(cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_file_roundtrip() {
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("test_orc_lite.orc");

        let ids: Vec<i64> = vec![10, 20, 30];
        let vals: Vec<f64> = vec![1.0, 2.0, 3.0];
        let cols = vec![
            ("id".to_string(), OrcColumnData::Int64(ids.clone())),
            ("val".to_string(), OrcColumnData::Float64(vals.clone())),
        ];

        write_orc_lite(&path, &cols).expect("write file");
        let result = read_orc_lite(&path).expect("read file");

        assert_eq!(result["id"], OrcColumnData::Int64(ids));
        assert_eq!(result["val"], OrcColumnData::Float64(vals));

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }
}
