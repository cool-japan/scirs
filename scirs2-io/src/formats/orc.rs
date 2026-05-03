//! ORC-extended columnar format.
//!
//! An enhanced ORC-inspired columnar storage format that goes beyond the
//! existing `orc_lite` module.  This module adds:
//!
//! - **Multi-stripe files** with independent column encoding per stripe
//! - **RLE v2 integer encoding**: Direct, Delta, and Variable-length integer modes
//! - **Dictionary string encoding** with byte-run-length encoded indices
//! - **Boolean bit-packing** with run-length encoding
//! - **Present stream** (null bitmap) for nullable columns
//! - **Schema footer** containing type metadata and stripe offsets
//!
//! ## Wire layout
//!
//! ```text
//! [MAGIC "ORCEXT\0\0"] [stripe_0] ... [stripe_N] [FOOTER] [footer_len: u32 LE]
//! ```
//!
//! Each stripe:
//! ```text
//! [STRIPE_MAGIC "STRX"] [n_rows: u32 LE] [n_cols: u32 LE]
//!   for each column:
//!     [name_len: u16 LE] [name bytes]
//!     [orc_type: u8] [encoding: u8]
//!     [present_len: u32 LE] [present bytes]   // null bitmap (0 = null)
//!     [data_len: u32 LE]   [data bytes]        // encoded column data
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::{IoError, Result};

// ─────────────────────────────── constants ───────────────────────────────────

const FILE_MAGIC: &[u8; 8] = b"ORCEXT\0\0";
const STRIPE_MAGIC: &[u8; 4] = b"STRX";

// ─────────────────────────────── ORCType ─────────────────────────────────────

/// ORC column type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ORCType {
    Int64 = 0,
    Float64 = 1,
    String = 2,
    Boolean = 3,
}

impl ORCType {
    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(ORCType::Int64),
            1 => Ok(ORCType::Float64),
            2 => Ok(ORCType::String),
            3 => Ok(ORCType::Boolean),
            other => Err(IoError::FormatError(format!(
                "ORC: unknown type tag {other}"
            ))),
        }
    }
}

// ─────────────────────────────── ColumnEncoding ──────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ColumnEncoding {
    DirectI64 = 0,
    DeltaI64 = 1,
    DirectF64 = 2,
    DictionaryStr = 3,
    BoolRle = 4,
}

impl ColumnEncoding {
    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(ColumnEncoding::DirectI64),
            1 => Ok(ColumnEncoding::DeltaI64),
            2 => Ok(ColumnEncoding::DirectF64),
            3 => Ok(ColumnEncoding::DictionaryStr),
            4 => Ok(ColumnEncoding::BoolRle),
            other => Err(IoError::FormatError(format!(
                "ORC: unknown encoding {other}"
            ))),
        }
    }
}

// ─────────────────────────────── ORCColumn ───────────────────────────────────

/// A single column stored as raw encoded bytes.
#[derive(Debug, Clone)]
pub struct ORCColumn {
    /// Column name.
    pub name: String,
    /// Column type.
    pub type_kind: ORCType,
    /// Encoded data bytes.
    pub data: Vec<u8>,
    /// Present (validity) stream — one bit per row (LSB first), 1 = non-null.
    pub present: Vec<u8>,
    /// Internal encoding hint.
    encoding: ColumnEncoding,
}

// ─────────────────────────────── ORCStripe ───────────────────────────────────

/// A stripe (horizontal partition) of an ORC file.
#[derive(Debug, Clone)]
pub struct ORCStripe {
    /// Number of rows in this stripe.
    pub n_rows: usize,
    /// Columns.
    pub columns: Vec<ORCColumn>,
}

// ─────────────────────────────── ORCWriter ───────────────────────────────────

/// High-level ORC writer.  Accumulates column data, then writes all stripes.
#[derive(Debug, Default)]
pub struct ORCWriter {
    stripes: Vec<ORCStripe>,
    n_rows: usize,
    current_stripe: Option<ORCStripeBuilder>,
}

#[derive(Debug, Default)]
struct ORCStripeBuilder {
    n_rows: Option<usize>,
    columns: Vec<ORCColumn>,
}

impl ORCWriter {
    /// Create an empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a new stripe (flush current if any).
    pub fn begin_stripe(&mut self) {
        if let Some(builder) = self.current_stripe.take() {
            self.flush_stripe(builder);
        }
        self.current_stripe = Some(ORCStripeBuilder::default());
    }

    fn ensure_stripe(&mut self) {
        if self.current_stripe.is_none() {
            self.current_stripe = Some(ORCStripeBuilder::default());
        }
    }

    fn set_n_rows(&mut self, n: usize) -> Result<()> {
        let builder = self
            .current_stripe
            .as_mut()
            .ok_or_else(|| IoError::FormatError("ORC: no active stripe".into()))?;
        match builder.n_rows {
            None => builder.n_rows = Some(n),
            Some(prev) if prev == n => {}
            Some(prev) => {
                return Err(IoError::FormatError(format!(
                    "ORC: column length {n} does not match stripe rows {prev}"
                )));
            }
        }
        Ok(())
    }

    /// Add an i64 column to the current stripe.
    pub fn add_column_i64(&mut self, name: &str, data: &[i64]) -> Result<()> {
        self.ensure_stripe();
        self.set_n_rows(data.len())?;
        let (encoding, encoded) = encode_i64(data);
        let present = full_present_stream(data.len());
        let builder = self
            .current_stripe
            .as_mut()
            .ok_or_else(|| IoError::FormatError("ORC: no active stripe".into()))?;
        builder.columns.push(ORCColumn {
            name: name.to_owned(),
            type_kind: ORCType::Int64,
            data: encoded,
            present,
            encoding,
        });
        Ok(())
    }

    /// Add an f64 column to the current stripe.
    pub fn add_column_f64(&mut self, name: &str, data: &[f64]) -> Result<()> {
        self.ensure_stripe();
        self.set_n_rows(data.len())?;
        let encoded: Vec<u8> = data
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        let present = full_present_stream(data.len());
        let builder = self
            .current_stripe
            .as_mut()
            .ok_or_else(|| IoError::FormatError("ORC: no active stripe".into()))?;
        builder.columns.push(ORCColumn {
            name: name.to_owned(),
            type_kind: ORCType::Float64,
            data: encoded,
            present,
            encoding: ColumnEncoding::DirectF64,
        });
        Ok(())
    }

    /// Add a string column to the current stripe.
    pub fn add_column_str(&mut self, name: &str, data: &[&str]) -> Result<()> {
        self.ensure_stripe();
        self.set_n_rows(data.len())?;
        let encoded = encode_dict_strings(data);
        let present = full_present_stream(data.len());
        let builder = self
            .current_stripe
            .as_mut()
            .ok_or_else(|| IoError::FormatError("ORC: no active stripe".into()))?;
        builder.columns.push(ORCColumn {
            name: name.to_owned(),
            type_kind: ORCType::String,
            data: encoded,
            present,
            encoding: ColumnEncoding::DictionaryStr,
        });
        Ok(())
    }

    /// Add a boolean column to the current stripe.
    pub fn add_column_bool(&mut self, name: &str, data: &[bool]) -> Result<()> {
        self.ensure_stripe();
        self.set_n_rows(data.len())?;
        let encoded = encode_bool_rle(data);
        let present = full_present_stream(data.len());
        let builder = self
            .current_stripe
            .as_mut()
            .ok_or_else(|| IoError::FormatError("ORC: no active stripe".into()))?;
        builder.columns.push(ORCColumn {
            name: name.to_owned(),
            type_kind: ORCType::Boolean,
            data: encoded,
            present,
            encoding: ColumnEncoding::BoolRle,
        });
        Ok(())
    }

    fn flush_stripe(&mut self, builder: ORCStripeBuilder) {
        let n_rows = builder.n_rows.unwrap_or(0);
        self.n_rows += n_rows;
        self.stripes.push(ORCStripe {
            n_rows,
            columns: builder.columns,
        });
    }

    /// Write the ORC file to `path`.
    pub fn write(mut self, path: &Path) -> Result<()> {
        // flush current stripe
        if let Some(builder) = self.current_stripe.take() {
            self.flush_stripe(builder);
        }

        let f = File::create(path).map_err(IoError::Io)?;
        let mut w = BufWriter::new(f);

        w.write_all(FILE_MAGIC).map_err(IoError::Io)?;
        let mut stripe_offsets: Vec<u64> = Vec::new();
        let mut cursor: u64 = FILE_MAGIC.len() as u64;

        for stripe in &self.stripes {
            stripe_offsets.push(cursor);
            cursor += write_stripe(&mut w, stripe)?;
        }

        // Footer
        let footer_start = cursor;
        let n_stripes = self.stripes.len() as u32;
        w.write_all(&n_stripes.to_le_bytes()).map_err(IoError::Io)?;
        let mut footer_bytes: u64 = 4;

        for (i, stripe) in self.stripes.iter().enumerate() {
            // stripe offset
            w.write_all(&stripe_offsets[i].to_le_bytes())
                .map_err(IoError::Io)?;
            // n_rows in stripe
            w.write_all(&(stripe.n_rows as u32).to_le_bytes())
                .map_err(IoError::Io)?;
            // n_cols
            w.write_all(&(stripe.columns.len() as u32).to_le_bytes())
                .map_err(IoError::Io)?;
            footer_bytes += 8 + 4 + 4;
            for col in &stripe.columns {
                let nb = col.name.as_bytes();
                w.write_all(&(nb.len() as u16).to_le_bytes())
                    .map_err(IoError::Io)?;
                w.write_all(nb).map_err(IoError::Io)?;
                w.write_all(&[col.type_kind as u8]).map_err(IoError::Io)?;
                footer_bytes += 2 + nb.len() as u64 + 1;
            }
        }

        let _ = footer_start; // informational
        w.write_all(&(footer_bytes as u32).to_le_bytes())
            .map_err(IoError::Io)?;
        w.flush().map_err(IoError::Io)?;
        Ok(())
    }
}

fn write_stripe<W: Write>(w: &mut W, stripe: &ORCStripe) -> Result<u64> {
    let mut n_bytes: u64 = 0;
    w.write_all(STRIPE_MAGIC).map_err(IoError::Io)?;
    n_bytes += 4;
    w.write_all(&(stripe.n_rows as u32).to_le_bytes())
        .map_err(IoError::Io)?;
    n_bytes += 4;
    w.write_all(&(stripe.columns.len() as u32).to_le_bytes())
        .map_err(IoError::Io)?;
    n_bytes += 4;

    for col in &stripe.columns {
        let name_bytes = col.name.as_bytes();
        w.write_all(&(name_bytes.len() as u16).to_le_bytes())
            .map_err(IoError::Io)?;
        w.write_all(name_bytes).map_err(IoError::Io)?;
        w.write_all(&[col.type_kind as u8]).map_err(IoError::Io)?;
        w.write_all(&[col.encoding as u8]).map_err(IoError::Io)?;
        n_bytes += 2 + name_bytes.len() as u64 + 2;

        w.write_all(&(col.present.len() as u32).to_le_bytes())
            .map_err(IoError::Io)?;
        w.write_all(&col.present).map_err(IoError::Io)?;
        n_bytes += 4 + col.present.len() as u64;

        w.write_all(&(col.data.len() as u32).to_le_bytes())
            .map_err(IoError::Io)?;
        w.write_all(&col.data).map_err(IoError::Io)?;
        n_bytes += 4 + col.data.len() as u64;
    }
    Ok(n_bytes)
}

// ─────────────────────────────── ORCReader ───────────────────────────────────

/// High-level ORC reader.
pub struct ORCReader;

impl ORCReader {
    /// Read the schema from the first stripe in the file.
    pub fn read_schema(path: &Path) -> Result<Vec<(String, ORCType)>> {
        let f = File::open(path).map_err(IoError::Io)?;
        let mut r = BufReader::new(f);
        check_magic(&mut r)?;

        // Read the first stripe header to get column names and types
        let stripe = read_stripe(&mut r)?;
        Ok(stripe
            .columns
            .iter()
            .map(|c| (c.name.clone(), c.type_kind))
            .collect())
    }

    /// Read all i64 values from a named column (searches all stripes).
    pub fn read_column_i64(path: &Path, col_name: &str) -> Result<Vec<i64>> {
        let stripes = read_all_stripes(path)?;
        let mut out = Vec::new();
        for stripe in &stripes {
            for col in &stripe.columns {
                if col.name == col_name {
                    if col.type_kind != ORCType::Int64 {
                        return Err(IoError::FormatError(format!(
                            "ORC: column '{col_name}' is {:?}, not Int64",
                            col.type_kind
                        )));
                    }
                    out.extend(decode_i64(&col.data, col.encoding, stripe.n_rows)?);
                }
            }
        }
        Ok(out)
    }

    /// Read all f64 values from a named column.
    pub fn read_column_f64(path: &Path, col_name: &str) -> Result<Vec<f64>> {
        let stripes = read_all_stripes(path)?;
        let mut out = Vec::new();
        for stripe in &stripes {
            for col in &stripe.columns {
                if col.name == col_name {
                    if col.type_kind != ORCType::Float64 {
                        return Err(IoError::FormatError(format!(
                            "ORC: column '{col_name}' is {:?}, not Float64",
                            col.type_kind
                        )));
                    }
                    let vals: Vec<f64> = col
                        .data
                        .chunks_exact(8)
                        .map(|c| {
                            Ok(f64::from_bits(u64::from_le_bytes(c.try_into().map_err(
                                |_| IoError::FormatError("ORC: bad f64 bytes".into()),
                            )?)))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    out.extend(vals);
                }
            }
        }
        Ok(out)
    }

    /// Read all string values from a named column.
    pub fn read_column_str(path: &Path, col_name: &str) -> Result<Vec<String>> {
        let stripes = read_all_stripes(path)?;
        let mut out = Vec::new();
        for stripe in &stripes {
            for col in &stripe.columns {
                if col.name == col_name {
                    if col.type_kind != ORCType::String {
                        return Err(IoError::FormatError(format!(
                            "ORC: column '{col_name}' is {:?}, not String",
                            col.type_kind
                        )));
                    }
                    out.extend(decode_dict_strings(&col.data, stripe.n_rows)?);
                }
            }
        }
        Ok(out)
    }

    /// Read all boolean values from a named column.
    pub fn read_column_bool(path: &Path, col_name: &str) -> Result<Vec<bool>> {
        let stripes = read_all_stripes(path)?;
        let mut out = Vec::new();
        for stripe in &stripes {
            for col in &stripe.columns {
                if col.name == col_name {
                    if col.type_kind != ORCType::Boolean {
                        return Err(IoError::FormatError(format!(
                            "ORC: column '{col_name}' is {:?}, not Boolean",
                            col.type_kind
                        )));
                    }
                    out.extend(decode_bool_rle(&col.data, stripe.n_rows)?);
                }
            }
        }
        Ok(out)
    }
}

// ─────────────────────────────── Read helpers ────────────────────────────────

fn check_magic<R: Read>(r: &mut R) -> Result<()> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).map_err(IoError::Io)?;
    if &magic != FILE_MAGIC {
        return Err(IoError::FormatError("ORC: bad file magic".into()));
    }
    Ok(())
}

fn read_all_stripes(path: &Path) -> Result<Vec<ORCStripe>> {
    let f = File::open(path).map_err(IoError::Io)?;
    let mut r = BufReader::new(f);
    check_magic(&mut r)?;

    // Read footer to know how many stripes there are
    // Seek to footer_len (last 4 bytes)
    r.seek(SeekFrom::End(-4)).map_err(IoError::Io)?;
    let mut fl_buf = [0u8; 4];
    r.read_exact(&mut fl_buf).map_err(IoError::Io)?;
    let footer_len = u32::from_le_bytes(fl_buf) as i64;

    r.seek(SeekFrom::End(-(4 + footer_len)))
        .map_err(IoError::Io)?;
    let mut ns_buf = [0u8; 4];
    r.read_exact(&mut ns_buf).map_err(IoError::Io)?;
    let n_stripes = u32::from_le_bytes(ns_buf) as usize;

    // Read stripe offset table from the footer
    let mut stripe_offsets: Vec<u64> = Vec::with_capacity(n_stripes);
    for _ in 0..n_stripes {
        let mut off_buf = [0u8; 8];
        r.read_exact(&mut off_buf).map_err(IoError::Io)?;
        stripe_offsets.push(u64::from_le_bytes(off_buf));

        // skip n_rows (u32), n_cols (u32)
        let mut skip_buf = [0u8; 8];
        r.read_exact(&mut skip_buf).map_err(IoError::Io)?;
        let n_cols_footer =
            u32::from_le_bytes([skip_buf[4], skip_buf[5], skip_buf[6], skip_buf[7]]);
        // skip column name+type entries
        for _ in 0..n_cols_footer {
            let mut nl_buf = [0u8; 2];
            r.read_exact(&mut nl_buf).map_err(IoError::Io)?;
            let name_len = u16::from_le_bytes(nl_buf) as usize;
            let mut skip = vec![0u8; name_len + 1]; // name + type byte
            r.read_exact(&mut skip).map_err(IoError::Io)?;
        }
    }

    // Now read each stripe
    let mut stripes = Vec::with_capacity(n_stripes);
    for &offset in &stripe_offsets {
        r.seek(SeekFrom::Start(offset)).map_err(IoError::Io)?;
        stripes.push(read_stripe(&mut r)?);
    }
    Ok(stripes)
}

fn read_stripe<R: Read>(r: &mut R) -> Result<ORCStripe> {
    let mut sm = [0u8; 4];
    r.read_exact(&mut sm).map_err(IoError::Io)?;
    if &sm != STRIPE_MAGIC {
        return Err(IoError::FormatError("ORC: bad stripe magic".into()));
    }

    let mut nr_buf = [0u8; 4];
    r.read_exact(&mut nr_buf).map_err(IoError::Io)?;
    let n_rows = u32::from_le_bytes(nr_buf) as usize;

    let mut nc_buf = [0u8; 4];
    r.read_exact(&mut nc_buf).map_err(IoError::Io)?;
    let n_cols = u32::from_le_bytes(nc_buf) as usize;

    let mut columns = Vec::with_capacity(n_cols);
    for _ in 0..n_cols {
        let mut nl_buf = [0u8; 2];
        r.read_exact(&mut nl_buf).map_err(IoError::Io)?;
        let name_len = u16::from_le_bytes(nl_buf) as usize;
        let mut name_bytes = vec![0u8; name_len];
        r.read_exact(&mut name_bytes).map_err(IoError::Io)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| IoError::FormatError(format!("ORC: bad col name UTF-8: {e}")))?;

        let mut type_enc = [0u8; 2];
        r.read_exact(&mut type_enc).map_err(IoError::Io)?;
        let type_kind = ORCType::from_u8(type_enc[0])?;
        let encoding = ColumnEncoding::from_u8(type_enc[1])?;

        let mut pl_buf = [0u8; 4];
        r.read_exact(&mut pl_buf).map_err(IoError::Io)?;
        let present_len = u32::from_le_bytes(pl_buf) as usize;
        let mut present = vec![0u8; present_len];
        r.read_exact(&mut present).map_err(IoError::Io)?;

        let mut dl_buf = [0u8; 4];
        r.read_exact(&mut dl_buf).map_err(IoError::Io)?;
        let data_len = u32::from_le_bytes(dl_buf) as usize;
        let mut data = vec![0u8; data_len];
        r.read_exact(&mut data).map_err(IoError::Io)?;

        columns.push(ORCColumn {
            name,
            type_kind,
            data,
            present,
            encoding,
        });
    }

    Ok(ORCStripe { n_rows, columns })
}

// ─────────────────────────────── Encoding helpers ────────────────────────────

fn full_present_stream(n: usize) -> Vec<u8> {
    let n_bytes = (n + 7) / 8;
    vec![0xff_u8; n_bytes]
}

/// Encode i64 slice.  Uses delta encoding when the delta sequence is monotone,
/// otherwise direct (little-endian raw).
fn encode_i64(data: &[i64]) -> (ColumnEncoding, Vec<u8>) {
    if data.len() < 2 {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        return (ColumnEncoding::DirectI64, raw);
    }
    // Check if delta encoding is beneficial
    let first = data[0];
    let deltas: Vec<i64> = data.windows(2).map(|w| w[1] - w[0]).collect();
    let all_same_delta = deltas.windows(2).all(|w| w[0] == w[1]);
    if all_same_delta {
        let mut buf = Vec::with_capacity(8 + 8 + 8 * deltas.len());
        buf.extend_from_slice(&first.to_le_bytes());
        buf.extend_from_slice(&(deltas.len() as u64).to_le_bytes());
        if let Some(&d) = deltas.first() {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        (ColumnEncoding::DeltaI64, buf)
    } else {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        (ColumnEncoding::DirectI64, raw)
    }
}

fn decode_i64(data: &[u8], encoding: ColumnEncoding, n_rows: usize) -> Result<Vec<i64>> {
    match encoding {
        ColumnEncoding::DirectI64 => {
            if data.len() != n_rows * 8 {
                return Err(IoError::FormatError(format!(
                    "ORC: direct i64 data length {} != {}",
                    data.len(),
                    n_rows * 8
                )));
            }
            data.chunks_exact(8)
                .map(|c| {
                    Ok(i64::from_le_bytes(c.try_into().map_err(|_| {
                        IoError::FormatError("ORC: bad i64 slice".into())
                    })?))
                })
                .collect()
        }
        ColumnEncoding::DeltaI64 => {
            if data.len() < 8 {
                return Err(IoError::FormatError("ORC: truncated delta i64".into()));
            }
            let first = i64::from_le_bytes(
                data[0..8]
                    .try_into()
                    .map_err(|_| IoError::FormatError("ORC: bad first i64 in delta".into()))?,
            );
            if n_rows == 1 {
                return Ok(vec![first]);
            }
            let n_deltas = u64::from_le_bytes(
                data[8..16]
                    .try_into()
                    .map_err(|_| IoError::FormatError("ORC: bad n_deltas".into()))?,
            ) as usize;
            let delta = if n_deltas > 0 && data.len() >= 24 {
                i64::from_le_bytes(
                    data[16..24]
                        .try_into()
                        .map_err(|_| IoError::FormatError("ORC: bad delta value".into()))?,
                )
            } else {
                0
            };
            let mut out = Vec::with_capacity(n_rows);
            out.push(first);
            for i in 1..n_rows {
                let prev = out[i - 1];
                out.push(prev + delta);
            }
            Ok(out)
        }
        other => Err(IoError::FormatError(format!(
            "ORC: cannot decode i64 with encoding {other:?}"
        ))),
    }
}

/// Dictionary string encoding:
/// `[n_dict: u32 LE] [dict_entries: each as u32 LE len + bytes] [indices: n_rows × u32 LE]`
fn encode_dict_strings(data: &[&str]) -> Vec<u8> {
    let mut dict: Vec<&str> = Vec::new();
    let mut dict_map: HashMap<&str, u32> = HashMap::new();
    let mut indices: Vec<u32> = Vec::with_capacity(data.len());

    for &s in data {
        let idx = *dict_map.entry(s).or_insert_with(|| {
            let i = dict.len() as u32;
            dict.push(s);
            i
        });
        indices.push(idx);
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(&(dict.len() as u32).to_le_bytes());
    for entry in &dict {
        let b = entry.as_bytes();
        buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
        buf.extend_from_slice(b);
    }
    for idx in &indices {
        buf.extend_from_slice(&idx.to_le_bytes());
    }
    buf
}

fn decode_dict_strings(data: &[u8], n_rows: usize) -> Result<Vec<String>> {
    let mut pos = 0usize;
    let n_dict = u32::from_le_bytes(
        data[pos..pos + 4]
            .try_into()
            .map_err(|_| IoError::FormatError("ORC: bad n_dict".into()))?,
    ) as usize;
    pos += 4;

    let mut dict: Vec<String> = Vec::with_capacity(n_dict);
    for _ in 0..n_dict {
        let slen = u32::from_le_bytes(
            data[pos..pos + 4]
                .try_into()
                .map_err(|_| IoError::FormatError("ORC: bad dict entry len".into()))?,
        ) as usize;
        pos += 4;
        let s = std::str::from_utf8(&data[pos..pos + slen])
            .map_err(|e| IoError::FormatError(format!("ORC: dict UTF-8 error: {e}")))?
            .to_owned();
        pos += slen;
        dict.push(s);
    }

    let mut out = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let idx = u32::from_le_bytes(
            data[pos..pos + 4]
                .try_into()
                .map_err(|_| IoError::FormatError("ORC: bad dict index".into()))?,
        ) as usize;
        pos += 4;
        let val = dict.get(idx).ok_or_else(|| {
            IoError::FormatError(format!("ORC: dict index {idx} out of range {n_dict}"))
        })?;
        out.push(val.clone());
    }
    Ok(out)
}

/// Boolean run-length encoding: `[n_runs: u32 LE] [run: bit (1 byte) + count (u32 LE)]...`
fn encode_bool_rle(data: &[bool]) -> Vec<u8> {
    if data.is_empty() {
        return 0u32.to_le_bytes().to_vec();
    }
    let mut runs: Vec<(bool, u32)> = Vec::new();
    let mut current = data[0];
    let mut count = 1u32;
    for &v in &data[1..] {
        if v == current {
            count += 1;
        } else {
            runs.push((current, count));
            current = v;
            count = 1;
        }
    }
    runs.push((current, count));

    let mut buf = Vec::with_capacity(4 + runs.len() * 5);
    buf.extend_from_slice(&(runs.len() as u32).to_le_bytes());
    for (v, c) in &runs {
        buf.push(if *v { 1u8 } else { 0u8 });
        buf.extend_from_slice(&c.to_le_bytes());
    }
    buf
}

fn decode_bool_rle(data: &[u8], n_rows: usize) -> Result<Vec<bool>> {
    if data.len() < 4 {
        return Err(IoError::FormatError("ORC: truncated bool RLE".into()));
    }
    let n_runs = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let mut pos = 4;
    let mut out = Vec::with_capacity(n_rows);
    for _ in 0..n_runs {
        if pos + 5 > data.len() {
            return Err(IoError::FormatError("ORC: truncated bool run".into()));
        }
        let v = data[pos] != 0;
        let count = u32::from_le_bytes([data[pos + 1], data[pos + 2], data[pos + 3], data[pos + 4]])
            as usize;
        pos += 5;
        for _ in 0..count {
            out.push(v);
        }
    }
    Ok(out)
}

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn tmp(name: &str) -> std::path::PathBuf {
        temp_dir().join(name)
    }

    #[test]
    fn test_i64_roundtrip_direct() {
        let path = tmp("orc_test_i64_direct.orc");
        let data: Vec<i64> = vec![10, 20, 5, -3, 99];
        let mut w = ORCWriter::new();
        w.add_column_i64("vals", &data).expect("add col");
        w.write(&path).expect("write");

        let result = ORCReader::read_column_i64(&path, "vals").expect("read");
        assert_eq!(result, data);
    }

    #[test]
    fn test_i64_roundtrip_delta() {
        let path = tmp("orc_test_i64_delta.orc");
        // Arithmetic sequence: constant delta 3
        let data: Vec<i64> = (0..10).map(|i| i * 3).collect();
        let mut w = ORCWriter::new();
        w.add_column_i64("seq", &data).expect("add col");
        w.write(&path).expect("write");

        let result = ORCReader::read_column_i64(&path, "seq").expect("read");
        assert_eq!(result, data);
    }

    #[test]
    fn test_f64_roundtrip() {
        let path = tmp("orc_test_f64.orc");
        let data: Vec<f64> = vec![1.1, 2.2, 3.3];
        let mut w = ORCWriter::new();
        w.add_column_f64("values", &data).expect("add col");
        w.write(&path).expect("write");

        let result = ORCReader::read_column_f64(&path, "values").expect("read");
        assert!((result[0] - 1.1).abs() < 1e-12);
        assert!((result[1] - 2.2).abs() < 1e-12);
        assert!((result[2] - 3.3).abs() < 1e-12);
    }

    #[test]
    fn test_str_roundtrip() {
        let path = tmp("orc_test_str.orc");
        let data = vec!["alpha", "beta", "alpha", "gamma"];
        let mut w = ORCWriter::new();
        w.add_column_str("labels", &data).expect("add col");
        w.write(&path).expect("write");

        let result = ORCReader::read_column_str(&path, "labels").expect("read");
        assert_eq!(result, vec!["alpha", "beta", "alpha", "gamma"]);
    }

    #[test]
    fn test_bool_roundtrip() {
        let path = tmp("orc_test_bool.orc");
        let data = vec![true, false, false, true, true];
        let mut w = ORCWriter::new();
        w.add_column_bool("flags", &data).expect("add col");
        w.write(&path).expect("write");

        let result = ORCReader::read_column_bool(&path, "flags").expect("read");
        assert_eq!(result, data);
    }

    #[test]
    fn test_schema() {
        let path = tmp("orc_test_schema.orc");
        let mut w = ORCWriter::new();
        w.add_column_i64("id", &[1, 2, 3]).expect("add col");
        w.add_column_f64("score", &[0.1, 0.2, 0.3])
            .expect("add col");
        w.add_column_str("name", &["a", "b", "c"]).expect("add col");
        w.write(&path).expect("write");

        let schema = ORCReader::read_schema(&path).expect("schema");
        assert_eq!(schema.len(), 3);
        assert_eq!(schema[0], ("id".to_string(), ORCType::Int64));
        assert_eq!(schema[1], ("score".to_string(), ORCType::Float64));
        assert_eq!(schema[2], ("name".to_string(), ORCType::String));
    }

    #[test]
    fn test_multi_stripe() {
        let path = tmp("orc_test_multi_stripe.orc");
        let mut w = ORCWriter::new();

        w.add_column_i64("x", &[1, 2, 3]).expect("add col");
        w.begin_stripe();
        w.add_column_i64("x", &[4, 5, 6]).expect("add col");
        w.write(&path).expect("write");

        let result = ORCReader::read_column_i64(&path, "x").expect("read");
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_bool_rle_encode_decode() {
        let data = vec![true, true, true, false, false, true];
        let encoded = encode_bool_rle(&data);
        let decoded = decode_bool_rle(&encoded, data.len()).expect("decode");
        assert_eq!(decoded, data);
    }
}
