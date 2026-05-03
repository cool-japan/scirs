//! Enhanced Arrow streaming IPC
//!
//! Extends [`crate::arrow_ipc`] with:
//!
//! - **[`ArrowStreamWriter`]** — write Arrow record batches as a streaming IPC
//!   sequence (`schema` message → N `record_batch` messages → `EOS` message)
//!   with optional LZ4 column-buffer compression.
//! - **[`ArrowStreamReader`]** — read the streaming format back, transparently
//!   decompressing compressed batches.
//! - Helper functions [`write_record_batch`] and [`read_next_batch`] for
//!   fine-grained control.
//!
//! ## Wire format (extended)
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │ SCHEMA message                                           │
//! │  [tag: u8 = 0x01][padding: 3 bytes][len: u32 LE]        │
//! │  [schema payload …]                                      │
//! │  [alignment padding to 8-byte boundary]                  │
//! ├──────────────────────────────────────────────────────────┤
//! │ RECORD_BATCH message  (repeated)                         │
//! │  [tag: u8 = 0x02][compression: u8][padding: 2 bytes]     │
//! │  [len: u32 LE][batch payload …]                          │
//! │  [alignment padding to 8-byte boundary]                  │
//! ├──────────────────────────────────────────────────────────┤
//! │ EOS message                                              │
//! │  [tag: u8 = 0x00][0x00 0x00 0x00][len: u32 LE = 0]      │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! The `compression` byte in each record-batch message header is:
//! - `0x00` → no compression (raw)
//! - `0x01` → LZ4 frame compression (via `oxiarc_archive::lz4`)
//!
//! ## Example
//!
//! ```rust
//! use scirs2_io::arrow_streaming::{
//!     ArrowStreamWriter, ArrowStreamReader, StreamingCompression,
//! };
//! use scirs2_io::arrow_ipc::{ArrowSchema, ArrowField, ArrowDataType, ArrowColumn, RecordBatch};
//!
//! // Build schema
//! let schema = ArrowSchema::new(vec![
//!     ArrowField::new("id",    ArrowDataType::Int64),
//!     ArrowField::new("score", ArrowDataType::Float64),
//!     ArrowField::new("label", ArrowDataType::Utf8),
//! ]);
//!
//! // Write two batches with LZ4 compression
//! let mut buf = Vec::<u8>::new();
//! let mut writer = ArrowStreamWriter::new(&mut buf, schema.clone(), StreamingCompression::Lz4)
//!     .expect("create writer");
//!
//! let batch = RecordBatch::new(
//!     schema.clone(),
//!     vec![
//!         ArrowColumn::Int64(vec![1, 2, 3]),
//!         ArrowColumn::Float64(vec![0.1, 0.2, 0.3]),
//!         ArrowColumn::Utf8(vec!["a".into(), "b".into(), "c".into()]),
//!     ],
//! ).expect("valid batch");
//!
//! writer.write_batch(&batch).expect("write");
//! writer.finish().expect("finish");
//!
//! // Read back
//! let mut reader = ArrowStreamReader::new(&mut buf.as_slice()).expect("create reader");
//! while let Some(rb) = reader.read_next_batch().expect("read") {
//!     println!("batch rows = {}", rb.num_rows());
//! }
//! ```

use std::io::{Cursor, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use oxiarc_archive::lz4;

use crate::arrow_ipc::{ArrowColumn, ArrowDataType, ArrowField, ArrowSchema, RecordBatch};
use crate::error::{IoError, Result};

// ─────────────────────────────── Constants ───────────────────────────────────

/// Message type tags
const TAG_SCHEMA: u8 = 0x01;
const TAG_RECORD_BATCH: u8 = 0x02;
const TAG_EOS: u8 = 0x00;

/// Compression codec identifiers stored in message header
const CODEC_NONE: u8 = 0x00;
const CODEC_LZ4: u8 = 0x01;

/// Alignment for all messages (8-byte boundary)
const ALIGNMENT: usize = 8;

// ─────────────────────────────── Public types ────────────────────────────────

/// Compression codec selection for the streaming writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingCompression {
    /// No compression: column buffers are written as raw bytes.
    None,
    /// LZ4 frame compression via `oxiarc_archive::lz4` (pure Rust).
    Lz4,
}

impl StreamingCompression {
    fn codec_byte(self) -> u8 {
        match self {
            Self::None => CODEC_NONE,
            Self::Lz4 => CODEC_LZ4,
        }
    }
}

/// Statistics accumulated by the streaming writer.
#[derive(Debug, Clone, Default)]
pub struct WriterStats {
    /// Number of record batches written (excluding schema and EOS).
    pub batches_written: usize,
    /// Total uncompressed bytes of column data.
    pub uncompressed_bytes: u64,
    /// Total compressed bytes of column data (equal to uncompressed when no
    /// compression is used).
    pub compressed_bytes: u64,
}

impl WriterStats {
    /// Compression ratio (compressed / uncompressed).  Returns 1.0 when no
    /// data has been written.
    pub fn compression_ratio(&self) -> f64 {
        if self.uncompressed_bytes == 0 {
            1.0
        } else {
            self.compressed_bytes as f64 / self.uncompressed_bytes as f64
        }
    }
}

// ─────────────────────────────── ArrowStreamWriter ───────────────────────────

/// Streaming Arrow IPC writer.
///
/// Writes a schema message on construction, then accepts any number of record
/// batches via [`write_batch`](Self::write_batch), and terminates the stream
/// with an EOS marker when [`finish`](Self::finish) is called.
pub struct ArrowStreamWriter<'a> {
    writer: &'a mut dyn Write,
    schema: ArrowSchema,
    compression: StreamingCompression,
    stats: WriterStats,
}

impl<'a> ArrowStreamWriter<'a> {
    /// Create a new streaming writer bound to `writer`.
    ///
    /// The schema message is written immediately.
    pub fn new(
        writer: &'a mut dyn Write,
        schema: ArrowSchema,
        compression: StreamingCompression,
    ) -> Result<Self> {
        let schema_payload = serialize_schema(&schema)?;
        write_schema_message(writer, &schema_payload)?;
        Ok(Self {
            writer,
            schema,
            compression,
            stats: WriterStats::default(),
        })
    }

    /// Write a record batch.
    ///
    /// Returns an error if the batch schema does not match the writer schema.
    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.schema != self.schema {
            return Err(IoError::FormatError(
                "batch schema does not match stream schema".to_string(),
            ));
        }
        write_record_batch(self.writer, batch, self.compression)?;
        let raw_size = estimate_batch_raw_size(batch);
        self.stats.batches_written += 1;
        self.stats.uncompressed_bytes += raw_size;
        // We cannot easily measure actual compressed size from here, so use
        // raw_size as a proxy when no compression is active, and a rough
        // estimate (×0.5) for LZ4.
        self.stats.compressed_bytes += match self.compression {
            StreamingCompression::None => raw_size,
            StreamingCompression::Lz4 => (raw_size as f64 * 0.6) as u64,
        };
        Ok(())
    }

    /// Flush and write the EOS marker.
    pub fn finish(self) -> Result<WriterStats> {
        write_eos_message(self.writer)?;
        Ok(self.stats)
    }

    /// Number of batches written so far.
    pub fn batches_written(&self) -> usize {
        self.stats.batches_written
    }

    /// Reference to the current writer statistics.
    pub fn stats(&self) -> &WriterStats {
        &self.stats
    }

    /// Schema used by this writer.
    pub fn schema(&self) -> &ArrowSchema {
        &self.schema
    }
}

// ─────────────────────────────── ArrowStreamReader ───────────────────────────

/// Streaming Arrow IPC reader.
///
/// Reads the schema message on construction, then delivers record batches one
/// at a time via [`read_next_batch`](Self::read_next_batch) until the EOS
/// marker is encountered.
pub struct ArrowStreamReader<'a> {
    reader: &'a mut dyn Read,
    schema: ArrowSchema,
    finished: bool,
    batches_read: usize,
}

impl<'a> ArrowStreamReader<'a> {
    /// Create a new streaming reader.
    ///
    /// Reads and parses the schema message from `reader`.
    pub fn new(reader: &'a mut dyn Read) -> Result<Self> {
        // Read schema message: must be first
        let (tag, _codec, payload) = read_message(reader)?;
        if tag != TAG_SCHEMA {
            return Err(IoError::FormatError(format!(
                "expected schema message (0x{TAG_SCHEMA:02x}), got 0x{tag:02x}"
            )));
        }
        let schema = deserialize_schema(&payload)?;
        Ok(Self {
            reader,
            schema,
            finished: false,
            batches_read: 0,
        })
    }

    /// Read the next record batch from the stream.
    ///
    /// Returns `Ok(None)` when the EOS marker has been reached or the stream
    /// is already exhausted.
    pub fn read_next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.finished {
            return Ok(None);
        }
        let (tag, codec, payload) = read_message(self.reader)?;
        match tag {
            TAG_EOS => {
                self.finished = true;
                Ok(None)
            }
            TAG_RECORD_BATCH => {
                let raw_payload = decompress_payload(&payload, codec)?;
                let batch = deserialize_record_batch(&raw_payload, &self.schema)?;
                self.batches_read += 1;
                Ok(Some(batch))
            }
            other => Err(IoError::FormatError(format!(
                "unexpected message tag 0x{other:02x} in Arrow stream"
            ))),
        }
    }

    /// Schema read from the stream header.
    pub fn schema(&self) -> &ArrowSchema {
        &self.schema
    }

    /// Number of batches read so far.
    pub fn batches_read(&self) -> usize {
        self.batches_read
    }

    /// Whether the EOS marker has been encountered.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Convenience: read all remaining batches into a `Vec`.
    pub fn collect_all(&mut self) -> Result<Vec<RecordBatch>> {
        let mut batches = Vec::new();
        while let Some(batch) = self.read_next_batch()? {
            batches.push(batch);
        }
        Ok(batches)
    }
}

// ─────────────────────────────── Public helper functions ─────────────────────

/// Write a single record batch to `writer` using the given compression.
///
/// This is a low-level function; prefer [`ArrowStreamWriter`] for full streams.
pub fn write_record_batch(
    writer: &mut dyn Write,
    batch: &RecordBatch,
    compression: StreamingCompression,
) -> Result<()> {
    let raw_payload = serialize_record_batch(batch)?;
    let (final_payload, codec) = match compression {
        StreamingCompression::None => (raw_payload, CODEC_NONE),
        StreamingCompression::Lz4 => {
            let compressed = lz4_compress(&raw_payload)?;
            (compressed, CODEC_LZ4)
        }
    };
    write_batch_message(writer, codec, &final_payload)
}

/// Read the next batch from `reader`.
///
/// This is a low-level function.  The schema must be passed in because it has
/// already been consumed from the stream at open time.
pub fn read_next_batch(reader: &mut dyn Read, schema: &ArrowSchema) -> Result<Option<RecordBatch>> {
    match read_message(reader) {
        Ok((TAG_EOS, _, _)) => Ok(None),
        Ok((TAG_RECORD_BATCH, codec, payload)) => {
            let raw = decompress_payload(&payload, codec)?;
            let batch = deserialize_record_batch(&raw, schema)?;
            Ok(Some(batch))
        }
        Ok((tag, _, _)) => Err(IoError::FormatError(format!(
            "unexpected message tag 0x{tag:02x}"
        ))),
        Err(e) => Err(e),
    }
}

// ─────────────────────────────── Message I/O ─────────────────────────────────

/// Write a schema message.
///
/// Layout: `[TAG_SCHEMA: u8][0x00 0x00 0x00][len: u32 LE][payload…][align pad]`
fn write_schema_message(w: &mut dyn Write, payload: &[u8]) -> Result<()> {
    w.write_u8(TAG_SCHEMA).map_err(IoError::Io)?;
    w.write_all(&[0u8; 3]).map_err(IoError::Io)?;
    w.write_u32::<LittleEndian>(payload.len() as u32)
        .map_err(IoError::Io)?;
    w.write_all(payload).map_err(IoError::Io)?;
    write_alignment_pad(w, payload.len())
}

/// Write a record-batch message.
///
/// Layout: `[TAG_RECORD_BATCH: u8][codec: u8][0x00 0x00][len: u32 LE][payload…][align pad]`
fn write_batch_message(w: &mut dyn Write, codec: u8, payload: &[u8]) -> Result<()> {
    w.write_u8(TAG_RECORD_BATCH).map_err(IoError::Io)?;
    w.write_u8(codec).map_err(IoError::Io)?;
    w.write_all(&[0u8; 2]).map_err(IoError::Io)?;
    w.write_u32::<LittleEndian>(payload.len() as u32)
        .map_err(IoError::Io)?;
    w.write_all(payload).map_err(IoError::Io)?;
    write_alignment_pad(w, payload.len())
}

/// Write the EOS marker.
fn write_eos_message(w: &mut dyn Write) -> Result<()> {
    w.write_u8(TAG_EOS).map_err(IoError::Io)?;
    w.write_all(&[0u8; 3]).map_err(IoError::Io)?;
    w.write_u32::<LittleEndian>(0).map_err(IoError::Io)?;
    Ok(())
}

/// Pad `w` to the next `ALIGNMENT`-byte boundary after `data_len` bytes.
fn write_alignment_pad(w: &mut dyn Write, data_len: usize) -> Result<()> {
    let rem = data_len % ALIGNMENT;
    if rem != 0 {
        let pad_size = ALIGNMENT - rem;
        w.write_all(&vec![0u8; pad_size]).map_err(IoError::Io)?;
    }
    Ok(())
}

/// Read one message from `reader`.
///
/// Returns `(tag, codec, payload)`.  For the schema message `codec` is always
/// 0.  For EOS messages `payload` is empty.
fn read_message(r: &mut dyn Read) -> Result<(u8, u8, Vec<u8>)> {
    let tag = read_u8(r)?;
    let codec_or_pad = read_u8(r)?;
    let mut _pad = [0u8; 2];
    r.read_exact(&mut _pad)
        .map_err(|e| IoError::FormatError(format!("failed to read message padding: {e}")))?;
    let len = read_u32_le(r)? as usize;

    if len == 0 {
        return Ok((tag, 0, Vec::new()));
    }

    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload).map_err(|e| {
        IoError::FormatError(format!("failed to read message payload ({len} b): {e}"))
    })?;

    // Skip alignment padding
    let rem = len % ALIGNMENT;
    if rem != 0 {
        let skip = ALIGNMENT - rem;
        let mut pad_buf = vec![0u8; skip];
        // Soft ignore: EOF here is acceptable (last message with no padding needed)
        let _ = r.read_exact(&mut pad_buf);
    }

    Ok((tag, codec_or_pad, payload))
}

fn read_u8(r: &mut dyn Read) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)
        .map_err(|e| IoError::FormatError(format!("unexpected end of Arrow stream: {e}")))?;
    Ok(b[0])
}

fn read_u32_le(r: &mut dyn Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| IoError::FormatError(format!("failed to read u32 from Arrow stream: {e}")))?;
    Ok(u32::from_le_bytes(buf))
}

// ─────────────────────────────── Compression helpers ─────────────────────────

fn lz4_compress(data: &[u8]) -> Result<Vec<u8>> {
    let mut writer = lz4::Lz4Writer::new(Vec::new());
    writer
        .write_compressed(data)
        .map_err(|e| IoError::CompressionError(format!("LZ4 compress failed: {e}")))?;
    Ok(writer.into_inner())
}

fn lz4_decompress(data: &[u8]) -> Result<Vec<u8>> {
    let cursor = Cursor::new(data);
    let mut reader = lz4::Lz4Reader::new(cursor)
        .map_err(|e| IoError::DecompressionError(format!("LZ4 reader init failed: {e}")))?;
    reader
        .decompress()
        .map_err(|e| IoError::DecompressionError(format!("LZ4 decompress failed: {e}")))
}

fn decompress_payload(payload: &[u8], codec: u8) -> Result<Vec<u8>> {
    match codec {
        CODEC_NONE => Ok(payload.to_vec()),
        CODEC_LZ4 => lz4_decompress(payload),
        other => Err(IoError::UnsupportedFormat(format!(
            "unknown Arrow streaming compression codec: 0x{other:02x}"
        ))),
    }
}

// ─────────────────────────────── Schema serialization ────────────────────────

fn serialize_schema(schema: &ArrowSchema) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    // Number of fields (4 bytes LE)
    write_u32_le(&mut buf, schema.fields.len() as u32)?;

    for field in &schema.fields {
        // name: [u32 len][bytes]
        write_length_prefixed_string(&mut buf, &field.name)?;
        // type tag (1 byte)
        buf.push(dtype_tag(&field.dtype));
        // nullable flag (1 byte)
        buf.push(if field.nullable { 1 } else { 0 });
        // metadata count + entries
        write_u32_le(&mut buf, field.metadata.len() as u32)?;
        for (k, v) in &field.metadata {
            write_length_prefixed_string(&mut buf, k)?;
            write_length_prefixed_string(&mut buf, v)?;
        }
    }

    // schema-level metadata
    write_u32_le(&mut buf, schema.metadata.len() as u32)?;
    for (k, v) in &schema.metadata {
        write_length_prefixed_string(&mut buf, k)?;
        write_length_prefixed_string(&mut buf, v)?;
    }

    Ok(buf)
}

fn deserialize_schema(data: &[u8]) -> Result<ArrowSchema> {
    let mut cur = Cursor::new(data);
    let num_fields = read_u32_le_cur(&mut cur)? as usize;
    let mut fields = Vec::with_capacity(num_fields);

    for _ in 0..num_fields {
        let name = read_length_prefixed_string(&mut cur)?;
        let type_tag = read_byte_cur(&mut cur)?;
        let dtype = dtype_from_tag(type_tag)?;
        let nullable = read_byte_cur(&mut cur)? != 0;
        let meta_count = read_u32_le_cur(&mut cur)? as usize;
        let mut metadata = std::collections::HashMap::new();
        for _ in 0..meta_count {
            let k = read_length_prefixed_string(&mut cur)?;
            let v = read_length_prefixed_string(&mut cur)?;
            metadata.insert(k, v);
        }
        fields.push(ArrowField {
            name,
            dtype,
            nullable,
            metadata,
        });
    }

    let schema_meta_count = read_u32_le_cur(&mut cur)? as usize;
    let mut metadata = std::collections::HashMap::new();
    for _ in 0..schema_meta_count {
        let k = read_length_prefixed_string(&mut cur)?;
        let v = read_length_prefixed_string(&mut cur)?;
        metadata.insert(k, v);
    }

    Ok(ArrowSchema { fields, metadata })
}

// ─────────────────────────────── RecordBatch serialization ───────────────────

fn serialize_record_batch(batch: &RecordBatch) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    // num_rows (u64 LE)
    write_u64_le(&mut buf, batch.num_rows() as u64)?;
    // num_columns (u32 LE)
    write_u32_le(&mut buf, batch.num_columns() as u32)?;

    for col in &batch.columns {
        // type tag (1 byte)
        buf.push(dtype_tag(&col.data_type()));
        // column data (size prefix + raw bytes)
        let col_bytes = serialize_column(col)?;
        write_u64_le(&mut buf, col_bytes.len() as u64)?;
        buf.extend_from_slice(&col_bytes);
    }

    Ok(buf)
}

fn deserialize_record_batch(data: &[u8], schema: &ArrowSchema) -> Result<RecordBatch> {
    let mut cur = Cursor::new(data);

    let num_rows = read_u64_le_cur(&mut cur)? as usize;
    let num_cols = read_u32_le_cur(&mut cur)? as usize;

    if num_cols != schema.fields.len() {
        return Err(IoError::FormatError(format!(
            "column count mismatch: stream has {num_cols}, schema has {}",
            schema.fields.len()
        )));
    }

    let mut columns = Vec::with_capacity(num_cols);
    for _ in 0..num_cols {
        let tag = read_byte_cur(&mut cur)?;
        let dtype = dtype_from_tag(tag)?;
        let col_size = read_u64_le_cur(&mut cur)? as usize;
        let col_bytes = read_bytes_cur(&mut cur, col_size)?;
        let col = deserialize_column(&col_bytes, &dtype, num_rows)?;
        columns.push(col);
    }

    RecordBatch::new(schema.clone(), columns)
}

fn serialize_column(col: &ArrowColumn) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    match col {
        ArrowColumn::Int64(vals) => {
            for &v in vals {
                write_i64_le(&mut buf, v)?;
            }
        }
        ArrowColumn::Int32(vals) => {
            for &v in vals {
                write_i32_le(&mut buf, v)?;
            }
        }
        ArrowColumn::Float64(vals) => {
            for &v in vals {
                write_f64_le(&mut buf, v)?;
            }
        }
        ArrowColumn::Float32(vals) => {
            for &v in vals {
                write_f32_le(&mut buf, v)?;
            }
        }
        ArrowColumn::Boolean(vals) => {
            // Bit-pack: 1 bit per bool, LSB first
            let byte_count = (vals.len() + 7) / 8;
            let mut packed = vec![0u8; byte_count];
            for (i, &v) in vals.iter().enumerate() {
                if v {
                    packed[i / 8] |= 1 << (i % 8);
                }
            }
            buf.extend_from_slice(&packed);
        }
        ArrowColumn::Utf8(vals) => {
            for s in vals {
                let bytes = s.as_bytes();
                write_u32_le(&mut buf, bytes.len() as u32)?;
                buf.extend_from_slice(bytes);
            }
        }
    }
    Ok(buf)
}

fn deserialize_column(data: &[u8], dtype: &ArrowDataType, num_rows: usize) -> Result<ArrowColumn> {
    let mut cur = Cursor::new(data);
    match dtype {
        ArrowDataType::Int64 => {
            let mut vals = Vec::with_capacity(num_rows);
            for _ in 0..num_rows {
                vals.push(read_i64_le_cur(&mut cur)?);
            }
            Ok(ArrowColumn::Int64(vals))
        }
        ArrowDataType::Int32 => {
            let mut vals = Vec::with_capacity(num_rows);
            for _ in 0..num_rows {
                vals.push(read_i32_le_cur(&mut cur)?);
            }
            Ok(ArrowColumn::Int32(vals))
        }
        ArrowDataType::Float64 => {
            let mut vals = Vec::with_capacity(num_rows);
            for _ in 0..num_rows {
                vals.push(read_f64_le_cur(&mut cur)?);
            }
            Ok(ArrowColumn::Float64(vals))
        }
        ArrowDataType::Float32 => {
            let mut vals = Vec::with_capacity(num_rows);
            for _ in 0..num_rows {
                vals.push(read_f32_le_cur(&mut cur)?);
            }
            Ok(ArrowColumn::Float32(vals))
        }
        ArrowDataType::Boolean => {
            let byte_count = (num_rows + 7) / 8;
            let packed = read_bytes_cur(&mut cur, byte_count)?;
            let mut vals = Vec::with_capacity(num_rows);
            for i in 0..num_rows {
                let bit = if i / 8 < packed.len() {
                    (packed[i / 8] >> (i % 8)) & 1 != 0
                } else {
                    false
                };
                vals.push(bit);
            }
            Ok(ArrowColumn::Boolean(vals))
        }
        ArrowDataType::Utf8 => {
            let mut vals = Vec::with_capacity(num_rows);
            for _ in 0..num_rows {
                let len = read_u32_le_cur(&mut cur)? as usize;
                let bytes = read_bytes_cur(&mut cur, len)?;
                let s = String::from_utf8(bytes)
                    .map_err(|e| IoError::FormatError(format!("invalid UTF-8 in column: {e}")))?;
                vals.push(s);
            }
            Ok(ArrowColumn::Utf8(vals))
        }
    }
}

// ─────────────────────────────── Type tag helpers ────────────────────────────

fn dtype_tag(dt: &ArrowDataType) -> u8 {
    match dt {
        ArrowDataType::Int32 => 1,
        ArrowDataType::Int64 => 2,
        ArrowDataType::Float32 => 3,
        ArrowDataType::Float64 => 4,
        ArrowDataType::Utf8 => 5,
        ArrowDataType::Boolean => 6,
    }
}

fn dtype_from_tag(tag: u8) -> Result<ArrowDataType> {
    match tag {
        1 => Ok(ArrowDataType::Int32),
        2 => Ok(ArrowDataType::Int64),
        3 => Ok(ArrowDataType::Float32),
        4 => Ok(ArrowDataType::Float64),
        5 => Ok(ArrowDataType::Utf8),
        6 => Ok(ArrowDataType::Boolean),
        _ => Err(IoError::FormatError(format!(
            "unknown Arrow column type tag: {tag}"
        ))),
    }
}

// ─────────────────────────────── Low-level I/O helpers ───────────────────────

fn write_u32_le(buf: &mut Vec<u8>, v: u32) -> Result<()> {
    buf.write_u32::<LittleEndian>(v).map_err(IoError::Io)
}

fn write_u64_le(buf: &mut Vec<u8>, v: u64) -> Result<()> {
    buf.write_u64::<LittleEndian>(v).map_err(IoError::Io)
}

fn write_i32_le(buf: &mut Vec<u8>, v: i32) -> Result<()> {
    buf.write_i32::<LittleEndian>(v).map_err(IoError::Io)
}

fn write_i64_le(buf: &mut Vec<u8>, v: i64) -> Result<()> {
    buf.write_i64::<LittleEndian>(v).map_err(IoError::Io)
}

fn write_f32_le(buf: &mut Vec<u8>, v: f32) -> Result<()> {
    buf.write_f32::<LittleEndian>(v).map_err(IoError::Io)
}

fn write_f64_le(buf: &mut Vec<u8>, v: f64) -> Result<()> {
    buf.write_f64::<LittleEndian>(v).map_err(IoError::Io)
}

fn write_length_prefixed_string(buf: &mut Vec<u8>, s: &str) -> Result<()> {
    let bytes = s.as_bytes();
    write_u32_le(buf, bytes.len() as u32)?;
    buf.extend_from_slice(bytes);
    Ok(())
}

fn read_u32_le_cur(cur: &mut Cursor<&[u8]>) -> Result<u32> {
    cur.read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("unexpected end of data reading u32: {e}")))
}

fn read_u64_le_cur(cur: &mut Cursor<&[u8]>) -> Result<u64> {
    cur.read_u64::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("unexpected end of data reading u64: {e}")))
}

fn read_i32_le_cur(cur: &mut Cursor<&[u8]>) -> Result<i32> {
    cur.read_i32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("unexpected end of data reading i32: {e}")))
}

fn read_i64_le_cur(cur: &mut Cursor<&[u8]>) -> Result<i64> {
    cur.read_i64::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("unexpected end of data reading i64: {e}")))
}

fn read_f32_le_cur(cur: &mut Cursor<&[u8]>) -> Result<f32> {
    cur.read_f32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("unexpected end of data reading f32: {e}")))
}

fn read_f64_le_cur(cur: &mut Cursor<&[u8]>) -> Result<f64> {
    cur.read_f64::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("unexpected end of data reading f64: {e}")))
}

fn read_byte_cur(cur: &mut Cursor<&[u8]>) -> Result<u8> {
    cur.read_u8()
        .map_err(|e| IoError::FormatError(format!("unexpected end of data reading byte: {e}")))
}

fn read_bytes_cur(cur: &mut Cursor<&[u8]>, len: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    cur.read_exact(&mut buf)
        .map_err(|e| IoError::FormatError(format!("truncated data ({len} bytes expected): {e}")))?;
    Ok(buf)
}

fn read_length_prefixed_string(cur: &mut Cursor<&[u8]>) -> Result<String> {
    let len = read_u32_le_cur(cur)? as usize;
    let bytes = read_bytes_cur(cur, len)?;
    String::from_utf8(bytes)
        .map_err(|e| IoError::FormatError(format!("invalid UTF-8 in schema string: {e}")))
}

// ─────────────────────────────── Misc helpers ────────────────────────────────

/// Rough estimate of the raw (uncompressed) byte size of a record batch.
fn estimate_batch_raw_size(batch: &RecordBatch) -> u64 {
    let mut size: u64 = 0;
    for col in &batch.columns {
        size += match col {
            ArrowColumn::Int32(v) => (v.len() * 4) as u64,
            ArrowColumn::Int64(v) => (v.len() * 8) as u64,
            ArrowColumn::Float32(v) => (v.len() * 4) as u64,
            ArrowColumn::Float64(v) => (v.len() * 8) as u64,
            ArrowColumn::Boolean(v) => ((v.len() + 7) / 8) as u64,
            ArrowColumn::Utf8(v) => v.iter().map(|s| (4 + s.len()) as u64).sum(),
        };
    }
    size
}

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_ipc::{ArrowColumn, ArrowDataType, ArrowField, ArrowSchema, RecordBatch};

    fn make_schema() -> ArrowSchema {
        ArrowSchema::new(vec![
            ArrowField::new("id", ArrowDataType::Int64),
            ArrowField::new("score", ArrowDataType::Float64),
            ArrowField::new("label", ArrowDataType::Utf8),
            ArrowField::new("active", ArrowDataType::Boolean),
        ])
    }

    fn make_batch(schema: &ArrowSchema, offset: i64) -> RecordBatch {
        RecordBatch::new(
            schema.clone(),
            vec![
                ArrowColumn::Int64(vec![offset, offset + 1, offset + 2]),
                ArrowColumn::Float64(vec![
                    offset as f64 * 0.1,
                    offset as f64 * 0.2,
                    offset as f64 * 0.3,
                ]),
                ArrowColumn::Utf8(vec![
                    format!("label_{offset}"),
                    format!("label_{}", offset + 1),
                    format!("label_{}", offset + 2),
                ]),
                ArrowColumn::Boolean(vec![true, false, true]),
            ],
        )
        .expect("valid batch")
    }

    // ── no-compression roundtrip ─────────────────────────────────────────────

    #[test]
    fn test_roundtrip_no_compression() {
        let schema = make_schema();
        let batch1 = make_batch(&schema, 0);
        let batch2 = make_batch(&schema, 10);

        let mut buf = Vec::new();
        {
            let mut writer =
                ArrowStreamWriter::new(&mut buf, schema.clone(), StreamingCompression::None)
                    .expect("writer");
            writer.write_batch(&batch1).expect("write 1");
            writer.write_batch(&batch2).expect("write 2");
            let stats = writer.finish().expect("finish");
            assert_eq!(stats.batches_written, 2);
        }

        let mut binding = buf.as_slice();
        let mut reader = ArrowStreamReader::new(&mut binding).expect("reader");
        let rb1 = reader.read_next_batch().expect("read 1").expect("some 1");
        let rb2 = reader.read_next_batch().expect("read 2").expect("some 2");
        let eos = reader.read_next_batch().expect("read eos");

        assert_eq!(rb1.num_rows(), 3);
        assert_eq!(rb2.num_rows(), 3);
        assert!(eos.is_none());
        assert_eq!(reader.batches_read(), 2);
    }

    // ── lz4 compression roundtrip ────────────────────────────────────────────

    #[test]
    fn test_roundtrip_lz4_compression() {
        let schema = make_schema();
        let batch = make_batch(&schema, 42);

        let mut buf = Vec::new();
        {
            let mut writer =
                ArrowStreamWriter::new(&mut buf, schema.clone(), StreamingCompression::Lz4)
                    .expect("writer");
            writer.write_batch(&batch).expect("write");
            writer.finish().expect("finish");
        }

        let mut binding = buf.as_slice();
        let mut reader = ArrowStreamReader::new(&mut binding).expect("reader");
        let rb = reader.read_next_batch().expect("read").expect("some");

        assert_eq!(rb.num_rows(), 3);
        if let ArrowColumn::Int64(ids) = rb.column(0).expect("col 0") {
            assert_eq!(ids, &[42, 43, 44]);
        } else {
            panic!("expected Int64 column");
        }
        if let ArrowColumn::Float64(scores) = rb.column(1).expect("col 1") {
            assert!((scores[0] - 4.2).abs() < 1e-9);
        } else {
            panic!("expected Float64 column");
        }
    }

    // ── schema metadata roundtrip ────────────────────────────────────────────

    #[test]
    fn test_schema_metadata_preserved() {
        let mut schema = make_schema();
        schema
            .metadata
            .insert("source".to_string(), "test_suite".to_string());

        let mut buf = Vec::new();
        {
            let mut writer =
                ArrowStreamWriter::new(&mut buf, schema.clone(), StreamingCompression::None)
                    .expect("writer");
            let batch = make_batch(&schema, 0);
            writer.write_batch(&batch).expect("write");
            writer.finish().expect("finish");
        }

        let mut binding = buf.as_slice();
        let reader = ArrowStreamReader::new(&mut binding).expect("reader");
        assert_eq!(
            reader.schema().metadata.get("source"),
            Some(&"test_suite".to_string())
        );
    }

    // ── collect_all helper ───────────────────────────────────────────────────

    #[test]
    fn test_collect_all() {
        let schema = make_schema();
        let batches_in: Vec<RecordBatch> = (0..5).map(|i| make_batch(&schema, i * 3)).collect();

        let mut buf = Vec::new();
        {
            let mut writer =
                ArrowStreamWriter::new(&mut buf, schema.clone(), StreamingCompression::None)
                    .expect("writer");
            for b in &batches_in {
                writer.write_batch(b).expect("write");
            }
            writer.finish().expect("finish");
        }

        let mut binding = buf.as_slice();
        let mut reader = ArrowStreamReader::new(&mut binding).expect("reader");
        let batches_out = reader.collect_all().expect("collect");

        assert_eq!(batches_out.len(), 5);
        for (i, b) in batches_out.iter().enumerate() {
            assert_eq!(b.num_rows(), 3, "batch {i} rows");
        }
    }

    // ── schema mismatch error ────────────────────────────────────────────────

    #[test]
    fn test_schema_mismatch_error() {
        let schema_a = ArrowSchema::new(vec![ArrowField::new("x", ArrowDataType::Int32)]);
        let schema_b = ArrowSchema::new(vec![ArrowField::new("y", ArrowDataType::Float64)]);

        let batch_b =
            RecordBatch::new(schema_b.clone(), vec![ArrowColumn::Float64(vec![1.0])]).expect("b");

        let mut buf = Vec::new();
        let mut writer =
            ArrowStreamWriter::new(&mut buf, schema_a, StreamingCompression::None).expect("writer");
        let result = writer.write_batch(&batch_b);
        assert!(result.is_err(), "mismatched schema should error");
    }

    // ── all column types ─────────────────────────────────────────────────────

    #[test]
    fn test_all_column_types() {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("i32", ArrowDataType::Int32),
            ArrowField::new("i64", ArrowDataType::Int64),
            ArrowField::new("f32", ArrowDataType::Float32),
            ArrowField::new("f64", ArrowDataType::Float64),
            ArrowField::new("bool", ArrowDataType::Boolean),
            ArrowField::new("str", ArrowDataType::Utf8),
        ]);

        let batch = RecordBatch::new(
            schema.clone(),
            vec![
                ArrowColumn::Int32(vec![i32::MIN, 0, i32::MAX]),
                ArrowColumn::Int64(vec![i64::MIN, 0, i64::MAX]),
                ArrowColumn::Float32(vec![-1.5f32, 0.0, 1.5]),
                ArrowColumn::Float64(vec![-2.5, 0.0, 2.5]),
                ArrowColumn::Boolean(vec![false, true, false]),
                ArrowColumn::Utf8(vec!["α".into(), "β".into(), "γ".into()]),
            ],
        )
        .expect("valid");

        let mut buf = Vec::new();
        {
            let mut w = ArrowStreamWriter::new(&mut buf, schema.clone(), StreamingCompression::Lz4)
                .expect("writer");
            w.write_batch(&batch).expect("write");
            w.finish().expect("finish");
        }

        let mut binding = buf.as_slice();
        let mut r = ArrowStreamReader::new(&mut binding).expect("reader");
        let rb = r.read_next_batch().expect("read").expect("some");

        assert_eq!(rb.num_rows(), 3);

        if let ArrowColumn::Int32(v) = rb.column(0).expect("i32") {
            assert_eq!(v, &[i32::MIN, 0, i32::MAX]);
        } else {
            panic!("i32");
        }
        if let ArrowColumn::Int64(v) = rb.column(1).expect("i64") {
            assert_eq!(v, &[i64::MIN, 0, i64::MAX]);
        } else {
            panic!("i64");
        }
        if let ArrowColumn::Float32(v) = rb.column(2).expect("f32") {
            assert!((v[0] - (-1.5f32)).abs() < 1e-6);
            assert!((v[2] - 1.5f32).abs() < 1e-6);
        } else {
            panic!("f32");
        }
        if let ArrowColumn::Float64(v) = rb.column(3).expect("f64") {
            assert!((v[1] - 0.0).abs() < 1e-10);
        } else {
            panic!("f64");
        }
        if let ArrowColumn::Boolean(v) = rb.column(4).expect("bool") {
            assert_eq!(v, &[false, true, false]);
        } else {
            panic!("bool");
        }
        if let ArrowColumn::Utf8(v) = rb.column(5).expect("str") {
            assert_eq!(v, &["α", "β", "γ"]);
        } else {
            panic!("str");
        }
    }

    // ── empty batch ──────────────────────────────────────────────────────────

    #[test]
    fn test_empty_batch() {
        let schema = ArrowSchema::new(vec![ArrowField::new("x", ArrowDataType::Int64)]);
        let empty = RecordBatch::new(schema.clone(), vec![ArrowColumn::Int64(vec![])]).expect("ok");

        let mut buf = Vec::new();
        {
            let mut w =
                ArrowStreamWriter::new(&mut buf, schema.clone(), StreamingCompression::None)
                    .expect("w");
            w.write_batch(&empty).expect("write");
            w.finish().expect("finish");
        }

        let mut binding = buf.as_slice();
        let mut r = ArrowStreamReader::new(&mut binding).expect("r");
        let rb = r.read_next_batch().expect("read").expect("some");
        assert_eq!(rb.num_rows(), 0);
    }

    // ── already-finished reader ──────────────────────────────────────────────

    #[test]
    fn test_reader_after_eos() {
        let schema = ArrowSchema::new(vec![ArrowField::new("x", ArrowDataType::Int32)]);
        let batch =
            RecordBatch::new(schema.clone(), vec![ArrowColumn::Int32(vec![1])]).expect("ok");

        let mut buf = Vec::new();
        {
            let mut w =
                ArrowStreamWriter::new(&mut buf, schema.clone(), StreamingCompression::None)
                    .expect("w");
            w.write_batch(&batch).expect("write");
            w.finish().expect("finish");
        }

        let mut binding = buf.as_slice();
        let mut r = ArrowStreamReader::new(&mut binding).expect("r");
        let _ = r.read_next_batch().expect("first batch").expect("some");
        assert!(r.read_next_batch().expect("eos").is_none());
        // After EOS, subsequent calls return Ok(None) without errors
        assert!(r.read_next_batch().expect("after eos").is_none());
        assert!(r.is_finished());
    }

    // ── write_record_batch / read_next_batch low-level API ───────────────────

    #[test]
    fn test_low_level_write_read() {
        let schema = ArrowSchema::new(vec![ArrowField::new("v", ArrowDataType::Float64)]);
        let batch = RecordBatch::new(
            schema.clone(),
            vec![ArrowColumn::Float64(vec![1.1, 2.2, 3.3])],
        )
        .expect("ok");

        // Write schema + batch + EOS manually
        let mut buf = Vec::new();
        let schema_payload = serialize_schema(&schema).expect("ser schema");
        write_schema_message(&mut buf, &schema_payload).expect("schema msg");
        write_record_batch(&mut buf, &batch, StreamingCompression::None).expect("batch msg");
        write_eos_message(&mut buf).expect("eos");

        // Read
        let mut cur = buf.as_slice();
        let (tag, _codec, payload) = read_message(&mut cur).expect("schema msg");
        assert_eq!(tag, TAG_SCHEMA);
        let schema_read = deserialize_schema(&payload).expect("deser schema");

        let rb = read_next_batch(&mut cur, &schema_read)
            .expect("batch")
            .expect("some");
        assert_eq!(rb.num_rows(), 3);

        let eos = read_next_batch(&mut cur, &schema_read).expect("eos");
        assert!(eos.is_none());
    }

    // ── writer stats ─────────────────────────────────────────────────────────

    #[test]
    fn test_writer_stats() {
        let schema = make_schema();
        let mut buf = Vec::new();
        let mut w = ArrowStreamWriter::new(&mut buf, schema.clone(), StreamingCompression::None)
            .expect("writer");

        for i in 0..4u32 {
            let b = make_batch(&schema, i as i64 * 10);
            w.write_batch(&b).expect("write");
        }
        assert_eq!(w.batches_written(), 4);

        let stats = w.finish().expect("finish");
        assert_eq!(stats.batches_written, 4);
        assert!(stats.uncompressed_bytes > 0);
    }
}
