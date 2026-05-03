//! Apache Avro Object Container File (OCF) format – pure-Rust implementation.
//!
//! This module provides a complete Avro binary encoding/decoding engine and
//! implements the Avro Object Container File format (magic `Obj\x01`).
//!
//! ## Features
//! - Full Avro binary encoding: null, boolean, int (zigzag), long (zigzag),
//!   float, double, bytes, string, array, map, record, enum, fixed, union
//! - Object Container File (OCF): header, sync marker, data blocks
//! - [`AvroWriter`] for streaming record writes
//! - [`AvroReader`] for iterating records from an OCF file
//! - [`write_avro_file`] / [`read_avro_file`] convenience path-based API
//!
//! ## Encoding notes
//! - `int` and `long` use zigzag + base-128 VLQ (same as Protocol Buffers)
//! - `float` = 4-byte IEEE 754 little-endian
//! - `double` = 8-byte IEEE 754 little-endian
//! - `bytes` / `string` = zigzag-encoded length prefix + raw bytes
//! - `array` / `map` = block count (long), items/pairs, zero terminator
//! - OCF sync marker is 16 bytes written after each data block

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;

use crate::error::IoError;

// ─────────────────────────────────────────────────────────────────────────────
// Schema
// ─────────────────────────────────────────────────────────────────────────────

/// Avro schema representation.
#[derive(Debug, Clone, PartialEq)]
pub enum AvroSchema {
    /// `"null"` – always encodes as zero bytes.
    Null,
    /// `"boolean"`.
    Boolean,
    /// `"int"` – 32-bit signed integer (zigzag VLQ).
    Int,
    /// `"long"` – 64-bit signed integer (zigzag VLQ).
    Long,
    /// `"float"` – IEEE 754 single (4 bytes LE).
    Float,
    /// `"double"` – IEEE 754 double (8 bytes LE).
    Double,
    /// `"bytes"` – length-prefixed raw bytes.
    Bytes,
    /// `"string"` – length-prefixed UTF-8.
    String,
    /// `{"type":"array","items":…}`
    Array(Box<AvroSchema>),
    /// `{"type":"map","values":…}` – keys are always strings.
    Map(Box<AvroSchema>),
    /// A named record with ordered fields.
    Record {
        name: String,
        fields: Vec<AvroField>,
    },
    /// A named enum with ordered symbols.
    Enum { name: String, symbols: Vec<String> },
    /// Fixed-length bytes.
    Fixed { name: String, size: usize },
    /// A union of two or more schemas.  The first matching branch is used.
    Union(Vec<AvroSchema>),
}

/// A single field in a [`AvroSchema::Record`].
#[derive(Debug, Clone, PartialEq)]
pub struct AvroField {
    pub name: String,
    pub schema: AvroSchema,
}

impl AvroField {
    pub fn new(name: impl Into<String>, schema: AvroSchema) -> Self {
        Self {
            name: name.into(),
            schema,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Value
// ─────────────────────────────────────────────────────────────────────────────

/// An Avro runtime value.
#[derive(Debug, Clone, PartialEq)]
pub enum AvroValue {
    Null,
    Boolean(bool),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    Bytes(Vec<u8>),
    String(String),
    Array(Vec<AvroValue>),
    Map(HashMap<String, AvroValue>),
    /// A record as an ordered list of `(field_name, value)` pairs.
    Record(Vec<(String, AvroValue)>),
    /// An enum value: the index into the symbols list.
    Enum(usize),
    /// Fixed-length bytes.
    Fixed(Vec<u8>),
    /// A union value: `(branch_index, boxed_value)`.
    Union(usize, Box<AvroValue>),
}

// ─────────────────────────────────────────────────────────────────────────────
// Zigzag / VLQ primitives
// ─────────────────────────────────────────────────────────────────────────────

/// Zigzag-encode then base-128 VLQ-encode a 32-bit integer.
pub fn encode_int(n: i32) -> Vec<u8> {
    encode_long(n as i64)
}

/// Zigzag-encode then base-128 VLQ-encode a 64-bit integer.
pub fn encode_long(n: i64) -> Vec<u8> {
    let zigzag = ((n << 1) ^ (n >> 63)) as u64;
    encode_varuint(zigzag)
}

/// Base-128 VLQ encode an unsigned 64-bit integer.
pub fn encode_varuint(mut v: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(10);
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
    buf
}

/// Decode a zigzag + VLQ encoded 32-bit integer from `bytes`.
///
/// Returns `(value, bytes_consumed)`.
pub fn decode_int(bytes: &[u8]) -> Result<(i32, usize), IoError> {
    let (v, consumed) = decode_long(bytes)?;
    if v > i32::MAX as i64 || v < i32::MIN as i64 {
        return Err(IoError::DeserializationError(
            "Avro: int value out of i32 range".into(),
        ));
    }
    Ok((v as i32, consumed))
}

/// Decode a zigzag + VLQ encoded 64-bit integer from `bytes`.
///
/// Returns `(value, bytes_consumed)`.
pub fn decode_long(bytes: &[u8]) -> Result<(i64, usize), IoError> {
    let (zigzag, consumed) = decode_varuint(bytes)?;
    let v = ((zigzag >> 1) as i64) ^ (-((zigzag & 1) as i64));
    Ok((v, consumed))
}

/// Decode a base-128 VLQ unsigned integer from `bytes`.
///
/// Returns `(value, bytes_consumed)`.
pub fn decode_varuint(bytes: &[u8]) -> Result<(u64, usize), IoError> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    for (i, &byte) in bytes.iter().enumerate() {
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
        shift += 7;
        if shift >= 64 {
            return Err(IoError::DeserializationError(
                "Avro: VLQ integer overflow".into(),
            ));
        }
    }
    Err(IoError::DeserializationError(
        "Avro: unexpected end of VLQ integer".into(),
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Binary encoder
// ─────────────────────────────────────────────────────────────────────────────

/// Encode an [`AvroValue`] into binary bytes using the provided schema.
pub fn encode_value(value: &AvroValue, schema: &AvroSchema) -> Result<Vec<u8>, IoError> {
    let mut buf = Vec::new();
    encode_value_into(value, schema, &mut buf)?;
    Ok(buf)
}

fn encode_value_into(
    value: &AvroValue,
    schema: &AvroSchema,
    buf: &mut Vec<u8>,
) -> Result<(), IoError> {
    match (value, schema) {
        (AvroValue::Null, AvroSchema::Null) => {}

        (AvroValue::Boolean(b), AvroSchema::Boolean) => {
            buf.push(if *b { 1 } else { 0 });
        }

        (AvroValue::Int(n), AvroSchema::Int) => {
            buf.extend_from_slice(&encode_int(*n));
        }

        (AvroValue::Long(n), AvroSchema::Long) => {
            buf.extend_from_slice(&encode_long(*n));
        }

        (AvroValue::Float(f), AvroSchema::Float) => {
            buf.extend_from_slice(&f.to_bits().to_le_bytes());
        }

        (AvroValue::Double(d), AvroSchema::Double) => {
            buf.extend_from_slice(&d.to_bits().to_le_bytes());
        }

        (AvroValue::Bytes(data), AvroSchema::Bytes) => {
            buf.extend_from_slice(&encode_long(data.len() as i64));
            buf.extend_from_slice(data);
        }

        (AvroValue::String(s), AvroSchema::String) => {
            buf.extend_from_slice(&encode_long(s.len() as i64));
            buf.extend_from_slice(s.as_bytes());
        }

        (AvroValue::Array(items), AvroSchema::Array(item_schema)) => {
            if !items.is_empty() {
                buf.extend_from_slice(&encode_long(items.len() as i64));
                for item in items {
                    encode_value_into(item, item_schema, buf)?;
                }
            }
            // Block terminator: count = 0
            buf.extend_from_slice(&encode_long(0));
        }

        (AvroValue::Map(map), AvroSchema::Map(val_schema)) => {
            if !map.is_empty() {
                buf.extend_from_slice(&encode_long(map.len() as i64));
                for (k, v) in map {
                    buf.extend_from_slice(&encode_long(k.len() as i64));
                    buf.extend_from_slice(k.as_bytes());
                    encode_value_into(v, val_schema, buf)?;
                }
            }
            buf.extend_from_slice(&encode_long(0));
        }

        (
            AvroValue::Record(fields),
            AvroSchema::Record {
                fields: field_schemas,
                ..
            },
        ) => {
            if fields.len() != field_schemas.len() {
                return Err(IoError::SerializationError(format!(
                    "Avro: record field count mismatch: got {}, expected {}",
                    fields.len(),
                    field_schemas.len()
                )));
            }
            for ((_name, val), schema_field) in fields.iter().zip(field_schemas.iter()) {
                encode_value_into(val, &schema_field.schema, buf)?;
            }
        }

        (AvroValue::Enum(idx), AvroSchema::Enum { symbols, .. }) => {
            if *idx >= symbols.len() {
                return Err(IoError::SerializationError(format!(
                    "Avro: enum index {idx} out of range (have {} symbols)",
                    symbols.len()
                )));
            }
            buf.extend_from_slice(&encode_int(*idx as i32));
        }

        (AvroValue::Fixed(data), AvroSchema::Fixed { size, .. }) => {
            if data.len() != *size {
                return Err(IoError::SerializationError(format!(
                    "Avro: fixed size mismatch: got {}, expected {size}",
                    data.len()
                )));
            }
            buf.extend_from_slice(data);
        }

        (AvroValue::Union(branch_idx, inner), AvroSchema::Union(branches)) => {
            if *branch_idx >= branches.len() {
                return Err(IoError::SerializationError(format!(
                    "Avro: union branch index {branch_idx} out of range"
                )));
            }
            buf.extend_from_slice(&encode_int(*branch_idx as i32));
            encode_value_into(inner, &branches[*branch_idx], buf)?;
        }

        _ => {
            return Err(IoError::SerializationError(format!(
                "Avro: value/schema type mismatch: {:?} vs {:?}",
                std::mem::discriminant(value),
                std::mem::discriminant(schema)
            )));
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Binary decoder
// ─────────────────────────────────────────────────────────────────────────────

/// Decode an [`AvroValue`] from `data` starting at `pos`.
///
/// Returns `(value, new_pos)`.
pub fn decode_value(
    data: &[u8],
    pos: usize,
    schema: &AvroSchema,
) -> Result<(AvroValue, usize), IoError> {
    match schema {
        AvroSchema::Null => Ok((AvroValue::Null, pos)),

        AvroSchema::Boolean => {
            if pos >= data.len() {
                return Err(IoError::DeserializationError(
                    "Avro: truncated boolean".into(),
                ));
            }
            Ok((AvroValue::Boolean(data[pos] != 0), pos + 1))
        }

        AvroSchema::Int => {
            let (v, consumed) = decode_int(&data[pos..])?;
            Ok((AvroValue::Int(v), pos + consumed))
        }

        AvroSchema::Long => {
            let (v, consumed) = decode_long(&data[pos..])?;
            Ok((AvroValue::Long(v), pos + consumed))
        }

        AvroSchema::Float => {
            if pos + 4 > data.len() {
                return Err(IoError::DeserializationError(
                    "Avro: truncated float".into(),
                ));
            }
            let bits = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            Ok((AvroValue::Float(f32::from_bits(bits)), pos + 4))
        }

        AvroSchema::Double => {
            if pos + 8 > data.len() {
                return Err(IoError::DeserializationError(
                    "Avro: truncated double".into(),
                ));
            }
            let bits = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]);
            Ok((AvroValue::Double(f64::from_bits(bits)), pos + 8))
        }

        AvroSchema::Bytes => {
            let (len, consumed) = decode_long(&data[pos..])?;
            if len < 0 {
                return Err(IoError::DeserializationError(
                    "Avro: negative bytes length".into(),
                ));
            }
            let start = pos + consumed;
            let len = len as usize;
            if start + len > data.len() {
                return Err(IoError::DeserializationError(
                    "Avro: truncated bytes".into(),
                ));
            }
            Ok((
                AvroValue::Bytes(data[start..start + len].to_vec()),
                start + len,
            ))
        }

        AvroSchema::String => {
            let (len, consumed) = decode_long(&data[pos..])?;
            if len < 0 {
                return Err(IoError::DeserializationError(
                    "Avro: negative string length".into(),
                ));
            }
            let start = pos + consumed;
            let len = len as usize;
            if start + len > data.len() {
                return Err(IoError::DeserializationError(
                    "Avro: truncated string".into(),
                ));
            }
            let s = std::str::from_utf8(&data[start..start + len]).map_err(|e| {
                IoError::DeserializationError(format!("Avro: invalid UTF-8 in string: {e}"))
            })?;
            Ok((AvroValue::String(s.to_string()), start + len))
        }

        AvroSchema::Array(item_schema) => {
            let mut items = Vec::new();
            let mut cur = pos;
            loop {
                let (count, consumed) = decode_long(&data[cur..])?;
                cur += consumed;
                if count == 0 {
                    break;
                }
                // Negative count means the block also has a byte-count prefix (ignored here).
                let abs_count = i64::unsigned_abs(count) as usize;
                if count < 0 {
                    // Skip the byte-count long.
                    let (_, consumed2) = decode_long(&data[cur..])?;
                    cur += consumed2;
                }
                for _ in 0..abs_count {
                    let (item, next) = decode_value(data, cur, item_schema)?;
                    items.push(item);
                    cur = next;
                }
            }
            Ok((AvroValue::Array(items), cur))
        }

        AvroSchema::Map(val_schema) => {
            let mut map = HashMap::new();
            let mut cur = pos;
            loop {
                let (count, consumed) = decode_long(&data[cur..])?;
                cur += consumed;
                if count == 0 {
                    break;
                }
                let abs_count = i64::unsigned_abs(count) as usize;
                if count < 0 {
                    let (_, consumed2) = decode_long(&data[cur..])?;
                    cur += consumed2;
                }
                for _ in 0..abs_count {
                    // Read key (string)
                    let (key_len, kc) = decode_long(&data[cur..])?;
                    cur += kc;
                    if key_len < 0 {
                        return Err(IoError::DeserializationError(
                            "Avro: negative map key length".into(),
                        ));
                    }
                    let kl = key_len as usize;
                    if cur + kl > data.len() {
                        return Err(IoError::DeserializationError(
                            "Avro: truncated map key".into(),
                        ));
                    }
                    let key = std::str::from_utf8(&data[cur..cur + kl])
                        .map_err(|e| {
                            IoError::DeserializationError(format!(
                                "Avro: invalid UTF-8 in map key: {e}"
                            ))
                        })?
                        .to_string();
                    cur += kl;
                    let (val, next) = decode_value(data, cur, val_schema)?;
                    map.insert(key, val);
                    cur = next;
                }
            }
            Ok((AvroValue::Map(map), cur))
        }

        AvroSchema::Record { fields, .. } => {
            let mut record_fields = Vec::with_capacity(fields.len());
            let mut cur = pos;
            for field in fields {
                let (val, next) = decode_value(data, cur, &field.schema)?;
                record_fields.push((field.name.clone(), val));
                cur = next;
            }
            Ok((AvroValue::Record(record_fields), cur))
        }

        AvroSchema::Enum { symbols, .. } => {
            let (idx, consumed) = decode_int(&data[pos..])?;
            if idx < 0 || (idx as usize) >= symbols.len() {
                return Err(IoError::DeserializationError(format!(
                    "Avro: enum index {idx} out of range"
                )));
            }
            Ok((AvroValue::Enum(idx as usize), pos + consumed))
        }

        AvroSchema::Fixed { size, .. } => {
            if pos + size > data.len() {
                return Err(IoError::DeserializationError(
                    "Avro: truncated fixed".into(),
                ));
            }
            Ok((AvroValue::Fixed(data[pos..pos + size].to_vec()), pos + size))
        }

        AvroSchema::Union(branches) => {
            let (branch_idx, consumed) = decode_int(&data[pos..])?;
            if branch_idx < 0 || (branch_idx as usize) >= branches.len() {
                return Err(IoError::DeserializationError(format!(
                    "Avro: union branch index {branch_idx} out of range"
                )));
            }
            let bidx = branch_idx as usize;
            let (inner, next) = decode_value(data, pos + consumed, &branches[bidx])?;
            Ok((AvroValue::Union(bidx, Box::new(inner)), next))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Schema JSON serialisation (for OCF header)
// ─────────────────────────────────────────────────────────────────────────────

/// Serialise an [`AvroSchema`] to a JSON string (used in the OCF header metadata).
pub fn schema_to_json(schema: &AvroSchema) -> String {
    match schema {
        AvroSchema::Null => "\"null\"".into(),
        AvroSchema::Boolean => "\"boolean\"".into(),
        AvroSchema::Int => "\"int\"".into(),
        AvroSchema::Long => "\"long\"".into(),
        AvroSchema::Float => "\"float\"".into(),
        AvroSchema::Double => "\"double\"".into(),
        AvroSchema::Bytes => "\"bytes\"".into(),
        AvroSchema::String => "\"string\"".into(),
        AvroSchema::Array(items) => {
            format!(r#"{{"type":"array","items":{}}}"#, schema_to_json(items))
        }
        AvroSchema::Map(values) => {
            format!(r#"{{"type":"map","values":{}}}"#, schema_to_json(values))
        }
        AvroSchema::Record { name, fields } => {
            let fields_json: Vec<String> = fields
                .iter()
                .map(|f| {
                    format!(
                        r#"{{"name":"{}","type":{}}}"#,
                        escape_json_string(&f.name),
                        schema_to_json(&f.schema)
                    )
                })
                .collect();
            format!(
                r#"{{"type":"record","name":"{}","fields":[{}]}}"#,
                escape_json_string(name),
                fields_json.join(",")
            )
        }
        AvroSchema::Enum { name, symbols } => {
            let sym_json: Vec<String> = symbols
                .iter()
                .map(|s| format!("\"{}\"", escape_json_string(s)))
                .collect();
            format!(
                r#"{{"type":"enum","name":"{}","symbols":[{}]}}"#,
                escape_json_string(name),
                sym_json.join(",")
            )
        }
        AvroSchema::Fixed { name, size } => {
            format!(
                r#"{{"type":"fixed","name":"{}","size":{}}}"#,
                escape_json_string(name),
                size
            )
        }
        AvroSchema::Union(branches) => {
            let branch_jsons: Vec<String> = branches.iter().map(schema_to_json).collect();
            format!("[{}]", branch_jsons.join(","))
        }
    }
}

fn escape_json_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ─────────────────────────────────────────────────────────────────────────────
// Object Container File (OCF) format
// ─────────────────────────────────────────────────────────────────────────────

/// Magic bytes for Avro OCF files.
const OCF_MAGIC: &[u8; 4] = b"Obj\x01";

/// Number of records per data block in [`AvroWriter`].
const DEFAULT_BLOCK_SIZE: usize = 100;

// ── OCF Header ────────────────────────────────────────────────────────────────

/// Encode an OCF header and return the bytes along with the sync marker.
///
/// Header layout (Avro binary):
/// ```text
/// magic(4) | metadata-map | sync-marker(16)
/// ```
/// The metadata map carries `"avro.schema"` → `schema_json` (bytes).
fn encode_ocf_header(schema: &AvroSchema) -> (Vec<u8>, [u8; 16]) {
    let schema_json = schema_to_json(schema);
    let sync_marker = generate_sync_marker();
    let mut buf = Vec::new();

    // Magic
    buf.extend_from_slice(OCF_MAGIC);

    // Metadata map: 1 key-value pair, then block terminator 0
    buf.extend_from_slice(&encode_long(1)); // one entry
                                            // Key: "avro.schema"
    let key = b"avro.schema";
    buf.extend_from_slice(&encode_long(key.len() as i64));
    buf.extend_from_slice(key);
    // Value: schema JSON as bytes
    let schema_bytes = schema_json.as_bytes();
    buf.extend_from_slice(&encode_long(schema_bytes.len() as i64));
    buf.extend_from_slice(schema_bytes);
    // Map block terminator
    buf.extend_from_slice(&encode_long(0));

    // Sync marker
    buf.extend_from_slice(&sync_marker);

    (buf, sync_marker)
}

fn generate_sync_marker() -> [u8; 16] {
    // Deterministic but unique-enough marker using process ID + stack address.
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(12345);
    let mut marker = [0u8; 16];
    marker[0..4].copy_from_slice(&pid.to_le_bytes());
    marker[4..12].copy_from_slice(&ts.to_le_bytes());
    marker[12] = 0xAB;
    marker[13] = 0xB0;
    marker[14] = 0x30;
    marker[15] = 0xCC;
    marker
}

// ── OCF Writer ────────────────────────────────────────────────────────────────

/// Streaming Avro Object Container File writer.
///
/// Buffers records into blocks and flushes them with a sync marker.
pub struct AvroWriter<W: Write> {
    writer: W,
    schema: AvroSchema,
    sync_marker: [u8; 16],
    pending_records: Vec<Vec<u8>>,
    block_size: usize,
}

impl<W: Write> AvroWriter<W> {
    /// Create a new writer, writing the OCF header immediately.
    pub fn new(mut writer: W, schema: AvroSchema) -> Result<Self, IoError> {
        let (header_bytes, sync_marker) = encode_ocf_header(&schema);
        writer.write_all(&header_bytes).map_err(|e| {
            IoError::SerializationError(format!("Avro: failed to write OCF header: {e}"))
        })?;
        Ok(Self {
            writer,
            schema,
            sync_marker,
            pending_records: Vec::new(),
            block_size: DEFAULT_BLOCK_SIZE,
        })
    }

    /// Set the number of records per block (default: `DEFAULT_BLOCK_SIZE`).
    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    /// Append a record.  Flushes automatically when the block is full.
    pub fn append(&mut self, value: &AvroValue) -> Result<(), IoError> {
        let encoded = encode_value(value, &self.schema)?;
        self.pending_records.push(encoded);
        if self.pending_records.len() >= self.block_size {
            self.flush_block()?;
        }
        Ok(())
    }

    /// Flush any remaining buffered records and write a final block with sync marker.
    pub fn flush(&mut self) -> Result<(), IoError> {
        self.flush_block()
    }

    fn flush_block(&mut self) -> Result<(), IoError> {
        if self.pending_records.is_empty() {
            return Ok(());
        }
        let count = self.pending_records.len();
        let block_bytes: Vec<u8> = self.pending_records.drain(..).flatten().collect();
        let byte_count = block_bytes.len();

        // Block header: object count (long), byte count (long)
        self.writer
            .write_all(&encode_long(count as i64))
            .map_err(|e| IoError::SerializationError(format!("Avro: block count write: {e}")))?;
        self.writer
            .write_all(&encode_long(byte_count as i64))
            .map_err(|e| {
                IoError::SerializationError(format!("Avro: block byte-count write: {e}"))
            })?;
        self.writer
            .write_all(&block_bytes)
            .map_err(|e| IoError::SerializationError(format!("Avro: block data write: {e}")))?;
        self.writer
            .write_all(&self.sync_marker)
            .map_err(|e| IoError::SerializationError(format!("Avro: sync marker write: {e}")))?;
        Ok(())
    }

    /// Finish writing – flush remaining records and return the inner writer.
    pub fn into_inner(mut self) -> Result<W, IoError> {
        self.flush_block()?;
        Ok(self.writer)
    }
}

// ── OCF Reader ────────────────────────────────────────────────────────────────

/// Parsed Avro OCF header.
struct OcfHeader {
    schema_json: String,
    sync_marker: [u8; 16],
    header_end: usize,
}

fn parse_ocf_header(data: &[u8]) -> Result<OcfHeader, IoError> {
    if data.len() < 4 {
        return Err(IoError::DeserializationError(
            "Avro OCF: too short for magic".into(),
        ));
    }
    if &data[0..4] != OCF_MAGIC {
        return Err(IoError::DeserializationError(
            "Avro OCF: invalid magic bytes".into(),
        ));
    }
    let mut pos = 4;

    // Read metadata map (Map<string, bytes>)
    let mut schema_json = String::new();
    loop {
        let (count, consumed) = decode_long(&data[pos..])?;
        pos += consumed;
        if count == 0 {
            break;
        }
        let abs_count = i64::unsigned_abs(count) as usize;
        if count < 0 {
            // Skip byte-count
            let (_, consumed2) = decode_long(&data[pos..])?;
            pos += consumed2;
        }
        for _ in 0..abs_count {
            // Key
            let (klen, kc) = decode_long(&data[pos..])?;
            pos += kc;
            if klen < 0 {
                return Err(IoError::DeserializationError(
                    "Avro OCF: negative metadata key length".into(),
                ));
            }
            let kl = klen as usize;
            if pos + kl > data.len() {
                return Err(IoError::DeserializationError(
                    "Avro OCF: truncated metadata key".into(),
                ));
            }
            let key = std::str::from_utf8(&data[pos..pos + kl])
                .map_err(|e| {
                    IoError::DeserializationError(format!("Avro OCF: invalid key UTF-8: {e}"))
                })?
                .to_string();
            pos += kl;
            // Value (bytes)
            let (vlen, vc) = decode_long(&data[pos..])?;
            pos += vc;
            if vlen < 0 {
                return Err(IoError::DeserializationError(
                    "Avro OCF: negative metadata value length".into(),
                ));
            }
            let vl = vlen as usize;
            if pos + vl > data.len() {
                return Err(IoError::DeserializationError(
                    "Avro OCF: truncated metadata value".into(),
                ));
            }
            if key == "avro.schema" {
                schema_json = std::str::from_utf8(&data[pos..pos + vl])
                    .map_err(|e| {
                        IoError::DeserializationError(format!(
                            "Avro OCF: invalid schema JSON UTF-8: {e}"
                        ))
                    })?
                    .to_string();
            }
            pos += vl;
        }
    }

    // Sync marker (16 bytes)
    if pos + 16 > data.len() {
        return Err(IoError::DeserializationError(
            "Avro OCF: truncated sync marker in header".into(),
        ));
    }
    let mut sync_marker = [0u8; 16];
    sync_marker.copy_from_slice(&data[pos..pos + 16]);
    pos += 16;

    Ok(OcfHeader {
        schema_json,
        sync_marker,
        header_end: pos,
    })
}

/// Streaming Avro Object Container File reader.
pub struct AvroReader {
    data: Vec<u8>,
    schema: AvroSchema,
    sync_marker: [u8; 16],
    pos: usize,
}

impl AvroReader {
    /// Create a reader from raw OCF bytes.
    ///
    /// The schema is extracted from the header; it is available via
    /// [`AvroReader::schema`].
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, IoError> {
        let header = parse_ocf_header(&data)?;
        let schema = parse_schema_json(&header.schema_json)?;
        Ok(Self {
            data,
            schema,
            sync_marker: header.sync_marker,
            pos: header.header_end,
        })
    }

    /// The schema extracted from the file header.
    pub fn schema(&self) -> &AvroSchema {
        &self.schema
    }

    /// Schema as a JSON string.
    pub fn schema_json(&self) -> String {
        schema_to_json(&self.schema)
    }

    /// Read all remaining records into a `Vec`.
    pub fn read_all(&mut self) -> Result<Vec<AvroValue>, IoError> {
        let mut records = Vec::new();
        while self.pos < self.data.len() {
            let block = self.read_block()?;
            records.extend(block);
        }
        Ok(records)
    }

    fn read_block(&mut self) -> Result<Vec<AvroValue>, IoError> {
        if self.pos >= self.data.len() {
            return Ok(Vec::new());
        }
        // Object count
        let (count, consumed) = decode_long(&self.data[self.pos..])?;
        self.pos += consumed;
        if count == 0 {
            return Ok(Vec::new());
        }
        // Byte count (we don't use it but must consume it)
        let (_, bc) = decode_long(&self.data[self.pos..])?;
        self.pos += bc;

        let mut records = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let (val, next) = decode_value(&self.data, self.pos, &self.schema)?;
            records.push(val);
            self.pos = next;
        }

        // Verify and skip sync marker
        if self.pos + 16 > self.data.len() {
            return Err(IoError::DeserializationError(
                "Avro OCF: truncated sync marker in data block".into(),
            ));
        }
        if self.data[self.pos..self.pos + 16] != self.sync_marker {
            return Err(IoError::DeserializationError(
                "Avro OCF: sync marker mismatch".into(),
            ));
        }
        self.pos += 16;

        Ok(records)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal JSON schema parser
// ─────────────────────────────────────────────────────────────────────────────

/// Parse an [`AvroSchema`] from a JSON string.
///
/// Supports all primitive types and the `record`, `enum`, `array`, `map`,
/// `fixed`, and `union` (JSON array) complex types.
pub fn parse_schema_json(json: &str) -> Result<AvroSchema, IoError> {
    let json = json.trim();
    parse_schema_json_inner(json)
}

fn parse_schema_json_inner(json: &str) -> Result<AvroSchema, IoError> {
    let json = json.trim();

    // Primitive quoted strings
    if json.starts_with('"') && json.ends_with('"') {
        let inner = &json[1..json.len() - 1];
        return match inner {
            "null" => Ok(AvroSchema::Null),
            "boolean" => Ok(AvroSchema::Boolean),
            "int" => Ok(AvroSchema::Int),
            "long" => Ok(AvroSchema::Long),
            "float" => Ok(AvroSchema::Float),
            "double" => Ok(AvroSchema::Double),
            "bytes" => Ok(AvroSchema::Bytes),
            "string" => Ok(AvroSchema::String),
            other => Err(IoError::DeserializationError(format!(
                "Avro schema: unknown primitive type \"{other}\""
            ))),
        };
    }

    // Union (JSON array)
    if json.starts_with('[') {
        let branches = split_json_array(json)?;
        let schemas: Result<Vec<AvroSchema>, IoError> = branches
            .iter()
            .map(|s| parse_schema_json_inner(s.as_str()))
            .collect();
        return Ok(AvroSchema::Union(schemas?));
    }

    // Complex type (JSON object)
    if json.starts_with('{') {
        let type_val = extract_json_string_field(json, "type").ok_or_else(|| {
            IoError::DeserializationError(
                "Avro schema: complex type object missing \"type\" field".into(),
            )
        })?;
        return match type_val.as_str() {
            "record" => parse_record_schema(json),
            "enum" => parse_enum_schema(json),
            "array" => parse_array_schema(json),
            "map" => parse_map_schema(json),
            "fixed" => parse_fixed_schema(json),
            other => Err(IoError::DeserializationError(format!(
                "Avro schema: unknown complex type \"{other}\""
            ))),
        };
    }

    Err(IoError::DeserializationError(format!(
        "Avro schema: cannot parse: {json}"
    )))
}

fn parse_record_schema(json: &str) -> Result<AvroSchema, IoError> {
    let name = extract_json_string_field(json, "name").ok_or_else(|| {
        IoError::DeserializationError("Avro schema: record missing \"name\"".into())
    })?;
    let fields_json = extract_json_array_field(json, "fields").ok_or_else(|| {
        IoError::DeserializationError("Avro schema: record missing \"fields\"".into())
    })?;
    let field_objects = split_json_array(&fields_json)?;
    let mut fields = Vec::new();
    for field_obj in &field_objects {
        let fname = extract_json_string_field(field_obj.as_str(), "name").ok_or_else(|| {
            IoError::DeserializationError("Avro schema: record field missing \"name\"".into())
        })?;
        let ftype_json = extract_json_value_field(field_obj.as_str(), "type").ok_or_else(|| {
            IoError::DeserializationError("Avro schema: record field missing \"type\"".into())
        })?;
        let fschema = parse_schema_json_inner(&ftype_json)?;
        fields.push(AvroField::new(fname, fschema));
    }
    Ok(AvroSchema::Record { name, fields })
}

fn parse_enum_schema(json: &str) -> Result<AvroSchema, IoError> {
    let name = extract_json_string_field(json, "name").ok_or_else(|| {
        IoError::DeserializationError("Avro schema: enum missing \"name\"".into())
    })?;
    let symbols_json = extract_json_array_field(json, "symbols").ok_or_else(|| {
        IoError::DeserializationError("Avro schema: enum missing \"symbols\"".into())
    })?;
    let sym_parts = split_json_array(&symbols_json)?;
    let symbols: Vec<String> = sym_parts
        .iter()
        .map(|s| {
            let s = s.trim();
            if s.starts_with('"') && s.ends_with('"') {
                Ok(s[1..s.len() - 1].to_string())
            } else {
                Err(IoError::DeserializationError(format!(
                    "Avro schema: enum symbol is not a string: {s}"
                )))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(AvroSchema::Enum { name, symbols })
}

fn parse_array_schema(json: &str) -> Result<AvroSchema, IoError> {
    let items_json = extract_json_value_field(json, "items").ok_or_else(|| {
        IoError::DeserializationError("Avro schema: array missing \"items\"".into())
    })?;
    let item_schema = parse_schema_json_inner(&items_json)?;
    Ok(AvroSchema::Array(Box::new(item_schema)))
}

fn parse_map_schema(json: &str) -> Result<AvroSchema, IoError> {
    let values_json = extract_json_value_field(json, "values").ok_or_else(|| {
        IoError::DeserializationError("Avro schema: map missing \"values\"".into())
    })?;
    let val_schema = parse_schema_json_inner(&values_json)?;
    Ok(AvroSchema::Map(Box::new(val_schema)))
}

fn parse_fixed_schema(json: &str) -> Result<AvroSchema, IoError> {
    let name = extract_json_string_field(json, "name").ok_or_else(|| {
        IoError::DeserializationError("Avro schema: fixed missing \"name\"".into())
    })?;
    let size_json = extract_json_value_field(json, "size").ok_or_else(|| {
        IoError::DeserializationError("Avro schema: fixed missing \"size\"".into())
    })?;
    let size: usize = size_json.trim().parse().map_err(|e| {
        IoError::DeserializationError(format!("Avro schema: fixed size parse error: {e}"))
    })?;
    Ok(AvroSchema::Fixed { name, size })
}

// ── Minimal JSON helpers ──────────────────────────────────────────────────────

/// Extract a string value for a key from a flat JSON object string.
fn extract_json_string_field(json: &str, key: &str) -> Option<String> {
    let search = format!("\"{}\"", key);
    let start = json.find(&search)?;
    let after_key = &json[start + search.len()..];
    // Skip whitespace and colon
    let after_colon = after_key.trim_start().strip_prefix(':')?.trim_start();
    if after_colon.starts_with('"') {
        let inner_start = 1;
        let mut i = inner_start;
        let chars: Vec<char> = after_colon.chars().collect();
        while i < chars.len() {
            if chars[i] == '\\' {
                i += 2;
            } else if chars[i] == '"' {
                let raw: String = after_colon[inner_start..i].to_string();
                return Some(raw.replace("\\\"", "\"").replace("\\\\", "\\"));
            } else {
                i += 1;
            }
        }
        None
    } else {
        None
    }
}

/// Extract an arbitrary JSON value (as raw JSON string) for a key.
fn extract_json_value_field(json: &str, key: &str) -> Option<String> {
    let search = format!("\"{}\"", key);
    let start = json.find(&search)?;
    let after_key = &json[start + search.len()..];
    let after_colon = after_key.trim_start().strip_prefix(':')?.trim_start();
    Some(extract_json_value_token(after_colon))
}

/// Extract a JSON array (as raw JSON string) for a key.
fn extract_json_array_field(json: &str, key: &str) -> Option<String> {
    let val = extract_json_value_field(json, key)?;
    if val.trim_start().starts_with('[') {
        Some(val)
    } else {
        None
    }
}

/// Extract one complete JSON value token from the beginning of `s`.
fn extract_json_value_token(s: &str) -> String {
    let s = s.trim_start();
    if s.starts_with('"') {
        // String token
        let mut i = 1;
        let bytes = s.as_bytes();
        while i < bytes.len() {
            if bytes[i] == b'\\' {
                i += 2;
            } else if bytes[i] == b'"' {
                return s[..i + 1].to_string();
            } else {
                i += 1;
            }
        }
        s.to_string()
    } else if s.starts_with('{') || s.starts_with('[') {
        let open = s.as_bytes()[0];
        let close = if open == b'{' { b'}' } else { b']' };
        let mut depth = 0usize;
        let mut in_string = false;
        for (i, &b) in s.as_bytes().iter().enumerate() {
            if in_string {
                if b == b'"' {
                    in_string = false;
                } else if b == b'\\' {
                    // skip next character handled by the loop incrementing i
                    // but we need to advance manually - handle differently
                }
            } else if b == b'"' {
                in_string = true;
            } else if b == open {
                depth += 1;
            } else if b == close {
                depth -= 1;
                if depth == 0 {
                    return s[..i + 1].to_string();
                }
            }
        }
        s.to_string()
    } else {
        // Number, boolean, null
        let end = s
            .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
            .unwrap_or(s.len());
        s[..end].to_string()
    }
}

/// Split a JSON array string (e.g. `[a, b, c]`) into element strings.
fn split_json_array(json: &str) -> Result<Vec<String>, IoError> {
    let json = json.trim();
    if !json.starts_with('[') || !json.ends_with(']') {
        return Err(IoError::DeserializationError(format!(
            "Avro schema: expected JSON array, got: {json}"
        )));
    }
    let inner = &json[1..json.len() - 1];
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut start = 0;
    let bytes = inner.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            if b == b'\\' {
                i += 1; // skip escaped char
            } else if b == b'"' {
                in_string = false;
            }
        } else {
            match b {
                b'"' => in_string = true,
                b'{' | b'[' => depth += 1,
                b'}' | b']' => depth -= 1,
                b',' if depth == 0 => {
                    let part = inner[start..i].trim();
                    if !part.is_empty() {
                        parts.push(part.to_string());
                    }
                    start = i + 1;
                }
                _ => {}
            }
        }
        i += 1;
    }
    let last = inner[start..].trim();
    if !last.is_empty() {
        parts.push(last.to_string());
    }
    Ok(parts)
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience path-based API
// ─────────────────────────────────────────────────────────────────────────────

/// Write records to an Avro OCF file.
///
/// `schema_json` is the Avro schema as a JSON string (e.g. the result of
/// [`schema_to_json`]).  Returns an error if encoding fails or the file cannot
/// be written.
pub fn write_avro_file(
    path: impl AsRef<Path>,
    schema_json: &str,
    records: &[AvroValue],
) -> Result<(), IoError> {
    let schema = parse_schema_json(schema_json)?;
    let buf: Vec<u8> = Vec::new();
    let cursor = io::Cursor::new(buf);
    let mut writer = AvroWriter::new(cursor, schema)?;
    for record in records {
        writer.append(record)?;
    }
    let cursor = writer.into_inner()?;
    fs::write(path.as_ref(), cursor.into_inner())
        .map_err(|e| IoError::SerializationError(format!("Avro: cannot write file: {e}")))
}

/// Read all records from an Avro OCF file.
///
/// Returns `(schema_json, records)`.
pub fn read_avro_file(path: impl AsRef<Path>) -> Result<(String, Vec<AvroValue>), IoError> {
    let data = fs::read(path.as_ref())
        .map_err(|e| IoError::DeserializationError(format!("Avro: cannot read file: {e}")))?;
    let mut reader = AvroReader::from_bytes(data)?;
    let schema_json = reader.schema_json();
    let records = reader.read_all()?;
    Ok((schema_json, records))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    // ── Zigzag / VLQ ──────────────────────────────────────────────────────────

    #[test]
    fn test_zigzag_int_roundtrip() {
        for &v in &[0i32, 1, -1, 64, -64, i32::MAX, i32::MIN] {
            let encoded = encode_int(v);
            let (decoded, consumed) = decode_int(&encoded).expect("decode_int failed");
            assert_eq!(decoded, v, "round-trip failed for {v}");
            assert_eq!(consumed, encoded.len(), "consumed mismatch for {v}");
        }
    }

    #[test]
    fn test_zigzag_long_roundtrip() {
        for &v in &[0i64, 1, -1, 1000, -1000, i64::MAX, i64::MIN] {
            let encoded = encode_long(v);
            let (decoded, consumed) = decode_long(&encoded).expect("decode_long failed");
            assert_eq!(decoded, v, "round-trip failed for {v}");
            assert_eq!(consumed, encoded.len(), "consumed mismatch for {v}");
        }
    }

    // ── Primitive value encoding ──────────────────────────────────────────────

    #[test]
    fn test_null_encoding() {
        let encoded = encode_value(&AvroValue::Null, &AvroSchema::Null).expect("encode failed");
        assert!(encoded.is_empty());
        let (val, pos) = decode_value(&encoded, 0, &AvroSchema::Null).expect("decode failed");
        assert_eq!(val, AvroValue::Null);
        assert_eq!(pos, 0);
    }

    #[test]
    fn test_boolean_encoding() {
        for &b in &[true, false] {
            let encoded =
                encode_value(&AvroValue::Boolean(b), &AvroSchema::Boolean).expect("encode");
            let (val, _) = decode_value(&encoded, 0, &AvroSchema::Boolean).expect("decode");
            assert_eq!(val, AvroValue::Boolean(b));
        }
    }

    #[test]
    fn test_int_encoding() {
        for &n in &[0i32, 42, -1, i32::MAX, i32::MIN] {
            let encoded = encode_value(&AvroValue::Int(n), &AvroSchema::Int).expect("encode");
            let (val, _) = decode_value(&encoded, 0, &AvroSchema::Int).expect("decode");
            assert_eq!(val, AvroValue::Int(n));
        }
    }

    #[test]
    fn test_long_encoding() {
        for &n in &[0i64, 1_000_000, -1_000_000, i64::MAX, i64::MIN] {
            let encoded = encode_value(&AvroValue::Long(n), &AvroSchema::Long).expect("encode");
            let (val, _) = decode_value(&encoded, 0, &AvroSchema::Long).expect("decode");
            assert_eq!(val, AvroValue::Long(n));
        }
    }

    #[test]
    fn test_float_encoding() {
        let f = std::f32::consts::PI;
        let encoded = encode_value(&AvroValue::Float(f), &AvroSchema::Float).expect("encode");
        let (val, _) = decode_value(&encoded, 0, &AvroSchema::Float).expect("decode");
        if let AvroValue::Float(decoded) = val {
            assert!((decoded - f).abs() < 1e-6);
        } else {
            panic!("expected Float");
        }
    }

    #[test]
    fn test_double_encoding() {
        let d = std::f64::consts::E;
        let encoded = encode_value(&AvroValue::Double(d), &AvroSchema::Double).expect("encode");
        let (val, _) = decode_value(&encoded, 0, &AvroSchema::Double).expect("decode");
        if let AvroValue::Double(decoded) = val {
            assert!((decoded - d).abs() < 1e-15);
        } else {
            panic!("expected Double");
        }
    }

    #[test]
    fn test_bytes_encoding() {
        let data = vec![0xde, 0xad, 0xbe, 0xef];
        let encoded =
            encode_value(&AvroValue::Bytes(data.clone()), &AvroSchema::Bytes).expect("encode");
        let (val, _) = decode_value(&encoded, 0, &AvroSchema::Bytes).expect("decode");
        assert_eq!(val, AvroValue::Bytes(data));
    }

    #[test]
    fn test_string_encoding() {
        let s = "hello, Avro!".to_string();
        let encoded =
            encode_value(&AvroValue::String(s.clone()), &AvroSchema::String).expect("encode");
        let (val, _) = decode_value(&encoded, 0, &AvroSchema::String).expect("decode");
        assert_eq!(val, AvroValue::String(s));
    }

    // ── Array ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_array_encoding() {
        let schema = AvroSchema::Array(Box::new(AvroSchema::Int));
        let value = AvroValue::Array(vec![
            AvroValue::Int(1),
            AvroValue::Int(2),
            AvroValue::Int(3),
        ]);
        let encoded = encode_value(&value, &schema).expect("encode");
        let (decoded, _) = decode_value(&encoded, 0, &schema).expect("decode");
        assert_eq!(decoded, value);
    }

    #[test]
    fn test_empty_array_encoding() {
        let schema = AvroSchema::Array(Box::new(AvroSchema::String));
        let value = AvroValue::Array(vec![]);
        let encoded = encode_value(&value, &schema).expect("encode");
        let (decoded, _) = decode_value(&encoded, 0, &schema).expect("decode");
        assert_eq!(decoded, value);
    }

    // ── Map ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_map_encoding() {
        let schema = AvroSchema::Map(Box::new(AvroSchema::Long));
        let mut map = HashMap::new();
        map.insert("alpha".to_string(), AvroValue::Long(100));
        map.insert("beta".to_string(), AvroValue::Long(200));
        let value = AvroValue::Map(map);
        let encoded = encode_value(&value, &schema).expect("encode");
        let (decoded, _) = decode_value(&encoded, 0, &schema).expect("decode");
        if let AvroValue::Map(m) = decoded {
            assert_eq!(m["alpha"], AvroValue::Long(100));
            assert_eq!(m["beta"], AvroValue::Long(200));
        } else {
            panic!("expected Map");
        }
    }

    // ── Record ────────────────────────────────────────────────────────────────

    #[test]
    fn test_record_encoding() {
        let schema = AvroSchema::Record {
            name: "Person".into(),
            fields: vec![
                AvroField::new("name", AvroSchema::String),
                AvroField::new("age", AvroSchema::Int),
                AvroField::new("active", AvroSchema::Boolean),
            ],
        };
        let value = AvroValue::Record(vec![
            ("name".into(), AvroValue::String("Alice".into())),
            ("age".into(), AvroValue::Int(30)),
            ("active".into(), AvroValue::Boolean(true)),
        ]);
        let encoded = encode_value(&value, &schema).expect("encode");
        let (decoded, _) = decode_value(&encoded, 0, &schema).expect("decode");
        assert_eq!(decoded, value);
    }

    // ── Enum ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_enum_encoding() {
        let schema = AvroSchema::Enum {
            name: "Color".into(),
            symbols: vec!["RED".into(), "GREEN".into(), "BLUE".into()],
        };
        for idx in 0..3usize {
            let value = AvroValue::Enum(idx);
            let encoded = encode_value(&value, &schema).expect("encode");
            let (decoded, _) = decode_value(&encoded, 0, &schema).expect("decode");
            assert_eq!(decoded, value);
        }
    }

    // ── Fixed ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_fixed_encoding() {
        let schema = AvroSchema::Fixed {
            name: "MD5".into(),
            size: 16,
        };
        let data: Vec<u8> = (0u8..16).collect();
        let value = AvroValue::Fixed(data);
        let encoded = encode_value(&value, &schema).expect("encode");
        let (decoded, _) = decode_value(&encoded, 0, &schema).expect("decode");
        assert_eq!(decoded, value);
    }

    // ── Union ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_union_null_or_string() {
        let schema = AvroSchema::Union(vec![AvroSchema::Null, AvroSchema::String]);
        // Branch 0: null
        let v0 = AvroValue::Union(0, Box::new(AvroValue::Null));
        let enc0 = encode_value(&v0, &schema).expect("encode null branch");
        let (dec0, _) = decode_value(&enc0, 0, &schema).expect("decode null branch");
        assert_eq!(dec0, v0);
        // Branch 1: string
        let v1 = AvroValue::Union(1, Box::new(AvroValue::String("hello".into())));
        let enc1 = encode_value(&v1, &schema).expect("encode string branch");
        let (dec1, _) = decode_value(&enc1, 0, &schema).expect("decode string branch");
        assert_eq!(dec1, v1);
    }

    // ── Schema JSON roundtrip ─────────────────────────────────────────────────

    #[test]
    fn test_schema_json_primitives() {
        for schema in &[
            AvroSchema::Null,
            AvroSchema::Boolean,
            AvroSchema::Int,
            AvroSchema::Long,
            AvroSchema::Float,
            AvroSchema::Double,
            AvroSchema::Bytes,
            AvroSchema::String,
        ] {
            let json = schema_to_json(schema);
            let parsed = parse_schema_json(&json).expect("parse failed");
            assert_eq!(&parsed, schema);
        }
    }

    #[test]
    fn test_schema_json_record() {
        let schema = AvroSchema::Record {
            name: "TestRecord".into(),
            fields: vec![
                AvroField::new("id", AvroSchema::Long),
                AvroField::new("label", AvroSchema::String),
            ],
        };
        let json = schema_to_json(&schema);
        let parsed = parse_schema_json(&json).expect("parse failed");
        assert_eq!(parsed, schema);
    }

    #[test]
    fn test_schema_json_enum() {
        let schema = AvroSchema::Enum {
            name: "Status".into(),
            symbols: vec!["ACTIVE".into(), "INACTIVE".into(), "PENDING".into()],
        };
        let json = schema_to_json(&schema);
        let parsed = parse_schema_json(&json).expect("parse failed");
        assert_eq!(parsed, schema);
    }

    #[test]
    fn test_schema_json_array_of_strings() {
        let schema = AvroSchema::Array(Box::new(AvroSchema::String));
        let json = schema_to_json(&schema);
        let parsed = parse_schema_json(&json).expect("parse failed");
        assert_eq!(parsed, schema);
    }

    #[test]
    fn test_schema_json_map_of_doubles() {
        let schema = AvroSchema::Map(Box::new(AvroSchema::Double));
        let json = schema_to_json(&schema);
        let parsed = parse_schema_json(&json).expect("parse failed");
        assert_eq!(parsed, schema);
    }

    #[test]
    fn test_schema_json_union() {
        let schema = AvroSchema::Union(vec![AvroSchema::Null, AvroSchema::Int]);
        let json = schema_to_json(&schema);
        let parsed = parse_schema_json(&json).expect("parse failed");
        assert_eq!(parsed, schema);
    }

    // ── OCF file round-trip ───────────────────────────────────────────────────

    #[test]
    fn test_ocf_simple_record_roundtrip() {
        let schema = AvroSchema::Record {
            name: "SensorReading".into(),
            fields: vec![
                AvroField::new("sensor_id", AvroSchema::Int),
                AvroField::new("value", AvroSchema::Double),
                AvroField::new("label", AvroSchema::String),
            ],
        };

        let records: Vec<AvroValue> = (0..5)
            .map(|i| {
                AvroValue::Record(vec![
                    ("sensor_id".into(), AvroValue::Int(i)),
                    ("value".into(), AvroValue::Double(i as f64 * 1.5)),
                    ("label".into(), AvroValue::String(format!("sensor_{i}"))),
                ])
            })
            .collect();

        let schema_json = schema_to_json(&schema);
        let path = temp_dir().join("avro_record_roundtrip.avro");
        write_avro_file(&path, &schema_json, &records).expect("write failed");
        let (returned_schema_json, decoded_records) = read_avro_file(&path).expect("read failed");

        // Schema should survive the round-trip
        let returned_schema =
            parse_schema_json(&returned_schema_json).expect("schema re-parse failed");
        assert_eq!(returned_schema, schema);
        assert_eq!(decoded_records.len(), records.len());
        assert_eq!(decoded_records[0], records[0]);
        assert_eq!(decoded_records[4], records[4]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_ocf_primitive_long_roundtrip() {
        let schema_json = schema_to_json(&AvroSchema::Long);
        let records: Vec<AvroValue> = vec![
            AvroValue::Long(0),
            AvroValue::Long(-1),
            AvroValue::Long(i64::MAX),
            AvroValue::Long(i64::MIN),
        ];
        let path = temp_dir().join("avro_long_roundtrip.avro");
        write_avro_file(&path, &schema_json, &records).expect("write");
        let (_, decoded) = read_avro_file(&path).expect("read");
        assert_eq!(decoded, records);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_ocf_large_record_count() {
        let schema = AvroSchema::Record {
            name: "Row".into(),
            fields: vec![
                AvroField::new("x", AvroSchema::Int),
                AvroField::new("y", AvroSchema::Double),
            ],
        };
        let n = 500usize;
        let records: Vec<AvroValue> = (0..n)
            .map(|i| {
                AvroValue::Record(vec![
                    ("x".into(), AvroValue::Int(i as i32)),
                    ("y".into(), AvroValue::Double(i as f64)),
                ])
            })
            .collect();
        let schema_json = schema_to_json(&schema);
        let path = temp_dir().join("avro_large_roundtrip.avro");
        write_avro_file(&path, &schema_json, &records).expect("write");
        let (_, decoded) = read_avro_file(&path).expect("read");
        assert_eq!(decoded.len(), n);
        assert_eq!(decoded[0], records[0]);
        assert_eq!(decoded[n - 1], records[n - 1]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_ocf_empty_records() {
        let schema_json = schema_to_json(&AvroSchema::String);
        let records: Vec<AvroValue> = vec![];
        let path = temp_dir().join("avro_empty_roundtrip.avro");
        write_avro_file(&path, &schema_json, &records).expect("write");
        let (_, decoded) = read_avro_file(&path).expect("read");
        assert_eq!(decoded.len(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_ocf_union_nullable_field() {
        let schema = AvroSchema::Record {
            name: "NullableTest".into(),
            fields: vec![
                AvroField::new("id", AvroSchema::Int),
                AvroField::new(
                    "optional_label",
                    AvroSchema::Union(vec![AvroSchema::Null, AvroSchema::String]),
                ),
            ],
        };
        let records = vec![
            AvroValue::Record(vec![
                ("id".into(), AvroValue::Int(1)),
                (
                    "optional_label".into(),
                    AvroValue::Union(0, Box::new(AvroValue::Null)),
                ),
            ]),
            AvroValue::Record(vec![
                ("id".into(), AvroValue::Int(2)),
                (
                    "optional_label".into(),
                    AvroValue::Union(1, Box::new(AvroValue::String("hello".into()))),
                ),
            ]),
        ];
        let schema_json = schema_to_json(&schema);
        let path = temp_dir().join("avro_union_nullable.avro");
        write_avro_file(&path, &schema_json, &records).expect("write");
        let (_, decoded) = read_avro_file(&path).expect("read");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0], records[0]);
        assert_eq!(decoded[1], records[1]);
        let _ = std::fs::remove_file(&path);
    }
}
