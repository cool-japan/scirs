//! Pure Rust Arrow IPC file format reader/writer
//!
//! Implements a simplified Arrow IPC (Inter-Process Communication) format for
//! efficient columnar data exchange. This is a pure Rust implementation that does
//! not depend on the Apache Arrow C++ library or the `arrow` crate.
//!
//! ## Supported Types
//!
//! - `Int32`, `Int64` (signed integers)
//! - `Float32`, `Float64` (IEEE 754 floating point)
//! - `Utf8` (variable-length UTF-8 strings)
//! - `Boolean` (bit-packed booleans)
//!
//! ## Format Overview
//!
//! The IPC file format consists of:
//! 1. Magic bytes "ARROW1" + padding
//! 2. Schema message (field names and types)
//! 3. One or more record batch messages (columnar data)
//! 4. Footer with schema copy and record batch locations
//! 5. Footer length (4 bytes, little-endian)
//! 6. Magic bytes "ARROW1"
//!
//! The streaming format omits the footer and trailing magic, suitable for
//! append-only and pipe-based workflows.
//!
//! # Example
//!
//! ```rust
//! use scirs2_io::arrow_ipc::*;
//!
//! // Define schema
//! let schema = ArrowSchema::new(vec![
//!     ArrowField::new("id", ArrowDataType::Int32),
//!     ArrowField::new("value", ArrowDataType::Float64),
//!     ArrowField::new("name", ArrowDataType::Utf8),
//! ]);
//!
//! // Create record batch
//! let batch = RecordBatch::new(schema.clone(), vec![
//!     ArrowColumn::Int32(vec![1, 2, 3]),
//!     ArrowColumn::Float64(vec![1.1, 2.2, 3.3]),
//!     ArrowColumn::Utf8(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
//! ]).expect("valid batch");
//!
//! // Write to bytes
//! let bytes = write_arrow_ipc_bytes(&schema, &[batch.clone()]).expect("write");
//!
//! // Read back
//! let (read_schema, batches) = read_arrow_ipc_bytes(&bytes).expect("read");
//! assert_eq!(read_schema.fields.len(), 3);
//! assert_eq!(batches.len(), 1);
//! assert_eq!(batches[0].num_rows(), 3);
//! ```

use crate::error::{IoError, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Magic bytes for Arrow IPC file format
const ARROW_MAGIC: &[u8; 6] = b"ARROW1";

/// Padding to 8-byte alignment
const ALIGNMENT: usize = 8;

// =====================================================================
// Public types
// =====================================================================

/// Arrow data types supported by this implementation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArrowDataType {
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 32-bit IEEE 754 float
    Float32,
    /// 64-bit IEEE 754 float
    Float64,
    /// Variable-length UTF-8 string
    Utf8,
    /// Boolean (bit-packed)
    Boolean,
}

impl ArrowDataType {
    /// Type tag byte for serialization
    fn tag(&self) -> u8 {
        match self {
            Self::Int32 => 1,
            Self::Int64 => 2,
            Self::Float32 => 3,
            Self::Float64 => 4,
            Self::Utf8 => 5,
            Self::Boolean => 6,
        }
    }

    /// Reconstruct from tag byte
    fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            1 => Ok(Self::Int32),
            2 => Ok(Self::Int64),
            3 => Ok(Self::Float32),
            4 => Ok(Self::Float64),
            5 => Ok(Self::Utf8),
            6 => Ok(Self::Boolean),
            _ => Err(IoError::FormatError(format!(
                "Unknown Arrow type tag: {tag}"
            ))),
        }
    }
}

/// A field in an Arrow schema
#[derive(Debug, Clone, PartialEq)]
pub struct ArrowField {
    /// Field name
    pub name: String,
    /// Field data type
    pub dtype: ArrowDataType,
    /// Whether the field is nullable
    pub nullable: bool,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl ArrowField {
    /// Create a new field (non-nullable by default)
    pub fn new(name: &str, dtype: ArrowDataType) -> Self {
        Self {
            name: name.to_string(),
            dtype,
            nullable: false,
            metadata: HashMap::new(),
        }
    }

    /// Create a nullable field
    pub fn new_nullable(name: &str, dtype: ArrowDataType) -> Self {
        Self {
            name: name.to_string(),
            dtype,
            nullable: true,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to this field
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// An Arrow schema (ordered list of fields)
#[derive(Debug, Clone, PartialEq)]
pub struct ArrowSchema {
    /// Fields in the schema
    pub fields: Vec<ArrowField>,
    /// Schema-level metadata
    pub metadata: HashMap<String, String>,
}

impl ArrowSchema {
    /// Create a new schema from fields
    pub fn new(fields: Vec<ArrowField>) -> Self {
        Self {
            fields,
            metadata: HashMap::new(),
        }
    }

    /// Add schema metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Number of fields
    pub fn num_fields(&self) -> usize {
        self.fields.len()
    }

    /// Get a field by index
    pub fn field(&self, index: usize) -> Option<&ArrowField> {
        self.fields.get(index)
    }

    /// Find a field by name
    pub fn field_by_name(&self, name: &str) -> Option<(usize, &ArrowField)> {
        self.fields.iter().enumerate().find(|(_, f)| f.name == name)
    }
}

/// A column of typed data
#[derive(Debug, Clone, PartialEq)]
pub enum ArrowColumn {
    /// 32-bit signed integers
    Int32(Vec<i32>),
    /// 64-bit signed integers
    Int64(Vec<i64>),
    /// 32-bit floats
    Float32(Vec<f32>),
    /// 64-bit floats
    Float64(Vec<f64>),
    /// UTF-8 strings
    Utf8(Vec<String>),
    /// Booleans
    Boolean(Vec<bool>),
}

impl ArrowColumn {
    /// Number of elements in this column
    pub fn len(&self) -> usize {
        match self {
            Self::Int32(v) => v.len(),
            Self::Int64(v) => v.len(),
            Self::Float32(v) => v.len(),
            Self::Float64(v) => v.len(),
            Self::Utf8(v) => v.len(),
            Self::Boolean(v) => v.len(),
        }
    }

    /// Whether the column is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the data type of this column
    pub fn data_type(&self) -> ArrowDataType {
        match self {
            Self::Int32(_) => ArrowDataType::Int32,
            Self::Int64(_) => ArrowDataType::Int64,
            Self::Float32(_) => ArrowDataType::Float32,
            Self::Float64(_) => ArrowDataType::Float64,
            Self::Utf8(_) => ArrowDataType::Utf8,
            Self::Boolean(_) => ArrowDataType::Boolean,
        }
    }

    /// Try to convert column to f64 values (returns None for string/boolean)
    pub fn as_f64(&self) -> Option<Vec<f64>> {
        match self {
            Self::Int32(v) => Some(v.iter().map(|x| *x as f64).collect()),
            Self::Int64(v) => Some(v.iter().map(|x| *x as f64).collect()),
            Self::Float32(v) => Some(v.iter().map(|x| *x as f64).collect()),
            Self::Float64(v) => Some(v.clone()),
            _ => None,
        }
    }
}

/// A record batch (table chunk with a fixed number of rows)
#[derive(Debug, Clone, PartialEq)]
pub struct RecordBatch {
    /// Schema for this batch
    pub schema: ArrowSchema,
    /// Column data (one per field)
    pub columns: Vec<ArrowColumn>,
    /// Number of rows
    num_rows: usize,
}

impl RecordBatch {
    /// Create a new record batch, validating schema and column lengths
    pub fn new(schema: ArrowSchema, columns: Vec<ArrowColumn>) -> Result<Self> {
        if columns.len() != schema.fields.len() {
            return Err(IoError::FormatError(format!(
                "Expected {} columns, got {}",
                schema.fields.len(),
                columns.len()
            )));
        }

        // Validate column types match schema
        for (i, (col, field)) in columns.iter().zip(schema.fields.iter()).enumerate() {
            if col.data_type() != field.dtype {
                return Err(IoError::FormatError(format!(
                    "Column {i} type {:?} does not match schema type {:?}",
                    col.data_type(),
                    field.dtype
                )));
            }
        }

        // All columns must have the same length
        let num_rows = columns.first().map_or(0, |c| c.len());
        for (i, col) in columns.iter().enumerate() {
            if col.len() != num_rows {
                return Err(IoError::FormatError(format!(
                    "Column {i} has {} rows, expected {num_rows}",
                    col.len()
                )));
            }
        }

        Ok(Self {
            schema,
            columns,
            num_rows,
        })
    }

    /// Number of rows
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of columns
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Get a column by index
    pub fn column(&self, index: usize) -> Option<&ArrowColumn> {
        self.columns.get(index)
    }

    /// Get a column by name
    pub fn column_by_name(&self, name: &str) -> Option<&ArrowColumn> {
        self.schema
            .field_by_name(name)
            .and_then(|(idx, _)| self.columns.get(idx))
    }
}

// =====================================================================
// Streaming writer
// =====================================================================

/// Arrow IPC streaming writer
///
/// Writes record batches in streaming format (no footer).
/// Useful for pipe-based and append-only workflows.
pub struct ArrowStreamWriter<W: Write> {
    writer: W,
    schema: ArrowSchema,
    batches_written: usize,
}

impl<W: Write> ArrowStreamWriter<W> {
    /// Create a new streaming writer and write the schema message
    pub fn new(mut writer: W, schema: ArrowSchema) -> Result<Self> {
        // Write schema message
        let schema_bytes = serialize_schema(&schema)?;
        write_message(&mut writer, 0x01, &schema_bytes)?;

        Ok(Self {
            writer,
            schema,
            batches_written: 0,
        })
    }

    /// Write a record batch
    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.schema != self.schema {
            return Err(IoError::FormatError(
                "Batch schema does not match writer schema".to_string(),
            ));
        }

        let batch_bytes = serialize_record_batch(batch)?;
        write_message(&mut self.writer, 0x02, &batch_bytes)?;
        self.batches_written += 1;
        Ok(())
    }

    /// Finish writing (writes EOS marker)
    pub fn finish(mut self) -> Result<W> {
        // End-of-stream marker: message type 0, size 0
        self.writer
            .write_u32::<LittleEndian>(0)
            .map_err(|e| IoError::Io(e))?;
        self.writer
            .write_u32::<LittleEndian>(0)
            .map_err(|e| IoError::Io(e))?;
        Ok(self.writer)
    }

    /// Number of batches written so far
    pub fn batches_written(&self) -> usize {
        self.batches_written
    }
}

// =====================================================================
// File format writer
// =====================================================================

/// Write Arrow IPC file format to a writer
pub fn write_arrow_ipc<W: Write + Seek>(
    writer: &mut W,
    schema: &ArrowSchema,
    batches: &[RecordBatch],
) -> Result<()> {
    // Write magic + padding
    writer.write_all(ARROW_MAGIC).map_err(|e| IoError::Io(e))?;
    // Pad to 8-byte boundary (6 + 2 = 8)
    writer.write_all(&[0u8; 2]).map_err(|e| IoError::Io(e))?;

    // Write schema message
    let schema_bytes = serialize_schema(schema)?;
    write_message(writer, 0x01, &schema_bytes)?;

    // Write record batches, recording their offsets
    let mut batch_offsets = Vec::with_capacity(batches.len());
    let mut batch_sizes = Vec::with_capacity(batches.len());

    for batch in batches {
        let offset = writer.stream_position().map_err(|e| IoError::Io(e))?;
        let batch_bytes = serialize_record_batch(batch)?;
        write_message(writer, 0x02, &batch_bytes)?;
        let end = writer.stream_position().map_err(|e| IoError::Io(e))?;
        batch_offsets.push(offset);
        batch_sizes.push(end - offset);
    }

    // Write footer
    let footer_offset = writer.stream_position().map_err(|e| IoError::Io(e))?;

    let footer_bytes = serialize_footer(schema, &batch_offsets, &batch_sizes)?;
    writer
        .write_all(&footer_bytes)
        .map_err(|e| IoError::Io(e))?;

    // Footer size (4 bytes)
    let footer_size =
        (writer.stream_position().map_err(|e| IoError::Io(e))? - footer_offset) as u32;
    writer
        .write_u32::<LittleEndian>(footer_size)
        .map_err(|e| IoError::Io(e))?;

    // Trailing magic
    writer.write_all(ARROW_MAGIC).map_err(|e| IoError::Io(e))?;

    Ok(())
}

/// Write Arrow IPC to a file path
pub fn write_arrow_ipc_file<P: AsRef<Path>>(
    path: P,
    schema: &ArrowSchema,
    batches: &[RecordBatch],
) -> Result<()> {
    let file = std::fs::File::create(path.as_ref()).map_err(|e| {
        IoError::FileError(format!(
            "Cannot create file '{}': {e}",
            path.as_ref().display()
        ))
    })?;
    let mut writer = std::io::BufWriter::new(file);
    write_arrow_ipc(&mut writer, schema, batches)
}

/// Write Arrow IPC to a byte vector
pub fn write_arrow_ipc_bytes(schema: &ArrowSchema, batches: &[RecordBatch]) -> Result<Vec<u8>> {
    let mut cursor = Cursor::new(Vec::new());
    write_arrow_ipc(&mut cursor, schema, batches)?;
    Ok(cursor.into_inner())
}

// =====================================================================
// File format reader
// =====================================================================

/// Read Arrow IPC file format from a reader
pub fn read_arrow_ipc<R: Read + Seek>(reader: &mut R) -> Result<(ArrowSchema, Vec<RecordBatch>)> {
    // Read and verify leading magic
    let mut magic = [0u8; 6];
    reader
        .read_exact(&mut magic)
        .map_err(|e| IoError::FormatError(format!("Failed to read Arrow magic: {e}")))?;
    if &magic != ARROW_MAGIC {
        return Err(IoError::FormatError(
            "Not a valid Arrow IPC file: magic mismatch".to_string(),
        ));
    }
    // Skip padding
    let mut pad = [0u8; 2];
    reader
        .read_exact(&mut pad)
        .map_err(|e| IoError::FormatError(format!("Failed to read padding: {e}")))?;

    // Read trailing magic to verify
    reader.seek(SeekFrom::End(-6)).map_err(|e| IoError::Io(e))?;
    let mut trail_magic = [0u8; 6];
    reader
        .read_exact(&mut trail_magic)
        .map_err(|e| IoError::FormatError(format!("Failed to read trailing magic: {e}")))?;
    if &trail_magic != ARROW_MAGIC {
        return Err(IoError::FormatError(
            "Not a valid Arrow IPC file: trailing magic mismatch".to_string(),
        ));
    }

    // Read footer size (4 bytes before trailing magic)
    reader
        .seek(SeekFrom::End(-10))
        .map_err(|e| IoError::Io(e))?;
    let footer_size = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read footer size: {e}")))?
        as usize;

    // Read footer
    let footer_start = reader
        .seek(SeekFrom::End(-10 - footer_size as i64))
        .map_err(|e| IoError::Io(e))?;
    let mut footer_data = vec![0u8; footer_size];
    reader
        .read_exact(&mut footer_data)
        .map_err(|e| IoError::FormatError(format!("Failed to read footer: {e}")))?;

    let (schema, batch_offsets, batch_sizes) = deserialize_footer(&footer_data)?;

    // Read record batches
    let mut batches = Vec::with_capacity(batch_offsets.len());
    for (offset, _size) in batch_offsets.iter().zip(batch_sizes.iter()) {
        reader
            .seek(SeekFrom::Start(*offset))
            .map_err(|e| IoError::Io(e))?;
        let (_msg_type, msg_data) = read_message(reader)?;
        let batch = deserialize_record_batch(&msg_data, &schema)?;
        batches.push(batch);
    }

    Ok((schema, batches))
}

/// Read Arrow IPC from a file path
pub fn read_arrow_ipc_file<P: AsRef<Path>>(path: P) -> Result<(ArrowSchema, Vec<RecordBatch>)> {
    let file = std::fs::File::open(path.as_ref()).map_err(|e| {
        IoError::FileError(format!(
            "Cannot open file '{}': {e}",
            path.as_ref().display()
        ))
    })?;
    let mut reader = std::io::BufReader::new(file);
    read_arrow_ipc(&mut reader)
}

/// Read Arrow IPC from bytes
pub fn read_arrow_ipc_bytes(data: &[u8]) -> Result<(ArrowSchema, Vec<RecordBatch>)> {
    let mut cursor = Cursor::new(data);
    read_arrow_ipc(&mut cursor)
}

/// Read Arrow IPC streaming format from a reader
pub fn read_arrow_ipc_stream<R: Read>(reader: &mut R) -> Result<(ArrowSchema, Vec<RecordBatch>)> {
    // First message should be schema
    let (msg_type, msg_data) = read_message(reader)?;
    if msg_type != 0x01 {
        return Err(IoError::FormatError(format!(
            "Expected schema message (0x01), got {msg_type:#x}"
        )));
    }
    let schema = deserialize_schema(&msg_data)?;

    // Read record batches until EOS
    let mut batches = Vec::new();
    loop {
        match read_message(reader) {
            Ok((0x00, _)) => break,
            Ok((_, ref d)) if d.is_empty() => break,
            Ok((0x02, msg_data)) => {
                let batch = deserialize_record_batch(&msg_data, &schema)?;
                batches.push(batch);
            }
            Ok((t, _)) => {
                return Err(IoError::FormatError(format!(
                    "Unexpected message type {t:#x} in stream"
                )));
            }
            Err(_) => break, // EOF
        }
    }

    Ok((schema, batches))
}

// =====================================================================
// Internal serialization helpers
// =====================================================================

/// Write a length-prefixed message
fn write_message<W: Write>(writer: &mut W, msg_type: u8, data: &[u8]) -> Result<()> {
    // Message header: type (1 byte) + padding (3 bytes) + data length (4 bytes)
    writer.write_u8(msg_type).map_err(|e| IoError::Io(e))?;
    writer.write_all(&[0u8; 3]).map_err(|e| IoError::Io(e))?;
    writer
        .write_u32::<LittleEndian>(data.len() as u32)
        .map_err(|e| IoError::Io(e))?;
    writer.write_all(data).map_err(|e| IoError::Io(e))?;

    // Pad to alignment
    let remainder = data.len() % ALIGNMENT;
    if remainder != 0 {
        let pad = ALIGNMENT - remainder;
        writer
            .write_all(&vec![0u8; pad])
            .map_err(|e| IoError::Io(e))?;
    }

    Ok(())
}

/// Read a length-prefixed message
fn read_message<R: Read>(reader: &mut R) -> Result<(u8, Vec<u8>)> {
    let msg_type = reader
        .read_u8()
        .map_err(|e| IoError::FormatError(format!("Failed to read message type: {e}")))?;
    let mut pad = [0u8; 3];
    reader
        .read_exact(&mut pad)
        .map_err(|e| IoError::FormatError(format!("Failed to read message padding: {e}")))?;
    let data_len = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read message length: {e}")))?
        as usize;

    if data_len == 0 && msg_type == 0 {
        return Ok((0, Vec::new()));
    }

    let mut data = vec![0u8; data_len];
    reader
        .read_exact(&mut data)
        .map_err(|e| IoError::FormatError(format!("Failed to read message data: {e}")))?;

    // Skip alignment padding
    let remainder = data_len % ALIGNMENT;
    if remainder != 0 {
        let pad_size = ALIGNMENT - remainder;
        let mut skip = vec![0u8; pad_size];
        let _ = reader.read_exact(&mut skip);
    }

    Ok((msg_type, data))
}

/// Serialize an ArrowSchema
fn serialize_schema(schema: &ArrowSchema) -> Result<Vec<u8>> {
    let mut buf = Vec::new();

    // Number of fields (4 bytes)
    buf.write_u32::<LittleEndian>(schema.fields.len() as u32)
        .map_err(|e| IoError::Io(e))?;

    for field in &schema.fields {
        // Field name length (4 bytes) + name bytes
        let name_bytes = field.name.as_bytes();
        buf.write_u32::<LittleEndian>(name_bytes.len() as u32)
            .map_err(|e| IoError::Io(e))?;
        buf.write_all(name_bytes).map_err(|e| IoError::Io(e))?;

        // Data type tag (1 byte)
        buf.write_u8(field.dtype.tag())
            .map_err(|e| IoError::Io(e))?;

        // Nullable flag (1 byte)
        buf.write_u8(if field.nullable { 1 } else { 0 })
            .map_err(|e| IoError::Io(e))?;

        // Metadata count (4 bytes)
        buf.write_u32::<LittleEndian>(field.metadata.len() as u32)
            .map_err(|e| IoError::Io(e))?;
        for (k, v) in &field.metadata {
            let kb = k.as_bytes();
            let vb = v.as_bytes();
            buf.write_u32::<LittleEndian>(kb.len() as u32)
                .map_err(|e| IoError::Io(e))?;
            buf.write_all(kb).map_err(|e| IoError::Io(e))?;
            buf.write_u32::<LittleEndian>(vb.len() as u32)
                .map_err(|e| IoError::Io(e))?;
            buf.write_all(vb).map_err(|e| IoError::Io(e))?;
        }
    }

    // Schema-level metadata
    buf.write_u32::<LittleEndian>(schema.metadata.len() as u32)
        .map_err(|e| IoError::Io(e))?;
    for (k, v) in &schema.metadata {
        let kb = k.as_bytes();
        let vb = v.as_bytes();
        buf.write_u32::<LittleEndian>(kb.len() as u32)
            .map_err(|e| IoError::Io(e))?;
        buf.write_all(kb).map_err(|e| IoError::Io(e))?;
        buf.write_u32::<LittleEndian>(vb.len() as u32)
            .map_err(|e| IoError::Io(e))?;
        buf.write_all(vb).map_err(|e| IoError::Io(e))?;
    }

    Ok(buf)
}

/// Deserialize an ArrowSchema
fn deserialize_schema(data: &[u8]) -> Result<ArrowSchema> {
    let mut cur = Cursor::new(data);

    let num_fields = cur
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read field count: {e}")))?
        as usize;

    let mut fields = Vec::with_capacity(num_fields);

    for _ in 0..num_fields {
        let name_len = cur
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read field name len: {e}")))?
            as usize;
        let mut name_bytes = vec![0u8; name_len];
        cur.read_exact(&mut name_bytes)
            .map_err(|e| IoError::FormatError(format!("Failed to read field name: {e}")))?;
        let name = String::from_utf8_lossy(&name_bytes).to_string();

        let type_tag = cur
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read type tag: {e}")))?;
        let dtype = ArrowDataType::from_tag(type_tag)?;

        let nullable = cur
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read nullable: {e}")))?
            != 0;

        let meta_count = cur
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read meta count: {e}")))?
            as usize;

        let mut metadata = HashMap::new();
        for _ in 0..meta_count {
            let kl = cur
                .read_u32::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read meta key len: {e}")))?
                as usize;
            let mut kb = vec![0u8; kl];
            cur.read_exact(&mut kb)
                .map_err(|e| IoError::FormatError(format!("Failed to read meta key: {e}")))?;
            let vl = cur
                .read_u32::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read meta val len: {e}")))?
                as usize;
            let mut vb = vec![0u8; vl];
            cur.read_exact(&mut vb)
                .map_err(|e| IoError::FormatError(format!("Failed to read meta val: {e}")))?;
            metadata.insert(
                String::from_utf8_lossy(&kb).to_string(),
                String::from_utf8_lossy(&vb).to_string(),
            );
        }

        fields.push(ArrowField {
            name,
            dtype,
            nullable,
            metadata,
        });
    }

    let schema_meta_count = cur
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read schema meta count: {e}")))?
        as usize;

    let mut metadata = HashMap::new();
    for _ in 0..schema_meta_count {
        let kl = cur
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read meta key len: {e}")))?
            as usize;
        let mut kb = vec![0u8; kl];
        cur.read_exact(&mut kb)
            .map_err(|e| IoError::FormatError(format!("Failed to read meta key: {e}")))?;
        let vl = cur
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read meta val len: {e}")))?
            as usize;
        let mut vb = vec![0u8; vl];
        cur.read_exact(&mut vb)
            .map_err(|e| IoError::FormatError(format!("Failed to read meta val: {e}")))?;
        metadata.insert(
            String::from_utf8_lossy(&kb).to_string(),
            String::from_utf8_lossy(&vb).to_string(),
        );
    }

    Ok(ArrowSchema { fields, metadata })
}

/// Serialize a record batch (column data)
fn serialize_record_batch(batch: &RecordBatch) -> Result<Vec<u8>> {
    let mut buf = Vec::new();

    // Number of rows (8 bytes)
    buf.write_u64::<LittleEndian>(batch.num_rows() as u64)
        .map_err(|e| IoError::Io(e))?;

    // Number of columns (4 bytes)
    buf.write_u32::<LittleEndian>(batch.num_columns() as u32)
        .map_err(|e| IoError::Io(e))?;

    for col in &batch.columns {
        // Column type tag (1 byte)
        buf.write_u8(col.data_type().tag())
            .map_err(|e| IoError::Io(e))?;

        match col {
            ArrowColumn::Int32(values) => {
                buf.write_u64::<LittleEndian>((values.len() * 4) as u64)
                    .map_err(|e| IoError::Io(e))?;
                for v in values {
                    buf.write_i32::<LittleEndian>(*v)
                        .map_err(|e| IoError::Io(e))?;
                }
            }
            ArrowColumn::Int64(values) => {
                buf.write_u64::<LittleEndian>((values.len() * 8) as u64)
                    .map_err(|e| IoError::Io(e))?;
                for v in values {
                    buf.write_i64::<LittleEndian>(*v)
                        .map_err(|e| IoError::Io(e))?;
                }
            }
            ArrowColumn::Float32(values) => {
                buf.write_u64::<LittleEndian>((values.len() * 4) as u64)
                    .map_err(|e| IoError::Io(e))?;
                for v in values {
                    buf.write_f32::<LittleEndian>(*v)
                        .map_err(|e| IoError::Io(e))?;
                }
            }
            ArrowColumn::Float64(values) => {
                buf.write_u64::<LittleEndian>((values.len() * 8) as u64)
                    .map_err(|e| IoError::Io(e))?;
                for v in values {
                    buf.write_f64::<LittleEndian>(*v)
                        .map_err(|e| IoError::Io(e))?;
                }
            }
            ArrowColumn::Utf8(values) => {
                // First compute total bytes
                let total: usize = values.iter().map(|s| 4 + s.len()).sum();
                buf.write_u64::<LittleEndian>(total as u64)
                    .map_err(|e| IoError::Io(e))?;
                for s in values {
                    let sb = s.as_bytes();
                    buf.write_u32::<LittleEndian>(sb.len() as u32)
                        .map_err(|e| IoError::Io(e))?;
                    buf.write_all(sb).map_err(|e| IoError::Io(e))?;
                }
            }
            ArrowColumn::Boolean(values) => {
                // Bit-pack: 1 bit per boolean, ceil(n/8) bytes
                let byte_count = (values.len() + 7) / 8;
                buf.write_u64::<LittleEndian>(byte_count as u64)
                    .map_err(|e| IoError::Io(e))?;
                let mut packed = vec![0u8; byte_count];
                for (i, &v) in values.iter().enumerate() {
                    if v {
                        packed[i / 8] |= 1 << (i % 8);
                    }
                }
                buf.write_all(&packed).map_err(|e| IoError::Io(e))?;
            }
        }
    }

    Ok(buf)
}

/// Deserialize a record batch
fn deserialize_record_batch(data: &[u8], schema: &ArrowSchema) -> Result<RecordBatch> {
    let mut cur = Cursor::new(data);

    let num_rows = cur
        .read_u64::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read row count: {e}")))?
        as usize;

    let num_columns = cur
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read column count: {e}")))?
        as usize;

    if num_columns != schema.fields.len() {
        return Err(IoError::FormatError(format!(
            "Column count mismatch: got {num_columns}, schema has {}",
            schema.fields.len()
        )));
    }

    let mut columns = Vec::with_capacity(num_columns);

    for _ in 0..num_columns {
        let type_tag = cur
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read col type: {e}")))?;
        let dtype = ArrowDataType::from_tag(type_tag)?;

        let data_size = cur
            .read_u64::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read col data size: {e}")))?
            as usize;

        let col =
            match dtype {
                ArrowDataType::Int32 => {
                    let count = data_size / 4;
                    let mut values = Vec::with_capacity(count);
                    for _ in 0..count {
                        values.push(cur.read_i32::<LittleEndian>().map_err(|e| {
                            IoError::FormatError(format!("Failed to read i32: {e}"))
                        })?);
                    }
                    ArrowColumn::Int32(values)
                }
                ArrowDataType::Int64 => {
                    let count = data_size / 8;
                    let mut values = Vec::with_capacity(count);
                    for _ in 0..count {
                        values.push(cur.read_i64::<LittleEndian>().map_err(|e| {
                            IoError::FormatError(format!("Failed to read i64: {e}"))
                        })?);
                    }
                    ArrowColumn::Int64(values)
                }
                ArrowDataType::Float32 => {
                    let count = data_size / 4;
                    let mut values = Vec::with_capacity(count);
                    for _ in 0..count {
                        values.push(cur.read_f32::<LittleEndian>().map_err(|e| {
                            IoError::FormatError(format!("Failed to read f32: {e}"))
                        })?);
                    }
                    ArrowColumn::Float32(values)
                }
                ArrowDataType::Float64 => {
                    let count = data_size / 8;
                    let mut values = Vec::with_capacity(count);
                    for _ in 0..count {
                        values.push(cur.read_f64::<LittleEndian>().map_err(|e| {
                            IoError::FormatError(format!("Failed to read f64: {e}"))
                        })?);
                    }
                    ArrowColumn::Float64(values)
                }
                ArrowDataType::Utf8 => {
                    let start = cur.position() as usize;
                    let end = start + data_size;
                    let mut values = Vec::with_capacity(num_rows);
                    while (cur.position() as usize) < end {
                        let slen = cur.read_u32::<LittleEndian>().map_err(|e| {
                            IoError::FormatError(format!("Failed to read str len: {e}"))
                        })? as usize;
                        let mut sbytes = vec![0u8; slen];
                        cur.read_exact(&mut sbytes).map_err(|e| {
                            IoError::FormatError(format!("Failed to read str data: {e}"))
                        })?;
                        values.push(String::from_utf8_lossy(&sbytes).to_string());
                    }
                    ArrowColumn::Utf8(values)
                }
                ArrowDataType::Boolean => {
                    let mut packed = vec![0u8; data_size];
                    cur.read_exact(&mut packed).map_err(|e| {
                        IoError::FormatError(format!("Failed to read bool data: {e}"))
                    })?;
                    let mut values = Vec::with_capacity(num_rows);
                    for i in 0..num_rows {
                        let byte_idx = i / 8;
                        let bit_idx = i % 8;
                        if byte_idx < packed.len() {
                            values.push(packed[byte_idx] & (1 << bit_idx) != 0);
                        } else {
                            values.push(false);
                        }
                    }
                    ArrowColumn::Boolean(values)
                }
            };

        columns.push(col);
    }

    RecordBatch::new(schema.clone(), columns)
}

/// Serialize footer (schema + batch locations)
fn serialize_footer(schema: &ArrowSchema, offsets: &[u64], sizes: &[u64]) -> Result<Vec<u8>> {
    let mut buf = Vec::new();

    // Schema
    let schema_bytes = serialize_schema(schema)?;
    buf.write_u32::<LittleEndian>(schema_bytes.len() as u32)
        .map_err(|e| IoError::Io(e))?;
    buf.write_all(&schema_bytes).map_err(|e| IoError::Io(e))?;

    // Number of record batches
    buf.write_u32::<LittleEndian>(offsets.len() as u32)
        .map_err(|e| IoError::Io(e))?;

    for (&offset, &size) in offsets.iter().zip(sizes.iter()) {
        buf.write_u64::<LittleEndian>(offset)
            .map_err(|e| IoError::Io(e))?;
        buf.write_u64::<LittleEndian>(size)
            .map_err(|e| IoError::Io(e))?;
    }

    Ok(buf)
}

/// Deserialize footer
fn deserialize_footer(data: &[u8]) -> Result<(ArrowSchema, Vec<u64>, Vec<u64>)> {
    let mut cur = Cursor::new(data);

    let schema_len = cur
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read footer schema len: {e}")))?
        as usize;

    let mut schema_data = vec![0u8; schema_len];
    cur.read_exact(&mut schema_data)
        .map_err(|e| IoError::FormatError(format!("Failed to read footer schema: {e}")))?;
    let schema = deserialize_schema(&schema_data)?;

    let num_batches = cur
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read batch count: {e}")))?
        as usize;

    let mut offsets = Vec::with_capacity(num_batches);
    let mut sizes = Vec::with_capacity(num_batches);

    for _ in 0..num_batches {
        offsets.push(
            cur.read_u64::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read batch offset: {e}")))?,
        );
        sizes.push(
            cur.read_u64::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read batch size: {e}")))?,
        );
    }

    Ok((schema, offsets, sizes))
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_creation() {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("a", ArrowDataType::Int32),
            ArrowField::new("b", ArrowDataType::Float64),
        ]);
        assert_eq!(schema.num_fields(), 2);
        assert_eq!(schema.field(0).map(|f| &f.name), Some(&"a".to_string()));
    }

    #[test]
    fn test_schema_field_by_name() {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("x", ArrowDataType::Int64),
            ArrowField::new("y", ArrowDataType::Float32),
        ]);
        let (idx, field) = schema.field_by_name("y").expect("found");
        assert_eq!(idx, 1);
        assert_eq!(field.dtype, ArrowDataType::Float32);
        assert!(schema.field_by_name("z").is_none());
    }

    #[test]
    fn test_record_batch_column_mismatch() {
        let schema = ArrowSchema::new(vec![ArrowField::new("a", ArrowDataType::Int32)]);
        let result = RecordBatch::new(
            schema,
            vec![ArrowColumn::Int32(vec![1]), ArrowColumn::Float64(vec![1.0])],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_record_batch_type_mismatch() {
        let schema = ArrowSchema::new(vec![ArrowField::new("a", ArrowDataType::Int32)]);
        let result = RecordBatch::new(schema, vec![ArrowColumn::Float64(vec![1.0])]);
        assert!(result.is_err());
    }

    #[test]
    fn test_record_batch_row_count_mismatch() {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("a", ArrowDataType::Int32),
            ArrowField::new("b", ArrowDataType::Int32),
        ]);
        let result = RecordBatch::new(
            schema,
            vec![ArrowColumn::Int32(vec![1, 2]), ArrowColumn::Int32(vec![1])],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_roundtrip_int32() {
        let schema = ArrowSchema::new(vec![ArrowField::new("values", ArrowDataType::Int32)]);
        let batch = RecordBatch::new(schema.clone(), vec![ArrowColumn::Int32(vec![10, -20, 30])])
            .expect("valid");

        let bytes = write_arrow_ipc_bytes(&schema, &[batch]).expect("write");
        let (read_schema, batches) = read_arrow_ipc_bytes(&bytes).expect("read");

        assert_eq!(read_schema.fields.len(), 1);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);

        if let ArrowColumn::Int32(vals) = batches[0].column(0).expect("col") {
            assert_eq!(vals, &[10, -20, 30]);
        } else {
            panic!("Expected Int32");
        }
    }

    #[test]
    fn test_roundtrip_float64() {
        let schema = ArrowSchema::new(vec![ArrowField::new("x", ArrowDataType::Float64)]);
        let batch = RecordBatch::new(
            schema.clone(),
            vec![ArrowColumn::Float64(vec![1.1, 2.2, 3.3])],
        )
        .expect("valid");

        let bytes = write_arrow_ipc_bytes(&schema, &[batch]).expect("write");
        let (_, batches) = read_arrow_ipc_bytes(&bytes).expect("read");

        if let ArrowColumn::Float64(vals) = batches[0].column(0).expect("col") {
            assert!((vals[0] - 1.1).abs() < 1e-10);
            assert!((vals[1] - 2.2).abs() < 1e-10);
            assert!((vals[2] - 3.3).abs() < 1e-10);
        } else {
            panic!("Expected Float64");
        }
    }

    #[test]
    fn test_roundtrip_utf8() {
        let schema = ArrowSchema::new(vec![ArrowField::new("name", ArrowDataType::Utf8)]);
        let batch = RecordBatch::new(
            schema.clone(),
            vec![ArrowColumn::Utf8(vec![
                "hello".to_string(),
                "world".to_string(),
            ])],
        )
        .expect("valid");

        let bytes = write_arrow_ipc_bytes(&schema, &[batch]).expect("write");
        let (_, batches) = read_arrow_ipc_bytes(&bytes).expect("read");

        if let ArrowColumn::Utf8(vals) = batches[0].column(0).expect("col") {
            assert_eq!(vals, &["hello", "world"]);
        } else {
            panic!("Expected Utf8");
        }
    }

    #[test]
    fn test_roundtrip_boolean() {
        let schema = ArrowSchema::new(vec![ArrowField::new("flag", ArrowDataType::Boolean)]);
        let batch = RecordBatch::new(
            schema.clone(),
            vec![ArrowColumn::Boolean(vec![true, false, true, true, false])],
        )
        .expect("valid");

        let bytes = write_arrow_ipc_bytes(&schema, &[batch]).expect("write");
        let (_, batches) = read_arrow_ipc_bytes(&bytes).expect("read");

        if let ArrowColumn::Boolean(vals) = batches[0].column(0).expect("col") {
            assert_eq!(vals, &[true, false, true, true, false]);
        } else {
            panic!("Expected Boolean");
        }
    }

    #[test]
    fn test_roundtrip_multiple_columns() {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("id", ArrowDataType::Int32),
            ArrowField::new("value", ArrowDataType::Float64),
            ArrowField::new("label", ArrowDataType::Utf8),
            ArrowField::new("active", ArrowDataType::Boolean),
        ]);
        let batch = RecordBatch::new(
            schema.clone(),
            vec![
                ArrowColumn::Int32(vec![1, 2]),
                ArrowColumn::Float64(vec![std::f64::consts::PI, std::f64::consts::E]),
                ArrowColumn::Utf8(vec!["pi".to_string(), "e".to_string()]),
                ArrowColumn::Boolean(vec![true, false]),
            ],
        )
        .expect("valid");

        let bytes = write_arrow_ipc_bytes(&schema, &[batch]).expect("write");
        let (read_schema, batches) = read_arrow_ipc_bytes(&bytes).expect("read");

        assert_eq!(read_schema.num_fields(), 4);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 2);
    }

    #[test]
    fn test_roundtrip_multiple_batches() {
        let schema = ArrowSchema::new(vec![ArrowField::new("x", ArrowDataType::Int64)]);
        let b1 = RecordBatch::new(schema.clone(), vec![ArrowColumn::Int64(vec![1, 2, 3])])
            .expect("valid");
        let b2 =
            RecordBatch::new(schema.clone(), vec![ArrowColumn::Int64(vec![4, 5])]).expect("valid");

        let bytes = write_arrow_ipc_bytes(&schema, &[b1, b2]).expect("write");
        let (_, batches) = read_arrow_ipc_bytes(&bytes).expect("read");

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), 3);
        assert_eq!(batches[1].num_rows(), 2);
    }

    #[test]
    fn test_roundtrip_empty_batch() {
        let schema = ArrowSchema::new(vec![ArrowField::new("x", ArrowDataType::Int32)]);
        let batch =
            RecordBatch::new(schema.clone(), vec![ArrowColumn::Int32(Vec::new())]).expect("valid");

        let bytes = write_arrow_ipc_bytes(&schema, &[batch]).expect("write");
        let (_, batches) = read_arrow_ipc_bytes(&bytes).expect("read");

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 0);
    }

    #[test]
    fn test_invalid_magic() {
        let data = b"NOT_ARROW_DATA_HERE";
        let result = read_arrow_ipc_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_column_as_f64() {
        let col = ArrowColumn::Int32(vec![1, 2, 3]);
        let f = col.as_f64().expect("numeric");
        assert_eq!(f, vec![1.0, 2.0, 3.0]);

        let col = ArrowColumn::Utf8(vec!["a".to_string()]);
        assert!(col.as_f64().is_none());

        let col = ArrowColumn::Boolean(vec![true]);
        assert!(col.as_f64().is_none());
    }

    #[test]
    fn test_column_data_type() {
        assert_eq!(ArrowColumn::Int32(vec![]).data_type(), ArrowDataType::Int32);
        assert_eq!(ArrowColumn::Int64(vec![]).data_type(), ArrowDataType::Int64);
        assert_eq!(
            ArrowColumn::Float32(vec![]).data_type(),
            ArrowDataType::Float32
        );
        assert_eq!(
            ArrowColumn::Float64(vec![]).data_type(),
            ArrowDataType::Float64
        );
        assert_eq!(ArrowColumn::Utf8(vec![]).data_type(), ArrowDataType::Utf8);
        assert_eq!(
            ArrowColumn::Boolean(vec![]).data_type(),
            ArrowDataType::Boolean
        );
    }

    #[test]
    fn test_schema_metadata() {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("a", ArrowDataType::Int32).with_metadata("unit", "meters")
        ])
        .with_metadata("version", "1.0");

        let batch =
            RecordBatch::new(schema.clone(), vec![ArrowColumn::Int32(vec![42])]).expect("valid");

        let bytes = write_arrow_ipc_bytes(&schema, &[batch]).expect("write");
        let (read_schema, _) = read_arrow_ipc_bytes(&bytes).expect("read");

        assert_eq!(
            read_schema.metadata.get("version"),
            Some(&"1.0".to_string())
        );
        assert_eq!(
            read_schema.fields[0].metadata.get("unit"),
            Some(&"meters".to_string())
        );
    }

    #[test]
    fn test_streaming_roundtrip() {
        let schema = ArrowSchema::new(vec![ArrowField::new("val", ArrowDataType::Float32)]);
        let b1 = RecordBatch::new(schema.clone(), vec![ArrowColumn::Float32(vec![1.0, 2.0])])
            .expect("valid");
        let b2 =
            RecordBatch::new(schema.clone(), vec![ArrowColumn::Float32(vec![3.0])]).expect("valid");

        let buf = Vec::new();
        let mut writer = ArrowStreamWriter::new(buf, schema.clone()).expect("writer");
        writer.write_batch(&b1).expect("batch1");
        writer.write_batch(&b2).expect("batch2");
        assert_eq!(writer.batches_written(), 2);
        let data = writer.finish().expect("finish");

        let mut cursor = Cursor::new(&data);
        let (read_schema, batches) = read_arrow_ipc_stream(&mut cursor).expect("read");

        assert_eq!(read_schema.num_fields(), 1);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), 2);
        assert_eq!(batches[1].num_rows(), 1);
    }

    #[test]
    fn test_column_by_name() {
        let schema = ArrowSchema::new(vec![
            ArrowField::new("a", ArrowDataType::Int32),
            ArrowField::new("b", ArrowDataType::Float64),
        ]);
        let batch = RecordBatch::new(
            schema.clone(),
            vec![ArrowColumn::Int32(vec![1]), ArrowColumn::Float64(vec![2.0])],
        )
        .expect("valid");

        let col = batch.column_by_name("b").expect("found");
        assert_eq!(col.data_type(), ArrowDataType::Float64);
        assert!(batch.column_by_name("c").is_none());
    }

    #[test]
    fn test_nullable_field() {
        let field = ArrowField::new_nullable("x", ArrowDataType::Int32);
        assert!(field.nullable);

        let field = ArrowField::new("y", ArrowDataType::Int32);
        assert!(!field.nullable);
    }

    #[test]
    fn test_float32_roundtrip() {
        let schema = ArrowSchema::new(vec![ArrowField::new("f", ArrowDataType::Float32)]);
        let batch = RecordBatch::new(
            schema.clone(),
            vec![ArrowColumn::Float32(vec![1.5f32, -2.5f32, 0.0f32])],
        )
        .expect("valid");

        let bytes = write_arrow_ipc_bytes(&schema, &[batch]).expect("write");
        let (_, batches) = read_arrow_ipc_bytes(&bytes).expect("read");

        if let ArrowColumn::Float32(vals) = batches[0].column(0).expect("col") {
            assert!((vals[0] - 1.5).abs() < 1e-6);
            assert!((vals[1] + 2.5).abs() < 1e-6);
            assert!((vals[2]).abs() < 1e-6);
        } else {
            panic!("Expected Float32");
        }
    }

    #[test]
    fn test_int64_roundtrip() {
        let schema = ArrowSchema::new(vec![ArrowField::new("big", ArrowDataType::Int64)]);
        let batch = RecordBatch::new(
            schema.clone(),
            vec![ArrowColumn::Int64(vec![i64::MAX, i64::MIN, 0])],
        )
        .expect("valid");

        let bytes = write_arrow_ipc_bytes(&schema, &[batch]).expect("write");
        let (_, batches) = read_arrow_ipc_bytes(&bytes).expect("read");

        if let ArrowColumn::Int64(vals) = batches[0].column(0).expect("col") {
            assert_eq!(vals[0], i64::MAX);
            assert_eq!(vals[1], i64::MIN);
            assert_eq!(vals[2], 0);
        } else {
            panic!("Expected Int64");
        }
    }

    #[test]
    fn test_file_io_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("arrow_ipc_test.arrow");

        let schema = ArrowSchema::new(vec![ArrowField::new("x", ArrowDataType::Int32)]);
        let batch = RecordBatch::new(schema.clone(), vec![ArrowColumn::Int32(vec![100, 200])])
            .expect("valid");

        write_arrow_ipc_file(&path, &schema, &[batch]).expect("write");
        let (read_schema, batches) = read_arrow_ipc_file(&path).expect("read");

        assert_eq!(read_schema.num_fields(), 1);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 2);

        let _ = std::fs::remove_file(&path);
    }
}
