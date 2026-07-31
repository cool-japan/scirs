//! Writer for the columnar binary format.
//!
//! File layout (format version 2):
//! - Magic bytes (8 bytes): "SCIRCOL\x01"
//! - Version (u32): format version
//! - Column count (u32)
//! - Row count (u64)
//! - For each column:
//!   - Name length (u32) + name bytes
//!   - Type tag (u8)
//!   - Encoding type (u8)
//!   - Data size in bytes (u64)
//!   - Encoded data bytes
//!   - Null-mask flag (u8): 1 if a null/validity bitmap follows, else 0
//!   - If the flag is 1: one byte per row (1 = null, 0 = valid)
//!
//! Version 1 files omit the null-mask flag/bitmap entirely; [`super::reader`]
//! treats every column in such a file as having no null mask.

use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::{IoError, Result};

use super::encoding::encode_column;
use super::types::{ColumnarTable, EncodingType, COLUMNAR_MAGIC, FORMAT_VERSION};

/// Options for writing columnar files
#[derive(Debug, Clone, Default)]
pub struct ColumnarWriteOptions {
    /// Force a specific encoding for all columns (None = auto-detect)
    pub encoding: Option<EncodingType>,
}

/// Write a columnar table to a file
pub fn write_columnar<P: AsRef<Path>>(path: P, table: &ColumnarTable) -> Result<()> {
    write_columnar_with_options(path, table, ColumnarWriteOptions::default())
}

/// Write a columnar table to a file with options
pub fn write_columnar_with_options<P: AsRef<Path>>(
    path: P,
    table: &ColumnarTable,
    options: ColumnarWriteOptions,
) -> Result<()> {
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    // Write magic bytes
    writer
        .write_all(COLUMNAR_MAGIC)
        .map_err(|e| IoError::FileError(format!("Failed to write magic: {}", e)))?;

    // Write version
    writer
        .write_u32::<LittleEndian>(FORMAT_VERSION)
        .map_err(|e| IoError::FileError(format!("Failed to write version: {}", e)))?;

    // Write column count
    writer
        .write_u32::<LittleEndian>(table.num_columns() as u32)
        .map_err(|e| IoError::FileError(format!("Failed to write column count: {}", e)))?;

    // Write row count
    writer
        .write_u64::<LittleEndian>(table.num_rows() as u64)
        .map_err(|e| IoError::FileError(format!("Failed to write row count: {}", e)))?;

    // Write each column
    for col in table.columns() {
        // Write column name
        let name_bytes = col.name.as_bytes();
        writer
            .write_u32::<LittleEndian>(name_bytes.len() as u32)
            .map_err(|e| {
                IoError::FileError(format!("Failed to write column name length: {}", e))
            })?;
        writer
            .write_all(name_bytes)
            .map_err(|e| IoError::FileError(format!("Failed to write column name: {}", e)))?;

        // Write type tag
        writer
            .write_u8(col.data.type_tag() as u8)
            .map_err(|e| IoError::FileError(format!("Failed to write type tag: {}", e)))?;

        // Determine encoding
        let encoding = options.encoding.unwrap_or_else(|| col.data.best_encoding());

        // Write encoding type
        writer
            .write_u8(encoding as u8)
            .map_err(|e| IoError::FileError(format!("Failed to write encoding type: {}", e)))?;

        // Encode data to a buffer to get the size
        let mut data_buf = Vec::new();
        encode_column(&mut data_buf, &col.data, encoding)?;

        // Write data size
        writer
            .write_u64::<LittleEndian>(data_buf.len() as u64)
            .map_err(|e| IoError::FileError(format!("Failed to write data size: {}", e)))?;

        // Write data
        writer
            .write_all(&data_buf)
            .map_err(|e| IoError::FileError(format!("Failed to write column data: {}", e)))?;

        // Write the null/validity bitmap, if any (format version 2+).
        match &col.null_mask {
            Some(mask) => {
                writer.write_u8(1).map_err(|e| {
                    IoError::FileError(format!("Failed to write null-mask flag: {}", e))
                })?;
                for &is_null in mask {
                    writer.write_u8(is_null as u8).map_err(|e| {
                        IoError::FileError(format!("Failed to write null-mask byte: {}", e))
                    })?;
                }
            }
            None => {
                writer.write_u8(0).map_err(|e| {
                    IoError::FileError(format!("Failed to write null-mask flag: {}", e))
                })?;
            }
        }
    }

    writer
        .flush()
        .map_err(|e| IoError::FileError(format!("Failed to flush writer: {}", e)))?;

    Ok(())
}
