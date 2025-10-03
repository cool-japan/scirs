//! MATLAB writing implementation
//!
//! This module provides a simplified implementation of MAT file writing functionality
//! that works with the existing MATLAB module structure.

use crate::error::{IoError, Result};
use crate::matlab::{
    MatType, MI_INT32, MI_INT8, MI_MATRIX, MI_UINT32, MX_CHAR_CLASS, MX_DOUBLE_CLASS,
    MX_INT32_CLASS, MX_SINGLE_CLASS, MX_UINT8_CLASS,
};
use byteorder::{LittleEndian, WriteBytesExt};
use scirs2_core::ndarray::ArrayD;
use std::io::{Seek, Write};

// Missing constants redefined locally
const MI_UINT8: i32 = 2;
#[allow(dead_code)]
const MI_INT16: i32 = 3;
const MI_UINT16: i32 = 4;
const MI_SINGLE: i32 = 7;
const MI_DOUBLE: i32 = 9;
#[allow(dead_code)]
const MI_INT64: i32 = 12;
#[allow(dead_code)]
const MI_UINT64: i32 = 13;

/// Write MAT file header (simplified version)
#[allow(dead_code)]
pub fn write_mat_header<W: Write>(writer: &mut W) -> Result<()> {
    // Write "MATLAB" magic string
    writer
        .write_all(b"MATLAB")
        .map_err(|e| IoError::FileError(format!("Failed to write header: {e}")))?;

    // Write descriptive text - MATLAB header is exactly 128 bytes total
    let description = b" 5.0 MAT-file, Created by: scirs2-io library";
    let description_len = description.len();

    // Write the description
    writer
        .write_all(description)
        .map_err(|e| IoError::FileError(format!("Failed to write description: {e}")))?;

    // Calculate padding needed to reach position 124 (subsystem data offset at 124-128)
    let headertext_target = 124;
    let already_written = 6 + description_len; // "MATLAB" (6 bytes) + description
    let padding_needed = headertext_target - already_written;

    // Write padding (spaces or nulls)
    if padding_needed > 0 {
        let padding = vec![0u8; padding_needed];
        writer
            .write_all(&padding)
            .map_err(|e| IoError::FileError(format!("Failed to write padding: {e}")))?;
    }

    // Write subsystem data offset (4 bytes: version + endianness at positions 124-128)
    writer
        .write_u16::<LittleEndian>(0x0100) // Version
        .map_err(|e| IoError::FileError(format!("Failed to write version: {e}")))?;
    writer
        .write_u16::<LittleEndian>(0x4D49) // Endianness indicator "MI"
        .map_err(|e| IoError::FileError(format!("Failed to write endianness: {e}")))?;

    Ok(())
}

/// Write a variable (simplified version that only handles basic types)
#[allow(dead_code)]
pub fn write_variable<W: Write + Seek>(
    writer: &mut W,
    name: &str,
    mat_type: &MatType,
) -> Result<()> {
    // Write matrix element header (MI_MATRIX)
    writer
        .write_i32::<LittleEndian>(MI_MATRIX)
        .map_err(|e| IoError::FileError(format!("Failed to write matrix type: {e}")))?;

    // Placeholder for size - we'll calculate and update this at the end
    let size_pos = writer
        .stream_position()
        .map_err(|e| IoError::FileError(format!("Failed to get size position: {e}")))?;
    writer
        .write_i32::<LittleEndian>(0)
        .map_err(|e| IoError::FileError(format!("Failed to write size placeholder: {e}")))?;

    let data_start = writer
        .stream_position()
        .map_err(|e| IoError::FileError(format!("Failed to get data start: {e}")))?;

    match mat_type {
        MatType::Double(array) => {
            write_matrix_header_content(writer, name, array.shape(), MX_DOUBLE_CLASS, false)?;
            write_double_data(writer, array)?;
        }
        MatType::Single(array) => {
            write_matrix_header_content(writer, name, array.shape(), MX_SINGLE_CLASS, false)?;
            write_single_data(writer, array)?;
        }
        MatType::Int32(array) => {
            write_matrix_header_content(writer, name, array.shape(), MX_INT32_CLASS, false)?;
            write_int32_data(writer, array)?;
        }
        MatType::Logical(array) => {
            write_matrix_header_content(writer, name, array.shape(), MX_UINT8_CLASS, true)?;
            write_logical_data(writer, array)?;
        }
        MatType::Char(string) => {
            write_char_data_content(writer, name, string)?;
        }
        _ => {
            return Err(IoError::Other(format!(
                "Data _type not yet supported in simplified writer: {:?}",
                std::any::type_name::<MatType>()
            )));
        }
    }

    // Calculate and write the actual matrix size
    let data_end = writer
        .stream_position()
        .map_err(|e| IoError::FileError(format!("Failed to get data end: {e}")))?;
    let total_size = (data_end - data_start) as i32;

    // Go back and write the actual size
    let current_pos = data_end;
    writer
        .seek(std::io::SeekFrom::Start(size_pos))
        .map_err(|e| IoError::FileError(format!("Failed to seek to size: {e}")))?;
    writer
        .write_i32::<LittleEndian>(total_size)
        .map_err(|e| IoError::FileError(format!("Failed to write actual size: {e}")))?;

    // Return to end of data
    writer
        .seek(std::io::SeekFrom::Start(current_pos))
        .map_err(|e| IoError::FileError(format!("Failed to seek back: {e}")))?;

    Ok(())
}

/// Write matrix header content (flags, dimensions, name) without the outer matrix element header
#[allow(dead_code)]
fn write_matrix_header_content<W: Write + Seek>(
    writer: &mut W,
    name: &str,
    shape: &[usize],
    class_type: i32,
    is_logical: bool,
) -> Result<()> {
    // Write array flags
    writer
        .write_i32::<LittleEndian>(MI_UINT32)
        .map_err(|e| IoError::FileError(format!("Failed to write flags type: {}", e)))?;
    writer
        .write_i32::<LittleEndian>(8)
        .map_err(|e| IoError::FileError(format!("Failed to write flags size: {}", e)))?;

    let mut flags = class_type as u32;
    if is_logical {
        flags |= 0x200;
    }
    writer
        .write_u32::<LittleEndian>(flags)
        .map_err(|e| IoError::FileError(format!("Failed to write flags: {}", e)))?;
    writer
        .write_u32::<LittleEndian>(0) // Reserved
        .map_err(|e| IoError::FileError(format!("Failed to write reserved: {}", e)))?;

    // Write dimensions
    let dims_size = (shape.len() * 4) as i32;
    writer
        .write_i32::<LittleEndian>(MI_INT32)
        .map_err(|e| IoError::FileError(format!("Failed to write dims type: {}", e)))?;
    writer
        .write_i32::<LittleEndian>(dims_size)
        .map_err(|e| IoError::FileError(format!("Failed to write dims size: {}", e)))?;

    // Write dimensions in reverse order (MATLAB column-major)
    for &dim in shape.iter().rev() {
        writer
            .write_i32::<LittleEndian>(dim as i32)
            .map_err(|e| IoError::FileError(format!("Failed to write dimension: {}", e)))?;
    }

    // Pad to 8-byte boundary
    let padding = (8 - (dims_size % 8)) % 8;
    if padding > 0 {
        writer
            .write_all(&vec![0u8; padding as usize])
            .map_err(|e| IoError::FileError(format!("Failed to write padding: {}", e)))?;
    }

    // Write array name
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len() as i32;
    writer
        .write_i32::<LittleEndian>(MI_INT8)
        .map_err(|e| IoError::FileError(format!("Failed to write name type: {}", e)))?;
    writer
        .write_i32::<LittleEndian>(name_len)
        .map_err(|e| IoError::FileError(format!("Failed to write name size: {}", e)))?;
    writer
        .write_all(name_bytes)
        .map_err(|e| IoError::FileError(format!("Failed to write name: {}", e)))?;

    // Pad name to 8-byte boundary
    let name_padding = (8 - (name_len % 8)) % 8;
    if name_padding > 0 {
        writer
            .write_all(&vec![0u8; name_padding as usize])
            .map_err(|e| IoError::FileError(format!("Failed to write name padding: {}", e)))?;
    }

    Ok(())
}

/// Write double precision data
#[allow(dead_code)]
fn write_double_data<W: Write>(writer: &mut W, array: &ArrayD<f64>) -> Result<()> {
    let data_size = (array.len() * 8) as i32;
    writer
        .write_i32::<LittleEndian>(MI_DOUBLE)
        .map_err(|e| IoError::FileError(format!("Failed to write double type: {}", e)))?;
    writer
        .write_i32::<LittleEndian>(data_size)
        .map_err(|e| IoError::FileError(format!("Failed to write double size: {}", e)))?;

    for &value in array.iter() {
        writer
            .write_f64::<LittleEndian>(value)
            .map_err(|e| IoError::FileError(format!("Failed to write double value: {}", e)))?;
    }

    // Pad to 8-byte boundary
    let padding = (8 - (data_size % 8)) % 8;
    if padding > 0 {
        writer
            .write_all(&vec![0u8; padding as usize])
            .map_err(|e| IoError::FileError(format!("Failed to write double padding: {}", e)))?;
    }

    Ok(())
}

/// Write single precision data
#[allow(dead_code)]
fn write_single_data<W: Write>(writer: &mut W, array: &ArrayD<f32>) -> Result<()> {
    let data_size = (array.len() * 4) as i32;
    writer
        .write_i32::<LittleEndian>(MI_SINGLE)
        .map_err(|e| IoError::FileError(format!("Failed to write single type: {}", e)))?;
    writer
        .write_i32::<LittleEndian>(data_size)
        .map_err(|e| IoError::FileError(format!("Failed to write single size: {}", e)))?;

    for &value in array.iter() {
        writer
            .write_f32::<LittleEndian>(value)
            .map_err(|e| IoError::FileError(format!("Failed to write single value: {}", e)))?;
    }

    // Pad to 8-byte boundary
    let padding = (8 - (data_size % 8)) % 8;
    if padding > 0 {
        writer
            .write_all(&vec![0u8; padding as usize])
            .map_err(|e| IoError::FileError(format!("Failed to write single padding: {}", e)))?;
    }

    Ok(())
}

/// Write int32 data
#[allow(dead_code)]
fn write_int32_data<W: Write>(writer: &mut W, array: &ArrayD<i32>) -> Result<()> {
    let data_size = (array.len() * 4) as i32;
    writer
        .write_i32::<LittleEndian>(MI_INT32)
        .map_err(|e| IoError::FileError(format!("Failed to write int32 type: {}", e)))?;
    writer
        .write_i32::<LittleEndian>(data_size)
        .map_err(|e| IoError::FileError(format!("Failed to write int32 size: {}", e)))?;

    for &value in array.iter() {
        writer
            .write_i32::<LittleEndian>(value)
            .map_err(|e| IoError::FileError(format!("Failed to write int32 value: {}", e)))?;
    }

    // Pad to 8-byte boundary
    let padding = (8 - (data_size % 8)) % 8;
    if padding > 0 {
        writer
            .write_all(&vec![0u8; padding as usize])
            .map_err(|e| IoError::FileError(format!("Failed to write int32 padding: {}", e)))?;
    }

    Ok(())
}

/// Write logical data (as uint8)
#[allow(dead_code)]
fn write_logical_data<W: Write>(writer: &mut W, array: &ArrayD<bool>) -> Result<()> {
    let data_size = array.len() as i32;
    writer
        .write_i32::<LittleEndian>(MI_UINT8)
        .map_err(|e| IoError::FileError(format!("Failed to write logical type: {}", e)))?;
    writer
        .write_i32::<LittleEndian>(data_size)
        .map_err(|e| IoError::FileError(format!("Failed to write logical size: {}", e)))?;

    for &value in array.iter() {
        writer
            .write_u8(if value { 1 } else { 0 })
            .map_err(|e| IoError::FileError(format!("Failed to write logical value: {}", e)))?;
    }

    // Pad to 8-byte boundary
    let padding = (8 - (data_size % 8)) % 8;
    if padding > 0 {
        writer
            .write_all(&vec![0u8; padding as usize])
            .map_err(|e| IoError::FileError(format!("Failed to write logical padding: {}", e)))?;
    }

    Ok(())
}

/// Write character data content (without outer matrix header)
#[allow(dead_code)]
fn write_char_data_content<W: Write + Seek>(
    writer: &mut W,
    name: &str,
    string: &str,
) -> Result<()> {
    // Convert to UTF-16
    let utf16_chars: Vec<u16> = string.encode_utf16().collect();
    let shape = [1, utf16_chars.len()];

    write_matrix_header_content(writer, name, &shape, MX_CHAR_CLASS, false)?;

    let data_size = (utf16_chars.len() * 2) as i32;
    writer
        .write_i32::<LittleEndian>(MI_UINT16)
        .map_err(|e| IoError::FileError(format!("Failed to write char type: {}", e)))?;
    writer
        .write_i32::<LittleEndian>(data_size)
        .map_err(|e| IoError::FileError(format!("Failed to write char size: {}", e)))?;

    for &ch in &utf16_chars {
        writer
            .write_u16::<LittleEndian>(ch)
            .map_err(|e| IoError::FileError(format!("Failed to write char: {}", e)))?;
    }

    // Pad to 8-byte boundary
    let padding = (8 - (data_size % 8)) % 8;
    if padding > 0 {
        writer
            .write_all(&vec![0u8; padding as usize])
            .map_err(|e| IoError::FileError(format!("Failed to write char padding: {}", e)))?;
    }

    Ok(())
}
