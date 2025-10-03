//! Fortran unformatted file handling module
//!
//! This module provides functionality for reading and writing Fortran unformatted files,
//! which are commonly used in scientific computing, particularly in computational physics,
//! weather modeling, and engineering applications.

#![allow(dead_code)]
#![allow(missing_docs)]
//!
//! # Fortran Unformatted File Format
//!
//! Fortran unformatted files use a record-based structure where each record is
//! delimited by a record marker that contains the byte count of the record.
//! The format varies between compilers and platforms:
//!
//! - **Sequential access**: Each record has a 4-byte (or 8-byte) marker before and after
//! - **Direct access**: Fixed-length records without markers
//! - **Stream access**: Continuous byte stream (Fortran 2003+)
//!
//! # Examples
//!
//! ```rust,no_run
//! use scirs2_io::fortran::{FortranFile, RecordType, EndianMode};
//! use scirs2_core::ndarray::Array2;
//!
//! // Read a Fortran unformatted file
//! let mut file = FortranFile::open("data.unf")?;
//!
//! // Read a 2D array of doubles
//! let array: Array2<f64> = file.read_array_2d(100, 50)?;
//!
//! // Write to a new file
//! let mut out_file = FortranFile::create("output.unf")?;
//! out_file.write_array(&array)?;
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```

use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use scirs2_core::ndarray::{Array, Array1, Array2, Array3, ShapeBuilder};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::{IoError, Result};
use scirs2_core::numeric::ScientificNumber;

/// Endianness mode for Fortran files
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndianMode {
    /// Little-endian byte order (x86, x86_64)
    Little,
    /// Big-endian byte order (PowerPC, SPARC)
    Big,
    /// Native byte order of the current platform
    Native,
}

impl Default for EndianMode {
    fn default() -> Self {
        if cfg!(target_endian = "big") {
            EndianMode::Big
        } else {
            EndianMode::Little
        }
    }
}

/// Record marker size for Fortran unformatted files
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecordMarkerSize {
    /// 4-byte record markers (default for most compilers)
    #[default]
    FourByte,
    /// 8-byte record markers (some compilers with large record support)
    EightByte,
}

/// Fortran file access mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccessMode {
    /// Sequential access (default)
    #[default]
    Sequential,
    /// Direct access with fixed record length
    Direct { record_length: usize },
    /// Stream access (Fortran 2003+)
    Stream,
}

/// Fortran data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FortranType {
    /// INTEGER*1 (8-bit integer)
    Integer1,
    /// INTEGER*2 (16-bit integer)
    Integer2,
    /// INTEGER*4 (32-bit integer)
    Integer4,
    /// INTEGER*8 (64-bit integer)
    Integer8,
    /// REAL*4 (32-bit float)
    Real4,
    /// REAL*8 (64-bit float)
    Real8,
    /// COMPLEX*8 (two 32-bit floats)
    Complex8,
    /// COMPLEX*16 (two 64-bit floats)
    Complex16,
    /// LOGICAL*1 (8-bit boolean)
    Logical1,
    /// LOGICAL*4 (32-bit boolean)
    Logical4,
    /// CHARACTER (with length)
    Character(usize),
}

/// Type of record in a Fortran file
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordType {
    /// Fixed-length record
    Fixed(usize),
    /// Variable-length record
    Variable,
}

/// Configuration for reading/writing Fortran unformatted files
#[derive(Debug, Clone)]
pub struct FortranConfig {
    /// Endianness mode
    pub endian: EndianMode,
    /// Record marker size
    pub marker_size: RecordMarkerSize,
    /// File access mode
    pub access_mode: AccessMode,
    /// Whether to validate record markers
    pub validate_markers: bool,
}

impl Default for FortranConfig {
    fn default() -> Self {
        Self {
            endian: EndianMode::default(),
            marker_size: RecordMarkerSize::default(),
            access_mode: AccessMode::default(),
            validate_markers: true,
        }
    }
}

/// Fortran unformatted file reader/writer
pub struct FortranFile<R> {
    reader: R,
    config: FortranConfig,
    #[allow(dead_code)]
    current_record_pos: u64,
}

impl FortranFile<BufReader<File>> {
    /// Open a Fortran unformatted file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|_e| IoError::FileNotFound(path.as_ref().to_string_lossy().to_string()))?;
        let reader = BufReader::new(file);
        Ok(Self {
            reader,
            config: FortranConfig::default(),
            current_record_pos: 0,
        })
    }

    /// Open a Fortran unformatted file with custom configuration
    pub fn open_with_config<P: AsRef<Path>>(path: P, config: FortranConfig) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|_e| IoError::FileNotFound(path.as_ref().to_string_lossy().to_string()))?;
        let reader = BufReader::new(file);
        Ok(Self {
            reader,
            config,
            current_record_pos: 0,
        })
    }
}

impl FortranFile<BufWriter<File>> {
    /// Create a new Fortran unformatted file for writing
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path.as_ref())
            .map_err(|e| IoError::FileError(format!("Failed to create file: {e}")))?;
        let writer = BufWriter::new(file);
        Ok(Self {
            reader: writer,
            config: FortranConfig::default(),
            current_record_pos: 0,
        })
    }

    /// Create a new Fortran unformatted file with custom configuration
    pub fn create_with_config<P: AsRef<Path>>(path: P, config: FortranConfig) -> Result<Self> {
        let file = File::create(path.as_ref())
            .map_err(|e| IoError::FileError(format!("Failed to create file: {e}")))?;
        let writer = BufWriter::new(file);
        Ok(Self {
            reader: writer,
            config,
            current_record_pos: 0,
        })
    }
}

impl<R: Read + Seek> FortranFile<R> {
    /// Read a record marker (4 or 8 bytes depending on configuration)
    fn read_record_marker(&mut self) -> Result<usize> {
        match self.config.marker_size {
            RecordMarkerSize::FourByte => match self.config.endian {
                EndianMode::Little | EndianMode::Native if cfg!(target_endian = "little") => self
                    .reader
                    .read_u32::<LittleEndian>()
                    .map(|v| v as usize)
                    .map_err(|e| IoError::ParseError(format!("Failed to read record marker: {e}"))),
                _ => self
                    .reader
                    .read_u32::<BigEndian>()
                    .map(|v| v as usize)
                    .map_err(|e| IoError::ParseError(format!("Failed to read record marker: {e}"))),
            },
            RecordMarkerSize::EightByte => match self.config.endian {
                EndianMode::Little | EndianMode::Native if cfg!(target_endian = "little") => self
                    .reader
                    .read_u64::<LittleEndian>()
                    .map(|v| v as usize)
                    .map_err(|e| IoError::ParseError(format!("Failed to read record marker: {e}"))),
                _ => self
                    .reader
                    .read_u64::<BigEndian>()
                    .map(|v| v as usize)
                    .map_err(|e| IoError::ParseError(format!("Failed to read record marker: {e}"))),
            },
        }
    }

    /// Read a complete record with validation
    fn read_record(&mut self) -> Result<Vec<u8>> {
        match self.config.access_mode {
            AccessMode::Sequential => {
                // Read start marker
                let start_marker = self.read_record_marker()?;

                // Read data
                let mut data = vec![0u8; start_marker];
                self.reader
                    .read_exact(&mut data)
                    .map_err(|e| IoError::ParseError(format!("Failed to read record data: {e}")))?;

                // Read end marker
                let end_marker = self.read_record_marker()?;

                // Validate markers if enabled
                if self.config.validate_markers && start_marker != end_marker {
                    return Err(IoError::ParseError(format!(
                        "Record marker mismatch: start={start_marker}, end={end_marker}"
                    )));
                }

                Ok(data)
            }
            AccessMode::Direct { record_length } => {
                // Read fixed-length record
                let mut data = vec![0u8; record_length];
                self.reader.read_exact(&mut data).map_err(|e| {
                    IoError::ParseError(format!("Failed to read direct access record: {e}"))
                })?;
                Ok(data)
            }
            AccessMode::Stream => {
                // Stream access requires explicit size specification
                Err(IoError::ParseError(
                    "Stream access requires explicit read size".to_string(),
                ))
            }
        }
    }

    /// Read a scalar value of type T
    pub fn read_scalar<T: ScientificNumber>(&mut self) -> Result<T> {
        let data = self.read_record()?;
        if data.len() != std::mem::size_of::<T>() {
            return Err(IoError::ParseError(format!(
                "Expected {} bytes for scalar, got {}",
                std::mem::size_of::<T>(),
                data.len()
            )));
        }

        // Read based on endianness
        let value = match self.config.endian {
            EndianMode::Little | EndianMode::Native if cfg!(target_endian = "little") => {
                T::from_le_bytes(&data)
            }
            _ => T::from_be_bytes(&data),
        };

        Ok(value)
    }

    /// Read a 1D array
    pub fn read_array_1d<T: ScientificNumber>(&mut self, n: usize) -> Result<Array1<T>> {
        let data = self.read_record()?;
        let expected_size = n * std::mem::size_of::<T>();
        if data.len() != expected_size {
            return Err(IoError::ParseError(format!(
                "Expected {expected_size} bytes for array, got {}",
                data.len()
            )));
        }

        let mut values = Vec::with_capacity(n);
        let item_size = std::mem::size_of::<T>();

        for i in 0..n {
            let start = i * item_size;
            let end = start + item_size;
            let value = match self.config.endian {
                EndianMode::Little | EndianMode::Native if cfg!(target_endian = "little") => {
                    T::from_le_bytes(&data[start..end])
                }
                _ => T::from_be_bytes(&data[start..end]),
            };
            values.push(value);
        }

        Ok(Array1::from_vec(values))
    }

    /// Read a 2D array (Fortran column-major order)
    pub fn read_array_2d<T: ScientificNumber>(
        &mut self,
        rows: usize,
        cols: usize,
    ) -> Result<Array2<T>> {
        let data = self.read_record()?;
        let expected_size = rows * cols * std::mem::size_of::<T>();
        if data.len() != expected_size {
            return Err(IoError::ParseError(format!(
                "Expected {expected_size} bytes for array, got {}",
                data.len()
            )));
        }

        let mut values = Vec::with_capacity(rows * cols);
        let item_size = std::mem::size_of::<T>();

        for i in 0..(rows * cols) {
            let start = i * item_size;
            let end = start + item_size;
            let value = match self.config.endian {
                EndianMode::Little | EndianMode::Native if cfg!(target_endian = "little") => {
                    T::from_le_bytes(&data[start..end])
                }
                _ => T::from_be_bytes(&data[start..end]),
            };
            values.push(value);
        }

        // Convert from Fortran column-major to row-major
        Array2::from_shape_vec((rows, cols).f(), values)
            .map_err(|e| IoError::ParseError(format!("Failed to create array: {e}")))
    }

    /// Read a 3D array (Fortran column-major order)
    pub fn read_array_3d<T: ScientificNumber>(
        &mut self,
        dim1: usize,
        dim2: usize,
        dim3: usize,
    ) -> Result<Array3<T>> {
        let data = self.read_record()?;
        let expected_size = dim1 * dim2 * dim3 * std::mem::size_of::<T>();
        if data.len() != expected_size {
            return Err(IoError::ParseError(format!(
                "Expected {expected_size} bytes for array, got {}",
                data.len()
            )));
        }

        let mut values = Vec::with_capacity(dim1 * dim2 * dim3);
        let item_size = std::mem::size_of::<T>();

        for i in 0..(dim1 * dim2 * dim3) {
            let start = i * item_size;
            let end = start + item_size;
            let value = match self.config.endian {
                EndianMode::Little | EndianMode::Native if cfg!(target_endian = "little") => {
                    T::from_le_bytes(&data[start..end])
                }
                _ => T::from_be_bytes(&data[start..end]),
            };
            values.push(value);
        }

        // Convert from Fortran column-major to row-major
        Array3::from_shape_vec((dim1, dim2, dim3).f(), values)
            .map_err(|e| IoError::ParseError(format!("Failed to create array: {e}")))
    }

    /// Read a character string
    pub fn read_string(&mut self, length: usize) -> Result<String> {
        let data = self.read_record()?;
        if data.len() != length {
            return Err(IoError::ParseError(format!(
                "Expected {} bytes for string, got {}",
                length,
                data.len()
            )));
        }

        // Fortran strings are space-padded
        let string = String::from_utf8_lossy(&data);
        Ok(string.trim_end().to_string())
    }

    /// Skip a record
    pub fn skip_record(&mut self) -> Result<()> {
        match self.config.access_mode {
            AccessMode::Sequential => {
                // Read start marker to get size
                let size = self.read_record_marker()?;

                // Skip data and end marker
                let skip_size = size
                    + match self.config.marker_size {
                        RecordMarkerSize::FourByte => 4,
                        RecordMarkerSize::EightByte => 8,
                    };

                self.reader
                    .seek(SeekFrom::Current(skip_size as i64))
                    .map_err(|e| IoError::ParseError(format!("Failed to skip record: {e}")))?;

                Ok(())
            }
            AccessMode::Direct { record_length } => {
                self.reader
                    .seek(SeekFrom::Current(record_length as i64))
                    .map_err(|e| IoError::ParseError(format!("Failed to skip record: {e}")))?;
                Ok(())
            }
            AccessMode::Stream => Err(IoError::ParseError(
                "Cannot skip record in stream access mode".to_string(),
            )),
        }
    }
}

impl<W: Write> FortranFile<W> {
    /// Write a record marker
    fn write_record_marker(&mut self, size: usize) -> Result<()> {
        match self.config.marker_size {
            RecordMarkerSize::FourByte => {
                let marker = size as u32;
                match self.config.endian {
                    EndianMode::Little | EndianMode::Native if cfg!(target_endian = "little") => {
                        self.reader.write_u32::<LittleEndian>(marker).map_err(|e| {
                            IoError::FileError(format!("Failed to write record marker: {e}"))
                        })
                    }
                    _ => self.reader.write_u32::<BigEndian>(marker).map_err(|e| {
                        IoError::FileError(format!("Failed to write record marker: {e}"))
                    }),
                }
            }
            RecordMarkerSize::EightByte => {
                let marker = size as u64;
                match self.config.endian {
                    EndianMode::Little | EndianMode::Native if cfg!(target_endian = "little") => {
                        self.reader.write_u64::<LittleEndian>(marker).map_err(|e| {
                            IoError::FileError(format!("Failed to write record marker: {e}"))
                        })
                    }
                    _ => self.reader.write_u64::<BigEndian>(marker).map_err(|e| {
                        IoError::FileError(format!("Failed to write record marker: {e}"))
                    }),
                }
            }
        }
    }

    /// Write a complete record
    fn write_record(&mut self, data: &[u8]) -> Result<()> {
        match self.config.access_mode {
            AccessMode::Sequential => {
                // Write start marker
                self.write_record_marker(data.len())?;

                // Write data
                self.reader
                    .write_all(data)
                    .map_err(|e| IoError::FileError(format!("Failed to write record data: {e}")))?;

                // Write end marker
                self.write_record_marker(data.len())?;

                Ok(())
            }
            AccessMode::Direct { record_length } => {
                if data.len() > record_length {
                    return Err(IoError::FileError(format!(
                        "Data size {} exceeds record length {record_length}",
                        data.len()
                    )));
                }

                // Write data
                self.reader
                    .write_all(data)
                    .map_err(|e| IoError::FileError(format!("Failed to write record data: {e}")))?;

                // Pad to record length
                if data.len() < record_length {
                    let padding = vec![0u8; record_length - data.len()];
                    self.reader
                        .write_all(&padding)
                        .map_err(|e| IoError::FileError(format!("Failed to write padding: {e}")))?;
                }

                Ok(())
            }
            AccessMode::Stream => {
                // Direct write for stream access
                self.reader
                    .write_all(data)
                    .map_err(|e| IoError::FileError(format!("Failed to write stream data: {e}")))
            }
        }
    }

    /// Write a scalar value
    pub fn write_scalar<T: ScientificNumber>(&mut self, value: T) -> Result<()> {
        let data = match self.config.endian {
            EndianMode::Little | EndianMode::Native if cfg!(target_endian = "little") => {
                value.to_le_bytes()
            }
            _ => value.to_be_bytes(),
        };

        self.write_record(&data)
    }

    /// Write a 1D array
    pub fn write_array_1d<T: ScientificNumber>(&mut self, array: &Array1<T>) -> Result<()> {
        let mut data = Vec::with_capacity(array.len() * std::mem::size_of::<T>());

        for value in array.iter() {
            let bytes = match self.config.endian {
                EndianMode::Little | EndianMode::Native if cfg!(target_endian = "little") => {
                    value.to_le_bytes()
                }
                _ => value.to_be_bytes(),
            };
            data.extend_from_slice(&bytes);
        }

        self.write_record(&data)
    }

    /// Write a 2D array (convert to Fortran column-major order)
    pub fn write_array_2d<T: ScientificNumber>(&mut self, array: &Array2<T>) -> Result<()> {
        let mut data = Vec::with_capacity(array.len() * std::mem::size_of::<T>());

        // Write in column-major order (Fortran convention)
        for col in 0..array.ncols() {
            for row in 0..array.nrows() {
                let value = &array[[row, col]];
                let bytes = match self.config.endian {
                    EndianMode::Little | EndianMode::Native if cfg!(target_endian = "little") => {
                        value.to_le_bytes()
                    }
                    _ => value.to_be_bytes(),
                };
                data.extend_from_slice(&bytes);
            }
        }

        self.write_record(&data)
    }

    /// Write a 3D array (convert to Fortran column-major order)
    pub fn write_array_3d<T: ScientificNumber>(&mut self, array: &Array3<T>) -> Result<()> {
        let shape = array.shape();
        let mut data = Vec::with_capacity(array.len() * std::mem::size_of::<T>());

        // Write in column-major order (Fortran convention)
        for k in 0..shape[2] {
            for j in 0..shape[1] {
                for i in 0..shape[0] {
                    let value = &array[[i, j, k]];
                    let bytes = match self.config.endian {
                        EndianMode::Little | EndianMode::Native
                            if cfg!(target_endian = "little") =>
                        {
                            value.to_le_bytes()
                        }
                        _ => value.to_be_bytes(),
                    };
                    data.extend_from_slice(&bytes);
                }
            }
        }

        self.write_record(&data)
    }

    /// Write a generic N-dimensional array
    pub fn write_array<T: ScientificNumber, D>(&mut self, array: &Array<T, D>) -> Result<()>
    where
        D: scirs2_core::ndarray::Dimension,
    {
        let mut data = Vec::with_capacity(array.len() * std::mem::size_of::<T>());

        // Convert to column-major order and write
        for value in array.t().iter() {
            let bytes = match self.config.endian {
                EndianMode::Little | EndianMode::Native if cfg!(target_endian = "little") => {
                    value.to_le_bytes()
                }
                _ => value.to_be_bytes(),
            };
            data.extend_from_slice(&bytes);
        }

        self.write_record(&data)
    }

    /// Write a character string (space-padded to length)
    pub fn write_string(&mut self, string: &str, length: usize) -> Result<()> {
        let mut data = string.as_bytes().to_vec();

        // Pad or truncate to specified length
        match data.len().cmp(&length) {
            std::cmp::Ordering::Less => data.resize(length, b' '),
            std::cmp::Ordering::Greater => data.truncate(length),
            std::cmp::Ordering::Equal => {}
        }

        self.write_record(&data)
    }

    /// Flush the writer
    pub fn flush(&mut self) -> Result<()> {
        self.reader
            .flush()
            .map_err(|e| IoError::FileError(format!("Failed to flush: {e}")))
    }
}

/// Read a complete Fortran unformatted file into memory
#[allow(dead_code)]
pub fn read_fortran_file<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<u8>>> {
    let mut file = FortranFile::open(path)?;
    let mut records = Vec::new();

    loop {
        match file.read_record() {
            Ok(record) => records.push(record),
            Err(IoError::ParseError(msg)) if msg.contains("Failed to read record marker") => break,
            Err(e) => return Err(e),
        }
    }

    Ok(records)
}

/// Detect the endianness and record marker size of a Fortran file
#[allow(dead_code)]
pub fn detect_fortran_format<P: AsRef<Path>>(path: P) -> Result<(EndianMode, RecordMarkerSize)> {
    let mut file = File::open(path.as_ref())
        .map_err(|_e| IoError::FileNotFound(path.as_ref().to_string_lossy().to_string()))?;

    // Read first 8 bytes
    let mut buffer = [0u8; 8];
    file.read_exact(&mut buffer)
        .map_err(|e| IoError::ParseError(format!("Failed to read file header: {e}")))?;

    // Try different interpretations
    let little_4 = LittleEndian::read_u32(&buffer[0..4]) as usize;
    let big_4 = BigEndian::read_u32(&buffer[0..4]) as usize;
    let little_8 = LittleEndian::read_u64(&buffer) as usize;
    let big_8 = BigEndian::read_u64(&buffer) as usize;

    // Get file size
    let file_size = file
        .metadata()
        .map_err(|e| IoError::ParseError(format!("Failed to get file metadata: {e}")))?
        .len() as usize;

    // Check which interpretation makes sense
    // A valid record size should be less than the file size
    if little_4 > 0 && file_size > 8 && little_4 <= file_size - 8 {
        // Try to read the end marker
        file.seek(SeekFrom::Start((4 + little_4) as u64))
            .map_err(|e| IoError::ParseError(format!("Failed to seek: {e}")))?;
        let mut end_marker = [0u8; 4];
        if file.read_exact(&mut end_marker).is_ok() {
            let end_value = LittleEndian::read_u32(&end_marker) as usize;
            if end_value == little_4 {
                return Ok((EndianMode::Little, RecordMarkerSize::FourByte));
            }
        }
    }

    if big_4 > 0 && file_size > 8 && big_4 <= file_size - 8 {
        file.seek(SeekFrom::Start((4 + big_4) as u64))
            .map_err(|e| IoError::ParseError(format!("Failed to seek: {e}")))?;
        let mut end_marker = [0u8; 4];
        if file.read_exact(&mut end_marker).is_ok() {
            let end_value = BigEndian::read_u32(&end_marker) as usize;
            if end_value == big_4 {
                return Ok((EndianMode::Big, RecordMarkerSize::FourByte));
            }
        }
    }

    // Try 8-byte markers
    if little_8 > 0 && file_size > 16 && little_8 <= file_size - 16 {
        file.seek(SeekFrom::Start((8 + little_8) as u64))
            .map_err(|e| IoError::ParseError(format!("Failed to seek: {e}")))?;
        let mut end_marker = [0u8; 8];
        if file.read_exact(&mut end_marker).is_ok() {
            let end_value = LittleEndian::read_u64(&end_marker) as usize;
            if end_value == little_8 {
                return Ok((EndianMode::Little, RecordMarkerSize::EightByte));
            }
        }
    }

    if big_8 > 0 && file_size > 16 && big_8 <= file_size - 16 {
        file.seek(SeekFrom::Start((8 + big_8) as u64))
            .map_err(|e| IoError::ParseError(format!("Failed to seek: {e}")))?;
        let mut end_marker = [0u8; 8];
        if file.read_exact(&mut end_marker).is_ok() {
            let end_value = BigEndian::read_u64(&end_marker) as usize;
            if end_value == big_8 {
                return Ok((EndianMode::Big, RecordMarkerSize::EightByte));
            }
        }
    }

    Err(IoError::ParseError(
        "Unable to detect Fortran file format".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;
    use tempfile::NamedTempFile;

    #[test]
    fn test_scalar_read_write() -> Result<()> {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Write scalars
        {
            let mut file = FortranFile::create(path)?;
            file.write_scalar(42i32)?;
            file.write_scalar(std::f64::consts::PI)?;
            file.flush()?;
        }

        // Read scalars
        {
            let mut file = FortranFile::open(path)?;
            let int_val: i32 = file.read_scalar()?;
            let float_val: f64 = file.read_scalar()?;

            assert_eq!(int_val, 42);
            assert!((float_val - std::f64::consts::PI).abs() < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_array_read_write() -> Result<()> {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let array = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        // Write array
        {
            let mut file = FortranFile::create(path)?;
            file.write_array_1d(&array)?;
            file.flush()?;
        }

        // Read array
        {
            let mut file = FortranFile::open(path)?;
            let read_array: Array1<f64> = file.read_array_1d(5)?;

            assert_eq!(array, read_array);
        }

        Ok(())
    }

    #[test]
    fn test_2d_array() -> Result<()> {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let array = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .map_err(|e| IoError::ParseError(format!("Shape error: {e}")))?;

        // Write array
        {
            let mut file = FortranFile::create(path)?;
            file.write_array_2d(&array)?;
            file.flush()?;
        }

        // Read array
        {
            let mut file = FortranFile::open(path)?;
            let read_array: Array2<f64> = file.read_array_2d(2, 3)?;

            assert_eq!(array, read_array);
        }

        Ok(())
    }

    #[test]
    fn test_string_read_write() -> Result<()> {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let test_string = "Hello Fortran";
        let string_length = 20;

        // Write string
        {
            let mut file = FortranFile::create(path)?;
            file.write_string(test_string, string_length)?;
            file.flush()?;
        }

        // Read string
        {
            let mut file = FortranFile::open(path)?;
            let read_string = file.read_string(string_length)?;

            assert_eq!(read_string, test_string);
        }

        Ok(())
    }

    #[test]
    fn test_format_detection() -> Result<()> {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Write a file with known format
        {
            let config = FortranConfig {
                endian: EndianMode::Little,
                marker_size: RecordMarkerSize::FourByte,
                ..Default::default()
            };
            let mut file = FortranFile::create_with_config(path, config)?;
            file.write_scalar(42i32)?;
            file.flush()?;
        }

        // Detect format
        let (endian, marker_size) = detect_fortran_format(path)?;
        assert_eq!(endian, EndianMode::Little);
        assert_eq!(marker_size, RecordMarkerSize::FourByte);

        Ok(())
    }
}
