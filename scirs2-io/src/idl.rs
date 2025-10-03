//! IDL (Interactive Data Language) save file format support
//!
//! This module provides functionality for reading and writing IDL save files (.sav),
//! which are commonly used in scientific data analysis, particularly in astronomy
//! and remote sensing applications.

use crate::error::{IoError, Result};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt, WriteBytesExt};
use scirs2_core::ndarray::{Array, ArrayD, IxDyn};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// IDL data types
#[derive(Debug, Clone)]
pub enum IdlType {
    /// Undefined type
    Undefined,
    /// Byte (8-bit unsigned integer)
    Byte(ArrayD<u8>),
    /// 16-bit signed integer
    Int(ArrayD<i16>),
    /// 32-bit signed integer
    Long(ArrayD<i32>),
    /// 32-bit floating point
    Float(ArrayD<f32>),
    /// 64-bit floating point
    Double(ArrayD<f64>),
    /// Complex (32-bit real + 32-bit imaginary)
    Complex(ArrayD<scirs2_core::numeric::Complex<f32>>),
    /// String
    String(String),
    /// String array
    StringArray(Vec<String>),
    /// Structure
    Structure(IdlStructure),
    /// Double complex (64-bit real + 64-bit imaginary)
    DoubleComplex(ArrayD<scirs2_core::numeric::Complex<f64>>),
    /// Pointer (heap variable)
    Pointer(Box<IdlType>),
    /// Object reference
    ObjectRef(IdlObject),
    /// 16-bit unsigned integer
    UInt(ArrayD<u16>),
    /// 32-bit unsigned integer
    ULong(ArrayD<u32>),
    /// 64-bit signed integer
    Long64(ArrayD<i64>),
    /// 64-bit unsigned integer
    ULong64(ArrayD<u64>),
}

/// IDL structure
#[derive(Debug, Clone)]
pub struct IdlStructure {
    /// Structure name
    pub name: String,
    /// Structure fields
    pub fields: HashMap<String, IdlType>,
}

/// IDL object
#[derive(Debug, Clone)]
pub struct IdlObject {
    /// Class name
    pub class_name: String,
    /// Object data
    pub data: HashMap<String, IdlType>,
}

/// IDL save file signature
const IDL_SIGNATURE: &[u8] = b"SR";

/// IDL record types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum RecordType {
    Timestamp = 0,
    Version = 1,
    Variable = 2,
    SystemVariable = 6,
    EndMarker = 10,
    CommonVariable = 3,
    Identifier = 12,
    Header = 13,
    HeapData = 16,
    HeapHeader = 17,
    CompressedData = 19,
}

/// IDL data type codes
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum TypeCode {
    Undefined = 0,
    Byte = 1,
    Int = 2,
    Long = 3,
    Float = 4,
    Double = 5,
    Complex = 6,
    String = 7,
    Structure = 8,
    DoubleComplex = 9,
    Pointer = 10,
    ObjectRef = 11,
    UInt = 12,
    ULong = 13,
    Long64 = 14,
    ULong64 = 15,
}

/// IDL save file reader
pub struct IdlReader {
    reader: BufReader<File>,
    endianness: Endianness,
    #[allow(dead_code)]
    heap_data: HashMap<u32, IdlType>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Endianness {
    Big,
    Little,
}

impl IdlReader {
    /// Create a new IDL reader
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
        let mut reader = BufReader::new(file);

        // Read and verify signature
        let mut signature = [0u8; 2];
        reader
            .read_exact(&mut signature)
            .map_err(|e| IoError::FileError(e.to_string()))?;

        if signature != IDL_SIGNATURE {
            return Err(IoError::FormatError(
                "Invalid IDL file signature".to_string(),
            ));
        }

        // Read format byte to determine endianness
        let format = reader
            .read_u8()
            .map_err(|e| IoError::FileError(e.to_string()))?;
        let endianness = if format & 0x04 != 0 {
            Endianness::Big
        } else {
            Endianness::Little
        };

        // Skip to the beginning of records
        reader
            .seek(SeekFrom::Start(4))
            .map_err(|e| IoError::FileError(e.to_string()))?;

        Ok(Self {
            reader,
            endianness,
            heap_data: HashMap::new(),
        })
    }

    /// Read all variables from the IDL save file
    pub fn read_all(&mut self) -> Result<HashMap<String, IdlType>> {
        let mut variables = HashMap::new();

        loop {
            match self.read_record() {
                Ok((record_type, name, data)) => {
                    match record_type {
                        RecordType::Variable => {
                            if let Some(name) = name {
                                if let Some(data) = data {
                                    variables.insert(name, data);
                                }
                            }
                        }
                        RecordType::HeapData => {
                            // Store heap data for later reference
                            if let Some(_data) = data {
                                // In a real implementation, we'd parse the heap index
                                // For now, just store it
                            }
                        }
                        RecordType::EndMarker => break,
                        _ => {} // Skip other record types
                    }
                }
                Err(e) => {
                    // Check if we've reached EOF
                    if let IoError::FileError(ref msg) = e {
                        if msg.contains("UnexpectedEof") {
                            break;
                        }
                    }
                    return Err(e);
                }
            }
        }

        Ok(variables)
    }

    /// Read a single record
    fn read_record(&mut self) -> Result<(RecordType, Option<String>, Option<IdlType>)> {
        // Read record header
        let rec_type = self.read_u32()? as u8;
        let next_offset = self.read_u32()?;
        let _unknown = self.read_u32()?;
        let _flags = self.read_u32()?;

        let record_type = match rec_type {
            0 => RecordType::Timestamp,
            1 => RecordType::Version,
            2 => RecordType::Variable,
            3 => RecordType::CommonVariable,
            6 => RecordType::SystemVariable,
            10 => RecordType::EndMarker,
            12 => RecordType::Identifier,
            13 => RecordType::Header,
            16 => RecordType::HeapData,
            17 => RecordType::HeapHeader,
            19 => RecordType::CompressedData,
            _ => {
                return Err(IoError::FormatError(format!(
                    "Unknown record type: {rec_type}",
                )))
            }
        };

        match record_type {
            RecordType::Variable => {
                let name = self.read_string()?;
                let data = self.read_variable_data()?;
                Ok((record_type, Some(name), Some(data)))
            }
            RecordType::EndMarker => Ok((record_type, None, None)),
            _ => {
                // Skip unknown record types
                if next_offset > 0 {
                    self.reader
                        .seek(SeekFrom::Start(next_offset as u64))
                        .map_err(|e| IoError::FileError(e.to_string()))?;
                }
                Ok((record_type, None, None))
            }
        }
    }

    /// Read variable data
    fn read_variable_data(&mut self) -> Result<IdlType> {
        let type_code = self.read_u32()? as u8;
        let _flags = self.read_u32()?;

        match type_code {
            1 => self.read_byte_array(),
            2 => self.read_int_array(),
            3 => self.read_long_array(),
            4 => self.read_float_array(),
            5 => self.read_double_array(),
            6 => self.read_complex_array(),
            7 => self.read_string_data(),
            8 => self.read_structure(),
            9 => self.read_double_complex_array(),
            12 => self.read_uint_array(),
            13 => self.read_ulong_array(),
            14 => self.read_long64_array(),
            15 => self.read_ulong64_array(),
            _ => Err(IoError::FormatError(format!(
                "Unknown type code: {type_code}",
            ))),
        }
    }

    /// Read array dimensions
    fn read_dimensions(&mut self) -> Result<Vec<usize>> {
        let ndims = self.read_u32()? as usize;
        let mut dims = Vec::with_capacity(ndims);

        // Skip padding
        self.read_u32()?;

        for _ in 0..ndims {
            dims.push(self.read_u32()? as usize);
        }

        // IDL uses column-major order, so reverse dimensions for row-major ndarray
        dims.reverse();

        Ok(dims)
    }

    /// Read byte array
    fn read_byte_array(&mut self) -> Result<IdlType> {
        let dims = self.read_dimensions()?;
        let total_size: usize = dims.iter().product();

        let mut data = vec![0u8; total_size];
        self.reader
            .read_exact(&mut data)
            .map_err(|e| IoError::FileError(e.to_string()))?;

        let shape = IxDyn(&dims);
        let array =
            Array::from_shape_vec(shape, data).map_err(|e| IoError::FormatError(e.to_string()))?;

        Ok(IdlType::Byte(array))
    }

    /// Read int array
    fn read_int_array(&mut self) -> Result<IdlType> {
        let dims = self.read_dimensions()?;
        let total_size: usize = dims.iter().product();

        let mut data = Vec::with_capacity(total_size);
        for _ in 0..total_size {
            data.push(self.read_i16()?);
        }

        let shape = IxDyn(&dims);
        let array =
            Array::from_shape_vec(shape, data).map_err(|e| IoError::FormatError(e.to_string()))?;

        Ok(IdlType::Int(array))
    }

    /// Read long array
    fn read_long_array(&mut self) -> Result<IdlType> {
        let dims = self.read_dimensions()?;
        let total_size: usize = dims.iter().product();

        let mut data = Vec::with_capacity(total_size);
        for _ in 0..total_size {
            data.push(self.read_i32()?);
        }

        let shape = IxDyn(&dims);
        let array =
            Array::from_shape_vec(shape, data).map_err(|e| IoError::FormatError(e.to_string()))?;

        Ok(IdlType::Long(array))
    }

    /// Read float array
    fn read_float_array(&mut self) -> Result<IdlType> {
        let dims = self.read_dimensions()?;
        let total_size: usize = dims.iter().product();

        let mut data = Vec::with_capacity(total_size);
        for _ in 0..total_size {
            data.push(self.read_f32()?);
        }

        let shape = IxDyn(&dims);
        let array =
            Array::from_shape_vec(shape, data).map_err(|e| IoError::FormatError(e.to_string()))?;

        Ok(IdlType::Float(array))
    }

    /// Read double array
    fn read_double_array(&mut self) -> Result<IdlType> {
        let dims = self.read_dimensions()?;
        let total_size: usize = dims.iter().product();

        let mut data = Vec::with_capacity(total_size);
        for _ in 0..total_size {
            data.push(self.read_f64()?);
        }

        let shape = IxDyn(&dims);
        let array =
            Array::from_shape_vec(shape, data).map_err(|e| IoError::FormatError(e.to_string()))?;

        Ok(IdlType::Double(array))
    }

    /// Read complex array
    fn read_complex_array(&mut self) -> Result<IdlType> {
        let dims = self.read_dimensions()?;
        let total_size: usize = dims.iter().product();

        let mut data = Vec::with_capacity(total_size);
        for _ in 0..total_size {
            let real = self.read_f32()?;
            let imag = self.read_f32()?;
            data.push(scirs2_core::numeric::Complex::new(real, imag));
        }

        let shape = IxDyn(&dims);
        let array =
            Array::from_shape_vec(shape, data).map_err(|e| IoError::FormatError(e.to_string()))?;

        Ok(IdlType::Complex(array))
    }

    /// Read double complex array
    fn read_double_complex_array(&mut self) -> Result<IdlType> {
        let dims = self.read_dimensions()?;
        let total_size: usize = dims.iter().product();

        let mut data = Vec::with_capacity(total_size);
        for _ in 0..total_size {
            let real = self.read_f64()?;
            let imag = self.read_f64()?;
            data.push(scirs2_core::numeric::Complex::new(real, imag));
        }

        let shape = IxDyn(&dims);
        let array =
            Array::from_shape_vec(shape, data).map_err(|e| IoError::FormatError(e.to_string()))?;

        Ok(IdlType::DoubleComplex(array))
    }

    /// Read unsigned int array
    fn read_uint_array(&mut self) -> Result<IdlType> {
        let dims = self.read_dimensions()?;
        let total_size: usize = dims.iter().product();

        let mut data = Vec::with_capacity(total_size);
        for _ in 0..total_size {
            data.push(self.read_u16()?);
        }

        let shape = IxDyn(&dims);
        let array =
            Array::from_shape_vec(shape, data).map_err(|e| IoError::FormatError(e.to_string()))?;

        Ok(IdlType::UInt(array))
    }

    /// Read unsigned long array
    fn read_ulong_array(&mut self) -> Result<IdlType> {
        let dims = self.read_dimensions()?;
        let total_size: usize = dims.iter().product();

        let mut data = Vec::with_capacity(total_size);
        for _ in 0..total_size {
            data.push(self.read_u32()?);
        }

        let shape = IxDyn(&dims);
        let array =
            Array::from_shape_vec(shape, data).map_err(|e| IoError::FormatError(e.to_string()))?;

        Ok(IdlType::ULong(array))
    }

    /// Read long64 array
    fn read_long64_array(&mut self) -> Result<IdlType> {
        let dims = self.read_dimensions()?;
        let total_size: usize = dims.iter().product();

        let mut data = Vec::with_capacity(total_size);
        for _ in 0..total_size {
            data.push(self.read_i64()?);
        }

        let shape = IxDyn(&dims);
        let array =
            Array::from_shape_vec(shape, data).map_err(|e| IoError::FormatError(e.to_string()))?;

        Ok(IdlType::Long64(array))
    }

    /// Read unsigned long64 array
    fn read_ulong64_array(&mut self) -> Result<IdlType> {
        let dims = self.read_dimensions()?;
        let total_size: usize = dims.iter().product();

        let mut data = Vec::with_capacity(total_size);
        for _ in 0..total_size {
            data.push(self.read_u64()?);
        }

        let shape = IxDyn(&dims);
        let array =
            Array::from_shape_vec(shape, data).map_err(|e| IoError::FormatError(e.to_string()))?;

        Ok(IdlType::ULong64(array))
    }

    /// Read string data
    fn read_string_data(&mut self) -> Result<IdlType> {
        let length = self.read_u32()? as usize;
        let mut buffer = vec![0u8; length];
        self.reader
            .read_exact(&mut buffer)
            .map_err(|e| IoError::FileError(e.to_string()))?;

        let string = String::from_utf8(buffer).map_err(|e| IoError::FormatError(e.to_string()))?;

        Ok(IdlType::String(string))
    }

    /// Read structure
    fn read_structure(&mut self) -> Result<IdlType> {
        // Read structure header
        let _flags = self.read_u32()?; // Structure flags
        let _struct_def_id = self.read_u32()?; // Structure definition ID
        let nfields = self.read_u32()? as usize; // Number of fields
        let _nbytes = self.read_u32()?; // Number of bytes in structure data

        // Read structure name
        let struct_name = self.read_string()?;

        // Read field names
        let mut field_names = Vec::with_capacity(nfields);
        for _ in 0..nfields {
            field_names.push(self.read_string()?);
        }

        // Read field data descriptors
        let mut field_types = Vec::with_capacity(nfields);
        for _ in 0..nfields {
            let type_code = self.read_u32()? as u8;
            let _flags = self.read_u32()?;
            field_types.push(type_code);
        }

        // Read field data
        let mut fields = HashMap::new();
        for (i, field_name) in field_names.iter().enumerate() {
            let type_code = field_types[i];

            // For structures, we recursively read the data based on type
            let field_data = match type_code {
                1 => self.read_byte_array()?,
                2 => self.read_int_array()?,
                3 => self.read_long_array()?,
                4 => self.read_float_array()?,
                5 => self.read_double_array()?,
                6 => self.read_complex_array()?,
                7 => self.read_string_data()?,
                8 => self.read_structure()?, // Nested structure
                9 => self.read_double_complex_array()?,
                12 => self.read_uint_array()?,
                13 => self.read_ulong_array()?,
                14 => self.read_long64_array()?,
                15 => self.read_ulong64_array()?,
                _ => {
                    // Unknown type, skip it by creating undefined
                    IdlType::Undefined
                }
            };

            fields.insert(field_name.clone(), field_data);
        }

        let structure = IdlStructure {
            name: struct_name,
            fields,
        };

        Ok(IdlType::Structure(structure))
    }

    /// Read string
    fn read_string(&mut self) -> Result<String> {
        let length = self.read_u32()? as usize;
        let mut buffer = vec![0u8; length];
        self.reader
            .read_exact(&mut buffer)
            .map_err(|e| IoError::FileError(e.to_string()))?;

        // Remove null terminator if present
        if buffer.last() == Some(&0) {
            buffer.pop();
        }

        String::from_utf8(buffer).map_err(|e| IoError::FormatError(e.to_string()))
    }

    // Helper methods for reading with correct endianness
    fn read_u16(&mut self) -> Result<u16> {
        match self.endianness {
            Endianness::Big => self.reader.read_u16::<BigEndian>(),
            Endianness::Little => self.reader.read_u16::<LittleEndian>(),
        }
        .map_err(|e| IoError::FileError(e.to_string()))
    }

    fn read_i16(&mut self) -> Result<i16> {
        match self.endianness {
            Endianness::Big => self.reader.read_i16::<BigEndian>(),
            Endianness::Little => self.reader.read_i16::<LittleEndian>(),
        }
        .map_err(|e| IoError::FileError(e.to_string()))
    }

    fn read_u32(&mut self) -> Result<u32> {
        match self.endianness {
            Endianness::Big => self.reader.read_u32::<BigEndian>(),
            Endianness::Little => self.reader.read_u32::<LittleEndian>(),
        }
        .map_err(|e| IoError::FileError(e.to_string()))
    }

    fn read_i32(&mut self) -> Result<i32> {
        match self.endianness {
            Endianness::Big => self.reader.read_i32::<BigEndian>(),
            Endianness::Little => self.reader.read_i32::<LittleEndian>(),
        }
        .map_err(|e| IoError::FileError(e.to_string()))
    }

    fn read_u64(&mut self) -> Result<u64> {
        match self.endianness {
            Endianness::Big => self.reader.read_u64::<BigEndian>(),
            Endianness::Little => self.reader.read_u64::<LittleEndian>(),
        }
        .map_err(|e| IoError::FileError(e.to_string()))
    }

    fn read_i64(&mut self) -> Result<i64> {
        match self.endianness {
            Endianness::Big => self.reader.read_i64::<BigEndian>(),
            Endianness::Little => self.reader.read_i64::<LittleEndian>(),
        }
        .map_err(|e| IoError::FileError(e.to_string()))
    }

    fn read_f32(&mut self) -> Result<f32> {
        match self.endianness {
            Endianness::Big => self.reader.read_f32::<BigEndian>(),
            Endianness::Little => self.reader.read_f32::<LittleEndian>(),
        }
        .map_err(|e| IoError::FileError(e.to_string()))
    }

    fn read_f64(&mut self) -> Result<f64> {
        match self.endianness {
            Endianness::Big => self.reader.read_f64::<BigEndian>(),
            Endianness::Little => self.reader.read_f64::<LittleEndian>(),
        }
        .map_err(|e| IoError::FileError(e.to_string()))
    }
}

/// Read variables from an IDL save file
#[allow(dead_code)]
pub fn read_idl<P: AsRef<Path>>(path: P) -> Result<HashMap<String, IdlType>> {
    let mut reader = IdlReader::new(path)?;
    reader.read_all()
}

/// IDL save file writer (basic implementation)
pub struct IdlWriter {
    writer: BufWriter<File>,
    endianness: Endianness,
}

impl IdlWriter {
    /// Create a new IDL writer
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
        let mut writer = BufWriter::new(file);

        // Write signature
        writer
            .write_all(IDL_SIGNATURE)
            .map_err(|e| IoError::FileError(e.to_string()))?;

        // Write format byte (little-endian)
        writer
            .write_u8(0x00)
            .map_err(|e| IoError::FileError(e.to_string()))?;

        // Reserved byte
        writer
            .write_u8(0x00)
            .map_err(|e| IoError::FileError(e.to_string()))?;

        Ok(Self {
            writer,
            endianness: Endianness::Little,
        })
    }

    /// Write variables to the IDL save file
    pub fn write_all(&mut self, variables: &HashMap<String, IdlType>) -> Result<()> {
        // Write timestamp record
        self.write_timestamp()?;

        // Write version record
        self.write_version()?;

        // Write variables
        for (name, data) in variables {
            self.write_variable(name, data)?;
        }

        // Write end marker
        self.write_end_marker()?;

        self.writer
            .flush()
            .map_err(|e| IoError::FileError(e.to_string()))?;
        Ok(())
    }

    /// Write timestamp record
    fn write_timestamp(&mut self) -> Result<()> {
        self.write_u32(RecordType::Timestamp as u32)?;
        self.write_u32(0)?; // next_offset
        self.write_u32(0)?; // unknown
        self.write_u32(0)?; // flags

        // Write timestamp data (simplified)
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.write_u64(timestamp)?;

        Ok(())
    }

    /// Write version record
    fn write_version(&mut self) -> Result<()> {
        self.write_u32(RecordType::Version as u32)?;
        self.write_u32(0)?; // next_offset
        self.write_u32(0)?; // unknown
        self.write_u32(0)?; // flags

        // Write version info (IDL 8.0 format)
        self.write_u32(0x00000800)?; // Version 8.0

        Ok(())
    }

    /// Write variable record
    fn write_variable(&mut self, name: &str, data: &IdlType) -> Result<()> {
        self.write_u32(RecordType::Variable as u32)?;
        self.write_u32(0)?; // next_offset (would need to calculate)
        self.write_u32(0)?; // unknown
        self.write_u32(0)?; // flags

        // Write variable name
        self.write_string(name)?;

        // Write variable data
        self.write_variable_data(data)?;

        Ok(())
    }

    /// Write variable data
    fn write_variable_data(&mut self, data: &IdlType) -> Result<()> {
        match data {
            IdlType::Byte(array) => self.write_byte_array(array),
            IdlType::Int(array) => self.write_int_array(array),
            IdlType::Long(array) => self.write_long_array(array),
            IdlType::Float(array) => self.write_float_array(array),
            IdlType::Double(array) => self.write_double_array(array),
            IdlType::String(string) => self.write_string_data(string),
            _ => Err(IoError::Other(
                "Unsupported IDL type for writing".to_string(),
            )),
        }
    }

    /// Write byte array
    fn write_byte_array(&mut self, array: &ArrayD<u8>) -> Result<()> {
        self.write_u32(TypeCode::Byte as u32)?;
        self.write_u32(0)?; // flags

        self.write_dimensions(array.shape())?;

        // Write data
        for &val in array.iter() {
            self.writer
                .write_u8(val)
                .map_err(|e| IoError::FileError(e.to_string()))?;
        }

        Ok(())
    }

    /// Write int array
    fn write_int_array(&mut self, array: &ArrayD<i16>) -> Result<()> {
        self.write_u32(TypeCode::Int as u32)?;
        self.write_u32(0)?; // flags

        self.write_dimensions(array.shape())?;

        // Write data
        for &val in array.iter() {
            self.write_i16(val)?;
        }

        Ok(())
    }

    /// Write long array
    fn write_long_array(&mut self, array: &ArrayD<i32>) -> Result<()> {
        self.write_u32(TypeCode::Long as u32)?;
        self.write_u32(0)?; // flags

        self.write_dimensions(array.shape())?;

        // Write data
        for &val in array.iter() {
            self.write_i32(val)?;
        }

        Ok(())
    }

    /// Write float array
    fn write_float_array(&mut self, array: &ArrayD<f32>) -> Result<()> {
        self.write_u32(TypeCode::Float as u32)?;
        self.write_u32(0)?; // flags

        self.write_dimensions(array.shape())?;

        // Write data
        for &val in array.iter() {
            self.write_f32(val)?;
        }

        Ok(())
    }

    /// Write double array
    fn write_double_array(&mut self, array: &ArrayD<f64>) -> Result<()> {
        self.write_u32(TypeCode::Double as u32)?;
        self.write_u32(0)?; // flags

        self.write_dimensions(array.shape())?;

        // Write data
        for &val in array.iter() {
            self.write_f64(val)?;
        }

        Ok(())
    }

    /// Write string data
    fn write_string_data(&mut self, string: &str) -> Result<()> {
        self.write_u32(TypeCode::String as u32)?;
        self.write_u32(0)?; // flags

        self.write_string(string)?;

        Ok(())
    }

    /// Write dimensions
    fn write_dimensions(&mut self, shape: &[usize]) -> Result<()> {
        self.write_u32(shape.len() as u32)?;
        self.write_u32(0)?; // padding

        // IDL uses column-major order, so reverse dimensions
        for &dim in shape.iter().rev() {
            self.write_u32(dim as u32)?;
        }

        Ok(())
    }

    /// Write string
    fn write_string(&mut self, string: &str) -> Result<()> {
        let bytes = string.as_bytes();
        self.write_u32(bytes.len() as u32)?;
        self.writer
            .write_all(bytes)
            .map_err(|e| IoError::FileError(e.to_string()))?;

        // Pad to 4-byte boundary
        let padding = (4 - (bytes.len() % 4)) % 4;
        for _ in 0..padding {
            self.writer
                .write_u8(0)
                .map_err(|e| IoError::FileError(e.to_string()))?;
        }

        Ok(())
    }

    /// Write end marker
    fn write_end_marker(&mut self) -> Result<()> {
        self.write_u32(RecordType::EndMarker as u32)?;
        self.write_u32(0)?; // next_offset
        self.write_u32(0)?; // unknown
        self.write_u32(0)?; // flags

        Ok(())
    }

    // Helper methods for writing with correct endianness
    fn write_i16(&mut self, val: i16) -> Result<()> {
        match self.endianness {
            Endianness::Big => self.writer.write_i16::<BigEndian>(val),
            Endianness::Little => self.writer.write_i16::<LittleEndian>(val),
        }
        .map_err(|e| IoError::FileError(e.to_string()))
    }

    fn write_u32(&mut self, val: u32) -> Result<()> {
        match self.endianness {
            Endianness::Big => self.writer.write_u32::<BigEndian>(val),
            Endianness::Little => self.writer.write_u32::<LittleEndian>(val),
        }
        .map_err(|e| IoError::FileError(e.to_string()))
    }

    fn write_i32(&mut self, val: i32) -> Result<()> {
        match self.endianness {
            Endianness::Big => self.writer.write_i32::<BigEndian>(val),
            Endianness::Little => self.writer.write_i32::<LittleEndian>(val),
        }
        .map_err(|e| IoError::FileError(e.to_string()))
    }

    fn write_u64(&mut self, val: u64) -> Result<()> {
        match self.endianness {
            Endianness::Big => self.writer.write_u64::<BigEndian>(val),
            Endianness::Little => self.writer.write_u64::<LittleEndian>(val),
        }
        .map_err(|e| IoError::FileError(e.to_string()))
    }

    fn write_f32(&mut self, val: f32) -> Result<()> {
        match self.endianness {
            Endianness::Big => self.writer.write_f32::<BigEndian>(val),
            Endianness::Little => self.writer.write_f32::<LittleEndian>(val),
        }
        .map_err(|e| IoError::FileError(e.to_string()))
    }

    fn write_f64(&mut self, val: f64) -> Result<()> {
        match self.endianness {
            Endianness::Big => self.writer.write_f64::<BigEndian>(val),
            Endianness::Little => self.writer.write_f64::<LittleEndian>(val),
        }
        .map_err(|e| IoError::FileError(e.to_string()))
    }
}

/// Write variables to an IDL save file
#[allow(dead_code)]
pub fn write_idl<P: AsRef<Path>>(path: P, variables: &HashMap<String, IdlType>) -> Result<()> {
    let mut writer = IdlWriter::new(path)?;
    writer.write_all(variables)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_idl_type_creation() {
        let byte_array = IdlType::Byte(arr1(&[1, 2, 3, 4]).into_dyn());
        if let IdlType::Byte(array) = byte_array {
            assert_eq!(array.len(), 4);
        } else {
            panic!("Expected IdlType::Byte, got {:?}", byte_array);
        }

        let double_array = IdlType::Double(arr1(&[1.0, 2.0, 3.0]).into_dyn());
        if let IdlType::Double(array) = double_array {
            assert_eq!(array.len(), 3);
        } else {
            panic!("Expected IdlType::Double, got {:?}", double_array);
        }
    }

    #[test]
    fn test_structure_creation() {
        let mut fields = HashMap::new();
        fields.insert(
            "x".to_string(),
            IdlType::Double(arr1(&[1.0, 2.0]).into_dyn()),
        );
        fields.insert(
            "y".to_string(),
            IdlType::Float(arr1(&[3.0, 4.0]).into_dyn()),
        );

        let structure = IdlStructure {
            name: "test_struct".to_string(),
            fields,
        };

        assert_eq!(structure.name, "test_struct");
        assert_eq!(structure.fields.len(), 2);
    }
}
