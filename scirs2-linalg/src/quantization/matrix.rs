//! Matrix quantization types and implementations
//!
//! This module contains the QuantizedMatrix struct and QuantizedData2D enum
//! along with their implementations for handling quantized matrix data.

use half::{bf16, f16};
use ndarray::{Array1, Array2};

use super::types::QuantizedDataType;

/// A matrix with quantized values
#[derive(Debug, Clone)]
pub struct QuantizedMatrix {
    /// The quantized values can be stored in different formats
    pub data: QuantizedData2D,

    /// The original shape of the matrix
    pub shape: (usize, usize),

    /// The data type used for quantization
    pub data_type: QuantizedDataType,
}

/// Storage for quantized 2D data (matrices) in different formats
#[derive(Debug, Clone)]
pub enum QuantizedData2D {
    /// 8-bit integer storage
    Int8(Array2<i8>),
    /// 16-bit float storage (IEEE 754 half-precision)
    Float16(Array2<f16>),
    /// 16-bit brain float storage
    BFloat16(Array2<bf16>),
}

impl QuantizedData2D {
    /// Get the number of elements in the storage
    pub fn len(&self) -> usize {
        match self {
            QuantizedData2D::Int8(arr) => arr.len(),
            QuantizedData2D::Float16(arr) => arr.len(),
            QuantizedData2D::BFloat16(arr) => arr.len(),
        }
    }

    /// Check if the storage is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Helper function to get the i8 data from a QuantizedMatrix if available
///
/// Returns None if the matrix does not use Int8 storage
#[allow(dead_code)]
pub fn get_quantizedmatrix_2d_i8(matrix: &QuantizedMatrix) -> Option<&Array2<i8>> {
    match &matrix.data {
        QuantizedData2D::Int8(data) => Some(data),
        _ => None,
    }
}

impl QuantizedMatrix {
    /// Creates a new quantized matrix with int8 storage
    pub fn new_i8(data: Array2<i8>, shape: (usize, usize), data_type: QuantizedDataType) -> Self {
        Self {
            data: QuantizedData2D::Int8(data),
            shape,
            data_type,
        }
    }

    /// Creates a new f16 quantized matrix
    pub fn new_f16(data: Array2<f16>, shape: (usize, usize)) -> Self {
        Self {
            data: QuantizedData2D::Float16(data),
            shape,
            data_type: QuantizedDataType::Float16,
        }
    }

    /// Creates a new bf16 quantized matrix
    pub fn new_bf16(data: Array2<bf16>, shape: (usize, usize)) -> Self {
        Self {
            data: QuantizedData2D::BFloat16(data),
            shape,
            data_type: QuantizedDataType::BFloat16,
        }
    }

    /// Creates a standard Int8 quantized matrix (for backward compatibility)
    pub fn from_i8(data: Array2<i8>, shape: (usize, usize)) -> Self {
        Self {
            data: QuantizedData2D::Int8(data),
            shape,
            data_type: QuantizedDataType::Int8,
        }
    }

    // This method stays for backward compatibility but will be deprecated in the future
    // Use get_i8 or get_f32 instead
    #[deprecated(since = "0.1.0", note = "Use get_i8 or get_f32 instead")]
    pub fn get(&self, row: usize, col: usize) -> i8 {
        self.get_i8(row, col)
    }

    /// Returns the shape of the matrix
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Returns the number of rows in the matrix
    pub fn nrows(&self) -> usize {
        self.shape.0
    }

    /// Returns the number of columns in the matrix
    pub fn ncols(&self) -> usize {
        self.shape.1
    }

    /// Get value at specified position as i8 (for int quantization)
    pub fn get_i8(&self, row: usize, col: usize) -> i8 {
        match &self.data {
            QuantizedData2D::Int8(arr) => {
                match self.data_type {
                    QuantizedDataType::Int8 => arr[[row, col]],
                    QuantizedDataType::Int4 => {
                        let idx = row * self.shape.1 + col;
                        let byte_idx = idx / 2;
                        let nibble_idx = idx % 2;
                        let byte = arr.as_slice().unwrap()[byte_idx];

                        if nibble_idx == 0 {
                            // Upper 4 bits
                            byte >> 4
                        } else {
                            // Lower 4 bits
                            byte & 0x0F
                        }
                    }
                    QuantizedDataType::UInt4 => {
                        let idx = row * self.shape.1 + col;
                        let byte_idx = idx / 2;
                        let nibble_idx = idx % 2;
                        let byte = arr.as_slice().unwrap()[byte_idx];

                        if nibble_idx == 0 {
                            // Upper 4 bits
                            (byte >> 4) & 0x0F
                        } else {
                            // Lower 4 bits
                            byte & 0x0F
                        }
                    }
                    _ => unreachable!(
                        "Invalid quantization type for Int8 storage: expected Int8, Int4, or UInt4"
                    ),
                }
            }
            _ => unreachable!("Cannot get i8 value from floating-point quantized matrix"),
        }
    }

    /// Get value at specified position as f32 (for all quantization types)
    pub fn get_f32(&self, row: usize, col: usize) -> f32 {
        match &self.data {
            QuantizedData2D::Int8(arr) => match self.data_type {
                QuantizedDataType::Int8 => arr[[row, col]] as f32,
                QuantizedDataType::Int4 => self.get_i8(row, col) as f32,
                QuantizedDataType::UInt4 => self.get_i8(row, col) as f32,
                _ => unreachable!(
                    "Invalid data type for Int8 storage: expected Int8, Int4, or UInt4"
                ),
            },
            QuantizedData2D::Float16(arr) => arr[[row, col]].to_f32(),
            QuantizedData2D::BFloat16(arr) => arr[[row, col]].to_f32(),
        }
    }
}
