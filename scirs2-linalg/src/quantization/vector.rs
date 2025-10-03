//! Vector quantization types and implementations
//!
//! This module contains the QuantizedVector struct and QuantizedData1D enum
//! along with their implementations for handling quantized vector data.

use half::{bf16, f16};
use scirs2_core::ndarray::Array1;

use super::types::QuantizedDataType;

/// A vector with quantized values
#[derive(Debug, Clone)]
pub struct QuantizedVector {
    /// The quantized values can be stored in different formats
    pub data: QuantizedData1D,

    /// The original length of the vector
    pub length: usize,

    /// The data type used for quantization
    pub data_type: QuantizedDataType,
}

/// Storage for quantized 1D data (vectors) in different formats
#[derive(Debug, Clone)]
pub enum QuantizedData1D {
    /// 8-bit integer storage
    Int8(Array1<i8>),
    /// 16-bit float storage (IEEE 754 half-precision)
    Float16(Array1<f16>),
    /// 16-bit brain float storage
    BFloat16(Array1<bf16>),
}

impl QuantizedData1D {
    /// Get the number of elements in the storage
    pub fn len(&self) -> usize {
        match self {
            QuantizedData1D::Int8(arr) => arr.len(),
            QuantizedData1D::Float16(arr) => arr.len(),
            QuantizedData1D::BFloat16(arr) => arr.len(),
        }
    }

    /// Check if the storage is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Helper function to get the i8 data from a QuantizedVector if available
///
/// Returns None if the vector does not use Int8 storage
#[allow(dead_code)]
pub fn get_quantized_vector_1d_i8(vector: &QuantizedVector) -> Option<&Array1<i8>> {
    match &vector.data {
        QuantizedData1D::Int8(data) => Some(data),
        _ => None,
    }
}

impl QuantizedVector {
    /// Creates a new quantized vector with int8 storage
    pub fn new_i8(data: Array1<i8>, length: usize, datatype: QuantizedDataType) -> Self {
        Self {
            data: QuantizedData1D::Int8(data),
            length,
            data_type: datatype,
        }
    }

    /// Creates a new f16 quantized vector
    pub fn new_f16(data: Array1<f16>, length: usize) -> Self {
        Self {
            data: QuantizedData1D::Float16(data),
            length,
            data_type: QuantizedDataType::Float16,
        }
    }

    /// Creates a new bf16 quantized vector
    pub fn new_bf16(data: Array1<bf16>, length: usize) -> Self {
        Self {
            data: QuantizedData1D::BFloat16(data),
            length,
            data_type: QuantizedDataType::BFloat16,
        }
    }

    /// Creates a standard Int8 quantized vector (for backward compatibility)
    pub fn from_i8(data: Array1<i8>, length: usize) -> Self {
        Self {
            data: QuantizedData1D::Int8(data),
            length,
            data_type: QuantizedDataType::Int8,
        }
    }

    // This method stays for backward compatibility but will be deprecated in the future
    // Use get_i8 or get_f32 instead
    #[deprecated(since = "0.1.0", note = "Use get_i8 or get_f32 instead")]
    pub fn get(&self, idx: usize) -> i8 {
        self.get_i8(idx)
    }

    /// Returns the length of the vector
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns true if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get value at specified position as i8 (for int quantization)
    pub fn get_i8(&self, idx: usize) -> i8 {
        match &self.data {
            QuantizedData1D::Int8(arr) => {
                match self.data_type {
                    QuantizedDataType::Int8 => arr[idx],
                    QuantizedDataType::Int4 => {
                        let byte_idx = idx / 2;
                        let nibble_idx = idx % 2;
                        let byte = arr[byte_idx];

                        if nibble_idx == 0 {
                            // Upper 4 bits (including sign bit)
                            byte >> 4
                        } else {
                            // Lower 4 bits (including sign bit)
                            byte & 0x0F
                        }
                    }
                    QuantizedDataType::UInt4 => {
                        let byte_idx = idx / 2;
                        let nibble_idx = idx % 2;
                        let byte = arr[byte_idx];

                        if nibble_idx == 0 {
                            // Upper 4 bits (no sign bit)
                            (byte >> 4) & 0x0F
                        } else {
                            // Lower 4 bits (no sign bit)
                            byte & 0x0F
                        }
                    }
                    _ => unreachable!(
                        "Invalid quantization type for Int8 storage: expected Int8, Int4, or UInt4"
                    ),
                }
            }
            _ => unreachable!("Cannot get i8 value from floating-point quantized vector"),
        }
    }

    /// Get value at specified position as f32 (for all quantization types)
    pub fn get_f32(&self, idx: usize) -> f32 {
        match &self.data {
            QuantizedData1D::Int8(arr) => match self.data_type {
                QuantizedDataType::Int8 => arr[idx] as f32,
                QuantizedDataType::Int4 => self.get_i8(idx) as f32,
                QuantizedDataType::UInt4 => self.get_i8(idx) as f32,
                _ => unreachable!(
                    "Invalid data type for Int8 storage: expected Int8, Int4, or UInt4"
                ),
            },
            QuantizedData1D::Float16(arr) => arr[idx].to_f32(),
            QuantizedData1D::BFloat16(arr) => arr[idx].to_f32(),
        }
    }
}
