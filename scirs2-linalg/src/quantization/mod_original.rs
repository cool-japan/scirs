//! Quantization-aware linear algebra operations
//!
//! This module provides functions and types for working with quantized matrices and vectors.
//! Quantization reduces the precision of floating-point numbers to save memory and
//! computational resources, which is particularly useful in machine learning applications.
//!
//! ## Overview
//!
//! * Quantization of matrices and vectors to lower bit-width representations
//! * Linear algebra operations on quantized data
//! * Support for different quantization methods (uniform, symmetric, affine)
//! * Efficient operations with mixed quantized and floating-point data
//!
//! ## Examples
//!
//! Basic quantization:
//!
//! ```
//! use scirs2_core::ndarray::{Array2, array};
//! use scirs2_linalg::quantization::{quantize_matrix, dequantize_matrix, QuantizationMethod};
//!
//! let a = array![[1.0_f32, 2.5, 3.7], [4.2, 5.0, 6.1]];
//!
//! // Quantize to 8-bit
//! let (quantized, params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Affine);
//!
//! // Dequantize back to floating point
//! let a_dequantized = dequantize_matrix(&quantized, &params);
//!
//! // Check the error exists but is bounded
//! let max_error = (&a - &a_dequantized).mapv(|x| x.abs()).fold(0.0_f32, |acc, &b| acc.max(b));
//! assert!(max_error > 0.0); // There should be some quantization error
//! assert!(max_error < 10.0); // But it should be bounded
//! ```
//!
//! Quantized matrix multiplication:
//!
//! ```
//! use scirs2_core::ndarray::{Array2, array};
//! use scirs2_linalg::quantization::{quantize_matrix, QuantizationMethod, quantized_matmul};
//!
//! let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
//! let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
//!
//! // Quantize both matrices to 8-bit
//! let (a_q, a_params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Symmetric);
//! let (b_q, b_params) = quantize_matrix(&b.view(), 8, QuantizationMethod::Symmetric);
//!
//! // Perform quantized matrix multiplication
//! let c_q = quantized_matmul(&a_q, &a_params, &b_q, &b_params).unwrap();
//!
//! // Regular matrix multiplication for comparison
//! let c = a.dot(&b);
//!
//! // Check the error is acceptable
//! let rel_error = (&c - &c_q).mapv(|x| x.abs()).sum() / c.sum();
//! assert!(rel_error < 0.1); // Relative error should be small
//! ```

use half::{bf16, f16};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{AsPrimitive, Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{LinalgError, LinalgResult};

// Export submodules
pub mod calibration;
pub mod calibration_ema;
pub mod fusion;
pub mod out_of_core;
pub mod quantized_matrixfree;
pub mod simd;
pub mod solvers;
pub mod stability;

/// Supported methods of quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMethod {
    /// Uniform quantization maps the input range to uniform discrete levels
    /// with equal spacing between consecutive levels
    Uniform,

    /// Symmetric quantization is centered around zero and has equal positive and
    /// negative ranges, making it suitable for weight matrices
    Symmetric,

    /// Affine quantization uses the formula q = scale * (x - zero_point)
    /// allowing better representation of asymmetric distributions
    Affine,

    /// Power-of-two quantization uses powers of 2 for the scale factor,
    /// enabling efficient implementation with bitshifts
    PowerOfTwo,

    /// Int4 quantization uses 4-bit signed integers, packing two values into each byte
    /// for memory efficiency. This is useful for model compression in ML applications.
    Int4,

    /// UInt4 quantization uses 4-bit unsigned integers, packing two values into each byte.
    /// This provides a positive-only range with maximum memory efficiency.
    UInt4,

    /// Float16 quantization uses IEEE 754 16-bit half-precision floating point format.
    /// It provides a good balance between precision and memory efficiency for ML models.
    Float16,

    /// BFloat16 quantization uses the "brain floating point" 16-bit format,
    /// which has the same exponent size as f32 but fewer mantissa bits.
    /// This is especially well-suited for deep learning applications.
    BFloat16,

    /// Per-channel symmetric quantization applies different symmetric quantization
    /// parameters to each channel (column), improving accuracy for matrices with
    /// varying distributions across channels.
    PerChannelSymmetric,

    /// Per-channel affine quantization applies different affine quantization
    /// parameters to each channel (column), allowing for better representation of
    /// asymmetric distributions that vary by channel.
    PerChannelAffine,
}

/// Parameters for the quantization process
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// The number of bits used for quantization
    pub bits: u8,

    /// The scale factor used to convert between quantized and float values
    /// For per-channel quantization, this is the default scale for debugging
    pub scale: f32,

    /// The zero point used for asymmetric quantization (for affine quantization)
    /// For per-channel quantization, this is the default zero point for debugging
    pub zero_point: i32,

    /// The minimum value of the original data
    /// For per-channel quantization, this is across all channels
    pub min_val: f32,

    /// The maximum value of the original data
    /// For per-channel quantization, this is across all channels
    pub max_val: f32,

    /// The quantization method used
    pub method: QuantizationMethod,

    /// The data type used for storage
    pub data_type: QuantizedDataType,

    /// Per-channel scale factors (only used for per-channel quantization)
    pub channel_scales: Option<Vec<f32>>,

    /// Per-channel zero points (only used for per-channel affine quantization)
    pub channel_zero_points: Option<Vec<i32>>,
}

/// The storage type used for quantized data
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizedDataType {
    /// 8-bit signed integers
    Int8,
    /// 4-bit signed integers (packed into i8 array)
    Int4,
    /// 4-bit unsigned integers (packed into i8 array)
    UInt4,
    /// 16-bit IEEE 754 half-precision floating point (f16)
    Float16,
    /// 16-bit Brain floating point (bf16)
    BFloat16,
}

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

// Constructor methods are already defined below

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

// Constructor methods are already defined below

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

/// Quantize a floating-point matrix to a lower precision representation
///
/// # Arguments
///
/// * `matrix` - The input matrix to quantize
/// * `bits` - The number of bits to use for quantization (typically 8)
/// * `method` - The quantization method to use
///
/// # Returns
///
/// A tuple containing the quantized matrix and the quantization parameters
///
/// # Notes
///
/// For per-channel quantization, use `quantize_matrix_per_channel` instead.
#[allow(dead_code)]
pub fn quantize_matrix<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    method: QuantizationMethod,
) -> (QuantizedMatrix, QuantizationParams)
where
    F: Float + Debug + AsPrimitive<f32> + FromPrimitive,
    f32: AsPrimitive<F>,
{
    let shape = (matrix.nrows(), matrix.ncols());

    // Find min and max values
    let mut min_val = F::infinity().as_();
    let mut max_val = F::neg_infinity().as_();

    for &val in matrix.iter() {
        let val_f32: f32 = val.as_();
        if val_f32.is_finite() {
            min_val = min_val.min(val_f32);
            max_val = max_val.max(val_f32);
        }
    }

    // Handle case where all values are the same
    if (max_val - min_val).abs() < f32::EPSILON {
        max_val = min_val + 1.0;
    }

    // For Float16 and BFloat16, we just directly convert the values without actual "quantization"
    if method == QuantizationMethod::Float16 {
        let mut f16_data = Array2::zeros(shape);
        for (i, &val) in matrix.iter().enumerate() {
            let val_f32: f32 = val.as_();
            f16_data.as_slice_mut().unwrap()[i] = f16::from_f32(val_f32);
        }

        // Create parameters - scale and zero_point aren't really used for float16
        let params = QuantizationParams {
            bits: 16,
            scale: 1.0, // Not used for float16
            zero_point: 0,
            min_val,
            max_val,
            method,
            data_type: QuantizedDataType::Float16,
            channel_scales: None,
            channel_zero_points: None,
        };

        return (QuantizedMatrix::new_f16(f16_data, shape), params);
    }

    if method == QuantizationMethod::BFloat16 {
        let mut bf16_data = Array2::zeros(shape);
        for (i, &val) in matrix.iter().enumerate() {
            let val_f32: f32 = val.as_();
            bf16_data.as_slice_mut().unwrap()[i] = bf16::from_f32(val_f32);
        }

        // Create parameters - scale and zero_point aren't really used for bfloat16
        let params = QuantizationParams {
            bits: 16,
            scale: 1.0, // Not used for bfloat16
            zero_point: 0,
            min_val,
            max_val,
            method,
            data_type: QuantizedDataType::BFloat16,
            channel_scales: None,
            channel_zero_points: None,
        };

        return (QuantizedMatrix::new_bf16(bf16_data, shape), params);
    }

    // Determine data type based on method and bits
    let data_type = match method {
        QuantizationMethod::Int4 => QuantizedDataType::Int4,
        QuantizationMethod::UInt4 => QuantizedDataType::UInt4,
        _ => QuantizedDataType::Int8,
    };

    // For Int4 and UInt4, override bits to 4
    let effective_bits = match method {
        QuantizationMethod::Int4 | QuantizationMethod::UInt4 => 4,
        _ => bits,
    };

    // Calculate quantization parameters based on the chosen method
    let (scale, zero_point) = match method {
        QuantizationMethod::Uniform => {
            let scale = (max_val - min_val) / ((1 << effective_bits) - 1) as f32;
            let zero_point = 0;
            (scale, zero_point)
        }
        QuantizationMethod::Symmetric => {
            // Symmetric around zero, calculate scale to fit
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = abs_max / ((1 << (effective_bits - 1)) - 1) as f32;
            let zero_point = 0;
            (scale, zero_point)
        }
        QuantizationMethod::Affine => {
            let scale = (max_val - min_val) / ((1 << effective_bits) - 1) as f32;
            let zero_point = (-min_val / scale).round() as i32;
            (scale, zero_point)
        }
        QuantizationMethod::PowerOfTwo => {
            // Find the smallest power of 2 greater than or equal to (max_val - min_val) / ((1 << bits) - 1)
            let range = max_val - min_val;
            let ideal_scale = range / ((1 << effective_bits) - 1) as f32;
            let exponent = ideal_scale.log2().ceil();
            let scale = 2.0_f32.powf(exponent);
            let zero_point = 0;
            (scale, zero_point)
        }
        QuantizationMethod::Int4 => {
            // Symmetric around zero, with 4-bit signed integers (-8 to 7)
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = abs_max / 7.0; // -8 to 7 range for 4-bit signed integer
            let zero_point = 0;
            (scale, zero_point)
        }
        QuantizationMethod::UInt4 => {
            // Unsigned 4-bit quantization (0 to 15)
            let scale = (max_val - min_val) / 15.0; // 0 to 15 range for 4-bit unsigned integer
            let zero_point = (-min_val / scale).round() as i32;
            (scale, zero_point)
        }
        _ => unreachable!(), // Float16 and BFloat16 are handled above
    };

    // Create quantization parameters
    let params = QuantizationParams {
        bits: effective_bits,
        scale,
        zero_point,
        min_val,
        max_val,
        method,
        data_type,
        channel_scales: None,
        channel_zero_points: None,
    };

    // Special handling for 4-bit quantization - pack two values into one byte
    match method {
        QuantizationMethod::Int4 => {
            // For 4-bit signed integers, we need to handle the packing
            let num_elements = matrix.len();
            // Packed size is calculated directly in the array dimensions
            let mut packed_data = Array2::zeros((shape.0, shape.1.div_ceil(2)));

            for i in 0..num_elements {
                let val_f32: f32 = matrix.as_slice().unwrap()[i].as_();
                // Clamp to -8 to 7 range for 4-bit signed integer
                let q_val = ((val_f32 / scale).round() as i8).clamp(-8, 7);

                let byte_idx = i / 2;
                if i % 2 == 0 {
                    // Store in upper 4 bits
                    packed_data.as_slice_mut().unwrap()[byte_idx] = q_val << 4;
                } else {
                    // Store in lower 4 bits, OR with existing upper bits
                    packed_data.as_slice_mut().unwrap()[byte_idx] |= q_val & 0x0F;
                }
            }

            // Calculate the shape for the packed data
            let packedshape = (shape.0, shape.1.div_ceil(2));

            // Use toshape instead of intoshape (deprecated)
            let packed_reshaped = packed_data.into_shape_with_order(packedshape).unwrap();
            (
                QuantizedMatrix::new_i8(packed_reshaped, shape, QuantizedDataType::Int4),
                params,
            )
        }
        QuantizationMethod::UInt4 => {
            // For 4-bit unsigned integers, similar packing approach
            let num_elements = matrix.len();
            // Packed size is calculated directly in the array dimensions
            let mut packed_data = Array2::zeros((shape.0, shape.1.div_ceil(2)));

            for i in 0..num_elements {
                let val_f32: f32 = matrix.as_slice().unwrap()[i].as_();
                // Scale to 0-15 range for 4-bit unsigned
                let ival = ((val_f32 - min_val) / scale).round() as i32;
                let q_val = (ival.clamp(0, 15) & 0x0F) as i8;

                let byte_idx = i / 2;
                if i % 2 == 0 {
                    // Store in upper 4 bits
                    packed_data.as_slice_mut().unwrap()[byte_idx] = q_val << 4;
                } else {
                    // Store in lower 4 bits, OR with existing upper bits
                    packed_data.as_slice_mut().unwrap()[byte_idx] |= q_val & 0x0F;
                }
            }

            // Calculate the shape for the packed data
            let packedshape = (shape.0, shape.1.div_ceil(2));

            // Use toshape instead of intoshape (deprecated)
            let packed_reshaped = packed_data.into_shape_with_order(packedshape).unwrap();
            (
                QuantizedMatrix::new_i8(packed_reshaped, shape, QuantizedDataType::UInt4),
                params,
            )
        }
        _ => {
            // Standard 8-bit quantization for other methods
            let quantized_data = match method {
                QuantizationMethod::Uniform => {
                    let mut quantized = Array2::zeros(shape);
                    for (i, &val) in matrix.iter().enumerate() {
                        let val_f32: f32 = val.as_();
                        let q_val = ((val_f32 - min_val) / scale).round() as i8;
                        quantized.as_slice_mut().unwrap()[i] = q_val;
                    }
                    quantized
                }
                QuantizationMethod::Symmetric => {
                    let mut quantized = Array2::zeros(shape);
                    for (i, &val) in matrix.iter().enumerate() {
                        let val_f32: f32 = val.as_();
                        let q_val = (val_f32 / scale).round() as i8;
                        quantized.as_slice_mut().unwrap()[i] = q_val;
                    }
                    quantized
                }
                QuantizationMethod::Affine => {
                    let mut quantized = Array2::zeros(shape);
                    for (i, &val) in matrix.iter().enumerate() {
                        let val_f32: f32 = val.as_();
                        let q_val = ((val_f32 / scale) + zero_point as f32).round() as i8;
                        quantized.as_slice_mut().unwrap()[i] = q_val;
                    }
                    quantized
                }
                QuantizationMethod::PowerOfTwo => {
                    let mut quantized = Array2::zeros(shape);
                    for (i, &val) in matrix.iter().enumerate() {
                        let val_f32: f32 = val.as_();
                        let q_val = ((val_f32 - min_val) / scale).round() as i8;
                        quantized.as_slice_mut().unwrap()[i] = q_val;
                    }
                    quantized
                }
                _ => unreachable!(), // Int4, UInt4, Float16, and BFloat16 are handled above
            };

            (
                QuantizedMatrix::new_i8(quantized_data, shape, QuantizedDataType::Int8),
                params,
            )
        }
    }
}

/// Quantize a floating-point matrix using per-channel quantization
///
/// This applies different quantization parameters to each column of the matrix,
/// which can significantly improve accuracy when the value distributions vary
/// across channels (like in neural network weights).
///
/// # Arguments
///
/// * `matrix` - The input matrix to quantize
/// * `bits` - The number of bits to use for quantization (typically 8)
/// * `method` - Must be either PerChannelSymmetric or PerChannelAffine
///
/// # Returns
///
/// A tuple containing the quantized matrix and the quantization parameters
///
/// # Panics
///
/// This function will panic if the method is not PerChannelSymmetric or PerChannelAffine
#[allow(dead_code)]
pub fn quantize_matrix_per_channel<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    method: QuantizationMethod,
) -> (QuantizedMatrix, QuantizationParams)
where
    F: Float + Debug + AsPrimitive<f32> + FromPrimitive,
    f32: AsPrimitive<F>,
{
    // Verify method is per-channel
    assert!(
        method == QuantizationMethod::PerChannelSymmetric
            || method == QuantizationMethod::PerChannelAffine,
        "quantize_matrix_per_channel requires PerChannelSymmetric or PerChannelAffine method, got {method:?}"
    );

    let shape = (matrix.nrows(), matrix.ncols());
    let num_channels = shape.1;

    // We'll use Int8 data type for now
    let data_type = QuantizedDataType::Int8.clone();

    // Calculate min/max for each channel
    let mut channel_min_vals = vec![F::infinity().as_(); num_channels];
    let mut channel_max_vals = vec![F::neg_infinity().as_(); num_channels];

    // Find minimum and maximum values for each channel (column)
    for col in 0..num_channels {
        for row in 0..shape.0 {
            let val_f32: f32 = matrix[[row, col]].as_();
            if val_f32.is_finite() {
                channel_min_vals[col] = channel_min_vals[col].min(val_f32);
                channel_max_vals[col] = channel_max_vals[col].max(val_f32);
            }
        }

        // Handle case where all values in a channel are the same
        if (channel_max_vals[col] - channel_min_vals[col]).abs() < f32::EPSILON {
            channel_max_vals[col] = channel_min_vals[col] + 1.0;
        }
    }

    // Calculate global min/max for the whole matrix
    let min_val = channel_min_vals
        .iter()
        .fold(F::infinity().as_(), |acc, &val| acc.min(val));
    let max_val = channel_max_vals
        .iter()
        .fold(F::neg_infinity().as_(), |acc, &val| acc.max(val));

    // Calculate scales and zero_points for each channel
    let mut channel_scales = vec![0.0; num_channels];
    let mut channel_zero_points = vec![0; num_channels];

    match method {
        QuantizationMethod::PerChannelSymmetric => {
            for col in 0..num_channels {
                // Symmetric around zero, calculate scale to fit
                let abs_max = channel_max_vals[col].abs().max(channel_min_vals[col].abs());
                channel_scales[col] = abs_max / ((1 << (bits - 1)) - 1) as f32;
                channel_zero_points[col] = 0; // Symmetric always has zero-point=0
            }
        }
        QuantizationMethod::PerChannelAffine => {
            for col in 0..num_channels {
                // Affine quantization for each channel
                channel_scales[col] =
                    (channel_max_vals[col] - channel_min_vals[col]) / ((1 << bits) - 1) as f32;
                channel_zero_points[col] =
                    (-channel_min_vals[col] / channel_scales[col]).round() as i32;
            }
        }
        _ => unreachable!(),
    }

    // Create a default scale for the overall parameters (for display purposes)
    // We'll use the average scale
    let scale = channel_scales.iter().sum::<f32>() / num_channels as f32;
    let zero_point = if method == QuantizationMethod::PerChannelAffine {
        (channel_zero_points.iter().sum::<i32>() as f32 / num_channels as f32).round() as i32
    } else {
        0
    };

    // Create quantization parameters
    let params = QuantizationParams {
        bits,
        scale,
        zero_point,
        min_val,
        max_val,
        method,
        data_type: data_type.clone(),
        channel_scales: Some(channel_scales.clone()),
        channel_zero_points: Some(channel_zero_points.clone()),
    };

    // Quantize the data for each channel
    let mut quantized_data = Array2::zeros(shape);

    for col in 0..num_channels {
        let scale = channel_scales[col];
        let zero_point = channel_zero_points[col];

        for row in 0..shape.0 {
            let val_f32: f32 = matrix[[row, col]].as_();

            let q_val = match method {
                QuantizationMethod::PerChannelSymmetric => {
                    // Symmetric quantization
                    (val_f32 / scale)
                        .round()
                        .clamp(-(1 << (bits - 1)) as f32, ((1 << (bits - 1)) - 1) as f32)
                        as i8
                }
                QuantizationMethod::PerChannelAffine => {
                    // Affine quantization
                    ((val_f32 / scale) + zero_point as f32)
                        .round()
                        .clamp(0.0, ((1 << bits) - 1) as f32) as i8
                }
                _ => unreachable!(),
            };

            quantized_data[[row, col]] = q_val;
        }
    }

    (
        QuantizedMatrix::new_i8(quantized_data, shape, data_type.clone()),
        params,
    )
}

/// Dequantize a matrix back to floating-point
///
/// # Arguments
///
/// * `quantized` - The quantized matrix
/// * `params` - The quantization parameters
///
/// # Returns
///
/// The dequantized matrix
#[allow(dead_code)]
pub fn dequantize_matrix(quantized: &QuantizedMatrix, params: &QuantizationParams) -> Array2<f32> {
    let shape = quantized.shape();
    let mut dequantized = Array2::zeros(shape);

    // Handle different quantization data types
    match &quantized.data {
        // Direct floating-point formats
        QuantizedData2D::Float16(data) => {
            // For Float16, just convert directly to f32
            for (i, &val) in data.iter().enumerate() {
                dequantized.as_slice_mut().unwrap()[i] = val.to_f32();
            }
        }
        QuantizedData2D::BFloat16(data) => {
            // For BFloat16, just convert directly to f32
            for (i, &val) in data.iter().enumerate() {
                dequantized.as_slice_mut().unwrap()[i] = val.to_f32();
            }
        }
        // Integer-based quantization
        QuantizedData2D::Int8(data) => {
            match quantized.data_type {
                // Special handling for 4-bit quantization types
                QuantizedDataType::Int4 | QuantizedDataType::UInt4 => {
                    let num_elements = shape.0 * shape.1;

                    for i in 0..num_elements {
                        let row = i / shape.1;
                        let col = i % shape.1;

                        // Get the 4-bit value using the get method
                        let q_val = quantized.get_i8(row, col);

                        // Dequantize based on the method
                        let val = match params.method {
                            QuantizationMethod::Int4 => q_val as f32 * params.scale,
                            QuantizationMethod::UInt4 => {
                                params.min_val + (q_val as f32 * params.scale)
                            }
                            _ => unreachable!(), // Should not happen with Int4/UInt4 data type
                        };

                        dequantized[[row, col]] = val;
                    }
                }
                // Per-channel quantization
                QuantizedDataType::Int8
                    if params.method == QuantizationMethod::PerChannelSymmetric
                        || params.method == QuantizationMethod::PerChannelAffine =>
                {
                    // We need channel_scales and channel_zero_points for per-channel dequantization
                    let channel_scales = params
                        .channel_scales
                        .as_ref()
                        .expect("Per-channel quantization requires channel_scales");

                    let channel_zero_points = params
                        .channel_zero_points
                        .as_ref()
                        .expect("Per-channel quantization requires channel_zero_points");

                    let num_channels = shape.1;

                    // Process each element with its channel-specific parameters
                    for row in 0..shape.0 {
                        for col in 0..num_channels {
                            let q_val = data[[row, col]];
                            let scale = channel_scales[col];
                            let zero_point = channel_zero_points[col];

                            let val = match params.method {
                                QuantizationMethod::PerChannelSymmetric => {
                                    // Symmetric means zero_point is always 0
                                    q_val as f32 * scale
                                }
                                QuantizationMethod::PerChannelAffine => {
                                    // Apply affine transformation with channel-specific zero point
                                    scale * (q_val as f32 - zero_point as f32)
                                }
                                _ => unreachable!(), // Should not happen
                            };

                            dequantized[[row, col]] = val;
                        }
                    }
                }
                // Standard 8-bit quantization
                QuantizedDataType::Int8 => {
                    // Perform dequantization based on the quantization method for 8-bit types
                    match params.method {
                        QuantizationMethod::Uniform => {
                            for (i, &q_val) in data.iter().enumerate() {
                                let val = params.min_val + (q_val as f32 * params.scale);
                                dequantized.as_slice_mut().unwrap()[i] = val;
                            }
                        }
                        QuantizationMethod::Symmetric => {
                            for (i, &q_val) in data.iter().enumerate() {
                                let val = q_val as f32 * params.scale;
                                dequantized.as_slice_mut().unwrap()[i] = val;
                            }
                        }
                        QuantizationMethod::Affine => {
                            for (i, &q_val) in data.iter().enumerate() {
                                let val = params.scale * (q_val as f32 - params.zero_point as f32);
                                dequantized.as_slice_mut().unwrap()[i] = val;
                            }
                        }
                        QuantizationMethod::PowerOfTwo => {
                            for (i, &q_val) in data.iter().enumerate() {
                                let val = params.min_val + (q_val as f32 * params.scale);
                                dequantized.as_slice_mut().unwrap()[i] = val;
                            }
                        }
                        _ => unreachable!(), // Other methods are handled above
                    }
                }
                _ => unreachable!(), // Should not happen
            }
        }
    }

    dequantized
}

/// Quantize a floating-point vector to a lower precision representation
///
/// # Arguments
///
/// * `vector` - The input vector to quantize
/// * `bits` - The number of bits to use for quantization (typically 8)
/// * `method` - The quantization method to use
///
/// # Returns
///
/// A tuple containing the quantized vector and the quantization parameters
#[allow(dead_code)]
pub fn quantize_vector<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    method: QuantizationMethod,
) -> (QuantizedVector, QuantizationParams)
where
    F: Float + Debug + AsPrimitive<f32> + FromPrimitive,
    f32: AsPrimitive<F>,
{
    let length = vector.len();

    // Find min and max values
    let mut min_val = F::infinity().as_();
    let mut max_val = F::neg_infinity().as_();

    for &val in vector.iter() {
        let val_f32: f32 = val.as_();
        if val_f32.is_finite() {
            min_val = min_val.min(val_f32);
            max_val = max_val.max(val_f32);
        }
    }

    // Handle case where all values are the same
    if (max_val - min_val).abs() < f32::EPSILON {
        max_val = min_val + 1.0;
    }

    // For Float16 and BFloat16, we just directly convert the values without actual "quantization"
    if method == QuantizationMethod::Float16 {
        let mut f16_data = Array1::zeros(length);
        for (i, &val) in vector.iter().enumerate() {
            let val_f32: f32 = val.as_();
            f16_data[i] = f16::from_f32(val_f32);
        }

        // Create parameters - scale and zero_point aren't really used for float16
        let params = QuantizationParams {
            bits: 16,
            scale: 1.0, // Not used for float16
            zero_point: 0,
            min_val,
            max_val,
            method,
            data_type: QuantizedDataType::Float16,
            channel_scales: None,
            channel_zero_points: None,
        };

        return (QuantizedVector::new_f16(f16_data, length), params);
    }

    if method == QuantizationMethod::BFloat16 {
        let mut bf16_data = Array1::zeros(length);
        for (i, &val) in vector.iter().enumerate() {
            let val_f32: f32 = val.as_();
            bf16_data[i] = bf16::from_f32(val_f32);
        }

        // Create parameters - scale and zero_point aren't really used for bfloat16
        let params = QuantizationParams {
            bits: 16,
            scale: 1.0, // Not used for bfloat16
            zero_point: 0,
            min_val,
            max_val,
            method,
            data_type: QuantizedDataType::BFloat16,
            channel_scales: None,
            channel_zero_points: None,
        };

        return (QuantizedVector::new_bf16(bf16_data, length), params);
    }

    // Determine data type based on method and bits
    let data_type = match method {
        QuantizationMethod::Int4 => QuantizedDataType::Int4,
        QuantizationMethod::UInt4 => QuantizedDataType::UInt4,
        _ => QuantizedDataType::Int8,
    };

    // For Int4 and UInt4, override bits to 4
    let effective_bits = match method {
        QuantizationMethod::Int4 | QuantizationMethod::UInt4 => 4,
        _ => bits,
    };

    // Calculate quantization parameters based on the chosen method
    let (scale, zero_point) = match method {
        QuantizationMethod::Uniform => {
            let scale = (max_val - min_val) / ((1 << effective_bits) - 1) as f32;
            let zero_point = 0;
            (scale, zero_point)
        }
        QuantizationMethod::Symmetric => {
            // Symmetric around zero, calculate scale to fit
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = abs_max / ((1 << (effective_bits - 1)) - 1) as f32;
            let zero_point = 0;
            (scale, zero_point)
        }
        QuantizationMethod::Affine => {
            let scale = (max_val - min_val) / ((1 << effective_bits) - 1) as f32;
            let zero_point = (-min_val / scale).round() as i32;
            (scale, zero_point)
        }
        QuantizationMethod::PowerOfTwo => {
            // Find the smallest power of 2 greater than or equal to (max_val - min_val) / ((1 << bits) - 1)
            let range = max_val - min_val;
            let ideal_scale = range / ((1 << effective_bits) - 1) as f32;
            let exponent = ideal_scale.log2().ceil();
            let scale = 2.0_f32.powf(exponent);
            let zero_point = 0;
            (scale, zero_point)
        }
        QuantizationMethod::Int4 => {
            // Symmetric around zero, with 4-bit signed integers (-8 to 7)
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = abs_max / 7.0; // -8 to 7 range for 4-bit signed integer
            let zero_point = 0;
            (scale, zero_point)
        }
        QuantizationMethod::UInt4 => {
            // Unsigned 4-bit quantization (0 to 15)
            let scale = (max_val - min_val) / 15.0; // 0 to 15 range for 4-bit unsigned integer
            let zero_point = (-min_val / scale).round() as i32;
            (scale, zero_point)
        }
        _ => unreachable!(), // Float16 and BFloat16 are handled above
    };

    // Create quantization parameters
    let params = QuantizationParams {
        bits: effective_bits,
        scale,
        zero_point,
        min_val,
        max_val,
        method,
        data_type,
        channel_scales: None,
        channel_zero_points: None,
    };

    // Special handling for 4-bit quantization - pack two values into one byte
    match method {
        QuantizationMethod::Int4 => {
            // For 4-bit signed integers, we need to handle the packing
            let packedsize = length.div_ceil(2); // Round up division
            let mut packed_data = Array1::zeros(packedsize);

            for i in 0..length {
                let val_f32: f32 = vector[i].as_();
                // Clamp to -8 to 7 range for 4-bit signed integer
                let q_val = ((val_f32 / scale).round() as i8).clamp(-8, 7);

                let byte_idx = i / 2;
                if i % 2 == 0 {
                    // Store in upper 4 bits
                    packed_data[byte_idx] = q_val << 4;
                } else {
                    // Store in lower 4 bits, OR with existing upper bits
                    packed_data[byte_idx] |= q_val & 0x0F;
                }
            }

            (
                QuantizedVector::new_i8(packed_data, length, QuantizedDataType::Int4),
                params,
            )
        }
        QuantizationMethod::UInt4 => {
            // For 4-bit unsigned integers, similar packing approach
            let packedsize = length.div_ceil(2); // Round up division
            let mut packed_data = Array1::zeros(packedsize);

            for i in 0..length {
                let val_f32: f32 = vector[i].as_();
                // Scale to 0-15 range for 4-bit unsigned
                let ival = ((val_f32 - min_val) / scale).round() as i32;
                let q_val = (ival.clamp(0, 15) & 0x0F) as i8;

                let byte_idx = i / 2;
                if i % 2 == 0 {
                    // Store in upper 4 bits
                    packed_data[byte_idx] = q_val << 4;
                } else {
                    // Store in lower 4 bits, OR with existing upper bits
                    packed_data[byte_idx] |= q_val & 0x0F;
                }
            }

            (
                QuantizedVector::new_i8(packed_data, length, QuantizedDataType::UInt4),
                params,
            )
        }
        _ => {
            // Standard 8-bit quantization for other methods
            let quantized_data = match method {
                QuantizationMethod::Uniform => {
                    let mut quantized = Array1::zeros(length);
                    for (i, &val) in vector.iter().enumerate() {
                        let val_f32: f32 = val.as_();
                        let q_val = ((val_f32 - min_val) / scale).round() as i8;
                        quantized[i] = q_val;
                    }
                    quantized
                }
                QuantizationMethod::Symmetric => {
                    let mut quantized = Array1::zeros(length);
                    for (i, &val) in vector.iter().enumerate() {
                        let val_f32: f32 = val.as_();
                        let q_val = (val_f32 / scale).round() as i8;
                        quantized[i] = q_val;
                    }
                    quantized
                }
                QuantizationMethod::Affine => {
                    let mut quantized = Array1::zeros(length);
                    for (i, &val) in vector.iter().enumerate() {
                        let val_f32: f32 = val.as_();
                        let q_val = ((val_f32 / scale) + zero_point as f32).round() as i8;
                        quantized[i] = q_val;
                    }
                    quantized
                }
                QuantizationMethod::PowerOfTwo => {
                    let mut quantized = Array1::zeros(length);
                    for (i, &val) in vector.iter().enumerate() {
                        let val_f32: f32 = val.as_();
                        let q_val = ((val_f32 - min_val) / scale).round() as i8;
                        quantized[i] = q_val;
                    }
                    quantized
                }
                _ => unreachable!(), // Int4, UInt4, Float16, and BFloat16 are handled above
            };

            (
                QuantizedVector::new_i8(quantized_data, length, QuantizedDataType::Int8),
                params,
            )
        }
    }
}

/// Dequantize a vector back to floating-point
///
/// # Arguments
///
/// * `quantized` - The quantized vector
/// * `params` - The quantization parameters
///
/// # Returns
///
/// The dequantized vector
#[allow(dead_code)]
pub fn dequantize_vector(quantized: &QuantizedVector, params: &QuantizationParams) -> Array1<f32> {
    let length = quantized.len();
    let mut dequantized = Array1::zeros(length);

    // Handle different quantization data types
    match &quantized.data {
        // Direct floating-point formats
        QuantizedData1D::Float16(data) => {
            // For Float16, just convert directly to f32
            for (i, &val) in data.iter().enumerate() {
                dequantized[i] = val.to_f32();
            }
        }
        QuantizedData1D::BFloat16(data) => {
            // For BFloat16, just convert directly to f32
            for (i, &val) in data.iter().enumerate() {
                dequantized[i] = val.to_f32();
            }
        }
        // Integer-based quantization
        QuantizedData1D::Int8(data) => {
            match quantized.data_type {
                // Special handling for 4-bit quantization types
                QuantizedDataType::Int4 | QuantizedDataType::UInt4 => {
                    for i in 0..length {
                        // Get the 4-bit value using the get method
                        let q_val = quantized.get_i8(i);

                        // Dequantize based on the method
                        let val = match params.method {
                            QuantizationMethod::Int4 => q_val as f32 * params.scale,
                            QuantizationMethod::UInt4 => {
                                params.min_val + (q_val as f32 * params.scale)
                            }
                            _ => unreachable!(), // Should not happen with Int4/UInt4 data type
                        };

                        dequantized[i] = val;
                    }
                }
                // Standard 8-bit quantization
                QuantizedDataType::Int8 => {
                    // Perform dequantization based on the quantization method for 8-bit types
                    match params.method {
                        QuantizationMethod::Uniform => {
                            for (i, &q_val) in data.iter().enumerate() {
                                let val = params.min_val + (q_val as f32 * params.scale);
                                dequantized[i] = val;
                            }
                        }
                        QuantizationMethod::Symmetric => {
                            for (i, &q_val) in data.iter().enumerate() {
                                let val = q_val as f32 * params.scale;
                                dequantized[i] = val;
                            }
                        }
                        QuantizationMethod::Affine => {
                            for (i, &q_val) in data.iter().enumerate() {
                                let val = params.scale * (q_val as f32 - params.zero_point as f32);
                                dequantized[i] = val;
                            }
                        }
                        QuantizationMethod::PowerOfTwo => {
                            for (i, &q_val) in data.iter().enumerate() {
                                let val = params.min_val + (q_val as f32 * params.scale);
                                dequantized[i] = val;
                            }
                        }
                        _ => unreachable!(), // Other methods are handled above
                    }
                }
                _ => unreachable!(), // Should not happen
            }
        }
    }

    dequantized
}

/// Perform matrix multiplication with quantized matrices
///
/// # Arguments
///
/// * `a` - The first quantized matrix
/// * `a_params` - Quantization parameters for the first matrix
/// * `b` - The second quantized matrix
/// * `b_params` - Quantization parameters for the second matrix
///
/// # Returns
///
/// The result of the matrix multiplication in floating-point
#[allow(dead_code)]
pub fn quantized_matmul(
    a: &QuantizedMatrix,
    a_params: &QuantizationParams,
    b: &QuantizedMatrix,
    b_params: &QuantizationParams,
) -> LinalgResult<Array2<f32>> {
    // Check dimensions
    if a.ncols() != b.nrows() {
        return Err(LinalgError::DimensionError(format!(
            "Cannot multiply matrices with shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }

    let (m, k) = a.shape();
    let (_, n) = b.shape();

    // Create result matrix
    let mut result = Array2::zeros((m, n));

    // For floating point quantization types, we use floating point operations
    if matches!(
        a.data_type,
        QuantizedDataType::Float16 | QuantizedDataType::BFloat16
    ) || matches!(
        b.data_type,
        QuantizedDataType::Float16 | QuantizedDataType::BFloat16
    ) {
        // Perform floating-point matrix multiplication
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f32;
                for l in 0..k {
                    let a_val = a.get_f32(i, l);
                    let b_val = b.get_f32(l, j);
                    sum += a_val * b_val;
                }
                result[[i, j]] = sum;
            }
        }
        return Ok(result);
    }

    // Check if either matrix uses per-channel quantization
    let a_per_channel = a_params.method == QuantizationMethod::PerChannelSymmetric
        || a_params.method == QuantizationMethod::PerChannelAffine;

    let b_per_channel = b_params.method == QuantizationMethod::PerChannelSymmetric
        || b_params.method == QuantizationMethod::PerChannelAffine;

    // If either matrix uses per-channel quantization, we'll dequantize to f32 and do regular matmul
    if a_per_channel || b_per_channel {
        // Dequantize both matrices
        let a_dequant = dequantize_matrix(a, a_params);
        let b_dequant = dequantize_matrix(b, b_params);

        // Perform standard matrix multiplication using dequantized matrices
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f32;
                for l in 0..k {
                    sum += a_dequant[[i, l]] * b_dequant[[l, j]];
                }
                result[[i, j]] = sum;
            }
        }

        return Ok(result);
    }

    // For integer quantization, use the original approach
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i32;
            for l in 0..k {
                // Use the get_i8 method for integer types
                let a_val = a.get_i8(i, l) as i32;
                let b_val = b.get_i8(l, j) as i32;
                sum += a_val * b_val;
            }

            // Dequantize the result - scale is the same regardless of method
            let a_scale = a_params.scale;
            let b_scale = b_params.scale;

            // Apply zero-point correction for affine quantization
            if (a_params.method == QuantizationMethod::Affine
                || a_params.method == QuantizationMethod::UInt4)
                && (b_params.method == QuantizationMethod::Affine
                    || b_params.method == QuantizationMethod::UInt4)
            {
                // For affine quantization, we need to correct for zero points
                let a_zero_sum: i32 =
                    (0..k).map(|l| b.get_i8(l, j) as i32).sum::<i32>() * a_params.zero_point;
                let b_zero_sum: i32 =
                    (0..k).map(|l| a.get_i8(i, l) as i32).sum::<i32>() * b_params.zero_point;
                let zero_product = k as i32 * a_params.zero_point * b_params.zero_point;

                sum = sum - a_zero_sum - b_zero_sum + zero_product;
            }

            result[[i, j]] = sum as f32 * a_scale * b_scale;
        }
    }

    Ok(result)
}

/// Perform matrix-vector multiplication with quantized matrix and vector
///
/// # Arguments
///
/// * `a` - The quantized matrix
/// * `a_params` - Quantization parameters for the matrix
/// * `x` - The quantized vector
/// * `x_params` - Quantization parameters for the vector
///
/// # Returns
///
/// The result of the matrix-vector multiplication in floating-point
#[allow(dead_code)]
pub fn quantized_matvec(
    a: &QuantizedMatrix,
    a_params: &QuantizationParams,
    x: &QuantizedVector,
    x_params: &QuantizationParams,
) -> LinalgResult<Array1<f32>> {
    // Check dimensions
    if a.ncols() != x.len() {
        return Err(LinalgError::DimensionError(format!(
            "Cannot multiply matrix with shape {:?} and vector with length {}",
            a.shape(),
            x.len()
        )));
    }

    let (m, n) = a.shape();

    // Create result vector
    let mut result = Array1::zeros(m);

    // For floating point quantization types, we use floating point operations
    if matches!(
        a.data_type,
        QuantizedDataType::Float16 | QuantizedDataType::BFloat16
    ) || matches!(
        x.data_type,
        QuantizedDataType::Float16 | QuantizedDataType::BFloat16
    ) {
        // Perform floating-point matrix-vector multiplication
        for i in 0..m {
            let mut sum = 0.0_f32;
            for j in 0..n {
                let a_val = a.get_f32(i, j);
                let x_val = x.get_f32(j);
                sum += a_val * x_val;
            }
            result[i] = sum;
        }
        return Ok(result);
    }

    // Check if matrix uses per-channel quantization
    let a_per_channel = a_params.method == QuantizationMethod::PerChannelSymmetric
        || a_params.method == QuantizationMethod::PerChannelAffine;

    // For per-channel quantization, we'll dequantize first
    if a_per_channel {
        // Dequantize the matrix and vector
        let a_dequant = dequantize_matrix(a, a_params);
        let x_dequant = dequantize_vector(x, x_params);

        // Perform standard matrix-vector multiplication using dequantized values
        for i in 0..m {
            let mut sum = 0.0_f32;
            for j in 0..n {
                sum += a_dequant[[i, j]] * x_dequant[j];
            }
            result[i] = sum;
        }

        return Ok(result);
    }

    // For integer quantization, use the original approach
    for i in 0..m {
        let mut sum = 0i32;
        for j in 0..n {
            // Use the get_i8 method for integer types
            let a_val = a.get_i8(i, j) as i32;
            let x_val = x.get_i8(j) as i32;
            sum += a_val * x_val;
        }

        // Dequantize the result - scale is the same regardless of method
        let a_scale = a_params.scale;
        let x_scale = x_params.scale;

        // Apply zero-point correction for affine quantization
        if (a_params.method == QuantizationMethod::Affine
            || a_params.method == QuantizationMethod::UInt4)
            && (x_params.method == QuantizationMethod::Affine
                || x_params.method == QuantizationMethod::UInt4)
        {
            // For affine quantization, we need to correct for zero points
            let a_zero_sum: i32 =
                (0..n).map(|j| x.get_i8(j) as i32).sum::<i32>() * a_params.zero_point;
            let x_zero_sum: i32 =
                (0..n).map(|j| a.get_i8(i, j) as i32).sum::<i32>() * x_params.zero_point;
            let zero_product = n as i32 * a_params.zero_point * x_params.zero_point;

            sum = sum - a_zero_sum - x_zero_sum + zero_product;
        }

        result[i] = sum as f32 * a_scale * x_scale;
    }

    Ok(result)
}

/// Compute the dot product of two quantized vectors
///
/// # Arguments
///
/// * `a` - The first quantized vector
/// * `a_params` - Quantization parameters for the first vector
/// * `b` - The second quantized vector
/// * `b_params` - Quantization parameters for the second vector
///
/// # Returns
///
/// The dot product as a floating-point value
#[allow(dead_code)]
pub fn quantized_dot(
    a: &QuantizedVector,
    a_params: &QuantizationParams,
    b: &QuantizedVector,
    b_params: &QuantizationParams,
) -> LinalgResult<f32> {
    // Check dimensions
    if a.len() != b.len() {
        return Err(LinalgError::DimensionError(format!(
            "Cannot compute dot product of vectors with lengths {} and {}",
            a.len(),
            b.len()
        )));
    }

    let n = a.len();

    // For floating point quantization types, we use floating point operations
    if matches!(
        a.data_type,
        QuantizedDataType::Float16 | QuantizedDataType::BFloat16
    ) || matches!(
        b.data_type,
        QuantizedDataType::Float16 | QuantizedDataType::BFloat16
    ) {
        // Perform floating-point dot product
        let mut sum = 0.0_f32;
        for i in 0..n {
            let a_val = a.get_f32(i);
            let b_val = b.get_f32(i);
            sum += a_val * b_val;
        }
        return Ok(sum);
    }

    // Per-channel quantization doesn't apply to vectors directly,
    // but we should check if either param uses per-channel methods
    let a_per_channel = a_params.method == QuantizationMethod::PerChannelSymmetric
        || a_params.method == QuantizationMethod::PerChannelAffine;

    let b_per_channel = b_params.method == QuantizationMethod::PerChannelSymmetric
        || b_params.method == QuantizationMethod::PerChannelAffine;

    // If either uses per-channel, convert to f32 first (unusual case)
    if a_per_channel || b_per_channel {
        // Just dequantize both vectors and compute normal dot product
        let a_dequant = dequantize_vector(a, a_params);
        let b_dequant = dequantize_vector(b, b_params);

        let mut sum = 0.0_f32;
        for i in 0..n {
            sum += a_dequant[i] * b_dequant[i];
        }

        return Ok(sum);
    }

    // For integer quantization, use the original approach
    let mut sum = 0i32;
    for i in 0..n {
        // Use the get_i8 method for integer types
        let a_val = a.get_i8(i) as i32;
        let b_val = b.get_i8(i) as i32;
        sum += a_val * b_val;
    }

    // Dequantize the result - scale is the same regardless of method
    let a_scale = a_params.scale;
    let b_scale = b_params.scale;

    // Apply zero-point correction for affine quantization
    if (a_params.method == QuantizationMethod::Affine
        || a_params.method == QuantizationMethod::UInt4)
        && (b_params.method == QuantizationMethod::Affine
            || b_params.method == QuantizationMethod::UInt4)
    {
        // For affine quantization, we need to correct for zero points
        let a_zero_sum: i32 = (0..n).map(|i| b.get_i8(i) as i32).sum::<i32>() * a_params.zero_point;
        let b_zero_sum: i32 = (0..n).map(|i| a.get_i8(i) as i32).sum::<i32>() * b_params.zero_point;
        let zero_product = n as i32 * a_params.zero_point * b_params.zero_point;

        sum = sum - a_zero_sum - b_zero_sum + zero_product;
    }

    let result = sum as f32 * a_scale * b_scale;

    Ok(result)
}

/// Apply fake quantization to a floating-point tensor
///
/// Fake quantization applies the quantization and dequantization steps
/// to simulate the effects of quantization while still working with
/// floating-point values. This is useful for training quantization-aware
/// neural networks.
///
/// # Arguments
///
/// * `matrix` - The input matrix to apply fake quantization to
/// * `bits` - The number of bits to use for quantization (typically 8)
/// * `method` - The quantization method to use
///
/// # Returns
///
/// The matrix after applying fake quantization
#[allow(dead_code)]
pub fn fake_quantize<F>(matrix: &ArrayView2<F>, bits: u8, method: QuantizationMethod) -> Array2<F>
where
    F: Float + Debug + AsPrimitive<f32> + FromPrimitive,
    f32: AsPrimitive<F>,
{
    // For Int4 and UInt4, we don't need the bits parameter
    let (quantized, params) = quantize_matrix(matrix, bits, method);
    let dequantized = dequantize_matrix(&quantized, &params);

    // Convert back to original type
    let mut result = Array2::zeros(matrix.dim());
    for (i, &val) in dequantized.iter().enumerate() {
        result.as_slice_mut().unwrap()[i] = F::from_f32(val).unwrap();
    }

    result
}

/// Apply fake quantization to a floating-point vector
///
/// # Arguments
///
/// * `vector` - The input vector to apply fake quantization to
/// * `bits` - The number of bits to use for quantization (typically 8)
/// * `method` - The quantization method to use
///
/// # Returns
///
/// The vector after applying fake quantization
#[allow(dead_code)]
pub fn fake_quantize_vector<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    method: QuantizationMethod,
) -> Array1<F>
where
    F: Float + Debug + AsPrimitive<f32> + FromPrimitive,
    f32: AsPrimitive<F>,
{
    // For Int4 and UInt4, we don't need the bits parameter
    let (quantized, params) = quantize_vector(vector, bits, method);
    let dequantized = dequantize_vector(&quantized, &params);

    // Convert back to original type
    let mut result = Array1::zeros(vector.dim());
    for (i, &val) in dequantized.iter().enumerate() {
        result[i] = F::from_f32(val).unwrap();
    }

    result
}

