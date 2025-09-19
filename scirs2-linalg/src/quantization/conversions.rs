//! Quantization conversion functions
//!
//! This module contains functions for converting between floating-point and quantized data,
//! including matrix and vector quantization/dequantization and fake quantization.

use half::{bf16, f16};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{AsPrimitive, Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{LinalgError, LinalgResult};

use super::matrix::{QuantizedData2D, QuantizedMatrix};
use super::types::{QuantizationMethod, QuantizationParams, QuantizedDataType};
use super::vector::{QuantizedData1D, QuantizedVector};

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

// TODO: Add the following functions from the original mod.rs:
// - quantize_vector (lines ~1089-1347)
// - dequantize_vector (lines ~1348-1439)
// - fake_quantize (lines ~1780-1809)
// - fake_quantize_vector (lines ~1810-1832)

// These functions will be added in subsequent iterations to complete the conversions module.
