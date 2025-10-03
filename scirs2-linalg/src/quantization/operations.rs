//! Quantized linear algebra operations
//!
//! This module contains functions for performing linear algebra operations
//! on quantized matrices and vectors, including matrix multiplication,
//! matrix-vector multiplication, and dot products.

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{LinalgError, LinalgResult};

use super::conversions::dequantize_matrix;
use super::matrix::QuantizedMatrix;
use super::types::{QuantizationMethod, QuantizationParams, QuantizedDataType};
use super::vector::QuantizedVector;

/// Perform matrix multiplication with two quantized matrices
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
/// * `b` - The quantized vector
/// * `b_params` - Quantization parameters for the vector
///
/// # Returns
///
/// The result of the matrix-vector multiplication in floating-point
pub fn quantized_matvec(
    a: &QuantizedMatrix,
    a_params: &QuantizationParams,
    b: &QuantizedVector,
    b_params: &QuantizationParams,
) -> LinalgResult<Array1<f32>> {
    // Check dimensions
    if a.ncols() != b.len() {
        return Err(LinalgError::DimensionError(format!(
            "Cannot multiply matrix with shape {:?} and vector with length {}",
            a.shape(),
            b.len()
        )));
    }

    let m = a.nrows();
    let n = a.ncols();

    // Create result vector
    let mut result = Array1::zeros(m);

    // For floating point quantization types
    if matches!(
        a_params.data_type,
        QuantizedDataType::Float16 | QuantizedDataType::BFloat16
    ) && matches!(
        b_params.data_type,
        QuantizedDataType::Float16 | QuantizedDataType::BFloat16
    ) {
        // Dequantize and compute in floating point
        let a_full = dequantize_matrix(a, a_params);
        let b_full = dequantize_vector(b, b_params);

        for i in 0..m {
            let mut sum = 0.0_f32;
            for j in 0..n {
                sum += a_full[[i, j]] * b_full[j];
            }
            result[i] = sum;
        }

        return Ok(result);
    }

    // For integer quantization
    let a_scale = a_params.scale;
    let b_scale = b_params.scale;

    for i in 0..m {
        let mut sum: i32 = 0;

        for j in 0..n {
            let a_val = a.get_i8(i, j) as i32;
            let b_val = b.get_i8(j) as i32;
            sum += a_val * b_val;
        }

        result[i] = sum as f32 * a_scale * b_scale;
    }

    Ok(result)
}

/// Perform dot product with two quantized vectors
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
/// The dot product result in floating-point
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

    // For floating point quantization types
    if matches!(
        a_params.data_type,
        QuantizedDataType::Float16 | QuantizedDataType::BFloat16
    ) && matches!(
        b_params.data_type,
        QuantizedDataType::Float16 | QuantizedDataType::BFloat16
    ) {
        // Dequantize and compute in floating point
        let a_full = dequantize_vector(a, a_params);
        let b_full = dequantize_vector(b, b_params);

        let mut sum = 0.0_f32;
        for i in 0..n {
            sum += a_full[i] * b_full[i];
        }

        return Ok(sum);
    }

    // For integer quantization
    let a_scale = a_params.scale;
    let b_scale = b_params.scale;

    let mut sum: i32 = 0;

    for i in 0..n {
        let a_val = a.get_i8(i) as i32;
        let b_val = b.get_i8(i) as i32;
        sum += a_val * b_val;
    }

    Ok(sum as f32 * a_scale * b_scale)
}

// Helper function to dequantize a vector
fn dequantize_vector(vec: &QuantizedVector, _params: &QuantizationParams) -> Array1<f32> {
    let n = vec.len();
    let mut result = Array1::zeros(n);

    for i in 0..n {
        result[i] = vec.get_f32(i);
    }

    result
}
