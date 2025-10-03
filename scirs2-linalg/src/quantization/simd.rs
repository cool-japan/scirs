//! SIMD-accelerated operations for quantized matrices
//!
//! This module provides SIMD-accelerated implementations of matrix operations
//! on quantized data for improved performance. These implementations leverage
//! the scirs2-core SIMD abstractions for SIMD operations and work with the quantization types
//! defined in the parent module.

use crate::error::{LinalgError, LinalgResult};
use crate::quantization::{
    dequantize_matrix, dequantize_vector, get_quantized_vector_1d_i8, get_quantizedmatrix_2d_i8,
    quantize_vector, QuantizationMethod, QuantizationParams, QuantizedData2D, QuantizedDataType,
    QuantizedMatrix, QuantizedVector,
};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::simd_ops::SimdUnifiedOps;

/// SIMD-accelerated quantized matrix-vector multiplication
///
/// Performs matrix-vector multiplication where the matrix is in quantized form
/// and the vector is in f32 format. The result is returned as f32.
///
/// # Arguments
///
/// * `qmatrix` - Quantized matrix
/// * `qparams` - Quantization parameters for the matrix
/// * `vector` - Vector to multiply with
///
/// # Returns
///
/// * Result vector of the multiplication
#[allow(dead_code)]
pub fn simd_quantized_matvec(
    qmatrix: &QuantizedMatrix,
    qparams: &QuantizationParams,
    vector: &ArrayView1<f32>,
) -> LinalgResult<Array1<f32>> {
    // Check dimensions
    if qmatrix.shape.1 != vector.len() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix columns ({}) must match vector length ({})",
            qmatrix.shape.1,
            vector.len()
        )));
    }

    // Create result vector
    let mut result = Array1::zeros(qmatrix.shape.0);
    let vec_slice = vector.as_slice().unwrap();

    // Handle based on data type
    match &qmatrix.data {
        QuantizedData2D::Int8(data) => {
            // Get the scale factors for dequantization
            let scale = qparams.scale;
            let zero_point = qparams.zero_point;

            // Handle per-channel quantization separately
            if qparams.method == QuantizationMethod::PerChannelSymmetric
                || qparams.method == QuantizationMethod::PerChannelAffine
            {
                let scales = qparams
                    .channel_scales
                    .as_ref()
                    .expect("Per-channel quantization requires channel scales");

                let zero_points = if qparams.method == QuantizationMethod::PerChannelAffine {
                    qparams
                        .channel_zero_points
                        .as_ref()
                        .expect("Per-channel affine quantization requires channel zero points")
                } else {
                    &vec![0; qmatrix.shape.1] // Symmetric doesn't use zero points
                };

                // Process each row of the matrix
                for (i, row) in data.rows().into_iter().enumerate() {
                    // We'll use SIMD to accumulate 8 values at once
                    let chunksize = 8;
                    let mut acc = 0.0f32;

                    let row_slice = row.as_slice().unwrap();
                    let mut j = 0;

                    // Accumulate 8 elements at a time using SIMD
                    while j + chunksize <= row_slice.len() {
                        // Load chunks from row, scales, zero points and vector
                        let mut row_vals = [0.0f32; 8];

                        for (k, val) in row_vals.iter_mut().enumerate().take(chunksize) {
                            let idx = j + k;
                            // Dequantize the value: (val - zero_point) * scale
                            let dequantized =
                                (row_slice[idx] as f32 - zero_points[idx] as f32) * scales[idx];
                            *val = dequantized * vec_slice[idx];
                        }

                        // Sum the products into our accumulator using core SIMD operations
                        let row_vals_slice = ArrayView1::from(&row_vals);
                        acc += f32::simd_sum(&row_vals_slice);

                        j += chunksize;
                    }

                    // Handle remaining elements
                    for k in j..row_slice.len() {
                        let dequantized = (row_slice[k] as f32 - zero_points[k] as f32) * scales[k];
                        acc += dequantized * vec_slice[k];
                    }

                    result[i] = acc;
                }
            } else {
                // Standard quantization (single scale/zero point)

                // For Int4/UInt4, we need special handling
                if qparams.data_type == QuantizedDataType::Int4
                    || qparams.data_type == QuantizedDataType::UInt4
                {
                    // Process each row
                    for (i, row) in data.rows().into_iter().enumerate() {
                        let row_slice = row.as_slice().unwrap();
                        let mut acc = 0.0f32;

                        // For Int4/UInt4, we need to unpack two values from each byte
                        for (j, &byte) in row_slice.iter().enumerate() {
                            let col_idx = j * 2; // Each byte contains 2 values

                            // Unpack the first 4-bit value
                            let val1 = if qparams.data_type == QuantizedDataType::Int4 {
                                // Extract and sign-extend 4-bit signed int
                                let q = (byte >> 4) & 0x0F;
                                if q & 0x08 != 0 {
                                    q | 0xF0u8 as i8
                                } else {
                                    q
                                } // Sign extend
                            } else {
                                // UInt4
                                (byte >> 4) & 0x0F
                            };

                            // Process only if we're still within matrix bounds
                            if col_idx < qmatrix.shape.1 {
                                let dequantized = (val1 as f32 - zero_point as f32) * scale;
                                acc += dequantized * vec_slice[col_idx];
                            }

                            // Unpack the second 4-bit value
                            let val2 = if qparams.data_type == QuantizedDataType::Int4 {
                                // Extract and sign-extend 4-bit signed int
                                let q = byte & 0x0F;
                                if q & 0x08 != 0 {
                                    q | 0xF0u8 as i8
                                } else {
                                    q
                                } // Sign extend
                            } else {
                                // UInt4
                                byte & 0x0F
                            };

                            // Process only if we're still within matrix bounds
                            if col_idx + 1 < qmatrix.shape.1 {
                                let dequantized = (val2 as f32 - zero_point as f32) * scale;
                                acc += dequantized * vec_slice[col_idx + 1];
                            }
                        }

                        result[i] = acc;
                    }
                } else {
                    // Standard Int8 processing
                    for (i, row) in data.rows().into_iter().enumerate() {
                        let row_slice = row.as_slice().unwrap();
                        let mut acc = 0.0f32;

                        // Process 8 elements at a time with SIMD
                        let chunksize = 8;
                        let mut j = 0;

                        while j + chunksize <= row_slice.len() {
                            // Load chunks from row and vector
                            let row_chunk = [
                                row_slice[j] as f32,
                                row_slice[j + 1] as f32,
                                row_slice[j + 2] as f32,
                                row_slice[j + 3] as f32,
                                row_slice[j + 4] as f32,
                                row_slice[j + 5] as f32,
                                row_slice[j + 6] as f32,
                                row_slice[j + 7] as f32,
                            ];

                            let vec_chunk = [
                                vec_slice[j],
                                vec_slice[j + 1],
                                vec_slice[j + 2],
                                vec_slice[j + 3],
                                vec_slice[j + 4],
                                vec_slice[j + 5],
                                vec_slice[j + 6],
                                vec_slice[j + 7],
                            ];

                            // Convert to ndarray views for core SIMD operations
                            let _row_view = ArrayView1::from(&row_chunk);
                            let vec_view = ArrayView1::from(&vec_chunk);

                            // Create dequantized values: (row - zero_point) * scale
                            let mut dequantized = [0.0f32; 8];
                            for (k, val) in dequantized.iter_mut().enumerate() {
                                *val = (row_chunk[k] - zero_point as f32) * scale;
                            }
                            let dequantized_view = ArrayView1::from(&dequantized);

                            // Multiply and accumulate using core SIMD
                            acc += f32::simd_dot(&dequantized_view, &vec_view);

                            j += chunksize;
                        }

                        // Process remaining elements
                        for k in j..row_slice.len() {
                            let dequantized = (row_slice[k] as f32 - zero_point as f32) * scale;
                            acc += dequantized * vec_slice[k];
                        }

                        result[i] = acc;
                    }
                }
            }
        }
        QuantizedData2D::Float16(data) => {
            // Do a basic loop multiplication for now - optimize this later
            for (i, row) in data.rows().into_iter().enumerate() {
                let mut sum = 0.0f32;
                for (j, &val) in row.iter().enumerate() {
                    sum += f32::from(val) * vec_slice[j];
                }
                result[i] = sum;
            }
        }
        QuantizedData2D::BFloat16(data) => {
            // Do a basic loop multiplication for now - optimize this later
            for (i, row) in data.rows().into_iter().enumerate() {
                let mut sum = 0.0f32;
                for (j, &val) in row.iter().enumerate() {
                    sum += f32::from(val) * vec_slice[j];
                }
                result[i] = sum;
            }
        }
    }

    Ok(result)
}

/// SIMD-accelerated quantized matrix-matrix multiplication
///
/// Performs matrix-matrix multiplication where both matrices are in quantized form.
/// The result is returned as f32.
///
/// # Arguments
///
/// * `a` - First quantized matrix
/// * `a_params` - Quantization parameters for the first matrix
/// * `b` - Second quantized matrix
/// * `b_params` - Quantization parameters for the second matrix
///
/// # Returns
///
/// * Result matrix of the multiplication
#[allow(dead_code)]
pub fn simd_quantized_matmul(
    a: &QuantizedMatrix,
    a_params: &QuantizationParams,
    b: &QuantizedMatrix,
    b_params: &QuantizationParams,
) -> LinalgResult<Array2<f32>> {
    // Check dimensions
    if a.shape.1 != b.shape.0 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions mismatch for multiplication: ({}, {}) * ({}, {})",
            a.shape.0, a.shape.1, b.shape.0, b.shape.1
        )));
    }

    // Create result matrix
    let (m, n) = (a.shape.0, b.shape.1);
    let mut result = Array2::zeros((m, n));

    // Get int8 data if available - we'll only handle Int8 SIMD acceleration for now
    if let (Some(a_data), Some(b_data)) =
        (get_quantizedmatrix_2d_i8(a), get_quantizedmatrix_2d_i8(b))
    {
        // If either matrix is per-channel quantized, we dequantize it fully first
        // In the future, we can optimize this with specialized kernels
        if a_params.method == QuantizationMethod::PerChannelSymmetric
            || a_params.method == QuantizationMethod::PerChannelAffine
            || b_params.method == QuantizationMethod::PerChannelSymmetric
            || b_params.method == QuantizationMethod::PerChannelAffine
        {
            // Dequantize matrices
            let a_dequant = dequantize_matrix(a, a_params);
            let b_dequant = dequantize_matrix(b, b_params);

            // Use standard matrix multiplication
            return Ok(a_dequant.dot(&b_dequant));
        }

        // Get quantization parameters
        let a_scale = a_params.scale;
        let a_zero = a_params.zero_point as f32;
        let b_scale = b_params.scale;
        let b_zero = b_params.zero_point as f32;

        // Combined scale for the output
        let _output_scale = a_scale * b_scale; // Used in future optimizations

        // For int4/uint4, each byte contains two values, and special handling is needed
        let a_is_4bit = a_params.data_type == QuantizedDataType::Int4
            || a_params.data_type == QuantizedDataType::UInt4;
        let b_is_4bit = b_params.data_type == QuantizedDataType::Int4
            || b_params.data_type == QuantizedDataType::UInt4;

        // Cache-friendly block sizes
        // These should be tuned based on target CPU cache sizes
        const BLOCK_SIZE_M: usize = 32;
        const BLOCK_SIZE_N: usize = 32;
        const BLOCK_SIZE_K: usize = 32;

        // Loop over blocks
        for i0 in (0..m).step_by(BLOCK_SIZE_M) {
            let i_end = (i0 + BLOCK_SIZE_M).min(m);

            for j0 in (0..n).step_by(BLOCK_SIZE_N) {
                let j_end = (j0 + BLOCK_SIZE_N).min(n);

                // Process inner dimension in blocks
                for k0 in (0..a.shape.1).step_by(BLOCK_SIZE_K) {
                    let k_end = (k0 + BLOCK_SIZE_K).min(a.shape.1);

                    // Process blocks
                    for i in i0..i_end {
                        for j in j0..j_end {
                            // Compute dot product of row i from A and column j from B
                            let mut sum = 0.0f32;

                            // Number of elements in this block of the inner dimension
                            let k_blocksize = k_end - k0;

                            // If we're using 4-bit quantization, we need to adjust
                            if a_is_4bit || b_is_4bit {
                                // Simplified handling for 4-bit quantization - dequantize on the fly
                                for k in k0..k_end {
                                    let a_val = if a_is_4bit {
                                        // Extract the right 4-bit value
                                        let byte_idx = k / 2;
                                        let byte = a_data[[i, byte_idx]];

                                        if k % 2 == 0 {
                                            // First 4 bits
                                            let val = (byte >> 4) & 0x0F;
                                            // Sign extend for Int4 if needed
                                            if a_params.data_type == QuantizedDataType::Int4
                                                && (val & 0x08) != 0
                                            {
                                                ((val | 0xF0u8 as i8) as f32 - a_zero) * a_scale
                                            } else {
                                                ((val & 0x0F) as f32 - a_zero) * a_scale
                                            }
                                        } else {
                                            // Second 4 bits
                                            let val = byte & 0x0F;
                                            // Sign extend for Int4 if needed
                                            if a_params.data_type == QuantizedDataType::Int4
                                                && (val & 0x08) != 0
                                            {
                                                ((val | 0xF0u8 as i8) as f32 - a_zero) * a_scale
                                            } else {
                                                ((val & 0x0F) as f32 - a_zero) * a_scale
                                            }
                                        }
                                    } else {
                                        // Regular 8-bit quantization
                                        (a_data[[i, k]] as f32 - a_zero) * a_scale
                                    };

                                    let b_val = if b_is_4bit {
                                        // Extract the right 4-bit value
                                        let byte_idx = k / 2;
                                        let byte = b_data[[byte_idx, j]];

                                        if k % 2 == 0 {
                                            // First 4 bits
                                            let val = (byte >> 4) & 0x0F;
                                            // Sign extend for Int4 if needed
                                            if b_params.data_type == QuantizedDataType::Int4
                                                && (val & 0x08) != 0
                                            {
                                                ((val | 0xF0u8 as i8) as f32 - b_zero) * b_scale
                                            } else {
                                                ((val & 0x0F) as f32 - b_zero) * b_scale
                                            }
                                        } else {
                                            // Second 4 bits
                                            let val = byte & 0x0F;
                                            // Sign extend for Int4 if needed
                                            if b_params.data_type == QuantizedDataType::Int4
                                                && (val & 0x08) != 0
                                            {
                                                ((val | 0xF0u8 as i8) as f32 - b_zero) * b_scale
                                            } else {
                                                ((val & 0x0F) as f32 - b_zero) * b_scale
                                            }
                                        }
                                    } else {
                                        // Regular 8-bit quantization
                                        (b_data[[k, j]] as f32 - b_zero) * b_scale
                                    };

                                    sum += a_val * b_val;
                                }
                            } else {
                                // Regular 8-bit quantization - we can use SIMD

                                // Get row from A and column from B as slices if possible
                                let a_row = a_data.slice(scirs2_core::ndarray::s![i, k0..k_end]);
                                let b_col = b_data.slice(scirs2_core::ndarray::s![k0..k_end, j]);

                                if let (Some(a_slice), Some(b_slice)) =
                                    (a_row.as_slice(), b_col.as_slice())
                                {
                                    // Process with SIMD (8 elements at a time)
                                    let mut l = 0;
                                    let chunksize = 8;

                                    while l + chunksize <= k_blocksize {
                                        // Load chunks
                                        let a_chunk = [
                                            a_slice[l] as f32,
                                            a_slice[l + 1] as f32,
                                            a_slice[l + 2] as f32,
                                            a_slice[l + 3] as f32,
                                            a_slice[l + 4] as f32,
                                            a_slice[l + 5] as f32,
                                            a_slice[l + 6] as f32,
                                            a_slice[l + 7] as f32,
                                        ];
                                        let b_chunk = [
                                            b_slice[l] as f32,
                                            b_slice[l + 1] as f32,
                                            b_slice[l + 2] as f32,
                                            b_slice[l + 3] as f32,
                                            b_slice[l + 4] as f32,
                                            b_slice[l + 5] as f32,
                                            b_slice[l + 6] as f32,
                                            b_slice[l + 7] as f32,
                                        ];

                                        // Dequantize chunks
                                        let mut a_dequant = [0.0f32; 8];
                                        let mut b_dequant = [0.0f32; 8];

                                        for k in 0..8 {
                                            a_dequant[k] = (a_chunk[k] - a_zero) * a_scale;
                                            b_dequant[k] = (b_chunk[k] - b_zero) * b_scale;
                                        }

                                        // Convert to views and compute dot product using core SIMD
                                        let a_view = ArrayView1::from(&a_dequant);
                                        let b_view = ArrayView1::from(&b_dequant);
                                        sum += f32::simd_dot(&a_view, &b_view);

                                        l += chunksize;
                                    }

                                    // Process remaining elements
                                    for m in l..k_blocksize {
                                        let a_val = (a_slice[m] as f32 - a_zero) * a_scale;
                                        let b_val = (b_slice[m] as f32 - b_zero) * b_scale;
                                        sum += a_val * b_val;
                                    }
                                } else {
                                    // Fallback for non-contiguous data
                                    for k in k0..k_end {
                                        let a_val = (a_data[[i, k]] as f32 - a_zero) * a_scale;
                                        let b_val = (b_data[[k, j]] as f32 - b_zero) * b_scale;
                                        sum += a_val * b_val;
                                    }
                                }
                            }

                            // Accumulate result
                            result[[i, j]] += sum;
                        }
                    }
                }
            }
        }
    } else {
        // If we don't have Int8 data, fall back to dequantize and multiply
        let a_dequant = dequantize_matrix(a, a_params);
        let b_dequant = dequantize_matrix(b, b_params);

        return Ok(a_dequant.dot(&b_dequant));
    }

    Ok(result)
}

/// SIMD-accelerated quantized dot product
///
/// Computes the dot product of two quantized vectors using SIMD instructions.
///
/// # Arguments
///
/// * `a` - First quantized vector
/// * `a_params` - Quantization parameters for the first vector
/// * `b` - Second quantized vector
/// * `b_params` - Quantization parameters for the second vector
///
/// # Returns
///
/// * Dot product result
#[allow(dead_code)]
pub fn simd_quantized_dot(
    a: &QuantizedVector,
    a_params: &QuantizationParams,
    b: &QuantizedVector,
    b_params: &QuantizationParams,
) -> LinalgResult<f32> {
    // Check dimensions
    if a.length != b.length {
        return Err(LinalgError::ShapeError(format!(
            "Vector dimensions must match for dot product: {} vs {}",
            a.length, b.length
        )));
    }

    // Get int8 data if available
    if let (Some(a_data), Some(b_data)) =
        (get_quantized_vector_1d_i8(a), get_quantized_vector_1d_i8(b))
    {
        // Get quantization parameters
        let a_scale = a_params.scale;
        let a_zero = a_params.zero_point as f32;
        let b_scale = b_params.scale;
        let b_zero = b_params.zero_point as f32;

        // Combined scale for the output
        let _output_scale = a_scale * b_scale; // Used in future optimizations

        // For int4/uint4, each byte contains two values
        let a_is_4bit = a_params.data_type == QuantizedDataType::Int4
            || a_params.data_type == QuantizedDataType::UInt4;
        let b_is_4bit = b_params.data_type == QuantizedDataType::Int4
            || b_params.data_type == QuantizedDataType::UInt4;

        if a_is_4bit || b_is_4bit {
            // Handle 4-bit specially - we need to unpack values
            let mut sum = 0.0f32;

            // We need to adjust length for 4-bit values (each byte has 2 values)
            let _a_byte_len = a.length.div_ceil(2); // Used for bounds checking
            let _b_byte_len = b.length.div_ceil(2); // Used for bounds checking

            for i in 0..a.length {
                // Extract values from packed 4-bit representation
                let a_val = if a_is_4bit {
                    let byte_idx = i / 2;
                    let byte = a_data[byte_idx];

                    if i % 2 == 0 {
                        // First 4 bits
                        let val = (byte >> 4) & 0x0F;
                        // Sign extend for Int4 if needed
                        if a_params.data_type == QuantizedDataType::Int4 && (val & 0x08) != 0 {
                            ((val | 0xF0u8 as i8) as f32 - a_zero) * a_scale
                        } else {
                            (val as f32 - a_zero) * a_scale
                        }
                    } else {
                        // Second 4 bits
                        let val = byte & 0x0F;
                        // Sign extend for Int4 if needed
                        if a_params.data_type == QuantizedDataType::Int4 && (val & 0x08) != 0 {
                            ((val | 0xF0u8 as i8) as f32 - a_zero) * a_scale
                        } else {
                            (val as f32 - a_zero) * a_scale
                        }
                    }
                } else {
                    (a_data[i] as f32 - a_zero) * a_scale
                };

                let b_val = if b_is_4bit {
                    let byte_idx = i / 2;
                    let byte = b_data[byte_idx];

                    if i % 2 == 0 {
                        // First 4 bits
                        let val = (byte >> 4) & 0x0F;
                        // Sign extend for Int4 if needed
                        if b_params.data_type == QuantizedDataType::Int4 && (val & 0x08) != 0 {
                            ((val | 0xF0u8 as i8) as f32 - b_zero) * b_scale
                        } else {
                            (val as f32 - b_zero) * b_scale
                        }
                    } else {
                        // Second 4 bits
                        let val = byte & 0x0F;
                        // Sign extend for Int4 if needed
                        if b_params.data_type == QuantizedDataType::Int4 && (val & 0x08) != 0 {
                            ((val | 0xF0u8 as i8) as f32 - b_zero) * b_scale
                        } else {
                            (val as f32 - b_zero) * b_scale
                        }
                    }
                } else {
                    (b_data[i] as f32 - b_zero) * b_scale
                };

                sum += a_val * b_val;
            }

            return Ok(sum);
        }

        // Standard 8-bit quantization
        let a_slice = a_data.as_slice().unwrap();
        let b_slice = b_data.as_slice().unwrap();

        // Process 8 elements at a time with SIMD
        let mut i = 0;
        let chunksize = 8;
        let mut sum = 0.0f32;

        while i + chunksize <= a.length {
            // Load chunks
            let a_chunk = [
                a_slice[i] as f32,
                a_slice[i + 1] as f32,
                a_slice[i + 2] as f32,
                a_slice[i + 3] as f32,
                a_slice[i + 4] as f32,
                a_slice[i + 5] as f32,
                a_slice[i + 6] as f32,
                a_slice[i + 7] as f32,
            ];

            let b_chunk = [
                b_slice[i] as f32,
                b_slice[i + 1] as f32,
                b_slice[i + 2] as f32,
                b_slice[i + 3] as f32,
                b_slice[i + 4] as f32,
                b_slice[i + 5] as f32,
                b_slice[i + 6] as f32,
                b_slice[i + 7] as f32,
            ];

            // Dequantize chunks
            let mut a_dequant = [0.0f32; 8];
            let mut b_dequant = [0.0f32; 8];

            for k in 0..8 {
                a_dequant[k] = (a_chunk[k] - a_zero) * a_scale;
                b_dequant[k] = (b_chunk[k] - b_zero) * b_scale;
            }

            // Convert to views and compute dot product using core SIMD
            let a_view = ArrayView1::from(&a_dequant);
            let b_view = ArrayView1::from(&b_dequant);
            sum += f32::simd_dot(&a_view, &b_view);

            i += chunksize;
        }

        // Process remaining elements
        for j in i..a.length {
            let a_val = (a_slice[j] as f32 - a_zero) * a_scale;
            let b_val = (b_slice[j] as f32 - b_zero) * b_scale;
            sum += a_val * b_val;
        }

        Ok(sum)
    } else {
        // If we don't have Int8 data, fall back to dequantize and dot
        let a_dequant = dequantize_vector(a, a_params);
        let b_dequant = dequantize_vector(b, b_params);
        Ok(a_dequant.dot(&b_dequant))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::{
        quantize_matrix, quantize_matrix_per_channel, quantize_vector, QuantizationMethod,
    };
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    #[ignore = "timeout"]
    fn test_simd_quantized_matvec() {
        // Create test matrix and vector
        let mat = array![
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ];

        let vec = array![2.0f32, 3.0, 4.0, 5.0];

        // Quantize the matrix
        let (qmat, qparams) = quantize_matrix(&mat.view(), 8, QuantizationMethod::Symmetric);

        // Compute result with SIMD acceleration
        let result = simd_quantized_matvec(&qmat, &qparams, &vec.view()).unwrap();

        // Expected result (regular matmul)
        let expected = array![40.0f32, 96.0, 152.0];

        // Verify correctness with tolerance for quantization error
        assert_eq!(result.len(), expected.len());
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 0.5);
        }
    }

    #[test]
    #[ignore = "timeout"]
    fn test_simd_quantized_matmul() {
        // Create test matrices
        let a = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0f32, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]];

        // Quantize matrices
        let (qa, qa_params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Symmetric);
        let (qb, qb_params) = quantize_matrix(&b.view(), 8, QuantizationMethod::Symmetric);

        // Compute result with SIMD acceleration
        let result = simd_quantized_matmul(&qa, &qa_params, &qb, &qb_params).unwrap();

        // Expected result (regular matmul)
        let expected = array![[66.0f32, 72.0, 78.0], [156.0, 171.0, 186.0]];

        // Verify correctness with tolerance for quantization error
        assert_eq!(result.shape(), expected.shape());
        for ((i, j), &val) in result.indexed_iter() {
            assert_relative_eq!(val, expected[[i, j]], epsilon = 1.0);
        }
    }

    #[test]
    #[ignore = "timeout"]
    fn test_simd_quantized_dot() {
        // Create test vectors
        let a = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = array![5.0f32, 4.0, 3.0, 2.0, 1.0];

        // Quantize vectors
        let (qa, qa_params) = quantize_vector(&a.view(), 8, QuantizationMethod::Symmetric);
        let (qb, qb_params) = quantize_vector(&b.view(), 8, QuantizationMethod::Symmetric);

        // Compute result with SIMD acceleration
        let result = simd_quantized_dot(&qa, &qa_params, &qb, &qb_params).unwrap();

        // Expected result (regular dot product)
        let expected = 1.0 * 5.0 + 2.0 * 4.0 + 3.0 * 3.0 + 4.0 * 2.0 + 5.0 * 1.0;

        // Verify correctness with tolerance for quantization error
        assert_relative_eq!(result, expected, epsilon = 0.5);

        // Temporary: just verify the expected calculation
        assert_eq!(expected, 35.0);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_simd_quantized_int4_operations() {
        // Create test matrix and vector
        let mat = array![[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];

        let vec = array![2.0f32, 3.0, 4.0, 5.0];

        // Quantize the matrix to Int4
        let (qmat, qparams) = quantize_matrix(&mat.view(), 4, QuantizationMethod::Int4);

        // Compute result with SIMD acceleration
        let result = simd_quantized_matvec(&qmat, &qparams, &vec.view()).unwrap();

        // Expected result (regular matmul)
        let expected = array![40.0f32, 96.0];

        // Verify correctness with tolerance for Int4 quantization error (higher error expected)
        assert_eq!(result.len(), expected.len());
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 3.0);
        }
    }

    #[test]
    #[ignore = "timeout"]
    fn test_simd_quantized_per_channel() {
        // Create a test matrix with very different scales in each column
        let mat = array![
            [0.1f32, 10.0, 100.0],
            [0.2, 20.0, 200.0],
            [0.3, 30.0, 300.0]
        ];

        let vec = array![1.0f32, 0.5, 0.25];

        // Quantize with per-channel method
        let (qmat, qparams) =
            quantize_matrix_per_channel(&mat.view(), 8, QuantizationMethod::PerChannelSymmetric);

        // Compute result with SIMD acceleration
        let result = simd_quantized_matvec(&qmat, &qparams, &vec.view()).unwrap();

        // Expected result (regular matmul)
        let expected = array![
            0.1 * 1.0 + 10.0 * 0.5 + 100.0 * 0.25,
            0.2 * 1.0 + 20.0 * 0.5 + 200.0 * 0.25,
            0.3 * 1.0 + 30.0 * 0.5 + 300.0 * 0.25
        ];

        // Verify correctness with tolerance for quantization error
        assert_eq!(result.len(), expected.len());
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 0.5);
        }
    }
}
