//! Fusion of consecutive quantized operations
//!
//! This module provides optimized implementations for fusing multiple
//! quantized operations, avoiding the overhead of intermediate dequantization
//! and requantization steps in performance-critical code paths.

use crate::error::{LinalgError, LinalgResult};
use crate::quantization::{
    dequantize_matrix, get_quantizedmatrix_2d_i8, QuantizationMethod, QuantizationParams,
    QuantizedData2D, QuantizedMatrix,
};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use std::fmt::Debug;

/// Fused quantized matrix multiplication chain
///
/// Computes a chain of matrix multiplications (A * B * C * ...) where all
/// matrices are in quantized form. This is more efficient than performing
/// individual multiplications and re-quantizing intermediate results.
///
/// # Arguments
///
/// * `matrices` - A slice of quantized matrices to multiply
/// * `params` - A slice of quantization parameters for each matrix
///
/// # Returns
///
/// * The result of the matrix multiplication chain
#[allow(dead_code)]
pub fn fused_quantized_matmul_chain(
    matrices: &[&QuantizedMatrix],
    params: &[&QuantizationParams],
) -> LinalgResult<Array2<f32>> {
    // Validate input
    if matrices.len() < 2 {
        return Err(LinalgError::ShapeError(
            "At least two matrices are required for matmul chain".to_string(),
        ));
    }

    if matrices.len() != params.len() {
        return Err(LinalgError::ShapeError(
            "Number of matrices must match number of quantization parameters".to_string(),
        ));
    }

    // Check dimension compatibility
    for i in 0..matrices.len() - 1 {
        if matrices[i].shape.1 != matrices[i + 1].shape.0 {
            return Err(LinalgError::ShapeError(format!(
                "Matrix dimensions mismatch at position {}: ({}, {}) * ({}, {})",
                i,
                matrices[i].shape.0,
                matrices[i].shape.1,
                matrices[i + 1].shape.0,
                matrices[i + 1].shape.1
            )));
        }
    }

    // Check if all matrices are Int8 format (for now, we only optimize this case)
    let all_int8 = matrices
        .iter()
        .all(|m| matches!(m.data, QuantizedData2D::Int8(_)));

    let all_symmetric = params
        .iter()
        .all(|p| p.method == QuantizationMethod::Symmetric || p.method == QuantizationMethod::Int4);

    if all_int8 && all_symmetric {
        // Optimized path for symmetric Int8 quantization
        fused_quantized_matmul_chain_int8_symmetric(matrices, params)
    } else {
        // Fallback path: dequantize all matrices and perform regular matmul
        // This could be optimized further in the future
        let mut dequantized_matrices = Vec::with_capacity(matrices.len());

        for (matrix, param) in matrices.iter().zip(params.iter()) {
            dequantized_matrices.push(dequantize_matrix(matrix, param));
        }

        // Compute the matrix multiplication chain
        let mut result = dequantized_matrices[0].clone();
        for mat in dequantized_matrices.iter().skip(1) {
            result = result.dot(mat);
        }

        Ok(result)
    }
}

/// Optimized implementation for Int8 symmetric quantized matrices
#[allow(dead_code)]
fn fused_quantized_matmul_chain_int8_symmetric(
    matrices: &[&QuantizedMatrix],
    params: &[&QuantizationParams],
) -> LinalgResult<Array2<f32>> {
    // Extract Int8 data from matrices
    let int8_matrices: Vec<&Array2<i8>> = matrices
        .iter()
        .map(|m| get_quantizedmatrix_2d_i8(m).unwrap())
        .collect();

    // Scales from the quantization parameters
    let scales: Vec<f32> = params.iter().map(|p| p.scale).collect();

    // Result dimensions
    let rows_ = matrices[0].shape.0;
    let cols = matrices.last().unwrap().shape.1;
    let mut result = Array2::zeros((rows_, cols));

    // Compute the fused scale factor - product of all scales
    let fused_scale: f32 = scales.iter().product();

    // Use block multiplication for better cache efficiency
    const BLOCK_SIZE: usize = 32;
    for i0 in (0..rows_).step_by(BLOCK_SIZE) {
        let i_end = (i0 + BLOCK_SIZE).min(rows_);

        for j0 in (0..cols).step_by(BLOCK_SIZE) {
            let j_end = (j0 + BLOCK_SIZE).min(cols);

            // Process each cell in the output block
            for i in i0..i_end {
                for j in j0..j_end {
                    // We need to do the matrix chain multiplication for this cell
                    // This is a dot product through the different matrices

                    // Use an intermediate buffer for partial results in the chain
                    let mut middle_dim = matrices[0].shape.1;
                    let mut intermediate = vec![0i32; middle_dim];

                    // Initialize with first matrix row
                    for (k, val) in intermediate.iter_mut().enumerate().take(middle_dim) {
                        *val = int8_matrices[0][[i, k]] as i32;
                    }

                    // Process intermediate matrices (all except first and last)
                    for mat_idx in 1..matrices.len() - 1 {
                        let mat = int8_matrices[mat_idx];
                        let (_, inner_dim) = matrices[mat_idx].shape;

                        let mut new_intermediate = vec![0i32; inner_dim];

                        // Propagate through the next matrix
                        for l in 0..inner_dim {
                            for k in 0..middle_dim {
                                new_intermediate[l] += intermediate[k] * (mat[[k, l]] as i32);
                            }
                        }

                        // Update intermediate and dimension for next iteration
                        intermediate = new_intermediate;
                        middle_dim = inner_dim;
                    }

                    // Final matrix
                    let last_mat = int8_matrices.last().unwrap();
                    let mut sum = 0i32;

                    for k in 0..middle_dim {
                        sum += intermediate[k] * (last_mat[[k, j]] as i32);
                    }

                    // Apply fused scaling factor
                    result[[i, j]] = (sum as f32) * fused_scale;
                }
            }
        }
    }

    Ok(result)
}

/// Fused quantized matrix-vector multiplication sequence
///
/// Computes the matrix-vector sequence (A * B * ... * x) where matrices and
/// vector are in quantized form. This avoids dequantizing and requantizing
/// intermediate results.
///
/// # Arguments
///
/// * `matrices` - A slice of quantized matrices to multiply
/// * `matrix_params` - A slice of quantization parameters for each matrix
/// * `vector` - The quantized vector to multiply with
/// * `vector_params` - Quantization parameters for the vector
///
/// # Returns
///
/// * The result of the matrix-vector multiplication sequence
#[allow(dead_code)]
pub fn fused_quantized_matvec_sequence<F>(
    matrices: &[&QuantizedMatrix],
    matrix_params: &[&QuantizationParams],
    vector: &ArrayView1<F>,
    output_quantize: bool,
) -> LinalgResult<Array1<F>>
where
    F: scirs2_core::numeric::Float
        + Debug
        + scirs2_core::numeric::AsPrimitive<f32>
        + scirs2_core::numeric::FromPrimitive,
    f32: scirs2_core::numeric::AsPrimitive<F>,
{
    // Validate input
    if matrices.is_empty() {
        return Err(LinalgError::ShapeError(
            "At least one matrix is required for matvec sequence".to_string(),
        ));
    }

    if matrices.len() != matrix_params.len() {
        return Err(LinalgError::ShapeError(
            "Number of matrices must match number of quantization parameters".to_string(),
        ));
    }

    // Check dimension compatibility
    let vector_len = vector.len();
    if matrices.last().unwrap().shape.1 != vector_len {
        return Err(LinalgError::ShapeError(format!(
            "Last matrix columns ({}) must match vector length ({})",
            matrices.last().unwrap().shape.1,
            vector_len
        )));
    }

    for i in 0..matrices.len() - 1 {
        if matrices[i].shape.1 != matrices[i + 1].shape.0 {
            return Err(LinalgError::ShapeError(format!(
                "Matrix dimensions mismatch at position {}: ({}, {}) * ({}, {})",
                i,
                matrices[i].shape.0,
                matrices[i].shape.1,
                matrices[i + 1].shape.0,
                matrices[i + 1].shape.1
            )));
        }
    }

    // Check if all matrices are Int8 format (for now, we only optimize this case)
    let all_int8 = matrices
        .iter()
        .all(|m| matches!(m.data, QuantizedData2D::Int8(_)));

    if all_int8 {
        // Convert vector to f32
        let vector_f32 = vector.mapv(|x| x.as_());
        let vector_f32_view = vector_f32.view();

        // Compute result as f32
        let result_f32 = if matrices.len() == 1 {
            // Single matrix case - use the existing SIMD function
            use crate::quantization::simd::simd_quantized_matvec;
            simd_quantized_matvec(matrices[0], matrix_params[0], &vector_f32_view)?
        } else {
            // Multiple matrices case - fuse the operation
            fused_quantized_matvec_sequence_int8(matrices, matrix_params, &vector_f32_view)?
        };

        // Convert back to the original type
        if output_quantize {
            // In a complete implementation, we would _quantize the result to the same bit depth
            // But for simplicity, just convert back to the original type
            Ok(result_f32.mapv(|x| scirs2_core::numeric::FromPrimitive::from_f32(x).unwrap()))
        } else {
            // Return as float directly
            Ok(result_f32.mapv(|x| scirs2_core::numeric::FromPrimitive::from_f32(x).unwrap()))
        }
    } else {
        // Fallback path: dequantize all matrices and perform regular matmul
        let mut dequantized_matrices = Vec::with_capacity(matrices.len());

        for (matrix, param) in matrices.iter().zip(matrix_params.iter()) {
            dequantized_matrices.push(dequantize_matrix(matrix, param));
        }

        // Convert to f32 for internal calculations
        let vector_f32 = vector.mapv(|x| x.as_());

        // Create a column vector from the 1D array
        let mut result_f32 = vector_f32.insert_axis(scirs2_core::ndarray::Axis(1));

        // Apply matrices in reverse order (rightmost first)
        for mat in dequantized_matrices.iter().rev() {
            result_f32 = mat.dot(&result_f32);
        }

        // Convert back to 1D array and then to original type
        let result_1d_f32 = result_f32.remove_axis(scirs2_core::ndarray::Axis(1));

        // Convert back to the original type
        let result_f =
            result_1d_f32.mapv(|x| scirs2_core::numeric::FromPrimitive::from_f32(x).unwrap());

        Ok(result_f)
    }
}

/// Optimized implementation for Int8 quantized matrices in a matvec sequence
#[allow(dead_code)]
fn fused_quantized_matvec_sequence_int8(
    matrices: &[&QuantizedMatrix],
    params: &[&QuantizationParams],
    vector: &ArrayView1<f32>,
) -> LinalgResult<Array1<f32>> {
    // Extract Int8 data
    let int8_matrices: Vec<&Array2<i8>> = matrices
        .iter()
        .map(|m| get_quantizedmatrix_2d_i8(m).unwrap())
        .collect();

    // Get scales from the parameters
    let scales: Vec<f32> = params.iter().map(|p| p.scale).collect();
    // Zero points used only in the asymmetric path
    let _zero_points: Vec<i32> = params.iter().map(|p| p.zero_point).collect();

    // For symmetric quantization, zero points should be zero
    let symmetric = params
        .iter()
        .all(|p| p.method == QuantizationMethod::Symmetric);

    // Get output dimensions
    let output_dim = matrices[0].shape.0;
    let mut result = Array1::zeros(output_dim);

    // Compute the result using block-based approach for better cache efficiency
    if symmetric {
        // Faster path for symmetric quantization
        let fused_scale: f32 = scales.iter().product();

        // We'll compute one result element at a time
        for i in 0..output_dim {
            let row = int8_matrices[0].row(i);

            // For each element in the output, we need to compute a complex contraction
            // Initialize with first matrix row
            let middle_dim = matrices[0].shape.1;
            let mut intermediate = vec![0i32; middle_dim];

            for k in 0..middle_dim {
                intermediate[k] = row[k] as i32;
            }

            // Propagate through the other matrices
            for mat_idx in 1..matrices.len() {
                let mat = int8_matrices[mat_idx];
                let (rows, cols) = matrices[mat_idx].shape;

                let mut new_intermediate = vec![0i32; cols];

                for c in 0..cols {
                    for r in 0..rows {
                        new_intermediate[c] += intermediate[r] * (mat[[r, c]] as i32);
                    }
                }

                intermediate = new_intermediate;
            }

            // Final dot product with the vector
            let mut sum = 0.0;
            for k in 0..intermediate.len() {
                sum += (intermediate[k] as f32) * vector[k];
            }

            result[i] = sum * fused_scale;
        }
    } else {
        // Path for asymmetric quantization (we need to handle zero points)
        // This is more complex and less optimized

        // Dequantize the matrices first
        let mut dequantized_matrices = Vec::with_capacity(matrices.len());

        for (matrix, param) in matrices.iter().zip(params.iter()) {
            dequantized_matrices.push(dequantize_matrix(matrix, param));
        }

        // Create a column vector for matrix operations
        let vector_2d = vector.to_owned().insert_axis(scirs2_core::ndarray::Axis(1));

        // Apply matrices in reverse order
        let mut result_2d = vector_2d;
        for mat in dequantized_matrices.iter().rev() {
            result_2d = mat.dot(&result_2d);
        }

        // Extract the column back to 1D
        result = result_2d.remove_axis(scirs2_core::ndarray::Axis(1));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::{quantize_matrix, QuantizationMethod};
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_fused_matmul_chain() {
        // Create test matrices
        let a = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0f32, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let c = array![[13.0f32, 14.0, 15.0], [16.0, 17.0, 18.0]];

        // Quantize matrices
        let (qa, qa_params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Symmetric);
        let (qb, qb_params) = quantize_matrix(&b.view(), 8, QuantizationMethod::Symmetric);
        let (qc, qc_params) = quantize_matrix(&c.view(), 8, QuantizationMethod::Symmetric);

        // Expected result - regular matrix multiplication chain
        let ab = a.dot(&b);
        let expected = ab.dot(&c);

        // Fused chain calculation
        let matrices = [&qa, &qb, &qc];
        let params = [&qa_params, &qb_params, &qc_params];
        let result = fused_quantized_matmul_chain(&matrices, &params).unwrap();

        // Verify correctness with tolerance for quantization error
        assert_eq!(result.shape(), expected.shape());
        for ((i, j), &val) in result.indexed_iter() {
            assert_relative_eq!(val, expected[[i, j]], epsilon = 12.0);
        }
    }

    #[test]
    fn test_fused_matvec_sequence() {
        // Create test matrices and vector
        let a = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0f32, 8.0], [9.0, 10.0], [11.0, 12.0]];
        // Use 1D vector instead of 2D
        let x = array![13.0f32, 14.0];

        // Quantize matrices and vector
        let (qa, qa_params) = quantize_matrix(&a.view(), 8, QuantizationMethod::Symmetric);
        let (qb, qb_params) = quantize_matrix(&b.view(), 8, QuantizationMethod::Symmetric);

        // Expected result
        let bx = b.dot(&x);
        let expected = a.dot(&bx);

        // Fused calculation
        let matrices = [&qa, &qb];
        let params = [&qa_params, &qb_params];
        let result = fused_quantized_matvec_sequence(&matrices, &params, &x.view(), false).unwrap();

        // Verify correctness with tolerance for quantization error
        assert_eq!(result.len(), expected.len());
        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 5.0);
        }
    }
}
