//! SIMD accelerated linear algebra operations
//!
//! This module provides SIMD-accelerated implementations of common linear algebra
//! operations for improved performance on modern CPUs. These implementations
//! leverage core SIMD support through the scirs2-core::simd module.

pub mod elementwise;
pub mod gemm;
pub mod norms;
pub mod transpose;

// Re-export commonly used SIMD operations
#[cfg(feature = "simd")]
pub use elementwise::{
    simd_matrix_add_f32, simd_matrix_add_f64, simd_matrix_add_inplace_f32,
    simd_matrix_mul_elementwise_f32, simd_matrix_scale_f32,
};
#[cfg(feature = "simd")]
pub use gemm::{
    simd_gemm_f32, simd_gemm_f64, simd_gemv_f32, simd_gemv_f64, simd_matmul_optimized_f32,
    simd_matmul_optimized_f64, GemmBlockSizes,
};
#[cfg(feature = "simd")]
pub use norms::{
    simd_frobenius_norm_f32, simd_frobenius_norm_f64, simd_vector_norm_f32, simd_vector_norm_f64,
};
#[cfg(feature = "simd")]
pub use transpose::{simd_transpose_f32, simd_transpose_f64};

#[allow(unused_imports)]
use crate::error::{LinalgError, LinalgResult};
#[allow(unused_imports)]
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
#[allow(unused_imports)]
use scirs2_core::simd::{
    simd_add_f32, simd_add_f64, simd_dot_f32 as core_simd_dot_f32,
    simd_dot_f64 as core_simd_dot_f64, simd_maximum_f32, simd_maximum_f64, simd_minimum_f32,
    simd_minimum_f64, simd_mul_f32, simd_mul_f64, simd_scalar_mul_f32, simd_scalar_mul_f64,
};
#[allow(unused_imports)]
use wide::{f32x8, f64x4};

/// Compute matrix-vector product using SIMD instructions for f32 values
///
/// # Arguments
///
/// * `matrix` - 2D matrix of shape (m, n)
/// * `vector` - 1D vector of shape (n,)
///
/// # Returns
///
/// * Result vector of shape (m,)
#[cfg(feature = "simd")]
pub fn simd_matvec_f32(
    matrix: &ArrayView2<f32>,
    vector: &ArrayView1<f32>,
) -> LinalgResult<Array1<f32>> {
    let (nrows, ncols) = matrix.dim();

    if ncols != vector.len() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix columns ({}) must match vector length ({})",
            ncols,
            vector.len()
        )));
    }

    let mut result = Array1::zeros(nrows);
    let vec_slice = vector.as_slice().unwrap();

    // For each row of the matrix
    for (i, row) in matrix.rows().into_iter().enumerate() {
        let row_slice = row.as_slice().unwrap();
        let mut sum = 0.0f32;

        // Process 8 elements at a time with SIMD
        let mut j = 0;
        let chunk_size = 8;

        // Accumulate SIMD vector sums
        let mut sum_vec = f32x8::splat(0.0);

        while j + chunk_size <= ncols {
            // Load chunks from row and vector
            let row_chunk = [
                row_slice[j],
                row_slice[j + 1],
                row_slice[j + 2],
                row_slice[j + 3],
                row_slice[j + 4],
                row_slice[j + 5],
                row_slice[j + 6],
                row_slice[j + 7],
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

            // Convert to SIMD vectors
            let row_vec = f32x8::new(row_chunk);
            let vec_vec = f32x8::new(vec_chunk);

            // Multiply and accumulate
            sum_vec += row_vec * vec_vec;

            j += chunk_size;
        }

        // Extract and sum the SIMD vector components
        let sum_arr: [f32; 8] = sum_vec.into();
        sum += sum_arr.iter().sum::<f32>();

        // Process remaining elements
        for k in j..ncols {
            sum += row_slice[k] * vec_slice[k];
        }

        result[i] = sum;
    }

    Ok(result)
}

/// Compute matrix-vector product using SIMD instructions for f64 values
///
/// # Arguments
///
/// * `matrix` - 2D matrix of shape (m, n)
/// * `vector` - 1D vector of shape (n,)
///
/// # Returns
///
/// * Result vector of shape (m,)
#[cfg(feature = "simd")]
pub fn simd_matvec_f64(
    matrix: &ArrayView2<f64>,
    vector: &ArrayView1<f64>,
) -> LinalgResult<Array1<f64>> {
    let (nrows, ncols) = matrix.dim();

    if ncols != vector.len() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix columns ({}) must match vector length ({})",
            ncols,
            vector.len()
        )));
    }

    let mut result = Array1::zeros(nrows);
    let vec_slice = vector.as_slice().unwrap();

    // For each row of the matrix
    for (i, row) in matrix.rows().into_iter().enumerate() {
        let row_slice = row.as_slice().unwrap();
        let mut sum = 0.0f64;

        // Process 4 elements at a time with SIMD
        let mut j = 0;
        let chunk_size = 4;

        // Accumulate SIMD vector sums
        let mut sum_vec = f64x4::splat(0.0);

        while j + chunk_size <= ncols {
            // Load chunks from row and vector
            let row_chunk = [
                row_slice[j],
                row_slice[j + 1],
                row_slice[j + 2],
                row_slice[j + 3],
            ];
            let vec_chunk = [
                vec_slice[j],
                vec_slice[j + 1],
                vec_slice[j + 2],
                vec_slice[j + 3],
            ];

            // Convert to SIMD vectors
            let row_vec = f64x4::new(row_chunk);
            let vec_vec = f64x4::new(vec_chunk);

            // Multiply and accumulate
            sum_vec += row_vec * vec_vec;

            j += chunk_size;
        }

        // Extract and sum the SIMD vector components
        let sum_arr: [f64; 4] = sum_vec.into();
        sum += sum_arr.iter().sum::<f64>();

        // Process remaining elements
        for k in j..ncols {
            sum += row_slice[k] * vec_slice[k];
        }

        result[i] = sum;
    }

    Ok(result)
}

/// Compute matrix-matrix product using SIMD instructions for f32 values
///
/// This implementation uses cache-friendly blocking and SIMD instructions
/// for improved performance.
///
/// # Arguments
///
/// * `a` - First matrix of shape (m, k)
/// * `b` - Second matrix of shape (k, n)
///
/// # Returns
///
/// * Result matrix of shape (m, n)
#[cfg(feature = "simd")]
pub fn simd_matmul_f32(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> LinalgResult<Array2<f32>> {
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions mismatch: a({}, {}) * b({}, {})",
            m, k1, k2, n
        )));
    }

    let k = k1; // Common dimension
    let mut result = Array2::zeros((m, n));

    // Cache-friendly block sizes
    // These should be tuned based on target CPU cache sizes
    const BLOCK_SIZE_M: usize = 64;
    const BLOCK_SIZE_N: usize = 64;
    const BLOCK_SIZE_K: usize = 64;

    // Loop over blocks
    for i0 in (0..m).step_by(BLOCK_SIZE_M) {
        let i_end = (i0 + BLOCK_SIZE_M).min(m);

        for j0 in (0..n).step_by(BLOCK_SIZE_N) {
            let j_end = (j0 + BLOCK_SIZE_N).min(n);

            for k0 in (0..k).step_by(BLOCK_SIZE_K) {
                let k_end = (k0 + BLOCK_SIZE_K).min(k);

                // Process blocks
                for i in i0..i_end {
                    for j in j0..j_end {
                        // Get row from A and column from B
                        let a_row = a.slice(s![i, k0..k_end]);
                        let b_col = b.slice(s![k0..k_end, j]);

                        // Calculate dot product of row from A and column from B
                        let mut sum = 0.0f32;

                        if let (Some(a_slice), Some(b_slice)) = (a_row.as_slice(), b_col.as_slice())
                        {
                            // Process with SIMD (8 elements at a time)
                            let mut l = 0;
                            let chunk_size = 8;

                            // Accumulate SIMD vector sums
                            let mut sum_vec = f32x8::splat(0.0);

                            while l + chunk_size <= a_row.len() {
                                // Load chunks
                                let a_chunk = [
                                    a_slice[l],
                                    a_slice[l + 1],
                                    a_slice[l + 2],
                                    a_slice[l + 3],
                                    a_slice[l + 4],
                                    a_slice[l + 5],
                                    a_slice[l + 6],
                                    a_slice[l + 7],
                                ];
                                let b_chunk = [
                                    b_slice[l],
                                    b_slice[l + 1],
                                    b_slice[l + 2],
                                    b_slice[l + 3],
                                    b_slice[l + 4],
                                    b_slice[l + 5],
                                    b_slice[l + 6],
                                    b_slice[l + 7],
                                ];

                                // Convert to SIMD vectors
                                let a_vec = f32x8::new(a_chunk);
                                let b_vec = f32x8::new(b_chunk);

                                // Multiply and accumulate
                                sum_vec += a_vec * b_vec;

                                l += chunk_size;
                            }

                            // Extract and sum the SIMD vector components
                            let sum_arr: [f32; 8] = sum_vec.into();
                            sum += sum_arr.iter().sum::<f32>();

                            // Process remaining elements
                            for m in l..a_row.len() {
                                sum += a_slice[m] * b_slice[m];
                            }
                        } else {
                            // Fallback for non-contiguous data
                            for k in 0..a_row.len() {
                                sum += a_row[k] * b_col[k];
                            }
                        }

                        // Accumulate result
                        result[[i, j]] += sum;
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Compute matrix-matrix product using SIMD instructions for f64 values
///
/// This implementation uses cache-friendly blocking and SIMD instructions
/// for improved performance.
///
/// # Arguments
///
/// * `a` - First matrix of shape (m, k)
/// * `b` - Second matrix of shape (k, n)
///
/// # Returns
///
/// * Result matrix of shape (m, n)
#[cfg(feature = "simd")]
pub fn simd_matmul_f64(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions mismatch: a({}, {}) * b({}, {})",
            m, k1, k2, n
        )));
    }

    let k = k1; // Common dimension
    let mut result = Array2::zeros((m, n));

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

            for k0 in (0..k).step_by(BLOCK_SIZE_K) {
                let k_end = (k0 + BLOCK_SIZE_K).min(k);

                // Process blocks
                for i in i0..i_end {
                    for j in j0..j_end {
                        // Get row from A and column from B
                        let a_row = a.slice(s![i, k0..k_end]);
                        let b_col = b.slice(s![k0..k_end, j]);

                        // Calculate dot product of row from A and column from B
                        let mut sum = 0.0f64;

                        if let (Some(a_slice), Some(b_slice)) = (a_row.as_slice(), b_col.as_slice())
                        {
                            // Process with SIMD (4 elements at a time)
                            let mut l = 0;
                            let chunk_size = 4;

                            // Accumulate SIMD vector sums
                            let mut sum_vec = f64x4::splat(0.0);

                            while l + chunk_size <= a_row.len() {
                                // Load chunks
                                let a_chunk =
                                    [a_slice[l], a_slice[l + 1], a_slice[l + 2], a_slice[l + 3]];
                                let b_chunk =
                                    [b_slice[l], b_slice[l + 1], b_slice[l + 2], b_slice[l + 3]];

                                // Convert to SIMD vectors
                                let a_vec = f64x4::new(a_chunk);
                                let b_vec = f64x4::new(b_chunk);

                                // Multiply and accumulate
                                sum_vec += a_vec * b_vec;

                                l += chunk_size;
                            }

                            // Extract and sum the SIMD vector components
                            let sum_arr: [f64; 4] = sum_vec.into();
                            sum += sum_arr.iter().sum::<f64>();

                            // Process remaining elements
                            for m in l..a_row.len() {
                                sum += a_slice[m] * b_slice[m];
                            }
                        } else {
                            // Fallback for non-contiguous data
                            for k in 0..a_row.len() {
                                sum += a_row[k] * b_col[k];
                            }
                        }

                        // Accumulate result
                        result[[i, j]] += sum;
                    }
                }
            }
        }
    }

    Ok(result)
}

/// SIMD accelerated element-wise maximum of two f32 matrices
///
/// # Arguments
///
/// * `a` - First matrix
/// * `b` - Second matrix
///
/// # Returns
///
/// * Matrix containing element-wise maximum values
#[cfg(feature = "simd")]
pub fn simd_matrix_max_f32(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> LinalgResult<Array2<f32>> {
    if a.shape() != b.shape() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions must match for element-wise operations: {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }

    // Create result matrix
    let mut result = Array2::zeros(a.dim());

    // For each row of the matrix, use SIMD on slices where possible
    for i in 0..a.shape()[0] {
        let a_row = a.slice(s![i, ..]);
        let b_row = b.slice(s![i, ..]);

        if let (Some(_), Some(_)) = (a_row.as_slice(), b_row.as_slice()) {
            // Use the core SIMD implementation for the row
            let max_row = simd_maximum_f32(&a_row.view(), &b_row.view());

            for (j, &val) in max_row.iter().enumerate() {
                result[[i, j]] = val;
            }
        } else {
            // Fallback for non-contiguous data
            for j in 0..a.shape()[1] {
                result[[i, j]] = a[[i, j]].max(b[[i, j]]);
            }
        }
    }

    Ok(result)
}

/// SIMD accelerated element-wise maximum of two f64 matrices
///
/// # Arguments
///
/// * `a` - First matrix
/// * `b` - Second matrix
///
/// # Returns
///
/// * Matrix containing element-wise maximum values
#[cfg(feature = "simd")]
pub fn simd_matrix_max_f64(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    if a.shape() != b.shape() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions must match for element-wise operations: {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }

    // Create result matrix
    let mut result = Array2::zeros(a.dim());

    // For each row of the matrix, use SIMD on slices where possible
    for i in 0..a.shape()[0] {
        let a_row = a.slice(s![i, ..]);
        let b_row = b.slice(s![i, ..]);

        if let (Some(_), Some(_)) = (a_row.as_slice(), b_row.as_slice()) {
            // Use the core SIMD implementation for the row
            let max_row = simd_maximum_f64(&a_row.view(), &b_row.view());

            for (j, &val) in max_row.iter().enumerate() {
                result[[i, j]] = val;
            }
        } else {
            // Fallback for non-contiguous data
            for j in 0..a.shape()[1] {
                result[[i, j]] = a[[i, j]].max(b[[i, j]]);
            }
        }
    }

    Ok(result)
}

/// SIMD accelerated element-wise minimum of two f32 matrices
///
/// # Arguments
///
/// * `a` - First matrix
/// * `b` - Second matrix
///
/// # Returns
///
/// * Matrix containing element-wise minimum values
#[cfg(feature = "simd")]
pub fn simd_matrix_min_f32(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> LinalgResult<Array2<f32>> {
    if a.shape() != b.shape() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions must match for element-wise operations: {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }

    // Create result matrix
    let mut result = Array2::zeros(a.dim());

    // For each row of the matrix, use SIMD on slices where possible
    for i in 0..a.shape()[0] {
        let a_row = a.slice(s![i, ..]);
        let b_row = b.slice(s![i, ..]);

        if let (Some(_), Some(_)) = (a_row.as_slice(), b_row.as_slice()) {
            // Use the core SIMD implementation for the row
            let min_row = simd_minimum_f32(&a_row.view(), &b_row.view());

            for (j, &val) in min_row.iter().enumerate() {
                result[[i, j]] = val;
            }
        } else {
            // Fallback for non-contiguous data
            for j in 0..a.shape()[1] {
                result[[i, j]] = a[[i, j]].min(b[[i, j]]);
            }
        }
    }

    Ok(result)
}

/// SIMD accelerated element-wise minimum of two f64 matrices
///
/// # Arguments
///
/// * `a` - First matrix
/// * `b` - Second matrix
///
/// # Returns
///
/// * Matrix containing element-wise minimum values
#[cfg(feature = "simd")]
pub fn simd_matrix_min_f64(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    if a.shape() != b.shape() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions must match for element-wise operations: {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }

    // Create result matrix
    let mut result = Array2::zeros(a.dim());

    // For each row of the matrix, use SIMD on slices where possible
    for i in 0..a.shape()[0] {
        let a_row = a.slice(s![i, ..]);
        let b_row = b.slice(s![i, ..]);

        if let (Some(_), Some(_)) = (a_row.as_slice(), b_row.as_slice()) {
            // Use the core SIMD implementation for the row
            let min_row = simd_minimum_f64(&a_row.view(), &b_row.view());

            for (j, &val) in min_row.iter().enumerate() {
                result[[i, j]] = val;
            }
        } else {
            // Fallback for non-contiguous data
            for j in 0..a.shape()[1] {
                result[[i, j]] = a[[i, j]].min(b[[i, j]]);
            }
        }
    }

    Ok(result)
}

/// SIMD implementation of axpy: y = alpha * x + y
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier
/// * `x` - Source vector
/// * `y` - Target vector (updated in-place)
///
/// # Returns
///
/// * Modified target vector
#[cfg(feature = "simd")]
pub fn simd_axpy_f32(alpha: f32, x: &ArrayView1<f32>, y: &mut Array1<f32>) -> LinalgResult<()> {
    if x.len() != y.len() {
        return Err(LinalgError::ShapeError(format!(
            "Vector dimensions must match: {} vs {}",
            x.len(),
            y.len()
        )));
    }

    let n = x.len();
    let x_slice = x.as_slice().unwrap();
    let y_slice = y.as_slice_mut().unwrap();

    // Process 8 elements at a time with SIMD
    let mut i = 0;
    let chunk_size = 8;
    let alpha_vec = f32x8::splat(alpha);

    while i + chunk_size <= n {
        // Load chunks
        let x_chunk = [
            x_slice[i],
            x_slice[i + 1],
            x_slice[i + 2],
            x_slice[i + 3],
            x_slice[i + 4],
            x_slice[i + 5],
            x_slice[i + 6],
            x_slice[i + 7],
        ];
        let y_chunk = [
            y_slice[i],
            y_slice[i + 1],
            y_slice[i + 2],
            y_slice[i + 3],
            y_slice[i + 4],
            y_slice[i + 5],
            y_slice[i + 6],
            y_slice[i + 7],
        ];

        // Convert to SIMD vectors
        let x_vec = f32x8::new(x_chunk);
        let y_vec = f32x8::new(y_chunk);

        // Compute alpha * x + y
        let result_vec = (alpha_vec * x_vec) + y_vec;

        // Extract and store results
        let result_arr: [f32; 8] = result_vec.into();
        y_slice[i..(i + chunk_size)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    // Process remaining elements
    for j in i..n {
        y_slice[j] += alpha * x_slice[j];
    }

    Ok(())
}

/// SIMD implementation of axpy: y = alpha * x + y
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier
/// * `x` - Source vector
/// * `y` - Target vector (updated in-place)
///
/// # Returns
///
/// * Modified target vector
#[cfg(feature = "simd")]
pub fn simd_axpy_f64(alpha: f64, x: &ArrayView1<f64>, y: &mut Array1<f64>) -> LinalgResult<()> {
    if x.len() != y.len() {
        return Err(LinalgError::ShapeError(format!(
            "Vector dimensions must match: {} vs {}",
            x.len(),
            y.len()
        )));
    }

    let n = x.len();
    let x_slice = x.as_slice().unwrap();
    let y_slice = y.as_slice_mut().unwrap();

    // Process 4 elements at a time with SIMD
    let mut i = 0;
    let chunk_size = 4;
    let alpha_vec = f64x4::splat(alpha);

    while i + chunk_size <= n {
        // Load chunks
        let x_chunk = [x_slice[i], x_slice[i + 1], x_slice[i + 2], x_slice[i + 3]];
        let y_chunk = [y_slice[i], y_slice[i + 1], y_slice[i + 2], y_slice[i + 3]];

        // Convert to SIMD vectors
        let x_vec = f64x4::new(x_chunk);
        let y_vec = f64x4::new(y_chunk);

        // Compute alpha * x + y
        let result_vec = (alpha_vec * x_vec) + y_vec;

        // Extract and store results
        let result_arr: [f64; 4] = result_vec.into();
        y_slice[i..(i + chunk_size)].copy_from_slice(&result_arr[..chunk_size]);

        i += chunk_size;
    }

    // Process remaining elements
    for j in i..n {
        y_slice[j] += alpha * x_slice[j];
    }

    Ok(())
}

/// SIMD accelerated vector dot product for f32 values
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// * Dot product result
#[cfg(feature = "simd")]
pub fn simd_dot_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> LinalgResult<f32> {
    if a.len() != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Vector dimensions must match for dot product: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    let n = a.len();
    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();

    // Process 8 elements at a time with SIMD
    let mut i = 0;
    let chunk_size = 8;
    let mut sum_vec = f32x8::splat(0.0);

    while i + chunk_size <= n {
        // Load chunks
        let a_chunk = [
            a_slice[i],
            a_slice[i + 1],
            a_slice[i + 2],
            a_slice[i + 3],
            a_slice[i + 4],
            a_slice[i + 5],
            a_slice[i + 6],
            a_slice[i + 7],
        ];
        let b_chunk = [
            b_slice[i],
            b_slice[i + 1],
            b_slice[i + 2],
            b_slice[i + 3],
            b_slice[i + 4],
            b_slice[i + 5],
            b_slice[i + 6],
            b_slice[i + 7],
        ];

        // Convert to SIMD vectors
        let a_vec = f32x8::new(a_chunk);
        let b_vec = f32x8::new(b_chunk);

        // Multiply and accumulate
        sum_vec += a_vec * b_vec;

        i += chunk_size;
    }

    // Extract and sum the SIMD vector components
    let sum_arr: [f32; 8] = sum_vec.into();
    let mut sum = sum_arr.iter().sum::<f32>();

    // Process remaining elements
    for j in i..n {
        sum += a_slice[j] * b_slice[j];
    }

    Ok(sum)
}

/// SIMD accelerated vector dot product for f64 values
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// * Dot product result
#[cfg(feature = "simd")]
pub fn simd_dot_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> LinalgResult<f64> {
    if a.len() != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Vector dimensions must match for dot product: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    let n = a.len();
    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();

    // Process 4 elements at a time with SIMD
    let mut i = 0;
    let chunk_size = 4;
    let mut sum_vec = f64x4::splat(0.0);

    while i + chunk_size <= n {
        // Load chunks
        let a_chunk = [a_slice[i], a_slice[i + 1], a_slice[i + 2], a_slice[i + 3]];
        let b_chunk = [b_slice[i], b_slice[i + 1], b_slice[i + 2], b_slice[i + 3]];

        // Convert to SIMD vectors
        let a_vec = f64x4::new(a_chunk);
        let b_vec = f64x4::new(b_chunk);

        // Multiply and accumulate
        sum_vec += a_vec * b_vec;

        i += chunk_size;
    }

    // Extract and sum the SIMD vector components
    let sum_arr: [f64; 4] = sum_vec.into();
    let mut sum = sum_arr.iter().sum::<f64>();

    // Process remaining elements
    for j in i..n {
        sum += a_slice[j] * b_slice[j];
    }

    Ok(sum)
}
