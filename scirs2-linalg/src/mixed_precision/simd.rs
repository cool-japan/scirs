//! SIMD accelerated mixed-precision operations
//!
//! This module provides SIMD-accelerated implementations of mixed-precision
//! linear algebra operations for improved performance while maintaining
//! numerical stability.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Perform mixed-precision matrix-vector multiplication with SIMD acceleration
///
/// This function computes the matrix-vector product using internal higher precision
/// while leveraging SIMD instructions for performance.
///
/// # Arguments
///
/// * `matrix` - Input matrix in f32 precision
/// * `vector` - Input vector in f32 precision
///
/// # Returns
///
/// * Result vector in precision C with computation in f64 precision
///
/// # Type Parameters
///
/// * `C` - Output vector precision
///
/// # Examples
///
/// ```
/// #[cfg(feature = "simd")]
/// {
///     use scirs2_core::ndarray::array;
///     use scirs2_linalg::mixed_precision::simd_mixed_precision_matvec_f32_f64;
///
///     let a_f32 = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
///     let x_f32 = array![0.5f32, 0.5f32];
///
///     // Compute result with SIMD-accelerated mixed precision
///     let y = simd_mixed_precision_matvec_f32_f64::<f32>(
///         &a_f32.view(),
///         &x_f32.view()
///     ).unwrap();
///
///     assert_eq!(y.len(), 2);
///     assert!((y[0] - 1.5f32).abs() < 1e-6);
///     assert!((y[1] - 3.5f32).abs() < 1e-6);
/// }
/// ```
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_mixed_precision_matvec_f32_f64<C>(
    matrix: &ArrayView2<f32>,
    vector: &ArrayView1<f32>,
) -> LinalgResult<Array1<C>>
where
    C: Clone + scirs2_core::numeric::Zero + scirs2_core::numeric::NumCast + std::fmt::Debug,
{
    let (nrows, ncols) = matrix.dim();

    if ncols != vector.len() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix columns ({}) must match vector length ({})",
            ncols,
            vector.len()
        )));
    }

    let mut result = Array1::<C>::zeros(nrows);

    if let (Some(matrix_slice), Some(vector_slice)) = (matrix.as_slice(), vector.as_slice()) {
        // Process matrix rows with SIMD
        for i in 0..nrows {
            // Extract current row
            let row_start = i * ncols;
            let row_end = row_start + ncols;
            let row_slice = &matrix_slice[row_start..row_end];

            // Compute dot product in higher precision using SIMD
            let mut j = 0;
            let chunksize = 4; // Process 4 elements at a time
            let mut sum = 0.0f64;

            while j + chunksize <= ncols {
                // Load chunks and convert to f64
                let row_chunk_f64 = [
                    row_slice[j] as f64,
                    row_slice[j + 1] as f64,
                    row_slice[j + 2] as f64,
                    row_slice[j + 3] as f64,
                ];
                let vec_chunk_f64 = [
                    vector_slice[j] as f64,
                    vector_slice[j + 1] as f64,
                    vector_slice[j + 2] as f64,
                    vector_slice[j + 3] as f64,
                ];

                // Convert to views and compute dot product using core SIMD in f64
                let row_view = ArrayView1::from(&row_chunk_f64);
                let vec_view = ArrayView1::from(&vec_chunk_f64);
                sum += f64::simd_dot(&row_view, &vec_view);

                j += chunksize;
            }

            // Process remaining elements
            for k in j..ncols {
                // Compute in higher precision
                sum += (row_slice[k] as f64) * (vector_slice[k] as f64);
            }

            // Convert back to desired output precision
            result[i] = C::from(sum).unwrap_or_else(|| C::zero());
        }
    } else {
        // Fallback for non-contiguous data
        for i in 0..nrows {
            let row = matrix.row(i);
            let mut sum = 0.0f64;

            for j in 0..ncols {
                // Manual element-wise in high precision
                sum += (row[j] as f64) * (vector[j] as f64);
            }

            // Convert back to desired output precision
            result[i] = C::from(sum).unwrap_or_else(|| C::zero());
        }
    }

    Ok(result)
}

/// Perform mixed-precision matrix multiplication with SIMD acceleration
///
/// This function computes the matrix-matrix product using internal higher precision
/// while leveraging SIMD instructions for performance.
///
/// # Arguments
///
/// * `a` - First matrix in f32 precision
/// * `b` - Second matrix in f32 precision
///
/// # Returns
///
/// * Result matrix in precision C with computation in f64 precision
///
/// # Type Parameters
///
/// * `C` - Output matrix precision
///
/// # Examples
///
/// ```
/// #[cfg(feature = "simd")]
/// {
///     use scirs2_core::ndarray::array;
///     use scirs2_linalg::mixed_precision::simd_mixed_precision_matmul_f32_f64;
///
///     let a_f32 = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
///     let b_f32 = array![[5.0f32, 6.0f32], [7.0f32, 8.0f32]];
///
///     // Compute result with SIMD-accelerated mixed precision
///     let c = simd_mixed_precision_matmul_f32_f64::<f32>(
///         &a_f32.view(),
///         &b_f32.view()
///     ).unwrap();
///
///     assert_eq!(c.shape(), &[2, 2]);
///     assert!((c[[0, 0]] - 19.0f32).abs() < 1e-5);
///     assert!((c[[0, 1]] - 22.0f32).abs() < 1e-5);
///     assert!((c[[1, 0]] - 43.0f32).abs() < 1e-5);
///     assert!((c[[1, 1]] - 50.0f32).abs() < 1e-5);
/// }
/// ```
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_mixed_precision_matmul_f32_f64<C>(
    a: &ArrayView2<f32>,
    b: &ArrayView2<f32>,
) -> LinalgResult<Array2<C>>
where
    C: Clone + scirs2_core::numeric::Zero + scirs2_core::numeric::NumCast + std::fmt::Debug,
{
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions mismatch: a({m}, {k1}) * b({k2}, {n})"
        )));
    }

    let k = k1; // Common dimension
    let mut result = Array2::<C>::zeros((m, n));

    // Cache-friendly block sizes
    const BLOCK_SIZE_M: usize = 32;
    const BLOCK_SIZE_N: usize = 32;
    const BLOCK_SIZE_K: usize = 32;

    // Loop over blocks
    for i0 in (0..m).step_by(BLOCK_SIZE_M) {
        let i_end = (i0 + BLOCK_SIZE_M).min(m);

        for j0 in (0..n).step_by(BLOCK_SIZE_N) {
            let j_end = (j0 + BLOCK_SIZE_N).min(n);

            // Initialize result block to zero in higher precision
            let mut c_high = Array2::<f64>::zeros((i_end - i0, j_end - j0));

            for k0 in (0..k).step_by(BLOCK_SIZE_K) {
                let k_end = (k0 + BLOCK_SIZE_K).min(k);

                // Process the current blocks with high precision and SIMD
                if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
                    // Process each cell in the current block
                    for i_local in 0..(i_end - i0) {
                        let i = i0 + i_local;

                        for j_local in 0..(j_end - j0) {
                            let j = j0 + j_local;

                            // Extract row from A and column from B for this k block
                            let a_row_start = i * k + k0;
                            let a_row_end = a_row_start + (k_end - k0);
                            let a_row_slice = &a_slice[a_row_start..a_row_end];

                            // Compute dot product in higher precision using SIMD
                            let mut l = 0;
                            let chunksize = 4; // Process 4 elements at a time
                            let mut block_sum = 0.0f64;

                            while l + chunksize <= (k_end - k0) {
                                // Extract and convert column slice from B (with stride handling)
                                let b_col_indices = [
                                    (k0 + l) * n + j,
                                    (k0 + l + 1) * n + j,
                                    (k0 + l + 2) * n + j,
                                    (k0 + l + 3) * n + j,
                                ];

                                // Load chunks and convert to f64
                                let a_chunk_f64 = [
                                    a_row_slice[l] as f64,
                                    a_row_slice[l + 1] as f64,
                                    a_row_slice[l + 2] as f64,
                                    a_row_slice[l + 3] as f64,
                                ];
                                let b_chunk_f64 = [
                                    b_slice[b_col_indices[0]] as f64,
                                    b_slice[b_col_indices[1]] as f64,
                                    b_slice[b_col_indices[2]] as f64,
                                    b_slice[b_col_indices[3]] as f64,
                                ];

                                // Convert to views and compute dot product using core SIMD in f64
                                let a_view = ArrayView1::from(&a_chunk_f64);
                                let b_view = ArrayView1::from(&b_chunk_f64);
                                block_sum += f64::simd_dot(&a_view, &b_view);

                                l += chunksize;
                            }

                            // Process remaining elements
                            for (offset, &a_val) in
                                a_row_slice.iter().enumerate().skip(l).take(k_end - k0 - l)
                            {
                                let l_remain = offset;
                                let b_idx = (k0 + l_remain) * n + j;
                                // Compute in higher precision
                                block_sum += (a_val as f64) * (b_slice[b_idx] as f64);
                            }

                            // Accumulate result in high precision
                            c_high[[i_local, j_local]] += block_sum;
                        }
                    }
                } else {
                    // Fallback for non-contiguous data
                    for i_local in 0..(i_end - i0) {
                        let i = i0 + i_local;

                        for j_local in 0..(j_end - j0) {
                            let j = j0 + j_local;

                            let mut sum = 0.0f64;

                            for k_idx in k0..k_end {
                                // Manual element-wise in high precision
                                sum += (a[[i, k_idx]] as f64) * (b[[k_idx, j]] as f64);
                            }

                            // Accumulate in high precision
                            c_high[[i_local, j_local]] += sum;
                        }
                    }
                }
            }

            // Convert back to desired output precision
            for i_local in 0..(i_end - i0) {
                let i = i0 + i_local;

                for j_local in 0..(j_end - j0) {
                    let j = j0 + j_local;

                    // Convert from high precision to output precision
                    result[[i, j]] =
                        C::from(c_high[[i_local, j_local]]).unwrap_or_else(|| C::zero());
                }
            }
        }
    }

    Ok(result)
}

/// Compute dot product with mixed precision using SIMD
///
/// This function computes the dot product of two vectors using internal
/// higher precision for increased accuracy with SIMD optimization.
///
/// # Arguments
///
/// * `a` - First vector in f32 precision
/// * `b` - Second vector in f32 precision
///
/// # Returns
///
/// * Dot product result in the requested precision
///
/// # Type Parameters
///
/// * `C` - Output scalar precision
///
/// # Examples
///
/// ```
/// #[cfg(feature = "simd")]
/// {
///     use scirs2_core::ndarray::array;
///     use scirs2_linalg::mixed_precision::simd_mixed_precision_dot_f32_f64;
///
///     let a_f32 = array![1.0f32, 2.0f32, 3.0f32];
///     let b_f32 = array![4.0f32, 5.0f32, 6.0f32];
///
///     // Compute dot product with SIMD-accelerated mixed precision
///     let result = simd_mixed_precision_dot_f32_f64::<f32>(
///         &a_f32.view(),
///         &b_f32.view()
///     ).unwrap();
///
///     // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
///     assert!((result - 32.0f32).abs() < 1e-6);
///
///     // For higher precision output
///     let result_f64 = simd_mixed_precision_dot_f32_f64::<f64>(
///         &a_f32.view(),
///         &b_f32.view()
///     ).unwrap();
///
///     assert!((result_f64 - 32.0f64).abs() < 1e-14);
/// }
/// ```
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_mixed_precision_dot_f32_f64<C>(
    a: &ArrayView1<f32>,
    b: &ArrayView1<f32>,
) -> LinalgResult<C>
where
    C: Clone + scirs2_core::numeric::Zero + scirs2_core::numeric::NumCast + std::fmt::Debug,
{
    if a.len() != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Vector dimensions must match for dot product: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    let n = a.len();

    if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
        // Process with SIMD in higher precision
        let mut i = 0;
        let chunksize = 4; // Process 4 elements at a time
        let mut sum = 0.0f64;

        while i + chunksize <= n {
            // Load chunks and convert to f64
            let a_chunk_f64 = [
                a_slice[i] as f64,
                a_slice[i + 1] as f64,
                a_slice[i + 2] as f64,
                a_slice[i + 3] as f64,
            ];
            let b_chunk_f64 = [
                b_slice[i] as f64,
                b_slice[i + 1] as f64,
                b_slice[i + 2] as f64,
                b_slice[i + 3] as f64,
            ];

            // Convert to views and compute dot product using core SIMD in f64
            let a_view = ArrayView1::from(&a_chunk_f64);
            let b_view = ArrayView1::from(&b_chunk_f64);
            sum += f64::simd_dot(&a_view, &b_view);

            i += chunksize;
        }

        // Process remaining elements
        for j in i..n {
            // Compute in higher precision
            sum += (a_slice[j] as f64) * (b_slice[j] as f64);
        }

        // Convert back to desired output precision
        C::from(sum).ok_or_else(|| {
            LinalgError::ComputationError("Failed to convert dot product result".to_string())
        })
    } else {
        // Fallback for non-contiguous data
        let mut sum = 0.0f64;

        for i in 0..n {
            // Manual element-wise in high precision
            sum += (a[i] as f64) * (b[i] as f64);
        }

        // Convert back to desired output precision
        C::from(sum).ok_or_else(|| {
            LinalgError::ComputationError("Failed to convert dot product result".to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simd_mixed_precision_matvec() {
        // Create test matrices and vectors
        let mat = array![[1.0e-4f32, 2.0e4, 3.0e-4], [4.0e4, 5.0e-4, 6.0e4]];

        let vec = array![7.0e-4f32, 8.0e4, 9.0e-4];

        // Compute using SIMD mixed precision
        let result_f32 =
            simd_mixed_precision_matvec_f32_f64::<f32>(&mat.view(), &vec.view()).unwrap();
        let result_f64 =
            simd_mixed_precision_matvec_f32_f64::<f64>(&mat.view(), &vec.view()).unwrap();

        // Compute reference using standard f32 and f64
        let mut expected_f32 = Array1::<f32>::zeros(2);
        let mut expected_f64 = Array1::<f64>::zeros(2);

        // Manual calculation with explicit conversions
        for i in 0..2 {
            let mut sum_f64 = 0.0f64;
            for j in 0..3 {
                sum_f64 += (mat[[i, j]] as f64) * (vec[j] as f64);
            }
            expected_f32[i] = sum_f64 as f32;
            expected_f64[i] = sum_f64;
        }

        // Verify correctness
        assert_eq!(result_f32.len(), expected_f32.len());
        assert_eq!(result_f64.len(), expected_f64.len());

        for i in 0..2 {
            assert_relative_eq!(result_f32[i], expected_f32[i], epsilon = 1e-5);
            assert_relative_eq!(result_f64[i], expected_f64[i], epsilon = 1e-14);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simd_mixed_precision_matmul() {
        // Create test matrices
        let a = array![[1.0e-4f32, 2.0e4, 3.0e-4], [4.0e4, 5.0e-4, 6.0e4]];

        let b = array![[7.0e-4f32, 8.0e-4], [9.0e4, 1.0e5], [2.0e-4, 3.0e-4]];

        // Compute using SIMD mixed precision
        let result_f32 = simd_mixed_precision_matmul_f32_f64::<f32>(&a.view(), &b.view()).unwrap();
        let result_f64 = simd_mixed_precision_matmul_f32_f64::<f64>(&a.view(), &b.view()).unwrap();

        // Compute reference using standard f32 and f64
        let mut expected_f32 = Array2::<f32>::zeros((2, 2));
        let mut expected_f64 = Array2::<f64>::zeros((2, 2));

        // Manual calculation with explicit conversions
        for i in 0..2 {
            for j in 0..2 {
                let mut sum_f64 = 0.0f64;
                for k in 0..3 {
                    sum_f64 += (a[[i, k]] as f64) * (b[[k, j]] as f64);
                }
                expected_f32[[i, j]] = sum_f64 as f32;
                expected_f64[[i, j]] = sum_f64;
            }
        }

        // Verify correctness
        assert_eq!(result_f32.shape(), expected_f32.shape());
        assert_eq!(result_f64.shape(), expected_f64.shape());

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(result_f32[[i, j]], expected_f32[[i, j]], epsilon = 1e-5);
                assert_relative_eq!(result_f64[[i, j]], expected_f64[[i, j]], epsilon = 1e-14);
            }
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simd_mixed_precision_dot() {
        // Create test vectors with very small and very large values to highlight precision issues
        let a = array![1.0e-7f32, 2.0e7, 3.0e-7, 4.0e7, 5.0e-7, 6.0e7, 7.0e-7, 8.0e7, 9.0e-7];
        let b = array![9.0e-7f32, 8.0e7, 7.0e-7, 6.0e7, 5.0e-7, 4.0e7, 3.0e-7, 2.0e7, 1.0e-7];

        // Compute using SIMD mixed precision
        let result_f32 = simd_mixed_precision_dot_f32_f64::<f32>(&a.view(), &b.view()).unwrap();
        let result_f64 = simd_mixed_precision_dot_f32_f64::<f64>(&a.view(), &b.view()).unwrap();

        // Compute reference using explicit f64 conversion
        let mut expected_f64 = 0.0f64;
        for i in 0..a.len() {
            expected_f64 += (a[i] as f64) * (b[i] as f64);
        }
        let expected_f32 = expected_f64 as f32;

        // Verify correctness
        assert_relative_eq!(result_f32, expected_f32, epsilon = 1e-5);
        assert_relative_eq!(result_f64, expected_f64, epsilon = 1e-14);

        // Verify the advantage of using mixed precision for values with large dynamic range
        let direct_f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f32>();
        println!("Direct f32 computation: {}", direct_f32);
        println!("Mixed precision computation (f64): {}", result_f64);
        println!("Mixed precision computation (f32): {}", result_f32);

        // The mixed precision result should be more accurate than direct f32 computation
        assert!(
            (direct_f32 as f64 - expected_f64).abs() >= (result_f32 as f64 - expected_f64).abs()
        );
    }
}
