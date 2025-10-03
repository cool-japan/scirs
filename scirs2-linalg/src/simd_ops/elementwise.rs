//! SIMD-accelerated element-wise matrix operations
//!
//! This module provides SIMD implementations of common element-wise operations
//! like addition, subtraction, multiplication, and more advanced operations.
//! All SIMD operations are delegated to scirs2-core::simd_ops for unified optimization.

#[cfg(feature = "simd")]
use crate::error::{LinalgError, LinalgResult};
#[cfg(feature = "simd")]
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2};
#[cfg(feature = "simd")]
use scirs2_core::simd_ops::SimdUnifiedOps;

/// SIMD-accelerated element-wise matrix addition for f32
///
/// Computes C = A + B using SIMD instructions
///
/// # Arguments
///
/// * `a` - First input matrix
/// * `b` - Second input matrix
///
/// # Returns
///
/// * Result matrix C = A + B
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simdmatrix_add_f32(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> LinalgResult<Array2<f32>> {
    if a.dim() != b.dim() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions must match: {:?} vs {:?}",
            a.dim(),
            b.dim()
        )));
    }

    let (rows, cols) = a.dim();
    let mut result = Array2::zeros((rows, cols));

    // Process row by row using unified SIMD operations
    for i in 0..rows {
        let a_row = a.row(i);
        let b_row = b.row(i);
        let sum_row = f32::simd_add(&a_row, &b_row);
        for (j, &val) in sum_row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    Ok(result)
}

/// SIMD-accelerated element-wise matrix addition for f64
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simdmatrix_add_f64(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    if a.dim() != b.dim() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions must match: {:?} vs {:?}",
            a.dim(),
            b.dim()
        )));
    }

    let (rows, cols) = a.dim();
    let mut result = Array2::zeros((rows, cols));

    // Process row by row using unified SIMD operations
    for i in 0..rows {
        let a_row = a.row(i);
        let b_row = b.row(i);
        let sum_row = f64::simd_add(&a_row, &b_row);
        for (j, &val) in sum_row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    Ok(result)
}

/// SIMD-accelerated in-place element-wise matrix addition for f32
///
/// Computes A += B using SIMD instructions
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simdmatrix_add_inplace_f32(
    a: &mut ArrayViewMut2<f32>,
    b: &ArrayView2<f32>,
) -> LinalgResult<()> {
    if a.dim() != b.dim() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions must match: {:?} vs {:?}",
            a.dim(),
            b.dim()
        )));
    }

    // Process row by row using unified SIMD operations
    for (mut a_row, b_row) in a.rows_mut().into_iter().zip(b.rows()) {
        let a_copy = Array1::from(a_row.to_vec());
        let sum_row = f32::simd_add(&a_copy.view(), &b_row);
        for (a_elem, &sum_elem) in a_row.iter_mut().zip(sum_row.iter()) {
            *a_elem = sum_elem;
        }
    }

    Ok(())
}

/// SIMD-accelerated element-wise matrix multiplication (Hadamard product) for f32
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simdmatrix_mul_elementwise_f32(
    a: &ArrayView2<f32>,
    b: &ArrayView2<f32>,
) -> LinalgResult<Array2<f32>> {
    if a.dim() != b.dim() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions must match: {:?} vs {:?}",
            a.dim(),
            b.dim()
        )));
    }

    let (rows, cols) = a.dim();
    let mut result = Array2::zeros((rows, cols));

    // Process row by row using unified SIMD operations
    for i in 0..rows {
        let a_row = a.row(i);
        let b_row = b.row(i);
        let mul_row = f32::simd_mul(&a_row, &b_row);
        for (j, &val) in mul_row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    Ok(result)
}

/// SIMD-accelerated scalar multiplication for f32
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simdmatrix_scale_f32(a: &ArrayView2<f32>, scalar: f32) -> LinalgResult<Array2<f32>> {
    let (rows, cols) = a.dim();
    let mut result = Array2::zeros((rows, cols));

    // Process row by row using unified SIMD operations
    for i in 0..rows {
        let a_row = a.row(i);
        let scaled_row = f32::simd_scalar_mul(&a_row, scalar);
        for (j, &val) in scaled_row.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    Ok(result)
}

// Note: Helper functions have been removed as we now use unified SIMD operations from scirs2-core

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simdmatrix_add_f32() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[5.0f32, 6.0], [7.0, 8.0]];

        let result = simdmatrix_add_f32(&a.view(), &b.view()).unwrap();

        assert_relative_eq!(result[[0, 0]], 6.0);
        assert_relative_eq!(result[[0, 1]], 8.0);
        assert_relative_eq!(result[[1, 0]], 10.0);
        assert_relative_eq!(result[[1, 1]], 12.0);
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simdmatrix_scale_f32() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let scalar = 2.5f32;

        let result = simdmatrix_scale_f32(&a.view(), scalar).unwrap();

        assert_relative_eq!(result[[0, 0]], 2.5);
        assert_relative_eq!(result[[0, 1]], 5.0);
        assert_relative_eq!(result[[1, 0]], 7.5);
        assert_relative_eq!(result[[1, 1]], 10.0);
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simdmatrix_mul_elementwise_f32() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[2.0f32, 3.0], [4.0, 5.0]];

        let result = simdmatrix_mul_elementwise_f32(&a.view(), &b.view()).unwrap();

        assert_relative_eq!(result[[0, 0]], 2.0);
        assert_relative_eq!(result[[0, 1]], 6.0);
        assert_relative_eq!(result[[1, 0]], 12.0);
        assert_relative_eq!(result[[1, 1]], 20.0);
    }
}
