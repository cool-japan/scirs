//! SIMD-accelerated element-wise matrix operations
//!
//! This module provides SIMD implementations of common element-wise operations
//! like addition, subtraction, multiplication, and more advanced operations.

#[cfg(feature = "simd")]
use crate::error::{LinalgError, LinalgResult};
#[cfg(feature = "simd")]
use ndarray::{Array2, ArrayView2, ArrayViewMut2};
#[cfg(feature = "simd")]
use wide::{f32x8, f64x4};

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
pub fn simd_matrix_add_f32(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> LinalgResult<Array2<f32>> {
    if a.dim() != b.dim() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions must match: {:?} vs {:?}",
            a.dim(),
            b.dim()
        )));
    }

    let (rows, cols) = a.dim();
    let mut result = Array2::zeros((rows, cols));

    // Try to use flat SIMD processing if matrices are contiguous
    if let (Some(a_slice), Some(b_slice), Some(result_slice)) =
        (a.as_slice(), b.as_slice(), result.as_slice_mut())
    {
        simd_add_flat_f32(a_slice, b_slice, result_slice);
    } else {
        // Process row by row
        for ((mut result_row, a_row), b_row) in
            result.rows_mut().into_iter().zip(a.rows()).zip(b.rows())
        {
            if let (Some(a_slice), Some(b_slice), Some(result_slice)) = (
                a_row.as_slice(),
                b_row.as_slice(),
                result_row.as_slice_mut(),
            ) {
                simd_add_flat_f32(a_slice, b_slice, result_slice);
            } else {
                // Fallback to scalar addition
                for ((result_elem, &a_elem), &b_elem) in
                    result_row.iter_mut().zip(a_row.iter()).zip(b_row.iter())
                {
                    *result_elem = a_elem + b_elem;
                }
            }
        }
    }

    Ok(result)
}

/// SIMD-accelerated element-wise matrix addition for f64
#[cfg(feature = "simd")]
pub fn simd_matrix_add_f64(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    if a.dim() != b.dim() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions must match: {:?} vs {:?}",
            a.dim(),
            b.dim()
        )));
    }

    let (rows, cols) = a.dim();
    let mut result = Array2::zeros((rows, cols));

    if let (Some(a_slice), Some(b_slice), Some(result_slice)) =
        (a.as_slice(), b.as_slice(), result.as_slice_mut())
    {
        simd_add_flat_f64(a_slice, b_slice, result_slice);
    } else {
        for ((mut result_row, a_row), b_row) in
            result.rows_mut().into_iter().zip(a.rows()).zip(b.rows())
        {
            if let (Some(a_slice), Some(b_slice), Some(result_slice)) = (
                a_row.as_slice(),
                b_row.as_slice(),
                result_row.as_slice_mut(),
            ) {
                simd_add_flat_f64(a_slice, b_slice, result_slice);
            } else {
                for ((result_elem, &a_elem), &b_elem) in
                    result_row.iter_mut().zip(a_row.iter()).zip(b_row.iter())
                {
                    *result_elem = a_elem + b_elem;
                }
            }
        }
    }

    Ok(result)
}

/// SIMD-accelerated in-place element-wise matrix addition for f32
///
/// Computes A += B using SIMD instructions
#[cfg(feature = "simd")]
pub fn simd_matrix_add_inplace_f32(
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

    if let (Some(a_slice), Some(b_slice)) = (a.as_slice_mut(), b.as_slice()) {
        simd_add_inplace_flat_f32(a_slice, b_slice);
    } else {
        for (mut a_row, b_row) in a.rows_mut().into_iter().zip(b.rows()) {
            if let (Some(a_slice), Some(b_slice)) = (a_row.as_slice_mut(), b_row.as_slice()) {
                simd_add_inplace_flat_f32(a_slice, b_slice);
            } else {
                for (a_elem, &b_elem) in a_row.iter_mut().zip(b_row.iter()) {
                    *a_elem += b_elem;
                }
            }
        }
    }

    Ok(())
}

/// SIMD-accelerated element-wise matrix multiplication (Hadamard product) for f32
#[cfg(feature = "simd")]
pub fn simd_matrix_mul_elementwise_f32(
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

    if let (Some(a_slice), Some(b_slice), Some(result_slice)) =
        (a.as_slice(), b.as_slice(), result.as_slice_mut())
    {
        simd_mul_flat_f32(a_slice, b_slice, result_slice);
    } else {
        for ((mut result_row, a_row), b_row) in
            result.rows_mut().into_iter().zip(a.rows()).zip(b.rows())
        {
            if let (Some(a_slice), Some(b_slice), Some(result_slice)) = (
                a_row.as_slice(),
                b_row.as_slice(),
                result_row.as_slice_mut(),
            ) {
                simd_mul_flat_f32(a_slice, b_slice, result_slice);
            } else {
                for ((result_elem, &a_elem), &b_elem) in
                    result_row.iter_mut().zip(a_row.iter()).zip(b_row.iter())
                {
                    *result_elem = a_elem * b_elem;
                }
            }
        }
    }

    Ok(result)
}

/// SIMD-accelerated scalar multiplication for f32
#[cfg(feature = "simd")]
pub fn simd_matrix_scale_f32(a: &ArrayView2<f32>, scalar: f32) -> LinalgResult<Array2<f32>> {
    let (rows, cols) = a.dim();
    let mut result = Array2::zeros((rows, cols));

    if let (Some(a_slice), Some(result_slice)) = (a.as_slice(), result.as_slice_mut()) {
        simd_scale_flat_f32(a_slice, scalar, result_slice);
    } else {
        for (mut result_row, a_row) in result.rows_mut().into_iter().zip(a.rows()) {
            if let (Some(a_slice), Some(result_slice)) =
                (a_row.as_slice(), result_row.as_slice_mut())
            {
                simd_scale_flat_f32(a_slice, scalar, result_slice);
            } else {
                for (result_elem, &a_elem) in result_row.iter_mut().zip(a_row.iter()) {
                    *result_elem = a_elem * scalar;
                }
            }
        }
    }

    Ok(result)
}

// Helper functions for flat array SIMD operations

#[cfg(feature = "simd")]
fn simd_add_flat_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;

    // Process 8 elements at a time with SIMD
    while i + 8 <= len {
        let a_chunk = f32x8::from([
            a[i],
            a[i + 1],
            a[i + 2],
            a[i + 3],
            a[i + 4],
            a[i + 5],
            a[i + 6],
            a[i + 7],
        ]);
        let b_chunk = f32x8::from([
            b[i],
            b[i + 1],
            b[i + 2],
            b[i + 3],
            b[i + 4],
            b[i + 5],
            b[i + 6],
            b[i + 7],
        ]);

        let result_chunk = a_chunk + b_chunk;
        let result_array: [f32; 8] = result_chunk.into();

        result[i..i + 8].copy_from_slice(&result_array);
        i += 8;
    }

    // Handle remaining elements
    for j in i..len {
        result[j] = a[j] + b[j];
    }
}

#[cfg(feature = "simd")]
fn simd_add_flat_f64(a: &[f64], b: &[f64], result: &mut [f64]) {
    let len = a.len();
    let mut i = 0;

    // Process 4 elements at a time with SIMD
    while i + 4 <= len {
        let a_chunk = f64x4::from([a[i], a[i + 1], a[i + 2], a[i + 3]]);
        let b_chunk = f64x4::from([b[i], b[i + 1], b[i + 2], b[i + 3]]);

        let result_chunk = a_chunk + b_chunk;
        let result_array: [f64; 4] = result_chunk.into();

        result[i..i + 4].copy_from_slice(&result_array);
        i += 4;
    }

    // Handle remaining elements
    for j in i..len {
        result[j] = a[j] + b[j];
    }
}

#[cfg(feature = "simd")]
fn simd_add_inplace_flat_f32(a: &mut [f32], b: &[f32]) {
    let len = a.len();
    let mut i = 0;

    while i + 8 <= len {
        let a_chunk = f32x8::from([
            a[i],
            a[i + 1],
            a[i + 2],
            a[i + 3],
            a[i + 4],
            a[i + 5],
            a[i + 6],
            a[i + 7],
        ]);
        let b_chunk = f32x8::from([
            b[i],
            b[i + 1],
            b[i + 2],
            b[i + 3],
            b[i + 4],
            b[i + 5],
            b[i + 6],
            b[i + 7],
        ]);

        let result_chunk = a_chunk + b_chunk;
        let result_array: [f32; 8] = result_chunk.into();

        a[i..i + 8].copy_from_slice(&result_array);
        i += 8;
    }

    for j in i..len {
        a[j] += b[j];
    }
}

#[cfg(feature = "simd")]
fn simd_mul_flat_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;

    while i + 8 <= len {
        let a_chunk = f32x8::from([
            a[i],
            a[i + 1],
            a[i + 2],
            a[i + 3],
            a[i + 4],
            a[i + 5],
            a[i + 6],
            a[i + 7],
        ]);
        let b_chunk = f32x8::from([
            b[i],
            b[i + 1],
            b[i + 2],
            b[i + 3],
            b[i + 4],
            b[i + 5],
            b[i + 6],
            b[i + 7],
        ]);

        let result_chunk = a_chunk * b_chunk;
        let result_array: [f32; 8] = result_chunk.into();

        result[i..i + 8].copy_from_slice(&result_array);
        i += 8;
    }

    for j in i..len {
        result[j] = a[j] * b[j];
    }
}

#[cfg(feature = "simd")]
fn simd_scale_flat_f32(a: &[f32], scalar: f32, result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    let scalar_vec = f32x8::splat(scalar);

    while i + 8 <= len {
        let a_chunk = f32x8::from([
            a[i],
            a[i + 1],
            a[i + 2],
            a[i + 3],
            a[i + 4],
            a[i + 5],
            a[i + 6],
            a[i + 7],
        ]);

        let result_chunk = a_chunk * scalar_vec;
        let result_array: [f32; 8] = result_chunk.into();

        result[i..i + 8].copy_from_slice(&result_array);
        i += 8;
    }

    for j in i..len {
        result[j] = a[j] * scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_matrix_add_f32() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[5.0f32, 6.0], [7.0, 8.0]];

        let result = simd_matrix_add_f32(&a.view(), &b.view()).unwrap();

        assert_relative_eq!(result[[0, 0]], 6.0);
        assert_relative_eq!(result[[0, 1]], 8.0);
        assert_relative_eq!(result[[1, 0]], 10.0);
        assert_relative_eq!(result[[1, 1]], 12.0);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_matrix_scale_f32() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let scalar = 2.5f32;

        let result = simd_matrix_scale_f32(&a.view(), scalar).unwrap();

        assert_relative_eq!(result[[0, 0]], 2.5);
        assert_relative_eq!(result[[0, 1]], 5.0);
        assert_relative_eq!(result[[1, 0]], 7.5);
        assert_relative_eq!(result[[1, 1]], 10.0);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_matrix_mul_elementwise_f32() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[2.0f32, 3.0], [4.0, 5.0]];

        let result = simd_matrix_mul_elementwise_f32(&a.view(), &b.view()).unwrap();

        assert_relative_eq!(result[[0, 0]], 2.0);
        assert_relative_eq!(result[[0, 1]], 6.0);
        assert_relative_eq!(result[[1, 0]], 12.0);
        assert_relative_eq!(result[[1, 1]], 20.0);
    }
}
