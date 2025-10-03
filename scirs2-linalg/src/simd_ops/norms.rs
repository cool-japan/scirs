//! SIMD-accelerated norm computations
//!
//! This module provides SIMD-accelerated implementations of various matrix
//! and vector norms for improved performance over scalar implementations.
//! All SIMD operations are delegated to scirs2-core::simd_ops for unified optimization.

#[cfg(feature = "simd")]
use scirs2_core::ndarray::{ArrayView1, ArrayView2};
#[cfg(feature = "simd")]
use scirs2_core::simd_ops::SimdUnifiedOps;

/// SIMD-accelerated Frobenius norm for f32 matrices
///
/// Computes the Frobenius norm (sqrt of sum of squares of all elements)
/// using SIMD instructions for improved performance.
///
/// # Arguments
///
/// * `matrix` - Input matrix
///
/// # Returns
///
/// * Frobenius norm of the matrix
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_frobenius_norm_f32(matrix: &ArrayView2<f32>) -> f32 {
    let mut sum_sq = 0.0f32;

    // Process row by row using unified SIMD operations
    for row in matrix.rows() {
        // Compute dot product of row with itself (sum of squares)
        sum_sq += f32::simd_dot(&row, &row);
    }

    sum_sq.sqrt()
}

/// SIMD-accelerated Frobenius norm for f64 matrices
///
/// Computes the Frobenius norm (sqrt of sum of squares of all elements)
/// using SIMD instructions for improved performance.
///
/// # Arguments
///
/// * `matrix` - Input matrix
///
/// # Returns
///
/// * Frobenius norm of the matrix
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_frobenius_norm_f64(matrix: &ArrayView2<f64>) -> f64 {
    let mut sum_sq = 0.0f64;

    // Process row by row using unified SIMD operations
    for row in matrix.rows() {
        // Compute dot product of row with itself (sum of squares)
        sum_sq += f64::simd_dot(&row, &row);
    }

    sum_sq.sqrt()
}

/// SIMD-accelerated vector 2-norm (Euclidean norm) for f32
///
/// # Arguments
///
/// * `vector` - Input vector
///
/// # Returns
///
/// * Euclidean norm of the vector
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_vector_norm_f32(vector: &ArrayView1<f32>) -> f32 {
    // Use unified SIMD norm operation
    f32::simd_norm(vector)
}

/// SIMD-accelerated vector 2-norm (Euclidean norm) for f64
///
/// # Arguments
///
/// * `vector` - Input vector
///
/// # Returns
///
/// * Euclidean norm of the vector
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_vector_norm_f64(vector: &ArrayView1<f64>) -> f64 {
    // Use unified SIMD norm operation
    f64::simd_norm(vector)
}

/// SIMD-accelerated vector 1-norm (Manhattan norm) for f32
///
/// # Arguments
///
/// * `vector` - Input vector
///
/// # Returns
///
/// * 1-norm (sum of absolute values) of the vector
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_vector_norm1_f32(vector: &ArrayView1<f32>) -> f32 {
    // Compute absolute values then sum
    let abs_vec = f32::simd_abs(vector);
    f32::simd_sum(&abs_vec.view())
}

/// SIMD-accelerated vector 1-norm (Manhattan norm) for f64
///
/// # Arguments
///
/// * `vector` - Input vector
///
/// # Returns
///
/// * 1-norm (sum of absolute values) of the vector
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_vector_norm1_f64(vector: &ArrayView1<f64>) -> f64 {
    // Compute absolute values then sum
    let abs_vec = f64::simd_abs(vector);
    f64::simd_sum(&abs_vec.view())
}

/// SIMD-accelerated vector infinity norm (maximum absolute value) for f32
///
/// # Arguments
///
/// * `vector` - Input vector
///
/// # Returns
///
/// * Infinity norm (maximum absolute value) of the vector
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_vector_norm_inf_f32(vector: &ArrayView1<f32>) -> f32 {
    // Compute absolute values then find maximum
    let abs_vec = f32::simd_abs(vector);
    f32::simd_max_element(&abs_vec.view())
}

/// SIMD-accelerated vector infinity norm (maximum absolute value) for f64
///
/// # Arguments
///
/// * `vector` - Input vector
///
/// # Returns
///
/// * Infinity norm (maximum absolute value) of the vector
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_vector_norm_inf_f64(vector: &ArrayView1<f64>) -> f64 {
    // Compute absolute values then find maximum
    let abs_vec = f64::simd_abs(vector);
    f64::simd_max_element(&abs_vec.view())
}

/// SIMD-accelerated matrix 1-norm (maximum column sum) for f32
///
/// # Arguments
///
/// * `matrix` - Input matrix
///
/// # Returns
///
/// * 1-norm of the matrix
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simdmatrix_norm1_f32(matrix: &ArrayView2<f32>) -> f32 {
    let mut max_col_sum = 0.0f32;

    for j in 0..matrix.ncols() {
        let col = matrix.column(j);
        let col_sum = simd_vector_norm1_f32(&col);
        max_col_sum = max_col_sum.max(col_sum);
    }

    max_col_sum
}

/// SIMD-accelerated matrix 1-norm (maximum column sum) for f64
///
/// # Arguments
///
/// * `matrix` - Input matrix
///
/// # Returns
///
/// * 1-norm of the matrix
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simdmatrix_norm1_f64(matrix: &ArrayView2<f64>) -> f64 {
    let mut max_col_sum = 0.0f64;

    for j in 0..matrix.ncols() {
        let col = matrix.column(j);
        let col_sum = simd_vector_norm1_f64(&col);
        max_col_sum = max_col_sum.max(col_sum);
    }

    max_col_sum
}

/// SIMD-accelerated matrix infinity norm (maximum row sum) for f32
///
/// # Arguments
///
/// * `matrix` - Input matrix
///
/// # Returns
///
/// * Infinity norm of the matrix
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simdmatrix_norm_inf_f32(matrix: &ArrayView2<f32>) -> f32 {
    let mut max_row_sum = 0.0f32;

    for i in 0..matrix.nrows() {
        let row = matrix.row(i);
        let row_sum = simd_vector_norm1_f32(&row);
        max_row_sum = max_row_sum.max(row_sum);
    }

    max_row_sum
}

/// SIMD-accelerated matrix infinity norm (maximum row sum) for f64
///
/// # Arguments
///
/// * `matrix` - Input matrix
///
/// # Returns
///
/// * Infinity norm of the matrix
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simdmatrix_norm_inf_f64(matrix: &ArrayView2<f64>) -> f64 {
    let mut max_row_sum = 0.0f64;

    for i in 0..matrix.nrows() {
        let row = matrix.row(i);
        let row_sum = simd_vector_norm1_f64(&row);
        max_row_sum = max_row_sum.max(row_sum);
    }

    max_row_sum
}

// Note: Helper functions have been removed as we now use unified SIMD operations from scirs2-core

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::{array, Array1};

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simd_frobenius_norm_f32() {
        let matrix = array![[3.0f32, 4.0, 0.0], [0.0, 0.0, 12.0], [5.0, 0.0, 0.0]];

        let result = simd_frobenius_norm_f32(&matrix.view());

        // Expected: sqrt(3^2 + 4^2 + 12^2 + 5^2) = sqrt(9 + 16 + 144 + 25) = sqrt(194)
        let expected = (9.0 + 16.0 + 144.0 + 25.0f32).sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simd_vector_norm_f32() {
        let vector = array![3.0f32, 4.0, 0.0, 12.0, 5.0];

        let result = simd_vector_norm_f32(&vector.view());

        // Expected: sqrt(3^2 + 4^2 + 0^2 + 12^2 + 5^2) = sqrt(9 + 16 + 0 + 144 + 25) = sqrt(194)
        let expected = (9.0 + 16.0 + 0.0 + 144.0 + 25.0f32).sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simd_vector_norm1_f32() {
        let vector = array![3.0f32, -4.0, 0.0, 12.0, -5.0];

        let result = simd_vector_norm1_f32(&vector.view());

        // Expected: |3| + |-4| + |0| + |12| + |-5| = 3 + 4 + 0 + 12 + 5 = 24
        let expected = 24.0f32;

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simd_vector_norm_inf_f32() {
        let vector = array![3.0f32, -4.0, 0.0, 12.0, -5.0];

        let result = simd_vector_norm_inf_f32(&vector.view());

        // Expected: max(|3|, |-4|, |0|, |12|, |-5|) = max(3, 4, 0, 12, 5) = 12
        let expected = 12.0f32;

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simdmatrix_norm1_f32() {
        let matrix = array![[1.0f32, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0]];

        let result = simdmatrix_norm1_f32(&matrix.view());

        // Column sums: |1| + |-4| + |7| = 12, |-2| + |5| + |-8| = 15, |3| + |-6| + |9| = 18
        // Maximum column sum: max(12, 15, 18) = 18
        let expected = 18.0f32;

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simdmatrix_norm_inf_f32() {
        let matrix = array![[1.0f32, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0]];

        let result = simdmatrix_norm_inf_f32(&matrix.view());

        // Row sums: |1| + |-2| + |3| = 6, |-4| + |5| + |-6| = 15, |7| + |-8| + |9| = 24
        // Maximum row sum: max(6, 15, 24) = 24
        let expected = 24.0f32;

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simd_frobenius_norm_f64() {
        let matrix = array![[3.0f64, 4.0], [0.0, 12.0]];

        let result = simd_frobenius_norm_f64(&matrix.view());

        // Expected: sqrt(3^2 + 4^2 + 0^2 + 12^2) = sqrt(9 + 16 + 0 + 144) = sqrt(169) = 13
        let expected = 13.0f64;

        assert_relative_eq!(result, expected, epsilon = 1e-12);
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simd_large_vector() {
        // Test with larger vector to exercise SIMD processing
        let size = 100;
        let vector: Array1<f32> = Array1::from_shape_fn(size, |i| (i as f32) * 0.1);

        let result = simd_vector_norm_f32(&vector.view());

        // Compute expected result
        let expected = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }
}
