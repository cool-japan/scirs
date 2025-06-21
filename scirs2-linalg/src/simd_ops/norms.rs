//! SIMD-accelerated norm computations
//!
//! This module provides SIMD-accelerated implementations of various matrix
//! and vector norms for improved performance over scalar implementations.

// LinalgError and LinalgResult imports removed as they are not used in this module
#[cfg(feature = "simd")]
use ndarray::{ArrayView1, ArrayView2};
#[cfg(feature = "simd")]
use wide::{f32x8, f64x4};

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
pub fn simd_frobenius_norm_f32(matrix: &ArrayView2<f32>) -> f32 {
    let _n = matrix.len();

    if let Some(flat_data) = matrix.as_slice() {
        // Process with SIMD when data is contiguous
        simd_frobenius_norm_flat_f32(flat_data)
    } else {
        // Fallback for non-contiguous data using row-wise processing
        let mut sum_sq = 0.0f32;

        for row in matrix.rows() {
            if let Some(row_slice) = row.as_slice() {
                sum_sq += simd_vector_norm_squared_f32(row_slice);
            } else {
                // Fallback to scalar computation
                for &val in row.iter() {
                    sum_sq += val * val;
                }
            }
        }

        sum_sq.sqrt()
    }
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
pub fn simd_frobenius_norm_f64(matrix: &ArrayView2<f64>) -> f64 {
    let _n = matrix.len();

    if let Some(flat_data) = matrix.as_slice() {
        // Process with SIMD when data is contiguous
        simd_frobenius_norm_flat_f64(flat_data)
    } else {
        // Fallback for non-contiguous data using row-wise processing
        let mut sum_sq = 0.0f64;

        for row in matrix.rows() {
            if let Some(row_slice) = row.as_slice() {
                sum_sq += simd_vector_norm_squared_f64(row_slice);
            } else {
                // Fallback to scalar computation
                for &val in row.iter() {
                    sum_sq += val * val;
                }
            }
        }

        sum_sq.sqrt()
    }
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
pub fn simd_vector_norm_f32(vector: &ArrayView1<f32>) -> f32 {
    if let Some(data) = vector.as_slice() {
        simd_vector_norm_squared_f32(data).sqrt()
    } else {
        // Fallback for non-contiguous data
        let mut sum_sq = 0.0f32;
        for &val in vector.iter() {
            sum_sq += val * val;
        }
        sum_sq.sqrt()
    }
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
pub fn simd_vector_norm_f64(vector: &ArrayView1<f64>) -> f64 {
    if let Some(data) = vector.as_slice() {
        simd_vector_norm_squared_f64(data).sqrt()
    } else {
        // Fallback for non-contiguous data
        let mut sum_sq = 0.0f64;
        for &val in vector.iter() {
            sum_sq += val * val;
        }
        sum_sq.sqrt()
    }
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
pub fn simd_vector_norm1_f32(vector: &ArrayView1<f32>) -> f32 {
    if let Some(data) = vector.as_slice() {
        simd_vector_norm1_flat_f32(data)
    } else {
        // Fallback for non-contiguous data
        vector.iter().map(|&x| x.abs()).sum()
    }
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
pub fn simd_vector_norm1_f64(vector: &ArrayView1<f64>) -> f64 {
    if let Some(data) = vector.as_slice() {
        simd_vector_norm1_flat_f64(data)
    } else {
        // Fallback for non-contiguous data
        vector.iter().map(|&x| x.abs()).sum()
    }
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
pub fn simd_vector_norm_inf_f32(vector: &ArrayView1<f32>) -> f32 {
    if let Some(data) = vector.as_slice() {
        simd_vector_norm_inf_flat_f32(data)
    } else {
        // Fallback for non-contiguous data
        vector.iter().map(|&x| x.abs()).fold(0.0f32, f32::max)
    }
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
pub fn simd_vector_norm_inf_f64(vector: &ArrayView1<f64>) -> f64 {
    if let Some(data) = vector.as_slice() {
        simd_vector_norm_inf_flat_f64(data)
    } else {
        // Fallback for non-contiguous data
        vector.iter().map(|&x| x.abs()).fold(0.0f64, f64::max)
    }
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
pub fn simd_matrix_norm1_f32(matrix: &ArrayView2<f32>) -> f32 {
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
pub fn simd_matrix_norm1_f64(matrix: &ArrayView2<f64>) -> f64 {
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
pub fn simd_matrix_norm_inf_f32(matrix: &ArrayView2<f32>) -> f32 {
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
pub fn simd_matrix_norm_inf_f64(matrix: &ArrayView2<f64>) -> f64 {
    let mut max_row_sum = 0.0f64;

    for i in 0..matrix.nrows() {
        let row = matrix.row(i);
        let row_sum = simd_vector_norm1_f64(&row);
        max_row_sum = max_row_sum.max(row_sum);
    }

    max_row_sum
}

// Helper functions for flat array processing

/// SIMD computation of sum of squares for flat f32 array
#[cfg(feature = "simd")]
fn simd_frobenius_norm_flat_f32(data: &[f32]) -> f32 {
    simd_vector_norm_squared_f32(data).sqrt()
}

/// SIMD computation of sum of squares for flat f64 array
#[cfg(feature = "simd")]
fn simd_frobenius_norm_flat_f64(data: &[f64]) -> f64 {
    simd_vector_norm_squared_f64(data).sqrt()
}

/// SIMD computation of sum of squares for f32 vector
#[cfg(feature = "simd")]
fn simd_vector_norm_squared_f32(data: &[f32]) -> f32 {
    let n = data.len();
    let mut i = 0;
    let chunk_size = 8;
    let mut sum_vec = f32x8::splat(0.0);

    // Process 8 elements at a time
    while i + chunk_size <= n {
        let chunk = [
            data[i],
            data[i + 1],
            data[i + 2],
            data[i + 3],
            data[i + 4],
            data[i + 5],
            data[i + 6],
            data[i + 7],
        ];
        let vec = f32x8::new(chunk);
        sum_vec += vec * vec;
        i += chunk_size;
    }

    // Extract sum from SIMD vector
    let sum_arr: [f32; 8] = sum_vec.into();
    let mut sum = sum_arr.iter().sum::<f32>();

    // Process remaining elements
    for &val in data.iter().skip(i) {
        sum += val * val;
    }

    sum
}

/// SIMD computation of sum of squares for f64 vector
#[cfg(feature = "simd")]
fn simd_vector_norm_squared_f64(data: &[f64]) -> f64 {
    let n = data.len();
    let mut i = 0;
    let chunk_size = 4;
    let mut sum_vec = f64x4::splat(0.0);

    // Process 4 elements at a time
    while i + chunk_size <= n {
        let chunk = [data[i], data[i + 1], data[i + 2], data[i + 3]];
        let vec = f64x4::new(chunk);
        sum_vec += vec * vec;
        i += chunk_size;
    }

    // Extract sum from SIMD vector
    let sum_arr: [f64; 4] = sum_vec.into();
    let mut sum = sum_arr.iter().sum::<f64>();

    // Process remaining elements
    for &val in data.iter().skip(i) {
        sum += val * val;
    }

    sum
}

/// SIMD computation of 1-norm for f32 vector
#[cfg(feature = "simd")]
fn simd_vector_norm1_flat_f32(data: &[f32]) -> f32 {
    let n = data.len();
    let mut i = 0;
    let chunk_size = 8;
    let mut sum_vec = f32x8::splat(0.0);

    // Process 8 elements at a time
    while i + chunk_size <= n {
        let chunk = [
            data[i],
            data[i + 1],
            data[i + 2],
            data[i + 3],
            data[i + 4],
            data[i + 5],
            data[i + 6],
            data[i + 7],
        ];
        // Compute absolute values (simplified - would use actual SIMD abs in production)
        let abs_vec = f32x8::new([
            chunk[0].abs(),
            chunk[1].abs(),
            chunk[2].abs(),
            chunk[3].abs(),
            chunk[4].abs(),
            chunk[5].abs(),
            chunk[6].abs(),
            chunk[7].abs(),
        ]);
        sum_vec += abs_vec;
        i += chunk_size;
    }

    // Extract sum from SIMD vector
    let sum_arr: [f32; 8] = sum_vec.into();
    let mut sum = sum_arr.iter().sum::<f32>();

    // Process remaining elements
    for &val in data.iter().skip(i) {
        sum += val.abs();
    }

    sum
}

/// SIMD computation of 1-norm for f64 vector
#[cfg(feature = "simd")]
fn simd_vector_norm1_flat_f64(data: &[f64]) -> f64 {
    let n = data.len();
    let mut i = 0;
    let chunk_size = 4;
    let mut sum_vec = f64x4::splat(0.0);

    // Process 4 elements at a time
    while i + chunk_size <= n {
        let chunk = [data[i], data[i + 1], data[i + 2], data[i + 3]];
        // Compute absolute values (simplified - would use actual SIMD abs in production)
        let abs_vec = f64x4::new([
            chunk[0].abs(),
            chunk[1].abs(),
            chunk[2].abs(),
            chunk[3].abs(),
        ]);
        sum_vec += abs_vec;
        i += chunk_size;
    }

    // Extract sum from SIMD vector
    let sum_arr: [f64; 4] = sum_vec.into();
    let mut sum = sum_arr.iter().sum::<f64>();

    // Process remaining elements
    for &val in data.iter().skip(i) {
        sum += val.abs();
    }

    sum
}

/// SIMD computation of infinity norm for f32 vector
#[cfg(feature = "simd")]
fn simd_vector_norm_inf_flat_f32(data: &[f32]) -> f32 {
    let n = data.len();
    let mut i = 0;
    let chunk_size = 8;
    let mut max_vec = f32x8::splat(0.0);

    // Process 8 elements at a time
    while i + chunk_size <= n {
        let chunk = [
            data[i],
            data[i + 1],
            data[i + 2],
            data[i + 3],
            data[i + 4],
            data[i + 5],
            data[i + 6],
            data[i + 7],
        ];
        // Compute absolute values (simplified - would use actual SIMD abs in production)
        let abs_vec = f32x8::new([
            chunk[0].abs(),
            chunk[1].abs(),
            chunk[2].abs(),
            chunk[3].abs(),
            chunk[4].abs(),
            chunk[5].abs(),
            chunk[6].abs(),
            chunk[7].abs(),
        ]);

        // Element-wise maximum (simplified - would use SIMD max)
        let max_arr: [f32; 8] = max_vec.into();
        let abs_arr: [f32; 8] = abs_vec.into();
        let new_max = [
            max_arr[0].max(abs_arr[0]),
            max_arr[1].max(abs_arr[1]),
            max_arr[2].max(abs_arr[2]),
            max_arr[3].max(abs_arr[3]),
            max_arr[4].max(abs_arr[4]),
            max_arr[5].max(abs_arr[5]),
            max_arr[6].max(abs_arr[6]),
            max_arr[7].max(abs_arr[7]),
        ];
        max_vec = f32x8::new(new_max);
        i += chunk_size;
    }

    // Extract maximum from SIMD vector
    let max_arr: [f32; 8] = max_vec.into();
    let mut max_val = max_arr.iter().fold(0.0f32, |a, &b| a.max(b));

    // Process remaining elements
    for &val in data.iter().skip(i) {
        max_val = max_val.max(val.abs());
    }

    max_val
}

/// SIMD computation of infinity norm for f64 vector
#[cfg(feature = "simd")]
fn simd_vector_norm_inf_flat_f64(data: &[f64]) -> f64 {
    let n = data.len();
    let mut i = 0;
    let chunk_size = 4;
    let mut max_vec = f64x4::splat(0.0);

    // Process 4 elements at a time
    while i + chunk_size <= n {
        let chunk = [data[i], data[i + 1], data[i + 2], data[i + 3]];
        // Compute absolute values (simplified - would use actual SIMD abs in production)
        let abs_vec = f64x4::new([
            chunk[0].abs(),
            chunk[1].abs(),
            chunk[2].abs(),
            chunk[3].abs(),
        ]);

        // Element-wise maximum (simplified - would use SIMD max)
        let max_arr: [f64; 4] = max_vec.into();
        let abs_arr: [f64; 4] = abs_vec.into();
        let new_max = [
            max_arr[0].max(abs_arr[0]),
            max_arr[1].max(abs_arr[1]),
            max_arr[2].max(abs_arr[2]),
            max_arr[3].max(abs_arr[3]),
        ];
        max_vec = f64x4::new(new_max);
        i += chunk_size;
    }

    // Extract maximum from SIMD vector
    let max_arr: [f64; 4] = max_vec.into();
    let mut max_val = max_arr.iter().fold(0.0f64, |a, &b| a.max(b));

    // Process remaining elements
    for &val in data.iter().skip(i) {
        max_val = max_val.max(val.abs());
    }

    max_val
}

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{array, Array1};

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_frobenius_norm_f32() {
        let matrix = array![[3.0f32, 4.0, 0.0], [0.0, 0.0, 12.0], [5.0, 0.0, 0.0]];

        let result = simd_frobenius_norm_f32(&matrix.view());

        // Expected: sqrt(3^2 + 4^2 + 12^2 + 5^2) = sqrt(9 + 16 + 144 + 25) = sqrt(194)
        let expected = (9.0 + 16.0 + 144.0 + 25.0f32).sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_vector_norm_f32() {
        let vector = array![3.0f32, 4.0, 0.0, 12.0, 5.0];

        let result = simd_vector_norm_f32(&vector.view());

        // Expected: sqrt(3^2 + 4^2 + 0^2 + 12^2 + 5^2) = sqrt(9 + 16 + 0 + 144 + 25) = sqrt(194)
        let expected = (9.0 + 16.0 + 0.0 + 144.0 + 25.0f32).sqrt();

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_vector_norm1_f32() {
        let vector = array![3.0f32, -4.0, 0.0, 12.0, -5.0];

        let result = simd_vector_norm1_f32(&vector.view());

        // Expected: |3| + |-4| + |0| + |12| + |-5| = 3 + 4 + 0 + 12 + 5 = 24
        let expected = 24.0f32;

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_vector_norm_inf_f32() {
        let vector = array![3.0f32, -4.0, 0.0, 12.0, -5.0];

        let result = simd_vector_norm_inf_f32(&vector.view());

        // Expected: max(|3|, |-4|, |0|, |12|, |-5|) = max(3, 4, 0, 12, 5) = 12
        let expected = 12.0f32;

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_matrix_norm1_f32() {
        let matrix = array![[1.0f32, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0]];

        let result = simd_matrix_norm1_f32(&matrix.view());

        // Column sums: |1| + |-4| + |7| = 12, |-2| + |5| + |-8| = 15, |3| + |-6| + |9| = 18
        // Maximum column sum: max(12, 15, 18) = 18
        let expected = 18.0f32;

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_matrix_norm_inf_f32() {
        let matrix = array![[1.0f32, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0]];

        let result = simd_matrix_norm_inf_f32(&matrix.view());

        // Row sums: |1| + |-2| + |3| = 6, |-4| + |5| + |-6| = 15, |7| + |-8| + |9| = 24
        // Maximum row sum: max(6, 15, 24) = 24
        let expected = 24.0f32;

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_frobenius_norm_f64() {
        let matrix = array![[3.0f64, 4.0], [0.0, 12.0]];

        let result = simd_frobenius_norm_f64(&matrix.view());

        // Expected: sqrt(3^2 + 4^2 + 0^2 + 12^2) = sqrt(9 + 16 + 0 + 144) = sqrt(169) = 13
        let expected = 13.0f64;

        assert_relative_eq!(result, expected, epsilon = 1e-12);
    }

    #[test]
    #[cfg(feature = "simd")]
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
