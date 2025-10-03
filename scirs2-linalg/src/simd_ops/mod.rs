//! SIMD accelerated linear algebra operations
//!
//! This module provides SIMD-accelerated implementations of common linear algebra
//! operations for improved performance on modern CPUs. All SIMD operations
//! are delegated to scirs2-core::simd_ops for unified optimization management.

pub mod elementwise;
pub mod gemm;
pub mod hardware_specific;
pub mod neural;
pub mod norms;
pub mod simd_memory_ops;
pub mod transpose;

// Re-export commonly used SIMD operations
#[cfg(feature = "simd")]
pub use elementwise::{
    simdmatrix_add_f32, simdmatrix_add_f64, simdmatrix_add_inplace_f32,
    simdmatrix_mul_elementwise_f32, simdmatrix_scale_f32,
};
#[cfg(feature = "simd")]
pub use gemm::{
    simd_gemm_f32, simd_gemm_f64, simd_gemv_f32, simd_gemv_f64, simd_matmul_optimized_f32,
    simd_matmul_optimized_f64, GemmBlockSizes,
};
#[cfg(feature = "simd")]
pub use hardware_specific::{
    hardware_optimized_dot, hardware_optimized_matvec, HardwareCapabilities,
};
#[cfg(feature = "simd")]
pub use norms::{
    simd_frobenius_norm_f32, simd_frobenius_norm_f64, simd_vector_norm_f32, simd_vector_norm_f64,
};
#[cfg(feature = "simd")]
pub use transpose::{simd_transpose_f32, simd_transpose_f64};

#[cfg(feature = "simd")]
use crate::{LinalgError, LinalgResult};
#[cfg(feature = "simd")]
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
#[cfg(feature = "simd")]
use scirs2_core::simd_ops::{AutoOptimizer, PlatformCapabilities, SimdUnifiedOps};

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
#[allow(dead_code)]
pub fn simd_matvec_f32(
    matrix: &ArrayView2<f32>,
    vector: &ArrayView1<f32>,
) -> LinalgResult<Array1<f32>> {
    let (nrows, ncols) = matrix.dim();

    if ncols != vector.len() {
        let vector_len = vector.len();
        return Err(LinalgError::ShapeError(format!(
            "Matrix columns ({ncols}) must match vector length ({vector_len})"
        )));
    }

    let mut result = Array1::zeros(nrows);

    // Use the unified SIMD operations
    f32::simd_gemv(matrix, vector, 0.0, &mut result);

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
#[allow(dead_code)]
pub fn simd_matvec_f64(
    matrix: &ArrayView2<f64>,
    vector: &ArrayView1<f64>,
) -> LinalgResult<Array1<f64>> {
    let (nrows, ncols) = matrix.dim();

    if ncols != vector.len() {
        let vector_len = vector.len();
        return Err(LinalgError::ShapeError(format!(
            "Matrix columns ({ncols}) must match vector length ({vector_len})"
        )));
    }

    let mut result = Array1::zeros(nrows);

    // Use the unified SIMD operations
    f64::simd_gemv(matrix, vector, 0.0, &mut result);

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
#[allow(dead_code)]
pub fn simd_matmul_f32(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> LinalgResult<Array2<f32>> {
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions mismatch: a({m}, {k1}) * b({k2}, {n})"
        )));
    }

    let mut result = Array2::zeros((m, n));

    // Use the unified SIMD operations
    f32::simd_gemm(1.0, a, b, 0.0, &mut result);

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
#[allow(dead_code)]
pub fn simd_matmul_f64(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions mismatch: a({m}, {k1}) * b({k2}, {n})"
        )));
    }

    let mut result = Array2::zeros((m, n));

    // Use the unified SIMD operations
    f64::simd_gemm(1.0, a, b, 0.0, &mut result);

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
#[allow(dead_code)]
pub fn simdmatrix_max_f32(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> LinalgResult<Array2<f32>> {
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
        let a_row = a.row(i);
        let b_row = b.row(i);

        // Use the unified SIMD operations for the row
        let max_row = f32::simd_max(&a_row, &b_row);

        for (j, &val) in max_row.iter().enumerate() {
            result[[i, j]] = val;
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
#[allow(dead_code)]
pub fn simdmatrix_max_f64(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
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
        let a_row = a.row(i);
        let b_row = b.row(i);

        // Use the unified SIMD operations for the row
        let max_row = f64::simd_max(&a_row, &b_row);

        for (j, &val) in max_row.iter().enumerate() {
            result[[i, j]] = val;
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
#[allow(dead_code)]
pub fn simdmatrix_min_f32(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> LinalgResult<Array2<f32>> {
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
        let a_row = a.row(i);
        let b_row = b.row(i);

        // Use the unified SIMD operations for the row
        let min_row = f32::simd_min(&a_row, &b_row);

        for (j, &val) in min_row.iter().enumerate() {
            result[[i, j]] = val;
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
#[allow(dead_code)]
pub fn simdmatrix_min_f64(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
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
        let a_row = a.row(i);
        let b_row = b.row(i);

        // Use the unified SIMD operations for the row
        let min_row = f64::simd_min(&a_row, &b_row);

        for (j, &val) in min_row.iter().enumerate() {
            result[[i, j]] = val;
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
#[allow(dead_code)]
pub fn simd_axpy_f32(alpha: f32, x: &ArrayView1<f32>, y: &mut Array1<f32>) -> LinalgResult<()> {
    if x.len() != y.len() {
        return Err(LinalgError::ShapeError(format!(
            "Vector dimensions must match: {} vs {}",
            x.len(),
            y.len()
        )));
    }

    // Compute _alpha * x
    let scaled_x = f32::simd_scalar_mul(x, alpha);

    // Add to y in-place
    let y_view = y.view();
    let sum = f32::simd_add(&scaled_x.view(), &y_view);
    y.assign(&sum);

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
#[allow(dead_code)]
pub fn simd_axpy_f64(alpha: f64, x: &ArrayView1<f64>, y: &mut Array1<f64>) -> LinalgResult<()> {
    if x.len() != y.len() {
        return Err(LinalgError::ShapeError(format!(
            "Vector dimensions must match: {} vs {}",
            x.len(),
            y.len()
        )));
    }

    // Compute _alpha * x
    let scaled_x = f64::simd_scalar_mul(x, alpha);

    // Add to y in-place
    let y_view = y.view();
    let sum = f64::simd_add(&scaled_x.view(), &y_view);
    y.assign(&sum);

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
#[allow(dead_code)]
pub fn simd_dot_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> LinalgResult<f32> {
    if a.len() != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Vector dimensions must match for dot product: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    Ok(f32::simd_dot(a, b))
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
#[allow(dead_code)]
pub fn simd_dot_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> LinalgResult<f64> {
    if a.len() != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Vector dimensions must match for dot product: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    Ok(f64::simd_dot(a, b))
}

/// Get platform capabilities for SIMD optimization
#[allow(dead_code)]
pub fn get_platform_capabilities() -> PlatformCapabilities {
    PlatformCapabilities::detect()
}

/// Create an auto-optimizer for automatic SIMD/GPU selection
#[allow(dead_code)]
pub fn create_auto_optimizer() -> AutoOptimizer {
    AutoOptimizer::new()
}
