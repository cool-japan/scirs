//! Mixed-precision linear algebra operations
//!
//! This module provides functionality for using different precisions
//! within the same computation to balance accuracy and performance.
//!
//! Mixed precision techniques offer the following benefits:
//! - Improved performance by using lower precision for bulk calculations
//! - Maintained accuracy by using higher precision for critical computations
//! - Reduced memory usage while preserving numerical stability
//! - Hardware acceleration utilization (e.g., tensor cores in modern GPUs)
//!
//! The implementation includes optimized versions with:
//! - Block-based algorithms for improved cache locality
//! - Parallel processing for large matrices when the "parallel" feature is enabled
//! - SIMD acceleration for specific data types when the "simd" feature is enabled
//! - Iterative refinement techniques for improved accuracy
//!
//! # Module Organization
//!
//! - [`conversions`] - Type conversion utilities between different precisions
//! - [`f32_ops`] - F32-optimized operations for basic computations
//! - [`f64_ops`] - F64-optimized operations with advanced algorithms
//! - [`adaptive`] - Adaptive algorithms with iterative refinement and decompositions

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, NumAssign, NumCast, ToPrimitive, Zero};
use std::fmt::Debug;

use crate::error::{LinalgError, LinalgResult};

// Re-export SIMD-accelerated functions when simd feature is enabled
#[cfg(feature = "simd")]
mod simd;

#[cfg(feature = "simd")]
pub use simd::{
    simd_mixed_precision_dot_f32_f64, simd_mixed_precision_matmul_f32_f64,
    simd_mixed_precision_matvec_f32_f64,
};

// Submodules
pub mod adaptive;
pub mod conversions;
pub mod f32_ops;
pub mod f64_ops;

// Re-export conversion functions for backward compatibility
pub use conversions::{convert, convert_2d};

// Re-export f32-optimized operations
pub use f32_ops::{
    mixed_precision_dot_f32, mixed_precision_matmul_f32_basic, mixed_precision_matvec_f32,
};

// Re-export f64-optimized operations
pub use f64_ops::mixed_precision_matmul_f64;

// Re-export adaptive algorithms
pub use adaptive::{
    iterative_refinement_solve, mixed_precision_cond, mixed_precision_qr, mixed_precision_solve,
    mixed_precision_svd,
};

/// Perform mixed-precision matrix-vector multiplication
///
/// Performs the computation in higher precision and returns the result in the desired precision.
/// Automatically selects the optimal implementation based on input precision.
///
/// # Arguments
/// * `a` - Matrix in precision A
/// * `x` - Vector in precision B
///
/// # Returns
/// * Vector in precision C
///
/// # Type Parameters
/// * `A` - Input matrix precision
/// * `B` - Input vector precision
/// * `C` - Output vector precision
/// * `H` - High precision used for computation
///
/// # Examples
/// ```
/// use ndarray::{array, ArrayView1, ArrayView2};
/// use scirs2_linalg::mixed_precision::mixed_precision_matvec;
///
/// let a_f32 = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
/// let x_f32 = array![0.5f32, 0.5f32];
///
/// // Compute result with internal f64 precision
/// let y = mixed_precision_matvec::<f32, f32, f32, f64>(
///     &a_f32.view(),
///     &x_f32.view()
/// ).unwrap();
///
/// assert_eq!(y.len(), 2);
/// assert!((y[0] - 1.5f32).abs() < 1e-6);
/// assert!((y[1] - 3.5f32).abs() < 1e-6);
/// ```
pub fn mixed_precision_matvec<A, B, C, H>(
    a: &ArrayView2<A>,
    x: &ArrayView1<B>,
) -> LinalgResult<Array1<C>>
where
    A: Clone + Debug + ToPrimitive + Copy,
    B: Clone + Debug + ToPrimitive + Copy,
    C: Clone + Zero + NumCast + Debug,
    H: Float + Clone + NumCast + Debug + ToPrimitive,
{
    // Delegate to f32-optimized implementation for simplicity
    // In a more sophisticated implementation, we could choose based on precision types
    f32_ops::mixed_precision_matvec_f32::<A, B, C, H>(a, x)
}

/// Mixed-precision matrix multiplication
///
/// Performs the matrix-matrix multiplication using high precision
/// internally while inputs and outputs use specified precision.
/// Automatically selects between parallel and serial implementations.
///
/// # Arguments
/// * `a` - First matrix in precision A
/// * `b` - Second matrix in precision B
///
/// # Returns
/// * Matrix in precision C
///
/// # Type Parameters
/// * `A` - First input matrix precision
/// * `B` - Second input matrix precision
/// * `C` - Output matrix precision
/// * `H` - High precision used for computation
///
/// # Examples
/// ```
/// use ndarray::{array, ArrayView2};
/// use scirs2_linalg::mixed_precision::mixed_precision_matmul;
///
/// let a_f32 = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
/// let b_f32 = array![[5.0f32, 6.0f32], [7.0f32, 8.0f32]];
///
/// // Compute result with internal f64 precision
/// let c = mixed_precision_matmul::<f32, f32, f32, f64>(
///     &a_f32.view(),
///     &b_f32.view()
/// ).unwrap();
///
/// assert_eq!(c.shape(), &[2, 2]);
/// assert!((c[[0, 0]] - 19.0f32).abs() < 1e-5);
/// assert!((c[[0, 1]] - 22.0f32).abs() < 1e-5);
/// assert!((c[[1, 0]] - 43.0f32).abs() < 1e-5);
/// assert!((c[[1, 1]] - 50.0f32).abs() < 1e-5);
/// ```
pub fn mixed_precision_matmul<A, B, C, H>(
    a: &ArrayView2<A>,
    b: &ArrayView2<B>,
) -> LinalgResult<Array2<C>>
where
    A: Clone + Debug + ToPrimitive + Copy + Sync,
    B: Clone + Debug + ToPrimitive + Copy + Sync,
    C: Clone + Zero + NumCast + Debug + Send,
    H: Float + Clone + NumCast + Debug + ToPrimitive + NumAssign + Zero + Send + Sync,
{
    // Choose implementation based on matrix size and available features
    let a_shape = a.shape();
    let b_shape = b.shape();
    let total_elements = a_shape[0] * a_shape[1] + b_shape[0] * b_shape[1];

    // For smaller matrices, use f32-optimized basic implementation
    if total_elements < 10000 {
        f32_ops::mixed_precision_matmul_f32_basic::<A, B, C, H>(a, b)
    } else {
        // For larger matrices, use f64-optimized implementation
        f64_ops::mixed_precision_matmul_f64::<A, B, C, H>(a, b)
    }
}

/// Compute the dot product of two vectors using mixed precision
///
/// This function computes the dot product using higher precision
/// internally while inputs and outputs use specified precision. It
/// employs the Kahan summation algorithm for improved numerical
/// stability, especially important for vectors with large magnitude
/// differences between elements.
///
/// # Arguments
/// * `a` - First vector in precision A
/// * `b` - Second vector in precision B
///
/// # Returns
/// * Dot product result in precision C
///
/// # Type Parameters
/// * `A` - First input vector precision
/// * `B` - Second input vector precision
/// * `C` - Output scalar precision
/// * `H` - High precision used for computation
///
/// # Examples
/// ```
/// use ndarray::array;
/// use scirs2_linalg::mixed_precision::mixed_precision_dot;
///
/// // Create two vectors with elements of vastly different magnitudes
/// let a_f32 = array![1.0e-6f32, 2.0f32, 3.0e6f32];
/// let b_f32 = array![4.0e6f32, 5.0f32, 6.0e-6f32];
///
/// // Compute with mixed precision using f64 internally
/// let dot_mp = mixed_precision_dot::<f32, f32, f32, f64>(
///     &a_f32.view(),
///     &b_f32.view()
/// ).unwrap();
///
/// // The mixed precision version may be more accurate
/// ```
pub fn mixed_precision_dot<A, B, C, H>(a: &ArrayView1<A>, b: &ArrayView1<B>) -> LinalgResult<C>
where
    A: Clone + Debug + ToPrimitive + Copy,
    B: Clone + Debug + ToPrimitive + Copy,
    C: Clone + Zero + NumCast + Debug,
    H: Float + Clone + NumCast + Debug + ToPrimitive,
{
    // Delegate to f32-optimized implementation
    f32_ops::mixed_precision_dot_f32::<A, B, C, H>(a, b)
}

/// Compute matrix inverse using mixed precision
///
/// This function computes the matrix inverse using higher precision
/// internally for improved numerical stability.
///
/// # Arguments
/// * `a` - Input matrix in precision A
///
/// # Returns
/// * Inverse matrix in precision C
///
/// # Type Parameters
/// * `A` - Input matrix precision
/// * `C` - Output matrix precision
/// * `H` - High precision used for computation
pub fn mixed_precision_inv<A, C, H>(a: &ArrayView2<A>) -> LinalgResult<Array2<C>>
where
    A: Clone + Debug + ToPrimitive + Copy,
    C: Clone + Zero + NumCast + Debug,
    H: Float + Clone + NumCast + Debug + ToPrimitive + NumAssign + Zero,
{
    // Check if matrix is square
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square for inversion, got shape {:?}",
            shape
        )));
    }

    let n = shape[0];

    // Convert to high precision
    let a_high = conversions::convert_2d::<A, H>(a);

    // Create augmented matrix [A|I]
    let mut aug = Array2::<H>::zeros((n, 2 * n));

    // Fill in A
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a_high[[i, j]];
        }
    }

    // Fill in I (identity matrix)
    for i in 0..n {
        aug[[i, n + i]] = H::one();
    }

    // Gaussian elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = aug[[i, i]].abs();

        for j in i + 1..n {
            let val = aug[[j, i]].abs();
            if val > max_val {
                max_row = j;
                max_val = val;
            }
        }

        // Check for singular matrix
        if max_val < H::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular and cannot be inverted".to_string(),
            ));
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..2 * n {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Scale pivot row
        let pivot = aug[[i, i]];
        for j in 0..2 * n {
            aug[[i, j]] /= pivot;
        }

        // Eliminate
        for j in 0..n {
            if j != i {
                let factor = aug[[j, i]];
                for k in 0..2 * n {
                    aug[[j, k]] = aug[[j, k]] - factor * aug[[i, k]];
                }
            }
        }
    }

    // Extract inverse matrix
    let mut inv_high = Array2::<H>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv_high[[i, j]] = aug[[i, n + j]];
        }
    }

    // Convert back to desired output precision
    let inv_c = conversions::convert_2d::<H, C>(&inv_high.view());

    Ok(inv_c)
}

/// Compute matrix determinant using mixed precision
///
/// This function computes the matrix determinant using higher precision
/// internally for improved numerical stability.
///
/// # Arguments
/// * `a` - Input matrix in precision A
///
/// # Returns
/// * Determinant in precision C
///
/// # Type Parameters
/// * `A` - Input matrix precision
/// * `C` - Output scalar precision
/// * `H` - High precision used for computation
pub fn mixed_precision_det<A, C, H>(a: &ArrayView2<A>) -> LinalgResult<C>
where
    A: Clone + Debug + ToPrimitive + Copy,
    C: Clone + Zero + NumCast + Debug,
    H: Float + Clone + NumCast + Debug + ToPrimitive + NumAssign + Zero,
{
    // Check if matrix is square
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square for determinant, got shape {:?}",
            shape
        )));
    }

    let n = shape[0];

    // Convert to high precision
    let mut a_high = conversions::convert_2d::<A, H>(a);

    let mut det = H::one();

    // Gaussian elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = a_high[[i, i]].abs();

        for j in i + 1..n {
            let val = a_high[[j, i]].abs();
            if val > max_val {
                max_row = j;
                max_val = val;
            }
        }

        // If pivot is zero, determinant is zero
        if max_val < H::epsilon() {
            return Ok(C::zero());
        }

        // Swap rows if needed (this changes sign of determinant)
        if max_row != i {
            for j in 0..n {
                let temp = a_high[[i, j]];
                a_high[[i, j]] = a_high[[max_row, j]];
                a_high[[max_row, j]] = temp;
            }
            det = -det; // Row swap changes sign
        }

        // Update determinant with diagonal element
        det *= a_high[[i, i]];

        // Eliminate
        for j in i + 1..n {
            let factor = a_high[[j, i]] / a_high[[i, i]];
            for k in i + 1..n {
                a_high[[j, k]] = a_high[[j, k]] - factor * a_high[[i, k]];
            }
        }
    }

    // Convert back to desired output precision
    C::from(det).ok_or_else(|| {
        LinalgError::ComputationError("Failed to convert determinant to output type".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_mixed_precision_matvec() {
        let a = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
        let x = array![0.5f32, 0.5f32];

        let result = mixed_precision_matvec::<f32, f32, f32, f64>(&a.view(), &x.view()).unwrap();

        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], 1.5f32, epsilon = 1e-6);
        assert_relative_eq!(result[1], 3.5f32, epsilon = 1e-6);
    }

    #[test]
    fn test_mixed_precision_matmul() {
        let a = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
        let b = array![[5.0f32, 6.0f32], [7.0f32, 8.0f32]];

        let result = mixed_precision_matmul::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_relative_eq!(result[[0, 0]], 19.0f32, epsilon = 1e-5);
        assert_relative_eq!(result[[0, 1]], 22.0f32, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 0]], 43.0f32, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 1]], 50.0f32, epsilon = 1e-5);
    }

    #[test]
    fn test_mixed_precision_dot() {
        let a = array![1.0f32, 2.0f32, 3.0f32];
        let b = array![4.0f32, 5.0f32, 6.0f32];

        let result = mixed_precision_dot::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();

        // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_relative_eq!(result, 32.0f32, epsilon = 1e-6);
    }

    #[test]
    fn test_mixed_precision_inv() {
        let a = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];

        let result = mixed_precision_inv::<f32, f32, f64>(&a.view()).unwrap();

        assert_eq!(result.shape(), &[2, 2]);

        // Expected inverse: [[-2, 1], [1.5, -0.5]]
        assert_relative_eq!(result[[0, 0]], -2.0f32, epsilon = 1e-5);
        assert_relative_eq!(result[[0, 1]], 1.0f32, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 0]], 1.5f32, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 1]], -0.5f32, epsilon = 1e-5);

        // Verify A * A^(-1) = I
        let identity = a.dot(&result);
        assert_relative_eq!(identity[[0, 0]], 1.0f32, epsilon = 1e-4);
        assert_relative_eq!(identity[[0, 1]], 0.0f32, epsilon = 1e-4);
        assert_relative_eq!(identity[[1, 0]], 0.0f32, epsilon = 1e-4);
        assert_relative_eq!(identity[[1, 1]], 1.0f32, epsilon = 1e-4);
    }

    #[test]
    fn test_mixed_precision_det() {
        // 2x2 matrix
        let a = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
        let det = mixed_precision_det::<f32, f32, f64>(&a.view()).unwrap();
        // Expected: 1*4 - 2*3 = 4 - 6 = -2
        assert_relative_eq!(det, -2.0f32, epsilon = 1e-5);

        // Identity matrix
        let identity = array![[1.0f32, 0.0f32], [0.0f32, 1.0f32]];
        let det_id = mixed_precision_det::<f32, f32, f64>(&identity.view()).unwrap();
        assert_relative_eq!(det_id, 1.0f32, epsilon = 1e-5);

        // Singular matrix
        let singular = array![[1.0f32, 2.0f32], [2.0f32, 4.0f32]];
        let det_sing = mixed_precision_det::<f32, f32, f64>(&singular.view()).unwrap();
        assert_relative_eq!(det_sing, 0.0f32, epsilon = 1e-5);
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules work together
        let a = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
        let b = array![[5.0f32, 6.0f32], [7.0f32, 8.0f32]];
        let x = array![1.0f32, 2.0f32];

        // Test matrix multiplication
        let matmul_result =
            mixed_precision_matmul::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();
        assert_eq!(matmul_result.shape(), &[2, 2]);

        // Test matrix-vector multiplication
        let matvec_result =
            mixed_precision_matvec::<f32, f32, f32, f64>(&a.view(), &x.view()).unwrap();
        assert_eq!(matvec_result.len(), 2);

        // Test linear solve
        let solve_result =
            mixed_precision_solve::<f32, f32, f32, f64>(&a.view(), &x.view()).unwrap();
        assert_eq!(solve_result.len(), 2);

        // Test decompositions
        let (q, r) = mixed_precision_qr::<f32, f32, f64>(&a.view()).unwrap();
        assert_eq!(q.shape(), &[2, 2]);
        assert_eq!(r.shape(), &[2, 2]);
    }
}
