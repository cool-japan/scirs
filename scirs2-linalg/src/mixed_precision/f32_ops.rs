//! F32-precision optimized operations for mixed-precision linear algebra
//!
//! This module provides specialized implementations optimized for f32 precision,
//! focusing on basic matrix operations with efficient algorithms for smaller matrices.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, NumAssign, NumCast, ToPrimitive, Zero};
use std::fmt::Debug;

use super::conversions::{convert, convert_2d};
use crate::error::{LinalgError, LinalgResult};

/// Perform mixed-precision matrix-vector multiplication using f32 optimization
///
/// Performs the computation in higher precision and returns the result in the desired precision.
/// Optimized for f32 inputs with efficient algorithms for small to medium matrices.
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
/// use scirs2_linalg::mixed_precision::f32_ops::mixed_precision_matvec_f32;
///
/// let a_f32 = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
/// let x_f32 = array![0.5f32, 0.5f32];
///
/// // Compute result with internal f64 precision
/// let y = mixed_precision_matvec_f32::<f32, f32, f32, f64>(
///     &a_f32.view(),
///     &x_f32.view()
/// ).unwrap();
///
/// assert_eq!(y.len(), 2);
/// assert!((y[0] - 1.5f32).abs() < 1e-6);
/// assert!((y[1] - 3.5f32).abs() < 1e-6);
/// ```
pub fn mixed_precision_matvec_f32<A, B, C, H>(
    a: &ArrayView2<A>,
    x: &ArrayView1<B>,
) -> LinalgResult<Array1<C>>
where
    A: Clone + Debug + ToPrimitive + Copy,
    B: Clone + Debug + ToPrimitive + Copy,
    C: Clone + Zero + NumCast + Debug,
    H: Float + Clone + NumCast + Debug + ToPrimitive,
{
    // Check dimensions
    let a_shape = a.shape();
    if a_shape[1] != x.len() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix columns ({}) must match vector length ({})",
            a_shape[1],
            x.len()
        )));
    }

    // Convert to high precision
    let a_high = convert_2d::<A, H>(a);
    let x_high = convert::<B, H>(x);

    // Perform computation in high precision
    let mut result_high = Array1::<H>::zeros(a_shape[0]);

    for i in 0..a_shape[0] {
        let row = a_high.index_axis(Axis(0), i);
        let mut sum = H::zero();

        for j in 0..a_shape[1] {
            sum = sum + row[j] * x_high[j];
        }

        result_high[i] = sum;
    }

    // Convert back to desired output precision
    let mut result = Array1::<C>::zeros(a_shape[0]);
    for (i, &val) in result_high.iter().enumerate() {
        result[i] = C::from(val).unwrap_or_else(|| C::zero());
    }

    Ok(result)
}

/// Basic matrix multiplication optimized for f32 precision (single-threaded)
///
/// This implementation uses simple algorithms optimized for f32 precision,
/// with efficient blocked algorithms for better cache usage on larger matrices.
pub fn mixed_precision_matmul_f32_basic<A, B, C, H>(
    a: &ArrayView2<A>,
    b: &ArrayView2<B>,
) -> LinalgResult<Array2<C>>
where
    A: Clone + Debug + ToPrimitive + Copy,
    B: Clone + Debug + ToPrimitive + Copy,
    C: Clone + Zero + NumCast + Debug,
    H: Float + Clone + NumCast + Debug + ToPrimitive + NumAssign + Zero,
{
    // Check dimensions
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape[1] != b_shape[0] {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions incompatible for multiplication: {}x{} and {}x{}",
            a_shape[0], a_shape[1], b_shape[0], b_shape[1]
        )));
    }

    let m = a_shape[0];
    let n = b_shape[1];
    let k = a_shape[1];

    // Convert to high precision
    let a_high = convert_2d::<A, H>(a);
    let b_high = convert_2d::<B, H>(b);

    // For small matrices, use the naive algorithm
    if m <= 32 && n <= 32 && k <= 32 {
        // Perform matrix multiplication in high precision
        let mut c_high = Array2::<H>::zeros((m, n));

        for i in 0..m {
            for j in 0..n {
                let mut sum = H::zero();
                for l in 0..k {
                    sum += a_high[[i, l]] * b_high[[l, j]];
                }
                c_high[[i, j]] = sum;
            }
        }

        // Convert back to desired output precision
        let mut c = Array2::<C>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                c[[i, j]] = C::from(c_high[[i, j]]).unwrap_or_else(|| C::zero());
            }
        }

        return Ok(c);
    }

    // For larger matrices, use blocked algorithm for better cache usage
    const BLOCK_SIZE: usize = 32;

    let mut c_high = Array2::<H>::zeros((m, n));

    // Block sizes
    let block_m = m.div_ceil(BLOCK_SIZE);
    let block_n = n.div_ceil(BLOCK_SIZE);
    let block_k = k.div_ceil(BLOCK_SIZE);

    // Blocked matrix multiplication algorithm
    for bi in 0..block_m {
        let i_start = bi * BLOCK_SIZE;
        let i_end = std::cmp::min(i_start + BLOCK_SIZE, m);

        for bj in 0..block_n {
            let j_start = bj * BLOCK_SIZE;
            let j_end = std::cmp::min(j_start + BLOCK_SIZE, n);

            // Initialize the result block to zero
            for i in i_start..i_end {
                for j in j_start..j_end {
                    c_high[[i, j]] = H::zero();
                }
            }

            // Compute the block result
            for bk in 0..block_k {
                let k_start = bk * BLOCK_SIZE;
                let k_end = std::cmp::min(k_start + BLOCK_SIZE, k);

                for i in i_start..i_end {
                    for j in j_start..j_end {
                        let mut sum = H::zero();
                        for l in k_start..k_end {
                            sum += a_high[[i, l]] * b_high[[l, j]];
                        }
                        c_high[[i, j]] += sum;
                    }
                }
            }
        }
    }

    // Convert back to desired output precision
    let mut c = Array2::<C>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            c[[i, j]] = C::from(c_high[[i, j]]).unwrap_or_else(|| C::zero());
        }
    }

    Ok(c)
}

/// Compute the dot product of two vectors using mixed precision (f32 optimized)
///
/// This function computes the dot product using higher precision
/// internally while inputs and outputs use specified precision. Optimized
/// for f32 vectors with simple but accurate algorithms.
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
pub fn mixed_precision_dot_f32<A, B, C, H>(a: &ArrayView1<A>, b: &ArrayView1<B>) -> LinalgResult<C>
where
    A: Clone + Debug + ToPrimitive + Copy,
    B: Clone + Debug + ToPrimitive + Copy,
    C: Clone + Zero + NumCast + Debug,
    H: Float + Clone + NumCast + Debug + ToPrimitive,
{
    // Check dimensions
    if a.len() != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Vector dimensions must match for dot product: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    // For very short vectors, use simple summation
    if a.len() <= 4 {
        // Convert to high precision
        let a_high = convert::<A, H>(a);
        let b_high = convert::<B, H>(b);

        // Simple dot product for short vectors
        let mut sum = H::zero();
        for i in 0..a.len() {
            sum = sum + a_high[i] * b_high[i];
        }

        // Convert back to desired output precision
        return C::from(sum).ok_or_else(|| {
            LinalgError::ComputationError("Failed to convert dot product result".to_string())
        });
    }

    // For longer vectors, use Kahan summation for numerical stability

    // Convert to high precision
    let a_high = convert::<A, H>(a);
    let b_high = convert::<B, H>(b);

    // Perform dot product in high precision using Kahan summation
    let mut sum = H::zero(); // Running sum
    let mut c = H::zero(); // Running error compensation term

    for i in 0..a.len() {
        let product = a_high[i] * b_high[i];
        let y = product - c; // Corrected value to add
        let t = sum + y; // New sum (may lose low-order bits)
        c = (t - sum) - y; // Recover lost low-order bits
        sum = t; // Update running sum
    }

    // Convert back to desired output precision
    C::from(sum).ok_or_else(|| {
        LinalgError::ComputationError("Failed to convert dot product result".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_mixed_precision_matvec_f32() {
        let a = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
        let x = array![0.5f32, 0.5f32];

        let result =
            mixed_precision_matvec_f32::<f32, f32, f32, f64>(&a.view(), &x.view()).unwrap();

        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], 1.5f32, epsilon = 1e-6);
        assert_relative_eq!(result[1], 3.5f32, epsilon = 1e-6);
    }

    #[test]
    fn test_mixed_precision_matmul_f32_basic_small() {
        let a = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
        let b = array![[5.0f32, 6.0f32], [7.0f32, 8.0f32]];

        let result =
            mixed_precision_matmul_f32_basic::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_relative_eq!(result[[0, 0]], 19.0f32, epsilon = 1e-5);
        assert_relative_eq!(result[[0, 1]], 22.0f32, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 0]], 43.0f32, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 1]], 50.0f32, epsilon = 1e-5);
    }

    #[test]
    fn test_mixed_precision_matmul_f32_basic_large() {
        // Test with larger matrix to exercise blocked algorithm
        let size = 64;
        let a = Array2::<f32>::ones((size, size));
        let b = Array2::<f32>::ones((size, size));

        let result =
            mixed_precision_matmul_f32_basic::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();

        assert_eq!(result.shape(), &[size, size]);
        // Each element should be size (sum of 1*1 over size elements)
        for i in 0..size {
            for j in 0..size {
                assert_relative_eq!(result[[i, j]], size as f32, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_mixed_precision_dot_f32() {
        let a = array![1.0f32, 2.0f32, 3.0f32];
        let b = array![4.0f32, 5.0f32, 6.0f32];

        let result = mixed_precision_dot_f32::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();

        // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_relative_eq!(result, 32.0f32, epsilon = 1e-6);
    }

    #[test]
    fn test_mixed_precision_dot_f32_short() {
        // Test very short vectors (should use simple algorithm)
        let a = array![1.0f32, 2.0f32];
        let b = array![3.0f32, 4.0f32];

        let result = mixed_precision_dot_f32::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();

        // Expected: 1*3 + 2*4 = 3 + 8 = 11
        assert_relative_eq!(result, 11.0f32, epsilon = 1e-6);
    }

    #[test]
    fn test_mixed_precision_dot_f32_precision() {
        // Test with values that might benefit from higher precision
        let a = array![1e-6f32, 2e6f32, 3e-6f32];
        let b = array![4e6f32, 5e-6f32, 6e6f32];

        let result = mixed_precision_dot_f32::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();

        // Expected: 1e-6*4e6 + 2e6*5e-6 + 3e-6*6e6 = 4 + 10 + 18 = 32
        assert_relative_eq!(result, 32.0f32, epsilon = 1e-3);
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let a = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
        let x = array![1.0f32, 2.0f32, 3.0f32]; // Wrong size

        let result = mixed_precision_matvec_f32::<f32, f32, f32, f64>(&a.view(), &x.view());
        assert!(result.is_err());

        let a = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]]; // 2x2
        let b = array![[1.0f32, 2.0f32, 3.0f32]]; // 1x3 - incompatible with 2x2

        let result = mixed_precision_matmul_f32_basic::<f32, f32, f32, f64>(&a.view(), &b.view());
        assert!(result.is_err());

        let a = array![1.0f32, 2.0f32];
        let b = array![1.0f32, 2.0f32, 3.0f32]; // Wrong size

        let result = mixed_precision_dot_f32::<f32, f32, f32, f64>(&a.view(), &b.view());
        assert!(result.is_err());
    }
}
