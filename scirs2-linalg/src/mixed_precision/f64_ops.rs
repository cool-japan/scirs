//! F64-precision optimized operations for mixed-precision linear algebra
//!
//! This module provides specialized implementations optimized for f64 precision,
//! featuring advanced algorithms including parallel processing and sophisticated
//! matrix multiplication strategies for large-scale computations.

use ndarray::{Array2, ArrayView2};
use num_traits::{Float, NumAssign, NumCast, ToPrimitive, Zero};
use std::fmt::Debug;

use super::conversions::convert_2d;
use crate::error::{LinalgError, LinalgResult};

/// Matrix multiplication using mixed precision with parallel processing (f64 optimized)
///
/// This implementation is automatically selected when the "parallel" feature is enabled.
/// Uses advanced blocked algorithms with parallel processing for optimal performance.
#[cfg(feature = "parallel")]
pub fn mixed_precision_matmul_f64_parallel<A, B, C, H>(
    a: &ArrayView2<A>,
    b: &ArrayView2<B>,
) -> LinalgResult<Array2<C>>
where
    A: Clone + Debug + ToPrimitive + Copy + Sync,
    B: Clone + Debug + ToPrimitive + Copy + Sync,
    C: Clone + Zero + NumCast + Debug + Send,
    H: Float + Clone + NumCast + Debug + ToPrimitive + NumAssign + Zero + Send + Sync,
{
    use ndarray::Zip;

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

    // For small matrices, use the naive algorithm without parallelism
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

    // For medium-sized matrices, use blocked algorithm for better cache locality
    if m <= 512 && n <= 512 && k <= 512 {
        // Block size for cache efficiency
        const BLOCK_SIZE: usize = 32;

        let mut c_high = Array2::<H>::zeros((m, n));

        // Block sizes
        let block_m = m.div_ceil(BLOCK_SIZE);
        let block_n = n.div_ceil(BLOCK_SIZE);
        let block_k = k.div_ceil(BLOCK_SIZE);

        // Blocked matrix multiplication algorithm - compute blocks sequentially to avoid borrowing issues
        for bi in 0..block_m {
            let i_start = bi * BLOCK_SIZE;
            let i_end = std::cmp::min(i_start + BLOCK_SIZE, m);

            for bj in 0..block_n {
                let j_start = bj * BLOCK_SIZE;
                let j_end = std::cmp::min(j_start + BLOCK_SIZE, n);

                // Process one block
                for i in i_start..i_end {
                    for j in j_start..j_end {
                        // Initialize to zero
                        c_high[[i, j]] = H::zero();

                        // Compute the dot product for this cell across all blocks in k dimension
                        for bk in 0..block_k {
                            let k_start = bk * BLOCK_SIZE;
                            let k_end = std::cmp::min(k_start + BLOCK_SIZE, k);

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

        // Use parallel iteration for the conversion back
        Zip::from(&mut c)
            .and(&c_high)
            .par_for_each(|c_val, &h_val| {
                *c_val = C::from(h_val).unwrap_or_else(|| C::zero());
            });

        return Ok(c);
    }

    // For very large matrices, use multi-level parallel approach
    let mut c_high = Array2::<H>::zeros((m, n));

    // Compute matrix multiplication with parallel row processing using proper synchronization
    // Use a safer approach to avoid borrowing conflicts
    Zip::from(c_high.rows_mut())
        .and(a_high.rows())
        .par_for_each(|mut c_row, a_row| {
            for (j, c_val) in c_row.iter_mut().enumerate() {
                let mut sum = H::zero();
                for (l, &a_val) in a_row.iter().enumerate() {
                    sum += a_val * b_high[[l, j]];
                }
                *c_val = sum;
            }
        });

    // Convert back to desired output precision in parallel
    let mut c = Array2::<C>::zeros((m, n));
    Zip::from(&mut c)
        .and(&c_high)
        .par_for_each(|c_val, &h_val| {
            *c_val = C::from(h_val).unwrap_or_else(|| C::zero());
        });

    Ok(c)
}

/// Matrix multiplication using mixed precision (f64 optimized, single-threaded)
///
/// This implementation is automatically selected when the "parallel" feature is disabled.
/// Uses efficient blocked algorithms optimized for f64 precision computations.
#[cfg(not(feature = "parallel"))]
pub fn mixed_precision_matmul_f64_serial<A, B, C, H>(
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
    // with a block size of 32 (typical L1 cache-friendly size)
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

/// Unified matrix multiplication interface that selects optimal implementation
///
/// Automatically chooses between parallel and serial implementations based
/// on compile-time feature flags.
pub fn mixed_precision_matmul_f64<A, B, C, H>(
    a: &ArrayView2<A>,
    b: &ArrayView2<B>,
) -> LinalgResult<Array2<C>>
where
    A: Clone + Debug + ToPrimitive + Copy + Sync,
    B: Clone + Debug + ToPrimitive + Copy + Sync,
    C: Clone + Zero + NumCast + Debug + Send,
    H: Float + Clone + NumCast + Debug + ToPrimitive + NumAssign + Zero + Send + Sync,
{
    #[cfg(feature = "parallel")]
    {
        mixed_precision_matmul_f64_parallel::<A, B, C, H>(a, b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        mixed_precision_matmul_f64_serial::<A, B, C, H>(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_mixed_precision_matmul_f64_small() {
        let a = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
        let b = array![[5.0f32, 6.0f32], [7.0f32, 8.0f32]];

        let result =
            mixed_precision_matmul_f64::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_relative_eq!(result[[0, 0]], 19.0f32, epsilon = 1e-5);
        assert_relative_eq!(result[[0, 1]], 22.0f32, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 0]], 43.0f32, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 1]], 50.0f32, epsilon = 1e-5);
    }

    #[test]
    fn test_mixed_precision_matmul_f64_medium() {
        // Test medium-sized matrix (should use blocked algorithm)
        let size = 64;
        let a = Array2::<f32>::ones((size, size));
        let b = Array2::<f32>::ones((size, size));

        let result =
            mixed_precision_matmul_f64::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();

        assert_eq!(result.shape(), &[size, size]);
        // Each element should be size (sum of 1*1 over size elements)
        for i in 0..size {
            for j in 0..size {
                assert_relative_eq!(result[[i, j]], size as f32, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_mixed_precision_matmul_f64_large() {
        // Test larger matrix (should use parallel algorithm if available)
        let size = 128;
        let a = Array2::<f32>::ones((size, size));
        let b = Array2::<f32>::ones((size, size));

        let result =
            mixed_precision_matmul_f64::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();

        assert_eq!(result.shape(), &[size, size]);
        // Each element should be size (sum of 1*1 over size elements)
        for i in 0..10 {
            // Check just a sample to avoid expensive testing
            for j in 0..10 {
                assert_relative_eq!(result[[i, j]], size as f32, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_mixed_precision_matmul_f64_precision() {
        // Test with values that might benefit from higher precision
        let a = array![[1e-6f32, 2e6f32], [3e-6f32, 4e6f32]];
        let b = array![[5e6f32, 6e-6f32], [7e-6f32, 8e6f32]];

        let result =
            mixed_precision_matmul_f64::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();

        assert_eq!(result.shape(), &[2, 2]);

        // Verify results are reasonable (specific values depend on precision)
        assert!(result[[0, 0]].is_finite());
        assert!(result[[0, 1]].is_finite());
        assert!(result[[1, 0]].is_finite());
        assert!(result[[1, 1]].is_finite());
    }

    #[test]
    fn test_mixed_precision_matmul_f64_errors() {
        let a = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]]; // 2x2
        let b = array![[1.0f32, 2.0f32, 3.0f32]]; // 1x3 - incompatible with 2x2

        let result = mixed_precision_matmul_f64::<f32, f32, f32, f64>(&a.view(), &b.view());
        assert!(result.is_err());

        if let Err(LinalgError::ShapeError(_)) = result {
            // Expected error
        } else {
            panic!("Expected ShapeError");
        }
    }

    #[test]
    fn test_mixed_precision_matmul_f64_rectangular() {
        // Test non-square matrices
        let a = array![[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]; // 2x3
        let b = array![[7.0f32, 8.0f32], [9.0f32, 10.0f32], [11.0f32, 12.0f32]]; // 3x2

        let result =
            mixed_precision_matmul_f64::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();

        assert_eq!(result.shape(), &[2, 2]);

        // Calculate expected values manually:
        // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
        // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]
        assert_relative_eq!(result[[0, 0]], 58.0f32, epsilon = 1e-5);
        assert_relative_eq!(result[[0, 1]], 64.0f32, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 0]], 139.0f32, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 1]], 154.0f32, epsilon = 1e-5);
    }
}
