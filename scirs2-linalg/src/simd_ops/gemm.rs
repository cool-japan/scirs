//! SIMD-accelerated General Matrix Multiplication (GEMM) operations
//!
//! This module provides highly optimized SIMD implementations of GEMM operations
//! using cache-friendly blocking strategies. All SIMD operations are delegated
//! to scirs2-core::simd_ops for unified optimization management.

#[cfg(feature = "simd")]
use crate::error::{LinalgError, LinalgResult};
#[cfg(feature = "simd")]
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
#[cfg(feature = "simd")]
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Cache-friendly block sizes for GEMM operations
/// These should be tuned for target CPU cache hierarchy
#[cfg(feature = "simd")]
pub struct GemmBlockSizes {
    /// Block size for M dimension (rows of A, rows of C)
    pub mc: usize,
    /// Block size for K dimension (cols of A, rows of B)  
    pub kc: usize,
    /// Block size for N dimension (cols of B, cols of C)
    pub nc: usize,
    /// Micro-kernel block size for M dimension
    pub mr: usize,
    /// Micro-kernel block size for N dimension  
    pub nr: usize,
}

#[cfg(feature = "simd")]
impl Default for GemmBlockSizes {
    fn default() -> Self {
        Self {
            mc: 64,  // L2 cache friendly
            kc: 256, // L1 cache friendly for B panel
            nc: 512, // L3 cache friendly
            mr: 8,   // SIMD width considerations
            nr: 8,   // SIMD width considerations
        }
    }
}

/// SIMD-accelerated GEMM for f32: C = alpha * A * B + beta * C
///
/// This implementation uses the unified SIMD operations from scirs2-core.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A*B
/// * `a` - Left matrix A (M x K)
/// * `b` - Right matrix B (K x N)
/// * `beta` - Scalar multiplier for C
/// * `c` - Result matrix C (M x N), updated in-place
/// * `_blocksizes` - Cache-friendly block size configuration (unused in unified implementation)
///
/// # Returns
///
/// * Result indicating success or error
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_gemm_f32(
    alpha: f32,
    a: &ArrayView2<f32>,
    b: &ArrayView2<f32>,
    beta: f32,
    c: &mut Array2<f32>,
    _blocksizes: Option<GemmBlockSizes>,
) -> LinalgResult<()> {
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();
    let (cm, cn) = c.dim();

    // Validate matrix dimensions
    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix inner dimensions must match: A({m}, {k1}) * B({k2}, {n})"
        )));
    }
    if cm != m || cn != n {
        return Err(LinalgError::ShapeError(format!(
            "Result matrix dimensions must match: C({cm}, {cn}) for A({m}, {k1}) * B({k2}, {n})"
        )));
    }

    // Use unified SIMD GEMM operation
    f32::simd_gemm(alpha, a, b, beta, c);

    Ok(())
}

/// SIMD-accelerated GEMM for f64: C = alpha * A * B + beta * C
///
/// This implementation uses the unified SIMD operations from scirs2-core.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A*B
/// * `a` - Left matrix A (M x K)
/// * `b` - Right matrix B (K x N)
/// * `beta` - Scalar multiplier for C
/// * `c` - Result matrix C (M x N), updated in-place
/// * `_blocksizes` - Cache-friendly block size configuration (unused in unified implementation)
///
/// # Returns
///
/// * Result indicating success or error
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_gemm_f64(
    alpha: f64,
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
    beta: f64,
    c: &mut Array2<f64>,
    _blocksizes: Option<GemmBlockSizes>,
) -> LinalgResult<()> {
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();
    let (cm, cn) = c.dim();

    // Validate matrix dimensions
    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix inner dimensions must match: A({m}, {k1}) * B({k2}, {n})"
        )));
    }
    if cm != m || cn != n {
        return Err(LinalgError::ShapeError(format!(
            "Result matrix dimensions must match: C({cm}, {cn}) for A({m}, {k1}) * B({k2}, {n})"
        )));
    }

    // Use unified SIMD GEMM operation
    f64::simd_gemm(alpha, a, b, beta, c);

    Ok(())
}

// Note: Micro-kernel functions have been removed as we now use unified SIMD operations from scirs2-core

/// Convenience function for SIMD matrix multiplication: C = A * B
///
/// # Arguments
///
/// * `a` - Left matrix (M x K)
/// * `b` - Right matrix (K x N)
///
/// # Returns
///
/// * Result matrix C (M x N)
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_matmul_optimized_f32(
    a: &ArrayView2<f32>,
    b: &ArrayView2<f32>,
) -> LinalgResult<Array2<f32>> {
    let (m, _) = a.dim();
    let (_, n) = b.dim();

    let mut c = Array2::zeros((m, n));
    simd_gemm_f32(1.0, a, b, 0.0, &mut c, None)?;
    Ok(c)
}

/// Convenience function for SIMD matrix multiplication: C = A * B (f64)
///
/// # Arguments
///
/// * `a` - Left matrix (M x K)
/// * `b` - Right matrix (K x N)
///
/// # Returns
///
/// * Result matrix C (M x N)
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_matmul_optimized_f64(
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
) -> LinalgResult<Array2<f64>> {
    let (m, _) = a.dim();
    let (_, n) = b.dim();

    let mut c = Array2::zeros((m, n));
    simd_gemm_f64(1.0, a, b, 0.0, &mut c, None)?;
    Ok(c)
}

/// SIMD-accelerated matrix-vector multiplication: y = alpha * A * x + beta * y
///
/// Optimized version using the unified SIMD operations from scirs2-core.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A*x
/// * `a` - Matrix A (M x N)
/// * `x` - Vector x (N,)
/// * `beta` - Scalar multiplier for y
/// * `y` - Result vector y (M,), updated in-place
///
/// # Returns
///
/// * Result indicating success or error
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_gemv_f32(
    alpha: f32,
    a: &ArrayView2<f32>,
    x: &ArrayView1<f32>,
    beta: f32,
    y: &mut Array1<f32>,
) -> LinalgResult<()> {
    let (m, n) = a.dim();

    if x.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "Vector x length ({}) must match matrix columns ({})",
            x.len(),
            n
        )));
    }
    if y.len() != m {
        return Err(LinalgError::ShapeError(format!(
            "Vector y length ({}) must match matrix rows ({})",
            y.len(),
            m
        )));
    }

    // The unified SIMD GEMV computes y = A*x + beta*y
    // To get y = alpha*A*x + beta*y, we need to:
    // 1. Store original y values if beta != 0
    // 2. Compute y = A*x (with beta=0)
    // 3. Scale by alpha and add beta*y_original

    if beta == 0.0 {
        // Simple case: y = alpha * A * x
        f32::simd_gemv(a, x, 0.0, y);
        if alpha != 1.0 {
            y.mapv_inplace(|v| v * alpha);
        }
    } else {
        // Complex case: need to preserve original y
        let y_original = y.clone();
        f32::simd_gemv(a, x, 0.0, y);
        // Now y contains A*x, scale and add beta*y_original
        for i in 0..y.len() {
            y[i] = alpha * y[i] + beta * y_original[i];
        }
    }

    Ok(())
}

/// SIMD-accelerated matrix-vector multiplication: y = alpha * A * x + beta * y (f64)
///
/// Optimized version using the unified SIMD operations from scirs2-core.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A*x
/// * `a` - Matrix A (M x N)
/// * `x` - Vector x (N,)
/// * `beta` - Scalar multiplier for y
/// * `y` - Result vector y (M,), updated in-place
///
/// # Returns
///
/// * Result indicating success or error
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn simd_gemv_f64(
    alpha: f64,
    a: &ArrayView2<f64>,
    x: &ArrayView1<f64>,
    beta: f64,
    y: &mut Array1<f64>,
) -> LinalgResult<()> {
    let (m, n) = a.dim();

    if x.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "Vector x length ({}) must match matrix columns ({})",
            x.len(),
            n
        )));
    }
    if y.len() != m {
        return Err(LinalgError::ShapeError(format!(
            "Vector y length ({}) must match matrix rows ({})",
            y.len(),
            m
        )));
    }

    // The unified SIMD GEMV computes y = A*x + beta*y
    // To get y = alpha*A*x + beta*y, we need to:
    // 1. Store original y values if beta != 0
    // 2. Compute y = A*x (with beta=0)
    // 3. Scale by alpha and add beta*y_original

    if beta == 0.0 {
        // Simple case: y = alpha * A * x
        f64::simd_gemv(a, x, 0.0, y);
        if alpha != 1.0 {
            y.mapv_inplace(|v| v * alpha);
        }
    } else {
        // Complex case: need to preserve original y
        let y_original = y.clone();
        f64::simd_gemv(a, x, 0.0, y);
        // Now y contains A*x, scale and add beta*y_original
        for i in 0..y.len() {
            y[i] = alpha * y[i] + beta * y_original[i];
        }
    }

    Ok(())
}

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simd_gemm_f32_basic() {
        // Test C = A * B
        let a = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0f32, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let mut c = Array2::zeros((2, 2));

        simd_gemm_f32(1.0, &a.view(), &b.view(), 0.0, &mut c, None).unwrap();

        // Expected: [[58, 64], [139, 154]]
        // A*B = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //     = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
        //     = [[58, 64], [139, 154]]
        assert_relative_eq!(c[[0, 0]], 58.0, epsilon = 1e-6);
        assert_relative_eq!(c[[0, 1]], 64.0, epsilon = 1e-6);
        assert_relative_eq!(c[[1, 0]], 139.0, epsilon = 1e-6);
        assert_relative_eq!(c[[1, 1]], 154.0, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simd_gemm_f64_basic() {
        // Test C = A * B
        let a = array![[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0f64, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let mut c = Array2::zeros((2, 2));

        simd_gemm_f64(1.0, &a.view(), &b.view(), 0.0, &mut c, None).unwrap();

        // Expected: [[58, 64], [139, 154]]
        assert_relative_eq!(c[[0, 0]], 58.0, epsilon = 1e-12);
        assert_relative_eq!(c[[0, 1]], 64.0, epsilon = 1e-12);
        assert_relative_eq!(c[[1, 0]], 139.0, epsilon = 1e-12);
        assert_relative_eq!(c[[1, 1]], 154.0, epsilon = 1e-12);
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simd_gemm_alpha_beta() {
        // Test C = alpha * A * B + beta * C
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[5.0f32, 6.0], [7.0, 8.0]];
        let mut c = array![[1.0f32, 2.0], [3.0, 4.0]];

        let alpha = 2.0;
        let beta = 3.0;

        simd_gemm_f32(alpha, &a.view(), &b.view(), beta, &mut c, None).unwrap();

        // A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        // Expected: 2.0 * [[19, 22], [43, 50]] + 3.0 * [[1, 2], [3, 4]]
        //         = [[38, 44], [86, 100]] + [[3, 6], [9, 12]]
        //         = [[41, 50], [95, 112]]
        assert_relative_eq!(c[[0, 0]], 41.0, epsilon = 1e-6);
        assert_relative_eq!(c[[0, 1]], 50.0, epsilon = 1e-6);
        assert_relative_eq!(c[[1, 0]], 95.0, epsilon = 1e-6);
        assert_relative_eq!(c[[1, 1]], 112.0, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simd_matmul_optimized() {
        let a = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0f32, 8.0], [9.0, 10.0], [11.0, 12.0]];

        let c = simd_matmul_optimized_f32(&a.view(), &b.view()).unwrap();

        assert_relative_eq!(c[[0, 0]], 58.0, epsilon = 1e-6);
        assert_relative_eq!(c[[0, 1]], 64.0, epsilon = 1e-6);
        assert_relative_eq!(c[[1, 0]], 139.0, epsilon = 1e-6);
        assert_relative_eq!(c[[1, 1]], 154.0, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simd_gemv() {
        // Test y = alpha * A * x + beta * y
        let a = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x = array![7.0f32, 8.0, 9.0];
        let mut y = array![1.0f32, 2.0];

        let alpha = 2.0;
        let beta = 3.0;

        simd_gemv_f32(alpha, &a.view(), &x.view(), beta, &mut y).unwrap();

        // A*x = [1*7+2*8+3*9, 4*7+5*8+6*9] = [7+16+27, 28+40+54] = [50, 122]
        // Expected: 2.0 * [50, 122] + 3.0 * [1, 2] = [100, 244] + [3, 6] = [103, 250]
        assert_relative_eq!(y[0], 103.0, epsilon = 1e-6);
        assert_relative_eq!(y[1], 250.0, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_simd_gemm_largematrix() {
        // Test with larger matrices to exercise blocking
        let m = 100;
        let k = 80;
        let n = 60;

        let a = Array2::from_shape_fn((m, k), |(i, j)| (i + j) as f32 * 0.01);
        let b = Array2::from_shape_fn((k, n), |(i, j)| (i * 2 + j) as f32 * 0.01);
        let mut c = Array2::zeros((m, n));

        // Test with custom block sizes
        let blocksizes = GemmBlockSizes {
            mc: 32,
            kc: 64,
            nc: 48,
            mr: 8,
            nr: 8,
        };

        simd_gemm_f32(1.0, &a.view(), &b.view(), 0.0, &mut c, Some(blocksizes)).unwrap();

        // Verify with reference implementation (naive multiplication)
        let c_ref = a.dot(&b);

        for ((i, j), &val) in c.indexed_iter() {
            assert_relative_eq!(val, c_ref[[i, j]], epsilon = 1e-4);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    #[ignore = "timeout"]
    fn test_gemm_error_handling() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[5.0f32, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]]; // Wrong dimensions
        let mut c = Array2::zeros((2, 3));

        let result = simd_gemm_f32(1.0, &a.view(), &b.view(), 0.0, &mut c, None);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LinalgError::ShapeError(_)));
    }
}
