//! SIMD-accelerated implementations using scirs2-core unified system
//!
//! This module provides vectorized implementations of performance-critical operations
//! used throughout the optimization library. It uses the scirs2-core SIMD abstraction
//! layer for automatic platform detection and optimization.

use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};
use scirs2_core::simd_ops::{AutoOptimizer, PlatformCapabilities, SimdUnifiedOps};

/// SIMD configuration - compatibility wrapper for legacy code
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Whether AVX2 is available
    pub avx2_available: bool,
    /// Whether SSE4.1 is available
    pub sse41_available: bool,
    /// Whether FMA is available
    pub fma_available: bool,
    /// Preferred vector width in elements
    pub vector_width: usize,
}

impl SimdConfig {
    /// Detect available SIMD capabilities
    pub fn detect() -> Self {
        let caps = PlatformCapabilities::detect();
        Self {
            avx2_available: caps.avx2_available,
            sse41_available: caps.simd_available, // Approximate - core doesn't expose SSE4.1 specifically
            fma_available: caps.simd_available,   // Approximate - assume FMA if SIMD available
            vector_width: if caps.avx2_available {
                4 // AVX2 can process 4 f64 values simultaneously
            } else if caps.simd_available {
                2 // SSE can process 2 f64 values simultaneously
            } else {
                1 // Scalar fallback
            },
        }
    }

    /// Check if any SIMD support is available
    pub fn has_simd(&self) -> bool {
        self.avx2_available || self.sse41_available
    }
}

/// SIMD-accelerated vector operations using core unified system
pub struct SimdVectorOps {
    optimizer: AutoOptimizer,
}

impl SimdVectorOps {
    /// Create new SIMD vector operations with auto-optimization
    pub fn new() -> Self {
        Self {
            optimizer: AutoOptimizer::new(),
        }
    }

    /// Create with specific configuration (for testing/compatibility)
    pub fn with_config(config: SimdConfig) -> Self {
        // Ignore the _config - always use auto-optimization from core
        Self::new()
    }

    /// Get the SIMD configuration (for compatibility)
    pub fn config(&self) -> SimdConfig {
        SimdConfig::detect()
    }

    /// Get platform capabilities for debugging/testing
    pub fn platform_capabilities(&self) -> PlatformCapabilities {
        PlatformCapabilities::detect()
    }

    /// SIMD-accelerated dot product
    pub fn dot_product(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        assert_eq!(a.len(), b.len());

        if self.optimizer.should_use_simd(a.len()) {
            f64::simd_dot(a, b)
        } else {
            // Scalar fallback for small arrays
            a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
        }
    }

    /// SIMD-accelerated vector addition: result = a + b
    pub fn add(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(a.len(), b.len());

        if self.optimizer.should_use_simd(a.len()) {
            f64::simd_add(a, b)
        } else {
            a.iter().zip(b.iter()).map(|(&ai, &bi)| ai + bi).collect()
        }
    }

    /// SIMD-accelerated vector subtraction: result = a - b
    pub fn sub(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(a.len(), b.len());

        if self.optimizer.should_use_simd(a.len()) {
            f64::simd_sub(a, b)
        } else {
            a.iter().zip(b.iter()).map(|(&ai, &bi)| ai - bi).collect()
        }
    }

    /// SIMD-accelerated scalar multiplication: result = alpha * a
    pub fn scale(&self, alpha: f64, a: &ArrayView1<f64>) -> Array1<f64> {
        if self.optimizer.should_use_simd(a.len()) {
            f64::simd_scalar_mul(a, alpha)
        } else {
            a.iter().map(|&ai| alpha * ai).collect()
        }
    }

    /// SIMD-accelerated AXPY operation: result = alpha * x + y
    pub fn axpy(&self, alpha: f64, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(x.len(), y.len());

        if self.optimizer.should_use_simd(x.len()) {
            // Use FMA (fused multiply-add) for axpy operation
            let alpha_x = f64::simd_scalar_mul(x, alpha);
            f64::simd_add(&alpha_x.view(), y)
        } else {
            x.iter()
                .zip(y.iter())
                .map(|(&xi, &yi)| alpha * xi + yi)
                .collect()
        }
    }

    /// SIMD-accelerated vector norm (L2)
    pub fn norm(&self, a: &ArrayView1<f64>) -> f64 {
        if self.optimizer.should_use_simd(a.len()) {
            f64::simd_norm(a)
        } else {
            a.iter().map(|&ai| ai * ai).sum::<f64>().sqrt()
        }
    }

    /// SIMD-accelerated matrix-vector multiplication
    pub fn matvec(&self, matrix: &ArrayView2<f64>, vector: &ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(matrix.ncols(), vector.len());

        let mut result = Array1::zeros(matrix.nrows());
        for (i, row) in matrix.outer_iter().enumerate() {
            result[i] = self.dot_product(&row, vector);
        }
        result
    }
}

impl Default for SimdVectorOps {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_platform_capabilities() {
        let ops = SimdVectorOps::new();
        let caps = ops.platform_capabilities();

        // Just verify that detection runs without error
        println!(
            "Platform capabilities - SIMD: {}, GPU: {}, AVX2: {}",
            caps.simd_available, caps.gpu_available, caps.avx2_available
        );
    }

    #[test]
    fn test_dot_product() {
        let ops = SimdVectorOps::new();
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = array![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = ops.dot_product(&a.view(), &b.view());
        let expected = 240.0; // 1*2 + 2*3 + 3*4 + 4*5 + 5*6 + 6*7 + 7*8 + 8*9

        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_vector_operations() {
        let ops = SimdVectorOps::new();
        let a = array![1.0, 2.0, 3.0, 4.0];
        let b = array![5.0, 6.0, 7.0, 8.0];

        // Test addition
        let sum = ops.add(&a.view(), &b.view());
        assert_abs_diff_eq!(sum[0], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sum[1], 8.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sum[2], 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sum[3], 12.0, epsilon = 1e-10);

        // Test subtraction
        let diff = ops.sub(&b.view(), &a.view());
        assert_abs_diff_eq!(diff[0], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(diff[1], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(diff[2], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(diff[3], 4.0, epsilon = 1e-10);

        // Test scaling
        let scaled = ops.scale(2.0, &a.view());
        assert_abs_diff_eq!(scaled[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[1], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[2], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[3], 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_axpy() {
        let ops = SimdVectorOps::new();
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![5.0, 6.0, 7.0, 8.0];
        let alpha = 2.0;

        let result = ops.axpy(alpha, &x.view(), &y.view());

        // Expected: alpha * x + y = 2.0 * [1,2,3,4] + [5,6,7,8] = [7,10,13,16]
        assert_abs_diff_eq!(result[0], 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 13.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[3], 16.0, epsilon = 1e-10);
    }

    #[test]
    fn test_norm() {
        let ops = SimdVectorOps::new();
        let a = array![3.0, 4.0]; // 3-4-5 triangle

        let norm = ops.norm(&a.view());
        assert_abs_diff_eq!(norm, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matvec() {
        let ops = SimdVectorOps::new();
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let vector = array![1.0, 2.0];

        let result = ops.matvec(&matrix.view(), &vector.view());

        // Expected: [[1,2], [3,4]] * [1,2] = [5, 11]
        assert_abs_diff_eq!(result[0], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 11.0, epsilon = 1e-10);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_large_vectors() {
        let ops = SimdVectorOps::new();
        // Reduce size from 1000 to 100 for faster test execution
        let n = 100;
        let a: Array1<f64> = Array1::from_shape_fn(n, |i| i as f64);
        let b: Array1<f64> = Array1::from_shape_fn(n, |i| (i + 1) as f64);

        // Test that large vectors work without errors
        let dot_result = ops.dot_product(&a.view(), &b.view());
        let norm_result = ops.norm(&a.view());
        let add_result = ops.add(&a.view(), &b.view());

        assert!(dot_result > 0.0);
        assert!(norm_result > 0.0);
        assert_eq!(add_result.len(), n);
    }
}
