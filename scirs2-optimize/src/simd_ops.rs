//! SIMD-accelerated implementations of key optimization algorithms
//!
//! This module provides vectorized implementations of performance-critical operations
//! used throughout the optimization library. It automatically detects CPU capabilities
//! and falls back to scalar implementations when SIMD is not available.

use ndarray::{Array1, ArrayView1, ArrayView2};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD configuration and capabilities
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
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2_available: is_x86_feature_detected!("avx2"),
                sse41_available: is_x86_feature_detected!("sse4.1"),
                fma_available: is_x86_feature_detected!("fma"),
                vector_width: if is_x86_feature_detected!("avx2") {
                    4 // AVX2 can process 4 f64 values simultaneously
                } else if is_x86_feature_detected!("sse4.1") {
                    2 // SSE can process 2 f64 values simultaneously
                } else {
                    1 // Scalar fallback
                },
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                avx2_available: false,
                sse41_available: false,
                fma_available: false,
                vector_width: 1,
            }
        }
    }

    /// Check if any SIMD support is available
    pub fn has_simd(&self) -> bool {
        self.avx2_available || self.sse41_available
    }
}

/// SIMD-accelerated vector operations
pub struct SimdVectorOps {
    config: SimdConfig,
}

impl SimdVectorOps {
    /// Create new SIMD vector operations with auto-detected capabilities
    pub fn new() -> Self {
        Self {
            config: SimdConfig::detect(),
        }
    }

    /// Create with specific configuration (for testing)
    pub fn with_config(config: SimdConfig) -> Self {
        Self { config }
    }

    /// Get the SIMD configuration
    pub fn config(&self) -> &SimdConfig {
        &self.config
    }

    /// SIMD-accelerated dot product
    pub fn dot_product(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        assert_eq!(a.len(), b.len());

        if self.config.has_simd() && a.len() >= 4 {
            self.simd_dot_product(a, b)
        } else {
            self.scalar_dot_product(a, b)
        }
    }

    /// SIMD-accelerated vector addition: result = a + b
    pub fn add(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(a.len(), b.len());

        if self.config.has_simd() && a.len() >= 4 {
            self.simd_add(a, b)
        } else {
            self.scalar_add(a, b)
        }
    }

    /// SIMD-accelerated vector subtraction: result = a - b
    pub fn sub(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(a.len(), b.len());

        if self.config.has_simd() && a.len() >= 4 {
            self.simd_sub(a, b)
        } else {
            self.scalar_sub(a, b)
        }
    }

    /// SIMD-accelerated scalar multiplication: result = alpha * a
    pub fn scale(&self, alpha: f64, a: &ArrayView1<f64>) -> Array1<f64> {
        if self.config.has_simd() && a.len() >= 4 {
            self.simd_scale(alpha, a)
        } else {
            self.scalar_scale(alpha, a)
        }
    }

    /// SIMD-accelerated AXPY operation: result = alpha * x + y
    pub fn axpy(&self, alpha: f64, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(x.len(), y.len());

        if self.config.has_simd() && x.len() >= 4 {
            self.simd_axpy(alpha, x, y)
        } else {
            self.scalar_axpy(alpha, x, y)
        }
    }

    /// SIMD-accelerated vector norm (L2)
    pub fn norm(&self, a: &ArrayView1<f64>) -> f64 {
        if self.config.has_simd() && a.len() >= 4 {
            self.simd_norm(a)
        } else {
            self.scalar_norm(a)
        }
    }

    /// SIMD-accelerated matrix-vector multiplication
    pub fn matvec(&self, matrix: &ArrayView2<f64>, vector: &ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(matrix.ncols(), vector.len());

        if self.config.has_simd() && vector.len() >= 4 {
            self.simd_matvec(matrix, vector)
        } else {
            self.scalar_matvec(matrix, vector)
        }
    }

    // Scalar implementations (fallback)

    fn scalar_dot_product(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
    }

    fn scalar_add(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        a.iter().zip(b.iter()).map(|(&ai, &bi)| ai + bi).collect()
    }

    fn scalar_sub(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        a.iter().zip(b.iter()).map(|(&ai, &bi)| ai - bi).collect()
    }

    fn scalar_scale(&self, alpha: f64, a: &ArrayView1<f64>) -> Array1<f64> {
        a.iter().map(|&ai| alpha * ai).collect()
    }

    fn scalar_axpy(&self, alpha: f64, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| alpha * xi + yi)
            .collect()
    }

    fn scalar_norm(&self, a: &ArrayView1<f64>) -> f64 {
        a.iter().map(|&ai| ai * ai).sum::<f64>().sqrt()
    }

    fn scalar_matvec(&self, matrix: &ArrayView2<f64>, vector: &ArrayView1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(matrix.nrows());
        for (i, row) in matrix.outer_iter().enumerate() {
            result[i] = self.scalar_dot_product(&row, vector);
        }
        result
    }

    // SIMD implementations

    #[cfg(target_arch = "x86_64")]
    fn simd_dot_product(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        let n = a.len();
        let simd_len = n - (n % self.config.vector_width);

        if self.config.avx2_available {
            self.avx2_dot_product(a, b, simd_len)
        } else if self.config.sse41_available {
            self.sse_dot_product(a, b, simd_len)
        } else {
            self.scalar_dot_product(a, b)
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn avx2_dot_product(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>, simd_len: usize) -> f64 {
        #[allow(unused_assignments)]
        let mut sum = 0.0;

        unsafe {
            let mut acc = _mm256_setzero_pd();

            for i in (0..simd_len).step_by(4) {
                let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
                let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));

                if self.config.fma_available {
                    acc = _mm256_fmadd_pd(a_vec, b_vec, acc);
                } else {
                    let prod = _mm256_mul_pd(a_vec, b_vec);
                    acc = _mm256_add_pd(acc, prod);
                }
            }

            // Horizontal sum of the accumulator
            let sum_vec = _mm256_hadd_pd(acc, acc);
            let sum_low = _mm256_extractf128_pd(sum_vec, 0);
            let sum_high = _mm256_extractf128_pd(sum_vec, 1);
            let final_sum = _mm_add_pd(sum_low, sum_high);

            sum = _mm_cvtsd_f64(final_sum);
        }

        // Handle remaining elements
        for i in simd_len..a.len() {
            sum += a[i] * b[i];
        }

        sum
    }

    #[cfg(target_arch = "x86_64")]
    fn sse_dot_product(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>, simd_len: usize) -> f64 {
        #[allow(unused_assignments)]
        let mut sum = 0.0;

        unsafe {
            let mut acc = _mm_setzero_pd();

            for i in (0..simd_len).step_by(2) {
                let a_vec = _mm_loadu_pd(a.as_ptr().add(i));
                let b_vec = _mm_loadu_pd(b.as_ptr().add(i));
                let prod = _mm_mul_pd(a_vec, b_vec);
                acc = _mm_add_pd(acc, prod);
            }

            // Horizontal sum
            let sum_vec = _mm_hadd_pd(acc, acc);
            sum = _mm_cvtsd_f64(sum_vec);
        }

        // Handle remaining elements
        for i in simd_len..a.len() {
            sum += a[i] * b[i];
        }

        sum
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_dot_product(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        self.scalar_dot_product(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_add(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        let n = a.len();
        let mut result = Array1::<f64>::zeros(n);
        let simd_len = n - (n % self.config.vector_width);

        if self.config.avx2_available {
            unsafe {
                for i in (0..simd_len).step_by(4) {
                    let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
                    let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
                    let sum_vec = _mm256_add_pd(a_vec, b_vec);
                    _mm256_storeu_pd(result.as_mut_ptr().add(i), sum_vec);
                }
            }
        } else if self.config.sse41_available {
            unsafe {
                for i in (0..simd_len).step_by(2) {
                    let a_vec = _mm_loadu_pd(a.as_ptr().add(i));
                    let b_vec = _mm_loadu_pd(b.as_ptr().add(i));
                    let sum_vec = _mm_add_pd(a_vec, b_vec);
                    _mm_storeu_pd(result.as_mut_ptr().add(i), sum_vec);
                }
            }
        }

        // Handle remaining elements
        for i in simd_len..n {
            result[i] = a[i] + b[i];
        }

        result
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_add(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        self.scalar_add(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_sub(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        let n = a.len();
        let mut result = Array1::<f64>::zeros(n);
        let simd_len = n - (n % self.config.vector_width);

        if self.config.avx2_available {
            unsafe {
                for i in (0..simd_len).step_by(4) {
                    let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
                    let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
                    let diff_vec = _mm256_sub_pd(a_vec, b_vec);
                    _mm256_storeu_pd(result.as_mut_ptr().add(i), diff_vec);
                }
            }
        } else if self.config.sse41_available {
            unsafe {
                for i in (0..simd_len).step_by(2) {
                    let a_vec = _mm_loadu_pd(a.as_ptr().add(i));
                    let b_vec = _mm_loadu_pd(b.as_ptr().add(i));
                    let diff_vec = _mm_sub_pd(a_vec, b_vec);
                    _mm_storeu_pd(result.as_mut_ptr().add(i), diff_vec);
                }
            }
        }

        // Handle remaining elements
        for i in simd_len..n {
            result[i] = a[i] - b[i];
        }

        result
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_sub(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        self.scalar_sub(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_scale(&self, alpha: f64, a: &ArrayView1<f64>) -> Array1<f64> {
        let n = a.len();
        let mut result = Array1::<f64>::zeros(n);
        let simd_len = n - (n % self.config.vector_width);

        if self.config.avx2_available {
            unsafe {
                let alpha_vec = _mm256_set1_pd(alpha);
                for i in (0..simd_len).step_by(4) {
                    let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
                    let scaled_vec = _mm256_mul_pd(alpha_vec, a_vec);
                    _mm256_storeu_pd(result.as_mut_ptr().add(i), scaled_vec);
                }
            }
        } else if self.config.sse41_available {
            unsafe {
                let alpha_vec = _mm_set1_pd(alpha);
                for i in (0..simd_len).step_by(2) {
                    let a_vec = _mm_loadu_pd(a.as_ptr().add(i));
                    let scaled_vec = _mm_mul_pd(alpha_vec, a_vec);
                    _mm_storeu_pd(result.as_mut_ptr().add(i), scaled_vec);
                }
            }
        }

        // Handle remaining elements
        for i in simd_len..n {
            result[i] = alpha * a[i];
        }

        result
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_scale(&self, alpha: f64, a: &ArrayView1<f64>) -> Array1<f64> {
        self.scalar_scale(alpha, a)
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_axpy(&self, alpha: f64, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        let n = x.len();
        let mut result = Array1::<f64>::zeros(n);
        let simd_len = n - (n % self.config.vector_width);

        if self.config.avx2_available {
            unsafe {
                let alpha_vec = _mm256_set1_pd(alpha);
                for i in (0..simd_len).step_by(4) {
                    let x_vec = _mm256_loadu_pd(x.as_ptr().add(i));
                    let y_vec = _mm256_loadu_pd(y.as_ptr().add(i));

                    let result_vec = if self.config.fma_available {
                        _mm256_fmadd_pd(alpha_vec, x_vec, y_vec)
                    } else {
                        let scaled_x = _mm256_mul_pd(alpha_vec, x_vec);
                        _mm256_add_pd(scaled_x, y_vec)
                    };

                    _mm256_storeu_pd(result.as_mut_ptr().add(i), result_vec);
                }
            }
        } else if self.config.sse41_available {
            unsafe {
                let alpha_vec = _mm_set1_pd(alpha);
                for i in (0..simd_len).step_by(2) {
                    let x_vec = _mm_loadu_pd(x.as_ptr().add(i));
                    let y_vec = _mm_loadu_pd(y.as_ptr().add(i));
                    let scaled_x = _mm_mul_pd(alpha_vec, x_vec);
                    let result_vec = _mm_add_pd(scaled_x, y_vec);
                    _mm_storeu_pd(result.as_mut_ptr().add(i), result_vec);
                }
            }
        }

        // Handle remaining elements
        for i in simd_len..n {
            result[i] = alpha * x[i] + y[i];
        }

        result
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_axpy(&self, alpha: f64, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        self.scalar_axpy(alpha, x, y)
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_norm(&self, a: &ArrayView1<f64>) -> f64 {
        let n = a.len();
        let simd_len = n - (n % self.config.vector_width);
        let mut sum = 0.0;

        if self.config.avx2_available {
            unsafe {
                let mut acc = _mm256_setzero_pd();

                for i in (0..simd_len).step_by(4) {
                    let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));

                    if self.config.fma_available {
                        acc = _mm256_fmadd_pd(a_vec, a_vec, acc);
                    } else {
                        let sq = _mm256_mul_pd(a_vec, a_vec);
                        acc = _mm256_add_pd(acc, sq);
                    }
                }

                // Horizontal sum
                let sum_vec = _mm256_hadd_pd(acc, acc);
                let sum_low = _mm256_extractf128_pd(sum_vec, 0);
                let sum_high = _mm256_extractf128_pd(sum_vec, 1);
                let final_sum = _mm_add_pd(sum_low, sum_high);
                sum = _mm_cvtsd_f64(final_sum);
            }
        } else if self.config.sse41_available {
            unsafe {
                let mut acc = _mm_setzero_pd();

                for i in (0..simd_len).step_by(2) {
                    let a_vec = _mm_loadu_pd(a.as_ptr().add(i));
                    let sq = _mm_mul_pd(a_vec, a_vec);
                    acc = _mm_add_pd(acc, sq);
                }

                // Horizontal sum
                let sum_vec = _mm_hadd_pd(acc, acc);
                sum = _mm_cvtsd_f64(sum_vec);
            }
        }

        // Handle remaining elements
        for i in simd_len..n {
            sum += a[i] * a[i];
        }

        sum.sqrt()
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_norm(&self, a: &ArrayView1<f64>) -> f64 {
        self.scalar_norm(a)
    }

    #[cfg(target_arch = "x86_64")]
    fn simd_matvec(&self, matrix: &ArrayView2<f64>, vector: &ArrayView1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(matrix.nrows());

        for (i, row) in matrix.outer_iter().enumerate() {
            result[i] = self.simd_dot_product(&row, vector);
        }

        result
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn simd_matvec(&self, matrix: &ArrayView2<f64>, vector: &ArrayView1<f64>) -> Array1<f64> {
        self.scalar_matvec(matrix, vector)
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
    use ndarray::array;

    #[test]
    fn test_simd_config_detection() {
        let config = SimdConfig::detect();

        // Just verify that detection runs without error
        assert!(config.vector_width >= 1);

        #[cfg(target_arch = "x86_64")]
        {
            // On x86_64, at least SSE should be available on most modern systems
            println!(
                "AVX2: {}, SSE4.1: {}, FMA: {}, Vector width: {}",
                config.avx2_available,
                config.sse41_available,
                config.fma_available,
                config.vector_width
            );
        }
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
    fn test_scalar_vs_simd_consistency() {
        let config_scalar = SimdConfig {
            avx2_available: false,
            sse41_available: false,
            fma_available: false,
            vector_width: 1,
        };
        let config_simd = SimdConfig::detect();

        let ops_scalar = SimdVectorOps::with_config(config_scalar);
        let ops_simd = SimdVectorOps::with_config(config_simd);

        let a = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = array![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];

        // Test dot product consistency
        let dot_scalar = ops_scalar.dot_product(&a.view(), &b.view());
        let dot_simd = ops_simd.dot_product(&a.view(), &b.view());
        assert_abs_diff_eq!(dot_scalar, dot_simd, epsilon = 1e-10);

        // Test norm consistency
        let norm_scalar = ops_scalar.norm(&a.view());
        let norm_simd = ops_simd.norm(&a.view());
        assert_abs_diff_eq!(norm_scalar, norm_simd, epsilon = 1e-10);

        // Test AXPY consistency
        let axpy_scalar = ops_scalar.axpy(2.0, &a.view(), &b.view());
        let axpy_simd = ops_simd.axpy(2.0, &a.view(), &b.view());
        for i in 0..axpy_scalar.len() {
            assert_abs_diff_eq!(axpy_scalar[i], axpy_simd[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_large_vectors() {
        let ops = SimdVectorOps::new();
        let n = 1000;
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
