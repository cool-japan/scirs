//! Unified SIMD operations abstraction layer
//!
//! This module provides a comprehensive abstraction layer for all SIMD operations
//! used across the scirs2 ecosystem. All modules should use these operations
//! instead of implementing their own SIMD code.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::Zero;

/// Unified SIMD operations trait
pub trait SimdUnifiedOps: Sized + Copy + PartialOrd + Zero {
    /// Element-wise addition
    fn simd_add(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise subtraction
    fn simd_sub(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise multiplication
    fn simd_mul(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise division
    fn simd_div(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Dot product
    fn simd_dot(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self;

    /// Matrix-vector multiplication (GEMV)
    fn simd_gemv(a: &ArrayView2<Self>, x: &ArrayView1<Self>, beta: Self, y: &mut Array1<Self>);

    /// Matrix-matrix multiplication (GEMM)
    fn simd_gemm(
        alpha: Self,
        a: &ArrayView2<Self>,
        b: &ArrayView2<Self>,
        beta: Self,
        c: &mut Array2<Self>,
    );

    /// Vector norm (L2)
    fn simd_norm(a: &ArrayView1<Self>) -> Self;

    /// Element-wise maximum
    fn simd_max(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise minimum
    fn simd_min(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Scalar multiplication
    fn simd_scalar_mul(a: &ArrayView1<Self>, scalar: Self) -> Array1<Self>;

    /// Sum reduction
    fn simd_sum(a: &ArrayView1<Self>) -> Self;

    /// Mean reduction
    fn simd_mean(a: &ArrayView1<Self>) -> Self;

    /// Find maximum element
    fn simd_max_element(a: &ArrayView1<Self>) -> Self;

    /// Find minimum element
    fn simd_min_element(a: &ArrayView1<Self>) -> Self;

    /// Fused multiply-add: a * b + c
    fn simd_fma(a: &ArrayView1<Self>, b: &ArrayView1<Self>, c: &ArrayView1<Self>) -> Array1<Self>;

    /// Matrix transpose
    fn simd_transpose(a: &ArrayView2<Self>) -> Array2<Self>;

    /// Element-wise absolute value
    fn simd_abs(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise square root
    fn simd_sqrt(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Check if SIMD is available for this type
    fn simd_available() -> bool;
}

// Implementation for f32
impl SimdUnifiedOps for f32 {
    #[cfg(feature = "simd")]
    fn simd_add(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_add_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_add(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        a + b
    }

    #[cfg(feature = "simd")]
    fn simd_sub(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        let n = a.len();
        let mut result = Array1::zeros(n);
        for i in 0..n {
            result[i] = a[i] - b[i];
        }
        result
    }

    #[cfg(not(feature = "simd"))]
    fn simd_sub(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        a - b
    }

    #[cfg(feature = "simd")]
    fn simd_mul(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_mul_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_mul(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        a * b
    }

    fn simd_div(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        a / b
    }

    #[cfg(feature = "simd")]
    fn simd_dot(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_dot_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_dot(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        a.dot(b)
    }

    fn simd_gemv(a: &ArrayView2<Self>, x: &ArrayView1<Self>, beta: Self, y: &mut Array1<Self>) {
        let m = a.nrows();
        let n = a.ncols();

        assert_eq!(n, x.len());
        assert_eq!(m, y.len());

        // Scale y by beta
        if beta == 0.0 {
            y.fill(0.0);
        } else if beta != 1.0 {
            y.mapv_inplace(|v| v * beta);
        }

        // Compute matrix-vector product
        for i in 0..m {
            let row = a.row(i);
            y[i] += Self::simd_dot(&row, x);
        }
    }

    fn simd_gemm(
        alpha: Self,
        a: &ArrayView2<Self>,
        b: &ArrayView2<Self>,
        beta: Self,
        c: &mut Array2<Self>,
    ) {
        let m = a.nrows();
        let k = a.ncols();
        let n = b.ncols();

        assert_eq!(k, b.nrows());
        assert_eq!((m, n), c.dim());

        // Scale C by beta
        if beta == 0.0 {
            c.fill(0.0);
        } else if beta != 1.0 {
            c.mapv_inplace(|v| v * beta);
        }

        // Compute matrix multiplication
        for i in 0..m {
            let a_row = a.row(i);
            for j in 0..n {
                let b_col = b.column(j);
                c[[i, j]] += alpha * Self::simd_dot(&a_row, &b_col);
            }
        }
    }

    fn simd_norm(a: &ArrayView1<Self>) -> Self {
        Self::simd_dot(a, a).sqrt()
    }

    #[cfg(feature = "simd")]
    fn simd_max(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_maximum_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_max(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for i in 0..a.len() {
            result[i] = a[i].max(b[i]);
        }
        result
    }

    #[cfg(feature = "simd")]
    fn simd_min(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_minimum_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_min(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for i in 0..a.len() {
            result[i] = a[i].min(b[i]);
        }
        result
    }

    #[cfg(feature = "simd")]
    fn simd_scalar_mul(a: &ArrayView1<Self>, scalar: Self) -> Array1<Self> {
        crate::simd::simd_scalar_mul_f32(a, scalar)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_scalar_mul(a: &ArrayView1<Self>, scalar: Self) -> Array1<Self> {
        a.mapv(|x| x * scalar)
    }

    #[cfg(feature = "simd")]
    fn simd_sum(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_sum_f32_enhanced(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_sum(a: &ArrayView1<Self>) -> Self {
        a.sum()
    }

    fn simd_mean(a: &ArrayView1<Self>) -> Self {
        if a.is_empty() {
            0.0
        } else {
            Self::simd_sum(a) / (a.len() as f32)
        }
    }

    fn simd_max_element(a: &ArrayView1<Self>) -> Self {
        a.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x))
    }

    fn simd_min_element(a: &ArrayView1<Self>) -> Self {
        a.fold(f32::INFINITY, |acc, &x| acc.min(x))
    }

    #[cfg(feature = "simd")]
    fn simd_fma(a: &ArrayView1<Self>, b: &ArrayView1<Self>, c: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_fused_multiply_add_f32(a, b, c)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_fma(a: &ArrayView1<Self>, b: &ArrayView1<Self>, c: &ArrayView1<Self>) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for i in 0..a.len() {
            result[i] = a[i] * b[i] + c[i];
        }
        result
    }

    fn simd_transpose(a: &ArrayView2<Self>) -> Array2<Self> {
        a.t().to_owned()
    }

    fn simd_abs(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.abs())
    }

    fn simd_sqrt(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.sqrt())
    }

    #[cfg(feature = "simd")]
    fn simd_available() -> bool {
        true
    }

    #[cfg(not(feature = "simd"))]
    fn simd_available() -> bool {
        false
    }
}

// Implementation for f64
impl SimdUnifiedOps for f64 {
    #[cfg(feature = "simd")]
    fn simd_add(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_add_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_add(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        a + b
    }

    fn simd_sub(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        a - b
    }

    #[cfg(feature = "simd")]
    fn simd_mul(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_mul_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_mul(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        a * b
    }

    fn simd_div(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        a / b
    }

    #[cfg(feature = "simd")]
    fn simd_dot(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_dot_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_dot(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        a.dot(b)
    }

    fn simd_gemv(a: &ArrayView2<Self>, x: &ArrayView1<Self>, beta: Self, y: &mut Array1<Self>) {
        let m = a.nrows();
        let n = a.ncols();

        assert_eq!(n, x.len());
        assert_eq!(m, y.len());

        // Scale y by beta
        if beta == 0.0 {
            y.fill(0.0);
        } else if beta != 1.0 {
            y.mapv_inplace(|v| v * beta);
        }

        // Compute matrix-vector product
        for i in 0..m {
            let row = a.row(i);
            y[i] += Self::simd_dot(&row, x);
        }
    }

    fn simd_gemm(
        alpha: Self,
        a: &ArrayView2<Self>,
        b: &ArrayView2<Self>,
        beta: Self,
        c: &mut Array2<Self>,
    ) {
        let m = a.nrows();
        let k = a.ncols();
        let n = b.ncols();

        assert_eq!(k, b.nrows());
        assert_eq!((m, n), c.dim());

        // Scale C by beta
        if beta == 0.0 {
            c.fill(0.0);
        } else if beta != 1.0 {
            c.mapv_inplace(|v| v * beta);
        }

        // Compute matrix multiplication
        for i in 0..m {
            let a_row = a.row(i);
            for j in 0..n {
                let b_col = b.column(j);
                c[[i, j]] += alpha * Self::simd_dot(&a_row, &b_col);
            }
        }
    }

    fn simd_norm(a: &ArrayView1<Self>) -> Self {
        Self::simd_dot(a, a).sqrt()
    }

    #[cfg(feature = "simd")]
    fn simd_max(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_maximum_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_max(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for i in 0..a.len() {
            result[i] = a[i].max(b[i]);
        }
        result
    }

    #[cfg(feature = "simd")]
    fn simd_min(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_minimum_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_min(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for i in 0..a.len() {
            result[i] = a[i].min(b[i]);
        }
        result
    }

    #[cfg(feature = "simd")]
    fn simd_scalar_mul(a: &ArrayView1<Self>, scalar: Self) -> Array1<Self> {
        crate::simd::simd_scalar_mul_f64(a, scalar)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_scalar_mul(a: &ArrayView1<Self>, scalar: Self) -> Array1<Self> {
        a.mapv(|x| x * scalar)
    }

    fn simd_sum(a: &ArrayView1<Self>) -> Self {
        a.sum()
    }

    fn simd_mean(a: &ArrayView1<Self>) -> Self {
        if a.is_empty() {
            0.0
        } else {
            Self::simd_sum(a) / (a.len() as f64)
        }
    }

    fn simd_max_element(a: &ArrayView1<Self>) -> Self {
        a.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
    }

    fn simd_min_element(a: &ArrayView1<Self>) -> Self {
        a.fold(f64::INFINITY, |acc, &x| acc.min(x))
    }

    fn simd_fma(a: &ArrayView1<Self>, b: &ArrayView1<Self>, c: &ArrayView1<Self>) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for i in 0..a.len() {
            result[i] = a[i] * b[i] + c[i];
        }
        result
    }

    fn simd_transpose(a: &ArrayView2<Self>) -> Array2<Self> {
        a.t().to_owned()
    }

    fn simd_abs(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.abs())
    }

    fn simd_sqrt(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.sqrt())
    }

    #[cfg(feature = "simd")]
    fn simd_available() -> bool {
        true
    }

    #[cfg(not(feature = "simd"))]
    fn simd_available() -> bool {
        false
    }
}

/// Platform capability detection
pub struct PlatformCapabilities {
    pub simd_available: bool,
    pub gpu_available: bool,
    pub cuda_available: bool,
    pub opencl_available: bool,
    pub metal_available: bool,
    pub avx2_available: bool,
    pub avx512_available: bool,
    pub neon_available: bool,
}

impl PlatformCapabilities {
    /// Detect current platform capabilities
    pub fn detect() -> Self {
        Self {
            simd_available: cfg!(feature = "simd"),
            gpu_available: cfg!(feature = "gpu"),
            cuda_available: cfg!(all(feature = "gpu", feature = "cuda")),
            opencl_available: cfg!(all(feature = "gpu", feature = "opencl")),
            metal_available: cfg!(all(feature = "gpu", feature = "metal", target_os = "macos")),
            avx2_available: cfg!(target_feature = "avx2"),
            avx512_available: cfg!(target_feature = "avx512f"),
            neon_available: cfg!(target_arch = "aarch64"),
        }
    }

    /// Get a summary of available acceleration features
    pub fn summary(&self) -> String {
        let mut features = Vec::new();

        if self.simd_available {
            features.push("SIMD");
        }
        if self.gpu_available {
            features.push("GPU");
        }
        if self.cuda_available {
            features.push("CUDA");
        }
        if self.opencl_available {
            features.push("OpenCL");
        }
        if self.metal_available {
            features.push("Metal");
        }
        if self.avx2_available {
            features.push("AVX2");
        }
        if self.avx512_available {
            features.push("AVX512");
        }
        if self.neon_available {
            features.push("NEON");
        }

        if features.is_empty() {
            "No acceleration features available".to_string()
        } else {
            format!("Available acceleration: {}", features.join(", "))
        }
    }
}

/// Automatic operation selection based on problem size and available features
pub struct AutoOptimizer {
    capabilities: PlatformCapabilities,
}

impl AutoOptimizer {
    pub fn new() -> Self {
        Self {
            capabilities: PlatformCapabilities::detect(),
        }
    }

    /// Determine if GPU should be used for a given problem size
    pub fn should_use_gpu(&self, size: usize) -> bool {
        // Use GPU for large problems when available
        self.capabilities.gpu_available && size > 10000
    }

    /// Determine if Metal should be used on macOS
    pub fn should_use_metal(&self, size: usize) -> bool {
        // Use Metal for medium to large problems on macOS
        // Metal has lower overhead than CUDA/OpenCL, so we can use it for smaller problems
        self.capabilities.metal_available && size > 1024
    }

    /// Determine if SIMD should be used
    pub fn should_use_simd(&self, size: usize) -> bool {
        // Use SIMD for medium to large problems
        self.capabilities.simd_available && size > 64
    }

    /// Select the best implementation for matrix multiplication
    pub fn select_gemm_impl(&self, m: usize, n: usize, k: usize) -> &'static str {
        let total_ops = m * n * k;

        // Metal-specific heuristics for macOS
        if self.capabilities.metal_available {
            // For Apple Silicon with unified memory, Metal is efficient even for smaller matrices
            if total_ops > 8192 {
                // 16x16x32 or larger
                return "Metal";
            }
        }

        if self.should_use_gpu(total_ops) {
            if self.capabilities.cuda_available {
                "CUDA"
            } else if self.capabilities.metal_available {
                "Metal"
            } else if self.capabilities.opencl_available {
                "OpenCL"
            } else {
                "GPU"
            }
        } else if self.should_use_simd(total_ops) {
            "SIMD"
        } else {
            "Scalar"
        }
    }

    /// Select the best implementation for vector operations
    pub fn select_vector_impl(&self, size: usize) -> &'static str {
        // Metal is efficient for vector operations on Apple Silicon
        if self.capabilities.metal_available && size > 1024 {
            return "Metal";
        }

        if self.should_use_gpu(size) {
            if self.capabilities.cuda_available {
                "CUDA"
            } else if self.capabilities.metal_available {
                "Metal"
            } else if self.capabilities.opencl_available {
                "OpenCL"
            } else {
                "GPU"
            }
        } else if self.should_use_simd(size) {
            if self.capabilities.avx512_available {
                "AVX512"
            } else if self.capabilities.avx2_available {
                "AVX2"
            } else if self.capabilities.neon_available {
                "NEON"
            } else {
                "SIMD"
            }
        } else {
            "Scalar"
        }
    }

    /// Select the best implementation for reduction operations
    pub fn select_reduction_impl(&self, size: usize) -> &'static str {
        // Reductions benefit from GPU parallelism at larger sizes
        // Metal has efficient reduction primitives
        if self.capabilities.metal_available && size > 4096 {
            return "Metal";
        }

        if self.should_use_gpu(size * 2) {
            // Higher threshold for reductions
            if self.capabilities.cuda_available {
                "CUDA"
            } else if self.capabilities.metal_available {
                "Metal"
            } else {
                "GPU"
            }
        } else if self.should_use_simd(size) {
            "SIMD"
        } else {
            "Scalar"
        }
    }

    /// Select the best implementation for FFT operations
    pub fn select_fft_impl(&self, size: usize) -> &'static str {
        // FFT benefits greatly from GPU acceleration
        // Metal Performance Shaders has optimized FFT
        if self.capabilities.metal_available && size > 512 {
            return "Metal-MPS";
        }

        if self.capabilities.cuda_available && size > 1024 {
            "cuFFT"
        } else if self.should_use_simd(size) {
            "SIMD"
        } else {
            "Scalar"
        }
    }

    /// Check if running on Apple Silicon with unified memory
    pub fn has_unified_memory(&self) -> bool {
        cfg!(all(target_os = "macos", target_arch = "aarch64"))
    }

    /// Get optimization recommendation for a specific operation
    pub fn recommend(&self, operation: &str, size: usize) -> String {
        let recommendation = match operation {
            "gemm" | "matmul" => self.select_gemm_impl(size, size, size),
            "vector" | "axpy" | "dot" => self.select_vector_impl(size),
            "reduction" | "sum" | "mean" => self.select_reduction_impl(size),
            "fft" => self.select_fft_impl(size),
            _ => "Scalar",
        };

        if self.has_unified_memory() && recommendation == "Metal" {
            format!("{} (Unified Memory)", recommendation)
        } else {
            recommendation.to_string()
        }
    }
}

impl Default for AutoOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_simd_unified_ops_f32() {
        let a = arr1(&[1.0f32, 2.0, 3.0, 4.0]);
        let b = arr1(&[5.0f32, 6.0, 7.0, 8.0]);

        let sum = f32::simd_add(&a.view(), &b.view());
        assert_eq!(sum, arr1(&[6.0f32, 8.0, 10.0, 12.0]));

        let product = f32::simd_mul(&a.view(), &b.view());
        assert_eq!(product, arr1(&[5.0f32, 12.0, 21.0, 32.0]));

        let dot = f32::simd_dot(&a.view(), &b.view());
        assert_eq!(dot, 70.0);
    }

    #[test]
    fn test_platform_capabilities() {
        let caps = PlatformCapabilities::detect();
        println!("{}", caps.summary());
    }

    #[test]
    fn test_auto_optimizer() {
        let optimizer = AutoOptimizer::new();

        // Small problem - should use scalar
        assert!(!optimizer.should_use_gpu(100));

        // Large problem - depends on GPU availability
        let large_size = 100000;
        if optimizer.capabilities.gpu_available {
            assert!(optimizer.should_use_gpu(large_size));
        }
    }
}
