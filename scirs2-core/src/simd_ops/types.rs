//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Number of f64 elements that fit in a typical L1 cache (32 KiB / 8 bytes).
pub const SIMD_BATCH_L1_F64: usize = 4_096;

/// Number of f64 elements that fit in a typical L2 cache (512 KiB / 8 bytes).
pub const SIMD_BATCH_L2_F64: usize = 65_536;

/// Number of f64 elements that fit in a typical L3 cache (8 MiB / 8 bytes).
pub const SIMD_BATCH_L3_F64: usize = 1_048_576;

/// Platform capability detection
#[derive(Debug, Clone, Copy)]
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
            cuda_available: false, // CUDA retired from scirs2-core in 0.6.x (moved to oxicuda-*)
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
            format!(
                "Available acceleration: {features}",
                features = features.join(", ")
            )
        }
    }
    /// Check if AVX2 is available
    pub fn has_avx2(&self) -> bool {
        self.avx2_available
    }
    /// Check if AVX512 is available
    pub fn has_avx512(&self) -> bool {
        self.avx512_available
    }
    /// Check if SSE is available (fallback to SIMD availability)
    pub fn has_sse(&self) -> bool {
        self.simd_available || self.neon_available || self.avx2_available
    }
    /// Get the number of CPU cores
    pub fn num_cores(&self) -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
    /// Get the cache line size in bytes
    pub fn cache_line_size(&self) -> usize {
        64
    }
}
/// Automatic operation selection based on problem size and available features
pub struct AutoOptimizer {
    pub(super) capabilities: PlatformCapabilities,
}
impl AutoOptimizer {
    pub fn new() -> Self {
        Self {
            capabilities: PlatformCapabilities::detect(),
        }
    }
    /// Determine if GPU should be used for a given problem size
    pub fn should_use_gpu(&self, size: usize) -> bool {
        self.capabilities.gpu_available && size > 10000
    }
    /// Determine if Metal should be used on macOS
    pub fn should_use_metal(&self, size: usize) -> bool {
        self.capabilities.metal_available && size > 1024
    }
    /// Determine if SIMD should be used
    pub fn should_use_simd(&self, size: usize) -> bool {
        self.capabilities.simd_available && size > 64
    }
    /// Select the best implementation for matrix multiplication
    pub fn select_gemm_impl(&self, m: usize, n: usize, k: usize) -> &'static str {
        let total_ops = m * n * k;
        if self.capabilities.metal_available && total_ops > 8192 {
            return "Metal";
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
        if self.capabilities.metal_available && size > 4096 {
            return "Metal";
        }
        if self.should_use_gpu(size * 2) {
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
            format!("{recommendation} (Unified Memory)")
        } else {
            recommendation.to_string()
        }
    }
}
