//! Advanced SIMD-optimized signal processing operations
//!
//! This module provides highly optimized SIMD implementations of common
//! signal processing operations that go beyond the basic operations in
//! scirs2-core, specifically targeting signal processing workloads.
//!
//! # Features
//!
//! ## Multi-platform SIMD Optimization
//! - **x86_64**: AVX512, AVX2, SSE4.1 instruction sets
//! - **AArch64**: NEON instruction sets
//! - **Automatic fallback**: Scalar implementations for unsupported platforms
//! - **Runtime detection**: Automatically selects best available instruction set
//!
//! ## Memory Optimization
//! - **Alignment detection**: Optimizes for cache line boundaries
//! - **Cache-friendly patterns**: Minimizes cache misses through data layout optimization
//! - **Prefetching**: Strategic memory prefetching for large datasets
//! - **Streaming**: Memory-efficient processing for datasets larger than cache
//!
//! ## Performance Features
//! - **Adaptive thresholds**: Automatically adjusts SIMD thresholds based on performance
//! - **Performance monitoring**: Tracks throughput, latency, and efficiency metrics
//! - **Batch processing**: Optimized algorithms for processing multiple signals simultaneously
//! - **Loop unrolling**: Configurable unroll factors for optimal performance
//!
//! ## Supported Operations
//!
//! ### Filtering Operations
//! - FIR filtering with arbitrary kernel lengths
//! - Enhanced convolution with memory optimization
//! - Real-time filtering with minimal latency
//!
//! ### Correlation Operations
//! - Auto-correlation with FFT-based acceleration
//! - Cross-correlation for signal alignment
//! - Matrix-vector operations for filter banks
//!
//! ### Spectral Analysis
//! - Batch spectral analysis for multiple signals
//! - Power spectrum computation with phase information
//! - Statistical analysis across signal batches
//! - SNR estimation and spectral centroid calculation
//! - Spectral centroid computation using SIMD acceleration
//! - Spectral rolloff analysis with configurable thresholds
//! - Complex number multiplication for frequency domain operations
//! - Weighted averaging for multitaper spectral estimation
//!
//! ### Platform-Specific Operations
//! - **SSE 4.1**: Low-level optimizations for older x86_64 processors
//! - **AVX2**: High-performance implementations for modern x86_64 processors
//! - **Peak detection**: Vectorized local maxima finding with configurable thresholds
//! - **Zero crossings**: Efficient sign change detection for audio analysis
//! - **Window functions**: Hardware-accelerated window application for spectral analysis
//!
//! # Usage Examples
//!
//! ## Basic FIR Filtering
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use scirs2_signal::simd_advanced::{SimdConfig, simd_fir_filter};
//!
//! let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let coeffs = vec![0.5, 0.3, 0.2];
//! let mut output = vec![0.0; input.len()];
//! let config = SimdConfig::default();
//!
//! simd_fir_filter(&input, &coeffs, &mut output, &config)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Batch Spectral Analysis
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use scirs2_signal::simd_advanced::{SimdConfig, simd_batch_spectral_analysis};
//! use scirs2_core::ndarray::Array2;
//!
//! let signals = Array2::zeros((4, 1024)); // 4 signals, 1024 samples each
//! let config = SimdConfig::default();
//!
//! let result = simd_batch_spectral_analysis(&signals, "hann", 1024, &config)?;
//! println!("Power spectra shape: {:?}", result.power_spectra.shape());
//! # Ok(())
//! # }
//! ```
//!
//! ## Spectral Centroid and Rolloff Analysis
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use scirs2_signal::simd_advanced::{SimdConfig, simd_spectral_centroid, simd_spectral_rolloff};
//!
//! let magnitude_spectrum = vec![1.0, 2.0, 3.0, 2.0, 1.0];
//! let frequencies = vec![0.0, 1000.0, 2000.0, 3000.0, 4000.0];
//! let config = SimdConfig::default();
//!
//! // Compute spectral centroid (weighted frequency center)
//! let centroid = simd_spectral_centroid(&magnitude_spectrum, &frequencies, &config)?;
//! println!("Spectral centroid: {:.1} Hz", centroid);
//!
//! // Compute spectral rolloff (85% energy point)
//! let rolloff = simd_spectral_rolloff(&magnitude_spectrum, &frequencies, 0.85, &config)?;
//! println!("Spectral rolloff: {:.1} Hz", rolloff);
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance Monitoring
//! ```rust
//! use scirs2_signal::simd_advanced::SimdConfig;
//!
//! let mut config = SimdConfig::default();
//! config.enable_monitoring = true;
//! config.adaptive_thresholds = true;
//!
//! // Operations will now collect performance metrics
//! ```
//!
//! # Configuration
//!
//! The [`SimdConfig`] structure provides fine-grained control over SIMD behavior:
//!
//! - **force_scalar**: Disable SIMD for testing and comparison
//! - **simd_threshold**: Minimum input size to trigger SIMD optimization
//! - **align_memory**: Enable memory alignment optimization
//! - **use_advanced**: Enable advanced instruction sets (AVX512, etc.)
//! - **enable_monitoring**: Collect performance metrics during execution
//! - **adaptive_thresholds**: Automatically adjust thresholds based on performance
//! - **cache_line_size**: Cache line size for memory optimization (typically 64 bytes)
//! - **max_unroll_factor**: Maximum loop unroll factor for optimization
//!
//! # Performance Considerations
//!
//! ## When to Use SIMD
//! - **Large datasets**: SIMD shows benefits primarily for datasets larger than the threshold
//! - **Repeated operations**: SIMD optimization pays off when processing many similar signals
//! - **Memory-bound operations**: SIMD can improve memory throughput for bandwidth-limited operations
//!
//! ## Platform-Specific Notes
//! - **x86_64**: Best performance with AVX2 or AVX512 support
//! - **AArch64**: NEON provides good performance for most operations
//! - **Scalar fallback**: Always available but may be significantly slower for large datasets
//!
//! # Validation and Testing
//!
//! All SIMD implementations include comprehensive validation against scalar reference
//! implementations to ensure numerical accuracy. Use [`SimdValidationResult`] to
//! verify SIMD correctness and performance characteristics.

pub mod basic_ops;
pub mod benchmarks;
pub mod convolution;
pub mod peak_detection;
pub mod platform_ops;
pub mod spectral_ops;
pub mod types;
pub mod validation;

// Re-export main types for easier access
pub use types::{
    BatchSpectralResult, BatchSpectralStats, SimdConfig, SimdPerformanceMetrics,
    SimdValidationResult,
};

// Re-export basic SIMD operations for backward compatibility
pub use basic_ops::{
    scalar_autocorrelation, scalar_cross_correlation, simd_apply_window, simd_autocorrelation,
    simd_complex_fft_butterfly, simd_cross_correlation, simd_fir_filter, simd_rms,
    simd_signal_energy, simd_zero_crossing_rate,
};

// Re-export spectral analysis SIMD operations for backward compatibility
pub use spectral_ops::{
    simd_batch_spectral_analysis, simd_complex_multiply, simd_power_spectrum,
    simd_spectral_centroid, simd_spectral_rolloff, simd_weighted_average_spectra,
};

// Re-export platform-specific SIMD operations for advanced users
// Note: These are unsafe functions that require specific CPU instruction set support
#[cfg(target_arch = "x86_64")]
pub use platform_ops::{
    avx2_apply_window,
    avx2_apply_window_v2,
    avx2_autocorrelation,
    avx2_complex_butterfly,
    avx2_complex_multiply,
    avx2_cross_correlation,
    // AVX2 implementations
    avx2_enhanced_convolution,
    avx2_fir_filter,
    avx2_peak_detection,
    avx2_power_spectrum,
    avx2_weighted_average_spectra,
    avx2_zero_crossings,
    sse_apply_window,
    sse_apply_window_v2,
    sse_autocorrelation,
    sse_complex_butterfly,
    sse_complex_multiply,
    sse_cross_correlation,
    // SSE 4.1 implementations
    sse_fir_filter,
    sse_power_spectrum,
    sse_weighted_average_spectra,
};

// Re-export benchmark functions
pub use benchmarks::benchmark_simd_operations;

// Re-export validation functions
pub use validation::comprehensive_simd_validation;

// Re-export peak detection functions
pub use peak_detection::simd_peak_detection;

// Re-export convolution functions
pub use convolution::simd_enhanced_convolution;

// Note: SingleSpectralResult is kept private (pub(crate)) as it's an internal type
// and will be re-exported through the parent implementation modules when they are created
