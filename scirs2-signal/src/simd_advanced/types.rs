//! Type definitions for SIMD-optimized signal processing operations
//!
//! This module contains all the configuration structures, result types, and
//! validation structures used throughout the SIMD-optimized signal processing
//! operations.

use scirs2_core::ndarray::Array2;

/// Configuration for SIMD operations
///
/// This structure controls various aspects of SIMD optimization including
/// memory alignment, instruction set usage, and performance monitoring.
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Force scalar fallback (for testing)
    pub force_scalar: bool,
    /// Minimum length for SIMD optimization
    pub simd_threshold: usize,
    /// Cache line alignment
    pub align_memory: bool,
    /// Use advanced instruction sets (AVX512, etc.)
    pub use_advanced: bool,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Adaptive threshold adjustment based on performance
    pub adaptive_thresholds: bool,
    /// Cache line size for optimization (typically 64 bytes)
    pub cache_line_size: usize,
    /// Maximum unroll factor for loops
    pub max_unroll_factor: usize,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            force_scalar: false,
            simd_threshold: 64,
            align_memory: true,
            use_advanced: false, // Disable until AVX-512 features are stabilized
            enable_monitoring: false,
            adaptive_thresholds: false,
            cache_line_size: 64,
            max_unroll_factor: 8,
        }
    }
}

/// Result of batch spectral analysis
///
/// Contains power spectra, phase information, and statistical analysis
/// for multiple signals processed simultaneously.
#[derive(Debug, Clone)]
pub struct BatchSpectralResult {
    /// Power spectra for all signals (n_signals x n_frequencies)
    pub power_spectra: Array2<f64>,
    /// Phase information for all signals (n_signals x n_frequencies)
    pub phases: Array2<f64>,
    /// Batch statistics
    pub statistics: BatchSpectralStats,
    /// Frequency bins (normalized)
    pub frequencies: Vec<f64>,
}

/// Batch spectral statistics
///
/// Statistical measures computed across all signals in a batch processing
/// operation, providing insights into signal characteristics.
#[derive(Debug, Clone)]
pub struct BatchSpectralStats {
    /// Mean power across all signals
    pub mean_power: Vec<f64>,
    /// Maximum power across all signals
    pub max_power: Vec<f64>,
    /// SNR estimates for each signal
    pub snr_estimates: Vec<f64>,
    /// Spectral centroids for each signal
    pub spectral_centroids: Vec<f64>,
}

/// Single signal spectral result
///
/// Contains spectral analysis results for a single signal including
/// power spectrum, phase information, and derived metrics.
#[derive(Debug, Clone)]
pub(crate) struct SingleSpectralResult {
    pub(crate) power_spectrum: Vec<f64>,
    pub(crate) phase: Vec<f64>,
    pub(crate) snr_estimate: f64,
    pub(crate) spectral_centroid: f64,
}

/// Performance monitoring structure for SIMD operations
///
/// Tracks various performance metrics during SIMD operation execution
/// including timing, throughput, and instruction set usage.
#[derive(Debug, Clone)]
pub struct SimdPerformanceMetrics {
    /// Operation name
    pub operation: String,
    /// Input size
    pub input_size: usize,
    /// Time taken in nanoseconds
    pub time_ns: u64,
    /// SIMD instruction set used
    pub instruction_set: String,
    /// Memory throughput (bytes/second)
    pub memory_throughput: f64,
    /// Computational throughput (operations/second)
    pub compute_throughput: f64,
}

/// SIMD validation result
///
/// Comprehensive validation results for SIMD operations including
/// timing measurements, error analysis, and consistency checks.
#[derive(Debug, Clone, Default)]
pub struct SimdValidationResult {
    /// FIR filter execution time in nanoseconds
    pub fir_filter_time_ns: u64,
    /// Autocorrelation execution time in nanoseconds
    pub autocorrelation_time_ns: u64,
    /// Cross-correlation execution time in nanoseconds
    pub cross_correlation_time_ns: u64,
    /// Matrix-vector multiplication time in nanoseconds
    pub matrix_vector_time_ns: u64,
    /// Maximum error between SIMD and scalar implementations
    pub simd_scalar_max_error: f64,
    /// Whether SIMD results are consistent with scalar results
    pub simd_consistency: bool,
    /// Total operations performed per second
    pub operations_per_second: f64,
    /// Memory throughput in bytes per second
    pub memory_throughput_bytes_per_sec: f64,
    /// Total validation execution time in milliseconds
    pub total_validation_time_ms: f64,
    /// Overall validation pass/fail status
    pub validation_passed: bool,
}
