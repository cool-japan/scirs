//! Type definitions for 2D Discrete Wavelet Transform operations.
//!
//! This module contains all the data structures used throughout the 2D DWT implementation,
//! including configuration types, result containers, validation structures, and metrics.

use crate::dwt::Wavelet;
use ndarray::Array2;
use std::collections::HashMap;

/// Memory-optimized configuration for 2D DWT operations
#[derive(Debug, Clone)]
pub struct Dwt2dConfig {
    /// Enable memory pre-allocation for better cache efficiency
    pub preallocate_memory: bool,
    /// Use in-place operations when possible
    pub use_inplace: bool,
    /// Memory alignment for SIMD operations (must be power of 2)
    pub memory_alignment: usize,
    /// Chunk size for large arrays to improve cache locality
    pub chunk_size: Option<usize>,
}

impl Default for Dwt2dConfig {
    fn default() -> Self {
        Self {
            preallocate_memory: true,
            use_inplace: false,            // Conservative default for safety
            memory_alignment: 32,          // AVX2 alignment
            chunk_size: Some(1024 * 1024), // 1MB chunks by default
        }
    }
}

/// Memory pool for efficient allocation/deallocation of temporary arrays
pub struct MemoryPool {
    pools: HashMap<usize, Vec<Vec<f64>>>,
    max_pool_size: usize,
}

impl MemoryPool {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            max_pool_size: 10, // Maximum number of arrays per size
        }
    }

    pub fn get_buffer(&mut self, size: usize) -> Vec<f64> {
        if let Some(pool) = self.pools.get_mut(&size) {
            if let Some(mut buffer) = pool.pop() {
                buffer.clear();
                buffer.resize(size, 0.0);
                return buffer;
            }
        }
        vec![0.0; size]
    }

    pub fn return_buffer(&mut self, buffer: Vec<f64>) {
        let size = buffer.capacity();
        let pool = self.pools.entry(size).or_insert_with(Vec::new);
        if pool.len() < self.max_pool_size {
            pool.push(buffer);
        }
    }
}

/// Result of a 2D DWT decomposition, containing the approximation and detail coefficients.
///
/// The 2D DWT decomposes an image into four subbands, each representing different
/// frequency components in horizontal and vertical directions. These subbands are
/// represented as separate 2D arrays (matrices) in this struct.
///
/// The coefficients represent the following subbands:
/// - `approx`: Low-frequency approximation coefficients (LL) - Represents the coarse, low-resolution version of the image
/// - `detail_h`: Horizontal detail coefficients (LH) - Captures horizontal edges (high frequency in horizontal direction)
/// - `detail_v`: Vertical detail coefficients (HL) - Captures vertical edges (high frequency in vertical direction)
/// - `detail_d`: Diagonal detail coefficients (HH) - Captures diagonal details (high frequency in both directions)
///
/// These four subbands together contain all the information needed to reconstruct
/// the original image. For multi-level decomposition, the approximation coefficients
/// are recursively decomposed into further subbands.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{dwt2d_decompose, Dwt2dResult};
///
/// // Create a simple 4x4 image with a gradient pattern
/// let mut image = Array2::zeros((4, 4));
/// for i in 0..4 {
///     for j in 0..4 {
///         image[[i, j]] = (i * j)  as f64;
///     }
/// }
///
/// // Decompose the image
/// let result: Dwt2dResult = dwt2d_decompose(&image, Wavelet::Haar, None).unwrap();
///
/// // The image is now decomposed into four subbands:
/// let ll = &result.approx;  // Approximation coefficients (low-resolution image)
/// let lh = &result.detail_h;  // Horizontal details
/// let hl = &result.detail_v;  // Vertical details
/// let hh = &result.detail_d;  // Diagonal details
///
/// // All subbands have the same shape (half the size in each dimension)
/// assert_eq!(ll.shape(), &[2, 2]);
/// assert_eq!(lh.shape(), &[2, 2]);
/// assert_eq!(hl.shape(), &[2, 2]);
/// assert_eq!(hh.shape(), &[2, 2]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Dwt2dResult {
    /// Approximation coefficients (LL subband)
    pub approx: Array2<f64>,
    /// Horizontal detail coefficients (LH subband)
    pub detail_h: Array2<f64>,
    /// Vertical detail coefficients (HL subband)
    pub detail_v: Array2<f64>,
    /// Diagonal detail coefficients (HH subband)
    pub detail_d: Array2<f64>,
}

/// Threshold method to apply to wavelet coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThresholdMethod {
    /// Hard thresholding: sets coefficients below threshold to zero, leaves others unchanged
    Hard,
    /// Soft thresholding: sets coefficients below threshold to zero, shrinks others toward zero by threshold amount
    Soft,
    /// Garrote thresholding: a non-linear thresholding approach with properties between hard and soft
    Garrote,
}

/// Structure containing energy values for each wavelet subband.
#[derive(Debug, Clone, Copy)]
pub struct WaveletEnergy {
    /// Energy in approximation coefficients (LL band)
    pub approx: f64,
    /// Energy in horizontal detail coefficients (LH band)
    pub detail_h: f64,
    /// Energy in vertical detail coefficients (HL band)
    pub detail_v: f64,
    /// Energy in diagonal detail coefficients (HH band)
    pub detail_d: f64,
}

/// Structure containing non-zero coefficient counts for each wavelet subband.
#[derive(Debug, Clone, Copy)]
pub struct WaveletCounts {
    /// Count in approximation coefficients (LL band)
    pub approx: usize,
    /// Count in horizontal detail coefficients (LH band)
    pub detail_h: usize,
    /// Count in vertical detail coefficients (HL band)
    pub detail_v: usize,
    /// Count in diagonal detail coefficients (HH band)
    pub detail_d: usize,
}

/// Enhanced validation metrics for 2D wavelet transforms
#[derive(Debug, Clone)]
pub struct Dwt2dValidationResult {
    /// Reconstruction error (RMSE)
    pub reconstruction_error: f64,
    /// Energy conservation error
    pub energy_conservation_error: f64,
    /// Orthogonality preservation
    pub orthogonality_error: f64,
    /// Memory efficiency metrics
    pub memory_efficiency: MemoryEfficiencyMetrics,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics2d,
    /// Overall validation score (0-100)
    pub overall_score: f64,
    /// Issues found during validation
    pub issues: Vec<String>,
}

/// Memory efficiency metrics for 2D DWT operations
#[derive(Debug, Clone)]
pub struct MemoryEfficiencyMetrics {
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Memory allocation count
    pub allocation_count: usize,
    /// Cache miss ratio (estimated)
    pub cache_miss_ratio: f64,
    /// Memory access pattern efficiency
    pub access_pattern_efficiency: f64,
}

/// Performance metrics for 2D DWT operations
#[derive(Debug, Clone)]
pub struct PerformanceMetrics2d {
    /// Total computation time (ms)
    pub total_time_ms: f64,
    /// Decomposition time (ms)
    pub decomposition_time_ms: f64,
    /// Reconstruction time (ms)
    pub reconstruction_time_ms: f64,
    /// SIMD utilization percentage
    pub simd_utilization: f64,
    /// Parallel efficiency
    pub parallel_efficiency: f64,
    /// Throughput (MB/s)
    pub throughput_mbs: f64,
}

/// Advanced configuration for 2D DWT validation
#[derive(Debug, Clone)]
pub struct Dwt2dValidationConfig {
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
    /// Test various image sizes
    pub test_sizes: Vec<(usize, usize)>,
    /// Test different wavelets
    pub test_wavelets: Vec<Wavelet>,
    /// Enable performance benchmarking
    pub benchmark_performance: bool,
    /// Test memory efficiency
    pub test_memory_efficiency: bool,
    /// Test numerical stability
    pub test_numerical_stability: bool,
    /// Test edge cases
    pub test_edge_cases: bool,
}

impl Default for Dwt2dValidationConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            test_sizes: vec![(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)],
            test_wavelets: vec![Wavelet::Haar, Wavelet::DB(2), Wavelet::DB(4)],
            benchmark_performance: true,
            test_memory_efficiency: true,
            test_numerical_stability: true,
            test_edge_cases: true,
        }
    }
}