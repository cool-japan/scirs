//! Type definitions for 2D Discrete Wavelet Transform
//!
//! This module contains the core data structures used throughout the DWT2D implementation:
//! - Configuration structures for memory optimization and algorithm parameters
//! - Result structures for decomposition output
//! - Enumerations for threshold methods
//! - Memory pool for efficient temporary buffer management

use scirs2_core::ndarray::Array2;
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

    pub fn return_buffer(&mut self, mut buffer: Vec<f64>) {
        let size = buffer.capacity();
        if let Some(pool) = self.pools.get_mut(&size) {
            if pool.len() < self.max_pool_size {
                buffer.clear();
                pool.push(buffer);
            }
        } else {
            let mut new_pool = Vec::new();
            buffer.clear();
            new_pool.push(buffer);
            self.pools.insert(size, new_pool);
        }
    }

    pub fn clear(&mut self) {
        self.pools.clear();
    }

    pub fn get_pool_statistics(&self) -> HashMap<usize, usize> {
        self.pools.iter().map(|(&size, pool)| (size, pool.len())).collect()
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a 2D discrete wavelet transform
///
/// Contains the four subbands produced by the 2D DWT:
/// - LL (approximation): Low-frequency content in both dimensions
/// - LH (horizontal detail): High-frequency horizontal, low-frequency vertical
/// - HL (vertical detail): Low-frequency horizontal, high-frequency vertical
/// - HH (diagonal detail): High-frequency content in both dimensions
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::dwt2d_decompose;
///
/// let data = Array2::from_shape_fn((4, 4), |(i, j)| (i + j) as f64);
/// let result = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();
///
/// let ll = &result.approx;    // Approximation coefficients
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