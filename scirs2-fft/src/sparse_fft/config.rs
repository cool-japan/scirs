//! Configuration types and utilities for Sparse FFT algorithms
//!
//! This module contains the configuration structures, enums, and utility functions
//! used to configure and control sparse FFT computations.

use scirs2_core::numeric::Complex64;
use std::fmt::Debug;

/// Helper function to extract complex values from various types (for doctests)
#[allow(dead_code)]
pub fn try_as_complex<T: 'static + Copy>(val: T) -> Option<Complex64> {
    use std::any::Any;

    // Try to use runtime type checking with Any for complex types
    if let Some(complex) = (&val as &dyn Any).downcast_ref::<Complex64>() {
        return Some(*complex);
    }

    // Try to handle f32 complex numbers
    if let Some(complex32) = (&val as &dyn Any).downcast_ref::<scirs2_core::numeric::Complex<f32>>()
    {
        return Some(Complex64::new(complex32.re as f64, complex32.im as f64));
    }

    None
}

/// Sparsity estimation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparsityEstimationMethod {
    /// Manual estimation (user provides the sparsity)
    Manual,
    /// Automatic estimation based on thresholding
    Threshold,
    /// Adaptive estimation based on signal properties
    Adaptive,
    /// Frequency domain pruning for high accuracy estimation
    FrequencyPruning,
    /// Spectral flatness measure for noise vs signal discrimination
    SpectralFlatness,
}

/// Sparse FFT algorithm variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFFTAlgorithm {
    /// Sublinear Sparse FFT
    Sublinear,
    /// Compressed Sensing-based Sparse FFT
    CompressedSensing,
    /// Iterative Method for Sparse FFT
    Iterative,
    /// Deterministic Sparse FFT
    Deterministic,
    /// Frequency-domain pruning approach
    FrequencyPruning,
    /// Advanced pruning using spectral flatness measure
    SpectralFlatness,
}

/// Window function to apply before FFT
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    /// No windowing (rectangular window)
    None,
    /// Hann window (reduces spectral leakage)
    Hann,
    /// Hamming window (good for speech)
    Hamming,
    /// Blackman window (excellent sidelobe suppression)
    Blackman,
    /// Flat top window (best amplitude accuracy)
    FlatTop,
    /// Kaiser window with adjustable parameter
    Kaiser,
}

/// Sparse FFT configuration
#[derive(Debug, Clone)]
pub struct SparseFFTConfig {
    /// The sparsity estimation method
    pub estimation_method: SparsityEstimationMethod,
    /// Expected sparsity (k) - number of significant frequency components
    pub sparsity: usize,
    /// Algorithm variant to use
    pub algorithm: SparseFFTAlgorithm,
    /// Threshold for frequency coefficient significance (when using threshold method)
    pub threshold: f64,
    /// Number of iterations for iterative methods
    pub iterations: usize,
    /// Random seed for probabilistic algorithms
    pub seed: Option<u64>,
    /// Maximum signal size to process (to prevent test timeouts)
    pub max_signal_size: usize,
    /// Adaptivity parameter (controls how aggressive adaptivity is)
    pub adaptivity_factor: f64,
    /// Pruning parameter (controls sensitivity of frequency pruning)
    pub pruning_sensitivity: f64,
    /// Spectral flatness threshold (0-1, lower values = more selective)
    pub flatness_threshold: f64,
    /// Analysis window size for spectral flatness calculations
    pub window_size: usize,
    /// Window function to apply before FFT
    pub window_function: WindowFunction,
    /// Kaiser window beta parameter (when using Kaiser window)
    pub kaiser_beta: f64,
}

impl Default for SparseFFTConfig {
    fn default() -> Self {
        Self {
            estimation_method: SparsityEstimationMethod::Threshold,
            sparsity: 10,
            algorithm: SparseFFTAlgorithm::Sublinear,
            threshold: 0.01,
            iterations: 3,
            seed: None,
            max_signal_size: 1024, // Default max size to avoid test timeouts
            adaptivity_factor: 0.25,
            pruning_sensitivity: 0.05,
            flatness_threshold: 0.3,
            window_size: 16,
            window_function: WindowFunction::None,
            kaiser_beta: 14.0, // Default beta for Kaiser window
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SparseFFTConfig::default();
        assert_eq!(config.sparsity, 10);
        assert_eq!(config.threshold, 0.01);
        assert_eq!(config.max_signal_size, 1024);
    }

    #[test]
    fn test_try_as_complex() {
        let val = Complex64::new(1.0, 2.0);
        assert_eq!(try_as_complex(val), Some(val));

        let val32 = scirs2_core::numeric::Complex::new(1.0f32, 2.0f32);
        assert_eq!(try_as_complex(val32), Some(Complex64::new(1.0, 2.0)));
    }
}
