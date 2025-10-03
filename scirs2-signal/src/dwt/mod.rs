// Discrete Wavelet Transform (DWT)
//
// This module provides implementations of the Discrete Wavelet Transform (DWT),
// inverse DWT, and associated wavelet filters. The DWT is useful for
// multi-resolution analysis, denoising, and compression of signals.
//
// The module is organized into submodules:
// - `filters`: Wavelet filter definitions and generation functions
// - `transform`: Core DWT decomposition and reconstruction functions
// - `boundary`: Signal extension methods for handling boundary conditions
// - `multiscale`: Multi-level transform functions for decomposition and reconstruction

use scirs2_core::ndarray::Array1;

#[allow(unused_imports)]
// Declare submodules
mod boundary;
mod filters;
mod multiscale;
mod transform;

// Re-export public items from submodules
pub use filters::{Wavelet, WaveletFilters};
pub use multiscale::{wavedec, wavedec_compat, waverec, waverec_compat};
pub use transform::{dwt_decompose, dwt_reconstruct};

// Export types - DecompositionResult is defined below

// Re-export boundary extension for advanced users
pub use boundary::extend_signal;

/// Result of multi-level wavelet decomposition
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// Approximation coefficients at the coarsest level
    pub approx: Array1<f64>,
    /// Detail coefficients at each level (from coarsest to finest)
    pub details: Vec<Array1<f64>>,
}

impl DecompositionResult {
    /// Create from wavedec result
    pub fn from_wavedec(coeffs: Vec<Vec<f64>>) -> Self {
        if coeffs.len() < 2 {
            panic!("wavedec result must have at least 2 arrays");
        }

        // Last element is approximation
        let approx = Array1::from_vec(coeffs[coeffs.len() - 1].clone());

        // All others are details (reverse order to go from coarsest to finest)
        let mut details = Vec::with_capacity(coeffs.len() - 1);
        for i in (0..coeffs.len() - 1).rev() {
            details.push(Array1::from_vec(coeffs[i].clone()));
        }

        Self { approx, details }
    }

    /// Convert back to wavedec format for reconstruction
    pub fn to_wavedec(&self) -> Vec<Vec<f64>> {
        let mut result = Vec::with_capacity(self.details.len() + 1);

        // Add details in reverse order (finest to coarsest)
        for detail in self.details.iter().rev() {
            result.push(detail.to_vec());
        }

        // Add approximation last
        result.push(self.approx.to_vec());

        result
    }
}
