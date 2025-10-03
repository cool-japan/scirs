// Wavelet type definitions and common traits

use crate::dwt::Wavelet;
use scirs2_core::numeric::Complex64;
use std::fmt::Debug;

#[allow(unused_imports)]
/// Trait for wavelet functions
pub trait WaveletType: Clone + Debug {
    /// Returns the central frequency of the wavelet
    fn central_frequency(&self) -> Option<f64> {
        // Default implementation returns None, individual wavelets should override this
        None
    }
}

// Implement the trait for Complex64
impl WaveletType for Complex64 {
    fn central_frequency(&self) -> Option<f64> {
        // For complex wavelet, we'll use a default value
        // This should be overridden by specific wavelet implementations
        Some(1.0)
    }
}

// Implement the trait for f64
impl WaveletType for f64 {
    fn central_frequency(&self) -> Option<f64> {
        // For real wavelet, we'll use a default value
        // This should be overridden by specific wavelet implementations
        Some(1.0)
    }
}
