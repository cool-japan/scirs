//! Wavelet transforms
//!
//! This module provides functions for continuous and discrete wavelet transforms,
//! useful for multi-resolution analysis of signals.

// Import internal modules
mod complex_wavelets;
mod cwt;
mod real_wavelets;
mod scalogram;
#[cfg(test)]
mod tests;
mod transform;
mod types;
mod utils;

// Re-export public components
pub use complex_wavelets::{complex_gaussian, complex_morlet, fbsp, morlet, paul, shannon};
pub use cwt::{convolve_complex_same_complex, convolve_complex_same_real};
pub use real_wavelets::ricker;
pub use scalogram::{cwt_magnitude, cwt_phase, scale_to_frequency, scalogram};
pub use transform::cwt;
pub use types::WaveletType;

// No common imports for internal use
