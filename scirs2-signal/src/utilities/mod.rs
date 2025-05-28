//! Signal processing utilities
//!
//! This module provides additional utility functions for signal processing
//! that extend beyond the basic utilities in the utils module.
//!
//! # Submodules
//!
//! * `spectral` - Utilities for spectral analysis

pub mod spectral;

// Re-export commonly used functions from submodules
pub use spectral::{
    dominant_frequencies, dominant_frequency, energy_spectral_density, normalized_psd,
    spectral_bandwidth, spectral_centroid, spectral_contrast, spectral_crest, spectral_decrease,
    spectral_flatness, spectral_flux, spectral_kurtosis, spectral_rolloff, spectral_skewness,
    spectral_slope, spectral_spread,
};

// Module with tests for utilities
#[cfg(test)]
mod tests;
