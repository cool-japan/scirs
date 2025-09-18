//! SciPy Validation Module
//!
//! This module provides comprehensive numerical validation against SciPy implementations
//! for signal processing algorithms. It ensures numerical accuracy and correctness
//! of our implementations by comparing against reference SciPy outputs.
//!
//! # Organization
//!
//! The module is organized into focused sub-modules:
//!
//! - [`types`] - Configuration and result data structures
//! - [`core`] - Main validation orchestration and entry points
//! - [`filtering`] - Digital filter validation (Butterworth, Chebyshev, etc.)
//! - [`spectral`] - Spectral analysis validation (periodogram, Welch, STFT, etc.)
//! - [`wavelets`] - Wavelet transform validation (DWT, CWT, wavelet families)
//! - [`windows`] - Window function validation (Hann, Kaiser, etc.)
//! - [`signal_generation`] - Signal generation validation (chirp, square, etc.)
//! - [`convolution`] - Convolution and correlation validation
//! - [`resampling`] - Resampling operation validation
//! - [`peak_detection`] - Peak detection algorithm validation
//! - [`reference`] - Reference implementation functions
//! - [`utils`] - Utility functions for error analysis and reporting
//!
//! # Usage
//!
//! ## Quick Validation
//!
//! ```rust,no_run
//! use scirs2_signal::scipy_validation::{validate_all, ValidationConfig};
//!
//! // Run quick validation with default settings
//! let config = ValidationConfig::default();
//! let results = validate_all(&config)?;
//!
//! if results.all_passed() {
//!     println!("All validations passed!");
//! } else {
//!     println!("Some validations failed:");
//!     for failure in results.failures() {
//!         println!("  {}: {:?}", failure.test_name, failure.error_message);
//!     }
//! }
//! # Ok::<(), scirs2_signal::error::SignalError>(())
//! ```
//!
//! ## Extensive Validation
//!
//! ```rust,no_run
//! use scirs2_signal::scipy_validation::{validate_all, ValidationConfig};
//!
//! // Run extensive validation (slower but more thorough)
//! let mut config = ValidationConfig::default();
//! config.extensive = true;
//! config.tolerance = 1e-12; // Stricter tolerance
//!
//! let results = validate_all(&config)?;
//! println!("{}", results.summary_report());
//! # Ok::<(), scirs2_signal::error::SignalError>(())
//! ```
//!
//! ## Focused Validation
//!
//! ```rust,no_run
//! use scirs2_signal::scipy_validation::{validate_filtering, ValidationConfig};
//! use std::collections::HashMap;
//!
//! // Test only filter implementations
//! let config = ValidationConfig::default();
//! let mut results = HashMap::new();
//! validate_filtering(&mut results, &config)?;
//!
//! for (name, result) in results {
//!     println!("{}: {}", name, if result.passed { "PASSED" } else { "FAILED" });
//! }
//! # Ok::<(), scirs2_signal::error::SignalError>(())
//! ```
//!
//! # Validation Coverage
//!
//! The validation suite covers:
//!
//! ## Digital Filters
//! - Butterworth filters (lowpass, highpass, bandpass, bandstop)
//! - Chebyshev Type I and II filters
//! - Elliptic filters (when implemented)
//! - Bessel filters (when implemented)
//! - Zero-phase filtering (filtfilt)
//!
//! ## Spectral Analysis
//! - Periodogram
//! - Welch's method
//! - Short-time Fourier transform (STFT)
//! - Multitaper spectral estimation
//! - Lomb-Scargle periodogram
//! - Parametric spectral estimation (AR methods)
//!
//! ## Wavelet Transforms
//! - Discrete wavelet transform (DWT)
//! - Continuous wavelet transform (CWT)
//! - Various wavelet families (Haar, Daubechies, Biorthogonal, etc.)
//!
//! ## Window Functions
//! - Hann, Hamming, Blackman windows
//! - Kaiser window with various beta parameters
//! - Tukey window
//! - Other standard window functions
//!
//! ## Signal Processing Operations
//! - Signal generation (chirp, square, sawtooth, etc.)
//! - Convolution and correlation
//! - Resampling operations
//! - Peak detection algorithms
//!
//! # Reference Implementations
//!
//! The validation framework uses simplified reference implementations that
//! approximate SciPy's behavior. In a production environment, these should
//! be replaced with:
//!
//! 1. **Python FFI**: Direct calls to SciPy via PyO3 or similar
//! 2. **Pre-computed Data**: Reference results computed offline and embedded
//! 3. **External Validation**: Subprocess calls to Python scripts
//!
//! # Error Analysis
//!
//! The framework computes comprehensive error metrics:
//! - Maximum absolute error
//! - Maximum relative error
//! - Root mean square error (RMSE)
//! - Mean absolute error
//! - Statistical error distributions
//!
//! # Performance Considerations
//!
//! - Use `ValidationConfig::extensive = false` for faster validation
//! - Reduce `test_lengths` for quicker testing
//! - Focus on specific validation categories as needed
//! - Enable parallel testing where possible

pub mod types;
pub mod core;
pub mod filtering;
pub mod spectral;
pub mod wavelets;
pub mod windows;
pub mod signal_generation;
pub mod convolution;
pub mod resampling;
pub mod peak_detection;
pub mod reference;
pub mod utils;

// Re-export main types and functions for convenience
pub use types::{
    ValidationConfig, ValidationResults, ValidationTestResult, ValidationSummary,
    FilterValidationConfig, SpectralValidationConfig, WaveletValidationConfig,
    WindowValidationConfig, SignalValidationConfig, ComprehensiveValidationConfig,
    ErrorAnalysis, TestParameters,
};

pub use core::{validate_all, validate_precision_levels, score_precision_level};

pub use filtering::{
    validate_filtering, validate_butterworth_filter, validate_chebyshev_filter,
    validate_elliptic_filter, validate_bessel_filter, validate_filtfilt,
};

pub use spectral::{
    validate_spectral_analysis, validate_periodogram, validate_welch, validate_stft,
    validate_multitaper_scipy, validate_lombscargle, validate_parametric_spectral,
};

pub use wavelets::{
    validate_wavelets, validate_dwt, validate_cwt, validate_wavelet_families,
};

pub use windows::validate_windows;

pub use signal_generation::{
    validate_signal_generation, reference_signal_generation,
};

pub use convolution::validate_convolution_correlation;
pub use resampling::validate_resampling;
pub use peak_detection::validate_peak_detection;

pub use reference::{
    reference_butter_filter, reference_cheby1_filter, reference_cheby2_filter,
    reference_multitaper_psd, reference_lombscargle, reference_ar_spectrum,
};

pub use utils::{
    calculate_errors, load_reference_data, generate_validation_report, validate_quick,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_imports() {
        // Test that all main types can be imported
        let _config = ValidationConfig::default();
        let _results = ValidationResults::new();
    }

    #[test]
    fn test_validate_quick() {
        // Test quick validation function
        let result = validate_quick();
        assert!(result.is_ok());
    }

    #[test]
    fn test_comprehensive_validation_config() {
        let config = ComprehensiveValidationConfig::default();
        assert!(config.filters.test_butterworth);
        assert!(config.spectral.test_periodogram);
        assert!(config.wavelets.test_dwt);
        assert!(config.windows.test_symmetric);
        assert!(config.signals.test_chirp);
    }
}