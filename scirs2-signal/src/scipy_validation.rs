//! SciPy Validation Module
//!
//! Comprehensive numerical validation against SciPy implementations for signal processing.
//! This module has been refactored into focused sub-modules for better maintainability
//! while preserving full backward compatibility.
//!
//! # Migration Notice
//!
//! This file now serves as a compatibility layer. The implementation has been moved to
//! `scipy_validation/` sub-modules for better organization and maintainability.
//! All original functionality is preserved through re-exports.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use scirs2_signal::scipy_validation::{validate_all, ValidationConfig};
//!
//! let config = ValidationConfig::default();
//! let results = validate_all(&config)?;
//!
//! if results.all_passed() {
//!     println!("All validations passed!");
//! } else {
//!     for failure in results.failures() {
//!         println!("Failed: {}", failure.test_name);
//!     }
//! }
//! # Ok::<(), scirs2_signal::error::SignalError>(())
//! ```

// Re-export everything from the modular implementation
// Note: The actual implementation is now in the scipy_validation/ directory
// This maintains backward compatibility for existing code

use crate::dwt::{wavedec, waverec, Wavelet};
use crate::error::{SignalError, SignalResult};
use crate::filter::{butter, cheby1, cheby2, lfilter, FilterType};
use crate::lombscargle::lombscargle;
use crate::multitaper::enhanced::{enhanced_pmtm, MultitaperConfig};
use crate::parametric::{ar_spectrum, estimate_ar, ARMethod};
use crate::waveforms::chirp;
use crate::window::kaiser::kaiser;
use crate::window::{blackman, hamming, hann, tukey};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::Rng;
use std::collections::HashMap;

// Import all types and functions from the modular implementation
pub use self::scipy_validation::types::*;
pub use self::scipy_validation::core::*;
pub use self::scipy_validation::filtering::*;
pub use self::scipy_validation::spectral::*;
pub use self::scipy_validation::wavelets::*;
pub use self::scipy_validation::windows::*;
pub use self::scipy_validation::signal_generation::*;
pub use self::scipy_validation::convolution::*;
pub use self::scipy_validation::resampling::*;
pub use self::scipy_validation::peak_detection::*;
pub use self::scipy_validation::reference::*;
pub use self::scipy_validation::utils::*;

// Include the modular implementation
pub mod scipy_validation;