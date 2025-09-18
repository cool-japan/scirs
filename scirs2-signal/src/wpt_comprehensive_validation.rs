//! Comprehensive Wavelet Packet Transform Validation Suite
//!
//! This module has been refactored into focused sub-modules for better maintainability
//! while preserving full backward compatibility through re-exports.
//!
//! # Migration Notice
//!
//! The implementation has been moved to `wpt_comprehensive_validation/` sub-modules.
//! All original functionality is preserved through comprehensive re-exports.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use scirs2_signal::wpt_comprehensive_validation::{
//!     validate_wpt_comprehensive, ComprehensiveWptValidationConfig
//! };
//!
//! let config = ComprehensiveWptValidationConfig::default();
//! let results = validate_wpt_comprehensive(&config)?;
//!
//! println!("Overall score: {:.2}/100", results.overall_score);
//! for issue in &results.issues {
//!     println!("Issue: {}", issue);
//! }
//! # Ok::<(), scirs2_signal::error::SignalError>(())
//! ```
//!
//! # Comprehensive Validation
//!
//! This validation suite provides extensive testing of wavelet packet transform
//! implementations including:
//!
//! - **Basic Validation**: Energy conservation, reconstruction accuracy, stability
//! - **Frame Theory**: Eigenvalue analysis, condition numbers, frame coherence
//! - **Multi-Scale Analysis**: Scale consistency, frequency/time localization
//! - **Best Basis Algorithm**: Convergence, repeatability, efficiency
//! - **Statistical Validation**: Hypothesis testing, bootstrap validation
//! - **Cross-Validation**: Comparison with reference implementations
//! - **Robustness Testing**: Noise resistance, parameter sensitivity, extreme conditions
//!
//! The suite generates a comprehensive score (0-100) and identifies critical issues
//! that require attention for robust wavelet packet transform implementation.

// Re-export everything from the modular implementation
pub use self::wpt_comprehensive_validation::*;

// Include the modular implementation
pub mod wpt_comprehensive_validation;