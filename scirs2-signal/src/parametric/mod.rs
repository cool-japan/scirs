//! Parametric spectral estimation methods
//!
//! This module implements parametric methods for spectral estimation, including:
//! - Autoregressive (AR) models using different estimation methods (Yule-Walker, Burg, least-squares)
//! - Moving Average (MA) models
//! - Autoregressive Moving Average (ARMA) models
//!
//! Parametric methods can provide better frequency resolution than non-parametric methods
//! (like periodogram) for shorter data records, and can model specific spectral characteristics.
//!
//! # Example
//! ```
//! use scirs2_core::ndarray::Array1;
//! use scirs2_signal::parametric::{ar_spectrum, burg_method};
//!
//! // Create a signal with spectral peaks
//! let n = 256;
//! let t = Array1::linspace(0.0, 1.0, n);
//! let f1 = 50.0;
//! let f2 = 120.0;
//! let signal = t.mapv(|ti| (2.0 * std::f64::consts::PI * f1 * ti).sin() +
//!                          0.5 * (2.0 * std::f64::consts::PI * f2 * ti).sin());
//!
//! // Estimate AR parameters using Burg's method (order 10)
//! let (ar_coeffs, reflection_coeffs, variance) = burg_method(&signal, 10).unwrap();
//!
//! // Burg method returns coefficients
//! assert_eq!(ar_coeffs.len(), 11); // order + 1 coefficients
//!
//! // Just check that we got valid outputs
//! assert!(variance > 0.0);
//! assert!(reflection_coeffs.is_some());
//!
//! // The coefficients exist
//! assert!(ar_coeffs.iter().any(|&x: &f64| x.abs() > 1e-10));
//! ```

// Module declarations - these will contain the actual implementations
pub mod ar_estimation;
pub mod arma_estimation;
pub mod ma_estimation;
pub mod robust_estimation;
// pub mod order_selection;
// pub mod spectrum_computation;
// pub mod state_space;
// pub mod adaptive_estimation;

// Type definitions
pub mod types;

// Re-export all public types for backward compatibility
pub use types::*;

// Re-export functions from submodules for backward compatibility
pub use ar_estimation::{
    ar_spectrum, burg_method, covariance_method, estimate_ar, least_squares_method,
    modified_covariance_method, select_arorder, yule_walker,
};

// Re-export MA estimation functions for backward compatibility
pub use ma_estimation::estimate_ma;

// Re-export ARMA estimation functions for backward compatibility
pub use arma_estimation::{
    adaptive_arma_estimator, arma_spectrum, arma_spectrum_enhanced, detect_spectral_peaks,
    estimate_arma, estimate_arma_enhanced, select_armaorder_enhanced,
};

// Re-export robust estimation functions for backward compatibility
pub use robust_estimation::{
    compute_parameter_change, robust_ar_estimation, update_robust_weights,
};

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use scirs2_core::validation::{check_finite, check_positive};
use statrs::statistics::Statistics;
use std::collections::HashMap;
use std::f64::consts::PI;

// AR estimation functions have been moved to ar_estimation.rs
// Functions are re-exported above for backward compatibility

// NOTE: Additional functions will be moved to their respective submodules as needed
// All AR estimation functions have been moved to ar_estimation.rs
// Other functions (MA, ARMA, etc.) will remain here until moved to their own modules
