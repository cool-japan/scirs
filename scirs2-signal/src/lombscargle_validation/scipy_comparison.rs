// SciPy comparison utilities for Lomb-Scargle validation
//
// This module provides functions for comparing our Lomb-Scargle implementation
// with SciPy's reference implementation for validation purposes.

use crate::error::{SignalError, SignalResult};
use super::types::SciPyComparisonResult;
use super::enhanced::{calculate_correlation, calculate_peak_detection_accuracy};

// Re-export comparison functions that are primarily used for SciPy validation
pub use super::enhanced::{scipy_reference_lombscargle, validate_against_scipy_reference};