//! Calibration utilities for quantization
//!
//! This module provides utilities for finding optimal quantization parameters.
//! Quantization calibration is the process of determining the optimal scaling
//! factors and zero points for a given dataset and quantization method.
//!
//! The module includes:
//!
//! * Histogram-based methods for range calibration
//! * Entropy-based methods using KL divergence minimization
//! * Per-channel calibration strategies
//! * Dynamic calibration based on data statistics

mod types;
mod matrix_calibration;
mod vector_calibration;
mod utils;

// Re-export the main types and functions to maintain backward compatibility
pub use types::{
    CalibrationMethod, CalibrationConfig,
    calibrate_matrix, calibrate_vector,
    get_weight_calibration_config, get_activation_calibration_config
};

// Re-export utility functions that were previously public
pub use utils::{find_min_max, find_min_max_vec, create_params_from_range, determine_data_type};

// Import all internal functions for cross-module usage
pub(super) use matrix_calibration::*;
pub(super) use vector_calibration::*;
pub(super) use utils::*;