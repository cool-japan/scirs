//! Types and configuration for quantization calibration
//!
//! This module provides the core types, enums, and configuration structures
//! for quantization calibration methods.

use super::super::{
    calibration_ema::{
        calibrate_matrix_ema, calibrate_matrix_per_channel_ema, calibrate_vector_ema,
    },
    QuantizationMethod, QuantizationParams,
};
use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use std::fmt::Debug;

use super::{matrix_calibration::*, utils::*, vector_calibration::*};

/// Calibration method for determining quantization parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMethod {
    /// Simple min-max calibration that uses the full range of the data
    MinMax,

    /// Moving average min-max that excludes outliers
    MovingAverageMinMax,

    /// Entropy-based calibration that minimizes the KL divergence
    EntropyCalibration,

    /// Percentile-based calibration that excludes outliers based on percentiles
    PercentileCalibration,

    /// Mean squared error minimization for finding optimal scale
    MSEOptimization,

    /// Exponential moving average for dynamic calibration
    ExponentialMovingAverage,
}

/// Configuration for quantization calibration
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Method used for calibration
    pub method: CalibrationMethod,

    /// Number of histogram bins for entropy-based methods
    pub num_bins: usize,

    /// Percentile value for percentile-based methods (0.0 to 1.0)
    pub percentile: f32,

    /// Moving average window size for min-max methods
    pub windowsize: usize,

    /// Whether to use per-channel calibration
    pub per_channel: bool,

    /// Whether to use symmetric quantization
    pub symmetric: bool,

    /// Smoothing factor for exponential moving average (0.0 to 1.0)
    /// Higher values give more weight to recent observations
    pub ema_factor: f32,

    /// Number of calibration iterations for dynamic methods
    pub max_iterations: usize,

    /// Convergence threshold for iterative calibration methods
    pub convergence_threshold: f32,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        CalibrationConfig {
            method: CalibrationMethod::MinMax,
            num_bins: 2048,
            percentile: 0.999,
            windowsize: 10,
            per_channel: false,
            symmetric: true,
            ema_factor: 0.1,
            max_iterations: 10,
            convergence_threshold: 1e-6,
        }
    }
}

/// Calibrate quantization parameters for a matrix using the specified method
///
/// # Arguments
///
/// * `matrix` - Input matrix to calibrate
/// * `bits` - Bit width for quantization
/// * `config` - Calibration configuration
///
/// # Returns
///
/// * Calibrated quantization parameters
#[allow(dead_code)]
pub fn calibrate_matrix<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    config: &CalibrationConfig,
) -> LinalgResult<QuantizationParams>
where
    F: scirs2_core::numeric::Float
        + Debug
        + scirs2_core::numeric::AsPrimitive<f32>
        + scirs2_core::numeric::FromPrimitive,
    f32: scirs2_core::numeric::AsPrimitive<F>,
{
    match config.method {
        CalibrationMethod::MinMax => {
            if config.per_channel {
                calibrate_matrix_per_channel_minmax(matrix, bits, config.symmetric)
            } else {
                calibrate_matrix_minmax(matrix, bits, config.symmetric)
            }
        }
        CalibrationMethod::MovingAverageMinMax => {
            if config.per_channel {
                calibrate_matrix_per_channel_moving_average(
                    matrix,
                    bits,
                    config.windowsize,
                    config.symmetric,
                )
            } else {
                calibrate_matrix_moving_average(matrix, bits, config.windowsize, config.symmetric)
            }
        }
        CalibrationMethod::PercentileCalibration => {
            if config.per_channel {
                calibrate_matrix_per_channel_percentile(
                    matrix,
                    bits,
                    config.percentile,
                    config.symmetric,
                )
            } else {
                calibrate_matrix_percentile(matrix, bits, config.percentile, config.symmetric)
            }
        }
        CalibrationMethod::EntropyCalibration => {
            if config.per_channel {
                calibrate_matrix_per_channel_entropy(
                    matrix,
                    bits,
                    config.num_bins,
                    config.symmetric,
                )
            } else {
                calibrate_matrix_entropy(matrix, bits, config.num_bins, config.symmetric)
            }
        }
        CalibrationMethod::MSEOptimization => {
            if config.per_channel {
                calibrate_matrix_per_channel_mse(matrix, bits, config.symmetric)
            } else {
                calibrate_matrix_mse(matrix, bits, config.symmetric)
            }
        }
        CalibrationMethod::ExponentialMovingAverage => {
            if config.per_channel {
                calibrate_matrix_per_channel_ema(
                    matrix,
                    bits,
                    config.ema_factor,
                    config.max_iterations,
                    config.convergence_threshold,
                    config.symmetric,
                )
            } else {
                calibrate_matrix_ema(
                    matrix,
                    bits,
                    config.ema_factor,
                    config.max_iterations,
                    config.convergence_threshold,
                    config.symmetric,
                )
            }
        }
    }
}

/// Calibrate quantization parameters for a vector using the specified method
///
/// # Arguments
///
/// * `vector` - Input vector to calibrate
/// * `bits` - Bit width for quantization
/// * `config` - Calibration configuration
///
/// # Returns
///
/// * Calibrated quantization parameters
#[allow(dead_code)]
pub fn calibrate_vector<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    config: &CalibrationConfig,
) -> LinalgResult<QuantizationParams>
where
    F: scirs2_core::numeric::Float
        + Debug
        + scirs2_core::numeric::AsPrimitive<f32>
        + scirs2_core::numeric::FromPrimitive,
    f32: scirs2_core::numeric::AsPrimitive<F>,
{
    // Modify configuration to disable per-channel for vectors
    let mut config = config.clone();
    config.per_channel = false;

    match config.method {
        CalibrationMethod::MinMax => calibrate_vector_minmax(vector, bits, config.symmetric),
        CalibrationMethod::MovingAverageMinMax => {
            calibrate_vector_moving_average(vector, bits, config.windowsize, config.symmetric)
        }
        CalibrationMethod::PercentileCalibration => {
            calibrate_vector_percentile(vector, bits, config.percentile, config.symmetric)
        }
        CalibrationMethod::EntropyCalibration => {
            calibrate_vector_entropy(vector, bits, config.num_bins, config.symmetric)
        }
        CalibrationMethod::MSEOptimization => calibrate_vector_mse(vector, bits, config.symmetric),
        CalibrationMethod::ExponentialMovingAverage => calibrate_vector_ema(
            vector,
            bits,
            config.ema_factor,
            config.max_iterations,
            config.convergence_threshold,
            config.symmetric,
        ),
    }
}

/// Helper function to get recommended calibration configuration for neural network weights
///
/// Neural network weights typically benefit from symmetric quantization with
/// entropy-based or percentile-based calibration to handle outliers.
///
/// # Arguments
///
/// * `bits` - Bit width to use for quantization
/// * `aggressive` - Whether to use more aggressive quantization (percentile vs entropy)
///
/// # Returns
///
/// Calibration configuration optimized for neural network weights
#[allow(dead_code)]
pub fn get_weight_calibration_config(bits: u8, aggressive: bool) -> CalibrationConfig {
    if aggressive {
        // More aggressive calibration - clips outliers more
        CalibrationConfig {
            method: CalibrationMethod::PercentileCalibration,
            symmetric: true,
            percentile: 0.99,  // Exclude 1% outliers on each tail
            per_channel: true, // Per-channel calibration often better for weights
            ..Default::default()
        }
    } else {
        // Default calibration - preserves more of the distribution
        CalibrationConfig {
            method: CalibrationMethod::EntropyCalibration,
            symmetric: true,
            num_bins: 2048, // Higher bin count for better precision
            per_channel: true,
            ..Default::default()
        }
    }
}

/// Helper function to get recommended calibration configuration for neural network activations
///
/// Activations typically benefit from asymmetric quantization, especially for
/// non-negative activation functions like ReLU.
///
/// # Arguments
///
/// * `bits` - Bit width to use for quantization
/// * `non_negative` - Whether the activations are known to be non-negative (e.g., from ReLU)
/// * `outlier_sensitive` - Whether the activations contain important outliers
///
/// # Returns
///
/// Calibration configuration optimized for neural network activations
#[allow(dead_code)]
pub fn get_activation_calibration_config(
    _bits: u8,
    non_negative: bool,
    outlier_sensitive: bool,
) -> CalibrationConfig {
    let mut config = if outlier_sensitive {
        // Outlier-sensitive activations benefit from MSE optimization
        CalibrationConfig {
            method: CalibrationMethod::MSEOptimization,
            num_bins: 1024,
            per_channel: false, // Activations usually don't need per-channel
            ..Default::default()
        }
    } else {
        // Standard activations benefit from percentile calibration to ignore outliers
        CalibrationConfig {
            method: CalibrationMethod::PercentileCalibration,
            percentile: 0.9995, // Keep more of the distribution than for weights
            per_channel: false,
            ..Default::default()
        }
    };

    // For activations like ReLU outputs, asymmetric is better
    // For activations with both positive and negative values, symmetric may be better
    config.symmetric = !non_negative;

    config
}
