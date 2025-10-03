//! Vector calibration implementations for quantization
//!
//! This module contains all vector-specific calibration methods for
//! one-dimensional array quantization.

use super::super::QuantizationParams;
use super::utils::*;
use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::ArrayView1;
use std::fmt::Debug;

// -------------------------------------------------------------------------
// Vector calibration implementations
// -------------------------------------------------------------------------

/// Simple min-max calibration for vectors
#[allow(dead_code)]
pub(super) fn calibrate_vector_minmax<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: scirs2_core::numeric::Float
        + Debug
        + scirs2_core::numeric::AsPrimitive<f32>
        + scirs2_core::numeric::FromPrimitive,
    f32: scirs2_core::numeric::AsPrimitive<F>,
{
    // Find min and max values in the vector
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for &val in vector.iter() {
        let val_f32 = val.as_();
        if val_f32.is_finite() {
            min_val = min_val.min(val_f32);
            max_val = max_val.max(val_f32);
        }
    }

    // Handle edge cases
    if !min_val.is_finite() || !max_val.is_finite() {
        return Err(LinalgError::ValueError(
            "Vector contains non-finite values".to_string(),
        ));
    }

    if min_val == max_val {
        min_val -= 1.0;
        max_val += 1.0;
    }

    create_params_from_range(bits, min_val, max_val, symmetric)
}

/// Moving average min-max calibration for vectors
#[allow(dead_code)]
pub(super) fn calibrate_vector_moving_average<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    windowsize: usize,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: scirs2_core::numeric::Float
        + Debug
        + scirs2_core::numeric::AsPrimitive<f32>
        + scirs2_core::numeric::FromPrimitive,
    f32: scirs2_core::numeric::AsPrimitive<F>,
{
    // Convert vector to a flattened vector of finite f32 values
    let mut values: Vec<f32> = vector
        .iter()
        .filter_map(|&x| {
            let val = x.as_();
            if val.is_finite() {
                Some(val)
            } else {
                None
            }
        })
        .collect();

    if values.is_empty() {
        return Err(LinalgError::ValueError(
            "Vector contains no finite values".to_string(),
        ));
    }

    // Sort values
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate moving averages to find stable min/max
    if values.len() <= windowsize {
        // Not enough data for moving average, fall back to min-max
        let min_val = *values.first().unwrap();
        let max_val = *values.last().unwrap();
        create_params_from_range(bits, min_val, max_val, symmetric)
    } else {
        // Calculate moving averages
        let min_val = values.iter().take(windowsize).sum::<f32>() / windowsize as f32;
        let max_val = values.iter().rev().take(windowsize).sum::<f32>() / windowsize as f32;

        create_params_from_range(bits, min_val, max_val, symmetric)
    }
}

/// Percentile-based calibration for vectors
#[allow(dead_code)]
pub(super) fn calibrate_vector_percentile<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    percentile: f32,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: scirs2_core::numeric::Float
        + Debug
        + scirs2_core::numeric::AsPrimitive<f32>
        + scirs2_core::numeric::FromPrimitive,
    f32: scirs2_core::numeric::AsPrimitive<F>,
{
    if !(0.0..=1.0).contains(&percentile) {
        return Err(LinalgError::ValueError(
            "Percentile must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Convert vector to a flattened vector of finite f32 values
    let mut values: Vec<f32> = vector
        .iter()
        .filter_map(|&x| {
            let val = x.as_();
            if val.is_finite() {
                Some(val)
            } else {
                None
            }
        })
        .collect();

    if values.is_empty() {
        return Err(LinalgError::ValueError(
            "Vector contains no finite values".to_string(),
        ));
    }

    // Sort values
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute percentile indexes
    let low_idx = ((1.0 - percentile) * (values.len() as f32)).round() as usize;
    let high_idx = ((percentile) * (values.len() as f32)).round() as usize;

    // Get percentile values
    let min_val = values[low_idx.min(values.len() - 1)];
    let max_val = values[high_idx.min(values.len() - 1)];

    create_params_from_range(bits, min_val, max_val, symmetric)
}

/// Entropy-based calibration for vectors
#[allow(dead_code)]
pub(super) fn calibrate_vector_entropy<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    num_bins: usize,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: scirs2_core::numeric::Float
        + Debug
        + scirs2_core::numeric::AsPrimitive<f32>
        + scirs2_core::numeric::FromPrimitive,
    f32: scirs2_core::numeric::AsPrimitive<F>,
{
    // First get min-max range to create histogram
    let (min_val, max_val) = find_min_max_vec(vector);

    // Create histogram of the data
    let histogram = create_histogram_vec(vector, min_val, max_val, num_bins);

    // Use KL divergence minimization to find optimal thresholds
    let (opt_min, opt_max) =
        optimize_thresholds_kl_divergence(&histogram, min_val, max_val, bits, symmetric);

    create_params_from_range(bits, opt_min, opt_max, symmetric)
}

/// MSE-based calibration for vectors
#[allow(dead_code)]
pub(super) fn calibrate_vector_mse<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: scirs2_core::numeric::Float
        + Debug
        + scirs2_core::numeric::AsPrimitive<f32>
        + scirs2_core::numeric::FromPrimitive,
    f32: scirs2_core::numeric::AsPrimitive<F>,
{
    // Start with min-max calibration
    let mut base_params = calibrate_vector_minmax(vector, bits, symmetric)?;

    // Define a range of scale factors to try
    let scales = if symmetric {
        optimize_symmetric_scale_vec(vector, bits, base_params.scale)
    } else {
        let (scale, zero_point) =
            optimize_affine_params_vec(vector, bits, base_params.scale, base_params.zero_point);
        base_params.scale = scale;
        base_params.zero_point = zero_point;
        base_params.scale
    };

    // Create QuantizationParams with optimized scale
    let mut opt_params = base_params.clone();
    opt_params.scale = scales;

    Ok(opt_params)
}
