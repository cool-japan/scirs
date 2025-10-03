//! Matrix calibration implementations for quantization
//!
//! This module contains all matrix-specific calibration methods including
//! both standard and per-channel calibration strategies.

use super::super::{QuantizationMethod, QuantizationParams};
use super::utils::*;
use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::ArrayView2;
use std::fmt::Debug;

// -------------------------------------------------------------------------
// Matrix calibration implementations
// -------------------------------------------------------------------------

/// Simple min-max calibration for matrices
#[allow(dead_code)]
pub(super) fn calibrate_matrix_minmax<F>(
    matrix: &ArrayView2<F>,
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
    // Find min and max values in the matrix
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for &val in matrix.iter() {
        let val_f32 = val.as_();
        if val_f32.is_finite() {
            min_val = min_val.min(val_f32);
            max_val = max_val.max(val_f32);
        }
    }

    // Handle edge cases
    if !min_val.is_finite() || !max_val.is_finite() {
        return Err(LinalgError::ValueError(
            "Matrix contains non-finite values".to_string(),
        ));
    }

    if min_val == max_val {
        min_val -= 1.0;
        max_val += 1.0;
    }

    create_params_from_range(bits, min_val, max_val, symmetric)
}

/// Moving average min-max calibration for matrices
#[allow(dead_code)]
pub(super) fn calibrate_matrix_moving_average<F>(
    matrix: &ArrayView2<F>,
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
    // Convert matrix to a flattened vector of finite f32 values
    let mut values: Vec<f32> = matrix
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
            "Matrix contains no finite values".to_string(),
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

/// Percentile-based calibration for matrices
#[allow(dead_code)]
pub(super) fn calibrate_matrix_percentile<F>(
    matrix: &ArrayView2<F>,
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

    // Convert matrix to a flattened vector of finite f32 values
    let mut values: Vec<f32> = matrix
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
            "Matrix contains no finite values".to_string(),
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

/// Entropy-based calibration for matrices
#[allow(dead_code)]
pub(super) fn calibrate_matrix_entropy<F>(
    matrix: &ArrayView2<F>,
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
    let (min_val, max_val) = find_min_max(matrix);

    // Create histogram of the data
    let histogram = create_histogram(matrix, min_val, max_val, num_bins);

    // Use KL divergence minimization to find optimal thresholds
    let (opt_min, opt_max) =
        optimize_thresholds_kl_divergence(&histogram, min_val, max_val, bits, symmetric);

    create_params_from_range(bits, opt_min, opt_max, symmetric)
}

/// MSE-based calibration for matrices
#[allow(dead_code)]
pub(super) fn calibrate_matrix_mse<F>(
    matrix: &ArrayView2<F>,
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
    let mut base_params = calibrate_matrix_minmax(matrix, bits, symmetric)?;

    // Define a range of scale factors to try
    let scales = if symmetric {
        optimize_symmetric_scale(matrix, bits, base_params.scale)
    } else {
        let (scale, zero_point) =
            optimize_affine_params(matrix, bits, base_params.scale, base_params.zero_point);
        base_params.scale = scale;
        base_params.zero_point = zero_point;
        base_params.scale
    };

    // Create QuantizationParams with optimized scale
    let mut opt_params = base_params.clone();
    opt_params.scale = scales;

    Ok(opt_params)
}

// -------------------------------------------------------------------------
// Per-channel matrix calibration implementations
// -------------------------------------------------------------------------

/// Per-channel min-max calibration for matrices
#[allow(dead_code)]
pub(super) fn calibrate_matrix_per_channel_minmax<F>(
    matrix: &ArrayView2<F>,
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
    let (_rows, cols) = matrix.dim();

    // Global min/max for the entire matrix
    let (global_min, global_max) = find_min_max(matrix);

    // Per-channel scales and zero points
    let mut channel_scales = Vec::with_capacity(cols);
    let mut channel_zero_points = Vec::with_capacity(if symmetric { 0 } else { cols });

    // For each channel (column)
    for col_idx in 0..cols {
        let column = matrix.column(col_idx);

        // Find min/max for this channel
        let mut col_min = f32::MAX;
        let mut col_max = f32::MIN;

        for &val in column.iter() {
            let val_f32 = val.as_();
            if val_f32.is_finite() {
                col_min = col_min.min(val_f32);
                col_max = col_max.max(val_f32);
            }
        }

        // Handle edge cases
        if !col_min.is_finite() || !col_max.is_finite() {
            col_min = 0.0;
            col_max = 1.0;
        }

        if col_min == col_max {
            col_min -= 1.0;
            col_max += 1.0;
        }

        // Calculate scale and zero point for this channel
        let (scale, zero_point) = if symmetric {
            let abs_max = col_max.abs().max(col_min.abs());
            let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
            (scale, 0)
        } else {
            let scale = (col_max - col_min) / ((1 << bits) - 1) as f32;
            let zero_point = (-col_min / scale).round() as i32;
            (scale, zero_point)
        };

        channel_scales.push(scale);
        if !symmetric {
            channel_zero_points.push(zero_point);
        }
    }

    // Create quantization params
    let q_method = if symmetric {
        QuantizationMethod::PerChannelSymmetric
    } else {
        QuantizationMethod::PerChannelAffine
    };

    Ok(QuantizationParams {
        bits,
        scale: 0.0,    // Not used for per-channel
        zero_point: 0, // Not used for per-channel symmetric
        min_val: global_min,
        max_val: global_max,
        method: q_method,
        data_type: determine_data_type(bits),
        channel_scales: Some(channel_scales),
        channel_zero_points: if symmetric {
            None
        } else {
            Some(channel_zero_points)
        },
    })
}

/// Per-channel moving average calibration for matrices
#[allow(dead_code)]
pub(super) fn calibrate_matrix_per_channel_moving_average<F>(
    matrix: &ArrayView2<F>,
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
    let (_rows, cols) = matrix.dim();

    // Global min/max for the entire matrix
    let (global_min, global_max) = find_min_max(matrix);

    // Per-channel scales and zero points
    let mut channel_scales = Vec::with_capacity(cols);
    let mut channel_zero_points = Vec::with_capacity(if symmetric { 0 } else { cols });

    // For each channel (column)
    for col_idx in 0..cols {
        let column = matrix.column(col_idx);

        // Convert column to a vector of finite f32 values
        let mut values: Vec<f32> = column
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
            // Handle empty or non-finite column
            channel_scales.push(1.0);
            if !symmetric {
                channel_zero_points.push(0);
            }
            continue;
        }

        // Sort values
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate moving averages to find stable min/max
        let (col_min, col_max) = if values.len() <= windowsize {
            // Not enough data for moving average, fall back to min-max
            (*values.first().unwrap(), *values.last().unwrap())
        } else {
            // Calculate moving averages
            let min_val = values.iter().take(windowsize).sum::<f32>() / windowsize as f32;
            let max_val = values.iter().rev().take(windowsize).sum::<f32>() / windowsize as f32;
            (min_val, max_val)
        };

        // Calculate scale and zero point for this channel
        let (scale, zero_point) = if symmetric {
            let abs_max = col_max.abs().max(col_min.abs());
            let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
            (scale, 0)
        } else {
            let scale = (col_max - col_min) / ((1 << bits) - 1) as f32;
            let zero_point = (-col_min / scale).round() as i32;
            (scale, zero_point)
        };

        channel_scales.push(scale);
        if !symmetric {
            channel_zero_points.push(zero_point);
        }
    }

    // Create quantization params
    let q_method = if symmetric {
        QuantizationMethod::PerChannelSymmetric
    } else {
        QuantizationMethod::PerChannelAffine
    };

    Ok(QuantizationParams {
        bits,
        scale: 0.0,    // Not used for per-channel
        zero_point: 0, // Not used for per-channel symmetric
        min_val: global_min,
        max_val: global_max,
        method: q_method,
        data_type: determine_data_type(bits),
        channel_scales: Some(channel_scales),
        channel_zero_points: if symmetric {
            None
        } else {
            Some(channel_zero_points)
        },
    })
}

/// Per-channel percentile calibration for matrices
#[allow(dead_code)]
pub(super) fn calibrate_matrix_per_channel_percentile<F>(
    matrix: &ArrayView2<F>,
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

    let (_rows, cols) = matrix.dim();

    // Global min/max for the entire matrix
    let (global_min, global_max) = find_min_max(matrix);

    // Per-channel scales and zero points
    let mut channel_scales = Vec::with_capacity(cols);
    let mut channel_zero_points = Vec::with_capacity(if symmetric { 0 } else { cols });

    // For each channel (column)
    for col_idx in 0..cols {
        let column = matrix.column(col_idx);

        // Convert column to a vector of finite f32 values
        let mut values: Vec<f32> = column
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
            // Handle empty or non-finite column
            channel_scales.push(1.0);
            if !symmetric {
                channel_zero_points.push(0);
            }
            continue;
        }

        // Sort values
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute percentile indexes
        let low_idx = ((1.0 - percentile) * (values.len() as f32)) as usize;
        let high_idx = ((percentile) * (values.len() as f32)) as usize;

        // Get percentile values
        let col_min = values[low_idx.min(values.len() - 1)];
        let col_max = values[high_idx.min(values.len() - 1)];

        // Calculate scale and zero point for this channel
        let (scale, zero_point) = if symmetric {
            let abs_max = col_max.abs().max(col_min.abs());
            let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
            (scale, 0)
        } else {
            let scale = (col_max - col_min) / ((1 << bits) - 1) as f32;
            let zero_point = (-col_min / scale).round() as i32;
            (scale, zero_point)
        };

        channel_scales.push(scale);
        if !symmetric {
            channel_zero_points.push(zero_point);
        }
    }

    // Create quantization params
    let q_method = if symmetric {
        QuantizationMethod::PerChannelSymmetric
    } else {
        QuantizationMethod::PerChannelAffine
    };

    Ok(QuantizationParams {
        bits,
        scale: 0.0,    // Not used for per-channel
        zero_point: 0, // Not used for per-channel symmetric
        min_val: global_min,
        max_val: global_max,
        method: q_method,
        data_type: determine_data_type(bits),
        channel_scales: Some(channel_scales),
        channel_zero_points: if symmetric {
            None
        } else {
            Some(channel_zero_points)
        },
    })
}

/// Per-channel entropy calibration for matrices
#[allow(dead_code)]
pub(super) fn calibrate_matrix_per_channel_entropy<F>(
    matrix: &ArrayView2<F>,
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
    let (_rows, cols) = matrix.dim();

    // Global min/max for the entire matrix
    let (global_min, global_max) = find_min_max(matrix);

    // Per-channel scales and zero points
    let mut channel_scales = Vec::with_capacity(cols);
    let mut channel_zero_points = Vec::with_capacity(if symmetric { 0 } else { cols });

    // For each channel (column)
    for col_idx in 0..cols {
        let column = matrix.column(col_idx);

        // Find min/max for this channel
        let (col_min, col_max) = find_min_max_vec(&column);

        // Create histogram of the channel data
        let histogram = create_histogram_vec(&column, col_min, col_max, num_bins);

        // Use KL divergence minimization to find optimal thresholds
        let (opt_min, opt_max) =
            optimize_thresholds_kl_divergence(&histogram, col_min, col_max, bits, symmetric);

        // Calculate scale and zero point for this channel
        let (scale, zero_point) = if symmetric {
            let abs_max = opt_max.abs().max(opt_min.abs());
            let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
            (scale, 0)
        } else {
            let scale = (opt_max - opt_min) / ((1 << bits) - 1) as f32;
            let zero_point = (-opt_min / scale).round() as i32;
            (scale, zero_point)
        };

        channel_scales.push(scale);
        if !symmetric {
            channel_zero_points.push(zero_point);
        }
    }

    // Create quantization params
    let q_method = if symmetric {
        QuantizationMethod::PerChannelSymmetric
    } else {
        QuantizationMethod::PerChannelAffine
    };

    Ok(QuantizationParams {
        bits,
        scale: 0.0,    // Not used for per-channel
        zero_point: 0, // Not used for per-channel symmetric
        min_val: global_min,
        max_val: global_max,
        method: q_method,
        data_type: determine_data_type(bits),
        channel_scales: Some(channel_scales),
        channel_zero_points: if symmetric {
            None
        } else {
            Some(channel_zero_points)
        },
    })
}

/// Per-channel MSE calibration for matrices
#[allow(dead_code)]
pub(super) fn calibrate_matrix_per_channel_mse<F>(
    matrix: &ArrayView2<F>,
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
    let (_rows, cols) = matrix.dim();

    // Global min/max for the entire matrix
    let (global_min, global_max) = find_min_max(matrix);

    // Per-channel scales and zero points
    let mut channel_scales = Vec::with_capacity(cols);
    let mut channel_zero_points = Vec::with_capacity(if symmetric { 0 } else { cols });

    // For each channel (column)
    for col_idx in 0..cols {
        let column = matrix.column(col_idx);

        // Start with min-max calibration for this channel
        let (col_min, col_max) = find_min_max_vec(&column);

        let base_scale = if symmetric {
            let abs_max = col_max.abs().max(col_min.abs());
            abs_max / ((1 << (bits - 1)) - 1) as f32
        } else {
            (col_max - col_min) / ((1 << bits) - 1) as f32
        };

        let base_zero_point = if symmetric {
            0
        } else {
            (-col_min / base_scale).round() as i32
        };

        // Optimize parameters for this channel
        if symmetric {
            let scale = optimize_symmetric_scale_vec(&column, bits, base_scale);
            channel_scales.push(scale);
        } else {
            let (scale, zero_point) =
                optimize_affine_params_vec(&column, bits, base_scale, base_zero_point);
            channel_scales.push(scale);
            channel_zero_points.push(zero_point);
        }
    }

    // Create quantization params
    let q_method = if symmetric {
        QuantizationMethod::PerChannelSymmetric
    } else {
        QuantizationMethod::PerChannelAffine
    };

    Ok(QuantizationParams {
        bits,
        scale: 0.0,    // Not used for per-channel
        zero_point: 0, // Not used for per-channel symmetric
        min_val: global_min,
        max_val: global_max,
        method: q_method,
        data_type: determine_data_type(bits),
        channel_scales: Some(channel_scales),
        channel_zero_points: if symmetric {
            None
        } else {
            Some(channel_zero_points)
        },
    })
}
