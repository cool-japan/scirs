// Implementation of Exponential Moving Average (EMA) calibration methods

use super::calibration::{
    create_params_from_range, determine_data_type, find_min_max, find_min_max_vec,
};
use super::{QuantizationMethod, QuantizationParams};
use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use std::fmt::Debug;

// Utility function to convert f32 to any Float type
#[allow(dead_code)]
fn to_f<F>(val: f32) -> F
where
    F: scirs2_core::numeric::Float + scirs2_core::numeric::FromPrimitive,
{
    F::from_f32(val).unwrap()
}

// Helper to convert from generic float to f32
#[allow(dead_code)]
fn to_f32<F>(val: F) -> f32
where
    F: scirs2_core::numeric::Float + scirs2_core::numeric::AsPrimitive<f32>,
{
    val.as_()
}

/// Exponential moving average calibration for matrices
///
/// This method uses iterative refinement with EMA to find optimal quantization parameters
/// where each step calibrates parameters, quantizes/dequantizes, and uses the error to
/// update the min/max values.
///
/// # Arguments
///
/// * `matrix` - Input matrix to calibrate
/// * `bits` - Bit width for quantization
/// * `ema_factor` - Smoothing factor (0-1) where higher values give more weight to recent observations
/// * `max_iterations` - Maximum number of iterations for convergence
/// * `convergence_threshold` - Threshold for early stopping
/// * `symmetric` - Whether to use symmetric quantization
///
/// # Returns
///
/// * Calibrated quantization parameters
#[allow(dead_code)]
pub fn calibrate_matrix_ema<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    ema_factor: f32,
    max_iterations: usize,
    convergence_threshold: f32,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: scirs2_core::numeric::Float
        + Debug
        + scirs2_core::numeric::AsPrimitive<f32>
        + scirs2_core::numeric::FromPrimitive,
    f32: scirs2_core::numeric::AsPrimitive<F>,
{
    // Validate parameters
    if !(0.0..=1.0).contains(&ema_factor) {
        return Err(LinalgError::ValueError(
            "EMA factor must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Start with min-max calibration
    let (min_val_f32, max_val_f32) = find_min_max(matrix);

    // Convert min/max values back to type F
    let mut min_val = to_f::<F>(min_val_f32);
    let mut max_val = to_f::<F>(max_val_f32);

    // Convert matrix to f32 for calculations
    let matrix_f32 = matrix.mapv(|x| x.as_());

    let mut prev_mse = f32::MAX;

    // Iterative refinement
    for _iter in 0..max_iterations {
        // Create parameters based on current min/max
        let min_val_f32 = to_f32(min_val);
        let max_val_f32 = to_f32(max_val);
        let params = create_params_from_range(bits, min_val_f32, max_val_f32, symmetric)?;

        // Simulate quantization and dequantization for error calculation
        let dequantized = simulate_quantization(&matrix_f32, &params, bits);

        // Calculate MSE
        let mse = (&matrix_f32 - &dequantized).mapv(|x| x * x).sum() / matrix_f32.len() as f32;

        // Check convergence
        if (prev_mse - mse).abs() < convergence_threshold {
            // Converged, return current parameters
            return Ok(params);
        }

        prev_mse = mse;

        // Update min/max values using error feedback
        if symmetric {
            // For symmetric quantization, adjust abs_max
            let abs_max = max_val.abs().max(min_val.abs());

            // Calculate error and update abs_max
            let mean_abs_error = (&matrix_f32 - &dequantized)
                .mapv(|x| x.abs())
                .mean()
                .unwrap_or(0.0);
            let scale_adjustment = if mean_abs_error > 0.01 {
                1.0 + mean_abs_error
            } else {
                1.0
            };

            // Apply EMA to abs_max - converting between F and f32 properly
            let abs_max_f32 = to_f32(abs_max);
            let new_abs_max_f32 = abs_max_f32 * scale_adjustment;
            let updated_abs_max_f32 =
                abs_max_f32 * (1.0 - ema_factor) + new_abs_max_f32 * ema_factor;
            let updated_abs_max = to_f::<F>(updated_abs_max_f32);

            // Update min/max
            min_val = -updated_abs_max;
            max_val = updated_abs_max;
        } else {
            // For asymmetric quantization, adjust min and max separately
            // Calculate per-direction errors
            let negative_errors = matrix_f32
                .iter()
                .zip(dequantized.iter())
                .filter_map(|(&orig, &deq)| {
                    if orig < deq {
                        Some((orig - deq).abs())
                    } else {
                        None
                    }
                })
                .fold(0.0, |sum, error| sum + error);

            let positive_errors = matrix_f32
                .iter()
                .zip(dequantized.iter())
                .filter_map(|(&orig, &deq)| {
                    if orig > deq {
                        Some((orig - deq).abs())
                    } else {
                        None
                    }
                })
                .fold(0.0, |sum, error| sum + error);

            // Calculate adjustment factors
            let neg_count = matrix_f32.iter().filter(|&&x| x < 0.0).count() as f32;
            let pos_count = matrix_f32.iter().filter(|&&x| x > 0.0).count() as f32;

            let neg_adjustment = if neg_count > 0.0 {
                1.0 + (negative_errors / neg_count)
            } else {
                1.0
            };
            let pos_adjustment = if pos_count > 0.0 {
                1.0 + (positive_errors / pos_count)
            } else {
                1.0
            };

            // Apply EMA to min and max - converting properly between types
            let min_val_f32 = to_f32(min_val);
            let max_val_f32 = to_f32(max_val);

            let new_min_f32 = min_val_f32 * neg_adjustment;
            let new_max_f32 = max_val_f32 * pos_adjustment;

            let updated_min_f32 = min_val_f32 * (1.0 - ema_factor) + new_min_f32 * ema_factor;
            let updated_max_f32 = max_val_f32 * (1.0 - ema_factor) + new_max_f32 * ema_factor;

            min_val = to_f::<F>(updated_min_f32);
            max_val = to_f::<F>(updated_max_f32);
        }

        // Debug output for last iteration
        if _iter == max_iterations - 1 {
            println!("EMA calibration reached max iterations with MSE: {mse}");
        }
    }

    // Return parameters from the final iteration
    let min_val_f32 = to_f32(min_val);
    let max_val_f32 = to_f32(max_val);
    create_params_from_range(bits, min_val_f32, max_val_f32, symmetric)
}

/// Per-channel exponential moving average calibration for matrices
///
/// Applies EMA calibration to each channel (column) independently
///
/// # Arguments
///
/// * `matrix` - Input matrix to calibrate
/// * `bits` - Bit width for quantization
/// * `ema_factor` - Smoothing factor (0-1) where higher values give more weight to recent observations
/// * `max_iterations` - Maximum number of iterations for convergence
/// * `convergence_threshold` - Threshold for early stopping
/// * `symmetric` - Whether to use symmetric quantization
///
/// # Returns
///
/// * Calibrated quantization parameters with per-channel scales
#[allow(dead_code)]
pub fn calibrate_matrix_per_channel_ema<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    ema_factor: f32,
    max_iterations: usize,
    convergence_threshold: f32,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: scirs2_core::numeric::Float
        + Debug
        + scirs2_core::numeric::AsPrimitive<f32>
        + scirs2_core::numeric::FromPrimitive,
    f32: scirs2_core::numeric::AsPrimitive<F>,
{
    // Validate parameters
    if !(0.0..=1.0).contains(&ema_factor) {
        return Err(LinalgError::ValueError(
            "EMA factor must be between 0.0 and 1.0".to_string(),
        ));
    }

    let (_rows, cols) = matrix.dim();

    // Global min/max for the entire matrix (for tracking only)
    let (global_min, global_max) = find_min_max(matrix);

    // Per-channel scales and zero points
    let mut channel_scales = Vec::with_capacity(cols);
    let mut channel_zero_points = Vec::with_capacity(if symmetric { 0 } else { cols });

    // Convert matrix to f32 for calculations
    let matrix_f32 = matrix.mapv(|x| x.as_());

    // For each channel (column)
    for col_idx in 0..cols {
        let column_view = matrix.column(col_idx);
        let column_f32_view = matrix_f32.column(col_idx);

        // Find initial min/max for this channel
        let (col_min_f32, col_max_f32) = find_min_max_vec(&column_view);
        let mut col_min = to_f::<F>(col_min_f32);
        let mut col_max = to_f::<F>(col_max_f32);
        let mut prev_mse = f32::MAX;

        // Iterative refinement for this channel
        for _iter in 0..max_iterations {
            // Create parameters based on current min/max
            let (scale, zero_point) = if symmetric {
                let abs_max = col_max.abs().max(col_min.abs());
                let abs_max_f32 = to_f32(abs_max);
                let scale_f32 = abs_max_f32 / ((1 << (bits - 1)) - 1) as f32;
                (scale_f32, 0)
            } else {
                let min_val_f32 = to_f32(col_min);
                let max_val_f32 = to_f32(col_max);
                let scale_f32 = (max_val_f32 - min_val_f32) / ((1 << bits) - 1) as f32;
                let zero_point = (-min_val_f32 / scale_f32).round() as i32;
                (scale_f32, zero_point)
            };

            // Simulate quantization for this column
            let dequantized_col = simulate_quantization_vector_f32(
                &column_f32_view,
                scale,
                zero_point,
                bits,
                symmetric,
            );

            // Calculate MSE for this column
            let mse = column_f32_view
                .iter()
                .zip(dequantized_col.iter())
                .map(|(&orig, &deq)| (orig - deq).powi(2))
                .sum::<f32>()
                / column_f32_view.len() as f32;

            // Check convergence
            if (prev_mse - mse).abs() < convergence_threshold {
                break;
            }

            prev_mse = mse;

            // Update min/max values using error feedback
            if symmetric {
                // For symmetric quantization, adjust abs_max
                let abs_max = col_max.abs().max(col_min.abs());

                // Calculate error and update abs_max
                let mean_abs_error = column_f32_view
                    .iter()
                    .zip(dequantized_col.iter())
                    .map(|(&orig, &deq)| (orig - deq).abs())
                    .sum::<f32>()
                    / column_f32_view.len() as f32;

                let scale_adjustment = if mean_abs_error > 0.01 {
                    1.0 + mean_abs_error
                } else {
                    1.0
                };

                // Apply EMA to abs_max - converting between F and f32 properly
                let abs_max_f32 = to_f32(abs_max);
                let new_abs_max_f32 = abs_max_f32 * scale_adjustment;
                let updated_abs_max_f32 =
                    abs_max_f32 * (1.0 - ema_factor) + new_abs_max_f32 * ema_factor;
                let updated_abs_max = to_f::<F>(updated_abs_max_f32);

                // Update min/max
                col_min = -updated_abs_max;
                col_max = updated_abs_max;
            } else {
                // For asymmetric quantization, adjust min and max separately
                // Similar to the global method but for this column only
                let negative_errors = column_f32_view
                    .iter()
                    .zip(dequantized_col.iter())
                    .filter_map(|(&orig, &deq)| {
                        if orig < deq {
                            Some((orig - deq).abs())
                        } else {
                            None
                        }
                    })
                    .sum::<f32>();

                let positive_errors = column_f32_view
                    .iter()
                    .zip(dequantized_col.iter())
                    .filter_map(|(&orig, &deq)| {
                        if orig > deq {
                            Some((orig - deq).abs())
                        } else {
                            None
                        }
                    })
                    .sum::<f32>();

                // Calculate adjustment factors
                let neg_count = column_f32_view.iter().filter(|&&x| x < 0.0).count() as f32;
                let pos_count = column_f32_view.iter().filter(|&&x| x > 0.0).count() as f32;

                let neg_adjustment = if neg_count > 0.0 {
                    1.0 + (negative_errors / neg_count)
                } else {
                    1.0
                };
                let pos_adjustment = if pos_count > 0.0 {
                    1.0 + (positive_errors / pos_count)
                } else {
                    1.0
                };

                // Apply EMA to min and max - converting properly between types
                let min_val_f32 = to_f32(col_min);
                let max_val_f32 = to_f32(col_max);

                let new_min_f32 = min_val_f32 * neg_adjustment;
                let new_max_f32 = max_val_f32 * pos_adjustment;

                let updated_min_f32 = min_val_f32 * (1.0 - ema_factor) + new_min_f32 * ema_factor;
                let updated_max_f32 = max_val_f32 * (1.0 - ema_factor) + new_max_f32 * ema_factor;

                col_min = to_f::<F>(updated_min_f32);
                col_max = to_f::<F>(updated_max_f32);
            }
        }

        // After convergence, calculate final scale and zero_point
        let (scale, zero_point) = if symmetric {
            let abs_max = col_max.abs().max(col_min.abs());
            let abs_max_f32 = to_f32(abs_max);
            let scale_f32 = abs_max_f32 / ((1 << (bits - 1)) - 1) as f32;
            (scale_f32, 0)
        } else {
            let min_val_f32 = to_f32(col_min);
            let max_val_f32 = to_f32(col_max);
            let scale_f32 = (max_val_f32 - min_val_f32) / ((1 << bits) - 1) as f32;
            let zero_point = (-min_val_f32 / scale_f32).round() as i32;
            (scale_f32, zero_point)
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

/// Exponential moving average calibration for vectors
///
/// # Arguments
///
/// * `vector` - Input vector to calibrate
/// * `bits` - Bit width for quantization
/// * `ema_factor` - Smoothing factor (0-1) where higher values give more weight to recent observations
/// * `max_iterations` - Maximum number of iterations for convergence
/// * `convergence_threshold` - Threshold for early stopping
/// * `symmetric` - Whether to use symmetric quantization
///
/// # Returns
///
/// * Calibrated quantization parameters
#[allow(dead_code)]
pub fn calibrate_vector_ema<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    ema_factor: f32,
    max_iterations: usize,
    convergence_threshold: f32,
    symmetric: bool,
) -> LinalgResult<QuantizationParams>
where
    F: scirs2_core::numeric::Float
        + Debug
        + scirs2_core::numeric::AsPrimitive<f32>
        + scirs2_core::numeric::FromPrimitive,
    f32: scirs2_core::numeric::AsPrimitive<F>,
{
    // Validate parameters
    if !(0.0..=1.0).contains(&ema_factor) {
        return Err(LinalgError::ValueError(
            "EMA factor must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Start with min-max calibration
    let (min_val_f32, max_val_f32) = find_min_max_vec(vector);

    // Convert min/max values back to type F
    let mut min_val = to_f::<F>(min_val_f32);
    let mut max_val = to_f::<F>(max_val_f32);

    // Convert vector to f32 for calculations
    let vector_f32 = vector.mapv(|x| x.as_());

    let mut prev_mse = f32::MAX;

    // Iterative refinement
    for _iter in 0..max_iterations {
        // Calculate scale and zero point
        let (scale, zero_point) = if symmetric {
            let abs_max = max_val.abs().max(min_val.abs());
            let abs_max_f32 = to_f32(abs_max);
            let scale_f32 = abs_max_f32 / ((1 << (bits - 1)) - 1) as f32;
            (scale_f32, 0)
        } else {
            let min_val_f32 = to_f32(min_val);
            let max_val_f32 = to_f32(max_val);
            let scale_f32 = (max_val_f32 - min_val_f32) / ((1 << bits) - 1) as f32;
            let zero_point = (-min_val_f32 / scale_f32).round() as i32;
            (scale_f32, zero_point)
        };

        // Simulate quantization
        let dequantized = simulate_quantization_vector_f32(
            &vector_f32.view(),
            scale,
            zero_point,
            bits,
            symmetric,
        );

        // Calculate MSE
        let mse = vector_f32
            .iter()
            .zip(dequantized.iter())
            .map(|(&orig, &deq)| (orig - deq).powi(2))
            .sum::<f32>()
            / vector_f32.len() as f32;

        // Check convergence
        if (prev_mse - mse).abs() < convergence_threshold {
            // Converged, return current parameters
            let min_val_f32 = to_f32(min_val);
            let max_val_f32 = to_f32(max_val);
            return create_params_from_range(bits, min_val_f32, max_val_f32, symmetric);
        }

        prev_mse = mse;

        // Update min/max values using error feedback
        if symmetric {
            // For symmetric quantization, adjust abs_max
            let abs_max = max_val.abs().max(min_val.abs());

            // Calculate error and update abs_max
            let mean_abs_error = vector_f32
                .iter()
                .zip(dequantized.iter())
                .map(|(&orig, &deq)| (orig - deq).abs())
                .sum::<f32>()
                / vector_f32.len() as f32;

            let scale_adjustment = if mean_abs_error > 0.01 {
                1.0 + mean_abs_error
            } else {
                1.0
            };

            // Apply EMA to abs_max - converting between F and f32 properly
            let abs_max_f32 = to_f32(abs_max);
            let new_abs_max_f32 = abs_max_f32 * scale_adjustment;
            let updated_abs_max_f32 =
                abs_max_f32 * (1.0 - ema_factor) + new_abs_max_f32 * ema_factor;
            let updated_abs_max = to_f::<F>(updated_abs_max_f32);

            // Update min/max
            min_val = -updated_abs_max;
            max_val = updated_abs_max;
        } else {
            // For asymmetric quantization, adjust min and max separately
            let negative_errors = vector_f32
                .iter()
                .zip(dequantized.iter())
                .filter_map(|(&orig, &deq)| {
                    if orig < deq {
                        Some((orig - deq).abs())
                    } else {
                        None
                    }
                })
                .sum::<f32>();

            let positive_errors = vector_f32
                .iter()
                .zip(dequantized.iter())
                .filter_map(|(&orig, &deq)| {
                    if orig > deq {
                        Some((orig - deq).abs())
                    } else {
                        None
                    }
                })
                .sum::<f32>();

            // Calculate adjustment factors
            let neg_count = vector_f32.iter().filter(|&&x| x < 0.0).count() as f32;
            let pos_count = vector_f32.iter().filter(|&&x| x > 0.0).count() as f32;

            let neg_adjustment = if neg_count > 0.0 {
                1.0 + (negative_errors / neg_count)
            } else {
                1.0
            };
            let pos_adjustment = if pos_count > 0.0 {
                1.0 + (positive_errors / pos_count)
            } else {
                1.0
            };

            // Apply EMA to min and max - converting properly between types
            let min_val_f32 = to_f32(min_val);
            let max_val_f32 = to_f32(max_val);

            let new_min_f32 = min_val_f32 * neg_adjustment;
            let new_max_f32 = max_val_f32 * pos_adjustment;

            let updated_min_f32 = min_val_f32 * (1.0 - ema_factor) + new_min_f32 * ema_factor;
            let updated_max_f32 = max_val_f32 * (1.0 - ema_factor) + new_max_f32 * ema_factor;

            min_val = to_f::<F>(updated_min_f32);
            max_val = to_f::<F>(updated_max_f32);
        }
    }

    // Return parameters from the final iteration
    let min_val_f32 = to_f32(min_val);
    let max_val_f32 = to_f32(max_val);
    create_params_from_range(bits, min_val_f32, max_val_f32, symmetric)
}

/// Simulate quantization for a vector (f32 version) using given scale and zero point
///
/// This is an overloaded version that takes f32 directly
#[allow(dead_code)]
fn simulate_quantization_vector_f32(
    vector: &ArrayView1<f32>,
    scale: f32,
    zero_point: i32,
    bits: u8,
    symmetric: bool,
) -> Array2<f32> {
    let mut result = Array2::zeros((vector.len(), 1));

    if symmetric {
        // Symmetric quantization
        let clamp_min = -(1 << (bits - 1)) as f32;
        let clamp_max = ((1 << (bits - 1)) - 1) as f32;

        for (i, &val) in vector.iter().enumerate() {
            let quantized = (val / scale).round().clamp(clamp_min, clamp_max);
            result[[i, 0]] = quantized * scale;
        }
    } else {
        // Affine quantization
        let clamp_max = ((1 << bits) - 1) as f32;
        let zero_point = zero_point as f32;

        for (i, &val) in vector.iter().enumerate() {
            let quantized = ((val / scale) + zero_point).round().clamp(0.0, clamp_max);
            result[[i, 0]] = (quantized - zero_point) * scale;
        }
    }

    result
}

/// Simulate quantization for a matrix using given quantization parameters
///
/// # Arguments
///
/// * `matrix` - Input matrix
/// * `params` - Quantization parameters
/// * `bits` - Bit width for quantization
///
/// # Returns
///
/// * Dequantized matrix after simulated quantization
#[allow(dead_code)]
fn simulate_quantization(
    matrix: &Array2<f32>,
    params: &QuantizationParams,
    bits: u8,
) -> Array2<f32> {
    match params.method {
        QuantizationMethod::Symmetric | QuantizationMethod::Int4 => {
            let scale = params.scale;
            let clamp_min = -(1 << (bits - 1)) as f32;
            let clamp_max = ((1 << (bits - 1)) - 1) as f32;

            matrix.mapv(|x| {
                let quantized = (x / scale).round().clamp(clamp_min, clamp_max);
                quantized * scale
            })
        }
        QuantizationMethod::Affine | QuantizationMethod::UInt4 => {
            let scale = params.scale;
            let zero_point = params.zero_point as f32;
            let clamp_max = ((1 << bits) - 1) as f32;

            matrix.mapv(|x| {
                let quantized = ((x / scale) + zero_point).round().clamp(0.0, clamp_max);
                (quantized - zero_point) * scale
            })
        }
        QuantizationMethod::PerChannelSymmetric | QuantizationMethod::PerChannelAffine => {
            // For per-channel quantization, we need to handle each column separately
            let mut result = Array2::zeros(matrix.dim());
            let (_, cols) = matrix.dim();

            if let Some(channel_scales) = &params.channel_scales {
                for col_idx in 0..cols {
                    let col_view = matrix.column(col_idx);
                    let scale = channel_scales[col_idx];

                    if params.method == QuantizationMethod::PerChannelSymmetric {
                        // Symmetric quantization
                        let clamp_min = -(1 << (bits - 1)) as f32;
                        let clamp_max = ((1 << (bits - 1)) - 1) as f32;

                        for (row_idx, &val) in col_view.iter().enumerate() {
                            let quantized = (val / scale).round().clamp(clamp_min, clamp_max);
                            result[[row_idx, col_idx]] = quantized * scale;
                        }
                    } else {
                        // Affine quantization (with zero point)
                        let clamp_max = ((1 << bits) - 1) as f32;
                        let zero_point =
                            params.channel_zero_points.as_ref().unwrap()[col_idx] as f32;

                        for (row_idx, &val) in col_view.iter().enumerate() {
                            let quantized =
                                ((val / scale) + zero_point).round().clamp(0.0, clamp_max);
                            result[[row_idx, col_idx]] = (quantized - zero_point) * scale;
                        }
                    }
                }
            }

            result
        }
        _ => {
            // For other methods (like float16, bfloat16, etc.), just return the original matrix
            // since we don't have simple simulation for these formats
            matrix.clone()
        }
    }
}
