//! Utility functions for quantization calibration
//!
//! This module contains helper functions used by both matrix and vector
//! calibration methods including min/max finding, histograms, KL divergence
//! optimization, and parameter creation.

use super::super::{QuantizationMethod, QuantizationParams};
use crate::error::LinalgResult;
use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use std::fmt::Debug;

// -------------------------------------------------------------------------
// Helper functions
// -------------------------------------------------------------------------

/// Find the minimum and maximum values in a matrix
#[allow(dead_code)]
pub fn find_min_max<F>(matrix: &ArrayView2<F>) -> (f32, f32)
where
    F: scirs2_core::numeric::Float + scirs2_core::numeric::AsPrimitive<f32>,
{
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
        min_val = 0.0;
        max_val = 1.0;
    }

    if min_val == max_val {
        min_val -= 1.0;
        max_val += 1.0;
    }

    (min_val, max_val)
}

/// Find the minimum and maximum values in a vector
#[allow(dead_code)]
pub fn find_min_max_vec<F>(vector: &ArrayView1<F>) -> (f32, f32)
where
    F: scirs2_core::numeric::Float + scirs2_core::numeric::AsPrimitive<f32>,
{
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
        min_val = 0.0;
        max_val = 1.0;
    }

    if min_val == max_val {
        min_val -= 1.0;
        max_val += 1.0;
    }

    (min_val, max_val)
}

/// Create a histogram of values from a matrix
#[allow(dead_code)]
pub fn create_histogram<F>(
    matrix: &ArrayView2<F>,
    min_val: f32,
    max_val: f32,
    num_bins: usize,
) -> Vec<usize>
where
    F: scirs2_core::numeric::Float + scirs2_core::numeric::AsPrimitive<f32>,
{
    let mut histogram = vec![0; num_bins];
    let bin_width = (max_val - min_val) / num_bins as f32;

    if bin_width == 0.0 {
        // All values are the same, put them all in the middle bin
        histogram[num_bins / 2] = matrix.len();
        return histogram;
    }

    for &_val in matrix.iter() {
        let val_f32 = _val.as_();
        if val_f32.is_finite() {
            let bin_idx = ((val_f32 - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(num_bins - 1); // Ensure we don't go out of bounds
            histogram[bin_idx] += 1;
        }
    }

    histogram
}

/// Create a histogram of values from a vector
#[allow(dead_code)]
pub fn create_histogram_vec<F>(
    vector: &ArrayView1<F>,
    min_val: f32,
    max_val: f32,
    num_bins: usize,
) -> Vec<usize>
where
    F: scirs2_core::numeric::Float + scirs2_core::numeric::AsPrimitive<f32>,
{
    let mut histogram = vec![0; num_bins];
    let bin_width = (max_val - min_val) / num_bins as f32;

    if bin_width == 0.0 {
        // All values are the same, put them all in the middle bin
        histogram[num_bins / 2] = vector.len();
        return histogram;
    }

    for &_val in vector.iter() {
        let val_f32 = _val.as_();
        if val_f32.is_finite() {
            let bin_idx = ((val_f32 - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(num_bins - 1); // Ensure we don't go out of bounds
            histogram[bin_idx] += 1;
        }
    }

    histogram
}

/// Optimize thresholds using KL divergence
#[allow(dead_code)]
pub fn optimize_thresholds_kl_divergence(
    histogram: &[usize],
    min_val: f32,
    max_val: f32,
    bits: u8,
    symmetric: bool,
) -> (f32, f32) {
    let num_bins = histogram.len();
    let bin_width = (max_val - min_val) / num_bins as f32;

    // Convert histogram to probability distribution
    let total_count = histogram.iter().sum::<usize>() as f32;
    let distribution: Vec<f32> = histogram
        .iter()
        .map(|&count| count as f32 / total_count)
        .collect();

    // Number of quantization levels
    let levels = if symmetric {
        (1 << (bits - 1)) as usize // For signed integers
    } else {
        (1 << bits) as usize // For unsigned integers
    };

    // For symmetric quantization, we want to find the optimal abs_max
    if symmetric {
        // Search for the optimal abs_max that minimizes KL divergence
        let mut best_abs_max = max_val.abs().max(min_val.abs());
        let mut min_kl = f32::MAX;

        // Try different abs_max values
        let step = (best_abs_max / 20.0).max(1e-6);
        for i in 0..40 {
            let abs_max = best_abs_max - 20.0 * step + i as f32 * step;
            if abs_max <= 0.0 {
                continue;
            }

            // Calculate quantization step
            let quantization_step = abs_max / (levels - 1) as f32;

            // Calculate KL divergence for this abs_max
            let kl = calculate_kl_divergence_symmetric(
                &distribution,
                min_val,
                max_val,
                bin_width,
                abs_max,
                quantization_step,
            );

            if kl < min_kl {
                min_kl = kl;
                best_abs_max = abs_max;
            }
        }

        // Return symmetric range
        (-best_abs_max, best_abs_max)
    } else {
        // For asymmetric quantization, find the best min and max
        let mut best_min = min_val;
        let mut best_max = max_val;
        let mut min_kl = f32::MAX;

        // Grid search for optimal min/max
        let min_step = (max_val - min_val) / 40.0;
        let max_step = min_step;

        for i in 0..10 {
            let trial_min = min_val + i as f32 * min_step;

            for j in 0..10 {
                let trial_max = max_val - j as f32 * max_step;

                if trial_min >= trial_max {
                    continue;
                }

                // Calculate quantization step
                let quantization_step = (trial_max - trial_min) / (levels - 1) as f32;

                // Calculate KL divergence for this range
                let kl = calculate_kl_divergence_asymmetric(
                    &distribution,
                    min_val,
                    max_val,
                    bin_width,
                    trial_min,
                    trial_max,
                    quantization_step,
                );

                if kl < min_kl {
                    min_kl = kl;
                    best_min = trial_min;
                    best_max = trial_max;
                }
            }
        }

        (best_min, best_max)
    }
}

/// Calculate KL divergence for symmetric quantization
#[allow(dead_code)]
fn calculate_kl_divergence_symmetric(
    distribution: &[f32],
    min_val: f32,
    _max_val: f32,
    bin_width: f32,
    abs_max: f32,
    quantization_step: f32,
) -> f32 {
    let num_bins = distribution.len();

    // Create quantized probability distribution
    let mut quantized_dist = vec![0.0; num_bins];

    for (bin_idx, &prob) in distribution.iter().enumerate() {
        // Original value at the center of this bin
        let orig_val = min_val + (bin_idx as f32 + 0.5) * bin_width;

        // Quantize the value
        let quantized_val = if orig_val > abs_max {
            abs_max
        } else if orig_val < -abs_max {
            -abs_max
        } else {
            // Round to nearest quantization step
            (orig_val / quantization_step).round() * quantization_step
        };

        // Map back to bin index
        let new_bin_idx = ((quantized_val - min_val) / bin_width).floor() as i32;

        if new_bin_idx >= 0 && new_bin_idx < num_bins as i32 {
            quantized_dist[new_bin_idx as usize] += prob;
        }
    }

    // Calculate KL divergence: sum(p * log(p / q))
    let mut kl = 0.0;
    for (i, &p) in distribution.iter().enumerate() {
        if p > 0.0 {
            let q = quantized_dist[i].max(1e-10); // Avoid division by zero
            kl += p * (p / q).ln();
        }
    }

    kl
}

/// Calculate KL divergence for asymmetric quantization
#[allow(dead_code)]
fn calculate_kl_divergence_asymmetric(
    distribution: &[f32],
    min_val: f32,
    _max_val: f32,
    bin_width: f32,
    quant_min: f32,
    quant_max: f32,
    quantization_step: f32,
) -> f32 {
    let num_bins = distribution.len();

    // Create quantized probability distribution
    let mut quantized_dist = vec![0.0; num_bins];

    for (bin_idx, &prob) in distribution.iter().enumerate() {
        // Original value at the center of this bin
        let orig_val = min_val + (bin_idx as f32 + 0.5) * bin_width;

        // Quantize the value
        let quantized_val = if orig_val > quant_max {
            quant_max
        } else if orig_val < quant_min {
            quant_min
        } else {
            // Round to nearest quantization step
            let steps = ((orig_val - quant_min) / quantization_step).round();
            quant_min + steps * quantization_step
        };

        // Map back to bin index
        let new_bin_idx = ((quantized_val - min_val) / bin_width).floor() as i32;

        if new_bin_idx >= 0 && new_bin_idx < num_bins as i32 {
            quantized_dist[new_bin_idx as usize] += prob;
        }
    }

    // Calculate KL divergence: sum(p * log(p / q))
    let mut kl = 0.0;
    for (i, &p) in distribution.iter().enumerate() {
        if p > 0.0 {
            let q = quantized_dist[i].max(1e-10); // Avoid division by zero
            kl += p * (p / q).ln();
        }
    }

    kl
}

/// Optimize symmetric scale factor using MSE
#[allow(dead_code)]
pub fn optimize_symmetric_scale<F>(matrix: &ArrayView2<F>, bits: u8, basescale: f32) -> f32
where
    F: scirs2_core::numeric::Float
        + Debug
        + scirs2_core::numeric::AsPrimitive<f32>
        + scirs2_core::numeric::FromPrimitive,
    f32: scirs2_core::numeric::AsPrimitive<F>,
{
    let num_trials = 20;
    let scales: Vec<f32> = (0..num_trials)
        .map(|i| {
            let factor = 0.5 + 1.5 * (i as f32 / (num_trials - 1) as f32);
            basescale * factor
        })
        .collect();

    let mut best_scale = basescale;
    let mut min_mse = f32::MAX;

    // Test each scale factor
    for &scale in &scales {
        // Create temporary quantization parameters
        let abs_max = matrix
            .mapv(|x| x.as_().abs())
            .fold(0.0, |a: f32, &b| a.max(b));
        let params = QuantizationParams {
            bits,
            scale,
            zero_point: 0,
            min_val: -abs_max,
            max_val: abs_max,
            method: if bits == 4 {
                QuantizationMethod::Int4
            } else {
                QuantizationMethod::Symmetric
            },
            data_type: determine_data_type(bits),
            channel_scales: None,
            channel_zero_points: None,
        };

        // Manually simulate quantization and dequantization for F type
        let matrix_f32 = matrix.mapv(|x| x.as_());
        let current_scale = params.scale;
        let dequantized = matrix_f32.mapv(|x| {
            let quantized = (x / scale)
                .round()
                .clamp(-(1 << (bits - 1)) as f32, ((1 << (bits - 1)) - 1) as f32);
            quantized * current_scale
        });

        // Calculate MSE
        let mse = (&matrix_f32 - &dequantized).mapv(|x| x * x).sum() / matrix.len() as f32;

        if mse < min_mse {
            min_mse = mse;
            best_scale = scale;
        }
    }

    best_scale
}

/// Optimize symmetric scale factor for vectors using MSE
#[allow(dead_code)]
pub fn optimize_symmetric_scale_vec<F>(_vector: &ArrayView1<F>, bits: u8, basescale: f32) -> f32
where
    F: scirs2_core::numeric::Float
        + Debug
        + scirs2_core::numeric::AsPrimitive<f32>
        + scirs2_core::numeric::FromPrimitive,
    f32: scirs2_core::numeric::AsPrimitive<F>,
{
    let num_trials = 20;
    let scales: Vec<f32> = (0..num_trials)
        .map(|i| {
            let factor = 0.5 + 1.5 * (i as f32 / (num_trials - 1) as f32);
            basescale * factor
        })
        .collect();

    let mut best_scale = basescale;
    let mut min_mse = f32::MAX;

    // Test each scale factor
    for &scale in &scales {
        // Create temporary QuantizationParams
        let abs_max = _vector
            .mapv(|x| x.as_().abs())
            .fold(0.0, |a: f32, &b| a.max(b));
        let params = QuantizationParams {
            bits,
            scale,
            zero_point: 0,
            min_val: -abs_max,
            max_val: abs_max,
            method: if bits == 4 {
                QuantizationMethod::Int4
            } else {
                QuantizationMethod::Symmetric
            },
            data_type: determine_data_type(bits),
            channel_scales: None,
            channel_zero_points: None,
        };

        // Manually simulate quantization and dequantization for F type
        let vector_f32 = _vector.mapv(|x| x.as_());
        let current_scale = params.scale;
        let dequantized = vector_f32.mapv(|x| {
            let quantized = (x / scale)
                .round()
                .clamp(-(1 << (bits - 1)) as f32, ((1 << (bits - 1)) - 1) as f32);
            quantized * current_scale
        });

        // Calculate MSE
        let mse = (&vector_f32 - &dequantized).mapv(|x| x * x).sum() / _vector.len() as f32;

        if mse < min_mse {
            min_mse = mse;
            best_scale = scale;
        }
    }

    best_scale
}

/// Optimize affine quantization parameters (scale and zero point) using MSE
#[allow(dead_code)]
pub fn optimize_affine_params<F>(
    matrix: &ArrayView2<F>,
    bits: u8,
    base_scale: f32,
    base_zero_point: i32,
) -> (f32, i32)
where
    F: scirs2_core::numeric::Float
        + Debug
        + scirs2_core::numeric::AsPrimitive<f32>
        + scirs2_core::numeric::FromPrimitive,
    f32: scirs2_core::numeric::AsPrimitive<F>,
{
    let num_scale_trials = 10;
    let num_zp_trials = 5;

    let scales: Vec<f32> = (0..num_scale_trials)
        .map(|i| {
            let factor = 0.8 + 0.4 * (i as f32 / (num_scale_trials - 1) as f32);
            base_scale * factor
        })
        .collect();

    let zero_points: Vec<i32> = (0..num_zp_trials)
        .map(|i| {
            let offset = -2 + i;
            base_zero_point + offset
        })
        .collect();

    let mut best_scale = base_scale;
    let mut best_zero_point = base_zero_point;
    let mut min_mse = f32::MAX;

    // Test each combination of scale and zero point
    for &_scale in &scales {
        for &zero_point in &zero_points {
            // Create temporary QuantizationParams
            let mut params = QuantizationParams {
                bits,
                scale: _scale,
                zero_point,
                min_val: 0.0, // Will be set by quantize_matrix
                max_val: 0.0, // Will be set by quantize_matrix
                method: QuantizationMethod::Affine,
                data_type: determine_data_type(bits),
                channel_scales: None,
                channel_zero_points: None,
            };

            // Manually simulate affine quantization and dequantization for F type
            let matrix_f32 = matrix.mapv(|x| x.as_());
            let scale = params.scale;
            let zero_point = params.zero_point;

            // Find min/max values for the matrix
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            for &val in matrix_f32.iter() {
                if val.is_finite() {
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
            }
            params.min_val = min_val;
            params.max_val = max_val;

            let dequantized = matrix_f32.mapv(|x| {
                let quantized = ((x / scale) + zero_point as f32)
                    .round()
                    .clamp(0.0, ((1 << bits) - 1) as f32);
                (quantized - zero_point as f32) * scale
            });

            // Calculate MSE
            let mse = (&matrix_f32 - &dequantized).mapv(|x| x * x).sum() / matrix.len() as f32;

            if mse < min_mse {
                min_mse = mse;
                best_scale = scale;
                best_zero_point = zero_point;
            }
        }
    }

    (best_scale, best_zero_point)
}

/// Optimize affine quantization parameters for vectors using MSE
#[allow(dead_code)]
pub fn optimize_affine_params_vec<F>(
    vector: &ArrayView1<F>,
    bits: u8,
    base_scale: f32,
    base_zero_point: i32,
) -> (f32, i32)
where
    F: scirs2_core::numeric::Float
        + Debug
        + scirs2_core::numeric::AsPrimitive<f32>
        + scirs2_core::numeric::FromPrimitive,
    f32: scirs2_core::numeric::AsPrimitive<F>,
{
    let num_scale_trials = 10;
    let num_zp_trials = 5;

    let scales: Vec<f32> = (0..num_scale_trials)
        .map(|i| {
            let factor = 0.8 + 0.4 * (i as f32 / (num_scale_trials - 1) as f32);
            base_scale * factor
        })
        .collect();

    let zero_points: Vec<i32> = (0..num_zp_trials)
        .map(|i| {
            let offset = -2 + i;
            base_zero_point + offset
        })
        .collect();

    let mut best_scale = base_scale;
    let mut best_zero_point = base_zero_point;
    let mut min_mse = f32::MAX;

    // Test each combination of scale and zero point
    for &_scale in &scales {
        for &zero_point in &zero_points {
            // Create temporary QuantizationParams
            let mut params = QuantizationParams {
                bits,
                scale: _scale,
                zero_point,
                min_val: 0.0, // Will be set by quantize_vector
                max_val: 0.0, // Will be set by quantize_vector
                method: QuantizationMethod::Affine,
                data_type: determine_data_type(bits),
                channel_scales: None,
                channel_zero_points: None,
            };

            // Manually simulate affine quantization and dequantization for F type
            let vector_f32 = vector.mapv(|x| x.as_());
            let scale = params.scale;
            let zero_point = params.zero_point;

            // Find min/max values for the vector
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            for &val in vector_f32.iter() {
                if val.is_finite() {
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
            }
            params.min_val = min_val;
            params.max_val = max_val;

            let dequantized = vector_f32.mapv(|x| {
                let quantized = ((x / scale) + zero_point as f32)
                    .round()
                    .clamp(0.0, ((1 << bits) - 1) as f32);
                (quantized - zero_point as f32) * scale
            });

            // Calculate MSE
            let mse = (&vector_f32 - &dequantized).mapv(|x| x * x).sum() / vector.len() as f32;

            if mse < min_mse {
                min_mse = mse;
                best_scale = scale;
                best_zero_point = zero_point;
            }
        }
    }

    (best_scale, best_zero_point)
}

/// Create QuantizationParams from a min-max range
#[allow(dead_code)]
pub fn create_params_from_range(
    bits: u8,
    min_val: f32,
    max_val: f32,
    symmetric: bool,
) -> LinalgResult<QuantizationParams> {
    let (method, scale, zero_point) = if symmetric {
        let abs_max = max_val.abs().max(min_val.abs());
        let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;
        (QuantizationMethod::Symmetric, scale, 0)
    } else {
        let method = QuantizationMethod::Affine;
        let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;
        let zero_point = (-min_val / scale).round() as i32;
        (method, scale, zero_point)
    };

    Ok(QuantizationParams {
        bits,
        scale,
        zero_point,
        min_val,
        max_val,
        method,
        data_type: determine_data_type(bits),
        channel_scales: None,
        channel_zero_points: None,
    })
}

/// Determine the appropriate data type based on bit width
#[allow(dead_code)]
pub fn determine_data_type(bits: u8) -> super::super::QuantizedDataType {
    use super::super::QuantizedDataType;

    match bits {
        4 => QuantizedDataType::Int4,     // Default to Int4 for 4-bit
        8 => QuantizedDataType::Int8,     // Default to Int8 for 8-bit
        16 => QuantizedDataType::Float16, // Default to Float16 for 16-bit
        _ => QuantizedDataType::Int8,     // Default to Int8 for other cases
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_find_min_max() {
        let matrix = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let (min_val, max_val) = find_min_max(&matrix.view());
        assert_eq!(min_val, 1.0);
        assert_eq!(max_val, 9.0);
    }

    #[test]
    fn test_find_min_max_vec() {
        let vector = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let (min_val, max_val) = find_min_max_vec(&vector.view());
        assert_eq!(min_val, 1.0);
        assert_eq!(max_val, 5.0);
    }

    #[test]
    fn test_create_histogram() {
        let matrix = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let histogram = create_histogram(&matrix.view(), 1.0, 6.0, 5);

        // Each value should be in a different bin
        assert_eq!(histogram.iter().sum::<usize>(), 6); // Total count
        assert!(histogram.iter().all(|&count| count <= 2)); // No bin has more than 2 values
    }

    #[test]
    fn test_create_params_from_range() {
        // Test symmetric quantization
        let params = create_params_from_range(8, -5.0, 5.0, true).unwrap();
        assert_eq!(params.method, QuantizationMethod::Symmetric);
        assert_eq!(params.zero_point, 0);
        assert_relative_eq!(params.scale, 5.0 / 127.0, epsilon = 1e-6);

        // Test affine quantization
        let params = create_params_from_range(8, 1.0, 9.0, false).unwrap();
        assert_eq!(params.method, QuantizationMethod::Affine);
        assert_relative_eq!(params.scale, 8.0 / 255.0, epsilon = 1e-6);
        assert_eq!(params.zero_point, (-1.0 / params.scale).round() as i32);
    }
}
