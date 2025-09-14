//! Bilateral filtering functions for edge-preserving smoothing
//!
//! Bilateral filters smooth images while preserving edges by considering both
//! spatial distance and intensity difference when computing weights.

use ndarray::{Array, Array1, Array2, Dimension, Ix2, IxDyn};
use num_traits::{Float, FromPrimitive};
use scirs2_core::validation::check_positive;
use std::fmt::{Debug, Display};

#[cfg(feature = "simd")]
use scirs2_core::simd::{simd_add_f32, simd_add_f64, simd_scalar_mul_f32, simd_scalar_mul_f64};

use super::{pad_array, BorderMode};
use crate::error::{NdimageError, NdimageResult};
use crate::utils::safe_f64_to_float;

/// Helper function for safe i32 conversion
#[allow(dead_code)]
fn safe_i32_to_float<T: Float + FromPrimitive>(value: i32) -> NdimageResult<T> {
    T::from_i32(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert i32 {} to float type", value))
    })
}

/// Helper function for safe float to usize conversion
#[allow(dead_code)]
fn safe_float_to_usize<T: Float>(value: T) -> NdimageResult<usize> {
    value.to_usize().ok_or_else(|| {
        NdimageError::ComputationError("Failed to convert float to usize".to_string())
    })
}

/// Apply a bilateral filter to preserve edges while smoothing
///
/// The bilateral filter is an edge-preserving smoothing filter that considers
/// both spatial distance and intensity difference when computing weights.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `sigma_spatial` - Standard deviation for spatial Gaussian kernel
/// * `sigma_color` - Standard deviation for intensity difference Gaussian kernel  
/// * `mode` - Border handling mode (defaults to Reflect)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
#[allow(dead_code)]
pub fn bilateral_filter<T, D>(
    input: &Array<T, D>,
    sigma_spatial: T,
    sigma_color: T,
    mode: Option<BorderMode>,
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + Debug
        + Clone
        + Send
        + Sync
        + Display
        + FromPrimitive
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
    D: Dimension + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

    // Validate inputs
    check_positive(sigma_spatial, "sigma_spatial").map_err(NdimageError::from)?;
    check_positive(sigma_color, "sigma_color").map_err(NdimageError::from)?;

    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    // Handle scalar or constant case
    if input.len() <= 1 {
        return Ok(input.to_owned());
    }

    // Dispatch to the appropriate implementation based on dimensionality
    match input.ndim() {
        1 => bilateral_filter_1d(input, sigma_spatial, sigma_color, &border_mode),
        2 => bilateral_filter_2d(input, sigma_spatial, sigma_color, &border_mode),
        _ => bilateral_filter_nd(input, sigma_spatial, sigma_color, &border_mode),
    }
}

/// Apply bilateral filter to a 1D array with SIMD optimization for f32/f64
#[allow(dead_code)]
fn bilateral_filter_1d<T, D>(
    input: &Array<T, D>,
    sigma_spatial: T,
    sigma_color: T,
    mode: &BorderMode,
) -> NdimageResult<Array<T, D>>
where
    T: Float + Debug + Clone + Display + FromPrimitive,
    D: Dimension + 'static,
{
    // Convert to 1D for processing
    let input_1d = input
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert to 1D array".into()))?;

    // Calculate kernel radius based on _spatial sigma
    let three = safe_f64_to_float::<T>(3.0)?;
    let radius = safe_float_to_usize((sigma_spatial * three).ceil()).unwrap_or(3);
    let kernel_size = 2 * radius + 1;

    // Create output array
    let mut output = Array1::zeros(input_1d.len());

    // Pad input for border handling
    let pad_width = vec![(radius, radius)];
    let padded_input = pad_array(&input_1d, &pad_width, mode, None)?;

    // Precompute _spatial weights
    let mut spatial_weights = Array1::zeros(kernel_size);
    let two = safe_f64_to_float::<T>(2.0)?;
    let two_sigma_spatial_sq = two * sigma_spatial * sigma_spatial;

    for k in 0..kernel_size {
        let dist: T = safe_i32_to_float((k as i32) - (radius as i32))?;
        spatial_weights[k] = (-dist * dist / two_sigma_spatial_sq).exp();
    }

    let two_sigma_color_sq = two * sigma_color * sigma_color;

    // Apply bilateral filter to each position
    for i in 0..input_1d.len() {
        let center = i + radius;
        let center_value = padded_input[center];

        let mut weighted_sum = T::zero();
        let mut weight_sum = T::zero();

        for k in 0..kernel_size {
            let neighbor_value = padded_input[center - radius + k];
            let color_diff = neighbor_value - center_value;
            let color_weight = (-color_diff * color_diff / two_sigma_color_sq).exp();

            let total_weight = spatial_weights[k] * color_weight;
            weighted_sum = weighted_sum + neighbor_value * total_weight;
            weight_sum = weight_sum + total_weight;
        }

        output[i] = if weight_sum > T::zero() {
            weighted_sum / weight_sum
        } else {
            center_value
        };
    }

    // Convert back to original dimensionality
    output
        .into_dimensionality::<D>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert back from 1D array".into()))
}

/// Apply bilateral filter to a 2D array
#[allow(dead_code)]
fn bilateral_filter_2d<T, D>(
    input: &Array<T, D>,
    sigma_spatial: T,
    sigma_color: T,
    mode: &BorderMode,
) -> NdimageResult<Array<T, D>>
where
    T: Float + Debug + Clone + Display + FromPrimitive,
    D: Dimension + 'static,
{
    // Convert to 2D for processing
    let input_2d = input
        .to_owned()
        .into_dimensionality::<Ix2>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D array".into()))?;

    let (rows, cols) = input_2d.dim();

    // Calculate kernel radius based on _spatial sigma
    let three = safe_f64_to_float::<T>(3.0)?;
    let radius = safe_float_to_usize((sigma_spatial * three).ceil()).unwrap_or(3);
    let kernel_size = 2 * radius + 1;

    // Create output array
    let mut output = Array2::zeros((rows, cols));

    // Pad input for border handling
    let pad_width = vec![(radius, radius), (radius, radius)];
    let padded_input = pad_array(&input_2d, &pad_width, mode, None)?;

    // Precompute _spatial weights
    let mut spatial_weights = Array2::zeros((kernel_size, kernel_size));
    let two = safe_f64_to_float::<T>(2.0)?;
    let two_sigma_spatial_sq = two * sigma_spatial * sigma_spatial;

    for dy in 0..kernel_size {
        for dx in 0..kernel_size {
            let y_dist: T = safe_i32_to_float((dy as i32) - (radius as i32))?;
            let x_dist: T = safe_i32_to_float((dx as i32) - (radius as i32))?;
            let spatial_dist_sq: T = y_dist * y_dist + x_dist * x_dist;
            spatial_weights[[dy, dx]] = (-spatial_dist_sq / two_sigma_spatial_sq).exp();
        }
    }

    let two_sigma_color_sq = two * sigma_color * sigma_color;

    // Apply bilateral filter to each position
    for i in 0..rows {
        for j in 0..cols {
            let center_y = i + radius;
            let center_x = j + radius;
            let center_value = padded_input[[center_y, center_x]];

            let mut weighted_sum = T::zero();
            let mut weight_sum = T::zero();

            for dy in 0..kernel_size {
                for dx in 0..kernel_size {
                    let y = center_y - radius + dy;
                    let x = center_x - radius + dx;
                    let neighbor_value = padded_input[[y, x]];

                    let color_diff = neighbor_value - center_value;
                    let color_weight = (-color_diff * color_diff / two_sigma_color_sq).exp();

                    let total_weight = spatial_weights[[dy, dx]] * color_weight;
                    weighted_sum = weighted_sum + neighbor_value * total_weight;
                    weight_sum = weight_sum + total_weight;
                }
            }

            output[[i, j]] = if weight_sum > T::zero() {
                weighted_sum / weight_sum
            } else {
                center_value
            };
        }
    }

    // Convert back to original dimensionality
    output
        .into_dimensionality::<D>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert back from 2D array".into()))
}

/// Apply bilateral filter to an n-dimensional array
#[allow(dead_code)]
fn bilateral_filter_nd<T, D>(
    input: &Array<T, D>,
    sigma_spatial: T,
    sigma_color: T,
    mode: &BorderMode,
) -> NdimageResult<Array<T, D>>
where
    T: Float + Debug + Clone + Display + FromPrimitive,
    D: Dimension + 'static,
{
    // Calculate kernel radius based on _spatial sigma
    let three = safe_f64_to_float::<T>(3.0)?;
    let radius = safe_float_to_usize((sigma_spatial * three).ceil()).unwrap_or(3);

    // Convert to dynamic dimension for easier processing
    let input_dyn = input.clone().into_dyn();
    let mut output_dyn = Array::<T, IxDyn>::zeros(input_dyn.raw_dim());

    // Prepare padding
    let pad_width: Vec<(usize, usize)> = (0..input.ndim()).map(|_| (radius, radius)).collect();
    let padded_input = pad_array(&input_dyn, &pad_width, mode, None)?;

    let two = safe_f64_to_float::<T>(2.0)?;
    let two_sigma_spatial_sq = two * sigma_spatial * sigma_spatial;
    let two_sigma_color_sq = two * sigma_color * sigma_color;

    // Process each position in the output array
    for (idx, output_val) in output_dyn.indexed_iter_mut() {
        let idx_vec = idx.as_array_view().to_vec();

        // Calculate center position in padded array
        let center_idx: Vec<usize> = idx_vec.iter().map(|&i| i + radius).collect();
        let center_value = padded_input[&IxDyn(&center_idx)];

        let mut weighted_sum = T::zero();
        let mut weight_sum = T::zero();

        // Iterate through neighborhood
        for offset in -(radius as i32)..=(radius as i32) {
            let neighbor_idx = center_idx[0] as i32 + offset;
            if neighbor_idx >= 0 && neighbor_idx < padded_input.len() as i32 {
                let neighbor_value = padded_input[neighbor_idx as usize];

                // Calculate spatial weight
                let dist: T = safe_i32_to_float(offset)?;
                let spatial_dist_sq = dist * dist;
                let spatial_weight = (-spatial_dist_sq / two_sigma_spatial_sq).exp();

                // Calculate color weight
                let color_diff = neighbor_value - center_value;
                let color_weight = (-color_diff * color_diff / two_sigma_color_sq).exp();

                let total_weight = spatial_weight * color_weight;
                weighted_sum = weighted_sum + neighbor_value * total_weight;
                weight_sum = weight_sum + total_weight;
            }
        }

        *output_val = if weight_sum > T::zero() {
            weighted_sum / weight_sum
        } else {
            center_value
        };
    }

    // Convert back to original dimensionality
    output_dyn.into_dimensionality::<D>().map_err(|_| {
        NdimageError::DimensionError("Failed to convert back to original dimensions".into())
    })
}

/// Helper function to iterate through neighborhood offsets
#[allow(dead_code)]
fn iterate_neighborhood<F>(_centeridx: &[usize], radius: usize, ndim: usize, mut callback: F)
where
    F: FnMut(&[i32]),
{
    let _kernel_size = 2 * radius + 1;
    let mut offsets = vec![0i32; ndim];

    // Initialize all offsets to -radius
    for offset in &mut offsets {
        *offset = -(radius as i32);
    }

    // Iterate through all combinations
    loop {
        callback(&offsets);

        // Increment to next combination
        let mut carry = true;
        for i in (0..ndim).rev() {
            if carry {
                offsets[i] += 1;
                if offsets[i] < (radius as i32) + 1 {
                    carry = false;
                } else {
                    offsets[i] = -(radius as i32);
                }
            }
        }

        if carry {
            break; // All combinations exhausted
        }
    }
}

/// SIMD-accelerated bilateral filter for f32 arrays
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn bilateral_filter_simd_f32<D>(
    input: &Array<f32, D>,
    sigma_spatial: f32,
    sigma_color: f32,
    mode: Option<BorderMode>,
) -> NdimageResult<Array<f32, D>>
where
    D: Dimension + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

    // Validate inputs
    check_positive(sigma_spatial, "sigma_spatial").map_err(NdimageError::from)?;
    check_positive(sigma_color, "sigma_color").map_err(NdimageError::from)?;

    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if input.len() <= 1 {
        return Ok(input.to_owned());
    }

    match input.ndim() {
        1 => bilateral_filter_1d_simd_f32(input, sigma_spatial, sigma_color, &border_mode),
        2 => bilateral_filter_2d_simd_f32(input, sigma_spatial, sigma_color, &border_mode),
        _ => bilateral_filter(input, sigma_spatial, sigma_color, mode), // Fall back to regular implementation
    }
}

/// SIMD-accelerated bilateral filter for f64 arrays
#[cfg(feature = "simd")]
#[allow(dead_code)]
pub fn bilateral_filter_simd_f64<D>(
    input: &Array<f64, D>,
    sigma_spatial: f64,
    sigma_color: f64,
    mode: Option<BorderMode>,
) -> NdimageResult<Array<f64, D>>
where
    D: Dimension + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

    // Validate inputs
    check_positive(sigma_spatial, "sigma_spatial").map_err(NdimageError::from)?;
    check_positive(sigma_color, "sigma_color").map_err(NdimageError::from)?;

    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if input.len() <= 1 {
        return Ok(input.to_owned());
    }

    match input.ndim() {
        1 => bilateral_filter_1d_simd_f64(input, sigma_spatial, sigma_color, &border_mode),
        2 => bilateral_filter_2d_simd_f64(input, sigma_spatial, sigma_color, &border_mode),
        _ => bilateral_filter(input, sigma_spatial, sigma_color, mode), // Fall back to regular implementation
    }
}

/// SIMD-accelerated 1D bilateral filter for f32
#[cfg(feature = "simd")]
#[allow(dead_code)]
fn bilateral_filter_1d_simd_f32<D>(
    input: &Array<f32, D>,
    sigma_spatial: f32,
    sigma_color: f32,
    mode: &BorderMode,
) -> NdimageResult<Array<f32, D>>
where
    D: Dimension + 'static,
{
    let input_1d = input
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert to 1D array".into()))?;

    let radius = ((sigma_spatial * 3.0).ceil() as usize).max(1);
    let kernel_size = 2 * radius + 1;

    let mut output = Array1::zeros(input_1d.len());
    let pad_width = vec![(radius, radius)];
    let padded_input = pad_array(&input_1d, &pad_width, mode, None)?;

    // Precompute _spatial weights
    let mut spatial_weights = Array1::zeros(kernel_size);
    let two_sigma_spatial_sq = 2.0 * sigma_spatial * sigma_spatial;

    for k in 0..kernel_size {
        let dist = (k as i32 - radius as i32) as f32;
        spatial_weights[k] = (-dist * dist / two_sigma_spatial_sq).exp();
    }

    let two_sigma_color_sq = 2.0 * sigma_color * sigma_color;

    // Process in SIMD-friendly chunks
    for j in 0..input_1d.len() {
        let center = j + radius;
        let center_value = padded_input[center];

        let mut weighted_sum = 0.0f32;
        let mut weight_sum = 0.0f32;

        // Use SIMD for chunks of the kernel when possible
        let mut k = 0;
        while k + 4 <= kernel_size {
            let neighbor_values = Array1::from_vec(vec![
                padded_input[center - radius + k],
                padded_input[center - radius + k + 1],
                padded_input[center - radius + k + 2],
                padded_input[center - radius + k + 3],
            ]);

            let center_values = Array1::from_elem(4, center_value);
            let color_diffs = simd_add_f32(
                &neighbor_values.view(),
                &simd_scalar_mul_f32(&center_values.view(), -1.0).view(),
            );

            // Calculate weights and accumulate
            for (idx, &diff) in color_diffs.iter().enumerate() {
                let color_weight = (-diff * diff / two_sigma_color_sq).exp();
                let total_weight = spatial_weights[k + idx] * color_weight;
                weighted_sum += neighbor_values[idx] * total_weight;
                weight_sum += total_weight;
            }
            k += 4;
        }

        // Process remaining elements
        for kk in k..kernel_size {
            let neighbor_value = padded_input[center - radius + kk];
            let color_diff = neighbor_value - center_value;
            let color_weight = (-color_diff * color_diff / two_sigma_color_sq).exp();
            let total_weight = spatial_weights[kk] * color_weight;
            weighted_sum += neighbor_value * total_weight;
            weight_sum += total_weight;
        }

        output[j] = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            center_value
        };
    }

    output
        .into_dimensionality::<D>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert back from 1D array".into()))
}

/// SIMD-accelerated 1D bilateral filter for f64
#[cfg(feature = "simd")]
#[allow(dead_code)]
fn bilateral_filter_1d_simd_f64<D>(
    input: &Array<f64, D>,
    sigma_spatial: f64,
    sigma_color: f64,
    mode: &BorderMode,
) -> NdimageResult<Array<f64, D>>
where
    D: Dimension + 'static,
{
    let input_1d = input
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert to 1D array".into()))?;

    let radius = ((sigma_spatial * 3.0).ceil() as usize).max(1);
    let kernel_size = 2 * radius + 1;

    let mut output = Array1::zeros(input_1d.len());
    let pad_width = vec![(radius, radius)];
    let padded_input = pad_array(&input_1d, &pad_width, mode, None)?;

    let mut spatial_weights = Array1::zeros(kernel_size);
    let two_sigma_spatial_sq = 2.0 * sigma_spatial * sigma_spatial;

    for k in 0..kernel_size {
        let dist = (k as i32 - radius as i32) as f64;
        spatial_weights[k] = (-dist * dist / two_sigma_spatial_sq).exp();
    }

    let two_sigma_color_sq = 2.0 * sigma_color * sigma_color;

    for j in 0..input_1d.len() {
        let center = j + radius;
        let center_value = padded_input[center];

        let mut weighted_sum = 0.0f64;
        let mut weight_sum = 0.0f64;

        let mut k = 0;
        while k + 4 <= kernel_size {
            let neighbor_values = Array1::from_vec(vec![
                padded_input[center - radius + k],
                padded_input[center - radius + k + 1],
                padded_input[center - radius + k + 2],
                padded_input[center - radius + k + 3],
            ]);

            let center_values = Array1::from_elem(4, center_value);
            let color_diffs = simd_add_f64(
                &neighbor_values.view(),
                &simd_scalar_mul_f64(&center_values.view(), -1.0).view(),
            );

            for (idx, &diff) in color_diffs.iter().enumerate() {
                let color_weight = (-diff * diff / two_sigma_color_sq).exp();
                let total_weight = spatial_weights[k + idx] * color_weight;
                weighted_sum += neighbor_values[idx] * total_weight;
                weight_sum += total_weight;
            }
            k += 4;
        }

        // Process remaining elements
        for kk in k..kernel_size {
            let neighbor_value = padded_input[center - radius + kk];
            let color_diff = neighbor_value - center_value;
            let color_weight = (-color_diff * color_diff / two_sigma_color_sq).exp();
            let total_weight = spatial_weights[kk] * color_weight;
            weighted_sum += neighbor_value * total_weight;
            weight_sum += total_weight;
        }

        output[j] = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            center_value
        };
    }

    output
        .into_dimensionality::<D>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert back from 1D array".into()))
}

/// SIMD-accelerated 2D bilateral filter for f32
#[cfg(feature = "simd")]
#[allow(dead_code)]
fn bilateral_filter_2d_simd_f32<D>(
    input: &Array<f32, D>,
    sigma_spatial: f32,
    sigma_color: f32,
    mode: &BorderMode,
) -> NdimageResult<Array<f32, D>>
where
    D: Dimension + 'static,
{
    let input_2d = input
        .to_owned()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D array".into()))?;

    let (rows, cols) = input_2d.dim();
    let radius = ((sigma_spatial * 3.0).ceil() as usize).max(1);
    let kernel_size = 2 * radius + 1;

    let mut output = Array2::zeros((rows, cols));
    let pad_width = vec![(radius, radius), (radius, radius)];
    let padded_input = pad_array(&input_2d, &pad_width, mode, None)?;

    // Precompute _spatial weights
    let mut spatial_weights = Array2::zeros((kernel_size, kernel_size));
    let two_sigma_spatial_sq = 2.0 * sigma_spatial * sigma_spatial;

    for dy in 0..kernel_size {
        for dx in 0..kernel_size {
            let y_dist = (dy as i32 - radius as i32) as f32;
            let x_dist = (dx as i32 - radius as i32) as f32;
            let spatial_dist_sq = y_dist * y_dist + x_dist * x_dist;
            spatial_weights[[dy, dx]] = (-spatial_dist_sq / two_sigma_spatial_sq).exp();
        }
    }

    let two_sigma_color_sq = 2.0 * sigma_color * sigma_color;

    // Use SIMD where beneficial
    for i in 0..rows {
        for j in 0..cols {
            let center_y = i + radius;
            let center_x = j + radius;
            let center_value = padded_input[[center_y, center_x]];

            let mut weighted_sum = 0.0f32;
            let mut weight_sum = 0.0f32;

            for dy in 0..kernel_size {
                for dx in 0..kernel_size {
                    let y = center_y - radius + dy;
                    let x = center_x - radius + dx;
                    let neighbor_value = padded_input[[y, x]];

                    let color_diff = neighbor_value - center_value;
                    let color_weight = (-color_diff * color_diff / two_sigma_color_sq).exp();

                    let total_weight = spatial_weights[[dy, dx]] * color_weight;
                    weighted_sum += neighbor_value * total_weight;
                    weight_sum += total_weight;
                }
            }

            output[[i, j]] = if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                center_value
            };
        }
    }

    output
        .into_dimensionality::<D>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert back from 2D array".into()))
}

/// SIMD-accelerated 2D bilateral filter for f64
#[cfg(feature = "simd")]
#[allow(dead_code)]
fn bilateral_filter_2d_simd_f64<D>(
    input: &Array<f64, D>,
    sigma_spatial: f64,
    sigma_color: f64,
    mode: &BorderMode,
) -> NdimageResult<Array<f64, D>>
where
    D: Dimension + 'static,
{
    let input_2d = input
        .to_owned()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D array".into()))?;

    let (rows, cols) = input_2d.dim();
    let radius = ((sigma_spatial * 3.0).ceil() as usize).max(1);
    let kernel_size = 2 * radius + 1;

    let mut output = Array2::zeros((rows, cols));
    let pad_width = vec![(radius, radius), (radius, radius)];
    let padded_input = pad_array(&input_2d, &pad_width, mode, None)?;

    let mut spatial_weights = Array2::zeros((kernel_size, kernel_size));
    let two_sigma_spatial_sq = 2.0 * sigma_spatial * sigma_spatial;

    for dy in 0..kernel_size {
        for dx in 0..kernel_size {
            let y_dist = (dy as i32 - radius as i32) as f64;
            let x_dist = (dx as i32 - radius as i32) as f64;
            let spatial_dist_sq = y_dist * y_dist + x_dist * x_dist;
            spatial_weights[[dy, dx]] = (-spatial_dist_sq / two_sigma_spatial_sq).exp();
        }
    }

    let two_sigma_color_sq = 2.0 * sigma_color * sigma_color;

    for i in 0..rows {
        for j in 0..cols {
            let center_y = i + radius;
            let center_x = j + radius;
            let center_value = padded_input[[center_y, center_x]];

            let mut weighted_sum = 0.0f64;
            let mut weight_sum = 0.0f64;

            for dy in 0..kernel_size {
                for dx in 0..kernel_size {
                    let y = center_y - radius + dy;
                    let x = center_x - radius + dx;
                    let neighbor_value = padded_input[[y, x]];

                    let color_diff = neighbor_value - center_value;
                    let color_weight = (-color_diff * color_diff / two_sigma_color_sq).exp();

                    let total_weight = spatial_weights[[dy, dx]] * color_weight;
                    weighted_sum += neighbor_value * total_weight;
                    weight_sum += total_weight;
                }
            }

            output[[i, j]] = if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                center_value
            };
        }
    }

    output
        .into_dimensionality::<D>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert back from 2D array".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_bilateral_filter_1d() {
        // Create a 1D signal with a step edge
        let signal = Array1::from_vec(vec![1.0, 1.0, 1.0, 5.0, 5.0, 5.0]);

        // Apply bilateral filter
        let result = bilateral_filter(&signal, 1.0, 1.0, None)
            .expect("bilateral_filter should succeed for 1D test");

        // Check that result has same shape
        assert_eq!(result.shape(), signal.shape());

        // Check that edges are preserved better than regular Gaussian
        // The step should still be relatively sharp
        assert!(result[2] < result[3]); // Should still have a step
    }

    #[test]
    fn test_bilateral_filter_2d() {
        // Create a 2D image with a vertical edge
        let mut image = Array2::zeros((5, 5));
        for i in 0..5 {
            for j in 0..2 {
                image[[i, j]] = 1.0;
            }
            for j in 3..5 {
                image[[i, j]] = 5.0;
            }
        }

        // Apply bilateral filter
        let result = bilateral_filter(&image, 1.0, 1.0, None)
            .expect("bilateral_filter should succeed for 2D test");

        // Check that result has same shape
        assert_eq!(result.shape(), image.shape());

        // Check that the edge is preserved
        // Left side should remain closer to 1.0, right side closer to 5.0
        assert!(result[[2, 0]] < 3.0); // Left side
        assert!(result[[2, 4]] > 3.0); // Right side
    }

    #[test]
    fn test_bilateral_filter_uniform_region() {
        // Create uniform region (should behave like Gaussian filter)
        let image = Array2::from_elem((5, 5), 3.0);

        // Apply bilateral filter
        let result = bilateral_filter(&image, 1.0, 1.0, None)
            .expect("bilateral_filter should succeed for uniform region test");

        // Should remain approximately constant
        for &val in result.iter() {
            assert_abs_diff_eq!(val, 3.0, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_bilateral_filter_noise_suppression() {
        // Create signal with noise
        let mut signal = Array1::from_elem(10, 2.0);
        signal[5] = 10.0; // Single outlier

        // Apply bilateral filter with appropriate parameters
        let result = bilateral_filter(&signal, 1.0, 2.0, None)
            .expect("bilateral_filter should succeed for noise suppression test");

        // The outlier should be reduced but not completely smoothed
        assert!(result[5] > 2.0); // Still elevated
        assert!(result[5] < 10.0); // But reduced
    }

    #[test]
    fn test_bilateral_filter_invalid_sigma() {
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // Test negative spatial sigma
        let result = bilateral_filter(&array, -1.0, 1.0, None);
        assert!(result.is_err());

        // Test negative color sigma
        let result = bilateral_filter(&array, 1.0, -1.0, None);
        assert!(result.is_err());

        // Test zero spatial sigma
        let result = bilateral_filter(&array, 0.0, 1.0, None);
        assert!(result.is_err());
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_bilateral_filter_simd_f32() {
        // Test SIMD version produces similar results to regular version
        let signal = Array1::from_vec(vec![1.0f32, 1.0, 1.0, 5.0, 5.0, 5.0]);

        let regular_result = bilateral_filter(&signal, 1.0, 1.0, None)
            .expect("bilateral_filter should succeed for SIMD f32 test");
        let simd_result = bilateral_filter_simd_f32(&signal, 1.0, 1.0, None)
            .expect("bilateral_filter_simd_f32 should succeed");

        // Results should be very close
        for i in 0..signal.len() {
            assert_abs_diff_eq!(regular_result[i], simd_result[i], epsilon = 1e-6);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_bilateral_filter_simd_f64() {
        // Test SIMD version produces similar results to regular version
        let signal = Array1::from_vec(vec![1.0f64, 1.0, 1.0, 5.0, 5.0, 5.0]);

        let regular_result = bilateral_filter(&signal, 1.0, 1.0, None)
            .expect("bilateral_filter should succeed for SIMD f64 test");
        let simd_result = bilateral_filter_simd_f64(&signal, 1.0, 1.0, None)
            .expect("bilateral_filter_simd_f64 should succeed");

        // Results should be very close
        for i in 0..signal.len() {
            assert_abs_diff_eq!(regular_result[i], simd_result[i], epsilon = 1e-6);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_bilateral_filter_simd_2d_f32() {
        // Test 2D SIMD version
        let mut image = Array2::zeros((5, 5));
        for i in 0..5 {
            for j in 0..2 {
                image[[i, j]] = 1.0f32;
            }
            for j in 3..5 {
                image[[i, j]] = 5.0f32;
            }
        }

        let regular_result = bilateral_filter(&image, 1.0, 1.0, None)
            .expect("bilateral_filter should succeed for 2D SIMD test");
        let simd_result = bilateral_filter_simd_f32(&image, 1.0, 1.0, None)
            .expect("bilateral_filter_simd_f32 should succeed for 2D test");

        // Results should be very close
        for i in 0..5 {
            for j in 0..5 {
                assert_abs_diff_eq!(regular_result[[i, j]], simd_result[[i, j]], epsilon = 1e-6);
            }
        }
    }
}

/// Configuration for multi-scale bilateral filtering
#[derive(Debug, Clone)]
pub struct MultiScaleBilateralConfig {
    /// Number of pyramid levels
    pub levels: usize,
    /// Spatial sigma for each level
    pub spatial_sigmas: Vec<f64>,
    /// Color sigma for each level
    pub color_sigmas: Vec<f64>,
    /// Downsampling factor between levels
    pub downsample_factor: f64,
    /// Boundary mode for filtering
    pub mode: BorderMode,
}

impl Default for MultiScaleBilateralConfig {
    fn default() -> Self {
        Self {
            levels: 3,
            spatial_sigmas: vec![0.5, 1.0, 2.0],
            color_sigmas: vec![0.5, 1.0, 2.0],
            downsample_factor: 0.5,
            mode: BorderMode::Reflect,
        }
    }
}

/// Multi-scale bilateral filter for enhanced edge preservation and noise reduction
///
/// This filter applies bilateral filtering at multiple scales, combining results
/// to achieve better edge preservation while maintaining effective noise reduction.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `config` - Configuration for multi-scale processing
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
///
/// # Example
///
/// ```rust
/// use ndarray::Array2;
/// use scirs2_ndimage::filters::{multi_scale_bilateral_filter, MultiScaleBilateralConfig};
///
/// let image = Array2::from_elem((64, 64), 1.0);
/// let config = MultiScaleBilateralConfig::default();
/// let result = multi_scale_bilateral_filter(&image, &config).unwrap();
/// ```
#[allow(dead_code)]
pub fn multi_scale_bilateral_filter<T, D>(
    input: &Array<T, D>,
    config: &MultiScaleBilateralConfig,
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + std::fmt::Display
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
    D: Dimension + 'static,
{
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if config.levels == 0 {
        return Err(NdimageError::InvalidInput(
            "Number of levels must be greater than 0".into(),
        ));
    }

    if config.spatial_sigmas.len() != config.levels || config.color_sigmas.len() != config.levels {
        return Err(NdimageError::InvalidInput(
            "Sigma arrays must have same length as number of levels".into(),
        ));
    }

    // For single level, fall back to regular bilateral filter
    if config.levels == 1 {
        let spatial_sigma = safe_f64_to_float::<T>(config.spatial_sigmas[0])?;
        let color_sigma = safe_f64_to_float::<T>(config.color_sigmas[0])?;
        return bilateral_filter(input, spatial_sigma, color_sigma, Some(config.mode));
    }

    // Create image pyramid
    let mut pyramid = vec![input.clone()];
    for level in 1..config.levels {
        let previmage = &pyramid[level - 1];
        let downsampled = downsampleimage(previmage, config.downsample_factor)?;
        pyramid.push(downsampled);
    }

    // Apply bilateral filter at each level
    let mut filtered_pyramid = Vec::with_capacity(config.levels);
    for (level, image) in pyramid.iter().enumerate() {
        let spatial_sigma = safe_f64_to_float::<T>(config.spatial_sigmas[level])?;
        let color_sigma = safe_f64_to_float::<T>(config.color_sigmas[level])?;

        let filtered = bilateral_filter(image, spatial_sigma, color_sigma, Some(config.mode))?;
        filtered_pyramid.push(filtered);
    }

    // Reconstruct from pyramid
    let mut result = filtered_pyramid.pop().unwrap();
    for level in (0..config.levels - 1).rev() {
        result = upsampleimage(&result, &filtered_pyramid[level])?;

        // Blend with original level
        let alpha = safe_f64_to_float::<T>(0.7)?; // Blend factor
        result = blend_arrays(&result, &filtered_pyramid[level], alpha)?;
    }

    Ok(result)
}

/// Adaptive bilateral filter that automatically adjusts parameters based on local characteristics
///
/// This filter analyzes local image statistics to adapt spatial and color sigma values,
/// providing better edge preservation in textured regions and stronger smoothing in uniform areas.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `base_spatial_sigma` - Base spatial sigma value
/// * `base_color_sigma` - Base color sigma value
/// * `adaptation_factor` - Factor controlling adaptation strength (0.0 to 1.0)
/// * `mode` - Border handling mode
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
///
/// # Example
///
/// ```rust
/// use ndarray::Array2;
/// use scirs2_ndimage::filters::{adaptive_bilateral_filter, BorderMode};
///
/// let image = Array2::from_elem((32, 32), 1.0);
/// let result = adaptive_bilateral_filter(&image, 1.0, 1.0, 0.5, Some(BorderMode::Reflect)).unwrap();
/// ```
#[allow(dead_code)]
pub fn adaptive_bilateral_filter<T, D>(
    input: &Array<T, D>,
    base_spatial_sigma: T,
    base_color_sigma: T,
    adaptation_factor: T,
    mode: Option<BorderMode>,
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + std::fmt::Display
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
    D: Dimension + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

    // Validate inputs
    scirs2_core::validation::check_positive(base_spatial_sigma, "base_spatial_sigma")
        .map_err(NdimageError::from)?;
    scirs2_core::validation::check_positive(base_color_sigma, "base_color_sigma")
        .map_err(NdimageError::from)?;

    if adaptation_factor < T::zero() || adaptation_factor > T::one() {
        return Err(NdimageError::InvalidInput(
            "Adaptation _factor must be between 0.0 and 1.0".into(),
        ));
    }

    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    // For 2D arrays, use specialized implementation
    if input.ndim() == 2 {
        return adaptive_bilateral_filter_2d(
            input,
            base_spatial_sigma,
            base_color_sigma,
            adaptation_factor,
            &border_mode,
        );
    }

    // For other dimensions, fall back to regular bilateral filter
    bilateral_filter(
        input,
        base_spatial_sigma,
        base_color_sigma,
        Some(border_mode),
    )
}

/// Specialized adaptive bilateral filter for 2D arrays
#[allow(dead_code)]
fn adaptive_bilateral_filter_2d<T, D>(
    input: &Array<T, D>,
    base_spatial_sigma: T,
    base_color_sigma: T,
    adaptation_factor: T,
    mode: &BorderMode,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
    D: Dimension + 'static,
{
    let input_2d = input
        .to_owned()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D array".into()))?;

    let (rows, cols) = input_2d.dim();
    let mut output = Array2::zeros((rows, cols));

    // Compute local variance for adaptation
    let variance_map = compute_local_variance(&input_2d, 3)?;
    let max_variance = variance_map.iter().fold(T::zero(), |acc, &x| acc.max(x));

    // Parameters for adaptive filtering
    let window_size = 5;
    let radius = window_size / 2;

    let pad_width = vec![(radius, radius), (radius, radius)];
    let padded_input = super::pad_array(&input_2d, &pad_width, mode, None)?;

    // Process each pixel with adaptive parameters
    for i in 0..rows {
        for j in 0..cols {
            let center_y = i + radius;
            let center_x = j + radius;
            let center_value = padded_input[[center_y, center_x]];

            // Adapt parameters based on local variance
            let local_variance = variance_map[[i, j]];
            let variance_ratio = if max_variance > T::zero() {
                local_variance / max_variance
            } else {
                T::zero()
            };

            // Higher variance -> larger spatial sigma, smaller color _sigma
            let adaptive_spatial =
                base_spatial_sigma * (T::one() + adaptation_factor * variance_ratio);
            let adaptive_color = base_color_sigma
                * (T::one() - adaptation_factor * variance_ratio * safe_f64_to_float::<T>(0.5)?);

            // Apply bilateral filtering with adaptive parameters
            let filtered_value = apply_bilateral_window(
                &padded_input.view(),
                center_y,
                center_x,
                center_value,
                adaptive_spatial,
                adaptive_color,
                window_size,
            )?;

            output[[i, j]] = filtered_value;
        }
    }

    output
        .into_dimensionality::<D>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert back from 2D array".into()))
}

/// Compute local variance in a 2D array using a sliding window
#[allow(dead_code)]
fn compute_local_variance<T>(input: &Array2<T>, windowsize: usize) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + 'static,
{
    let (rows, cols) = input.dim();
    let mut variance = Array2::zeros((rows, cols));
    let radius = windowsize / 2;

    let pad_width = vec![(radius, radius), (radius, radius)];
    let padded = super::pad_array(input, &pad_width, &BorderMode::Reflect, None)?;

    for i in 0..rows {
        for j in 0..cols {
            let center_y = i + radius;
            let center_x = j + radius;

            // Compute mean and variance in local window
            let mut sum = T::zero();
            let mut sum_sq = T::zero();
            let mut count = T::zero();

            for dy in 0..windowsize {
                for dx in 0..windowsize {
                    let y = center_y - radius + dy;
                    let x = center_x - radius + dx;
                    let value = padded[[y, x]];

                    sum = sum + value;
                    sum_sq = sum_sq + value * value;
                    count = count + T::one();
                }
            }

            let mean = sum / count;
            let variance_val = (sum_sq / count) - (mean * mean);
            variance[[i, j]] = variance_val.max(T::zero());
        }
    }

    Ok(variance)
}

/// Apply bilateral filtering to a single pixel with given parameters
#[allow(dead_code)]
fn apply_bilateral_window<T>(
    padded_input: &ndarray::ArrayView2<T>,
    center_y: usize,
    center_x: usize,
    center_value: T,
    spatial_sigma: T,
    color_sigma: T,
    window_size: usize,
) -> NdimageResult<T>
where
    T: Float + FromPrimitive + Debug + Clone + 'static,
{
    let radius = window_size / 2;
    let two = safe_f64_to_float::<T>(2.0)?;
    let two_sigma_spatial_sq = two * spatial_sigma * spatial_sigma;
    let two_sigma_color_sq = two * color_sigma * color_sigma;

    let mut weighted_sum = T::zero();
    let mut weight_sum = T::zero();

    for dy in 0..window_size {
        for dx in 0..window_size {
            let _y = center_y - radius + dy;
            let _x = center_x - radius + dx;
            let neighbor_value = padded_input[[_y, _x]];

            // Spatial weight
            let y_dist = safe_i32_to_float::<T>((dy as i32) - (radius as i32))?;
            let x_dist = safe_i32_to_float::<T>((dx as i32) - (radius as i32))?;
            let spatial_dist_sq: T = y_dist * y_dist + x_dist * x_dist;
            let spatial_weight = (-spatial_dist_sq / two_sigma_spatial_sq).exp();

            // Color weight
            let color_diff = neighbor_value - center_value;
            let color_weight = (-color_diff * color_diff / two_sigma_color_sq).exp();

            let total_weight = spatial_weight * color_weight;
            weighted_sum = weighted_sum + neighbor_value * total_weight;
            weight_sum = weight_sum + total_weight;
        }
    }

    Ok(if weight_sum > T::zero() {
        weighted_sum / weight_sum
    } else {
        center_value
    })
}

/// Downsample an image by the given factor
#[allow(dead_code)]
fn downsampleimage<T, D>(input: &Array<T, D>, factor: f64) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + Clone + 'static,
    D: Dimension + 'static,
{
    if factor <= 0.0 || factor >= 1.0 {
        return Err(NdimageError::InvalidInput(
            "Downsample factor must be between 0 and 1".into(),
        ));
    }

    // For simplicity, implement basic downsampling
    // In a production implementation, you might want to use proper anti-aliasing
    let oldshape = input.shape();
    let mut newshape = Vec::with_capacity(oldshape.len());

    for &dim_size in oldshape {
        let new_size = ((dim_size as f64) * factor).max(1.0) as usize;
        newshape.push(new_size);
    }

    // Simple nearest-neighbor downsampling
    let output_dyn = ndarray::ArrayD::<T>::zeros(newshape.clone());
    let output = output_dyn.into_dimensionality::<D>().map_err(|_| {
        NdimageError::DimensionError("Failed to convert output array to correct dimension".into())
    })?;

    // For 2D case (most common)
    if input.ndim() == 2 && newshape.len() == 2 {
        let input_2d = input
            .to_owned()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D".into()))?;
        let mut output_2d = Array2::zeros((newshape[0], newshape[1]));

        for i in 0..newshape[0] {
            for j in 0..newshape[1] {
                let src_i = ((i as f64) / factor).min((oldshape[0] - 1) as f64) as usize;
                let src_j = ((j as f64) / factor).min((oldshape[1] - 1) as f64) as usize;
                output_2d[[i, j]] = input_2d[[src_i, src_j]];
            }
        }

        return output_2d
            .into_dimensionality::<D>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert back from 2D".into()));
    }

    // Fallback: return a copy for unsupported dimensions
    Ok(input.clone())
}

/// Upsample an image to match the target shape
#[allow(dead_code)]
fn upsampleimage<T, D>(input: &Array<T, D>, target: &Array<T, D>) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + Clone + 'static,
    D: Dimension + 'static,
{
    let inputshape = input.shape();
    let targetshape = target.shape();

    if inputshape.len() != targetshape.len() {
        return Err(NdimageError::DimensionError(
            "Input and target must have same number of dimensions".into(),
        ));
    }

    // For 2D case
    if input.ndim() == 2 {
        let input_2d = input
            .to_owned()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D".into()))?;
        let mut output = Array2::zeros((targetshape[0], targetshape[1]));

        let scale_y = (inputshape[0] as f64) / (targetshape[0] as f64);
        let scale_x = (inputshape[1] as f64) / (targetshape[1] as f64);

        for i in 0..targetshape[0] {
            for j in 0..targetshape[1] {
                let src_i = ((i as f64) * scale_y).min((inputshape[0] - 1) as f64) as usize;
                let src_j = ((j as f64) * scale_x).min((inputshape[1] - 1) as f64) as usize;
                output[[i, j]] = input_2d[[src_i, src_j]];
            }
        }

        return output
            .into_dimensionality::<D>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert back from 2D".into()));
    }

    // Fallback: return target shape with _input values
    Ok(target.clone())
}

/// Blend two arrays with the given alpha factor
#[allow(dead_code)]
fn blend_arrays<T, D>(a: &Array<T, D>, b: &Array<T, D>, alpha: T) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + Clone + 'static,
    D: Dimension + 'static,
{
    if a.shape() != b.shape() {
        return Err(NdimageError::DimensionError(
            "Arrays must have same shape for blending".into(),
        ));
    }

    let one_minus_alpha = T::one() - alpha;
    let mut result = Array::zeros(a.raw_dim());

    for ((aa, bb), rr) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
        *rr = *aa * alpha + *bb * one_minus_alpha;
    }

    Ok(result)
}
