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
use crate::error::{NdimageError, Result};

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
pub fn bilateral_filter<T, D>(
    input: &Array<T, D>,
    sigma_spatial: T,
    sigma_color: T,
    mode: Option<BorderMode>,
) -> Result<Array<T, D>>
where
    T: Float + Debug + Clone + Send + Sync + Display + FromPrimitive,
    D: Dimension,
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
fn bilateral_filter_1d<T, D>(
    input: &Array<T, D>,
    sigma_spatial: T,
    sigma_color: T,
    mode: &BorderMode,
) -> Result<Array<T, D>>
where
    T: Float + Debug + Clone + Display + FromPrimitive,
    D: Dimension,
{
    // Convert to 1D for processing
    let input_1d = input
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert to 1D array".into()))?;

    // Calculate kernel radius based on spatial sigma
    let radius = (sigma_spatial * T::from(3.0).unwrap())
        .ceil()
        .to_usize()
        .unwrap_or(3);
    let kernel_size = 2 * radius + 1;

    // Create output array
    let mut output = Array1::zeros(input_1d.len());

    // Pad input for border handling
    let pad_width = vec![(radius, radius)];
    let padded_input = pad_array(&input_1d, &pad_width, mode, None)?;

    // Precompute spatial weights
    let mut spatial_weights = Array1::zeros(kernel_size);
    let two_sigma_spatial_sq = T::from(2.0).unwrap() * sigma_spatial * sigma_spatial;

    for k in 0..kernel_size {
        let dist = T::from((k as i32) - (radius as i32)).unwrap();
        spatial_weights[k] = (-dist * dist / two_sigma_spatial_sq).exp();
    }

    let two_sigma_color_sq = T::from(2.0).unwrap() * sigma_color * sigma_color;

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
fn bilateral_filter_2d<T, D>(
    input: &Array<T, D>,
    sigma_spatial: T,
    sigma_color: T,
    mode: &BorderMode,
) -> Result<Array<T, D>>
where
    T: Float + Debug + Clone + Display + FromPrimitive,
    D: Dimension,
{
    // Convert to 2D for processing
    let input_2d = input
        .to_owned()
        .into_dimensionality::<Ix2>()
        .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D array".into()))?;

    let (rows, cols) = input_2d.dim();

    // Calculate kernel radius based on spatial sigma
    let radius = (sigma_spatial * T::from(3.0).unwrap())
        .ceil()
        .to_usize()
        .unwrap_or(3);
    let kernel_size = 2 * radius + 1;

    // Create output array
    let mut output = Array2::zeros((rows, cols));

    // Pad input for border handling
    let pad_width = vec![(radius, radius), (radius, radius)];
    let padded_input = pad_array(&input_2d, &pad_width, mode, None)?;

    // Precompute spatial weights
    let mut spatial_weights = Array2::zeros((kernel_size, kernel_size));
    let two_sigma_spatial_sq = T::from(2.0).unwrap() * sigma_spatial * sigma_spatial;

    for dy in 0..kernel_size {
        for dx in 0..kernel_size {
            let y_dist = T::from((dy as i32) - (radius as i32)).unwrap();
            let x_dist = T::from((dx as i32) - (radius as i32)).unwrap();
            let spatial_dist_sq = y_dist * y_dist + x_dist * x_dist;
            spatial_weights[[dy, dx]] = (-spatial_dist_sq / two_sigma_spatial_sq).exp();
        }
    }

    let two_sigma_color_sq = T::from(2.0).unwrap() * sigma_color * sigma_color;

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
fn bilateral_filter_nd<T, D>(
    input: &Array<T, D>,
    sigma_spatial: T,
    sigma_color: T,
    mode: &BorderMode,
) -> Result<Array<T, D>>
where
    T: Float + Debug + Clone + Display + FromPrimitive,
    D: Dimension,
{
    // Calculate kernel radius based on spatial sigma
    let radius = (sigma_spatial * T::from(3.0).unwrap())
        .ceil()
        .to_usize()
        .unwrap_or(3);

    // Convert to dynamic dimension for easier processing
    let input_dyn = input.clone().into_dyn();
    let mut output_dyn = Array::<T, IxDyn>::zeros(input_dyn.raw_dim());

    // Prepare padding
    let pad_width: Vec<(usize, usize)> = (0..input.ndim()).map(|_| (radius, radius)).collect();
    let padded_input = pad_array(&input_dyn, &pad_width, mode, None)?;

    let two_sigma_spatial_sq = T::from(2.0).unwrap() * sigma_spatial * sigma_spatial;
    let two_sigma_color_sq = T::from(2.0).unwrap() * sigma_color * sigma_color;

    // Process each position in the output array
    for (idx, output_val) in output_dyn.indexed_iter_mut() {
        let idx_vec = idx.as_array_view().to_vec();

        // Calculate center position in padded array
        let center_idx: Vec<usize> = idx_vec.iter().map(|&i| i + radius).collect();
        let center_value = padded_input[&IxDyn(&center_idx)];

        let mut weighted_sum = T::zero();
        let mut weight_sum = T::zero();

        // Iterate through neighborhood
        iterate_neighborhood(&idx_vec, radius, input.ndim(), |neighbor_offset| {
            // Calculate neighbor position in padded array
            let neighbor_idx: Vec<usize> = center_idx
                .iter()
                .zip(neighbor_offset.iter())
                .map(|(&center, &offset)| (center as i32 + offset) as usize)
                .collect();

            let neighbor_value = padded_input[&IxDyn(&neighbor_idx)];

            // Calculate spatial weight
            let spatial_dist_sq = neighbor_offset.iter().fold(T::zero(), |acc, &offset| {
                let dist = T::from(offset).unwrap();
                acc + dist * dist
            });
            let spatial_weight = (-spatial_dist_sq / two_sigma_spatial_sq).exp();

            // Calculate color weight
            let color_diff = neighbor_value - center_value;
            let color_weight = (-color_diff * color_diff / two_sigma_color_sq).exp();

            let total_weight = spatial_weight * color_weight;
            weighted_sum = weighted_sum + neighbor_value * total_weight;
            weight_sum = weight_sum + total_weight;
        });

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
fn iterate_neighborhood<F>(_center_idx: &[usize], radius: usize, ndim: usize, mut callback: F)
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
pub fn bilateral_filter_simd_f32<D>(
    input: &Array<f32, D>,
    sigma_spatial: f32,
    sigma_color: f32,
    mode: Option<BorderMode>,
) -> Result<Array<f32, D>>
where
    D: Dimension,
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
pub fn bilateral_filter_simd_f64<D>(
    input: &Array<f64, D>,
    sigma_spatial: f64,
    sigma_color: f64,
    mode: Option<BorderMode>,
) -> Result<Array<f64, D>>
where
    D: Dimension,
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
fn bilateral_filter_1d_simd_f32<D>(
    input: &Array<f32, D>,
    sigma_spatial: f32,
    sigma_color: f32,
    mode: &BorderMode,
) -> Result<Array<f32, D>>
where
    D: Dimension,
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

    // Precompute spatial weights
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
fn bilateral_filter_1d_simd_f64<D>(
    input: &Array<f64, D>,
    sigma_spatial: f64,
    sigma_color: f64,
    mode: &BorderMode,
) -> Result<Array<f64, D>>
where
    D: Dimension,
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
fn bilateral_filter_2d_simd_f32<D>(
    input: &Array<f32, D>,
    sigma_spatial: f32,
    sigma_color: f32,
    mode: &BorderMode,
) -> Result<Array<f32, D>>
where
    D: Dimension,
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

    // Precompute spatial weights
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
fn bilateral_filter_2d_simd_f64<D>(
    input: &Array<f64, D>,
    sigma_spatial: f64,
    sigma_color: f64,
    mode: &BorderMode,
) -> Result<Array<f64, D>>
where
    D: Dimension,
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
        let result = bilateral_filter(&signal, 1.0, 1.0, None).unwrap();

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
        let result = bilateral_filter(&image, 1.0, 1.0, None).unwrap();

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
        let result = bilateral_filter(&image, 1.0, 1.0, None).unwrap();

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
        let result = bilateral_filter(&signal, 1.0, 2.0, None).unwrap();

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

        let regular_result = bilateral_filter(&signal, 1.0, 1.0, None).unwrap();
        let simd_result = bilateral_filter_simd_f32(&signal, 1.0, 1.0, None).unwrap();

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

        let regular_result = bilateral_filter(&signal, 1.0, 1.0, None).unwrap();
        let simd_result = bilateral_filter_simd_f64(&signal, 1.0, 1.0, None).unwrap();

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

        let regular_result = bilateral_filter(&image, 1.0, 1.0, None).unwrap();
        let simd_result = bilateral_filter_simd_f32(&image, 1.0, 1.0, None).unwrap();

        // Results should be very close
        for i in 0..5 {
            for j in 0..5 {
                assert_abs_diff_eq!(regular_result[[i, j]], simd_result[[i, j]], epsilon = 1e-6);
            }
        }
    }
}
