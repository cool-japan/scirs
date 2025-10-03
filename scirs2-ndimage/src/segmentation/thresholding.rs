//! Thresholding algorithms for image segmentation
//!
//! This module provides functions for thresholding images to create binary masks or segmentations.

use crate::error::{NdimageError, NdimageResult};
use crate::utils::safe_f64_to_float;
use scirs2_core::ndarray::{Array, Dimension, Ix2};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign};

/// Helper function for safe conversion from usize to float
#[allow(dead_code)]
fn safe_usize_to_float<T: Float + FromPrimitive>(value: usize) -> NdimageResult<T> {
    T::from_usize(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert usize {} to float type", value))
    })
}

/// Apply a threshold to an image to create a binary image
///
/// # Arguments
///
/// * `image` - Input array
/// * `threshold` - Threshold value
///
/// # Returns
///
/// * Binary mask where values equal to or above the threshold are set to true
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::segmentation::threshold_binary;
///
/// let image = array![
///     [0.0, 0.2, 0.5],
///     [0.3, 0.8, 0.1],
///     [0.7, 0.4, 0.6],
/// ];
///
/// let mask = threshold_binary(&image, 0.5).unwrap();
/// ```
#[allow(dead_code)]
pub fn threshold_binary<T, D>(image: &Array<T, D>, threshold: T) -> NdimageResult<Array<T, D>>
where
    T: Float + NumAssign + std::fmt::Debug + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
{
    // Apply threshold by mapping over the input array
    let result = image.mapv(|val| if val > threshold { T::one() } else { T::zero() });

    Ok(result)
}

/// Apply Otsu's thresholding method
///
/// Otsu's method determines an optimal threshold by maximizing
/// the variance between foreground and background classes.
///
/// # Arguments
///
/// * `image` - Input array
/// * `bins` - Number of bins for the histogram
///
/// # Returns
///
/// * Tuple containing (binary_image, threshold_value)
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::segmentation::otsu_threshold;
///
/// let image = array![
///     [0.1, 0.2, 0.3],
///     [0.4, 0.5, 0.6],
///     [0.7, 0.8, 0.9],
/// ];
///
/// let (binary, threshold) = otsu_threshold(&image, 256).unwrap();
/// ```
/// ```
#[allow(dead_code)]
pub fn otsu_threshold<T, D>(image: &Array<T, D>, bins: usize) -> NdimageResult<(Array<T, D>, T)>
where
    T: Float + NumAssign + std::fmt::Debug + std::ops::DivAssign + FromPrimitive + 'static,
    D: Dimension + 'static,
{
    let nbins = bins;

    // Get min and max values
    let mut min_val = Float::infinity();
    let mut max_val = Float::neg_infinity();

    for &val in image.iter() {
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }

    // Handle edge case of flat image
    if min_val == max_val {
        // Create a binary image with all zeros (as all values == threshold)
        let binary = threshold_binary(image, min_val)?;
        return Ok((binary, min_val));
    }

    // Calculate histogram
    let mut hist = vec![0; nbins];
    let bin_width = (max_val - min_val) / safe_usize_to_float(nbins)?;

    for &val in image.iter() {
        let bin = ((val - min_val) / bin_width).to_usize().unwrap_or(0);
        let bin_index = std::cmp::min(bin, nbins - 1);
        hist[bin_index] += 1;
    }

    // Calculate total pixels
    let total_pixels = image.len();

    // Compute cumulative sums
    let mut cum_sum = vec![0; nbins];
    cum_sum[0] = hist[0];
    for i in 1..nbins {
        cum_sum[i] = cum_sum[i - 1] + hist[i];
    }

    // Compute cumulative means
    let mut cum_val = vec![T::zero(); nbins];
    for i in 0..nbins {
        if i > 0 {
            cum_val[i] = cum_val[i - 1] + safe_usize_to_float(i * hist[i])?;
        } else {
            cum_val[i] = safe_usize_to_float(i * hist[i])?
        }
    }

    // Compute maximum inter-class variance
    let mut max_var = T::zero();
    let mut threshold_idx = 0;

    for i in 0..(nbins - 1) {
        let bg_pixels = cum_sum[i];
        let fg_pixels = total_pixels - bg_pixels;

        // Skip cases where all pixels are in one class
        if bg_pixels == 0 || fg_pixels == 0 {
            continue;
        }

        let bg_mean = cum_val[i] / safe_usize_to_float::<T>(bg_pixels)?;
        let fg_mean = (cum_val[nbins - 1] - cum_val[i]) / safe_usize_to_float::<T>(fg_pixels)?;

        // Calculate inter-class variance
        let variance = safe_usize_to_float::<T>(bg_pixels * fg_pixels)?
            * (bg_mean - fg_mean)
            * (bg_mean - fg_mean);

        // Update threshold if variance is higher
        if variance > max_var {
            max_var = variance;
            threshold_idx = i;
        }
    }

    // Convert threshold index back to intensity value
    let threshold = min_val + safe_usize_to_float::<T>(threshold_idx)? * bin_width;

    // Create binary image using the threshold
    let binary = threshold_binary(image, threshold)?;

    Ok((binary, threshold))
}

/// Apply adaptive thresholding
///
/// Adaptive thresholding computes a local threshold for each pixel based on
/// the statistics of its neighborhood.
///
/// # Arguments
///
/// * `image` - Input 2D array
/// * `block_size` - Size of the neighborhood for calculating local threshold
/// * `method` - Thresholding method ('mean' or 'gaussian')
/// * `c` - Constant subtracted from the local threshold
///
/// # Returns
///
/// * Result containing the binary mask
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::segmentation::{adaptive_threshold, AdaptiveMethod};
///
/// let image = array![
///     [0.1, 0.2, 0.7],
///     [0.3, 0.8, 0.1],
///     [0.7, 0.4, 0.2],
/// ];
///
/// let mask = adaptive_threshold(&image, 3, AdaptiveMethod::Mean, 0.05).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub enum AdaptiveMethod {
    Mean,
    Gaussian,
}

#[allow(dead_code)]
pub fn adaptive_threshold<T>(
    image: &Array<T, Ix2>,
    block_size: usize,
    method: AdaptiveMethod,
    c: T,
) -> NdimageResult<Array<bool, Ix2>>
where
    T: Float + NumAssign + std::fmt::Debug + FromPrimitive,
{
    // Check block _size (must be odd)
    if block_size % 2 == 0 || block_size < 3 {
        return Err(NdimageError::InvalidInput(
            "block_size must be odd and at least 3".to_string(),
        ));
    }

    let shape = image.raw_dim();
    let (rows, cols) = (shape[0], shape[1]);
    let mut result = Array::from_elem(shape, false);
    let radius = block_size / 2;

    // For each pixel, compute local threshold based on its neighborhood
    for i in 0..rows {
        for j in 0..cols {
            // Define neighborhood bounds with padding at the edges
            let start_row = i.saturating_sub(radius);
            let end_row = std::cmp::min(i + radius + 1, rows);
            let start_col = j.saturating_sub(radius);
            let end_col = std::cmp::min(j + radius + 1, cols);

            // Slice the neighborhood
            let neighborhood = image.slice(scirs2_core::ndarray::s![
                start_row..end_row,
                start_col..end_col
            ]);

            // Compute local threshold based on method
            let threshold = match method {
                AdaptiveMethod::Mean => {
                    // Simple mean of neighborhood
                    let sum = neighborhood.iter().fold(T::zero(), |acc, &x| acc + x);
                    sum / safe_usize_to_float(neighborhood.len())? - c
                }
                AdaptiveMethod::Gaussian => {
                    // Gaussian weighted mean
                    // Simplified implementation with distance-based weighting
                    let center_row = i - start_row;
                    let center_col = j - start_col;

                    let mut weighted_sum = T::zero();
                    let mut weight_sum = T::zero();

                    for (idx, &val) in neighborhood.indexed_iter() {
                        let dist_sq = (idx.0 as isize - center_row as isize).pow(2)
                            + (idx.1 as isize - center_col as isize).pow(2);
                        let dist = safe_usize_to_float::<T>(dist_sq as usize)?.sqrt();

                        // Gaussian weight
                        let sigma =
                            safe_usize_to_float::<T>(radius)? / safe_f64_to_float::<T>(2.0)?;
                        let weight =
                            (-dist * dist / (safe_f64_to_float::<T>(2.0)? * sigma * sigma)).exp();

                        weighted_sum += val * weight;
                        weight_sum += weight;
                    }

                    weighted_sum / weight_sum - c
                }
            };

            // Apply threshold
            result[(i, j)] = image[(i, j)] > threshold;
        }
    }

    Ok(result)
}
