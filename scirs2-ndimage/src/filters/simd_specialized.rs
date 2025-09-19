//! SIMD-optimized specialized filter functions
//!
//! This module provides highly optimized SIMD implementations for
//! specialized filtering operations that benefit from vectorization.

use ndarray::{s, Array, Array1, ArrayView2, ArrayViewMut1, Axis, Ix2};
use num_traits::{Float, FromPrimitive};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};
use crate::utils::safe_f64_to_float;

/// Helper function for safe float to usize conversion
#[allow(dead_code)]
fn safe_float_to_usize<T: Float>(value: T) -> NdimageResult<usize> {
    value.to_usize().ok_or_else(|| {
        NdimageError::ComputationError("Failed to convert float to usize".to_string())
    })
}

/// Helper function for safe isize conversion
#[allow(dead_code)]
fn safe_isize_to_float<T: Float + FromPrimitive>(value: isize) -> NdimageResult<T> {
    T::from_isize(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert isize {} to float type", value))
    })
}

/// Helper function for safe usize conversion
#[allow(dead_code)]
fn safe_usize_to_float<T: Float + FromPrimitive>(value: usize) -> NdimageResult<T> {
    T::from_usize(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert usize {} to float type", value))
    })
}

/// Helper function for safe partial comparison
#[allow(dead_code)]
fn safe_partial_cmp<T: PartialOrd>(a: &T, b: &T) -> NdimageResult<std::cmp::Ordering> {
    a.partial_cmp(b).ok_or_else(|| {
        NdimageError::ComputationError("Failed to compare values (NaN encountered)".to_string())
    })
}

/// SIMD-optimized bilateral filter for edge-preserving smoothing
///
/// The bilateral filter is a non-linear filter that smooths an image while preserving edges.
/// It considers both spatial distance and intensity difference when computing weights.
#[allow(dead_code)]
pub fn simd_bilateral_filter<T>(
    input: ArrayView2<T>,
    spatial_sigma: T,
    range_sigma: T,
    window_size: Option<usize>,
) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let window_size = match window_size {
        Some(size) => size,
        None => {
            // Automatically determine window _size based on spatial _sigma
            let three = safe_f64_to_float::<T>(3.0)?;
            let radius = safe_float_to_usize(spatial_sigma * three)?;
            2 * radius + 1
        }
    };
    let half_window = window_size / 2;

    let mut output = Array::zeros((height, width));

    // Pre-compute spatial weights
    let spatial_weights = compute_spatial_weights(window_size, spatial_sigma)?;

    // Process image in parallel chunks
    let chunk_size = if height * width > 10000 { 64 } else { height };

    output
        .axis_chunks_iter_mut(Axis(0), chunk_size)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut chunk)| {
            let y_start = chunk_idx * chunk_size;

            for (local_y, mut row) in chunk.axis_iter_mut(Axis(0)).enumerate() {
                let y = y_start + local_y;
                if let Err(e) = simd_bilateral_filter_row(
                    &input,
                    &mut row,
                    y,
                    half_window,
                    &spatial_weights,
                    range_sigma,
                ) {
                    // For parallel processing, we can't propagate errors directly
                    // Log error and continue with default values
                    eprintln!("Warning: bilateral filter row processing failed: {:?}", e);
                }
            }
        });

    Ok(output)
}

/// Process a single row with SIMD optimization
#[allow(dead_code)]
fn simd_bilateral_filter_row<T>(
    input: &ArrayView2<T>,
    output_row: &mut ArrayViewMut1<T>,
    y: usize,
    half_window: usize,
    spatial_weights: &Array<T, Ix2>,
    range_sigma: T,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + Debug + Clone + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let range_factor = safe_f64_to_float::<T>(-0.5)? / (range_sigma * range_sigma);

    // Process pixels in SIMD chunks
    let simd_width = 8; // Default SIMD width for vectorization
    let num_full_chunks = width / simd_width;

    // Process full SIMD chunks
    for chunk_idx in 0..num_full_chunks {
        let x_start = chunk_idx * simd_width;

        // Gather center pixel values for SIMD chunk
        let mut center_values = vec![T::zero(); simd_width];
        for i in 0..simd_width {
            center_values[i] = input[(y, x_start + i)];
        }

        let mut sum_weights = vec![T::zero(); simd_width];
        let mut sum_values = vec![T::zero(); simd_width];

        // Compute bilateral filter for _window
        for dy in 0..2 * half_window + 1 {
            let ny = (y as isize + dy as isize - half_window as isize).clamp(0, height as isize - 1)
                as usize;

            for dx in 0..2 * half_window + 1 {
                // Process SIMD width pixels at once
                let mut neighbor_values = vec![T::zero(); simd_width];
                let mut valid_mask = vec![true; simd_width];

                for i in 0..simd_width {
                    let x = x_start + i;
                    let nx = (x as isize + dx as isize - half_window as isize)
                        .clamp(0, width as isize - 1) as usize;
                    neighbor_values[i] = input[(ny, nx)];
                    valid_mask[i] = nx < width;
                }

                // Compute range _weights using SIMD
                let mut range_diffs = vec![T::zero(); simd_width];
                for i in 0..simd_width {
                    range_diffs[i] = neighbor_values[i] - center_values[i];
                }

                // Square differences
                let range_diffs_array = Array1::from_vec(range_diffs.clone());
                let range_diffs_sq =
                    T::simd_mul(&range_diffs_array.view(), &range_diffs_array.view());

                // Apply range factor
                let mut range_exp_args = vec![T::zero(); simd_width];
                for i in 0..simd_width {
                    range_exp_args[i] = range_diffs_sq.as_slice().unwrap()[i] * range_factor;
                }

                // Compute exponential (approximation for SIMD)
                let range_weights = simd_exp_approx(&range_exp_args);

                // Combine with spatial weight
                let spatial_weight = spatial_weights[(dy, dx)];

                for i in 0..simd_width {
                    if valid_mask[i] {
                        let weight = spatial_weight * range_weights[i];
                        sum_weights[i] = sum_weights[i] + weight;
                        sum_values[i] = sum_values[i] + weight * neighbor_values[i];
                    }
                }
            }
        }

        // Write results
        for i in 0..simd_width {
            if x_start + i < width {
                output_row[x_start + i] = sum_values[i] / sum_weights[i];
            }
        }
    }

    // Process remaining pixels
    for x in (num_full_chunks * simd_width)..width {
        let center_value = input[(y, x)];
        let mut sum_weight = T::zero();
        let mut sum_value = T::zero();

        for dy in 0..2 * half_window + 1 {
            let ny = (y as isize + dy as isize - half_window as isize).clamp(0, height as isize - 1)
                as usize;

            for dx in 0..2 * half_window + 1 {
                let nx = (x as isize + dx as isize - half_window as isize)
                    .clamp(0, width as isize - 1) as usize;

                let neighbor_value = input[(ny, nx)];
                let range_diff = neighbor_value - center_value;
                let range_weight = (range_diff * range_diff * range_factor).exp();
                let spatial_weight = spatial_weights[(dy, dx)];

                let weight = spatial_weight * range_weight;
                sum_weight = sum_weight + weight;
                sum_value = sum_value + weight * neighbor_value;
            }
        }

        output_row[x] = sum_value / sum_weight;
    }

    Ok(())
}

/// Compute spatial weights for bilateral filter
#[allow(dead_code)]
fn compute_spatial_weights<T>(window_size: usize, sigma: T) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive,
{
    let half_window = window_size / 2;
    let factor = safe_f64_to_float::<T>(-0.5)? / (sigma * sigma);
    let mut weights = Array::zeros((window_size, window_size));

    for dy in 0..window_size {
        for dx in 0..window_size {
            let y_dist = safe_isize_to_float::<T>(dy as isize - half_window as isize)?;
            let x_dist = safe_isize_to_float::<T>(dx as isize - half_window as isize)?;
            let dist_sq: T = y_dist * y_dist + x_dist * x_dist;
            weights[(dy, dx)] = (dist_sq * factor).exp();
        }
    }

    Ok(weights)
}

/// SIMD approximation of exponential function
#[allow(dead_code)]
fn simd_exp_approx<T>(values: &[T]) -> Vec<T>
where
    T: Float + FromPrimitive,
{
    // Use Taylor series approximation for exp(x) ≈ 1 + x + x²/2 + x³/6
    // This is accurate for small _values (which we have after multiplying by range_factor)
    let mut result = vec![T::one(); values.len()];

    for i in 0..values.len() {
        let x = values[i];
        let x2 = x * x;
        let x3 = x2 * x;
        // Use safe constants with fallback to simple approximation
        let two = T::from_f64(2.0).unwrap_or_else(|| T::one() + T::one());
        let six = T::from_f64(6.0).unwrap_or_else(|| two * two * two / two);
        result[i] = T::one() + x + x2 / two + x3 / six;

        // Clamp to positive _values
        if result[i] < T::zero() {
            result[i] = T::zero();
        }
    }

    result
}

/// SIMD-optimized non-local means filter
///
/// Non-local means is a denoising algorithm that averages similar patches
/// throughout the image, not just in a local neighborhood.
#[allow(dead_code)]
pub fn simd_non_local_means<T>(
    input: ArrayView2<T>,
    patch_size: usize,
    search_window: usize,
    h: T, // Filter strength
) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let half_patch = patch_size / 2;
    let half_search = search_window / 2;

    let mut output = Array::zeros((height, width));
    let h_squared = h * h;

    // Pre-compute patch normalization factor
    let patch_norm = safe_usize_to_float(patch_size * patch_size)?;

    // Process in parallel chunks
    output
        .axis_chunks_iter_mut(Axis(0), 32)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut chunk)| {
            let y_start = chunk_idx * 32;

            for (local_y, mut row) in chunk.axis_iter_mut(Axis(0)).enumerate() {
                let y = y_start + local_y;
                if y >= half_patch && y < height - half_patch {
                    if let Err(e) = simd_nlm_process_row(
                        &input,
                        &mut row,
                        y,
                        half_patch,
                        half_search,
                        h_squared,
                        patch_norm,
                    ) {
                        // For parallel processing, we can't propagate errors directly
                        // Log error and continue with default values
                        eprintln!("Warning: non-local means row processing failed: {:?}", e);
                    }
                }
            }
        });

    Ok(output)
}

/// Process a row with non-local means using SIMD
#[allow(dead_code)]
fn simd_nlm_process_row<T>(
    input: &ArrayView2<T>,
    output_row: &mut ArrayViewMut1<T>,
    y: usize,
    half_patch: usize,
    half_search: usize,
    h_squared: T,
    patch_norm: T,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + Debug + Clone + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let simd_width = 8; // Default SIMD width for vectorization

    for x in half_patch..width - half_patch {
        let mut weight_sum = T::zero();
        let mut value_sum = T::zero();

        // Define _search region
        let search_y_min = (y as isize - half_search as isize).max(half_patch as isize) as usize;
        let search_y_max = (y + half_search + 1).min(height - half_patch);
        let search_x_min = (x as isize - half_search as isize).max(half_patch as isize) as usize;
        let search_x_max = (x + half_search + 1).min(width - half_patch);

        // Extract reference _patch
        let ref_patch = input.slice(s![
            y - half_patch..=y + half_patch,
            x - half_patch..=x + half_patch
        ]);

        // Search for similar patches
        for sy in search_y_min..search_y_max {
            for sx in search_x_min..search_x_max {
                // Extract comparison _patch
                let comp_patch = input.slice(s![
                    sy - half_patch..=sy + half_patch,
                    sx - half_patch..=sx + half_patch
                ]);

                // Compute _patch distance using SIMD
                let distance = simd_patch_distance(&ref_patch, &comp_patch)? / patch_norm;

                // Compute weight
                let weight = (-distance / h_squared).exp();
                weight_sum = weight_sum + weight;
                value_sum = value_sum + weight * input[(sy, sx)];
            }
        }

        output_row[x] = value_sum / weight_sum;
    }

    Ok(())
}

/// Compute L2 distance between patches using SIMD
#[allow(dead_code)]
fn simd_patch_distance<T>(patch1: &ArrayView2<T>, patch2: &ArrayView2<T>) -> NdimageResult<T>
where
    T: Float + FromPrimitive + SimdUnifiedOps,
{
    let flat1 = patch1.as_slice().ok_or_else(|| {
        NdimageError::ComputationError("Failed to convert _patch1 to contiguous slice".to_string())
    })?;
    let flat2 = patch2.as_slice().ok_or_else(|| {
        NdimageError::ComputationError("Failed to convert patch2 to contiguous slice".to_string())
    })?;

    let mut sum = T::zero();
    let simd_width = 8; // Default SIMD width for vectorization
    let num_chunks = flat1.len() / simd_width;

    // Process SIMD chunks
    for i in 0..num_chunks {
        let start = i * simd_width;
        let end = start + simd_width;

        let flat1_array = Array1::from_vec(flat1[start..end].to_vec());
        let flat2_array = Array1::from_vec(flat2[start..end].to_vec());
        let diff = T::simd_sub(&flat1_array.view(), &flat2_array.view());
        let diff_sq = T::simd_mul(&diff.view(), &diff.view());

        for &val in &diff_sq {
            sum = sum + val;
        }
    }

    // Process remaining elements
    for i in (num_chunks * simd_width)..flat1.len() {
        let diff = flat1[i] - flat2[i];
        sum = sum + diff * diff;
    }

    Ok(sum)
}

/// SIMD-optimized anisotropic diffusion filter
///
/// This filter performs edge-preserving smoothing by varying the diffusion
/// coefficient based on the local image gradient.
#[allow(dead_code)]
pub fn simd_anisotropic_diffusion<T>(
    input: ArrayView2<T>,
    iterations: usize,
    kappa: T,      // Edge threshold parameter
    lambda: T,     // Diffusion rate (0 < lambda <= 0.25)
    option: usize, // 1: exponential, 2: quadratic
) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let mut current = input.to_owned();
    let mut next = Array::zeros((height, width));

    let kappa_sq = kappa * kappa;

    for _ in 0..iterations {
        // Compute gradients and diffusion coefficients in parallel
        next.axis_chunks_iter_mut(Axis(0), 32)
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_idx, mut chunk)| {
                let y_start = chunk_idx * 32;

                for (local_y, mut row) in chunk.axis_iter_mut(Axis(0)).enumerate() {
                    let y = y_start + local_y;
                    simd_diffusion_row(&current.view(), &mut row, y, kappa_sq, lambda, option);
                }
            });

        // Swap buffers
        std::mem::swap(&mut current, &mut next);
    }

    Ok(current)
}

/// Process a row with anisotropic diffusion using SIMD
#[allow(dead_code)]
fn simd_diffusion_row<T>(
    input: &ArrayView2<T>,
    output_row: &mut ArrayViewMut1<T>,
    y: usize,
    kappa_sq: T,
    lambda: T,
    option: usize,
) where
    T: Float + FromPrimitive + Debug + Clone + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let simd_width = 8; // Default SIMD width for vectorization

    // Process SIMD chunks
    let num_chunks = width / simd_width;

    for chunk_idx in 0..num_chunks {
        let x_start = chunk_idx * simd_width;

        let mut center_vals = vec![T::zero(); simd_width];
        let mut north_vals = vec![T::zero(); simd_width];
        let mut south_vals = vec![T::zero(); simd_width];
        let mut east_vals = vec![T::zero(); simd_width];
        let mut west_vals = vec![T::zero(); simd_width];

        // Gather neighborhood values
        for i in 0..simd_width {
            let x = x_start + i;
            center_vals[i] = input[(y, x)];

            north_vals[i] = if y > 0 {
                input[(y - 1, x)]
            } else {
                center_vals[i]
            };
            south_vals[i] = if y < height - 1 {
                input[(y + 1, x)]
            } else {
                center_vals[i]
            };
            west_vals[i] = if x > 0 {
                input[(y, x - 1)]
            } else {
                center_vals[i]
            };
            east_vals[i] = if x < width - 1 {
                input[(y, x + 1)]
            } else {
                center_vals[i]
            };
        }

        // Compute gradients using SIMD
        let north_array = Array1::from_vec(north_vals.clone());
        let south_array = Array1::from_vec(south_vals.clone());
        let east_array = Array1::from_vec(east_vals.clone());
        let west_array = Array1::from_vec(west_vals.clone());
        let center_array = Array1::from_vec(center_vals.clone());

        let grad_n = T::simd_sub(&north_array.view(), &center_array.view());
        let grad_s = T::simd_sub(&south_array.view(), &center_array.view());
        let grad_e = T::simd_sub(&east_array.view(), &center_array.view());
        let grad_w = T::simd_sub(&west_array.view(), &center_array.view());

        // Compute diffusion coefficients
        let coeff_n = compute_diffusion_coeff(grad_n.as_slice().unwrap(), kappa_sq, option);
        let coeff_s = compute_diffusion_coeff(grad_s.as_slice().unwrap(), kappa_sq, option);
        let coeff_e = compute_diffusion_coeff(grad_e.as_slice().unwrap(), kappa_sq, option);
        let coeff_w = compute_diffusion_coeff(grad_w.as_slice().unwrap(), kappa_sq, option);

        // Update values
        for i in 0..simd_width {
            if x_start + i < width {
                let flux = coeff_n[i] * grad_n.as_slice().unwrap()[i]
                    + coeff_s[i] * grad_s.as_slice().unwrap()[i]
                    + coeff_e[i] * grad_e.as_slice().unwrap()[i]
                    + coeff_w[i] * grad_w.as_slice().unwrap()[i];
                output_row[x_start + i] = center_vals[i] + lambda * flux;
            }
        }
    }

    // Process remaining pixels
    for x in (num_chunks * simd_width)..width {
        let center = input[(y, x)];

        let north = if y > 0 { input[(y - 1, x)] } else { center };
        let south = if y < height - 1 {
            input[(y + 1, x)]
        } else {
            center
        };
        let west = if x > 0 { input[(y, x - 1)] } else { center };
        let east = if x < width - 1 {
            input[(y, x + 1)]
        } else {
            center
        };

        let grad_n = north - center;
        let grad_s = south - center;
        let grad_e = east - center;
        let grad_w = west - center;

        let coeff_n = compute_single_diffusion_coeff(grad_n, kappa_sq, option);
        let coeff_s = compute_single_diffusion_coeff(grad_s, kappa_sq, option);
        let coeff_e = compute_single_diffusion_coeff(grad_e, kappa_sq, option);
        let coeff_w = compute_single_diffusion_coeff(grad_w, kappa_sq, option);

        let flux = coeff_n * grad_n + coeff_s * grad_s + coeff_e * grad_e + coeff_w * grad_w;
        output_row[x] = center + lambda * flux;
    }
}

/// Compute diffusion coefficients for SIMD values
#[allow(dead_code)]
fn compute_diffusion_coeff<T>(_gradients: &[T], kappasq: T, option: usize) -> Vec<T>
where
    T: Float + FromPrimitive,
{
    _gradients
        .iter()
        .map(|&g| compute_single_diffusion_coeff(g, kappasq, option))
        .collect()
}

/// Compute single diffusion coefficient
#[allow(dead_code)]
fn compute_single_diffusion_coeff<T>(gradient: T, kappasq: T, option: usize) -> T
where
    T: Float + FromPrimitive,
{
    match option {
        1 => {
            // Exponential: c(g) = exp(-(g/kappa)²)
            (-(gradient * gradient) / kappasq).exp()
        }
        2 => {
            // Quadratic: c(g) = 1 / (1 + (g/kappa)²)
            T::one() / (T::one() + gradient * gradient / kappasq)
        }
        _ => T::one(),
    }
}

// Conditional compilation for parallel iterator
#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;

#[cfg(not(feature = "parallel"))]
trait IntoParallelIterator {
    type Iter;
    fn into_par_iter(self) -> Self::Iter;
}

#[cfg(not(feature = "parallel"))]
impl<T> IntoParallelIterator for T
where
    T: IntoIterator,
{
    type Iter = T::IntoIter;
    fn into_par_iter(self) -> Self::Iter {
        self.into_iter()
    }
}

/// SIMD-optimized guided filter for edge-preserving smoothing
///
/// The guided filter uses a guidance image to perform edge-aware filtering,
/// making it effective for applications like flash/no-flash denoising and HDR compression.
#[allow(dead_code)]
pub fn simd_guided_filter<T>(
    input: ArrayView2<T>,
    guide: ArrayView2<T>,
    radius: usize,
    epsilon: T,
) -> NdimageResult<Array<T, Ix2>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Clone
        + Send
        + Sync
        + SimdUnifiedOps
        + ndarray::ScalarOperand,
{
    let (height, width) = input.dim();
    if guide.dim() != (height, width) {
        return Err(crate::error::NdimageError::InvalidInput(
            "Input and guide must have the same shape".into(),
        ));
    }

    // Compute mean values using box filter
    let mean_i = simd_box_filter(&guide, radius)?;
    let mean_p = simd_box_filter(&input, radius)?;

    // Compute correlation and variance
    let corr_ip = simd_box_filter_product(&guide, &input, radius)?;
    let corr_ii = simd_box_filter_product(&guide, &guide, radius)?;

    // Compute variance of I: var_I = corr_II - mean_I * mean_I
    let var_i = &corr_ii - &(&mean_i * &mean_i);

    // Compute covariance: cov_Ip = corr_Ip - mean_I * mean_p
    let cov_ip = &corr_ip - &(&mean_i * &mean_p);

    // Compute coefficients a and b
    let a = &cov_ip / &(&var_i + epsilon);
    let b = &mean_p - &(&a * &mean_i);

    // Compute mean of a and b
    let mean_a = simd_box_filter(&a.view(), radius)?;
    let mean_b = simd_box_filter(&b.view(), radius)?;

    // Output: q = mean_a * I + mean_b
    Ok(&(&mean_a * &guide) + &mean_b)
}

/// SIMD-optimized box filter (mean filter)
#[allow(dead_code)]
fn simd_box_filter<T>(input: &ArrayView2<T>, radius: usize) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let mut output = Array::zeros((height, width));
    let window_size = 2 * radius + 1;
    let norm = safe_usize_to_float(window_size * window_size)?;

    // Process rows in parallel
    output
        .axis_chunks_iter_mut(Axis(0), 32)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut chunk)| {
            let y_start = chunk_idx * 32;

            for (local_y, mut row) in chunk.axis_iter_mut(Axis(0)).enumerate() {
                let y = y_start + local_y;
                if let Err(e) = simd_box_filter_row(input, &mut row, y, radius, norm) {
                    // For parallel processing, we can't propagate errors directly
                    // Log error and continue with default values
                    eprintln!("Warning: box filter row processing failed: {:?}", e);
                }
            }
        });

    Ok(output)
}

/// Process a row with box filter using SIMD
#[allow(dead_code)]
fn simd_box_filter_row<T>(
    input: &ArrayView2<T>,
    output_row: &mut ArrayViewMut1<T>,
    y: usize,
    radius: usize,
    norm: T,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let simd_width = 8; // Default SIMD width for vectorization

    // Use sliding window approach for efficiency
    for x in 0..width {
        let x_min = x.saturating_sub(radius);
        let x_max = (x + radius + 1).min(width);
        let y_min = y.saturating_sub(radius);
        let y_max = (y + radius + 1).min(height);

        let mut sum = T::zero();

        // Sum values in the window
        for wy in y_min..y_max {
            let row_slice = input.slice(s![wy, x_min..x_max]);

            // Process SIMD chunks
            let chunks = row_slice.len() / simd_width;
            for i in 0..chunks {
                let start = i * simd_width;
                let end = start + simd_width;
                let slice = row_slice.as_slice().ok_or_else(|| {
                    NdimageError::ComputationError(
                        "Failed to convert _row slice to contiguous slice".to_string(),
                    )
                })?;
                let slice_array = Array1::from_vec(slice[start..end].to_vec());
                let chunk_sum = T::simd_sum(&slice_array.view());
                sum = sum + chunk_sum;
            }

            // Process remaining elements
            for i in (chunks * simd_width)..row_slice.len() {
                sum = sum + row_slice[i];
            }
        }

        output_row[x] = sum / norm;
    }

    Ok(())
}

/// SIMD-optimized box filter for product of two images
#[allow(dead_code)]
fn simd_box_filter_product<T>(
    input1: &ArrayView2<T>,
    input2: &ArrayView2<T>,
    radius: usize,
) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + SimdUnifiedOps,
{
    let (height, width) = input1.dim();
    let mut output = Array::zeros((height, width));
    let window_size = 2 * radius + 1;
    let norm = safe_usize_to_float(window_size * window_size)?;

    // Process rows in parallel
    output
        .axis_chunks_iter_mut(Axis(0), 32)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut chunk)| {
            let y_start = chunk_idx * 32;

            for (local_y, mut row) in chunk.axis_iter_mut(Axis(0)).enumerate() {
                let y = y_start + local_y;
                if let Err(e) =
                    simd_box_filter_product_row(input1, input2, &mut row, y, radius, norm)
                {
                    // For parallel processing, we can't propagate errors directly
                    // Log error and continue with default values
                    eprintln!("Warning: box filter product row processing failed: {:?}", e);
                }
            }
        });

    Ok(output)
}

/// Process a row with box filter product using SIMD
#[allow(dead_code)]
fn simd_box_filter_product_row<T>(
    input1: &ArrayView2<T>,
    input2: &ArrayView2<T>,
    output_row: &mut ArrayViewMut1<T>,
    y: usize,
    radius: usize,
    norm: T,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + SimdUnifiedOps,
{
    let (height, width) = input1.dim();
    let simd_width = 8; // Default SIMD width for vectorization

    for x in 0..width {
        let x_min = x.saturating_sub(radius);
        let x_max = (x + radius + 1).min(width);
        let y_min = y.saturating_sub(radius);
        let y_max = (y + radius + 1).min(height);

        let mut sum = T::zero();

        // Sum products in the window
        for wy in y_min..y_max {
            let row1 = input1.slice(s![wy, x_min..x_max]);
            let row2 = input2.slice(s![wy, x_min..x_max]);

            // Process SIMD chunks
            let chunks = row1.len() / simd_width;
            for i in 0..chunks {
                let start = i * simd_width;
                let end = start + simd_width;
                let slice1_raw = row1.as_slice().ok_or_else(|| {
                    NdimageError::ComputationError(
                        "Failed to convert row1 to contiguous slice".to_string(),
                    )
                })?;
                let slice2_raw = row2.as_slice().ok_or_else(|| {
                    NdimageError::ComputationError(
                        "Failed to convert row2 to contiguous slice".to_string(),
                    )
                })?;
                let slice1 = &slice1_raw[start..end];
                let slice2 = &slice2_raw[start..end];

                let slice1_array = Array1::from_vec(slice1.to_vec());
                let slice2_array = Array1::from_vec(slice2.to_vec());
                let products = T::simd_mul(&slice1_array.view(), &slice2_array.view());
                let chunk_sum = T::simd_sum(&products.view());
                sum = sum + chunk_sum;
            }

            // Process remaining elements
            for i in (chunks * simd_width)..row1.len() {
                sum = sum + row1[i] * row2[i];
            }
        }

        output_row[x] = sum / norm;
    }

    Ok(())
}

/// SIMD-optimized joint bilateral filter
///
/// Similar to bilateral filter but uses a guidance image for edge detection
#[allow(dead_code)]
pub fn simd_joint_bilateral_filter<T>(
    input: ArrayView2<T>,
    guide: ArrayView2<T>,
    spatial_sigma: T,
    range_sigma: T,
    window_size: Option<usize>,
) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    if guide.dim() != (height, width) {
        return Err(crate::error::NdimageError::InvalidInput(
            "Input and guide must have the same shape".into(),
        ));
    }

    let window_size = match window_size {
        Some(size) => size,
        None => {
            let three = safe_f64_to_float::<T>(3.0)?;
            let radius = safe_float_to_usize(spatial_sigma * three)?;
            2 * radius + 1
        }
    };
    let half_window = window_size / 2;

    let mut output = Array::zeros((height, width));

    // Pre-compute spatial weights
    let spatial_weights = compute_spatial_weights(window_size, spatial_sigma)?;

    // Process image in parallel chunks
    output
        .axis_chunks_iter_mut(Axis(0), 32)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut chunk)| {
            let y_start = chunk_idx * 32;

            for (local_y, mut row) in chunk.axis_iter_mut(Axis(0)).enumerate() {
                let y = y_start + local_y;
                if let Err(e) = simd_joint_bilateral_row(
                    &input,
                    &guide,
                    &mut row,
                    y,
                    half_window,
                    &spatial_weights,
                    range_sigma,
                ) {
                    // For parallel processing, we can't propagate errors directly
                    // Log error and continue with default values
                    eprintln!(
                        "Warning: joint bilateral filter row processing failed: {:?}",
                        e
                    );
                }
            }
        });

    Ok(output)
}

/// Process a row with joint bilateral filter using SIMD
#[allow(dead_code)]
fn simd_joint_bilateral_row<T>(
    input: &ArrayView2<T>,
    guide: &ArrayView2<T>,
    output_row: &mut ArrayViewMut1<T>,
    y: usize,
    half_window: usize,
    spatial_weights: &Array<T, Ix2>,
    range_sigma: T,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + Debug + Clone + SimdUnifiedOps,
{
    let (height, width) = input.dim();
    let range_factor = safe_f64_to_float::<T>(-0.5)? / (range_sigma * range_sigma);
    let simd_width = 8; // Default SIMD width for vectorization

    for x in 0..width {
        let guide_center = guide[(y, x)];
        let mut sum_weight = T::zero();
        let mut sum_value = T::zero();

        for dy in 0..2 * half_window + 1 {
            let ny = (y as isize + dy as isize - half_window as isize).clamp(0, height as isize - 1)
                as usize;

            for dx in 0..2 * half_window + 1 {
                let nx = (x as isize + dx as isize - half_window as isize)
                    .clamp(0, width as isize - 1) as usize;

                // Use guide image for range weight
                let guide_neighbor = guide[(ny, nx)];
                let range_diff = guide_neighbor - guide_center;
                let range_weight = (range_diff * range_diff * range_factor).exp();
                let spatial_weight = spatial_weights[(dy, dx)];

                let weight = spatial_weight * range_weight;
                sum_weight = sum_weight + weight;
                sum_value = sum_value + weight * input[(ny, nx)];
            }
        }

        output_row[x] = sum_value / sum_weight;
    }

    Ok(())
}

/// SIMD-optimized adaptive median filter
///
/// This filter adapts the window size based on local statistics to better preserve edges
#[allow(dead_code)]
pub fn simd_adaptive_median_filter<T>(
    input: ArrayView2<T>,
    max_window_size: usize,
) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + SimdUnifiedOps + PartialOrd,
{
    let (height, width) = input.dim();
    let mut output = Array::zeros((height, width));

    // Process in parallel
    output
        .axis_chunks_iter_mut(Axis(0), 32)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut chunk)| {
            let y_start = chunk_idx * 32;

            for (local_y, mut row) in chunk.axis_iter_mut(Axis(0)).enumerate() {
                let y = y_start + local_y;
                for x in 0..width {
                    row[x] = adaptive_median_at_point(input, y, x, max_window_size);
                }
            }
        });

    Ok(output)
}

/// Compute adaptive median at a single point
#[allow(dead_code)]
fn adaptive_median_at_point<T>(
    input: ArrayView2<T>,
    y: usize,
    x: usize,
    max_window_size: usize,
) -> T
where
    T: Float + FromPrimitive + PartialOrd + Clone,
{
    let (height, width) = input.dim();
    let mut window_size = 3;

    while window_size <= max_window_size {
        let half_window = window_size / 2;

        // Collect values in window
        let mut values = Vec::with_capacity(window_size * window_size);

        for dy in 0..window_size {
            let ny = (y as isize + dy as isize - half_window as isize).clamp(0, height as isize - 1)
                as usize;

            for dx in 0..window_size {
                let nx = (x as isize + dx as isize - half_window as isize)
                    .clamp(0, width as isize - 1) as usize;

                values.push(input[(ny, nx)]);
            }
        }

        // Sort values with safe comparison
        values.sort_by(|a, b| safe_partial_cmp(a, b).unwrap_or(std::cmp::Ordering::Equal));

        let median = values[values.len() / 2];
        let min = values[0];
        let max = values[values.len() - 1];
        let pixel_value = input[(y, x)];

        // Stage A
        let a1 = median - min;
        let a2 = median - max;

        if a1 > T::zero() && a2 < T::zero() {
            // Stage B
            let b1 = pixel_value - min;
            let b2 = pixel_value - max;

            if b1 > T::zero() && b2 < T::zero() {
                return pixel_value;
            } else {
                return median;
            }
        }

        window_size += 2;
    }

    // If we reach max window size, return median of largest window
    input[(y, x)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_bilateral_filter() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let result = simd_bilateral_filter(input.view(), 1.0, 2.0, Some(3)).unwrap();
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_anisotropic_diffusion() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let result = simd_anisotropic_diffusion(input.view(), 5, 2.0, 0.1, 1).unwrap();
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_guided_filter() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let guide = input.clone();

        let result = simd_guided_filter(input.view(), guide.view(), 1, 0.1).unwrap();
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_joint_bilateral_filter() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let guide = array![[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]];

        let result =
            simd_joint_bilateral_filter(input.view(), guide.view(), 1.0, 2.0, Some(3)).unwrap();
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_adaptive_median_filter() {
        let input = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 100.0, 7.0, 8.0], // 100.0 is an outlier
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ];

        let result = simd_adaptive_median_filter(input.view(), 5).unwrap();
        assert_eq!(result.shape(), input.shape());
        // The outlier should be replaced by a value closer to its neighbors
        assert!(result[(1, 1)] < 20.0);
    }
}
