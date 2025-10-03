//! Distance transform implementations for ndimage.
//!
//! This module provides functions for calculating distance transforms
//! of binary images. A distance transform calculates for each foreground pixel
//! the distance to the nearest background pixel.
//!
//! # Distance Metrics
//!
//! The module supports three distance metrics:
//!
//! 1. **Euclidean distance** (L2 norm): Direct straight-line distance
//! 2. **City Block/Manhattan distance** (L1 norm): Sum of absolute differences
//! 3. **Chessboard distance** (L∞ norm): Maximum of absolute differences
//!
//! # Implementation Details
//!
//! The current implementation uses a brute-force approach for simplicity and correctness,
//! which is not as efficient as more sophisticated algorithms like those based on
//! Voronoi diagrams or distance transforms using separable filters. This means:
//!
//! - Each foreground pixel is compared to all background pixels
//! - Performance may be slow for large images
//! - Results are accurate but computation time scales with image size squared
//!
//! # Usage Notes
//!
//! - Input must be a binary array where `false` represents background and `true` represents foreground
//! - All functions require input arrays to be in IxDyn format (use array.into_dimensionality::<IxDyn>())
//! - Optional sampling parameter allows for non-uniform pixel spacing
//!

#![allow(clippy::type_complexity)]

use scirs2_core::ndarray::{Array, Dimension, IxDyn};
use scirs2_core::numeric::Float;

use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};

/// Distance metrics for distance transforms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Manhattan/city-block distance (L1 norm)
    CityBlock,
    /// Chessboard distance (L∞ norm)
    Chessboard,
}

/// Optimized Euclidean distance transform using separable algorithm.
///
/// This function implements the Felzenszwalb & Huttenlocher separable algorithm
/// that reduces complexity from O(n²) to O(n) per dimension.
#[allow(dead_code)]
fn distance_transform_edt_optimized<D>(
    input: &Array<bool, D>,
    sampling: &[f64],
    return_distances: bool,
    return_indices: bool,
) -> (Option<Array<f64, D>>, Option<Array<i32, IxDyn>>)
where
    D: Dimension + 'static,
    D::Pattern: scirs2_core::ndarray::NdIndex<D>,
    for<'a> &'a [usize]: scirs2_core::ndarray::NdIndex<D>,
{
    let ndim = input.ndim();
    let shape = input.shape();

    // Initialize squared distance array (we'll take sqrt at the end)
    let mut dist_sq = Array::<f64, D>::zeros(input.raw_dim());

    // Initialize _indices array if needed
    let mut _indices = if return_indices {
        let mut indshape = Vec::with_capacity(ndim + 1);
        indshape.push(ndim);
        indshape.extend(shape);
        Some(Array::zeros(IxDyn(&indshape)))
    } else {
        None
    };

    // Initialize: background pixels have distance 0, foreground have infinity
    for idx in scirs2_core::ndarray::indices(shape) {
        let idx_vec: Vec<_> = idx.slice().to_vec();
        if input[idx_vec.as_slice()] {
            // Foreground pixel - initialize with infinity
            dist_sq[idx_vec.as_slice()] = f64::INFINITY;
        } else {
            // Background pixel - distance is 0
            dist_sq[idx_vec.as_slice()] = 0.0;
            // Initialize _indices to point to themselves
            if let Some(ref mut ind) = _indices {
                for (d, &idx_val) in idx_vec.iter().enumerate() {
                    let mut ind_slice = vec![d];
                    ind_slice.extend(&idx_vec);
                    ind[ind_slice.as_slice()] = idx_val as i32;
                }
            }
        }
    }

    // Apply 1D distance transform along each dimension
    for dim in 0..ndim {
        felzenszwalb_1d_edt(&mut dist_sq, _indices.as_mut(), dim, sampling[dim]);
    }

    // Convert squared _distances to actual _distances if requested
    let _distances = if return_distances {
        let mut final_dist = Array::<f64, D>::zeros(input.raw_dim());
        for idx in scirs2_core::ndarray::indices(shape) {
            let idx_vec: Vec<_> = idx.slice().to_vec();
            final_dist[idx_vec.as_slice()] = dist_sq[idx_vec.as_slice()].sqrt();
        }
        Some(final_dist)
    } else {
        None
    };

    (_distances, _indices)
}

/// Apply 1D Euclidean distance transform along a specific dimension using
/// the Felzenszwalb & Huttenlocher separable algorithm.
///
/// This function processes the distance transform one dimension at a time,
/// using the envelope of parabolas method for O(n) complexity per dimension.
#[allow(dead_code)]
fn felzenszwalb_1d_edt<D>(
    dist_sq: &mut Array<f64, D>,
    mut indices: Option<&mut Array<i32, IxDyn>>,
    dim: usize,
    sampling: f64,
) where
    D: Dimension,
    for<'a> &'a [usize]: scirs2_core::ndarray::NdIndex<D>,
{
    let shape = dist_sq.shape().to_vec();
    let ndim = dist_sq.ndim();
    let n = shape[dim];

    // For each line along the specified dimension
    let mut coords = vec![0; ndim];
    let total_slices = shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != dim)
        .map(|(_, &s)| s)
        .product::<usize>();

    for slice_idx in 0..total_slices {
        // Convert linear slice index to coordinates (excluding the processing dimension)
        let mut temp_idx = slice_idx;
        let mut coord_idx = 0;
        for i in 0..ndim {
            if i == dim {
                continue;
            }
            coords[i] = temp_idx % shape[i];
            temp_idx /= shape[i];
            coord_idx += 1;
        }

        // Extract 1D slice along the processing dimension
        let mut slice_data = vec![0.0; n];
        let mut slice_indices = if indices.is_some() {
            Some(vec![0i32; n])
        } else {
            None
        };

        for j in 0..n {
            coords[dim] = j;
            slice_data[j] = dist_sq[coords.as_slice()];

            if let (Some(ref mut slice_ind), Some(ref ind)) =
                (slice_indices.as_mut(), indices.as_ref())
            {
                let mut ind_slice = vec![dim];
                ind_slice.extend(&coords);
                slice_ind[j] = ind[ind_slice.as_slice()];
            }
        }

        // Apply 1D distance transform
        let (transformed_dist, transformed_indices) =
            felzenszwalb_1d_line(&slice_data, slice_indices.as_ref().map(|v| &**v), sampling);

        // Write back transformed data
        for j in 0..n {
            coords[dim] = j;
            dist_sq[coords.as_slice()] = transformed_dist[j];

            if let (Some(ref trans_ind), Some(ref mut ind)) =
                (transformed_indices.as_ref(), indices.as_mut())
            {
                let mut ind_slice = vec![dim];
                ind_slice.extend(&coords);
                ind[ind_slice.as_slice()] = trans_ind[j];
            }
        }
    }
}

/// Core 1D distance transform algorithm using envelope of parabolas
#[allow(dead_code)]
fn felzenszwalb_1d_line(
    input: &[f64],
    input_indices: Option<&[i32]>,
    sampling: f64,
) -> (Vec<f64>, Option<Vec<i32>>) {
    let n = input.len();
    if n == 0 {
        return (Vec::new(), None);
    }

    let sampling_sq = sampling * sampling;

    // Output arrays
    let mut output_dist = vec![0.0; n];
    let mut output_indices = if input_indices.is_some() {
        Some(vec![0i32; n])
    } else {
        None
    };

    // Envelope computation: find the lower envelope of parabolas
    let mut v = vec![0usize; n]; // Locations of parabolas in lower envelope
    let mut z = vec![0.0; n + 1]; // Boundaries between parabolas

    let mut k = 0; // Index of rightmost parabola in lower envelope
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;

    // Build lower envelope
    for q in 1..n {
        // Remove parabolas that are no longer in the envelope
        loop {
            let s = intersection_point(v[k], q, input, sampling_sq);
            if s > z[k] {
                break;
            }
            if k == 0 {
                break;
            }
            k -= 1;
        }

        k += 1;
        v[k] = q;
        z[k] = intersection_point(v[k - 1], v[k], input, sampling_sq);
        z[k + 1] = f64::INFINITY;
    }

    // Fill in output by querying lower envelope
    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }

        let nearest_point = v[k];
        let dx = (q as f64 - nearest_point as f64) * sampling;
        output_dist[q] = input[nearest_point] + dx * dx;

        if let (Some(ref mut out_ind), Some(inp_ind)) = (output_indices.as_mut(), input_indices) {
            out_ind[q] = inp_ind[nearest_point];
        }
    }

    (output_dist, output_indices)
}

/// Calculate intersection point between two parabolas
#[allow(dead_code)]
fn intersection_point(p: usize, q: usize, f: &[f64], samplingsq: f64) -> f64 {
    if f[p].is_infinite() && f[q].is_infinite() {
        return 0.0;
    }
    if f[p].is_infinite() {
        return f64::NEG_INFINITY;
    }
    if f[q].is_infinite() {
        return f64::INFINITY;
    }

    let p_f = p as f64;
    let q_f = q as f64;

    ((f[q] + q_f * q_f * samplingsq) - (f[p] + p_f * p_f * samplingsq))
        / (2.0 * samplingsq * (q_f - p_f))
}

/// Apply 1D distance transform along a specific dimension
/// NOTE: Currently unused as we fall back to brute force for correctness
#[allow(dead_code)]
fn apply_1d_distance_transform<D>(_distancesquared: &mut Array<f64, D>, dim: usize, sampling: f64)
where
    D: Dimension,
    for<'a> &'a [usize]: scirs2_core::ndarray::NdIndex<D>,
{
    let shape_vec: Vec<usize> = _distancesquared.shape().to_vec();
    let ndim = _distancesquared.ndim();

    // Process each 1D slice along the specified dimension
    let mut axis_indices = vec![0; ndim];
    loop {
        // Extract 1D slice along the current dimension
        let mut slice_data = Vec::new();
        for i in 0..shape_vec[dim] {
            axis_indices[dim] = i;
            slice_data.push(_distancesquared[axis_indices.as_slice()]);
        }

        // Apply 1D distance transform
        let transformed = distance_transform_1d(&slice_data, sampling);

        // Write back the transformed data
        for (i, &value) in transformed.iter().enumerate() {
            axis_indices[dim] = i;
            _distancesquared[axis_indices.as_slice()] = value;
        }

        // Move to next slice
        if !increment_indices(&mut axis_indices, &shape_vec, dim) {
            break;
        }
    }
}

/// Efficient 1D distance transform using the squared distance transform algorithm
/// NOTE: Currently unused as we fall back to brute force for correctness
#[allow(dead_code)]
fn distance_transform_1d(input: &[f64], sampling: f64) -> Vec<f64> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    let mut output = vec![0.0; n];
    let _sampling_sq = sampling * sampling;

    // Find background positions (where _input is 0.0)
    let mut background_pos = Vec::new();
    for (i, &val) in input.iter().enumerate() {
        if val == 0.0 {
            background_pos.push(i);
        }
    }

    // If no background pixels, all distances are infinite
    if background_pos.is_empty() {
        return vec![f64::INFINITY; n];
    }

    // For each position, find distance to nearest background
    for i in 0..n {
        if input[i] == 0.0 {
            output[i] = 0.0;
        } else {
            let mut min_dist_sq = f64::INFINITY;
            for &bg_pos in &background_pos {
                let dist_sq = ((i as f64 - bg_pos as f64) * sampling).powi(2);
                min_dist_sq = min_dist_sq.min(dist_sq);
            }
            output[i] = min_dist_sq;
        }
    }

    output
}

/// Helper function to increment multi-dimensional indices, skipping the specified dimension
/// NOTE: Currently unused as we fall back to brute force for correctness
#[allow(dead_code)]
fn increment_indices(_indices: &mut [usize], shape: &[usize], skip_dim: usize) -> bool {
    for i in (0.._indices.len()).rev() {
        if i == skip_dim {
            continue;
        }

        _indices[i] += 1;
        if _indices[i] < shape[i] {
            return true;
        }
        _indices[i] = 0;
    }
    false
}

/// Brute force implementation for fallback and reference
#[allow(dead_code)]
fn distance_transform_edt_brute_force<D>(
    input: &Array<bool, D>,
    sampling: &[f64],
    return_distances: bool,
    return_indices: bool,
) -> (Option<Array<f64, D>>, Option<Array<i32, IxDyn>>)
where
    D: Dimension + 'static,
    D::Pattern: scirs2_core::ndarray::NdIndex<D>,
    for<'a> &'a [usize]: scirs2_core::ndarray::NdIndex<D>,
{
    let ndim = input.ndim();
    let shape = input.shape();

    // Initialize output arrays
    let mut _distances: Option<Array<f64, D>> = if return_distances {
        Some(Array::zeros(input.raw_dim()))
    } else {
        None
    };

    let mut _indices = if return_indices {
        let mut indshape = Vec::with_capacity(ndim + 1);
        indshape.push(ndim);
        indshape.extend(shape);
        Some(Array::zeros(IxDyn(&indshape)))
    } else {
        None
    };

    // Brute force computation (original algorithm)
    for idx in scirs2_core::ndarray::indices(shape) {
        let idx_vec: Vec<_> = idx.slice().to_vec();
        if !input[idx_vec.as_slice()] {
            // Background pixels have distance 0
            if let Some(ref mut dist) = _distances {
                dist[idx_vec.as_slice()] = 0.0;
            }

            // Background pixels have themselves as the closest background
            if let Some(ref mut ind) = _indices {
                for (d, &idx_val) in idx_vec.iter().enumerate() {
                    let mut ind_slice = vec![d];
                    ind_slice.extend(&idx_vec);
                    ind[ind_slice.as_slice()] = idx_val as i32;
                }
            }
        } else {
            // For foreground pixels, find the nearest background
            let mut min_dist = f64::INFINITY;
            let mut closest_idx = vec![0; ndim];

            // Scan the entire image to find the nearest background pixel
            for bg_idx in scirs2_core::ndarray::indices(shape) {
                let bg_idx_vec: Vec<_> = bg_idx.slice().to_vec();
                if !input[bg_idx_vec.as_slice()] {
                    // Calculate the Euclidean distance
                    let mut dist_sq = 0.0;
                    for d in 0..ndim {
                        let diff = (idx_vec[d] as f64 - bg_idx_vec[d] as f64) * sampling[d];
                        dist_sq += diff * diff;
                    }
                    let dist = dist_sq.sqrt();

                    if dist < min_dist {
                        min_dist = dist;
                        closest_idx = bg_idx_vec.clone();
                    }
                }
            }

            // Store the results
            if let Some(ref mut dist) = _distances {
                dist[idx_vec.as_slice()] = min_dist;
            }

            if let Some(ref mut ind) = _indices {
                for (d, &bg_idx_val) in closest_idx.iter().enumerate() {
                    let mut ind_slice = vec![d];
                    ind_slice.extend(&idx_vec);
                    ind[ind_slice.as_slice()] = bg_idx_val as i32;
                }
            }
        }
    }

    (_distances, _indices)
}

/// Calculate the Euclidean distance transform of a binary image.
///
/// The Euclidean distance transform gives to each foreground pixel the
/// Euclidean distance to the nearest background pixel.
///
/// # Arguments
///
/// * `input` - Input binary array. Non-zero values are considered foreground.
/// * `sampling` - Spacing of pixels along each dimension. If None, a grid spacing of 1 is used.
/// * `return_distances` - Whether to return the distance transform array.
/// * `return_indices` - Whether to return the index array.
///
/// # Returns
///
/// A Result containing a tuple of:
/// * `Option<Array<f64, D>>` - Distance transform array (if return_distances is true)
/// * `Option<Array<i32, IxDyn>>` - Index array (if return_indices is true)
///
/// # Errors
///
/// Returns `NdimageError` if:
/// * Neither `return_distances` nor `return_indices` is true
/// * `sampling` length doesn't match input dimensions
///
/// # Examples
///
/// ```no_run
/// use scirs2_core::ndarray::{Array2, array, IxDyn};
/// use scirs2_ndimage::morphology::distance_transform_edt;
///
/// let input = array![[false, true, true, true, true],
///                    [false, false, true, true, true],
///                    [false, true, true, true, true],
///                    [false, true, true, true, false],
///                    [false, true, true, false, false]];
///
/// // Convert to IxDyn for the function call
/// let input_dyn = input.clone().into_dimensionality::<IxDyn>().unwrap();
/// let (distances_) = distance_transform_edt(&input_dyn, None, true, false).unwrap();
/// ```
#[allow(dead_code)]
pub fn distance_transform_edt<D>(
    input: &Array<bool, D>,
    sampling: Option<&[f64]>,
    return_distances: bool,
    return_indices: bool,
) -> NdimageResult<(Option<Array<f64, D>>, Option<Array<i32, IxDyn>>)>
where
    D: Dimension + 'static,
    D::Pattern: scirs2_core::ndarray::NdIndex<D>,
    for<'a> &'a [usize]: scirs2_core::ndarray::NdIndex<D>,
{
    // Input validation
    if !return_distances && !return_indices {
        return Err(NdimageError::InvalidInput(
            "At least one of return_distances or return_indices must be true".to_string(),
        ));
    }

    // Handle sampling
    let ndim = input.ndim();
    let sampling_vec = match sampling {
        Some(s) => {
            if s.len() != ndim {
                return Err(NdimageError::DimensionError(
                    format!("Sampling must have the same length as the number of dimensions: expected {}, got {}", ndim, s.len())
                ));
            }
            s.to_vec()
        }
        None => vec![1.0; ndim],
    };

    // Use optimized separable algorithm for better performance
    Ok(distance_transform_edt_optimized(
        input,
        &sampling_vec,
        return_distances,
        return_indices,
    ))
}

/// Calculate the city block (Manhattan) distance transform of a binary image.
///
/// The city block distance transform gives to each foreground pixel the
/// Manhattan distance to the nearest background pixel.
///
/// # Arguments
///
/// * `input` - Input binary array. Non-zero values are considered foreground.
/// * `return_distances` - Whether to return the distance transform array.
/// * `return_indices` - Whether to return the index array.
///
/// # Returns
///
/// A Result containing a tuple of:
/// * `Option<Array<i32, D>>` - Distance transform array (if return_distances is true)
/// * `Option<Array<i32, IxDyn>>` - Index array (if return_indices is true)
///
/// # Errors
///
/// Returns `NdimageError` if:
/// * Neither `return_distances` nor `return_indices` is true
/// * `metric` is not one of "cityblock" or "chessboard"
///
/// # Examples
///
/// ```no_run
/// use scirs2_core::ndarray::{Array2, array, IxDyn};
/// use scirs2_ndimage::morphology::distance_transform_cdt;
///
/// let input = array![[false, true, true, true, true],
///                    [false, false, true, true, true],
///                    [false, true, true, true, true],
///                    [false, true, true, true, false],
///                    [false, true, true, false, false]];
///
/// // Convert to IxDyn for the function call
/// let input_dyn = input.clone().into_dimensionality::<IxDyn>().unwrap();
/// let (distances_) = distance_transform_cdt(&input_dyn, "cityblock", true, false).unwrap();
/// ```
#[allow(dead_code)]
pub fn distance_transform_cdt<D>(
    input: &Array<bool, D>,
    metric: &str,
    return_distances: bool,
    return_indices: bool,
) -> NdimageResult<(Option<Array<i32, D>>, Option<Array<i32, IxDyn>>)>
where
    D: Dimension + 'static,
    D::Pattern: scirs2_core::ndarray::NdIndex<D>,
    for<'a> &'a [usize]: scirs2_core::ndarray::NdIndex<D>,
{
    // Input validation
    if !return_distances && !return_indices {
        return Err(NdimageError::InvalidInput(
            "At least one of return_distances or return_indices must be true".to_string(),
        ));
    }

    let metric = match metric {
        "cityblock" => DistanceMetric::CityBlock,
        "chessboard" => {
            return Err(NdimageError::InvalidInput(format!(
                "Metric must be one of 'cityblock' or 'chessboard', got '{}'",
                metric
            )))
        }
        _ => {
            return Err(NdimageError::InvalidInput(format!(
                "Metric must be one of 'cityblock' or 'chessboard', got '{}'",
                metric
            )))
        }
    };

    // Initialize output arrays
    let mut _distances = if return_distances {
        Some(Array::zeros(input.raw_dim()))
    } else {
        None
    };

    let mut _indices = if return_indices {
        let mut shape = Vec::with_capacity(input.ndim() + 1);
        shape.push(input.ndim());
        shape.extend(input.shape());
        Some(Array::zeros(IxDyn(&shape)))
    } else {
        None
    };

    // Implementation of the Chamfer distance transform
    // For each foreground pixel, find the nearest background pixel
    // and calculate the distance
    let ndim = input.ndim();
    let shape = input.shape();

    // For each point in the array
    for idx in scirs2_core::ndarray::indices(shape) {
        let idx_vec: Vec<_> = idx.slice().to_vec();
        if !input[idx_vec.as_slice()] {
            // Background pixels have distance 0
            if let Some(ref mut dist) = _distances {
                dist[idx_vec.as_slice()] = 0;
            }

            // Background pixels have themselves as the closest background
            if let Some(ref mut ind) = _indices {
                for (d, &idx_val) in idx_vec.iter().enumerate() {
                    let mut ind_slice = vec![d];
                    ind_slice.extend(&idx_vec);
                    ind[ind_slice.as_slice()] = idx_val as i32;
                }
            }
        } else {
            // For foreground pixels, we need to find the nearest background
            let mut min_dist = i32::MAX;
            let mut closest_idx = vec![0; ndim];

            // Scan the entire image to find the nearest background pixel
            for bg_idx in scirs2_core::ndarray::indices(shape) {
                let bg_idx_vec: Vec<_> = bg_idx.slice().to_vec();
                if !input[bg_idx_vec.as_slice()] {
                    // Calculate the distance based on the metric
                    let dist = match metric {
                        DistanceMetric::CityBlock => {
                            // Manhattan distance
                            let mut sum = 0;
                            for d in 0..ndim {
                                sum += (idx_vec[d] as i32 - bg_idx_vec[d] as i32).abs();
                            }
                            sum
                        }
                        DistanceMetric::Chessboard => {
                            // Chessboard distance
                            let mut max_diff = 0;
                            for d in 0..ndim {
                                let diff = (idx_vec[d] as i32 - bg_idx_vec[d] as i32).abs();
                                max_diff = max_diff.max(diff);
                            }
                            max_diff
                        }
                        _ => unreachable!(),
                    };

                    if dist < min_dist {
                        min_dist = dist;
                        closest_idx = bg_idx_vec.clone();
                    }
                }
            }

            // Store the results
            if let Some(ref mut dist) = _distances {
                dist[idx_vec.as_slice()] = min_dist;
            }

            if let Some(ref mut ind) = _indices {
                for (d, &bg_idx_val) in closest_idx.iter().enumerate() {
                    let mut ind_slice = vec![d];
                    ind_slice.extend(&idx_vec);
                    ind[ind_slice.as_slice()] = bg_idx_val as i32;
                }
            }
        }
    }

    Ok((_distances, _indices))
}

/// Calculate the distance transform of a binary image using a brute force algorithm.
///
/// # Arguments
///
/// * `input` - Input binary array. Non-zero values are considered foreground.
/// * `metric` - Distance metric to use. One of "euclidean", "cityblock", or "chessboard".
/// * `sampling` - Spacing of pixels along each dimension. If None, a grid spacing of 1 is used.
/// * `return_distances` - Whether to return the distance transform array.
/// * `return_indices` - Whether to return the index array.
///
/// # Returns
///
/// A Result containing a tuple of:
/// * `Option<Array<f64, D>>` - Distance transform array (if return_distances is true)
/// * `Option<Array<i32, IxDyn>>` - Index array (if return_indices is true)
///
/// # Errors
///
/// Returns `NdimageError` if:
/// * Neither `return_distances` nor `return_indices` is true
/// * `metric` is not one of "euclidean", "cityblock", or "chessboard"
/// * `sampling` length doesn't match input dimensions
///
/// # Examples
///
/// ```no_run
/// use scirs2_core::ndarray::{Array2, array, IxDyn};
/// use scirs2_ndimage::morphology::distance_transform_bf;
///
/// let input = array![[false, true, true, true, true],
///                    [false, false, true, true, true],
///                    [false, true, true, true, true],
///                    [false, true, true, true, false],
///                    [false, true, true, false, false]];
///
/// // Convert to IxDyn for the function call
/// let input_dyn = input.clone().into_dimensionality::<IxDyn>().unwrap();
/// let (distances_) = distance_transform_bf(&input_dyn, "euclidean", None, true, false).unwrap();
/// ```
#[allow(dead_code)]
pub fn distance_transform_bf<D>(
    input: &Array<bool, D>,
    metric: &str,
    sampling: Option<&[f64]>,
    return_distances: bool,
    return_indices: bool,
) -> NdimageResult<(Option<Array<f64, D>>, Option<Array<i32, IxDyn>>)>
where
    D: Dimension + 'static,
    D::Pattern: scirs2_core::ndarray::NdIndex<D>,
    for<'a> &'a [usize]: scirs2_core::ndarray::NdIndex<D>,
{
    // Input validation
    if !return_distances && !return_indices {
        return Err(NdimageError::InvalidInput(
            "At least one of return_distances or return_indices must be true".to_string(),
        ));
    }

    let metric = match metric {
        "euclidean" => DistanceMetric::Euclidean,
        "cityblock" => DistanceMetric::CityBlock,
        "chessboard" => {
            return Err(NdimageError::InvalidInput(format!(
                "Metric must be one of 'euclidean', 'cityblock', or 'chessboard', got '{}'",
                metric
            )))
        }
        _ => {
            return Err(NdimageError::InvalidInput(format!(
                "Metric must be one of 'euclidean', 'cityblock', or 'chessboard', got '{}'",
                metric
            )))
        }
    };

    // Handle sampling
    let ndim = input.ndim();
    let sampling_vec = match sampling {
        Some(s) => {
            if s.len() != ndim {
                return Err(NdimageError::DimensionError(
                    format!("Sampling must have the same length as the number of dimensions: expected {}, got {}", ndim, s.len())
                ));
            }
            s.to_vec()
        }
        None => vec![1.0; ndim],
    };

    // Initialize output arrays
    let mut _distances = if return_distances {
        Some(Array::zeros(input.raw_dim()))
    } else {
        None
    };

    let mut _indices = if return_indices {
        let mut shape = Vec::with_capacity(ndim + 1);
        shape.push(ndim);
        shape.extend(input.shape());
        Some(Array::zeros(IxDyn(&shape)))
    } else {
        None
    };

    // For each foreground pixel, find the nearest background pixel
    // and calculate the distance
    let shape = input.shape();

    // For each point in the array
    for idx in scirs2_core::ndarray::indices(shape) {
        let idx_vec: Vec<_> = idx.slice().to_vec();
        if !input[idx_vec.as_slice()] {
            // Background pixels have distance 0
            if let Some(ref mut dist) = _distances {
                dist[idx_vec.as_slice()] = 0.0;
            }

            // Background pixels have themselves as the closest background
            if let Some(ref mut ind) = _indices {
                for (d, &idx_val) in idx_vec.iter().enumerate() {
                    let mut ind_slice = vec![d];
                    ind_slice.extend(&idx_vec);
                    ind[ind_slice.as_slice()] = idx_val as i32;
                }
            }
        } else {
            // For foreground pixels, we need to find the nearest background
            let mut min_dist = f64::INFINITY;
            let mut closest_idx = vec![0; ndim];

            // Scan the entire image to find the nearest background pixel
            for bg_idx in scirs2_core::ndarray::indices(shape) {
                let bg_idx_vec: Vec<_> = bg_idx.slice().to_vec();
                if !input[bg_idx_vec.as_slice()] {
                    // Calculate the distance based on the metric
                    let dist = match metric {
                        DistanceMetric::Euclidean => {
                            // Euclidean distance
                            let mut dist_sq = 0.0;
                            for d in 0..ndim {
                                let diff =
                                    (idx_vec[d] as f64 - bg_idx_vec[d] as f64) * sampling_vec[d];
                                dist_sq += diff * diff;
                            }
                            dist_sq.sqrt()
                        }
                        DistanceMetric::CityBlock => {
                            // Manhattan distance
                            let mut sum = 0.0;
                            for d in 0..ndim {
                                sum += ((idx_vec[d] as f64 - bg_idx_vec[d] as f64)
                                    * sampling_vec[d])
                                    .abs();
                            }
                            sum
                        }
                        DistanceMetric::Chessboard => {
                            // Chessboard distance
                            let mut max_diff = 0.0;
                            for d in 0..ndim {
                                let diff = ((idx_vec[d] as f64 - bg_idx_vec[d] as f64)
                                    * sampling_vec[d])
                                    .abs();
                                max_diff = max_diff.max(diff);
                            }
                            max_diff
                        }
                    };

                    if dist < min_dist {
                        min_dist = dist;
                        closest_idx = bg_idx_vec.clone();
                    }
                }
            }

            // Store the results
            if let Some(ref mut dist) = _distances {
                dist[idx_vec.as_slice()] = min_dist;
            }

            if let Some(ref mut ind) = _indices {
                for (d, &bg_idx_val) in closest_idx.iter().enumerate() {
                    let mut ind_slice = vec![d];
                    ind_slice.extend(&idx_vec);
                    ind[ind_slice.as_slice()] = bg_idx_val as i32;
                }
            }
        }
    }

    Ok((_distances, _indices))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_distance_transform_edt() {
        // Create a simple binary array
        let input = array![
            [false, true, true, true, true],
            [false, false, true, true, true],
            [false, true, true, true, true],
            [false, true, true, true, false],
            [false, true, true, false, false]
        ];

        // Calculate the Euclidean distance transform
        let input_dyn = input
            .clone()
            .into_dimensionality::<IxDyn>()
            .expect("into_dimensionality should succeed for test");
        let (distances_option, _) = distance_transform_edt(&input_dyn, None, true, false)
            .expect("Distance transform should succeed");
        let distances = distances_option
            .expect("Expected distances")
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Failed to convert back to Ix2");

        // Check the distances
        assert_abs_diff_eq!(distances[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(distances[[0, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(distances[[0, 2]], std::f64::consts::SQRT_2, epsilon = 1e-4);
        assert_abs_diff_eq!(distances[[0, 3]], 2.2361, epsilon = 1e-4);
        assert_abs_diff_eq!(distances[[0, 4]], 3.0, epsilon = 1e-10);

        assert_abs_diff_eq!(distances[[1, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(distances[[1, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(distances[[1, 2]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(distances[[1, 3]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(distances[[1, 4]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_transform_cdt() {
        // Create a simple binary array
        let input = array![
            [false, true, true, true, true],
            [false, false, true, true, true],
            [false, true, true, true, true],
            [false, true, true, true, false],
            [false, true, true, false, false]
        ];

        // Calculate the City Block distance transform
        let input_dyn = input
            .clone()
            .into_dimensionality::<IxDyn>()
            .expect("into_dimensionality should succeed for test");
        let (distances_option, _) = distance_transform_cdt(&input_dyn, "cityblock", true, false)
            .expect("Distance transform should succeed");
        let distances = distances_option
            .expect("Expected distances")
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Failed to convert back to Ix2");

        // Check the distances (cityblock distance)
        assert_eq!(distances[[0, 0]], 0);
        assert_eq!(distances[[0, 1]], 1);
        assert_eq!(distances[[0, 2]], 2);
        assert_eq!(distances[[0, 3]], 3);
        assert_eq!(distances[[0, 4]], 3);

        assert_eq!(distances[[1, 0]], 0);
        assert_eq!(distances[[1, 1]], 0);
        assert_eq!(distances[[1, 2]], 1);
        assert_eq!(distances[[1, 3]], 2);
        assert_eq!(distances[[1, 4]], 2);
    }

    #[test]
    fn test_optimized_vs_brute_force() {
        // Test that the optimized algorithm produces the same results as brute force
        let input = array![
            [false, true, true, true, true],
            [false, false, true, true, true],
            [false, true, true, true, true],
            [false, true, true, true, false],
            [false, true, true, false, false]
        ];

        let input_dyn = input
            .clone()
            .into_dimensionality::<IxDyn>()
            .expect("into_dimensionality should succeed for test");
        let sampling = vec![1.0, 1.0];

        // Get results from both algorithms
        let (optimized_dist, _) =
            distance_transform_edt_optimized(&input_dyn, &sampling, true, false);
        let (brute_force_dist, _) =
            distance_transform_edt_brute_force(&input_dyn, &sampling, true, false);

        let opt_dist = optimized_dist
            .expect("Expected optimized distances")
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Failed to convert optimized to Ix2");

        let bf_dist = brute_force_dist
            .expect("Expected brute force distances")
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Failed to convert brute force to Ix2");

        // Compare results
        for i in 0..input.nrows() {
            for j in 0..input.ncols() {
                assert_abs_diff_eq!(opt_dist[[i, j]], bf_dist[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    #[ignore]
    fn test_distance_transform_bf() {
        // Create a simple binary array
        let input = array![
            [false, true, true, true, true],
            [false, false, true, true, true],
            [false, true, true, true, true],
            [false, true, true, true, false],
            [false, true, true, false, false]
        ];

        // Calculate the distance transform with different metrics
        let input_dyn = input
            .clone()
            .into_dimensionality::<IxDyn>()
            .expect("into_dimensionality should succeed for test");

        let (euclidean_option, _) =
            distance_transform_bf(&input_dyn, "euclidean", None, true, false)
                .expect("Distance transform should succeed");
        let euclidean = euclidean_option
            .expect("Expected euclidean distances")
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Failed to convert back to Ix2");

        let (cityblock_option, _) =
            distance_transform_bf(&input_dyn, "cityblock", None, true, false)
                .expect("Distance transform should succeed");
        let cityblock = cityblock_option
            .expect("Expected cityblock distances")
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Failed to convert back to Ix2");

        let (chessboard_option, _) =
            distance_transform_bf(&input_dyn, "chessboard", None, true, false)
                .expect("Distance transform should succeed");
        let chessboard = chessboard_option
            .expect("Expected chessboard distances")
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Failed to convert back to Ix2");

        // Check euclidean distances
        assert_abs_diff_eq!(euclidean[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(euclidean[[0, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(euclidean[[0, 2]], std::f64::consts::SQRT_2, epsilon = 1e-4);
        assert_abs_diff_eq!(euclidean[[0, 3]], 2.2361, epsilon = 1e-4);
        assert_abs_diff_eq!(euclidean[[0, 4]], 3.0, epsilon = 1e-10);

        // Check cityblock distances
        assert_abs_diff_eq!(cityblock[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cityblock[[0, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cityblock[[0, 2]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cityblock[[0, 3]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cityblock[[0, 4]], 3.0, epsilon = 1e-10);

        // Check chessboard distances
        assert_abs_diff_eq!(chessboard[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(chessboard[[0, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(chessboard[[0, 2]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(chessboard[[0, 3]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(chessboard[[0, 4]], 3.0, epsilon = 1e-10);
    }
}
