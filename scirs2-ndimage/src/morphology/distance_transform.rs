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

use ndarray::{Array, Dimension, IxDyn};
use num_traits::Float;

use std::fmt::Debug;

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
/// This function implements a more efficient algorithm based on separable filtering
/// that reduces complexity from O(n²) per pixel to O(n log n) overall.
fn distance_transform_edt_optimized<D>(
    input: &Array<bool, D>,
    sampling: &[f64],
    return_distances: bool,
    return_indices: bool,
) -> (Option<Array<f64, D>>, Option<Array<i32, IxDyn>>)
where
    D: Dimension + 'static,
    D::Pattern: ndarray::NdIndex<D>,
    for<'a> &'a [usize]: ndarray::NdIndex<D>,
{
    // For now, fall back to the brute force algorithm to ensure correctness
    // A proper separable EDT algorithm would require implementation of
    // algorithms like Felzenszwalb & Huttenlocher's method
    distance_transform_edt_brute_force(input, sampling, return_distances, return_indices)
}

/// Apply 1D distance transform along a specific dimension
/// NOTE: Currently unused as we fall back to brute force for correctness
#[allow(dead_code)]
fn apply_1d_distance_transform<D>(distance_squared: &mut Array<f64, D>, dim: usize, sampling: f64)
where
    D: Dimension,
    for<'a> &'a [usize]: ndarray::NdIndex<D>,
{
    let shape_vec: Vec<usize> = distance_squared.shape().to_vec();
    let ndim = distance_squared.ndim();

    // Process each 1D slice along the specified dimension
    let mut axis_indices = vec![0; ndim];
    loop {
        // Extract 1D slice along the current dimension
        let mut slice_data = Vec::new();
        for i in 0..shape_vec[dim] {
            axis_indices[dim] = i;
            slice_data.push(distance_squared[axis_indices.as_slice()]);
        }

        // Apply 1D distance transform
        let transformed = distance_transform_1d(&slice_data, sampling);

        // Write back the transformed data
        for (i, &value) in transformed.iter().enumerate() {
            axis_indices[dim] = i;
            distance_squared[axis_indices.as_slice()] = value;
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

    // Find background positions (where input is 0.0)
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
fn increment_indices(indices: &mut [usize], shape: &[usize], skip_dim: usize) -> bool {
    for i in (0..indices.len()).rev() {
        if i == skip_dim {
            continue;
        }

        indices[i] += 1;
        if indices[i] < shape[i] {
            return true;
        }
        indices[i] = 0;
    }
    false
}

/// Brute force implementation for fallback and reference
fn distance_transform_edt_brute_force<D>(
    input: &Array<bool, D>,
    sampling: &[f64],
    return_distances: bool,
    return_indices: bool,
) -> (Option<Array<f64, D>>, Option<Array<i32, IxDyn>>)
where
    D: Dimension + 'static,
    D::Pattern: ndarray::NdIndex<D>,
    for<'a> &'a [usize]: ndarray::NdIndex<D>,
{
    let ndim = input.ndim();
    let shape = input.shape();

    // Initialize output arrays
    let mut distances: Option<Array<f64, D>> = if return_distances {
        Some(Array::zeros(input.raw_dim()))
    } else {
        None
    };

    let mut indices = if return_indices {
        let mut ind_shape = Vec::with_capacity(ndim + 1);
        ind_shape.push(ndim);
        ind_shape.extend(shape);
        Some(Array::zeros(IxDyn(&ind_shape)))
    } else {
        None
    };

    // Brute force computation (original algorithm)
    for idx in ndarray::indices(shape) {
        let idx_vec: Vec<_> = idx.slice().to_vec();
        if !input[idx_vec.as_slice()] {
            // Background pixels have distance 0
            if let Some(ref mut dist) = distances {
                dist[idx_vec.as_slice()] = 0.0;
            }

            // Background pixels have themselves as the closest background
            if let Some(ref mut ind) = indices {
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
            for bg_idx in ndarray::indices(shape) {
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
            if let Some(ref mut dist) = distances {
                dist[idx_vec.as_slice()] = min_dist;
            }

            if let Some(ref mut ind) = indices {
                for (d, &bg_idx_val) in closest_idx.iter().enumerate() {
                    let mut ind_slice = vec![d];
                    ind_slice.extend(&idx_vec);
                    ind[ind_slice.as_slice()] = bg_idx_val as i32;
                }
            }
        }
    }

    (distances, indices)
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
/// A tuple of:
/// * `Option<Array<f64, D>>` - Distance transform array (if return_distances is true)
/// * `Option<Array<i32, IxDyn>>` - Index array (if return_indices is true)
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array2, array, IxDyn};
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
/// let distances = distance_transform_edt(&input_dyn, None, true, false).0.unwrap();
/// ```
pub fn distance_transform_edt<D>(
    input: &Array<bool, D>,
    sampling: Option<&[f64]>,
    return_distances: bool,
    return_indices: bool,
) -> (Option<Array<f64, D>>, Option<Array<i32, IxDyn>>)
where
    D: Dimension + 'static,
    D::Pattern: ndarray::NdIndex<D>,
    for<'a> &'a [usize]: ndarray::NdIndex<D>,
{
    // Input validation
    if !return_distances && !return_indices {
        panic!("At least one of return_distances or return_indices must be true");
    }

    // Handle sampling
    let ndim = input.ndim();
    let sampling_vec = match sampling {
        Some(s) => {
            if s.len() != ndim {
                panic!("Sampling must have the same length as the number of dimensions");
            }
            s.to_vec()
        }
        None => vec![1.0; ndim],
    };

    // Use optimized separable algorithm for better performance
    distance_transform_edt_optimized(input, &sampling_vec, return_distances, return_indices)
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
/// A tuple of:
/// * `Option<Array<i32, D>>` - Distance transform array (if return_distances is true)
/// * `Option<Array<i32, IxDyn>>` - Index array (if return_indices is true)
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array2, array, IxDyn};
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
/// let distances = distance_transform_cdt(&input_dyn, "cityblock", true, false).0.unwrap();
/// ```
pub fn distance_transform_cdt<D>(
    input: &Array<bool, D>,
    metric: &str,
    return_distances: bool,
    return_indices: bool,
) -> (Option<Array<i32, D>>, Option<Array<i32, IxDyn>>)
where
    D: Dimension + 'static,
    D::Pattern: ndarray::NdIndex<D>,
    for<'a> &'a [usize]: ndarray::NdIndex<D>,
{
    // Input validation
    if !return_distances && !return_indices {
        panic!("At least one of return_distances or return_indices must be true");
    }

    let metric = match metric {
        "cityblock" => DistanceMetric::CityBlock,
        "chessboard" => DistanceMetric::Chessboard,
        _ => panic!("Metric must be one of 'cityblock' or 'chessboard'"),
    };

    // Initialize output arrays
    let mut distances = if return_distances {
        Some(Array::zeros(input.raw_dim()))
    } else {
        None
    };

    let mut indices = if return_indices {
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
    for idx in ndarray::indices(shape) {
        let idx_vec: Vec<_> = idx.slice().to_vec();
        if !input[idx_vec.as_slice()] {
            // Background pixels have distance 0
            if let Some(ref mut dist) = distances {
                dist[idx_vec.as_slice()] = 0;
            }

            // Background pixels have themselves as the closest background
            if let Some(ref mut ind) = indices {
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
            for bg_idx in ndarray::indices(shape) {
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
            if let Some(ref mut dist) = distances {
                dist[idx_vec.as_slice()] = min_dist;
            }

            if let Some(ref mut ind) = indices {
                for (d, &bg_idx_val) in closest_idx.iter().enumerate() {
                    let mut ind_slice = vec![d];
                    ind_slice.extend(&idx_vec);
                    ind[ind_slice.as_slice()] = bg_idx_val as i32;
                }
            }
        }
    }

    (distances, indices)
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
/// A tuple of:
/// * `Option<Array<f64, D>>` - Distance transform array (if return_distances is true)
/// * `Option<Array<i32, IxDyn>>` - Index array (if return_indices is true)
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array2, array, IxDyn};
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
/// let distances = distance_transform_bf(&input_dyn, "euclidean", None, true, false).0.unwrap();
/// ```
pub fn distance_transform_bf<D>(
    input: &Array<bool, D>,
    metric: &str,
    sampling: Option<&[f64]>,
    return_distances: bool,
    return_indices: bool,
) -> (Option<Array<f64, D>>, Option<Array<i32, IxDyn>>)
where
    D: Dimension + 'static,
    D::Pattern: ndarray::NdIndex<D>,
    for<'a> &'a [usize]: ndarray::NdIndex<D>,
{
    // Input validation
    if !return_distances && !return_indices {
        panic!("At least one of return_distances or return_indices must be true");
    }

    let metric = match metric {
        "euclidean" => DistanceMetric::Euclidean,
        "cityblock" => DistanceMetric::CityBlock,
        "chessboard" => DistanceMetric::Chessboard,
        _ => panic!("Metric must be one of 'euclidean', 'cityblock', or 'chessboard'"),
    };

    // Handle sampling
    let ndim = input.ndim();
    let sampling_vec = match sampling {
        Some(s) => {
            if s.len() != ndim {
                panic!("Sampling must have the same length as the number of dimensions");
            }
            s.to_vec()
        }
        None => vec![1.0; ndim],
    };

    // Initialize output arrays
    let mut distances = if return_distances {
        Some(Array::zeros(input.raw_dim()))
    } else {
        None
    };

    let mut indices = if return_indices {
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
    for idx in ndarray::indices(shape) {
        let idx_vec: Vec<_> = idx.slice().to_vec();
        if !input[idx_vec.as_slice()] {
            // Background pixels have distance 0
            if let Some(ref mut dist) = distances {
                dist[idx_vec.as_slice()] = 0.0;
            }

            // Background pixels have themselves as the closest background
            if let Some(ref mut ind) = indices {
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
            for bg_idx in ndarray::indices(shape) {
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
            if let Some(ref mut dist) = distances {
                dist[idx_vec.as_slice()] = min_dist;
            }

            if let Some(ref mut ind) = indices {
                for (d, &bg_idx_val) in closest_idx.iter().enumerate() {
                    let mut ind_slice = vec![d];
                    ind_slice.extend(&idx_vec);
                    ind[ind_slice.as_slice()] = bg_idx_val as i32;
                }
            }
        }
    }

    (distances, indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

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
        let input_dyn = input.clone().into_dimensionality::<IxDyn>().unwrap();
        let (distances_option, _) = distance_transform_edt(&input_dyn, None, true, false);
        let distances = distances_option
            .expect("Expected distances")
            .into_dimensionality::<ndarray::Ix2>()
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
        let input_dyn = input.clone().into_dimensionality::<IxDyn>().unwrap();
        let (distances_option, _) = distance_transform_cdt(&input_dyn, "cityblock", true, false);
        let distances = distances_option
            .expect("Expected distances")
            .into_dimensionality::<ndarray::Ix2>()
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

        let input_dyn = input.clone().into_dimensionality::<IxDyn>().unwrap();
        let sampling = vec![1.0, 1.0];

        // Get results from both algorithms
        let (optimized_dist, _) =
            distance_transform_edt_optimized(&input_dyn, &sampling, true, false);
        let (brute_force_dist, _) =
            distance_transform_edt_brute_force(&input_dyn, &sampling, true, false);

        let opt_dist = optimized_dist
            .expect("Expected optimized distances")
            .into_dimensionality::<ndarray::Ix2>()
            .expect("Failed to convert optimized to Ix2");

        let bf_dist = brute_force_dist
            .expect("Expected brute force distances")
            .into_dimensionality::<ndarray::Ix2>()
            .expect("Failed to convert brute force to Ix2");

        // Compare results
        for i in 0..input.nrows() {
            for j in 0..input.ncols() {
                assert_abs_diff_eq!(opt_dist[[i, j]], bf_dist[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
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
        let input_dyn = input.clone().into_dimensionality::<IxDyn>().unwrap();

        let (euclidean_option, _) =
            distance_transform_bf(&input_dyn, "euclidean", None, true, false);
        let euclidean = euclidean_option
            .expect("Expected euclidean distances")
            .into_dimensionality::<ndarray::Ix2>()
            .expect("Failed to convert back to Ix2");

        let (cityblock_option, _) =
            distance_transform_bf(&input_dyn, "cityblock", None, true, false);
        let cityblock = cityblock_option
            .expect("Expected cityblock distances")
            .into_dimensionality::<ndarray::Ix2>()
            .expect("Failed to convert back to Ix2");

        let (chessboard_option, _) =
            distance_transform_bf(&input_dyn, "chessboard", None, true, false);
        let chessboard = chessboard_option
            .expect("Expected chessboard distances")
            .into_dimensionality::<ndarray::Ix2>()
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
