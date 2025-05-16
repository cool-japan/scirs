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
/// ```rust,ignore
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

    // We'll implement a simple brute-force algorithm first
    // TODO: Implement a more efficient algorithm, like the one based on Voronoi diagrams

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
            // This is inefficient but simple for now
            for bg_idx in ndarray::indices(shape) {
                let bg_idx_vec: Vec<_> = bg_idx.slice().to_vec();
                if !input[bg_idx_vec.as_slice()] {
                    // Calculate the Euclidean distance
                    let mut dist_sq = 0.0;
                    for d in 0..ndim {
                        let diff = (idx_vec[d] as f64 - bg_idx_vec[d] as f64) * sampling_vec[d];
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
/// ```rust,ignore
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
/// ```rust,ignore
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
        assert_abs_diff_eq!(distances[[0, 2]], 1.4142, epsilon = 1e-4);
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
        assert_abs_diff_eq!(euclidean[[0, 2]], 1.4142, epsilon = 1e-4);
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
