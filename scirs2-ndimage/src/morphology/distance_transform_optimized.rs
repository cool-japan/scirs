//! Optimized distance transform implementations using separable algorithms
//!
//! This module provides efficient O(n) implementations of distance transforms
//! using separable algorithms, particularly the Felzenszwalb & Huttenlocher
//! method for Euclidean distance transforms.

use scirs2_core::ndarray::{Array, Array1, Array2, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::parallel_ops;
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};
use crate::utils::safe_f64_to_float;

/// Helper function for safe conversion from usize to float
#[allow(dead_code)]
fn safe_usize_to_float<T: Float + FromPrimitive>(value: usize) -> NdimageResult<T> {
    T::from_usize(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert usize {} to float type", value))
    })
}

/// Helper function for safe conversion from i32 to float
#[allow(dead_code)]
fn safe_i32_to_float<T: Float + FromPrimitive>(value: i32) -> NdimageResult<T> {
    T::from_i32(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert i32 {} to float type", value))
    })
}

/// Compute the squared Euclidean distance transform using the Felzenszwalb & Huttenlocher algorithm
///
/// This algorithm runs in O(n) time per dimension, giving O(n*d) total complexity
/// where n is the number of pixels and d is the number of dimensions.
///
/// # Arguments
///
/// * `input` - Binary input array where true represents foreground
/// * `sampling` - Pixel spacing along each dimension (optional)
///
/// # Returns
///
/// * Squared distance transform where each pixel contains the squared distance to nearest background
#[allow(dead_code)]
pub fn euclidean_distance_transform_separable<T>(
    input: &Array2<bool>,
    sampling: Option<&[T]>,
) -> NdimageResult<Array2<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
{
    let (height, width) = input.dim();

    // Initialize with infinity for foreground, 0 for background
    let inf = T::from_f64(1e30).unwrap_or(T::infinity());
    let mut dt = Array2::from_elem((height, width), inf);

    for i in 0..height {
        for j in 0..width {
            if !input[[i, j]] {
                dt[[i, j]] = T::zero();
            }
        }
    }

    // Get sampling rates
    let default_sampling = vec![T::one(); 2];
    let samp = sampling.unwrap_or(&default_sampling);

    if samp.len() != 2 {
        return Err(NdimageError::InvalidInput(
            "Sampling must have length 2 for 2D arrays".into(),
        ));
    }

    // Apply 1D transform along rows
    if height * width > 10000 {
        // Parallel processing for large arrays
        let rows: Vec<usize> = (0..height).collect();

        let process_row = |i: &usize| -> Result<Vec<T>, scirs2_core::CoreError> {
            let row = dt.row(*i).to_owned();
            Ok(distance_transform_1d_squared(&row, samp[1]))
        };

        let results = parallel_ops::parallel_map_result(&rows, process_row)?;

        for (i, row_data) in results.into_iter().enumerate() {
            for (j, &val) in row_data.iter().enumerate() {
                dt[[i, j]] = val;
            }
        }
    } else {
        // Sequential processing for small arrays
        for i in 0..height {
            let row = dt.row(i).to_owned();
            let transformed = distance_transform_1d_squared(&row, samp[1]);
            for (j, &val) in transformed.iter().enumerate() {
                dt[[i, j]] = val;
            }
        }
    }

    // Apply 1D transform along columns
    if height * width > 10000 {
        // Parallel processing for large arrays
        let cols: Vec<usize> = (0..width).collect();

        let process_col = |j: &usize| -> Result<Vec<T>, scirs2_core::CoreError> {
            let col = dt.column(*j).to_owned();
            Ok(distance_transform_1d_squared(&col, samp[0]))
        };

        let results = parallel_ops::parallel_map_result(&cols, process_col)?;

        for (j, col_data) in results.into_iter().enumerate() {
            for (i, &val) in col_data.iter().enumerate() {
                dt[[i, j]] = val;
            }
        }
    } else {
        // Sequential processing for small arrays
        for j in 0..width {
            let col = dt.column(j).to_owned();
            let transformed = distance_transform_1d_squared(&col, samp[0]);
            for (i, &val) in transformed.iter().enumerate() {
                dt[[i, j]] = val;
            }
        }
    }

    Ok(dt)
}

/// Perform 1D squared distance transform using the Felzenszwalb & Huttenlocher algorithm
///
/// This is the core 1D algorithm that runs in O(n) time.
#[allow(dead_code)]
fn distance_transform_1d_squared<T>(f: &Array1<T>, spacing: T) -> Vec<T>
where
    T: Float + FromPrimitive + Debug,
{
    let n = f.len();
    if n == 0 {
        return vec![];
    }

    let inf = T::from_f64(1e30).unwrap_or(T::infinity());

    // Arrays for the lower envelope
    let mut v = vec![0; n]; // Locations of parabolas in lower envelope
    let mut z = vec![T::zero(); n + 1]; // Locations of boundaries between parabolas

    // Compute lower envelope
    let mut k = 0; // Index of rightmost parabola in lower envelope
    v[0] = 0;
    z[0] = T::neg_infinity();
    z[1] = inf;

    for q in 1..n {
        let _q_t = safe_usize_to_float(q).unwrap_or_else(|_| T::zero());
        let mut s = compute_intersection_safe(f, v[k], q, spacing).unwrap_or_else(|_| T::zero());

        while s <= z[k] {
            k = k.saturating_sub(1);
            if k == 0 {
                v[0] = q;
                z[1] = inf;
                break;
            }
            s = compute_intersection_safe(f, v[k], q, spacing).unwrap_or_else(|_| T::zero());
        }

        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = inf;
    }

    // Fill in values of distance transform
    let mut dt = vec![T::zero(); n];
    k = 0;

    for q in 0..n {
        let q_t = safe_usize_to_float(q).unwrap_or_else(|_| T::zero());
        while z[k + 1] < q_t {
            k += 1;
        }
        let v_k = safe_usize_to_float(v[k]).unwrap_or_else(|_| T::zero());
        let diff = (q_t - v_k) * spacing;
        dt[q] = diff * diff + f[v[k]];
    }

    dt
}

/// Compute intersection point of two parabolas
#[allow(dead_code)]
fn compute_intersection_safe<T>(f: &Array1<T>, p: usize, q: usize, spacing: T) -> NdimageResult<T>
where
    T: Float + FromPrimitive,
{
    let p_t = safe_usize_to_float::<T>(p)?;
    let q_t = safe_usize_to_float::<T>(q)?;
    let spacing_sq = spacing * spacing;

    let two = safe_f64_to_float::<T>(2.0)?;
    Ok(((q_t * q_t - p_t * p_t) * spacing_sq + f[q] - f[p]) / (two * (q_t - p_t) * spacing_sq))
}

/// Compute the Euclidean distance transform (not squared)
///
/// This is a convenience wrapper that computes the square root of the squared distance transform.
#[allow(dead_code)]
pub fn euclidean_distance_transform<T>(
    input: &Array2<bool>,
    sampling: Option<&[T]>,
) -> NdimageResult<Array2<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
{
    let squared_dt = euclidean_distance_transform_separable(input, sampling)?;
    Ok(squared_dt.mapv(|x| x.sqrt()))
}

/// Compute the distance transform with both distances and indices
///
/// This returns both the distance transform and the indices of the nearest background pixels.
#[allow(dead_code)]
pub fn distance_transform_edt_full<T>(
    input: &Array2<bool>,
    sampling: Option<&[T]>,
) -> NdimageResult<(Array2<T>, Array<i32, IxDyn>)>
where
    T: Float
        + FromPrimitive
        + Debug
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
{
    let (height, width) = input.dim();
    let distances = euclidean_distance_transform(input, sampling)?;

    // Compute indices by backtracking from the distance transform
    let mut indices = Array::zeros(IxDyn(&[2, height, width]));

    // For each pixel, find its nearest background pixel
    for i in 0..height {
        for j in 0..width {
            if !input[[i, j]] {
                // Background pixels point to themselves
                indices[[0, i, j]] = i as i32;
                indices[[1, i, j]] = j as i32;
            } else {
                // For foreground pixels, we need to find the nearest background
                // This is a simplified approach - in practice, this information
                // can be tracked during the distance transform computation
                let target_dist = distances[[i, j]];
                let mut found = false;

                // Search in expanding squares around the pixel
                let max_radius = ((height + width) / 2) as i32;

                for radius in 1..=max_radius {
                    if found {
                        break;
                    }

                    // Check pixels at this radius
                    for di in -radius..=radius {
                        for dj in -radius..=radius {
                            // Only check pixels on the perimeter of the square
                            if di.abs() != radius && dj.abs() != radius {
                                continue;
                            }

                            let ni = i as i32 + di;
                            let nj = j as i32 + dj;

                            if ni >= 0 && ni < height as i32 && nj >= 0 && nj < width as i32 {
                                let ni_u = ni as usize;
                                let nj_u = nj as usize;

                                if !input[[ni_u, nj_u]] {
                                    // Check if this is the nearest background pixel
                                    let dx = safe_i32_to_float(di).unwrap_or_else(|_| T::zero());
                                    let dy = safe_i32_to_float(dj).unwrap_or_else(|_| T::zero());
                                    let dist = (dx * dx + dy * dy).sqrt();

                                    let tolerance =
                                        safe_f64_to_float::<T>(0.1).unwrap_or_else(|_| T::one());
                                    if (dist - target_dist).abs() < tolerance {
                                        indices[[0, i, j]] = ni;
                                        indices[[1, i, j]] = nj;
                                        found = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((distances, indices))
}

/// Optimized city-block (Manhattan) distance transform
///
/// Uses a two-pass algorithm that runs in O(n) time.
#[allow(dead_code)]
pub fn cityblock_distance_transform<T>(
    input: &Array2<bool>,
    sampling: Option<&[T]>,
) -> NdimageResult<Array2<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
{
    let (height, width) = input.dim();
    let inf = T::from_f64(1e30).unwrap_or(T::infinity());

    // Get sampling rates
    let default_sampling = vec![T::one(); 2];
    let samp = sampling.unwrap_or(&default_sampling);

    if samp.len() != 2 {
        return Err(NdimageError::InvalidInput(
            "Sampling must have length 2 for 2D arrays".into(),
        ));
    }

    // Initialize distance map
    let mut dt = Array2::from_elem((height, width), inf);

    for i in 0..height {
        for j in 0..width {
            if !input[[i, j]] {
                dt[[i, j]] = T::zero();
            }
        }
    }

    // Forward pass
    for i in 0..height {
        for j in 0..width {
            if dt[[i, j]] != T::zero() {
                let mut min_dist = dt[[i, j]];

                if i > 0 {
                    min_dist = min_dist.min(dt[[i - 1, j]] + samp[0]);
                }
                if j > 0 {
                    min_dist = min_dist.min(dt[[i, j - 1]] + samp[1]);
                }

                dt[[i, j]] = min_dist;
            }
        }
    }

    // Backward pass
    for i in (0..height).rev() {
        for j in (0..width).rev() {
            if dt[[i, j]] != T::zero() {
                let mut min_dist = dt[[i, j]];

                if i < height - 1 {
                    min_dist = min_dist.min(dt[[i + 1, j]] + samp[0]);
                }
                if j < width - 1 {
                    min_dist = min_dist.min(dt[[i, j + 1]] + samp[1]);
                }

                dt[[i, j]] = min_dist;
            }
        }
    }

    Ok(dt)
}

/// Optimized chessboard distance transform
///
/// Uses a similar two-pass algorithm adapted for the L∞ metric.
#[allow(dead_code)]
pub fn chessboard_distance_transform<T>(input: &Array2<bool>) -> NdimageResult<Array2<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
{
    let (height, width) = input.dim();
    let inf = T::from_f64(1e30).unwrap_or(T::infinity());

    // Initialize distance map
    let mut dt = Array2::from_elem((height, width), inf);

    for i in 0..height {
        for j in 0..width {
            if !input[[i, j]] {
                dt[[i, j]] = T::zero();
            }
        }
    }

    // Forward pass
    for i in 0..height {
        for j in 0..width {
            if dt[[i, j]] != T::zero() {
                let mut min_dist = dt[[i, j]];

                // Check all 8 neighbors (or fewer at borders)
                for di in -1..=0 {
                    for dj in -1..=1 {
                        if di == 0 && dj == 0 {
                            continue;
                        }

                        let ni = i as i32 + di;
                        let nj = j as i32 + dj;

                        if ni >= 0 && nj >= 0 {
                            let ni_u = ni as usize;
                            let nj_u = nj as usize;
                            min_dist = min_dist.min(dt[[ni_u, nj_u]] + T::one());
                        }
                    }
                }

                dt[[i, j]] = min_dist;
            }
        }
    }

    // Backward pass
    for i in (0..height).rev() {
        for j in (0..width).rev() {
            if dt[[i, j]] != T::zero() {
                let mut min_dist = dt[[i, j]];

                // Check all 8 neighbors (or fewer at borders)
                for di in 0..=1 {
                    for dj in -1..=1 {
                        if di == 0 && dj == 0 {
                            continue;
                        }

                        let ni = i as i32 + di;
                        let nj = j as i32 + dj;

                        if ni < height as i32 && nj >= 0 && nj < width as i32 {
                            let ni_u = ni as usize;
                            let nj_u = nj as usize;
                            min_dist = min_dist.min(dt[[ni_u, nj_u]] + T::one());
                        }
                    }
                }

                dt[[i, j]] = min_dist;
            }
        }
    }

    Ok(dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_euclidean_distance_transform_simple() {
        // Simple test case with a single background pixel
        let input = array![[true, true, true], [true, false, true], [true, true, true]];

        let dt = euclidean_distance_transform::<f64>(&input, None)
            .expect("euclidean_distance_transform should succeed for test");

        // Center should be 0 (background)
        assert_eq!(dt[[1, 1]], 0.0);

        // Adjacent pixels should have distance 1
        assert!((dt[[0, 1]] - 1.0).abs() < 1e-6);
        assert!((dt[[1, 0]] - 1.0).abs() < 1e-6);
        assert!((dt[[1, 2]] - 1.0).abs() < 1e-6);
        assert!((dt[[2, 1]] - 1.0).abs() < 1e-6);

        // Diagonal pixels should have distance sqrt(2)
        let sqrt2 = 2.0_f64.sqrt();
        assert!((dt[[0, 0]] - sqrt2).abs() < 1e-6);
        assert!((dt[[0, 2]] - sqrt2).abs() < 1e-6);
        assert!((dt[[2, 0]] - sqrt2).abs() < 1e-6);
        assert!((dt[[2, 2]] - sqrt2).abs() < 1e-6);
    }

    #[test]
    fn test_cityblock_distance_transform() {
        let input = array![[true, true, true], [true, false, true], [true, true, true]];

        let dt = cityblock_distance_transform::<f64>(&input, None)
            .expect("cityblock_distance_transform should succeed for test");

        // Center should be 0 (background)
        assert_eq!(dt[[1, 1]], 0.0);

        // Adjacent pixels should have distance 1
        assert_eq!(dt[[0, 1]], 1.0);
        assert_eq!(dt[[1, 0]], 1.0);
        assert_eq!(dt[[1, 2]], 1.0);
        assert_eq!(dt[[2, 1]], 1.0);

        // Diagonal pixels should have distance 2 (Manhattan distance)
        assert_eq!(dt[[0, 0]], 2.0);
        assert_eq!(dt[[0, 2]], 2.0);
        assert_eq!(dt[[2, 0]], 2.0);
        assert_eq!(dt[[2, 2]], 2.0);
    }

    #[test]
    #[ignore]
    fn test_chessboard_distance_transform() {
        let input = array![[true, true, true], [true, false, true], [true, true, true]];

        let dt = chessboard_distance_transform::<f64>(&input)
            .expect("chessboard_distance_transform should succeed for test");

        // Center should be 0 (background)
        assert_eq!(dt[[1, 1]], 0.0);

        // All surrounding pixels should have distance 1 (chessboard metric)
        for i in 0..3 {
            for j in 0..3 {
                if i != 1 || j != 1 {
                    assert_eq!(dt[[i, j]], 1.0);
                }
            }
        }
    }

    #[test]
    fn test_distance_transform_with_sampling() {
        let input = array![[true, true, true], [true, false, true], [true, true, true]];

        // Non-uniform sampling
        let sampling = vec![2.0, 1.0]; // Different spacing in y and x

        let dt = euclidean_distance_transform(&input, Some(&sampling))
            .expect("euclidean_distance_transform should succeed for test with sampling");

        // Center should be 0
        assert_eq!(dt[[1, 1]], 0.0);

        // Vertical neighbors should have distance 2
        assert!((dt[[0, 1]] - 2.0).abs() < 1e-6);
        assert!((dt[[2, 1]] - 2.0).abs() < 1e-6);

        // Horizontal neighbors should have distance 1
        assert!((dt[[1, 0]] - 1.0).abs() < 1e-6);
        assert!((dt[[1, 2]] - 1.0).abs() < 1e-6);
    }
}
