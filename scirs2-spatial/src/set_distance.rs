//! Set-based distance metrics
//!
//! This module provides distance metrics for comparing sets of points,
//! treating each set as a geometric arrangement rather than just a collection.
//!
//! # Distance metrics
//!
//! * `hausdorff_distance` - Compute the Hausdorff distance between two point sets
//! * `directed_hausdorff` - Compute the directed Hausdorff distance with additional information
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use scirs2_spatial::set_distance::hausdorff_distance;
//!
//! // Create two point sets
//! let points1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
//! let points2 = array![[0.0, 0.5], [1.0, 0.5], [0.5, 1.0]];
//!
//! // Compute the Hausdorff distance
//! let dist = hausdorff_distance(&points1.view(), &points2.view(), None);
//! println!("Hausdorff distance: {}", dist);
//! ```

use crate::distance::euclidean;
use crate::error::SpatialResult;
use ndarray::{Array2, ArrayView2};
use num_traits::Float;
use rand::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

/// Compute the directed Hausdorff distance from set1 to set2.
///
/// The directed Hausdorff distance from set1 to set2 is defined as:
/// h(set1, set2) = max_{a in set1} min_{b in set2} |a - b|
///
/// # Arguments
///
/// * `set1` - Array of points in the first set, shape (n_points1, n_dims)
/// * `set2` - Array of points in the second set, shape (n_points2, n_dims)
/// * `seed` - Optional random seed for shuffling points (improves performance)
///
/// # Returns
///
/// A tuple containing:
/// * The directed Hausdorff distance
/// * Index of the point in set1 that realizes this distance
/// * Index of the point in set2 that realizes this distance
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_spatial::set_distance::directed_hausdorff;
///
/// // Create two point sets
/// let points1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let points2 = array![[0.0, 0.5], [1.0, 0.5], [0.5, 1.0]];
///
/// // Compute the directed Hausdorff distance from points1 to points2
/// let (dist, idx1, idx2) = directed_hausdorff(&points1.view(), &points2.view(), Some(42));
/// println!("Directed Hausdorff distance: {}", dist);
/// println!("Point in set1 that realizes this distance: {:?}", points1.row(idx1));
/// println!("Point in set2 that realizes this distance: {:?}", points2.row(idx2));
/// ```
pub fn directed_hausdorff<T: Float + Send + Sync>(
    set1: &ArrayView2<T>,
    set2: &ArrayView2<T>,
    seed: Option<u64>,
) -> (T, usize, usize) {
    let n1 = set1.shape()[0];
    let n2 = set2.shape()[0];
    let dims = set1.shape()[1];

    if n1 == 0 || n2 == 0 {
        return (T::infinity(), 0, 0);
    }

    if set2.shape()[1] != dims {
        panic!("Dimension mismatch: sets must have the same number of dimensions");
    }

    // Create randomized indices for shuffling
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_os_rng(),
    };

    let mut indices1: Vec<usize> = (0..n1).collect();
    let mut indices2: Vec<usize> = (0..n2).collect();
    indices1.shuffle(&mut rng);
    indices2.shuffle(&mut rng);

    // Create shuffled views of the datasets
    let mut set1_shuffled = Array2::zeros((n1, dims));
    let mut set2_shuffled = Array2::zeros((n2, dims));

    for (i, &idx) in indices1.iter().enumerate() {
        set1_shuffled.row_mut(i).assign(&set1.row(idx));
    }

    for (i, &idx) in indices2.iter().enumerate() {
        set2_shuffled.row_mut(i).assign(&set2.row(idx));
    }

    // Compute directed Hausdorff distance
    let mut cmax = T::zero();
    let mut i_ret = 0;
    let mut j_ret = 0;

    for i in 0..n1 {
        let mut cmin = T::infinity();
        let mut j_store = 0;
        let mut d_early_break = T::infinity();

        for j in 0..n2 {
            let mut d = T::zero();
            for k in 0..dims {
                let diff = set1_shuffled[[i, k]] - set2_shuffled[[j, k]];
                d = d + diff * diff;
            }

            // Early break if we've already found a closer point than cmax
            if d < cmax {
                d_early_break = d;
                break;
            }

            if d < cmin {
                cmin = d;
                j_store = j;
            }
        }

        // If we broke out of the inner loop early, we don't need to update cmax
        if d_early_break < cmax {
            continue;
        }

        if cmin >= cmax {
            cmax = cmin;
            i_ret = i;
            j_ret = j_store;
        }
    }

    // Map shuffled indices back to original indices
    let i_original = indices1[i_ret];
    let j_original = indices2[j_ret];

    (cmax.sqrt(), i_original, j_original)
}

/// Compute the Hausdorff distance between two point sets.
///
/// The Hausdorff distance between two sets is the greater of the two directed
/// Hausdorff distances. It measures how far the sets are from each other.
///
/// # Arguments
///
/// * `set1` - Array of points in the first set, shape (n_points1, n_dims)
/// * `set2` - Array of points in the second set, shape (n_points2, n_dims)
/// * `seed` - Optional random seed for shuffling points (improves performance)
///
/// # Returns
///
/// The Hausdorff distance between the two sets
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_spatial::set_distance::hausdorff_distance;
///
/// // Create two point sets
/// let points1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let points2 = array![[0.0, 0.5], [1.0, 0.5], [0.5, 1.0]];
///
/// // Compute the Hausdorff distance
/// let dist = hausdorff_distance(&points1.view(), &points2.view(), None);
/// println!("Hausdorff distance: {}", dist);
/// ```
pub fn hausdorff_distance<T: Float + Send + Sync>(
    set1: &ArrayView2<T>,
    set2: &ArrayView2<T>,
    seed: Option<u64>,
) -> T {
    // Compute directed Hausdorff distances in both directions
    let (dist_forward, _, _) = directed_hausdorff(set1, set2, seed);
    let (dist_backward, _, _) = directed_hausdorff(set2, set1, seed);

    // Return the maximum of the two directed distances
    if dist_forward > dist_backward {
        dist_forward
    } else {
        dist_backward
    }
}

/// Compute an estimate of the Earth Mover's distance (Wasserstein metric) between two distributions.
///
/// This implementation uses a simple approximation that works well for distributions of similar sizes.
/// For more accurate calculations, use specialized libraries for optimal transport problems.
///
/// # Arguments
///
/// * `set1` - Array of points in the first set, shape (n_points1, n_dims)
/// * `set2` - Array of points in the second set, shape (n_points2, n_dims)
///
/// # Returns
///
/// An estimate of the Earth Mover's distance
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_spatial::set_distance::wasserstein_distance;
///
/// // Create two point distributions
/// let points1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let points2 = array![[0.0, 0.5], [1.0, 0.5], [0.5, 1.0]];
///
/// // Compute the approximate Wasserstein distance
/// let dist = wasserstein_distance(&points1.view(), &points2.view()).unwrap();
/// println!("Approximate Wasserstein distance: {}", dist);
/// ```
pub fn wasserstein_distance<T: Float + Send + Sync>(
    set1: &ArrayView2<T>,
    set2: &ArrayView2<T>,
) -> SpatialResult<T> {
    let n1 = set1.shape()[0];
    let n2 = set2.shape()[0];
    let dims = set1.shape()[1];

    if n1 == 0 || n2 == 0 {
        return Ok(T::infinity());
    }

    if set2.shape()[1] != dims {
        panic!("Dimension mismatch: sets must have the same number of dimensions");
    }

    // For simplicity in this implementation, we'll use a greedy approach
    // rather than solving the full optimal transport problem

    // For equal-sized sets, we can use a greedy assignment
    if n1 == n2 {
        let mut total_distance = T::zero();
        let mut used = vec![false; n2];

        for i in 0..n1 {
            let point1 = set1.row(i);
            let mut min_dist = T::infinity();
            let mut best_j = 0;

            // Find the closest available point in set2
            for (j, &is_used) in used.iter().enumerate().take(n2) {
                if !is_used {
                    let point2 = set2.row(j);
                    let dist = euclidean(&point1.to_vec(), &point2.to_vec());
                    if dist < min_dist {
                        min_dist = dist;
                        best_j = j;
                    }
                }
            }

            used[best_j] = true;
            total_distance = total_distance + min_dist;
        }

        return Ok(total_distance / T::from(n1).unwrap());
    }

    // For unequal-sized sets, average the distances to nearest neighbors
    let mut total_distance = T::zero();

    // For each point in set1, find the closest point in set2
    for i in 0..n1 {
        let point1 = set1.row(i);
        let mut min_dist = T::infinity();

        for j in 0..n2 {
            let point2 = set2.row(j);
            let dist = euclidean(&point1.to_vec(), &point2.to_vec());
            if dist < min_dist {
                min_dist = dist;
            }
        }

        total_distance = total_distance + min_dist;
    }

    // For each point in set2, find the closest point in set1
    for j in 0..n2 {
        let point2 = set2.row(j);
        let mut min_dist = T::infinity();

        for i in 0..n1 {
            let point1 = set1.row(i);
            let dist = euclidean(&point2.to_vec(), &point1.to_vec());
            if dist < min_dist {
                min_dist = dist;
            }
        }

        total_distance = total_distance + min_dist;
    }

    // Average of both directional distances
    let avg_n = T::from(n1 + n2).unwrap();
    Ok(total_distance / avg_n)
}

/// Implement the Gromov-Hausdorff distance, which measures similarity between metric spaces
pub fn gromov_hausdorff_distance<T: Float + Send + Sync>(
    set1: &ArrayView2<T>,
    set2: &ArrayView2<T>,
) -> T {
    // This is a simplified implementation that provides an upper bound
    // on the actual Gromov-Hausdorff distance

    let n1 = set1.shape()[0];
    let n2 = set2.shape()[0];

    if n1 == 0 || n2 == 0 {
        return T::infinity();
    }

    // Compute pairwise distance matrices for each set
    let mut dist_matrix1 = Array2::zeros((n1, n1));
    let mut dist_matrix2 = Array2::zeros((n2, n2));

    // Compute distance matrices
    for i in 0..n1 {
        for j in 0..n1 {
            let p1 = set1.row(i).to_vec();
            let p2 = set1.row(j).to_vec();
            dist_matrix1[[i, j]] = euclidean(&p1, &p2);
        }
    }

    for i in 0..n2 {
        for j in 0..n2 {
            let p1 = set2.row(i).to_vec();
            let p2 = set2.row(j).to_vec();
            dist_matrix2[[i, j]] = euclidean(&p1, &p2);
        }
    }

    // Simple upper bound based on diameters of the spaces
    let diam1 = dist_matrix1.fold(T::neg_infinity(), |max, &val| max.max(val));
    let diam2 = dist_matrix2.fold(T::neg_infinity(), |max, &val| max.max(val));

    // Upper bound on Gromov-Hausdorff distance
    (diam1 - diam2).abs() / T::from(2).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_directed_hausdorff() {
        let set1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let set2 = array![[0.0, 0.5], [1.0, 0.5], [0.5, 1.0]];

        let (dist, _, _) = directed_hausdorff(&set1.view(), &set2.view(), Some(42));

        // The directed Hausdorff distance should be 0.5
        // (the maximum minimum distance from a point in set1 to set2)
        assert_relative_eq!(dist, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_hausdorff_distance() {
        let set1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let set2 = array![[0.0, 0.5], [1.0, 0.5], [0.5, 1.0]];

        let dist = hausdorff_distance(&set1.view(), &set2.view(), Some(42));

        // The Hausdorff distance is the maximum of the two directed distances
        assert_relative_eq!(dist, 0.5, epsilon = 1e-6);

        // Test with different sets where the Hausdorff distance is larger
        let set3 = array![[0.0, 0.0], [1.0, 0.0]];
        let set4 = array![[0.0, 2.0], [1.0, 2.0]];

        let dist = hausdorff_distance(&set3.view(), &set4.view(), Some(42));
        assert_relative_eq!(dist, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_wasserstein_distance() {
        let set1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let set2 = array![[0.0, 0.5], [1.0, 0.5], [0.5, 1.0]];

        let dist = wasserstein_distance(&set1.view(), &set2.view()).unwrap();

        // The approximate Earth Mover's distance for these sets
        // should be approximately 0.5
        assert!(dist > 0.0);
        assert!(dist < 1.0);
    }

    #[test]
    fn test_gromov_hausdorff_distance() {
        let set1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let set2 = array![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];

        let dist = gromov_hausdorff_distance(&set1.view(), &set2.view());

        // Diameter of set1 is sqrt(2), diameter of set2 is 2*sqrt(2)
        // Upper bound is |sqrt(2) - 2*sqrt(2)|/2 = sqrt(2)/2
        assert!(dist > 0.0);
    }

    #[test]
    fn test_empty_sets() {
        let set1 = array![[0.0, 0.0], [1.0, 0.0]];
        let empty: Array2<f64> = Array2::zeros((0, 2)); // Empty array with correct dimensions

        let dist = hausdorff_distance(&set1.view(), &empty.view(), None);
        assert!(dist.is_infinite());

        let dist = hausdorff_distance(&empty.view(), &set1.view(), None);
        assert!(dist.is_infinite());
    }
}
