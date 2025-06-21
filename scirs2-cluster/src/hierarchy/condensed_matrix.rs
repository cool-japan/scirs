//! Condensed distance matrix utilities
//!
//! This module provides utilities for working with condensed distance matrices,
//! which store only the upper triangular portion of a symmetric distance matrix
//! in a flattened 1D array format for memory efficiency.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive, Zero};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Convert a square distance matrix to condensed form
///
/// Takes a symmetric distance matrix and returns only the upper triangular
/// portion in a flattened 1D array. The diagonal is assumed to be zero and
/// is not stored.
///
/// # Arguments
///
/// * `square_matrix` - Square symmetric distance matrix (n × n)
///
/// # Returns
///
/// * `Result<Array1<F>>` - Condensed distance matrix with n*(n-1)/2 elements
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_cluster::hierarchy::condensed_matrix::square_to_condensed;
///
/// let square = Array2::from_shape_vec((3, 3), vec![
///     0.0, 1.0, 2.0,
///     1.0, 0.0, 3.0,
///     2.0, 3.0, 0.0,
/// ]).unwrap();
///
/// let condensed = square_to_condensed(square.view()).unwrap();
/// assert_eq!(condensed.to_vec(), vec![1.0, 2.0, 3.0]);
/// ```
pub fn square_to_condensed<F: Float + Zero + Copy>(
    square_matrix: ArrayView2<F>,
) -> Result<Array1<F>> {
    let n = square_matrix.shape()[0];
    let m = square_matrix.shape()[1];

    if n != m {
        return Err(ClusteringError::InvalidInput(format!(
            "Distance matrix must be square, got {}x{}",
            n, m
        )));
    }

    if n < 2 {
        return Err(ClusteringError::InvalidInput(
            "Distance matrix must be at least 2x2".to_string(),
        ));
    }

    let condensed_size = n * (n - 1) / 2;
    let mut condensed = Array1::zeros(condensed_size);

    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            condensed[idx] = square_matrix[[i, j]];
            idx += 1;
        }
    }

    Ok(condensed)
}

/// Convert a condensed distance matrix to square form
///
/// Takes a condensed distance matrix and reconstructs the full symmetric
/// square matrix with zeros on the diagonal.
///
/// # Arguments
///
/// * `condensed_matrix` - Condensed distance matrix (length n*(n-1)/2)
///
/// # Returns
///
/// * `Result<Array2<F>>` - Square symmetric distance matrix (n × n)
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_cluster::hierarchy::condensed_matrix::condensed_to_square;
///
/// let condensed = Array1::from_vec(vec![1.0, 2.0, 3.0]);
/// let square = condensed_to_square(condensed.view()).unwrap();
///
/// assert_eq!(square[[0, 1]], 1.0);
/// assert_eq!(square[[1, 0]], 1.0);
/// assert_eq!(square[[0, 2]], 2.0);
/// assert_eq!(square[[2, 0]], 2.0);
/// assert_eq!(square[[1, 2]], 3.0);
/// assert_eq!(square[[2, 1]], 3.0);
/// ```
pub fn condensed_to_square<F: Float + Zero + Copy>(
    condensed_matrix: ArrayView1<F>,
) -> Result<Array2<F>> {
    let condensed_len = condensed_matrix.len();

    // Solve n*(n-1)/2 = condensed_len for n
    let n_float = (1.0 + (1.0 + 8.0 * condensed_len as f64).sqrt()) / 2.0;
    let n = n_float as usize;

    if n * (n - 1) / 2 != condensed_len {
        return Err(ClusteringError::InvalidInput(format!(
            "Invalid condensed matrix size: {} elements doesn't correspond to n*(n-1)/2 for any integer n",
            condensed_len
        )));
    }

    let mut square = Array2::zeros((n, n));

    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let value = condensed_matrix[idx];
            square[[i, j]] = value;
            square[[j, i]] = value; // Symmetric
            idx += 1;
        }
    }

    Ok(square)
}

/// Get the distance between two points from a condensed distance matrix
///
/// Retrieves the distance between points i and j from a condensed distance matrix
/// without converting to square form.
///
/// # Arguments
///
/// * `condensed_matrix` - Condensed distance matrix
/// * `i` - First point index
/// * `j` - Second point index
/// * `n` - Total number of points
///
/// # Returns
///
/// * `Result<F>` - Distance between points i and j
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_cluster::hierarchy::condensed_matrix::get_distance;
///
/// let condensed = Array1::from_vec(vec![1.0, 2.0, 3.0]);
/// let distance = get_distance(condensed.view(), 0, 2, 3).unwrap();
/// assert_eq!(distance, 2.0);
/// ```
pub fn get_distance<F: Float + Zero + Copy>(
    condensed_matrix: ArrayView1<F>,
    i: usize,
    j: usize,
    n: usize,
) -> Result<F> {
    if i == j {
        return Ok(F::zero());
    }

    if i >= n || j >= n {
        return Err(ClusteringError::InvalidInput(format!(
            "Point indices {} and {} must be less than n={}",
            i, j, n
        )));
    }

    let expected_len = n * (n - 1) / 2;
    if condensed_matrix.len() != expected_len {
        return Err(ClusteringError::InvalidInput(format!(
            "Condensed matrix length {} doesn't match expected {} for n={}",
            condensed_matrix.len(),
            expected_len,
            n
        )));
    }

    // Ensure i < j for indexing
    let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };

    // Calculate the index in the condensed matrix
    let condensed_idx = n * min_idx - (min_idx * (min_idx + 1)) / 2 + (max_idx - min_idx - 1);

    if condensed_idx >= condensed_matrix.len() {
        return Err(ClusteringError::InvalidInput(format!(
            "Computed index {} is out of bounds for condensed matrix of length {}",
            condensed_idx,
            condensed_matrix.len()
        )));
    }

    Ok(condensed_matrix[condensed_idx])
}

/// Set the distance between two points in a condensed distance matrix
///
/// Updates the distance between points i and j in a condensed distance matrix
/// without converting to square form.
///
/// # Arguments
///
/// * `condensed_matrix` - Mutable condensed distance matrix
/// * `i` - First point index  
/// * `j` - Second point index
/// * `n` - Total number of points
/// * `distance` - New distance value
///
/// # Returns
///
/// * `Result<()>` - Ok if successful, error otherwise
pub fn set_distance<F: Float + Zero + Copy>(
    condensed_matrix: ArrayView1<F>,
    i: usize,
    j: usize,
    n: usize,
    distance: F,
) -> Result<()> {
    if i == j {
        if !distance.is_zero() {
            return Err(ClusteringError::InvalidInput(
                "Cannot set non-zero distance for identical points".to_string(),
            ));
        }
        return Ok(()); // Distance between identical points is always zero
    }

    if i >= n || j >= n {
        return Err(ClusteringError::InvalidInput(format!(
            "Point indices {} and {} must be less than n={}",
            i, j, n
        )));
    }

    let expected_len = n * (n - 1) / 2;
    if condensed_matrix.len() != expected_len {
        return Err(ClusteringError::InvalidInput(format!(
            "Condensed matrix length {} doesn't match expected {} for n={}",
            condensed_matrix.len(),
            expected_len,
            n
        )));
    }

    // Ensure i < j for indexing
    let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };

    // Calculate the index in the condensed matrix
    let condensed_idx = n * min_idx - (min_idx * (min_idx + 1)) / 2 + (max_idx - min_idx - 1);

    if condensed_idx >= condensed_matrix.len() {
        return Err(ClusteringError::InvalidInput(format!(
            "Computed index {} is out of bounds for condensed matrix of length {}",
            condensed_idx,
            condensed_matrix.len()
        )));
    }

    // Note: This is a limitation - we can't actually modify through ArrayView1
    // In practice, this would need to work with Array1 or a mutable slice
    Err(ClusteringError::ComputationError(
        "Cannot modify through immutable view - use mutable Array1 instead".to_string(),
    ))
}

/// Calculate the size of a condensed matrix for n points
///
/// Returns the number of elements needed to store a condensed distance matrix
/// for n points.
///
/// # Arguments
///
/// * `n` - Number of points
///
/// # Returns
///
/// * `usize` - Size of condensed matrix (n*(n-1)/2)
pub fn condensed_size(n: usize) -> usize {
    n * (n - 1) / 2
}

/// Calculate the number of points from condensed matrix size
///
/// Given the size of a condensed distance matrix, calculate the number
/// of original points.
///
/// # Arguments
///
/// * `condensed_len` - Length of condensed matrix
///
/// # Returns
///
/// * `Result<usize>` - Number of points, or error if size is invalid
pub fn points_from_condensed_size(condensed_len: usize) -> Result<usize> {
    let n_float = (1.0 + (1.0 + 8.0 * condensed_len as f64).sqrt()) / 2.0;
    let n = n_float as usize;

    if n * (n - 1) / 2 != condensed_len {
        return Err(ClusteringError::InvalidInput(format!(
            "Invalid condensed matrix size: {} elements doesn't correspond to n*(n-1)/2 for any integer n",
            condensed_len
        )));
    }

    Ok(n)
}

/// Validate a condensed distance matrix
///
/// Checks that a condensed distance matrix has the correct size and
/// contains valid distance values.
///
/// # Arguments
///
/// * `condensed_matrix` - Condensed distance matrix to validate
///
/// # Returns
///
/// * `Result<usize>` - Number of points if valid, error otherwise
pub fn validate_condensed_matrix<F: Float + FromPrimitive + Debug + PartialOrd>(
    condensed_matrix: ArrayView1<F>,
) -> Result<usize> {
    let condensed_len = condensed_matrix.len();

    if condensed_len == 0 {
        return Err(ClusteringError::InvalidInput(
            "Condensed matrix cannot be empty".to_string(),
        ));
    }

    // Check if size corresponds to valid n
    let n = points_from_condensed_size(condensed_len)?;

    if n < 2 {
        return Err(ClusteringError::InvalidInput(
            "Condensed matrix must represent at least 2 points".to_string(),
        ));
    }

    // Check all distances are non-negative and finite
    for (idx, &distance) in condensed_matrix.iter().enumerate() {
        if !distance.is_finite() {
            return Err(ClusteringError::InvalidInput(format!(
                "Non-finite distance at index {}",
                idx
            )));
        }

        if distance < F::zero() {
            return Err(ClusteringError::InvalidInput(format!(
                "Negative distance at index {}: {:?}",
                idx, distance
            )));
        }
    }

    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_square_to_condensed() {
        let square = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 4.0, 5.0, 2.0, 4.0, 0.0, 6.0, 3.0, 5.0, 6.0, 0.0,
            ],
        )
        .unwrap();

        let condensed = square_to_condensed(square.view()).unwrap();
        assert_eq!(condensed.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_condensed_to_square() {
        let condensed = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let square = condensed_to_square(condensed.view()).unwrap();

        let expected = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 4.0, 5.0, 2.0, 4.0, 0.0, 6.0, 3.0, 5.0, 6.0, 0.0,
            ],
        )
        .unwrap();

        assert_eq!(square, expected);
    }

    #[test]
    fn test_round_trip_conversion() {
        let original =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.5, 2.5, 1.5, 0.0, 3.5, 2.5, 3.5, 0.0])
                .unwrap();

        let condensed = square_to_condensed(original.view()).unwrap();
        let reconstructed = condensed_to_square(condensed.view()).unwrap();

        assert_eq!(original, reconstructed);
    }

    #[test]
    fn test_get_distance() {
        let condensed = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // Test all pairwise distances for 3 points
        assert_eq!(get_distance(condensed.view(), 0, 1, 3).unwrap(), 1.0);
        assert_eq!(get_distance(condensed.view(), 1, 0, 3).unwrap(), 1.0); // Symmetric
        assert_eq!(get_distance(condensed.view(), 0, 2, 3).unwrap(), 2.0);
        assert_eq!(get_distance(condensed.view(), 2, 0, 3).unwrap(), 2.0); // Symmetric
        assert_eq!(get_distance(condensed.view(), 1, 2, 3).unwrap(), 3.0);
        assert_eq!(get_distance(condensed.view(), 2, 1, 3).unwrap(), 3.0); // Symmetric

        // Test diagonal (should be zero)
        assert_eq!(get_distance(condensed.view(), 0, 0, 3).unwrap(), 0.0);
        assert_eq!(get_distance(condensed.view(), 1, 1, 3).unwrap(), 0.0);
        assert_eq!(get_distance(condensed.view(), 2, 2, 3).unwrap(), 0.0);
    }

    #[test]
    fn test_condensed_size_calculations() {
        assert_eq!(condensed_size(2), 1);
        assert_eq!(condensed_size(3), 3);
        assert_eq!(condensed_size(4), 6);
        assert_eq!(condensed_size(5), 10);

        assert_eq!(points_from_condensed_size(1).unwrap(), 2);
        assert_eq!(points_from_condensed_size(3).unwrap(), 3);
        assert_eq!(points_from_condensed_size(6).unwrap(), 4);
        assert_eq!(points_from_condensed_size(10).unwrap(), 5);

        // Invalid sizes should fail
        assert!(points_from_condensed_size(2).is_err()); // No integer n such that n*(n-1)/2 = 2
        assert!(points_from_condensed_size(5).is_err()); // No integer n such that n*(n-1)/2 = 5
    }

    #[test]
    fn test_validate_condensed_matrix() {
        // Valid matrix
        let valid = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(validate_condensed_matrix(valid.view()).unwrap(), 3);

        // Invalid size
        let invalid_size = Array1::from_vec(vec![1.0, 2.0]);
        assert!(validate_condensed_matrix(invalid_size.view()).is_err());

        // Negative distance
        let negative = Array1::from_vec(vec![-1.0, 2.0, 3.0]);
        assert!(validate_condensed_matrix(negative.view()).is_err());

        // Non-finite distance
        let non_finite = Array1::from_vec(vec![f64::NAN, 2.0, 3.0]);
        assert!(validate_condensed_matrix(non_finite.view()).is_err());
    }

    #[test]
    fn test_error_cases() {
        // Non-square matrix
        let non_square =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(square_to_condensed(non_square.view()).is_err());

        // Too small matrix
        let too_small = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        assert!(square_to_condensed(too_small.view()).is_err());

        // Out of bounds point indices
        let condensed = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(get_distance(condensed.view(), 0, 3, 3).is_err()); // j=3 >= n=3
        assert!(get_distance(condensed.view(), 4, 1, 3).is_err()); // i=4 >= n=3
    }
}
