//! Covariance and correlation related functions
//!
//! This module provides functions for computing covariance and correlation matrices.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, Zero};

use crate::error::{LinalgError, LinalgResult};

/// Compute the covariance matrix from a data matrix.
///
/// The input data matrix has samples as rows and features as columns.
/// The covariance matrix will have features as both rows and columns.
///
/// # Arguments
///
/// * `data` - Data matrix with samples as rows and features as columns
/// * `ddof` - Delta degrees of freedom (defaults to 1 for unbiased estimate)
///
/// # Returns
///
/// * Covariance matrix with shape (n_features, n_features)
pub fn covariance_matrix<F>(data: &ArrayView2<F>, ddof: Option<usize>) -> LinalgResult<Array2<F>>
where
    F: Float + Zero + num_traits::FromPrimitive,
{
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if n_samples <= 1 {
        return Err(LinalgError::InvalidInputError(
            "Need at least 2 samples to compute covariance".to_string(),
        ));
    }

    // Delta degrees of freedom (default: 1 for unbiased estimator)
    let ddof = ddof.unwrap_or(1);
    if ddof >= n_samples {
        return Err(LinalgError::InvalidInputError(format!(
            "Delta degrees of freedom ({}) must be less than sample count ({})",
            ddof, n_samples
        )));
    }

    // Compute mean for each feature
    let mean = data.mean_axis(Axis(0)).unwrap();

    // Center the data
    let centered = data.to_owned() - &mean;

    // Compute covariance matrix: X^T * X / (n - ddof)
    let mut cov = Array2::zeros((n_features, n_features));
    let normalizer = F::from(n_samples - ddof).unwrap();

    for i in 0..n_features {
        for j in 0..=i {
            // Only compute lower triangle due to symmetry
            let mut sum = F::zero();
            for k in 0..n_samples {
                sum = sum + centered[[k, i]] * centered[[k, j]];
            }
            let val = sum / normalizer;
            cov[[i, j]] = val;
            cov[[j, i]] = val; // Fill upper triangle (symmetric matrix)
        }
    }

    Ok(cov)
}

/// Compute the correlation matrix from a data matrix.
///
/// The correlation matrix is the normalized covariance matrix,
/// with values in the range [-1, 1].
///
/// # Arguments
///
/// * `data` - Data matrix with samples as rows and features as columns
/// * `ddof` - Delta degrees of freedom (defaults to 1 for unbiased estimate)
///
/// # Returns
///
/// * Correlation matrix with shape (n_features, n_features)
pub fn correlation_matrix<F>(data: &ArrayView2<F>, ddof: Option<usize>) -> LinalgResult<Array2<F>>
where
    F: Float + Zero + num_traits::FromPrimitive,
{
    // Compute covariance matrix
    let cov = covariance_matrix(data, ddof)?;
    let n_features = cov.nrows();

    // Extract standard deviations (sqrt of diagonal elements)
    let mut std_devs = Array1::zeros(n_features);
    for i in 0..n_features {
        std_devs[i] = cov[[i, i]].sqrt();
    }

    // Normalize covariance to get correlation
    let mut corr = Array2::zeros((n_features, n_features));

    for i in 0..n_features {
        for j in 0..n_features {
            if std_devs[i] > F::epsilon() && std_devs[j] > F::epsilon() {
                corr[[i, j]] = cov[[i, j]] / (std_devs[i] * std_devs[j]);
            } else {
                // If either standard deviation is zero, correlation is undefined
                // We'll set it to zero by convention
                corr[[i, j]] = F::zero();
            }
        }
    }

    // Ensure diagonal is exactly 1.0
    for i in 0..n_features {
        corr[[i, i]] = F::one();
    }

    Ok(corr)
}

/// Compute the Mahalanobis distance between a point and a distribution.
///
/// The Mahalanobis distance measures how many standard deviations away
/// a point is from the mean of a distribution.
///
/// # Arguments
///
/// * `x` - Point to measure
/// * `mean` - Mean of the distribution
/// * `cov` - Covariance matrix of the distribution
///
/// # Returns
///
/// * Mahalanobis distance (scalar)
pub fn mahalanobis_distance<F>(
    x: &ArrayView1<F>,
    mean: &ArrayView1<F>,
    cov: &ArrayView2<F>,
) -> LinalgResult<F>
where
    F: Float + Zero + num_traits::One + num_traits::NumAssign + std::iter::Sum + 'static,
{
    if x.len() != mean.len() || x.len() != cov.nrows() || x.len() != cov.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Incompatible dimensions: x: {:?}, mean: {:?}, cov: {:?}",
            x.shape(),
            mean.shape(),
            cov.shape()
        )));
    }

    // Compute the deviation from the mean
    let dev = x - mean;

    // Solve the system cov * y = dev to get cov^-1 * dev
    let inv_dev = crate::solve::solve(cov, &dev.view(), None)?;

    // Compute the square distance: dev^T * cov^-1 * dev
    let dist_sq = dev.dot(&inv_dev);

    Ok(dist_sq.sqrt())
}
