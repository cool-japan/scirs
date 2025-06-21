//! Correlation functions for ndarray arrays
//!
//! This module provides functions for calculating correlation coefficients
//! and covariance matrices.

use ndarray::{Array, ArrayView, Ix1, Ix2};
use num_traits::{Float, FromPrimitive};

/// Calculate correlation coefficient between two 1D arrays
///
/// # Arguments
///
/// * `x` - First input array
/// * `y` - Second input array (must have same shape as x)
///
/// # Returns
///
/// Pearson's correlation coefficient
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::corrcoef;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
/// let corr = corrcoef(x.view(), y.view()).unwrap();
/// assert!((corr + 1.0_f64).abs() < 1e-10); // Perfect negative correlation (-1.0)
/// ```
///
/// This function is similar to ``NumPy``'s `np.corrcoef` function but returns a single value.
pub fn corrcoef<T>(x: ArrayView<T, Ix1>, y: ArrayView<T, Ix1>) -> Result<T, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    if x.is_empty() || y.is_empty() {
        return Err("Cannot compute correlation of empty arrays");
    }

    if x.len() != y.len() {
        return Err("Arrays must have the same length");
    }

    // Calculate means
    let n = T::from_usize(x.len()).unwrap();
    let mut sum_x = T::zero();
    let mut sum_y = T::zero();

    for (&x_val, &y_val) in x.iter().zip(y.iter()) {
        sum_x = sum_x + x_val;
        sum_y = sum_y + y_val;
    }

    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    // Calculate covariance and variances
    let mut cov_xy = T::zero();
    let mut var_x = T::zero();
    let mut var_y = T::zero();

    for (&x_val, &y_val) in x.iter().zip(y.iter()) {
        let dx = x_val - mean_x;
        let dy = y_val - mean_y;
        cov_xy = cov_xy + dx * dy;
        var_x = var_x + dx * dx;
        var_y = var_y + dy * dy;
    }

    // Calculate correlation coefficient
    if var_x.is_zero() || var_y.is_zero() {
        return Err("Correlation coefficient is not defined when either array has zero variance");
    }

    Ok(cov_xy / (var_x * var_y).sqrt())
}

/// Calculate the covariance matrix of a 2D array
///
/// # Arguments
///
/// * `array` - The input 2D array where rows are observations and columns are variables
/// * `ddof` - Delta degrees of freedom (default 1)
///
/// # Returns
///
/// The covariance matrix
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::stats::cov;
///
/// let data = array![
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
///     [10.0, 11.0, 12.0]
/// ];
/// let cov_matrix = cov(data.view(), 1).unwrap();
/// assert_eq!(cov_matrix.shape(), &[3, 3]);
/// ```
///
/// This function is equivalent to ``NumPy``'s `np.cov` function.
pub fn cov<T>(array: ArrayView<T, Ix2>, ddof: usize) -> Result<Array<T, Ix2>, &'static str>
where
    T: Clone + Float + FromPrimitive,
{
    if array.is_empty() {
        return Err("Cannot compute covariance of an empty array");
    }

    let (n_samples, n_features) = (array.shape()[0], array.shape()[1]);

    if n_samples <= ddof {
        return Err("Not enough data points for covariance calculation with given ddof");
    }

    // Calculate means for each feature
    let mut feature_means = Array::<T, Ix1>::zeros(n_features);

    for j in 0..n_features {
        let mut sum = T::zero();
        for i in 0..n_samples {
            sum = sum + array[[i, j]];
        }
        feature_means[j] = sum / T::from_usize(n_samples).unwrap();
    }

    // Calculate covariance matrix
    let mut cov_matrix = Array::<T, Ix2>::zeros((n_features, n_features));
    let scale = T::from_usize(n_samples - ddof).unwrap();

    for i in 0..n_features {
        for j in 0..=i {
            let mut cov_ij = T::zero();

            for k in 0..n_samples {
                let dev_i = array[[k, i]] - feature_means[i];
                let dev_j = array[[k, j]] - feature_means[j];
                cov_ij = cov_ij + dev_i * dev_j;
            }

            cov_ij = cov_ij / scale;
            cov_matrix[[i, j]] = cov_ij;

            // Fill symmetric part
            if i != j {
                cov_matrix[[j, i]] = cov_ij;
            }
        }
    }

    Ok(cov_matrix)
}
