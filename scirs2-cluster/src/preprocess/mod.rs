//! Data preprocessing utilities for clustering algorithms
//!
//! This module provides functions for preprocessing data before applying
//! clustering algorithms, such as:
//! - Whitening: Scaling features to have unit variance
//! - Normalization: Scaling data to a specific range or norm
//! - Standardization: Transforming data to have zero mean and unit variance

use ndarray::{Array1, Array2, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Whiten a dataset by rescaling each feature to have unit variance
///
/// This is useful for preprocessing before applying certain clustering algorithms
/// like K-means. Each feature is divided by its standard deviation to give it
/// unit variance.
///
/// # Arguments
///
/// * `data` - Input data as a 2D array (n_samples × n_features)
/// * `check_finite` - Whether to check for NaN or infinite values
///
/// # Returns
///
/// * `Result<Array2<F>>` - The whitened data
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_cluster::preprocess::whiten;
///
/// // Example data with 3 features
/// let data = Array2::from_shape_vec((3, 3), vec![
///     1.9, 2.3, 1.7,
///     1.5, 2.5, 2.2,
///     0.8, 0.6, 1.7,
/// ]).unwrap();
///
/// // Whiten the data
/// let whitened = whiten(data.view(), true).unwrap();
/// ```
pub fn whiten<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    check_finite: bool,
) -> Result<Array2<F>> {
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if n_samples == 0 || n_features == 0 {
        return Err(ClusteringError::InvalidInput("Input data is empty".into()));
    }

    if check_finite {
        // Check for NaN or infinite values
        for element in data.iter() {
            if !element.is_finite() {
                return Err(ClusteringError::InvalidInput(
                    "Input data contains NaN or infinite values".into(),
                ));
            }
        }
    }

    // Calculate the standard deviation for each feature
    let std_dev = standard_deviation(data, Axis(0))?;

    // Create the whitened data
    let mut result = Array2::zeros(data.dim());

    // Scale each feature by its standard deviation
    for j in 0..n_features {
        let std_j = std_dev[j];
        if std_j <= F::epsilon() {
            // If the standard deviation is close to zero, don't scale
            for i in 0..n_samples {
                result[[i, j]] = data[[i, j]];
            }
        } else {
            for i in 0..n_samples {
                result[[i, j]] = data[[i, j]] / std_j;
            }
        }
    }

    Ok(result)
}

/// Standardize a dataset by rescaling each feature to have zero mean and unit variance
///
/// # Arguments
///
/// * `data` - Input data as a 2D array (n_samples × n_features)
/// * `check_finite` - Whether to check for NaN or infinite values
///
/// # Returns
///
/// * `Result<Array2<F>>` - The standardized data
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_cluster::preprocess::standardize;
///
/// // Example data with 3 features
/// let data = Array2::from_shape_vec((3, 3), vec![
///     1.9, 2.3, 1.7,
///     1.5, 2.5, 2.2,
///     0.8, 0.6, 1.7,
/// ]).unwrap();
///
/// // Standardize the data
/// let standardized = standardize(data.view(), true).unwrap();
/// ```
pub fn standardize<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    check_finite: bool,
) -> Result<Array2<F>> {
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if n_samples == 0 || n_features == 0 {
        return Err(ClusteringError::InvalidInput("Input data is empty".into()));
    }

    if check_finite {
        // Check for NaN or infinite values
        for element in data.iter() {
            if !element.is_finite() {
                return Err(ClusteringError::InvalidInput(
                    "Input data contains NaN or infinite values".into(),
                ));
            }
        }
    }

    // Calculate the mean for each feature
    let mean = data.mean_axis(Axis(0)).unwrap();

    // Calculate the standard deviation for each feature
    let std_dev = standard_deviation(data, Axis(0))?;

    // Create the standardized data
    let mut result = Array2::zeros(data.dim());

    // Scale each feature to zero mean and unit variance
    for j in 0..n_features {
        let mean_j = mean[j];
        let std_j = std_dev[j];

        if std_j <= F::epsilon() {
            // If the standard deviation is close to zero, just subtract the mean
            for i in 0..n_samples {
                result[[i, j]] = data[[i, j]] - mean_j;
            }
        } else {
            for i in 0..n_samples {
                result[[i, j]] = (data[[i, j]] - mean_j) / std_j;
            }
        }
    }

    Ok(result)
}

/// Normalize each sample to a given norm
///
/// # Arguments
///
/// * `data` - Input data as a 2D array (n_samples × n_features)
/// * `norm` - Type of normalization: L1, L2, or Max
/// * `check_finite` - Whether to check for NaN or infinite values
///
/// # Returns
///
/// * `Result<Array2<F>>` - The normalized data
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_cluster::preprocess::{normalize, NormType};
///
/// // Example data with 3 features
/// let data = Array2::from_shape_vec((3, 3), vec![
///     1.9, 2.3, 1.7,
///     1.5, 2.5, 2.2,
///     0.8, 0.6, 1.7,
/// ]).unwrap();
///
/// // Normalize the data using L2 norm
/// let normalized = normalize(data.view(), NormType::L2, true).unwrap();
/// ```
pub fn normalize<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    norm: NormType,
    check_finite: bool,
) -> Result<Array2<F>> {
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if n_samples == 0 || n_features == 0 {
        return Err(ClusteringError::InvalidInput("Input data is empty".into()));
    }

    if check_finite {
        // Check for NaN or infinite values
        for element in data.iter() {
            if !element.is_finite() {
                return Err(ClusteringError::InvalidInput(
                    "Input data contains NaN or infinite values".into(),
                ));
            }
        }
    }

    // Calculate the norm for each sample
    let norms = match norm {
        NormType::L1 => {
            // L1 norm (sum of absolute values)
            let mut norms = Array1::zeros(n_samples);
            for i in 0..n_samples {
                let row = data.row(i);
                let row_norm = row.iter().fold(F::zero(), |acc, &x| acc + x.abs());
                norms[i] = row_norm;
            }
            norms
        }
        NormType::L2 => {
            // L2 norm (square root of sum of squares)
            let mut norms = Array1::zeros(n_samples);
            for i in 0..n_samples {
                let row = data.row(i);
                let row_norm = row.iter().fold(F::zero(), |acc, &x| acc + x * x).sqrt();
                norms[i] = row_norm;
            }
            norms
        }
        NormType::Max => {
            // Max norm (maximum absolute value)
            let mut norms = Array1::zeros(n_samples);
            for i in 0..n_samples {
                let row = data.row(i);
                let row_norm = row.iter().fold(F::zero(), |acc, &x| acc.max(x.abs()));
                norms[i] = row_norm;
            }
            norms
        }
    };

    // Create the normalized data
    let mut result = Array2::zeros(data.dim());

    // Scale each sample by its norm
    for i in 0..n_samples {
        let norm_i = norms[i];
        if norm_i <= F::epsilon() {
            // If the norm is close to zero, don't normalize
            for j in 0..n_features {
                result[[i, j]] = data[[i, j]];
            }
        } else {
            for j in 0..n_features {
                result[[i, j]] = data[[i, j]] / norm_i;
            }
        }
    }

    Ok(result)
}

/// Normalize data to a specified range (min-max scaling)
///
/// # Arguments
///
/// * `data` - Input data as a 2D array (n_samples × n_features)
/// * `feature_range` - Tuple of (min, max) values to scale to
/// * `check_finite` - Whether to check for NaN or infinite values
///
/// # Returns
///
/// * `Result<Array2<F>>` - The scaled data
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_cluster::preprocess::min_max_scale;
///
/// // Example data with 3 features
/// let data = Array2::from_shape_vec((3, 3), vec![
///     1.9, 2.3, 1.7,
///     1.5, 2.5, 2.2,
///     0.8, 0.6, 1.7,
/// ]).unwrap();
///
/// // Scale the data to the range [0, 1]
/// let scaled = min_max_scale(data.view(), (0.0, 1.0), true).unwrap();
/// ```
pub fn min_max_scale<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    feature_range: (f64, f64),
    check_finite: bool,
) -> Result<Array2<F>> {
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if n_samples == 0 || n_features == 0 {
        return Err(ClusteringError::InvalidInput("Input data is empty".into()));
    }

    if check_finite {
        // Check for NaN or infinite values
        for element in data.iter() {
            if !element.is_finite() {
                return Err(ClusteringError::InvalidInput(
                    "Input data contains NaN or infinite values".into(),
                ));
            }
        }
    }

    let (min_val, max_val) = feature_range;
    if min_val >= max_val {
        return Err(ClusteringError::InvalidInput(
            "Feature range minimum must be less than maximum".into(),
        ));
    }

    let feature_min = F::from_f64(min_val).unwrap();
    let feature_max = F::from_f64(max_val).unwrap();

    // Calculate the min and max for each feature
    let mut min_values = Array1::zeros(n_features);
    let mut max_values = Array1::zeros(n_features);

    for j in 0..n_features {
        let column = data.column(j);
        let (min_j, max_j) = column.iter().fold(
            (F::infinity(), F::neg_infinity()),
            |(min_val, max_val), &x| (min_val.min(x), max_val.max(x)),
        );
        min_values[j] = min_j;
        max_values[j] = max_j;
    }

    // Create the scaled data
    let mut result = Array2::zeros(data.dim());

    // Scale each feature to the specified range
    for j in 0..n_features {
        let min_j = min_values[j];
        let max_j = max_values[j];
        let range_j = max_j - min_j;

        if range_j <= F::epsilon() {
            // If the feature has no variation, set to the middle of the feature range
            let middle = (feature_min + feature_max) / F::from_f64(2.0).unwrap();
            for i in 0..n_samples {
                result[[i, j]] = middle;
            }
        } else {
            for i in 0..n_samples {
                // Scale to [0, 1]
                let scaled = (data[[i, j]] - min_j) / range_j;
                // Scale to [feature_min, feature_max]
                result[[i, j]] = scaled * (feature_max - feature_min) + feature_min;
            }
        }
    }

    Ok(result)
}

/// Normalization types for the normalize function
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// L1 norm (sum of absolute values)
    L1,
    /// L2 norm (square root of sum of squares)
    L2,
    /// Max norm (maximum absolute value)
    Max,
}

/// Calculate the standard deviation along the specified axis
fn standard_deviation<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    axis: Axis,
) -> Result<Array1<F>> {
    let mean = data.mean_axis(axis).unwrap();
    let n = F::from_usize(match axis {
        Axis(0) => data.shape()[0],
        Axis(1) => data.shape()[1],
        _ => return Err(ClusteringError::InvalidInput("Invalid axis".into())),
    })
    .unwrap();

    let mut variance = match axis {
        Axis(0) => Array1::zeros(data.shape()[1]),
        Axis(1) => Array1::zeros(data.shape()[0]),
        _ => return Err(ClusteringError::InvalidInput("Invalid axis".into())),
    };

    if axis == Axis(0) {
        // Calculate variance along rows (for each feature)
        let n_features = data.shape()[1];
        for j in 0..n_features {
            let mut sum_squared_diff = F::zero();
            for i in 0..data.shape()[0] {
                let diff = data[[i, j]] - mean[j];
                sum_squared_diff = sum_squared_diff + diff * diff;
            }
            // Avoid division by zero for single sample
            if n > F::one() {
                variance[j] = sum_squared_diff / (n - F::one());
            } else {
                variance[j] = F::zero();
            }
        }
    } else {
        // Calculate variance along columns (for each sample)
        let n_samples = data.shape()[0];
        for i in 0..n_samples {
            let mut sum_squared_diff = F::zero();
            for j in 0..data.shape()[1] {
                let diff = data[[i, j]] - mean[i];
                sum_squared_diff = sum_squared_diff + diff * diff;
            }
            // Avoid division by zero for single feature
            if n > F::one() {
                variance[i] = sum_squared_diff / (n - F::one());
            } else {
                variance[i] = F::zero();
            }
        }
    }

    // Calculate standard deviation
    let std_dev = variance.mapv(|x| x.sqrt());

    // Replace zeros with ones to avoid division by zero
    let std_dev = std_dev.mapv(|x| if x <= F::epsilon() { F::one() } else { x });

    Ok(std_dev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    #[test]
    fn test_whiten() {
        let data =
            Array2::from_shape_vec((3, 3), vec![1.9, 2.3, 1.7, 1.5, 2.5, 2.2, 0.8, 0.6, 1.7])
                .unwrap();

        let whitened = whiten(data.view(), true).unwrap();

        // Check that each feature has approximately unit variance
        let std_dev = standard_deviation(whitened.view(), Axis(0)).unwrap();
        for &std in std_dev.iter() {
            assert_abs_diff_eq!(std, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_standardize() {
        let data =
            Array2::from_shape_vec((3, 3), vec![1.9, 2.3, 1.7, 1.5, 2.5, 2.2, 0.8, 0.6, 1.7])
                .unwrap();

        let standardized = standardize(data.view(), true).unwrap();

        // Check that each feature has approximately zero mean
        let mean = standardized.mean_axis(Axis(0)).unwrap();
        for mean_val in mean.iter() {
            assert_abs_diff_eq!(*mean_val, 0.0, epsilon = 1e-10);
        }

        // Check that each feature has approximately unit variance
        let std_dev = standard_deviation(standardized.view(), Axis(0)).unwrap();
        for std_val in std_dev.iter() {
            assert_abs_diff_eq!(*std_val, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normalize_l2() {
        let data =
            Array2::from_shape_vec((3, 3), vec![1.9, 2.3, 1.7, 1.5, 2.5, 2.2, 0.8, 0.6, 1.7])
                .unwrap();

        let normalized = normalize(data.view(), NormType::L2, true).unwrap();

        // Check that each sample has L2 norm approximately equal to 1
        for i in 0..data.shape()[0] {
            let row = normalized.row(i);
            let norm_sq: f64 = row.iter().fold(0.0, |acc, &x| acc + x * x);
            let norm = norm_sq.sqrt();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_min_max_scale() {
        let data =
            Array2::from_shape_vec((3, 3), vec![1.9, 2.3, 1.7, 1.5, 2.5, 2.2, 0.8, 0.6, 1.7])
                .unwrap();

        let scaled = min_max_scale(data.view(), (0.0, 1.0), true).unwrap();

        // Check that all values are in the range [0, 1]
        for val in scaled.iter() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }

        // Check that for each feature, the minimum value is 0.0 and the maximum is 1.0
        for j in 0..data.shape()[1] {
            let column = scaled.column(j);

            // Convert the values to f64 for stable comparison
            let column_values: Vec<f64> = column.iter().copied().collect();

            if !column_values.is_empty() {
                let min_val = column_values
                    .iter()
                    .fold(f64::INFINITY, |min, &x| min.min(x));
                let max_val = column_values
                    .iter()
                    .fold(f64::NEG_INFINITY, |max, &x| max.max(x));

                // Only check if the feature had different values in the original data
                if data.column(j).iter().any(|&x| x != data[[0, j]]) {
                    assert_abs_diff_eq!(min_val, 0.0, epsilon = 1e-10);
                    assert_abs_diff_eq!(max_val, 1.0, epsilon = 1e-10);
                }
            }
        }
    }
}
