//! Data scaling and normalization utilities
//!
//! This module provides various methods for scaling and normalizing data to improve
//! the performance of machine learning algorithms. It includes standard normalization
//! (z-score), min-max scaling, and robust scaling that is resistant to outliers.

use ndarray::Array2;

/// Helper function to normalize data (zero mean, unit variance)
///
/// This function normalizes each feature (column) in the dataset to have zero mean
/// and unit variance. This is commonly used as a preprocessing step for machine learning.
/// Also known as z-score normalization or standardization.
///
/// # Arguments
///
/// * `data` - A mutable reference to the data array to normalize in-place
///
/// # Examples
///
/// ```rust
/// use ndarray::Array2;
/// use scirs2_datasets::utils::normalize;
///
/// let mut data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// normalize(&mut data);
/// // data is now normalized with zero mean and unit variance for each feature
/// ```
pub fn normalize(data: &mut Array2<f64>) {
    let n_features = data.ncols();

    for j in 0..n_features {
        let mut column = data.column_mut(j);

        // Calculate mean and std
        let mean = column.mean().unwrap_or(0.0);
        let std = column.std(0.0);

        // Avoid division by zero
        if std > 1e-10 {
            column.mapv_inplace(|x| (x - mean) / std);
        }
    }
}

/// Performs Min-Max scaling to scale features to a specified range
///
/// Transforms features by scaling each feature to a given range, typically [0, 1].
/// The transformation is: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
///
/// # Arguments
///
/// * `data` - Feature matrix to scale in-place (n_samples, n_features)
/// * `feature_range` - Target range as (min, max) tuple
///
/// # Examples
///
/// ```rust
/// use ndarray::Array2;
/// use scirs2_datasets::utils::min_max_scale;
///
/// let mut data = Array2::from_shape_vec((3, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap();
/// min_max_scale(&mut data, (0.0, 1.0));
/// // Features are now scaled to [0, 1] range
/// ```
pub fn min_max_scale(data: &mut Array2<f64>, feature_range: (f64, f64)) {
    let (range_min, range_max) = feature_range;
    let range_size = range_max - range_min;

    for j in 0..data.ncols() {
        let mut column = data.column_mut(j);

        // Find min and max values in the column
        let col_min = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let col_max = column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Avoid division by zero
        if (col_max - col_min).abs() > 1e-10 {
            column.mapv_inplace(|x| (x - col_min) / (col_max - col_min) * range_size + range_min);
        } else {
            // If all values are the same, set to the middle of the range
            column.fill(range_min + range_size / 2.0);
        }
    }
}

/// Performs robust scaling using median and interquartile range
///
/// Scales features using statistics that are robust to outliers. Each feature is
/// scaled by: X_scaled = (X - median) / IQR, where IQR is the interquartile range.
/// This scaling method is less sensitive to outliers compared to standard normalization.
///
/// # Arguments
///
/// * `data` - Feature matrix to scale in-place (n_samples, n_features)
///
/// # Examples
///
/// ```rust
/// use ndarray::Array2;
/// use scirs2_datasets::utils::robust_scale;
///
/// let mut data = Array2::from_shape_vec((5, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 100.0, 500.0]).unwrap();
/// robust_scale(&mut data);
/// // Features are now robustly scaled using median and IQR
/// ```
pub fn robust_scale(data: &mut Array2<f64>) {
    for j in 0..data.ncols() {
        let mut column_values: Vec<f64> = data.column(j).to_vec();
        column_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = column_values.len();
        if n == 0 {
            continue;
        }

        // Calculate median
        let median = if n % 2 == 0 {
            (column_values[n / 2 - 1] + column_values[n / 2]) / 2.0
        } else {
            column_values[n / 2]
        };

        // Calculate Q1 and Q3
        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;
        let q1 = column_values[q1_idx];
        let q3 = column_values[q3_idx];
        let iqr = q3 - q1;

        // Scale the column
        let mut column = data.column_mut(j);
        if iqr > 1e-10 {
            column.mapv_inplace(|x| (x - median) / iqr);
        } else {
            // If IQR is zero, center around median but don't scale
            column.mapv_inplace(|x| x - median);
        }
    }
}

/// Trait extension for Array1 to calculate mean and standard deviation
///
/// This trait provides statistical methods for ndarray's ArrayView1 type,
/// enabling easy calculation of mean and standard deviation for scaling operations.
pub trait StatsExt {
    /// Calculate the mean of the array
    fn mean(&self) -> Option<f64>;
    /// Calculate the standard deviation with specified degrees of freedom
    fn std(&self, ddof: f64) -> f64;
}

impl StatsExt for ndarray::ArrayView1<'_, f64> {
    /// Calculate the mean of the array
    ///
    /// # Returns
    ///
    /// Some(mean) if the array is not empty, None otherwise
    fn mean(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }

        let sum: f64 = self.sum();
        Some(sum / self.len() as f64)
    }

    /// Calculate the standard deviation
    ///
    /// # Arguments
    ///
    /// * `ddof` - Degrees of freedom (delta degrees of freedom)
    ///
    /// # Returns
    ///
    /// Standard deviation of the array
    fn std(&self, ddof: f64) -> f64 {
        if self.is_empty() {
            return 0.0;
        }

        let n = self.len() as f64;
        let mean = self.mean().unwrap_or(0.0);

        let mut sum_sq = 0.0;
        for &x in self.iter() {
            let diff = x - mean;
            sum_sq += diff * diff;
        }

        let divisor = n - ddof;
        if divisor <= 0.0 {
            return 0.0;
        }

        (sum_sq / divisor).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};

    #[test]
    fn test_normalize() {
        let mut data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        normalize(&mut data);

        // Check that each column has approximately zero mean
        for j in 0..data.ncols() {
            let column = data.column(j);
            let mean = column.mean().unwrap();
            assert!(mean.abs() < 1e-10);
        }
    }

    #[test]
    fn test_normalize_constant_values() {
        let mut data = Array2::from_shape_vec((3, 1), vec![5.0, 5.0, 5.0]).unwrap();
        normalize(&mut data);

        // Constant values should remain unchanged (avoid division by zero)
        for i in 0..data.nrows() {
            assert_eq!(data[[i, 0]], 5.0);
        }
    }

    #[test]
    fn test_min_max_scale() {
        let mut data =
            Array2::from_shape_vec((3, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap();
        min_max_scale(&mut data, (0.0, 1.0));

        // Check that values are scaled to [0, 1]
        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                let value = data[[i, j]];
                assert!(value >= 0.0 && value <= 1.0);
            }
        }

        // Check specific scaling: first column should be [0, 0.5, 1]
        assert!((data[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((data[[1, 0]] - 0.5).abs() < 1e-10);
        assert!((data[[2, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_max_scale_custom_range() {
        let mut data = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        min_max_scale(&mut data, (-1.0, 1.0));

        // Check that values are scaled to [-1, 1]
        assert!((data[[0, 0]] - (-1.0)).abs() < 1e-10);
        assert!((data[[1, 0]] - 0.0).abs() < 1e-10);
        assert!((data[[2, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_max_scale_constant_values() {
        let mut data = Array2::from_shape_vec((3, 1), vec![5.0, 5.0, 5.0]).unwrap();
        min_max_scale(&mut data, (0.0, 1.0));

        // All values should be 0.5 (middle of range) when all values are the same
        for i in 0..data.nrows() {
            assert!((data[[i, 0]] - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_robust_scale() {
        let mut data = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 100.0, 500.0],
        )
        .unwrap(); // Last row has outliers

        robust_scale(&mut data);

        // Check that the scaling was applied (data should have different values than original)
        // and that extreme outliers have limited influence
        let col1_values: Vec<f64> = data.column(0).to_vec();
        let col2_values: Vec<f64> = data.column(1).to_vec();

        // Verify that the data has been transformed (not all values are the same)
        let col1_range = col1_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - col1_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let col2_range = col2_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - col2_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        // After robust scaling, the range should be reasonable (not infinite)
        assert!(col1_range.is_finite());
        assert!(col2_range.is_finite());
        assert!(col1_range > 0.0); // Some variation should remain
        assert!(col2_range > 0.0); // Some variation should remain
    }

    #[test]
    fn test_robust_scale_constant_values() {
        let mut data = Array2::from_shape_vec((3, 1), vec![5.0, 5.0, 5.0]).unwrap();
        robust_scale(&mut data);

        // With constant values, robust scaling should center around 0 (median subtraction)
        for i in 0..data.nrows() {
            assert!((data[[i, 0]] - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_robust_vs_standard_scaling() {
        // Create data with outliers
        let mut data_robust = Array2::from_shape_vec(
            (5, 1),
            vec![1.0, 2.0, 3.0, 4.0, 100.0], // 100.0 is an outlier
        )
        .unwrap();
        let mut data_standard = data_robust.clone();

        // Apply different scaling methods
        robust_scale(&mut data_robust);
        normalize(&mut data_standard); // Standard z-score normalization

        // Both scaling methods should produce finite, transformed data
        let robust_range = data_robust.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - data_robust.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let standard_range = data_standard
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - data_standard.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        // Both scaling methods should produce finite ranges
        assert!(robust_range.is_finite());
        assert!(standard_range.is_finite());
        assert!(robust_range > 0.0);
        assert!(standard_range > 0.0);

        // The scaled data should be different from the original
        assert!(data_robust[[0, 0]] != 1.0); // First value should be transformed
        assert!(data_standard[[0, 0]] != 1.0); // First value should be transformed
    }

    #[test]
    fn test_stats_ext_trait() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let view = data.view();

        // Test mean calculation
        let mean = view.mean().unwrap();
        assert!((mean - 3.0_f64).abs() < 1e-10);

        // Test standard deviation calculation
        let std = view.std(0.0); // Population standard deviation
        let expected_std = (10.0_f64 / 5.0).sqrt(); // sqrt(variance)
        assert!((std - expected_std).abs() < 1e-10);

        // Test with ddof = 1 (sample standard deviation)
        let std_sample = view.std(1.0);
        let expected_std_sample = (10.0_f64 / 4.0).sqrt();
        assert!((std_sample - expected_std_sample).abs() < 1e-10);
    }

    #[test]
    fn test_stats_ext_empty_array() {
        let data: Array1<f64> = array![];
        let view = data.view();

        // Mean of empty array should be None
        assert!(StatsExt::mean(&view).is_none());

        // Standard deviation of empty array should be 0
        assert_eq!(StatsExt::std(&view, 0.0), 0.0);
    }

    #[test]
    fn test_scaling_pipeline() {
        // Test a complete scaling pipeline
        let mut data1 =
            Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0])
                .unwrap();
        let mut data2 = data1.clone();
        let mut data3 = data1.clone();

        // Apply different scaling methods
        normalize(&mut data1); // Z-score normalization
        min_max_scale(&mut data2, (0.0, 1.0)); // Min-max scaling
        robust_scale(&mut data3); // Robust scaling

        // All methods should produce finite transformed data
        assert!(data1.iter().all(|&x| x.is_finite()));
        assert!(data2.iter().all(|&x| x.is_finite()));
        assert!(data3.iter().all(|&x| x.is_finite()));

        // Min-max scaled data should be in [0, 1] range
        assert!(data2.iter().all(|&x| x >= 0.0 && x <= 1.0));

        // All scaling methods should preserve shape
        assert_eq!(data1.shape(), data2.shape());
        assert_eq!(data2.shape(), data3.shape());
    }
}
