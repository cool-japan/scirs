//! Feature selection utilities
//!
//! This module provides methods for selecting relevant features from datasets,
//! which can help reduce dimensionality and improve model performance.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use num_traits::{Float, NumCast};

use crate::error::{Result, TransformError};

/// VarianceThreshold for removing low-variance features
///
/// Features with variance below the threshold are removed. This is useful for
/// removing features that are mostly constant and don't provide much information.
pub struct VarianceThreshold {
    /// Variance threshold for feature selection
    threshold: f64,
    /// Variances computed for each feature (learned during fit)
    variances_: Option<Array1<f64>>,
    /// Indices of selected features
    selected_features_: Option<Vec<usize>>,
}

impl VarianceThreshold {
    /// Creates a new VarianceThreshold selector
    ///
    /// # Arguments
    /// * `threshold` - Features with variance below this threshold are removed (default: 0.0)
    ///
    /// # Returns
    /// * A new VarianceThreshold instance
    ///
    /// # Examples
    /// ```
    /// use scirs2_transform::selection::VarianceThreshold;
    ///
    /// // Remove features with variance less than 0.1
    /// let selector = VarianceThreshold::new(0.1);
    /// ```
    pub fn new(threshold: f64) -> Result<Self> {
        if threshold < 0.0 {
            return Err(TransformError::InvalidInput(
                "Threshold must be non-negative".to_string(),
            ));
        }

        Ok(VarianceThreshold {
            threshold,
            variances_: None,
            selected_features_: None,
        })
    }

    /// Creates a VarianceThreshold with default threshold (0.0)
    ///
    /// This will only remove features that are completely constant.
    pub fn with_defaults() -> Self {
        Self::new(0.0).unwrap()
    }

    /// Fits the VarianceThreshold to the input data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if n_samples < 2 {
            return Err(TransformError::InvalidInput(
                "At least 2 samples required to compute variance".to_string(),
            ));
        }

        // Compute variance for each feature
        let mut variances = Array1::zeros(n_features);
        let mut selected_features = Vec::new();

        for j in 0..n_features {
            let feature_data = x_f64.column(j);

            // Calculate mean
            let mean = feature_data.iter().sum::<f64>() / n_samples as f64;

            // Calculate variance (using population variance for consistency with sklearn)
            let variance = feature_data
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / n_samples as f64;

            variances[j] = variance;

            // Select feature if variance is above threshold
            if variance > self.threshold {
                selected_features.push(j);
            }
        }

        self.variances_ = Some(variances);
        self.selected_features_ = Some(selected_features);

        Ok(())
    }

    /// Transforms the input data by removing low-variance features
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data with selected features only
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if self.selected_features_.is_none() {
            return Err(TransformError::TransformationError(
                "VarianceThreshold has not been fitted".to_string(),
            ));
        }

        let selected_features = self.selected_features_.as_ref().unwrap();

        // Check feature consistency
        if let Some(ref variances) = self.variances_ {
            if n_features != variances.len() {
                return Err(TransformError::InvalidInput(format!(
                    "x has {} features, but VarianceThreshold was fitted with {} features",
                    n_features,
                    variances.len()
                )));
            }
        }

        let n_selected = selected_features.len();
        let mut transformed = Array2::zeros((n_samples, n_selected));

        // Copy selected features
        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            for i in 0..n_samples {
                transformed[[i, new_idx]] = x_f64[[i, old_idx]];
            }
        }

        Ok(transformed)
    }

    /// Fits the VarianceThreshold to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data with selected features only
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the variances computed for each feature
    ///
    /// # Returns
    /// * `Option<&Array1<f64>>` - The variances for each feature
    pub fn variances(&self) -> Option<&Array1<f64>> {
        self.variances_.as_ref()
    }

    /// Returns the indices of selected features
    ///
    /// # Returns
    /// * `Option<&Vec<usize>>` - Indices of features that pass the variance threshold
    pub fn get_support(&self) -> Option<&Vec<usize>> {
        self.selected_features_.as_ref()
    }

    /// Returns a boolean mask indicating which features are selected
    ///
    /// # Returns
    /// * `Option<Array1<bool>>` - Boolean mask where true indicates selected features
    pub fn get_support_mask(&self) -> Option<Array1<bool>> {
        if let (Some(ref variances), Some(ref selected)) =
            (&self.variances_, &self.selected_features_)
        {
            let n_features = variances.len();
            let mut mask = Array1::from_elem(n_features, false);

            for &idx in selected {
                mask[idx] = true;
            }

            Some(mask)
        } else {
            None
        }
    }

    /// Returns the number of selected features
    ///
    /// # Returns
    /// * `Option<usize>` - Number of features that pass the variance threshold
    pub fn n_features_selected(&self) -> Option<usize> {
        self.selected_features_.as_ref().map(|s| s.len())
    }

    /// Inverse transform - not applicable for feature selection
    ///
    /// This method is not implemented for feature selection as it's not possible
    /// to reconstruct removed features.
    pub fn inverse_transform<S>(&self, _x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        Err(TransformError::TransformationError(
            "inverse_transform is not supported for feature selection".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    #[test]
    fn test_variance_threshold_basic() {
        // Create test data with different variances
        // Feature 0: [1, 1, 1] - constant, variance = 0
        // Feature 1: [1, 2, 3] - varying, variance > 0
        // Feature 2: [5, 5, 5] - constant, variance = 0
        // Feature 3: [1, 3, 5] - varying, variance > 0
        let data = Array::from_shape_vec(
            (3, 4),
            vec![1.0, 1.0, 5.0, 1.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 5.0, 5.0],
        )
        .unwrap();

        let mut selector = VarianceThreshold::with_defaults();
        let transformed = selector.fit_transform(&data).unwrap();

        // Should keep features 1 and 3 (indices 1 and 3)
        assert_eq!(transformed.shape(), &[3, 2]);

        // Check that we kept the right features
        let selected = selector.get_support().unwrap();
        assert_eq!(selected, &[1, 3]);

        // Check transformed values
        assert_abs_diff_eq!(transformed[[0, 0]], 1.0, epsilon = 1e-10); // Feature 1, sample 0
        assert_abs_diff_eq!(transformed[[1, 0]], 2.0, epsilon = 1e-10); // Feature 1, sample 1
        assert_abs_diff_eq!(transformed[[2, 0]], 3.0, epsilon = 1e-10); // Feature 1, sample 2

        assert_abs_diff_eq!(transformed[[0, 1]], 1.0, epsilon = 1e-10); // Feature 3, sample 0
        assert_abs_diff_eq!(transformed[[1, 1]], 3.0, epsilon = 1e-10); // Feature 3, sample 1
        assert_abs_diff_eq!(transformed[[2, 1]], 5.0, epsilon = 1e-10); // Feature 3, sample 2
    }

    #[test]
    fn test_variance_threshold_custom() {
        // Create test data with specific variances
        let data = Array::from_shape_vec(
            (4, 3),
            vec![
                1.0, 1.0, 1.0, // Sample 0
                2.0, 1.1, 2.0, // Sample 1
                3.0, 1.0, 3.0, // Sample 2
                4.0, 1.1, 4.0, // Sample 3
            ],
        )
        .unwrap();

        // Set threshold to remove features with very low variance
        let mut selector = VarianceThreshold::new(0.1).unwrap();
        let transformed = selector.fit_transform(&data).unwrap();

        // Feature 1 has very low variance (between 1.0 and 1.1), should be removed
        // Features 0 and 2 have higher variance, should be kept
        assert_eq!(transformed.shape(), &[4, 2]);

        let selected = selector.get_support().unwrap();
        assert_eq!(selected, &[0, 2]);

        // Check variances
        let variances = selector.variances().unwrap();
        assert!(variances[0] > 0.1); // Feature 0 variance
        assert!(variances[1] <= 0.1); // Feature 1 variance (should be low)
        assert!(variances[2] > 0.1); // Feature 2 variance
    }

    #[test]
    fn test_variance_threshold_support_mask() {
        let data = Array::from_shape_vec(
            (3, 4),
            vec![1.0, 1.0, 5.0, 1.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 5.0, 5.0],
        )
        .unwrap();

        let mut selector = VarianceThreshold::with_defaults();
        selector.fit(&data).unwrap();

        let mask = selector.get_support_mask().unwrap();
        assert_eq!(mask.len(), 4);
        assert!(!mask[0]); // Feature 0 is constant
        assert!(mask[1]); // Feature 1 has variance
        assert!(!mask[2]); // Feature 2 is constant
        assert!(mask[3]); // Feature 3 has variance

        assert_eq!(selector.n_features_selected().unwrap(), 2);
    }

    #[test]
    fn test_variance_threshold_all_removed() {
        // Create data where all features are constant
        let data = Array::from_shape_vec((3, 2), vec![5.0, 10.0, 5.0, 10.0, 5.0, 10.0]).unwrap();

        let mut selector = VarianceThreshold::with_defaults();
        let transformed = selector.fit_transform(&data).unwrap();

        // All features should be removed
        assert_eq!(transformed.shape(), &[3, 0]);
        assert_eq!(selector.n_features_selected().unwrap(), 0);
    }

    #[test]
    fn test_variance_threshold_errors() {
        // Test negative threshold
        assert!(VarianceThreshold::new(-0.1).is_err());

        // Test with insufficient samples
        let small_data = Array::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let mut selector = VarianceThreshold::with_defaults();
        assert!(selector.fit(&small_data).is_err());

        // Test transform before fit
        let data = Array::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let selector_unfitted = VarianceThreshold::with_defaults();
        assert!(selector_unfitted.transform(&data).is_err());

        // Test inverse transform (should always fail)
        let mut selector = VarianceThreshold::with_defaults();
        selector.fit(&data).unwrap();
        assert!(selector.inverse_transform(&data).is_err());
    }

    #[test]
    fn test_variance_threshold_feature_mismatch() {
        let train_data =
            Array::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();
        let test_data = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(); // Different number of features

        let mut selector = VarianceThreshold::with_defaults();
        selector.fit(&train_data).unwrap();
        assert!(selector.transform(&test_data).is_err());
    }

    #[test]
    fn test_variance_calculation() {
        // Test variance calculation manually
        // Data: [1, 2, 3] should have variance = ((1-2)² + (2-2)² + (3-2)²) / 3 = (1 + 0 + 1) / 3 = 2/3
        let data = Array::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();

        let mut selector = VarianceThreshold::with_defaults();
        selector.fit(&data).unwrap();

        let variances = selector.variances().unwrap();
        let expected_variance = 2.0 / 3.0;
        assert_abs_diff_eq!(variances[0], expected_variance, epsilon = 1e-10);
    }
}
