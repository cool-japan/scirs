//! Feature selection utilities
//!
//! This module provides methods for selecting relevant features from datasets,
//! which can help reduce dimensionality and improve model performance.

use scirs2_core::ndarray::ArrayStatCompat;
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};

use crate::error::{Result, TransformError};
use statrs::statistics::Statistics;

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
        let x_f64 = x.mapv(|x| NumCast::from(x).unwrap_or(0.0));

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
        let x_f64 = x.mapv(|x| NumCast::from(x).unwrap_or(0.0));

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

/// Recursive Feature Elimination (RFE) for feature selection
///
/// RFE works by recursively removing features and evaluating model performance.
/// This implementation uses a feature importance scoring function to rank features.
#[derive(Debug, Clone)]
pub struct RecursiveFeatureElimination<F>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Result<Array1<f64>>,
{
    /// Number of features to select
    n_features_to_select: usize,
    /// Number of features to remove at each iteration
    step: usize,
    /// Feature importance scoring function
    /// Takes (X, y) and returns importance scores for each feature
    importance_func: F,
    /// Indices of selected features
    selected_features_: Option<Vec<usize>>,
    /// Feature rankings (1 is best)
    ranking_: Option<Array1<usize>>,
    /// Feature importance scores
    scores_: Option<Array1<f64>>,
}

impl<F> RecursiveFeatureElimination<F>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Result<Array1<f64>>,
{
    /// Creates a new RFE selector
    ///
    /// # Arguments
    /// * `n_features_to_select` - Number of features to select
    /// * `importance_func` - Function that computes feature importance scores
    pub fn new(n_features_to_select: usize, importancefunc: F) -> Self {
        RecursiveFeatureElimination {
            n_features_to_select,
            step: 1,
            importance_func: importancefunc,
            selected_features_: None,
            ranking_: None,
            scores_: None,
        }
    }

    /// Set the number of features to remove at each iteration
    pub fn with_step(mut self, step: usize) -> Self {
        self.step = step.max(1);
        self
    }

    /// Fit the RFE selector
    ///
    /// # Arguments
    /// * `x` - Training data, shape (n_samples, n_features)
    /// * `y` - Target values, shape (n_samples,)
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_samples != y.len() {
            return Err(TransformError::InvalidInput(format!(
                "X has {} samples but y has {} samples",
                n_samples,
                y.len()
            )));
        }

        if self.n_features_to_select > n_features {
            return Err(TransformError::InvalidInput(format!(
                "n_features_to_select={} must be <= n_features={}",
                self.n_features_to_select, n_features
            )));
        }

        // Initialize with all features
        let mut remaining_features: Vec<usize> = (0..n_features).collect();
        let mut ranking = Array1::zeros(n_features);
        let mut current_rank = 1;

        // Recursively eliminate features
        while remaining_features.len() > self.n_features_to_select {
            // Create subset of data with remaining features
            let x_subset = self.subset_features(x, &remaining_features);

            // Get feature importances
            let importances = (self.importance_func)(&x_subset, y)?;

            if importances.len() != remaining_features.len() {
                return Err(TransformError::InvalidInput(
                    "Importance function returned wrong number of scores".to_string(),
                ));
            }

            // Find features to eliminate
            let n_to_remove = (self.step).min(remaining_features.len() - self.n_features_to_select);

            // Get indices of features with lowest importance
            let mut indices: Vec<usize> = (0..importances.len()).collect();
            indices.sort_by(|&i, &j| importances[i].partial_cmp(&importances[j]).unwrap());

            // Mark eliminated features with current rank
            for i in 0..n_to_remove {
                let feature_idx = remaining_features[indices[i]];
                ranking[feature_idx] = n_features - current_rank + 1;
                current_rank += 1;
            }

            // Remove eliminated features
            let eliminated: std::collections::HashSet<usize> =
                indices.iter().take(n_to_remove).cloned().collect();
            let features_to_retain: Vec<usize> = remaining_features
                .iter()
                .filter(|&&idx| !eliminated.contains(&idx))
                .cloned()
                .collect();
            remaining_features = features_to_retain;
        }

        // Mark remaining features as rank 1
        for &feature_idx in &remaining_features {
            ranking[feature_idx] = 1;
        }

        // Compute final scores for selected features
        let x_final = self.subset_features(x, &remaining_features);
        let final_scores = (self.importance_func)(&x_final, y)?;

        let mut scores = Array1::zeros(n_features);
        for (i, &feature_idx) in remaining_features.iter().enumerate() {
            scores[feature_idx] = final_scores[i];
        }

        self.selected_features_ = Some(remaining_features);
        self.ranking_ = Some(ranking);
        self.scores_ = Some(scores);

        Ok(())
    }

    /// Create a subset of features
    fn subset_features(&self, x: &Array2<f64>, features: &[usize]) -> Array2<f64> {
        let n_samples = x.shape()[0];
        let n_selected = features.len();
        let mut subset = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in features.iter().enumerate() {
            subset.column_mut(new_idx).assign(&x.column(old_idx));
        }

        subset
    }

    /// Transform data by selecting features
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if self.selected_features_.is_none() {
            return Err(TransformError::TransformationError(
                "RFE has not been fitted".to_string(),
            ));
        }

        let selected = self.selected_features_.as_ref().unwrap();
        Ok(self.subset_features(x, selected))
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array2<f64>> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Get selected feature indices
    pub fn get_support(&self) -> Option<&Vec<usize>> {
        self.selected_features_.as_ref()
    }

    /// Get feature rankings (1 is best)
    pub fn ranking(&self) -> Option<&Array1<usize>> {
        self.ranking_.as_ref()
    }

    /// Get feature scores
    pub fn scores(&self) -> Option<&Array1<f64>> {
        self.scores_.as_ref()
    }
}

/// Mutual information based feature selection
///
/// Selects features based on mutual information between features and target.
#[derive(Debug, Clone)]
pub struct MutualInfoSelector {
    /// Number of features to select
    k: usize,
    /// Whether to use discrete mutual information
    discrete_target: bool,
    /// Number of neighbors for KNN estimation
    n_neighbors: usize,
    /// Selected feature indices
    selected_features_: Option<Vec<usize>>,
    /// Mutual information scores
    scores_: Option<Array1<f64>>,
}

impl MutualInfoSelector {
    /// Create a new mutual information selector
    ///
    /// # Arguments
    /// * `k` - Number of top features to select
    pub fn new(k: usize) -> Self {
        MutualInfoSelector {
            k,
            discrete_target: false,
            n_neighbors: 3,
            selected_features_: None,
            scores_: None,
        }
    }

    /// Use discrete mutual information (for classification)
    pub fn with_discrete_target(mut self) -> Self {
        self.discrete_target = true;
        self
    }

    /// Set number of neighbors for KNN estimation
    pub fn with_n_neighbors(mut self, nneighbors: usize) -> Self {
        self.n_neighbors = nneighbors;
        self
    }

    /// Estimate mutual information using KNN method (simplified)
    fn estimate_mutual_info(&self, x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        let n = x.len();
        if n < self.n_neighbors + 1 {
            return 0.0;
        }

        // Simple correlation-based approximation for continuous variables
        if !self.discrete_target {
            // Standardize variables
            let x_mean = x.mean_or(0.0);
            let y_mean = y.mean_or(0.0);
            let x_std = x.std(0.0);
            let y_std = y.std(0.0);

            if x_std < 1e-10 || y_std < 1e-10 {
                return 0.0;
            }

            let mut correlation = 0.0;
            for i in 0..n {
                correlation += (x[i] - x_mean) * (y[i] - y_mean);
            }
            correlation /= (n as f64 - 1.0) * x_std * y_std;

            // Convert correlation to mutual information approximation
            // MI ≈ -0.5 * log(1 - r²) for Gaussian variables
            if correlation.abs() >= 1.0 {
                return 5.0; // Cap at reasonable value
            }
            (-0.5 * (1.0 - correlation * correlation).ln()).max(0.0)
        } else {
            // For discrete targets, use a simple grouping approach
            let mut groups = std::collections::HashMap::new();

            for i in 0..n {
                let key = y[i].round() as i64;
                groups.entry(key).or_insert_with(Vec::new).push(x[i]);
            }

            // Calculate between-group variance / total variance ratio
            let total_mean = x.mean_or(0.0);
            let total_var = x.variance();

            if total_var < 1e-10 {
                return 0.0;
            }

            let mut between_var = 0.0;
            for (_, values) in groups {
                let group_mean = values.iter().sum::<f64>() / values.len() as f64;
                let weight = values.len() as f64 / n as f64;
                between_var += weight * (group_mean - total_mean).powi(2);
            }

            (between_var / total_var).min(1.0) * 2.0 // Scale to reasonable range
        }
    }

    /// Fit the selector
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_features = x.shape()[1];

        if self.k > n_features {
            return Err(TransformError::InvalidInput(format!(
                "k={} must be <= n_features={}",
                self.k, n_features
            )));
        }

        // Compute mutual information for each feature
        let mut scores = Array1::zeros(n_features);

        for j in 0..n_features {
            let feature = x.column(j).to_owned();
            scores[j] = self.estimate_mutual_info(&feature, y);
        }

        // Select top k features
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap());

        let selected_features = indices.into_iter().take(self.k).collect();

        self.scores_ = Some(scores);
        self.selected_features_ = Some(selected_features);

        Ok(())
    }

    /// Transform data by selecting features
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if self.selected_features_.is_none() {
            return Err(TransformError::TransformationError(
                "MutualInfoSelector has not been fitted".to_string(),
            ));
        }

        let selected = self.selected_features_.as_ref().unwrap();
        let n_samples = x.shape()[0];
        let mut transformed = Array2::zeros((n_samples, self.k));

        for (new_idx, &old_idx) in selected.iter().enumerate() {
            transformed.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(transformed)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array2<f64>> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Get selected feature indices
    pub fn get_support(&self) -> Option<&Vec<usize>> {
        self.selected_features_.as_ref()
    }

    /// Get mutual information scores
    pub fn scores(&self) -> Option<&Array1<f64>> {
        self.scores_.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array;

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

    #[test]
    fn test_rfe_basic() {
        // Create test data where features have clear importance
        let n_samples = 100;
        let mut data_vec = Vec::new();
        let mut target_vec = Vec::new();

        for i in 0..n_samples {
            let x1 = i as f64 / n_samples as f64;
            let x2 = (i as f64 / n_samples as f64).sin();
            let x3 = scirs2_core::random::random::<f64>(); // Noise
            let x4 = 2.0 * x1; // Highly correlated with target

            data_vec.extend_from_slice(&[x1, x2, x3, x4]);
            target_vec.push(3.0 * x1 + x4 + 0.1 * scirs2_core::random::random::<f64>());
        }

        let x = Array::from_shape_vec((n_samples, 4), data_vec).unwrap();
        let y = Array::from_vec(target_vec);

        // Simple importance function based on correlation
        let importance_func = |x: &Array2<f64>, y: &Array1<f64>| -> Result<Array1<f64>> {
            let n_features = x.shape()[1];
            let mut scores = Array1::zeros(n_features);

            for j in 0..n_features {
                let feature = x.column(j);
                let corr = pearson_correlation(&feature.to_owned(), y);
                scores[j] = corr.abs();
            }

            Ok(scores)
        };

        let mut rfe = RecursiveFeatureElimination::new(2, importance_func);
        let transformed = rfe.fit_transform(&x, &y).unwrap();

        // Should select 2 features
        assert_eq!(transformed.shape()[1], 2);

        // Check that features 0 and 3 (most important) were selected
        let selected = rfe.get_support().unwrap();
        assert!(selected.contains(&0) || selected.contains(&3));
    }

    #[test]
    fn test_mutual_info_continuous() {
        // Create data with clear relationships
        let n_samples = 100;
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        for i in 0..n_samples {
            let t = i as f64 / n_samples as f64 * 2.0 * std::f64::consts::PI;

            // Feature 0: Strongly related to target
            let x0 = t;
            // Feature 1: Noise
            let x1 = scirs2_core::random::random::<f64>();
            // Feature 2: Non-linearly related
            let x2 = t.sin();

            x_data.extend_from_slice(&[x0, x1, x2]);
            y_data.push(t + 0.5 * t.sin());
        }

        let x = Array::from_shape_vec((n_samples, 3), x_data).unwrap();
        let y = Array::from_vec(y_data);

        let mut selector = MutualInfoSelector::new(2);
        selector.fit(&x, &y).unwrap();

        let scores = selector.scores().unwrap();

        // Feature 0 should have highest score (linear relationship)
        // Feature 2 should have second highest (non-linear relationship)
        // Feature 1 should have lowest score (noise)
        assert!(scores[0] > scores[1]);
        assert!(scores[2] > scores[1]);
    }

    #[test]
    fn test_mutual_info_discrete() {
        // Create classification-like data
        let x = Array::from_shape_vec(
            (6, 3),
            vec![
                1.0, 0.1, 5.0, // Class 0
                1.1, 0.2, 5.1, // Class 0
                2.0, 0.1, 4.0, // Class 1
                2.1, 0.2, 4.1, // Class 1
                3.0, 0.1, 3.0, // Class 2
                3.1, 0.2, 3.1, // Class 2
            ],
        )
        .unwrap();

        let y = Array::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

        let mut selector = MutualInfoSelector::new(2).with_discrete_target();
        let transformed = selector.fit_transform(&x, &y).unwrap();

        assert_eq!(transformed.shape(), &[6, 2]);

        // Feature 1 (middle column) has low variance within groups, should be excluded
        let selected = selector.get_support().unwrap();
        assert!(!selected.contains(&1));
    }

    // Helper function for correlation
    fn pearson_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        #[allow(unused_variables)]
        let n = x.len() as f64;
        let x_mean = x.mean_or(0.0);
        let y_mean = y.mean_or(0.0);

        let mut num = 0.0;
        let mut x_var = 0.0;
        let mut y_var = 0.0;

        for i in 0..x.len() {
            let x_diff = x[i] - x_mean;
            let y_diff = y[i] - y_mean;
            num += x_diff * y_diff;
            x_var += x_diff * x_diff;
            y_var += y_diff * y_diff;
        }

        if x_var * y_var > 0.0 {
            num / (x_var * y_var).sqrt()
        } else {
            0.0
        }
    }
}
