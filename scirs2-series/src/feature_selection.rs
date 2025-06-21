//! Time series feature selection methods
//!
//! This module provides comprehensive feature selection methods specifically designed for time series data.
//! It includes filter methods, wrapper methods, embedded methods, and time series specific approaches.

use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};

use crate::error::{Result, TimeSeriesError};
use crate::utils::{autocorrelation, cross_correlation};

/// Feature selection result containing selected features and their scores
#[derive(Debug, Clone)]
pub struct FeatureSelectionResult {
    /// Indices of selected features
    pub selected_features: Vec<usize>,
    /// Feature scores (higher is better)
    pub feature_scores: Array1<f64>,
    /// Selection method used
    pub method: String,
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

/// Configuration for feature selection methods
#[derive(Debug, Clone)]
pub struct FeatureSelectionConfig {
    /// Number of features to select (None = automatic)
    pub n_features: Option<usize>,
    /// Scoring method for wrapper methods
    pub scoring_method: ScoringMethod,
    /// Cross-validation folds for wrapper methods
    pub cv_folds: usize,
    /// Significance level for statistical tests
    pub alpha: f64,
    /// Minimum correlation threshold for filter methods
    pub correlation_threshold: f64,
    /// Minimum variance threshold for variance-based filtering
    pub variance_threshold: f64,
    /// Maximum number of iterations for wrapper methods
    pub max_iterations: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Regularization parameter for embedded methods
    pub regularization_alpha: f64,
    /// Maximum lag for time series specific methods
    pub max_lag: usize,
    /// Seasonal period for seasonal feature selection
    pub seasonal_period: Option<usize>,
}

impl Default for FeatureSelectionConfig {
    fn default() -> Self {
        Self {
            n_features: None,
            scoring_method: ScoringMethod::MeanSquaredError,
            cv_folds: 5,
            alpha: 0.05,
            correlation_threshold: 0.1,
            variance_threshold: 0.01,
            max_iterations: 100,
            random_seed: None,
            regularization_alpha: 1.0,
            max_lag: 10,
            seasonal_period: None,
        }
    }
}

/// Scoring methods for feature selection
#[derive(Debug, Clone)]
pub enum ScoringMethod {
    /// Mean squared error
    MeanSquaredError,
    /// Mean absolute error
    MeanAbsoluteError,
    /// R-squared
    RSquared,
    /// Akaike Information Criterion
    AIC,
    /// Bayesian Information Criterion
    BIC,
    /// Cross-validation score
    CrossValidation,
}

/// Filter-based feature selection methods
pub struct FilterMethods;

impl FilterMethods {
    /// Variance-based feature selection
    ///
    /// Removes features with low variance (quasi-constant features).
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `threshold` - Minimum variance threshold
    ///
    /// # Returns
    ///
    /// * Feature selection result
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::Array2;
    /// use scirs2_series::feature_selection::FilterMethods;
    ///
    /// let features = Array2::from_shape_vec((100, 5), (0..500).map(|x| x as f64).collect()).unwrap();
    /// let result = FilterMethods::variance_threshold(&features, 0.1).unwrap();
    /// println!("Selected {} features", result.selected_features.len());
    /// ```
    pub fn variance_threshold(
        features: &Array2<f64>,
        threshold: f64,
    ) -> Result<FeatureSelectionResult> {
        let (n_samples, n_features) = features.dim();

        if n_samples < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 2 samples for variance calculation".to_string(),
                required: 2,
                actual: n_samples,
            });
        }

        let mut selected_features = Vec::new();
        let mut feature_scores = Array1::zeros(n_features);

        for i in 0..n_features {
            let feature_col = features.column(i);
            let mean = feature_col.sum() / n_samples as f64;
            let variance = feature_col.mapv(|x| (x - mean).powi(2)).sum() / (n_samples - 1) as f64;

            feature_scores[i] = variance;

            if variance >= threshold {
                selected_features.push(i);
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("threshold".to_string(), threshold);
        metadata.insert("n_selected".to_string(), selected_features.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: "VarianceThreshold".to_string(),
            metadata,
        })
    }

    /// Correlation-based feature selection
    ///
    /// Selects features based on their correlation with the target variable.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `target` - Target variable
    /// * `threshold` - Minimum absolute correlation threshold
    ///
    /// # Returns
    ///
    /// * Feature selection result
    pub fn correlation_selection(
        features: &Array2<f64>,
        target: &Array1<f64>,
        threshold: f64,
    ) -> Result<FeatureSelectionResult> {
        let (n_samples, n_features) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        if n_samples < 3 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 3 samples for correlation calculation".to_string(),
                required: 3,
                actual: n_samples,
            });
        }

        let mut selected_features = Vec::new();
        let mut feature_scores = Array1::zeros(n_features);

        let target_mean = target.sum() / n_samples as f64;

        for i in 0..n_features {
            let feature_col = features.column(i);
            let feature_mean = feature_col.sum() / n_samples as f64;

            let correlation =
                Self::calculate_correlation(&feature_col, target, feature_mean, target_mean)?;
            let abs_correlation = correlation.abs();

            feature_scores[i] = abs_correlation;

            if abs_correlation >= threshold {
                selected_features.push(i);
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("threshold".to_string(), threshold);
        metadata.insert("n_selected".to_string(), selected_features.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: "CorrelationSelection".to_string(),
            metadata,
        })
    }

    /// Mutual information-based feature selection for time series
    ///
    /// Estimates mutual information between features and target using histogram-based approach.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `target` - Target variable
    /// * `n_bins` - Number of bins for histogram estimation
    /// * `n_features` - Number of top features to select
    ///
    /// # Returns
    ///
    /// * Feature selection result
    pub fn mutual_information_selection(
        features: &Array2<f64>,
        target: &Array1<f64>,
        n_bins: usize,
        n_features: Option<usize>,
    ) -> Result<FeatureSelectionResult> {
        let (n_samples, n_feat) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        if n_samples < n_bins * 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Insufficient samples for mutual information estimation".to_string(),
                required: n_bins * 2,
                actual: n_samples,
            });
        }

        let mut feature_scores = Array1::zeros(n_feat);

        for i in 0..n_feat {
            let feature_col = features.column(i);
            let mi = Self::calculate_mutual_information(&feature_col, target, n_bins)?;
            feature_scores[i] = mi;
        }

        // Select top features
        let n_to_select = n_features.unwrap_or(n_feat / 2).min(n_feat);
        let mut indexed_scores: Vec<(usize, f64)> = feature_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected_features: Vec<usize> = indexed_scores
            .into_iter()
            .take(n_to_select)
            .map(|(idx, _)| idx)
            .collect();

        let mut metadata = HashMap::new();
        metadata.insert("n_bins".to_string(), n_bins as f64);
        metadata.insert("n_selected".to_string(), selected_features.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: "MutualInformation".to_string(),
            metadata,
        })
    }

    /// F-test based feature selection
    ///
    /// Performs F-test for feature relevance.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `target` - Target variable
    /// * `alpha` - Significance level
    ///
    /// # Returns
    ///
    /// * Feature selection result
    pub fn f_test_selection(
        features: &Array2<f64>,
        target: &Array1<f64>,
        alpha: f64,
    ) -> Result<FeatureSelectionResult> {
        let (n_samples, n_features) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        if n_samples < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 4 samples for F-test".to_string(),
                required: 4,
                actual: n_samples,
            });
        }

        let mut selected_features = Vec::new();
        let mut feature_scores = Array1::zeros(n_features);

        let target_mean = target.sum() / n_samples as f64;
        let sst = target.mapv(|y| (y - target_mean).powi(2)).sum();

        for i in 0..n_features {
            let feature_col = features.column(i);
            let f_stat = Self::calculate_f_statistic(&feature_col, target, target_mean, sst)?;

            feature_scores[i] = f_stat;

            // Critical value for F(1, n-2) distribution at significance level alpha
            // Simplified approximation - in practice, use proper F-distribution
            let critical_value = Self::f_critical_value(alpha, n_samples);

            if f_stat > critical_value {
                selected_features.push(i);
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("alpha".to_string(), alpha);
        metadata.insert("n_selected".to_string(), selected_features.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: "FTest".to_string(),
            metadata,
        })
    }

    /// Autocorrelation-based feature filtering for time series
    ///
    /// Selects features based on their autocorrelation structure.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `max_lag` - Maximum lag to consider
    /// * `threshold` - Minimum autocorrelation threshold
    ///
    /// # Returns
    ///
    /// * Feature selection result
    pub fn autocorrelation_filter(
        features: &Array2<f64>,
        max_lag: usize,
        threshold: f64,
    ) -> Result<FeatureSelectionResult> {
        let (n_samples, n_features) = features.dim();

        if n_samples <= max_lag + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Insufficient samples for autocorrelation calculation".to_string(),
                required: max_lag + 2,
                actual: n_samples,
            });
        }

        let mut selected_features = Vec::new();
        let mut feature_scores = Array1::zeros(n_features);

        for i in 0..n_features {
            let feature_col = features.column(i).to_owned();

            // Calculate autocorrelation function, handle constant series
            let max_acf = match autocorrelation(&feature_col, Some(max_lag)) {
                Ok(acf) => {
                    // Score is the maximum significant autocorrelation
                    acf.slice(ndarray::s![1..])
                        .iter()
                        .map(|&x| x.abs())
                        .fold(0.0f64, |a, b| a.max(b))
                }
                Err(_) => {
                    // Constant series or other error - assign zero autocorrelation
                    0.0
                }
            };

            feature_scores[i] = max_acf;

            if max_acf >= threshold {
                selected_features.push(i);
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("max_lag".to_string(), max_lag as f64);
        metadata.insert("threshold".to_string(), threshold);
        metadata.insert("n_selected".to_string(), selected_features.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: "AutocorrelationFilter".to_string(),
            metadata,
        })
    }

    // Helper methods

    fn calculate_correlation(
        x: &ndarray::ArrayView1<f64>,
        y: &Array1<f64>,
        x_mean: f64,
        y_mean: f64,
    ) -> Result<f64> {
        let n = x.len();

        let mut numerator = 0.0;
        let mut x_var = 0.0;
        let mut y_var = 0.0;

        for i in 0..n {
            let x_dev = x[i] - x_mean;
            let y_dev = y[i] - y_mean;

            numerator += x_dev * y_dev;
            x_var += x_dev * x_dev;
            y_var += y_dev * y_dev;
        }

        let denominator = (x_var * y_var).sqrt();

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    fn calculate_mutual_information(
        x: &ndarray::ArrayView1<f64>,
        y: &Array1<f64>,
        n_bins: usize,
    ) -> Result<f64> {
        let n = x.len();

        // Create bins
        let x_min = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if x_max == x_min || y_max == y_min {
            return Ok(0.0);
        }

        let x_bin_size = (x_max - x_min) / n_bins as f64;
        let y_bin_size = (y_max - y_min) / n_bins as f64;

        // Count occurrences
        let mut joint_counts = vec![vec![0; n_bins]; n_bins];
        let mut x_counts = vec![0; n_bins];
        let mut y_counts = vec![0; n_bins];

        for i in 0..n {
            let x_bin = ((x[i] - x_min) / x_bin_size).floor() as usize;
            let y_bin = ((y[i] - y_min) / y_bin_size).floor() as usize;

            let x_bin = x_bin.min(n_bins - 1);
            let y_bin = y_bin.min(n_bins - 1);

            joint_counts[x_bin][y_bin] += 1;
            x_counts[x_bin] += 1;
            y_counts[y_bin] += 1;
        }

        // Calculate mutual information
        let mut mi = 0.0;

        for (i, _) in x_counts.iter().enumerate().take(n_bins) {
            for (j, _) in y_counts.iter().enumerate().take(n_bins) {
                if joint_counts[i][j] > 0 && x_counts[i] > 0 && y_counts[j] > 0 {
                    let p_xy = joint_counts[i][j] as f64 / n as f64;
                    let p_x = x_counts[i] as f64 / n as f64;
                    let p_y = y_counts[j] as f64 / n as f64;

                    mi += p_xy * (p_xy / (p_x * p_y)).ln();
                }
            }
        }

        Ok(mi)
    }

    fn calculate_f_statistic(
        x: &ndarray::ArrayView1<f64>,
        y: &Array1<f64>,
        y_mean: f64,
        sst: f64,
    ) -> Result<f64> {
        let n = x.len();

        // Calculate regression coefficients
        let x_mean = x.sum() / n as f64;

        let mut num = 0.0;
        let mut den = 0.0;

        for i in 0..n {
            let x_dev = x[i] - x_mean;
            num += x_dev * (y[i] - y_mean);
            den += x_dev * x_dev;
        }

        if den == 0.0 {
            return Ok(0.0);
        }

        let beta = num / den;
        let alpha = y_mean - beta * x_mean;

        // Calculate sum of squared residuals
        let mut ssr = 0.0;
        for i in 0..n {
            let y_pred = alpha + beta * x[i];
            ssr += (y[i] - y_pred).powi(2);
        }

        let sse = sst - ssr;

        if ssr == 0.0 {
            return Ok(f64::INFINITY);
        }

        // F-statistic = (SSE/1) / (SSR/(n-2))
        let f_stat = (sse * (n - 2) as f64) / ssr;

        Ok(f_stat)
    }

    fn f_critical_value(alpha: f64, n: usize) -> f64 {
        // Simplified approximation for F(1, n-2) critical values
        // In practice, use proper F-distribution tables or functions
        match alpha {
            a if a <= 0.01 => 6.635 + 10.0 / (n as f64),
            a if a <= 0.05 => 3.841 + 5.0 / (n as f64),
            a if a <= 0.10 => 2.706 + 3.0 / (n as f64),
            _ => 1.0,
        }
    }
}

/// Wrapper-based feature selection methods
pub struct WrapperMethods;

impl WrapperMethods {
    /// Forward feature selection
    ///
    /// Starts with no features and iteratively adds the best feature.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `target` - Target variable
    /// * `config` - Configuration for selection
    ///
    /// # Returns
    ///
    /// * Feature selection result
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array1, Array2};
    /// use scirs2_series::feature_selection::{WrapperMethods, FeatureSelectionConfig};
    ///
    /// let features = Array2::from_shape_vec((100, 10), (0..1000).map(|x| x as f64).collect()).unwrap();
    /// let target = Array1::from_vec((0..100).map(|x| x as f64).collect());
    /// let config = FeatureSelectionConfig::default();
    ///
    /// let result = WrapperMethods::forward_selection(&features, &target, &config).unwrap();
    /// println!("Selected {} features", result.selected_features.len());
    /// ```
    pub fn forward_selection(
        features: &Array2<f64>,
        target: &Array1<f64>,
        config: &FeatureSelectionConfig,
    ) -> Result<FeatureSelectionResult> {
        let (n_samples, n_features) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        let max_features = config.n_features.unwrap_or(n_features.min(10));
        let mut selected_features = Vec::new();
        let mut remaining_features: HashSet<usize> = (0..n_features).collect();
        let mut feature_scores = Array1::zeros(n_features);
        let mut best_score = f64::NEG_INFINITY;

        for _iteration in 0..max_features.min(config.max_iterations) {
            let mut best_feature = None;
            let mut best_iteration_score = f64::NEG_INFINITY;

            // Try adding each remaining feature
            for &feature_idx in &remaining_features {
                let mut current_features = selected_features.clone();
                current_features.push(feature_idx);

                let score =
                    Self::evaluate_feature_subset(features, target, &current_features, config)?;

                if score > best_iteration_score {
                    best_iteration_score = score;
                    best_feature = Some(feature_idx);
                }
            }

            if let Some(feature_idx) = best_feature {
                // Check if adding this feature improves the score
                if best_iteration_score > best_score {
                    selected_features.push(feature_idx);
                    remaining_features.remove(&feature_idx);
                    feature_scores[feature_idx] = best_iteration_score;
                    best_score = best_iteration_score;
                } else {
                    // No improvement, stop
                    break;
                }
            } else {
                break;
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("final_score".to_string(), best_score);
        metadata.insert("iterations".to_string(), selected_features.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: "ForwardSelection".to_string(),
            metadata,
        })
    }

    /// Backward feature elimination
    ///
    /// Starts with all features and iteratively removes the worst feature.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `target` - Target variable
    /// * `config` - Configuration for selection
    ///
    /// # Returns
    ///
    /// * Feature selection result
    pub fn backward_elimination(
        features: &Array2<f64>,
        target: &Array1<f64>,
        config: &FeatureSelectionConfig,
    ) -> Result<FeatureSelectionResult> {
        let (n_samples, n_features) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        let min_features = config.n_features.unwrap_or(1).max(1);
        let mut selected_features: Vec<usize> = (0..n_features).collect();
        let mut feature_scores = Array1::zeros(n_features);
        let mut best_score =
            Self::evaluate_feature_subset(features, target, &selected_features, config)?;

        while selected_features.len() > min_features {
            let mut worst_feature = None;
            let mut best_iteration_score = f64::NEG_INFINITY;

            // Try removing each feature
            for (i, &_feature_idx) in selected_features.iter().enumerate() {
                let mut current_features = selected_features.clone();
                current_features.remove(i);

                if current_features.is_empty() {
                    continue;
                }

                let score =
                    Self::evaluate_feature_subset(features, target, &current_features, config)?;

                if score > best_iteration_score {
                    best_iteration_score = score;
                    worst_feature = Some(i);
                }
            }

            if let Some(worst_idx) = worst_feature {
                // Check if removing this feature improves or maintains the score
                if best_iteration_score >= best_score * 0.99 {
                    // Allow small degradation
                    let removed_feature = selected_features.remove(worst_idx);
                    feature_scores[removed_feature] = best_score - best_iteration_score;
                    best_score = best_iteration_score;
                } else {
                    // Removing would hurt too much, stop
                    break;
                }
            } else {
                break;
            }
        }

        // Set scores for remaining features
        for &idx in &selected_features {
            feature_scores[idx] = best_score;
        }

        let mut metadata = HashMap::new();
        metadata.insert("final_score".to_string(), best_score);
        metadata.insert(
            "features_removed".to_string(),
            (n_features - selected_features.len()) as f64,
        );

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: "BackwardElimination".to_string(),
            metadata,
        })
    }

    /// Recursive feature elimination
    ///
    /// Recursively eliminates features by fitting a model and removing the least important features.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `target` - Target variable
    /// * `config` - Configuration for selection
    ///
    /// # Returns
    ///
    /// * Feature selection result
    pub fn recursive_feature_elimination(
        features: &Array2<f64>,
        target: &Array1<f64>,
        config: &FeatureSelectionConfig,
    ) -> Result<FeatureSelectionResult> {
        let (n_samples, n_features) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        let target_features = config.n_features.unwrap_or(n_features / 2).max(1);
        let mut selected_features: Vec<usize> = (0..n_features).collect();
        let mut feature_scores = Array1::ones(n_features);

        let mut iteration = 0;
        while selected_features.len() > target_features && iteration < config.max_iterations {
            // Fit a simple linear model to get feature importance
            let importance =
                Self::calculate_feature_importance(features, target, &selected_features)?;

            // Remove the least important features (remove 10% or at least 1)
            let n_to_remove = ((selected_features.len() as f64 * 0.1).ceil() as usize).max(1);
            let n_to_remove = n_to_remove.min(selected_features.len() - target_features);

            if n_to_remove == 0 {
                break;
            }

            // Sort by importance and remove the worst
            let mut indexed_importance: Vec<(usize, f64)> = selected_features
                .iter()
                .map(|&idx| (idx, importance[idx]))
                .collect();

            indexed_importance
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Remove the least important features
            for &(feature_idx, importance) in indexed_importance.iter().take(n_to_remove) {
                feature_scores[feature_idx] = importance;
                if let Some(pos) = selected_features.iter().position(|&x| x == feature_idx) {
                    selected_features.remove(pos);
                }
            }

            iteration += 1;
        }

        // Set final scores for remaining features
        let final_score =
            Self::evaluate_feature_subset(features, target, &selected_features, config)?;
        for &idx in &selected_features {
            feature_scores[idx] = final_score;
        }

        let mut metadata = HashMap::new();
        metadata.insert("final_score".to_string(), final_score);
        metadata.insert("iterations".to_string(), iteration as f64);

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: "RecursiveFeatureElimination".to_string(),
            metadata,
        })
    }

    /// Bidirectional feature selection
    ///
    /// Combines forward selection and backward elimination.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `target` - Target variable
    /// * `config` - Configuration for selection
    ///
    /// # Returns
    ///
    /// * Feature selection result
    pub fn bidirectional_selection(
        features: &Array2<f64>,
        target: &Array1<f64>,
        config: &FeatureSelectionConfig,
    ) -> Result<FeatureSelectionResult> {
        let (n_samples, n_features) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        let max_features = config.n_features.unwrap_or(n_features.min(10));
        let mut selected_features = Vec::new();
        let mut remaining_features: HashSet<usize> = (0..n_features).collect();
        let mut feature_scores = Array1::zeros(n_features);
        let mut best_score = f64::NEG_INFINITY;

        for _iteration in 0..config.max_iterations {
            let mut improved = false;

            // Forward step: try adding a feature
            if selected_features.len() < max_features {
                let mut best_add_feature = None;
                let mut best_add_score = best_score;

                for &feature_idx in &remaining_features {
                    let mut current_features = selected_features.clone();
                    current_features.push(feature_idx);

                    let score =
                        Self::evaluate_feature_subset(features, target, &current_features, config)?;

                    if score > best_add_score {
                        best_add_score = score;
                        best_add_feature = Some(feature_idx);
                    }
                }

                if let Some(feature_idx) = best_add_feature {
                    selected_features.push(feature_idx);
                    remaining_features.remove(&feature_idx);
                    feature_scores[feature_idx] = best_add_score;
                    best_score = best_add_score;
                    improved = true;
                }
            }

            // Backward step: try removing a feature
            if selected_features.len() > 1 {
                let mut best_remove_idx = None;
                let mut best_remove_score = best_score;

                for (i, &_feature_idx) in selected_features.iter().enumerate() {
                    let mut current_features = selected_features.clone();
                    current_features.remove(i);

                    let score =
                        Self::evaluate_feature_subset(features, target, &current_features, config)?;

                    if score > best_remove_score {
                        best_remove_score = score;
                        best_remove_idx = Some(i);
                    }
                }

                if let Some(remove_idx) = best_remove_idx {
                    let removed_feature = selected_features.remove(remove_idx);
                    remaining_features.insert(removed_feature);
                    feature_scores[removed_feature] = best_score - best_remove_score;
                    best_score = best_remove_score;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("final_score".to_string(), best_score);
        metadata.insert("n_selected".to_string(), selected_features.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: "BidirectionalSelection".to_string(),
            metadata,
        })
    }

    // Helper methods

    fn evaluate_feature_subset(
        features: &Array2<f64>,
        target: &Array1<f64>,
        feature_indices: &[usize],
        config: &FeatureSelectionConfig,
    ) -> Result<f64> {
        if feature_indices.is_empty() {
            return Ok(f64::NEG_INFINITY);
        }

        let n_samples = features.nrows();

        // Extract selected features
        let selected_features =
            Array2::from_shape_fn((n_samples, feature_indices.len()), |(i, j)| {
                features[[i, feature_indices[j]]]
            });

        match config.scoring_method {
            ScoringMethod::MeanSquaredError => {
                Self::calculate_mse_score(&selected_features, target)
            }
            ScoringMethod::MeanAbsoluteError => {
                Self::calculate_mae_score(&selected_features, target)
            }
            ScoringMethod::RSquared => Self::calculate_r2_score(&selected_features, target),
            ScoringMethod::AIC => Self::calculate_aic_score(&selected_features, target),
            ScoringMethod::BIC => Self::calculate_bic_score(&selected_features, target),
            ScoringMethod::CrossValidation => {
                Self::calculate_cv_score(&selected_features, target, config.cv_folds)
            }
        }
    }

    fn calculate_mse_score(features: &Array2<f64>, target: &Array1<f64>) -> Result<f64> {
        let predictions = Self::fit_predict_linear(features, target)?;
        let mse = target
            .iter()
            .zip(predictions.iter())
            .map(|(y_true, y_pred)| (y_true - y_pred).powi(2))
            .sum::<f64>()
            / target.len() as f64;

        Ok(-mse) // Negative because we want higher scores to be better
    }

    fn calculate_mae_score(features: &Array2<f64>, target: &Array1<f64>) -> Result<f64> {
        let predictions = Self::fit_predict_linear(features, target)?;
        let mae = target
            .iter()
            .zip(predictions.iter())
            .map(|(y_true, y_pred)| (y_true - y_pred).abs())
            .sum::<f64>()
            / target.len() as f64;

        Ok(-mae) // Negative because we want higher scores to be better
    }

    fn calculate_r2_score(features: &Array2<f64>, target: &Array1<f64>) -> Result<f64> {
        let predictions = Self::fit_predict_linear(features, target)?;
        let y_mean = target.sum() / target.len() as f64;

        let ss_res = target
            .iter()
            .zip(predictions.iter())
            .map(|(y_true, y_pred)| (y_true - y_pred).powi(2))
            .sum::<f64>();

        let ss_tot = target.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>();

        if ss_tot == 0.0 {
            Ok(0.0)
        } else {
            Ok(1.0 - ss_res / ss_tot)
        }
    }

    fn calculate_aic_score(features: &Array2<f64>, target: &Array1<f64>) -> Result<f64> {
        let predictions = Self::fit_predict_linear(features, target)?;
        let n = target.len() as f64;
        let k = features.ncols() as f64 + 1.0; // +1 for intercept

        let mse = target
            .iter()
            .zip(predictions.iter())
            .map(|(y_true, y_pred)| (y_true - y_pred).powi(2))
            .sum::<f64>()
            / n;

        let aic = n * mse.ln() + 2.0 * k;
        Ok(-aic) // Negative because lower AIC is better
    }

    fn calculate_bic_score(features: &Array2<f64>, target: &Array1<f64>) -> Result<f64> {
        let predictions = Self::fit_predict_linear(features, target)?;
        let n = target.len() as f64;
        let k = features.ncols() as f64 + 1.0; // +1 for intercept

        let mse = target
            .iter()
            .zip(predictions.iter())
            .map(|(y_true, y_pred)| (y_true - y_pred).powi(2))
            .sum::<f64>()
            / n;

        let bic = n * mse.ln() + k * n.ln();
        Ok(-bic) // Negative because lower BIC is better
    }

    fn calculate_cv_score(
        features: &Array2<f64>,
        target: &Array1<f64>,
        cv_folds: usize,
    ) -> Result<f64> {
        let n_samples = features.nrows();
        let fold_size = n_samples / cv_folds;
        let mut scores = Vec::new();

        for fold in 0..cv_folds {
            let test_start = fold * fold_size;
            let test_end = if fold == cv_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Split data
            let train_indices: Vec<usize> = (0..test_start).chain(test_end..n_samples).collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();

            if train_indices.is_empty() || test_indices.is_empty() {
                continue;
            }

            // Create training and test sets
            let train_features =
                Array2::from_shape_fn((train_indices.len(), features.ncols()), |(i, j)| {
                    features[[train_indices[i], j]]
                });
            let train_target =
                Array1::from_shape_fn(train_indices.len(), |i| target[train_indices[i]]);
            let test_features =
                Array2::from_shape_fn((test_indices.len(), features.ncols()), |(i, j)| {
                    features[[test_indices[i], j]]
                });
            let test_target =
                Array1::from_shape_fn(test_indices.len(), |i| target[test_indices[i]]);

            // Fit on training and predict on test
            let coefficients = Self::fit_linear_regression(&train_features, &train_target)?;
            let predictions = Self::predict_linear(&test_features, &coefficients);

            // Calculate R²
            let score = Self::calculate_r2_from_predictions(&test_target, &predictions);
            scores.push(score);
        }

        if scores.is_empty() {
            Ok(0.0)
        } else {
            Ok(scores.iter().sum::<f64>() / scores.len() as f64)
        }
    }

    fn fit_predict_linear(features: &Array2<f64>, target: &Array1<f64>) -> Result<Array1<f64>> {
        let coefficients = Self::fit_linear_regression(features, target)?;
        Ok(Self::predict_linear(features, &coefficients))
    }

    fn fit_linear_regression(features: &Array2<f64>, target: &Array1<f64>) -> Result<Array1<f64>> {
        let n_samples = features.nrows();
        let n_features = features.ncols();

        // Add intercept term
        let mut x_matrix = Array2::ones((n_samples, n_features + 1));
        for i in 0..n_samples {
            for j in 0..n_features {
                x_matrix[[i, j + 1]] = features[[i, j]];
            }
        }

        // Normal equation: (X^T X)^(-1) X^T y
        let xt = x_matrix.t();
        let xtx = xt.dot(&x_matrix);
        let xty = xt.dot(target);

        // Simple matrix inversion for small matrices
        let coefficients = Self::solve_linear_system(&xtx, &xty)?;

        Ok(coefficients)
    }

    fn predict_linear(features: &Array2<f64>, coefficients: &Array1<f64>) -> Array1<f64> {
        let n_samples = features.nrows();
        let n_features = features.ncols();

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut pred = coefficients[0]; // Intercept
            for j in 0..n_features {
                pred += coefficients[j + 1] * features[[i, j]];
            }
            predictions[i] = pred;
        }

        predictions
    }

    fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        let n = a.nrows();
        if n != a.ncols() || n != b.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: b.len(),
            });
        }

        // Simple Gaussian elimination with partial pivoting
        let mut aug_matrix = Array2::zeros((n, n + 1));

        // Create augmented matrix
        for i in 0..n {
            for j in 0..n {
                aug_matrix[[i, j]] = a[[i, j]];
            }
            aug_matrix[[i, n]] = b[i];
        }

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug_matrix[[k, i]].abs() > aug_matrix[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..=n {
                    let temp = aug_matrix[[i, j]];
                    aug_matrix[[i, j]] = aug_matrix[[max_row, j]];
                    aug_matrix[[max_row, j]] = temp;
                }
            }

            // Check for zero pivot
            if aug_matrix[[i, i]].abs() < 1e-10 {
                // Use regularization for ill-conditioned matrix
                aug_matrix[[i, i]] += 1e-6;
            }

            // Eliminate
            for k in (i + 1)..n {
                let factor = aug_matrix[[k, i]] / aug_matrix[[i, i]];
                for j in i..=n {
                    aug_matrix[[k, j]] -= factor * aug_matrix[[i, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = aug_matrix[[i, n]];
            for j in (i + 1)..n {
                x[i] -= aug_matrix[[i, j]] * x[j];
            }
            x[i] /= aug_matrix[[i, i]];
        }

        Ok(x)
    }

    fn calculate_r2_from_predictions(target: &Array1<f64>, predictions: &Array1<f64>) -> f64 {
        let y_mean = target.sum() / target.len() as f64;

        let ss_res = target
            .iter()
            .zip(predictions.iter())
            .map(|(y_true, y_pred)| (y_true - y_pred).powi(2))
            .sum::<f64>();

        let ss_tot = target.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>();

        if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }

    fn calculate_feature_importance(
        features: &Array2<f64>,
        target: &Array1<f64>,
        feature_indices: &[usize],
    ) -> Result<Array1<f64>> {
        let n_features = features.ncols();
        let mut importance = Array1::zeros(n_features);

        // Extract selected features
        let selected_features =
            Array2::from_shape_fn((features.nrows(), feature_indices.len()), |(i, j)| {
                features[[i, feature_indices[j]]]
            });

        // Fit linear regression
        let coefficients = Self::fit_linear_regression(&selected_features, target)?;

        // Use absolute coefficients as importance (skip intercept)
        for (i, &feature_idx) in feature_indices.iter().enumerate() {
            importance[feature_idx] = coefficients[i + 1].abs();
        }

        Ok(importance)
    }
}

/// Embedded feature selection methods
pub struct EmbeddedMethods;

impl EmbeddedMethods {
    /// LASSO-based feature selection
    ///
    /// Uses LASSO regularization to select features.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `target` - Target variable
    /// * `alpha` - Regularization parameter
    /// * `max_iterations` - Maximum iterations for optimization
    ///
    /// # Returns
    ///
    /// * Feature selection result
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array1, Array2};
    /// use scirs2_series::feature_selection::EmbeddedMethods;
    ///
    /// let features = Array2::from_shape_vec((100, 10), (0..1000).map(|x| x as f64).collect()).unwrap();
    /// let target = Array1::from_vec((0..100).map(|x| x as f64).collect());
    ///
    /// let result = EmbeddedMethods::lasso_selection(&features, &target, 1.0, 1000).unwrap();
    /// println!("Selected {} features", result.selected_features.len());
    /// ```
    pub fn lasso_selection(
        features: &Array2<f64>,
        target: &Array1<f64>,
        alpha: f64,
        max_iterations: usize,
    ) -> Result<FeatureSelectionResult> {
        let (n_samples, n_features) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        // Normalize features
        let (normalized_features, _feature_means, _feature_stds) =
            Self::normalize_features(features);
        let target_mean = target.sum() / n_samples as f64;
        let normalized_target = target.mapv(|x| x - target_mean);

        // Initialize coefficients
        let mut coefficients = Array1::zeros(n_features);
        let _learning_rate = 0.01;

        // Coordinate descent for LASSO
        for _iteration in 0..max_iterations {
            let mut max_change = 0.0f64;

            for j in 0..n_features {
                let old_coef = coefficients[j];

                // Calculate residual without feature j
                let mut residual = normalized_target.clone();
                for k in 0..n_features {
                    if k != j {
                        for i in 0..n_samples {
                            residual[i] -= coefficients[k] * normalized_features[[i, k]];
                        }
                    }
                }

                // Calculate correlation with residual
                let correlation = normalized_features.column(j).dot(&residual) / n_samples as f64;

                // Soft thresholding
                let new_coef = Self::soft_threshold(correlation, alpha);
                coefficients[j] = new_coef;

                max_change = max_change.max((new_coef - old_coef).abs());
            }

            // Check convergence
            if max_change < 1e-6 {
                break;
            }
        }

        // Select features with non-zero coefficients
        let mut selected_features = Vec::new();
        let mut feature_scores = Array1::zeros(n_features);

        for i in 0..n_features {
            let abs_coef = coefficients[i].abs();
            feature_scores[i] = abs_coef;

            if abs_coef > 1e-6 {
                selected_features.push(i);
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("alpha".to_string(), alpha);
        metadata.insert("n_selected".to_string(), selected_features.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: "LASSO".to_string(),
            metadata,
        })
    }

    /// Ridge regression-based feature ranking
    ///
    /// Uses Ridge regularization to rank features by importance.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `target` - Target variable
    /// * `alpha` - Regularization parameter
    /// * `n_features` - Number of top features to select
    ///
    /// # Returns
    ///
    /// * Feature selection result
    pub fn ridge_selection(
        features: &Array2<f64>,
        target: &Array1<f64>,
        alpha: f64,
        n_features: Option<usize>,
    ) -> Result<FeatureSelectionResult> {
        let (n_samples, n_feat) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        // Normalize features
        let (normalized_features, _, _) = Self::normalize_features(features);
        let target_mean = target.sum() / n_samples as f64;
        let normalized_target = target.mapv(|x| x - target_mean);

        // Ridge regression: (X^T X + αI)^(-1) X^T y
        let xt = normalized_features.t();
        let mut xtx = xt.dot(&normalized_features);

        // Add regularization
        for i in 0..n_feat {
            xtx[[i, i]] += alpha;
        }

        let xty = xt.dot(&normalized_target);
        let coefficients = WrapperMethods::solve_linear_system(&xtx, &xty)?;

        // Rank features by absolute coefficient values
        let mut feature_scores = Array1::zeros(n_feat);
        let mut indexed_scores: Vec<(usize, f64)> = Vec::new();

        for i in 0..n_feat {
            let abs_coef = coefficients[i].abs();
            feature_scores[i] = abs_coef;
            indexed_scores.push((i, abs_coef));
        }

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top features
        let n_to_select = n_features.unwrap_or(n_feat / 2).min(n_feat);
        let selected_features: Vec<usize> = indexed_scores
            .into_iter()
            .take(n_to_select)
            .map(|(idx, _)| idx)
            .collect();

        let mut metadata = HashMap::new();
        metadata.insert("alpha".to_string(), alpha);
        metadata.insert("n_selected".to_string(), selected_features.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: "Ridge".to_string(),
            metadata,
        })
    }

    /// Tree-based feature importance (simplified decision tree)
    ///
    /// Estimates feature importance using a simplified decision tree approach.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `target` - Target variable
    /// * `n_features` - Number of top features to select
    ///
    /// # Returns
    ///
    /// * Feature selection result
    pub fn tree_based_selection(
        features: &Array2<f64>,
        target: &Array1<f64>,
        n_features: Option<usize>,
    ) -> Result<FeatureSelectionResult> {
        let (n_samples, n_feat) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        let mut feature_scores = Array1::zeros(n_feat);

        // Calculate feature importance based on variance reduction
        for i in 0..n_feat {
            let importance = Self::calculate_feature_importance_tree(&features.column(i), target)?;
            feature_scores[i] = importance;
        }

        // Select top features
        let n_to_select = n_features.unwrap_or(n_feat / 2).min(n_feat);
        let mut indexed_scores: Vec<(usize, f64)> = feature_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected_features: Vec<usize> = indexed_scores
            .into_iter()
            .take(n_to_select)
            .map(|(idx, _)| idx)
            .collect();

        let mut metadata = HashMap::new();
        metadata.insert("n_selected".to_string(), selected_features.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: "TreeBased".to_string(),
            metadata,
        })
    }

    // Helper methods

    fn normalize_features(features: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let (n_samples, n_features) = features.dim();
        let mut normalized = Array2::zeros((n_samples, n_features));
        let mut means = Array1::zeros(n_features);
        let mut stds = Array1::zeros(n_features);

        for j in 0..n_features {
            let col = features.column(j);
            let mean = col.sum() / n_samples as f64;
            let variance = col.mapv(|x| (x - mean).powi(2)).sum() / n_samples as f64;
            let std = variance.sqrt().max(1e-8); // Avoid division by zero

            means[j] = mean;
            stds[j] = std;

            for i in 0..n_samples {
                normalized[[i, j]] = (features[[i, j]] - mean) / std;
            }
        }

        (normalized, means, stds)
    }

    fn soft_threshold(x: f64, threshold: f64) -> f64 {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            0.0
        }
    }

    fn calculate_feature_importance_tree(
        feature: &ndarray::ArrayView1<f64>,
        target: &Array1<f64>,
    ) -> Result<f64> {
        let n = feature.len();

        if n < 4 {
            return Ok(0.0);
        }

        // Calculate initial variance
        let target_mean = target.sum() / n as f64;
        let initial_variance = target.mapv(|y| (y - target_mean).powi(2)).sum() / n as f64;

        if initial_variance == 0.0 {
            return Ok(0.0);
        }

        // Try different split points
        let mut feature_values: Vec<f64> = feature.iter().cloned().collect();
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut best_gain = 0.0f64;

        for i in 1..n {
            if feature_values[i] != feature_values[i - 1] {
                let threshold = (feature_values[i] + feature_values[i - 1]) / 2.0;

                let mut left_targets = Vec::new();
                let mut right_targets = Vec::new();

                for j in 0..n {
                    if feature[j] <= threshold {
                        left_targets.push(target[j]);
                    } else {
                        right_targets.push(target[j]);
                    }
                }

                if left_targets.is_empty() || right_targets.is_empty() {
                    continue;
                }

                let left_mean = left_targets.iter().sum::<f64>() / left_targets.len() as f64;
                let right_mean = right_targets.iter().sum::<f64>() / right_targets.len() as f64;

                let left_variance = left_targets
                    .iter()
                    .map(|&y| (y - left_mean).powi(2))
                    .sum::<f64>()
                    / left_targets.len() as f64;

                let right_variance = right_targets
                    .iter()
                    .map(|&y| (y - right_mean).powi(2))
                    .sum::<f64>()
                    / right_targets.len() as f64;

                let weighted_variance = (left_targets.len() as f64 * left_variance
                    + right_targets.len() as f64 * right_variance)
                    / n as f64;

                let gain = initial_variance - weighted_variance;
                best_gain = best_gain.max(gain);
            }
        }

        Ok(best_gain / initial_variance)
    }
}

/// Time series specific feature selection methods
pub struct TimeSeriesMethods;

impl TimeSeriesMethods {
    /// Lag-based feature selection
    ///
    /// Selects features based on their predictive power at different lags.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `target` - Target variable
    /// * `max_lag` - Maximum lag to consider
    /// * `n_features` - Number of top features to select
    ///
    /// # Returns
    ///
    /// * Feature selection result
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array1, Array2};
    /// use scirs2_series::feature_selection::TimeSeriesMethods;
    ///
    /// let features = Array2::from_shape_vec((100, 5), (0..500).map(|x| x as f64).collect()).unwrap();
    /// let target = Array1::from_vec((0..100).map(|x| x as f64).collect());
    ///
    /// let result = TimeSeriesMethods::lag_based_selection(&features, &target, 5, Some(3)).unwrap();
    /// println!("Selected {} features", result.selected_features.len());
    /// ```
    pub fn lag_based_selection(
        features: &Array2<f64>,
        target: &Array1<f64>,
        max_lag: usize,
        n_features: Option<usize>,
    ) -> Result<FeatureSelectionResult> {
        let (n_samples, n_feat) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        if n_samples <= max_lag + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Insufficient samples for lag-based selection".to_string(),
                required: max_lag + 2,
                actual: n_samples,
            });
        }

        let mut feature_scores = Array1::zeros(n_feat);

        for i in 0..n_feat {
            let feature_col = features.column(i).to_owned();
            let mut max_correlation = 0.0f64;

            // Test correlations at different lags
            for lag in 1..=max_lag {
                if n_samples > lag {
                    let lagged_feature = feature_col.slice(ndarray::s![..n_samples - lag]);
                    let future_target = target.slice(ndarray::s![lag..]);

                    let correlation =
                        Self::calculate_correlation_arrays(&lagged_feature, &future_target)?;
                    max_correlation = max_correlation.max(correlation.abs());
                }
            }

            feature_scores[i] = max_correlation;
        }

        // Select top features
        let n_to_select = n_features.unwrap_or(n_feat / 2).min(n_feat);
        let mut indexed_scores: Vec<(usize, f64)> = feature_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected_features: Vec<usize> = indexed_scores
            .into_iter()
            .take(n_to_select)
            .map(|(idx, _)| idx)
            .collect();

        let mut metadata = HashMap::new();
        metadata.insert("max_lag".to_string(), max_lag as f64);
        metadata.insert("n_selected".to_string(), selected_features.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: "LagBased".to_string(),
            metadata,
        })
    }

    /// Seasonal feature importance
    ///
    /// Selects features based on their seasonal patterns.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `seasonal_period` - Seasonal period
    /// * `n_features` - Number of top features to select
    ///
    /// # Returns
    ///
    /// * Feature selection result
    pub fn seasonal_importance_selection(
        features: &Array2<f64>,
        seasonal_period: usize,
        n_features: Option<usize>,
    ) -> Result<FeatureSelectionResult> {
        let (n_samples, n_feat) = features.dim();

        if n_samples < seasonal_period * 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Insufficient samples for seasonal analysis".to_string(),
                required: seasonal_period * 2,
                actual: n_samples,
            });
        }

        let mut feature_scores = Array1::zeros(n_feat);

        for i in 0..n_feat {
            let feature_col = features.column(i).to_owned();
            let seasonal_strength =
                Self::calculate_seasonal_strength(&feature_col, seasonal_period)?;
            feature_scores[i] = seasonal_strength;
        }

        // Select top features
        let n_to_select = n_features.unwrap_or(n_feat / 2).min(n_feat);
        let mut indexed_scores: Vec<(usize, f64)> = feature_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected_features: Vec<usize> = indexed_scores
            .into_iter()
            .take(n_to_select)
            .map(|(idx, _)| idx)
            .collect();

        let mut metadata = HashMap::new();
        metadata.insert("seasonal_period".to_string(), seasonal_period as f64);
        metadata.insert("n_selected".to_string(), selected_features.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: "SeasonalImportance".to_string(),
            metadata,
        })
    }

    /// Cross-correlation based feature selection
    ///
    /// Selects features based on cross-correlation with target variable.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `target` - Target variable
    /// * `max_lag` - Maximum lag for cross-correlation
    /// * `threshold` - Minimum cross-correlation threshold
    ///
    /// # Returns
    ///
    /// * Feature selection result
    pub fn cross_correlation_selection(
        features: &Array2<f64>,
        target: &Array1<f64>,
        max_lag: usize,
        threshold: f64,
    ) -> Result<FeatureSelectionResult> {
        let (n_samples, n_feat) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        if n_samples <= max_lag + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Insufficient samples for cross-correlation".to_string(),
                required: max_lag + 2,
                actual: n_samples,
            });
        }

        let mut selected_features = Vec::new();
        let mut feature_scores = Array1::zeros(n_feat);

        for i in 0..n_feat {
            let feature_col = features.column(i).to_owned();

            // Calculate cross-correlation
            let ccf = cross_correlation(&feature_col, target, Some(max_lag))?;

            // Find maximum absolute cross-correlation
            let max_ccf = ccf.iter().map(|&x| x.abs()).fold(0.0_f64, |a, b| a.max(b));

            feature_scores[i] = max_ccf;

            if max_ccf >= threshold {
                selected_features.push(i);
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("max_lag".to_string(), max_lag as f64);
        metadata.insert("threshold".to_string(), threshold);
        metadata.insert("n_selected".to_string(), selected_features.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: "CrossCorrelation".to_string(),
            metadata,
        })
    }

    /// Granger causality-based feature selection
    ///
    /// Selects features based on Granger causality test.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `target` - Target variable
    /// * `max_lag` - Maximum lag for Granger test
    /// * `alpha` - Significance level
    ///
    /// # Returns
    ///
    /// * Feature selection result
    pub fn granger_causality_selection(
        features: &Array2<f64>,
        target: &Array1<f64>,
        max_lag: usize,
        alpha: f64,
    ) -> Result<FeatureSelectionResult> {
        let (n_samples, n_feat) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        if n_samples <= max_lag * 2 + 5 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Insufficient samples for Granger causality test".to_string(),
                required: max_lag * 2 + 6,
                actual: n_samples,
            });
        }

        let mut selected_features = Vec::new();
        let mut feature_scores = Array1::zeros(n_feat);

        for i in 0..n_feat {
            let feature_col = features.column(i).to_owned();

            let (f_stat, p_value) = Self::granger_causality_test(&feature_col, target, max_lag)?;

            feature_scores[i] = f_stat;

            if p_value < alpha {
                selected_features.push(i);
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("max_lag".to_string(), max_lag as f64);
        metadata.insert("alpha".to_string(), alpha);
        metadata.insert("n_selected".to_string(), selected_features.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores,
            method: "GrangerCausality".to_string(),
            metadata,
        })
    }

    // Helper methods

    fn calculate_correlation_arrays(
        x: &ndarray::ArrayView1<f64>,
        y: &ndarray::ArrayView1<f64>,
    ) -> Result<f64> {
        let n = x.len().min(y.len());

        if n < 2 {
            return Ok(0.0);
        }

        let x_mean = x.iter().take(n).sum::<f64>() / n as f64;
        let y_mean = y.iter().take(n).sum::<f64>() / n as f64;

        let mut numerator = 0.0;
        let mut x_var = 0.0;
        let mut y_var = 0.0;

        for i in 0..n {
            let x_dev = x[i] - x_mean;
            let y_dev = y[i] - y_mean;

            numerator += x_dev * y_dev;
            x_var += x_dev * x_dev;
            y_var += y_dev * y_dev;
        }

        let denominator = (x_var * y_var).sqrt();

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    fn calculate_seasonal_strength(ts: &Array1<f64>, period: usize) -> Result<f64> {
        let n = ts.len();

        if n < period * 2 {
            return Ok(0.0);
        }

        // Calculate seasonal differences
        let mut seasonal_diff = Vec::new();
        for i in period..n {
            seasonal_diff.push(ts[i] - ts[i - period]);
        }

        if seasonal_diff.is_empty() {
            return Ok(0.0);
        }

        // Calculate variance of original series
        let mean = ts.sum() / n as f64;
        let var_original = ts.mapv(|x| (x - mean).powi(2)).sum() / n as f64;

        // Calculate variance of seasonal differences
        let diff_mean = seasonal_diff.iter().sum::<f64>() / seasonal_diff.len() as f64;
        let var_seasonal = seasonal_diff
            .iter()
            .map(|&x| (x - diff_mean).powi(2))
            .sum::<f64>()
            / seasonal_diff.len() as f64;

        if var_original == 0.0 {
            Ok(0.0)
        } else {
            // Seasonal strength: 1 - var(seasonal_diff) / var(original)
            let strength = 1.0 - var_seasonal / var_original;
            Ok(strength.clamp(0.0, 1.0))
        }
    }

    fn granger_causality_test(
        x: &Array1<f64>,
        y: &Array1<f64>,
        max_lag: usize,
    ) -> Result<(f64, f64)> {
        let n = x.len();

        if n <= max_lag * 2 + 1 {
            return Ok((0.0, 1.0));
        }

        // Prepare lagged data
        let effective_n = n - max_lag;
        let mut y_lagged = Array2::zeros((effective_n, max_lag));
        let mut x_lagged = Array2::zeros((effective_n, max_lag));
        let mut y_current = Array1::zeros(effective_n);

        for i in 0..effective_n {
            y_current[i] = y[i + max_lag];

            for lag in 0..max_lag {
                y_lagged[[i, lag]] = y[i + max_lag - lag - 1];
                x_lagged[[i, lag]] = x[i + max_lag - lag - 1];
            }
        }

        // Restricted model: y_t = α + Σ β_i * y_{t-i} + ε_t
        let rss_restricted = Self::fit_autoregressive_model(&y_lagged, &y_current)?;

        // Unrestricted model: y_t = α + Σ β_i * y_{t-i} + Σ γ_i * x_{t-i} + ε_t
        let mut combined_features = Array2::zeros((effective_n, max_lag * 2));
        for i in 0..effective_n {
            for j in 0..max_lag {
                combined_features[[i, j]] = y_lagged[[i, j]];
                combined_features[[i, j + max_lag]] = x_lagged[[i, j]];
            }
        }

        let rss_unrestricted = Self::fit_autoregressive_model(&combined_features, &y_current)?;

        // F-statistic
        let df1 = max_lag as f64;
        let df2 = (effective_n - max_lag * 2 - 1) as f64;

        if df2 <= 0.0 || rss_unrestricted <= 0.0 {
            return Ok((0.0, 1.0));
        }

        let f_stat = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2);

        // Approximate p-value (simplified)
        let p_value = if f_stat > 3.0 {
            0.01
        } else if f_stat > 2.0 {
            0.05
        } else {
            0.1
        };

        Ok((f_stat.max(0.0), p_value))
    }

    fn fit_autoregressive_model(features: &Array2<f64>, target: &Array1<f64>) -> Result<f64> {
        let predictions = WrapperMethods::fit_predict_linear(features, target)?;

        let rss = target
            .iter()
            .zip(predictions.iter())
            .map(|(y_true, y_pred)| (y_true - y_pred).powi(2))
            .sum::<f64>();

        Ok(rss)
    }
}

/// High-level feature selection interface
pub struct FeatureSelector;

impl FeatureSelector {
    /// Automatic feature selection using multiple methods
    ///
    /// Combines different feature selection methods for robust selection.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature matrix (samples x features)
    /// * `target` - Target variable (optional for unsupervised methods)
    /// * `config` - Configuration for selection
    ///
    /// # Returns
    ///
    /// * Feature selection result
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array1, Array2};
    /// use scirs2_series::feature_selection::{FeatureSelector, FeatureSelectionConfig};
    ///
    /// let features = Array2::from_shape_vec((100, 10), (0..1000).map(|x| x as f64).collect()).unwrap();
    /// let target = Array1::from_vec((0..100).map(|x| x as f64).collect());
    /// let config = FeatureSelectionConfig::default();
    ///
    /// let result = FeatureSelector::auto_select(&features, Some(&target), &config).unwrap();
    /// println!("Selected {} features using automatic selection", result.selected_features.len());
    /// ```
    pub fn auto_select(
        features: &Array2<f64>,
        target: Option<&Array1<f64>>,
        config: &FeatureSelectionConfig,
    ) -> Result<FeatureSelectionResult> {
        let (_, n_features) = features.dim();

        // Start with variance filtering
        let mut results = Vec::new();

        // 1. Variance threshold
        let variance_result =
            FilterMethods::variance_threshold(features, config.variance_threshold)?;
        results.push(variance_result);

        if let Some(target) = target {
            // 2. Correlation-based selection
            let correlation_result = FilterMethods::correlation_selection(
                features,
                target,
                config.correlation_threshold,
            )?;
            results.push(correlation_result);

            // 3. Mutual information
            let mi_result = FilterMethods::mutual_information_selection(
                features,
                target,
                10,
                config.n_features,
            )?;
            results.push(mi_result);

            // 4. Forward selection (if not too many features)
            if n_features <= 50 {
                let forward_result = WrapperMethods::forward_selection(features, target, config)?;
                results.push(forward_result);
            }

            // 5. LASSO selection
            let lasso_result = EmbeddedMethods::lasso_selection(
                features,
                target,
                config.regularization_alpha,
                1000,
            )?;
            results.push(lasso_result);

            // 6. Cross-correlation for time series
            let ccf_result = TimeSeriesMethods::cross_correlation_selection(
                features,
                target,
                config.max_lag,
                0.1,
            )?;
            results.push(ccf_result);
        }

        // Combine results using voting
        let combined_result = Self::combine_results(&results, config.n_features)?;

        Ok(combined_result)
    }

    /// Combine multiple feature selection results using voting
    fn combine_results(
        results: &[FeatureSelectionResult],
        n_features: Option<usize>,
    ) -> Result<FeatureSelectionResult> {
        if results.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "No feature selection results to combine".to_string(),
            ));
        }

        let n_feat = results[0].feature_scores.len();
        let mut vote_counts = Array1::<f64>::zeros(n_feat);
        let mut combined_scores = Array1::<f64>::zeros(n_feat);

        // Count votes and combine scores
        for result in results {
            for &feature_idx in &result.selected_features {
                vote_counts[feature_idx] += 1.0;
            }

            // Normalize and add scores
            let max_score = result.feature_scores.iter().fold(0.0_f64, |a, &b| a.max(b));
            if max_score > 0.0 {
                for i in 0..n_feat {
                    combined_scores[i] += result.feature_scores[i] / max_score;
                }
            }
        }

        // Combine vote counts and scores
        for i in 0..n_feat {
            combined_scores[i] = vote_counts[i] + combined_scores[i] / results.len() as f64;
        }

        // Select features based on combined scores
        let target_features = n_features.unwrap_or(n_feat / 3).min(n_feat);
        let mut indexed_scores: Vec<(usize, f64)> = combined_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected_features: Vec<usize> = indexed_scores
            .into_iter()
            .take(target_features)
            .map(|(idx, _)| idx)
            .collect();

        let mut metadata = HashMap::new();
        metadata.insert("n_methods".to_string(), results.len() as f64);
        metadata.insert("n_selected".to_string(), selected_features.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features,
            feature_scores: combined_scores,
            method: "AutoSelect".to_string(),
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn create_test_data() -> (Array2<f64>, Array1<f64>) {
        // Create synthetic data with known relationships
        let n_samples = 100;
        let n_features = 10;

        let mut features = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let t = i as f64;

            // Feature 0: strongly correlated with target
            features[[i, 0]] = t + (i as f64 * 0.1).sin();

            // Feature 1: weakly correlated with target
            features[[i, 1]] = t * 0.3 + (i as f64 * 0.2).cos();

            // Feature 2: noise
            features[[i, 2]] = (i as f64 * 0.05).sin() * 10.0;

            // Feature 3: constant (low variance)
            features[[i, 3]] = 5.0;

            // Features 4-9: random noise
            for j in 4..n_features {
                features[[i, j]] = (i * j) as f64 * 0.01;
            }

            // Target is mainly based on features 0 and 1
            target[i] = features[[i, 0]] * 2.0 + features[[i, 1]] * 0.5 + (i as f64 * 0.03).sin();
        }

        (features, target)
    }

    #[test]
    fn test_variance_threshold() {
        let (features, _) = create_test_data();

        let result = FilterMethods::variance_threshold(&features, 0.1).unwrap();

        // Feature 3 (constant) should be filtered out
        assert!(!result.selected_features.contains(&3));
        assert!(result.selected_features.len() < features.ncols());
    }

    #[test]
    fn test_correlation_selection() {
        let (features, target) = create_test_data();

        let result = FilterMethods::correlation_selection(&features, &target, 0.3).unwrap();

        // Should select features 0 and 1 which are correlated with target
        assert!(result.selected_features.contains(&0));
        assert!(result.selected_features.contains(&1));
        assert!(result.feature_scores[0] > result.feature_scores[2]);
    }

    #[test]
    fn test_mutual_information_selection() {
        let (features, target) = create_test_data();

        let result =
            FilterMethods::mutual_information_selection(&features, &target, 5, Some(5)).unwrap();

        assert_eq!(result.selected_features.len(), 5);
        assert!(!result.feature_scores.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_f_test_selection() {
        let (features, target) = create_test_data();

        let result = FilterMethods::f_test_selection(&features, &target, 0.05).unwrap();

        // Should select some features based on F-test
        assert!(!result.selected_features.is_empty());
        assert!(result.feature_scores[0] > 0.0);
    }

    #[test]
    fn test_autocorrelation_filter() {
        let (features, _) = create_test_data();

        let result = FilterMethods::autocorrelation_filter(&features, 5, 0.1).unwrap();

        assert!(!result.selected_features.is_empty());
        assert!(result.feature_scores.len() == features.ncols());
    }

    #[test]
    fn test_forward_selection() {
        let (features, target) = create_test_data();
        let config = FeatureSelectionConfig {
            n_features: Some(3),
            ..Default::default()
        };

        let result = WrapperMethods::forward_selection(&features, &target, &config).unwrap();

        assert!(result.selected_features.len() <= 3);
        // Should prefer features 0 and 1
        assert!(result.selected_features.contains(&0) || result.selected_features.contains(&1));
    }

    #[test]
    fn test_backward_elimination() {
        let (features, target) = create_test_data();
        let config = FeatureSelectionConfig {
            n_features: Some(5),
            ..Default::default()
        };

        let result = WrapperMethods::backward_elimination(&features, &target, &config).unwrap();

        assert!(result.selected_features.len() >= 5);
    }

    #[test]
    fn test_recursive_feature_elimination() {
        let (features, target) = create_test_data();
        let config = FeatureSelectionConfig {
            n_features: Some(4),
            ..Default::default()
        };

        let result =
            WrapperMethods::recursive_feature_elimination(&features, &target, &config).unwrap();

        assert_eq!(result.selected_features.len(), 4);
    }

    #[test]
    fn test_bidirectional_selection() {
        let (features, target) = create_test_data();
        let config = FeatureSelectionConfig {
            n_features: Some(5),
            max_iterations: 10,
            ..Default::default()
        };

        let result = WrapperMethods::bidirectional_selection(&features, &target, &config).unwrap();

        assert!(result.selected_features.len() <= 5);
    }

    #[test]
    fn test_lasso_selection() {
        let (features, target) = create_test_data();

        let result = EmbeddedMethods::lasso_selection(&features, &target, 0.1, 100).unwrap();

        // LASSO should select some features and zero out others
        assert!(!result.selected_features.is_empty());
        assert!(result.selected_features.len() < features.ncols());
    }

    #[test]
    fn test_ridge_selection() {
        let (features, target) = create_test_data();

        let result = EmbeddedMethods::ridge_selection(&features, &target, 1.0, Some(5)).unwrap();

        assert_eq!(result.selected_features.len(), 5);
    }

    #[test]
    fn test_tree_based_selection() {
        let (features, target) = create_test_data();

        let result = EmbeddedMethods::tree_based_selection(&features, &target, Some(4)).unwrap();

        assert_eq!(result.selected_features.len(), 4);
    }

    #[test]
    fn test_lag_based_selection() {
        let (features, target) = create_test_data();

        let result =
            TimeSeriesMethods::lag_based_selection(&features, &target, 3, Some(5)).unwrap();

        assert_eq!(result.selected_features.len(), 5);
    }

    #[test]
    fn test_seasonal_importance_selection() {
        let (features, _) = create_test_data();

        let result =
            TimeSeriesMethods::seasonal_importance_selection(&features, 12, Some(4)).unwrap();

        assert_eq!(result.selected_features.len(), 4);
    }

    #[test]
    fn test_cross_correlation_selection() {
        let (features, target) = create_test_data();

        let result =
            TimeSeriesMethods::cross_correlation_selection(&features, &target, 5, 0.1).unwrap();

        assert!(!result.selected_features.is_empty());
    }

    #[test]
    fn test_granger_causality_selection() {
        let (features, target) = create_test_data();

        let result =
            TimeSeriesMethods::granger_causality_selection(&features, &target, 3, 0.05).unwrap();

        // May or may not select features depending on causality
        assert!(result.feature_scores.len() == features.ncols());
    }

    #[test]
    fn test_auto_select() {
        let (features, target) = create_test_data();
        let config = FeatureSelectionConfig {
            n_features: Some(5),
            ..Default::default()
        };

        let result = FeatureSelector::auto_select(&features, Some(&target), &config).unwrap();

        assert!(result.selected_features.len() <= 5);
        assert_eq!(result.method, "AutoSelect");
    }

    #[test]
    fn test_edge_cases() {
        // Test with insufficient data
        let small_features = Array2::zeros((2, 5));
        let small_target = Array1::zeros(2);

        let result = FilterMethods::correlation_selection(&small_features, &small_target, 0.1);
        assert!(result.is_err());

        // Test with dimension mismatch
        let features = Array2::zeros((10, 5));
        let mismatched_target = Array1::zeros(5);

        let result = FilterMethods::correlation_selection(&features, &mismatched_target, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_selection_result() {
        let result = FeatureSelectionResult {
            selected_features: vec![0, 2, 4],
            feature_scores: array![0.8, 0.2, 0.9, 0.1, 0.7],
            method: "Test".to_string(),
            metadata: HashMap::new(),
        };

        assert_eq!(result.selected_features.len(), 3);
        assert_eq!(result.method, "Test");
        assert_eq!(result.feature_scores.len(), 5);
    }
}
