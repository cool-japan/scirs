//! Wrapper-based feature selection methods

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};

use super::{FeatureSelectionConfig, FeatureSelectionResult, ScoringMethod};
use crate::error::{Result, TimeSeriesError};

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
    /// use scirs2_core::ndarray::{Array1, Array2};
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
        let (n_samples, nfeatures) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        let maxfeatures = config.n_features.unwrap_or(nfeatures.min(10));
        let mut selectedfeatures = Vec::new();
        let mut remainingfeatures: HashSet<usize> = (0..nfeatures).collect();
        let mut feature_scores = Array1::zeros(nfeatures);
        let mut best_score = f64::NEG_INFINITY;

        for _iteration in 0..maxfeatures.min(config.max_iterations) {
            let mut best_feature = None;
            let mut best_iteration_score = f64::NEG_INFINITY;

            // Try adding each remaining feature
            for &feature_idx in &remainingfeatures {
                let mut currentfeatures = selectedfeatures.clone();
                currentfeatures.push(feature_idx);

                let score =
                    Self::evaluate_feature_subset(features, target, &currentfeatures, config)?;

                if score > best_iteration_score {
                    best_iteration_score = score;
                    best_feature = Some(feature_idx);
                }
            }

            if let Some(feature_idx) = best_feature {
                // Check if adding this feature improves the score
                if best_iteration_score > best_score {
                    selectedfeatures.push(feature_idx);
                    remainingfeatures.remove(&feature_idx);
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
        metadata.insert("iterations".to_string(), selectedfeatures.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features: selectedfeatures,
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
        let (n_samples, nfeatures) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        let minfeatures = config.n_features.unwrap_or(1).max(1);
        let mut selectedfeatures: Vec<usize> = (0..nfeatures).collect();
        let mut feature_scores = Array1::zeros(nfeatures);
        let mut best_score =
            Self::evaluate_feature_subset(features, target, &selectedfeatures, config)?;

        while selectedfeatures.len() > minfeatures {
            let mut worst_feature = None;
            let mut best_iteration_score = f64::NEG_INFINITY;

            // Try removing each feature
            for (i, &_feature_idx) in selectedfeatures.iter().enumerate() {
                let mut currentfeatures = selectedfeatures.clone();
                currentfeatures.remove(i);

                if currentfeatures.is_empty() {
                    continue;
                }

                let score =
                    Self::evaluate_feature_subset(features, target, &currentfeatures, config)?;

                if score > best_iteration_score {
                    best_iteration_score = score;
                    worst_feature = Some(i);
                }
            }

            if let Some(worst_idx) = worst_feature {
                // Check if removing this feature improves or maintains the score
                if best_iteration_score >= best_score * 0.99 {
                    // Allow small degradation
                    let removed_feature = selectedfeatures.remove(worst_idx);
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
        for &idx in &selectedfeatures {
            feature_scores[idx] = best_score;
        }

        let mut metadata = HashMap::new();
        metadata.insert("final_score".to_string(), best_score);
        metadata.insert(
            "features_removed".to_string(),
            (nfeatures - selectedfeatures.len()) as f64,
        );

        Ok(FeatureSelectionResult {
            selected_features: selectedfeatures,
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
        let (n_samples, nfeatures) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        let targetfeatures = config.n_features.unwrap_or(nfeatures / 2).max(1);
        let mut selectedfeatures: Vec<usize> = (0..nfeatures).collect();
        let mut feature_scores = Array1::ones(nfeatures);

        let mut iteration = 0;
        while selectedfeatures.len() > targetfeatures && iteration < config.max_iterations {
            // Fit a simple linear model to get feature importance
            let importance =
                Self::calculate_feature_importance(features, target, &selectedfeatures)?;

            // Remove the least important features (remove 10% or at least 1)
            let n_to_remove = ((selectedfeatures.len() as f64 * 0.1).ceil() as usize).max(1);
            let n_to_remove = n_to_remove.min(selectedfeatures.len() - targetfeatures);

            if n_to_remove == 0 {
                break;
            }

            // Sort by importance and remove the worst
            let mut indexed_importance: Vec<(usize, f64)> = selectedfeatures
                .iter()
                .map(|&idx| (idx, importance[idx]))
                .collect();

            indexed_importance
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Remove the least important features
            for &(feature_idx, importance) in indexed_importance.iter().take(n_to_remove) {
                feature_scores[feature_idx] = importance;
                if let Some(pos) = selectedfeatures.iter().position(|&x| x == feature_idx) {
                    selectedfeatures.remove(pos);
                }
            }

            iteration += 1;
        }

        // Set final scores for remaining features
        let final_score =
            Self::evaluate_feature_subset(features, target, &selectedfeatures, config)?;
        for &idx in &selectedfeatures {
            feature_scores[idx] = final_score;
        }

        let mut metadata = HashMap::new();
        metadata.insert("final_score".to_string(), final_score);
        metadata.insert("iterations".to_string(), iteration as f64);

        Ok(FeatureSelectionResult {
            selected_features: selectedfeatures,
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
        let (n_samples, nfeatures) = features.dim();

        if n_samples != target.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n_samples,
                actual: target.len(),
            });
        }

        let maxfeatures = config.n_features.unwrap_or(nfeatures.min(10));
        let mut selectedfeatures = Vec::new();
        let mut remainingfeatures: HashSet<usize> = (0..nfeatures).collect();
        let mut feature_scores = Array1::zeros(nfeatures);
        let mut best_score = f64::NEG_INFINITY;

        for _iteration in 0..config.max_iterations {
            let mut improved = false;

            // Forward step: try adding a feature
            if selectedfeatures.len() < maxfeatures {
                let mut best_add_feature = None;
                let mut best_add_score = best_score;

                for &feature_idx in &remainingfeatures {
                    let mut currentfeatures = selectedfeatures.clone();
                    currentfeatures.push(feature_idx);

                    let score =
                        Self::evaluate_feature_subset(features, target, &currentfeatures, config)?;

                    if score > best_add_score {
                        best_add_score = score;
                        best_add_feature = Some(feature_idx);
                    }
                }

                if let Some(feature_idx) = best_add_feature {
                    selectedfeatures.push(feature_idx);
                    remainingfeatures.remove(&feature_idx);
                    feature_scores[feature_idx] = best_add_score;
                    best_score = best_add_score;
                    improved = true;
                }
            }

            // Backward step: try removing a feature
            if selectedfeatures.len() > 1 {
                let mut best_remove_idx = None;
                let mut best_remove_score = best_score;

                for (i, &_feature_idx) in selectedfeatures.iter().enumerate() {
                    let mut currentfeatures = selectedfeatures.clone();
                    currentfeatures.remove(i);

                    let score =
                        Self::evaluate_feature_subset(features, target, &currentfeatures, config)?;

                    if score > best_remove_score {
                        best_remove_score = score;
                        best_remove_idx = Some(i);
                    }
                }

                if let Some(remove_idx) = best_remove_idx {
                    let removed_feature = selectedfeatures.remove(remove_idx);
                    remainingfeatures.insert(removed_feature);
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
        metadata.insert("n_selected".to_string(), selectedfeatures.len() as f64);

        Ok(FeatureSelectionResult {
            selected_features: selectedfeatures,
            feature_scores,
            method: "BidirectionalSelection".to_string(),
            metadata,
        })
    }

    // Helper methods

    /// Evaluate the performance of a feature subset using the configured scoring method
    pub fn evaluate_feature_subset(
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
        let selectedfeatures =
            Array2::from_shape_fn((n_samples, feature_indices.len()), |(i, j)| {
                features[[i, feature_indices[j]]]
            });

        match config.scoring_method {
            ScoringMethod::MeanSquaredError => Self::calculate_mse_score(&selectedfeatures, target),
            ScoringMethod::MeanAbsoluteError => {
                Self::calculate_mae_score(&selectedfeatures, target)
            }
            ScoringMethod::RSquared => Self::calculate_r2_score(&selectedfeatures, target),
            ScoringMethod::AIC => Self::calculate_aic_score(&selectedfeatures, target),
            ScoringMethod::BIC => Self::calculate_bic_score(&selectedfeatures, target),
            ScoringMethod::CrossValidation => {
                Self::calculate_cv_score(&selectedfeatures, target, config.cv_folds)
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
            let trainfeatures =
                Array2::from_shape_fn((train_indices.len(), features.ncols()), |(i, j)| {
                    features[[train_indices[i], j]]
                });
            let traintarget =
                Array1::from_shape_fn(train_indices.len(), |i| target[train_indices[i]]);
            let testfeatures =
                Array2::from_shape_fn((test_indices.len(), features.ncols()), |(i, j)| {
                    features[[test_indices[i], j]]
                });
            let testtarget = Array1::from_shape_fn(test_indices.len(), |i| target[test_indices[i]]);

            // Fit on training and predict on test
            let coefficients = Self::fit_linear_regression(&trainfeatures, &traintarget)?;
            let predictions = Self::predict_linear(&testfeatures, &coefficients);

            // Calculate R²
            let score = Self::calculate_r2_from_predictions(&testtarget, &predictions);
            scores.push(score);
        }

        if scores.is_empty() {
            Ok(0.0)
        } else {
            Ok(scores.iter().sum::<f64>() / scores.len() as f64)
        }
    }

    /// Fit a linear regression model and return predictions
    pub fn fit_predict_linear(features: &Array2<f64>, target: &Array1<f64>) -> Result<Array1<f64>> {
        let coefficients = Self::fit_linear_regression(features, target)?;
        Ok(Self::predict_linear(features, &coefficients))
    }

    /// Fit a linear regression model and return coefficients
    pub fn fit_linear_regression(
        features: &Array2<f64>,
        target: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let n_samples = features.nrows();
        let nfeatures = features.ncols();

        // Add intercept term
        let mut x_matrix = Array2::ones((n_samples, nfeatures + 1));
        for i in 0..n_samples {
            for j in 0..nfeatures {
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
        let nfeatures = features.ncols();

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut pred = coefficients[0]; // Intercept
            for j in 0..nfeatures {
                pred += coefficients[j + 1] * features[[i, j]];
            }
            predictions[i] = pred;
        }

        predictions
    }

    /// Solve a linear system using Gaussian elimination with partial pivoting
    pub fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
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
        let nfeatures = features.ncols();
        let mut importance = Array1::zeros(nfeatures);

        // Extract selected features
        let selectedfeatures =
            Array2::from_shape_fn((features.nrows(), feature_indices.len()), |(i, j)| {
                features[[i, feature_indices[j]]]
            });

        // Fit linear regression
        let coefficients = Self::fit_linear_regression(&selectedfeatures, target)?;

        // Use absolute coefficients as importance (skip intercept)
        for (i, &feature_idx) in feature_indices.iter().enumerate() {
            importance[feature_idx] = coefficients[i + 1].abs();
        }

        Ok(importance)
    }
}
