//! Embedded feature selection methods

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

use super::wrapper::WrapperMethods;
use super::FeatureSelectionResult;
use crate::error::{Result, TimeSeriesError};

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
    /// use scirs2_core::ndarray::{Array1, Array2};
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

        // Rank _features by absolute coefficient values
        let mut feature_scores = Array1::zeros(n_feat);
        let mut indexed_scores: Vec<(usize, f64)> = Vec::new();

        for i in 0..n_feat {
            let abs_coef = coefficients[i].abs();
            feature_scores[i] = abs_coef;
            indexed_scores.push((i, abs_coef));
        }

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top _features
        let n_to_select = n_features.unwrap_or(n_feat / 2).min(n_feat);
        let selected_features: Vec<usize> = indexed_scores
            .into_iter()
            .take(n_to_select)
            .map(|(idx_, _)| idx_)
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

        // Select top _features
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
            .map(|(idx_, _)| idx_)
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
        feature: &scirs2_core::ndarray::ArrayView1<f64>,
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
