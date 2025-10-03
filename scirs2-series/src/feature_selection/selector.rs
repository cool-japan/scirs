//! High-level feature selection interface

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

use super::embedded::EmbeddedMethods;
use super::filter::FilterMethods;
use super::time_series::TimeSeriesMethods;
use super::wrapper::WrapperMethods;
use super::{FeatureSelectionConfig, FeatureSelectionResult};
use crate::error::{Result, TimeSeriesError};

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
    /// use scirs2_core::ndarray::{Array1, Array2};
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

        // Select _features based on combined scores
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
            .map(|(idx_, _)| idx_)
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
