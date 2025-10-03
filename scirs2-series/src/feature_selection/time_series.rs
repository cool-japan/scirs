//! Time series specific feature selection methods

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

use super::wrapper::WrapperMethods;
use super::FeatureSelectionResult;
use crate::error::{Result, TimeSeriesError};
use crate::utils::cross_correlation;

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
    /// use scirs2_core::ndarray::{Array1, Array2};
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
                    let lagged_feature =
                        feature_col.slice(scirs2_core::ndarray::s![..n_samples - lag]);
                    let future_target = target.slice(scirs2_core::ndarray::s![lag..]);

                    let correlation =
                        Self::calculate_correlation_arrays(&lagged_feature, &future_target)?;
                    max_correlation = max_correlation.max(correlation.abs());
                }
            }

            feature_scores[i] = max_correlation;
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
        x: &scirs2_core::ndarray::ArrayView1<f64>,
        y: &scirs2_core::ndarray::ArrayView1<f64>,
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
