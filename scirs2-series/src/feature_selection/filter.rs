//! Filter-based feature selection methods

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

use super::FeatureSelectionResult;
use crate::error::{Result, TimeSeriesError};
use crate::utils::autocorrelation;

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
    /// use scirs2_core::ndarray::Array2;
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
                    acf.slice(scirs2_core::ndarray::s![1..])
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
        x: &scirs2_core::ndarray::ArrayView1<f64>,
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
        x: &scirs2_core::ndarray::ArrayView1<f64>,
        y: &Array1<f64>,
        n_bins: usize,
    ) -> Result<f64> {
        let n = x.len();

        // Create _bins
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

        for (i_, _) in x_counts.iter().enumerate().take(n_bins) {
            for (j_, _) in y_counts.iter().enumerate().take(n_bins) {
                if joint_counts[i_][j_] > 0 && x_counts[i_] > 0 && y_counts[j_] > 0 {
                    let p_xy = joint_counts[i_][j_] as f64 / n as f64;
                    let p_x = x_counts[i_] as f64 / n as f64;
                    let p_y = y_counts[j_] as f64 / n as f64;

                    mi += p_xy * (p_xy / (p_x * p_y)).ln();
                }
            }
        }

        Ok(mi)
    }

    fn calculate_f_statistic(
        x: &scirs2_core::ndarray::ArrayView1<f64>,
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
