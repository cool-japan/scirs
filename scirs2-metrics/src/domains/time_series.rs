//! Time series domain metrics
//!
//! This module provides specialized metric collections for time series analysis
//! including forecasting, anomaly detection, and trend analysis.

use crate::domains::{DomainEvaluationResult, DomainMetrics};
use crate::error::{MetricsError, Result};
use crate::regression::{mean_absolute_error, mean_squared_error, r2_score};
use ndarray::{s, Array1, Array2};
use std::collections::HashMap;

/// Forecasting evaluation results
#[derive(Debug, Clone)]
pub struct ForecastingResults {
    /// Mean Absolute Error
    pub mae: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Percentage Error
    pub mape: f64,
    /// Symmetric Mean Absolute Percentage Error
    pub smape: f64,
    /// Mean Absolute Scaled Error
    pub mase: f64,
    /// Directional Accuracy
    pub directional_accuracy: f64,
    /// Forecast bias (mean error)
    pub forecast_bias: f64,
    /// Theil's U statistic
    pub theil_u: f64,
}

/// Anomaly detection evaluation results for time series
#[derive(Debug, Clone)]
pub struct TimeSeriesAnomalyResults {
    /// Precision for anomaly detection
    pub precision: f64,
    /// Recall for anomaly detection
    pub recall: f64,
    /// F1 score for anomaly detection
    pub f1_score: f64,
    /// Area Under ROC Curve
    pub auc: f64,
    /// False alarm rate
    pub false_alarm_rate: f64,
    /// Miss detection rate
    pub miss_detection_rate: f64,
    /// Point-adjust precision (accounting for delay tolerance)
    pub point_adjust_precision: f64,
    /// Point-adjust recall (accounting for delay tolerance)
    pub point_adjust_recall: f64,
}

/// Trend analysis evaluation results
#[derive(Debug, Clone)]
pub struct TrendAnalysisResults {
    /// Trend strength coefficient
    pub trend_strength: f64,
    /// Seasonality strength coefficient
    pub seasonality_strength: f64,
    /// Autocorrelation at lag 1
    pub autocorr_lag1: f64,
    /// Ljung-Box test p-value for residuals
    pub ljung_box_pvalue: f64,
    /// Augmented Dickey-Fuller test statistic
    pub adf_statistic: f64,
    /// Stationarity indicator
    pub is_stationary: bool,
}

/// Change point detection evaluation results
#[derive(Debug, Clone)]
pub struct ChangePointResults {
    /// Precision for change point detection
    pub precision: f64,
    /// Recall for change point detection
    pub recall: f64,
    /// F1 score for change point detection
    pub f1_score: f64,
    /// Mean absolute error in change point timing
    pub timing_error: f64,
    /// Hausdorff distance between true and detected change points
    pub hausdorff_distance: f64,
}

/// Time series similarity evaluation results
#[derive(Debug, Clone)]
pub struct SimilarityResults {
    /// Dynamic Time Warping distance
    pub dtw_distance: f64,
    /// Pearson correlation coefficient
    pub pearson_correlation: f64,
    /// Spearman rank correlation
    pub spearman_correlation: f64,
    /// Cross-correlation maximum
    pub cross_correlation_max: f64,
    /// Euclidean distance
    pub euclidean_distance: f64,
}

/// Forecasting metrics calculator
pub struct ForecastingMetrics {
    naive_forecast: Option<Array1<f64>>,
}

impl Default for ForecastingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ForecastingMetrics {
    /// Create new forecasting metrics calculator
    pub fn new() -> Self {
        Self {
            naive_forecast: None,
        }
    }

    /// Set naive forecast for MASE calculation (typically last known value)
    pub fn with_naive_forecast(mut self, naive_forecast: Array1<f64>) -> Self {
        self.naive_forecast = Some(naive_forecast);
        self
    }

    /// Evaluate forecasting performance
    pub fn evaluate_forecast(
        &self,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        y_train: Option<&Array1<f64>>,
    ) -> Result<ForecastingResults> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::InvalidInput(
                "True and predicted values must have same length".to_string(),
            ));
        }

        let mae = mean_absolute_error(y_true, y_pred)?;
        let mse = mean_squared_error(y_true, y_pred)?;
        let rmse = mse.sqrt();

        // Calculate MAPE
        let mape = self.calculate_mape(y_true, y_pred)?;

        // Calculate SMAPE
        let smape = self.calculate_smape(y_true, y_pred)?;

        // Calculate MASE
        let mase = if let Some(train) = y_train {
            self.calculate_mase(y_true, y_pred, train)?
        } else if let Some(naive) = &self.naive_forecast {
            self.calculate_mase_with_naive(y_true, y_pred, naive)?
        } else {
            0.0 // Can't calculate without reference
        };

        // Calculate directional accuracy
        let directional_accuracy = self.calculate_directional_accuracy(y_true, y_pred, y_train)?;

        // Calculate forecast bias (mean error)
        let forecast_bias = y_pred
            .iter()
            .zip(y_true.iter())
            .map(|(pred, true_val)| pred - true_val)
            .sum::<f64>()
            / y_true.len() as f64;

        // Calculate Theil's U statistic
        let theil_u = self.calculate_theil_u(y_true, y_pred, y_train)?;

        Ok(ForecastingResults {
            mae,
            rmse,
            mape,
            smape,
            mase,
            directional_accuracy,
            forecast_bias,
            theil_u,
        })
    }

    /// Calculate Mean Absolute Percentage Error
    fn calculate_mape(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
        let mut sum = 0.0;
        let mut count = 0;

        for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
            if true_val.abs() > 1e-10 {
                // Avoid division by zero
                sum += ((true_val - pred_val) / true_val).abs();
                count += 1;
            }
        }

        if count > 0 {
            Ok(100.0 * sum / count as f64)
        } else {
            Err(MetricsError::ComputationError(
                "All true values are zero".to_string(),
            ))
        }
    }

    /// Calculate Symmetric Mean Absolute Percentage Error
    fn calculate_smape(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
        let mut sum = 0.0;
        let mut count = 0;

        for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
            let denominator = (true_val.abs() + pred_val.abs()) / 2.0;
            if denominator > 1e-10 {
                sum += (true_val - pred_val).abs() / denominator;
                count += 1;
            }
        }

        if count > 0 {
            Ok(100.0 * sum / count as f64)
        } else {
            Err(MetricsError::ComputationError(
                "All values are zero".to_string(),
            ))
        }
    }

    /// Calculate Mean Absolute Scaled Error using training data
    fn calculate_mase(
        &self,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        y_train: &Array1<f64>,
    ) -> Result<f64> {
        if y_train.len() < 2 {
            return Err(MetricsError::InvalidInput(
                "Training data too short for MASE".to_string(),
            ));
        }

        // Calculate MAE of forecast
        let forecast_mae = mean_absolute_error(y_true, y_pred)?;

        // Calculate MAE of naive forecast on training data
        let mut naive_errors = Vec::new();
        for i in 1..y_train.len() {
            naive_errors.push((y_train[i] - y_train[i - 1]).abs());
        }

        let naive_mae = naive_errors.iter().sum::<f64>() / naive_errors.len() as f64;

        if naive_mae > 1e-10 {
            Ok(forecast_mae / naive_mae)
        } else {
            Err(MetricsError::ComputationError(
                "Naive forecast MAE is zero".to_string(),
            ))
        }
    }

    /// Calculate MASE with provided naive forecast
    fn calculate_mase_with_naive(
        &self,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        naive: &Array1<f64>,
    ) -> Result<f64> {
        if naive.len() != y_true.len() {
            return Err(MetricsError::InvalidInput(
                "Naive forecast length mismatch".to_string(),
            ));
        }

        let forecast_mae = mean_absolute_error(y_true, y_pred)?;
        let naive_mae = mean_absolute_error(y_true, naive)?;

        if naive_mae > 1e-10 {
            Ok(forecast_mae / naive_mae)
        } else {
            Err(MetricsError::ComputationError(
                "Naive forecast MAE is zero".to_string(),
            ))
        }
    }

    /// Calculate directional accuracy
    fn calculate_directional_accuracy(
        &self,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        y_train: Option<&Array1<f64>>,
    ) -> Result<f64> {
        if y_true.len() < 2 {
            return Ok(0.0);
        }

        let _baseline = if let Some(train) = y_train {
            if !train.is_empty() {
                train[train.len() - 1]
            } else {
                y_true[0]
            }
        } else {
            y_true[0]
        };

        let mut correct = 0;
        for i in 1..y_true.len() {
            let true_direction = y_true[i] > y_true[i - 1];
            let pred_direction = y_pred[i] > y_pred[i - 1];

            if true_direction == pred_direction {
                correct += 1;
            }
        }

        Ok(correct as f64 / (y_true.len() - 1) as f64)
    }

    /// Calculate Theil's U statistic
    fn calculate_theil_u(
        &self,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        y_train: Option<&Array1<f64>>,
    ) -> Result<f64> {
        let mse_forecast = mean_squared_error(y_true, y_pred)?;

        // Calculate MSE of naive forecast (no-change forecast)
        let baseline = if let Some(train) = y_train {
            if !train.is_empty() {
                train[train.len() - 1]
            } else {
                y_true[0]
            }
        } else {
            y_true[0]
        };

        let naive_forecast = Array1::from_elem(y_true.len(), baseline);
        let mse_naive = mean_squared_error(y_true, &naive_forecast)?;

        if mse_naive > 1e-10 {
            Ok((mse_forecast / mse_naive).sqrt())
        } else {
            Ok(f64::INFINITY)
        }
    }
}

/// Time series anomaly detection metrics calculator
pub struct TimeSeriesAnomalyMetrics {
    tolerance_window: usize,
}

impl Default for TimeSeriesAnomalyMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSeriesAnomalyMetrics {
    /// Create new time series anomaly detection metrics calculator
    pub fn new() -> Self {
        Self {
            tolerance_window: 5,
        }
    }

    /// Set tolerance window for point-adjust metrics
    pub fn with_tolerance_window(mut self, window: usize) -> Self {
        self.tolerance_window = window;
        self
    }

    /// Evaluate anomaly detection performance
    pub fn evaluate_anomaly_detection(
        &self,
        y_true: &Array1<i32>,          // 1 for anomaly, 0 for normal
        y_pred: &Array1<i32>,          // 1 for detected anomaly, 0 for normal
        y_score: Option<&Array1<f64>>, // Anomaly scores for AUC calculation
    ) -> Result<TimeSeriesAnomalyResults> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::InvalidInput(
                "True and predicted labels must have same length".to_string(),
            ));
        }

        // Calculate standard metrics
        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_count = 0;

        for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
            match (true_val, pred_val) {
                (1, 1) => tp += 1,
                (0, 1) => fp += 1,
                (0, 0) => tn += 1,
                (1, 0) => fn_count += 1,
                _ => {
                    return Err(MetricsError::InvalidInput(
                        "Invalid label values".to_string(),
                    ))
                }
            }
        }

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };

        let recall = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };

        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let false_alarm_rate = if fp + tn > 0 {
            fp as f64 / (fp + tn) as f64
        } else {
            0.0
        };

        let miss_detection_rate = if fn_count + tp > 0 {
            fn_count as f64 / (fn_count + tp) as f64
        } else {
            0.0
        };

        // Calculate AUC if scores are provided
        let auc = if let Some(scores) = y_score {
            self.calculate_auc(y_true, scores)?
        } else {
            0.0
        };

        // Calculate point-adjust metrics
        let (point_adjust_precision, point_adjust_recall) =
            self.calculate_point_adjust_metrics(y_true, y_pred)?;

        Ok(TimeSeriesAnomalyResults {
            precision,
            recall,
            f1_score,
            auc,
            false_alarm_rate,
            miss_detection_rate,
            point_adjust_precision,
            point_adjust_recall,
        })
    }

    /// Calculate AUC for time series anomaly detection
    fn calculate_auc(&self, y_true: &Array1<i32>, y_score: &Array1<f64>) -> Result<f64> {
        if y_true.len() != y_score.len() {
            return Err(MetricsError::InvalidInput("Length mismatch".to_string()));
        }

        // Create pairs of (score, label) and sort by score
        let mut pairs: Vec<(f64, i32)> = y_score
            .iter()
            .zip(y_true.iter())
            .map(|(&score, &label)| (score, label))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut tp = 0;
        let mut fp = 0;
        let total_positive = y_true.iter().filter(|&&x| x == 1).count();
        let total_negative = y_true.len() - total_positive;

        if total_positive == 0 || total_negative == 0 {
            return Ok(0.5); // No meaningful AUC
        }

        let mut auc = 0.0;
        let mut prev_fpr = 0.0;

        for (_, label) in pairs {
            if label == 1 {
                tp += 1;
            } else {
                fp += 1;
            }

            let tpr = tp as f64 / total_positive as f64;
            let fpr = fp as f64 / total_negative as f64;

            auc += (fpr - prev_fpr) * tpr;
            prev_fpr = fpr;
        }

        Ok(auc)
    }

    /// Calculate point-adjust precision and recall
    fn calculate_point_adjust_metrics(
        &self,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
    ) -> Result<(f64, f64)> {
        // Find true anomaly windows
        let true_anomalies = self.find_anomaly_segments(y_true);
        let pred_anomalies = self.find_anomaly_segments(y_pred);

        // For each true anomaly, check if any prediction falls within tolerance window
        let mut true_positive_segments = 0;
        for (start, end) in &true_anomalies {
            let window_start = start.saturating_sub(self.tolerance_window);
            let window_end = (end + self.tolerance_window).min(y_true.len() - 1);

            let has_detection = pred_anomalies.iter().any(|(pred_start, pred_end)| {
                *pred_start <= window_end && *pred_end >= window_start
            });

            if has_detection {
                true_positive_segments += 1;
            }
        }

        // For each predicted anomaly, check if it's within tolerance of any true anomaly
        let mut correctly_detected_segments = 0;
        for (pred_start, pred_end) in &pred_anomalies {
            let has_true_anomaly = true_anomalies.iter().any(|(true_start, true_end)| {
                let window_start = true_start.saturating_sub(self.tolerance_window);
                let window_end = (true_end + self.tolerance_window).min(y_true.len() - 1);

                *pred_start <= window_end && *pred_end >= window_start
            });

            if has_true_anomaly {
                correctly_detected_segments += 1;
            }
        }

        let point_adjust_recall = if !true_anomalies.is_empty() {
            true_positive_segments as f64 / true_anomalies.len() as f64
        } else {
            1.0
        };

        let point_adjust_precision = if !pred_anomalies.is_empty() {
            correctly_detected_segments as f64 / pred_anomalies.len() as f64
        } else {
            1.0
        };

        Ok((point_adjust_precision, point_adjust_recall))
    }

    /// Find contiguous anomaly segments
    fn find_anomaly_segments(&self, labels: &Array1<i32>) -> Vec<(usize, usize)> {
        let mut segments = Vec::new();
        let mut start = None;

        for (i, &label) in labels.iter().enumerate() {
            if label == 1 && start.is_none() {
                start = Some(i);
            } else if label == 0 && start.is_some() {
                segments.push((start.unwrap(), i - 1));
                start = None;
            }
        }

        // Handle case where anomaly extends to end
        if let Some(start_idx) = start {
            segments.push((start_idx, labels.len() - 1));
        }

        segments
    }
}

/// Trend analysis metrics calculator
pub struct TrendAnalysisMetrics;

impl Default for TrendAnalysisMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl TrendAnalysisMetrics {
    /// Create new trend analysis metrics calculator
    pub fn new() -> Self {
        Self
    }

    /// Evaluate trend analysis performance
    pub fn evaluate_trend_analysis(
        &self,
        time_series: &Array1<f64>,
        period: Option<usize>,
    ) -> Result<TrendAnalysisResults> {
        if time_series.len() < 10 {
            return Err(MetricsError::InvalidInput(
                "Time series too short for trend analysis".to_string(),
            ));
        }

        // Calculate trend strength using linear regression
        let trend_strength = self.calculate_trend_strength(time_series)?;

        // Calculate seasonality strength
        let seasonality_strength = if let Some(p) = period {
            self.calculate_seasonality_strength(time_series, p)?
        } else {
            0.0
        };

        // Calculate autocorrelation at lag 1
        let autocorr_lag1 = self.calculate_autocorrelation(time_series, 1)?;

        // Simplified stationarity test (based on variance of differences)
        let is_stationary = self.test_stationarity(time_series)?;

        // Simplified Ljung-Box test (placeholder)
        let ljung_box_pvalue = 0.5; // Would need proper implementation

        // Simplified ADF test (placeholder)
        let adf_statistic = -2.0; // Would need proper implementation

        Ok(TrendAnalysisResults {
            trend_strength,
            seasonality_strength,
            autocorr_lag1,
            ljung_box_pvalue,
            adf_statistic,
            is_stationary,
        })
    }

    /// Calculate trend strength using correlation with linear trend
    fn calculate_trend_strength(&self, time_series: &Array1<f64>) -> Result<f64> {
        let n = time_series.len();
        if n < 2 {
            return Ok(0.0);
        }

        // Calculate correlation with linear trend
        let x: Array1<f64> = Array1::linspace(0.0, (n - 1) as f64, n);

        let x_mean = x.mean().unwrap_or(0.0);
        let y_mean = time_series.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut x_var = 0.0;
        let mut y_var = 0.0;

        for i in 0..n {
            let x_diff = x[i] - x_mean;
            let y_diff = time_series[i] - y_mean;
            numerator += x_diff * y_diff;
            x_var += x_diff * x_diff;
            y_var += y_diff * y_diff;
        }

        if x_var > 1e-10 && y_var > 1e-10 {
            let correlation = numerator / (x_var * y_var).sqrt();
            Ok(correlation.abs()) // Use absolute correlation as trend strength
        } else {
            Ok(0.0)
        }
    }

    /// Calculate seasonality strength
    fn calculate_seasonality_strength(
        &self,
        time_series: &Array1<f64>,
        period: usize,
    ) -> Result<f64> {
        if time_series.len() < period * 2 {
            return Ok(0.0);
        }

        // Calculate variance of seasonal differences
        let mut seasonal_diffs = Vec::new();
        for i in period..time_series.len() {
            seasonal_diffs.push(time_series[i] - time_series[i - period]);
        }

        if seasonal_diffs.is_empty() {
            return Ok(0.0);
        }

        let seasonal_var = self.calculate_variance(&seasonal_diffs)?;
        let total_var = self.calculate_variance(&time_series.to_vec())?;

        if total_var > 1e-10 {
            Ok(1.0 - seasonal_var / total_var)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate autocorrelation at given lag
    fn calculate_autocorrelation(&self, time_series: &Array1<f64>, lag: usize) -> Result<f64> {
        if time_series.len() <= lag {
            return Ok(0.0);
        }

        let n = time_series.len() - lag;
        let x1 = &time_series.slice(s![..n]);
        let x2 = &time_series.slice(s![lag..]);

        let mean1 = x1.mean().unwrap_or(0.0);
        let mean2 = x2.mean().unwrap_or(0.0);

        let mut covariance = 0.0_f64;
        let mut var1 = 0.0_f64;
        let mut var2 = 0.0_f64;

        for i in 0..n {
            let diff1 = x1[i] - mean1;
            let diff2 = x2[i] - mean2;
            covariance += diff1 * diff2;
            var1 += diff1 * diff1;
            var2 += diff2 * diff2;
        }

        if var1 > 1e-10 && var2 > 1e-10 {
            Ok(covariance / (var1 * var2).sqrt())
        } else {
            Ok(0.0)
        }
    }

    /// Simple stationarity test based on variance stability
    fn test_stationarity(&self, time_series: &Array1<f64>) -> Result<bool> {
        let n = time_series.len();
        if n < 20 {
            return Ok(true); // Too short to determine
        }

        // Split into first and second half
        let mid = n / 2;
        let first_half = time_series.slice(s![..mid]);
        let second_half = time_series.slice(s![mid..]);

        let var1 = self.calculate_variance(&first_half.to_vec())?;
        let var2 = self.calculate_variance(&second_half.to_vec())?;

        // Simple test: variance ratio should be close to 1 for stationary series
        let ratio = if var2 > 1e-10 { var1 / var2 } else { 1.0 };
        Ok(ratio > 0.5 && ratio < 2.0)
    }

    /// Calculate variance of a vector
    fn calculate_variance(&self, data: &[f64]) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;

        Ok(variance)
    }
}

/// Complete time series metrics suite
pub struct TimeSeriesSuite {
    forecasting: ForecastingMetrics,
    anomaly: TimeSeriesAnomalyMetrics,
    trend: TrendAnalysisMetrics,
}

impl Default for TimeSeriesSuite {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSeriesSuite {
    /// Create a new time series metrics suite
    pub fn new() -> Self {
        Self {
            forecasting: ForecastingMetrics::new(),
            anomaly: TimeSeriesAnomalyMetrics::new(),
            trend: TrendAnalysisMetrics::new(),
        }
    }

    /// Get forecasting metrics calculator
    pub fn forecasting(&self) -> &ForecastingMetrics {
        &self.forecasting
    }

    /// Get anomaly detection metrics calculator
    pub fn anomaly(&self) -> &TimeSeriesAnomalyMetrics {
        &self.anomaly
    }

    /// Get trend analysis metrics calculator
    pub fn trend(&self) -> &TrendAnalysisMetrics {
        &self.trend
    }
}

impl DomainMetrics for TimeSeriesSuite {
    type Result = DomainEvaluationResult;

    fn domain_name(&self) -> &'static str {
        "Time Series"
    }

    fn available_metrics(&self) -> Vec<&'static str> {
        vec![
            "forecasting_mape",
            "forecasting_smape",
            "forecasting_mase",
            "forecasting_directional_accuracy",
            "anomaly_precision",
            "anomaly_recall",
            "anomaly_f1",
            "anomaly_auc",
            "trend_strength",
            "seasonality_strength",
            "autocorrelation_lag1",
            "is_stationary",
        ]
    }

    fn metric_descriptions(&self) -> HashMap<&'static str, &'static str> {
        let mut descriptions = HashMap::new();
        descriptions.insert(
            "forecasting_mape",
            "Mean Absolute Percentage Error for forecasting",
        );
        descriptions.insert(
            "forecasting_smape",
            "Symmetric Mean Absolute Percentage Error",
        );
        descriptions.insert("forecasting_mase", "Mean Absolute Scaled Error");
        descriptions.insert(
            "forecasting_directional_accuracy",
            "Directional accuracy for trend prediction",
        );
        descriptions.insert("anomaly_precision", "Precision for anomaly detection");
        descriptions.insert("anomaly_recall", "Recall for anomaly detection");
        descriptions.insert("anomaly_f1", "F1 score for anomaly detection");
        descriptions.insert("anomaly_auc", "Area Under ROC Curve for anomaly detection");
        descriptions.insert("trend_strength", "Strength of linear trend component");
        descriptions.insert("seasonality_strength", "Strength of seasonal component");
        descriptions.insert("autocorrelation_lag1", "Autocorrelation at lag 1");
        descriptions.insert("is_stationary", "Whether the time series is stationary");
        descriptions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_forecasting_metrics() {
        let metrics = ForecastingMetrics::new();

        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = Array1::from_vec(vec![1.1, 2.1, 2.9, 4.1, 4.9]);
        let y_train = Array1::from_vec(vec![0.0, 1.0]);

        let results = metrics
            .evaluate_forecast(&y_true, &y_pred, Some(&y_train))
            .unwrap();

        assert!(results.mae >= 0.0);
        assert!(results.rmse >= 0.0);
        assert!(results.mape >= 0.0);
        assert!(results.smape >= 0.0);
        assert!(results.directional_accuracy >= 0.0 && results.directional_accuracy <= 1.0);
    }

    #[test]
    fn test_mape_calculation() {
        let metrics = ForecastingMetrics::new();

        let y_true = Array1::from_vec(vec![100.0, 200.0, 300.0]);
        let y_pred = Array1::from_vec(vec![110.0, 190.0, 320.0]);

        let mape = metrics.calculate_mape(&y_true, &y_pred).unwrap();

        // Expected: (10/100 + 10/200 + 20/300) / 3 * 100 = (0.1 + 0.05 + 0.067) / 3 * 100 ≈ 7.22%
        assert!(mape > 6.0 && mape < 8.0);
    }

    #[test]
    fn test_anomaly_detection_metrics() {
        let metrics = TimeSeriesAnomalyMetrics::new();

        let y_true = Array1::from_vec(vec![0, 0, 1, 1, 0, 0, 1, 0]);
        let y_pred = Array1::from_vec(vec![0, 0, 1, 0, 0, 1, 1, 0]);
        let y_score = Array1::from_vec(vec![0.1, 0.2, 0.9, 0.6, 0.3, 0.8, 0.95, 0.1]);

        let results = metrics
            .evaluate_anomaly_detection(&y_true, &y_pred, Some(&y_score))
            .unwrap();

        assert!(results.precision >= 0.0 && results.precision <= 1.0);
        assert!(results.recall >= 0.0 && results.recall <= 1.0);
        assert!(results.f1_score >= 0.0 && results.f1_score <= 1.0);
        assert!(results.auc >= 0.0 && results.auc <= 1.0);
    }

    #[test]
    fn test_trend_analysis_metrics() {
        let metrics = TrendAnalysisMetrics::new();

        // Create a simple linear trend
        let trend_data = Array1::linspace(0.0, 10.0, 20);

        let results = metrics
            .evaluate_trend_analysis(&trend_data, Some(4))
            .unwrap();

        assert!(results.trend_strength >= 0.0 && results.trend_strength <= 1.0);
        assert!(results.seasonality_strength >= 0.0 && results.seasonality_strength <= 1.0);
        assert!(results.autocorr_lag1 >= -1.0 && results.autocorr_lag1 <= 1.0);
    }

    #[test]
    fn test_autocorrelation_calculation() {
        let metrics = TrendAnalysisMetrics::new();

        // Perfect autocorrelation (constant series)
        let constant_series = Array1::from_elem(10, 5.0);
        let autocorr = metrics
            .calculate_autocorrelation(&constant_series, 1)
            .unwrap();

        // Should be close to 1 or 0 (undefined for constant series)
        assert!((-1.0..=1.0).contains(&autocorr));
    }

    #[test]
    fn test_time_series_suite() {
        let suite = TimeSeriesSuite::new();

        assert_eq!(suite.domain_name(), "Time Series");
        assert!(!suite.available_metrics().is_empty());
        assert!(!suite.metric_descriptions().is_empty());
    }
}
