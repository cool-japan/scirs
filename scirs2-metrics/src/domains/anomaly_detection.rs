//! Anomaly detection domain metrics
//!
//! This module provides specialized metric collections for anomaly detection
//! including detection accuracy, distribution analysis, and business metrics.

use crate::anomaly::{
    anomaly_auc_score, detection_accuracy, false_alarm_rate, miss_detection_rate,
};
use crate::domains::{DomainEvaluationResult, DomainMetrics};
use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Detection performance evaluation results
#[derive(Debug, Clone)]
pub struct DetectionResults {
    /// Detection accuracy
    pub accuracy: f64,
    /// Precision for anomaly detection
    pub precision: f64,
    /// Recall for anomaly detection (sensitivity)
    pub recall: f64,
    /// F1 score for anomaly detection
    pub f1_score: f64,
    /// Specificity (true negative rate)
    pub specificity: f64,
    /// Area Under ROC Curve
    pub auc_roc: f64,
    /// Area Under Precision-Recall Curve
    pub auc_pr: f64,
    /// False alarm rate (Type I error)
    pub false_alarm_rate: f64,
    /// Miss detection rate (Type II error)
    pub miss_detection_rate: f64,
    /// Matthews Correlation Coefficient
    pub mcc: f64,
}

/// Distribution-based anomaly evaluation results
#[derive(Debug, Clone)]
pub struct DistributionResults {
    /// Kullback-Leibler divergence
    pub kl_divergence: f64,
    /// Jensen-Shannon divergence
    pub js_divergence: f64,
    /// Wasserstein distance (Earth Mover's Distance)
    pub wasserstein_distance: f64,
    /// Maximum Mean Discrepancy
    pub mmd: f64,
    /// Energy distance
    pub energy_distance: f64,
    /// Kolmogorov-Smirnov test statistic
    pub ks_statistic: f64,
    /// Anderson-Darling test statistic
    pub ad_statistic: f64,
}

/// Time series anomaly evaluation results
#[derive(Debug, Clone)]
pub struct TimeSeriesAnomalyResults {
    /// Standard precision
    pub precision: f64,
    /// Standard recall
    pub recall: f64,
    /// Point-adjust precision (with tolerance window)
    pub point_adjust_precision: f64,
    /// Point-adjust recall (with tolerance window)
    pub point_adjust_recall: f64,
    /// Range-based precision
    pub range_precision: f64,
    /// Range-based recall
    pub range_recall: f64,
    /// Numenta Anomaly Benchmark (NAB) score
    pub nab_score: f64,
    /// Time-weighted precision
    pub time_weighted_precision: f64,
    /// Average delay in detection
    pub average_delay: f64,
}

/// Clustering-based anomaly evaluation results
#[derive(Debug, Clone)]
pub struct ClusteringAnomalyResults {
    /// Isolation score (average path length)
    pub isolation_score: f64,
    /// Local Outlier Factor (LOF) score
    pub lof_score: f64,
    /// Silhouette score for anomalies
    pub silhouette_score: f64,
    /// DBSCAN outlier ratio
    pub dbscan_outlier_ratio: f64,
    /// Cluster validity for normal points
    pub cluster_validity: f64,
    /// Average distance to nearest normal cluster
    pub distance_to_normal: f64,
}

/// Business-oriented anomaly evaluation results
#[derive(Debug, Clone)]
pub struct BusinessAnomalyResults {
    /// Cost-sensitive accuracy
    pub cost_sensitive_accuracy: f64,
    /// Economic value of detection
    pub economic_value: f64,
    /// Alert fatigue score (too many false alarms)
    pub alert_fatigue_score: f64,
    /// Response time efficiency
    pub response_efficiency: f64,
    /// Coverage of critical anomalies
    pub critical_coverage: f64,
    /// ROI of anomaly detection system
    pub detection_roi: f64,
}

/// Detection performance metrics calculator
pub struct DetectionMetrics {
    threshold: f64,
}

impl Default for DetectionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl DetectionMetrics {
    /// Create new detection metrics calculator
    pub fn new() -> Self {
        Self { threshold: 0.5 }
    }

    /// Set detection threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Evaluate anomaly detection performance
    pub fn evaluate_detection(
        &self,
        y_true: &Array1<f64>,  // 1.0 for anomalies, 0.0 for normal
        y_score: &Array1<f64>, // Anomaly scores
    ) -> Result<DetectionResults> {
        if y_true.len() != y_score.len() {
            return Err(MetricsError::InvalidInput(
                "True labels and scores must have same length".to_string(),
            ));
        }

        // Convert to binary predictions
        let y_pred = y_score.mapv(|score| if score >= self.threshold { 1.0 } else { 0.0 });

        // Calculate basic metrics
        let accuracy = detection_accuracy(y_true, &y_pred)?;
        let far = false_alarm_rate(y_true, &y_pred)?;
        let mdr = miss_detection_rate(y_true, &y_pred)?;
        let auc_roc = anomaly_auc_score(y_true, y_score)?;

        // Calculate confusion matrix components
        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut tn = 0.0;
        let mut fn_count = 0.0;

        for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
            match (true_val > 0.5, pred_val > 0.5) {
                (true, true) => tp += 1.0,
                (false, true) => fp += 1.0,
                (false, false) => tn += 1.0,
                (true, false) => fn_count += 1.0,
            }
        }

        // Calculate derived metrics
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_count > 0.0 {
            tp / (tp + fn_count)
        } else {
            0.0
        };
        let specificity = if tn + fp > 0.0 { tn / (tn + fp) } else { 0.0 };

        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        // Calculate Matthews Correlation Coefficient
        let mcc = if (tp + fp) * (tp + fn_count) * (tn + fp) * (tn + fn_count) > 0.0 {
            let denominator: f64 = (tp + fp) * (tp + fn_count) * (tn + fp) * (tn + fn_count);
            (tp * tn - fp * fn_count) / denominator.sqrt()
        } else {
            0.0
        };

        // Calculate AUC-PR (simplified)
        let auc_pr = self.calculate_auc_pr(y_true, y_score)?;

        Ok(DetectionResults {
            accuracy,
            precision,
            recall,
            f1_score,
            specificity,
            auc_roc,
            auc_pr,
            false_alarm_rate: far,
            miss_detection_rate: mdr,
            mcc,
        })
    }

    /// Calculate Area Under Precision-Recall Curve
    fn calculate_auc_pr(&self, y_true: &Array1<f64>, y_score: &Array1<f64>) -> Result<f64> {
        // Create score-label pairs and sort by score (descending)
        let mut pairs: Vec<(f64, f64)> = y_score
            .iter()
            .zip(y_true.iter())
            .map(|(&score, &label)| (score, label))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_positives = y_true.iter().filter(|&&x| x > 0.5).count() as f64;
        if total_positives == 0.0 {
            return Ok(0.0);
        }

        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut auc = 0.0;
        let mut prev_recall = 0.0;

        for (_, label) in pairs {
            if label > 0.5 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }

            let recall = tp / total_positives;
            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };

            // Add area of trapezoid
            auc += (recall - prev_recall) * precision;
            prev_recall = recall;
        }

        Ok(auc)
    }
}

/// Distribution-based anomaly metrics calculator
pub struct DistributionMetrics;

impl Default for DistributionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl DistributionMetrics {
    /// Create new distribution metrics calculator
    pub fn new() -> Self {
        Self
    }

    /// Evaluate distribution-based anomaly detection
    pub fn evaluate_distribution_anomalies(
        &self,
        normal_samples: &Array1<f64>,
        anomaly_samples: &Array1<f64>,
    ) -> Result<DistributionResults> {
        // Calculate KL divergence (requires probability distributions)
        let kl_divergence =
            self.calculate_kl_divergence_from_samples(normal_samples, anomaly_samples)?;

        // Calculate JS divergence
        let js_divergence =
            self.calculate_js_divergence_from_samples(normal_samples, anomaly_samples)?;

        // Calculate Wasserstein distance (1D case)
        let wasserstein_distance =
            self.calculate_wasserstein_distance(normal_samples, anomaly_samples)?;

        // Calculate Maximum Mean Discrepancy (simplified)
        let mmd = self.calculate_mmd(normal_samples, anomaly_samples)?;

        // Calculate energy distance
        let energy_distance = self.calculate_energy_distance(normal_samples, anomaly_samples)?;

        // Calculate Kolmogorov-Smirnov statistic
        let ks_statistic = self.calculate_ks_statistic(normal_samples, anomaly_samples)?;

        // Calculate Anderson-Darling statistic (simplified)
        let ad_statistic = self.calculate_ad_statistic(normal_samples, anomaly_samples)?;

        Ok(DistributionResults {
            kl_divergence,
            js_divergence,
            wasserstein_distance,
            mmd,
            energy_distance,
            ks_statistic,
            ad_statistic,
        })
    }

    /// Calculate KL divergence from samples (using histogram approximation)
    fn calculate_kl_divergence_from_samples(
        &self,
        normal_samples: &Array1<f64>,
        anomaly_samples: &Array1<f64>,
    ) -> Result<f64> {
        let bins = 50;
        let (p_hist, q_hist) = self.create_histograms(normal_samples, anomaly_samples, bins)?;

        let mut kl_div = 0.0;
        for (p, q) in p_hist.iter().zip(q_hist.iter()) {
            if *p > 0.0 && *q > 0.0 {
                kl_div += p * (p / q).ln();
            } else if *p > 0.0 && *q == 0.0 {
                return Ok(f64::INFINITY); // KL divergence is infinite
            }
        }

        Ok(kl_div)
    }

    /// Calculate JS divergence from samples
    fn calculate_js_divergence_from_samples(
        &self,
        normal_samples: &Array1<f64>,
        anomaly_samples: &Array1<f64>,
    ) -> Result<f64> {
        let bins = 50;
        let (p_hist, q_hist) = self.create_histograms(normal_samples, anomaly_samples, bins)?;

        let mut js_div = 0.0;
        for (p, q) in p_hist.iter().zip(q_hist.iter()) {
            let m = (p + q) / 2.0;
            if *p > 0.0 && m > 0.0 {
                js_div += 0.5 * p * (p / m).ln();
            }
            if *q > 0.0 && m > 0.0 {
                js_div += 0.5 * q * (q / m).ln();
            }
        }

        Ok(js_div)
    }

    /// Create normalized histograms for two sample sets
    fn create_histograms(
        &self,
        samples1: &Array1<f64>,
        samples2: &Array1<f64>,
        bins: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let min_val = samples1
            .iter()
            .chain(samples2.iter())
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = samples1
            .iter()
            .chain(samples2.iter())
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if (max_val - min_val).abs() < 1e-10 {
            return Ok((vec![1.0 / bins as f64; bins], vec![1.0 / bins as f64; bins]));
        }

        let bin_width = (max_val - min_val) / bins as f64;

        let mut hist1 = vec![0; bins];
        let mut hist2 = vec![0; bins];

        for &sample in samples1.iter() {
            let bin_idx = ((sample - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(bins - 1);
            hist1[bin_idx] += 1;
        }

        for &sample in samples2.iter() {
            let bin_idx = ((sample - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(bins - 1);
            hist2[bin_idx] += 1;
        }

        // Normalize to probabilities
        let sum1 = hist1.iter().sum::<usize>() as f64;
        let sum2 = hist2.iter().sum::<usize>() as f64;

        let norm_hist1 = hist1.iter().map(|&x| x as f64 / sum1).collect();
        let norm_hist2 = hist2.iter().map(|&x| x as f64 / sum2).collect();

        Ok((norm_hist1, norm_hist2))
    }

    /// Calculate Wasserstein distance (1D case - optimal transport)
    fn calculate_wasserstein_distance(
        &self,
        samples1: &Array1<f64>,
        samples2: &Array1<f64>,
    ) -> Result<f64> {
        let mut sorted1 = samples1.to_vec();
        let mut sorted2 = samples2.to_vec();
        sorted1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted2.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // For equal-length samples, Wasserstein-1 is the L1 distance between sorted samples
        if sorted1.len() == sorted2.len() {
            let distance = sorted1
                .iter()
                .zip(sorted2.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / sorted1.len() as f64;
            Ok(distance)
        } else {
            // For different lengths, use interpolation (simplified)
            let n = sorted1.len().min(sorted2.len());
            let distance = (0..n)
                .map(|i| {
                    let idx1 = i * sorted1.len() / n;
                    let idx2 = i * sorted2.len() / n;
                    (sorted1[idx1] - sorted2[idx2]).abs()
                })
                .sum::<f64>()
                / n as f64;
            Ok(distance)
        }
    }

    /// Calculate Maximum Mean Discrepancy (simplified with RBF kernel)
    fn calculate_mmd(&self, samples1: &Array1<f64>, samples2: &Array1<f64>) -> Result<f64> {
        let gamma = 1.0; // RBF kernel parameter

        let n1 = samples1.len() as f64;
        let n2 = samples2.len() as f64;

        // K(X, X)
        let mut kxx = 0.0;
        for i in 0..samples1.len() {
            for j in 0..samples1.len() {
                kxx += (-gamma * (samples1[i] - samples1[j]).powi(2)).exp();
            }
        }
        kxx /= n1 * n1;

        // K(Y, Y)
        let mut kyy = 0.0;
        for i in 0..samples2.len() {
            for j in 0..samples2.len() {
                kyy += (-gamma * (samples2[i] - samples2[j]).powi(2)).exp();
            }
        }
        kyy /= n2 * n2;

        // K(X, Y)
        let mut kxy = 0.0;
        for i in 0..samples1.len() {
            for j in 0..samples2.len() {
                kxy += (-gamma * (samples1[i] - samples2[j]).powi(2)).exp();
            }
        }
        kxy /= n1 * n2;

        Ok(kxx + kyy - 2.0 * kxy)
    }

    /// Calculate energy distance
    fn calculate_energy_distance(
        &self,
        samples1: &Array1<f64>,
        samples2: &Array1<f64>,
    ) -> Result<f64> {
        let n1 = samples1.len() as f64;
        let n2 = samples2.len() as f64;

        // E[|X - Y|]
        let mut exy = 0.0;
        for &x in samples1.iter() {
            for &y in samples2.iter() {
                exy += (x - y).abs();
            }
        }
        exy /= n1 * n2;

        // E[|X - X'|]
        let mut exx = 0.0;
        for i in 0..samples1.len() {
            for j in 0..samples1.len() {
                if i != j {
                    exx += (samples1[i] - samples1[j]).abs();
                }
            }
        }
        exx /= n1 * (n1 - 1.0);

        // E[|Y - Y'|]
        let mut eyy = 0.0;
        for i in 0..samples2.len() {
            for j in 0..samples2.len() {
                if i != j {
                    eyy += (samples2[i] - samples2[j]).abs();
                }
            }
        }
        eyy /= n2 * (n2 - 1.0);

        Ok(2.0 * exy - exx - eyy)
    }

    /// Calculate Kolmogorov-Smirnov test statistic
    fn calculate_ks_statistic(
        &self,
        samples1: &Array1<f64>,
        samples2: &Array1<f64>,
    ) -> Result<f64> {
        let mut all_values = samples1
            .iter()
            .chain(samples2.iter())
            .copied()
            .collect::<Vec<_>>();
        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        all_values.dedup();

        let mut max_diff = 0.0_f64;

        for &value in &all_values {
            let cdf1 =
                samples1.iter().filter(|&&x| x <= value).count() as f64 / samples1.len() as f64;
            let cdf2 =
                samples2.iter().filter(|&&x| x <= value).count() as f64 / samples2.len() as f64;
            max_diff = max_diff.max((cdf1 - cdf2).abs());
        }

        Ok(max_diff)
    }

    /// Calculate Anderson-Darling test statistic (simplified)
    fn calculate_ad_statistic(
        &self,
        samples1: &Array1<f64>,
        samples2: &Array1<f64>,
    ) -> Result<f64> {
        // Simplified version - in practice, this would require more complex implementation
        let ks_stat = self.calculate_ks_statistic(samples1, samples2)?;
        Ok(ks_stat * (samples1.len() + samples2.len()) as f64)
    }
}

/// Time series anomaly metrics calculator
pub struct TimeSeriesAnomalyMetrics {
    tolerance_window: usize,
    nab_weights: NABWeights,
}

/// NAB (Numenta Anomaly Benchmark) scoring weights
#[derive(Debug, Clone)]
pub struct NABWeights {
    pub true_positive: f64,
    pub false_positive: f64,
    pub false_negative: f64,
    pub early_detection_bonus: f64,
    pub late_detection_penalty: f64,
}

impl Default for NABWeights {
    fn default() -> Self {
        Self {
            true_positive: 1.0,
            false_positive: -0.11,
            false_negative: -1.0,
            early_detection_bonus: 1.0,
            late_detection_penalty: -1.0,
        }
    }
}

impl Default for TimeSeriesAnomalyMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSeriesAnomalyMetrics {
    /// Create new time series anomaly metrics calculator
    pub fn new() -> Self {
        Self {
            tolerance_window: 5,
            nab_weights: NABWeights::default(),
        }
    }

    /// Set tolerance window for point-adjust metrics
    pub fn with_tolerance_window(mut self, window: usize) -> Self {
        self.tolerance_window = window;
        self
    }

    /// Set NAB scoring weights
    pub fn with_nab_weights(mut self, weights: NABWeights) -> Self {
        self.nab_weights = weights;
        self
    }

    /// Evaluate time series anomaly detection
    pub fn evaluate_time_series_anomalies(
        &self,
        y_true: &Array1<i32>,             // 1 for anomaly, 0 for normal
        y_pred: &Array1<i32>,             // 1 for detected anomaly, 0 for normal
        timestamps: Option<&Array1<f64>>, // Optional timestamps for delay calculation
    ) -> Result<TimeSeriesAnomalyResults> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::InvalidInput(
                "True and predicted labels must have same length".to_string(),
            ));
        }

        // Calculate standard precision and recall
        let (precision, recall) = self.calculate_standard_metrics(y_true, y_pred)?;

        // Calculate point-adjust metrics
        let (point_adjust_precision, point_adjust_recall) =
            self.calculate_point_adjust_metrics(y_true, y_pred)?;

        // Calculate range-based metrics
        let (range_precision, range_recall) = self.calculate_range_based_metrics(y_true, y_pred)?;

        // Calculate NAB score
        let nab_score = self.calculate_nab_score(y_true, y_pred)?;

        // Calculate time-weighted precision
        let time_weighted_precision = if let Some(ts) = timestamps {
            self.calculate_time_weighted_precision(y_true, y_pred, ts)?
        } else {
            precision
        };

        // Calculate average detection delay
        let average_delay = if let Some(ts) = timestamps {
            self.calculate_average_delay(y_true, y_pred, ts)?
        } else {
            0.0
        };

        Ok(TimeSeriesAnomalyResults {
            precision,
            recall,
            point_adjust_precision,
            point_adjust_recall,
            range_precision,
            range_recall,
            nab_score,
            time_weighted_precision,
            average_delay,
        })
    }

    /// Calculate standard precision and recall
    fn calculate_standard_metrics(
        &self,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
    ) -> Result<(f64, f64)> {
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_count = 0;

        for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
            match (true_val, pred_val) {
                (1, 1) => tp += 1,
                (0, 1) => fp += 1,
                (1, 0) => fn_count += 1,
                _ => {} // TN
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

        Ok((precision, recall))
    }

    /// Calculate point-adjust precision and recall with tolerance window
    fn calculate_point_adjust_metrics(
        &self,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
    ) -> Result<(f64, f64)> {
        let true_anomaly_points: Vec<usize> = y_true
            .iter()
            .enumerate()
            .filter(|(_, &val)| val == 1)
            .map(|(idx, _)| idx)
            .collect();

        let pred_anomaly_points: Vec<usize> = y_pred
            .iter()
            .enumerate()
            .filter(|(_, &val)| val == 1)
            .map(|(idx, _)| idx)
            .collect();

        // Point-adjust recall: fraction of true anomalies detected within tolerance
        let mut detected_true_anomalies = 0;
        for &true_point in &true_anomaly_points {
            let detected = pred_anomaly_points.iter().any(|&pred_point| {
                (pred_point as i32 - true_point as i32).abs() <= self.tolerance_window as i32
            });
            if detected {
                detected_true_anomalies += 1;
            }
        }

        let pa_recall = if !true_anomaly_points.is_empty() {
            detected_true_anomalies as f64 / true_anomaly_points.len() as f64
        } else {
            1.0
        };

        // Point-adjust precision: fraction of predictions that are near true anomalies
        let mut correct_predictions = 0;
        for &pred_point in &pred_anomaly_points {
            let correct = true_anomaly_points.iter().any(|&true_point| {
                (pred_point as i32 - true_point as i32).abs() <= self.tolerance_window as i32
            });
            if correct {
                correct_predictions += 1;
            }
        }

        let pa_precision = if !pred_anomaly_points.is_empty() {
            correct_predictions as f64 / pred_anomaly_points.len() as f64
        } else {
            1.0
        };

        Ok((pa_precision, pa_recall))
    }

    /// Calculate range-based precision and recall
    fn calculate_range_based_metrics(
        &self,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
    ) -> Result<(f64, f64)> {
        let true_ranges = self.find_anomaly_ranges(y_true);
        let pred_ranges = self.find_anomaly_ranges(y_pred);

        // Range-based recall: fraction of true ranges that overlap with predicted ranges
        let mut detected_ranges = 0;
        for (true_start, true_end) in &true_ranges {
            let detected = pred_ranges.iter().any(|(pred_start, pred_end)| {
                // Check for overlap
                *pred_start <= *true_end && *pred_end >= *true_start
            });
            if detected {
                detected_ranges += 1;
            }
        }

        let range_recall = if !true_ranges.is_empty() {
            detected_ranges as f64 / true_ranges.len() as f64
        } else {
            1.0
        };

        // Range-based precision: fraction of predicted ranges that overlap with true ranges
        let mut correct_pred_ranges = 0;
        for (pred_start, pred_end) in &pred_ranges {
            let correct = true_ranges.iter().any(|(true_start, true_end)| {
                // Check for overlap
                *pred_start <= *true_end && *pred_end >= *true_start
            });
            if correct {
                correct_pred_ranges += 1;
            }
        }

        let range_precision = if !pred_ranges.is_empty() {
            correct_pred_ranges as f64 / pred_ranges.len() as f64
        } else {
            1.0
        };

        Ok((range_precision, range_recall))
    }

    /// Find contiguous anomaly ranges
    fn find_anomaly_ranges(&self, labels: &Array1<i32>) -> Vec<(usize, usize)> {
        let mut ranges = Vec::new();
        let mut start = None;

        for (i, &label) in labels.iter().enumerate() {
            if label == 1 && start.is_none() {
                start = Some(i);
            } else if label == 0 && start.is_some() {
                ranges.push((start.unwrap(), i - 1));
                start = None;
            }
        }

        // Handle case where anomaly extends to end
        if let Some(start_idx) = start {
            ranges.push((start_idx, labels.len() - 1));
        }

        ranges
    }

    /// Calculate NAB (Numenta Anomaly Benchmark) score
    fn calculate_nab_score(&self, y_true: &Array1<i32>, y_pred: &Array1<i32>) -> Result<f64> {
        let mut score = 0.0;
        let true_ranges = self.find_anomaly_ranges(y_true);

        for (true_start, true_end) in true_ranges {
            let mut range_detected = false;

            // Check for detections in this range
            for i in true_start..=true_end {
                if y_pred[i] == 1 && !range_detected {
                    // First detection in this range
                    let detection_position = i;
                    let range_middle = (true_start + true_end) / 2;

                    if detection_position <= range_middle {
                        // Early detection bonus
                        score +=
                            self.nab_weights.true_positive + self.nab_weights.early_detection_bonus;
                    } else {
                        // Late detection penalty
                        score += self.nab_weights.true_positive
                            + self.nab_weights.late_detection_penalty;
                    }
                    range_detected = true;
                    // Additional detections in same range don't add score
                }
            }

            if !range_detected {
                // Missed anomaly
                score += self.nab_weights.false_negative;
            }
        }

        // Count false positives (detections outside true anomaly ranges)
        for (i, &pred_val) in y_pred.iter().enumerate() {
            if pred_val == 1 && y_true[i] == 0 {
                score += self.nab_weights.false_positive;
            }
        }

        Ok(score)
    }

    /// Calculate time-weighted precision
    fn calculate_time_weighted_precision(
        &self,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
        timestamps: &Array1<f64>,
    ) -> Result<f64> {
        // Simple implementation: weight by time gaps
        let mut weighted_tp = 0.0;
        let mut weighted_fp = 0.0;

        for i in 0..y_true.len() {
            let weight = if i > 0 {
                timestamps[i] - timestamps[i - 1]
            } else {
                1.0
            };

            match (y_true[i], y_pred[i]) {
                (1, 1) => weighted_tp += weight,
                (0, 1) => weighted_fp += weight,
                _ => {}
            }
        }

        if weighted_tp + weighted_fp > 0.0 {
            Ok(weighted_tp / (weighted_tp + weighted_fp))
        } else {
            Ok(0.0)
        }
    }

    /// Calculate average detection delay
    fn calculate_average_delay(
        &self,
        y_true: &Array1<i32>,
        y_pred: &Array1<i32>,
        timestamps: &Array1<f64>,
    ) -> Result<f64> {
        let true_ranges = self.find_anomaly_ranges(y_true);
        let mut delays = Vec::new();

        for (true_start, true_end) in true_ranges {
            // Find first detection in this range
            for i in true_start..=true_end {
                if y_pred[i] == 1 {
                    let delay = timestamps[i] - timestamps[true_start];
                    delays.push(delay);
                    break;
                }
            }
        }

        if delays.is_empty() {
            Ok(0.0)
        } else {
            Ok(delays.iter().sum::<f64>() / delays.len() as f64)
        }
    }
}

/// Complete anomaly detection metrics suite
pub struct AnomalyDetectionSuite {
    detection: DetectionMetrics,
    distribution: DistributionMetrics,
    time_series: TimeSeriesAnomalyMetrics,
}

impl Default for AnomalyDetectionSuite {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyDetectionSuite {
    /// Create a new anomaly detection metrics suite
    pub fn new() -> Self {
        Self {
            detection: DetectionMetrics::new(),
            distribution: DistributionMetrics::new(),
            time_series: TimeSeriesAnomalyMetrics::new(),
        }
    }

    /// Get detection metrics calculator
    pub fn detection(&self) -> &DetectionMetrics {
        &self.detection
    }

    /// Get distribution metrics calculator
    pub fn distribution(&self) -> &DistributionMetrics {
        &self.distribution
    }

    /// Get time series anomaly metrics calculator
    pub fn time_series(&self) -> &TimeSeriesAnomalyMetrics {
        &self.time_series
    }
}

impl DomainMetrics for AnomalyDetectionSuite {
    type Result = DomainEvaluationResult;

    fn domain_name(&self) -> &'static str {
        "Anomaly Detection"
    }

    fn available_metrics(&self) -> Vec<&'static str> {
        vec![
            "detection_accuracy",
            "detection_precision",
            "detection_recall",
            "detection_f1",
            "detection_auc_roc",
            "detection_auc_pr",
            "distribution_kl_divergence",
            "distribution_js_divergence",
            "distribution_wasserstein",
            "time_series_point_adjust_precision",
            "time_series_point_adjust_recall",
            "time_series_nab_score",
            "time_series_average_delay",
        ]
    }

    fn metric_descriptions(&self) -> HashMap<&'static str, &'static str> {
        let mut descriptions = HashMap::new();
        descriptions.insert("detection_accuracy", "Overall detection accuracy");
        descriptions.insert("detection_precision", "Precision for anomaly detection");
        descriptions.insert("detection_recall", "Recall for anomaly detection");
        descriptions.insert("detection_f1", "F1 score for anomaly detection");
        descriptions.insert("detection_auc_roc", "Area under ROC curve");
        descriptions.insert("detection_auc_pr", "Area under precision-recall curve");
        descriptions.insert(
            "distribution_kl_divergence",
            "Kullback-Leibler divergence between distributions",
        );
        descriptions.insert("distribution_js_divergence", "Jensen-Shannon divergence");
        descriptions.insert(
            "distribution_wasserstein",
            "Wasserstein distance between distributions",
        );
        descriptions.insert(
            "time_series_point_adjust_precision",
            "Point-adjust precision with tolerance window",
        );
        descriptions.insert(
            "time_series_point_adjust_recall",
            "Point-adjust recall with tolerance window",
        );
        descriptions.insert("time_series_nab_score", "Numenta Anomaly Benchmark score");
        descriptions.insert(
            "time_series_average_delay",
            "Average delay in anomaly detection",
        );
        descriptions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_metrics() {
        let metrics = DetectionMetrics::new().with_threshold(0.5);

        let y_true = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
        let y_score = Array1::from_vec(vec![0.2, 0.8, 0.9, 0.3, 0.7, 0.1]);

        let results = metrics.evaluate_detection(&y_true, &y_score).unwrap();

        assert!(results.accuracy >= 0.0 && results.accuracy <= 1.0);
        assert!(results.precision >= 0.0 && results.precision <= 1.0);
        assert!(results.recall >= 0.0 && results.recall <= 1.0);
        assert!(results.f1_score >= 0.0 && results.f1_score <= 1.0);
        assert!(results.auc_roc >= 0.0 && results.auc_roc <= 1.0);
    }

    #[test]
    fn test_distribution_metrics() {
        let metrics = DistributionMetrics::new();

        let normal_samples = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.1, 0.0, -0.1]);
        let anomaly_samples = Array1::from_vec(vec![2.0, 2.5, 3.0, 2.8, 2.2, 2.7]);

        let results = metrics
            .evaluate_distribution_anomalies(&normal_samples, &anomaly_samples)
            .unwrap();

        assert!(results.kl_divergence >= 0.0);
        assert!(results.js_divergence >= 0.0);
        assert!(results.wasserstein_distance >= 0.0);
        assert!(results.ks_statistic >= 0.0 && results.ks_statistic <= 1.0);
    }

    #[test]
    fn test_time_series_anomaly_metrics() {
        let metrics = TimeSeriesAnomalyMetrics::new().with_tolerance_window(2);

        let y_true = Array1::from_vec(vec![0, 0, 1, 1, 0, 0, 1, 0, 0]);
        let y_pred = Array1::from_vec(vec![0, 0, 0, 1, 0, 1, 1, 0, 0]);
        let timestamps = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        let results = metrics
            .evaluate_time_series_anomalies(&y_true, &y_pred, Some(&timestamps))
            .unwrap();

        assert!(results.precision >= 0.0 && results.precision <= 1.0);
        assert!(results.recall >= 0.0 && results.recall <= 1.0);
        assert!(results.point_adjust_precision >= 0.0 && results.point_adjust_precision <= 1.0);
        assert!(results.point_adjust_recall >= 0.0 && results.point_adjust_recall <= 1.0);
        assert!(results.average_delay >= 0.0);
    }

    #[test]
    fn test_anomaly_range_finding() {
        let metrics = TimeSeriesAnomalyMetrics::new();

        let labels = Array1::from_vec(vec![0, 1, 1, 0, 0, 1, 0, 1, 1, 1]);
        let ranges = metrics.find_anomaly_ranges(&labels);

        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0], (1, 2));
        assert_eq!(ranges[1], (5, 5));
        assert_eq!(ranges[2], (7, 9));
    }

    #[test]
    fn test_nab_score_calculation() {
        let metrics = TimeSeriesAnomalyMetrics::new();

        let y_true = Array1::from_vec(vec![0, 1, 1, 0, 0, 1, 0]);
        let y_pred = Array1::from_vec(vec![0, 1, 0, 0, 0, 1, 0]);

        let score = metrics.calculate_nab_score(&y_true, &y_pred).unwrap();

        // Should get positive score for detecting both anomaly ranges
        assert!(score > 0.0);
    }

    #[test]
    fn test_anomaly_detection_suite() {
        let suite = AnomalyDetectionSuite::new();

        assert_eq!(suite.domain_name(), "Anomaly Detection");
        assert!(!suite.available_metrics().is_empty());
        assert!(!suite.metric_descriptions().is_empty());
    }
}
