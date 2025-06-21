//! Anomaly detection algorithms for time series
//!
//! This module provides various algorithms for detecting anomalies and outliers
//! in time series data, including statistical process control, isolation forest,
//! one-class SVM, distance-based, and prediction-based approaches.

use ndarray::{s, Array1, Array2};
use num_traits::{Float, FromPrimitive, NumCast};
use rand::prelude::*;
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// Method for anomaly detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyMethod {
    /// Statistical process control (SPC)
    StatisticalProcessControl,
    /// Isolation forest for time series
    IsolationForest,
    /// One-class SVM for time series
    OneClassSVM,
    /// Distance-based anomaly detection
    DistanceBased,
    /// Prediction-based anomaly detection
    PredictionBased,
    /// Z-score based detection
    ZScore,
    /// Modified Z-score using median absolute deviation
    ModifiedZScore,
    /// Interquartile range (IQR) method
    InterquartileRange,
}

/// Statistical process control method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SPCMethod {
    /// Shewhart control charts
    Shewhart,
    /// CUSUM control charts
    CUSUM,
    /// Exponentially weighted moving average (EWMA)
    EWMA,
}

/// Distance metric for distance-based anomaly detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Mahalanobis distance
    Mahalanobis,
    /// Dynamic time warping distance
    DTW,
}

/// Options for anomaly detection
#[derive(Debug, Clone)]
pub struct AnomalyOptions {
    /// Detection method to use
    pub method: AnomalyMethod,
    /// Threshold for anomaly detection (interpretation depends on method)
    pub threshold: Option<f64>,
    /// Window size for local anomaly detection
    pub window_size: Option<usize>,
    /// Number of trees for isolation forest
    pub n_trees: usize,
    /// Subsampling size for isolation forest
    pub subsample_size: Option<usize>,
    /// SPC method (if using SPC)
    pub spc_method: SPCMethod,
    /// Alpha for EWMA (if using EWMA SPC)
    pub ewma_alpha: f64,
    /// Distance metric (if using distance-based method)
    pub distance_metric: DistanceMetric,
    /// Number of nearest neighbors (for distance-based methods)
    pub k_neighbors: usize,
    /// Contamination rate (expected fraction of anomalies)
    pub contamination: f64,
    /// Whether to use seasonal adjustment
    pub seasonal_adjustment: bool,
    /// Seasonal period (if using seasonal adjustment)
    pub seasonal_period: Option<usize>,
}

impl Default for AnomalyOptions {
    fn default() -> Self {
        Self {
            method: AnomalyMethod::StatisticalProcessControl,
            threshold: None,
            window_size: None,
            n_trees: 100,
            subsample_size: None,
            spc_method: SPCMethod::Shewhart,
            ewma_alpha: 0.3,
            distance_metric: DistanceMetric::Euclidean,
            k_neighbors: 5,
            contamination: 0.1,
            seasonal_adjustment: false,
            seasonal_period: None,
        }
    }
}

/// Result of anomaly detection
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Anomaly scores for each point (higher scores indicate more anomalous)
    pub scores: Array1<f64>,
    /// Binary indicators of anomalies (true = anomaly)
    pub is_anomaly: Array1<bool>,
    /// Threshold used for binary classification
    pub threshold: f64,
    /// Method used for detection
    pub method: AnomalyMethod,
    /// Additional information specific to the method
    pub method_info: Option<MethodInfo>,
}

/// Method-specific information
#[derive(Debug, Clone)]
pub enum MethodInfo {
    /// SPC-specific information
    SPC {
        /// Control limits (lower, upper)
        control_limits: (f64, f64),
        /// Center line value
        center_line: f64,
    },
    /// Isolation Forest-specific information
    IsolationForest {
        /// Average path length for normal points
        average_path_length: f64,
    },
    /// Distance-based information
    DistanceBased {
        /// Distance scores for each point
        distances: Array1<f64>,
    },
}

/// Detects anomalies in a time series
///
/// This function applies various anomaly detection algorithms to identify
/// points in the time series that deviate significantly from normal behavior.
///
/// # Arguments
///
/// * `ts` - The time series to analyze
/// * `options` - Options controlling the anomaly detection
///
/// # Returns
///
/// * A result containing anomaly scores and binary classifications
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_series::anomaly::{detect_anomalies, AnomalyOptions, AnomalyMethod};
///
/// // Create a time series with some anomalies
/// let mut ts = Array1::from_vec((0..100).map(|i| (i as f64 / 10.0).sin()).collect());
/// ts[25] = 5.0; // Anomaly
/// ts[75] = -5.0; // Anomaly
///
/// let options = AnomalyOptions {
///     method: AnomalyMethod::ZScore,
///     threshold: Some(3.0),
///     ..Default::default()
/// };
///
/// let result = detect_anomalies(&ts, &options).unwrap();
/// println!("Anomalies detected: {}", result.is_anomaly.iter().filter(|&&x| x).count());
/// ```
pub fn detect_anomalies<F>(ts: &Array1<F>, options: &AnomalyOptions) -> Result<AnomalyResult>
where
    F: Float + FromPrimitive + Debug + NumCast + std::iter::Sum,
{
    let n = ts.len();

    if n < 3 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Time series too short for anomaly detection".to_string(),
            required: 3,
            actual: n,
        });
    }

    // Apply seasonal adjustment if requested
    let adjusted_ts = if options.seasonal_adjustment {
        if let Some(period) = options.seasonal_period {
            seasonally_adjust(ts, period)?
        } else {
            ts.clone()
        }
    } else {
        ts.clone()
    };

    // Apply the selected anomaly detection method
    match options.method {
        AnomalyMethod::StatisticalProcessControl => detect_anomalies_spc(&adjusted_ts, options),
        AnomalyMethod::IsolationForest => detect_anomalies_isolation_forest(&adjusted_ts, options),
        AnomalyMethod::OneClassSVM => detect_anomalies_one_class_svm(&adjusted_ts, options),
        AnomalyMethod::DistanceBased => detect_anomalies_distance_based(&adjusted_ts, options),
        AnomalyMethod::PredictionBased => detect_anomalies_prediction_based(&adjusted_ts, options),
        AnomalyMethod::ZScore => detect_anomalies_zscore(&adjusted_ts, options),
        AnomalyMethod::ModifiedZScore => detect_anomalies_modified_zscore(&adjusted_ts, options),
        AnomalyMethod::InterquartileRange => detect_anomalies_iqr(&adjusted_ts, options),
    }
}

/// Statistical Process Control (SPC) anomaly detection
fn detect_anomalies_spc<F>(ts: &Array1<F>, options: &AnomalyOptions) -> Result<AnomalyResult>
where
    F: Float + FromPrimitive + Debug + NumCast + std::iter::Sum,
{
    let n = ts.len();
    let mut scores = Array1::zeros(n);
    let mut is_anomaly = Array1::from_elem(n, false);

    match options.spc_method {
        SPCMethod::Shewhart => {
            // Calculate control limits using the first portion of data
            let training_size = (n as f64 * 0.5).min(100.0) as usize;
            let training_data = ts.slice(s![0..training_size]);

            let mean = training_data
                .mean()
                .unwrap_or(F::zero())
                .to_f64()
                .unwrap_or(0.0);
            let std_dev = calculate_std_dev(&training_data.to_owned())
                .to_f64()
                .unwrap_or(1.0);

            let multiplier = 3.0; // 3-sigma control limits
            let ucl = mean + multiplier * std_dev; // Upper control limit
            let lcl = mean - multiplier * std_dev; // Lower control limit

            for i in 0..n {
                let value = ts[i].to_f64().unwrap_or(0.0);
                let distance_from_center = (value - mean).abs();
                scores[i] = distance_from_center / std_dev;
                is_anomaly[i] = value > ucl || value < lcl;
            }

            Ok(AnomalyResult {
                scores,
                is_anomaly,
                threshold: multiplier,
                method: AnomalyMethod::StatisticalProcessControl,
                method_info: Some(MethodInfo::SPC {
                    control_limits: (lcl, ucl),
                    center_line: mean,
                }),
            })
        }
        SPCMethod::CUSUM => {
            // CUSUM control chart implementation
            let training_size = (n as f64 * 0.5).min(100.0) as usize;
            let training_data = ts.slice(s![0..training_size]);

            let target = training_data
                .mean()
                .unwrap_or(F::zero())
                .to_f64()
                .unwrap_or(0.0);
            let std_dev = calculate_std_dev(&training_data.to_owned())
                .to_f64()
                .unwrap_or(1.0);

            let k = 0.5 * std_dev; // Reference value
            let h = 5.0 * std_dev; // Decision interval

            let mut cusum_pos = 0.0;
            let mut cusum_neg = 0.0;

            for i in 0..n {
                let value = ts[i].to_f64().unwrap_or(0.0);
                cusum_pos = f64::max(0.0, cusum_pos + (value - target) - k);
                cusum_neg = f64::max(0.0, cusum_neg - (value - target) - k);

                let cusum_max = f64::max(cusum_pos, cusum_neg);
                scores[i] = cusum_max / std_dev;
                is_anomaly[i] = cusum_max > h;
            }

            Ok(AnomalyResult {
                scores,
                is_anomaly,
                threshold: h / std_dev,
                method: AnomalyMethod::StatisticalProcessControl,
                method_info: Some(MethodInfo::SPC {
                    control_limits: (-h, h),
                    center_line: target,
                }),
            })
        }
        SPCMethod::EWMA => {
            // EWMA control chart implementation
            let alpha = options.ewma_alpha;
            let training_size = (n as f64 * 0.5).min(100.0) as usize;
            let training_data = ts.slice(s![0..training_size]);

            let target = training_data
                .mean()
                .unwrap_or(F::zero())
                .to_f64()
                .unwrap_or(0.0);
            let sigma = calculate_std_dev(&training_data.to_owned())
                .to_f64()
                .unwrap_or(1.0);

            let mut ewma = target;
            let l = 3.0; // Control limit multiplier

            for i in 0..n {
                let value = ts[i].to_f64().unwrap_or(0.0);
                ewma = alpha * value + (1.0 - alpha) * ewma;

                let ewma_variance = sigma * sigma * alpha / (2.0 - alpha)
                    * (1.0 - (1.0 - alpha).powi(2 * (i as i32 + 1)));
                let ewma_std = ewma_variance.sqrt();

                let ucl = target + l * ewma_std;
                let lcl = target - l * ewma_std;

                scores[i] = (ewma - target).abs() / ewma_std;
                is_anomaly[i] = ewma > ucl || ewma < lcl;
            }

            Ok(AnomalyResult {
                scores,
                is_anomaly,
                threshold: l,
                method: AnomalyMethod::StatisticalProcessControl,
                method_info: Some(MethodInfo::SPC {
                    control_limits: (target - l * sigma, target + l * sigma),
                    center_line: target,
                }),
            })
        }
    }
}

/// Isolation Forest anomaly detection (simplified version)
fn detect_anomalies_isolation_forest<F>(
    ts: &Array1<F>,
    options: &AnomalyOptions,
) -> Result<AnomalyResult>
where
    F: Float + FromPrimitive + Debug + NumCast + std::iter::Sum,
{
    let n = ts.len();
    let n_trees = options.n_trees;
    let subsample_size = options
        .subsample_size
        .unwrap_or((n as f64 * 0.5).min(256.0) as usize);

    // Convert to sliding windows for multivariate representation
    let window_size = options.window_size.unwrap_or(10.min(n / 4));
    let windowed_data = create_sliding_windows(ts, window_size)?;
    let n_windows = windowed_data.nrows();

    let mut path_lengths = Array1::<f64>::zeros(n_windows);
    let mut rng = rand::rng();

    // Build isolation trees
    for _ in 0..n_trees {
        let tree_path_lengths = build_isolation_tree(&windowed_data, subsample_size, &mut rng)?;
        for i in 0..n_windows {
            path_lengths[i] += tree_path_lengths[i];
        }
    }

    // Average path lengths
    path_lengths.mapv_inplace(|x| x / n_trees as f64);

    // Calculate expected path length for normal data
    let c_n = if subsample_size > 2 {
        2.0 * (subsample_size as f64 - 1.0).ln() + 0.5772156649
            - 2.0 * (subsample_size - 1) as f64 / subsample_size as f64
    } else {
        1.0
    };

    // Calculate anomaly scores (higher for anomalies)
    let mut scores = Array1::zeros(n);
    let mut anomaly_scores = Array1::zeros(n_windows);

    for i in 0..n_windows {
        let score = 2.0_f64.powf(-path_lengths[i] / c_n);
        anomaly_scores[i] = score;
    }

    // Map window scores back to time series
    for i in 0..n {
        let window_idx = if i >= window_size {
            i - window_size + 1
        } else {
            0
        }
        .min(n_windows - 1);
        scores[i] = anomaly_scores[window_idx];
    }

    // Determine threshold and anomalies
    let threshold = determine_threshold(&scores, options.contamination);
    let is_anomaly = scores.mapv(|x| x > threshold);

    Ok(AnomalyResult {
        scores,
        is_anomaly,
        threshold,
        method: AnomalyMethod::IsolationForest,
        method_info: Some(MethodInfo::IsolationForest {
            average_path_length: path_lengths.mean().unwrap_or(0.0),
        }),
    })
}

/// Z-score based anomaly detection
fn detect_anomalies_zscore<F>(ts: &Array1<F>, options: &AnomalyOptions) -> Result<AnomalyResult>
where
    F: Float + FromPrimitive + Debug + NumCast + std::iter::Sum,
{
    let n = ts.len();
    let mean = ts.mean().unwrap_or(F::zero()).to_f64().unwrap_or(0.0);
    let std_dev = calculate_std_dev(ts).to_f64().unwrap_or(1.0);

    let threshold = options.threshold.unwrap_or(3.0);

    let mut scores = Array1::zeros(n);
    let mut is_anomaly = Array1::from_elem(n, false);

    for i in 0..n {
        let value = ts[i].to_f64().unwrap_or(0.0);
        let zscore = (value - mean).abs() / std_dev;
        scores[i] = zscore;
        is_anomaly[i] = zscore > threshold;
    }

    Ok(AnomalyResult {
        scores,
        is_anomaly,
        threshold,
        method: AnomalyMethod::ZScore,
        method_info: None,
    })
}

/// Modified Z-score using median absolute deviation
fn detect_anomalies_modified_zscore<F>(
    ts: &Array1<F>,
    options: &AnomalyOptions,
) -> Result<AnomalyResult>
where
    F: Float + FromPrimitive + Debug + NumCast + std::iter::Sum,
{
    let n = ts.len();
    let threshold = options.threshold.unwrap_or(3.5);

    // Calculate median
    let mut sorted_values: Vec<f64> = ts.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if n % 2 == 0 {
        (sorted_values[n / 2 - 1] + sorted_values[n / 2]) / 2.0
    } else {
        sorted_values[n / 2]
    };

    // Calculate MAD (Median Absolute Deviation)
    let mut abs_deviations: Vec<f64> = ts
        .iter()
        .map(|&x| (x.to_f64().unwrap_or(0.0) - median).abs())
        .collect();
    abs_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mad = if n % 2 == 0 {
        (abs_deviations[n / 2 - 1] + abs_deviations[n / 2]) / 2.0
    } else {
        abs_deviations[n / 2]
    };

    // Scale MAD for consistency with normal distribution
    let mad_scaled = mad / 0.6745;

    let mut scores = Array1::zeros(n);
    let mut is_anomaly = Array1::from_elem(n, false);

    for i in 0..n {
        let value = ts[i].to_f64().unwrap_or(0.0);
        let modified_zscore = if mad_scaled > 1e-10 {
            0.6745 * (value - median) / mad
        } else {
            0.0
        };
        scores[i] = modified_zscore.abs();
        is_anomaly[i] = modified_zscore.abs() > threshold;
    }

    Ok(AnomalyResult {
        scores,
        is_anomaly,
        threshold,
        method: AnomalyMethod::ModifiedZScore,
        method_info: None,
    })
}

/// Interquartile Range (IQR) anomaly detection
fn detect_anomalies_iqr<F>(ts: &Array1<F>, options: &AnomalyOptions) -> Result<AnomalyResult>
where
    F: Float + FromPrimitive + Debug + NumCast + std::iter::Sum,
{
    let n = ts.len();
    let multiplier = options.threshold.unwrap_or(1.5);

    // Calculate quartiles
    let mut sorted_values: Vec<f64> = ts.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q1_idx = n / 4;
    let q3_idx = 3 * n / 4;
    let q1 = sorted_values[q1_idx];
    let q3 = sorted_values[q3_idx];
    let iqr = q3 - q1;

    let lower_bound = q1 - multiplier * iqr;
    let upper_bound = q3 + multiplier * iqr;

    let mut scores = Array1::zeros(n);
    let mut is_anomaly = Array1::from_elem(n, false);

    for i in 0..n {
        let value = ts[i].to_f64().unwrap_or(0.0);
        let score = if value < lower_bound {
            (lower_bound - value) / iqr
        } else if value > upper_bound {
            (value - upper_bound) / iqr
        } else {
            0.0
        };
        scores[i] = score;
        is_anomaly[i] = value < lower_bound || value > upper_bound;
    }

    Ok(AnomalyResult {
        scores,
        is_anomaly,
        threshold: multiplier,
        method: AnomalyMethod::InterquartileRange,
        method_info: None,
    })
}

/// Placeholder implementations for complex methods
fn detect_anomalies_one_class_svm<F>(
    ts: &Array1<F>,
    options: &AnomalyOptions,
) -> Result<AnomalyResult>
where
    F: Float + FromPrimitive + Debug + NumCast + std::iter::Sum,
{
    // Simplified implementation using distance-based approach as a substitute
    detect_anomalies_distance_based(ts, options)
}

fn detect_anomalies_distance_based<F>(
    ts: &Array1<F>,
    options: &AnomalyOptions,
) -> Result<AnomalyResult>
where
    F: Float + FromPrimitive + Debug + NumCast + std::iter::Sum,
{
    let n = ts.len();
    let window_size = options.window_size.unwrap_or(10.min(n / 4));
    let k = options.k_neighbors;

    // Create sliding windows
    let windowed_data = create_sliding_windows(ts, window_size)?;
    let n_windows = windowed_data.nrows();

    let mut distances = Array1::zeros(n_windows);

    // Calculate average distance to k nearest neighbors for each window
    for i in 0..n_windows {
        let mut window_distances = Vec::new();
        let current_window = windowed_data.row(i);

        for j in 0..n_windows {
            if i != j {
                let other_window = windowed_data.row(j);
                let dist = euclidean_distance(&current_window.to_owned(), &other_window.to_owned());
                window_distances.push(dist);
            }
        }

        window_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let avg_k_distance: f64 = window_distances.iter().take(k).sum::<f64>() / k as f64;
        distances[i] = avg_k_distance;
    }

    // Map back to time series
    let mut scores = Array1::zeros(n);
    for i in 0..n {
        let window_idx = if i >= window_size {
            i - window_size + 1
        } else {
            0
        }
        .min(n_windows - 1);
        scores[i] = distances[window_idx];
    }

    let threshold = determine_threshold(&scores, options.contamination);
    let is_anomaly = scores.mapv(|x| x > threshold);

    Ok(AnomalyResult {
        scores,
        is_anomaly,
        threshold,
        method: AnomalyMethod::DistanceBased,
        method_info: Some(MethodInfo::DistanceBased {
            distances: distances.to_owned(),
        }),
    })
}

fn detect_anomalies_prediction_based<F>(
    ts: &Array1<F>,
    _options: &AnomalyOptions,
) -> Result<AnomalyResult>
where
    F: Float + FromPrimitive + Debug + NumCast + std::iter::Sum,
{
    // Simplified prediction-based approach using moving average prediction
    let n = ts.len();
    let window_size = 10.min(n / 4);

    let mut scores = Array1::zeros(n);
    let mut is_anomaly = Array1::from_elem(n, false);

    for i in window_size..n {
        // Calculate moving average as prediction
        let window = ts.slice(s![i - window_size..i]);
        let prediction = window.mean().unwrap_or(F::zero()).to_f64().unwrap_or(0.0);
        let actual = ts[i].to_f64().unwrap_or(0.0);

        // Calculate prediction error
        let error = (actual - prediction).abs();
        scores[i] = error;
    }

    // Calculate threshold based on prediction errors
    let valid_scores: Vec<f64> = scores.iter().skip(window_size).copied().collect();

    if !valid_scores.is_empty() {
        let mean_error = valid_scores.iter().sum::<f64>() / valid_scores.len() as f64;
        let std_error = {
            let variance = valid_scores
                .iter()
                .map(|&x| (x - mean_error).powi(2))
                .sum::<f64>()
                / valid_scores.len() as f64;
            variance.sqrt()
        };

        let threshold = mean_error + 3.0 * std_error;

        for i in window_size..n {
            is_anomaly[i] = scores[i] > threshold;
        }

        Ok(AnomalyResult {
            scores,
            is_anomaly,
            threshold,
            method: AnomalyMethod::PredictionBased,
            method_info: None,
        })
    } else {
        Err(TimeSeriesError::InsufficientData {
            message: "Not enough data for prediction-based anomaly detection".to_string(),
            required: window_size + 1,
            actual: n,
        })
    }
}

// Helper functions

fn seasonally_adjust<F>(ts: &Array1<F>, period: usize) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + NumCast + std::iter::Sum,
{
    let n = ts.len();
    if n < period * 2 {
        return Ok(ts.clone());
    }

    let mut adjusted = ts.clone();

    // Simple seasonal adjustment using period-wise detrending
    for season in 0..period {
        let mut seasonal_values = Vec::new();
        let mut indices = Vec::new();

        for i in (season..n).step_by(period) {
            seasonal_values.push(ts[i]);
            indices.push(i);
        }

        if seasonal_values.len() > 1 {
            let seasonal_mean = seasonal_values.iter().cloned().sum::<F>()
                / F::from_usize(seasonal_values.len()).unwrap();

            for &idx in &indices {
                adjusted[idx] = adjusted[idx] - seasonal_mean;
            }
        }
    }

    Ok(adjusted)
}

fn create_sliding_windows<F>(ts: &Array1<F>, window_size: usize) -> Result<Array2<f64>>
where
    F: Float + FromPrimitive + Debug + NumCast,
{
    let n = ts.len();
    if n < window_size {
        return Err(TimeSeriesError::InsufficientData {
            message: "Time series too short for windowing".to_string(),
            required: window_size,
            actual: n,
        });
    }

    let n_windows = n - window_size + 1;
    let mut windows = Array2::zeros((n_windows, window_size));

    for i in 0..n_windows {
        for j in 0..window_size {
            windows[[i, j]] = ts[i + j].to_f64().unwrap_or(0.0);
        }
    }

    Ok(windows)
}

fn build_isolation_tree(
    data: &Array2<f64>,
    subsample_size: usize,
    rng: &mut ThreadRng,
) -> Result<Array1<f64>> {
    let n_samples = data.nrows();
    let _n_features = data.ncols();

    // Subsample data
    let actual_subsample_size = subsample_size.min(n_samples);
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(rng);
    let _subsample_indices = &indices[0..actual_subsample_size];

    let mut path_lengths = Array1::zeros(n_samples);

    // Build tree using subsample, but calculate path lengths for all points
    for idx in 0..n_samples {
        path_lengths[idx] =
            calculate_isolation_path_length(&data.row(idx).to_owned(), data, 0, rng);
    }

    Ok(path_lengths)
}

fn calculate_isolation_path_length(
    point: &Array1<f64>,
    data: &Array2<f64>,
    depth: usize,
    rng: &mut ThreadRng,
) -> f64 {
    const MAX_DEPTH: usize = 20; // Prevent infinite recursion

    if depth >= MAX_DEPTH || data.nrows() <= 1 {
        return depth as f64;
    }

    // Randomly select a feature and split value
    let feature_idx = rng.random_range(0..data.ncols());
    let feature_values: Vec<f64> = data.column(feature_idx).to_vec();
    let min_val = feature_values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = feature_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    if (max_val - min_val).abs() < 1e-10 {
        return depth as f64; // No variation in this feature
    }

    // Simplified: return path length based on how far the point is from the mean
    let mean_val = feature_values.iter().sum::<f64>() / feature_values.len() as f64;
    let deviation = (point[feature_idx] - mean_val).abs();
    let max_deviation = (max_val - min_val) / 2.0;

    // Anomalies (points far from mean) should have shorter path lengths
    if max_deviation > 0.0 {
        let normalized_deviation = deviation / max_deviation;
        // Points far from mean get shorter paths (are isolated faster)
        depth as f64 + (1.0 - normalized_deviation.min(1.0))
    } else {
        depth as f64 + 1.0
    }
}

fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn determine_threshold(scores: &Array1<f64>, contamination: f64) -> f64 {
    let mut sorted_scores: Vec<f64> = scores.to_vec();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let threshold_idx = ((1.0 - contamination) * sorted_scores.len() as f64) as usize;
    sorted_scores[threshold_idx.min(sorted_scores.len() - 1)]
}

fn calculate_std_dev<F>(data: &Array1<F>) -> F
where
    F: Float + FromPrimitive + Debug + NumCast + std::iter::Sum,
{
    let n = data.len();
    if n <= 1 {
        return F::zero();
    }

    let mean = data.mean().unwrap_or(F::zero());
    let variance =
        data.iter().map(|&x| (x - mean) * (x - mean)).sum::<F>() / F::from_usize(n - 1).unwrap();

    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_zscore_anomaly_detection() {
        // Create a time series with clear anomalies
        let mut ts = Array1::from_vec((0..100).map(|i| (i as f64 / 10.0).sin()).collect());
        ts[25] = 5.0; // Clear anomaly
        ts[75] = -5.0; // Clear anomaly

        let options = AnomalyOptions {
            method: AnomalyMethod::ZScore,
            threshold: Some(3.0),
            ..Default::default()
        };

        let result = detect_anomalies(&ts, &options).unwrap();

        // Should detect the two anomalies
        let anomaly_count = result.is_anomaly.iter().filter(|&&x| x).count();
        assert!(
            anomaly_count >= 2,
            "Should detect at least 2 anomalies, found {}",
            anomaly_count
        );

        // Anomalies should have high scores
        assert!(result.scores[25] > 3.0);
        assert!(result.scores[75] > 3.0);
    }

    #[test]
    fn test_modified_zscore() {
        let ts = array![1.0, 2.0, 1.5, 2.2, 1.8, 10.0, 2.1, 1.9]; // 10.0 is an outlier

        let options = AnomalyOptions {
            method: AnomalyMethod::ModifiedZScore,
            threshold: Some(3.5),
            ..Default::default()
        };

        let result = detect_anomalies(&ts, &options).unwrap();

        // Should detect the outlier at index 5
        assert!(result.is_anomaly[5], "Should detect anomaly at index 5");
        assert!(result.scores[5] > 3.5);
    }

    #[test]
    fn test_iqr_anomaly_detection() {
        let mut ts = Array1::from_elem(100, 1.0);
        ts[50] = 10.0; // Clear outlier

        let options = AnomalyOptions {
            method: AnomalyMethod::InterquartileRange,
            threshold: Some(1.5),
            ..Default::default()
        };

        let result = detect_anomalies(&ts, &options).unwrap();

        // Should detect the outlier
        assert!(result.is_anomaly[50], "Should detect anomaly at index 50");
    }

    #[test]
    fn test_spc_shewhart() {
        // Create a time series with a shift in mean
        let mut ts = Array1::zeros(100);
        for i in 0..50 {
            ts[i] = 1.0 + 0.1 * (i as f64 * 0.1).sin();
        }
        for i in 50..100 {
            ts[i] = 5.0 + 0.1 * (i as f64 * 0.1).sin(); // Shift in mean
        }

        let options = AnomalyOptions {
            method: AnomalyMethod::StatisticalProcessControl,
            spc_method: SPCMethod::Shewhart,
            ..Default::default()
        };

        let result = detect_anomalies(&ts, &options).unwrap();

        // Should detect anomalies in the second half
        let anomalies_second_half = result
            .is_anomaly
            .slice(s![50..])
            .iter()
            .filter(|&&x| x)
            .count();
        assert!(
            anomalies_second_half > 10,
            "Should detect many anomalies in second half"
        );
    }

    #[test]
    fn test_isolation_forest() {
        let mut ts = Array1::from_vec((0..50).map(|i| (i as f64 / 5.0).sin()).collect());
        ts[25] = 10.0; // Anomaly

        let options = AnomalyOptions {
            method: AnomalyMethod::IsolationForest,
            contamination: 0.1,
            window_size: Some(5),
            n_trees: 10, // Fewer trees for faster testing
            ..Default::default()
        };

        let result = detect_anomalies(&ts, &options).unwrap();

        // Should detect some anomalies
        let anomaly_count = result.is_anomaly.iter().filter(|&&x| x).count();
        assert!(anomaly_count > 0, "Should detect at least one anomaly");
    }

    #[test]
    fn test_edge_cases() {
        // Test with very short time series
        let ts = array![1.0, 2.0];
        let options = AnomalyOptions::default();

        let result = detect_anomalies(&ts, &options);
        assert!(result.is_err());

        // Test with constant time series
        let ts = Array1::from_elem(50, 1.0);
        let options = AnomalyOptions {
            method: AnomalyMethod::ZScore,
            threshold: Some(3.0),
            ..Default::default()
        };

        let result = detect_anomalies(&ts, &options).unwrap();
        // Should detect no anomalies in constant series
        let anomaly_count = result.is_anomaly.iter().filter(|&&x| x).count();
        assert_eq!(
            anomaly_count, 0,
            "Should detect no anomalies in constant series"
        );
    }
}
