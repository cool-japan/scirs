use crate::error::{SignalError, SignalResult};
use crate::features::options::FeatureOptions;
use crate::features::{
    entropy::extract_entropy_features, peaks::extract_peak_features,
    spectral::extract_spectral_features, statistical::extract_statistical_features,
    trend::extract_trend_features, zero_crossing::extract_zero_crossing_features,
};
use ndarray::{s, Array2};
use num_traits::{Float, NumCast};
use std::collections::HashMap;
use std::fmt::Debug;

/// Run feature extraction on a time series
///
/// # Arguments
///
/// * `signal` - Input time series
/// * `options` - Feature extraction options
///
/// # Returns
///
/// * A HashMap containing the extracted features
///
/// # Examples
///
/// ```
/// use scirs2_signal::features::{extract_features, FeatureOptions};
/// use std::f64::consts::PI;
///
/// // Generate a sinusoidal signal
/// let signal: Vec<f64> = (0..1000)
///     .map(|i| (2.0 * PI * 10.0 * i as f64 / 1000.0).sin())
///     .collect();
///
/// // Configure feature extraction
/// let mut options = FeatureOptions::default();
/// options.sample_rate = Some(1000.0);
///
/// // Extract features
/// let features = extract_features(&signal, &options).unwrap();
///
/// // Access individual features
/// let mean = *features.get("mean").unwrap();
/// let std_dev = *features.get("std").unwrap();
/// let spectral_centroid = *features.get("spectral_centroid").unwrap();
///
/// // Mean should be close to zero for a sine wave
/// assert!(mean.abs() < 0.01);
/// // Standard deviation should be close to 1/sqrt(2) for a sine wave
/// assert!((std_dev - 1.0 / 2.0_f64.sqrt()).abs() < 0.01);
/// ```
pub fn extract_features<T>(
    signal: &[T],
    options: &FeatureOptions,
) -> SignalResult<HashMap<String, f64>>
where
    T: Float + NumCast + Debug,
{
    // Validate input
    if signal.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    // Convert input to f64
    let signal_f64: Vec<f64> = signal
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    let mut features = HashMap::new();

    // Extract statistical features
    if options.statistical {
        extract_statistical_features(&signal_f64, &mut features)?;
    }

    // Extract spectral features
    if options.spectral {
        extract_spectral_features(&signal_f64, options, &mut features)?;
    }

    // Extract entropy features
    if options.entropy {
        extract_entropy_features(&signal_f64, &mut features)?;
    }

    // Extract trend features
    if options.trend {
        extract_trend_features(&signal_f64, &mut features)?;
    }

    // Extract zero-crossing features
    if options.zero_crossings {
        extract_zero_crossing_features(&signal_f64, options, &mut features)?;
    }

    // Extract peak features
    if options.peaks {
        extract_peak_features(&signal_f64, &mut features)?;
    }

    Ok(features)
}

/// Extract all available features from multiple time series
///
/// # Arguments
///
/// * `signals` - Array of time series (each row is a time series)
/// * `options` - Feature extraction options
///
/// # Returns
///
/// * A 2D array where each row contains features for a time series,
///   and a vector of feature names corresponding to columns
///
/// # Examples
///
/// ```
/// use scirs2_signal::features::{extract_features_batch, FeatureOptions};
/// use std::f64::consts::PI;
/// use ndarray::Array2;
///
/// // Generate multiple signals
/// let n_signals = 5;
/// let n_samples = 1000;
/// let mut signals = Array2::zeros((n_signals, n_samples));
///
/// for i in 0..n_signals {
///     let freq = 5.0 + i as f64 * 2.0; // Different frequency for each signal
///     for j in 0..n_samples {
///         signals[[i, j]] = (2.0 * PI * freq * j as f64 / n_samples as f64).sin();
///     }
/// }
///
/// // Configure feature extraction
/// let mut options = FeatureOptions::default();
/// options.sample_rate = Some(1000.0);
///
/// // Extract features for all signals
/// let (feature_matrix, feature_names) = extract_features_batch(&signals, &options).unwrap();
///
/// // Check results
/// assert_eq!(feature_matrix.shape()[0], n_signals);
/// assert_eq!(feature_matrix.shape()[1], feature_names.len());
/// ```
pub fn extract_features_batch(
    signals: &Array2<f64>,
    options: &FeatureOptions,
) -> SignalResult<(Array2<f64>, Vec<String>)> {
    // Validate input
    if signals.is_empty() {
        return Err(SignalError::ValueError(
            "Input signals array is empty".to_string(),
        ));
    }

    let n_signals = signals.shape()[0];

    // Extract features for the first signal to get feature names
    let first_signal = signals.slice(s![0, ..]).to_vec();
    let first_features = extract_features(&first_signal, options)?;

    // Get feature names and sort them for consistent order
    let mut feature_names: Vec<String> = first_features.keys().cloned().collect();
    feature_names.sort();

    let n_features = feature_names.len();
    let mut feature_matrix = Array2::zeros((n_signals, n_features));

    // Set the first row
    for (i, name) in feature_names.iter().enumerate() {
        feature_matrix[[0, i]] = *first_features.get(name).unwrap();
    }

    // Process remaining signals
    for i in 1..n_signals {
        let signal = signals.slice(s![i, ..]).to_vec();
        let features = extract_features(&signal, options)?;

        for (j, name) in feature_names.iter().enumerate() {
            if let Some(&value) = features.get(name) {
                feature_matrix[[i, j]] = value;
            } else {
                // If a feature is missing (shouldn't happen), set to NaN
                feature_matrix[[i, j]] = f64::NAN;
            }
        }
    }

    Ok((feature_matrix, feature_names))
}
