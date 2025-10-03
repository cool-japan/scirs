use crate::error::SignalResult;
use crate::features::batch::extract_features;
use crate::features::options::FeatureOptions;
use scirs2_core::numeric::{Float, NumCast};
use std::collections::HashMap;
use std::fmt::Debug;

#[allow(unused_imports)]
/// Calculate features commonly used for activity recognition
///
/// This is a convenience function that extracts a specific set of features
/// that are commonly used for human activity recognition.
///
/// # Arguments
///
/// * `signal` - Input time series
/// * `fs` - Sampling frequency
///
/// # Returns
///
/// * A HashMap containing the extracted features
#[allow(dead_code)]
pub fn activity_recognition_features<T>(
    _signal: &[T],
    fs: f64,
) -> SignalResult<HashMap<String, f64>>
where
    T: Float + NumCast + Debug,
{
    let options = FeatureOptions {
        sample_rate: Some(fs),
        spectral: true,
        entropy: true,
        zero_crossings: true,
        peaks: true,
        ..Default::default()
    };

    let mut features = extract_features(_signal, &options)?;

    // Calculate additional features specific to activity recognition
    // 1. Signal magnitude area
    if let Some(signal_f64) = _signal
        .iter()
        .map(|&val| NumCast::from(val))
        .collect::<Option<Vec<f64>>>()
    {
        let sma = signal_f64.iter().map(|&x: &f64| x.abs()).sum::<f64>() / signal_f64.len() as f64;
        features.insert("signal_magnitude_area".to_string(), sma);
    }

    // 2. Energy in specific frequency bands (e.g., for accelerometer data)
    let bands = [(0.0, 1.0), (1.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, 12.0)];

    if let Some(&energy_ratio) = features.get("low_freq_energy_ratio") {
        for (low, high) in bands.iter() {
            features.insert(
                format!("band_{}_{}Hz_energy", low, high),
                energy_ratio * (1.0 - (*low / bands[bands.len() - 1].1).powi(2)),
            );
        }
    }

    // 3. Autocorrelation features
    if let Some(signal_f64) = _signal
        .iter()
        .map(|&val| NumCast::from(val))
        .collect::<Option<Vec<f64>>>()
    {
        let max_lag = (0.5 * fs).min((signal_f64.len() as f64) / 3.0) as usize;
        let autocorr = calculate_autocorrelation(&signal_f64, max_lag);

        // Peak in autocorrelation
        let (peak_lag, peak_value) = find_first_peak(&autocorr);

        features.insert("autocorr_peak_lag".to_string(), peak_lag as f64);
        features.insert("autocorr_peak_value".to_string(), peak_value);

        // Autocorrelation at specific lags (e.g., 1/4, 1/2, 3/4 of max_lag)
        for d in [0.25, 0.5, 0.75] {
            let lag = (d * max_lag as f64) as usize;
            if lag < autocorr.len() {
                features.insert(format!("autocorr_lag_{:.2}", d), autocorr[lag]);
            }
        }
    }

    Ok(features)
}

/// Calculate autocorrelation up to a given lag
#[allow(dead_code)]
fn calculate_autocorrelation(_signal: &[f64], maxlag: usize) -> Vec<f64> {
    let n = signal.len();
    let mean = signal.iter().sum::<f64>() / n as f64;

    // Subtract mean for unbiased correlation
    let signal_centered: Vec<f64> = signal.iter().map(|&x| x - mean).collect();

    // Calculate autocorrelation for lags from 0 to max_lag
    let mut autocorr = Vec::with_capacity(max_lag + 1);

    // Autocorrelation at _lag 0 (variance)
    let variance = signal_centered.iter().map(|&x| x * x).sum::<f64>() / n as f64;
    autocorr.push(1.0); // Normalized autocorrelation at _lag 0 is always 1

    // Autocorrelation for lags 1 to max_lag
    for _lag in 1..=max_lag {
        if _lag >= n {
            autocorr.push(0.0);
            continue;
        }

        let mut sum = 0.0;
        for i in 0..n - _lag {
            sum += signal_centered[i] * signal_centered[i + _lag];
        }

        // Normalize by variance
        let corr = if variance > 0.0 {
            sum / (variance * (n - lag) as f64)
        } else {
            0.0
        };
        autocorr.push(corr);
    }

    autocorr
}

/// Find the first peak in a vector
#[allow(dead_code)]
fn find_first_peak(signal: &[f64]) -> (usize, f64) {
    if signal.len() <= 2 {
        return (0, if signal.is_empty() { 0.0 } else { signal[0] });
    }

    // Skip first point as it's always 1.0 for autocorrelation
    for i in 2.._signal.len() - 1 {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] > 0.1 {
            return (i, signal[i]);
        }
    }

    (0, signal[0])
}
