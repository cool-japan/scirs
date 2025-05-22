use crate::error::SignalResult;
use crate::measurements;
use std::collections::HashMap;

/// Extract peak-based features from a time series
pub fn extract_peak_features(
    signal: &[f64],
    features: &mut HashMap<String, f64>,
) -> SignalResult<()> {
    let n = signal.len();

    // Find local maxima and minima
    let mut maxima = Vec::new();
    let mut minima = Vec::new();

    // First point
    if n > 1 && signal[0] > signal[1] {
        maxima.push(0);
    } else if n > 1 && signal[0] < signal[1] {
        minima.push(0);
    }

    // Middle points
    for i in 1..n - 1 {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            maxima.push(i);
        } else if signal[i] < signal[i - 1] && signal[i] < signal[i + 1] {
            minima.push(i);
        }
    }

    // Last point
    if n > 1 && signal[n - 1] > signal[n - 2] {
        maxima.push(n - 1);
    } else if n > 1 && signal[n - 1] < signal[n - 2] {
        minima.push(n - 1);
    }

    // Calculate peak features
    features.insert("num_peaks".to_string(), maxima.len() as f64);
    features.insert("num_troughs".to_string(), minima.len() as f64);
    features.insert(
        "peak_trough_ratio".to_string(),
        if !minima.is_empty() {
            maxima.len() as f64 / minima.len() as f64
        } else {
            f64::MAX
        },
    );

    // Calculate peak-to-peak amplitude (max - min)
    if !signal.is_empty() {
        let min = signal.iter().copied().fold(f64::INFINITY, f64::min);
        let max = signal.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        features.insert("peak_to_peak".to_string(), max - min);
    }

    // Calculate peak density (peaks per sample)
    features.insert("peak_density".to_string(), maxima.len() as f64 / n as f64);

    // If at least two peaks, calculate mean peak distance
    if maxima.len() >= 2 {
        let mut peak_distances = Vec::with_capacity(maxima.len() - 1);
        for i in 1..maxima.len() {
            peak_distances.push((maxima[i] - maxima[i - 1]) as f64);
        }

        let mean_peak_distance = peak_distances.iter().sum::<f64>() / peak_distances.len() as f64;
        features.insert("mean_peak_distance".to_string(), mean_peak_distance);

        // Calculate std of peak distances to measure irregularity
        if peak_distances.len() >= 2 {
            let mean = mean_peak_distance;
            let variance = peak_distances
                .iter()
                .map(|&d| (d - mean).powi(2))
                .sum::<f64>()
                / peak_distances.len() as f64;

            features.insert("peak_distance_std".to_string(), variance.sqrt());
        }
    }

    // Calculate crest factor (peak amplitude / RMS)
    if let Ok(rms) = measurements::rms(signal) {
        if rms > 0.0 {
            let max_abs = signal.iter().map(|&x| x.abs()).fold(0.0, f64::max);
            features.insert("crest_factor".to_string(), max_abs / rms);
        }
    }

    Ok(())
}
