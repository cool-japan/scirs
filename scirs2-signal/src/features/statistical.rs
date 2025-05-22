use crate::error::SignalResult;
use crate::measurements;
use std::collections::HashMap;

/// Extract statistical features from a time series
pub fn extract_statistical_features(
    signal: &[f64],
    features: &mut HashMap<String, f64>,
) -> SignalResult<()> {
    let n = signal.len();

    // Calculate basic statistics
    let sum: f64 = signal.iter().sum();
    let mean = sum / n as f64;
    features.insert("mean".to_string(), mean);

    // Calculate variance and std
    let sum_squared_diff: f64 = signal.iter().map(|&x| (x - mean).powi(2)).sum();
    let variance = sum_squared_diff / n as f64;
    let std_dev = variance.sqrt();
    features.insert("variance".to_string(), variance);
    features.insert("std".to_string(), std_dev);

    // Calculate median
    let mut sorted = signal.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    features.insert("median".to_string(), median);

    // Calculate min, max, range
    let min = *signal
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max = *signal
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    features.insert("min".to_string(), min);
    features.insert("max".to_string(), max);
    features.insert("range".to_string(), max - min);

    // Calculate higher-order moments
    let skewness = calculate_skewness(signal, mean, std_dev);
    let kurtosis = calculate_kurtosis(signal, mean, std_dev);
    features.insert("skewness".to_string(), skewness);
    features.insert("kurtosis".to_string(), kurtosis);

    // Calculate quantiles
    features.insert("q25".to_string(), calculate_quantile(&sorted, 0.25));
    features.insert("q75".to_string(), calculate_quantile(&sorted, 0.75));
    features.insert(
        "iqr".to_string(),
        calculate_quantile(&sorted, 0.75) - calculate_quantile(&sorted, 0.25),
    );

    // Calculate RMS
    let rms = measurements::rms(signal)?;
    features.insert("rms".to_string(), rms);

    // Calculate energy and power
    let energy: f64 = signal.iter().map(|&x| x * x).sum();
    let power = energy / n as f64;
    features.insert("energy".to_string(), energy);
    features.insert("power".to_string(), power);

    // Calculate mean absolute deviation
    let mad: f64 = signal.iter().map(|&x| (x - mean).abs()).sum::<f64>() / n as f64;
    features.insert("mad".to_string(), mad);

    // Calculate coefficient of variation
    if mean != 0.0 {
        features.insert("cv".to_string(), std_dev / mean.abs());
    } else {
        features.insert("cv".to_string(), f64::NAN);
    }

    Ok(())
}

/// Calculate skewness of a signal
pub fn calculate_skewness(signal: &[f64], mean: f64, std_dev: f64) -> f64 {
    if std_dev <= 0.0 || signal.len() < 3 {
        return 0.0;
    }

    let n = signal.len() as f64;
    let sum_cubed_diff: f64 = signal.iter().map(|&x| (x - mean).powi(3)).sum();

    sum_cubed_diff / ((n - 1.0) * std_dev.powi(3))
}

/// Calculate kurtosis of a signal
pub fn calculate_kurtosis(signal: &[f64], mean: f64, std_dev: f64) -> f64 {
    if std_dev <= 0.0 || signal.len() < 4 {
        return 0.0;
    }

    let n = signal.len() as f64;
    let sum_quartic_diff: f64 = signal.iter().map(|&x| (x - mean).powi(4)).sum();

    sum_quartic_diff / ((n - 1.0) * std_dev.powi(4)) - 3.0 // Excess kurtosis
}

/// Calculate a quantile (percentile) of a sorted array
pub fn calculate_quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }

    let pos = q * (sorted.len() - 1) as f64;
    let idx = pos.floor() as usize;
    let frac = pos - idx as f64;

    if idx + 1 < sorted.len() {
        sorted[idx] * (1.0 - frac) + sorted[idx + 1] * frac
    } else {
        sorted[idx]
    }
}

/// Calculate standard deviation
pub fn calculate_std(signal: &[f64]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }

    let n = signal.len();
    let mean = signal.iter().sum::<f64>() / n as f64;
    let variance = signal.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_statistical_features() {
        // Create a simple signal
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut features = HashMap::new();
        extract_statistical_features(&signal, &mut features).unwrap();

        // Check that basic statistics are calculated correctly
        assert_eq!(*features.get("mean").unwrap(), 3.0);
        assert_eq!(*features.get("median").unwrap(), 3.0);
        assert_eq!(*features.get("min").unwrap(), 1.0);
        assert_eq!(*features.get("max").unwrap(), 5.0);
        assert_eq!(*features.get("range").unwrap(), 4.0);

        // Variance should be (1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²) / 5 = 2
        assert!((features.get("variance").unwrap() - 2.0).abs() < 1e-10);
        assert!((features.get("std").unwrap() - 2.0_f64.sqrt()).abs() < 1e-10);
    }
}
