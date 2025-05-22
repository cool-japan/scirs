use crate::error::SignalResult;
use crate::features::options::FeatureOptions;
use std::collections::HashMap;

/// Extract zero-crossing-based features from a time series
pub fn extract_zero_crossing_features(
    signal: &[f64],
    options: &FeatureOptions,
    features: &mut HashMap<String, f64>,
) -> SignalResult<()> {
    let n = signal.len();

    // Count zero crossings
    let mut zero_crossings = 0;
    for i in 1..n {
        if (signal[i] >= 0.0 && signal[i - 1] < 0.0) || (signal[i] < 0.0 && signal[i - 1] >= 0.0) {
            zero_crossings += 1;
        }
    }

    features.insert("zero_crossings".to_string(), zero_crossings as f64);

    // Calculate zero crossing rate
    features.insert(
        "zero_crossing_rate".to_string(),
        zero_crossings as f64 / n as f64,
    );

    // If sample rate is provided, estimate frequency from zero crossings
    if let Some(fs) = options.sample_rate {
        // Frequency is half the zero crossing rate (since each cycle has 2 crossings)
        let frequency_estimate = (zero_crossings as f64) * fs / (2.0 * n as f64);
        features.insert("zero_crossing_frequency".to_string(), frequency_estimate);
    }

    // Calculate mean crossing rate
    let mean = signal.iter().sum::<f64>() / n as f64;
    let mut mean_crossings = 0;
    for i in 1..n {
        if (signal[i] >= mean && signal[i - 1] < mean)
            || (signal[i] < mean && signal[i - 1] >= mean)
        {
            mean_crossings += 1;
        }
    }

    features.insert("mean_crossings".to_string(), mean_crossings as f64);
    features.insert(
        "mean_crossing_rate".to_string(),
        mean_crossings as f64 / n as f64,
    );

    Ok(())
}
