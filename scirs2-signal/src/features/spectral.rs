use crate::error::SignalResult;
use crate::features::options::FeatureOptions;
use crate::periodogram;
use crate::utilities::spectral;
use std::collections::HashMap;

/// Extract spectral features from a time series
pub fn extract_spectral_features(
    signal: &[f64],
    options: &FeatureOptions,
    features: &mut HashMap<String, f64>,
) -> SignalResult<()> {
    // Default to 1.0 Hz if not provided
    let fs = options.sample_rate.unwrap_or(1.0);

    // Compute the periodogram
    let (freqs, psd) = periodogram(
        signal,
        Some(fs),
        Some("hann"),
        None,
        Some("constant"),
        Some("density"),
    )?;

    // Calculate spectral features
    let total_power: f64 = psd.iter().sum();

    // Handle potential zero power case
    if total_power <= 0.0 {
        // Insert zero or small value for all spectral features
        features.insert("spectral_power".to_string(), 0.0);
        features.insert("spectral_centroid".to_string(), 0.0);
        features.insert("spectral_spread".to_string(), 0.0);
        features.insert("spectral_skewness".to_string(), 0.0);
        features.insert("spectral_kurtosis".to_string(), 0.0);
        features.insert("spectral_flatness".to_string(), 0.0);
        features.insert("spectral_rolloff".to_string(), 0.0);
        features.insert("spectral_slope".to_string(), 0.0);
        features.insert("spectral_decrease".to_string(), 0.0);
        features.insert("dominant_frequency".to_string(), 0.0);
        features.insert("dominant_frequency_magnitude".to_string(), 0.0);

        if freqs.len() > 10 {
            features.insert("low_freq_energy_ratio".to_string(), 0.0);
            features.insert("mid_freq_energy_ratio".to_string(), 0.0);
            features.insert("high_freq_energy_ratio".to_string(), 0.0);
        }

        return Ok(());
    }

    features.insert("spectral_power".to_string(), total_power);

    // Calculate spectral centroid
    let centroid = spectral::spectral_centroid(&freqs, &psd)?;
    features.insert("spectral_centroid".to_string(), centroid);

    // Calculate spectral spread
    let spread = spectral::spectral_spread(&freqs, &psd, Some(centroid))?;
    features.insert("spectral_spread".to_string(), spread);

    // Calculate spectral skewness
    let skew = spectral::spectral_skewness(&freqs, &psd, Some(centroid), Some(spread))?;
    features.insert("spectral_skewness".to_string(), skew);

    // Calculate spectral kurtosis
    let kurt = spectral::spectral_kurtosis(&freqs, &psd, Some(centroid), Some(spread))?;
    features.insert("spectral_kurtosis".to_string(), kurt);

    // Calculate spectral flatness
    let flatness = spectral::spectral_flatness(&psd)?;
    features.insert("spectral_flatness".to_string(), flatness);

    // Calculate spectral rolloff
    let rolloff = spectral::spectral_rolloff(&freqs, &psd, 0.85)?;
    features.insert("spectral_rolloff".to_string(), rolloff);

    // Calculate spectral slope
    let slope = spectral::spectral_slope(&freqs, &psd)?;
    features.insert("spectral_slope".to_string(), slope);

    // Calculate spectral decrease
    let decrease = spectral::spectral_decrease(&freqs, &psd)?;
    features.insert("spectral_decrease".to_string(), decrease);

    // Get dominant frequency
    let (dom_freq, dom_magnitude) = spectral::dominant_frequency(&freqs, &psd)?;
    features.insert("dominant_frequency".to_string(), dom_freq);
    features.insert("dominant_frequency_magnitude".to_string(), dom_magnitude);

    // Calculate ratio of energy in different frequency bands
    if freqs.len() > 10 {
        let max_freq = freqs[freqs.len() - 1];
        let low_band_end = max_freq * 0.25;
        let mid_band_end = max_freq * 0.75;

        let mut low_band_energy = 0.0;
        let mut mid_band_energy = 0.0;
        let mut high_band_energy = 0.0;

        for i in 0..freqs.len() {
            let f = freqs[i];
            let p = psd[i];

            if f < low_band_end {
                low_band_energy += p;
            } else if f < mid_band_end {
                mid_band_energy += p;
            } else {
                high_band_energy += p;
            }
        }

        let total = low_band_energy + mid_band_energy + high_band_energy;
        if total > 0.0 {
            features.insert("low_freq_energy_ratio".to_string(), low_band_energy / total);
            features.insert("mid_freq_energy_ratio".to_string(), mid_band_energy / total);
            features.insert(
                "high_freq_energy_ratio".to_string(),
                high_band_energy / total,
            );
        }
    }

    Ok(())
}
