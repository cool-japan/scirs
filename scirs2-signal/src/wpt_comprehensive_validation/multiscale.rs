//! Multi-scale analysis validation for wavelet packet transforms
//!
//! This module provides validation of multi-scale properties including
//! scale consistency, frequency localization, and inter-scale correlations.

use super::types::*;
use super::utils::generate_test_signal;
use crate::error::SignalResult;
use ndarray::Array2;

/// Validate multi-scale properties of wavelet packet decompositions
pub fn validate_multiscale_properties(
    config: &ComprehensiveWptValidationConfig,
) -> SignalResult<MultiscaleValidationMetrics> {
    let mut scale_energies = Vec::new();
    let mut correlations = Vec::new();

    // Test multi-scale properties for different configurations
    for &wavelet in &config.test_wavelets {
        for &signal_length in &config.test_signal_lengths {
            if signal_length < 64 {
                continue; // Skip small signals for multi-scale analysis
            }

            let signal = generate_test_signal(TestSignalType::WhiteNoise, signal_length, 0)?;

            // Analyze scale energy distribution
            let scale_energy_dist = analyze_scale_energy_distribution(&signal, wavelet, &config.test_levels)?;
            scale_energies.push(scale_energy_dist);

            // Compute inter-scale correlations
            let inter_scale_corr = compute_inter_scale_correlations(&signal, wavelet, &config.test_levels)?;
            correlations.push(inter_scale_corr);
        }
    }

    // Aggregate results
    let scale_energy_distribution = if !scale_energies.is_empty() {
        // Average energy distributions
        let max_len = scale_energies.iter().map(|e| e.len()).max().unwrap_or(0);
        let mut avg_energy = vec![0.0; max_len];
        let mut count = vec![0; max_len];

        for energy_dist in &scale_energies {
            for (i, &energy) in energy_dist.iter().enumerate() {
                avg_energy[i] += energy;
                count[i] += 1;
            }
        }

        for i in 0..max_len {
            if count[i] > 0 {
                avg_energy[i] /= count[i] as f64;
            }
        }

        avg_energy
    } else {
        vec![1.0] // Default single scale
    };

    // Average inter-scale correlations
    let inter_scale_correlations = if !correlations.is_empty() {
        let total_corr = correlations.iter().fold(
            Array2::zeros((config.test_levels.len(), config.test_levels.len())),
            |acc, corr| acc + corr
        );
        total_corr / correlations.len() as f64
    } else {
        Array2::eye(config.test_levels.len())
    };

    // Compute derived metrics
    let scale_consistency = compute_scale_consistency(&scale_energy_distribution);
    let frequency_localization = compute_frequency_localization(&scale_energy_distribution);
    let time_localization = compute_time_localization(&scale_energy_distribution);

    Ok(MultiscaleValidationMetrics {
        scale_energy_distribution,
        inter_scale_correlations,
        scale_consistency,
        frequency_localization,
        time_localization,
    })
}

/// Analyze energy distribution across scales
fn analyze_scale_energy_distribution(
    signal: &[f64],
    wavelet: Wavelet,
    levels: &[usize],
) -> SignalResult<Vec<f64>> {
    let mut energy_distribution = Vec::new();

    for &level in levels {
        // Simplified energy calculation for each scale
        // In practice, this would involve actual WPT decomposition
        let scale_energy = compute_scale_energy(signal, level);
        energy_distribution.push(scale_energy);
    }

    // Normalize energy distribution
    let total_energy: f64 = energy_distribution.iter().sum();
    if total_energy > 1e-15 {
        for energy in &mut energy_distribution {
            *energy /= total_energy;
        }
    }

    Ok(energy_distribution)
}

/// Compute energy at a specific scale (simplified)
fn compute_scale_energy(signal: &[f64], level: usize) -> f64 {
    // Simplified scale energy computation
    // In practice, this would use actual wavelet packet coefficients
    let scale_factor = 2.0_f64.powi(level as i32);
    let decimated_length = (signal.len() as f64 / scale_factor).ceil() as usize;

    let mut scale_signal = Vec::with_capacity(decimated_length);
    for i in (0..signal.len()).step_by(scale_factor as usize) {
        scale_signal.push(signal[i]);
    }

    scale_signal.iter().map(|&x| x * x).sum()
}

/// Compute inter-scale correlations
fn compute_inter_scale_correlations(
    signal: &[f64],
    wavelet: Wavelet,
    levels: &[usize],
) -> SignalResult<Array2<f64>> {
    let n_levels = levels.len();
    let mut correlations = Array2::eye(n_levels);

    // Compute scale signals
    let mut scale_signals = Vec::new();
    for &level in levels {
        let scale_signal = compute_scale_signal(signal, level);
        scale_signals.push(scale_signal);
    }

    // Compute pairwise correlations
    for i in 0..n_levels {
        for j in (i + 1)..n_levels {
            let correlation = compute_signal_correlation(&scale_signals[i], &scale_signals[j]);
            correlations[[i, j]] = correlation;
            correlations[[j, i]] = correlation;
        }
    }

    Ok(correlations)
}

/// Compute signal at specific scale
fn compute_scale_signal(signal: &[f64], level: usize) -> Vec<f64> {
    let scale_factor = 2.0_f64.powi(level as i32);
    let decimated_length = (signal.len() as f64 / scale_factor).ceil() as usize;

    let mut scale_signal = Vec::with_capacity(decimated_length);
    for i in (0..signal.len()).step_by(scale_factor as usize) {
        scale_signal.push(signal[i]);
    }

    scale_signal
}

/// Compute correlation between two signals
fn compute_signal_correlation(signal1: &[f64], signal2: &[f64]) -> f64 {
    let min_len = signal1.len().min(signal2.len());
    if min_len == 0 {
        return 0.0;
    }

    let mean1 = signal1[..min_len].iter().sum::<f64>() / min_len as f64;
    let mean2 = signal2[..min_len].iter().sum::<f64>() / min_len as f64;

    let mut numerator = 0.0;
    let mut sum_sq1 = 0.0;
    let mut sum_sq2 = 0.0;

    for i in 0..min_len {
        let diff1 = signal1[i] - mean1;
        let diff2 = signal2[i] - mean2;

        numerator += diff1 * diff2;
        sum_sq1 += diff1 * diff1;
        sum_sq2 += diff2 * diff2;
    }

    let denominator = (sum_sq1 * sum_sq2).sqrt();
    if denominator > 1e-15 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Compute scale consistency measure
fn compute_scale_consistency(energy_distribution: &[f64]) -> f64 {
    if energy_distribution.len() < 2 {
        return 1.0;
    }

    // Measure how evenly distributed the energy is across scales
    let mean_energy = energy_distribution.iter().sum::<f64>() / energy_distribution.len() as f64;
    let variance = energy_distribution
        .iter()
        .map(|&e| (e - mean_energy) * (e - mean_energy))
        .sum::<f64>()
        / energy_distribution.len() as f64;

    // Convert variance to consistency score (lower variance = higher consistency)
    1.0 / (1.0 + variance * 10.0)
}

/// Compute frequency localization measure
fn compute_frequency_localization(energy_distribution: &[f64]) -> f64 {
    // Simplified frequency localization based on energy concentration
    if energy_distribution.is_empty() {
        return 0.0;
    }

    let max_energy = energy_distribution.iter().cloned().fold(0.0f64, f64::max);
    let total_energy: f64 = energy_distribution.iter().sum();

    if total_energy > 1e-15 {
        max_energy / total_energy
    } else {
        0.0
    }
}

/// Compute time localization measure
fn compute_time_localization(energy_distribution: &[f64]) -> f64 {
    // Simplified time localization (similar to frequency for this implementation)
    compute_frequency_localization(energy_distribution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dwt::Wavelet;

    #[test]
    fn test_multiscale_validation() {
        let mut config = ComprehensiveWptValidationConfig::default();

        // Use minimal configuration for testing
        config.test_wavelets = vec![Wavelet::Haar];
        config.test_signal_lengths = vec![64];
        config.test_levels = vec![1, 2];

        let result = validate_multiscale_properties(&config);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(!metrics.scale_energy_distribution.is_empty());
        assert_eq!(metrics.inter_scale_correlations.nrows(), 2);
        assert_eq!(metrics.inter_scale_correlations.ncols(), 2);
    }

    #[test]
    fn test_scale_consistency() {
        let uniform_energy = vec![0.25, 0.25, 0.25, 0.25];
        let non_uniform_energy = vec![0.8, 0.1, 0.05, 0.05];

        let consistency1 = compute_scale_consistency(&uniform_energy);
        let consistency2 = compute_scale_consistency(&non_uniform_energy);

        // Uniform distribution should have higher consistency
        assert!(consistency1 > consistency2);
    }
}