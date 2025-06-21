//! Sparsity estimation methods for Sparse FFT
//!
//! This module provides various methods to estimate the sparsity of a signal,
//! which determines how many significant frequency components are present.

use crate::error::FFTResult;
use crate::fft::fft;
// Complex64 is used through the FFT functions
use num_traits::NumCast;
use std::fmt::Debug;

use super::config::{SparseFFTConfig, SparsityEstimationMethod};

/// Estimate sparsity of a signal using various methods
pub fn estimate_sparsity<T>(signal: &[T], config: &SparseFFTConfig) -> FFTResult<usize>
where
    T: NumCast + Copy + Debug + 'static,
{
    match config.estimation_method {
        SparsityEstimationMethod::Manual => Ok(config.sparsity),

        SparsityEstimationMethod::Threshold => {
            estimate_sparsity_threshold(signal, config.threshold)
        }

        SparsityEstimationMethod::Adaptive => {
            estimate_sparsity_adaptive(signal, config.adaptivity_factor, config.sparsity)
        }

        SparsityEstimationMethod::FrequencyPruning => {
            estimate_sparsity_frequency_pruning(signal, config.pruning_sensitivity)
        }

        SparsityEstimationMethod::SpectralFlatness => estimate_sparsity_spectral_flatness(
            signal,
            config.flatness_threshold,
            config.window_size,
        ),
    }
}

/// Estimate sparsity using magnitude thresholding
pub fn estimate_sparsity_threshold<T>(signal: &[T], threshold: f64) -> FFTResult<usize>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Compute regular FFT
    let spectrum = fft(signal, None)?;

    // Find magnitudes
    let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();

    // Find maximum magnitude
    let max_magnitude = magnitudes.iter().cloned().fold(0.0, f64::max);

    // Count coefficients above threshold
    let threshold_value = max_magnitude * threshold;
    let count = magnitudes.iter().filter(|&&m| m > threshold_value).count();

    Ok(count)
}

/// Estimate sparsity using adaptive energy-based method
pub fn estimate_sparsity_adaptive<T>(
    signal: &[T],
    adaptivity_factor: f64,
    fallback_sparsity: usize,
) -> FFTResult<usize>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Compute regular FFT
    let spectrum = fft(signal, None)?;

    // Find magnitudes and sort them
    let mut magnitudes: Vec<(usize, f64)> = spectrum
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.norm()))
        .collect();

    magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find "elbow" in the magnitude curve using adaptivity factor
    let signal_energy: f64 = magnitudes.iter().map(|(_, m)| m * m).sum();
    let mut cumulative_energy = 0.0;
    let energy_threshold = signal_energy * (1.0 - adaptivity_factor);

    for (i, (_, mag)) in magnitudes.iter().enumerate() {
        cumulative_energy += mag * mag;
        if cumulative_energy >= energy_threshold {
            return Ok(i + 1);
        }
    }

    // Fallback: return a default small value if we couldn't determine sparsity
    Ok(fallback_sparsity)
}

/// Estimate sparsity using frequency pruning method
pub fn estimate_sparsity_frequency_pruning<T>(
    signal: &[T],
    pruning_sensitivity: f64,
) -> FFTResult<usize>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Compute regular FFT
    let spectrum = fft(signal, None)?;

    // Use frequency pruning approach
    let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
    let n = magnitudes.len();

    // Compute local variance in frequency domain
    let mut local_variances = Vec::with_capacity(n);
    let window_size = (n / 16).max(3).min(n);

    for i in 0..n {
        let start = if i >= window_size / 2 {
            i - window_size / 2
        } else {
            0
        };
        let end = (i + window_size / 2 + 1).min(n);

        let window_mags = &magnitudes[start..end];
        let mean = window_mags.iter().sum::<f64>() / window_mags.len() as f64;
        let variance =
            window_mags.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window_mags.len() as f64;

        local_variances.push(variance);
    }

    // Count significant components based on local variance
    let mean_variance = local_variances.iter().sum::<f64>() / local_variances.len() as f64;
    let variance_threshold = mean_variance * pruning_sensitivity;

    let significant_count = local_variances
        .iter()
        .zip(magnitudes.iter())
        .filter(|(&var, &mag)| var > variance_threshold && mag > 0.0)
        .count();

    Ok(significant_count.max(1))
}

/// Estimate sparsity using spectral flatness measure
pub fn estimate_sparsity_spectral_flatness<T>(
    signal: &[T],
    flatness_threshold: f64,
    window_size: usize,
) -> FFTResult<usize>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Compute regular FFT
    let spectrum = fft(signal, None)?;

    // Compute power spectrum
    let power_spectrum: Vec<f64> = spectrum.iter().map(|c| c.norm_sqr()).collect();
    let n = power_spectrum.len();

    // Compute spectral flatness for overlapping windows
    let mut significant_components = 0;
    let step_size = window_size / 2;

    for start in (0..n).step_by(step_size) {
        let end = (start + window_size).min(n);
        let window_power = &power_spectrum[start..end];

        // Skip if window is too small or contains only zeros
        if window_power.len() < 2 || window_power.iter().all(|&x| x == 0.0) {
            continue;
        }

        // Compute geometric mean
        let geometric_mean = {
            let log_sum = window_power
                .iter()
                .filter(|&&x| x > 0.0)
                .map(|&x| x.ln())
                .sum::<f64>();
            let count = window_power.iter().filter(|&&x| x > 0.0).count() as f64;
            if count > 0.0 {
                (log_sum / count).exp()
            } else {
                0.0
            }
        };

        // Compute arithmetic mean
        let arithmetic_mean = window_power.iter().sum::<f64>() / window_power.len() as f64;

        // Compute spectral flatness
        let spectral_flatness = if arithmetic_mean > 0.0 {
            geometric_mean / arithmetic_mean
        } else {
            0.0
        };

        // Count as significant if flatness is below threshold (indicating peaks)
        if spectral_flatness < flatness_threshold {
            significant_components += window_power
                .iter()
                .filter(|&&x| x > arithmetic_mean * 0.1)
                .count();
        }
    }

    Ok(significant_components.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn create_sparse_signal(n: usize, frequencies: &[(usize, f64)]) -> Vec<f64> {
        let mut signal = vec![0.0; n];

        for i in 0..n {
            let t = 2.0 * PI * (i as f64) / (n as f64);
            for &(freq, amp) in frequencies {
                signal[i] += amp * (freq as f64 * t).sin();
            }
        }

        signal
    }

    #[test]
    fn test_estimate_sparsity_threshold() {
        let n = 64;
        let frequencies = vec![(3, 1.0), (7, 0.5)];
        let signal = create_sparse_signal(n, &frequencies);

        let result = estimate_sparsity_threshold(&signal, 0.1).unwrap();
        // Should find approximately 4 components (positive and negative frequencies)
        assert!(result >= 2 && result <= 8);
    }

    #[test]
    fn test_estimate_sparsity_adaptive() {
        let n = 64;
        let frequencies = vec![(3, 1.0), (7, 0.5), (15, 0.25)];
        let signal = create_sparse_signal(n, &frequencies);

        let result = estimate_sparsity_adaptive(&signal, 0.25, 10).unwrap();
        // Should find some components (adaptive method can vary)
        assert!(result >= 2 && result <= 15);
    }

    #[test]
    fn test_estimate_sparsity_frequency_pruning() {
        let n = 64;
        let frequencies = vec![(3, 1.0), (7, 0.5)];
        let signal = create_sparse_signal(n, &frequencies);

        let result = estimate_sparsity_frequency_pruning(&signal, 2.0).unwrap();
        assert!(result >= 1);
    }

    #[test]
    fn test_estimate_sparsity_spectral_flatness() {
        let n = 64;
        let frequencies = vec![(3, 1.0), (7, 0.5)];
        let signal = create_sparse_signal(n, &frequencies);

        let result = estimate_sparsity_spectral_flatness(&signal, 0.3, 8).unwrap();
        assert!(result >= 1);
    }
}
