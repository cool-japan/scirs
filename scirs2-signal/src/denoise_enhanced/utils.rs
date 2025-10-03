//! Utility functions for enhanced denoising
//!
//! This module provides validation, memory optimization, quality assessment,
//! and other utility functions used across the enhanced denoising algorithms.

use super::types::*;
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::validation::{check_finite, check_positive};

/// Validate denoising configuration
pub fn validate_denoise_config(config: &DenoiseConfig) -> SignalResult<()> {
    // Validate noise estimation flag
    if config.adaptive_noise && config.levels.is_some() && config.levels.unwrap() == 0 {
        return Err(SignalError::ValueError(
            "Adaptive noise estimation requires at least one decomposition level".to_string(),
        ));
    }

    // Validate translation-invariant parameters
    if config.translation_invariant && config.n_shifts == 0 {
        return Err(SignalError::ValueError(
            "Translation-invariant denoising requires n_shifts > 0".to_string(),
        ));
    }

    // Validate block thresholding parameters
    if let ThresholdMethod::Block { block_size } = config.method {
        check_positive(block_size, "block_size")?;
    }

    // Validate SCAD parameters
    if let ThresholdMethod::SCAD { a } = config.method {
        if a <= 2.0 {
            return Err(SignalError::ValueError(
                "SCAD parameter 'a' must be greater than 2".to_string(),
            ));
        }
    }

    // Validate Firm thresholding parameters
    if let ThresholdMethod::Firm { alpha } = config.method {
        if alpha <= 1.0 {
            return Err(SignalError::ValueError(
                "Firm parameter 'alpha' must be greater than 1".to_string(),
            ));
        }
    }

    // Validate FDR parameters
    if let ThresholdRule::FDR { q } = config.threshold_rule {
        if q <= 0.0 || q >= 1.0 {
            return Err(SignalError::ValueError(
                "FDR parameter 'q' must be between 0 and 1".to_string(),
            ));
        }
    }

    // Validate custom threshold
    if let ThresholdRule::Custom(threshold) = config.threshold_rule {
        check_positive(threshold, "custom_threshold")?;
    }

    // Validate memory optimization settings
    if config.memory_optimized && config.block_size.is_some() {
        check_positive(config.block_size.unwrap(), "block_size")?;
    }

    Ok(())
}

/// Estimate memory usage for denoising operation
pub fn estimate_memory_usage(signal_length: usize) -> usize {
    // Rough estimate: original signal + decomposition + thresholded + temp arrays
    let element_size = std::mem::size_of::<f64>();
    let decomposition_overhead = 2.0; // Approximation for wavelet decomposition
    let temp_arrays = 3; // Various temporary arrays during processing

    (signal_length as f64 * decomposition_overhead * temp_arrays as f64 * element_size as f64)
        as usize
}

/// Compute numerical stability score for denoising result
pub fn compute_numerical_stability(signal: &Array1<f64>) -> SignalResult<f64> {
    if signal.is_empty() {
        return Err(SignalError::ValueError(
            "Signal cannot be empty".to_string(),
        ));
    }

    // Check for finite values
    let finite_count = signal.iter().filter(|&&x| x.is_finite()).count();
    let finite_ratio = finite_count as f64 / signal.len() as f64;

    if finite_ratio < 1.0 {
        return Ok(finite_ratio); // Return ratio of finite values
    }

    // Check for reasonable dynamic range
    let max_val = signal.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    let min_val = signal.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));

    if max_val == min_val {
        return Ok(1.0); // Constant signal is numerically stable
    }

    let dynamic_range = (max_val / min_val.abs()).abs();
    let stability_score = if dynamic_range > 1e12 {
        0.5 // Poor numerical conditioning
    } else if dynamic_range > 1e6 {
        0.8 // Moderate conditioning
    } else {
        1.0 // Good conditioning
    };

    Ok(stability_score)
}

/// Adaptive noise estimation from wavelet coefficients
pub fn adaptive_noise_estimation(coeffs: &crate::dwt::DecompositionResult) -> SignalResult<f64> {
    if coeffs.details.is_empty() {
        return Err(SignalError::ValueError(
            "No detail coefficients available for noise estimation".to_string(),
        ));
    }

    // Use the finest scale detail coefficients
    let finest_detail = &coeffs.details[coeffs.details.len() - 1];

    // Median absolute deviation estimator
    let mut abs_coeffs: Vec<f64> = finest_detail.iter().map(|&x| x.abs()).collect();
    abs_coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = if abs_coeffs.len().is_multiple_of(2) {
        (abs_coeffs[abs_coeffs.len() / 2 - 1] + abs_coeffs[abs_coeffs.len() / 2]) / 2.0
    } else {
        abs_coeffs[abs_coeffs.len() / 2]
    };

    // Scale factor for Gaussian noise
    let noise_sigma = median / 0.6745;

    Ok(noise_sigma)
}

/// Memory-optimized 1D denoising for large signals
pub fn memory_optimized_denoise_1d(
    signal: &Array1<f64>,
    config: &DenoiseConfig,
) -> SignalResult<DenoiseResult> {
    let n = signal.len();
    let memory_limit = 100_000_000; // 100MB limit
    let estimated_memory = estimate_memory_usage(n);

    if estimated_memory < memory_limit || !config.memory_optimized {
        // Use standard algorithm
        return crate::denoise_enhanced::wavelet::denoise_wavelet_1d(signal, config);
    }

    // Process in blocks
    let block_size = config.block_size.unwrap_or(4096);
    let overlap = 256; // Overlap to handle boundary effects

    let mut denoised_signal = Array1::zeros(n);
    let mut combined_noise_sigma = 0.0;
    let mut combined_thresholds = Vec::new();
    let mut combined_retention_rate = 0.0;
    let mut block_count = 0;

    for start in (0..n).step_by(block_size - overlap) {
        let end = (start + block_size).min(n);
        let block = signal
            .slice(scirs2_core::ndarray::s![start..end])
            .to_owned();

        // Denoise block
        let block_result = crate::denoise_enhanced::wavelet::denoise_wavelet_1d(&block, config)?;

        // Merge results
        let actual_start = if start == 0 { 0 } else { start + overlap / 2 };
        let actual_end = if end == n { end } else { end - overlap / 2 };

        for (i, &val) in block_result
            .signal
            .slice(scirs2_core::ndarray::s![
                actual_start - start..actual_end - start
            ])
            .iter()
            .enumerate()
        {
            denoised_signal[actual_start + i] = val;
        }

        combined_noise_sigma += block_result.noise_sigma;
        if combined_thresholds.is_empty() {
            combined_thresholds = block_result.thresholds;
        }
        combined_retention_rate += block_result.retention_rate;
        block_count += 1;
    }

    // Average the results
    combined_noise_sigma /= block_count as f64;
    combined_retention_rate /= block_count as f64;

    Ok(DenoiseResult {
        signal: denoised_signal,
        noise_sigma: combined_noise_sigma,
        thresholds: combined_thresholds,
        retention_rate: combined_retention_rate,
        effective_df: n as f64 * combined_retention_rate,
        risk_estimate: None,
        processing_time_ms: 0.0,
        memory_usage_bytes: estimated_memory,
        stability_score: 1.0,
        snr_improvement_db: 0.0,
    })
}

/// Compute SNR improvement between original and denoised signals
pub fn compute_snr_improvement(
    original: &Array1<f64>,
    noisy: &Array1<f64>,
    denoised: &Array1<f64>,
) -> SignalResult<f64> {
    if original.len() != noisy.len() || original.len() != denoised.len() {
        return Err(SignalError::ValueError(
            "All signals must have the same length".to_string(),
        ));
    }

    // Compute noise power in original noisy signal
    let noise_power_original: f64 = original
        .iter()
        .zip(noisy.iter())
        .map(|(&orig, &noisy_val)| (orig - noisy_val).powi(2))
        .sum();

    // Compute noise power in denoised signal
    let noise_power_denoised: f64 = original
        .iter()
        .zip(denoised.iter())
        .map(|(&orig, &denoised_val)| (orig - denoised_val).powi(2))
        .sum();

    if noise_power_original <= 0.0 || noise_power_denoised <= 0.0 {
        return Ok(0.0);
    }

    // SNR improvement in dB
    let snr_improvement = 10.0 * (noise_power_original / noise_power_denoised).log10();

    Ok(snr_improvement)
}

/// Compute signal energy
pub fn compute_energy(signal: &[f64]) -> f64 {
    signal.iter().map(|&x| x * x).sum()
}

/// Compute energy of wavelet packet tree
pub fn compute_tree_energy(_tree_data: &[f64]) -> SignalResult<f64> {
    // Simplified implementation - placeholder until wpt module is available
    let total_energy = _tree_data.iter().map(|&x| x * x).sum::<f64>();
    Ok(total_energy)
}

/// Compute next power of two
pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut power = 1;
    while power < n {
        power <<= 1;
    }
    power
}

/// Compute patch similarity for non-local means algorithms
pub fn compute_patch_similarity_1d(
    signal: &Array1<f64>,
    i: usize,
    j: usize,
    patch_size: usize,
    h_sq: f64,
) -> f64 {
    let n = signal.len();
    let half_patch = patch_size / 2;

    let start_i = i.saturating_sub(half_patch);
    let end_i = (i + half_patch + 1).min(n);
    let start_j = j.saturating_sub(half_patch);
    let end_j = (j + half_patch + 1).min(n);

    // Compute patch difference
    let mut diff_sum = 0.0;
    let mut count = 0;

    for (ki, kj) in (start_i..end_i).zip(start_j..end_j) {
        let diff = signal[ki] - signal[kj];
        diff_sum += diff * diff;
        count += 1;
    }

    if count > 0 {
        let normalized_diff = diff_sum / count as f64;
        (-normalized_diff / h_sq).exp()
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dwt::{DecompositionResult, Wavelet};

    #[test]
    fn test_validate_denoise_config() {
        // Test valid configuration
        let valid_config = DenoiseConfig::default();
        let result = validate_denoise_config(&valid_config);
        assert!(result.is_ok());

        // Test invalid SCAD parameter
        let mut invalid_config = DenoiseConfig::default();
        invalid_config.method = ThresholdMethod::SCAD { a: 1.5 }; // Should be > 2
        let result = validate_denoise_config(&invalid_config);
        assert!(result.is_err());

        // Test invalid Firm parameter
        invalid_config.method = ThresholdMethod::Firm { alpha: 0.5 }; // Should be > 1
        let result = validate_denoise_config(&invalid_config);
        assert!(result.is_err());

        // Test invalid FDR parameter
        invalid_config.method = ThresholdMethod::Soft;
        invalid_config.threshold_rule = ThresholdRule::FDR { q: 1.5 }; // Should be < 1
        let result = validate_denoise_config(&invalid_config);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_memory_usage() {
        let usage = estimate_memory_usage(1000);
        assert!(usage > 0);
        assert!(usage > 1000 * std::mem::size_of::<f64>()); // Should be larger than just the signal
    }

    #[test]
    fn test_compute_numerical_stability() {
        // Test with normal signal
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let stability = compute_numerical_stability(&signal).unwrap();
        assert_eq!(stability, 1.0);

        // Test with signal containing infinity
        let signal = Array1::from_vec(vec![1.0, f64::INFINITY, 3.0]);
        let stability = compute_numerical_stability(&signal).unwrap();
        assert!(stability < 1.0);

        // Test with empty signal
        let signal = Array1::from_vec(vec![]);
        let result = compute_numerical_stability(&signal);
        assert!(result.is_err());
    }

    #[test]
    fn test_adaptive_noise_estimation() {
        // Create mock decomposition result
        let detail1 = Array1::from_vec(vec![0.1, -0.2, 0.15, -0.1]);
        let detail2 = Array1::from_vec(vec![0.05, -0.08, 0.03]);
        let approx = Array1::from_vec(vec![1.0, 2.0]);

        let coeffs = DecompositionResult {
            approx,
            details: vec![detail1, detail2],
        };

        let noise_sigma = adaptive_noise_estimation(&coeffs).unwrap();
        assert!(noise_sigma > 0.0);
        assert!(noise_sigma < 1.0); // Reasonable range for test data

        // Test tree energy computation
        let tree_data = vec![1.0, 2.0, 3.0];
        let energy = compute_tree_energy(&tree_data).unwrap();
        assert_eq!(energy, 14.0); // 1^2 + 2^2 + 3^2 = 14
    }

    #[test]
    fn test_compute_snr_improvement() {
        let original = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let noisy = Array1::from_vec(vec![1.1, 2.1, 3.1, 2.1, 1.1]);
        let denoised = Array1::from_vec(vec![1.05, 2.05, 3.05, 2.05, 1.05]);

        let snr_improvement = compute_snr_improvement(&original, &noisy, &denoised).unwrap();
        assert!(snr_improvement > 0.0); // Should show improvement

        // Test error condition - mismatched lengths
        let short_signal = Array1::from_vec(vec![1.0, 2.0]);
        let result = compute_snr_improvement(&original, &short_signal, &denoised);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_energy() {
        let signal = vec![1.0, 2.0, 3.0];
        let energy = compute_energy(&signal);
        assert_eq!(energy, 14.0); // 1^2 + 2^2 + 3^2 = 14
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(7), 8);
        assert_eq!(next_power_of_two(8), 8);
        assert_eq!(next_power_of_two(9), 16);
    }

    #[test]
    fn test_compute_patch_similarity_1d() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let similarity = compute_patch_similarity_1d(&signal, 1, 3, 3, 1.0);
        assert!(similarity >= 0.0 && similarity <= 1.0);

        // Identical patches should have high similarity
        let similarity_identical = compute_patch_similarity_1d(&signal, 1, 1, 3, 1.0);
        assert_eq!(similarity_identical, 1.0);
    }
}
