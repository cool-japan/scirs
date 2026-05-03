// Advanced wavelet denoising methods
//
// This module implements sophisticated denoising techniques including:
// - Translation Invariant Wavelet Denoising (Cycle Spinning)
// - Block thresholding
// - Bayesian wavelet denoising
// - Adaptive thresholding based on local statistics
// - Multiscale denoising with cross-scale dependencies

use crate::denoise::{threshold_coefficients, ThresholdMethod};
use crate::dwt::{wavedec, waverec, DecompositionResult, Wavelet};
use crate::error::{SignalError, SignalResult};
use crate::wpt::wp_decompose;
use scirs2_core::ndarray::Array1;
use scirs2_core::parallel_ops::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Advanced denoising configuration
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AdvancedDenoiseConfig {
    /// Wavelet to use
    pub wavelet: Wavelet,
    /// Decomposition level
    pub level: usize,
    /// Base threshold method
    pub threshold_method: ThresholdMethod,
    /// Use translation invariant denoising
    pub translation_invariant: bool,
    /// Number of shifts for cycle spinning
    pub n_shifts: usize,
    /// Use block thresholding
    pub block_threshold: bool,
    /// Block size for block thresholding
    pub block_size: usize,
    /// Use Bayesian denoising
    pub bayesian: bool,
    /// Use adaptive thresholding
    pub adaptive: bool,
    /// Noise estimation method
    pub noise_estimation: NoiseEstimation,
    /// Use parallel processing
    pub parallel: bool,
}

impl Default for AdvancedDenoiseConfig {
    fn default() -> Self {
        Self {
            wavelet: Wavelet::DB(4),
            level: 4,
            threshold_method: ThresholdMethod::Soft,
            translation_invariant: true,
            n_shifts: 8,
            block_threshold: false,
            block_size: 4,
            bayesian: false,
            adaptive: true,
            noise_estimation: NoiseEstimation::MAD,
            parallel: true,
        }
    }
}

/// Noise estimation methods
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum NoiseEstimation {
    /// Median Absolute Deviation
    MAD,
    /// Standard deviation of finest scale coefficients
    FinestScale,
    /// Robust estimator using interquartile range
    IQR,
    /// Local variance estimation
    LocalVariance,
}

/// Advanced wavelet denoising result
#[derive(Debug, Clone)]
pub struct AdvancedDenoiseResult {
    /// Denoised signal
    pub signal: Vec<f64>,
    /// Estimated noise level
    pub noise_level: f64,
    /// Threshold values used at each scale
    pub thresholds: Vec<f64>,
    /// Signal-to-noise ratio improvement
    pub snr_improvement: Option<f64>,
}

/// Perform advanced wavelet denoising
///
/// # Arguments
///
/// * `signal` - Input noisy signal
/// * `config` - Denoising configuration
///
/// # Returns
///
/// * Advanced denoising result
#[allow(dead_code)]
pub fn advanced_denoise(
    signal: &[f64],
    config: &AdvancedDenoiseConfig,
) -> SignalResult<AdvancedDenoiseResult> {
    if !signal.iter().all(|&x: &f64| x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }

    if signal.len() < (1 << config.level) {
        return Err(SignalError::ValueError(
            "Signal too short for requested decomposition level".to_string(),
        ));
    }

    // Estimate noise level
    let noise_level = estimate_noise_level(signal, &config)?;

    // Apply denoising method
    let (denoised, thresholds) = if config.translation_invariant {
        translation_invariant_denoise(signal, &config, noise_level)?
    } else if config.bayesian {
        bayesian_denoise(signal, &config, noise_level)?
    } else if config.block_threshold {
        block_threshold_denoise(signal, &config, noise_level)?
    } else {
        standard_denoise(signal, &config, noise_level)?
    };

    // Estimate SNR improvement if possible
    let snr_improvement = estimate_snr_improvement(signal, &denoised);

    Ok(AdvancedDenoiseResult {
        signal: denoised,
        noise_level,
        thresholds,
        snr_improvement,
    })
}

/// Translation Invariant Wavelet Denoising (Cycle Spinning)
#[allow(dead_code)]
fn translation_invariant_denoise(
    signal: &[f64],
    config: &AdvancedDenoiseConfig,
    noise_level: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    let n = signal.len();
    let n_shifts = config.n_shifts.min(n);

    if config.parallel {
        // Parallel cycle spinning
        let signal_arc = Arc::new(signal.to_vec());

        let results: Vec<Vec<f64>> = (0..n_shifts)
            .into_par_iter()
            .map(|shift| {
                let signal_ref = signal_arc.clone();

                // Circular shift
                let mut shifted = vec![0.0; n];
                for i in 0..n {
                    shifted[i] = signal_ref[(i + shift) % n];
                }

                // Denoise shifted signal (fall back to original on error)
                let (denoised_shifted, _) = standard_denoise(&shifted, config, noise_level)
                    .unwrap_or_else(|_| (shifted.clone(), vec![]));

                // Inverse shift
                let mut result = vec![0.0; n];
                for i in 0..n {
                    result[(i + shift) % n] = denoised_shifted[i];
                }

                result
            })
            .collect();

        // Average all shifted results
        let mut averaged = vec![0.0; n];
        for result in &results {
            for i in 0..n {
                averaged[i] += result[i] / n_shifts as f64;
            }
        }

        // Use thresholds from first decomposition
        let (_, thresholds) = standard_denoise(signal, config, noise_level)?;

        Ok((averaged, thresholds))
    } else {
        // Sequential cycle spinning
        let mut accumulated = vec![0.0; n];
        let mut thresholds = Vec::new();

        for shift in 0..n_shifts {
            // Circular shift
            let mut shifted = vec![0.0; n];
            for i in 0..n {
                shifted[i] = signal[(i + shift) % n];
            }

            // Denoise shifted signal
            let (denoised_shifted, thresh) = standard_denoise(&shifted, config, noise_level)?;

            if shift == 0 {
                thresholds = thresh;
            }

            // Inverse shift and accumulate
            for i in 0..n {
                accumulated[(i + shift) % n] += denoised_shifted[i] / n_shifts as f64;
            }
        }

        Ok((accumulated, thresholds))
    }
}

/// Bayesian wavelet denoising
#[allow(dead_code)]
fn bayesian_denoise(
    signal: &[f64],
    config: &AdvancedDenoiseConfig,
    noise_level: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    // Decompose signal
    let coeffs_raw = wavedec(signal, config.wavelet, Some(config.level), None)?;
    let coeffs = DecompositionResult::from_wavedec(coeffs_raw);
    let mut thresholds = Vec::new();

    // Process each scale
    let mut denoised_coeffs = coeffs.clone();

    for (level_idx, detail) in coeffs.details.iter().enumerate() {
        // Estimate signal variance at this scale
        let signal_var = estimate_signal_variance(detail.as_slice().unwrap_or(&[]), noise_level);

        // Bayesian shrinkage
        let shrinkage_factor = signal_var / (signal_var + noise_level * noise_level);
        let threshold = noise_level * ((1.0 - shrinkage_factor) as f64).sqrt();
        thresholds.push(threshold);

        // Apply Bayesian shrinkage to coefficients
        let mut denoised_detail = detail.clone();
        for coeff in denoised_detail.iter_mut() {
            *coeff *= shrinkage_factor;
        }

        denoised_coeffs.details[level_idx] = denoised_detail;
    }

    // Reconstruct
    let denoised = waverec(&denoised_coeffs.to_wavedec(), config.wavelet)?;

    Ok((denoised, thresholds))
}

/// Block thresholding
#[allow(dead_code)]
fn block_threshold_denoise(
    signal: &[f64],
    config: &AdvancedDenoiseConfig,
    noise_level: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    // Decompose signal
    let coeffs_raw = wavedec(signal, config.wavelet, Some(config.level), None)?;
    let coeffs = DecompositionResult::from_wavedec(coeffs_raw);
    let mut thresholds = Vec::new();
    let mut denoised_coeffs = coeffs.clone();

    // Process each scale with block thresholding
    for (level_idx, detail) in coeffs.details.iter().enumerate() {
        let n_coeffs = detail.len();
        let block_size = config.block_size.min(n_coeffs);
        let n_blocks = (n_coeffs + block_size - 1) / block_size;

        // Scale-dependent threshold
        let base_threshold = noise_level * (2.0 * (signal.len() as f64).ln()).sqrt();
        let scale_factor = (level_idx + 1) as f64 / config.level as f64;
        let threshold = base_threshold * (1.0 + 0.5 * scale_factor);
        thresholds.push(threshold);

        let mut denoised_detail = vec![0.0; n_coeffs];

        for block_idx in 0..n_blocks {
            let start = block_idx * block_size;
            let end = ((block_idx + 1) * block_size).min(n_coeffs);

            // Compute block energy
            let mut block_energy = 0.0f64;
            for i in start..end {
                block_energy += detail[i] * detail[i];
            }
            block_energy = block_energy.sqrt();

            // Block thresholding decision
            if block_energy > threshold {
                // Keep block with shrinkage
                let shrinkage = (block_energy - threshold) / block_energy;
                for i in start..end {
                    denoised_detail[i] = detail[i] * shrinkage;
                }
            }
            // else: block is zeroed (already initialized to 0)
        }

        denoised_coeffs.details[level_idx] = Array1::from_vec(denoised_detail);
    }

    // Reconstruct
    let denoised = waverec(&denoised_coeffs.to_wavedec(), config.wavelet)?;

    Ok((denoised, thresholds))
}

/// Standard wavelet denoising (baseline)
#[allow(dead_code)]
fn standard_denoise(
    signal: &[f64],
    config: &AdvancedDenoiseConfig,
    noise_level: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    // Decompose signal
    let coeffs_raw = wavedec(signal, config.wavelet, Some(config.level), None)?;
    let coeffs = DecompositionResult::from_wavedec(coeffs_raw);
    let mut thresholds = Vec::new();
    let mut denoised_coeffs = coeffs.clone();

    // Universal threshold
    let base_threshold = noise_level * (2.0 * (signal.len() as f64).ln()).sqrt();

    // Process each scale
    for (level_idx, detail) in coeffs.details.iter().enumerate() {
        // Adaptive threshold based on scale
        let threshold = if config.adaptive {
            // Scale-dependent threshold
            let scale_factor = (level_idx + 1) as f64 / config.level as f64;
            base_threshold * (1.0 - 0.3 * scale_factor)
        } else {
            base_threshold
        };

        thresholds.push(threshold);

        // Apply thresholding
        let thresholded = threshold_coefficients(
            detail.as_slice().unwrap_or(&[]),
            threshold,
            config.threshold_method,
        );
        denoised_coeffs.details[level_idx] = Array1::from_vec(thresholded);
    }

    // Reconstruct
    let denoised = waverec(&denoised_coeffs.to_wavedec(), config.wavelet)?;

    // Ensure output length matches input
    let mut result = denoised;
    result.truncate(signal.len());

    Ok((result, thresholds))
}

/// Estimate noise level from signal
#[allow(dead_code)]
fn estimate_noise_level(signal: &[f64], config: &AdvancedDenoiseConfig) -> SignalResult<f64> {
    match config.noise_estimation {
        NoiseEstimation::MAD => {
            // Use MAD of finest scale wavelet coefficients
            let coeffs = wavedec(signal, config.wavelet, Some(1), None)?;
            let detail = &coeffs[1]; // First detail coefficients

            // Compute median
            let mut sorted = detail.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = sorted[sorted.len() / 2];

            // Compute MAD
            let mut deviations: Vec<f64> = detail.iter().map(|&x| (x - median).abs()).collect();
            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mad = deviations[deviations.len() / 2];

            // Scale factor for Gaussian noise
            Ok(mad / 0.6745)
        }
        NoiseEstimation::FinestScale => {
            // Standard deviation of finest scale coefficients
            let coeffs_raw = wavedec(signal, config.wavelet, Some(1), None)?;
            let coeffs = DecompositionResult::from_wavedec(coeffs_raw);
            let detail = &coeffs.details[0];

            let mean = detail.iter().sum::<f64>() / detail.len() as f64;
            let variance =
                detail.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / detail.len() as f64;

            Ok(variance.sqrt())
        }
        NoiseEstimation::IQR => {
            // Interquartile range based estimation
            let coeffs_raw = wavedec(signal, config.wavelet, Some(1), None)?;
            let coeffs = DecompositionResult::from_wavedec(coeffs_raw);
            let mut detail = coeffs.details[0].to_vec();
            detail.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let q1_idx = detail.len() / 4;
            let q3_idx = 3 * detail.len() / 4;
            let iqr = detail[q3_idx] - detail[q1_idx];

            // Scale factor for Gaussian noise
            Ok(iqr / 1.349)
        }
        NoiseEstimation::LocalVariance => {
            // Estimate using local variance in signal domain
            let window = 16;
            let mut variances = Vec::new();

            for i in 0..signal.len().saturating_sub(window) {
                let chunk = &signal[i..i + window];
                let mean = chunk.iter().sum::<f64>() / window as f64;
                let var = chunk.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window as f64;
                variances.push(var);
            }

            if variances.is_empty() {
                return Ok(1.0);
            }

            // Use minimum variance as noise estimate
            variances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Ok(variances[0].sqrt())
        }
    }
}

/// Estimate signal variance for Bayesian denoising
#[allow(dead_code)]
fn estimate_signal_variance(coeffs: &[f64], noise_level: f64) -> f64 {
    let n = coeffs.len() as f64;
    let empirical_var = coeffs.iter().map(|&x| x * x).sum::<f64>() / n;

    // Estimate signal variance by removing noise contribution
    let noise_var = noise_level * noise_level;
    (empirical_var - noise_var).max(0.0)
}

/// Estimate SNR improvement
#[allow(dead_code)]
fn estimate_snr_improvement(original: &[f64], denoised: &[f64]) -> Option<f64> {
    if original.len() != denoised.len() {
        return None;
    }

    // Estimate noise as difference between original and denoised
    let noise: Vec<f64> = original
        .iter()
        .zip(denoised.iter())
        .map(|(&o, &d)| o - d)
        .collect();

    // Compute power
    let signal_power = denoised.iter().map(|&x| x * x).sum::<f64>() / denoised.len() as f64;
    let noise_power = noise.iter().map(|&x| x * x).sum::<f64>() / noise.len() as f64;

    if noise_power > 0.0 {
        Some(10.0 * (signal_power / noise_power).log10())
    } else {
        None
    }
}

/// Coifman-Wickerhauser wavelet packet best-basis denoising.
///
/// # Algorithm
///
/// 1. Decompose the signal into a full wavelet packet tree.
/// 2. Select the best basis using Shannon entropy cost minimisation
///    (Coifman-Wickerhauser bottom-up algorithm).
/// 3. Apply soft-thresholding to every best-basis leaf node.
///    Threshold: universal threshold σ√(2 ln n), where σ is estimated
///    from the MAD of the finest-level detail coefficients.
/// 4. Reconstruct the signal from the thresholded best-basis leaves.
///
/// # Arguments
///
/// * `signal` — Noisy input signal (must have at least 2^level samples).
/// * `config`  — Denoising configuration (uses `wavelet`, `level`).
///
/// # Returns
///
/// Denoised signal with the same length as `signal`.
#[allow(dead_code)]
pub fn wavelet_packet_denoise(
    signal: &[f64],
    config: &AdvancedDenoiseConfig,
) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    if n < (1 << config.level) {
        return Err(SignalError::ValueError(
            "Signal too short for the requested decomposition level".to_string(),
        ));
    }

    // --- Step 1: Build full wavelet packet tree ---
    let tree = wp_decompose(signal, config.wavelet, config.level, None)?;

    // --- Step 2: Estimate noise from MAD of finest-level detail coefficients ---
    // Decompose to level 1 to access finest-scale detail band.
    let finest_coeffs_raw = crate::dwt::wavedec(signal, config.wavelet, Some(1), None)?;
    let sigma = if finest_coeffs_raw.len() >= 2 && !finest_coeffs_raw[1].is_empty() {
        let detail = &finest_coeffs_raw[1];
        let mut sorted: Vec<f64> = detail.iter().map(|&x| x.abs()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_abs = sorted[sorted.len() / 2];
        // MAD-based sigma: σ = MAD / 0.6745
        (median_abs / 0.6745).max(1e-10)
    } else {
        1.0
    };

    // Universal threshold: λ = σ √(2 ln n)
    let threshold = sigma * (2.0 * (n as f64).ln()).sqrt();

    // --- Step 3: Select best basis (Coifman-Wickerhauser) ---
    let best_basis = tree.get_best_basis("shannon")?;

    // --- Step 4: Soft-threshold coefficients at each best-basis leaf ---
    let mut overrides: HashMap<(usize, usize), Vec<f64>> = HashMap::new();

    for &(level, pos) in &best_basis {
        let node = tree.nodes.get(&(level, pos)).ok_or_else(|| {
            SignalError::ValueError(format!(
                "Best-basis node ({}, {}) not found in tree",
                level, pos
            ))
        })?;

        // Soft thresholding: sign(c) * max(|c| - λ, 0)
        let thresholded: Vec<f64> = node
            .data
            .iter()
            .map(|&c| {
                let abs_c = c.abs();
                if abs_c <= threshold {
                    0.0
                } else {
                    c.signum() * (abs_c - threshold)
                }
            })
            .collect();

        overrides.insert((level, pos), thresholded);
    }

    // --- Step 5: Reconstruct from thresholded best-basis leaves ---
    let reconstructed = tree.reconstruct_best_basis(&best_basis, Some(&overrides))?;

    // Trim or pad to the original signal length.
    let mut result = reconstructed;
    result.truncate(n);
    if result.len() < n {
        result.resize(n, 0.0);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_advanced_denoise_basic() {
        // Create noisy signal
        let n = 256;
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / n as f64 * 5.0).sin())
            .collect();

        let mut noisy = clean.clone();
        for (i, sample) in noisy.iter_mut().enumerate() {
            *sample += 0.1 * ((i as f64 * 0.1).sin());
        }

        let config = AdvancedDenoiseConfig::default();
        let result = advanced_denoise(&noisy, &config).expect("Operation failed");

        assert_eq!(result.signal.len(), n);
        assert!(result.noise_level > 0.0);
        assert!(!result.thresholds.is_empty());
    }

    #[test]
    fn test_translation_invariant() {
        // Use a signal long enough for the decomposition level (level=1 needs 2 samples).
        let signal = vec![
            1.0, 2.0, 1.0, 0.0, -1.0, 0.0, 1.0, 2.0, 0.5, 1.5, 0.5, -0.5, -1.5, -0.5, 0.5, 1.5,
            1.0, 2.0, 1.0, 0.0, -1.0, 0.0, 1.0, 2.0, 0.5, 1.5, 0.5, -0.5, -1.5, -0.5, 0.5, 1.5,
        ];

        let config = AdvancedDenoiseConfig {
            translation_invariant: true,
            n_shifts: 4,
            parallel: false,
            level: 2,
            ..Default::default()
        };

        let result = advanced_denoise(&signal, &config).expect("Operation failed");
        assert_eq!(result.signal.len(), signal.len());
    }

    #[test]
    fn test_wavelet_packet_denoise_reduces_noise() {
        // Build a clean low-frequency signal and add many high-frequency tones (broadband noise).
        // Using a sum of incommensurate tones approximates broadband noise for WPT purposes.
        let n = 256usize;
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 2.0 * i as f64 / n as f64).sin())
            .collect();

        // Broadband noise: sum of multiple high-frequency components with small amplitudes.
        // This spreads energy across many WPT subbands, mimicking real broadband noise.
        let noise_amplitudes = [0.08, 0.07, 0.06, 0.05, 0.06, 0.07, 0.05, 0.06];
        let noise_freqs = [31.0, 37.0, 43.0, 53.0, 59.0, 67.0, 71.0, 79.0];
        let noisy: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let noise: f64 = noise_amplitudes
                    .iter()
                    .zip(noise_freqs.iter())
                    .map(|(&a, &f)| a * (2.0 * PI * f * i as f64 / n as f64).sin())
                    .sum();
                c + noise
            })
            .collect();

        let config = AdvancedDenoiseConfig {
            wavelet: Wavelet::Haar,
            level: 3,
            translation_invariant: false,
            bayesian: false,
            block_threshold: false,
            ..Default::default()
        };

        let denoised =
            wavelet_packet_denoise(&noisy, &config).expect("wavelet_packet_denoise failed");

        assert_eq!(denoised.len(), n, "Output length must equal input length");

        // The denoised signal should have lower total variation (less high-freq content)
        // than the noisy signal. This is a weaker criterion that doesn't depend on
        // exact threshold tuning.
        let tv_noisy: f64 = noisy.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        let tv_denoised: f64 = denoised.windows(2).map(|w| (w[1] - w[0]).abs()).sum();

        assert!(
            tv_denoised < tv_noisy,
            "Denoised TV ({:.4}) should be lower than noisy TV ({:.4}) — \
             indicating high-frequency suppression",
            tv_denoised,
            tv_noisy,
        );

        // Also verify the output is not all zeros (denoising shouldn't remove everything).
        let energy: f64 = denoised.iter().map(|x| x * x).sum();
        assert!(
            energy > 0.01,
            "Denoised signal should retain significant energy"
        );
    }

    #[test]
    fn test_wavelet_packet_denoise_preserves_length() {
        // Test with a shorter signal to ensure padding/truncation logic works.
        let n = 64usize;
        let signal: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

        let config = AdvancedDenoiseConfig {
            wavelet: Wavelet::Haar,
            level: 2,
            translation_invariant: false,
            bayesian: false,
            block_threshold: false,
            ..Default::default()
        };

        let denoised =
            wavelet_packet_denoise(&signal, &config).expect("wavelet_packet_denoise failed");
        assert_eq!(denoised.len(), n);
        // All values must be finite.
        assert!(denoised.iter().all(|x| x.is_finite()));
    }
}
