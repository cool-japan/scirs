//! Multitaper parametric spectral estimation
//!
//! This module combines the robustness of multitaper methods with the
//! high resolution of parametric methods using DPSS tapers and weighted combining.

use super::core::advanced_enhanced_arma;
use super::types::*;
use super::utils::{compute_ar_psd, compute_arma_psd, generate_frequency_grid};
use crate::error::{SignalError, SignalResult};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Multi-taper parametric spectral estimation
///
/// This function combines the robustness of multitaper methods with the
/// high resolution of parametric methods
pub fn multitaper_parametric_estimation(
    signal: &[f64],
    config: &MultitaperParametricConfig,
) -> SignalResult<MultitaperParametricResult> {
    // Signal validation - check for finite values
    if signal.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }

    let n = signal.len();
    let nw = config.time_bandwidth;
    let k = config.num_tapers;

    if n < config.ar_order * 5 {
        return Err(SignalError::ValueError(
            "Signal too short for multitaper parametric estimation".to_string(),
        ));
    }

    // Generate DPSS tapers
    let tapers = generate_dpss_tapers(n, nw, k)?;

    // Estimate AR parameters for each tapered signal
    let mut individual_estimates = Vec::new();
    let mut individual_spectra = Vec::new();

    for taper_idx in 0..k {
        let taper = tapers.row(taper_idx);
        let tapered_signal: Vec<f64> = signal
            .iter()
            .zip(taper.iter())
            .map(|(&s, &t)| s * t)
            .collect();

        let tapered_signal_array = Array1::from_vec(tapered_signal);

        // AR estimation for this taper
        let ar_config = AdvancedEnhancedConfig::default();
        let ar_result =
            advanced_enhanced_arma(&tapered_signal_array, config.ar_order, 0, &ar_config)?;

        // Compute PSD for this taper
        let freqs = generate_frequency_grid(512, 1.0);
        let psd = compute_ar_psd(&ar_result.ar_coeffs, ar_result.noise_variance, &freqs)?;

        // Create spectral estimate structure
        let spectral_estimate = SpectralEstimate {
            psd: Array1::from_vec(psd),
            frequencies: Array1::from_vec(freqs),
            method: "Multitaper AR".to_string(),
            order: config.ar_order,
            confidence_intervals: None,
        };

        individual_estimates.push(spectral_estimate.clone());
        individual_spectra.push(spectral_estimate);
    }

    // Combine spectral estimates using appropriate weights
    let combined_spectrum =
        combine_multitaper_spectra(&individual_spectra, &config.combination_method)?;

    // Compute coherence estimates between tapers
    let coherence_estimates = compute_taper_coherence(&individual_spectra)?;

    // Compute variance estimates if jackknife is enabled
    let variance_estimates = if config.jackknife_variance {
        Some(compute_jackknife_variance(&individual_spectra)?)
    } else {
        None
    };

    // Compute bias correction factors
    let bias_correction_factors = compute_bias_correction_factors(k, nw)?;

    Ok(MultitaperParametricResult {
        combined_spectrum,
        individual_estimates,
        coherence_estimates,
        variance_estimates,
        degrees_of_freedom: 2.0 * k as f64,
        bias_correction_factors,
    })
}

/// Generate DPSS (Discrete Prolate Spheroidal Sequence) tapers
fn generate_dpss_tapers(n: usize, nw: f64, k: usize) -> SignalResult<Array2<f64>> {
    // Simplified DPSS implementation
    // In a full implementation, this would solve the eigenvalue problem for DPSS
    let mut tapers = Array2::zeros((k, n));

    for taper_idx in 0..k {
        for i in 0..n {
            let t = i as f64 / n as f64;
            let phase = 2.0 * PI * (taper_idx as f64 + 1.0) * t;

            // Simple sinusoidal tapers (approximation)
            // Real DPSS would be computed from eigenvalue decomposition
            let window = 0.5 * (1.0 - (2.0 * PI * t).cos()); // Hanning-like base
            tapers[(taper_idx, i)] = window * phase.sin();
        }

        // Normalize the taper
        let norm = tapers.row(taper_idx).mapv(|x| x * x).sum().sqrt();
        if norm > 0.0 {
            for i in 0..n {
                tapers[(taper_idx, i)] /= norm;
            }
        }
    }

    Ok(tapers)
}

/// Combine multitaper spectral estimates
fn combine_multitaper_spectra(
    spectra: &[SpectralEstimate],
    method: &CombinationMethod,
) -> SignalResult<SpectralEstimate> {
    if spectra.is_empty() {
        return Err(SignalError::ValueError(
            "No spectral estimates to combine".to_string(),
        ));
    }

    let n_freq = spectra[0].psd.len();
    let frequencies = spectra[0].frequencies.clone();
    let mut combined_psd = Array1::zeros(n_freq);

    match method {
        CombinationMethod::ArithmeticMean => {
            for spectrum in spectra {
                combined_psd = combined_psd + &spectrum.psd;
            }
            combined_psd /= spectra.len() as f64;
        }
        CombinationMethod::GeometricMean => {
            // Initialize with ones for geometric mean
            combined_psd = Array1::ones(n_freq);
            for spectrum in spectra {
                for i in 0..n_freq {
                    combined_psd[i] *= spectrum.psd[i].max(1e-12); // Avoid log(0)
                }
            }
            combined_psd = combined_psd.mapv(|x| x.powf(1.0 / spectra.len() as f64));
        }
        CombinationMethod::MedianCombination => {
            for i in 0..n_freq {
                let mut values: Vec<f64> = spectra.iter().map(|s| s.psd[i]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                combined_psd[i] = if values.len() % 2 == 0 {
                    (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                } else {
                    values[values.len() / 2]
                };
            }
        }
        CombinationMethod::WeightedAverage => {
            // Use inverse variance weighting (simplified)
            let mut weights = vec![1.0; spectra.len()];
            for (j, spectrum) in spectra.iter().enumerate() {
                let variance = spectrum.psd.var(0.0);
                weights[j] = if variance > 0.0 { 1.0 / variance } else { 1.0 };
            }

            let weight_sum: f64 = weights.iter().sum();
            for (j, spectrum) in spectra.iter().enumerate() {
                let w = weights[j] / weight_sum;
                combined_psd = combined_psd + &spectrum.psd.mapv(|x| x * w);
            }
        }
        CombinationMethod::AdaptiveWeighting => {
            // Adaptive weighting based on local spectral characteristics
            combine_adaptive_weighting(spectra, &mut combined_psd)?;
        }
    }

    Ok(SpectralEstimate {
        psd: combined_psd,
        frequencies,
        method: format!("Multitaper Combined ({:?})", method),
        order: spectra[0].order,
        confidence_intervals: None,
    })
}

/// Adaptive weighting combination
fn combine_adaptive_weighting(
    spectra: &[SpectralEstimate],
    combined_psd: &mut Array1<f64>,
) -> SignalResult<()> {
    let n_freq = combined_psd.len();

    for i in 0..n_freq {
        let values: Vec<f64> = spectra.iter().map(|s| s.psd[i]).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        // Compute weights based on deviation from mean
        let mut weights = Vec::new();
        for &val in &values {
            let deviation = (val - mean).abs();
            let weight = 1.0 / (1.0 + deviation / mean.max(1e-12));
            weights.push(weight);
        }

        let weight_sum: f64 = weights.iter().sum();
        if weight_sum > 0.0 {
            let mut weighted_sum = 0.0;
            for (j, &val) in values.iter().enumerate() {
                weighted_sum += val * weights[j] / weight_sum;
            }
            combined_psd[i] = weighted_sum;
        } else {
            combined_psd[i] = mean;
        }
    }

    Ok(())
}

/// Compute coherence estimates between tapers
fn compute_taper_coherence(spectra: &[SpectralEstimate]) -> SignalResult<Array1<f64>> {
    if spectra.len() < 2 {
        return Ok(Array1::ones(spectra[0].psd.len()));
    }

    let n_freq = spectra[0].psd.len();
    let mut coherence = Array1::zeros(n_freq);

    // Compute average coherence across all taper pairs
    let mut pair_count = 0;
    for i in 0..spectra.len() {
        for j in i + 1..spectra.len() {
            for k in 0..n_freq {
                let p1 = spectra[i].psd[k];
                let p2 = spectra[j].psd[k];
                let cross_power = (p1 * p2).sqrt();
                let coherence_val = if p1 > 0.0 && p2 > 0.0 {
                    cross_power / ((p1 + p2) / 2.0)
                } else {
                    0.0
                };
                coherence[k] += coherence_val;
            }
            pair_count += 1;
        }
    }

    if pair_count > 0 {
        coherence /= pair_count as f64;
    }

    Ok(coherence)
}

/// Compute jackknife variance estimates
fn compute_jackknife_variance(spectra: &[SpectralEstimate]) -> SignalResult<Array1<f64>> {
    let n_tapers = spectra.len();
    let n_freq = spectra[0].psd.len();
    let mut variance = Array1::zeros(n_freq);

    if n_tapers < 2 {
        return Ok(variance);
    }

    // Compute jackknife estimates
    for i in 0..n_freq {
        let mut jackknife_estimates = Vec::new();

        // For each leave-one-out combination
        for leave_out in 0..n_tapers {
            let mut sum = 0.0;
            let mut count = 0;

            for (j, spectrum) in spectra.iter().enumerate() {
                if j != leave_out {
                    sum += spectrum.psd[i];
                    count += 1;
                }
            }

            if count > 0 {
                jackknife_estimates.push(sum / count as f64);
            }
        }

        // Compute variance of jackknife estimates
        if !jackknife_estimates.is_empty() {
            let mean = jackknife_estimates.iter().sum::<f64>() / jackknife_estimates.len() as f64;
            let var = jackknife_estimates
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / (jackknife_estimates.len() - 1) as f64;
            variance[i] = var * (n_tapers - 1) as f64; // Jackknife variance correction
        }
    }

    Ok(variance)
}

/// Compute bias correction factors
fn compute_bias_correction_factors(k: usize, nw: f64) -> SignalResult<Array1<f64>> {
    // Simplified bias correction based on number of tapers and time-bandwidth product
    let n_freq = 512; // Default frequency grid size
    let mut factors = Array1::ones(n_freq);

    // Apply frequency-dependent bias correction
    for i in 0..n_freq {
        let freq_norm = i as f64 / n_freq as f64;
        let bias_factor = 1.0 - (freq_norm * nw / k as f64).min(0.1);
        factors[i] = bias_factor.max(0.5); // Ensure reasonable correction
    }

    Ok(factors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multitaper_parametric_estimation() {
        // Use longer signal for multitaper parametric estimation
        let mut signal = vec![];
        for i in 0..256 {
            signal.push(
                1.0 + 0.7 * (i as f64 * 0.08).sin()
                    + 0.5 * (i as f64 * 0.12).cos()
                    + 0.3 * (i as f64 * 0.04).sin()
                    + 0.1 * rand::random::<f64>(), // Add small amount of noise
            );
        }

        let config = MultitaperParametricConfig::default();

        let result = multitaper_parametric_estimation(&signal, &config);
        if result.is_err() {
            println!("Multitaper error: {:?}", result.as_ref().err());
        }
        assert!(result.is_ok());

        let mt_result = result.unwrap();
        assert!(!mt_result.combined_spectrum.psd.is_empty());
        assert!(!mt_result.individual_estimates.is_empty());
        assert!(mt_result.degrees_of_freedom > 0.0);
    }

    #[test]
    fn test_generate_dpss_tapers() {
        let n = 32;
        let nw = 2.5;
        let k = 4;

        let result = generate_dpss_tapers(n, nw, k);
        assert!(result.is_ok());

        let tapers = result.unwrap();
        assert_eq!(tapers.dim(), (k, n));

        // Check that tapers are normalized
        for i in 0..k {
            let norm = tapers.row(i).mapv(|x| x * x).sum().sqrt();
            assert!((norm - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_combine_multitaper_spectra() {
        let psd1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let psd2 = Array1::from_vec(vec![1.5, 2.5, 2.5, 3.5]);
        let freqs = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.3]);

        let spectra = vec![
            SpectralEstimate {
                psd: psd1,
                frequencies: freqs.clone(),
                method: "Test".to_string(),
                order: 2,
                confidence_intervals: None,
            },
            SpectralEstimate {
                psd: psd2,
                frequencies: freqs,
                method: "Test".to_string(),
                order: 2,
                confidence_intervals: None,
            },
        ];

        let result = combine_multitaper_spectra(&spectra, &CombinationMethod::ArithmeticMean);
        assert!(result.is_ok());

        let combined = result.unwrap();
        assert_eq!(combined.psd[0], 1.25); // (1.0 + 1.5) / 2
        assert_eq!(combined.psd[1], 2.25); // (2.0 + 2.5) / 2
    }
}
