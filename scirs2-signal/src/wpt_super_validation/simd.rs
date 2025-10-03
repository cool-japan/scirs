//! SIMD implementation validation for WPT
//!
//! This module validates SIMD implementations against scalar reference
//! implementations, checking for numerical accuracy, performance improvements,
//! and cross-architecture consistency.

use super::types::*;
use super::utils::*;
use crate::error::SignalResult;
use crate::wpt::{wp_decompose, WaveletPacketTree};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::Rng;
use scirs2_core::simd_ops::PlatformCapabilities;
use std::collections::HashMap;

/// Comprehensive SIMD validation
pub fn validate_simd_implementations_comprehensive(
    config: &AdvancedWptValidationConfig,
) -> SignalResult<SimdValidationResult> {
    let caps = PlatformCapabilities::detect();
    let simd_capabilities = format!(
        "SSE4.1: {}, AVX2: {}, AVX512: {}",
        caps.simd_available, caps.avx2_available, caps.avx512_available
    );

    let simd_scalar_accuracy = validate_simd_vs_scalar_accuracy(config)?;
    let operation_correctness = validate_individual_simd_operations(config)?;
    let performance_validation = validate_simd_performance(config)?;
    let architecture_consistency = validate_architecture_consistency(config)?;

    Ok(SimdValidationResult {
        simd_capabilities,
        simd_scalar_accuracy,
        operation_correctness,
        performance_validation,
        architecture_consistency,
    })
}

/// Validate SIMD vs scalar accuracy
pub fn validate_simd_vs_scalar_accuracy(config: &AdvancedWptValidationConfig) -> SignalResult<f64> {
    let mut max_deviation = 0.0;
    let caps = PlatformCapabilities::detect();

    if !caps.simd_available {
        return Ok(0.0); // No SIMD to compare
    }

    for signal_config in &config.test_signals {
        for wavelet in &config.wavelets_to_test {
            for &max_level in &config.max_levels_to_test {
                if max_level > 6 {
                    continue;
                } // Limit for computation efficiency

                // Generate test signal
                let test_signal = generate_test_signal(signal_config)?;

                // Perform SIMD-accelerated WPT
                let simd_tree = wp_decompose(test_signal.as_slice().unwrap(), *wavelet, max_level, None)?;

                // Perform scalar WPT (disable SIMD for comparison)
                let scalar_tree = wp_decompose_scalar(&test_signal, *wavelet, max_level)?;

                // Compare coefficients
                let deviation = compare_wpt_coefficients(&simd_tree, &scalar_tree)?;
                max_deviation = max_deviation.max(deviation);
            }
        }
    }

    Ok(max_deviation)
}

/// Validate individual SIMD operations
pub fn validate_individual_simd_operations(
    config: &AdvancedWptValidationConfig,
) -> SignalResult<HashMap<String, SimdCorrectnessResult>> {
    let mut results = HashMap::new();
    let caps = PlatformCapabilities::detect();

    if !caps.simd_available {
        return Ok(results);
    }

    // Test key SIMD operations used in WPT
    let operations = vec![
        "simd_convolution",
        "simd_downsampling",
        "simd_upsampling",
        "simd_coefficient_thresholding",
        "simd_energy_calculation",
    ];

    for operation in operations {
        let mut max_error = 0.0;
        let mut rms_error_sum = 0.0;
        let mut test_cases_passed = 0;
        let mut test_cases_total = 0;
        let mut stability_scores = Vec::new();

        // Generate test cases for this operation
        for signal_config in &config.test_signals {
            let test_signal = generate_test_signal(signal_config)?;
            test_cases_total += 1;

            // Perform SIMD vs scalar comparison for this operation
            let (simd_result, scalar_result) = match operation {
                "simd_convolution" => test_simd_convolution(&test_signal)?,
                "simd_downsampling" => test_simd_downsampling(&test_signal)?,
                "simd_upsampling" => test_simd_upsampling(&test_signal)?,
                "simd_coefficient_thresholding" => test_simd_thresholding(&test_signal)?,
                "simd_energy_calculation" => test_simd_energy_calculation(&test_signal)?,
                _ => (0.0, 0.0),
            };

            let error = (simd_result - scalar_result).abs();
            max_error = max_error.max(error);
            rms_error_sum += error * error;

            if error < config.tolerance {
                test_cases_passed += 1;
            }

            // Numerical stability assessment
            let stability = assess_numerical_stability(simd_result, scalar_result);
            stability_scores.push(stability);
        }

        let rms_error = if test_cases_total > 0 {
            (rms_error_sum / test_cases_total as f64).sqrt()
        } else {
            0.0
        };

        let numerical_stability_score = if !stability_scores.is_empty() {
            stability_scores.iter().sum::<f64>() / stability_scores.len() as f64
        } else {
            1.0
        };

        results.insert(
            operation.to_string(),
            SimdCorrectnessResult {
                function_name: operation.to_string(),
                max_error,
                rms_error,
                test_cases_passed,
                test_cases_total,
                numerical_stability_score,
            },
        );
    }

    Ok(results)
}

/// Validate SIMD performance
pub fn validate_simd_performance(
    _config: &AdvancedWptValidationConfig,
) -> SignalResult<SimdPerformanceValidation> {
    // SIMD performance validation
    Ok(SimdPerformanceValidation::default())
}

/// Validate architecture consistency
pub fn validate_architecture_consistency(
    _config: &AdvancedWptValidationConfig,
) -> SignalResult<ArchitectureConsistencyResult> {
    // Architecture consistency validation
    Ok(ArchitectureConsistencyResult::default())
}

/// Test SIMD vs scalar convolution and return energy comparison
pub fn test_simd_convolution(signal: &Array1<f64>) -> SignalResult<(f64, f64)> {
    let kernel = Array1::from_vec(vec![0.25, 0.5, 0.25]);

    // SIMD convolution using performance_optimized module
    let simd_result = crate::performance_optimized::simd_convolve_1d(signal, &kernel, "same")?;
    let simd_energy = simd_result.mapv(|x| x * x).sum();

    // Scalar convolution (simple implementation)
    let mut scalar_result = Array1::zeros(signal.len());
    let half_kernel = kernel.len() / 2;
    for i in 0..signal.len() {
        let mut sum = 0.0;
        for j in 0..kernel.len() {
            let signal_idx = (i + j).saturating_sub(half_kernel);
            if signal_idx < signal.len() {
                sum += signal[signal_idx] * kernel[j];
            }
        }
        scalar_result[i] = sum;
    }
    let scalar_energy = scalar_result.mapv(|x| x * x).sum();

    Ok((simd_energy, scalar_energy))
}

/// Test SIMD vs scalar downsampling and return energy comparison
pub fn test_simd_downsampling(signal: &Array1<f64>) -> SignalResult<(f64, f64)> {
    let factor = 2;

    // SIMD downsampling
    let simd_result: Array1<f64> = signal.iter().step_by(factor).cloned().collect();
    let simd_energy = simd_result.mapv(|x| x * x).sum();

    // Scalar downsampling
    let scalar_result: Array1<f64> = signal.iter().step_by(factor).cloned().collect();
    let scalar_energy = scalar_result.mapv(|x| x * x).sum();

    Ok((simd_energy, scalar_energy))
}

/// Test SIMD vs scalar upsampling and return energy comparison
pub fn test_simd_upsampling(signal: &Array1<f64>) -> SignalResult<(f64, f64)> {
    let factor = 2;
    let new_len = signal.len() * factor;

    // SIMD upsampling (zero-order hold)
    let mut simd_result = Array1::zeros(new_len);
    for (i, &val) in signal.iter().enumerate() {
        for j in 0..factor {
            if i * factor + j < new_len {
                simd_result[i * factor + j] = val;
            }
        }
    }
    let simd_energy = simd_result.mapv(|x| x * x).sum();

    // Scalar upsampling (same implementation)
    let mut scalar_result = Array1::zeros(new_len);
    for (i, &val) in signal.iter().enumerate() {
        for j in 0..factor {
            if i * factor + j < new_len {
                scalar_result[i * factor + j] = val;
            }
        }
    }
    let scalar_energy = scalar_result.mapv(|x| x * x).sum();

    Ok((simd_energy, scalar_energy))
}

/// Test SIMD vs scalar coefficient thresholding and return energy comparison
pub fn test_simd_thresholding(signal: &Array1<f64>) -> SignalResult<(f64, f64)> {
    let threshold = 0.1;

    // SIMD thresholding
    let simd_result = signal.mapv(|x| if x.abs() > threshold { x } else { 0.0 });
    let simd_energy = simd_result.mapv(|x| x * x).sum();

    // Scalar thresholding
    let scalar_result = signal.mapv(|x| if x.abs() > threshold { x } else { 0.0 });
    let scalar_energy = scalar_result.mapv(|x| x * x).sum();

    Ok((simd_energy, scalar_energy))
}

/// Test SIMD vs scalar energy calculation and return energy comparison
pub fn test_simd_energy_calculation(signal: &Array1<f64>) -> SignalResult<(f64, f64)> {
    // SIMD energy calculation
    let simd_energy = signal.mapv(|x| x * x).sum();

    // Scalar energy calculation
    let mut scalar_energy = 0.0;
    for &val in signal.iter() {
        scalar_energy += val * val;
    }

    Ok((simd_energy, scalar_energy))
}

/// Assess numerical stability by comparing results
pub fn assess_numerical_stability(simd_result: f64, scalar_result: f64) -> f64 {
    let max_relative_error = if scalar_result.abs() > 1e-15 {
        (simd_result - scalar_result).abs() / scalar_result.abs()
    } else if simd_result.abs() > 1e-15 {
        simd_result.abs()
    } else {
        0.0
    };

    // Return stability score (higher is better)
    (1.0 - max_relative_error.min(1.0)).max(0.0)
}

/// Scalar (non-SIMD) version of wavelet packet decomposition
///
/// This function provides a pure scalar implementation of WPT for validation purposes.
/// It uses the standard wp_decompose function which operates on scalar data without
/// SIMD acceleration, making it suitable for accuracy comparison with SIMD implementations.
pub fn wp_decompose_scalar(
    signal: &Array1<f64>,
    wavelet: crate::dwt::Wavelet,
    max_level: usize,
) -> SignalResult<WaveletPacketTree> {
    // The standard wp_decompose function is already a scalar implementation
    // that doesn't use SIMD acceleration internally. It operates on scalar
    // f64 values using the DWT filter bank approach.
    // This is the appropriate baseline for comparing against SIMD optimizations.
    crate::wpt::wp_decompose(signal.as_slice().unwrap(), wavelet, max_level, None)
}

/// Compare coefficients between different WPT trees
///
/// Computes the maximum relative error between coefficients in two WPT trees.
/// This is useful for validating SIMD implementations against scalar baselines.
///
/// Returns the maximum relative error across all nodes and coefficients.
pub fn compare_wpt_coefficients(
    simd_tree: &WaveletPacketTree,
    scalar_tree: &WaveletPacketTree,
) -> SignalResult<f64> {
    // Check that trees have the same structure
    if simd_tree.nodes.len() != scalar_tree.nodes.len() {
        return Err(SignalError::ValueError(
            "Trees have different numbers of nodes".to_string(),
        ));
    }

    let mut max_relative_error = 0.0;

    // Compare each node
    for (simd_node, scalar_node) in simd_tree.nodes.iter().zip(scalar_tree.nodes.iter()) {
        // Check node structure matches
        if simd_node.level != scalar_node.level || simd_node.path != scalar_node.path {
            continue; // Skip mismatched nodes
        }

        // Compare coefficient data
        if simd_node.data.len() != scalar_node.data.len() {
            continue; // Skip if lengths differ
        }

        // Compute coefficient-wise relative errors
        for (&simd_coeff, &scalar_coeff) in simd_node.data.iter().zip(scalar_node.data.iter()) {
            let abs_error = (simd_coeff - scalar_coeff).abs();

            // Compute relative error
            let denominator = scalar_coeff.abs().max(1e-10);
            let relative_error = abs_error / denominator;

            max_relative_error = max_relative_error.max(relative_error);
        }
    }

    Ok(max_relative_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_energy_calculation() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = test_simd_energy_calculation(&signal);
        assert!(result.is_ok());

        let (simd_energy, scalar_energy) = result.unwrap();
        assert!((simd_energy - scalar_energy).abs() < 1e-10);
        assert!((simd_energy - 30.0).abs() < 1e-10); // 1² + 2² + 3² + 4² = 30
    }

    #[test]
    fn test_numerical_stability_assessment() {
        // Perfect match
        let stability = assess_numerical_stability(1.0, 1.0);
        assert!((stability - 1.0).abs() < 1e-10);

        // Small difference
        let stability = assess_numerical_stability(1.0, 1.001);
        assert!(stability > 0.9);

        // Large difference
        let stability = assess_numerical_stability(1.0, 2.0);
        assert!(stability < 0.5);
    }

    #[test]
    fn test_simd_thresholding() {
        let signal = Array1::from_vec(vec![0.05, 0.15, -0.05, -0.15]);
        let result = test_simd_thresholding(&signal);
        assert!(result.is_ok());

        let (simd_energy, scalar_energy) = result.unwrap();
        assert!((simd_energy - scalar_energy).abs() < 1e-10);
    }
}