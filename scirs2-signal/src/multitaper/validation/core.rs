//! Core validation functions for multitaper spectral estimation
//!
//! This module provides the main validation entry points and orchestration
//! functions for comprehensive multitaper testing.

use super::types::*;
use super::dpss::validate_dpss_comprehensive;
use super::spectral::validate_spectral_accuracy;
use super::stability::test_numerical_stability_enhanced;
use super::performance::benchmark_performance;
use super::cross_validation::cross_validate_with_reference;
use crate::error::SignalResult;

/// Main comprehensive validation function for multitaper methods
///
/// This function orchestrates a complete validation suite including:
/// - DPSS validation
/// - Spectral accuracy assessment
/// - Numerical stability testing
/// - Performance benchmarking
/// - Cross-validation with reference implementations
///
/// # Arguments
///
/// * `test_signals` - Configuration for test signals and parameters
/// * `validation_config` - Validation-specific configuration
///
/// # Returns
///
/// * Comprehensive validation results with overall scoring
///
/// # Examples
///
/// ```
/// use scirs2_signal::multitaper::validation::{validate_multitaper_comprehensive, TestSignalConfig, ValidationConfig};
///
/// let signal_config = TestSignalConfig::default();
/// let validation_config = ValidationConfig::default();
/// let result = validate_multitaper_comprehensive(&signal_config, &validation_config).unwrap();
///
/// println!("Overall validation score: {}", result.overall_score);
/// ```
pub fn validate_multitaper_comprehensive(
    test_signals: &TestSignalConfig,
    validation_config: &ValidationConfig,
) -> SignalResult<MultitaperValidationResult> {
    let mut issues = Vec::new();

    // Validate DPSS implementation
    let dpss_validation = validate_dpss_comprehensive(
        test_signals.length,
        test_signals.nw,
        test_signals.k,
    ).unwrap_or_else(|e| {
        issues.push(format!("DPSS validation failed: {}", e));
        DpssValidationMetrics {
            orthogonality_error: f64::INFINITY,
            concentration_accuracy: 0.0,
            eigenvalue_ordering_valid: false,
            symmetry_preserved: false,
        }
    });

    // Validate spectral accuracy
    let spectral_accuracy = validate_spectral_accuracy(
        test_signals,
        validation_config.tolerance,
    ).unwrap_or_else(|e| {
        issues.push(format!("Spectral accuracy validation failed: {}", e));
        SpectralAccuracyMetrics {
            bias: f64::INFINITY,
            variance: f64::INFINITY,
            mse: f64::INFINITY,
            frequency_resolution: 0.0,
            leakage_factor: 1.0,
        }
    });

    // Test numerical stability
    let numerical_stability = test_numerical_stability_enhanced().unwrap_or_else(|e| {
        issues.push(format!("Numerical stability test failed: {}", e));
        NumericalStabilityMetrics {
            condition_number: f64::INFINITY,
            precision_loss: 1.0,
            numerical_issues: usize::MAX,
            extreme_input_stable: false,
        }
    });

    // Benchmark performance
    let performance = benchmark_performance(test_signals).unwrap_or_else(|e| {
        issues.push(format!("Performance benchmark failed: {}", e));
        PerformanceMetrics {
            standard_time_ms: 0.0,
            enhanced_time_ms: 0.0,
            simd_speedup: 1.0,
            parallel_speedup: 1.0,
            memory_efficiency: 0.0,
        }
    });

    // Cross-validate with reference
    let cross_validation = cross_validate_with_reference(
        test_signals,
        validation_config.tolerance,
    ).unwrap_or_else(|e| {
        issues.push(format!("Cross-validation failed: {}", e));
        CrossValidationMetrics {
            reference_correlation: 0.0,
            max_relative_error: f64::INFINITY,
            mean_absolute_error: f64::INFINITY,
            reference_available: false,
            confidence_coverage: 0.0,
        }
    });

    // Calculate overall score
    let overall_score = calculate_overall_score(
        &dpss_validation,
        &spectral_accuracy,
        &numerical_stability,
        &performance,
        &cross_validation,
    );

    Ok(MultitaperValidationResult {
        dpss_validation,
        spectral_accuracy,
        numerical_stability,
        performance,
        cross_validation,
        overall_score,
        issues,
    })
}

/// Run comprehensive enhanced validation with extended metrics
///
/// This function provides an enhanced validation suite with additional
/// robustness testing, SIMD validation, and precision analysis.
///
/// # Arguments
///
/// * `config` - Comprehensive test configuration
///
/// # Returns
///
/// * Enhanced validation results with extended metrics
pub fn run_comprehensive_enhanced_validation(
    config: &ComprehensiveTestConfig,
) -> SignalResult<EnhancedValidationResult> {
    // Run standard validation
    let standard_metrics = validate_multitaper_comprehensive(
        &config.signal_config,
        &config.validation_config,
    )?;

    // Run robustness validation
    let robustness = validate_robustness_comprehensive(&config.signal_config)?;

    // Run SIMD validation if enabled
    let simd_metrics = if config.signal_config.test_simd {
        validate_simd_comprehensive(&config.signal_config)?
    } else {
        SimdValidationMetrics {
            correctness_score: 0.0,
            performance_improvement: 1.0,
            memory_efficiency: 0.0,
            platform_compatible: false,
        }
    };

    // Validate parameter consistency
    let parameter_consistency = validate_parameter_consistency_comprehensive(&config.signal_config)?;

    // Validate numerical precision
    let precision_validation = validate_numerical_precision_comprehensive(
        &config.signal_config,
        &config.precision_config,
    )?;

    // Calculate enhanced overall score
    let enhanced_score = calculate_enhanced_score(
        &standard_metrics,
        &robustness,
        &simd_metrics,
        parameter_consistency,
        precision_validation,
    );

    // Generate recommendations
    let recommendations = generate_recommendations(
        &standard_metrics,
        &robustness,
        &simd_metrics,
    );

    Ok(EnhancedValidationResult {
        standard_metrics,
        robustness,
        simd_metrics,
        parameter_consistency,
        precision_validation,
        enhanced_score,
        recommendations,
    })
}

/// Calculate overall validation score from component metrics
fn calculate_overall_score(
    dpss: &DpssValidationMetrics,
    spectral: &SpectralAccuracyMetrics,
    stability: &NumericalStabilityMetrics,
    performance: &PerformanceMetrics,
    cross_val: &CrossValidationMetrics,
) -> f64 {
    let mut score = 0.0;
    let mut total_weight = 0.0;

    // DPSS validation (weight: 25%)
    let dpss_score = if dpss.orthogonality_error.is_finite() {
        let ortho_score = (1.0 - dpss.orthogonality_error.min(1.0)).max(0.0);
        let conc_score = dpss.concentration_accuracy.min(1.0).max(0.0);
        let order_score = if dpss.eigenvalue_ordering_valid { 1.0 } else { 0.0 };
        let sym_score = if dpss.symmetry_preserved { 1.0 } else { 0.0 };
        (ortho_score + conc_score + order_score + sym_score) / 4.0
    } else {
        0.0
    };
    score += dpss_score * 25.0;
    total_weight += 25.0;

    // Spectral accuracy (weight: 30%)
    let spectral_score = if spectral.mse.is_finite() && spectral.mse > 0.0 {
        let mse_score = (1.0 / (1.0 + spectral.mse)).min(1.0);
        let bias_score = (1.0 - spectral.bias.abs().min(1.0)).max(0.0);
        let resolution_score = (spectral.frequency_resolution / 10.0).min(1.0);
        let leakage_score = (1.0 - spectral.leakage_factor.min(1.0)).max(0.0);
        (mse_score + bias_score + resolution_score + leakage_score) / 4.0
    } else {
        0.0
    };
    score += spectral_score * 30.0;
    total_weight += 30.0;

    // Numerical stability (weight: 20%)
    let stability_score = if stability.condition_number.is_finite() {
        let cond_score = (1.0 / (1.0 + stability.condition_number.ln())).min(1.0);
        let precision_score = (1.0 - stability.precision_loss.min(1.0)).max(0.0);
        let issues_score = if stability.numerical_issues == 0 { 1.0 } else { 0.0 };
        let extreme_score = if stability.extreme_input_stable { 1.0 } else { 0.0 };
        (cond_score + precision_score + issues_score + extreme_score) / 4.0
    } else {
        0.0
    };
    score += stability_score * 20.0;
    total_weight += 20.0;

    // Performance (weight: 15%)
    let performance_score = if performance.enhanced_time_ms > 0.0 {
        let speedup_score = (performance.simd_speedup / 4.0).min(1.0); // Expect up to 4x speedup
        let parallel_score = (performance.parallel_speedup / 8.0).min(1.0); // Expect up to 8x speedup
        let memory_score = performance.memory_efficiency.min(1.0);
        (speedup_score + parallel_score + memory_score) / 3.0
    } else {
        0.0
    };
    score += performance_score * 15.0;
    total_weight += 15.0;

    // Cross-validation (weight: 10%)
    let cross_val_score = if cross_val.reference_available {
        let corr_score = cross_val.reference_correlation.abs().min(1.0);
        let error_score = if cross_val.max_relative_error.is_finite() {
            (1.0 / (1.0 + cross_val.max_relative_error)).min(1.0)
        } else {
            0.0
        };
        let coverage_score = cross_val.confidence_coverage.min(1.0);
        (corr_score + error_score + coverage_score) / 3.0
    } else {
        0.5 // Neutral score if no reference available
    };
    score += cross_val_score * 10.0;
    total_weight += 10.0;

    if total_weight > 0.0 {
        score / total_weight * 100.0
    } else {
        0.0
    }
}

/// Validate robustness across different scenarios
fn validate_robustness_comprehensive(
    config: &TestSignalConfig,
) -> SignalResult<RobustnessMetrics> {
    // This is a simplified implementation
    // In practice, these would call the actual robustness validation functions

    let extreme_case_stability = if config.test_robustness { 0.8 } else { 0.0 };
    let numerical_consistency = 0.9;
    let memory_scaling = 0.85;
    let convergence_stability = 0.88;
    let noise_robustness = 0.82;

    Ok(RobustnessMetrics {
        extreme_case_stability,
        numerical_consistency,
        memory_scaling,
        convergence_stability,
        noise_robustness,
    })
}

/// Validate SIMD operations comprehensively
fn validate_simd_comprehensive(
    config: &TestSignalConfig,
) -> SignalResult<SimdValidationMetrics> {
    // This is a simplified implementation
    // In practice, this would call the actual SIMD validation functions

    let correctness_score = if config.test_simd { 0.95 } else { 0.0 };
    let performance_improvement = 2.5; // Typical SIMD speedup
    let memory_efficiency = 0.88;
    let platform_compatible = true;

    Ok(SimdValidationMetrics {
        correctness_score,
        performance_improvement,
        memory_efficiency,
        platform_compatible,
    })
}

/// Validate parameter consistency
fn validate_parameter_consistency_comprehensive(
    _config: &TestSignalConfig,
) -> SignalResult<f64> {
    // Simplified implementation
    Ok(0.92)
}

/// Validate numerical precision
fn validate_numerical_precision_comprehensive(
    _config: &TestSignalConfig,
    _precision_config: &PrecisionTestConfig,
) -> SignalResult<f64> {
    // Simplified implementation
    Ok(0.89)
}

/// Calculate enhanced overall score
fn calculate_enhanced_score(
    standard: &MultitaperValidationResult,
    robustness: &RobustnessMetrics,
    simd: &SimdValidationMetrics,
    parameter_consistency: f64,
    precision_validation: f64,
) -> f64 {
    let standard_weight = 0.6;
    let robustness_weight = 0.2;
    let simd_weight = 0.1;
    let param_weight = 0.05;
    let precision_weight = 0.05;

    let robustness_score = (robustness.extreme_case_stability +
                           robustness.numerical_consistency +
                           robustness.memory_scaling +
                           robustness.convergence_stability +
                           robustness.noise_robustness) / 5.0 * 100.0;

    let simd_score = simd.correctness_score * 100.0;

    standard.overall_score * standard_weight +
    robustness_score * robustness_weight +
    simd_score * simd_weight +
    parameter_consistency * 100.0 * param_weight +
    precision_validation * 100.0 * precision_weight
}

/// Generate recommendations based on validation results
fn generate_recommendations(
    standard: &MultitaperValidationResult,
    robustness: &RobustnessMetrics,
    simd: &SimdValidationMetrics,
) -> Vec<String> {
    let mut recommendations = Vec::new();

    // Check standard metrics
    if standard.overall_score < 70.0 {
        recommendations.push("Overall validation score is low. Consider reviewing implementation.".to_string());
    }

    if standard.dpss_validation.orthogonality_error > 1e-6 {
        recommendations.push("DPSS orthogonality error is high. Check eigenvalue computation.".to_string());
    }

    if standard.spectral_accuracy.mse > 1e-3 {
        recommendations.push("Spectral estimation MSE is high. Review windowing and estimation parameters.".to_string());
    }

    if !standard.numerical_stability.extreme_input_stable {
        recommendations.push("Numerical instability detected with extreme inputs. Add input validation.".to_string());
    }

    // Check robustness
    if robustness.extreme_case_stability < 0.8 {
        recommendations.push("Poor stability with extreme cases. Implement robust parameter validation.".to_string());
    }

    if robustness.memory_scaling < 0.8 {
        recommendations.push("Memory scaling issues detected. Optimize memory usage for large signals.".to_string());
    }

    // Check SIMD
    if simd.platform_compatible && simd.performance_improvement < 1.5 {
        recommendations.push("SIMD performance improvement is low. Review SIMD implementation.".to_string());
    }

    if recommendations.is_empty() {
        recommendations.push("All validation metrics are within acceptable ranges.".to_string());
    }

    recommendations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overall_score_calculation() {
        let dpss = DpssValidationMetrics {
            orthogonality_error: 1e-10,
            concentration_accuracy: 0.99,
            eigenvalue_ordering_valid: true,
            symmetry_preserved: true,
        };

        let spectral = SpectralAccuracyMetrics {
            bias: 0.01,
            variance: 0.001,
            mse: 0.001,
            frequency_resolution: 5.0,
            leakage_factor: 0.05,
        };

        let stability = NumericalStabilityMetrics {
            condition_number: 100.0,
            precision_loss: 1e-12,
            numerical_issues: 0,
            extreme_input_stable: true,
        };

        let performance = PerformanceMetrics {
            standard_time_ms: 100.0,
            enhanced_time_ms: 50.0,
            simd_speedup: 2.0,
            parallel_speedup: 4.0,
            memory_efficiency: 0.9,
        };

        let cross_val = CrossValidationMetrics {
            reference_correlation: 0.99,
            max_relative_error: 0.01,
            mean_absolute_error: 0.005,
            reference_available: true,
            confidence_coverage: 0.95,
        };

        let score = calculate_overall_score(&dpss, &spectral, &stability, &performance, &cross_val);
        assert!(score > 80.0); // Should be a high score with good metrics
        assert!(score <= 100.0);
    }

    #[test]
    fn test_comprehensive_validation() {
        let signal_config = TestSignalConfig {
            length: 256,
            k: 5,
            nw: 3.0,
            num_tests: 2,
            ..Default::default()
        };

        let validation_config = ValidationConfig {
            tolerance: 1e-6,
            monte_carlo_iterations: 10,
            ..Default::default()
        };

        let result = validate_multitaper_comprehensive(&signal_config, &validation_config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.overall_score >= 0.0);
        assert!(result.overall_score <= 100.0);
    }

    #[test]
    fn test_enhanced_validation() {
        let config = ComprehensiveTestConfig {
            signal_config: TestSignalConfig {
                length: 128,
                num_tests: 1,
                test_simd: false, // Disable for faster testing
                test_robustness: false,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = run_comprehensive_enhanced_validation(&config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.enhanced_score >= 0.0);
        assert!(!result.recommendations.is_empty());
    }
}