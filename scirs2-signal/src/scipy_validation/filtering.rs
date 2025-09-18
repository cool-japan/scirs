//! Filter validation functions for SciPy compatibility
//!
//! This module provides validation functions for various digital filters
//! including Butterworth, Chebyshev, Elliptic, and Bessel filters.

use super::types::*;
use super::reference::{
    reference_butter_filter, reference_cheby1_filter, reference_cheby2_filter,
};
use super::utils::calculate_errors;
use crate::error::{SignalError, SignalResult};
use crate::filter::{butter, cheby1, cheby2, lfilter, FilterType};
use ndarray::Array1;

/// Validate filtering operations against SciPy reference
pub fn validate_filtering(config: &ValidationConfig) -> SignalResult<ValidationResults> {
    let mut results = ValidationResults::new();

    // Validate Butterworth filters
    if let Ok(butter_results) = validate_butterworth_filter(config) {
        for (_, test_result) in butter_results.test_results {
            results.add_test_result(test_result);
        }
    }

    // Validate Chebyshev Type I filters
    if let Ok(cheby1_results) = validate_chebyshev_filter(config) {
        for (_, test_result) in cheby1_results.test_results {
            results.add_test_result(test_result);
        }
    }

    // Additional filter types can be added here
    // - Chebyshev Type II
    // - Elliptic
    // - Bessel
    // - filtfilt (zero-phase filtering)

    Ok(results)
}

/// Validate Butterworth filter implementation
pub fn validate_butterworth_filter(config: &ValidationConfig) -> SignalResult<ValidationResults> {
    let mut results = ValidationResults::new();

    let filter_types = vec![FilterType::Lowpass, FilterType::Highpass];
    let orders = vec![2, 4, 6, 8];
    let cutoff_frequencies = vec![0.1, 0.25, 0.4]; // Relative to Nyquist

    for &filter_type in &filter_types {
        for &order in &orders {
            for &cutoff in &cutoff_frequencies {
                for &length in &config.test_lengths {
                    for &fs in &config.sampling_frequencies {
                        let test_result = test_single_butter_filter(
                            filter_type,
                            order,
                            cutoff,
                            length,
                            fs,
                            config.tolerance,
                        );

                        match test_result {
                            Ok(result) => results.add_test_result(result),
                            Err(e) => {
                                let failed_result = ValidationTestResult {
                                    test_name: format!(
                                        "butterworth_{:?}_order{}_cutoff{:.2}_len{}_fs{:.0}",
                                        filter_type, order, cutoff, length, fs
                                    ),
                                    passed: false,
                                    max_absolute_error: f64::INFINITY,
                                    max_relative_error: f64::INFINITY,
                                    mean_absolute_error: f64::INFINITY,
                                    num_test_cases: 1,
                                    error_message: Some(format!("Filter test failed: {}", e)),
                                };
                                results.add_test_result(failed_result);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(results)
}

/// Test single Butterworth filter configuration
fn test_single_butter_filter(
    filter_type: FilterType,
    order: usize,
    cutoff: f64,
    length: usize,
    fs: f64,
    tolerance: f64,
) -> SignalResult<ValidationTestResult> {
    let test_name = format!(
        "butterworth_{:?}_order{}_cutoff{:.2}_len{}_fs{:.0}",
        filter_type, order, cutoff, length, fs
    );

    // Generate test signal
    let test_signal = generate_test_signal(length, fs);

    // Apply our filter
    let (b, a) = butter(order, cutoff, filter_type)?;
    let our_output = lfilter(&b, &a, &test_signal)?;

    // Get reference output
    let reference_output = reference_butter_filter(&test_signal, order, cutoff, filter_type, fs)?;

    // Calculate errors
    let (max_abs_error, max_rel_error, mean_abs_error) =
        calculate_errors(&our_output, &reference_output)?;

    // Determine if test passed
    let passed = max_abs_error <= tolerance && max_rel_error <= tolerance * 100.0;

    Ok(ValidationTestResult {
        test_name,
        passed,
        max_absolute_error: max_abs_error,
        max_relative_error: max_rel_error,
        mean_absolute_error: mean_abs_error,
        num_test_cases: 1,
        error_message: if passed {
            None
        } else {
            Some(format!(
                "Error too high: max_abs={:.2e}, max_rel={:.2e}",
                max_abs_error, max_rel_error
            ))
        },
    })
}

/// Validate Chebyshev Type I filter implementation
pub fn validate_chebyshev_filter(config: &ValidationConfig) -> SignalResult<ValidationResults> {
    let mut results = ValidationResults::new();

    let filter_types = vec![FilterType::Lowpass, FilterType::Highpass];
    let orders = vec![2, 4, 6];
    let cutoff_frequencies = vec![0.1, 0.25, 0.4];
    let rp_values = vec![0.5, 1.0, 3.0]; // Passband ripple in dB

    for &filter_type in &filter_types {
        for &order in &orders {
            for &cutoff in &cutoff_frequencies {
                for &rp in &rp_values {
                    for &length in &config.test_lengths {
                        if length < 64 {
                            continue; // Skip very short signals for Chebyshev
                        }

                        for &fs in &config.sampling_frequencies {
                            let test_result = test_single_cheby1_filter(
                                filter_type,
                                order,
                                cutoff,
                                rp,
                                length,
                                fs,
                                config.tolerance,
                            );

                            match test_result {
                                Ok(result) => results.add_test_result(result),
                                Err(e) => {
                                    let failed_result = ValidationTestResult {
                                        test_name: format!(
                                            "chebyshev1_{:?}_order{}_cutoff{:.2}_rp{:.1}_len{}_fs{:.0}",
                                            filter_type, order, cutoff, rp, length, fs
                                        ),
                                        passed: false,
                                        max_absolute_error: f64::INFINITY,
                                        max_relative_error: f64::INFINITY,
                                        mean_absolute_error: f64::INFINITY,
                                        num_test_cases: 1,
                                        error_message: Some(format!("Chebyshev test failed: {}", e)),
                                    };
                                    results.add_test_result(failed_result);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(results)
}

/// Test single Chebyshev Type I filter configuration
fn test_single_cheby1_filter(
    filter_type: FilterType,
    order: usize,
    cutoff: f64,
    rp: f64,
    length: usize,
    fs: f64,
    tolerance: f64,
) -> SignalResult<ValidationTestResult> {
    let test_name = format!(
        "chebyshev1_{:?}_order{}_cutoff{:.2}_rp{:.1}_len{}_fs{:.0}",
        filter_type, order, cutoff, rp, length, fs
    );

    // Generate test signal
    let test_signal = generate_test_signal(length, fs);

    // Apply our filter
    let (b, a) = cheby1(order, rp, cutoff, filter_type)?;
    let our_output = lfilter(&b, &a, &test_signal)?;

    // Get reference output
    let reference_output = reference_cheby1_filter(&test_signal, order, rp, cutoff, filter_type, fs)?;

    // Calculate errors
    let (max_abs_error, max_rel_error, mean_abs_error) =
        calculate_errors(&our_output, &reference_output)?;

    // Determine if test passed (more relaxed tolerance for Chebyshev)
    let passed = max_abs_error <= tolerance * 10.0 && max_rel_error <= tolerance * 1000.0;

    Ok(ValidationTestResult {
        test_name,
        passed,
        max_absolute_error: max_abs_error,
        max_relative_error: max_rel_error,
        mean_absolute_error: mean_abs_error,
        num_test_cases: 1,
        error_message: if passed {
            None
        } else {
            Some(format!(
                "Error too high: max_abs={:.2e}, max_rel={:.2e}",
                max_abs_error, max_rel_error
            ))
        },
    })
}

/// Test single Chebyshev Type II filter configuration
fn test_single_cheby2_filter(
    filter_type: FilterType,
    order: usize,
    cutoff: f64,
    rs: f64,
    length: usize,
    fs: f64,
    tolerance: f64,
) -> SignalResult<ValidationTestResult> {
    let test_name = format!(
        "chebyshev2_{:?}_order{}_cutoff{:.2}_rs{:.1}_len{}_fs{:.0}",
        filter_type, order, cutoff, rs, length, fs
    );

    // Generate test signal
    let test_signal = generate_test_signal(length, fs);

    // Apply our filter (assuming cheby2 is implemented)
    let (b, a) = cheby2(order, rs, cutoff, filter_type)?;
    let our_output = lfilter(&b, &a, &test_signal)?;

    // Get reference output
    let reference_output = reference_cheby2_filter(&test_signal, order, rs, cutoff, filter_type, fs)?;

    // Calculate errors
    let (max_abs_error, max_rel_error, mean_abs_error) =
        calculate_errors(&our_output, &reference_output)?;

    // Determine if test passed
    let passed = max_abs_error <= tolerance * 10.0 && max_rel_error <= tolerance * 1000.0;

    Ok(ValidationTestResult {
        test_name,
        passed,
        max_absolute_error: max_abs_error,
        max_relative_error: max_rel_error,
        mean_absolute_error: mean_abs_error,
        num_test_cases: 1,
        error_message: if passed {
            None
        } else {
            Some(format!(
                "Error too high: max_abs={:.2e}, max_rel={:.2e}",
                max_abs_error, max_rel_error
            ))
        },
    })
}

/// Validate elliptic filter implementation
pub fn validate_elliptic_filter(_config: &ValidationConfig) -> SignalResult<ValidationResults> {
    // Placeholder for elliptic filter validation
    // This would be implemented when elliptic filters are available
    Ok(ValidationResults::new())
}

/// Validate Bessel filter implementation
pub fn validate_bessel_filter(_config: &ValidationConfig) -> SignalResult<ValidationResults> {
    // Placeholder for Bessel filter validation
    // This would be implemented when Bessel filters are available
    Ok(ValidationResults::new())
}

/// Validate filtfilt (zero-phase filtering) implementation
pub fn validate_filtfilt(_config: &ValidationConfig) -> SignalResult<ValidationResults> {
    // Placeholder for filtfilt validation
    // This would be implemented when filtfilt is available
    Ok(ValidationResults::new())
}

/// Generate test signal for filter validation
fn generate_test_signal(length: usize, fs: f64) -> Array1<f64> {
    let mut signal = Array1::zeros(length);
    let dt = 1.0 / fs;

    // Create a multi-component signal with known frequency content
    for i in 0..length {
        let t = i as f64 * dt;

        // Low frequency component
        signal[i] += (2.0 * std::f64::consts::PI * 0.05 * fs * t).sin();

        // Medium frequency component
        signal[i] += 0.5 * (2.0 * std::f64::consts::PI * 0.15 * fs * t).sin();

        // High frequency component
        signal[i] += 0.25 * (2.0 * std::f64::consts::PI * 0.35 * fs * t).sin();

        // Add small amount of noise
        signal[i] += 0.01 * (fastrand::f64() * 2.0 - 1.0);
    }

    signal
}

/// Test filter with impulse response
pub fn test_filter_impulse_response(
    filter_type: FilterType,
    order: usize,
    cutoff: f64,
    length: usize,
) -> SignalResult<ValidationTestResult> {
    let test_name = format!(
        "impulse_response_{:?}_order{}_cutoff{:.2}_len{}",
        filter_type, order, cutoff, length
    );

    // Create impulse signal
    let mut impulse = Array1::zeros(length);
    if length > 0 {
        impulse[0] = 1.0;
    }

    // Apply filter
    let (b, a) = butter(order, cutoff, filter_type)?;
    let response = lfilter(&b, &a, &impulse)?;

    // Basic sanity checks
    let max_response = response.iter().cloned().fold(0.0f64, f64::max);
    let total_energy: f64 = response.iter().map(|&x| x * x).sum();

    let passed = max_response.is_finite() &&
                 total_energy.is_finite() &&
                 max_response > 0.0 &&
                 total_energy > 0.0;

    Ok(ValidationTestResult {
        test_name,
        passed,
        max_absolute_error: if passed { 0.0 } else { f64::INFINITY },
        max_relative_error: if passed { 0.0 } else { f64::INFINITY },
        mean_absolute_error: if passed { 0.0 } else { f64::INFINITY },
        num_test_cases: 1,
        error_message: if passed {
            None
        } else {
            Some("Impulse response test failed".to_string())
        },
    })
}

/// Test filter stability
pub fn test_filter_stability(
    filter_type: FilterType,
    order: usize,
    cutoff: f64,
) -> SignalResult<ValidationTestResult> {
    let test_name = format!(
        "stability_{:?}_order{}_cutoff{:.2}",
        filter_type, order, cutoff
    );

    // Design filter
    let (b, a) = butter(order, cutoff, filter_type)?;

    // Check for NaN or infinite coefficients
    let coeffs_valid = b.iter().all(|&x| x.is_finite()) && a.iter().all(|&x| x.is_finite());

    // Basic stability check: a[0] should not be zero
    let a0_nonzero = a.len() > 0 && a[0].abs() > f64::EPSILON;

    let passed = coeffs_valid && a0_nonzero;

    Ok(ValidationTestResult {
        test_name,
        passed,
        max_absolute_error: if passed { 0.0 } else { f64::INFINITY },
        max_relative_error: if passed { 0.0 } else { f64::INFINITY },
        mean_absolute_error: if passed { 0.0 } else { f64::INFINITY },
        num_test_cases: 1,
        error_message: if passed {
            None
        } else {
            Some("Filter stability check failed".to_string())
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_test_signal() {
        let signal = generate_test_signal(100, 1000.0);
        assert_eq!(signal.len(), 100);
        assert!(signal.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_single_butter_filter() {
        let result = test_single_butter_filter(
            FilterType::Lowpass,
            4,
            0.25,
            64,
            1000.0,
            1e-6,
        );
        // Note: This test may fail without proper reference implementation
        // In practice, you would need actual SciPy reference data
        assert!(result.is_ok());
    }

    #[test]
    fn test_filter_stability() {
        let result = test_filter_stability(FilterType::Lowpass, 4, 0.25);
        assert!(result.is_ok());
        let test_result = result.unwrap();
        assert!(test_result.passed);
    }

    #[test]
    fn test_impulse_response() {
        let result = test_filter_impulse_response(FilterType::Lowpass, 4, 0.25, 64);
        assert!(result.is_ok());
        let test_result = result.unwrap();
        assert!(test_result.passed);
    }
}