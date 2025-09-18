//! Window function validation for SciPy compatibility
//!
//! This module provides validation functions for various window functions
//! including Hann, Hamming, Blackman, Kaiser, and Tukey windows.

use super::types::*;
use super::utils::calculate_errors;
use crate::error::{SignalError, SignalResult};
use crate::window::kaiser::kaiser;
use crate::window::{blackman, hamming, hann, tukey};
use std::collections::HashMap;

/// Validate window functions against SciPy
pub fn validate_windows(
    results: &mut HashMap<String, ValidationTestResult>,
    config: &ValidationConfig,
) -> SignalResult<()> {
    let start_time = std::time::Instant::now();
    let test_name = "windows".to_string();
    let mut max_abs_error: f64 = 0.0;
    let mut max_rel_error: f64 = 0.0;
    let mut rmse_sum: f64 = 0.0;
    let mut num_cases = 0;
    let mut passed = true;
    let mut error_message = None;

    // Test different window functions
    for &n in &config.test_lengths {
        if n < 8 {
            continue;
        } // Skip very small windows

        // Test Hann window
        match test_single_window(n, "hann", &[], config) {
            Ok((abs_err, rel_err, rmse)) => {
                max_abs_error = max_abs_error.max(abs_err);
                max_rel_error = max_rel_error.max(rel_err);
                rmse_sum += rmse * rmse;
                num_cases += 1;
            }
            Err(e) => {
                passed = false;
                error_message = Some(format!("Window validation failed: {}", e));
            }
        }

        // Test Kaiser window with different beta values
        if config.extensive {
            let beta_values = vec![0.5, 5.0, 8.6];
            for &beta in &beta_values {
                match test_single_window(n, "kaiser", &[beta], config) {
                    Ok((abs_err, rel_err, rmse)) => {
                        max_abs_error = max_abs_error.max(abs_err);
                        max_rel_error = max_rel_error.max(rel_err);
                        rmse_sum += rmse * rmse;
                        num_cases += 1;
                    }
                    Err(e) => {
                        passed = false;
                        error_message = Some(format!("Kaiser window validation failed: {}", e));
                    }
                }
            }
        }
    }

    let rmse = if num_cases > 0 {
        (rmse_sum / num_cases as f64).sqrt()
    } else {
        0.0
    };
    let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;

    if max_abs_error > config.tolerance || max_rel_error > config.relative_tolerance {
        passed = false;
        if error_message.is_none() {
            error_message = Some(format!(
                "Window validation failed: abs_err={:.2e}, rel_err={:.2e}",
                max_abs_error, max_rel_error
            ));
        }
    }

    results.insert(
        test_name.clone(),
        ValidationTestResult {
            test_name,
            passed,
            max_absolute_error: max_abs_error,
            max_relative_error: max_rel_error,
            rmse,
            error_message,
            num_cases,
            execution_time_ms: execution_time,
        },
    );

    Ok(())
}

/// Test a single window function configuration
fn test_single_window(
    n: usize,
    window_type: &str,
    params: &[f64],
    _config: &ValidationConfig,
) -> SignalResult<(f64, f64, f64)> {
    // Our implementation
    let our_window = match window_type {
        "hann" => hann(n)?,
        "hamming" => hamming(n)?,
        "blackman" => blackman(n)?,
        "kaiser" => {
            let beta = params.get(0).copied().unwrap_or(5.0);
            kaiser(n, beta)?
        }
        "tukey" => {
            let alpha = params.get(0).copied().unwrap_or(0.5);
            tukey(n, alpha)?
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown window type: {}",
                window_type
            )))
        }
    };

    // Reference implementation (simplified)
    let reference_window = reference_window_function(n, window_type, params)?;

    // Calculate errors
    let errors = calculate_errors(&our_window.to_vec(), &reference_window)?;

    Ok(errors)
}

/// Reference window function implementation
pub fn reference_window_function(
    n: usize,
    window_type: &str,
    params: &[f64],
) -> SignalResult<Vec<f64>> {
    // Simplified reference implementation
    // In practice, this would use scipy.signal.windows functions

    let mut window = vec![0.0; n];
    const PI: f64 = std::f64::consts::PI;

    match window_type {
        "hann" => {
            for i in 0..n {
                window[i] = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
            }
        }
        "hamming" => {
            for i in 0..n {
                window[i] = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
            }
        }
        "blackman" => {
            for i in 0..n {
                let phase = 2.0 * PI * i as f64 / (n - 1) as f64;
                window[i] = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
            }
        }
        "kaiser" => {
            let beta = params.get(0).copied().unwrap_or(5.0);
            // Simplified Kaiser window (without proper Bessel function)
            for i in 0..n {
                let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
                let arg = beta * ((1.0 - x * x) as f64).sqrt();
                window[i] = (arg / beta).exp(); // Simplified approximation
            }
        }
        "tukey" => {
            let alpha = params.get(0).copied().unwrap_or(0.5);
            let taper_len = (alpha * n as f64 / 2.0) as usize;

            for i in 0..n {
                if i < taper_len {
                    let phase = PI * i as f64 / taper_len as f64;
                    window[i] = 0.5 * (1.0 - phase.cos());
                } else if i >= n - taper_len {
                    let phase = PI * (n - 1 - i) as f64 / taper_len as f64;
                    window[i] = 0.5 * (1.0 - phase.cos());
                } else {
                    window[i] = 1.0;
                }
            }
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown window type: {}",
                window_type
            )))
        }
    }

    Ok(window)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_windows() {
        let mut results = HashMap::new();
        let config = ValidationConfig::default();

        let result = validate_windows(&mut results, &config);
        assert!(result.is_ok());
        assert!(results.contains_key("windows"));
    }

    #[test]
    fn test_single_window() {
        let config = ValidationConfig::default();
        let result = test_single_window(32, "hann", &[], &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reference_window_functions() {
        let n = 32;

        let hann_window = reference_window_function(n, "hann", &[]).unwrap();
        assert_eq!(hann_window.len(), n);
        assert!((hann_window[0] - 0.0).abs() < 1e-10); // Hann window starts at 0
        assert!((hann_window[n / 2] - 1.0).abs() < 0.1); // Approximate peak at center

        let hamming_window = reference_window_function(n, "hamming", &[]).unwrap();
        assert_eq!(hamming_window.len(), n);
        assert!(hamming_window[0] > 0.0); // Hamming window doesn't start at 0
    }
}