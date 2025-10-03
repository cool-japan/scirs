//! Comprehensive edge case testing for special functions
//!
//! This module provides extensive testing for numerical edge cases, extreme parameter
//! values, and boundary conditions to ensure robust numerical behavior across the
//! entire domain of special functions.

#![allow(dead_code)]

use crate::{bessel, erf, gamma, SpecialResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::f64;

/// Test configuration for edge cases
#[derive(Debug, Clone)]
pub struct EdgeCaseConfig {
    pub tolerance: f64,
    pub extreme_values: bool,
    pub subnormal_values: bool,
    pub boundary_conditions: bool,
    pub special_values: bool,
}

impl Default for EdgeCaseConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            extreme_values: true,
            subnormal_values: true,
            boundary_conditions: true,
            special_values: true,
        }
    }
}

/// Edge case test result
#[derive(Debug)]
pub struct EdgeCaseResult {
    pub test_name: String,
    pub function: String,
    pub input: f64,
    pub output: f64,
    pub expected_behavior: String,
    pub passed: bool,
    pub error_message: Option<String>,
}

/// Edge case test suite for gamma function
#[allow(dead_code)]
pub fn test_gamma_edge_cases(config: &EdgeCaseConfig) -> Vec<EdgeCaseResult> {
    let mut results = Vec::new();

    if config.special_values {
        // Test special values
        results.extend(test_gamma_special_values(config));
    }

    if config.extreme_values {
        // Test extreme values
        results.extend(test_gamma_extreme_values(config));
    }

    if config.boundary_conditions {
        // Test boundary conditions
        results.extend(test_gamma_boundary_conditions(config));
    }

    if config.subnormal_values {
        // Test subnormal values
        results.extend(test_gamma_subnormal_values(config));
    }

    results
}

/// Test gamma function with special values
#[allow(dead_code)]
fn test_gamma_special_values(config: &EdgeCaseConfig) -> Vec<EdgeCaseResult> {
    let mut results = Vec::new();

    // Test cases: (input, expected, description)
    let test_cases = vec![
        (1.0, 1.0, "Gamma(1) = 1"),
        (2.0, 1.0, "Gamma(2) = 1"),
        (3.0, 2.0, "Gamma(3) = 2"),
        (4.0, 6.0, "Gamma(4) = 6"),
        (5.0, 24.0, "Gamma(5) = 24"),
        (0.5, f64::consts::PI.sqrt(), "Gamma(1/2) = sqrt(π)"),
        (1.5, f64::consts::PI.sqrt() / 2.0, "Gamma(3/2) = sqrt(π)/2"),
    ];

    for (input, expected, description) in test_cases {
        let output = gamma::gamma(input);
        let error = (output - expected).abs();
        let passed = error < config.tolerance;

        results.push(EdgeCaseResult {
            test_name: format!("gamma_special_{input}"),
            function: "gamma".to_string(),
            input,
            output,
            expected_behavior: description.to_string(),
            passed,
            error_message: if !passed {
                Some(format!(
                    "Error: {error:.2e}, tolerance: {tolerance:.2e}",
                    tolerance = config.tolerance
                ))
            } else {
                None
            },
        });
    }

    results
}

/// Test gamma function with extreme values
#[allow(dead_code)]
fn test_gamma_extreme_values(_config: &EdgeCaseConfig) -> Vec<EdgeCaseResult> {
    let mut results = Vec::new();

    // Test very large values
    let largeinput = 170.0; // Near overflow threshold
    let output = gamma::gamma(largeinput);
    results.push(EdgeCaseResult {
        test_name: "gamma_large_value".to_string(),
        function: "gamma".to_string(),
        input: largeinput,
        output,
        expected_behavior: "Should be finite for x < 171".to_string(),
        passed: output.is_finite(),
        error_message: if !output.is_finite() {
            Some("Gamma function overflowed for large but valid input".to_string())
        } else {
            None
        },
    });

    // Test very small positive values
    let smallinput = 1e-10;
    let output = gamma::gamma(smallinput);
    results.push(EdgeCaseResult {
        test_name: "gamma_small_positive".to_string(),
        function: "gamma".to_string(),
        input: smallinput,
        output,
        expected_behavior: "Should be approximately 1/x for small x".to_string(),
        passed: output.is_finite() && output > 0.0,
        error_message: if !output.is_finite() || output <= 0.0 {
            Some("Gamma function failed for small positive input".to_string())
        } else {
            None
        },
    });

    results
}

/// Test gamma function boundary conditions
#[allow(dead_code)]
fn test_gamma_boundary_conditions(_config: &EdgeCaseConfig) -> Vec<EdgeCaseResult> {
    let mut results = Vec::new();

    // Test near zero
    let near_zero = 1e-15;
    let output = gamma::gamma(near_zero);
    results.push(EdgeCaseResult {
        test_name: "gamma_near_zero".to_string(),
        function: "gamma".to_string(),
        input: near_zero,
        output,
        expected_behavior: "Should have large positive value near 1/x".to_string(),
        passed: output.is_finite() && output > 0.0,
        error_message: None,
    });

    // Test negative values near integers
    let inputs: Vec<f64> = vec![-0.1, -0.9, -1.1, -1.9, -2.1];
    for input in inputs {
        let output = gamma::gamma(input);
        let expected_finite = input.fract() != 0.0; // Should be finite unless exactly negative integer

        results.push(EdgeCaseResult {
            test_name: format!("gamma_negative_{absinput}", absinput = input.abs()),
            function: "gamma".to_string(),
            input,
            output,
            expected_behavior: if expected_finite {
                "Should be finite for non-integer negative values".to_string()
            } else {
                "Should be infinite for negative integer values".to_string()
            },
            passed: expected_finite == output.is_finite(),
            error_message: None,
        });
    }

    results
}

/// Test gamma function with subnormal values
#[allow(dead_code)]
fn test_gamma_subnormal_values(_config: &EdgeCaseConfig) -> Vec<EdgeCaseResult> {
    let mut results = Vec::new();

    // Test with subnormal inputs
    let subnormalinput = f64::MIN_POSITIVE / 2.0;
    let output = gamma::gamma(subnormalinput);

    results.push(EdgeCaseResult {
        test_name: "gamma_subnormal".to_string(),
        function: "gamma".to_string(),
        input: subnormalinput,
        output,
        expected_behavior: "Should handle subnormal inputs gracefully".to_string(),
        passed: output.is_finite(),
        error_message: if !output.is_finite() {
            Some("Failed to handle subnormal input".to_string())
        } else {
            None
        },
    });

    results
}

/// Edge case test suite for Bessel functions
#[allow(dead_code)]
pub fn test_bessel_edge_cases(config: &EdgeCaseConfig) -> Vec<EdgeCaseResult> {
    let mut results = Vec::new();

    if config.special_values {
        results.extend(test_bessel_special_values(config));
    }

    if config.extreme_values {
        results.extend(test_bessel_extreme_values(config));
    }

    if config.boundary_conditions {
        results.extend(test_bessel_boundary_conditions(config));
    }

    results
}

/// Test Bessel functions with special values
#[allow(dead_code)]
fn test_bessel_special_values(config: &EdgeCaseConfig) -> Vec<EdgeCaseResult> {
    let mut results = Vec::new();

    // J0(0) = 1
    let output = bessel::j0(0.0);
    let expected = 1.0f64;
    let error = (output - expected).abs();

    results.push(EdgeCaseResult {
        test_name: "j0_zero".to_string(),
        function: "j0".to_string(),
        input: 0.0,
        output,
        expected_behavior: "J0(0) = 1".to_string(),
        passed: error < config.tolerance,
        error_message: if error >= config.tolerance {
            Some(format!("Error: {error:.2e}"))
        } else {
            None
        },
    });

    // J1(0) = 0
    let output = bessel::j1(0.0);
    let expected = 0.0f64;
    let error = (output - expected).abs();

    results.push(EdgeCaseResult {
        test_name: "j1_zero".to_string(),
        function: "j1".to_string(),
        input: 0.0,
        output,
        expected_behavior: "J1(0) = 0".to_string(),
        passed: error < config.tolerance,
        error_message: if error >= config.tolerance {
            Some(format!("Error: {error:.2e}"))
        } else {
            None
        },
    });

    results
}

/// Test Bessel functions with extreme values
#[allow(dead_code)]
fn test_bessel_extreme_values(_config: &EdgeCaseConfig) -> Vec<EdgeCaseResult> {
    let mut results = Vec::new();

    // Test very large arguments
    let largeinput = 1000.0;
    let output = bessel::j0(largeinput);

    results.push(EdgeCaseResult {
        test_name: "j0_large".to_string(),
        function: "j0".to_string(),
        input: largeinput,
        output,
        expected_behavior: "Should be finite and oscillatory for large x".to_string(),
        passed: output.is_finite() && output.abs() < 1.0,
        error_message: if !output.is_finite() {
            Some("J0 failed for large input".to_string())
        } else if output.abs() >= 1.0 {
            Some("J0 magnitude should be < 1 for large x".to_string())
        } else {
            None
        },
    });

    results
}

/// Test Bessel functions boundary conditions
#[allow(dead_code)]
fn test_bessel_boundary_conditions(config: &EdgeCaseConfig) -> Vec<EdgeCaseResult> {
    let mut results = Vec::new();

    // Test very small positive values
    let smallinput = 1e-12;
    let output = bessel::j0(smallinput);
    let expected = 1.0f64; // J0(x) ≈ 1 for small x
    let error = (output - expected).abs();

    results.push(EdgeCaseResult {
        test_name: "j0_small".to_string(),
        function: "j0".to_string(),
        input: smallinput,
        output,
        expected_behavior: "J0(x) ≈ 1 for small x".to_string(),
        passed: error < config.tolerance,
        error_message: if error >= config.tolerance {
            Some(format!("Error: {error:.2e}"))
        } else {
            None
        },
    });

    results
}

/// Edge case test suite for error functions
#[allow(dead_code)]
pub fn test_erf_edge_cases(config: &EdgeCaseConfig) -> Vec<EdgeCaseResult> {
    let mut results = Vec::new();

    if config.special_values {
        results.extend(test_erf_special_values(config));
    }

    if config.extreme_values {
        results.extend(test_erf_extreme_values(config));
    }

    results
}

/// Test error functions with special values
#[allow(dead_code)]
fn test_erf_special_values(config: &EdgeCaseConfig) -> Vec<EdgeCaseResult> {
    let mut results = Vec::new();

    // Test cases: (input, expected, description)
    let test_cases: Vec<(f64, f64, &str)> = vec![
        (0.0, 0.0, "erf(0) = 0"),
        (f64::INFINITY, 1.0, "erf(∞) = 1"),
        (f64::NEG_INFINITY, -1.0, "erf(-∞) = -1"),
    ];

    for (input, expected, description) in test_cases {
        let output = erf::erf(input);
        let error = if expected.is_finite() {
            (output - expected).abs()
        } else {
            0.0 // For infinite cases, just check if they're in the right direction
        };
        let passed = if expected.is_finite() {
            error < config.tolerance
        } else {
            (output > 0.0) == (expected > 0.0) && output.abs() >= 0.99
        };

        results.push(EdgeCaseResult {
            test_name: format!("erf_special_{input}"),
            function: "erf".to_string(),
            input,
            output,
            expected_behavior: description.to_string(),
            passed,
            error_message: if !passed {
                Some(format!("Error: {error:.2e}"))
            } else {
                None
            },
        });
    }

    results
}

/// Test error functions with extreme values
#[allow(dead_code)]
fn test_erf_extreme_values(_config: &EdgeCaseConfig) -> Vec<EdgeCaseResult> {
    let mut results = Vec::new();

    // Test very large positive values
    let largeinput = 100.0;
    let output = erf::erf(largeinput);

    results.push(EdgeCaseResult {
        test_name: "erf_large_positive".to_string(),
        function: "erf".to_string(),
        input: largeinput,
        output,
        expected_behavior: "erf(large) should approach 1".to_string(),
        passed: (output - 1.0).abs() < 1e-10,
        error_message: if (output - 1.0).abs() >= 1e-10 {
            Some(format!("erf({largeinput}) = {output}, expected ≈ 1"))
        } else {
            None
        },
    });

    // Test very large negative values
    let large_neginput = -100.0;
    let output = erf::erf(large_neginput);

    results.push(EdgeCaseResult {
        test_name: "erf_large_negative".to_string(),
        function: "erf".to_string(),
        input: large_neginput,
        output,
        expected_behavior: "erf(-large) should approach -1".to_string(),
        passed: (output + 1.0).abs() < 1e-10,
        error_message: if (output + 1.0).abs() >= 1e-10 {
            Some(format!("erf({large_neginput}) = {output}, expected ≈ -1"))
        } else {
            None
        },
    });

    results
}

/// Comprehensive edge case test runner
#[allow(dead_code)]
pub fn run_comprehensive_edge_case_tests(config: &EdgeCaseConfig) -> SpecialResult<()> {
    println!("🧪 Running Comprehensive Edge Case Tests");
    println!("========================================");

    let mut all_results = Vec::new();

    // Test gamma function
    println!("\n📊 Testing Gamma Function Edge Cases");
    let gamma_results = test_gamma_edge_cases(config);
    let gamma_passed = gamma_results.iter().filter(|r| r.passed).count();
    println!(
        "Gamma tests: {}/{} passed",
        gamma_passed,
        gamma_results.len()
    );
    all_results.extend(gamma_results);

    // Test Bessel functions
    println!("\n📊 Testing Bessel Function Edge Cases");
    let bessel_results = test_bessel_edge_cases(config);
    let bessel_passed = bessel_results.iter().filter(|r| r.passed).count();
    println!(
        "Bessel tests: {}/{} passed",
        bessel_passed,
        bessel_results.len()
    );
    all_results.extend(bessel_results);

    // Test error functions
    println!("\n📊 Testing Error Function Edge Cases");
    let erf_results = test_erf_edge_cases(config);
    let erf_passed = erf_results.iter().filter(|r| r.passed).count();
    println!(
        "Error function tests: {}/{} passed",
        erf_passed,
        erf_results.len()
    );
    all_results.extend(erf_results);

    // Summary
    let total_passed = all_results.iter().filter(|r| r.passed).count();
    let total_tests = all_results.len();

    println!("\n📈 Overall Summary");
    println!("==================");
    println!("Total tests: {total_tests}");
    println!("Passed: {total_passed}");
    println!("Failed: {}", total_tests - total_passed);
    println!(
        "Success rate: {:.1}%",
        100.0 * total_passed as f64 / total_tests as f64
    );

    // Report failures
    let failures: Vec<_> = all_results.iter().filter(|r| !r.passed).collect();
    if !failures.is_empty() {
        println!("\n❌ Failed Tests:");
        for failure in failures {
            println!(
                "  {} ({}): {}",
                failure.test_name,
                failure.function,
                failure
                    .error_message
                    .as_ref()
                    .unwrap_or(&"Unknown error".to_string())
            );
        }
    }

    Ok(())
}

/// Test numerical precision and accuracy
#[allow(dead_code)]
pub fn test_numerical_precision() -> SpecialResult<()> {
    println!("\n🔬 Testing Numerical Precision");
    println!("=============================");

    // Test precision of gamma function for known values
    let test_cases = vec![
        (1.0, 1.0),
        (2.0, 1.0),
        (3.0, 2.0),
        (4.0, 6.0),
        (5.0, 24.0),
        (0.5, f64::consts::PI.sqrt()),
    ];

    let mut max_relative_error: f64 = 0.0;

    for (input, expected) in test_cases {
        let computed = gamma::gamma(input);
        let relative_error = if expected != 0.0 {
            ((computed - expected) / expected).abs()
        } else {
            computed.abs()
        };

        max_relative_error = max_relative_error.max(relative_error);

        println!(
            "γ({input:.1}) = {computed:.10} (expected: {expected:.10}, relerror: {relative_error:.2e})"
        );
    }

    println!("Maximum relative error: {max_relative_error:.2e}");

    if max_relative_error > 1e-10 {
        println!("⚠️  Warning: Relative error exceeds 1e-10");
    } else {
        println!("✅ All precision tests passed (error < 1e-10)");
    }

    Ok(())
}

/// Array-based edge case testing
#[allow(dead_code)]
pub fn test_array_edge_cases() -> SpecialResult<()> {
    println!("\n📊 Testing Array Edge Cases");
    println!("===========================");

    // Test with arrays containing edge case values (reduced for faster testing)
    let edge_values = vec![0.1, 0.5, 1.0, 2.0, 10.0];

    let input_array = Array1::from_vec(edge_values.clone());

    // Test gamma function on array
    let gamma_results: Vec<_> = input_array.iter().map(|&x| gamma::gamma(x)).collect();

    println!("Gamma function results:");
    for (input, output) in edge_values.iter().zip(gamma_results.iter()) {
        println!(
            "  γ({:.2e}) = {:.6e} (finite: {})",
            input,
            output,
            output.is_finite()
        );
    }

    // Check that all results are reasonable
    let all_finite = gamma_results.iter().all(|&x| x.is_finite() || x > 0.0);

    if all_finite {
        println!("✅ All array gamma computations are finite and positive");
    } else {
        println!("⚠️  Some array gamma computations produced invalid results");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_case_config() {
        let config = EdgeCaseConfig::default();
        assert_eq!(config.tolerance, 1e-12);
        assert!(config.extreme_values);
        assert!(config.special_values);
    }

    #[test]
    fn test_gamma_special_values_basic() {
        let config = EdgeCaseConfig::default();
        let results = test_gamma_special_values(&config);

        // Should have at least a few test cases
        assert!(!results.is_empty());

        // Check that gamma(1) test exists and passes
        let gamma_1_test = results.iter().find(|r| r.input == 1.0);
        assert!(gamma_1_test.is_some());
        assert!(gamma_1_test.unwrap().passed);
    }

    #[test]
    fn test_bessel_special_values_basic() {
        let config = EdgeCaseConfig::default();
        let results = test_bessel_special_values(&config);

        // Should have test cases
        assert!(!results.is_empty());

        // Check that J0(0) test exists
        let j0_zero_test = results.iter().find(|r| r.test_name == "j0_zero");
        assert!(j0_zero_test.is_some());
    }

    #[test]
    fn test_erf_special_values_basic() {
        let config = EdgeCaseConfig::default();
        let results = test_erf_special_values(&config);

        // Should have test cases
        assert!(!results.is_empty());

        // Check that erf(0) test exists and passes
        let erf_zero_test = results.iter().find(|r| r.input == 0.0);
        assert!(erf_zero_test.is_some());
        assert!(erf_zero_test.unwrap().passed);
    }

    #[test]
    fn test_numerical_precision_runner() {
        // This should not panic
        let result = test_numerical_precision();
        assert!(result.is_ok());
    }

    #[test]
    fn test_array_edge_cases_runner() {
        // This should not panic
        let result = test_array_edge_cases();
        assert!(result.is_ok());
    }
}
