//! Edge Case Validation Demo
//!
//! This example demonstrates comprehensive edge case testing for special functions,
//! covering extreme values, boundary conditions, and numerical precision validation.

use scirs2_special::edge_case_tests::{
    run_comprehensive_edge_case_tests, test_array_edge_cases, test_numerical_precision,
    EdgeCaseConfig,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Special Functions Edge Case Validation Demo");
    println!("==============================================");

    // Example 1: Basic edge case testing with default configuration
    println!("\n📊 Example 1: Default Edge Case Testing");
    println!("--------------------------------------");

    let config = EdgeCaseConfig::default();
    println!("Configuration:");
    println!("  Tolerance: {:.2e}", config.tolerance);
    println!("  Extreme values: {}", config.extreme_values);
    println!("  Subnormal values: {}", config.subnormal_values);
    println!("  Boundary conditions: {}", config.boundary_conditions);
    println!("  Special values: {}", config.special_values);

    run_comprehensive_edge_case_tests(&config)?;

    // Example 2: Strict tolerance testing
    println!("\n📊 Example 2: Strict Tolerance Testing");
    println!("-------------------------------------");

    let strict_config = EdgeCaseConfig {
        tolerance: 1e-15,
        extreme_values: true,
        subnormal_values: false, // Skip subnormal for strict testing
        boundary_conditions: true,
        special_values: true,
    };

    println!("Running with stricter tolerance (1e-15)...");
    run_comprehensive_edge_case_tests(&strict_config)?;

    // Example 3: Numerical precision analysis
    test_numerical_precision()?;

    // Example 4: Array edge case testing
    test_array_edge_cases()?;

    // Example 5: Custom edge case scenarios
    println!("\n🧪 Example 5: Custom Edge Case Scenarios");
    println!("---------------------------------------");

    demonstrate_custom_edge_cases()?;

    // Example 6: Performance impact of edge cases
    println!("\n⚡ Example 6: Performance Impact Analysis");
    println!("---------------------------------------");

    analyze_edge_case_performance()?;

    // Example 7: Robustness testing
    println!("\n🛡️ Example 7: Robustness Testing");
    println!("------------------------------");

    test_robustness()?;

    println!("\n✅ All edge case validation examples completed!");
    println!("================================================");

    Ok(())
}

/// Demonstrate custom edge case scenarios
#[allow(dead_code)]
fn demonstrate_custom_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_special::{bessel, erf, erfc, gamma};

    println!("Testing challenging numerical scenarios:");

    // Scenario 1: Values near gamma function poles
    println!("\n1. Gamma function near poles:");
    let near_poles = vec![-0.000001, -0.999999, -1.000001, -1.999999];
    for value in near_poles {
        let result: f64 = gamma(value);
        println!(
            "  γ({:.6}) = {:.6e} (finite: {})",
            value,
            result,
            result.is_finite()
        );
    }

    // Scenario 2: Bessel functions with challenging arguments
    println!("\n2. Bessel functions with challenging arguments:");
    let bessel_args = vec![0.0, 1e-15, 1e15, f64::MAX.sqrt()];
    for arg in bessel_args {
        let j0_result = bessel::j0(arg);
        let j1_result = bessel::j1(arg);
        println!(
            "  J0({:.2e}) = {:.6e}, J1({:.2e}) = {:.6e}",
            arg, j0_result, arg, j1_result
        );
    }

    // Scenario 3: Error function saturation
    println!("\n3. Error function saturation:");
    let erf_args = vec![0.0, 1.0, 5.0, 10.0, 100.0, 1000.0];
    for arg in erf_args {
        let erf_result = erf(arg);
        let erfc_result = erfc(arg);
        println!(
            "  erf({:.1}) = {:.10}, erfc({:.1}) = {:.2e}",
            arg, erf_result, arg, erfc_result
        );
    }

    Ok(())
}

/// Analyze performance impact of edge cases
#[allow(dead_code)]
fn analyze_edge_case_performance() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_special::gamma;
    use std::time::Instant;

    println!("Measuring performance for different value ranges:");

    // Normal values
    let normal_values: Vec<f64> = (1..1000).map(|i| i as f64 * 0.01).collect();
    let start = Instant::now();
    let _: Vec<_> = normal_values.iter().map(|&x| gamma(x)).collect();
    let normal_time = start.elapsed();

    // Edge case values
    let edge_values = [
        1e-15, 1e-10, 1e-5, 0.1, 0.5, 1.0, 1.5, 2.0, 10.0, 50.0, 100.0, 170.0,
    ];
    let start = Instant::now();
    let _: Vec<_> = edge_values.iter().map(|&x| gamma(x)).collect();
    let edge_time = start.elapsed();

    println!("  Normal values (1000 items): {:.2?}", normal_time);
    println!("  Edge values (12 items): {:.2?}", edge_time);
    println!(
        "  Average time per normal value: {:.2?}",
        normal_time / 1000
    );
    println!("  Average time per edge value: {:.2?}", edge_time / 12);

    if edge_time / 12 > normal_time / 1000 {
        println!("  ⚠️  Edge cases are slower on average");
    } else {
        println!("  ✅ Edge cases have similar performance");
    }

    Ok(())
}

/// Test robustness against various inputs
#[allow(dead_code)]
fn test_robustness() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_special::{bessel, erf, gamma};

    println!("Testing robustness against various challenging inputs:");

    // Test special IEEE 754 values
    let special_values = vec![
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::EPSILON,
        0.0,
        -0.0,
    ];

    println!("\n1. IEEE 754 special values:");
    for value in special_values {
        let gamma_result = gamma(value);
        let j0_result = bessel::j0(value);
        let erf_result = erf(value);

        println!("  Input: {:.2e}", value);
        println!(
            "    γ: {:.2e} (finite: {})",
            gamma_result,
            gamma_result.is_finite()
        );
        println!(
            "    J0: {:.2e} (finite: {})",
            j0_result,
            j0_result.is_finite()
        );
        println!(
            "    erf: {:.2e} (finite: {})",
            erf_result,
            erf_result.is_finite()
        );
    }

    // Test monotonicity where expected
    println!("\n2. Monotonicity testing:");
    test_gamma_monotonicity()?;
    test_erf_monotonicity()?;

    // Test symmetry properties
    println!("\n3. Symmetry properties:");
    test_erf_symmetry()?;

    Ok(())
}

/// Test gamma function monotonicity in valid ranges
#[allow(dead_code)]
fn test_gamma_monotonicity() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_special::gamma;

    println!("  Testing gamma function monotonicity:");

    // Gamma should be increasing for x > 1.46...
    let test_points: Vec<f64> = (150..200).map(|i| 1.5 + i as f64 * 0.01).collect();
    let mut monotonic = true;

    for i in 1..test_points.len() {
        let prev = gamma(test_points[i - 1]);
        let curr = gamma(test_points[i]);

        if curr <= prev {
            monotonic = false;
            println!(
                "    ❌ Non-monotonic at {:.2}: {:.6e} -> {:.6e}",
                test_points[i], prev, curr
            );
            break;
        }
    }

    if monotonic {
        println!("    ✅ Gamma is monotonic increasing for x > 1.5");
    }

    Ok(())
}

/// Test error function monotonicity
#[allow(dead_code)]
fn test_erf_monotonicity() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_special::erf;

    println!("  Testing error function monotonicity:");

    let test_points: Vec<f64> = (-100..101).map(|i| i as f64 * 0.1).collect();
    let mut monotonic = true;

    for i in 1..test_points.len() {
        let prev = erf(test_points[i - 1]);
        let curr = erf(test_points[i]);

        if curr <= prev {
            monotonic = false;
            println!(
                "    ❌ Non-monotonic at {:.2}: {:.6e} -> {:.6e}",
                test_points[i], prev, curr
            );
            break;
        }
    }

    if monotonic {
        println!("    ✅ Error function is monotonic increasing");
    }

    Ok(())
}

/// Test error function symmetry: erf(-x) = -erf(x)
#[allow(dead_code)]
fn test_erf_symmetry() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_special::erf;

    println!("  Testing error function symmetry:");

    let test_values = vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
    let mut symmetric = true;
    let mut max_error: f64 = 0.0;

    for value in test_values {
        let pos_erf: f64 = erf(value);
        let neg_erf: f64 = erf(-value);
        let expected_neg: f64 = -pos_erf;
        let error: f64 = (neg_erf - expected_neg).abs();
        max_error = max_error.max(error);

        if error > 1e-14 {
            symmetric = false;
            println!("    ❌ Symmetry violation at {}: erf({}) = {:.6e}, erf(-{}) = {:.6e}, error = {:.2e}", 
                     value, value, pos_erf, value, neg_erf, error);
        }
    }

    if symmetric {
        println!(
            "    ✅ Error function symmetry preserved (max error: {:.2e})",
            max_error
        );
    }

    Ok(())
}
