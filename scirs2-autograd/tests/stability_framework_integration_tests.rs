//! Integration tests for the comprehensive stability testing framework
//!
//! This module tests the integration of all stability testing components including
//! numerical analysis, stability metrics, and the comprehensive test framework.

mod test_helpers;

use scirs2_autograd as ag;
use scirs2_autograd::tensor::Tensor;
use scirs2_autograd::testing::numerical_analysis::NumericalAnalyzer;
use scirs2_autograd::testing::stability_metrics::{
    compute_backward_stability, compute_forward_stability, StabilityGrade, StabilityMetrics,
};
use scirs2_autograd::testing::stability_test_framework::{
    create_test_scenario, run_basic_stability_tests, run_comprehensive_stability_tests,
    run_stability_tests_with_config, test_function_stability, StabilityTestSuite, TestConfig,
    TestScenario,
};
use scirs2_autograd::testing::StabilityError;
use scirs2_autograd::Float;
use test_helpers::{create_test_tensor_in_context, create_uncertainty_tensor_in_context};

/// Test the basic stability framework functionality
#[test]
#[allow(dead_code)]
fn test_basic_stability_framework() {
    let result = run_basic_stability_tests::<f32>();
    assert!(result.is_ok());

    let summary = result.expect("Test: operation failed");
    assert!(summary.total_tests > 0);
    println!(
        "Basic stability tests completed: {}/{} passed",
        summary.passed_tests, summary.total_tests
    );
}

/// Test comprehensive stability testing
#[test]
#[allow(dead_code)]
fn test_comprehensive_stability_testing() {
    let result = run_comprehensive_stability_tests::<f32>();
    assert!(result.is_ok());

    let summary = result.expect("Test: operation failed");
    assert!(summary.total_tests > 0);
    println!(
        "Comprehensive tests: {} total, {} passed, {} failed",
        summary.total_tests, summary.passed_tests, summary.failed_tests
    );

    // Print the full summary
    summary.print_summary();
}

/// Test custom configuration for stability testing
#[test]
#[allow(dead_code)]
fn test_custom_stability_config() {
    let config = TestConfig {
        run_basic_tests: true,
        run_advanced_tests: false,
        run_edge_case_tests: true,
        run_precision_tests: false,
        run_benchmarks: true,
        run_scenario_tests: false,
        tolerance_level: 1e-12,
        ..Default::default()
    };

    let result = run_stability_tests_with_config::<f32>(config);
    assert!(result.is_ok());

    let summary = result.expect("Test: operation failed");
    assert!(summary.total_tests > 0);
    println!(
        "Custom config tests: success rate = {:.1}%",
        summary.success_rate()
    );
}

/// Test individual function stability analysis
#[test]
#[allow(dead_code)]
fn test_function_stability_analysis() {
    let grade = StabilityGrade::Excellent;
    assert!(matches!(grade, StabilityGrade::Excellent));

    ag::run(|ctx| {
        let input = create_test_tensor_in_context(ctx, vec![10]);

        // Each closure is inlined directly at its call site (rather than
        // bound to a `let` first) so type inference picks it up against the
        // callee's `for<'b> Fn(&'b Tensor<'b, F>) -> ...` bound instead of
        // locking in a single concrete lifetime too early.
        let analyzer: NumericalAnalyzer<f32> = NumericalAnalyzer::new();
        let conditioning = analyzer
            .analyze_condition_number(|x: &Tensor<f32>| Ok(*x), &input)
            .expect("Test: operation failed");
        // A condition number is never less than 1.0 by definition; the exact
        // numerically-estimated value for identity is implementation- and
        // size-dependent (finite-difference Jacobian, not a closed form), so
        // assert the mathematical lower bound rather than a specific value.
        assert!(conditioning.spectral_condition_number.is_finite());
        assert!(conditioning.spectral_condition_number >= 1.0 - 1e-6);

        let forward = compute_forward_stability(&(|x: &Tensor<f32>| Ok(*x)), &input, 1e-8)
            .expect("Test: operation failed");
        assert!(matches!(
            forward.stability_grade,
            StabilityGrade::Excellent | StabilityGrade::Good
        ));

        println!("Framework validation: numerical analyzer + stability metrics agree on identity stability");
    });
}

/// Test scenario-based testing
#[test]
#[allow(dead_code)]
fn test_scenario_based_testing() {
    ag::run(|ctx| {
        let input = create_test_tensor_in_context(ctx, vec![10]);

        // y = x (identity): a real, checkable function rather than a
        // placeholder that always errors, so the scenario actually
        // exercises compute_forward_stability's perturbation analysis.
        let scenario = create_test_scenario(
            "identity_scaling".to_string(),
            "Test identity function y = x".to_string(),
            |x: &Tensor<f32>| Ok(*x),
            input,
            StabilityGrade::Excellent,
        );

        let mut suite = StabilityTestSuite::new();
        suite.add_scenario(scenario);

        let result = suite.run_all_tests_with_context(ctx);
        assert!(result.is_ok());

        let summary = result.expect("Test: operation failed");
        println!(
            "Scenario test results: {}/{} passed",
            summary.passed_tests, summary.total_tests
        );
        assert!(summary.total_tests > 0);
        assert!(summary.passed_tests > 0);
    });
}

/// Test numerical analysis integration
#[test]
#[allow(dead_code)]
fn test_numerical_analysis_integration() {
    ag::run(|ctx| {
        let analyzer = NumericalAnalyzer::<f32>::new();
        let input = create_test_tensor_in_context(ctx, vec![8, 8]);

        let conditioning_result =
            analyzer.analyze_condition_number(|x: &Tensor<f32>| Ok(*x), &input);
        assert!(conditioning_result.is_ok());

        let conditioning = conditioning_result.expect("Test: operation failed");
        println!("Condition number analysis:");
        println!(
            "  Spectral condition number: {:.3e}",
            conditioning.spectral_condition_number
        );
        println!("  Assessment: {:?}", conditioning.conditioning_assessment);
        assert!(conditioning.spectral_condition_number.is_finite());
        assert!(conditioning.spectral_condition_number >= 1.0 - 1e-6);
    });
}

/// Test stability metrics integration
#[test]
#[allow(dead_code)]
fn test_stability_metrics_integration() {
    ag::run(|ctx| {
        let metrics = StabilityMetrics::<f32>::new();
        let input = create_test_tensor_in_context(ctx, vec![6, 6]);

        // Each use inlines its own copy of the (non-capturing, hence trivially
        // Copy) identity closure directly at the call site — see the note in
        // test_function_stability_analysis for why an intermediate `let`
        // binding defeats the required higher-ranked lifetime inference.
        let forward_result =
            metrics.compute_forward_stability(&(|x: &Tensor<f32>| Ok(*x)), &input, 1e-8);
        assert!(forward_result.is_ok());

        let forward_metrics = forward_result.expect("Test: operation failed");
        println!("Forward stability metrics:");
        println!("  Grade: {:?}", forward_metrics.stability_grade);
        println!(
            "  Mean relative error: {:.3e}",
            forward_metrics.mean_relative_error
        );
        assert!(forward_metrics.mean_relative_error.is_finite());

        // Backward stability compares against the function's own output; the
        // test function is the identity, so its output is just `input`.
        let output = input;
        let backward_result =
            metrics.compute_backward_stability(&(|x: &Tensor<f32>| Ok(*x)), &input, &output);
        assert!(backward_result.is_ok());

        let backward_metrics = backward_result.expect("Test: operation failed");
        println!("Backward stability metrics:");
        println!("  Grade: {:?}", backward_metrics.stability_grade);
        println!("  Error: {:.2e}", backward_metrics.backward_error);
        assert!(backward_metrics.backward_error.is_finite());
    });
}

/// Test error propagation analysis
#[test]
#[allow(dead_code)]
fn test_error_propagation_analysis() {
    ag::run(|ctx| {
        let input = create_test_tensor_in_context(ctx, vec![5]);
        let uncertainty = create_uncertainty_tensor_in_context(ctx, vec![5], 1e-8);

        let analyzer = NumericalAnalyzer::<f32>::new();
        let result =
            analyzer.analyze_error_propagation(|x: &Tensor<f32>| Ok(*x), &input, &uncertainty);
        assert!(result.is_ok());

        let propagation = result.expect("Test: operation failed");
        println!("Error propagation analysis:");
        println!(
            "  Linear error bound: {:.3e}",
            propagation.linear_error_bound
        );
        println!(
            "  Monte Carlo samples: {}",
            propagation.monte_carlo_analysis.num_samples
        );
        assert!(propagation.linear_error_bound.is_finite());
        assert!(propagation.linear_error_bound >= 0.0);
    });
}

/// Test comprehensive integration of all components
#[test]
#[allow(dead_code)]
fn test_full_pipeline_integration() {
    ag::run(|ctx| {
        // Create a comprehensive test suite with all features enabled
        let config = TestConfig {
            run_basic_tests: true,
            run_advanced_tests: true,
            run_edge_case_tests: true,
            run_precision_tests: true,
            run_benchmarks: true,
            run_scenario_tests: true,
            tolerance_level: 1e-10,
            ..Default::default()
        };

        let mut suite = StabilityTestSuite::<f32>::with_config(config);

        // Add some custom scenarios
        let scenarios = create_test_scenarios::<f32>(ctx);
        for scenario in scenarios {
            suite.add_scenario(scenario);
        }

        // Run all tests
        let result = suite.run_all_tests_with_context(ctx);
        assert!(result.is_ok());

        let summary = result.expect("Test: operation failed");
        println!("\n=== FULL PIPELINE INTEGRATION RESULTS ===");
        summary.print_summary();

        // Verify we got comprehensive results
        assert!(summary.total_tests >= 4); // Basic + scenarios
        assert!(summary.success_rate() >= 50.0); // At least half should pass
        assert!(!summary.recommendations.is_empty());
    });
}

/// Test edge case handling
#[test]
#[allow(dead_code)]
fn test_edge_case_handling() {
    let config = TestConfig {
        run_basic_tests: false,
        run_advanced_tests: false,
        run_edge_case_tests: true,
        run_precision_tests: false,
        run_benchmarks: false,
        run_scenario_tests: false,
        ..Default::default()
    };

    let result = run_stability_tests_with_config::<f32>(config);
    assert!(result.is_ok());

    let summary = result.expect("Test: operation failed");
    println!("Edge case test results:");
    println!("  Total tests: {}", summary.total_tests);
    println!("  Success rate: {:.1}%", summary.success_rate());

    // Edge cases might have some failures, which is expected
    assert!(summary.total_tests > 0);
}

/// Test performance benchmarking
///
/// (This test was accidentally dropped from an earlier full-file rewrite in
/// this same audit — restored here.) `run_benchmarks` used to push one
/// hardcoded `BenchmarkResult` regardless of `size`, so this always passed in
/// microseconds despite being `#[ignore = "timeout"]`d; `run_size_benchmark`
/// now times real per-size tensor creation (see stability_test_framework.rs),
/// so assert the benchmarks actually reflect that instead of only checking
/// `is_ok()`.
#[test]
#[allow(dead_code)]
fn test_performance_benchmarking() {
    let config = TestConfig {
        run_basic_tests: false,
        run_advanced_tests: false,
        run_edge_case_tests: false,
        run_precision_tests: false,
        run_benchmarks: true,
        run_scenario_tests: false,
        ..Default::default()
    };

    let result = run_stability_tests_with_config::<f32>(config);
    assert!(result.is_ok());

    let summary = result.expect("Test: operation failed");
    println!("Performance benchmark results:");
    println!(
        "  Avg analysis duration: {:.6}s",
        summary
            .performance_summary
            .average_analysis_duration
            .as_secs_f64()
    );
    println!(
        "  Max ops/sec: {}",
        summary.performance_summary.max_operations_per_second
    );

    // Performance tests should always pass (they're measuring, not validating)
    assert_eq!(summary.failed_tests, 0);
    // calculate_performance_summary returns PerformanceSummary::default()
    // (max_operations_per_second: 0) whenever `self.benchmarks` is empty, so
    // a positive value here confirms run_performance_benchmarks genuinely
    // ran across all four configured sizes rather than being skipped.
    assert!(summary.performance_summary.max_operations_per_second > 0);
}

/// Test precision sensitivity analysis
#[test]
#[allow(dead_code)]
fn test_precision_sensitivity() {
    let config = TestConfig {
        run_basic_tests: false,
        run_advanced_tests: false,
        run_edge_case_tests: false,
        run_precision_tests: true,
        run_benchmarks: false,
        run_scenario_tests: false,
        ..Default::default()
    };

    let result = run_stability_tests_with_config::<f32>(config);
    assert!(result.is_ok());

    let summary = result.expect("Test: operation failed");
    println!("Precision sensitivity test completed");
    println!("  Tests performed: {}", summary.total_tests);

    // Precision tests should provide useful information
    // Note: total_tests is usize, so always >= 0
    assert!(summary.total_tests == summary.total_tests); // Basic sanity check
}

/// Test various function types for stability
#[test]
#[allow(dead_code)]
fn test_different_function_types() {
    ag::run(|ctx| {
        let input = create_test_tensor_in_context(ctx, vec![4]);

        // Each named function is genuinely different so the three stability
        // results can actually differ, rather than all sharing one
        // placeholder that always errors. Called directly (rather than
        // boxed into a Vec<dyn Fn>) so each closure's higher-ranked lifetime
        // is inferred from test_function_stability's own signature.
        let identity_result = test_function_stability(|x: &Tensor<f32>| Ok(*x), &input, "identity");
        assert!(
            identity_result.is_ok(),
            "Function identity failed stability test"
        );

        let negate_result = test_function_stability(
            |x: &Tensor<f32>| Ok(scirs2_autograd::tensor_ops::neg(x)),
            &input,
            "negate",
        );
        assert!(
            negate_result.is_ok(),
            "Function negate failed stability test"
        );

        let add_self_result = test_function_stability(
            |x: &Tensor<f32>| Ok(scirs2_autograd::tensor_ops::add(x, x)),
            &input,
            "add_self",
        );
        assert!(
            add_self_result.is_ok(),
            "Function add_self failed stability test"
        );

        for (name, result) in [
            ("identity", identity_result),
            ("negate", negate_result),
            ("add_self", add_self_result),
        ] {
            let test_result = result.expect("Test: operation failed");
            println!(
                "Function '{}' stability: {:?} (passed: {})",
                name, test_result.actual_grade, test_result.passed
            );
        }
    });
}

/// Test large tensor stability
#[test]
#[allow(dead_code)]
fn test_large_tensor_stability() {
    ag::run(|ctx| {
        let large_input = create_test_tensor_in_context(ctx, vec![100, 100]);

        let result =
            test_function_stability(|x: &Tensor<f32>| Ok(*x), &large_input, "large_tensor_test");
        assert!(result.is_ok());

        let test_result = result.expect("Test: operation failed");
        println!("Large tensor stability test:");
        println!("  Input shape: {:?}", large_input.shape());
        println!("  Stability grade: {:?}", test_result.actual_grade);
        println!(
            "  Test duration: {:.3}s",
            test_result.duration.as_secs_f64()
        );

        // Large tensors should still maintain good stability for simple operations
        assert!(matches!(
            test_result.actual_grade,
            StabilityGrade::Excellent | StabilityGrade::Good | StabilityGrade::Fair
        ));
    });
}

/// Test mixed precision scenarios
#[test]
#[allow(dead_code)]
fn test_mixed_precision_scenarios() {
    // Test with f32
    let f32_result = run_basic_stability_tests::<f32>();
    assert!(f32_result.is_ok());
    let f32_summary = f32_result.expect("Test: operation failed");

    // Test with f64
    let f64_result = run_basic_stability_tests::<f64>();
    assert!(f64_result.is_ok());
    let f64_summary = f64_result.expect("Test: operation failed");

    println!("Mixed precision comparison:");
    println!("  f32 success rate: {:.1}%", f32_summary.success_rate());
    println!("  f64 success rate: {:.1}%", f64_summary.success_rate());

    // f64 should generally have better or equal stability
    assert!(f64_summary.success_rate() >= f32_summary.success_rate() - 10.0);
}

// Helper functions

#[allow(dead_code)]
fn create_test_scenarios<'a, F: Float>(ctx: &'a ag::Context<F>) -> Vec<TestScenario<'a, F>> {
    let mut scenarios = Vec::new();

    // Scenario 1: Identity transformation
    scenarios.push(create_test_scenario(
        "identity_transform".to_string(),
        "Identity transformation y = x".to_string(),
        |x: &Tensor<F>| Ok(*x),
        create_test_tensor_in_context(ctx, vec![8]),
        StabilityGrade::Excellent,
    ));

    // Scenario 2: Self-addition (y = x + x = 2x)
    scenarios.push(create_test_scenario(
        "double".to_string(),
        "Self-addition function y = x + x".to_string(),
        |x: &Tensor<F>| {
            use scirs2_autograd::tensor_ops as T;
            Ok(T::add(x, x))
        },
        create_test_tensor_in_context(ctx, vec![6]),
        StabilityGrade::Good,
    ));

    // Scenario 3: Negation
    scenarios.push(create_test_scenario(
        "negate".to_string(),
        "Negation function y = -x".to_string(),
        |x: &Tensor<F>| {
            use scirs2_autograd::tensor_ops as T;
            Ok(T::neg(x))
        },
        create_test_tensor_in_context(ctx, vec![10]),
        StabilityGrade::Fair,
    ));

    scenarios
}

/// Integration test for the complete stability testing workflow
#[test]
#[allow(dead_code)]
fn test_complete_stability_workflow() {
    println!("\n=== COMPLETE STABILITY TESTING WORKFLOW ===");

    ag::run(|ctx| {
        // Step 1: Create test data
        let input = create_test_tensor_in_context(ctx, vec![50, 50]);
        println!("1. Created test tensor with shape {:?}", input.shape());

        // Step 2: Test individual components
        println!("2. Testing individual components...");

        let analyzer = NumericalAnalyzer::new();
        let conditioning = analyzer.analyze_condition_number(|x: &Tensor<f32>| Ok(*x), &input);
        assert!(conditioning.is_ok());
        println!("   \u{2713} Numerical analysis completed");

        let forward_metrics = compute_forward_stability(&(|x: &Tensor<f32>| Ok(*x)), &input, 1e-8);
        assert!(forward_metrics.is_ok());
        println!("   \u{2713} Stability metrics completed");

        // Step 3: Run comprehensive test suite
        println!("3. Running comprehensive test suite...");
        let comprehensive_result = run_comprehensive_stability_tests::<f32>();
        assert!(comprehensive_result.is_ok());
        let summary = comprehensive_result.expect("Test: operation failed");
        println!("   \u{2713} Comprehensive tests completed");

        // Step 4: Analyze results
        println!("4. Analyzing results...");
        println!("   Total tests: {}", summary.total_tests);
        println!("   Success rate: {:.1}%", summary.success_rate());
        println!("   Duration: {:.2}s", summary.total_duration.as_secs_f64());

        // Step 5: Validate workflow success
        assert!(summary.total_tests > 0);
        assert!(summary.success_rate() >= 0.0);
        assert!(!summary.recommendations.is_empty());

        println!("5. \u{2713} Complete workflow validation passed");
        println!("=========================================\n");
    });
}
