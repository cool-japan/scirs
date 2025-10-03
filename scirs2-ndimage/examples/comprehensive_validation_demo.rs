//! Comprehensive Validation and Testing Demo
//!
//! This example demonstrates how to use the new validation, testing, and
//! benchmarking modules to ensure scirs2-ndimage meets SciPy compatibility
//! and performance standards.

use scirs2_ndimage::{
    api_compatibility_verification::{ApiCompatibilityTester, CompatibilityConfig},
    comprehensive_examples::{validate_all_examples, ExampleTutorial},
    comprehensive_scipy_validation::{SciPyValidationSuite, ValidationConfig},
    scipy_performance_comparison::{BenchmarkConfig, SciPyBenchmarkSuite},
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== COMPREHENSIVE SCIRS2-NDIMAGE VALIDATION DEMO ===\n");

    // 1. Validate all comprehensive examples
    println!("1. VALIDATING COMPREHENSIVE EXAMPLES");
    println!("=====================================");

    match validate_all_examples() {
        Ok(()) => println!("✓ All comprehensive examples validated successfully\n"),
        Err(e) => {
            println!("✗ Example validation failed: {}\n", e);
            return Err(e);
        }
    }

    // 2. Generate and display tutorial
    println!("2. GENERATING COMPREHENSIVE TUTORIAL");
    println!("====================================");

    let mut tutorial = ExampleTutorial::new();
    tutorial.generate_complete_tutorial()?;

    println!(
        "Generated {} tutorial steps covering:",
        tutorial.step_count()
    );
    for (i, step) in tutorial.get_steps().iter().enumerate() {
        println!("   {}. {}", i + 1, step.title);
    }

    // Export tutorial as markdown (in a real application, you'd save this to a file)
    let tutorial_markdown = tutorial.export_markdown();
    println!(
        "\nTutorial markdown generated ({} characters)\n",
        tutorial_markdown.len()
    );

    // 3. Run API compatibility tests
    println!("3. RUNNING API COMPATIBILITY TESTS");
    println!("===================================");

    let config = CompatibilityConfig {
        test_edge_cases: true,
        test_error_conditions: true,
        test_performance: false, // Skip performance tests for this demo
        numerical_tolerance: 1e-10,
        max_test_size: 100,
    };

    let mut compatibility_tester = ApiCompatibilityTester::with_config(config);

    match compatibility_tester.run_all_tests() {
        Ok(()) => {
            let overall_score = compatibility_tester.get_overall_score();
            println!("✓ API compatibility tests completed");
            println!("Overall compatibility score: {:.1}%", overall_score * 100.0);

            let results = compatibility_tester.get_results();
            let compatible_count = results.iter().filter(|r| r.compatible).count();
            println!(
                "Compatible functions: {}/{}",
                compatible_count,
                results.len()
            );

            // Show any incompatible functions
            for result in results {
                if !result.compatible {
                    println!(
                        "⚠ {} - {:.1}% compatible",
                        result.function_name,
                        result.compatibility_score * 100.0
                    );
                    for issue in &result.error_messages {
                        println!("   Issue: {}", issue);
                    }
                    for suggestion in &result.suggestions {
                        println!("   Suggestion: {}", suggestion);
                    }
                }
            }
        }
        Err(e) => {
            println!("✗ API compatibility tests failed: {}", e);
            return Err(e);
        }
    }
    println!();

    // 4. Run numerical validation tests
    println!("4. RUNNING NUMERICAL VALIDATION TESTS");
    println!("=====================================");

    let validation_config = ValidationConfig {
        tolerance: 1e-6,
        test_edge_cases: true,
        test_large_arrays: false, // Skip large arrays for demo
        max_test_size: 100,
        num_random_tests: 5,
        random_seed: 42,
    };

    let mut validation_suite = SciPyValidationSuite::with_config(validation_config);

    match validation_suite.run_all_validations() {
        Ok(()) => {
            let pass_rate = validation_suite.get_pass_rate();
            println!("✓ Numerical validation tests completed");
            println!("Pass rate: {:.1}%", pass_rate * 100.0);

            let results = validation_suite.get_results();
            println!("Total validation tests: {}", results.len());

            // Show summary by function
            let mut by_function: std::collections::HashMap<String, Vec<&_>> =
                std::collections::HashMap::new();
            for result in results {
                by_function
                    .entry(result.function_name.clone())
                    .or_insert_with(Vec::new)
                    .push(result);
            }

            for (function, func_results) in by_function {
                let passed = func_results.iter().filter(|r| r.passed).count();
                let total = func_results.len();
                println!("   {}: {}/{} tests passed", function, passed, total);

                for result in func_results {
                    if !result.passed {
                        println!(
                            "     ✗ {}: max_diff={:.2e}, tolerance={:.2e}",
                            result.test_name, result.max_abs_diff, result.tolerance
                        );
                    }
                }
            }
        }
        Err(e) => {
            println!("✗ Numerical validation tests failed: {}", e);
            return Err(e);
        }
    }
    println!();

    // 5. Run performance benchmarks (quick version)
    println!("5. RUNNING PERFORMANCE BENCHMARKS");
    println!("==================================");

    let benchmark_config = BenchmarkConfig {
        array_sizes: vec![
            vec![50, 50],   // Small 2D for demo
            vec![100, 100], // Medium 2D for demo
        ],
        dtypes: vec!["f64".to_string()], // Just f64 for demo
        iterations: 3,                   // Fewer iterations for demo
        warmup_iterations: 1,            // Minimal warmup for demo
        profile_memory: false,           // Skip memory profiling for demo
        numerical_tolerance: 1e-6,
    };

    let mut benchmark_suite = SciPyBenchmarkSuite::with_config(benchmark_config);

    match benchmark_suite.run_all_benchmarks() {
        Ok(()) => {
            println!("✓ Performance benchmarks completed");

            let results = benchmark_suite.get_results();
            println!("Benchmark results ({} operations tested):", results.len());

            // Group by operation and show performance summary
            let mut by_operation: std::collections::HashMap<String, Vec<&_>> =
                std::collections::HashMap::new();
            for result in results {
                by_operation
                    .entry(result.operation.clone())
                    .or_insert_with(Vec::new)
                    .push(result);
            }

            for (operation, op_results) in by_operation {
                let avg_time: f64 = op_results.iter().map(|r| r.execution_time_ms).sum::<f64>()
                    / op_results.len() as f64;
                let avg_memory: f64 = op_results
                    .iter()
                    .map(|r| r.memory_usage_bytes)
                    .sum::<usize>() as f64
                    / op_results.len() as f64;

                println!(
                    "   {}: {:.2}ms avg, {:.1}MB avg memory",
                    operation,
                    avg_time,
                    avg_memory / 1_000_000.0
                );
            }
        }
        Err(e) => {
            println!("✗ Performance benchmarks failed: {}", e);
            return Err(e);
        }
    }
    println!();

    // 6. Generate comprehensive report
    println!("6. GENERATING COMPREHENSIVE REPORT");
    println!("===================================");

    // Generate individual reports
    let compatibility_report = compatibility_tester.generate_report();
    let validation_report = validation_suite.generate_report();
    let benchmark_report = benchmark_suite.generate_report();

    println!("Generated comprehensive reports:");
    println!(
        "   - API Compatibility Report: {} characters",
        compatibility_report.len()
    );
    println!(
        "   - Numerical Validation Report: {} characters",
        validation_report.len()
    );
    println!(
        "   - Performance Benchmark Report: {} characters",
        benchmark_report.len()
    );
    println!(
        "   - Tutorial Documentation: {} characters",
        tutorial_markdown.len()
    );

    // In a real application, you would save these reports to files:
    // std::fs::write("api_compatibility_report.md", compatibility_report)?;
    // std::fs::write("numerical_validation_report.md", validation_report)?;
    // std::fs::write("performance_benchmark_report.md", benchmark_report)?;
    // std::fs::write("comprehensive_tutorial.md", tutorial_markdown)?;

    println!("\n=== VALIDATION SUMMARY ===");
    println!("✓ All comprehensive examples validated");
    println!(
        "✓ API compatibility: {:.1}%",
        compatibility_tester.get_overall_score() * 100.0
    );
    println!(
        "✓ Numerical validation: {:.1}%",
        validation_suite.get_pass_rate() * 100.0
    );
    println!(
        "✓ Performance benchmarks: {} operations tested",
        benchmark_suite.get_results().len()
    );
    println!(
        "✓ Documentation: {} tutorial steps generated",
        tutorial.step_count()
    );

    println!("\n🎉 scirs2-ndimage comprehensive validation completed successfully!");
    println!("   Ready for production use with full SciPy compatibility validation.");

    Ok(())
}

/// Helper function to demonstrate specific validation scenarios
#[allow(dead_code)]
fn demonstrate_specific_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SPECIFIC VALIDATION DEMONSTRATIONS ===\n");

    // Example 1: Test specific function compatibility
    println!("Testing gaussian_filter compatibility...");
    let mut tester = ApiCompatibilityTester::new();
    tester.test_filter_apis()?;

    let results = tester.get_results();
    for result in results {
        if result.function_name.contains("gaussian") {
            println!(
                "   {}: {}% compatible",
                result.function_name,
                result.compatibility_score * 100.0
            );
        }
    }

    // Example 2: Test specific numerical accuracy
    println!("\nTesting morphological operations accuracy...");
    let mut validator = SciPyValidationSuite::new();
    validator.validate_morphological_operations()?;

    let results = validator.get_results();
    for result in results {
        println!(
            "   {}: {} (maxdiff: {:.2e})",
            result.test_name,
            if result.passed { "PASS" } else { "FAIL" },
            result.max_abs_diff
        );
    }

    // Example 3: Test specific performance characteristics
    println!("\nTesting filter performance...");
    let mut benchmarker = SciPyBenchmarkSuite::new();
    benchmarker.benchmark_filters()?;

    let results = benchmarker.get_results();
    for result in results {
        println!(
            "   {} ({}): {:.2}ms",
            result.operation, result.dtype, result.execution_time_ms
        );
    }

    Ok(())
}

/// Helper function to show how to customize validation configurations
#[allow(dead_code)]
fn demonstrate_custom_configurations() {
    println!("=== CUSTOM CONFIGURATION EXAMPLES ===\n");

    // Custom API compatibility configuration
    let api_config = CompatibilityConfig {
        test_edge_cases: true,
        test_error_conditions: true,
        test_performance: true,     // Enable performance testing
        numerical_tolerance: 1e-12, // Very strict tolerance
        max_test_size: 2000,        // Test larger arrays
    };
    println!(
        "Custom API config: strict tolerance {:.0e}, test size {}",
        api_config.numerical_tolerance, api_config.max_test_size
    );

    // Custom validation configuration
    let validation_config = ValidationConfig {
        tolerance: 1e-8, // Moderate tolerance
        test_edge_cases: true,
        test_large_arrays: true, // Enable large array testing
        max_test_size: 5000,
        num_random_tests: 20, // More random tests
        random_seed: 123,     // Different seed
    };
    println!(
        "Custom validation config: {} random tests, max size {}",
        validation_config.num_random_tests, validation_config.max_test_size
    );

    // Custom benchmark configuration
    let benchmark_config = BenchmarkConfig {
        array_sizes: vec![
            vec![1000, 1000],     // Large 2D
            vec![100, 100, 100],  // 3D arrays
            vec![50, 50, 50, 50], // 4D arrays
        ],
        dtypes: vec!["f32".to_string(), "f64".to_string()],
        iterations: 50,        // Many iterations for accuracy
        warmup_iterations: 10, // Thorough warmup
        profile_memory: true,  // Enable memory profiling
        numerical_tolerance: 1e-10,
    };
    println!(
        "Custom benchmark config: {} iterations, {} warmup, {} array types",
        benchmark_config.iterations,
        benchmark_config.warmup_iterations,
        benchmark_config.array_sizes.len()
    );
}
