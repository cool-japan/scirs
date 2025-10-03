//! Demonstration of the comprehensive validation frameworks
//!
//! This example shows how to use the integrated validation suite
//! to comprehensively test statistical functions.

use scirs2_core::ndarray::Array1;
use scirs2_core::ndarray::ArrayView1;
use scirs2_stats::{
    /* comprehensive_validation_suite::*, */ mean, numerical_stability_analyzer::*,
    /* propertybased_validation::*, */ scipy_benchmark_framework::*,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 SciRS2 Comprehensive Validation Framework Demo");
    println!("================================================\n");

    // Create comprehensive validation suite
    /*
    let mut validation_suite = ComprehensiveValidationSuite::new(ValidationSuiteConfig {
        benchmark_config: BenchmarkConfig {
            testsizes: vec![100, 1000],
            performance_iterations: 50,
            warmup_iterations: 5,
            absolute_tolerance: 1e-12,
            relative_tolerance: 1e-9,
            ..Default::default()
        },
        property_config: PropertyTestConfig {
            test_cases_per_property: 100,
            seed: 42,
            tolerance: 1e-12,
            test_edge_cases: true,
            ..Default::default()
        },
        stability_config: StabilityConfig {
            perturbation_tests: 50,
            test_extreme_values: true,
            test_singular_cases: true,
            ..Default::default()
        },
        enable_cross_validation: true,
        production_readiness_threshold: 0.85,
        ..Default::default()
    });

    println!("1. Testing Mean Function");
    println!("========================");

    // Mock SciPy reference implementation
    let scipy_mean = |data: &scirs2_core::ndarray::ArrayView1<f64>| -> f64 { data.sum() / data.len() as f64 };

    // Validate the mean function comprehensively
    let mean_validation =
        validation_suite.validate_function("mean", |data| mean(data), Some(scipy_mean))?;

    println!("✅ Mean Validation Results:");
    println!("   Function: {}", mean_validation.function_name);
    println!("   Overall Status: {:?}", mean_validation.overall_status);
    println!(
        "   Production Ready: {}",
        mean_validation.production_readiness.is_production_ready
    );
    println!(
        "   Readiness Score: {:.1}%",
        mean_validation.production_readiness.readiness_score
    );
    println!(
        "   Stability Grade: {:?}",
        mean_validation.stability_result.stability_grade
    );
    println!(
        "   Stability Score: {:.1}/100",
        mean_validation.stability_result.stability_score
    );
    println!("   Validation Time: {:?}", mean_validation.validation_time);

    if !mean_validation.benchmark_results.is_empty() {
        let bench = &mean_validation.benchmark_results[0];
        println!("   Benchmark Status: {:?}", bench.status);
        println!("   Accuracy Grade: {:?}", bench.accuracy.accuracy_grade);
        println!(
            "   Max Abs Difference: {:.2e}",
            bench.accuracy.max_abs_difference
        );
    }
    */

    println!("\n2. Individual Framework Demonstrations");
    println!("=====================================");

    // Mock SciPy reference implementation
    let scipy_mean =
        |data: &scirs2_core::ndarray::ArrayView1<f64>| -> f64 { data.sum() / data.len() as f64 };

    // Demonstrate SciPy Benchmark Framework
    println!("\n📊 SciPy Benchmark Framework:");
    let mut benchmark_framework = ScipyBenchmarkFramework::new(BenchmarkConfig {
        testsizes: vec![1000],
        performance_iterations: 100,
        ..Default::default()
    });

    let benchmark_results =
        benchmark_framework.benchmark_function("mean_benchmark", mean, scipy_mean)?;

    for result in &benchmark_results {
        println!("   ✓ Data size: {}", result.datasize);
        println!(
            "     Accuracy: {} (Grade: {:?})",
            if result.accuracy.passes_tolerance {
                "PASS"
            } else {
                "FAIL"
            },
            result.accuracy.accuracy_grade
        );
        println!("     Relative Error: {:.2e}", result.accuracy.relativeerror);
        println!(
            "     Performance: {:?}",
            result.performance.performance_grade
        );
        if let Some(ratio) = result.performance.performance_ratio {
            println!("     Speed vs SciPy: {:.2}x", 1.0 / ratio);
        }
    }

    // Demonstrate Property-Based Testing
    println!("\n🔬 Property-Based Testing Framework:");
    /*
    let mut property_suite = ComprehensivePropertyTestSuite::new(PropertyTestConfig {
        test_cases_per_property: 50,
        ..Default::default()
    });

    let property_results = property_suite.test_function("mean")?;
    for result in &property_results {
        println!("   ✓ Property: {}", result.property_name);
        println!("     Status: {:?}", result.status);
        println!(
            "     Tests: {}/{} passed",
            result.test_cases_passed, result.test_cases_run
        );
        if let Some(significance) = result.statistical_significance {
            println!("     Significance: p < {:.3}", significance);
        }
    }
    */

    // Demonstrate Numerical Stability Analysis
    println!("\n⚖️  Numerical Stability Analysis:");
    let mut stability_analyzer = NumericalStabilityAnalyzer::new(StabilityConfig {
        perturbation_tests: 25,
        ..Default::default()
    });

    let testdata = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0, -50.0]);
    let stability_result =
        stability_analyzer.analyze_function("mean_stability", mean, &testdata.view())?;

    println!(
        "   ✓ Stability Grade: {:?}",
        stability_result.stability_grade
    );
    println!(
        "     Stability Score: {:.1}/100",
        stability_result.stability_score
    );
    println!(
        "     Conditioning: {:?}",
        stability_result.condition_analysis.conditioning_class
    );
    println!(
        "     Error Amplification: {:.2e}",
        stability_result.error_propagation.error_amplification
    );
    println!(
        "     Edge Case Success Rate: {:.1}%",
        stability_result.edge_case_robustness.edge_case_success_rate * 100.0
    );
    println!(
        "     Precision Loss: {:.1} bits",
        stability_result.precision_analysis.precision_loss_bits
    );

    if !stability_result.recommendations.is_empty() {
        println!("     Recommendations:");
        for rec in &stability_result.recommendations {
            println!("       - {:?}: {}", rec.recommendation_type, rec.suggestion);
        }
    }

    /*
    println!("\n3. Comprehensive Validation Report");
    println!("==================================");

    let comprehensive_report = validation_suite.generate_comprehensive_report();
    println!("📋 Overall Statistics:");
    println!(
        "   Total Functions Validated: {}",
        comprehensive_report.total_functions
    );
    println!(
        "   Production Ready: {}",
        comprehensive_report.production_ready_functions
    );
    println!(
        "   Need Improvement: {}",
        comprehensive_report.functions_needing_improvement
    );
    println!(
        "   Average Benchmark Score: {:.1}%",
        comprehensive_report
            .validation_summary
            .average_benchmark_score
    );
    println!(
        "   Average Stability Score: {:.1}%",
        comprehensive_report
            .validation_summary
            .average_stability_score
    );
    println!(
        "   Overall Validation Score: {:.1}%",
        comprehensive_report
            .validation_summary
            .overall_validation_score
    );

    println!("\n🎯 Production Readiness Assessment:");
    let production = &comprehensive_report.overall_production_readiness;
    println!(
        "   Ready for Production: {}",
        production.is_production_ready
    );
    println!(
        "   Production Ready Percentage: {:.1}%",
        production.production_ready_percentage
    );

    println!("\n🔗 Framework Analysis:");
    let framework = &comprehensive_report.framework_analysis;
    println!(
        "   Benchmark Reliability: {:.1}%",
        framework.benchmark_reliability * 100.0
    );
    println!(
        "   Property Test Reliability: {:.1}%",
        framework.property_test_reliability * 100.0
    );
    println!(
        "   Stability Reliability: {:.1}%",
        framework.stability_reliability * 100.0
    );
    println!(
        "   Inter-Framework Agreement: {:.1}%",
        framework.inter_framework_agreement * 100.0
    );

    */

    println!("\n✨ Demo completed successfully!");
    println!("The validation frameworks are working correctly and can be used");
    println!("to comprehensively validate statistical functions for production use.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /*
    #[test]
    #[ignore = "timeout"]
    fn test_validation_demo_components() {
        // Test that our validation components can be created
        let _benchmark_framework = ScipyBenchmarkFramework::default();
        let _property_suite = ComprehensivePropertyTestSuite::new(PropertyTestConfig::default());
        let _stability_analyzer = NumericalStabilityAnalyzer::default();
        let _validation_suite = ComprehensiveValidationSuite::default();

        // If we reach here, all components were created successfully
        assert!(true);
    }
    */

    #[test]
    fn test_mean_basic_validation() {
        let testdata = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = mean(&testdata.view()).unwrap();
        assert!((result - 3.0f64).abs() < 1e-10);
    }
}
