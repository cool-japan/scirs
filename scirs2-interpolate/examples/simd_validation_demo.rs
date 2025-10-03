//! SIMD Performance Validation Demonstration
//!
//! This example demonstrates the comprehensive SIMD performance validation system
//! across different architectures and instruction sets.

use scirs2_interpolate::{
    run_simd_validation, run_simd_validation_with_config, SimdValidationConfig,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SIMD Performance Validation Demo");
    println!("{}", "=".repeat(50));

    // Run quick validation with smaller test sizes for demo
    let config = SimdValidationConfig {
        test_sizes: vec![100, 1_000, 5_000], // Smaller sizes for quick demo
        timing_iterations: 10,               // Fewer iterations for speed
        warmup_iterations: 3,
        correctness_tolerance: 1e-10,
        test_all_instruction_sets: true,
        validate_memory_alignment: true,
        run_regression_detection: false, // Skip for demo
        max_benchmark_time: 10.0,        // Shorter time limit
    };

    println!("Starting SIMD validation with custom configuration...");
    match run_simd_validation_with_config::<f64>(config) {
        Ok(summary) => {
            summary.print_report();

            if summary.meets_quality_standards() {
                println!("\n🎉 SIMD validation PASSED - meets quality standards!");
                println!(
                    "✅ Success rate: {:.1}%",
                    summary.overall_success_rate * 100.0
                );
                println!("✅ Average speedup: {:.2}x", summary.average_speedup);
            } else {
                println!("\n⚠️  SIMD validation completed with issues:");
                if summary.overall_success_rate < 0.95 {
                    println!(
                        "  - Success rate too low: {:.1}%",
                        summary.overall_success_rate * 100.0
                    );
                }
                if summary.average_speedup < 1.5 {
                    println!(
                        "  - Average speedup too low: {:.2}x",
                        summary.average_speedup
                    );
                }
            }

            // Show JSON report for CI/CD integration
            println!("\nJSON Report for CI/CD:");
            println!("{}", summary.to_json());
        }
        Err(e) => {
            eprintln!("❌ SIMD validation failed: {}", e);
            return Err(e.into());
        }
    }

    // Also run default validation for f32
    println!("\n{}", "=".repeat(50));
    println!("Running default f32 validation...");

    match run_simd_validation::<f32>() {
        Ok(summary) => {
            println!("f32 validation completed:");
            println!(
                "  Tests: {} passed, {} failed",
                summary.passed_tests, summary.failed_tests
            );
            println!("  Average speedup: {:.2}x", summary.average_speedup);
        }
        Err(e) => {
            eprintln!("f32 validation failed: {}", e);
        }
    }

    Ok(())
}
