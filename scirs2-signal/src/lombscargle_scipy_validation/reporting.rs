//! Comprehensive validation reporting
//!
//! This module provides functions to run comprehensive validation and generate
//! detailed reports of the Lomb-Scargle implementation performance.

use super::core::validate_lombscargle_against_scipy;
use super::types::*;
use crate::error::SignalResult;

/// Run comprehensive validation and print detailed report
#[allow(dead_code)]
pub fn run_comprehensive_validation() -> SignalResult<()> {
    println!("Running comprehensive Lomb-Scargle validation against SciPy...");

    let config = ScipyValidationConfig::default();
    let results = validate_lombscargle_against_scipy(&config)?;

    println!("\n=== Validation Results ===");
    println!(
        "Overall Status: {}",
        if results.summary.passed {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    println!("Overall Score: {:.1}/100", results.summary.overall_score);
    println!("Accuracy Score: {:.1}/100", results.summary.accuracy_score);
    println!(
        "Performance Score: {:.1}/100",
        results.summary.performance_score
    );
    println!(
        "Reliability Score: {:.1}/100",
        results.summary.reliability_score
    );

    println!("\n=== Accuracy Metrics ===");
    println!(
        "Maximum Relative Error: {:.2e}",
        results.accuracy_results.max_relative_error
    );
    println!("RMSE: {:.2e}", results.accuracy_results.rmse);
    println!(
        "Correlation with SciPy: {:.6}",
        results.accuracy_results.correlation
    );
    println!(
        "Passed Cases: {}/{}",
        results.accuracy_results.passed_cases, results.accuracy_results.total_cases
    );

    if let Some(ref norm_results) = results.normalization_results {
        println!("\n=== Normalization Methods ===");
        println!("Best Method: {}", norm_results.best_method);
        println!("Consistency Score: {:.3}", norm_results.consistency_score);
        for (method, result) in &norm_results.method_results {
            println!(
                "  {}: correlation={:.4}, passed={}/{}",
                method, result.correlation, result.passed_cases, result.total_cases
            );
        }
    }

    if let Some(ref edge_results) = results.edge_case_results {
        println!("\n=== Edge Cases ===");
        println!(
            "Sparse Sampling: {}",
            if edge_results.sparse_sampling {
                "PASS"
            } else {
                "FAIL"
            }
        );
        println!(
            "Extreme Dynamic Range: {}",
            if edge_results.extreme_dynamic_range {
                "PASS"
            } else {
                "FAIL"
            }
        );
        println!(
            "Short Time Series: {}",
            if edge_results.short_time_series {
                "PASS"
            } else {
                "FAIL"
            }
        );
        println!(
            "High Freq Resolution: {}",
            if edge_results.high_freq_resolution {
                "PASS"
            } else {
                "FAIL"
            }
        );
        println!("Stability Score: {:.3}", edge_results.stability_score);
    }

    println!("\n=== Statistical Properties ===");
    println!(
        "False Alarm Rate: {:.3}",
        results.statistical_results.false_alarm_rate
    );
    println!(
        "Detection Power: {:.3}",
        results.statistical_results.detection_power
    );
    println!(
        "CI Coverage: {:.3}",
        results.statistical_results.ci_coverage
    );

    println!("\n=== Performance ===");
    println!(
        "Speed Ratio (ours/scipy): {:.2}",
        results.performance_results.speed_ratio
    );
    println!(
        "Memory Ratio (ours/scipy): {:.2}",
        results.performance_results.memory_ratio
    );
    println!(
        "Scalability Score: {:.1}",
        results.performance_results.scalability_score
    );

    if !results.issues.is_empty() {
        println!("\n=== Issues Found ===");
        for issue in &results.issues {
            println!("  - {}", issue);
        }
    }

    println!("\nValidation completed!");

    Ok(())
}
