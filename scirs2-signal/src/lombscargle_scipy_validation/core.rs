//! Core validation functions for Lomb-Scargle SciPy validation
//!
//! This module contains the main validation orchestration functions that
//! coordinate different aspects of the validation process.

use super::accuracy::validate_basic_accuracy;
use super::edge_cases::validate_edge_cases;
use super::normalization::validate_normalization_methods;
use super::performance::validate_performance_characteristics;
use super::statistical::validate_statistical_properties;
use super::types::*;
use crate::error::SignalResult;

/// Run comprehensive Lomb-Scargle validation against SciPy
#[allow(dead_code)]
pub fn validate_lombscargle_against_scipy(
    config: &ScipyValidationConfig,
) -> SignalResult<ScipyValidationResult> {
    let mut issues: Vec<String> = Vec::new();

    // 1. Basic accuracy validation
    let accuracy_results = validate_basic_accuracy(config)?;

    // 2. Normalization method validation
    let normalization_results = if config.test_normalizations {
        Some(validate_normalization_methods(config)?)
    } else {
        None
    };

    // 3. Edge case validation
    let edge_case_results = if config.test_edge_cases {
        Some(validate_edge_cases(config)?)
    } else {
        None
    };

    // 4. Statistical properties validation
    let statistical_results = validate_statistical_properties(config)?;

    // 5. Performance validation
    let performance_results = validate_performance_characteristics(config)?;

    // 6. Calculate overall summary
    let summary = calculate_overall_summary(
        &accuracy_results,
        &normalization_results,
        &edge_case_results,
        &statistical_results,
        &performance_results,
    );

    // Check for critical issues
    if accuracy_results.max_relative_error > config.max_error_percent / 100.0 {
        issues.push(format!(
            "Maximum relative error {:.4}% exceeds threshold {:.4}%",
            accuracy_results.max_relative_error * 100.0,
            config.max_error_percent
        ));
    }

    if accuracy_results.correlation < 0.999 {
        issues.push(format!(
            "Correlation with SciPy {:.6} is below expected 0.999",
            accuracy_results.correlation
        ));
    }

    Ok(ScipyValidationResult {
        accuracy_results,
        normalization_results,
        edge_case_results,
        statistical_results,
        performance_results,
        summary,
        issues,
    })
}

/// Calculate overall validation summary
#[allow(dead_code)]
pub fn calculate_overall_summary(
    accuracy: &AccuracyValidationResult,
    normalization: &Option<NormalizationValidationResult>,
    edge_cases: &Option<EdgeCaseValidationResult>,
    statistical: &StatisticalValidationResult,
    performance: &PerformanceValidationResult,
) -> ValidationSummary {
    let accuracy_score =
        (accuracy.correlation * 100.0).min(100.0 - accuracy.max_relative_error * 10000.0);

    let performance_score =
        (performance.speed_ratio * 50.0 + performance.scalability_score * 0.5).min(100.0);

    let reliability_score = statistical.consistency_score * 100.0;

    let overall_score =
        (accuracy_score * 0.5 + performance_score * 0.3 + reliability_score * 0.2).min(100.0);

    let passed =
        overall_score >= 85.0 && accuracy.max_relative_error < 0.01 && accuracy.correlation > 0.999;

    ValidationSummary {
        passed,
        accuracy_score,
        performance_score,
        reliability_score,
        overall_score,
    }
}
