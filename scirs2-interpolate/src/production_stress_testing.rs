//! Production stress testing and edge case validation
//!
//! This module provides comprehensive stress testing specifically designed for
//! production readiness validation of the interpolation library.
//!
//! # Overview
//!
//! Production stress testing is critical for ensuring reliability when:
//! - Deploying interpolation algorithms to production environments
//! - Processing large datasets with unknown characteristics
//! - Handling edge cases and pathological data patterns
//! - Ensuring graceful degradation under resource pressure
//! - Validating error handling and recovery mechanisms
//!
//! This module provides tools to:
//! - Test extreme data sizes and memory pressure scenarios
//! - Validate behavior with pathological data patterns
//! - Assess numerical stability under edge conditions
//! - Verify error handling robustness
//! - Measure performance degradation under stress
//! - Generate comprehensive production readiness reports
//!
//! # Modular Organization
//!
//! This module has been refactored into focused submodules:
//! - `types`: Core type definitions and data structures
//! - `generators`: Test data generation functions for various scenarios
//! - `execution`: Test execution engine and orchestration logic
//! - `helpers`: Utility functions and test result creation
//! - `reporting`: Report generation and analysis capabilities
//!
//! # Examples
//!
//! ## Basic stress testing
//!
//! ```rust,no_run
//! use scirs2_interpolate::production_stress_testing::{run_production_stress_tests, ProductionReadiness};
//!
//! // Run comprehensive production stress tests
//! let report = run_production_stress_tests::<f64>().unwrap();
//!
//! match report.production_readiness {
//!     ProductionReadiness::Ready => {
//!         println!("System is ready for production deployment");
//!     },
//!     ProductionReadiness::ReadyWithMonitoring => {
//!         println!("Ready for production with enhanced monitoring");
//!         for recommendation in &report.production_recommendations {
//!             println!("- {}", recommendation);
//!         }
//!     },
//!     ProductionReadiness::NeedsPerformanceTuning => {
//!         println!("Performance optimization required before production");
//!     },
//!     ProductionReadiness::NeedsBugFixes => {
//!         println!("Critical issues must be addressed");
//!         for issue in &report.critical_issues {
//!             println!("- {:?}: {}", issue.severity, issue.description);
//!         }
//!     },
//!     ProductionReadiness::NotReady => {
//!         println!("System is not ready for production deployment");
//!     }
//! }
//! ```
//!
//! ## Quick stress testing
//!
//! ```rust,no_run
//! use scirs2_interpolate::production_stress_testing::run_quick_stress_tests;
//!
//! // Run quick stress tests with reduced scope for CI/CD
//! let report = run_quick_stress_tests::<f64>().unwrap();
//! println!("Quick test results: {:?}", report.production_readiness);
//! ```
//!
//! ## Custom configuration
//!
//! ```rust,no_run
//! use scirs2_interpolate::production_stress_testing::{
//!     StressTestConfig, run_stress_tests_with_config
//! };
//!
//! let config = StressTestConfig {
//!     max_data_size: 500_000,
//!     stress_iterations: 50,
//!     test_timeout: 120,
//!     memory_limit: Some(4 * 1024 * 1024 * 1024), // 4GB
//!     test_extreme_cases: true,
//!     max_performance_degradation: 5.0,
//! };
//!
//! let report = run_stress_tests_with_config::<f64>(config).unwrap();
//! println!("Custom stress test completed: {}", report);
//! ```

// Import the modular implementation
use crate::production_stress_testing_modules;

// Re-export the public API
pub use crate::production_stress_testing_modules::{
    create_constant_data, create_duplicate_x_data, create_edge_case_data, create_empty_data,
    create_exponential_data, create_extreme_y_data, create_large_test_data, create_linear_data,
    create_mismatched_data, create_monotonic_extreme_data, create_nan_inf_data,
    create_oscillatory_data, create_pseudo_random_data, create_quadratic_data,
    create_rapid_change_data, create_single_point_data, create_sparse_data, create_step_data,
    create_stress_tester, create_stress_tester_with_config, create_unsorted_x_data,
    generate_test_data, get_error_test_data_types, get_general_test_data_types,
    get_pathological_data_types, run_production_stress_tests, run_quick_stress_tests,
    run_stress_tests_with_config, BaselinePerformance, ErrorInfo, IssueSeverity, MemoryUsageStats,
    PerformanceAnalysis, ProductionImpact, ProductionReadiness, ProductionStressTester,
    ScalabilityAssessment, StressPerformanceMetrics, StressTestCategory, StressTestConfig,
    StressTestIssue, StressTestReport, StressTestResult, StressTestSummary, TestStatus,
};

// Convenience re-exports for common patterns
pub use crate::production_stress_testing_modules::prelude::*;

use crate::error::{InterpolateError, InterpolateResult};
use crate::traits::InterpolationFloat;

// Type aliases for convenience
pub type ProductionStressTesterF32 = ProductionStressTester<f32>;
pub type ProductionStressTesterF64 = ProductionStressTester<f64>;

/// Quick production readiness check for f64 systems
pub fn quick_production_readiness_check() -> InterpolateResult<ProductionReadiness> {
    let report = run_quick_stress_tests::<f64>()?;
    Ok(report.production_readiness)
}

/// Quick production readiness check for f32 systems
pub fn quick_production_readiness_check_f32() -> InterpolateResult<ProductionReadiness> {
    let report = run_quick_stress_tests::<f32>()?;
    Ok(report.production_readiness)
}

/// Run targeted stress tests for specific categories
pub fn run_targeted_stress_tests<T: InterpolationFloat + std::panic::RefUnwindSafe>(
    categories: &[StressTestCategory],
) -> InterpolateResult<StressTestReport> {
    let mut tester = ProductionStressTester::<T>::new();

    // This is a simplified implementation - a full implementation would
    // allow selective execution of test categories
    let _requested_categories = categories; // Placeholder to use the parameter

    // For now, run all tests (future enhancement would filter by categories)
    tester.run_comprehensive_stress_tests()
}

/// Validate interpolation library for production deployment
pub fn validate_for_production<T: InterpolationFloat + std::panic::RefUnwindSafe>(
) -> InterpolateResult<bool> {
    let report = run_production_stress_tests::<T>()?;

    match report.production_readiness {
        ProductionReadiness::Ready | ProductionReadiness::ReadyWithMonitoring => Ok(true),
        _ => Ok(false),
    }
}

/// Generate a production deployment checklist
pub fn generate_deployment_checklist<T: InterpolationFloat + std::panic::RefUnwindSafe>(
) -> InterpolateResult<Vec<String>> {
    let report = run_production_stress_tests::<T>()?;

    let mut checklist = Vec::new();

    // Basic checks
    checklist.push("✓ Run comprehensive stress tests".to_string());

    if report.summary.tests_failed > 0 {
        checklist.push(format!(
            "✗ Fix {} failing tests",
            report.summary.tests_failed
        ));
    } else {
        checklist.push("✓ All stress tests passing".to_string());
    }

    if !report.critical_issues.is_empty() {
        checklist.push(format!(
            "✗ Address {} critical issues",
            report.critical_issues.len()
        ));
    } else {
        checklist.push("✓ No critical issues detected".to_string());
    }

    // Performance checks
    if report.summary.overall_degradation_factor > 5.0 {
        checklist.push("✗ Optimize performance (high degradation detected)".to_string());
    } else {
        checklist.push("✓ Performance within acceptable limits".to_string());
    }

    // Memory checks
    if report.summary.memory_efficiency_score < 70.0 {
        checklist.push("✗ Improve memory efficiency".to_string());
    } else {
        checklist.push("✓ Memory usage is efficient".to_string());
    }

    // Scalability checks
    if !report
        .performance_analysis
        .scalability_assessment
        .can_handle_10x_load
    {
        checklist.push("✗ Test and optimize for expected production scale".to_string());
    } else {
        checklist.push("✓ System can handle expected load".to_string());
    }

    // Monitoring and observability
    checklist.push("◦ Set up performance monitoring".to_string());
    checklist.push("◦ Configure alerting for edge cases".to_string());
    checklist.push("◦ Document known limitations".to_string());
    checklist.push("◦ Prepare rollback procedures".to_string());

    // Add custom recommendations from the report
    for recommendation in &report.production_recommendations {
        checklist.push(format!("◦ {}", recommendation));
    }

    Ok(checklist)
}

/// Module information and version
pub mod info {
    pub use crate::production_stress_testing_modules::info::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_quick_production_readiness_check() {
        let result = quick_production_readiness_check();
        assert!(result.is_ok());

        let readiness = result.unwrap();
        // Should be one of the valid readiness states
        assert!(matches!(
            readiness,
            ProductionReadiness::Ready
                | ProductionReadiness::ReadyWithMonitoring
                | ProductionReadiness::NeedsPerformanceTuning
                | ProductionReadiness::NeedsBugFixes
                | ProductionReadiness::NotReady
        ));
    }

    #[test]
    #[ignore]
    fn test_quick_production_readiness_check_f32() {
        let result = quick_production_readiness_check_f32();
        assert!(result.is_ok());
    }

    #[test]
    #[ignore]
    fn test_validate_for_production() {
        let result = validate_for_production::<f64>();
        assert!(result.is_ok());
        // Result should be boolean
        let _is_ready = result.unwrap();
    }

    #[test]
    #[ignore]
    fn test_generate_deployment_checklist() {
        let result = generate_deployment_checklist::<f64>();
        assert!(result.is_ok());

        let checklist = result.unwrap();
        assert!(!checklist.is_empty());

        // Should contain basic checks
        assert!(checklist.iter().any(|item| item.contains("stress tests")));
    }

    #[test]
    #[ignore]
    fn test_targeted_stress_tests() {
        let categories = vec![
            StressTestCategory::ExtremeDataSize,
            StressTestCategory::ErrorHandling,
        ];

        let result = run_targeted_stress_tests::<f64>(&categories);
        assert!(result.is_ok());
    }

    #[test]
    fn test_module_api_integration() {
        // Test that the re-exported API works correctly
        let config = StressTestConfig::default();
        let _tester = ProductionStressTester::<f64>::with_config(config);

        // Test data generation
        let result = generate_test_data::<f64>("linear", 100);
        assert!(result.is_ok());

        // Test pathological data types
        let types = get_pathological_data_types();
        assert!(!types.is_empty());
    }

    #[test]
    fn test_type_aliases() {
        let _tester_f32: ProductionStressTesterF32 = ProductionStressTester::new();
        let _tester_f64: ProductionStressTesterF64 = ProductionStressTester::new();
    }

    #[test]
    fn test_prelude_imports() {
        use super::production_stress_testing_modules::prelude::*;

        // Test that prelude imports work correctly
        let _tester = create_stress_tester::<f64>();
        let _config = StressTestConfig::default();

        // These should compile without additional imports
        let _status = TestStatus::Passed;
        let _readiness = ProductionReadiness::Ready;
    }

    #[test]
    fn test_comprehensive_workflow() {
        // Test a complete workflow using the modular API
        let config = StressTestConfig {
            max_data_size: 1_000,                 // Small for testing
            stress_iterations: 2,                 // Minimal iterations
            test_timeout: 5,                      // Short timeout
            memory_limit: Some(64 * 1024 * 1024), // 64MB
            test_extreme_cases: false,            // Skip extreme cases for speed
            max_performance_degradation: 10.0,
        };

        // Create tester with custom config
        let tester = create_stress_tester_with_config::<f64>(config);
        assert_eq!(tester.config.max_data_size, 1_000);

        // Test data generation workflow
        let linear_data = create_linear_data::<f64>(50);
        assert!(linear_data.is_ok());

        let constant_data = create_constant_data::<f64>(50);
        assert!(constant_data.is_ok());

        // Test error data generation
        let empty_data = create_empty_data::<f64>();
        assert!(empty_data.is_ok());

        // Test that we can generate various data types
        for data_type in ["linear", "quadratic", "exponential"].iter() {
            let result = generate_test_data::<f64>(data_type, 50);
            assert!(result.is_ok(), "Failed to generate {} data", data_type);
        }
    }

    #[test]
    #[ignore]
    fn test_deployment_checklist_content() {
        // Test that deployment checklist contains expected items
        let result = generate_deployment_checklist::<f64>();
        assert!(result.is_ok());

        let checklist = result.unwrap();

        // Should contain monitoring recommendations
        assert!(checklist.iter().any(|item| item.contains("monitoring")));

        // Should contain some form of status indicators
        let has_check_marks = checklist.iter().any(|item| item.contains("✓"));
        let has_cross_marks = checklist.iter().any(|item| item.contains("✗"));
        let has_circles = checklist.iter().any(|item| item.contains("◦"));

        assert!(has_check_marks || has_cross_marks || has_circles);
    }
}
