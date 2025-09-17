//! Production stress testing modules
//!
//! This module provides comprehensive stress testing specifically designed for
//! production readiness validation of the interpolation library.
//!
//! ## Organization
//!
//! - `types`: Core type definitions and data structures
//! - `generators`: Test data generation functions
//! - `execution`: Test execution engine and orchestration
//! - `helpers`: Utility functions and test result creation
//! - `reporting`: Report generation and analysis
//!
//! ## Key Features
//!
//! - **Extreme input stress testing**: Test with pathological data
//! - **Memory pressure testing**: Validate behavior under memory constraints
//! - **Numerical edge case validation**: Test boundary conditions
//! - **Error handling verification**: Ensure graceful error handling
//! - **Performance under stress**: Measure performance degradation
//! - **Resource exhaustion scenarios**: Test system limits

pub mod execution;
pub mod generators;
pub mod helpers;
pub mod reporting;
pub mod types;

// Re-export core types and functions
pub use types::{
    BaselinePerformance, ErrorInfo, IssueSeverity, MemoryUsageStats, PerformanceAnalysis,
    ProductionImpact, ProductionReadiness, ProductionStressTester, ScalabilityAssessment,
    StressPerformanceMetrics, StressTestCategory, StressTestConfig, StressTestIssue,
    StressTestReport, StressTestResult, StressTestSummary, TestStatus,
};

pub use generators::{
    create_constant_data, create_duplicate_x_data, create_edge_case_data, create_empty_data,
    create_exponential_data, create_extreme_y_data, create_large_test_data, create_linear_data,
    create_mismatched_data, create_monotonic_extreme_data, create_nan_inf_data,
    create_oscillatory_data, create_pseudo_random_data, create_quadratic_data,
    create_rapid_change_data, create_single_point_data, create_sparse_data, create_step_data,
    create_unsorted_x_data, generate_test_data, get_error_test_data_types,
    get_general_test_data_types, get_pathological_data_types,
};

use crate::error::InterpolateResult;
use crate::traits::InterpolationFloat;

/// Run production stress tests with default configuration
pub fn run_production_stress_tests<T: InterpolationFloat + std::panic::RefUnwindSafe>(
) -> InterpolateResult<StressTestReport> {
    let mut tester = ProductionStressTester::<T>::new();
    tester.run_comprehensive_stress_tests()
}

/// Run quick stress tests with reduced scope
pub fn run_quick_stress_tests<T: InterpolationFloat + std::panic::RefUnwindSafe>(
) -> InterpolateResult<StressTestReport> {
    let config = StressTestConfig {
        max_data_size: 50_000,
        stress_iterations: 10,
        test_timeout: 60,
        memory_limit: Some(1024 * 1024 * 1024), // 1GB
        test_extreme_cases: false,
        max_performance_degradation: 5.0,
    };

    let mut tester = ProductionStressTester::<T>::with_config(config);
    tester.run_comprehensive_stress_tests()
}

/// Run stress tests with custom configuration
pub fn run_stress_tests_with_config<T: InterpolationFloat + std::panic::RefUnwindSafe>(
    config: StressTestConfig,
) -> InterpolateResult<StressTestReport> {
    let mut tester = ProductionStressTester::<T>::with_config(config);
    tester.run_comprehensive_stress_tests()
}

/// Create a stress tester with default configuration
pub fn create_stress_tester<T: InterpolationFloat>() -> ProductionStressTester<T> {
    ProductionStressTester::new()
}

/// Create a stress tester with custom configuration
pub fn create_stress_tester_with_config<T: InterpolationFloat>(
    config: StressTestConfig,
) -> ProductionStressTester<T> {
    ProductionStressTester::with_config(config)
}

/// Module information and version
pub mod info {
    /// Module version
    pub const VERSION: &str = "0.1.0";

    /// List of available stress test features
    pub const FEATURES: &[&str] = &[
        "extreme_data_size_testing",
        "pathological_data_patterns",
        "numerical_edge_cases",
        "memory_pressure_testing",
        "error_handling_validation",
        "performance_stress_testing",
        "numerical_stability_analysis",
        "error_message_clarity",
        "resource_exhaustion_recovery",
        "production_readiness_assessment",
        "scalability_analysis",
        "comprehensive_reporting",
    ];

    /// Get feature availability
    pub fn has_feature(feature: &str) -> bool {
        FEATURES.contains(&feature)
    }

    /// Get module information
    pub fn module_info() -> String {
        format!(
            "Production Stress Testing v{}\\nFeatures: {}\\nTest Categories: 9\\nReport Types: Comprehensive",
            VERSION,
            FEATURES.len()
        )
    }
}

/// Convenience re-exports for common usage patterns
pub mod prelude {
    pub use super::types::{
        ProductionReadiness, ProductionStressTester, StressTestConfig, StressTestReport, TestStatus,
    };
    pub use super::{create_stress_tester, run_production_stress_tests, run_quick_stress_tests};
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::traits::InterpolationFloat;

    #[test]
    fn test_stress_tester_creation() {
        let _tester = create_stress_tester::<f64>();
        assert!(true); // Basic creation test
    }

    #[test]
    fn test_custom_config_creation() {
        let config = StressTestConfig {
            max_data_size: 10_000,
            stress_iterations: 5,
            test_timeout: 30,
            memory_limit: Some(512 * 1024 * 1024),
            test_extreme_cases: false,
            max_performance_degradation: 3.0,
        };

        let _tester = create_stress_tester_with_config::<f64>(config);
        assert!(true); // Basic creation test with config
    }

    #[test]
    fn test_data_generation() {
        // Test various data generators
        let result = generate_test_data::<f64>("linear", 100);
        assert!(result.is_ok());

        let (x, y) = result.unwrap();
        assert_eq!(x.len(), 100);
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_pathological_data_types() {
        let types = get_pathological_data_types();
        assert!(!types.is_empty());
        assert!(types.contains(&"constant"));
        assert!(types.contains(&"duplicate_x"));
        assert!(types.contains(&"extreme_y"));
    }

    #[test]
    fn test_error_test_data_types() {
        let types = get_error_test_data_types();
        assert!(!types.is_empty());
        assert!(types.contains(&"empty"));
        assert!(types.contains(&"single_point"));
        assert!(types.contains(&"mismatched"));
    }

    #[test]
    fn test_general_test_data_types() {
        let types = get_general_test_data_types();
        assert!(!types.is_empty());
        assert!(types.contains(&"linear"));
        assert!(types.contains(&"quadratic"));
        assert!(types.contains(&"exponential"));
    }

    #[test]
    fn test_specific_data_generators() {
        // Test individual generators
        assert!(create_constant_data::<f64>(50).is_ok());
        assert!(create_linear_data::<f64>(50).is_ok());
        assert!(create_quadratic_data::<f64>(50).is_ok());
        assert!(create_exponential_data::<f64>(50).is_ok());
        assert!(create_step_data::<f64>(50).is_ok());
        assert!(create_pseudo_random_data::<f64>(50).is_ok());
        assert!(create_oscillatory_data::<f64>(50).is_ok());
        assert!(create_sparse_data::<f64>(100).is_ok());
        assert!(create_extreme_y_data::<f64>(50).is_ok());
        assert!(create_nan_inf_data::<f64>(50).is_ok());
        assert!(create_monotonic_extreme_data::<f64>(50).is_ok());
        assert!(create_rapid_change_data::<f64>(50).is_ok());
    }

    #[test]
    fn test_edge_case_generators() {
        assert!(create_edge_case_data::<f64>(50, 0.0, 10.0).is_ok());
        assert!(create_empty_data::<f64>().is_ok());
        assert!(create_single_point_data::<f64>().is_ok());
        assert!(create_mismatched_data::<f64>().is_ok());
        assert!(create_unsorted_x_data::<f64>().is_ok());
    }

    #[test]
    fn test_stress_test_config_default() {
        let config = StressTestConfig::default();
        assert_eq!(config.max_data_size, 1_000_000);
        assert_eq!(config.stress_iterations, 100);
        assert_eq!(config.test_timeout, 300);
        assert!(config.memory_limit.is_some());
        assert!(config.test_extreme_cases);
        assert_eq!(config.max_performance_degradation, 10.0);
    }

    #[test]
    fn test_production_stress_tester_default() {
        let tester = ProductionStressTester::<f64>::default();
        assert_eq!(tester.results.len(), 0);
        assert!(tester.baseline_performance.is_none());
        assert_eq!(tester.error_patterns.len(), 0);
    }

    #[test]
    fn test_memory_usage_stats_default() {
        let stats = MemoryUsageStats::default();
        assert_eq!(stats.peak_usage, 0);
        assert_eq!(stats.average_usage, 0);
        assert_eq!(stats.initial_usage, 0);
        assert_eq!(stats.final_usage, 0);
        assert_eq!(stats.memory_leaked, 0);
    }

    #[test]
    fn test_stress_performance_metrics_default() {
        let metrics = StressPerformanceMetrics::default();
        assert_eq!(metrics.mean_execution_time.as_millis(), 0);
        assert_eq!(metrics.max_execution_time.as_millis(), 0);
        assert_eq!(metrics.min_execution_time.as_millis(), 0);
        assert!(metrics.degradation_factor.is_none());
        assert!(metrics.throughput.is_none());
        assert!(metrics.memory_efficiency.is_none());
    }

    #[test]
    fn test_module_info() {
        let info = info::module_info();
        assert!(info.contains("Production Stress Testing"));
        assert!(info.contains("0.1.0"));

        assert!(info::has_feature("extreme_data_size_testing"));
        assert!(info::has_feature("pathological_data_patterns"));
        assert!(info::has_feature("numerical_edge_cases"));
        assert!(!info::has_feature("nonexistent_feature"));
    }

    #[test]
    fn test_test_status_display() {
        use std::fmt::Write;
        let mut output = String::new();

        write!(&mut output, "{}", TestStatus::Passed).unwrap();
        assert_eq!(output, "PASSED");

        output.clear();
        write!(&mut output, "{}", TestStatus::Failed).unwrap();
        assert_eq!(output, "FAILED");

        output.clear();
        write!(&mut output, "{}", TestStatus::PassedWithWarnings).unwrap();
        assert_eq!(output, "PASSED (with warnings)");
    }

    #[test]
    fn test_issue_severity_display() {
        use std::fmt::Write;
        let mut output = String::new();

        write!(&mut output, "{}", IssueSeverity::Critical).unwrap();
        assert_eq!(output, "CRITICAL");

        output.clear();
        write!(&mut output, "{}", IssueSeverity::High).unwrap();
        assert_eq!(output, "HIGH");
    }

    #[test]
    fn test_production_impact_display() {
        use std::fmt::Write;
        let mut output = String::new();

        write!(&mut output, "{}", ProductionImpact::Blocking).unwrap();
        assert_eq!(output, "BLOCKING");

        output.clear();
        write!(&mut output, "{}", ProductionImpact::Major).unwrap();
        assert_eq!(output, "MAJOR");
    }

    #[test]
    fn test_production_readiness_display() {
        use std::fmt::Write;
        let mut output = String::new();

        write!(&mut output, "{}", ProductionReadiness::Ready).unwrap();
        assert_eq!(output, "Ready for Production");

        output.clear();
        write!(&mut output, "{}", ProductionReadiness::NotReady).unwrap();
        assert_eq!(output, "Not Ready for Production");
    }

    #[test]
    fn test_stress_test_category_display() {
        use std::fmt::Write;
        let mut output = String::new();

        write!(&mut output, "{}", StressTestCategory::ExtremeDataSize).unwrap();
        assert_eq!(output, "Extreme Data Size");

        output.clear();
        write!(&mut output, "{}", StressTestCategory::PathologicalData).unwrap();
        assert_eq!(output, "Pathological Data");
    }

    #[test]
    fn test_comprehensive_workflow() {
        // This test verifies the complete workflow integration
        let config = StressTestConfig {
            max_data_size: 1_000,                  // Small for testing
            stress_iterations: 2,                  // Minimal iterations
            test_timeout: 10,                      // Short timeout
            memory_limit: Some(128 * 1024 * 1024), // 128MB
            test_extreme_cases: false,             // Skip extreme cases for speed
            max_performance_degradation: 5.0,
        };

        // Test that we can create a tester and it has the expected config
        let tester = create_stress_tester_with_config::<f64>(config.clone());
        assert_eq!(tester.config.max_data_size, 1_000);
        assert_eq!(tester.config.stress_iterations, 2);
        assert_eq!(tester.config.test_timeout, 10);

        // Test that various data generation functions work together
        let data_types = get_pathological_data_types();
        for data_type in data_types.iter().take(3) {
            // Test first 3 to keep test fast
            let result = generate_test_data::<f64>(data_type, 100);
            assert!(result.is_ok(), "Failed to generate {} data", data_type);
        }

        // Test error data types
        let error_types = get_error_test_data_types();
        for error_type in error_types.iter().take(2) {
            // Test first 2
            let result = generate_test_data::<f64>(error_type, 0);
            assert!(
                result.is_ok(),
                "Failed to generate {} error data",
                error_type
            );
        }
    }
}
