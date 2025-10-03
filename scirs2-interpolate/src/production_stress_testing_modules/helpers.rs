//! Helper functions and utilities for production stress testing
//!
//! This module contains utility functions for creating test results, calculating metrics,
//! and other common operations used throughout the stress testing system.

use super::generators::*;
use super::types::*;
use crate::error::{InterpolateError, InterpolateResult};
use crate::traits::InterpolationFloat;
use scirs2_core::ndarray::Array1;
use std::time::{Duration, Instant};

impl<T: InterpolationFloat + std::panic::RefUnwindSafe> ProductionStressTester<T> {
    /// Estimate memory usage for a given data size
    pub fn estimate_memory_usage(&self, size: usize) -> u64 {
        // Rough estimation: 2 arrays of T + overhead
        (size * 2 * std::mem::size_of::<T>() + 1024) as u64
    }

    /// Calculate performance metrics from execution times
    pub fn calculate_performance_metrics(
        &self,
        execution_times: &[Duration],
        baseline: Option<&BaselinePerformance>,
    ) -> StressPerformanceMetrics {
        if execution_times.is_empty() {
            return StressPerformanceMetrics::default();
        }

        let total_time: Duration = execution_times.iter().sum();
        let mean_time = total_time / (execution_times.len() as u32);
        let min_time = *execution_times
            .iter()
            .min()
            .unwrap_or(&Duration::from_millis(0));
        let max_time = *execution_times
            .iter()
            .max()
            .unwrap_or(&Duration::from_millis(0));

        let degradation_factor = baseline.map(|b| {
            if b.execution_time.as_nanos() > 0 {
                mean_time.as_nanos() as f64 / b.execution_time.as_nanos() as f64
            } else {
                1.0
            }
        });

        let throughput = if mean_time.as_secs_f64() > 0.0 {
            Some(1.0 / mean_time.as_secs_f64())
        } else {
            None
        };

        StressPerformanceMetrics {
            mean_execution_time: mean_time,
            max_execution_time: max_time,
            min_execution_time: min_time,
            degradation_factor,
            throughput,
            memory_efficiency: Some(0.8), // Placeholder
        }
    }

    /// Create a success result for a test
    pub fn create_success_result(
        &self,
        test_name: &str,
        data_size: usize,
        execution_time: Duration,
        category: StressTestCategory,
    ) -> StressTestResult {
        StressTestResult {
            test_name: test_name.to_string(),
            category,
            input_characteristics: format!("{} data points", data_size),
            status: TestStatus::Passed,
            execution_time,
            performance: self.calculate_performance_metrics(
                &[execution_time],
                self.baseline_performance.as_ref(),
            ),
            error_info: None,
            memory_usage: MemoryUsageStats {
                peak_usage: self.estimate_memory_usage(data_size),
                average_usage: self.estimate_memory_usage(data_size),
                initial_usage: 0,
                final_usage: 0,
                memory_leaked: 0,
            },
            duration: execution_time,
            issues: Vec::new(),
            recommendations: Vec::new(),
        }
    }

    /// Create error result for failed tests
    pub fn create_error_result(
        &self,
        test_name: &str,
        data_size: usize,
        error: InterpolateError,
        category: StressTestCategory,
    ) -> StressTestResult {
        StressTestResult {
            test_name: test_name.to_string(),
            category,
            input_characteristics: format!("Error case with {} data points", data_size),
            execution_time: Duration::from_millis(1),
            status: TestStatus::Error,
            performance: StressPerformanceMetrics::default(),
            error_info: Some(ErrorInfo {
                error_type: "InterpolateError".to_string(),
                error_message: error.to_string(),
                iteration: Some(0),
                data_size: Some(data_size),
                recovery_attempted: false,
                recovery_successful: false,
            }),
            memory_usage: MemoryUsageStats::default(),
            duration: Duration::from_millis(1),
            issues: vec![StressTestIssue {
                description: "Test failed due to error".to_string(),
                severity: IssueSeverity::High,
                production_impact: ProductionImpact::Major,
                suggested_fix: Some("Add error handling for this scenario".to_string()),
                iteration: Some(0),
            }],
            recommendations: vec![
                "Investigate error cause and add appropriate handling".to_string()
            ],
        }
    }

    /// Create panic result for tests that panicked
    pub fn create_panic_result(
        &self,
        test_name: &str,
        data_size: usize,
        category: StressTestCategory,
    ) -> StressTestResult {
        StressTestResult {
            test_name: test_name.to_string(),
            category,
            input_characteristics: format!("Panic case with {} data points", data_size),
            execution_time: Duration::from_millis(1),
            status: TestStatus::Failed,
            performance: StressPerformanceMetrics::default(),
            error_info: Some(ErrorInfo {
                error_type: "Panic".to_string(),
                error_message: "Test panicked during execution".to_string(),
                iteration: Some(0),
                data_size: Some(data_size),
                recovery_attempted: false,
                recovery_successful: false,
            }),
            memory_usage: MemoryUsageStats::default(),
            duration: Duration::from_millis(1),
            issues: vec![StressTestIssue {
                description: "Test caused panic - critical production risk".to_string(),
                severity: IssueSeverity::Critical,
                production_impact: ProductionImpact::Blocking,
                suggested_fix: Some("Add panic handling and input validation".to_string()),
                iteration: Some(0),
            }],
            recommendations: vec![
                "Critical: Fix panic condition before production deployment".to_string()
            ],
        }
    }

    /// Generate recommendations for a specific pathological pattern
    pub fn generate_recommendations_for_pattern(&self, pattern: &str) -> Vec<String> {
        match pattern {
            "constant" => vec![
                "Consider adding validation for constant data".to_string(),
                "Provide meaningful error messages for degenerate cases".to_string(),
            ],
            "duplicate_x" => vec![
                "Add input validation to detect duplicate x values".to_string(),
                "Consider data preprocessing options".to_string(),
            ],
            "extreme_y" => vec![
                "Add range checking for extreme values".to_string(),
                "Consider data normalization techniques".to_string(),
            ],
            "nan_inf" => vec![
                "Add robust NaN/infinity detection and handling".to_string(),
                "Provide clear error messages for invalid data".to_string(),
            ],
            "sparse" => vec![
                "Document minimum data density requirements".to_string(),
                "Consider alternative methods for sparse data".to_string(),
            ],
            "oscillatory" => vec![
                "Test with various interpolation methods".to_string(),
                "Consider adaptive sampling strategies".to_string(),
            ],
            "monotonic_extreme" => vec![
                "Verify numerical stability with extreme monotonic data".to_string(),
                "Consider logarithmic scaling for extreme ranges".to_string(),
            ],
            _ => vec!["Review numerical stability for this data pattern".to_string()],
        }
    }

    /// Enhanced numerical stability testing
    pub fn test_enhanced_numerical_stability(&mut self) -> InterpolateResult<()> {
        println!("Testing enhanced numerical stability...");

        // Test matrix conditioning
        let matrix_test = self.test_matrix_conditioning()?;
        self.results.push(matrix_test);

        // Test precision limits
        let precision_test = self.test_precision_limits()?;
        self.results.push(precision_test);

        // Test gradient stability
        let gradient_test = self.test_gradient_stability()?;
        self.results.push(gradient_test);

        // Test oscillatory stability
        let oscillatory_test = self.test_oscillatory_stability()?;
        self.results.push(oscillatory_test);

        // Test scaling stability
        let scaling_test = self.test_scaling_stability()?;
        self.results.push(scaling_test);

        Ok(())
    }

    /// Test matrix conditioning effects
    fn test_matrix_conditioning(&self) -> InterpolateResult<StressTestResult> {
        let start = Instant::now();

        // Create ill-conditioned data
        let (x, y) = create_duplicate_x_data(100)?;
        let x_query = Array1::linspace(T::from_f64(1.0).unwrap(), T::from_f64(9.0).unwrap(), 10);

        let result = crate::interp1d::cubic_interpolate(&x.view(), &y.view(), &x_query.view());
        let duration = start.elapsed();

        let mut issues = Vec::new();
        let status = match result {
            Ok(values) => {
                if values.iter().any(|v| !v.is_finite()) {
                    issues.push(StressTestIssue {
                        description: "Matrix conditioning produced non-finite results".to_string(),
                        severity: IssueSeverity::High,
                        production_impact: ProductionImpact::Major,
                        suggested_fix: Some("Improve matrix conditioning handling".to_string()),
                        iteration: None,
                    });
                    TestStatus::PassedWithWarnings
                } else {
                    TestStatus::Passed
                }
            }
            Err(_) => {
                issues.push(StressTestIssue {
                    description: "Matrix conditioning caused interpolation failure".to_string(),
                    severity: IssueSeverity::Medium,
                    production_impact: ProductionImpact::Minor,
                    suggested_fix: Some(
                        "Add graceful handling for ill-conditioned matrices".to_string(),
                    ),
                    iteration: None,
                });
                TestStatus::PassedWithWarnings
            }
        };

        Ok(StressTestResult {
            test_name: "matrix_conditioning".to_string(),
            category: StressTestCategory::NumericalStability,
            input_characteristics: "Ill-conditioned matrix data".to_string(),
            status,
            execution_time: duration,
            performance: self.calculate_performance_metrics(&[duration], None),
            error_info: None,
            memory_usage: MemoryUsageStats::default(),
            duration,
            issues,
            recommendations: vec![
                "Monitor matrix condition numbers in production".to_string(),
                "Consider regularization for ill-conditioned problems".to_string(),
            ],
        })
    }

    /// Test precision limits
    fn test_precision_limits(&self) -> InterpolateResult<StressTestResult> {
        let start = Instant::now();

        // Test with very small differences
        let (x, y) = create_edge_case_data(100, 0.0, f64::EPSILON * 100.0)?;
        let x_query = Array1::linspace(
            T::from_f64(f64::EPSILON).unwrap(),
            T::from_f64(f64::EPSILON * 50.0).unwrap(),
            10,
        );

        let result = crate::interp1d::linear_interpolate(&x.view(), &y.view(), &x_query.view());
        let duration = start.elapsed();

        let mut issues = Vec::new();
        let status = match result {
            Ok(values) => {
                if values.iter().any(|v| !v.is_finite()) {
                    issues.push(StressTestIssue {
                        description: "Precision limits caused numerical instability".to_string(),
                        severity: IssueSeverity::Medium,
                        production_impact: ProductionImpact::Minor,
                        suggested_fix: Some("Add epsilon-based tolerance checking".to_string()),
                        iteration: None,
                    });
                    TestStatus::PassedWithWarnings
                } else {
                    TestStatus::Passed
                }
            }
            Err(_) => TestStatus::Failed,
        };

        Ok(StressTestResult {
            test_name: "precision_limits".to_string(),
            category: StressTestCategory::NumericalStability,
            input_characteristics: "Near machine precision data".to_string(),
            status,
            execution_time: duration,
            performance: self.calculate_performance_metrics(&[duration], None),
            error_info: None,
            memory_usage: MemoryUsageStats::default(),
            duration,
            issues,
            recommendations: vec!["Document precision requirements for production data".to_string()],
        })
    }

    /// Test gradient stability
    fn test_gradient_stability(&self) -> InterpolateResult<StressTestResult> {
        let start = Instant::now();

        // Create data with steep gradients
        let (x, y) = create_rapid_change_data(1000)?;
        let x_query = Array1::linspace(T::from_f64(4.5).unwrap(), T::from_f64(5.5).unwrap(), 20);

        let result = crate::interp1d::cubic_interpolate(&x.view(), &y.view(), &x_query.view());
        let duration = start.elapsed();

        let mut issues = Vec::new();
        let status = match result {
            Ok(values) => {
                // Check for oscillations or overshoots
                let max_val = values.iter().fold(T::neg_infinity(), |acc, &x| acc.max(x));
                let min_val = values.iter().fold(T::infinity(), |acc, &x| acc.min(x));

                if max_val > T::from_f64(150.0).unwrap() || min_val < T::from_f64(-50.0).unwrap() {
                    issues.push(StressTestIssue {
                        description: "Steep gradients caused interpolation overshoots".to_string(),
                        severity: IssueSeverity::Medium,
                        production_impact: ProductionImpact::Minor,
                        suggested_fix: Some("Consider monotonic interpolation methods".to_string()),
                        iteration: None,
                    });
                    TestStatus::PassedWithWarnings
                } else {
                    TestStatus::Passed
                }
            }
            Err(_) => TestStatus::Failed,
        };

        Ok(StressTestResult {
            test_name: "gradient_stability".to_string(),
            category: StressTestCategory::NumericalStability,
            input_characteristics: "Data with steep gradients".to_string(),
            status,
            execution_time: duration,
            performance: self.calculate_performance_metrics(&[duration], None),
            error_info: None,
            memory_usage: MemoryUsageStats::default(),
            duration,
            issues,
            recommendations: vec![
                "Monitor for interpolation overshoots with steep data".to_string()
            ],
        })
    }

    /// Test oscillatory stability
    fn test_oscillatory_stability(&self) -> InterpolateResult<StressTestResult> {
        let start = Instant::now();

        let (x, y) = create_oscillatory_data(1000)?;
        let x_query = Array1::linspace(T::from_f64(1.0).unwrap(), T::from_f64(9.0).unwrap(), 100);

        let result = crate::interp1d::cubic_interpolate(&x.view(), &y.view(), &x_query.view());
        let duration = start.elapsed();

        let status = match result {
            Ok(_) => TestStatus::Passed,
            Err(_) => TestStatus::Failed,
        };

        Ok(StressTestResult {
            test_name: "oscillatory_stability".to_string(),
            category: StressTestCategory::NumericalStability,
            input_characteristics: "Highly oscillatory data".to_string(),
            status,
            execution_time: duration,
            performance: self.calculate_performance_metrics(&[duration], None),
            error_info: None,
            memory_usage: MemoryUsageStats::default(),
            duration,
            issues: Vec::new(),
            recommendations: vec!["Consider adaptive sampling for oscillatory data".to_string()],
        })
    }

    /// Test scaling stability
    fn test_scaling_stability(&self) -> InterpolateResult<StressTestResult> {
        let start = Instant::now();

        let (x, y) = create_extreme_y_data(100)?;
        let x_query = Array1::linspace(T::from_f64(1.0).unwrap(), T::from_f64(9.0).unwrap(), 10);

        let result = crate::interp1d::linear_interpolate(&x.view(), &y.view(), &x_query.view());
        let duration = start.elapsed();

        let status = match result {
            Ok(_) => TestStatus::Passed,
            Err(_) => TestStatus::Failed,
        };

        Ok(StressTestResult {
            test_name: "scaling_stability".to_string(),
            category: StressTestCategory::NumericalStability,
            input_characteristics: "Extreme value scaling".to_string(),
            status,
            execution_time: duration,
            performance: self.calculate_performance_metrics(&[duration], None),
            error_info: None,
            memory_usage: MemoryUsageStats::default(),
            duration,
            issues: Vec::new(),
            recommendations: vec!["Consider data normalization for extreme ranges".to_string()],
        })
    }

    /// Test error message clarity
    pub fn test_error_message_clarity(&mut self) -> InterpolateResult<()> {
        println!("Testing error message clarity...");

        // Test empty data error messages
        let empty_test = self.test_empty_data_error_messages()?;
        self.results.push(empty_test);

        // Test dimension mismatch errors
        let dimension_test = self.test_dimension_mismatch_errors()?;
        self.results.push(dimension_test);

        // Test parameter validation errors
        let param_test = self.test_parameter_validation_errors()?;
        self.results.push(param_test);

        // Test numerical error messages
        let numerical_test = self.test_numerical_error_messages()?;
        self.results.push(numerical_test);

        Ok(())
    }

    /// Test empty data error messages
    fn test_empty_data_error_messages(&self) -> InterpolateResult<StressTestResult> {
        let (x, y) = create_empty_data()?;
        let x_query = Array1::from_vec(vec![T::from_f64(1.0).unwrap()]);

        let result = crate::interp1d::linear_interpolate(&x.view(), &y.view(), &x_query.view());

        let status = match result {
            Err(e) => {
                let error_msg = e.to_string();
                if error_msg.contains("empty") || error_msg.contains("zero") {
                    TestStatus::Passed
                } else {
                    TestStatus::PassedWithWarnings
                }
            }
            Ok(_) => TestStatus::Failed,
        };

        Ok(StressTestResult {
            test_name: "empty_data_error_clarity".to_string(),
            category: StressTestCategory::ErrorHandling,
            input_characteristics: "Empty data arrays".to_string(),
            status,
            execution_time: Duration::from_millis(1),
            performance: StressPerformanceMetrics::default(),
            error_info: None,
            memory_usage: MemoryUsageStats::default(),
            duration: Duration::from_millis(1),
            issues: Vec::new(),
            recommendations: vec!["Ensure error messages are clear and actionable".to_string()],
        })
    }

    /// Test dimension mismatch errors
    fn test_dimension_mismatch_errors(&self) -> InterpolateResult<StressTestResult> {
        let (x, y) = create_mismatched_data()?;
        let x_query = Array1::from_vec(vec![T::from_f64(1.0).unwrap()]);

        let result = crate::interp1d::linear_interpolate(&x.view(), &y.view(), &x_query.view());

        let status = match result {
            Err(e) => {
                let error_msg = e.to_string();
                if error_msg.contains("length") || error_msg.contains("dimension") {
                    TestStatus::Passed
                } else {
                    TestStatus::PassedWithWarnings
                }
            }
            Ok(_) => TestStatus::Failed,
        };

        Ok(StressTestResult {
            test_name: "dimension_mismatch_error_clarity".to_string(),
            category: StressTestCategory::ErrorHandling,
            input_characteristics: "Mismatched array dimensions".to_string(),
            status,
            execution_time: Duration::from_millis(1),
            performance: StressPerformanceMetrics::default(),
            error_info: None,
            memory_usage: MemoryUsageStats::default(),
            duration: Duration::from_millis(1),
            issues: Vec::new(),
            recommendations: vec![
                "Provide specific dimension information in error messages".to_string()
            ],
        })
    }

    /// Test parameter validation errors
    fn test_parameter_validation_errors(&self) -> InterpolateResult<StressTestResult> {
        // This would test parameter validation, but our current API doesn't have many parameters
        // to validate. This is a placeholder for future parameter validation tests.

        Ok(StressTestResult {
            test_name: "parameter_validation_error_clarity".to_string(),
            category: StressTestCategory::ErrorHandling,
            input_characteristics: "Invalid parameters".to_string(),
            status: TestStatus::Skipped,
            execution_time: Duration::from_millis(1),
            performance: StressPerformanceMetrics::default(),
            error_info: None,
            memory_usage: MemoryUsageStats::default(),
            duration: Duration::from_millis(1),
            issues: Vec::new(),
            recommendations: vec!["Add parameter validation as API expands".to_string()],
        })
    }

    /// Test numerical error messages
    fn test_numerical_error_messages(&self) -> InterpolateResult<StressTestResult> {
        let (x, y) = create_nan_inf_data(100)?;
        let x_query = Array1::linspace(T::from_f64(1.0).unwrap(), T::from_f64(9.0).unwrap(), 10);

        let result = crate::interp1d::linear_interpolate(&x.view(), &y.view(), &x_query.view());

        let status = match result {
            Err(e) => {
                let error_msg = e.to_string();
                if error_msg.contains("NaN")
                    || error_msg.contains("infinite")
                    || error_msg.contains("numerical")
                {
                    TestStatus::Passed
                } else {
                    TestStatus::PassedWithWarnings
                }
            }
            Ok(_) => TestStatus::PassedWithWarnings, // Should probably detect NaN/inf
        };

        Ok(StressTestResult {
            test_name: "numerical_error_clarity".to_string(),
            category: StressTestCategory::ErrorHandling,
            input_characteristics: "Data with NaN/infinite values".to_string(),
            status,
            execution_time: Duration::from_millis(1),
            performance: StressPerformanceMetrics::default(),
            error_info: None,
            memory_usage: MemoryUsageStats::default(),
            duration: Duration::from_millis(1),
            issues: Vec::new(),
            recommendations: vec!["Improve detection and reporting of numerical issues".to_string()],
        })
    }

    /// Test resource exhaustion recovery
    pub fn test_resource_exhaustion_recovery(&mut self) -> InterpolateResult<()> {
        println!("Testing resource exhaustion recovery...");

        // Test memory exhaustion scenarios
        let large_sizes = [500_000, 1_000_000];

        for (i, &size) in large_sizes.iter().enumerate() {
            let test_result = match std::panic::catch_unwind(|| create_large_test_data(size)) {
                Ok(Ok((x, y))) => {
                    let start = Instant::now();

                    // Try to process the large dataset
                    let x_query = Array1::linspace(
                        T::from_f64(1.0).unwrap(),
                        T::from_f64(9.0).unwrap(),
                        1000,
                    );

                    match crate::interp1d::linear_interpolate(&x.view(), &y.view(), &x_query.view())
                    {
                        Ok(_) => self.create_success_result(
                            &format!("resource_exhaustion_{}", i),
                            size,
                            start.elapsed(),
                            StressTestCategory::ResourceExhaustion,
                        ),
                        Err(e) => self.create_error_result(
                            &format!("resource_exhaustion_{}", i),
                            size,
                            e,
                            StressTestCategory::ResourceExhaustion,
                        ),
                    }
                }
                Ok(Err(e)) => self.create_error_result(
                    &format!("resource_exhaustion_{}", i),
                    size,
                    e,
                    StressTestCategory::ResourceExhaustion,
                ),
                Err(_) => self.create_panic_result(
                    &format!("resource_exhaustion_{}", i),
                    size,
                    StressTestCategory::ResourceExhaustion,
                ),
            };

            self.results.push(test_result);
        }

        Ok(())
    }
}
