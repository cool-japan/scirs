//! Test execution engine for production stress testing
//!
//! This module contains the core execution logic for running various types of stress tests.

use super::generators::*;
use super::types::*;
use crate::error::{InterpolateError, InterpolateResult};
use crate::traits::InterpolationFloat;
use ndarray::{Array1, ArrayView1};
use std::collections::HashMap;
use std::time::{Duration, Instant};

impl<T: InterpolationFloat + std::panic::RefUnwindSafe> ProductionStressTester<T> {
    /// Run comprehensive production stress tests
    pub fn run_comprehensive_stress_tests(&mut self) -> InterpolateResult<StressTestReport> {
        println!("Starting comprehensive production stress testing...");

        // Establish baseline performance if not available
        if self.baseline_performance.is_none() {
            self.establish_baseline_performance()?;
        }

        // 1. Extreme data size testing
        self.test_extreme_data_sizes()?;

        // 2. Pathological data patterns
        self.test_pathological_data()?;

        // 3. Numerical edge cases
        self.test_numerical_edge_cases()?;

        // 4. Memory pressure testing
        self.test_memory_pressure()?;

        // 5. Error handling validation
        self.test_error_handling()?;

        // 6. Performance under stress
        self.test_performance_under_stress()?;

        // 7. Enhanced numerical stability analysis
        self.test_enhanced_numerical_stability()?;

        // 8. Critical error message clarity validation
        self.test_error_message_clarity()?;

        // 9. Resource exhaustion recovery testing
        self.test_resource_exhaustion_recovery()?;

        // Generate comprehensive report
        let report = self.generate_stress_test_report();

        println!("Production stress testing completed.");
        Ok(report)
    }

    /// Establish baseline performance for comparison
    fn establish_baseline_performance(&mut self) -> InterpolateResult<()> {
        println!("Establishing baseline performance...");

        let mut total_time = Duration::from_millis(0);
        let mut total_memory = 0u64;
        let mut test_count = 0;

        // Test standard interpolation methods with normal data
        let normal_sizes = vec![100, 1_000, 10_000];

        for size in normal_sizes {
            let (x, y) = create_large_test_data(size)?;
            let x_query = Array1::linspace(
                T::from_f64(0.5).unwrap(),
                T::from_f64(9.5).unwrap(),
                size / 10,
            );

            // Baseline linear interpolation
            let start = Instant::now();
            let _ = crate::interp1d::linear_interpolate(&x.view(), &y.view(), &x_query.view())?;
            let duration = start.elapsed();
            total_time += duration;
            total_memory += self.estimate_memory_usage(size);
            test_count += 1;

            // Baseline cubic interpolation
            let start = Instant::now();
            let _ = crate::interp1d::cubic_interpolate(&x.view(), &y.view(), &x_query.view())?;
            let duration = start.elapsed();
            total_time += duration;
            total_memory += self.estimate_memory_usage(size);
            test_count += 1;
        }

        self.baseline_performance = Some(BaselinePerformance {
            execution_time: total_time / (test_count as u32),
            memory_usage: total_memory / (test_count as u64),
            throughput: (test_count as f64) / total_time.as_secs_f64(),
        });

        Ok(())
    }

    /// Test with extreme data sizes
    fn test_extreme_data_sizes(&mut self) -> InterpolateResult<()> {
        println!("Testing extreme data sizes...");

        let extreme_sizes = vec![100_000, 500_000, self.config.max_data_size];

        for size in extreme_sizes {
            println!("  Testing size: {}", size);

            let test_result = match std::panic::catch_unwind(|| create_large_test_data(size)) {
                Ok(Ok((x, y))) => {
                    let start = Instant::now();
                    let x_query = Array1::linspace(
                        T::from_f64(1.0).unwrap(),
                        T::from_f64(9.0).unwrap(),
                        (size / 100).min(1000),
                    );

                    let result =
                        crate::interp1d::linear_interpolate(&x.view(), &y.view(), &x_query.view());
                    let duration = start.elapsed();

                    match result {
                        Ok(_) => self.create_success_result(
                            &format!("extreme_size_{}", size),
                            size,
                            duration,
                            StressTestCategory::ExtremeDataSize,
                        ),
                        Err(e) => self.create_error_result(
                            &format!("extreme_size_{}", size),
                            size,
                            e,
                            StressTestCategory::ExtremeDataSize,
                        ),
                    }
                }
                Ok(Err(e)) => self.create_error_result(
                    &format!("extreme_size_{}", size),
                    size,
                    e,
                    StressTestCategory::ExtremeDataSize,
                ),
                Err(_) => self.create_panic_result(
                    &format!("extreme_size_{}", size),
                    size,
                    StressTestCategory::ExtremeDataSize,
                ),
            };

            self.results.push(test_result);
        }

        Ok(())
    }

    /// Test pathological data patterns
    fn test_pathological_data(&mut self) -> InterpolateResult<()> {
        println!("Testing pathological data patterns...");

        let pathological_patterns = get_pathological_data_types();

        for pattern in pathological_patterns {
            println!("  Testing pattern: {}", pattern);

            let test_result = match generate_test_data(pattern, 1000) {
                Ok((x, y)) => {
                    let start = Instant::now();
                    let x_query =
                        Array1::linspace(T::from_f64(1.0).unwrap(), T::from_f64(9.0).unwrap(), 10);

                    // Test multiple interpolation methods
                    let methods = vec!["linear", "cubic"];
                    let mut issues = Vec::new();
                    let mut method_results = Vec::new();

                    for method in methods {
                        let method_result = match method {
                            "linear" => crate::interp1d::linear_interpolate(
                                &x.view(),
                                &y.view(),
                                &x_query.view(),
                            ),
                            "cubic" => crate::interp1d::cubic_interpolate(
                                &x.view(),
                                &y.view(),
                                &x_query.view(),
                            ),
                            _ => continue,
                        };

                        match method_result {
                            Ok(result) => {
                                // Check result validity
                                if result.iter().any(|v| !v.is_finite()) {
                                    issues.push(StressTestIssue {
                                        description: format!(
                                            "{} method produced non-finite values",
                                            method
                                        ),
                                        severity: IssueSeverity::High,
                                        production_impact: ProductionImpact::Major,
                                        suggested_fix: Some(
                                            "Add input validation and result checking".to_string(),
                                        ),
                                        iteration: None,
                                    });
                                }
                                method_results.push((method, true));
                            }
                            Err(_) => {
                                method_results.push((method, false));
                            }
                        }
                    }

                    let duration = start.elapsed();
                    let status = if method_results.iter().any(|(_, success)| *success) {
                        if issues.is_empty() {
                            TestStatus::Passed
                        } else {
                            TestStatus::PassedWithWarnings
                        }
                    } else {
                        TestStatus::Failed
                    };

                    StressTestResult {
                        test_name: format!("pathological_{}", pattern),
                        category: StressTestCategory::PathologicalData,
                        input_characteristics: format!("Pathological pattern: {}", pattern),
                        status,
                        execution_time: duration,
                        performance: self.calculate_performance_metrics(&[duration], None),
                        error_info: None,
                        memory_usage: MemoryUsageStats {
                            peak_usage: self.estimate_memory_usage(1000),
                            average_usage: self.estimate_memory_usage(1000),
                            initial_usage: 0,
                            final_usage: 0,
                            memory_leaked: 0,
                        },
                        duration,
                        issues,
                        recommendations: self.generate_recommendations_for_pattern(pattern),
                    }
                }
                Err(e) => self.create_error_result(
                    &format!("pathological_{}", pattern),
                    1000,
                    e,
                    StressTestCategory::PathologicalData,
                ),
            };

            self.results.push(test_result);
        }

        Ok(())
    }

    /// Test numerical edge cases
    fn test_numerical_edge_cases(&mut self) -> InterpolateResult<()> {
        println!("Testing numerical edge cases...");

        let edge_cases = [
            (1e-15, 1e-10),      // Very small values
            (-1e15, 1e15),       // Very large range
            (0.0, f64::EPSILON), // Near-zero range
        ];

        for (i, (min_val, max_val)) in edge_cases.iter().enumerate() {
            println!("  Testing edge case {}: [{}, {}]", i + 1, min_val, max_val);

            let test_result = match create_edge_case_data(1000, *min_val, *max_val) {
                Ok((x, y)) => {
                    let start = Instant::now();
                    let x_query = Array1::linspace(
                        T::from_f64(*min_val + (*max_val - *min_val) * 0.1).unwrap(),
                        T::from_f64(*min_val + (*max_val - *min_val) * 0.9).unwrap(),
                        10,
                    );

                    let result =
                        crate::interp1d::linear_interpolate(&x.view(), &y.view(), &x_query.view());
                    let duration = start.elapsed();

                    match result {
                        Ok(interpolated) => {
                            let mut issues = Vec::new();

                            // Check for numerical issues
                            if interpolated.iter().any(|v| !v.is_finite()) {
                                issues.push(StressTestIssue {
                                    description: "Interpolation produced non-finite values"
                                        .to_string(),
                                    severity: IssueSeverity::High,
                                    production_impact: ProductionImpact::Major,
                                    suggested_fix: Some(
                                        "Improve numerical stability handling".to_string(),
                                    ),
                                    iteration: None,
                                });
                            }

                            StressTestResult {
                                test_name: format!("edge_case_{}", i),
                                category: StressTestCategory::NumericalEdgeCases,
                                input_characteristics: format!(
                                    "Range: [{:.2e}, {:.2e}]",
                                    min_val, max_val
                                ),
                                status: if issues.is_empty() {
                                    TestStatus::Passed
                                } else {
                                    TestStatus::PassedWithWarnings
                                },
                                execution_time: duration,
                                performance: self.calculate_performance_metrics(&[duration], None),
                                error_info: None,
                                memory_usage: MemoryUsageStats::default(),
                                duration,
                                issues,
                                recommendations: vec![
                                    "Monitor numerical precision in production".to_string()
                                ],
                            }
                        }
                        Err(e) => self.create_error_result(
                            &format!("edge_case_{}", i),
                            1000,
                            e,
                            StressTestCategory::NumericalEdgeCases,
                        ),
                    }
                }
                Err(e) => self.create_error_result(
                    &format!("edge_case_{}", i),
                    1000,
                    e,
                    StressTestCategory::NumericalEdgeCases,
                ),
            };

            self.results.push(test_result);
        }

        Ok(())
    }

    /// Test memory pressure scenarios
    fn test_memory_pressure(&mut self) -> InterpolateResult<()> {
        println!("Testing memory pressure scenarios...");

        let memory_test_sizes = [50_000, 100_000, 250_000];

        for (i, &data_size) in memory_test_sizes.iter().enumerate() {
            println!("  Testing memory pressure with {} data points", data_size);

            let test_result = match create_large_test_data(data_size) {
                Ok((x, y)) => {
                    let initial_memory = self.estimate_memory_usage(0);
                    let start = Instant::now();

                    let x_query = Array1::linspace(
                        T::from_f64(1.0).unwrap(),
                        T::from_f64(9.0).unwrap(),
                        data_size / 100,
                    );

                    let result =
                        crate::interp1d::linear_interpolate(&x.view(), &y.view(), &x_query.view());
                    let duration = start.elapsed();
                    let final_memory = self.estimate_memory_usage(data_size);

                    match result {
                        Ok(_) => {
                            let memory_growth = final_memory.saturating_sub(initial_memory) as i64;
                            let mut result = StressTestResult {
                                test_name: format!("memory_pressure_{}", i),
                                category: StressTestCategory::MemoryPressure,
                                input_characteristics: format!("{} data points", data_size),
                                status: TestStatus::Passed,
                                execution_time: duration,
                                performance: self.calculate_performance_metrics(&[duration], None),
                                error_info: None,
                                memory_usage: MemoryUsageStats {
                                    peak_usage: final_memory,
                                    average_usage: (initial_memory + final_memory) / 2,
                                    initial_usage: initial_memory,
                                    final_usage: final_memory,
                                    memory_leaked: memory_growth,
                                },
                                duration,
                                issues: Vec::new(),
                                recommendations: vec![
                                    "Monitor memory usage in production".to_string()
                                ],
                            };

                            // Check for memory leaks
                            if memory_growth > (data_size * std::mem::size_of::<T>() * 10) as i64 {
                                result.issues.push(StressTestIssue {
                                    description: "Potential memory leak detected".to_string(),
                                    severity: IssueSeverity::High,
                                    production_impact: ProductionImpact::Major,
                                    suggested_fix: Some(
                                        "Investigate memory allocation patterns".to_string(),
                                    ),
                                    iteration: None,
                                });
                            }

                            result
                        }
                        Err(e) => self.create_error_result(
                            &format!("memory_pressure_{}", i),
                            data_size,
                            e,
                            StressTestCategory::MemoryPressure,
                        ),
                    }
                }
                Err(e) => self.create_error_result(
                    &format!("memory_pressure_{}", i),
                    data_size,
                    e,
                    StressTestCategory::MemoryPressure,
                ),
            };

            self.results.push(test_result);
        }

        Ok(())
    }

    /// Test error handling robustness
    fn test_error_handling(&mut self) -> InterpolateResult<()> {
        println!("Testing error handling robustness...");

        let error_test_types = get_error_test_data_types();

        for error_type in error_test_types {
            println!("  Testing error scenario: {}", error_type);

            let test_result = match generate_test_data(error_type, 0) {
                Ok((x, y)) => {
                    // For error scenarios, we expect graceful failures
                    let mut issues = Vec::new();
                    let mut error_count = 0;
                    let mut total_tests = 0;

                    // Test each interpolation method
                    let methods = vec!["linear", "cubic"];
                    for method in methods {
                        total_tests += 1;
                        let query_x = Array1::from_vec(vec![T::from_f64(2.5).unwrap()]);

                        let result = match method {
                            "linear" => crate::interp1d::linear_interpolate(
                                &x.view(),
                                &y.view(),
                                &query_x.view(),
                            ),
                            "cubic" => crate::interp1d::cubic_interpolate(
                                &x.view(),
                                &y.view(),
                                &query_x.view(),
                            ),
                            _ => continue,
                        };

                        match result {
                            Ok(_) => {
                                // For most error scenarios, we shouldn't succeed
                                if error_type != "single_point" {
                                    issues.push(StressTestIssue {
                                        description: format!(
                                            "{} method should have failed with {}",
                                            method, error_type
                                        ),
                                        severity: IssueSeverity::Medium,
                                        production_impact: ProductionImpact::Minor,
                                        suggested_fix: Some(
                                            "Add more robust input validation".to_string(),
                                        ),
                                        iteration: None,
                                    });
                                }
                            }
                            Err(_) => {
                                error_count += 1;
                                // This is expected for most error scenarios
                            }
                        }
                    }

                    let status = if error_count == total_tests && error_type != "single_point" {
                        TestStatus::Passed // Expected to fail
                    } else if issues.is_empty() {
                        TestStatus::Passed
                    } else {
                        TestStatus::PassedWithWarnings
                    };

                    StressTestResult {
                        test_name: format!("error_handling_{}", error_type),
                        category: StressTestCategory::ErrorHandling,
                        input_characteristics: format!("Error scenario: {}", error_type),
                        status,
                        execution_time: Duration::from_millis(1),
                        performance: StressPerformanceMetrics::default(),
                        error_info: None,
                        memory_usage: MemoryUsageStats::default(),
                        duration: Duration::from_millis(1),
                        issues,
                        recommendations: vec![
                            "Ensure consistent error handling across all methods".to_string(),
                        ],
                    }
                }
                Err(e) => self.create_error_result(
                    &format!("error_handling_{}", error_type),
                    0,
                    e,
                    StressTestCategory::ErrorHandling,
                ),
            };

            self.results.push(test_result);
        }

        Ok(())
    }

    /// Test performance under stress
    fn test_performance_under_stress(&mut self) -> InterpolateResult<()> {
        println!("Testing performance under stress...");

        let stress_sizes = [10_000, 50_000, 100_000];

        for (i, &size) in stress_sizes.iter().enumerate() {
            println!("  Testing performance stress with {} data points", size);

            let stressed_size = size * self.config.stress_iterations / 100;

            let test_result = match create_large_test_data(stressed_size) {
                Ok((x, y)) => {
                    let mut execution_times = Vec::new();
                    let start_time = Instant::now();

                    // Run multiple iterations
                    for _iteration in 0..5 {
                        let x_query = Array1::linspace(
                            T::from_f64(1.0).unwrap(),
                            T::from_f64(9.0).unwrap(),
                            size / 100,
                        );

                        let iter_start = Instant::now();
                        let _ = crate::interp1d::linear_interpolate(
                            &x.view(),
                            &y.view(),
                            &x_query.view(),
                        )?;
                        execution_times.push(iter_start.elapsed());
                    }

                    let total_duration = start_time.elapsed();
                    let performance = self.calculate_performance_metrics(
                        &execution_times,
                        self.baseline_performance.as_ref(),
                    );

                    let mut issues = Vec::new();
                    if let Some(degradation) = performance.degradation_factor {
                        if degradation > self.config.max_performance_degradation {
                            issues.push(StressTestIssue {
                                description: format!("Performance degraded by {:.1}x", degradation),
                                severity: IssueSeverity::High,
                                production_impact: ProductionImpact::Major,
                                suggested_fix: Some(
                                    "Optimize performance for large datasets".to_string(),
                                ),
                                iteration: None,
                            });
                        }
                    }

                    StressTestResult {
                        test_name: format!("performance_stress_{}", i),
                        category: StressTestCategory::PerformanceStress,
                        input_characteristics: format!(
                            "{} data points, {} iterations",
                            stressed_size, 5
                        ),
                        status: if issues.is_empty() {
                            TestStatus::Passed
                        } else {
                            TestStatus::PassedWithWarnings
                        },
                        execution_time: total_duration,
                        performance,
                        error_info: None,
                        memory_usage: MemoryUsageStats {
                            peak_usage: self.estimate_memory_usage(stressed_size),
                            average_usage: self.estimate_memory_usage(stressed_size),
                            initial_usage: 0,
                            final_usage: 0,
                            memory_leaked: 0,
                        },
                        duration: total_duration,
                        issues,
                        recommendations: vec![
                            "Monitor performance metrics in production".to_string()
                        ],
                    }
                }
                Err(e) => self.create_error_result(
                    &format!("performance_stress_{}", i),
                    stressed_size,
                    e,
                    StressTestCategory::PerformanceStress,
                ),
            };

            self.results.push(test_result);
        }

        Ok(())
    }
}
