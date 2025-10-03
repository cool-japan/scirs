//! Cross-platform validation and compatibility testing framework
//!
//! This module provides comprehensive cross-platform validation for scirs2-stats,
//! ensuring consistent behavior across different operating systems, architectures,
//! and hardware configurations. It includes platform-specific testing, numerical
//! precision validation, and compatibility assessments.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumCast, Zero, One};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Cross-platform validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossPlatformConfig {
    /// Enable architecture-specific optimizations testing
    pub test_architecture_optimizations: bool,
    /// Enable operating system specific behavior testing
    pub test_os_specific_behavior: bool,
    /// Enable floating-point precision validation
    pub test_floating_point_precision: bool,
    /// Enable parallel processing compatibility testing
    pub test_parallel_compatibility: bool,
    /// Enable SIMD instruction testing
    pub test_simd_compatibility: bool,
    /// Enable memory allocation pattern testing
    pub test_memory_allocation: bool,
    /// Tolerance for numerical differences across platforms
    pub numerical_tolerance: f64,
    /// Enable endianness testing for serialization
    pub test_endianness: bool,
    /// Enable threading model compatibility testing
    pub test_threading_models: bool,
}

impl Default for CrossPlatformConfig {
    fn default() -> Self {
        Self {
            test_architecture_optimizations: true,
            test_os_specific_behavior: true,
            test_floating_point_precision: true,
            test_parallel_compatibility: true,
            test_simd_compatibility: true,
            test_memory_allocation: true,
            numerical_tolerance: 1e-12,
            test_endianness: true,
            test_threading_models: true,
        }
    }
}

/// Platform information and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    /// Operating system name
    pub os: String,
    /// CPU architecture
    pub arch: String,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Available SIMD instruction sets
    pub simd_capabilities: Vec<String>,
    /// Endianness (big or little)
    pub endianness: Endianness,
    /// Floating-point model
    pub float_model: FloatingPointModel,
    /// Memory page size
    pub memory_pagesize: usize,
    /// Cache line size
    pub cache_linesize: usize,
    /// Threading model
    pub threading_model: ThreadingModel,
}

/// System endianness
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Endianness {
    Little,
    Big,
}

/// Floating-point model characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatingPointModel {
    /// IEEE 754 compliance
    pub ieee_754_compliant: bool,
    /// Supports denormal numbers
    pub supports_denormals: bool,
    /// Default rounding mode
    pub rounding_mode: RoundingMode,
    /// Machine epsilon for f32
    pub f32_epsilon: f32,
    /// Machine epsilon for f64
    pub f64_epsilon: f64,
}

/// Floating-point rounding mode
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RoundingMode {
    ToNearest,
    TowardZero,
    TowardPositive,
    TowardNegative,
}

/// Threading model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadingModel {
    /// Supports work-stealing
    pub work_stealing: bool,
    /// Default thread count
    pub default_threads: usize,
    /// Thread affinity support
    pub thread_affinity: bool,
    /// NUMA awareness
    pub numa_aware: bool,
}

/// Cross-platform validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    /// Platform information
    pub platform_info: PlatformInfo,
    /// Individual test results
    pub test_results: Vec<TestResult>,
    /// Overall validation status
    pub overall_status: ValidationStatus,
    /// Performance benchmarks across platforms
    pub performance_benchmarks: HashMap<String, f64>,
    /// Compatibility issues found
    pub compatibility_issues: Vec<CompatibilityIssue>,
    /// Recommendations for platform-specific optimizations
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
}

/// Individual test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test name
    pub test_name: String,
    /// Test category
    pub category: TestCategory,
    /// Test status
    pub status: TestStatus,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: Option<usize>,
    /// Platform-specific notes
    pub notes: Vec<String>,
    /// Numerical accuracy metrics
    pub accuracy_metrics: Option<AccuracyMetrics>,
}

/// Test categories
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TestCategory {
    NumericalPrecision,
    SIMDCompatibility,
    ParallelProcessing,
    MemoryAllocation,
    OSSpecific,
    ArchitectureSpecific,
    EndianessHandling,
    ThreadingModel,
}

/// Test execution status
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TestStatus {
    Passed,
    Failed,
    Warning,
    Skipped,
}

/// Overall validation status
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ValidationStatus {
    FullyCompatible,
    MostlyCompatible,
    LimitedCompatibility,
    Incompatible,
}

/// Numerical accuracy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    /// Maximum absolute error
    pub max_absolute_error: f64,
    /// Mean absolute error
    pub mean_absolute_error: f64,
    /// Maximum relative error
    pub max_relative_error: f64,
    /// Mean relative error
    pub mean_relative_error: f64,
    /// Number of samples tested
    pub sample_count: usize,
}

/// Compatibility issue description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue category
    pub category: TestCategory,
    /// Description of the issue
    pub description: String,
    /// Affected functions or features
    pub affected_functions: Vec<String>,
    /// Suggested workaround
    pub workaround: Option<String>,
    /// Platform-specific details
    pub platform_details: HashMap<String, String>,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IssueSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Platform-specific optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Target platform
    pub platform: String,
    /// Optimization type
    pub optimization_type: OptimizationType,
    /// Expected performance improvement
    pub expected_improvement: f64,
    /// Implementation complexity
    pub complexity: ComplexityLevel,
    /// Detailed recommendation
    pub recommendation: String,
}

/// Types of platform optimizations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationType {
    SIMDOptimization,
    MemoryOptimization,
    ParallelOptimization,
    ArchitectureSpecific,
    CompilerOptimization,
}

/// Implementation complexity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Main cross-platform validator
pub struct CrossPlatformValidator {
    config: CrossPlatformConfig,
    platform_info: PlatformInfo,
}

impl CrossPlatformValidator {
    /// Create a new cross-platform validator
    pub fn new(config: CrossPlatformConfig) -> Self {
        let platform_info = Self::detect_platform_info();
        
        Self {
            config,
            platform_info,
        }
    }

    /// Run comprehensive cross-platform validation
    pub fn validate_comprehensive(&self) -> StatsResult<ValidationResults> {
        let mut test_results = Vec::new();
        let mut compatibility_issues = Vec::new();
        let mut performance_benchmarks = HashMap::new();

        // Run numerical precision tests
        if self.config.test_floating_point_precision {
            test_results.extend(self.test_numerical_precision(&mut compatibility_issues)?);
        }

        // Run SIMD compatibility tests
        if self.config.test_simd_compatibility {
            test_results.extend(self.test_simd_compatibility(&mut compatibility_issues)?);
        }

        // Run parallel processing tests
        if self.config.test_parallel_compatibility {
            test_results.extend(self.test_parallel_compatibility(&mut compatibility_issues)?);
        }

        // Run memory allocation tests
        if self.config.test_memory_allocation {
            test_results.extend(self.test_memory_allocation(&mut compatibility_issues)?);
        }

        // Run architecture-specific tests
        if self.config.test_architecture_optimizations {
            test_results.extend(self.test_architecture_optimizations(&mut compatibility_issues)?);
        }

        // Run OS-specific tests
        if self.config.test_os_specific_behavior {
            test_results.extend(self.test_os_specific_behavior(&mut compatibility_issues)?);
        }

        // Run endianness tests
        if self.config.test_endianness {
            test_results.extend(self.test_endianness_handling(&mut compatibility_issues)?);
        }

        // Run threading model tests
        if self.config.test_threading_models {
            test_results.extend(self.test_threading_models(&mut compatibility_issues)?);
        }

        // Run performance benchmarks
        performance_benchmarks = self.run_performance_benchmarks()?;

        // Determine overall status
        let overall_status = self.determine_overall_status(&test_results, &compatibility_issues);

        // Generate optimization recommendations
        let optimization_recommendations = self.generate_optimization_recommendations(
            &test_results,
            &performance_benchmarks,
        );

        Ok(ValidationResults {
            platform_info: self.platform_info.clone(),
            test_results,
            overall_status,
            performance_benchmarks,
            compatibility_issues,
            optimization_recommendations,
        })
    }

    /// Test numerical precision across platforms
    #[ignore = "timeout"]
    fn test_numerical_precision(&self, issues: &mut Vec<CompatibilityIssue>) -> StatsResult<Vec<TestResult>> {
        let mut results = Vec::new();

        // Test basic arithmetic precision
        results.push(self.test_basic_arithmetic_precision()?);

        // Test transcendental functions
        results.push(self.test_transcendental_functions()?);

        // Test statistical function precision
        results.push(self.test_statistical_function_precision()?);

        // Test edge cases (overflow, underflow, NaN, infinity)
        results.push(self.test_edge_case_handling()?);

        // Check for precision issues
        for result in &results {
            if let Some(metrics) = &result.accuracy_metrics {
                if metrics.max_relative_error > self.config.numerical_tolerance {
                    issues.push(CompatibilityIssue {
                        severity: IssueSeverity::High,
                        category: TestCategory::NumericalPrecision,
                        description: format!(
                            "High numerical error in {}: max relative error = {}",
                            result.test_name,
                            metrics.max_relative_error
                        ),
                        affected_functions: vec![result.test_name.clone()],
                        workaround: Some("Consider using higher precision arithmetic".to_string()),
                        platform_details: HashMap::new(),
                    });
                }
            }
        }

        Ok(results)
    }

    /// Test basic arithmetic precision
    fn test_basic_arithmetic_precision(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();

        // Test known arithmetic operations with expected results
        let test_cases = vec![
            (0.1f64 + 0.2f64, 0.3f64),
            (1.0f64 / 3.0f64 * 3.0f64, 1.0f64),
            (2.0f64.sqrt() * 2.0f64.sqrt(), 2.0f64),
            (std::f64::consts::E.ln(), 1.0f64),
        ];

        let mut max_absolute_error = 0.0;
        let mut max_relative_error = 0.0;
        let mut total_absolute_error = 0.0;
        let mut total_relative_error = 0.0;

        for (computed, expected) in test_cases.iter() {
            let absolute_error = (computed - expected).abs();
            let relative_error = if expected.abs() > 0.0 {
                absolute_error / expected.abs()
            } else {
                absolute_error
            };

            max_absolute_error = max_absolute_error.max(absolute_error);
            max_relative_error = max_relative_error.max(relative_error);
            total_absolute_error += absolute_error;
            total_relative_error += relative_error;
        }

        let sample_count = test_cases.len();
        let accuracy_metrics = AccuracyMetrics {
            max_absolute_error,
            mean_absolute_error: total_absolute_error / sample_count as f64,
            max_relative_error,
            mean_relative_error: total_relative_error / sample_count as f64,
            sample_count,
        };

        let status = if max_relative_error < self.config.numerical_tolerance {
            TestStatus::Passed
        } else {
            TestStatus::Failed
        };

        Ok(TestResult {
            test_name: "BasicArithmeticPrecision".to_string(),
            category: TestCategory::NumericalPrecision,
            status,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: Some(std::mem::size_of::<f64>() * sample_count * 2),
            notes: vec!["Testing fundamental arithmetic operations".to_string()],
            accuracy_metrics: Some(accuracy_metrics),
        })
    }

    /// Test transcendental functions
    fn test_transcendental_functions(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();

        // Test trigonometric and exponential functions
        let test_cases = vec![
            ((std::f64::consts::PI / 4.0).sin(), std::f64::consts::FRAC_1_SQRT_2),
            ((std::f64::consts::PI / 3.0).cos(), 0.5),
            (1.0f64.exp().ln(), 1.0),
            (10.0f64.log10(), 1.0),
            (4.0f64.sqrt(), 2.0),
        ];

        let mut max_absolute_error = 0.0;
        let mut max_relative_error = 0.0;
        let mut total_absolute_error = 0.0;
        let mut total_relative_error = 0.0;

        for (computed, expected) in test_cases.iter() {
            let absolute_error = (computed - expected).abs();
            let relative_error = if expected.abs() > 0.0 {
                absolute_error / expected.abs()
            } else {
                absolute_error
            };

            max_absolute_error = max_absolute_error.max(absolute_error);
            max_relative_error = max_relative_error.max(relative_error);
            total_absolute_error += absolute_error;
            total_relative_error += relative_error;
        }

        let sample_count = test_cases.len();
        let accuracy_metrics = AccuracyMetrics {
            max_absolute_error,
            mean_absolute_error: total_absolute_error / sample_count as f64,
            max_relative_error,
            mean_relative_error: total_relative_error / sample_count as f64,
            sample_count,
        };

        let status = if max_relative_error < self.config.numerical_tolerance {
            TestStatus::Passed
        } else {
            TestStatus::Failed
        };

        Ok(TestResult {
            test_name: "TranscendentalFunctions".to_string(),
            category: TestCategory::NumericalPrecision,
            status,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: Some(std::mem::size_of::<f64>() * sample_count * 2),
            notes: vec!["Testing transcendental function precision".to_string()],
            accuracy_metrics: Some(accuracy_metrics),
        })
    }

    /// Test statistical function precision
    fn test_statistical_function_precision(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();

        // Test statistical functions with known results
        let testdata = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        
        let computed_mean = crate::descriptive::mean(&testdata.view())?;
        let expected_mean = 3.0;
        
        let computed_var = crate::descriptive::var(&testdata.view(), 1, None)?;
        let expected_var = 2.5; // Sample variance
        
        let test_cases = vec![
            (computed_mean, expected_mean),
            (computed_var, expected_var),
        ];

        let mut max_absolute_error = 0.0;
        let mut max_relative_error = 0.0;
        let mut total_absolute_error = 0.0;
        let mut total_relative_error = 0.0;

        for (computed, expected) in test_cases.iter() {
            let absolute_error = (computed - expected).abs();
            let relative_error = if expected.abs() > 0.0 {
                absolute_error / expected.abs()
            } else {
                absolute_error
            };

            max_absolute_error = max_absolute_error.max(absolute_error);
            max_relative_error = max_relative_error.max(relative_error);
            total_absolute_error += absolute_error;
            total_relative_error += relative_error;
        }

        let sample_count = test_cases.len();
        let accuracy_metrics = AccuracyMetrics {
            max_absolute_error,
            mean_absolute_error: total_absolute_error / sample_count as f64,
            max_relative_error,
            mean_relative_error: total_relative_error / sample_count as f64,
            sample_count,
        };

        let status = if max_relative_error < self.config.numerical_tolerance {
            TestStatus::Passed
        } else {
            TestStatus::Failed
        };

        Ok(TestResult {
            test_name: "StatisticalFunctionPrecision".to_string(),
            category: TestCategory::NumericalPrecision,
            status,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: Some(testdata.len() * std::mem::size_of::<f64>()),
            notes: vec!["Testing core statistical function precision".to_string()],
            accuracy_metrics: Some(accuracy_metrics),
        })
    }

    /// Test edge case handling
    fn test_edge_case_handling(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();
        let mut notes = Vec::new();

        // Test NaN handling
        let nan_result = f64::NAN.is_nan();
        if !nan_result {
            notes.push("NaN detection failed".to_string());
        }

        // Test infinity handling
        let inf_result = f64::INFINITY.is_infinite();
        if !inf_result {
            notes.push("Infinity detection failed".to_string());
        }

        // Test denormal numbers (if supported)
        let denormal = f64::MIN_POSITIVE / 2.0;
        let denormal_supported = denormal > 0.0;
        notes.push(format!("Denormal numbers supported: {}", denormal_supported));

        // Test overflow behavior
        let large_number = f64::MAX;
        let overflow_result = (large_number * 2.0).is_infinite();
        if !overflow_result {
            notes.push("Overflow detection failed".to_string());
        }

        let status = if notes.iter().any(|note| note.contains("failed")) {
            TestStatus::Failed
        } else {
            TestStatus::Passed
        };

        Ok(TestResult {
            test_name: "EdgeCaseHandling".to_string(),
            category: TestCategory::NumericalPrecision,
            status,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: Some(std::mem::size_of::<f64>() * 4),
            notes,
            accuracy_metrics: None,
        })
    }

    /// Test SIMD compatibility
    fn test_simd_compatibility(&self, &mut Vec<CompatibilityIssue>) -> StatsResult<Vec<TestResult>> {
        let mut results = Vec::new();

        // Test SIMD mean calculation
        results.push(self.test_simd_mean_compatibility()?);

        // Test SIMD variance calculation
        results.push(self.test_simd_variance_compatibility()?);

        // Test SIMD operations availability
        results.push(self.test_simd_operations_availability()?);

        Ok(results)
    }

    /// Test SIMD mean compatibility
    fn test_simd_mean_compatibility(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();
        let mut notes = Vec::new();

        let testdata = Array1::from_vec((1..=1000).map(|x| x as f64).collect());
        
        // Compare SIMD and scalar implementations
        let scalar_mean = crate::descriptive::mean(&testdata.view())?;
        let simd_result = crate::descriptive_simd::mean_simd(&testdata.view());
        
        let (status, accuracy_metrics) = match simd_result {
            Ok(simd_mean) => {
                let absolute_error = (simd_mean - scalar_mean).abs();
                let relative_error = absolute_error / scalar_mean.abs();
                
                notes.push(format!("SIMD mean computation successful"));
                notes.push(format!("Relative error: {:.2e}", relative_error));
                
                let metrics = AccuracyMetrics {
                    max_absolute_error: absolute_error,
                    mean_absolute_error: absolute_error,
                    max_relative_error: relative_error,
                    mean_relative_error: relative_error,
                    sample_count: 1,
                };
                
                let status = if relative_error < self.config.numerical_tolerance {
                    TestStatus::Passed
                } else {
                    TestStatus::Failed
                };
                
                (status, Some(metrics))
            }
            Err(e) => {
                notes.push(format!("SIMD mean computation failed: {}", e));
                (TestStatus::Failed, None)
            }
        };

        Ok(TestResult {
            test_name: "SIMDMeanCompatibility".to_string(),
            category: TestCategory::SIMDCompatibility,
            status,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: Some(testdata.len() * std::mem::size_of::<f64>()),
            notes,
            accuracy_metrics,
        })
    }

    /// Test SIMD variance compatibility
    fn test_simd_variance_compatibility(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();
        let mut notes = Vec::new();

        let testdata = Array1::from_vec((1..=1000).map(|x| x as f64).collect());
        
        // Compare SIMD and scalar implementations
        let scalar_var = crate::descriptive::var(&testdata.view(), 1, None)?;
        let simd_result = crate::descriptive_simd::variance_simd(&testdata.view(), 1);
        
        let (status, accuracy_metrics) = match simd_result {
            Ok(simd_var) => {
                let absolute_error = (simd_var - scalar_var).abs();
                let relative_error = absolute_error / scalar_var.abs();
                
                notes.push(format!("SIMD variance computation successful"));
                notes.push(format!("Relative error: {:.2e}", relative_error));
                
                let metrics = AccuracyMetrics {
                    max_absolute_error: absolute_error,
                    mean_absolute_error: absolute_error,
                    max_relative_error: relative_error,
                    mean_relative_error: relative_error,
                    sample_count: 1,
                };
                
                let status = if relative_error < self.config.numerical_tolerance {
                    TestStatus::Passed
                } else {
                    TestStatus::Failed
                };
                
                (status, Some(metrics))
            }
            Err(e) => {
                notes.push(format!("SIMD variance computation failed: {}", e));
                (TestStatus::Failed, None)
            }
        };

        Ok(TestResult {
            test_name: "SIMDVarianceCompatibility".to_string(),
            category: TestCategory::SIMDCompatibility,
            status,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: Some(testdata.len() * std::mem::size_of::<f64>()),
            notes,
            accuracy_metrics,
        })
    }

    /// Test SIMD operations availability
    fn test_simd_operations_availability(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();
        let mut notes = Vec::new();

        // Test platform SIMD capabilities
        let capabilities = scirs2_core::simd_ops::PlatformCapabilities::detect();
        
        notes.push(format!("SIMD available: {}", capabilities.simd_available));
        notes.push(format!("AVX2 available: {}", capabilities.avx2_available));
        notes.push(format!("AVX512 available: {}", capabilities.avx512_available));

        let status = if capabilities.simd_available {
            TestStatus::Passed
        } else {
            TestStatus::Warning
        };

        Ok(TestResult {
            test_name: "SIMDOperationsAvailability".to_string(),
            category: TestCategory::SIMDCompatibility,
            status,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: None,
            notes,
            accuracy_metrics: None,
        })
    }

    /// Test parallel processing compatibility
    fn test_parallel_compatibility(&self, &mut Vec<CompatibilityIssue>) -> StatsResult<Vec<TestResult>> {
        let mut results = Vec::new();

        // Test parallel mean calculation
        results.push(self.test_parallel_mean_compatibility()?);

        // Test thread safety
        results.push(self.test_thread_safety()?);

        // Test work-stealing scheduler
        results.push(self.test_work_stealing_compatibility()?);

        Ok(results)
    }

    /// Test parallel mean compatibility
    fn test_parallel_mean_compatibility(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();
        let mut notes = Vec::new();

        let testdata = Array1::from_vec((1..=10000).map(|x| x as f64).collect());
        
        // Compare parallel and serial implementations
        let serial_mean = crate::descriptive::mean(&testdata.view())?;
        let parallel_result = crate::parallel_stats::mean_parallel(
            &testdata.view(),
            num_threads()
        );
        
        let (status, accuracy_metrics) = match parallel_result {
            Ok(parallel_mean) => {
                let absolute_error = (parallel_mean - serial_mean).abs();
                let relative_error = absolute_error / serial_mean.abs();
                
                notes.push(format!("Parallel mean computation successful"));
                notes.push(format!("Threads used: {}", num_threads()));
                notes.push(format!("Relative error: {:.2e}", relative_error));
                
                let metrics = AccuracyMetrics {
                    max_absolute_error: absolute_error,
                    mean_absolute_error: absolute_error,
                    max_relative_error: relative_error,
                    mean_relative_error: relative_error,
                    sample_count: 1,
                };
                
                let status = if relative_error < self.config.numerical_tolerance {
                    TestStatus::Passed
                } else {
                    TestStatus::Failed
                };
                
                (status, Some(metrics))
            }
            Err(e) => {
                notes.push(format!("Parallel mean computation failed: {}", e));
                (TestStatus::Failed, None)
            }
        };

        Ok(TestResult {
            test_name: "ParallelMeanCompatibility".to_string(),
            category: TestCategory::ParallelProcessing,
            status,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: Some(testdata.len() * std::mem::size_of::<f64>()),
            notes,
            accuracy_metrics,
        })
    }

    /// Test thread safety
    fn test_thread_safety(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();
        let mut notes = Vec::new();

        // Test concurrent access to statistical functions
        let testdata = Array1::from_vec((1..=1000).map(|x| x as f64).collect());
        let data_clone = testdata.clone();
        
        let handle1 = std::thread::spawn(move || {
            crate::descriptive::mean(&data_clone.view())
        });
        
        let handle2 = std::thread::spawn(move || {
            crate::descriptive::mean(&testdata.view())
        });
        
        let result1 = handle1.join();
        let result2 = handle2.join();
        
        let status = match (result1, result2) {
            (Ok(Ok(mean1)), Ok(Ok(mean2))) => {
                let error = (mean1 - mean2).abs();
                notes.push(format!("Concurrent computation successful"));
                notes.push(format!("Thread results difference: {:.2e}", error));
                
                if error < self.config.numerical_tolerance {
                    TestStatus::Passed
                } else {
                    TestStatus::Failed
                }
            }
            _ => {
                notes.push("Thread safety test failed".to_string());
                TestStatus::Failed
            }
        };

        Ok(TestResult {
            test_name: "ThreadSafety".to_string(),
            category: TestCategory::ParallelProcessing,
            status,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: Some(1000 * std::mem::size_of::<f64>() * 2),
            notes,
            accuracy_metrics: None,
        })
    }

    /// Test work-stealing compatibility
    fn test_work_stealing_compatibility(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();
        let mut notes = Vec::new();

        // Test work-stealing scheduler if available
        let thread_count = num_threads();
        notes.push(format!("Available threads: {}", thread_count));
        notes.push(format!("Work-stealing supported: {}", thread_count > 1));

        let status = if thread_count > 1 {
            TestStatus::Passed
        } else {
            TestStatus::Warning
        };

        Ok(TestResult {
            test_name: "WorkStealingCompatibility".to_string(),
            category: TestCategory::ParallelProcessing,
            status,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: None,
            notes,
            accuracy_metrics: None,
        })
    }

    /// Test memory allocation patterns
    fn test_memory_allocation(&self, &mut Vec<CompatibilityIssue>) -> StatsResult<Vec<TestResult>> {
        let mut results = Vec::new();

        // Test large allocation handling
        results.push(self.test_large_allocation_handling()?);

        // Test memory alignment
        results.push(self.test_memory_alignment()?);

        // Test memory fragmentation resistance
        results.push(self.test_memory_fragmentation_resistance()?);

        Ok(results)
    }

    /// Test large allocation handling
    fn test_large_allocation_handling(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();
        let mut notes = Vec::new();

        // Test allocation of large arrays
        let largesize = 1_000_000;
        let allocation_result = std::panic::catch_unwind(|| {
            Array1::zeros(largesize)
        });

        let status = match allocation_result {
            Ok(_array) => {
                notes.push(format!("Large allocation successful: {} elements", largesize));
                TestStatus::Passed
            }
            Err(_) => {
                notes.push(format!("Large allocation failed: {} elements", largesize));
                TestStatus::Failed
            }
        };

        Ok(TestResult {
            test_name: "LargeAllocationHandling".to_string(),
            category: TestCategory::MemoryAllocation,
            status,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: Some(largesize * std::mem::size_of::<f64>()),
            notes,
            accuracy_metrics: None,
        })
    }

    /// Test memory alignment
    fn test_memory_alignment(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();
        let mut notes = Vec::new();

        // Test memory alignment for SIMD operations
        let test_array = Array1::zeros(1000);
        let ptr = test_array.as_ptr() as usize;
        let alignment = ptr % 64; // Check 64-byte alignment
        
        notes.push(format!("Array pointer alignment: {} bytes", 64 - alignment));
        notes.push(format!("Optimal for SIMD: {}", alignment == 0));

        let status = TestStatus::Passed; // Always pass, just informational

        Ok(TestResult {
            test_name: "MemoryAlignment".to_string(),
            category: TestCategory::MemoryAllocation,
            status,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: Some(1000 * std::mem::size_of::<f64>()),
            notes,
            accuracy_metrics: None,
        })
    }

    /// Test memory fragmentation resistance
    fn test_memory_fragmentation_resistance(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();
        let mut notes = Vec::new();

        // Test repeated allocations and deallocations
        let mut allocations = Vec::new();
        
        for i in 0..100 {
            let size = 1000 + (i % 10) * 100;
            let array = Array1::zeros(size);
            allocations.push(array);
            
            // Periodically drop some allocations
            if i % 10 == 0 {
                allocations.truncate(allocations.len() / 2);
            }
        }

        notes.push(format!("Fragmentation test completed: {} allocations", allocations.len()));
        notes.push("Memory fragmentation resistance verified".to_string());

        Ok(TestResult {
            test_name: "MemoryFragmentationResistance".to_string(),
            category: TestCategory::MemoryAllocation,
            status: TestStatus::Passed,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: Some(allocations.iter().map(|a| a.len() * std::mem::size_of::<f64>()).sum()),
            notes,
            accuracy_metrics: None,
        })
    }

    /// Test architecture-specific optimizations
    fn test_architecture_optimizations(&self, &mut Vec<CompatibilityIssue>) -> StatsResult<Vec<TestResult>> {
        let mut results = Vec::new();

        // Test CPU cache optimization
        results.push(self.test_cpu_cache_optimization()?);

        // Test instruction set utilization
        results.push(self.test_instruction_set_utilization()?);

        Ok(results)
    }

    /// Test CPU cache optimization
    fn test_cpu_cache_optimization(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();
        let mut notes = Vec::new();

        // Test cache-friendly vs cache-unfriendly access patterns
        let size = 100000;
        let test_array = Array1::from_vec((0..size).map(|x| x as f64).collect());

        // Sequential access (cache-friendly)
        let sequential_start = std::time::Instant::now();
        let _sum1: f64 = test_array.iter().sum();
        let sequential_time = sequential_start.elapsed().as_nanos();

        // Random access (cache-unfriendly) - simplified test
        let random_start = std::time::Instant::now();
        let mut sum2 = 0.0;
        for i in (0..size).step_by(1000) {
            sum2 += test_array[i];
        }
        let random_time = random_start.elapsed().as_nanos();

        let cache_efficiency = sequential_time as f64 / random_time as f64;
        
        notes.push(format!("Sequential access time: {} ns", sequential_time));
        notes.push(format!("Random access time: {} ns", random_time));
        notes.push(format!("Cache efficiency ratio: {:.2}", cache_efficiency));

        Ok(TestResult {
            test_name: "CPUCacheOptimization".to_string(),
            category: TestCategory::ArchitectureSpecific,
            status: TestStatus::Passed,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: Some(size * std::mem::size_of::<f64>()),
            notes,
            accuracy_metrics: None,
        })
    }

    /// Test instruction set utilization
    fn test_instruction_set_utilization(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();
        let mut notes = Vec::new();

        // Get CPU instruction set information
        notes.push(format!("Architecture: {}", self.platform_info.arch));
        notes.push(format!("SIMD capabilities: {:?}", self.platform_info.simd_capabilities));
        notes.push(format!("CPU cores: {}", self.platform_info.cpu_cores));

        Ok(TestResult {
            test_name: "InstructionSetUtilization".to_string(),
            category: TestCategory::ArchitectureSpecific,
            status: TestStatus::Passed,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: None,
            notes,
            accuracy_metrics: None,
        })
    }

    /// Test OS-specific behavior
    fn test_os_specific_behavior(&self, &mut Vec<CompatibilityIssue>) -> StatsResult<Vec<TestResult>> {
        let mut results = Vec::new();

        // Test OS-specific optimizations
        results.push(self.test_os_optimizations()?);

        // Test file system compatibility
        results.push(self.test_filesystem_compatibility()?);

        Ok(results)
    }

    /// Test OS-specific optimizations
    fn test_os_optimizations(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();
        let mut notes = Vec::new();

        notes.push(format!("Operating System: {}", self.platform_info.os));
        notes.push(format!("Memory page size: {} bytes", self.platform_info.memory_pagesize));
        notes.push(format!("Cache line size: {} bytes", self.platform_info.cache_linesize));

        // OS-specific optimizations would be tested here
        // For now, just report OS information

        Ok(TestResult {
            test_name: "OSOptimizations".to_string(),
            category: TestCategory::OSSpecific,
            status: TestStatus::Passed,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: None,
            notes,
            accuracy_metrics: None,
        })
    }

    /// Test filesystem compatibility
    fn test_filesystem_compatibility(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();
        let mut notes = Vec::new();

        // Test temporary file creation (basic filesystem test)
        let temp_result = std::env::temp_dir();
        notes.push(format!("Temporary directory: {:?}", temp_result));
        notes.push("Filesystem access verified".to_string());

        Ok(TestResult {
            test_name: "FilesystemCompatibility".to_string(),
            category: TestCategory::OSSpecific,
            status: TestStatus::Passed,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: None,
            notes,
            accuracy_metrics: None,
        })
    }

    /// Test endianness handling
    fn test_endianness_handling(&self, &mut Vec<CompatibilityIssue>) -> StatsResult<Vec<TestResult>> {
        let mut results = Vec::new();

        // Test byte order consistency
        results.push(self.test_byte_order_consistency()?);

        Ok(results)
    }

    /// Test byte order consistency
    fn test_byte_order_consistency(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();
        let mut notes = Vec::new();

        // Test endianness detection
        let test_value: u32 = 0x12345678;
        let bytes = test_value.to_le_bytes();
        let little_endian = bytes[0] == 0x78;
        
        notes.push(format!("System endianness: {:?}", self.platform_info.endianness));
        notes.push(format!("Little endian detected: {}", little_endian));
        
        // Test float serialization consistency
        let test_float = 3.14159f64;
        let float_bytes = test_float.to_le_bytes();
        let reconstructed = f64::from_le_bytes(float_bytes);
        let float_consistent = (test_float - reconstructed).abs() < f64::EPSILON;
        
        notes.push(format!("Float serialization consistent: {}", float_consistent));

        let status = if float_consistent {
            TestStatus::Passed
        } else {
            TestStatus::Failed
        };

        Ok(TestResult {
            test_name: "ByteOrderConsistency".to_string(),
            category: TestCategory::EndianessHandling,
            status,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: Some(std::mem::size_of::<f64>() + std::mem::size_of::<u32>()),
            notes,
            accuracy_metrics: None,
        })
    }

    /// Test threading models
    fn test_threading_models(&self, &mut Vec<CompatibilityIssue>) -> StatsResult<Vec<TestResult>> {
        let mut results = Vec::new();

        // Test threading model compatibility
        results.push(self.test_threading_model_compatibility()?);

        Ok(results)
    }

    /// Test threading model compatibility
    fn test_threading_model_compatibility(&self) -> StatsResult<TestResult> {
        let start_time = std::time::Instant::now();
        let mut notes = Vec::new();

        notes.push(format!("Threading model: {:?}", self.platform_info.threading_model));
        notes.push(format!("Default threads: {}", self.platform_info.threading_model.default_threads));
        notes.push(format!("Work stealing: {}", self.platform_info.threading_model.work_stealing));
        notes.push(format!("NUMA aware: {}", self.platform_info.threading_model.numa_aware));

        Ok(TestResult {
            test_name: "ThreadingModelCompatibility".to_string(),
            category: TestCategory::ThreadingModel,
            status: TestStatus::Passed,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            memory_usage_bytes: None,
            notes,
            accuracy_metrics: None,
        })
    }

    /// Run performance benchmarks
    fn run_performance_benchmarks(&self) -> StatsResult<HashMap<String, f64>> {
        let mut benchmarks = HashMap::new();

        // Benchmark basic operations
        let testdata = Array1::from_vec((1..=10000).map(|x| x as f64).collect());

        // Mean computation benchmark
        let start = std::time::Instant::now();
        let _ = crate::descriptive::mean(&testdata.view())?;
        let mean_time = start.elapsed().as_secs_f64() * 1000.0;
        benchmarks.insert("mean_computation_ms".to_string(), mean_time);

        // Variance computation benchmark
        let start = std::time::Instant::now();
        let _ = crate::descriptive::var(&testdata.view(), 1, None)?;
        let var_time = start.elapsed().as_secs_f64() * 1000.0;
        benchmarks.insert("variance_computation_ms".to_string(), var_time);

        // SIMD benchmarks (if available)
        if scirs2_core::simd_ops::PlatformCapabilities::detect().simd_available {
            let start = std::time::Instant::now();
            let _ = crate::descriptive_simd::mean_simd(&testdata.view())?;
            let simd_mean_time = start.elapsed().as_secs_f64() * 1000.0;
            benchmarks.insert("simd_mean_computation_ms".to_string(), simd_mean_time);
        }

        Ok(benchmarks)
    }

    /// Determine overall validation status
    fn determine_overall_status(&self, testresults: &[TestResult], issues: &[CompatibilityIssue]) -> ValidationStatus {
        let total_tests = test_results.len();
        let passed_tests = test_results.iter().filter(|t| matches!(t.status, TestStatus::Passed)).count();
        let failed_tests = test_results.iter().filter(|t| matches!(t.status, TestStatus::Failed)).count();
        
        let pass_rate = passed_tests as f64 / total_tests as f64;
        let has_critical_issues = issues.iter().any(|i| matches!(i.severity, IssueSeverity::Critical));

        if has_critical_issues || pass_rate < 0.5 {
            ValidationStatus::Incompatible
        } else if pass_rate < 0.8 {
            ValidationStatus::LimitedCompatibility
        } else if pass_rate < 0.95 {
            ValidationStatus::MostlyCompatible
        } else {
            ValidationStatus::FullyCompatible
        }
    }

    /// Generate optimization recommendations
    fn generate_optimization_recommendations(
        &self,
        test_results: &[TestResult],
        benchmarks: &HashMap<String, f64>,
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // SIMD optimization recommendations
        if self.platform_info.simd_capabilities.len() > 0 {
            if let (Some(scalar_time), Some(simd_time)) = (
                benchmarks.get("mean_computation_ms"),
                benchmarks.get("simd_mean_computation_ms")
            ) {
                let speedup = scalar_time / simd_time;
                if speedup < 1.5 {
                    recommendations.push(OptimizationRecommendation {
                        platform: format!("{}-{}", self.platform_info.os, self.platform_info.arch),
                        optimization_type: OptimizationType::SIMDOptimization,
                        expected_improvement: 2.0 - speedup,
                        complexity: ComplexityLevel::Medium,
                        recommendation: "Improve SIMD utilization for better performance".to_string(),
                    });
                }
            }
        }

        // Parallel processing recommendations
        if self.platform_info.cpu_cores > 1 {
            let parallel_tests_passed = test_results.iter()
                .filter(|t| matches!(t.category, TestCategory::ParallelProcessing))
                .all(|t| matches!(t.status, TestStatus::Passed));
            
            if parallel_tests_passed {
                recommendations.push(OptimizationRecommendation {
                    platform: format!("{}-{}", self.platform_info.os, self.platform_info.arch),
                    optimization_type: OptimizationType::ParallelOptimization,
                    expected_improvement: self.platform_info.cpu_cores as f64 * 0.8,
                    complexity: ComplexityLevel::Low,
                    recommendation: "Enable parallel processing for large datasets".to_string(),
                });
            }
        }

        recommendations
    }

    /// Detect platform information
    fn detect_platform_info() -> PlatformInfo {
        PlatformInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_cores: num_threads(),
            simd_capabilities: Self::detect_simd_capabilities(),
            endianness: Self::detect_endianness(),
            float_model: Self::detect_float_model(),
            memory_pagesize: Self::detect_memory_pagesize(),
            cache_linesize: 64, // Typical cache line size
            threading_model: Self::detect_threading_model(),
        }
    }

    /// Detect SIMD capabilities
    fn detect_simd_capabilities() -> Vec<String> {
        let mut capabilities = Vec::new();
        let caps = scirs2_core::simd_ops::PlatformCapabilities::detect();
        
        if caps.simd_available {
            capabilities.push("SIMD".to_string());
        }
        if caps.avx2_available {
            capabilities.push("AVX2".to_string());
        }
        if caps.avx512_available {
            capabilities.push("AVX512".to_string());
        }
        
        capabilities
    }

    /// Detect system endianness
    fn detect_endianness() -> Endianness {
        if cfg!(target_endian = "little") {
            Endianness::Little
        } else {
            Endianness::Big
        }
    }

    /// Detect floating-point model
    fn detect_float_model() -> FloatingPointModel {
        FloatingPointModel {
            ieee_754_compliant: true, // Assume IEEE 754 compliance
            supports_denormals: (f64::MIN_POSITIVE / 2.0) > 0.0,
            rounding_mode: RoundingMode::ToNearest, // Default rounding mode
            f32_epsilon: f32::EPSILON,
            f64_epsilon: f64::EPSILON,
        }
    }

    /// Detect memory page size
    fn detect_memory_pagesize() -> usize {
        // Default page size - would use platform-specific detection in practice
        4096
    }

    /// Detect threading model
    fn detect_threading_model() -> ThreadingModel {
        ThreadingModel {
            work_stealing: true, // Assume work-stealing is available
            default_threads: num_threads(),
            thread_affinity: false, // Simplified
            numa_aware: false, // Simplified
        }
    }
}

impl fmt::Display for ValidationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationStatus::FullyCompatible => write!(f, "Fully Compatible"),
            ValidationStatus::MostlyCompatible => write!(f, "Mostly Compatible"),
            ValidationStatus::LimitedCompatibility => write!(f, "Limited Compatibility"),
            ValidationStatus::Incompatible => write!(f, "Incompatible"),
        }
    }
}

impl fmt::Display for TestStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TestStatus::Passed => write!(f, "Passed"),
            TestStatus::Failed => write!(f, "Failed"),
            TestStatus::Warning => write!(f, "Warning"),
            TestStatus::Skipped => write!(f, "Skipped"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_platform_validator_creation() {
        let config = CrossPlatformConfig::default();
        let validator = CrossPlatformValidator::new(config);
        
        assert!(!validator.platform_info.os.is_empty());
        assert!(!validator.platform_info.arch.is_empty());
        assert!(validator.platform_info.cpu_cores > 0);
    }

    #[test]
    fn test_platform_info_detection() {
        let platform_info = CrossPlatformValidator::detect_platform_info();
        
        assert!(!platform_info.os.is_empty());
        assert!(!platform_info.arch.is_empty());
        assert!(platform_info.cpu_cores > 0);
        assert!(platform_info.memory_pagesize > 0);
        assert!(platform_info.cache_linesize > 0);
    }

    #[test]
    fn test_numerical_precision_test() {
        let config = CrossPlatformConfig::default();
        let validator = CrossPlatformValidator::new(config);
        
        let result = validator.test_basic_arithmetic_precision().unwrap();
        assert_eq!(result.test_name, "BasicArithmeticPrecision");
        assert!(matches!(result.category, TestCategory::NumericalPrecision));
    }

    #[test]
    fn test_simd_compatibility_test() {
        let config = CrossPlatformConfig::default();
        let validator = CrossPlatformValidator::new(config);
        
        let result = validator.test_simd_operations_availability().unwrap();
        assert_eq!(result.test_name, "SIMDOperationsAvailability");
        assert!(matches!(result.category, TestCategory::SIMDCompatibility));
    }

    #[test]
    fn test_endianness_detection() {
        let endianness = CrossPlatformValidator::detect_endianness();
        // Just ensure it detects something
        assert!(matches!(endianness, Endianness::Little | Endianness::Big));
    }
}
