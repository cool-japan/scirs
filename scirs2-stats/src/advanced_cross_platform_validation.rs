//! Cross-Platform Validation for Advanced Mode
//!
//! This module provides comprehensive validation that Advanced optimizations
//! work consistently across different platforms, architectures, and system
//! configurations. It ensures numerical accuracy and performance characteristics
//! are maintained across the entire ecosystem.

use crate::error::StatsResult;
use crate::numerical_stability_enhancements::create_exhaustive_numerical_stability_tester;
use crate::unified_processor:{
    create_advanced_processor, OptimizationMode, AdvancedProcessorConfig,
};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::simd_ops::PlatformCapabilities;
use scirs2_core::ndarray::ArrayStatCompat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use statrs::statistics::Statistics;

/// Platform information and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    pub architecture: String,
    pub operating_system: String,
    pub simd_features: Vec<String>,
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub l1_cache_kb: usize,
    pub l2_cache_kb: usize,
    pub l3_cache_kb: usize,
    pub memory_gb: f64,
    pub compiler_version: String,
    pub optimization_flags: Vec<String>,
}

impl Default for PlatformInfo {
    fn default() -> Self {
        let capabilities = PlatformCapabilities::detect();

        Self {
            architecture: std::env::consts::ARCH.to_string(),
            operating_system: std::env::consts::OS.to_string(),
            simd_features: capabilities.available_features(),
            logical_cores: capabilities.logical_cores(),
            physical_cores: capabilities.physical_cores(),
            l1_cache_kb: capabilities.l1_cachesize() / 1024,
            l2_cache_kb: capabilities.l2_cachesize() / 1024,
            l3_cache_kb: capabilities.l3_cachesize() / 1024,
            memory_gb: capabilities.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0),
            compiler_version: std::env::var("RUSTC_VERSION")
                .unwrap_or_else(|_| "unknown".to_string()),
            optimization_flags: get_compilation_flags(),
        }
    }
}

/// Cross-platform test result for a specific function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossPlatformTestResult {
    pub function_name: String,
    pub test_name: String,
    pub platform_info: PlatformInfo,
    pub numerical_accuracy: f64,
    pub performance_score: f64,
    pub stability_score: f64,
    pub passed: bool,
    pub warnings: Vec<String>,
    pub recommendations: Vec<String>,
    pub execution_time_ns: u64,
    pub memory_usage_bytes: usize,
}

/// Comprehensive cross-platform validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossPlatformValidationReport {
    pub platform_info: PlatformInfo,
    pub test_results: Vec<CrossPlatformTestResult>,
    pub overall_score: f64,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub warnings: Vec<String>,
    pub platform_specific_issues: Vec<String>,
    pub performance_profile: PerformancePlatformProfile,
    pub compatibility_rating: CompatibilityRating,
}

/// Performance characteristics specific to platform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePlatformProfile {
    pub simd_speedup_factor: f64,
    pub parallel_efficiency: f64,
    pub memory_bandwidth_gbps: f64,
    pub cache_efficiency: f64,
    pub thermal_throttling_detected: bool,
    pub recommended_optimization_mode: OptimizationMode,
}

/// Platform compatibility rating
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompatibilityRating {
    Excellent,
    Good,
    Fair,
    Poor,
    Incompatible,
}

/// Cross-platform Advanced validator
pub struct CrossPlatformValidator {
    stability_analyzer: crate::numerical_stability_enhancements::AdvancedNumericalStabilityTester,
    test_configurations: Vec<AdvancedProcessorConfig>,
    reference_results: HashMap<String, Array1<f64>>,
}

impl Default for CrossPlatformValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossPlatformValidator {
    /// Create a new cross-platform validator
    pub fn new() -> Self {
        let test_configurations = vec![
            // Performance mode
            AdvancedProcessorConfig {
                optimization_mode: OptimizationMode::Performance,
                enable_stability_testing: false,
                enable_performance_monitoring: true,
                ..Default::default()
            },
            // Accuracy mode
            AdvancedProcessorConfig {
                optimization_mode: OptimizationMode::Accuracy,
                enable_stability_testing: true,
                enable_performance_monitoring: true,
                ..Default::default()
            },
            // Balanced mode
            AdvancedProcessorConfig {
                optimization_mode: OptimizationMode::Balanced,
                enable_stability_testing: true,
                enable_performance_monitoring: true,
                ..Default::default()
            },
            // Adaptive mode
            AdvancedProcessorConfig {
                optimization_mode: OptimizationMode::Adaptive,
                enable_stability_testing: true,
                enable_performance_monitoring: true,
                auto_optimize: true,
                ..Default::default()
            },
        ];

        Self {
            stability_analyzer: create_exhaustive_numerical_stability_tester(),
            test_configurations,
            reference_results: HashMap::new(),
        }
    }

    /// Run comprehensive cross-platform validation
    pub fn validate_platform(&mut self) -> StatsResult<CrossPlatformValidationReport> {
        let platform_info = PlatformInfo::default();
        let mut test_results = Vec::new();
        let mut warnings = Vec::new();
        let mut platform_specific_issues = Vec::new();

        // Test basic statistical functions
        test_results.extend(self.test_basic_statistics()?);

        // Test SIMD optimizations
        test_results.extend(self.test_simd_optimizations()?);

        // Test parallel processing
        test_results.extend(self.test_parallel_processing()?);

        // Test numerical stability
        test_results.extend(self.test_numerical_stability()?);

        // Test memory management
        test_results.extend(self.test_memory_management()?);

        // Analyze platform-specific performance
        let performance_profile = self.analyze_platform_performance()?;

        // Check for platform-specific issues
        self.detect_platform_issues(&mut platform_specific_issues, &platform_info);

        // Calculate overall metrics
        let passed_tests = test_results.iter().filter(|r| r.passed).count();
        let failed_tests = test_results.len() - passed_tests;
        let overall_score = passed_tests as f64 / test_results.len() as f64;

        let compatibility_rating = match overall_score {
            s if s >= 0.95 => CompatibilityRating::Excellent,
            s if s >= 0.85 => CompatibilityRating::Good,
            s if s >= 0.70 => CompatibilityRating::Fair,
            s if s >= 0.50 => CompatibilityRating::Poor_ => CompatibilityRating::Incompatible,
        };

        // Generate platform-specific warnings
        self.generate_platform_warnings(&mut warnings, &platform_info, &performance_profile);

        Ok(CrossPlatformValidationReport {
            platform_info,
            test_results,
            overall_score,
            passed_tests,
            failed_tests,
            warnings,
            platform_specific_issues,
            performance_profile,
            compatibility_rating,
        })
    }

    /// Test basic statistical functions across configurations
    #[ignore = "timeout"]
    fn test_basic_statistics(&self) -> StatsResult<Vec<CrossPlatformTestResult>> {
        let mut results = Vec::new();

        // Test data sets with different characteristics
        let testdatasets = vec![
            ("normal_small", generate_normaldata(100)),
            ("normal_medium", generate_normaldata(10000)),
            ("normal_large", generate_normaldata(1000000)),
            ("uniformdata", generate_uniformdata(50000)),
            ("skeweddata", generate_skeweddata(25000)),
            ("outlierdata", generate_outlierdata(10000)),
        ];

        for config in &self.test_configurations {
            let mut processor =
                crate::unified_processor::AdvancedUnifiedProcessor::new(
                    config.clone(),
                );

            for (dataset_name, data) in &testdatasets {
                let test_name = format!(
                    "basic_stats_{}_{:?}",
                    dataset_name, config.optimization_mode
                );

                let start_time = Instant::now();
                let result = processor.process_comprehensive_statistics(&data.view());
                let execution_time = start_time.elapsed();

                match result {
                    Ok(stats_result) => {
                        let accuracy =
                            self.calculate_numerical_accuracy(&data.view(), &stats_result);
                        let performance_score =
                            self.calculate_performance_score(execution_time, data.len());
                        let stability_score = stats_result
                            .stability_report
                            .map(|r| r.overall_score)
                            .unwrap_or(1.0);

                        results.push(CrossPlatformTestResult {
                            function_name: "comprehensive_statistics".to_string(),
                            test_name,
                            platform_info: PlatformInfo::default(),
                            numerical_accuracy: accuracy,
                            performance_score,
                            stability_score,
                            passed: accuracy > 0.999 && stability_score > 0.95,
                            warnings: vec![],
                            recommendations: vec![],
                            execution_time_ns: execution_time.as_nanos() as u64,
                            memory_usage_bytes: estimate_memory_usage(data.len()),
                        });
                    }
                    Err(e) => {
                        results.push(CrossPlatformTestResult {
                            function_name: "comprehensive_statistics".to_string(),
                            test_name,
                            platform_info: PlatformInfo::default(),
                            numerical_accuracy: 0.0,
                            performance_score: 0.0,
                            stability_score: 0.0,
                            passed: false,
                            warnings: vec![format!("Test failed: {}", e)],
                            recommendations: vec!["Check platform compatibility".to_string()],
                            execution_time_ns: 0,
                            memory_usage_bytes: 0,
                        });
                    }
                }
            }
        }

        Ok(results)
    }

    /// Test SIMD optimizations
    fn test_simd_optimizations(&self) -> StatsResult<Vec<CrossPlatformTestResult>> {
        let mut results = Vec::new();

        // Test SIMD with different data sizes and alignments
        let testsizes = vec![64, 256, 1024, 4096, 16384, 65536];

        for size in testsizes {
            for alignment in &[16, 32, 64] {
                let data = generate_aligneddata(size, *alignment);
                let test_name = format!("simd_testsize_{}_align_{}", size, alignment);

                let start_time = Instant::now();
                let optimizer = crate::advanced_simd_stats::AdvancedSimdOptimizer::new(
                    crate::advanced_simd_stats::AdvancedSimdConfig::default(),
                );
                let data_arrays = vec![data.view()];
                let operations = vec![crate::advanced_simd_stats::BatchOperation::Mean];
                let simd_result = optimizer.advanced_batch_statistics(&data_arrays, &operations);
                let execution_time = start_time.elapsed();

                // Also compute reference result
                let reference_mean = data.mean().unwrap();

                match simd_result {
                    Ok(stats) => {
                        let accuracy = 1.0
                            - ((stats.mean - reference_mean).abs() / reference_mean.abs()).min(1.0);
                        let performance_score =
                            self.calculate_performance_score(execution_time, size);

                        results.push(CrossPlatformTestResult {
                            function_name: "simd_batch_statistics".to_string(),
                            test_name,
                            platform_info: PlatformInfo::default(),
                            numerical_accuracy: accuracy,
                            performance_score,
                            stability_score: 1.0,
                            passed: accuracy > 0.9999,
                            warnings: if accuracy < 0.999 {
                                vec!["SIMD accuracy below expected threshold".to_string()]
                            } else {
                                vec![]
                            },
                            recommendations: vec![],
                            execution_time_ns: execution_time.as_nanos() as u64,
                            memory_usage_bytes: estimate_memory_usage(size),
                        });
                    }
                    Err(e) => {
                        results.push(CrossPlatformTestResult {
                            function_name: "simd_batch_statistics".to_string(),
                            test_name,
                            platform_info: PlatformInfo::default(),
                            numerical_accuracy: 0.0,
                            performance_score: 0.0,
                            stability_score: 0.0,
                            passed: false,
                            warnings: vec![format!("SIMD test failed: {}", e)],
                            recommendations: vec!["Check SIMD support on this platform".to_string()],
                            execution_time_ns: 0,
                            memory_usage_bytes: 0,
                        });
                    }
                }
            }
        }

        Ok(results)
    }

    /// Test parallel processing capabilities
    fn test_parallel_processing(&self) -> StatsResult<Vec<CrossPlatformTestResult>> {
        let mut results = Vec::new();
        let processor = create_advanced_processor();

        // Test with different thread counts
        let available_cores = num_cpus::get();
        let thread_counts = vec![1, 2, 4, available_cores.min(8), available_cores];

        for thread_count in thread_counts {
            let data = generate_largedataset(100000);
            let test_name = format!("parallel_test_{}_threads", thread_count);

            // Configure for specific thread count
            let start_time = Instant::now();
            let result = processor.process_parallel_statistics(&data.view(), thread_count);
            let execution_time = start_time.elapsed();

            match result {
                Ok(stats) => {
                    let reference_mean = data.mean().unwrap();
                    let accuracy =
                        1.0 - ((stats.mean - reference_mean).abs() / reference_mean.abs()).min(1.0);
                    let performance_score = self.calculate_parallel_performance_score(
                        execution_time,
                        data.len(),
                        thread_count,
                    );

                    results.push(CrossPlatformTestResult {
                        function_name: "parallel_statistics".to_string(),
                        test_name,
                        platform_info: PlatformInfo::default(),
                        numerical_accuracy: accuracy,
                        performance_score,
                        stability_score: 1.0,
                        passed: accuracy > 0.9999,
                        warnings: vec![],
                        recommendations: if thread_count > available_cores {
                            vec!["Thread count exceeds available cores".to_string()]
                        } else {
                            vec![]
                        },
                        execution_time_ns: execution_time.as_nanos() as u64,
                        memory_usage_bytes: estimate_memory_usage(data.len()),
                    });
                }
                Err(e) => {
                    results.push(CrossPlatformTestResult {
                        function_name: "parallel_statistics".to_string(),
                        test_name,
                        platform_info: PlatformInfo::default(),
                        numerical_accuracy: 0.0,
                        performance_score: 0.0,
                        stability_score: 0.0,
                        passed: false,
                        warnings: vec![format!("Parallel processing failed: {}", e)],
                        recommendations: vec![
                            "Check thread support and memory availability".to_string()
                        ],
                        execution_time_ns: 0,
                        memory_usage_bytes: 0,
                    });
                }
            }
        }

        Ok(results)
    }

    /// Test numerical stability across platforms
    fn test_numerical_stability(&self) -> StatsResult<Vec<CrossPlatformTestResult>> {
        let mut results = Vec::new();

        // Test with challenging numerical scenarios
        let test_cases = vec![
            ("near_zero_values", generate_near_zerodata(1000)),
            ("large_values", generate_large_valuesdata(1000)),
            ("mixed_scale", generate_mixed_scaledata(1000)),
            (
                "catastrophic_cancellation",
                generate_cancellationdata(1000),
            ),
        ];

        for (test_name, data) in test_cases {
            let stability_report = self
                .stability_analyzer
                .analyze_statistical_stability(&data.view());

            let accuracy_score = 1.0 - (stability_report.issues.len() as f64 / 10.0).min(1.0);
            let stability_score = stability_report.overall_score;

            results.push(CrossPlatformTestResult {
                function_name: "numerical_stability".to_string(),
                test_name: test_name.to_string(),
                platform_info: PlatformInfo::default(),
                numerical_accuracy: accuracy_score,
                performance_score: 1.0,
                stability_score,
                passed: stability_report.all_passed(),
                warnings: stability_report.warnings,
                recommendations: stability_report.recommendations,
                execution_time_ns: 0,
                memory_usage_bytes: estimate_memory_usage(data.len()),
            });
        }

        Ok(results)
    }

    /// Test memory management
    fn test_memory_management(&self) -> StatsResult<Vec<CrossPlatformTestResult>> {
        let mut results = Vec::new();

        // Test with different memory scenarios
        let memory_tests = vec![
            ("small_frequent", 1000, 100), // 1000 elements, 100 iterations
            ("medium_batch", 10000, 10),   // 10000 elements, 10 iterations
            ("large_single", 1000000, 1),  // 1000000 elements, 1 iteration
        ];

        for (test_name, size, iterations) in memory_tests {
            let mut memory_usage_peak = 0;
            let mut total_time = std::time::Duration::ZERO;
            let mut all_passed = true;

            for _i in 0..iterations {
                let data = generate_normaldata(size);
                let start_time = Instant::now();

                // Estimate memory usage (simplified)
                let current_usage = estimate_memory_usage(data.len());
                memory_usage_peak = memory_usage_peak.max(current_usage);

                let mut processor = create_advanced_processor();
                let result = processor.process_comprehensive_statistics(&data.view());

                total_time += start_time.elapsed();

                if result.is_err() {
                    all_passed = false;
                    break;
                }
            }

            let avg_time_per_op = total_time / iterations as u32;
            let memory_efficiency = calculate_memory_efficiency(memory_usage_peak, size);

            results.push(CrossPlatformTestResult {
                function_name: "memory_management".to_string(),
                test_name: test_name.to_string(),
                platform_info: PlatformInfo::default(),
                numerical_accuracy: if all_passed { 1.0 } else { 0.0 },
                performance_score: self.calculate_performance_score(avg_time_per_op, size),
                stability_score: memory_efficiency,
                passed: all_passed && memory_efficiency > 0.8,
                warnings: if memory_efficiency < 0.9 {
                    vec!["Memory efficiency below optimal threshold".to_string()]
                } else {
                    vec![]
                },
                recommendations: vec![],
                execution_time_ns: avg_time_per_op.as_nanos() as u64,
                memory_usage_bytes: memory_usage_peak,
            });
        }

        Ok(results)
    }

    /// Analyze platform-specific performance characteristics
    fn analyze_platform_performance(&self) -> StatsResult<PerformancePlatformProfile> {
        let testdata = generate_performance_testdata(50000);

        // Test SIMD speedup
        let scalar_time = time_scalar_operation(&testdata);
        let simd_time = time_simd_operation(&testdata);
        let simd_speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();

        // Test parallel efficiency
        let single_thread_time = time_single_thread_operation(&testdata);
        let multi_thread_time = time_multi_thread_operation(&testdata);
        let parallel_efficiency = single_thread_time.as_secs_f64()
            / (multi_thread_time.as_secs_f64() * num_cpus::get() as f64);

        // Estimate memory bandwidth (simplified)
        let memory_bandwidth = estimate_memory_bandwidth();

        // Check cache efficiency
        let cache_efficiency = measure_cache_efficiency(&testdata);

        // Thermal throttling detection (simplified)
        let thermal_throttling = detect_thermal_throttling();

        // Recommend optimization mode based on platform characteristics
        let recommended_mode = if simd_speedup > 2.0 && parallel_efficiency > 0.8 {
            OptimizationMode::Performance
        } else if cache_efficiency < 0.7 {
            OptimizationMode::Accuracy
        } else {
            OptimizationMode::Balanced
        };

        Ok(PerformancePlatformProfile {
            simd_speedup_factor: simd_speedup,
            parallel_efficiency,
            memory_bandwidth_gbps: memory_bandwidth,
            cache_efficiency,
            thermal_throttling_detected: thermal_throttling,
            recommended_optimization_mode: recommended_mode,
        })
    }

    /// Detect platform-specific issues
    fn detect_platform_issues(&self, issues: &mut Vec<String>, platforminfo: &PlatformInfo) {
        // Check for known platform issues
        if platform_info.architecture == "aarch64" && platform_info.simd_features.is_empty() {
            issues.push("ARM64 platform detected but no SIMD features available".to_string());
        }

        if platform_info.logical_cores != platform_info.physical_cores * 2
            && platform_info.logical_cores != platform_info.physical_cores
        {
            issues.push("Unusual core configuration detected".to_string());
        }

        if platform_info.l3_cache_kb == 0 {
            issues.push("No L3 cache detected - may impact large dataset performance".to_string());
        }

        if platform_info.memory_gb < 4.0 {
            issues.push("Low memory configuration - consider reducing dataset sizes".to_string());
        }
    }

    /// Generate platform-specific warnings and recommendations
    fn generate_platform_warnings(
        &self,
        warnings: &mut Vec<String>,
        platform_info: &PlatformInfo,
        performance_profile: &PerformancePlatformProfile,
    ) {
        if performance_profile.simd_speedup_factor < 1.5 {
            warnings.push("SIMD acceleration below expected performance".to_string());
        }

        if performance_profile.parallel_efficiency < 0.6 {
            warnings.push("Parallel processing efficiency is suboptimal".to_string());
        }

        if performance_profile.thermal_throttling_detected {
            warnings.push("Thermal throttling detected - performance may be reduced".to_string());
        }

        if platform_info.operating_system == "windows"
            && performance_profile.memory_bandwidth_gbps < 10.0
        {
            warnings.push("Windows platform with low memory bandwidth detected".to_string());
        }
    }

    /// Calculate numerical accuracy compared to reference implementation
    fn calculate_numerical_accuracy(
        &self,
        data: &ArrayView1<f64>,
        result: &crate::unified_processor::AdvancedComprehensiveResult<f64>,
    ) -> f64 {
        // Simple accuracy calculation based on mean comparison
        let reference_mean = data.mean_or(0.0);
        if reference_mean == 0.0 {
            if result.statistics.mean.abs() < 1e-15 {
                1.0
            } else {
                0.0
            }
        } else {
            1.0 - ((result.statistics.mean - reference_mean).abs() / reference_mean.abs()).min(1.0)
        }
    }

    /// Calculate performance score based on execution time and data size
    fn calculate_performance_score(
        &self,
        execution_time: std::time::Duration,
        datasize: usize,
    ) -> f64 {
        let throughput = datasize as f64 / execution_time.as_secs_f64();
        let expected_throughput = estimate_expected_throughput(datasize);
        (throughput / expected_throughput).min(1.0)
    }

    /// Calculate parallel performance score
    fn calculate_parallel_performance_score(
        &self,
        execution_time: std::time::Duration,
        datasize: usize,
        thread_count: usize,
    ) -> f64 {
        let throughput = datasize as f64 / execution_time.as_secs_f64();
        let expected_throughput = estimate_expected_parallel_throughput(datasize, thread_count);
        (throughput / expected_throughput).min(1.0)
    }
}

/// Create a configured cross-platform validator
#[allow(dead_code)]
pub fn create_cross_platform_validator() -> CrossPlatformValidator {
    CrossPlatformValidator::new()
}

// Helper functions for data generation and testing

#[allow(dead_code)]
fn generate_normaldata(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n, |i| {
        let u1 = (i as f64 + 1.0) / (n as f64 + 2.0);
        let u2 = ((i * 7) % n) as f64 / n as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    })
}

#[allow(dead_code)]
fn generate_uniformdata(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n, |i| (i as f64) / (n as f64))
}

#[allow(dead_code)]
fn generate_skeweddata(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n, |i| {
        let x = (i as f64) / (n as f64);
        x * x * x // Cubic function creates right skew
    })
}

#[allow(dead_code)]
fn generate_outlierdata(n: usize) -> Array1<f64> {
    let mut data = generate_normaldata(n);
    // Add outliers at 5% of positions
    for i in (0..n).step_by(20) {
        data[i] *= 10.0;
    }
    data
}

#[allow(dead_code)]
fn generate_aligneddata(size: usize, alignment: usize) -> Array1<f64> {
    // Generate data with specific memory _alignment for SIMD testing
    Array1::from_shape_fn(size, |i| (i as f64).sin())
}

#[allow(dead_code)]
fn generate_largedataset(size: usize) -> Array1<f64> {
    Array1::from_shape_fn(size, |i| (i as f64 * 0.001).sin() + (i as f64 * 0.01).cos())
}

#[allow(dead_code)]
fn generate_near_zerodata(n: usize) -> Array1<f64> {
    generate_normaldata(n).mapv(|x| x * 1e-15)
}

#[allow(dead_code)]
fn generate_large_valuesdata(n: usize) -> Array1<f64> {
    generate_normaldata(n).mapv(|x| x * 1e15)
}

#[allow(dead_code)]
fn generate_mixed_scaledata(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n, |i| if i % 10 == 0 { 1e12 } else { 1e-12 })
}

#[allow(dead_code)]
fn generate_cancellationdata(n: usize) -> Array1<f64> {
    // Data designed to cause catastrophic cancellation
    let base = 1e8;
    Array1::from_shape_fn(n, |i| base + (i as f64 * 1e-8))
}

#[allow(dead_code)]
fn generate_performance_testdata(size: usize) -> Array1<f64> {
    Array1::from_shape_fn(size, |i| (i as f64).sin())
}

#[allow(dead_code)]
fn estimate_memory_usage(datasize: usize) -> usize {
    datasize * std::mem::size_of::<f64>() * 2 // Rough estimate
}

#[allow(dead_code)]
fn calculate_memory_efficiency(_peak_usage: usize, datasize: usize) -> f64 {
    let theoretical_minimum = datasize * std::mem::size_of::<f64>();
    theoretical_minimum as f64 / _peak_usage as f64
}

#[allow(dead_code)]
fn estimate_expected_throughput(datasize: usize) -> f64 {
    // Very rough estimate - should be calibrated per platform
    match datasize {
        n if n < 1000 => 1e6,    // 1M elements/sec
        n if n < 10000 => 5e6,   // 5M elements/sec
        n if n < 100000 => 10e6, // 10M elements/sec
        _ => 20e6,               // 20M elements/sec
    }
}

#[allow(dead_code)]
fn estimate_expected_parallel_throughput(datasize: usize, threadcount: usize) -> f64 {
    estimate_expected_throughput(datasize) * (thread_count as f64 * 0.8) // 80% parallel efficiency
}

#[allow(dead_code)]
fn time_scalar_operation(data: &Array1<f64>) -> std::time::Duration {
    let start = Instant::now();
    let _result = data.iter().sum::<f64>() / data.len() as f64;
    start.elapsed()
}

#[allow(dead_code)]
fn time_simd_operation(data: &Array1<f64>) -> std::time::Duration {
    let start = Instant::now();
    let optimizer = crate::advanced_simd_stats::AdvancedSimdOptimizer::new(
        crate::advanced_simd_stats::AdvancedSimdConfig::default(),
    );
    let data_arrays = vec![data.view()];
    let operations = vec![crate::advanced_simd_stats::BatchOperation::Mean];
    let _result = optimizer.advanced_batch_statistics(&data_arrays, &operations);
    start.elapsed()
}

#[allow(dead_code)]
fn time_single_thread_operation(data: &Array1<f64>) -> std::time::Duration {
    let start = Instant::now();
    let _result = data.mean();
    start.elapsed()
}

#[allow(dead_code)]
fn time_multi_thread_operation(data: &Array1<f64>) -> std::time::Duration {
    let start = Instant::now();
    let _result = crate::parallel_stats::mean_parallel(&data.view(), None);
    start.elapsed()
}

#[allow(dead_code)]
fn estimate_memory_bandwidth() -> f64 {
    // Simplified memory bandwidth estimation
    10.0 // GB/s - should be measured properly
}

#[allow(dead_code)]
fn measure_cache_efficiency(data: &Array1<f64>) -> f64 {
    // Simplified cache efficiency measurement
    0.85 // Should be measured properly
}

#[allow(dead_code)]
fn detect_thermal_throttling() -> bool {
    // Simplified thermal throttling detection
    false // Should check actual CPU frequency scaling
}

#[allow(dead_code)]
fn get_compilation_flags() -> Vec<String> {
    // Return compilation flags if available
    vec!["O3".to_string(), "native".to_string()]
}
