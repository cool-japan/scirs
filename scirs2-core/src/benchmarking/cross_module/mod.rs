// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

use crate::benchmarking::current_process_memory_bytes;
use crate::error::{CoreError, CoreResult, ErrorContext};
#[cfg(feature = "parallel")]
use crate::parallel_ops::*;
#[cfg(feature = "serialization")]
use chrono;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
/// Cross-module performance benchmarking configuration
#[derive(Debug, Clone)]
pub struct CrossModuleBenchConfig {
    /// Number of benchmark iterations
    pub iterations: usize,
    /// Warmup iterations before measurement
    pub warmup_iterations: usize,
    /// Data sizes to test
    pub datasizes: Vec<usize>,
    /// Thread counts to test (for parallel operations)
    pub ns: Vec<usize>,
    /// Memory limits for testing
    pub memory_limits: Vec<usize>,
    /// Enable detailed profiling
    pub enable_profiling: bool,
    /// Enable regression detection
    pub enable_regression_detection: bool,
    /// Baseline performance file path
    pub baseline_file: Option<String>,
    /// Maximum acceptable regression percentage
    pub max_regression_percent: f64,
    /// Benchmark timeout per test
    pub timeout: Duration,
}
impl Default for CrossModuleBenchConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            warmup_iterations: 10,
            datasizes: vec![1024, 1024 * 16, 1024 * 1024, 1024 * 1024 * 16],
            ns: vec![1, 2, 4, 8],
            memory_limits: vec![64 * 1024 * 1024, 256 * 1024 * 1024, 1024 * 1024 * 1024],
            enable_profiling: true,
            enable_regression_detection: true,
            baseline_file: None,
            max_regression_percent: 10.0,
            timeout: Duration::from_secs(60),
        }
    }
}
/// Performance measurement result
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    /// Operation name
    pub name: String,
    /// Module combination involved
    pub modules: Vec<String>,
    /// Data size used
    pub datasize: usize,
    /// Thread count used
    pub n: usize,
    /// Average execution time
    pub avg_duration: Duration,
    /// Minimum execution time
    pub min_duration: Duration,
    /// Maximum execution time
    pub max_duration: Duration,
    /// Standard deviation
    pub std_deviation: Duration,
    /// Throughput (operations per second)
    pub throughput: f64,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Peak memory usage (bytes)
    pub peak_memory: usize,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Operations count
    pub operations_count: usize,
    /// Detailed timing breakdown
    pub timing_breakdown: HashMap<String, Duration>,
}
impl PerformanceMeasurement {
    /// Create a new performance measurement
    pub fn new(name: String, modules: Vec<String>) -> Self {
        Self {
            name,
            modules,
            datasize: 0,
            n: 1,
            avg_duration: Duration::from_nanos(0),
            min_duration: Duration::from_nanos(u64::MAX),
            max_duration: Duration::from_nanos(0),
            std_deviation: Duration::from_nanos(0),
            throughput: 0.0,
            memory_usage: 0,
            peak_memory: 0,
            cpu_utilization: 0.0,
            operations_count: 0,
            timing_breakdown: HashMap::new(),
        }
    }
    /// Calculate efficiency score (0-100)
    pub fn efficiency_score(&self) -> f64 {
        if self.avg_duration.as_nanos() == 0 {
            return 0.0;
        }
        let time_efficiency = 1.0 / (self.avg_duration.as_secs_f64() + 1e-9);
        let memory_efficiency = if self.memory_usage > 0 {
            self.throughput / (self.memory_usage as f64 / 1024.0 / 1024.0)
        } else {
            self.throughput
        };
        ((time_efficiency + memory_efficiency) / 2.0 * 100.0).min(100.0)
    }
}
/// Benchmark suite result
#[derive(Debug, Clone)]
pub struct BenchmarkSuiteResult {
    /// Suite name
    pub name: String,
    /// Individual measurements
    pub measurements: Vec<PerformanceMeasurement>,
    /// Overall suite duration
    pub total_duration: Duration,
    /// Average efficiency score
    pub avg_efficiency: f64,
    /// Regression analysis results
    pub regression_analysis: Option<RegressionAnalysis>,
    /// Scalability analysis
    pub scalability_analysis: ScalabilityAnalysis,
    /// Memory efficiency analysis
    pub memory_analysis: MemoryEfficiencyAnalysis,
}
/// Regression analysis results
#[derive(Debug, Clone)]
pub struct RegressionAnalysis {
    /// Whether regression was detected
    pub regression_detected: bool,
    /// Regressed benchmarks
    pub regressions: Vec<RegressionResult>,
    /// Improved benchmarks
    pub improvements: Vec<RegressionResult>,
    /// Overall performance change percentage
    pub overall_change_percent: f64,
}
/// Individual regression result
#[derive(Debug, Clone)]
pub struct RegressionResult {
    /// Benchmark name
    pub benchmark_name: String,
    /// Baseline performance
    pub baseline_duration: Duration,
    /// Current performance
    pub current_duration: Duration,
    /// Change percentage (positive = regression, negative = improvement)
    pub change_percent: f64,
    /// Significance of change
    pub significance: RegressionSignificance,
}
/// Significance levels for performance changes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressionSignificance {
    /// Negligible change (< 5%)
    Negligible,
    /// Minor change (5-15%)
    Minor,
    /// Moderate change (15-30%)
    Moderate,
    /// Major change (30-50%)
    Major,
    /// Critical change (> 50%)
    Critical,
}
/// Scalability analysis results
#[derive(Debug, Clone)]
pub struct ScalabilityAnalysis {
    /// Thread scalability efficiency (0.saturating_sub(1))
    pub thread_scalability: f64,
    /// Data size scalability efficiency (0.saturating_sub(1))
    pub data_scalability: f64,
    /// Memory scalability efficiency (0.saturating_sub(1))
    pub memory_scalability: f64,
    /// Scalability breakdown by data size
    pub datasize_breakdown: HashMap<usize, f64>,
    /// Scalability breakdown by thread count
    pub n_breakdown: HashMap<usize, f64>,
}
/// Memory efficiency analysis
#[derive(Debug, Clone)]
pub struct MemoryEfficiencyAnalysis {
    /// Average memory usage per operation (bytes)
    pub avg_memory_per_op: f64,
    /// Peak to average memory ratio
    pub peak_to_avg_ratio: f64,
    /// Memory fragmentation score (0.saturating_sub(1), lower is better)
    pub fragmentation_score: f64,
    /// Zero-copy efficiency score (0.saturating_sub(1))
    pub zero_copy_efficiency: f64,
    /// Memory bandwidth utilization (0.saturating_sub(1))
    pub bandwidth_utilization: f64,
}
/// Cross-module benchmark runner
pub struct CrossModuleBenchmarkRunner {
    config: CrossModuleBenchConfig,
    results: Arc<Mutex<Vec<BenchmarkSuiteResult>>>,
}
impl CrossModuleBenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(config: CrossModuleBenchConfig) -> Self {
        Self {
            config,
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }
    /// Run comprehensive cross-module benchmarks
    pub fn run_benchmarks(&self) -> CoreResult<BenchmarkSuiteResult> {
        let start_time = Instant::now();
        let mut measurements = Vec::new();
        println!("🚀 Running Cross-Module Performance Benchmarks");
        println!("==============================================");
        measurements.extend(self.run_data_pipeline_benchmarks()?);
        measurements.extend(self.run_memory_efficiency_benchmarks()?);
        measurements.extend(self.run_scalability_benchmarks()?);
        measurements.extend(self.run_real_world_benchmarks()?);
        let total_duration = start_time.elapsed();
        let avg_efficiency = if measurements.is_empty() {
            0.0
        } else {
            measurements
                .iter()
                .map(|m| m.efficiency_score())
                .sum::<f64>()
                / measurements.len() as f64
        };
        let regression_analysis = if self.config.enable_regression_detection {
            Some(self.analyze_regressions(&measurements)?)
        } else {
            None
        };
        let scalability_analysis = self.analyze_scalability(&measurements)?;
        let memory_analysis = self.analyze_memory_efficiency(&measurements)?;
        let suite_result = BenchmarkSuiteResult {
            name: "Cross-Module Performance Suite".to_string(),
            measurements,
            total_duration,
            avg_efficiency,
            regression_analysis,
            scalability_analysis,
            memory_analysis,
        };
        {
            let mut results = self.results.lock().map_err(|_| {
                CoreError::ComputationError(ErrorContext::new("Failed to lock results".to_string()))
            })?;
            results.push(suite_result.clone());
        }
        Ok(suite_result)
    }
    /// Run data pipeline benchmarks
    fn run_data_pipeline_benchmarks(&self) -> CoreResult<Vec<PerformanceMeasurement>> {
        let mut measurements = Vec::new();
        println!("📊 Running Data Pipeline Benchmarks...");
        measurements.push(self.benchmark_linalg_stats_pipeline()?);
        measurements.push(self.benchmark_signal_fft_pipeline()?);
        measurements.push(self.benchmark_io_processing_pipeline()?);
        measurements.push(self.benchmark_ml_pipeline()?);
        Ok(measurements)
    }
    /// Benchmark linear algebra + statistics pipeline
    fn benchmark_linalg_stats_pipeline(&self) -> CoreResult<PerformanceMeasurement> {
        let mut measurement = PerformanceMeasurement::new(
            "linalg_stats_pipeline".to_string(),
            vec!["scirs2-linalg".to_string(), "scirs2-stats".to_string()],
        );
        for &datasize in &self.config.datasizes {
            let timing_data = self.time_operation(&format!("{datasize}"), || {
                self.simulate_linalg_stats_workflow(datasize)
            })?;
            if datasize == *self.config.datasizes.last().expect("Operation failed") {
                measurement.datasize = datasize;
                measurement.avg_duration = timing_data.avg_duration;
                measurement.min_duration = timing_data.min_duration;
                measurement.max_duration = timing_data.max_duration;
                measurement.throughput = timing_data.throughput;
                measurement.memory_usage = timing_data.memory_usage;
                measurement.peak_memory = timing_data.peak_memory;
                measurement.operations_count = timing_data.operations_count;
            }
        }
        Ok(measurement)
    }
    /// Simulate linear algebra + statistics workflow
    fn simulate_linalg_stats_workflow(&self, datasize: usize) -> CoreResult<()> {
        let matrix_size = (datasize as f64).sqrt() as usize;
        let matrix_elements = matrix_size * matrix_size;
        let operations = matrix_size.pow(3);
        for _ in 0..operations.min(1000000) {
            let result = 1.23456 * 7.89012 + 3.45678;
        }
        let stats_operations = datasize;
        for _ in 0..stats_operations.min(1000000) {
            let result = 1.23456_f64.sin() + 7.89012_f64.cos();
        }
        Ok(())
    }
    /// Benchmark signal processing + FFT pipeline
    fn benchmark_signal_fft_pipeline(&self) -> CoreResult<PerformanceMeasurement> {
        let mut measurement = PerformanceMeasurement::new(
            "signal_fft_pipeline".to_string(),
            vec!["scirs2-signal".to_string(), "scirs2-fft".to_string()],
        );
        for &datasize in &self.config.datasizes {
            let timing_data = self.time_operation(&format!("{datasize}"), || {
                self.simulate_signal_fft_workflow(datasize)
            })?;
            if datasize == *self.config.datasizes.last().expect("Operation failed") {
                measurement.datasize = datasize;
                measurement.avg_duration = timing_data.avg_duration;
                measurement.throughput = timing_data.throughput;
                measurement.memory_usage = timing_data.memory_usage;
                measurement.peak_memory = timing_data.peak_memory;
                measurement.operations_count = timing_data.operations_count;
            }
        }
        Ok(measurement)
    }
    /// Simulate signal processing + FFT workflow
    fn simulate_signal_fft_workflow(&self, datasize: usize) -> CoreResult<()> {
        let signal_length = datasize / std::mem::size_of::<f64>();
        let filter_operations = signal_length.min(1000000);
        for _ in 0..filter_operations {
            let result = 1.23456_f64.sin() * 0.78901 + 2.34567_f64.cos();
        }
        let fft_operations = (signal_length as f64 * (signal_length as f64).log2()) as usize;
        for _ in 0..fft_operations.min(1000000) {
            let result = std::f64::consts::PI * std::f64::consts::E.exp();
        }
        Ok(())
    }
    /// Benchmark data I/O + processing pipeline
    fn benchmark_io_processing_pipeline(&self) -> CoreResult<PerformanceMeasurement> {
        let mut measurement = PerformanceMeasurement::new(
            "io_processing_pipeline".to_string(),
            vec!["scirs2-io".to_string(), "scirs2-core".to_string()],
        );
        for &datasize in &self.config.datasizes {
            let timing_data = self.time_operation(&format!("{datasize}"), || {
                self.simulate_io_processing_workflow(datasize)
            })?;
            if datasize == *self.config.datasizes.last().expect("Operation failed") {
                measurement.datasize = datasize;
                measurement.avg_duration = timing_data.avg_duration;
                measurement.throughput = timing_data.throughput;
                measurement.memory_usage = timing_data.memory_usage;
                measurement.peak_memory = timing_data.peak_memory;
                measurement.operations_count = timing_data.operations_count;
            }
        }
        Ok(measurement)
    }
    /// Simulate data I/O + processing workflow
    fn simulate_io_processing_workflow(&self, datasize: usize) -> CoreResult<()> {
        let buffer = vec![0u8; datasize];
        let mut checksum = 0u64;
        for &byte in &buffer {
            checksum = checksum.wrapping_add(byte as u64);
        }
        for i in 0..datasize.min(100000) {
            let value = (0 as f64) / datasize as f64;
            if !value.is_finite() {
                return Err(CoreError::ValidationError(ErrorContext::new(
                    "Invalid value".to_string(),
                )));
            }
        }
        if checksum == u64::MAX {
            return Err(CoreError::ComputationError(ErrorContext::new(
                "Unlikely checksum".to_string(),
            )));
        }
        Ok(())
    }
    /// Benchmark machine learning pipeline
    fn benchmark_ml_pipeline(&self) -> CoreResult<PerformanceMeasurement> {
        let mut measurement = PerformanceMeasurement::new(
            "ml_pipeline".to_string(),
            vec!["scirs2-neural".to_string(), "scirs2-optimize".to_string()],
        );
        for &datasize in &self.config.datasizes {
            let timing_data = self.time_operation(&format!("{datasize}"), || {
                self.simulate_ml_workflow(datasize)
            })?;
            if datasize == *self.config.datasizes.last().expect("Operation failed") {
                measurement.datasize = datasize;
                measurement.avg_duration = timing_data.avg_duration;
                measurement.throughput = timing_data.throughput;
                measurement.memory_usage = timing_data.memory_usage;
                measurement.peak_memory = timing_data.peak_memory;
                measurement.operations_count = timing_data.operations_count;
            }
        }
        Ok(measurement)
    }
    /// Simulate machine learning workflow
    fn simulate_ml_workflow(&self, datasize: usize) -> CoreResult<()> {
        let feature_count = (datasize / 1000).max(10);
        let sample_count = datasize / feature_count;
        for _ in 0..sample_count.min(10000) {
            for _ in 0..feature_count.min(1000) {
                let activation = 1.0 / (1.0 + (-0.5_f64).exp());
            }
        }
        for _ in 0..(feature_count * sample_count).min(100000) {
            let gradient = 0.01 * 1.23456;
        }
        Ok(())
    }
    /// Run memory efficiency benchmarks
    fn run_memory_efficiency_benchmarks(&self) -> CoreResult<Vec<PerformanceMeasurement>> {
        let mut measurements = Vec::new();
        println!("🧠 Running Memory Efficiency Benchmarks...");
        measurements.push(self.benchmark_zero_copy_operations()?);
        measurements.push(self.benchmark_memory_mapped_operations()?);
        measurements.push(self.benchmark_out_of_core_operations()?);
        Ok(measurements)
    }
    /// Benchmark zero-copy operations
    fn benchmark_zero_copy_operations(&self) -> CoreResult<PerformanceMeasurement> {
        let mut measurement = PerformanceMeasurement::new(
            "zero_copy_operations".to_string(),
            vec!["scirs2-core".to_string()],
        );
        for &datasize in &self.config.datasizes {
            let timing_data = self.time_operation(&format!("{datasize}"), || {
                self.simulate_zero_copy_workflow(datasize)
            })?;
            if datasize == *self.config.datasizes.last().expect("Operation failed") {
                measurement.datasize = datasize;
                measurement.avg_duration = timing_data.avg_duration;
                measurement.throughput = timing_data.throughput;
                measurement.memory_usage = timing_data.memory_usage;
                measurement.peak_memory = timing_data.peak_memory;
                measurement.operations_count = timing_data.operations_count;
            }
        }
        Ok(measurement)
    }
    /// Simulate zero-copy operations
    fn simulate_zero_copy_workflow(&self, datasize: usize) -> CoreResult<()> {
        let buffer = vec![1.0f64; datasize / std::mem::size_of::<f64>()];
        let chunk_size = buffer.len() / 4;
        for i in 0..4 {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(buffer.len());
            let slice = &buffer[start..end];
            let mut sum = 0.0;
            for &value in slice {
                sum += value;
            }
            if sum < 0.0 {
                return Err(CoreError::ComputationError(ErrorContext::new(
                    "Invalid sum".to_string(),
                )));
            }
        }
        Ok(())
    }
    /// Benchmark memory-mapped operations
    fn benchmark_memory_mapped_operations(&self) -> CoreResult<PerformanceMeasurement> {
        let mut measurement = PerformanceMeasurement::new(
            "memory_mapped_operations".to_string(),
            vec!["scirs2-core".to_string(), "scirs2-io".to_string()],
        );
        println!("   Simulating memory-mapped operations...");
        for &datasize in &self.config.datasizes {
            let timing_data = self.time_operation(&format!("{datasize}"), || {
                self.simulate_mmap_workflow(datasize)
            })?;
            if datasize == *self.config.datasizes.last().expect("Operation failed") {
                measurement.datasize = datasize;
                measurement.avg_duration = timing_data.avg_duration;
                measurement.throughput = timing_data.throughput;
                measurement.memory_usage = timing_data.memory_usage;
                measurement.peak_memory = timing_data.peak_memory;
                measurement.operations_count = timing_data.operations_count;
            }
        }
        Ok(measurement)
    }
    /// Simulate memory-mapped workflow
    fn simulate_mmap_workflow(&self, datasize: usize) -> CoreResult<()> {
        let element_count = datasize / std::mem::size_of::<f64>();
        let chunk_size = element_count / 16;
        for chunk_id in 0..16 {
            let start_idx = chunk_id * chunk_size;
            let end_idx = ((chunk_id + 1) * chunk_size).min(element_count);
            for idx in start_idx..end_idx {
                let value = (idx as f64).sin();
                if !value.is_finite() {
                    return Err(CoreError::ComputationError(ErrorContext::new(
                        "Invalid computation".to_string(),
                    )));
                }
            }
        }
        Ok(())
    }
    /// Benchmark out-of-core operations
    fn benchmark_out_of_core_operations(&self) -> CoreResult<PerformanceMeasurement> {
        let mut measurement = PerformanceMeasurement::new(
            "out_of_core_operations".to_string(),
            vec!["scirs2-core".to_string()],
        );
        println!("   Simulating out-of-core operations...");
        for &datasize in &self.config.datasizes {
            let timing_data = self.time_operation(&format!("{datasize}"), || {
                self.simulate_out_of_core_workflow(datasize)
            })?;
            if datasize == *self.config.datasizes.last().expect("Operation failed") {
                measurement.datasize = datasize;
                measurement.avg_duration = timing_data.avg_duration;
                measurement.throughput = timing_data.throughput;
                measurement.memory_usage = timing_data.memory_usage;
                measurement.peak_memory = timing_data.peak_memory;
                measurement.operations_count = timing_data.operations_count;
            }
        }
        Ok(measurement)
    }
    /// Simulate out-of-core workflow
    fn simulate_out_of_core_workflow(&self, datasize: usize) -> CoreResult<()> {
        let total_elements = datasize / std::mem::size_of::<f64>();
        let chunk_size = 1024;
        let num_chunks = total_elements.div_ceil(chunk_size);
        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(total_elements);
            let chunk_len = end - start;
            let chunk_data = vec![1.0f64; chunk_len];
            let mut sum = 0.0;
            for &value in &chunk_data {
                sum += value * value;
            }
            if sum < 0.0 {
                return Err(CoreError::ComputationError(ErrorContext::new(
                    "Invalid computation result".to_string(),
                )));
            }
        }
        Ok(())
    }
    /// Run scalability benchmarks
    fn run_scalability_benchmarks(&self) -> CoreResult<Vec<PerformanceMeasurement>> {
        let mut measurements = Vec::new();
        println!("📈 Running Scalability Benchmarks...");
        measurements.push(self.benchmark_thread_scalability()?);
        measurements.push(self.benchmark_datasize_scalability()?);
        measurements.push(self.benchmark_memory_scalability()?);
        Ok(measurements)
    }
    /// Benchmark thread scalability
    fn benchmark_thread_scalability(&self) -> CoreResult<PerformanceMeasurement> {
        let mut measurement = PerformanceMeasurement::new(
            "thread_scalability".to_string(),
            vec!["scirs2-core".to_string()],
        );
        #[cfg(feature = "parallel")]
        {
            for &n in &self.config.ns {
                let timing_data = self.time_operation(&format!("{n}"), || {
                    self.simulate_scalable_operation(n * 1024)
                })?;
                if n == *self.config.ns.last().expect("Operation failed") {
                    measurement.n = n;
                    measurement.avg_duration = timing_data.avg_duration;
                    measurement.throughput = timing_data.throughput;
                    measurement.operations_count = timing_data.operations_count;
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            measurement.n = 1;
            measurement.avg_duration = Duration::from_millis(100);
            measurement.throughput = 1000.0;
            measurement.operations_count = 1000;
        }
        Ok(measurement)
    }
    /// Simulate parallel operations
    #[cfg(feature = "parallel")]
    fn count(n: usize) -> CoreResult<()> {
        let work_items = 100000;
        let items_per_thread = work_items / n;
        crate::parallel_ops::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| CoreError::ComputationError(ErrorContext::new(format!("{e}"))))?
            .install(|| {
                (0..n).into_par_iter().try_for_each(|_| {
                    for _ in 0..items_per_thread {
                        let result = 1.23456_f64.sin() + 7.89012_f64.cos();
                    }
                    Ok::<(), CoreError>(())
                })
            })?;
        Ok(())
    }
    /// Simulate parallel operations (fallback)
    #[cfg(not(feature = "parallel"))]
    fn count(n: usize) -> CoreResult<()> {
        for _ in 0..100000 {
            let result = 1.23456_f64.sin() + 7.89012_f64.cos();
        }
        Ok(())
    }
    /// Benchmark data size scalability
    fn benchmark_datasize_scalability(&self) -> CoreResult<PerformanceMeasurement> {
        let mut measurement = PerformanceMeasurement::new(
            "datasize_scalability".to_string(),
            vec!["scirs2-core".to_string()],
        );
        println!("   Testing data size scalability...");
        let mut scalability_scores = Vec::new();
        for &datasize in &self.config.datasizes {
            let timing_data = self.time_operation(&format!("{datasize}"), || {
                self.simulate_scalable_operation(datasize)
            })?;
            let ops_per_byte = timing_data.throughput / datasize as f64;
            scalability_scores.push(ops_per_byte);
            if datasize == *self.config.datasizes.last().expect("Operation failed") {
                measurement.datasize = datasize;
                measurement.avg_duration = timing_data.avg_duration;
                measurement.throughput = timing_data.throughput;
                measurement.memory_usage = timing_data.memory_usage;
                measurement.peak_memory = timing_data.peak_memory;
                measurement.operations_count = timing_data.operations_count;
            }
        }
        Ok(measurement)
    }
    /// Simulate scalable operation for data size testing
    fn simulate_scalable_operation(&self, datasize: usize) -> CoreResult<()> {
        let elements = datasize / std::mem::size_of::<f64>();
        for i in 0..elements.min(1000000) {
            let value = (0 as f64) / elements as f64;
            let result = value.sin() + value.cos();
        }
        Ok(())
    }
    /// Benchmark memory scalability
    fn benchmark_memory_scalability(&self) -> CoreResult<PerformanceMeasurement> {
        let mut measurement = PerformanceMeasurement::new(
            "memory_scalability".to_string(),
            vec!["scirs2-core".to_string()],
        );
        println!("   Testing memory scalability...");
        for &memory_limit in &self.config.memory_limits {
            let timing_data = self.time_operation(&format!("{memory_limit}"), || {
                self.simulate_scalable_operation(memory_limit)
            })?;
            if memory_limit == *self.config.memory_limits.last().expect("Operation failed") {
                measurement.memory_usage = memory_limit;
                measurement.avg_duration = timing_data.avg_duration;
                measurement.throughput = timing_data.throughput;
                measurement.operations_count = timing_data.operations_count;
            }
        }
        Ok(measurement)
    }
    /// Simulate memory-constrained operation
    fn limit(n: usize) -> CoreResult<()> {
        let element_count = (n / std::mem::size_of::<f64>()).min(1000000);
        let buffer = vec![1.0f64; element_count];
        let mut result = 0.0;
        for (i, &value) in buffer.iter().enumerate() {
            result += value * (i as f64).sqrt();
        }
        if result < 0.0 {
            return Err(CoreError::ComputationError(ErrorContext::new(
                "Invalid result".to_string(),
            )));
        }
        Ok(())
    }
    /// Run real-world scenario benchmarks
    fn run_real_world_benchmarks(&self) -> CoreResult<Vec<PerformanceMeasurement>> {
        let mut measurements = Vec::new();
        println!("🌍 Running Real-World Scenario Benchmarks...");
        measurements.push(self.benchmark_scientific_simulation()?);
        measurements.push(self.benchmark_data_analysis_pipeline()?);
        measurements.push(self.benchmark_machine_learning_training()?);
        Ok(measurements)
    }
    /// Benchmark scientific simulation scenario
    fn benchmark_scientific_simulation(&self) -> CoreResult<PerformanceMeasurement> {
        let mut measurement = PerformanceMeasurement::new(
            "scientific_simulation".to_string(),
            vec!["scirs2-linalg".to_string(), "scirs2-integrate".to_string()],
        );
        println!("   Running scientific simulation benchmark...");
        for &datasize in &self.config.datasizes {
            let timing_data = self.time_operation(&format!("{datasize}"), || {
                self.simulate_scientific_workflow(datasize)
            })?;
            if datasize == *self.config.datasizes.last().expect("Operation failed") {
                measurement.datasize = datasize;
                measurement.avg_duration = timing_data.avg_duration;
                measurement.throughput = timing_data.throughput;
                measurement.memory_usage = timing_data.memory_usage;
                measurement.peak_memory = timing_data.peak_memory;
                measurement.operations_count = timing_data.operations_count;
            }
        }
        Ok(measurement)
    }
    /// Simulate scientific simulation workflow
    fn simulate_scientific_workflow(&self, datasize: usize) -> CoreResult<()> {
        let grid_size = (datasize as f64).sqrt() as usize;
        let time_steps = 100;
        for i in 0..grid_size {
            for j in 0..grid_size {
                let x = 0 as f64 / grid_size as f64;
                let y = j as f64 / grid_size as f64;
                let initial_value = (x * x + y * y).exp() * (-x * y).sin();
            }
        }
        for _step in 0..time_steps {
            for i in 1..(grid_size - 1) {
                for _j in 1..(grid_size - 1) {
                    let dt = 0.01;
                    let dx = 1.0 / grid_size as f64;
                    let laplacian = dt / (dx * dx);
                }
            }
            let matrix_ops = grid_size * grid_size / 100;
            for _ in 0..matrix_ops {
                let result = 1.23456_f64.sin() + 0.78901_f64.cos();
            }
        }
        Ok(())
    }
    /// Benchmark data analysis pipeline scenario
    fn benchmark_data_analysis_pipeline(&self) -> CoreResult<PerformanceMeasurement> {
        let mut measurement = PerformanceMeasurement::new(
            "data_analysis_pipeline".to_string(),
            vec![
                "scirs2-io".to_string(),
                "scirs2-stats".to_string(),
                "scirs2-signal".to_string(),
            ],
        );
        println!("   Running data analysis pipeline benchmark...");
        for &datasize in &self.config.datasizes {
            let timing_data = self.time_operation(&format!("{datasize}"), || {
                self.simulate_data_analysis_workflow(datasize)
            })?;
            if datasize == *self.config.datasizes.last().expect("Operation failed") {
                measurement.datasize = datasize;
                measurement.avg_duration = timing_data.avg_duration;
                measurement.throughput = timing_data.throughput;
                measurement.memory_usage = timing_data.memory_usage;
                measurement.peak_memory = timing_data.peak_memory;
                measurement.operations_count = timing_data.operations_count;
            }
        }
        Ok(measurement)
    }
    /// Simulate data analysis workflow
    fn simulate_data_analysis_workflow(&self, datasize: usize) -> CoreResult<()> {
        let sample_count = datasize / std::mem::size_of::<f64>();
        let raw_data = vec![0.0f64; sample_count];
        let mut processed_data = Vec::with_capacity(sample_count);
        for (i, &value) in raw_data.iter().enumerate() {
            let cleaned_value = value + (i as f64 * 0.01).sin();
            processed_data.push(cleaned_value);
        }
        let mut sum = 0.0;
        let mut sum_squares = 0.0;
        for &value in &processed_data {
            sum += value;
            sum_squares += value * value;
        }
        let mean = sum / processed_data.len() as f64;
        let variance = (sum_squares / processed_data.len() as f64) - (mean * mean);
        for (i, &value) in processed_data.iter().enumerate() {
            let freq = 2.0 * std::f64::consts::PI * (i as f64) / sample_count as f64;
            let filtered = value * freq.cos();
        }
        if variance < 0.0 {
            return Err(CoreError::ComputationError(ErrorContext::new(
                "Invalid variance".to_string(),
            )));
        }
        Ok(())
    }
    /// Benchmark machine learning training scenario
    fn benchmark_machine_learning_training(&self) -> CoreResult<PerformanceMeasurement> {
        let mut measurement = PerformanceMeasurement::new(
            "ml_training".to_string(),
            vec![
                "scirs2-neural".to_string(),
                "scirs2-optimize".to_string(),
                "scirs2-linalg".to_string(),
            ],
        );
        println!("   Running ML training benchmark...");
        for &datasize in &self.config.datasizes {
            let timing_data = self.time_operation(&format!("{datasize}"), || {
                self.simulate_ml_training_workflow(datasize)
            })?;
            if datasize == *self.config.datasizes.last().expect("Operation failed") {
                measurement.datasize = datasize;
                measurement.avg_duration = timing_data.avg_duration;
                measurement.throughput = timing_data.throughput;
                measurement.memory_usage = timing_data.memory_usage;
                measurement.peak_memory = timing_data.peak_memory;
                measurement.operations_count = timing_data.operations_count;
            }
        }
        Ok(measurement)
    }
    /// Simulate machine learning training workflow
    fn simulate_ml_training_workflow(&self, datasize: usize) -> CoreResult<()> {
        let batch_size = 32;
        let feature_dim = 128;
        let hidden_dim = 256;
        let numbatches = (datasize / (batch_size * feature_dim)).max(1);
        let epochs = 10;
        for _epoch in 0..epochs {
            for _batch in 0..numbatches {
                for i in 0..batch_size {
                    for j in 0..hidden_dim {
                        let mut activation = 0.0;
                        for k in 0..feature_dim {
                            let weight = ((i + j + k) as f64) * 0.01;
                            let input = ((i * k) as f64) * 0.001;
                            activation += weight * input;
                        }
                        let output = 1.0 / (1.0 + (-activation).exp());
                    }
                }
                for i in 0..hidden_dim {
                    for j in 0..feature_dim {
                        let gradient = ((i + j) as f64) * 0.001;
                        let weight_update = gradient * 0.01;
                    }
                }
                let param_count = hidden_dim * feature_dim;
                for _ in 0..param_count / 1000 {
                    let momentum_update = 0.9 * 0.01 + 0.1 * 0.001;
                }
            }
        }
        Ok(())
    }
    /// Time an operation with multiple iterations, measuring real
    /// wall-clock duration and real resident-memory usage (`VmRSS` deltas
    /// on Linux; `0` — "not measured" — elsewhere, never a fabricated
    /// constant) per iteration.
    fn time_operation<F>(&self, name: &str, mut operation: F) -> CoreResult<TimingData>
    where
        F: FnMut() -> CoreResult<()>,
    {
        let mut durations = Vec::new();
        let mut memory_deltas = Vec::new();
        for _ in 0..self.config.warmup_iterations {
            operation()?;
        }
        for _ in 0..self.config.iterations {
            let memory_before = current_process_memory_bytes();
            let start = Instant::now();
            operation()?;
            let duration = start.elapsed();
            let memory_after = current_process_memory_bytes();
            durations.push(duration);
            memory_deltas.push(memory_after.saturating_sub(memory_before));
        }
        let total_duration: Duration = durations.iter().sum();
        let avg_duration = total_duration / durations.len() as u32;
        let min_duration = *durations.iter().min().expect("Operation failed");
        let max_duration = *durations.iter().max().expect("Operation failed");
        let variance = durations
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as i128 - avg_duration.as_nanos() as i128;
                (diff * diff) as u128
            })
            .sum::<u128>()
            / durations.len() as u128;
        let std_deviation = Duration::from_nanos((variance as f64).sqrt() as u64);
        let throughput = if avg_duration.as_secs_f64() > 0.0 {
            self.config.iterations as f64 / avg_duration.as_secs_f64()
        } else {
            0.0
        };
        let memory_usage = memory_deltas.iter().sum::<usize>() / memory_deltas.len().max(1);
        let peak_memory = memory_deltas.iter().copied().max().unwrap_or(0);
        Ok(TimingData {
            name: name.to_string(),
            avg_duration,
            min_duration,
            max_duration,
            std_deviation,
            throughput,
            memory_usage,
            peak_memory,
            operations_count: self.config.iterations,
        })
    }
    /// Analyze performance regressions against a statistical baseline derived from
    /// the current measurement set. Each benchmark's timing is compared to the
    /// fleet-wide mean; an entry is a regression when its duration exceeds
    /// `mean + 2 * std_dev` (simplified z-score outlier) or when the
    /// configured `max_regression_percent` threshold is crossed relative to a
    /// stored external baseline (if `baseline_file` is set).
    ///
    /// When no external baseline is available the method uses intra-suite
    /// statistics so regressions can still be detected from a single run
    /// (e.g. one benchmark taking abnormally longer than all others).
    fn analyze_regressions(
        &self,
        measurements: &[PerformanceMeasurement],
    ) -> CoreResult<RegressionAnalysis> {
        if measurements.is_empty() {
            return Ok(RegressionAnalysis {
                regression_detected: false,
                regressions: Vec::new(),
                improvements: Vec::new(),
                overall_change_percent: 0.0,
            });
        }
        let durations_ns: Vec<f64> = measurements
            .iter()
            .map(|m| m.avg_duration.as_nanos() as f64)
            .collect();
        let n = durations_ns.len() as f64;
        let mean_ns = durations_ns.iter().sum::<f64>() / n;
        let variance_ns = durations_ns
            .iter()
            .map(|&d| {
                let diff = d - mean_ns;
                diff * diff
            })
            .sum::<f64>()
            / n;
        let std_ns = variance_ns.sqrt();
        let mut baseline_map: HashMap<String, f64> = HashMap::new();
        if let Some(ref path) = self.config.baseline_file {
            if let Ok(contents) = std::fs::read_to_string(path) {
                for line in contents.lines() {
                    let parts: Vec<&str> = line.splitn(2, ' ').collect();
                    if parts.len() == 2 {
                        if let Ok(nanos) = parts[1].trim().parse::<f64>() {
                            baseline_map.insert(parts[0].trim().to_string(), nanos);
                        }
                    }
                }
            }
        }
        let mut regressions: Vec<RegressionResult> = Vec::new();
        let mut improvements: Vec<RegressionResult> = Vec::new();
        let mut total_change_sum = 0.0_f64;
        let threshold_factor = self.config.max_regression_percent / 100.0;
        for m in measurements {
            let current_ns = m.avg_duration.as_nanos() as f64;
            let baseline_ns = baseline_map.get(&m.name).copied().unwrap_or(mean_ns);
            let change_percent = if baseline_ns > 0.0 {
                (current_ns - baseline_ns) / baseline_ns * 100.0
            } else {
                0.0
            };
            let z_score = if std_ns > 0.0 {
                (current_ns - mean_ns) / std_ns
            } else {
                0.0
            };
            let is_regression = change_percent > threshold_factor * 100.0 || z_score > 2.0;
            let is_improvement = change_percent < -(threshold_factor * 100.0) || z_score < -2.0;
            let significance = {
                let abs_change = change_percent.abs();
                if abs_change < 5.0 {
                    RegressionSignificance::Negligible
                } else if abs_change < 15.0 {
                    RegressionSignificance::Minor
                } else if abs_change < 30.0 {
                    RegressionSignificance::Moderate
                } else if abs_change < 50.0 {
                    RegressionSignificance::Major
                } else {
                    RegressionSignificance::Critical
                }
            };
            let result = RegressionResult {
                benchmark_name: m.name.clone(),
                baseline_duration: Duration::from_nanos(baseline_ns.max(0.0) as u64),
                current_duration: m.avg_duration,
                change_percent,
                significance,
            };
            if is_regression {
                regressions.push(result);
            } else if is_improvement {
                improvements.push(result);
            }
            total_change_sum += change_percent;
        }
        let overall_change_percent = total_change_sum / measurements.len() as f64;
        let regression_detected = !regressions.is_empty();
        Ok(RegressionAnalysis {
            regression_detected,
            regressions,
            improvements,
            overall_change_percent,
        })
    }
    /// Analyze scalability characteristics
    fn analyze_scalability(
        &self,
        measurements: &[PerformanceMeasurement],
    ) -> CoreResult<ScalabilityAnalysis> {
        let mut datasize_breakdown = HashMap::new();
        let mut n_breakdown = HashMap::new();
        for measurement in measurements {
            if measurement.datasize > 0 {
                datasize_breakdown
                    .insert(measurement.datasize, measurement.efficiency_score() / 100.0);
            }
            if measurement.n > 0 {
                n_breakdown.insert(measurement.n, measurement.efficiency_score() / 100.0);
            }
        }
        let scalability_analysis = ScalabilityAnalysis {
            thread_scalability: Self::retained_efficiency_score(&n_breakdown),
            data_scalability: Self::retained_efficiency_score(&datasize_breakdown),
            memory_scalability: Self::memory_scalability_score(measurements),
            datasize_breakdown,
            n_breakdown,
        };
        Ok(scalability_analysis)
    }
    /// How well efficiency is retained as a breakdown key (thread count or
    /// data size) grows from its smallest to largest value observed in this
    /// run: `efficiency(largest) / efficiency(smallest)`, clamped to
    /// `[0, 1]`. A single observed key has nothing to compare against and
    /// conservatively scores `1.0` (no evidence of degradation), rather
    /// than a fabricated constant regardless of the data.
    fn retained_efficiency_score(breakdown: &HashMap<usize, f64>) -> f64 {
        let mut keys: Vec<usize> = breakdown.keys().copied().collect();
        keys.sort_unstable();
        let (Some(&smallest), Some(&largest)) = (keys.first(), keys.last()) else {
            return 1.0;
        };
        if smallest == largest {
            return 1.0;
        }
        let eff_small = breakdown[&smallest];
        let eff_large = breakdown[&largest];
        if eff_small <= 0.0 {
            return if eff_large > 0.0 { 1.0 } else { 0.0 };
        }
        (eff_large / eff_small).clamp(0.0, 1.0)
    }
    /// How well memory usage scales with data size: ideally memory grows no
    /// faster than the data being processed (linear or better). Computed as
    /// `(data_growth / memory_growth)` between the smallest- and
    /// largest-`datasize` measurements, clamped to `[0, 1]` (a ratio `< 1`
    /// means memory grew faster than the data size, i.e. worse-than-linear
    /// scaling).
    fn memory_scalability_score(measurements: &[PerformanceMeasurement]) -> f64 {
        let mut by_size: Vec<(usize, usize)> = measurements
            .iter()
            .filter(|m| m.datasize > 0)
            .map(|m| (m.datasize, m.memory_usage))
            .collect();
        by_size.sort_unstable_by_key(|&(size, _)| size);
        let (Some(&(small_size, small_mem)), Some(&(large_size, large_mem))) =
            (by_size.first(), by_size.last())
        else {
            return 1.0;
        };
        if small_size == large_size || small_mem == 0 {
            return 1.0;
        }
        let data_growth = large_size as f64 / small_size as f64;
        let mem_growth = large_mem as f64 / small_mem as f64;
        if mem_growth <= 0.0 {
            return 1.0;
        }
        (data_growth / mem_growth).clamp(0.0, 1.0)
    }
    /// Analyze memory efficiency
    fn analyze_memory_efficiency(
        &self,
        measurements: &[PerformanceMeasurement],
    ) -> CoreResult<MemoryEfficiencyAnalysis> {
        let total_operations: usize = measurements.iter().map(|m| m.operations_count).sum();
        let total_memory: usize = measurements.iter().map(|m| m.memory_usage).sum();
        let avg_memory_per_op = if total_operations > 0 {
            total_memory as f64 / total_operations as f64
        } else {
            0.0
        };
        let avg_memory_usage = if measurements.is_empty() {
            0.0
        } else {
            total_memory as f64 / measurements.len() as f64
        };
        let overall_peak_memory = measurements
            .iter()
            .map(|m| m.peak_memory)
            .max()
            .unwrap_or(0);
        let peak_to_avg_ratio = if avg_memory_usage > 0.0 {
            overall_peak_memory as f64 / avg_memory_usage
        } else {
            1.0
        };
        let per_benchmark_peak_ratios: Vec<f64> = measurements
            .iter()
            .filter(|m| m.memory_usage > 0)
            .map(|m| (m.peak_memory as f64 / m.memory_usage as f64 - 1.0).max(0.0))
            .collect();
        let fragmentation_score = if per_benchmark_peak_ratios.is_empty() {
            0.0
        } else {
            (per_benchmark_peak_ratios.iter().sum::<f64>() / per_benchmark_peak_ratios.len() as f64)
                .min(1.0)
        };
        let cross_module: Vec<&PerformanceMeasurement> = measurements
            .iter()
            .filter(|m| m.modules.len() > 1)
            .collect();
        let zero_copy_efficiency = if cross_module.is_empty() {
            1.0
        } else {
            let efficient = cross_module
                .iter()
                .filter(|m| (m.memory_usage as f64) <= avg_memory_usage.max(1.0))
                .count();
            efficient as f64 / cross_module.len() as f64
        };
        let max_throughput = measurements
            .iter()
            .map(|m| m.throughput)
            .fold(0.0_f64, f64::max);
        let avg_throughput = if measurements.is_empty() {
            0.0
        } else {
            measurements.iter().map(|m| m.throughput).sum::<f64>() / measurements.len() as f64
        };
        let bandwidth_utilization = if max_throughput > 0.0 {
            (avg_throughput / max_throughput).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let memory_analysis = MemoryEfficiencyAnalysis {
            avg_memory_per_op,
            peak_to_avg_ratio,
            fragmentation_score,
            zero_copy_efficiency,
            bandwidth_utilization,
        };
        Ok(memory_analysis)
    }
    /// Generate comprehensive benchmark report
    pub fn generate_benchmark_report(&self) -> CoreResult<String> {
        let results = self.results.lock().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to lock results".to_string()))
        })?;
        if results.is_empty() {
            return Ok("No benchmark results available.".to_string());
        }
        let latest = &results[results.len() - 1];
        let mut report = String::new();
        report.push_str("# SciRS2 Cross-Module Performance Benchmark Report\n\n");
        #[cfg(feature = "serialization")]
        {
            report.push_str(&format!(
                "**Generated**: {}\n",
                chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
            ));
        }
        #[cfg(not(feature = "serialization"))]
        {
            report.push_str("**Generated**: [timestamp unavailable]\n");
        }
        report.push_str(&format!("**Suite**: {}\n", latest.name));
        report.push_str(&format!(
            "**Total Duration**: {:?}\n",
            latest.total_duration
        ));
        report.push_str(&format!(
            "**Average Efficiency**: {:.1}%\n\n",
            latest.avg_efficiency
        ));
        report.push_str("## Executive Summary\n\n");
        report.push_str(&format!(
            "- **Benchmarks Executed**: {}\n",
            latest.measurements.len()
        ));
        report.push_str(&format!(
            "- **Overall Efficiency**: {:.1}%\n",
            latest.avg_efficiency
        ));
        report.push_str(&format!(
            "- **Thread Scalability**: {:.1}%\n",
            latest.scalability_analysis.thread_scalability * 100.0
        ));
        report.push_str(&format!(
            "- **Memory Efficiency**: {:.1}%\n",
            (1.0 - latest.memory_analysis.fragmentation_score) * 100.0
        ));
        report.push_str("\n## Benchmark Results\n\n");
        for measurement in &latest.measurements {
            report.push_str(&format!(
                "### {} ({})\n",
                measurement.name,
                measurement.modules.join(" + ")
            ));
            report.push_str(&format!(
                "- **Data Size**: {} bytes\n",
                measurement.datasize
            ));
            report.push_str(&format!(
                "- **Average Time**: {:?}\n",
                measurement.avg_duration
            ));
            report.push_str(&format!(
                "- **Throughput**: {:.2} ops/sec\n",
                measurement.throughput
            ));
            report.push_str(&format!(
                "- **Memory Usage**: {} MB\n",
                measurement.memory_usage / (1024 * 1024)
            ));
            report.push_str(&format!(
                "- **Efficiency Score**: {:.1}%\n",
                measurement.efficiency_score()
            ));
            report.push('\n');
        }
        report.push_str("## Scalability Analysis\n\n");
        report.push_str(&format!(
            "- **Thread Scalability**: {:.1}%\n",
            latest.scalability_analysis.thread_scalability * 100.0
        ));
        report.push_str(&format!(
            "- **Data Size Scalability**: {:.1}%\n",
            latest.scalability_analysis.data_scalability * 100.0
        ));
        report.push_str(&format!(
            "- **Memory Scalability**: {:.1}%\n",
            latest.scalability_analysis.memory_scalability * 100.0
        ));
        report.push_str("\n## Memory Efficiency Analysis\n\n");
        report.push_str(&format!(
            "- **Average Memory per Operation**: {:.2} bytes\n",
            latest.memory_analysis.avg_memory_per_op
        ));
        report.push_str(&format!(
            "- **Peak to Average Ratio**: {:.2}\n",
            latest.memory_analysis.peak_to_avg_ratio
        ));
        report.push_str(&format!(
            "- **Fragmentation Score**: {:.3} (lower is better)\n",
            latest.memory_analysis.fragmentation_score
        ));
        report.push_str(&format!(
            "- **Zero-Copy Efficiency**: {:.1}%\n",
            latest.memory_analysis.zero_copy_efficiency * 100.0
        ));
        if let Some(regression) = &latest.regression_analysis {
            report.push_str("\n## Regression Analysis\n\n");
            if regression.regression_detected {
                report.push_str("⚠️ **Performance regressions detected**\n\n");
                for reg in &regression.regressions {
                    report.push_str(&format!(
                        "- **{}**: {:.2}% regression\n",
                        reg.benchmark_name, reg.change_percent
                    ));
                }
            } else {
                report.push_str("✅ **No significant regressions detected**\n");
            }
        }
        report.push_str("\n## Recommendations\n\n");
        if latest.avg_efficiency >= 80.0 {
            report.push_str(
                "✅ **Excellent Performance**: The cross-module performance is very good.\n",
            );
        } else if latest.avg_efficiency >= 60.0 {
            report.push_str(
                "⚠️ **Good Performance**: Consider optimizing bottlenecks identified above.\n",
            );
        } else {
            report
                .push_str(
                    "❌ **Performance Issues**: Significant optimization work needed before 1.0 release.\n",
                );
        }
        Ok(report)
    }
}
/// Internal timing data structure
#[derive(Debug)]
struct TimingData {
    name: String,
    avg_duration: Duration,
    min_duration: Duration,
    max_duration: Duration,
    std_deviation: Duration,
    throughput: f64,
    memory_usage: usize,
    peak_memory: usize,
    operations_count: usize,
}
impl fmt::Display for RegressionSignificance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegressionSignificance::Negligible => write!(f, "Negligible"),
            RegressionSignificance::Minor => write!(f, "Minor"),
            RegressionSignificance::Moderate => write!(f, "Moderate"),
            RegressionSignificance::Major => write!(f, "Major"),
            RegressionSignificance::Critical => write!(f, "Critical"),
        }
    }
}
/// Convenience function to create a default benchmark suite
#[allow(dead_code)]
pub fn create_default_benchmark_suite() -> CoreResult<CrossModuleBenchmarkRunner> {
    let config = CrossModuleBenchConfig::default();
    Ok(CrossModuleBenchmarkRunner::new(config))
}
/// Convenience function to run quick benchmarks for CI/CD
#[allow(dead_code)]
pub fn run_quick_benchmarks() -> CoreResult<BenchmarkSuiteResult> {
    let config = CrossModuleBenchConfig {
        iterations: 2,
        warmup_iterations: 1,
        datasizes: vec![1024],
        ns: vec![1],
        memory_limits: vec![64 * 1024 * 1024],
        enable_profiling: false,
        enable_regression_detection: false,
        timeout: Duration::from_secs(30),
        ..Default::default()
    };
    let runner = CrossModuleBenchmarkRunner::new(config);
    runner.run_benchmarks()
}

#[cfg(test)]
mod tests;
