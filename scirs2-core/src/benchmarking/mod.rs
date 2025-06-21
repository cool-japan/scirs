//! # Comprehensive Benchmarking System for `SciRS2` Core
//!
//! This module provides a production-ready benchmarking infrastructure that includes:
//! - Performance regression testing
//! - Optimization validation
//! - Comparative benchmarking against reference implementations
//! - Automated performance monitoring
//! - Statistical analysis of benchmark results
//! - Hardware-specific optimization verification

pub mod performance;
pub mod regression;

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Target measurement time
    pub measurement_time: Duration,
    /// Confidence level for statistical analysis (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Maximum acceptable coefficient of variation
    pub max_cv: f64,
    /// Enable detailed profiling
    pub enable_profiling: bool,
    /// Enable memory tracking
    pub enable_memory_tracking: bool,
    /// Custom tags for benchmark categorization
    pub tags: Vec<String>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
            measurement_time: Duration::from_secs(5),
            confidence_level: 0.95,
            max_cv: 0.1, // 10% coefficient of variation
            enable_profiling: false,
            enable_memory_tracking: true,
            tags: Vec::new(),
        }
    }
}

impl BenchmarkConfig {
    /// Create a new benchmark configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set warmup iterations
    pub fn with_warmup_iterations(mut self, iterations: usize) -> Self {
        self.warmup_iterations = iterations;
        self
    }

    /// Set measurement iterations
    pub fn with_measurement_iterations(mut self, iterations: usize) -> Self {
        self.measurement_iterations = iterations;
        self
    }

    /// Set measurement time
    pub fn with_measurement_time(mut self, time: Duration) -> Self {
        self.measurement_time = time;
        self
    }

    /// Set confidence level
    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Set maximum coefficient of variation
    pub fn with_max_cv(mut self, cv: f64) -> Self {
        self.max_cv = cv;
        self
    }

    /// Enable profiling
    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.enable_profiling = enabled;
        self
    }

    /// Enable memory tracking
    pub fn with_memory_tracking(mut self, enabled: bool) -> Self {
        self.enable_memory_tracking = enabled;
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Add a single tag
    pub fn with_tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }
}

/// Benchmark measurement result
#[derive(Debug, Clone)]
pub struct BenchmarkMeasurement {
    /// Execution time for this measurement
    pub execution_time: Duration,
    /// Memory usage during this measurement (in bytes)
    pub memory_usage: usize,
    /// Custom metrics collected during this measurement
    pub custom_metrics: HashMap<String, f64>,
    /// Timestamp when measurement was taken
    pub timestamp: std::time::SystemTime,
}

impl BenchmarkMeasurement {
    /// Create a new benchmark measurement
    pub fn new(execution_time: Duration) -> Self {
        Self {
            execution_time,
            memory_usage: 0,
            custom_metrics: HashMap::new(),
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Set memory usage
    pub fn with_memory_usage(mut self, memory: usize) -> Self {
        self.memory_usage = memory;
        self
    }

    /// Add a custom metric
    pub fn with_custom_metric(mut self, name: String, value: f64) -> Self {
        self.custom_metrics.insert(name, value);
        self
    }

    /// Get execution time in nanoseconds
    pub fn execution_time_nanos(&self) -> u64 {
        self.execution_time.as_nanos() as u64
    }

    /// Get execution time in microseconds
    pub fn execution_time_micros(&self) -> u64 {
        self.execution_time.as_micros() as u64
    }

    /// Get execution time in milliseconds
    pub fn execution_time_millis(&self) -> u64 {
        self.execution_time.as_millis() as u64
    }
}

/// Comprehensive benchmark result with statistical analysis
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the benchmark
    pub name: String,
    /// All measurements collected
    pub measurements: Vec<BenchmarkMeasurement>,
    /// Statistical summary
    pub statistics: BenchmarkStatistics,
    /// Configuration used for this benchmark
    pub config: BenchmarkConfig,
    /// Total benchmark execution time
    pub total_time: Duration,
    /// Whether the benchmark met quality criteria
    pub quality_criteria_met: bool,
    /// Warnings or issues encountered
    pub warnings: Vec<String>,
}

impl BenchmarkResult {
    /// Create a new benchmark result
    pub fn new(name: String, config: BenchmarkConfig) -> Self {
        Self {
            name,
            measurements: Vec::new(),
            statistics: BenchmarkStatistics::default(),
            config,
            total_time: Duration::from_secs(0),
            quality_criteria_met: false,
            warnings: Vec::new(),
        }
    }

    /// Add a measurement
    pub fn add_measurement(&mut self, measurement: BenchmarkMeasurement) {
        self.measurements.push(measurement);
    }

    /// Finalize the benchmark and compute statistics
    pub fn finalize(&mut self) -> CoreResult<()> {
        if self.measurements.is_empty() {
            return Err(CoreError::ValidationError(crate::error::ErrorContext::new(
                "No measurements collected",
            )));
        }

        self.statistics = BenchmarkStatistics::from_measurements(&self.measurements)?;

        // Check quality criteria
        self.quality_criteria_met = self.statistics.coefficient_of_variation <= self.config.max_cv;

        if !self.quality_criteria_met {
            self.warnings.push(format!(
                "High coefficient of variation: {:.3} > {:.3}",
                self.statistics.coefficient_of_variation, self.config.max_cv
            ));
        }

        Ok(())
    }

    /// Get throughput in operations per second
    pub fn throughput_ops_per_sec(&self, operations_per_iteration: u64) -> f64 {
        let avg_time_seconds = self.statistics.mean_execution_time.as_secs_f64();
        operations_per_iteration as f64 / avg_time_seconds
    }

    /// Get memory efficiency (operations per MB)
    pub fn memory_efficiency(&self, operations_per_iteration: u64) -> f64 {
        if self.statistics.mean_memory_usage == 0 {
            return f64::INFINITY;
        }
        let memory_mb = self.statistics.mean_memory_usage as f64 / (1024.0 * 1024.0);
        operations_per_iteration as f64 / memory_mb
    }
}

/// Statistical summary of benchmark measurements
#[derive(Debug, Clone, Default)]
pub struct BenchmarkStatistics {
    /// Mean execution time
    pub mean_execution_time: Duration,
    /// Median execution time
    pub median_execution_time: Duration,
    /// Standard deviation of execution times
    pub std_dev_execution_time: Duration,
    /// Minimum execution time
    pub min_execution_time: Duration,
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Coefficient of variation for execution time
    pub coefficient_of_variation: f64,
    /// 95% confidence interval for mean
    pub confidence_interval: (Duration, Duration),
    /// Mean memory usage
    pub mean_memory_usage: usize,
    /// Standard deviation of memory usage
    pub std_dev_memory_usage: usize,
    /// Total number of measurements
    pub sample_count: usize,
}

impl BenchmarkStatistics {
    /// Compute statistics from measurements
    pub fn from_measurements(measurements: &[BenchmarkMeasurement]) -> CoreResult<Self> {
        if measurements.is_empty() {
            return Err(CoreError::ValidationError(crate::error::ErrorContext::new(
                "Cannot compute statistics from empty measurements",
            )));
        }

        // Extract execution times
        let mut execution_times: Vec<Duration> =
            measurements.iter().map(|m| m.execution_time).collect();
        execution_times.sort();

        // Compute execution time statistics
        let mean_nanos = execution_times
            .iter()
            .map(|d| d.as_nanos() as f64)
            .sum::<f64>()
            / execution_times.len() as f64;
        let mean_execution_time = Duration::from_nanos(mean_nanos as u64);

        let median_execution_time = if execution_times.len() % 2 == 0 {
            let mid = execution_times.len() / 2;
            Duration::from_nanos(
                ((execution_times[mid - 1].as_nanos() + execution_times[mid].as_nanos()) / 2)
                    as u64,
            )
        } else {
            execution_times[execution_times.len() / 2]
        };

        let variance = execution_times
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>()
            / execution_times.len() as f64;
        let std_dev_execution_time = Duration::from_nanos(variance.sqrt() as u64);

        let min_execution_time = execution_times[0];
        let max_execution_time = execution_times[execution_times.len() - 1];

        let coefficient_of_variation = if mean_nanos > 0.0 {
            (variance.sqrt()) / mean_nanos
        } else {
            0.0
        };

        // Compute 95% confidence interval (assuming normal distribution)
        let t_value = 1.96; // For 95% confidence with large sample size
        let standard_error = variance.sqrt() / (execution_times.len() as f64).sqrt();
        let margin_of_error = t_value * standard_error;
        let confidence_interval = (
            Duration::from_nanos((mean_nanos - margin_of_error).max(0.0) as u64),
            Duration::from_nanos((mean_nanos + margin_of_error) as u64),
        );

        // Memory statistics
        let mean_memory = measurements
            .iter()
            .map(|m| m.memory_usage as f64)
            .sum::<f64>()
            / measurements.len() as f64;
        let memory_variance = measurements
            .iter()
            .map(|m| {
                let diff = m.memory_usage as f64 - mean_memory;
                diff * diff
            })
            .sum::<f64>()
            / measurements.len() as f64;

        Ok(BenchmarkStatistics {
            mean_execution_time,
            median_execution_time,
            std_dev_execution_time,
            min_execution_time,
            max_execution_time,
            coefficient_of_variation,
            confidence_interval,
            mean_memory_usage: mean_memory as usize,
            std_dev_memory_usage: memory_variance.sqrt() as usize,
            sample_count: measurements.len(),
        })
    }

    /// Check if the measurements are statistically reliable
    pub fn is_reliable(&self, max_cv: f64) -> bool {
        self.coefficient_of_variation <= max_cv && self.sample_count >= 10
    }

    /// Get execution time percentile
    pub fn execution_time_percentile(
        &self,
        measurements: &[BenchmarkMeasurement],
        percentile: f64,
    ) -> Duration {
        if measurements.is_empty() {
            return Duration::from_secs(0);
        }

        let mut times: Vec<Duration> = measurements.iter().map(|m| m.execution_time).collect();
        times.sort();

        let index = (percentile / 100.0 * (times.len() - 1) as f64).round() as usize;
        times[index.min(times.len() - 1)]
    }
}

/// Benchmark runner that executes and measures performance
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run a benchmark function
    pub fn run<F, T>(&self, name: &str, mut benchmark_fn: F) -> CoreResult<BenchmarkResult>
    where
        F: FnMut() -> CoreResult<T>,
    {
        let total_start = Instant::now();
        let mut result = BenchmarkResult::new(name.to_string(), self.config.clone());

        // Warmup phase
        for _ in 0..self.config.warmup_iterations {
            benchmark_fn()?;
        }

        // Measurement phase
        let measurement_start = Instant::now();
        let mut iteration_count = 0;

        while iteration_count < self.config.measurement_iterations
            && measurement_start.elapsed() < self.config.measurement_time
        {
            let memory_before = if self.config.enable_memory_tracking {
                self.get_memory_usage().unwrap_or(0)
            } else {
                0
            };

            let start = Instant::now();
            benchmark_fn()?;
            let execution_time = start.elapsed();

            let memory_after = if self.config.enable_memory_tracking {
                self.get_memory_usage().unwrap_or(0)
            } else {
                0
            };

            let memory_usage = memory_after.saturating_sub(memory_before);

            result.add_measurement(
                BenchmarkMeasurement::new(execution_time).with_memory_usage(memory_usage),
            );

            iteration_count += 1;
        }

        result.total_time = total_start.elapsed();
        result.finalize()?;

        Ok(result)
    }

    /// Run a benchmark with setup and teardown
    pub fn run_with_setup<F, G, H, T, S>(
        &self,
        name: &str,
        mut setup: F,
        mut benchmark_fn: G,
        mut teardown: H,
    ) -> CoreResult<BenchmarkResult>
    where
        F: FnMut() -> CoreResult<S>,
        G: FnMut(&mut S) -> CoreResult<T>,
        H: FnMut(S) -> CoreResult<()>,
    {
        let total_start = Instant::now();
        let mut result = BenchmarkResult::new(name.to_string(), self.config.clone());

        // Warmup phase
        for _ in 0..self.config.warmup_iterations {
            let mut state = setup()?;
            benchmark_fn(&mut state)?;
            teardown(state)?;
        }

        // Measurement phase
        let measurement_start = Instant::now();
        let mut iteration_count = 0;

        while iteration_count < self.config.measurement_iterations
            && measurement_start.elapsed() < self.config.measurement_time
        {
            let mut state = setup()?;

            let memory_before = if self.config.enable_memory_tracking {
                self.get_memory_usage().unwrap_or(0)
            } else {
                0
            };

            let start = Instant::now();
            benchmark_fn(&mut state)?;
            let execution_time = start.elapsed();

            let memory_after = if self.config.enable_memory_tracking {
                self.get_memory_usage().unwrap_or(0)
            } else {
                0
            };

            teardown(state)?;

            let memory_usage = memory_after.saturating_sub(memory_before);

            result.add_measurement(
                BenchmarkMeasurement::new(execution_time).with_memory_usage(memory_usage),
            );

            iteration_count += 1;
        }

        result.total_time = total_start.elapsed();
        result.finalize()?;

        Ok(result)
    }

    /// Run a parameterized benchmark
    pub fn run_parameterized<F, T, P>(
        &self,
        name: &str,
        parameters: Vec<P>,
        mut benchmark_fn: F,
    ) -> CoreResult<Vec<(P, BenchmarkResult)>>
    where
        F: FnMut(&P) -> CoreResult<T>,
        P: Clone + std::fmt::Debug,
    {
        let mut results = Vec::new();

        for param in parameters {
            let param_name = format!("{}({:?})", name, param);
            let param_clone = param.clone();

            let result = self.run(&param_name, || benchmark_fn(&param_clone))?;
            results.push((param, result));
        }

        Ok(results)
    }

    /// Get current memory usage
    fn get_memory_usage(&self) -> CoreResult<usize> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let status = fs::read_to_string("/proc/self/status").map_err(|e| {
                CoreError::IoError(ErrorContext::new(format!(
                    "Failed to read memory status: {}",
                    e
                )))
            })?;

            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let kb: usize = parts[1].parse().map_err(|e| {
                            CoreError::ValidationError(crate::error::ErrorContext::new(format!(
                                "Failed to parse memory: {}",
                                e
                            )))
                        })?;
                        return Ok(kb * 1024);
                    }
                }
            }
        }

        // Fallback for non-Linux systems
        Ok(0)
    }
}

/// Type alias for benchmark functions
type BenchmarkFn = Box<dyn Fn(&BenchmarkRunner) -> CoreResult<BenchmarkResult> + Send + Sync>;

/// Benchmark suite for organizing multiple related benchmarks
pub struct BenchmarkSuite {
    name: String,
    benchmarks: Vec<BenchmarkFn>,
    config: BenchmarkConfig,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(name: &str, config: BenchmarkConfig) -> Self {
        Self {
            name: name.to_string(),
            benchmarks: Vec::new(),
            config,
        }
    }

    /// Add a benchmark to the suite
    pub fn add_benchmark<F>(&mut self, benchmark_fn: F)
    where
        F: Fn(&BenchmarkRunner) -> CoreResult<BenchmarkResult> + Send + Sync + 'static,
    {
        self.benchmarks.push(Box::new(benchmark_fn));
    }

    /// Run all benchmarks in the suite
    pub fn run(&self) -> CoreResult<Vec<BenchmarkResult>> {
        let runner = BenchmarkRunner::new(self.config.clone());
        let mut results = Vec::new();

        println!("Running benchmark suite: {}", self.name);

        for (i, benchmark) in self.benchmarks.iter().enumerate() {
            println!("Running benchmark {}/{}", i + 1, self.benchmarks.len());

            match benchmark(&runner) {
                Ok(result) => {
                    println!(
                        "  {} completed: {:.3}ms ± {:.3}ms",
                        result.name,
                        result.statistics.mean_execution_time.as_millis(),
                        result.statistics.std_dev_execution_time.as_millis()
                    );
                    results.push(result);
                }
                Err(e) => {
                    println!("  Benchmark failed: {:?}", e);
                    return Err(e);
                }
            }
        }

        // Print summary
        self.print_summary(&results);

        Ok(results)
    }

    /// Print a summary of benchmark results
    fn print_summary(&self, results: &[BenchmarkResult]) {
        println!("\nBenchmark Suite '{}' Summary:", self.name);
        println!("----------------------------------------");

        for result in results {
            let quality_indicator = if result.quality_criteria_met {
                "✓"
            } else {
                "⚠"
            };
            println!(
                "{} {}: {:.3}ms (CV: {:.2}%)",
                quality_indicator,
                result.name,
                result.statistics.mean_execution_time.as_millis(),
                result.statistics.coefficient_of_variation * 100.0
            );

            for warning in &result.warnings {
                println!("    Warning: {}", warning);
            }
        }

        let reliable_count = results.iter().filter(|r| r.quality_criteria_met).count();
        println!(
            "\nReliable benchmarks: {}/{}",
            reliable_count,
            results.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::new()
            .with_warmup_iterations(5)
            .with_measurement_iterations(50)
            .with_confidence_level(0.99)
            .with_tag("test".to_string());

        assert_eq!(config.warmup_iterations, 5);
        assert_eq!(config.measurement_iterations, 50);
        assert_eq!(config.confidence_level, 0.99);
        assert_eq!(config.tags, vec!["test"]);
    }

    #[test]
    fn test_benchmark_measurement() {
        let measurement = BenchmarkMeasurement::new(Duration::from_millis(100))
            .with_memory_usage(1024)
            .with_custom_metric("ops".to_string(), 1000.0);

        assert_eq!(measurement.execution_time, Duration::from_millis(100));
        assert_eq!(measurement.memory_usage, 1024);
        assert_eq!(measurement.custom_metrics["ops"], 1000.0);
    }

    #[test]
    fn test_benchmark_statistics() {
        let measurements = vec![
            BenchmarkMeasurement::new(Duration::from_millis(100)),
            BenchmarkMeasurement::new(Duration::from_millis(110)),
            BenchmarkMeasurement::new(Duration::from_millis(90)),
            BenchmarkMeasurement::new(Duration::from_millis(105)),
        ];

        let stats = BenchmarkStatistics::from_measurements(&measurements).unwrap();

        assert_eq!(stats.sample_count, 4);
        assert!(stats.mean_execution_time > Duration::from_millis(95));
        assert!(stats.mean_execution_time < Duration::from_millis(110));
        assert!(stats.coefficient_of_variation > 0.0);
    }

    #[test]
    fn test_benchmark_runner() {
        let config = BenchmarkConfig::new()
            .with_warmup_iterations(1)
            .with_measurement_iterations(5);
        let runner = BenchmarkRunner::new(config);

        let result = runner
            .run("test_benchmark", || {
                // Simulate some work
                std::thread::sleep(Duration::from_micros(100));
                Ok(())
            })
            .unwrap();

        assert_eq!(result.name, "test_benchmark");
        assert_eq!(result.measurements.len(), 5);
        assert!(result.statistics.mean_execution_time > Duration::from_micros(50));
    }
}
