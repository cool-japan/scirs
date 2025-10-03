//! # Comprehensive Benchmarking System for `SciRS2` Core
//!
//! This module provides a production-ready benchmarking infrastructure that includes:
//! - Performance regression testing
//! - Optimization validation
//! - Comparative benchmarking against reference implementations
//! - Automated performance monitoring
//! - Statistical analysis of benchmark results
//! - Hardware-specific optimization verification
//! - Cross-module performance benchmarking for 1.0 release validation

pub mod cross_module;
pub mod performance;
pub mod regression;

// Re-export commonly used cross-module types
pub use cross_module::{
    create_default_benchmark_suite, run_quick_benchmarks,
    BenchmarkSuiteResult as CrossModuleBenchmarkSuiteResult, CrossModuleBenchConfig,
    CrossModuleBenchmarkRunner, PerformanceMeasurement as CrossModulePerformanceMeasurement,
};

use crate::error::{CoreError, CoreResult, ErrorContext};
use crate::performance_optimization::OptimizationStrategy;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Optimization strategies to benchmark
    pub strategies: HashSet<OptimizationStrategy>,
    /// Sample sizes for benchmarking
    pub sample_sizes: Vec<usize>,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Target measurement time
    pub measurement_time: Duration,
    /// Minimum duration for each measurement
    pub min_duration: Duration,
    /// Maximum duration for each measurement  
    pub max_duration: Duration,
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
        let mut strategies = HashSet::new();
        strategies.insert(OptimizationStrategy::Scalar);

        Self {
            strategies,
            sample_sizes: vec![1000, 10000, 100000],
            warmup_iterations: 10,
            measurement_iterations: 100,
            measurement_time: Duration::from_secs(5),
            min_duration: Duration::from_millis(1),
            max_duration: Duration::from_secs(30),
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

    /// Set minimum duration
    pub fn with_min_duration(mut self, duration: Duration) -> Self {
        self.min_duration = std::time::Duration::from_secs(1);
        self
    }

    /// Set maximum duration
    pub fn with_max_duration(mut self, duration: Duration) -> Self {
        self.max_duration = std::time::Duration::from_secs(1);
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

    /// Set strategies
    pub fn with_strategies(mut self, strategies: HashSet<OptimizationStrategy>) -> Self {
        self.strategies = strategies;
        self
    }

    /// Add a single strategy
    pub fn with_strategy(mut self, strategy: OptimizationStrategy) -> Self {
        self.strategies.insert(strategy);
        self
    }

    /// Set sample sizes
    pub fn with_sample_sizes(mut self, sample_sizes: Vec<usize>) -> Self {
        self.sample_sizes = sample_sizes;
        self
    }
}

/// Benchmark measurement result
#[derive(Debug, Clone)]
pub struct BenchmarkMeasurement {
    /// Execution time for this measurement
    pub execution_time: Duration,
    /// Duration field (alias for execution_time for compatibility)
    pub duration: Duration,
    /// Strategy used for this measurement  
    pub strategy: OptimizationStrategy,
    /// Input size used for this measurement
    pub input_size: usize,
    /// Throughput achieved in operations per second
    pub throughput: f64,
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
            duration: execution_time,
            strategy: OptimizationStrategy::Scalar,
            input_size: 0,
            throughput: 0.0,
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

    /// Set strategy
    pub fn with_strategy(mut self, strategy: OptimizationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set input size
    pub fn with_input_size(mut self, input_size: usize) -> Self {
        self.input_size = input_size;
        self
    }

    /// Set throughput
    pub fn with_throughput(mut self, throughput: f64) -> Self {
        self.throughput = throughput;
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
    pub fn get_throughput(&self, operations_periteration: u64) -> f64 {
        let avg_time_seconds = self.statistics.mean_execution_time.as_secs_f64();
        operations_periteration as f64 / avg_time_seconds
    }

    /// Get memory efficiency (operations per MB)
    pub fn get_memory_efficiency(&self, operations_periteration: u64) -> f64 {
        if self.statistics.mean_memory_usage == 0 {
            return f64::INFINITY;
        }
        let memory_mb = self.statistics.mean_memory_usage as f64 / (1024.0 * 1024.0);
        operations_periteration as f64 / memory_mb
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

        let median_execution_time = if execution_times.len().is_multiple_of(2) {
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
        let standarderror = variance.sqrt() / (execution_times.len() as f64).sqrt();
        let margin_oferror = t_value * standarderror;
        let confidence_interval = (
            Duration::from_nanos((mean_nanos - margin_oferror).max(0.0) as u64),
            Duration::from_nanos((mean_nanos + margin_oferror) as u64),
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
    pub fn run<F, T>(&self, name: &str, mut benchmarkfn: F) -> CoreResult<BenchmarkResult>
    where
        F: FnMut() -> CoreResult<T>,
    {
        let total_start = Instant::now();
        let mut result = BenchmarkResult::new(name.to_string(), self.config.clone());

        // Warmup phase
        for _ in 0..self.config.warmup_iterations {
            benchmarkfn()?;
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
            benchmarkfn()?;
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
            let param_name = format!("{name}({param:?})");
            let param_clone = param.clone();

            let result = self.run(&param_name, || benchmark_fn(&param_clone))?;
            results.push((param, result));
        }

        Ok(results)
    }

    /// Benchmark an operation with different strategies
    #[allow(dead_code)]
    pub fn benchmark_operation<F, T>(
        &self,
        name: &str,
        mut operation: F,
    ) -> CoreResult<Vec<BenchmarkMeasurement>>
    where
        F: FnMut(&[u8], OptimizationStrategy) -> CoreResult<T>,
    {
        let mut measurements = Vec::new();

        // Generate some dummy data for testing
        let data = vec![0u8; 1000];

        for strategy in &self.config.strategies {
            let start = std::time::Instant::now();
            let _ = operation(&data, *strategy)?;
            let elapsed = start.elapsed();

            let measurement = BenchmarkMeasurement::new(elapsed)
                .with_strategy(*strategy)
                .with_input_size(data.len())
                .with_throughput(data.len() as f64 / elapsed.as_secs_f64());

            measurements.push(measurement);
        }

        Ok(measurements)
    }

    /// Get current memory usage
    fn get_memory_usage(&self) -> CoreResult<usize> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let status = fs::read_to_string("/proc/self/status").map_err(|e| {
                CoreError::IoError(ErrorContext::new(format!(
                    "Failed to read memory status: {e}"
                )))
            })?;

            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let kb: usize = parts[1].parse().map_err(|e| {
                            CoreError::ValidationError(crate::error::ErrorContext::new(format!(
                                "Failed to parse memory: {e}"
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
                    println!("  Benchmark failed: {e:?}");
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
                println!("    Warning: {warning}");
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

/// Strategy performance measurement
#[derive(Debug, Clone)]
pub struct StrategyPerformance {
    pub strategy: OptimizationStrategy,
    pub throughput: f64,
    pub latency: Duration,
    pub memory_efficiency: f64,
    pub cache_hit_rate: f64,
    pub avg_throughput: f64,
    pub throughput_stddev: f64,
    pub avg_memory_usage: f64,
    pub optimal_size: usize,
    pub efficiency_score: f64,
}

impl StrategyPerformance {
    /// Create a new strategy performance measurement
    #[allow(dead_code)]
    pub fn new(strategy: OptimizationStrategy) -> Self {
        Self {
            strategy,
            throughput: 0.0,
            latency: Duration::from_secs(0),
            memory_efficiency: 0.0,
            cache_hit_rate: 0.0,
            avg_throughput: 0.0,
            throughput_stddev: 0.0,
            avg_memory_usage: 0.0,
            optimal_size: 0,
            efficiency_score: 0.0,
        }
    }
}

/// Memory scaling characteristics  
#[derive(Debug, Clone)]
pub struct MemoryScaling {
    pub linear_factor: f64,
    pub logarithmic_factor: f64,
    pub constant_overhead: usize,
    pub linear_coefficient: f64,
    pub constant_coefficient: f64,
    pub r_squared: f64,
}

impl Default for MemoryScaling {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryScaling {
    /// Create a new memory scaling measurement
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            linear_factor: 1.0,
            logarithmic_factor: 0.0,
            constant_overhead: 0,
            linear_coefficient: 1.0,
            constant_coefficient: 0.0,
            r_squared: 1.0,
        }
    }
}

/// Performance bottleneck identification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BottleneckType {
    CpuBound,
    MemoryBandwidth,
    CacheMisses,
    BranchMisprediction,
    IoWait,
    AlgorithmicComplexity,
    CacheLatency,
    ComputeBound,
    SynchronizationOverhead,
}

#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    pub bottleneck_type: BottleneckType,
    pub severity: f64,
    pub description: String,
    pub mitigation_strategy: OptimizationStrategy,
    pub size_range: (usize, usize),
    pub impact: f64,
    pub mitigation: String,
}

impl PerformanceBottleneck {
    /// Create a new performance bottleneck
    #[allow(dead_code)]
    pub fn new(bottleneck_type: BottleneckType) -> Self {
        Self {
            bottleneck_type,
            severity: 0.0,
            description: String::new(),
            mitigation_strategy: OptimizationStrategy::Scalar,
            size_range: (0, 0),
            impact: 0.0,
            mitigation: String::new(),
        }
    }
}

/// Scalability analysis
#[derive(Debug, Clone)]
pub struct ScalabilityAnalysis {
    pub parallel_efficiency: HashMap<usize, f64>,
    pub memory_scaling: MemoryScaling,
    pub bottlenecks: Vec<PerformanceBottleneck>,
}

impl Default for ScalabilityAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalabilityAnalysis {
    /// Create a new scalability analysis
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            parallel_efficiency: HashMap::new(),
            memory_scaling: MemoryScaling::new(),
            bottlenecks: Vec::new(),
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub operation_name: String,
    pub measurements: Vec<BenchmarkMeasurement>,
    pub strategy_summary: HashMap<OptimizationStrategy, StrategyPerformance>,
    pub scalability_analysis: ScalabilityAnalysis,
    pub recommendations: Vec<String>,
    pub total_duration: Duration,
}

impl BenchmarkResults {
    /// Create a new benchmark results
    #[allow(dead_code)]
    pub fn new(operation_name: String) -> Self {
        Self {
            operation_name,
            measurements: Vec::new(),
            strategy_summary: HashMap::new(),
            scalability_analysis: ScalabilityAnalysis::new(),
            recommendations: Vec::new(),
            total_duration: Duration::from_secs(0),
        }
    }
}

/// Benchmark configuration presets
pub mod presets {
    use super::*;

    /// Comprehensive benchmark configuration for Advanced mode
    ///
    /// This configuration includes all available optimization strategies
    /// and provides extensive sample size coverage for thorough testing.
    #[allow(dead_code)]
    pub fn advanced_comprehensive() -> BenchmarkConfig {
        let mut strategies = HashSet::new();
        strategies.insert(OptimizationStrategy::Scalar);
        strategies.insert(OptimizationStrategy::Simd);
        strategies.insert(OptimizationStrategy::Parallel);
        strategies.insert(OptimizationStrategy::Gpu);
        strategies.insert(OptimizationStrategy::Hybrid);
        strategies.insert(OptimizationStrategy::CacheOptimized);
        strategies.insert(OptimizationStrategy::MemoryBound);
        strategies.insert(OptimizationStrategy::ComputeBound);
        strategies.insert(OptimizationStrategy::ModernArchOptimized);
        strategies.insert(OptimizationStrategy::VectorOptimized);
        strategies.insert(OptimizationStrategy::EnergyEfficient);
        strategies.insert(OptimizationStrategy::HighThroughput);

        let sample_sizes = vec![
            100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000,
            10_000_000,
        ];

        BenchmarkConfig {
            strategies,
            sample_sizes,
            warmup_iterations: 15,
            measurement_iterations: 50,
            measurement_time: Duration::from_secs(10),
            min_duration: Duration::from_millis(10),
            max_duration: Duration::from_secs(60),
            confidence_level: 0.95,
            max_cv: 0.1,
            enable_profiling: true,
            enable_memory_tracking: true,
            tags: vec!["Advanced".to_string(), "comprehensive".to_string()],
        }
    }

    /// Modern architecture-focused benchmark configuration
    ///
    /// This configuration focuses on modern CPU architectures and
    /// advanced optimization strategies while excluding basic scalar approaches.
    #[allow(dead_code)]
    pub fn modern_architectures() -> BenchmarkConfig {
        let mut strategies = HashSet::new();
        strategies.insert(OptimizationStrategy::ModernArchOptimized);
        strategies.insert(OptimizationStrategy::VectorOptimized);
        strategies.insert(OptimizationStrategy::EnergyEfficient);
        strategies.insert(OptimizationStrategy::HighThroughput);

        let sample_sizes = vec![1_000, 10_000, 100_000, 1_000_000, 10_000_000];

        BenchmarkConfig {
            strategies,
            sample_sizes,
            warmup_iterations: 10,
            measurement_iterations: 30,
            measurement_time: Duration::from_secs(8),
            min_duration: Duration::from_millis(5),
            max_duration: Duration::from_secs(30),
            confidence_level: 0.95,
            max_cv: 0.1,
            enable_profiling: true,
            enable_memory_tracking: true,
            tags: vec!["modern".to_string(), "architecture".to_string()],
        }
    }

    /// Array operations benchmark configuration
    ///
    /// This configuration is optimized for benchmarking array operations
    /// with strategies focused on SIMD and parallel processing.
    #[allow(dead_code)]
    pub fn array_operations() -> BenchmarkConfig {
        let mut strategies = HashSet::new();
        strategies.insert(OptimizationStrategy::Scalar);
        strategies.insert(OptimizationStrategy::Simd);
        strategies.insert(OptimizationStrategy::VectorOptimized);
        strategies.insert(OptimizationStrategy::CacheOptimized);

        let sample_sizes = vec![100, 1_000, 10_000, 100_000];

        BenchmarkConfig {
            strategies,
            sample_sizes,
            warmup_iterations: 10,
            measurement_iterations: 25,
            measurement_time: Duration::from_secs(5),
            min_duration: Duration::from_millis(1),
            max_duration: Duration::from_secs(15),
            confidence_level: 0.95,
            max_cv: 0.15,
            enable_profiling: false,
            enable_memory_tracking: true,
            tags: vec!["array".to_string(), "operations".to_string()],
        }
    }

    /// Matrix operations benchmark configuration
    ///
    /// This configuration is optimized for benchmarking matrix operations
    /// with strategies focused on cache optimization and parallel processing.
    #[allow(dead_code)]
    pub fn matrix_operations() -> BenchmarkConfig {
        let mut strategies = HashSet::new();
        strategies.insert(OptimizationStrategy::Scalar);
        strategies.insert(OptimizationStrategy::Parallel);
        strategies.insert(OptimizationStrategy::CacheOptimized);
        strategies.insert(OptimizationStrategy::ModernArchOptimized);

        let sample_sizes = vec![100, 500, 1_000, 5_000];

        BenchmarkConfig {
            strategies,
            sample_sizes,
            warmup_iterations: 5,
            measurement_iterations: 20,
            measurement_time: Duration::from_secs(8),
            min_duration: Duration::from_millis(2),
            max_duration: Duration::from_secs(20),
            confidence_level: 0.95,
            max_cv: 0.12,
            enable_profiling: true,
            enable_memory_tracking: true,
            tags: vec!["matrix".to_string(), "operations".to_string()],
        }
    }

    /// Memory intensive benchmark configuration
    ///
    /// This configuration is optimized for benchmarking memory-intensive operations
    /// with strategies focused on memory optimization and throughput.
    #[allow(dead_code)]
    pub fn memory_intensive() -> BenchmarkConfig {
        let mut strategies = HashSet::new();
        strategies.insert(OptimizationStrategy::MemoryBound);
        strategies.insert(OptimizationStrategy::CacheOptimized);
        strategies.insert(OptimizationStrategy::HighThroughput);

        let sample_sizes = vec![10_000, 100_000, 1_000_000];

        BenchmarkConfig {
            strategies,
            sample_sizes,
            warmup_iterations: 3,
            measurement_iterations: 15,
            measurement_time: Duration::from_secs(12),
            min_duration: Duration::from_millis(5),
            max_duration: Duration::from_secs(45),
            confidence_level: 0.95,
            max_cv: 0.2,
            enable_profiling: true,
            enable_memory_tracking: true,
            tags: vec!["memory".to_string(), "intensive".to_string()],
        }
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
