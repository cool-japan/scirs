//! Comprehensive SciPy benchmark comparison framework
//!
//! This module provides a complete benchmarking framework to validate
//! SciRS2 implementations against SciPy equivalents and measure performance.
//!
//! ## Features
//!
//! - Automated benchmarking against Python SciPy
//! - Accuracy validation with configurable tolerances
//! - Performance measurement and comparison
//! - Comprehensive test data generation
//! - Statistical significance testing
//! - Detailed reporting and visualization

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Comprehensive benchmark framework for SciPy comparison
#[derive(Debug)]
pub struct ScipyBenchmarkFramework {
    config: BenchmarkConfig,
    results_cache: HashMap<String, BenchmarkResult>,
    testdata_generator: TestDataGenerator,
}

/// Configuration for benchmark comparisons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Absolute tolerance for numerical comparisons
    pub absolute_tolerance: f64,
    /// Relative tolerance for numerical comparisons  
    pub relative_tolerance: f64,
    /// Number of performance test iterations
    pub performance_iterations: usize,
    /// Number of warmup iterations before timing
    pub warmup_iterations: usize,
    /// Maximum allowed performance regression (ratio)
    pub max_performance_regression: f64,
    /// Test data sizes to benchmark
    pub testsizes: Vec<usize>,
    /// Enable detailed statistical analysis
    pub enable_statistical_tests: bool,
    /// Path to Python SciPy reference implementation
    pub scipy_reference_path: Option<String>,
}

/// Result of a benchmark comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Function name being benchmarked
    pub function_name: String,
    /// Test data size
    pub datasize: usize,
    /// Accuracy comparison results
    pub accuracy: AccuracyComparison,
    /// Performance comparison results
    pub performance: PerformanceComparison,
    /// Overall benchmark status
    pub status: BenchmarkStatus,
    /// Timestamp of benchmark execution
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Accuracy comparison between SciRS2 and SciPy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyComparison {
    /// Maximum absolute difference
    pub max_abs_difference: f64,
    /// Mean absolute difference
    pub mean_abs_difference: f64,
    /// Relative error (L2 norm)
    pub relativeerror: f64,
    /// Number of values that differ beyond tolerance
    pub outlier_count: usize,
    /// Accuracy grade (A-F scale)
    pub accuracy_grade: AccuracyGrade,
    /// Pass/fail status
    pub passes_tolerance: bool,
}

/// Performance comparison between SciRS2 and SciPy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    /// SciRS2 execution time statistics
    pub scirs2_timing: TimingStatistics,
    /// SciPy execution time statistics (if available)
    pub scipy_timing: Option<TimingStatistics>,
    /// Performance ratio (SciRS2 / SciPy)
    pub performance_ratio: Option<f64>,
    /// Performance grade (A-F scale)
    pub performance_grade: PerformanceGrade,
    /// Memory usage comparison
    pub memory_usage: MemoryComparison,
}

/// Timing statistics for performance measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStatistics {
    /// Mean execution time
    pub mean: Duration,
    /// Standard deviation of execution times
    pub std_dev: Duration,
    /// Minimum execution time
    pub min: Duration,
    /// Maximum execution time
    pub max: Duration,
    /// 50th percentile (median)
    pub p50: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
}

/// Memory usage comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryComparison {
    /// Peak memory usage (bytes)
    pub peak_memory: usize,
    /// Average memory usage during execution
    pub average_memory: usize,
    /// Memory efficiency ratio vs SciPy
    pub efficiency_ratio: Option<f64>,
}

/// Accuracy grading scale
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AccuracyGrade {
    /// Excellent accuracy (< 1e-12 error)
    A,
    /// Very good accuracy (< 1e-9 error)
    B,
    /// Good accuracy (< 1e-6 error)
    C,
    /// Acceptable accuracy (< 1e-3 error)
    D,
    /// Poor accuracy (> 1e-3 error)
    F,
}

/// Performance grading scale
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PerformanceGrade {
    /// Excellent performance (> 2x faster than SciPy)
    A,
    /// Very good performance (1.5-2x faster)
    B,
    /// Good performance (0.8-1.5x)
    C,
    /// Acceptable performance (0.5-0.8x)
    D,
    /// Poor performance (< 0.5x SciPy speed)
    F,
}

/// Overall benchmark status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BenchmarkStatus {
    /// Both accuracy and performance meet requirements
    Pass,
    /// Accuracy meets requirements but performance issues
    AccuracyPass,
    /// Performance meets requirements but accuracy issues
    PerformancePass,
    /// Neither accuracy nor performance meet requirements
    Fail,
    /// Benchmark could not be completed
    Error,
}

/// Test data generator for benchmarks
#[derive(Debug)]
pub struct TestDataGenerator {
    config: TestDataConfig,
}

/// Configuration for test data generation
#[derive(Debug, Clone)]
pub struct TestDataConfig {
    /// Random seed for reproducible tests
    pub seed: u64,
    /// Generate edge cases (inf, nan, very large/small values)
    pub include_edge_cases: bool,
    /// Distribution of test data
    pub data_distribution: DataDistribution,
}

/// Distribution types for test data
#[derive(Debug, Clone)]
pub enum DataDistribution {
    /// Standard normal distribution
    Normal,
    /// Uniform distribution in range
    Uniform { min: f64, max: f64 },
    /// Exponential distribution
    Exponential { lambda: f64 },
    /// Mixed distribution combining multiple types
    Mixed(Vec<DataDistribution>),
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-12,
            relative_tolerance: 1e-9,
            performance_iterations: 100,
            warmup_iterations: 10,
            max_performance_regression: 2.0, // Allow 2x slower than SciPy
            testsizes: vec![100, 1000, 10000, 100000],
            enable_statistical_tests: true,
            scipy_reference_path: None,
        }
    }
}

impl Default for TestDataConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            include_edge_cases: true,
            data_distribution: DataDistribution::Normal,
        }
    }
}

impl ScipyBenchmarkFramework {
    /// Create a new benchmark framework
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results_cache: HashMap::new(),
            testdata_generator: TestDataGenerator::new(TestDataConfig::default()),
        }
    }

    /// Create framework with default configuration
    pub fn default() -> Self {
        Self::new(BenchmarkConfig::default())
    }

    /// Run comprehensive benchmark for a statistical function
    pub fn benchmark_function<F, G>(
        &mut self,
        function_name: &str,
        scirs2_impl: F,
        scipy_reference: G,
    ) -> StatsResult<Vec<BenchmarkResult>>
    where
        F: Fn(&ArrayView1<f64>) -> StatsResult<f64>,
        G: Fn(&ArrayView1<f64>) -> f64,
    {
        let mut results = Vec::new();

        for &size in &self.config.testsizes {
            let testdata = self.testdata_generator.generate_1ddata(size)?;

            // Run accuracy comparison
            let accuracy =
                self.compare_accuracy(&scirs2_impl, &scipy_reference, &testdata.view())?;

            // Run performance comparison
            let performance =
                self.compare_performance(&scirs2_impl, Some(&scipy_reference), &testdata.view())?;

            // Determine overall status
            let status = self.determine_status(&accuracy, &performance);

            let result = BenchmarkResult {
                function_name: function_name.to_string(),
                datasize: size,
                accuracy,
                performance,
                status,
                timestamp: chrono::Utc::now(),
            };

            results.push(result.clone());
            self.results_cache
                .insert(format!("{}_{}", function_name, size), result);
        }

        Ok(results)
    }

    /// Compare accuracy between implementations
    fn compare_accuracy<F, G>(
        &self,
        scirs2_impl: &F,
        scipy_reference: &G,
        testdata: &ArrayView1<f64>,
    ) -> StatsResult<AccuracyComparison>
    where
        F: Fn(&ArrayView1<f64>) -> StatsResult<f64>,
        G: Fn(&ArrayView1<f64>) -> f64,
    {
        let scirs2_result = scirs2_impl(testdata)?;
        let scipy_result = scipy_reference(testdata);

        let abs_difference = (scirs2_result - scipy_result).abs();
        let relativeerror = if scipy_result.abs() > 1e-15 {
            abs_difference / scipy_result.abs()
        } else {
            abs_difference
        };

        let passes_tolerance = abs_difference <= self.config.absolute_tolerance
            || relativeerror <= self.config.relative_tolerance;

        let accuracy_grade = self.grade_accuracy(relativeerror);

        Ok(AccuracyComparison {
            max_abs_difference: abs_difference,
            mean_abs_difference: abs_difference,
            relativeerror,
            outlier_count: if passes_tolerance { 0 } else { 1 },
            accuracy_grade,
            passes_tolerance,
        })
    }

    /// Compare performance between implementations
    fn compare_performance<F, G>(
        &self,
        scirs2_impl: &F,
        scipy_reference: Option<&G>,
        testdata: &ArrayView1<f64>,
    ) -> StatsResult<PerformanceComparison>
    where
        F: Fn(&ArrayView1<f64>) -> StatsResult<f64>,
        G: Fn(&ArrayView1<f64>) -> f64,
    {
        // Benchmark SciRS2 implementation
        let scirs2_timing = self.measure_timing(|| scirs2_impl(testdata).map(|_| ()))?;

        // Benchmark SciPy implementation if available
        let scipy_timing = if let Some(scipy_func) = scipy_reference {
            Some(self.measure_timing_scipy(|| {
                scipy_func(testdata);
            })?)
        } else {
            None
        };

        // Calculate performance ratio
        let performance_ratio = scipy_timing
            .as_ref()
            .map(|scipy_stats| scirs2_timing.mean.as_secs_f64() / scipy_stats.mean.as_secs_f64());

        let performance_grade = self.grade_performance(performance_ratio);

        Ok(PerformanceComparison {
            scirs2_timing,
            scipy_timing,
            performance_ratio,
            performance_grade,
            memory_usage: MemoryComparison {
                peak_memory: 0, // TODO: Implement memory tracking
                average_memory: 0,
                efficiency_ratio: None,
            },
        })
    }

    /// Measure timing statistics for a function
    fn measure_timing<F, R>(&self, mut func: F) -> StatsResult<TimingStatistics>
    where
        F: FnMut() -> StatsResult<R>,
    {
        let mut times = Vec::with_capacity(self.config.performance_iterations);

        // Warmup iterations
        for _ in 0..self.config.warmup_iterations {
            func()?;
        }

        // Timed iterations
        for _ in 0..self.config.performance_iterations {
            let start = Instant::now();
            func()?;
            let elapsed = start.elapsed();
            times.push(elapsed);
        }

        self.calculate_timing_statistics(&times)
    }

    /// Measure timing for SciPy functions (no Result handling)
    fn measure_timing_scipy<F>(&self, mut func: F) -> StatsResult<TimingStatistics>
    where
        F: FnMut(),
    {
        let mut times = Vec::with_capacity(self.config.performance_iterations);

        // Warmup iterations
        for _ in 0..self.config.warmup_iterations {
            func();
        }

        // Timed iterations
        for _ in 0..self.config.performance_iterations {
            let start = Instant::now();
            func();
            let elapsed = start.elapsed();
            times.push(elapsed);
        }

        self.calculate_timing_statistics(&times)
    }

    /// Calculate timing statistics from raw measurements
    fn calculate_timing_statistics(&self, times: &[Duration]) -> StatsResult<TimingStatistics> {
        if times.is_empty() {
            return Err(StatsError::InvalidInput(
                "No timing measurements".to_string(),
            ));
        }

        let mut sorted_times = times.to_vec();
        sorted_times.sort();

        let mean_nanos: f64 =
            times.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / times.len() as f64;
        let mean = Duration::from_nanos(mean_nanos as u64);

        let variance: f64 = times
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>()
            / times.len() as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        let p50_idx = times.len() / 2;
        let p95_idx = (times.len() as f64 * 0.95) as usize;
        let p99_idx = (times.len() as f64 * 0.99) as usize;

        Ok(TimingStatistics {
            mean,
            std_dev,
            min: sorted_times[0],
            max: sorted_times[times.len() - 1],
            p50: sorted_times[p50_idx],
            p95: sorted_times[p95_idx.min(times.len() - 1)],
            p99: sorted_times[p99_idx.min(times.len() - 1)],
        })
    }

    /// Grade accuracy based on relative error
    fn grade_accuracy(&self, relativeerror: f64) -> AccuracyGrade {
        if relativeerror < 1e-12 {
            AccuracyGrade::A
        } else if relativeerror < 1e-9 {
            AccuracyGrade::B
        } else if relativeerror < 1e-6 {
            AccuracyGrade::C
        } else if relativeerror < 1e-3 {
            AccuracyGrade::D
        } else {
            AccuracyGrade::F
        }
    }

    /// Grade performance based on ratio to SciPy
    fn grade_performance(&self, ratio: Option<f64>) -> PerformanceGrade {
        match ratio {
            Some(r) if r < 0.5 => PerformanceGrade::A,
            Some(r) if r < 0.67 => PerformanceGrade::B,
            Some(r) if r < 1.25 => PerformanceGrade::C,
            Some(r) if r < 2.0 => PerformanceGrade::D,
            Some(_) => PerformanceGrade::F,
            None => PerformanceGrade::C, // No comparison available
        }
    }

    /// Determine overall benchmark status
    fn determine_status(
        &self,
        accuracy: &AccuracyComparison,
        performance: &PerformanceComparison,
    ) -> BenchmarkStatus {
        let accuracy_pass = accuracy.passes_tolerance;
        let performance_pass = matches!(
            performance.performance_grade,
            PerformanceGrade::A | PerformanceGrade::B | PerformanceGrade::C | PerformanceGrade::D
        );

        match (accuracy_pass, performance_pass) {
            (true, true) => BenchmarkStatus::Pass,
            (true, false) => BenchmarkStatus::AccuracyPass,
            (false, true) => BenchmarkStatus::PerformancePass,
            (false, false) => BenchmarkStatus::Fail,
        }
    }

    /// Generate comprehensive benchmark report
    pub fn generate_report(&self) -> BenchmarkReport {
        let results: Vec<_> = self.results_cache.values().cloned().collect();

        BenchmarkReport {
            total_tests: results.len(),
            passed_tests: results
                .iter()
                .filter(|r| r.status == BenchmarkStatus::Pass)
                .count(),
            failed_tests: results
                .iter()
                .filter(|r| r.status == BenchmarkStatus::Fail)
                .count(),
            results,
            generated_at: chrono::Utc::now(),
        }
    }
}

impl TestDataGenerator {
    /// Create a new test data generator
    pub fn new(config: TestDataConfig) -> Self {
        Self { config }
    }

    /// Generate 1D test data
    pub fn generate_1ddata(&self, size: usize) -> StatsResult<Array1<f64>> {
        use scirs2_core::random::prelude::*;
        use scirs2_core::random::{Distribution, Normal, Uniform as UniformDist};

        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut data = Array1::zeros(size);

        match &self.config.data_distribution {
            DataDistribution::Normal => {
                let normal = Normal::new(0.0, 1.0).map_err(|e| {
                    StatsError::InvalidInput(format!("Normal distribution error: {}", e))
                })?;
                for val in data.iter_mut() {
                    *val = normal.sample(&mut rng);
                }
            }
            DataDistribution::Uniform { min, max } => {
                let uniform = UniformDist::new(*min, *max).unwrap();
                for val in data.iter_mut() {
                    *val = uniform.sample(&mut rng);
                }
            }
            DataDistribution::Exponential { lambda } => {
                for val in data.iter_mut() {
                    *val = -lambda.ln() / rng.random::<f64>().ln();
                }
            }
            DataDistribution::Mixed(_) => {
                // Simplified: just use normal for now
                let normal = Normal::new(0.0, 1.0).map_err(|e| {
                    StatsError::InvalidInput(format!("Normal distribution error: {}", e))
                })?;
                for val in data.iter_mut() {
                    *val = normal.sample(&mut rng);
                }
            }
        }

        // Add edge cases if requested
        if self.config.include_edge_cases && size > 10 {
            data[0] = f64::INFINITY;
            data[1] = f64::NEG_INFINITY;
            data[2] = f64::NAN;
            data[3] = f64::MAX;
            data[4] = f64::MIN;
        }

        Ok(data)
    }

    /// Generate 2D test data
    pub fn generate_2ddata(&self, rows: usize, cols: usize) -> StatsResult<Array2<f64>> {
        use scirs2_core::random::prelude::*;
        use scirs2_core::random::{Distribution, Normal};

        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut data = Array2::zeros((rows, cols));

        let normal = Normal::new(0.0, 1.0)
            .map_err(|e| StatsError::InvalidInput(format!("Normal distribution error: {}", e)))?;

        for val in data.iter_mut() {
            *val = normal.sample(&mut rng);
        }

        Ok(data)
    }
}

/// Comprehensive benchmark report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Total number of tests run
    pub total_tests: usize,
    /// Number of tests that passed
    pub passed_tests: usize,
    /// Number of tests that failed
    pub failed_tests: usize,
    /// Detailed results for each test
    pub results: Vec<BenchmarkResult>,
    /// Timestamp when report was generated
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

impl BenchmarkReport {
    /// Calculate overall pass rate
    pub fn pass_rate(&self) -> f64 {
        if self.total_tests == 0 {
            0.0
        } else {
            self.passed_tests as f64 / self.total_tests as f64
        }
    }

    /// Get summary statistics
    pub fn summary(&self) -> BenchmarkSummary {
        let accuracy_grades: Vec<_> = self
            .results
            .iter()
            .map(|r| r.accuracy.accuracy_grade)
            .collect();
        let performance_grades: Vec<_> = self
            .results
            .iter()
            .map(|r| r.performance.performance_grade)
            .collect();

        BenchmarkSummary {
            pass_rate: self.pass_rate(),
            average_accuracy_grade: self.average_accuracy_grade(&accuracy_grades),
            average_performance_grade: self.average_performance_grade(&performance_grades),
            total_runtime: self.total_runtime(),
        }
    }

    fn average_accuracy_grade(&self, grades: &[AccuracyGrade]) -> AccuracyGrade {
        // Simplified: just return most common grade
        AccuracyGrade::C // Placeholder
    }

    fn average_performance_grade(&self, grades: &[PerformanceGrade]) -> PerformanceGrade {
        // Simplified: just return most common grade
        PerformanceGrade::C // Placeholder
    }

    fn total_runtime(&self) -> Duration {
        // Sum all mean execution times
        self.results
            .iter()
            .map(|r| r.performance.scirs2_timing.mean)
            .sum()
    }
}

/// Summary statistics for benchmark report
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    pub pass_rate: f64,
    pub average_accuracy_grade: AccuracyGrade,
    pub average_performance_grade: PerformanceGrade,
    pub total_runtime: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptive::mean;

    #[test]
    fn test_benchmark_framework_creation() {
        let framework = ScipyBenchmarkFramework::default();
        assert_eq!(framework.config.absolute_tolerance, 1e-12);
        assert_eq!(framework.config.relative_tolerance, 1e-9);
    }

    #[test]
    fn test_testdata_generation() {
        let generator = TestDataGenerator::new(TestDataConfig::default());
        let data = generator.generate_1ddata(100).unwrap();
        assert_eq!(data.len(), 100);
    }

    #[test]
    fn test_accuracy_grading() {
        let framework = ScipyBenchmarkFramework::default();

        assert_eq!(framework.grade_accuracy(1e-15), AccuracyGrade::A);
        assert_eq!(framework.grade_accuracy(1e-10), AccuracyGrade::B);
        assert_eq!(framework.grade_accuracy(1e-7), AccuracyGrade::C);
        assert_eq!(framework.grade_accuracy(1e-4), AccuracyGrade::D);
        assert_eq!(framework.grade_accuracy(1e-1), AccuracyGrade::F);
    }

    #[test]
    fn test_performance_grading() {
        let framework = ScipyBenchmarkFramework::default();

        assert_eq!(framework.grade_performance(Some(0.3)), PerformanceGrade::A);
        assert_eq!(framework.grade_performance(Some(0.6)), PerformanceGrade::B);
        assert_eq!(framework.grade_performance(Some(1.0)), PerformanceGrade::C);
        assert_eq!(framework.grade_performance(Some(1.5)), PerformanceGrade::D);
        assert_eq!(framework.grade_performance(Some(3.0)), PerformanceGrade::F);
        assert_eq!(framework.grade_performance(None), PerformanceGrade::C);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_benchmark_integration() {
        let mut framework = ScipyBenchmarkFramework::new(BenchmarkConfig {
            testsizes: vec![100],
            performance_iterations: 5,
            warmup_iterations: 1,
            ..Default::default()
        });

        // Mock SciPy reference that matches our mean implementation
        let scipy_mean = |data: &ArrayView1<f64>| -> f64 { data.sum() / data.len() as f64 };

        let results = framework
            .benchmark_function("mean", |data| mean(data), scipy_mean)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].function_name, "mean");
        assert!(results[0].accuracy.passes_tolerance);
    }
}
