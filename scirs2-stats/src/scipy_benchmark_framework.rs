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
///
/// Populated from real resident-memory (RSS) samples taken immediately before and
/// after each timed iteration when the crate's `memory_tracking` feature is enabled
/// (see the internal `ScipyBenchmarkFramework::measure_timing` helper). Without that
/// feature, both fields are honest zeros rather than fabricated numbers.
///
/// RSS deltas are inherently approximate: the OS does not always reclaim freed pages
/// immediately, and other allocator/thread activity in the process can perturb an
/// individual sample. Treat these figures as directional evidence of memory pressure
/// rather than exact byte counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryComparison {
    /// Peak memory usage (bytes) — the largest single-iteration RSS delta observed
    pub peak_memory: usize,
    /// Average memory usage during execution (bytes) — mean RSS delta across iterations
    pub average_memory: usize,
    /// Memory efficiency ratio vs SciPy (SciRS2 average memory / SciPy average memory)
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

        // Edge-case datasets (see generate_1ddata) deliberately inject NaN/Inf,
        // and the two implementations are expected to propagate them the same
        // way. An arithmetic difference against a NaN operand is itself NaN,
        // which fails every `<=` comparison below even when both sides agree
        // exactly — so non-finite results must be compared structurally
        // instead of arithmetically.
        if !scirs2_result.is_finite() || !scipy_result.is_finite() {
            let agree =
                (scirs2_result.is_nan() && scipy_result.is_nan()) || scirs2_result == scipy_result; // handles matching +-inf
            return Ok(AccuracyComparison {
                max_abs_difference: if agree { 0.0 } else { f64::INFINITY },
                mean_abs_difference: if agree { 0.0 } else { f64::INFINITY },
                relativeerror: if agree { 0.0 } else { f64::INFINITY },
                outlier_count: if agree { 0 } else { 1 },
                accuracy_grade: if agree {
                    AccuracyGrade::A
                } else {
                    AccuracyGrade::F
                },
                passes_tolerance: agree,
            });
        }

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
        // Benchmark SciRS2 implementation (timing + resident-memory sampling)
        let (scirs2_timing, scirs2_memory) =
            self.measure_timing(|| scirs2_impl(testdata).map(|_| ()))?;

        // Benchmark SciPy implementation if available (timing + resident-memory sampling)
        let (scipy_timing, scipy_memory) = if let Some(scipy_func) = scipy_reference {
            let (timing, memory) = self.measure_timing_scipy(|| {
                scipy_func(testdata);
            })?;
            (Some(timing), Some(memory))
        } else {
            (None, None)
        };

        // Calculate performance ratio
        let performance_ratio = scipy_timing
            .as_ref()
            .map(|scipy_stats| scirs2_timing.mean.as_secs_f64() / scipy_stats.mean.as_secs_f64());

        let performance_grade = self.grade_performance(performance_ratio);

        // Memory efficiency ratio (SciRS2 / SciPy average memory), mirroring how
        // `performance_ratio` compares SciRS2 vs SciPy timing above. Only meaningful
        // when a SciPy baseline measurement with nonzero average memory is available.
        let efficiency_ratio = scipy_memory.as_ref().and_then(|scipy_mem| {
            if scipy_mem.average_memory > 0 {
                Some(scirs2_memory.average_memory as f64 / scipy_mem.average_memory as f64)
            } else {
                None
            }
        });

        Ok(PerformanceComparison {
            scirs2_timing,
            scipy_timing,
            performance_ratio,
            performance_grade,
            memory_usage: MemoryComparison {
                peak_memory: scirs2_memory.peak_memory,
                average_memory: scirs2_memory.average_memory,
                efficiency_ratio,
            },
        })
    }

    /// Measure timing statistics for a function, together with resident-memory (RSS)
    /// statistics sampled around each timed iteration.
    ///
    /// When the `memory_tracking` feature is enabled, [`scirs2_core::profiling::MemoryStats::current`]
    /// (a Pure-Rust RSS profiler — Mach `task_info` on macOS, `/proc/self/statm` on Linux)
    /// is sampled immediately before and after every timed call to `func`, and the
    /// (saturating) per-iteration delta feeds the returned [`MemoryComparison`]. RSS
    /// deltas are inherently approximate — the OS does not always reclaim freed pages
    /// immediately, and other allocator/thread activity in the process can perturb a
    /// given sample — so treat the reported figures as directional rather than exact.
    ///
    /// Without the `memory_tracking` feature, the memory component is an honest zero
    /// (documented as such) rather than a fabricated measurement.
    #[cfg(feature = "memory_tracking")]
    fn measure_timing<F, R>(&self, mut func: F) -> StatsResult<(TimingStatistics, MemoryComparison)>
    where
        F: FnMut() -> StatsResult<R>,
    {
        use scirs2_core::profiling::MemoryStats;

        let mut times = Vec::with_capacity(self.config.performance_iterations);
        let mut memory_deltas = Vec::with_capacity(self.config.performance_iterations);

        // Warmup iterations
        for _ in 0..self.config.warmup_iterations {
            func()?;
        }

        // Timed iterations, sampling RSS immediately before/after each call. The
        // call's return value is deliberately kept alive (bound to `result`) until
        // after the "after" sample, then dropped — so memory owned by the return
        // value itself (e.g. a freshly allocated buffer) is captured in the delta
        // instead of being silently freed before we get a chance to observe it.
        for _ in 0..self.config.performance_iterations {
            let before_resident = MemoryStats::current()?.resident;
            let start = Instant::now();
            let result = func()?;
            let elapsed = start.elapsed();
            let after_resident = MemoryStats::current()?.resident;
            drop(result);

            times.push(elapsed);
            // Memory can also decrease between samples (deallocation, OS page
            // reclamation); clamp negative deltas to 0 instead of treating them as
            // meaningful growth (or wrapping, since these are unsigned byte counts).
            memory_deltas.push(after_resident.saturating_sub(before_resident));
        }

        let timing_stats = self.calculate_timing_statistics(&times)?;
        let memory_stats = Self::summarize_memory_deltas(&memory_deltas);

        Ok((timing_stats, memory_stats))
    }

    /// Measure timing statistics for a function (memory-tracking disabled build).
    ///
    /// Real RSS-based memory tracking requires the `memory_tracking` feature (which
    /// enables scirs2-core's Pure-Rust `profiling_memory` RSS profiler). Without it we
    /// report honest zeros for memory rather than fabricating a measurement.
    #[cfg(not(feature = "memory_tracking"))]
    fn measure_timing<F, R>(&self, mut func: F) -> StatsResult<(TimingStatistics, MemoryComparison)>
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

        let timing_stats = self.calculate_timing_statistics(&times)?;
        // `memory_tracking` feature not enabled: report honest zeros rather than a
        // fabricated measurement (see struct docs on `MemoryComparison`).
        let memory_stats = MemoryComparison {
            peak_memory: 0,
            average_memory: 0,
            efficiency_ratio: None,
        };

        Ok((timing_stats, memory_stats))
    }

    /// Measure timing (and RSS memory, when `memory_tracking` is enabled) for SciPy
    /// functions (no `Result` handling). Mirrors [`Self::measure_timing`]'s loop
    /// structure and sampling strategy.
    #[cfg(feature = "memory_tracking")]
    fn measure_timing_scipy<F>(
        &self,
        mut func: F,
    ) -> StatsResult<(TimingStatistics, MemoryComparison)>
    where
        F: FnMut(),
    {
        use scirs2_core::profiling::MemoryStats;

        let mut times = Vec::with_capacity(self.config.performance_iterations);
        let mut memory_deltas = Vec::with_capacity(self.config.performance_iterations);

        // Warmup iterations
        for _ in 0..self.config.warmup_iterations {
            func();
        }

        // Timed iterations, sampling RSS immediately before/after each call
        for _ in 0..self.config.performance_iterations {
            let before_resident = MemoryStats::current()?.resident;
            let start = Instant::now();
            func();
            let elapsed = start.elapsed();
            let after_resident = MemoryStats::current()?.resident;

            times.push(elapsed);
            memory_deltas.push(after_resident.saturating_sub(before_resident));
        }

        let timing_stats = self.calculate_timing_statistics(&times)?;
        let memory_stats = Self::summarize_memory_deltas(&memory_deltas);

        Ok((timing_stats, memory_stats))
    }

    /// Measure timing for SciPy functions (no `Result` handling; memory-tracking
    /// disabled build — see [`Self::measure_timing`] for the rationale).
    #[cfg(not(feature = "memory_tracking"))]
    fn measure_timing_scipy<F>(
        &self,
        mut func: F,
    ) -> StatsResult<(TimingStatistics, MemoryComparison)>
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

        let timing_stats = self.calculate_timing_statistics(&times)?;
        let memory_stats = MemoryComparison {
            peak_memory: 0,
            average_memory: 0,
            efficiency_ratio: None,
        };

        Ok((timing_stats, memory_stats))
    }

    /// Fold a series of per-iteration RSS deltas (bytes) into a [`MemoryComparison`].
    ///
    /// `peak_memory` is the largest single-iteration delta (saturating growth only);
    /// `average_memory` is the mean delta across all iterations. `efficiency_ratio` is
    /// left `None` here — it is filled in by the caller once a SciPy baseline (if any)
    /// is also available.
    #[cfg(feature = "memory_tracking")]
    fn summarize_memory_deltas(deltas: &[usize]) -> MemoryComparison {
        let peak_memory = deltas.iter().copied().max().unwrap_or(0);
        let average_memory = if deltas.is_empty() {
            0
        } else {
            (deltas.iter().sum::<usize>() as f64 / deltas.len() as f64).round() as usize
        };

        MemoryComparison {
            peak_memory,
            average_memory,
            efficiency_ratio: None,
        }
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
                let uniform = UniformDist::new(*min, *max).expect("Operation failed");
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
        let data = generator.generate_1ddata(100).expect("Operation failed");
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
            .expect("Operation failed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].function_name, "mean");
        assert!(results[0].accuracy.passes_tolerance);
    }

    // ------------------------------------------------------------------
    // Real RSS memory-tracking tests (require the `memory_tracking` feature,
    // e.g. `cargo test -p scirs2-stats --features memory_tracking`, or any
    // invocation with `--all-features`).
    //
    // RSS sampling is page-granularity and OS/allocator-dependent (freed pages
    // are not always reclaimed immediately), so these tests assert relative /
    // ordering properties rather than exact byte counts.
    // ------------------------------------------------------------------

    /// Per-call growth size (~1.6 MiB of f64) for the monotonically-growing buffer
    /// used by the memory-tracking tests below — big enough that its resident-memory
    /// footprint is unambiguously distinguishable from sampling noise (page-granularity
    /// jitter, allocator bookkeeping, etc).
    #[cfg(feature = "memory_tracking")]
    const MEMORY_TEST_GROWTH_LEN: usize = 200_000;

    #[cfg(feature = "memory_tracking")]
    #[test]
    fn test_memory_tracking_allocating_closure_reports_nonzero_memory() {
        let framework = ScipyBenchmarkFramework::new(BenchmarkConfig {
            performance_iterations: 20,
            warmup_iterations: 2,
            ..Default::default()
        });

        // Deliberately *grow* a buffer captured by the (`FnMut`) closure on every call,
        // rather than allocating-then-freeing a fresh same-sized `Vec` each time. The
        // latter was tried first and reliably measured a peak/average of exactly 0 on
        // macOS: `measure_timing`'s 2 warmup calls already prime the allocator's
        // same-size free list/large-allocation cache, so every "after" sample in the
        // measured loop finds the identical (already-resident) pages reused for the
        // new allocation, showing zero incremental RSS growth. A buffer that only ever
        // grows (never freed until the closure itself drops at the end of this test)
        // sidesteps that reuse entirely and gives a deterministic, platform-independent
        // nonzero delta on every iteration.
        let mut buffer: Vec<f64> = Vec::new();
        let (_, memory) = framework
            .measure_timing(move || -> StatsResult<()> {
                buffer.extend(std::iter::repeat_n(1.0_f64, MEMORY_TEST_GROWTH_LEN));
                Ok(())
            })
            .expect("Operation failed");

        assert!(
            memory.peak_memory > 0,
            "expected nonzero peak resident-memory delta for an allocating closure, got {}",
            memory.peak_memory
        );
        assert!(
            memory.average_memory > 0,
            "expected nonzero average resident-memory delta for an allocating closure, got {}",
            memory.average_memory
        );
    }

    #[cfg(feature = "memory_tracking")]
    #[test]
    fn test_memory_tracking_trivial_closure_much_smaller_than_allocating() {
        let framework = ScipyBenchmarkFramework::new(BenchmarkConfig {
            performance_iterations: 20,
            warmup_iterations: 2,
            ..Default::default()
        });

        // See `test_memory_tracking_allocating_closure_reports_nonzero_memory` for why
        // this uses a monotonically-growing captured buffer rather than a fresh
        // allocate-then-free `Vec` per call.
        let mut buffer: Vec<f64> = Vec::new();
        let (_, allocating_memory) = framework
            .measure_timing(move || -> StatsResult<()> {
                buffer.extend(std::iter::repeat_n(1.0_f64, MEMORY_TEST_GROWTH_LEN));
                Ok(())
            })
            .expect("Operation failed");

        // A trivial closure that touches no heap memory at all.
        let (_, trivial_memory) = framework
            .measure_timing(|| -> StatsResult<i32> { Ok(1 + 1) })
            .expect("Operation failed");

        assert!(
            allocating_memory.peak_memory > 0,
            "sanity check: allocating closure should itself report nonzero peak memory, got {}",
            allocating_memory.peak_memory
        );
        // Contrast rather than asserting an arbitrary absolute bound: the trivial
        // closure's footprint must be much smaller than the ~15.3 MiB allocating
        // closure's, not merely nonnegative (which would be vacuous for a usize).
        assert!(
            trivial_memory.peak_memory < allocating_memory.peak_memory,
            "expected trivial closure's peak memory ({}) to be much smaller than the \
             allocating closure's ({})",
            trivial_memory.peak_memory,
            allocating_memory.peak_memory
        );
        assert!(
            trivial_memory.average_memory < allocating_memory.average_memory,
            "expected trivial closure's average memory ({}) to be much smaller than the \
             allocating closure's ({})",
            trivial_memory.average_memory,
            allocating_memory.average_memory
        );
    }

    #[cfg(feature = "memory_tracking")]
    #[test]
    fn test_memory_tracking_wired_into_compare_performance() {
        use std::cell::RefCell;

        // End-to-end: `compare_performance` (used by `benchmark_function`) should
        // surface the same real memory tracking, including a computed
        // `efficiency_ratio` once both SciRS2 and SciPy sides report nonzero
        // average memory.
        //
        // `compare_performance`'s SciRS2 timing/memory loop discards the closure's
        // own return value (`.map(|_| ())`), so an allocation that is built *and*
        // freed entirely inside `scirs2_impl`'s body would depend on whether the
        // allocator/OS happens to reclaim those pages before the "after" sample —
        // exactly the kind of nondeterminism this feature's docs warn about. To get
        // a deterministic, platform-independent signal instead, each closure here
        // appends to a `RefCell`-captured buffer that is never freed until the test
        // itself ends, so resident memory only ever grows across iterations.
        let scirs2_growing: RefCell<Vec<f64>> = RefCell::new(Vec::new());
        let scipy_growing: RefCell<Vec<f64>> = RefCell::new(Vec::new());
        const GROWTH_PER_CALL: usize = 200_000; // ~1.5 MiB of f64 per call

        let framework = ScipyBenchmarkFramework::new(BenchmarkConfig {
            performance_iterations: 10,
            warmup_iterations: 1,
            ..Default::default()
        });

        let testdata = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Both "implementations" deliberately grow captured state so both sides of
        // the comparison report nonzero average memory (letting us exercise the
        // `efficiency_ratio` computation, not just the raw peak/average fields).
        // `RefCell` gives interior mutability so these closures can still satisfy
        // the `Fn` bound `compare_performance` requires.
        let scirs2_impl = |data: &ArrayView1<f64>| -> StatsResult<f64> {
            scirs2_growing
                .borrow_mut()
                .extend(std::iter::repeat_n(1.0_f64, GROWTH_PER_CALL));
            Ok(data.sum())
        };
        let scipy_reference = |data: &ArrayView1<f64>| -> f64 {
            scipy_growing
                .borrow_mut()
                .extend(std::iter::repeat_n(1.0_f64, GROWTH_PER_CALL));
            data.sum()
        };

        let performance = framework
            .compare_performance(&scirs2_impl, Some(&scipy_reference), &testdata.view())
            .expect("Operation failed");

        assert!(
            performance.memory_usage.peak_memory > 0,
            "expected nonzero peak memory from an allocating benchmarked closure, got {}",
            performance.memory_usage.peak_memory
        );
        assert!(
            performance.memory_usage.average_memory > 0,
            "expected nonzero average memory from an allocating benchmarked closure, got {}",
            performance.memory_usage.average_memory
        );
        assert!(
            performance.memory_usage.efficiency_ratio.is_some(),
            "expected an efficiency_ratio once both SciRS2 and SciPy sides allocate"
        );

        // Keep the growing buffers alive (and their growth "used") through the end
        // of the test, rather than letting the borrow checker/optimizer treat the
        // accumulated data as dead.
        assert!(scirs2_growing.borrow().len() >= GROWTH_PER_CALL);
        assert!(scipy_growing.borrow().len() >= GROWTH_PER_CALL);
    }
}
