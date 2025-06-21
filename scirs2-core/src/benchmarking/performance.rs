//! # Performance Benchmarking
//!
//! This module provides specific benchmarking tools for measuring and validating
//! performance characteristics of `SciRS2` Core functions and algorithms.

use crate::benchmarking::{BenchmarkConfig, BenchmarkResult, BenchmarkRunner, BenchmarkSuite};
use crate::error::{CoreError, CoreResult, ErrorContext};
use std::time::Duration;

/// Performance benchmark categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkCategory {
    /// CPU-intensive computations
    Computation,
    /// Memory access patterns
    Memory,
    /// I/O operations
    InputOutput,
    /// Parallel processing
    Parallel,
    /// SIMD operations
    Simd,
    /// Algorithm complexity
    Algorithmic,
}

/// Performance target specification
#[derive(Debug, Clone)]
pub struct PerformanceTarget {
    /// Category of the benchmark
    pub category: BenchmarkCategory,
    /// Target execution time (maximum acceptable)
    pub target_time: Duration,
    /// Target throughput (minimum acceptable, operations per second)
    pub target_throughput: Option<f64>,
    /// Target memory usage (maximum acceptable, in bytes)
    pub target_memory: Option<usize>,
    /// Scaling factor for different input sizes
    pub scaling_factor: f64,
}

impl PerformanceTarget {
    /// Create a new performance target
    pub fn new(category: BenchmarkCategory, target_time: Duration) -> Self {
        Self {
            category,
            target_time,
            target_throughput: None,
            target_memory: None,
            scaling_factor: 1.0,
        }
    }

    /// Set target throughput
    pub fn with_throughput(mut self, throughput: f64) -> Self {
        self.target_throughput = Some(throughput);
        self
    }

    /// Set target memory usage
    pub fn with_memory(mut self, memory: usize) -> Self {
        self.target_memory = Some(memory);
        self
    }

    /// Set scaling factor
    pub fn with_scaling_factor(mut self, factor: f64) -> Self {
        self.scaling_factor = factor;
        self
    }

    /// Check if benchmark result meets this target
    pub fn is_met_by(&self, result: &BenchmarkResult, input_scale: f64) -> bool {
        let scaled_target_time = Duration::from_nanos(
            (self.target_time.as_nanos() as f64 * input_scale.powf(self.scaling_factor)) as u64,
        );

        // Check execution time
        if result.statistics.mean_execution_time > scaled_target_time {
            return false;
        }

        // Check throughput if specified
        if let Some(target_throughput) = self.target_throughput {
            let actual_throughput = 1.0 / result.statistics.mean_execution_time.as_secs_f64();
            if actual_throughput < target_throughput {
                return false;
            }
        }

        // Check memory usage if specified
        if let Some(target_memory) = self.target_memory {
            if result.statistics.mean_memory_usage > target_memory {
                return false;
            }
        }

        true
    }
}

/// Performance benchmark result with target validation
#[derive(Debug, Clone)]
pub struct PerformanceBenchmarkResult {
    /// Base benchmark result
    pub benchmark_result: BenchmarkResult,
    /// Performance target
    pub target: PerformanceTarget,
    /// Input scale factor used
    pub input_scale: f64,
    /// Whether target was met
    pub target_met: bool,
    /// Performance ratio (actual vs target)
    pub performance_ratio: f64,
}

impl PerformanceBenchmarkResult {
    /// Create a new performance benchmark result
    pub fn new(
        benchmark_result: BenchmarkResult,
        target: PerformanceTarget,
        input_scale: f64,
    ) -> Self {
        let target_met = target.is_met_by(&benchmark_result, input_scale);

        let scaled_target_time = Duration::from_nanos(
            (target.target_time.as_nanos() as f64 * input_scale.powf(target.scaling_factor)) as u64,
        );

        let performance_ratio = benchmark_result
            .statistics
            .mean_execution_time
            .as_secs_f64()
            / scaled_target_time.as_secs_f64();

        Self {
            benchmark_result,
            target,
            input_scale,
            target_met,
            performance_ratio,
        }
    }

    /// Get performance grade
    pub fn performance_grade(&self) -> PerformanceGrade {
        if self.performance_ratio <= 0.5 {
            PerformanceGrade::Excellent
        } else if self.performance_ratio <= 0.8 {
            PerformanceGrade::Good
        } else if self.performance_ratio <= 1.0 {
            PerformanceGrade::Acceptable
        } else if self.performance_ratio <= 1.5 {
            PerformanceGrade::Poor
        } else {
            PerformanceGrade::Unacceptable
        }
    }
}

/// Performance grade classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceGrade {
    /// Performance significantly exceeds target
    Excellent,
    /// Performance exceeds target
    Good,
    /// Performance meets target
    Acceptable,
    /// Performance slightly below target
    Poor,
    /// Performance significantly below target
    Unacceptable,
}

/// Performance benchmarking utilities
pub struct PerformanceBenchmarker {
    runner: BenchmarkRunner,
}

impl PerformanceBenchmarker {
    /// Create a new performance benchmarker
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            runner: BenchmarkRunner::new(config),
        }
    }

    /// Run a performance benchmark with target validation
    pub fn run_with_target<F, T>(
        &self,
        name: &str,
        target: PerformanceTarget,
        input_scale: f64,
        benchmark_fn: F,
    ) -> CoreResult<PerformanceBenchmarkResult>
    where
        F: FnMut() -> CoreResult<T>,
    {
        let benchmark_result = self.runner.run(name, benchmark_fn)?;
        Ok(PerformanceBenchmarkResult::new(
            benchmark_result,
            target,
            input_scale,
        ))
    }

    /// Run computational performance benchmarks
    pub fn benchmark_computation<F, T>(
        &self,
        name: &str,
        computation_fn: F,
        expected_complexity: f64,
    ) -> CoreResult<PerformanceBenchmarkResult>
    where
        F: FnMut() -> CoreResult<T>,
    {
        let target =
            PerformanceTarget::new(BenchmarkCategory::Computation, Duration::from_millis(100))
                .with_scaling_factor(expected_complexity);

        self.run_with_target(name, target, 1.0, computation_fn)
    }

    /// Run memory access performance benchmarks
    pub fn benchmark_memory_access<F, T>(
        &self,
        name: &str,
        memory_fn: F,
        data_size: usize,
    ) -> CoreResult<PerformanceBenchmarkResult>
    where
        F: FnMut() -> CoreResult<T>,
    {
        let target = PerformanceTarget::new(
            BenchmarkCategory::Memory,
            Duration::from_micros(10),
        )
        .with_memory(data_size * 2) // Allow 2x memory overhead
        .with_scaling_factor(1.0); // Linear scaling with data size

        let scale = data_size as f64 / 1024.0; // Scale relative to 1KB
        self.run_with_target(name, target, scale, memory_fn)
    }

    /// Run algorithmic complexity benchmarks
    pub fn benchmark_algorithm_scaling<F, T>(
        &self,
        name: &str,
        algorithm_fn: F,
        input_sizes: Vec<usize>,
        expected_complexity: f64,
    ) -> CoreResult<Vec<PerformanceBenchmarkResult>>
    where
        F: Fn(usize) -> CoreResult<T> + Clone,
    {
        let mut results = Vec::new();
        let base_target =
            PerformanceTarget::new(BenchmarkCategory::Algorithmic, Duration::from_millis(10))
                .with_scaling_factor(expected_complexity);

        for size in &input_sizes {
            let size_name = format!("{}(n={})", name, size);
            let algorithm_clone = algorithm_fn.clone();

            let benchmark_result = self.runner.run(&size_name, || algorithm_clone(*size))?;
            let scale = *size as f64 / input_sizes[0] as f64;
            let performance_result =
                PerformanceBenchmarkResult::new(benchmark_result, base_target.clone(), scale);

            results.push(performance_result);
        }

        Ok(results)
    }

    /// Benchmark SIMD operations
    #[cfg(feature = "simd")]
    pub fn benchmark_simd<F, G, T>(
        &self,
        name: &str,
        simd_fn: F,
        scalar_fn: G,
        data_size: usize,
    ) -> CoreResult<(PerformanceBenchmarkResult, PerformanceBenchmarkResult, f64)>
    where
        F: FnMut() -> CoreResult<T>,
        G: FnMut() -> CoreResult<T>,
    {
        // Benchmark SIMD version
        let simd_target =
            PerformanceTarget::new(BenchmarkCategory::Simd, Duration::from_micros(100));
        let simd_result = self.run_with_target(
            &format!("{}_simd", name),
            simd_target,
            data_size as f64 / 1000.0,
            simd_fn,
        )?;

        // Benchmark scalar version
        let scalar_target =
            PerformanceTarget::new(BenchmarkCategory::Computation, Duration::from_millis(1));
        let scalar_result = self.run_with_target(
            &format!("{}_scalar", name),
            scalar_target,
            data_size as f64 / 1000.0,
            scalar_fn,
        )?;

        // Calculate speedup
        let speedup = scalar_result
            .benchmark_result
            .statistics
            .mean_execution_time
            .as_secs_f64()
            / simd_result
                .benchmark_result
                .statistics
                .mean_execution_time
                .as_secs_f64();

        Ok((simd_result, scalar_result, speedup))
    }

    /// Benchmark parallel operations
    #[cfg(feature = "parallel")]
    pub fn benchmark_parallel<F, G, T>(
        &self,
        name: &str,
        parallel_fn: F,
        sequential_fn: G,
        thread_count: usize,
    ) -> CoreResult<(PerformanceBenchmarkResult, PerformanceBenchmarkResult, f64)>
    where
        F: FnMut() -> CoreResult<T>,
        G: FnMut() -> CoreResult<T>,
    {
        // Benchmark parallel version
        let parallel_target =
            PerformanceTarget::new(BenchmarkCategory::Parallel, Duration::from_millis(100));
        let parallel_result = self.run_with_target(
            &format!("{}_parallel", name),
            parallel_target,
            1.0,
            parallel_fn,
        )?;

        // Benchmark sequential version
        let sequential_target =
            PerformanceTarget::new(BenchmarkCategory::Computation, Duration::from_millis(500));
        let sequential_result = self.run_with_target(
            &format!("{}_sequential", name),
            sequential_target,
            1.0,
            sequential_fn,
        )?;

        // Calculate efficiency
        let theoretical_speedup = thread_count as f64;
        let actual_speedup = sequential_result
            .benchmark_result
            .statistics
            .mean_execution_time
            .as_secs_f64()
            / parallel_result
                .benchmark_result
                .statistics
                .mean_execution_time
                .as_secs_f64();
        let efficiency = actual_speedup / theoretical_speedup;

        Ok((parallel_result, sequential_result, efficiency))
    }
}

/// Create standard performance benchmark suites
pub struct StandardBenchmarks;

impl StandardBenchmarks {
    /// Create a computational benchmark suite
    pub fn create_computation_suite(config: BenchmarkConfig) -> BenchmarkSuite {
        let mut suite = BenchmarkSuite::new("computation_performance", config);

        // Basic arithmetic operations
        suite.add_benchmark(|runner| {
            runner.run("arithmetic_operations", || {
                let mut sum = 0.0f64;
                for i in 0..10000 {
                    sum += (i as f64).sin().cos().sqrt();
                }
                Ok(sum)
            })
        });

        // Vector operations
        suite.add_benchmark(|runner| {
            runner.run("vector_operations", || {
                let a: Vec<f64> = (0..10000).map(|i| i as f64).collect();
                let b: Vec<f64> = (0..10000).map(|i| (i as f64) * 2.0).collect();
                let result: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
                Ok(result.iter().sum::<f64>())
            })
        });

        // Matrix operations (simplified)
        suite.add_benchmark(|runner| {
            runner.run("matrix_multiplication", || {
                let size = 100;
                let a: Vec<Vec<f64>> = (0..size)
                    .map(|i| (0..size).map(|j| (i * j) as f64).collect())
                    .collect();
                let b: Vec<Vec<f64>> = (0..size)
                    .map(|i| (0..size).map(|j| (i + j) as f64).collect())
                    .collect();

                let mut c = vec![vec![0.0; size]; size];
                for i in 0..size {
                    for j in 0..size {
                        #[allow(clippy::needless_range_loop)]
                        for k in 0..size {
                            c[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }

                Ok(c[0][0])
            })
        });

        suite
    }

    /// Create a memory benchmark suite
    pub fn create_memory_suite(config: BenchmarkConfig) -> BenchmarkSuite {
        let mut suite = BenchmarkSuite::new("memory_performance", config);

        // Memory allocation
        suite.add_benchmark(|runner| {
            runner.run("memory_allocation", || {
                let mut vectors = Vec::new();
                for i in 0..1000 {
                    vectors.push(vec![i as f64; 1000]);
                }
                Ok(vectors.len())
            })
        });

        // Sequential memory access
        suite.add_benchmark(|runner| {
            runner.run("sequential_access", || {
                let data: Vec<f64> = (0..1000000).map(|i| i as f64).collect();
                let sum: f64 = data.iter().sum();
                Ok(sum)
            })
        });

        // Random memory access
        suite.add_benchmark(|runner| {
            runner.run("random_access", || {
                let data: Vec<f64> = (0..100000).map(|i| i as f64).collect();
                let mut sum = 0.0;
                for i in (0..data.len()).step_by(1000) {
                    sum += data[i];
                }
                Ok(sum)
            })
        });

        suite
    }

    /// Create an I/O benchmark suite
    pub fn create_io_suite(config: BenchmarkConfig) -> BenchmarkSuite {
        let mut suite = BenchmarkSuite::new("io_performance", config);

        // File I/O
        suite.add_benchmark(|runner| {
            runner.run_with_setup(
                "file_io",
                || {
                    use tempfile::NamedTempFile;
                    let temp_file = NamedTempFile::new().map_err(|e| {
                        CoreError::IoError(ErrorContext::new(format!(
                            "Failed to create temp file: {}",
                            e
                        )))
                    })?;
                    Ok(temp_file)
                },
                |temp_file| {
                    use std::io::Write;
                    let data = vec![42u8; 10000];
                    temp_file.write_all(&data).map_err(|e| {
                        CoreError::IoError(ErrorContext::new(format!("Failed to write: {}", e)))
                    })?;
                    temp_file.flush().map_err(|e| {
                        CoreError::IoError(ErrorContext::new(format!("Failed to flush: {}", e)))
                    })?;
                    Ok(data.len())
                },
                |temp_file| {
                    drop(temp_file);
                    Ok(())
                },
            )
        });

        suite
    }

    /// Create a comprehensive benchmark suite
    pub fn create_comprehensive_suite(config: BenchmarkConfig) -> BenchmarkSuite {
        let mut suite = BenchmarkSuite::new("comprehensive_performance", config.clone());

        // Add computation benchmarks
        let _comp_suite = Self::create_computation_suite(config.clone());
        // Note: This is a simplified version - in practice you'd need to extract benchmarks
        suite.add_benchmark(|runner| {
            runner.run("comprehensive_computation", || {
                // Simplified computation benchmark
                let mut result = 0.0;
                for i in 0..1000 {
                    result += (i as f64).sin();
                }
                Ok(result)
            })
        });

        // Add memory benchmarks
        suite.add_benchmark(|runner| {
            runner.run("comprehensive_memory", || {
                let data: Vec<f64> = (0..10000).map(|i| i as f64).collect();
                Ok(data.iter().sum::<f64>())
            })
        });

        suite
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_target() {
        let target =
            PerformanceTarget::new(BenchmarkCategory::Computation, Duration::from_millis(100))
                .with_throughput(1000.0)
                .with_memory(1024)
                .with_scaling_factor(2.0);

        assert_eq!(target.category, BenchmarkCategory::Computation);
        assert_eq!(target.target_time, Duration::from_millis(100));
        assert_eq!(target.target_throughput, Some(1000.0));
        assert_eq!(target.target_memory, Some(1024));
        assert_eq!(target.scaling_factor, 2.0);
    }

    #[test]
    fn test_performance_grade() {
        let config = BenchmarkConfig::default();
        let mut result = crate::benchmarking::BenchmarkResult::new("test".to_string(), config);

        // Add some measurements
        result.add_measurement(crate::benchmarking::BenchmarkMeasurement::new(
            Duration::from_millis(50),
        ));
        result.finalize().unwrap();

        let target =
            PerformanceTarget::new(BenchmarkCategory::Computation, Duration::from_millis(100));

        let perf_result = PerformanceBenchmarkResult::new(result, target, 1.0);
        // With measurement of 50ms and target of 100ms, ratio is 0.5, which is Excellent
        assert_eq!(perf_result.performance_grade(), PerformanceGrade::Excellent);
    }

    #[test]
    fn test_performance_benchmarker() {
        let config = BenchmarkConfig::new()
            .with_warmup_iterations(1)
            .with_measurement_iterations(5);
        let benchmarker = PerformanceBenchmarker::new(config);

        let result = benchmarker
            .benchmark_computation(
                "test_computation",
                || {
                    std::thread::sleep(Duration::from_micros(100));
                    Ok(42)
                },
                1.0,
            )
            .unwrap();

        assert!(result.benchmark_result.statistics.mean_execution_time > Duration::from_micros(50));
        assert_eq!(result.target.category, BenchmarkCategory::Computation);
    }
}
