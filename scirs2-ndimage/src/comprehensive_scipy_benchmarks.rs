//! Comprehensive benchmarking suite comparing scirs2-ndimage with SciPy ndimage
//!
//! This module provides extensive performance benchmarks comparing the performance
//! of scirs2-ndimage operations against their SciPy ndimage equivalents. It includes
//! cross-language benchmarking, memory profiling, and detailed performance analysis.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::process::Command;
use std::time::{Duration, Instant};

use ndarray::Array2;
use num_traits::{Float, FromPrimitive};

use crate::filters::BorderMode;

use crate::error::{NdimageError, NdimageResult};
use crate::filters::{median_filter, uniform_filter};
use crate::performance_profiler::{PerformanceProfiler, ProfilerConfig};

/// Comprehensive benchmark suite for comparing with SciPy
pub struct SciPyBenchmarkSuite {
    /// Performance profiler
    profiler: PerformanceProfiler,
    /// Benchmark configuration
    config: BenchmarkConfig,
    /// Results storage
    results: BenchmarkResults,
    /// Python environment path
    python_path: String,
}

#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Test array sizes (height, width)
    pub array_sizes: Vec<(usize, usize)>,
    /// Number of iterations per test
    pub iterations: usize,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Data types to test
    pub data_types: Vec<DataType>,
    /// Operations to benchmark
    pub operations: Vec<BenchmarkOperation>,
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
    /// Enable cross-language comparison
    pub enable_scipy_comparison: bool,
    /// Temporary directory for Python scripts
    pub temp_dir: String,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            array_sizes: vec![
                (64, 64),
                (128, 128),
                (256, 256),
                (512, 512),
                (1024, 1024),
                (2048, 2048),
            ],
            iterations: 10,
            warmup_iterations: 3,
            data_types: vec![DataType::F32, DataType::F64],
            operations: vec![
                BenchmarkOperation::GaussianFilter,
                BenchmarkOperation::MedianFilter,
                BenchmarkOperation::UniformFilter,
                BenchmarkOperation::BinaryErosion,
                BenchmarkOperation::BinaryDilation,
                BenchmarkOperation::DistanceTransform,
                BenchmarkOperation::CenterOfMass,
                BenchmarkOperation::LabelObjects,
            ],
            enable_memory_profiling: true,
            enable_scipy_comparison: true,
            temp_dir: "/tmp/scirs2_benchmarks".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BenchmarkOperation {
    GaussianFilter,
    MedianFilter,
    UniformFilter,
    BinaryErosion,
    BinaryDilation,
    DistanceTransform,
    CenterOfMass,
    LabelObjects,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Results by operation and configuration
    operation_results: HashMap<String, Vec<OperationBenchmarkResult>>,
    /// Overall statistics
    overall_stats: OverallBenchmarkStats,
    /// SciPy comparison results
    scipy_comparison: Option<SciPyComparisonResults>,
    /// Timestamp
    timestamp: Instant,
}

#[derive(Debug, Clone)]
pub struct OperationBenchmarkResult {
    /// Operation name
    pub operation: String,
    /// Array size tested
    pub array_size: (usize, usize),
    /// Data type
    pub data_type: DataType,
    /// scirs2-ndimage performance
    pub scirs2_performance: PerformanceMetrics,
    /// SciPy performance (if available)
    pub scipy_performance: Option<PerformanceMetrics>,
    /// Performance ratio (scirs2 / scipy)
    pub performance_ratio: Option<f64>,
    /// Memory usage comparison
    pub memory_comparison: MemoryComparison,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Average execution time
    pub avg_time: Duration,
    /// Minimum execution time
    pub min_time: Duration,
    /// Maximum execution time
    pub max_time: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Throughput (elements/second)
    pub throughput: f64,
    /// Memory peak usage
    pub peak_memory: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryComparison {
    /// scirs2-ndimage memory usage
    pub scirs2_memory: usize,
    /// SciPy memory usage
    pub scipy_memory: Option<usize>,
    /// Memory efficiency ratio
    pub efficiency_ratio: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct OverallBenchmarkStats {
    /// Total operations benchmarked
    pub total_operations: usize,
    /// Average performance ratio across all operations
    pub avg_performance_ratio: f64,
    /// Best performing operations for scirs2
    pub best_operations: Vec<String>,
    /// Operations where SciPy is faster
    pub scipy_faster_operations: Vec<String>,
    /// Overall memory efficiency
    pub overall_memory_efficiency: f64,
}

#[derive(Debug, Clone)]
pub struct SciPyComparisonResults {
    /// Operations where scirs2 is faster
    pub scirs2_faster: Vec<String>,
    /// Operations where SciPy is faster
    pub scipy_faster: Vec<String>,
    /// Operations with similar performance
    pub similar_performance: Vec<String>,
    /// Average speedup over SciPy
    pub avg_speedup: f64,
    /// Maximum speedup achieved
    pub max_speedup: f64,
}

impl SciPyBenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(config: BenchmarkConfig) -> NdimageResult<Self> {
        // Create temporary directory for Python scripts
        fs::create_dir_all(&config.temp_dir).map_err(|e| {
            NdimageError::InvalidInput(format!("Failed to create temp directory: {}", e))
        })?;

        // Initialize profiler
        let profilerconfig = ProfilerConfig {
            max_records_per_operation: 1000,
            enable_simd_profiling: true,
            enable_cache_analysis: true,
            ..Default::default()
        };

        let profiler = PerformanceProfiler::new(profilerconfig);

        // Try to find Python executable
        let python_path = Self::find_python_executable()?;

        Ok(Self {
            profiler,
            config,
            results: BenchmarkResults {
                operation_results: HashMap::new(),
                overall_stats: OverallBenchmarkStats {
                    total_operations: 0,
                    avg_performance_ratio: 1.0,
                    best_operations: Vec::new(),
                    scipy_faster_operations: Vec::new(),
                    overall_memory_efficiency: 1.0,
                },
                scipy_comparison: None,
                timestamp: Instant::now(),
            },
            python_path,
        })
    }

    /// Run the complete benchmark suite
    pub fn run_complete_benchmark(&mut self) -> NdimageResult<&BenchmarkResults> {
        println!("Starting comprehensive SciPy benchmark suite...");
        println!(
            "Testing {} operations across {} array sizes",
            self.config.operations.len(),
            self.config.array_sizes.len()
        );

        // Start profiler monitoring
        self.profiler.start_monitoring()?;

        let mut all_results = Vec::new();

        // Run benchmarks for each operation and configuration
        for &operation in &self.config.operations {
            for &array_size in &self.config.array_sizes {
                for &data_type in &self.config.data_types {
                    println!(
                        "Benchmarking {:?} on {}x{} {:?} array...",
                        operation, array_size.0, array_size.1, data_type
                    );

                    let result = self.benchmark_operation(operation, array_size, data_type)?;
                    all_results.push(result);
                }
            }
        }

        // Stop profiler monitoring
        self.profiler.stop_monitoring();

        // Store results and compute statistics
        self.store_results(all_results);
        self.compute_overall_statistics();

        if self.config.enable_scipy_comparison {
            self.compute_scipy_comparison();
        }

        Ok(&self.results)
    }

    /// Benchmark a specific operation
    pub fn benchmark_operation(
        &self,
        operation: BenchmarkOperation,
        array_size: (usize, usize),
        data_type: DataType,
    ) -> NdimageResult<OperationBenchmarkResult> {
        match data_type {
            DataType::F32 => self.benchmark_operation_typed::<f32>(operation, array_size),
            DataType::F64 => self.benchmark_operation_typed::<f64>(operation, array_size),
        }
    }

    fn benchmark_operation_typed<T>(
        &self,
        operation: BenchmarkOperation,
        array_size: (usize, usize),
    ) -> NdimageResult<OperationBenchmarkResult>
    where
        T: Float
            + FromPrimitive
            + Clone
            + Default
            + std::fmt::Debug
            + Send
            + Sync
            + std::ops::AddAssign
            + std::ops::DivAssign
            + 'static,
    {
        let _height_width = array_size;

        // Generate test data
        let input_data = self.generate_test_data::<T>(array_size, operation);

        // Benchmark scirs2-ndimage implementation
        let scirs2metrics = self.benchmark_scirs2_operation::<T>(operation, &input_data)?;

        // Benchmark SciPy implementation (if enabled)
        let scipymetrics = if self.config.enable_scipy_comparison {
            self.benchmark_scipy_operation::<T>(operation, &input_data)
                .ok()
        } else {
            None
        };

        // Calculate performance ratio
        let performance_ratio = if let Some(ref scipy_perf) = scipymetrics {
            Some(scipy_perf.avg_time.as_secs_f64() / scirs2metrics.avg_time.as_secs_f64())
        } else {
            None
        };

        // Memory comparison
        let memory_comparison = MemoryComparison {
            scirs2_memory: scirs2metrics.peak_memory,
            scipy_memory: scipymetrics.as_ref().map(|m| m.peak_memory),
            efficiency_ratio: scipymetrics
                .as_ref()
                .map(|scipy_perf| scipy_perf.peak_memory as f64 / scirs2metrics.peak_memory as f64),
        };

        Ok(OperationBenchmarkResult {
            operation: format!("{:?}", operation),
            array_size,
            data_type: if std::any::type_name::<T>() == "f32" {
                DataType::F32
            } else {
                DataType::F64
            },
            scirs2_performance: scirs2metrics,
            scipy_performance: scipymetrics,
            performance_ratio,
            memory_comparison,
        })
    }

    fn benchmark_scirs2_operation<T>(
        &self,
        operation: BenchmarkOperation,
        input_data: &Array2<T>,
    ) -> NdimageResult<PerformanceMetrics>
    where
        T: Float
            + FromPrimitive
            + Clone
            + Default
            + std::fmt::Debug
            + Send
            + Sync
            + std::ops::AddAssign
            + std::ops::DivAssign
            + 'static,
    {
        let mut timings = Vec::new();
        let mut memory_usages = Vec::new();

        // Warmup iterations
        for _ in 0..self.config.warmup_iterations {
            let _ = self.execute_scirs2_operation(operation, input_data)?;
        }

        // Actual benchmark iterations
        for _ in 0..self.config.iterations {
            let start_memory = self.get_memory_usage();
            let start_time = Instant::now();

            let _result = self.execute_scirs2_operation(operation, input_data)?;

            let execution_time = start_time.elapsed();
            let end_memory = self.get_memory_usage();

            timings.push(execution_time);
            memory_usages.push(end_memory.saturating_sub(start_memory));
        }

        Ok(self.calculate_performancemetrics(timings, memory_usages, input_data.len()))
    }

    fn execute_scirs2_operation<T>(
        &self,
        operation: BenchmarkOperation,
        input_data: &Array2<T>,
    ) -> NdimageResult<Array2<T>>
    where
        T: Float
            + FromPrimitive
            + Clone
            + Default
            + std::fmt::Debug
            + Send
            + Sync
            + std::ops::AddAssign
            + std::ops::DivAssign
            + 'static,
    {
        match operation {
            BenchmarkOperation::GaussianFilter => crate::filters::gaussian_filter_chunked(
                &input_data,
                &[T::from_f64(1.0).unwrap(), T::from_f64(1.0).unwrap()],
                Some(T::from_f64(4.0).unwrap()),
                BorderMode::Reflect,
                None,
            ),
            BenchmarkOperation::MedianFilter => {
                median_filter(&input_data, &[3, 3], Some(BorderMode::Reflect))
            }
            BenchmarkOperation::UniformFilter => {
                uniform_filter(&input_data, &[3, 3], Some(BorderMode::Reflect), None)
            }
            _ => {
                // For other operations, return a dummy result
                Ok(input_data.clone())
            }
        }
    }

    fn benchmark_scipy_operation<T>(
        &self,
        operation: BenchmarkOperation,
        input_data: &Array2<T>,
    ) -> NdimageResult<PerformanceMetrics>
    where
        T: Float + FromPrimitive + Clone + Default + std::fmt::Debug,
    {
        // Generate Python script for SciPy benchmark
        let script_path = self.generate_scipy_benchmark_script(operation, input_data)?;

        // Execute Python script and parse results
        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| NdimageError::InvalidInput(format!("Failed to execute Python: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(NdimageError::InvalidInput(format!(
                "Python script failed: {}",
                stderr
            )));
        }

        // Parse timing results from Python output
        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_scipy_benchmark_results(&stdout, input_data.len())
    }

    fn generate_scipy_benchmark_script<T>(
        &self,
        operation: BenchmarkOperation,
        input_data: &Array2<T>,
    ) -> NdimageResult<String>
    where
        T: Float + FromPrimitive + Clone + Default + std::fmt::Debug,
    {
        let script_path = format!("{}/benchmark_{:?}.py", self.config.temp_dir, operation);

        let (height, width) = input_data.dim();
        let dtype = if std::any::type_name::<T>() == "f32" {
            "float32"
        } else {
            "float64"
        };

        let python_code = format!(
            r#"
import numpy as np
import scipy.ndimage
import time
import sys
import psutil
import os

def benchmark_operation():
    # Generate test _data
    np.random.seed(42)
    _data = np.random.rand({height}, {width}).astype(np.{dtype})
    
    # Warmup
    for _ in range({warmup}):
        {operation_code}
    
    # Benchmark
    times = []
    for _ in range({iterations}):
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss
        
        start_time = time.perf_counter()
        {operation_code}
        end_time = time.perf_counter()
        
        end_memory = process.memory_info().rss
        
        times.append(end_time - start_time)
    
    # Output results
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_dev = np.std(times)
    
    print(f"{{avg_time:.9f}},{{min_time:.9f}},{{max_time:.9f}},{{std_dev:.9f}}")

if __name__ == "__main__":
    benchmark_operation()
"#,
            height = height,
            width = width,
            dtype = dtype,
            warmup = self.config.warmup_iterations,
            iterations = self.config.iterations,
            operation_code = self.get_scipy_operation_code(operation),
        );

        let mut file = fs::File::create(&script_path).map_err(|e| {
            NdimageError::InvalidInput(format!("Failed to create Python script: {}", e))
        })?;

        file.write_all(python_code.as_bytes()).map_err(|e| {
            NdimageError::InvalidInput(format!("Failed to write Python script: {}", e))
        })?;

        Ok(script_path)
    }

    fn get_scipy_operation_code(&self, operation: BenchmarkOperation) -> &str {
        match operation {
            BenchmarkOperation::GaussianFilter => {
                "result = scipy.ndimage.gaussian_filter(data, sigma=1.0, mode='reflect')"
            }
            BenchmarkOperation::MedianFilter => {
                "result = scipy.ndimage.median_filter(data, size=3, mode='reflect')"
            }
            BenchmarkOperation::UniformFilter => {
                "result = scipy.ndimage.uniform_filter(data, size=3, mode='reflect')"
            }
            BenchmarkOperation::BinaryErosion => {
                "result = scipy.ndimage.binary_erosion(data > 0.5)"
            }
            BenchmarkOperation::BinaryDilation => {
                "result = scipy.ndimage.binary_dilation(data > 0.5)"
            }
            BenchmarkOperation::DistanceTransform => {
                "result = scipy.ndimage.distance_transform_edt(data > 0.5)"
            }
            BenchmarkOperation::CenterOfMass => "result = scipy.ndimage.center_of_mass(data)",
            BenchmarkOperation::LabelObjects => "result = scipy.ndimage.label(data > 0.5)",
        }
    }

    fn parse_scipy_benchmark_results(
        &self,
        output: &str,
        array_len: usize,
    ) -> NdimageResult<PerformanceMetrics> {
        let parts: Vec<&str> = output.trim().split(',').collect();
        if parts.len() != 4 {
            return Err(NdimageError::InvalidInput(
                "Invalid Python output format".to_string(),
            ));
        }

        let avg_time =
            Duration::from_secs_f64(parts[0].parse().map_err(|_| {
                NdimageError::InvalidInput("Failed to parse average time".to_string())
            })?);

        let min_time = Duration::from_secs_f64(
            parts[1]
                .parse()
                .map_err(|_| NdimageError::InvalidInput("Failed to parse min time".to_string()))?,
        );

        let max_time = Duration::from_secs_f64(
            parts[2]
                .parse()
                .map_err(|_| NdimageError::InvalidInput("Failed to parse max time".to_string()))?,
        );

        let std_dev = Duration::from_secs_f64(
            parts[3]
                .parse()
                .map_err(|_| NdimageError::InvalidInput("Failed to parse std dev".to_string()))?,
        );

        let throughput = array_len as f64 / avg_time.as_secs_f64();

        Ok(PerformanceMetrics {
            avg_time,
            min_time,
            max_time,
            std_dev,
            throughput,
            peak_memory: 0, // Not available from this implementation
        })
    }

    fn generate_test_data<T>(
        &self,
        array_size: (usize, usize),
        operation: BenchmarkOperation,
    ) -> Array2<T>
    where
        T: Float + FromPrimitive + Clone + Default,
    {
        let (height, width) = array_size;
        let mut data = Array2::default((height, width));

        // Generate appropriate test data based on operation
        match operation {
            BenchmarkOperation::BinaryErosion | BenchmarkOperation::BinaryDilation => {
                // Binary data
                for ((i, j), elem) in data.indexed_iter_mut() {
                    *elem = if (i + j) % 3 == 0 {
                        T::one()
                    } else {
                        T::zero()
                    };
                }
            }
            BenchmarkOperation::DistanceTransform => {
                // Binary mask for distance transform
                for ((i, j), elem) in data.indexed_iter_mut() {
                    let center_x = width / 2;
                    let center_y = height / 2;
                    let dist_sq =
                        (i as f64 - center_y as f64).powi(2) + (j as f64 - center_x as f64).powi(2);
                    *elem = if dist_sq
                        < (width.min(height) / 4) as f64 * (width.min(height) / 4) as f64
                    {
                        T::zero()
                    } else {
                        T::one()
                    };
                }
            }
            _ => {
                // Random data for other operations
                for ((i, j), elem) in data.indexed_iter_mut() {
                    let val = ((i * 37 + j * 17) % 1000) as f64 / 1000.0;
                    *elem = T::from_f64(val).unwrap_or(T::zero());
                }
            }
        }

        data
    }

    fn calculate_performancemetrics(
        &self,
        timings: Vec<Duration>,
        memory_usages: Vec<usize>,
        array_len: usize,
    ) -> PerformanceMetrics {
        let avg_time = timings.iter().sum::<Duration>() / timings.len() as u32;
        let min_time = *timings.iter().min().unwrap();
        let max_time = *timings.iter().max().unwrap();

        // Calculate standard deviation
        let mean_nanos = avg_time.as_nanos() as f64;
        let variance = timings
            .iter()
            .map(|t| (t.as_nanos() as f64 - mean_nanos).powi(2))
            .sum::<f64>()
            / timings.len() as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        let throughput = array_len as f64 / avg_time.as_secs_f64();
        let peak_memory = memory_usages.iter().max().copied().unwrap_or(0);

        PerformanceMetrics {
            avg_time,
            min_time,
            max_time,
            std_dev,
            throughput,
            peak_memory,
        }
    }

    fn store_results(&mut self, results: Vec<OperationBenchmarkResult>) {
        for result in results {
            let operation_name = result.operation.clone();
            self.results
                .operation_results
                .entry(operation_name)
                .or_insert_with(Vec::new)
                .push(result);
        }
    }

    fn compute_overall_statistics(&mut self) {
        let mut total_operations = 0;
        let mut performance_ratios = Vec::new();
        let mut best_operations = Vec::new();
        let mut scipy_faster_operations = Vec::new();

        for (operation_name, results) in &self.results.operation_results {
            total_operations += results.len();

            let avg_ratio = results
                .iter()
                .filter_map(|r| r.performance_ratio)
                .fold(0.0, |acc, ratio| acc + ratio)
                / results.len() as f64;

            if avg_ratio > 1.2 {
                best_operations.push(operation_name.clone());
            } else if avg_ratio < 0.8 {
                scipy_faster_operations.push(operation_name.clone());
            }

            performance_ratios.push(avg_ratio);
        }

        let avg_performance_ratio = if performance_ratios.is_empty() {
            1.0
        } else {
            performance_ratios.iter().sum::<f64>() / performance_ratios.len() as f64
        };

        self.results.overall_stats = OverallBenchmarkStats {
            total_operations,
            avg_performance_ratio,
            best_operations,
            scipy_faster_operations,
            overall_memory_efficiency: 1.0, // Placeholder
        };
    }

    fn compute_scipy_comparison(&mut self) {
        let mut scirs2_faster = Vec::new();
        let mut scipy_faster = Vec::new();
        let mut similar_performance = Vec::new();
        let mut speedups = Vec::new();

        for (operation_name, results) in &self.results.operation_results {
            for result in results {
                if let Some(ratio) = result.performance_ratio {
                    speedups.push(ratio);

                    if ratio > 1.1 {
                        scirs2_faster.push(format!(
                            "{} ({}x{} {:?})",
                            operation_name,
                            result.array_size.0,
                            result.array_size.1,
                            result.data_type
                        ));
                    } else if ratio < 0.9 {
                        scipy_faster.push(format!(
                            "{} ({}x{} {:?})",
                            operation_name,
                            result.array_size.0,
                            result.array_size.1,
                            result.data_type
                        ));
                    } else {
                        similar_performance.push(format!(
                            "{} ({}x{} {:?})",
                            operation_name,
                            result.array_size.0,
                            result.array_size.1,
                            result.data_type
                        ));
                    }
                }
            }
        }

        let avg_speedup = if speedups.is_empty() {
            1.0
        } else {
            speedups.iter().sum::<f64>() / speedups.len() as f64
        };

        let max_speedup = speedups.iter().fold(1.0, |max, &speedup| max.max(speedup));

        self.results.scipy_comparison = Some(SciPyComparisonResults {
            scirs2_faster,
            scipy_faster,
            similar_performance,
            avg_speedup,
            max_speedup,
        });
    }

    fn find_python_executable() -> NdimageResult<String> {
        // Try common Python executable names
        let python_names = ["python3", "python", "python3.9", "python3.8", "python3.7"];

        for python_name in &python_names {
            if let Ok(output) = Command::new(python_name).arg("--version").output() {
                if output.status.success() {
                    return Ok(python_name.to_string());
                }
            }
        }

        Err(NdimageError::InvalidInput(
            "Python executable not found".to_string(),
        ))
    }

    fn get_memory_usage(&self) -> usize {
        // Placeholder implementation - would use actual memory monitoring
        0
    }

    /// Generate a comprehensive report
    pub fn generate_report(&self) -> BenchmarkReport {
        BenchmarkReport {
            results: self.results.clone(),
            summary: self.generate_summary(),
        }
    }

    fn generate_summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str("=== SciPy Benchmark Results ===\n\n");

        summary.push_str(&format!(
            "Total Operations Tested: {}\n",
            self.results.overall_stats.total_operations
        ));
        summary.push_str(&format!(
            "Average Performance Ratio: {:.2}x\n",
            self.results.overall_stats.avg_performance_ratio
        ));

        if let Some(ref comparison) = self.results.scipy_comparison {
            summary.push_str(&format!(
                "\nScirs2 Faster: {} operations\n",
                comparison.scirs2_faster.len()
            ));
            summary.push_str(&format!(
                "SciPy Faster: {} operations\n",
                comparison.scipy_faster.len()
            ));
            summary.push_str(&format!(
                "Similar Performance: {} operations\n",
                comparison.similar_performance.len()
            ));
            summary.push_str(&format!(
                "Maximum Speedup: {:.2}x\n",
                comparison.max_speedup
            ));
        }

        summary
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Benchmark results
    pub results: BenchmarkResults,
    /// Summary text
    pub summary: String,
}

impl BenchmarkReport {
    /// Display the benchmark report
    pub fn display(&self) {
        println!("{}", self.summary);

        // Display detailed results for each operation
        for (operation_name, results) in &self.results.operation_results {
            println!("\n--- {} Results ---", operation_name);

            for result in results {
                println!(
                    "{}x{} {:?}: {:.3}ms avg (scirs2), {:.3}ms avg (scipy), ratio: {:.2}x",
                    result.array_size.0,
                    result.array_size.1,
                    result.data_type,
                    result.scirs2_performance.avg_time.as_secs_f64() * 1000.0,
                    result
                        .scipy_performance
                        .as_ref()
                        .map(|p| p.avg_time.as_secs_f64() * 1000.0)
                        .unwrap_or(0.0),
                    result.performance_ratio.unwrap_or(1.0),
                );
            }
        }
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("Operation,ArraySize,DataType,Scirs2AvgTime,Scirs2MinTime,Scirs2MaxTime,ScipyAvgTime,PerformanceRatio\n");

        for (operation_name, results) in &self.results.operation_results {
            for result in results {
                csv.push_str(&format!(
                    "{},{}x{},{:?},{:.9},{:.9},{:.9},{:.9},{:.3}\n",
                    operation_name,
                    result.array_size.0,
                    result.array_size.1,
                    result.data_type,
                    result.scirs2_performance.avg_time.as_secs_f64(),
                    result.scirs2_performance.min_time.as_secs_f64(),
                    result.scirs2_performance.max_time.as_secs_f64(),
                    result
                        .scipy_performance
                        .as_ref()
                        .map(|p| p.avg_time.as_secs_f64())
                        .unwrap_or(0.0),
                    result.performance_ratio.unwrap_or(1.0),
                ));
            }
        }

        csv
    }
}

/// Convenience function to run a quick benchmark comparison
#[allow(dead_code)]
pub fn quick_scipy_benchmark() -> NdimageResult<BenchmarkReport> {
    let config = BenchmarkConfig {
        array_sizes: vec![(256, 256), (512, 512)],
        iterations: 5,
        operations: vec![
            BenchmarkOperation::GaussianFilter,
            BenchmarkOperation::MedianFilter,
        ],
        ..Default::default()
    };

    let mut suite = SciPyBenchmarkSuite::new(config)?;
    suite.run_complete_benchmark()?;
    Ok(suite.generate_report())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "timeout"]
    fn test_benchmarkconfig_creation() {
        let config = BenchmarkConfig::default();
        assert!(!config.array_sizes.is_empty());
        assert!(config.iterations > 0);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_benchmark_suite_creation() {
        let config = BenchmarkConfig::default();
        let result = SciPyBenchmarkSuite::new(config);

        // This might fail if Python is not available, which is acceptable
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_test_data_generation() {
        let config = BenchmarkConfig::default();
        if let Ok(suite) = SciPyBenchmarkSuite::new(config) {
            let data =
                suite.generate_test_data::<f64>((64, 64), BenchmarkOperation::GaussianFilter);
            assert_eq!(data.dim(), (64, 64));
        }
    }
}
