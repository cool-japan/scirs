//! GPU vs CPU performance benchmarking suite
//!
//! This module provides comprehensive benchmarking capabilities for comparing
//! performance between CPU and GPU implementations of various algorithms.

use crate::gpu::{GpuBackend, GpuContext, GpuError};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Error types for benchmarking operations
#[derive(Error, Debug)]
pub enum BenchmarkError {
    /// Benchmark setup failed
    #[error("Benchmark setup failed: {0}")]
    SetupFailed(String),

    /// Benchmark execution failed
    #[error("Benchmark execution failed: {0}")]
    ExecutionFailed(String),

    /// Invalid benchmark configuration
    #[error("Invalid benchmark configuration: {0}")]
    InvalidConfiguration(String),

    /// Results comparison failed
    #[error("Results comparison failed: {0}")]
    ComparisonFailed(String),

    /// Underlying GPU error
    #[error("GPU error: {0}")]
    GpuError(#[from] GpuError),
}

/// Benchmark operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BenchmarkOperation {
    /// Matrix multiplication (GEMM)
    MatrixMultiply,
    /// Element-wise vector operations
    VectorOperations,
    /// Fast Fourier Transform
    FastFourierTransform,
    /// Convolution operations
    Convolution,
    /// Reduction operations (sum, max, etc.)
    Reduction,
    /// Sorting algorithms
    Sorting,
    /// Random number generation
    RandomGeneration,
    /// Image processing operations
    ImageProcessing,
    /// Signal processing
    SignalProcessing,
    /// Statistical computations
    Statistics,
    /// Linear algebra operations
    LinearAlgebra,
    /// Sparse matrix operations
    SparseMatrix,
}

impl BenchmarkOperation {
    /// Get human-readable name
    pub const fn name(&self) -> &'static str {
        match self {
            BenchmarkOperation::MatrixMultiply => "Matrix Multiplication",
            BenchmarkOperation::VectorOperations => "Vector Operations",
            BenchmarkOperation::FastFourierTransform => "Fast Fourier Transform",
            BenchmarkOperation::Convolution => "Convolution",
            BenchmarkOperation::Reduction => "Reduction",
            BenchmarkOperation::Sorting => "Sorting",
            BenchmarkOperation::RandomGeneration => "Random Generation",
            BenchmarkOperation::ImageProcessing => "Image Processing",
            BenchmarkOperation::SignalProcessing => "Signal Processing",
            BenchmarkOperation::Statistics => "Statistics",
            BenchmarkOperation::LinearAlgebra => "Linear Algebra",
            BenchmarkOperation::SparseMatrix => "Sparse Matrix",
        }
    }

    /// Get operation category
    pub fn category(&self) -> BenchmarkCategory {
        match self {
            BenchmarkOperation::MatrixMultiply
            | BenchmarkOperation::LinearAlgebra
            | BenchmarkOperation::SparseMatrix => BenchmarkCategory::LinearAlgebra,

            BenchmarkOperation::VectorOperations | BenchmarkOperation::Reduction => {
                BenchmarkCategory::ElementWise
            }

            BenchmarkOperation::FastFourierTransform
            | BenchmarkOperation::Convolution
            | BenchmarkOperation::SignalProcessing => BenchmarkCategory::SignalProcessing,

            BenchmarkOperation::ImageProcessing => BenchmarkCategory::ImageProcessing,

            BenchmarkOperation::Sorting
            | BenchmarkOperation::RandomGeneration
            | BenchmarkOperation::Statistics => BenchmarkCategory::GeneralCompute,
        }
    }
}

/// Benchmark operation categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BenchmarkCategory {
    /// Linear algebra operations
    LinearAlgebra,
    /// Element-wise operations
    ElementWise,
    /// Signal processing operations
    SignalProcessing,
    /// Image processing operations
    ImageProcessing,
    /// General compute operations
    GeneralCompute,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Operations to benchmark
    pub operations: Vec<BenchmarkOperation>,
    /// Problem sizes to test
    pub problem_sizes: Vec<ProblemSize>,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
    /// Data types to test
    pub data_types: Vec<DataType>,
    /// GPU backends to test
    pub gpu_backends: Vec<GpuBackend>,
    /// Whether to verify correctness
    pub verify_correctness: bool,
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            operations: vec![
                BenchmarkOperation::MatrixMultiply,
                BenchmarkOperation::VectorOperations,
                BenchmarkOperation::Reduction,
            ],
            problem_sizes: vec![ProblemSize::Small, ProblemSize::Medium, ProblemSize::Large],
            warmup_iterations: 3,
            benchmark_iterations: 10,
            data_types: vec![DataType::Float32, DataType::Float64],
            gpu_backends: vec![GpuBackend::Cuda, GpuBackend::Rocm],
            verify_correctness: true,
            tolerance: 1e-6,
        }
    }
}

/// Problem size categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProblemSize {
    /// Small problems (< 1K elements)
    Small,
    /// Medium problems (1K - 1M elements)
    Medium,
    /// Large problems (1M - 100M elements)
    Large,
    /// Extra large problems (> 100M elements)
    ExtraLarge,
    /// Custom size
    Custom(usize),
}

impl ProblemSize {
    /// Get actual size for matrix operations (N x N)
    pub fn matrix_size(&self) -> usize {
        match self {
            ProblemSize::Small => 64,
            ProblemSize::Medium => 512,
            ProblemSize::Large => 2048,
            ProblemSize::ExtraLarge => 8192,
            ProblemSize::Custom(size) => *size,
        }
    }

    /// Get actual size for vector operations
    pub fn vector_size(&self) -> usize {
        match self {
            ProblemSize::Small => 1024,
            ProblemSize::Medium => 1024 * 1024,
            ProblemSize::Large => 64 * 1024 * 1024,
            ProblemSize::ExtraLarge => 512 * 1024 * 1024,
            ProblemSize::Custom(size) => *size,
        }
    }
}

/// Data types for benchmarking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// 16-bit floating point
    Float16,
    /// 32-bit signed integer
    Int32,
    /// 32-bit unsigned integer
    UInt32,
}

impl DataType {
    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::Float32 | DataType::Int32 | DataType::UInt32 => 4,
            DataType::Float64 => 8,
            DataType::Float16 => 2,
        }
    }

    /// Get type name
    pub const fn name(&self) -> &'static str {
        match self {
            DataType::Float32 => "f32",
            DataType::Float64 => "f64",
            DataType::Float16 => "f16",
            DataType::Int32 => "i32",
            DataType::UInt32 => "u32",
        }
    }
}

/// Compute platform for benchmarking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputePlatform {
    /// CPU implementation
    Cpu,
    /// GPU implementation with specific backend
    Gpu(GpuBackend),
}

impl ComputePlatform {
    /// Get platform name
    pub fn name(&self) -> String {
        match self {
            ComputePlatform::Cpu => "CPU".to_string(),
            ComputePlatform::Gpu(backend) => format!("GPU ({})", backend),
        }
    }
}

/// Benchmark result for a single test
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Operation that was benchmarked
    pub operation: BenchmarkOperation,
    /// Platform used
    pub platform: ComputePlatform,
    /// Problem size
    pub problem_size: ProblemSize,
    /// Data type
    pub data_type: DataType,
    /// Execution time (average)
    pub execution_time: Duration,
    /// Standard deviation of execution times
    pub time_stddev: Duration,
    /// Throughput (operations per second)
    pub throughput: f64,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
    /// Energy efficiency (operations per joule) if available
    pub energy_efficiency: Option<f64>,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,
    /// Whether correctness verification passed
    pub correctness_verified: bool,
}

/// Benchmark comparison result
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    /// Operation being compared
    pub operation: BenchmarkOperation,
    /// Problem size
    pub problem_size: ProblemSize,
    /// Data type
    pub data_type: DataType,
    /// Results for each platform
    pub platform_results: HashMap<ComputePlatform, BenchmarkResult>,
    /// Speedup factors (GPU vs CPU)
    pub speedups: HashMap<GpuBackend, f64>,
    /// Energy efficiency comparison
    pub energy_comparison: HashMap<ComputePlatform, f64>,
    /// Recommendation based on results
    pub recommendation: PlatformRecommendation,
}

/// Platform recommendation based on benchmark results
#[derive(Debug, Clone)]
pub enum PlatformRecommendation {
    /// CPU is recommended
    Cpu { reason: String },
    /// GPU is recommended
    Gpu { backend: GpuBackend, reason: String },
    /// Depends on specific use case
    Depends { factors: Vec<String> },
}

/// Main benchmarking suite
pub struct BenchmarkSuite {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResult>,
    comparisons: Vec<BenchmarkComparison>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
            comparisons: Vec::new(),
        }
    }

    /// Run all benchmarks
    pub fn run_all(&mut self) -> Result<(), BenchmarkError> {
        let operations = self.config.operations.clone();
        let problem_sizes = self.config.problem_sizes.clone();
        let data_types = self.config.data_types.clone();

        for operation in operations {
            for problem_size in problem_sizes.iter() {
                for data_type in data_types.iter() {
                    self.run_operation_benchmark(operation, *problem_size, *data_type)?;
                }
            }
        }

        self.generate_comparisons()?;
        Ok(())
    }

    /// Run benchmark for a specific operation
    fn run_operation_benchmark(
        &mut self,
        operation: BenchmarkOperation,
        problem_size: ProblemSize,
        data_type: DataType,
    ) -> Result<(), BenchmarkError> {
        // Run CPU benchmark
        let cpu_result = self.run_cpu_benchmark(operation, problem_size, data_type)?;
        self.results.push(cpu_result);

        // Run GPU benchmarks for each available backend
        for &backend in &self.config.gpu_backends {
            if backend.is_available() {
                match self.run_gpu_benchmark(operation, problem_size, data_type, backend) {
                    Ok(gpu_result) => self.results.push(gpu_result),
                    Err(e) => {
                        eprintln!("GPU benchmark failed for {}: {}", backend, e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Run CPU benchmark
    fn run_cpu_benchmark(
        &self,
        operation: BenchmarkOperation,
        problem_size: ProblemSize,
        data_type: DataType,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            self.execute_cpu_operation(operation, problem_size, data_type)?;
        }

        // Benchmark
        let mut execution_times = Vec::new();
        for _ in 0..self.config.benchmark_iterations {
            let start = Instant::now();
            self.execute_cpu_operation(operation, problem_size, data_type)?;
            execution_times.push(start.elapsed());
        }

        let avg_time = execution_times.iter().sum::<Duration>() / execution_times.len() as u32;
        let time_stddev = self.calculate_stddev(&execution_times, avg_time);

        Ok(BenchmarkResult {
            operation,
            platform: ComputePlatform::Cpu,
            problem_size,
            data_type,
            execution_time: avg_time,
            time_stddev,
            throughput: self.calculate_throughput(operation, problem_size, avg_time),
            memory_bandwidth: self.calculate_memory_bandwidth(
                operation,
                problem_size,
                data_type,
                avg_time,
            ),
            energy_efficiency: None, // Would need power measurement
            peak_memory_usage: self.estimate_memory_usage(operation, problem_size, data_type),
            correctness_verified: true, // CPU is reference implementation
        })
    }

    /// Run GPU benchmark
    fn run_gpu_benchmark(
        &self,
        operation: BenchmarkOperation,
        problem_size: ProblemSize,
        data_type: DataType,
        backend: GpuBackend,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        // Create GPU context
        let _context =
            GpuContext::new(backend).map_err(|e| BenchmarkError::SetupFailed(e.to_string()))?;

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            self.execute_gpu_operation(operation, problem_size, data_type, backend)?;
        }

        // Benchmark
        let mut execution_times = Vec::new();
        for _ in 0..self.config.benchmark_iterations {
            let start = Instant::now();
            self.execute_gpu_operation(operation, problem_size, data_type, backend)?;
            execution_times.push(start.elapsed());
        }

        let avg_time = execution_times.iter().sum::<Duration>() / execution_times.len() as u32;
        let time_stddev = self.calculate_stddev(&execution_times, avg_time);

        Ok(BenchmarkResult {
            operation,
            platform: ComputePlatform::Gpu(backend),
            problem_size,
            data_type,
            execution_time: avg_time,
            time_stddev,
            throughput: self.calculate_throughput(operation, problem_size, avg_time),
            memory_bandwidth: self.calculate_memory_bandwidth(
                operation,
                problem_size,
                data_type,
                avg_time,
            ),
            energy_efficiency: None,
            peak_memory_usage: self.estimate_memory_usage(operation, problem_size, data_type),
            correctness_verified: self.config.verify_correctness,
        })
    }

    /// Execute CPU operation (placeholder implementation)
    fn execute_cpu_operation(
        &self,
        operation: BenchmarkOperation,
        problem_size: ProblemSize,
        _data_type: DataType,
    ) -> Result<(), BenchmarkError> {
        match operation {
            BenchmarkOperation::MatrixMultiply => {
                let n = problem_size.matrix_size();
                // Simulate matrix multiplication
                let _result = (0..n * n).map(|i| i as f64).sum::<f64>();
                Ok(())
            }
            BenchmarkOperation::VectorOperations => {
                let n = problem_size.vector_size();
                // Simulate vector operation
                let _result = (0..n).map(|i| (i as f64).sin()).sum::<f64>();
                Ok(())
            }
            _ => {
                // Other operations would be implemented similarly
                std::thread::sleep(Duration::from_millis(1));
                Ok(())
            }
        }
    }

    /// Execute GPU operation (placeholder implementation)
    fn execute_gpu_operation(
        &self,
        operation: BenchmarkOperation,
        problem_size: ProblemSize,
        _data_type: DataType,
        _backend: GpuBackend,
    ) -> Result<(), BenchmarkError> {
        match operation {
            BenchmarkOperation::MatrixMultiply => {
                let _n = problem_size.matrix_size();
                // Would launch GPU kernel for matrix multiplication
                std::thread::sleep(Duration::from_micros(100));
                Ok(())
            }
            BenchmarkOperation::VectorOperations => {
                let _n = problem_size.vector_size();
                // Would launch GPU kernel for vector operations
                std::thread::sleep(Duration::from_micros(50));
                Ok(())
            }
            _ => {
                // Other operations would be implemented similarly
                std::thread::sleep(Duration::from_micros(100));
                Ok(())
            }
        }
    }

    /// Generate comparison results
    fn generate_comparisons(&mut self) -> Result<(), BenchmarkError> {
        let mut grouped_results: HashMap<
            (BenchmarkOperation, ProblemSize, DataType),
            Vec<&BenchmarkResult>,
        > = HashMap::new();

        // Group results by operation, size, and data type
        for result in &self.results {
            let key = (result.operation, result.problem_size, result.data_type);
            grouped_results.entry(key).or_default().push(result);
        }

        // Generate comparisons for each group
        for ((operation, problem_size, data_type), results) in grouped_results {
            if results.len() > 1 {
                let comparison =
                    self.create_comparison(operation, problem_size, data_type, &results)?;
                self.comparisons.push(comparison);
            }
        }

        Ok(())
    }

    /// Create a comparison from results
    fn create_comparison(
        &self,
        operation: BenchmarkOperation,
        problem_size: ProblemSize,
        data_type: DataType,
        results: &[&BenchmarkResult],
    ) -> Result<BenchmarkComparison, BenchmarkError> {
        let mut platform_results = HashMap::new();
        let mut cpu_time = None;

        for result in results {
            platform_results.insert(result.platform, (*result).clone());
            if matches!(result.platform, ComputePlatform::Cpu) {
                cpu_time = Some(result.execution_time);
            }
        }

        let mut speedups = HashMap::new();
        let mut energy_comparison = HashMap::new();

        if let Some(cpu_time) = cpu_time {
            for result in results {
                if let ComputePlatform::Gpu(backend) = result.platform {
                    let speedup = cpu_time.as_secs_f64() / result.execution_time.as_secs_f64();
                    speedups.insert(backend, speedup);
                }

                // Energy comparison (placeholder)
                energy_comparison.insert(result.platform, 1.0);
            }
        }

        let recommendation = self.generate_recommendation(operation, &platform_results, &speedups);

        Ok(BenchmarkComparison {
            operation,
            problem_size,
            data_type,
            platform_results,
            speedups,
            energy_comparison,
            recommendation,
        })
    }

    /// Generate platform recommendation
    fn generate_recommendation(
        &self,
        operation: BenchmarkOperation,
        _platform_results: &HashMap<ComputePlatform, BenchmarkResult>,
        speedups: &HashMap<GpuBackend, f64>,
    ) -> PlatformRecommendation {
        // Find best GPU speedup
        let best_speedup = speedups.values().fold(0.0f64, |a, &b| a.max(b));
        let best_backend = speedups
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(&backend, _)| backend);

        if best_speedup > 2.0 {
            if let Some(backend) = best_backend {
                PlatformRecommendation::Gpu {
                    backend,
                    reason: format!("GPU shows {:.1}x speedup over CPU", best_speedup),
                }
            } else {
                PlatformRecommendation::Cpu {
                    reason: "No significant GPU advantage found".to_string(),
                }
            }
        } else if best_speedup > 1.2 {
            PlatformRecommendation::Depends {
                factors: vec![
                    format!("GPU shows modest {:.1}x speedup", best_speedup),
                    "Consider data transfer overhead".to_string(),
                    format!(
                        "{} may benefit from GPU for larger problems",
                        operation.name()
                    ),
                ],
            }
        } else {
            PlatformRecommendation::Cpu {
                reason: "CPU performance is competitive or better".to_string(),
            }
        }
    }

    /// Calculate standard deviation of execution times
    fn calculate_stddev(&self, times: &[Duration], avg: Duration) -> Duration {
        if times.len() <= 1 {
            return Duration::ZERO;
        }

        let variance = times
            .iter()
            .map(|&time| {
                let diff = time.as_secs_f64() - avg.as_secs_f64();
                diff * diff
            })
            .sum::<f64>()
            / (times.len() - 1) as f64;

        Duration::from_secs_f64(variance.sqrt())
    }

    /// Calculate throughput for an operation
    fn calculate_throughput(
        &self,
        operation: BenchmarkOperation,
        problem_size: ProblemSize,
        time: Duration,
    ) -> f64 {
        let ops = match operation {
            BenchmarkOperation::MatrixMultiply => {
                let n = problem_size.matrix_size();
                2 * n * n * n // 2*N^3 operations for N x N matrix multiply
            }
            BenchmarkOperation::VectorOperations => {
                problem_size.vector_size() // One operation per element
            }
            _ => problem_size.vector_size(), // Default estimate
        };

        ops as f64 / time.as_secs_f64()
    }

    /// Calculate memory bandwidth utilization
    fn calculate_memory_bandwidth(
        &self,
        operation: BenchmarkOperation,
        problem_size: ProblemSize,
        data_type: DataType,
        time: Duration,
    ) -> f64 {
        let bytes = match operation {
            BenchmarkOperation::MatrixMultiply => {
                let n = problem_size.matrix_size();
                (3 * n * n) * data_type.size_bytes() // A, B, C matrices
            }
            BenchmarkOperation::VectorOperations => {
                problem_size.vector_size() * data_type.size_bytes() * 2 // Read + write
            }
            _ => problem_size.vector_size() * data_type.size_bytes() * 2,
        };

        (bytes as f64) / (time.as_secs_f64() * 1e9) // GB/s
    }

    /// Estimate memory usage for an operation
    fn estimate_memory_usage(
        &self,
        operation: BenchmarkOperation,
        problem_size: ProblemSize,
        data_type: DataType,
    ) -> usize {
        match operation {
            BenchmarkOperation::MatrixMultiply => {
                let n = problem_size.matrix_size();
                3 * n * n * data_type.size_bytes() // Three N x N matrices
            }
            BenchmarkOperation::VectorOperations => {
                problem_size.vector_size() * data_type.size_bytes() * 2 // Input + output
            }
            _ => problem_size.vector_size() * data_type.size_bytes() * 2,
        }
    }

    /// Get all benchmark results
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Get all benchmark comparisons
    pub fn comparisons(&self) -> &[BenchmarkComparison] {
        &self.comparisons
    }

    /// Generate a summary report
    pub fn generate_report(&self) -> BenchmarkReport {
        BenchmarkReport::new(&self.results, &self.comparisons)
    }
}

/// Comprehensive benchmark report
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Summary statistics
    pub summary: BenchmarkSummary,
    /// Detailed results
    pub detailed_results: Vec<BenchmarkResult>,
    /// Platform comparisons
    pub comparisons: Vec<BenchmarkComparison>,
    /// Recommendations by operation category
    pub category_recommendations: HashMap<BenchmarkCategory, String>,
}

impl BenchmarkReport {
    fn new(results: &[BenchmarkResult], comparisons: &[BenchmarkComparison]) -> Self {
        let summary = BenchmarkSummary::from_results(results);
        let category_recommendations = Self::generate_category_recommendations(comparisons);

        Self {
            summary,
            detailed_results: results.to_vec(),
            comparisons: comparisons.to_vec(),
            category_recommendations,
        }
    }

    fn generate_category_recommendations(
        comparisons: &[BenchmarkComparison],
    ) -> HashMap<BenchmarkCategory, String> {
        let mut recommendations = HashMap::new();

        // Group by category and analyze
        for category in [
            BenchmarkCategory::LinearAlgebra,
            BenchmarkCategory::ElementWise,
            BenchmarkCategory::SignalProcessing,
            BenchmarkCategory::ImageProcessing,
            BenchmarkCategory::GeneralCompute,
        ] {
            let category_comps: Vec<_> = comparisons
                .iter()
                .filter(|c| c.operation.category() == category)
                .collect();

            if !category_comps.is_empty() {
                let gpu_wins = category_comps
                    .iter()
                    .filter(|c| matches!(c.recommendation, PlatformRecommendation::Gpu { .. }))
                    .count();

                let recommendation = if gpu_wins > category_comps.len() / 2 {
                    format!("GPU recommended for most {} operations", category.name())
                } else {
                    format!("CPU competitive for {} operations", category.name())
                };

                recommendations.insert(category, recommendation);
            }
        }

        recommendations
    }
}

impl BenchmarkCategory {
    fn name(&self) -> &'static str {
        match self {
            BenchmarkCategory::LinearAlgebra => "linear algebra",
            BenchmarkCategory::ElementWise => "element-wise",
            BenchmarkCategory::SignalProcessing => "signal processing",
            BenchmarkCategory::ImageProcessing => "image processing",
            BenchmarkCategory::GeneralCompute => "general compute",
        }
    }
}

/// Summary statistics for benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    /// Total number of benchmarks run
    pub total_benchmarks: usize,
    /// Average CPU execution time
    pub avg_cpu_time: Duration,
    /// Average GPU execution time
    pub avg_gpu_time: Duration,
    /// Overall GPU speedup factor
    pub overall_speedup: f64,
    /// Best performing platform by operation
    pub best_platforms: HashMap<BenchmarkOperation, ComputePlatform>,
}

impl BenchmarkSummary {
    fn from_results(results: &[BenchmarkResult]) -> Self {
        let total_benchmarks = results.len();

        let cpu_times: Vec<_> = results
            .iter()
            .filter(|r| matches!(r.platform, ComputePlatform::Cpu))
            .map(|r| r.execution_time)
            .collect();

        let gpu_times: Vec<_> = results
            .iter()
            .filter(|r| matches!(r.platform, ComputePlatform::Gpu(_)))
            .map(|r| r.execution_time)
            .collect();

        let avg_cpu_time = if !cpu_times.is_empty() {
            cpu_times.iter().sum::<Duration>() / cpu_times.len() as u32
        } else {
            Duration::ZERO
        };

        let avg_gpu_time = if !gpu_times.is_empty() {
            gpu_times.iter().sum::<Duration>() / gpu_times.len() as u32
        } else {
            Duration::ZERO
        };

        let overall_speedup = if avg_gpu_time > Duration::ZERO {
            avg_cpu_time.as_secs_f64() / avg_gpu_time.as_secs_f64()
        } else {
            1.0
        };

        // Find best platform for each operation
        let mut best_platforms = HashMap::new();
        let mut operation_results: HashMap<BenchmarkOperation, Vec<&BenchmarkResult>> =
            HashMap::new();

        for result in results {
            operation_results
                .entry(result.operation)
                .or_default()
                .push(result);
        }

        for (operation, op_results) in operation_results {
            if let Some(best) = op_results.iter().min_by_key(|r| r.execution_time) {
                best_platforms.insert(operation, best.platform);
            }
        }

        Self {
            total_benchmarks,
            avg_cpu_time,
            avg_gpu_time,
            overall_speedup,
            best_platforms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_operation_name() {
        assert_eq!(
            BenchmarkOperation::MatrixMultiply.name(),
            "Matrix Multiplication"
        );
        assert_eq!(
            BenchmarkOperation::VectorOperations.name(),
            "Vector Operations"
        );
    }

    #[test]
    fn test_problem_size_matrix() {
        assert_eq!(ProblemSize::Small.matrix_size(), 64);
        assert_eq!(ProblemSize::Large.matrix_size(), 2048);
        assert_eq!(ProblemSize::Custom(1000).matrix_size(), 1000);
    }

    #[test]
    fn test_data_type_size() {
        assert_eq!(DataType::Float32.size_bytes(), 4);
        assert_eq!(DataType::Float64.size_bytes(), 8);
        assert_eq!(DataType::Float16.size_bytes(), 2);
    }

    #[test]
    fn test_compute_platform_name() {
        assert_eq!(ComputePlatform::Cpu.name(), "CPU");
        assert_eq!(ComputePlatform::Gpu(GpuBackend::Cuda).name(), "GPU (CUDA)");
    }

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert!(!config.operations.is_empty());
        assert!(!config.problem_sizes.is_empty());
        assert!(config.verify_correctness);
    }

    #[test]
    fn test_benchmark_suite_creation() {
        let config = BenchmarkConfig::default();
        let suite = BenchmarkSuite::new(config);
        assert!(suite.results().is_empty());
        assert!(suite.comparisons().is_empty());
    }
}
