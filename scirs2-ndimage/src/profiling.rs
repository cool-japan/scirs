//! Performance profiling and optimization tools
//!
//! This module provides comprehensive tools for profiling and optimizing ndimage operations,
//! including timing measurements, memory usage tracking, performance analysis, backend
//! comparison, and automatic optimization recommendations.

use scirs2_core::numeric::Float;
use std::cmp;
use std::collections::HashMap;
use std::fmt::{self, Debug, Display};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::backend::Backend;
use crate::error::NdimageResult;

// Global profiler instance
lazy_static::lazy_static! {
    static ref PROFILER: Arc<Mutex<Profiler>> = Arc::new(Mutex::new(Profiler::new()));
}

/// Performance metrics for a single operation
#[derive(Debug, Clone)]
pub struct OperationMetrics {
    pub name: String,
    pub duration: Duration,
    pub memory_allocated: usize,
    pub memory_deallocated: usize,
    pub arrayshape: Vec<usize>,
    pub backend: Backend,
    pub thread_count: usize,
    pub timestamp: Instant,
}

impl Display for OperationMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.3}ms, shape={:?}, backend={:?}, threads={}",
            self.name,
            self.duration.as_secs_f64() * 1000.0,
            self.arrayshape,
            self.backend,
            self.thread_count
        )
    }
}

/// Profiler for tracking performance metrics
#[derive(Debug)]
pub struct Profiler {
    metrics: Vec<OperationMetrics>,
    enabled: bool,
    memory_tracking: bool,
    current_memory: usize,
    peak_memory: usize,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            enabled: false,
            memory_tracking: false,
            current_memory: 0,
            peak_memory: 0,
        }
    }

    /// Enable profiling
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable profiling
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable memory tracking
    pub fn enable_memory_tracking(&mut self) {
        self.memory_tracking = true;
    }

    /// Record a metric
    pub fn record(&mut self, metric: OperationMetrics) {
        if self.enabled {
            self.metrics.push(metric);
        }
    }

    /// Clear all metrics
    pub fn clear(&mut self) {
        self.metrics.clear();
        self.current_memory = 0;
        self.peak_memory = 0;
    }

    /// Get all metrics
    pub fn metrics(&self) -> &[OperationMetrics] {
        &self.metrics
    }

    /// Generate a performance report
    pub fn report(&self) -> PerformanceReport {
        PerformanceReport::frommetrics(&self.metrics)
    }

    /// Track memory allocation
    pub fn track_allocation(&mut self, bytes: usize) {
        if self.memory_tracking {
            self.current_memory += bytes;
            self.peak_memory = self.peak_memory.max(self.current_memory);
        }
    }

    /// Track memory deallocation
    pub fn track_deallocation(&mut self, bytes: usize) {
        if self.memory_tracking {
            self.current_memory = self.current_memory.saturating_sub(bytes);
        }
    }
}

/// Performance report with analysis
#[derive(Debug)]
pub struct PerformanceReport {
    pub total_time: Duration,
    pub operation_breakdown: HashMap<String, OperationSummary>,
    pub backend_usage: HashMap<String, usize>,
    pub memory_stats: MemoryStats,
    pub recommendations: Vec<String>,
}

/// Summary statistics for an operation type
#[derive(Debug)]
pub struct OperationSummary {
    pub count: usize,
    pub total_time: Duration,
    pub mean_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub std_dev: f64,
}

/// Memory usage statistics
#[derive(Debug)]
pub struct MemoryStats {
    pub peak_usage: usize,
    pub total_allocated: usize,
    pub total_deallocated: usize,
}

impl PerformanceReport {
    fn frommetrics(metrics: &[OperationMetrics]) -> Self {
        let total_time = metrics.iter().map(|m| m.duration).sum();

        // Group metrics by operation name
        let mut op_groups: HashMap<String, Vec<&OperationMetrics>> = HashMap::new();
        let mut backend_usage: HashMap<String, usize> = HashMap::new();

        for metric in metrics {
            op_groups
                .entry(metric.name.clone())
                .or_default()
                .push(metric);

            *backend_usage
                .entry(format!("{:?}", metric.backend))
                .or_default() += 1;
        }

        // Compute operation summaries
        let operation_breakdown: HashMap<String, OperationSummary> = op_groups
            .into_iter()
            .map(|(name, group)| {
                let count = group.len();
                let total: Duration = group.iter().map(|m| m.duration).sum();
                let mean = total / count as u32;

                let times: Vec<f64> = group.iter().map(|m| m.duration.as_secs_f64()).collect();

                let min = times
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(&0.0);
                let max = times
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(&0.0);

                let mean_f64 = times.iter().sum::<f64>() / count as f64;
                let variance =
                    times.iter().map(|t| (t - mean_f64).powi(2)).sum::<f64>() / count as f64;
                let std_dev = variance.sqrt();

                (
                    name,
                    OperationSummary {
                        count,
                        total_time: total,
                        mean_time: mean,
                        min_time: Duration::from_secs_f64(*min),
                        max_time: Duration::from_secs_f64(*max),
                        std_dev,
                    },
                )
            })
            .collect();

        // Compute memory statistics
        let total_allocated: usize = metrics.iter().map(|m| m.memory_allocated).sum();
        let total_deallocated: usize = metrics.iter().map(|m| m.memory_deallocated).sum();
        let peak_usage = metrics
            .iter()
            .scan(0isize, |acc, m| {
                *acc += m.memory_allocated as isize - m.memory_deallocated as isize;
                Some(*acc as usize)
            })
            .max()
            .unwrap_or(0);

        let memory_stats = MemoryStats {
            peak_usage,
            total_allocated,
            total_deallocated,
        };

        // Generate recommendations
        let recommendations =
            generate_recommendations(&operation_breakdown, &backend_usage, metrics);

        Self {
            total_time,
            operation_breakdown,
            backend_usage,
            memory_stats,
            recommendations,
        }
    }

    /// Display the report in a human-readable format
    pub fn display(&self) {
        println!("\n=== Performance Report ===\n");

        println!(
            "Total execution time: {:.3}ms",
            self.total_time.as_secs_f64() * 1000.0
        );
        println!();

        println!("Operation Breakdown:");
        let mut ops: Vec<_> = self.operation_breakdown.iter().collect();
        ops.sort_by_key(|(_, summary)| std::cmp::Reverse(summary.total_time));

        for (name, summary) in ops {
            println!("  {}: {} calls", name, summary.count);
            println!(
                "    Total: {:.3}ms ({:.1}%)",
                summary.total_time.as_secs_f64() * 1000.0,
                (summary.total_time.as_secs_f64() / self.total_time.as_secs_f64()) * 100.0
            );
            println!(
                "    Mean: {:.3}ms, Min: {:.3}ms, Max: {:.3}ms, StdDev: {:.3}ms",
                summary.mean_time.as_secs_f64() * 1000.0,
                summary.min_time.as_secs_f64() * 1000.0,
                summary.max_time.as_secs_f64() * 1000.0,
                summary.std_dev * 1000.0
            );
        }
        println!();

        println!("Backend Usage:");
        for (backend, count) in &self.backend_usage {
            println!("  {}: {} operations", backend, count);
        }
        println!();

        println!("Memory Statistics:");
        println!(
            "  Peak usage: {} MB",
            self.memory_stats.peak_usage / (1024 * 1024)
        );
        println!(
            "  Total allocated: {} MB",
            self.memory_stats.total_allocated / (1024 * 1024)
        );
        println!(
            "  Total deallocated: {} MB",
            self.memory_stats.total_deallocated / (1024 * 1024)
        );
        println!();

        if !self.recommendations.is_empty() {
            println!("Recommendations:");
            for rec in &self.recommendations {
                println!("  • {}", rec);
            }
        }
    }
}

/// Generate performance recommendations
#[allow(dead_code)]
fn generate_recommendations(
    operation_breakdown: &HashMap<String, OperationSummary>,
    backend_usage: &HashMap<String, usize>,
    metrics: &[OperationMetrics],
) -> Vec<String> {
    let mut recommendations = Vec::new();

    // Check for operations that could benefit from GPU acceleration
    let cpu_only = backend_usage.get("Cpu").copied().unwrap_or(0);
    let total_ops = backend_usage.values().sum::<usize>();

    if cpu_only == total_ops && total_ops > 10 {
        // Check if there are large arrays that could benefit from GPU
        let large_arrays = metrics
            .iter()
            .filter(|m| m.arrayshape.iter().product::<usize>() > 1_000_000)
            .count();

        if large_arrays > 0 {
            recommendations.push(format!(
                "Consider enabling GPU acceleration - {} operations processed large arrays (>1M elements)",
                large_arrays
            ));
        }
    }

    // Check for operations with high variance in execution time
    for (name, summary) in operation_breakdown {
        let cv = summary.std_dev / summary.mean_time.as_secs_f64(); // Coefficient of variation
        if cv > 0.5 && summary.count > 5 {
            recommendations.push(format!(
                "High variance in '{}' execution times (CV={:.2}) - consider investigating data-dependent performance",
                name, cv
            ));
        }
    }

    // Check for potential memory issues
    let total_time_ms = metrics.iter().map(|m| m.duration.as_millis()).sum::<u128>();
    let ops_per_ms = total_ops as f64 / total_time_ms as f64;

    if ops_per_ms < 0.1 {
        recommendations.push(
            "Low throughput detected - consider batch processing or parallelization".to_string(),
        );
    }

    recommendations
}

/// Profiling scope guard
pub struct ProfilingScope {
    name: String,
    start: Instant,
    shape: Vec<usize>,
    backend: Backend,
    initial_memory: usize,
}

impl ProfilingScope {
    pub fn new(name: impl Into<String>, shape: &[usize], backend: Backend) -> Self {
        let profiler = PROFILER
            .lock()
            .expect("PROFILER mutex should not be poisoned");
        let initial_memory = profiler.current_memory;
        drop(profiler);

        Self {
            name: name.into(),
            start: Instant::now(),
            shape: shape.to_vec(),
            backend,
            initial_memory,
        }
    }
}

impl Drop for ProfilingScope {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        let thread_count = scirs2_core::parallel_ops::get_num_threads();

        let mut profiler = PROFILER
            .lock()
            .expect("PROFILER mutex should not be poisoned");
        let memory_allocated = profiler.current_memory.saturating_sub(self.initial_memory);

        let metric = OperationMetrics {
            name: self.name.clone(),
            duration,
            memory_allocated,
            memory_deallocated: 0,
            arrayshape: self.shape.clone(),
            backend: self.backend,
            thread_count,
            timestamp: self.start,
        };

        profiler.record(metric);
    }
}

/// Profile an operation
#[macro_export]
macro_rules! profile_op {
    ($name:expr, $shape:expr, $backend:expr, $body:expr) => {{
        let _scope = $crate::profiling::ProfilingScope::new($name, $shape, $backend);
        $body
    }};
}

/// Enable global profiling
#[allow(dead_code)]
pub fn enable_profiling() {
    PROFILER
        .lock()
        .expect("PROFILER mutex should not be poisoned")
        .enable();
}

/// Disable global profiling
#[allow(dead_code)]
pub fn disable_profiling() {
    PROFILER
        .lock()
        .expect("PROFILER mutex should not be poisoned")
        .disable();
}

/// Enable memory tracking
#[allow(dead_code)]
pub fn enable_memory_tracking() {
    PROFILER
        .lock()
        .expect("PROFILER mutex should not be poisoned")
        .enable_memory_tracking();
}

/// Clear all profiling data
#[allow(dead_code)]
pub fn clear_profiling_data() {
    PROFILER
        .lock()
        .expect("PROFILER mutex should not be poisoned")
        .clear();
}

/// Get performance report
#[allow(dead_code)]
pub fn get_performance_report() -> PerformanceReport {
    PROFILER
        .lock()
        .expect("PROFILER mutex should not be poisoned")
        .report()
}

/// Display performance report
#[allow(dead_code)]
pub fn display_performance_report() {
    let report = get_performance_report();
    report.display();
}

/// Benchmark utility for comparing implementations
pub struct Benchmark<T> {
    name: String,
    iterations: usize,
    warmup_iterations: usize,
    results: Vec<BenchmarkResult<T>>,
}

#[derive(Debug)]
pub struct BenchmarkResult<T> {
    pub variant: String,
    pub times: Vec<Duration>,
    pub result: T,
}

impl<T> Benchmark<T> {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            iterations: 100,
            warmup_iterations: 10,
            results: Vec::new(),
        }
    }

    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn warmup_iterations(mut self, warmup: usize) -> Self {
        self.warmup_iterations = warmup;
        self
    }

    pub fn run<F>(&mut self, variant: impl Into<String>, mut f: F) -> NdimageResult<()>
    where
        F: FnMut() -> NdimageResult<T>,
    {
        let variant = variant.into();

        // Warmup
        for _ in 0..self.warmup_iterations {
            f()?;
        }

        // Actual benchmark
        let mut times = Vec::with_capacity(self.iterations);
        let mut result = None;

        for _ in 0..self.iterations {
            let start = Instant::now();
            result = Some(f()?);
            times.push(start.elapsed());
        }

        self.results.push(BenchmarkResult {
            variant,
            times,
            result: result.expect("Benchmark result should be available after iterations"),
        });

        Ok(())
    }

    pub fn compare(&self) -> BenchmarkComparison {
        BenchmarkComparison::from_results(&self.name, &self.results)
    }
}

/// Comparison of benchmark results
#[derive(Debug)]
pub struct BenchmarkComparison {
    pub name: String,
    pub variants: Vec<VariantStats>,
    pub fastest: String,
    pub baseline: String,
}

#[derive(Debug)]
pub struct VariantStats {
    pub name: String,
    pub mean: Duration,
    pub median: Duration,
    pub std_dev: Duration,
    pub min: Duration,
    pub max: Duration,
    pub speedup: f64,
}

impl BenchmarkComparison {
    fn from_results<T>(name: &str, results: &[BenchmarkResult<T>]) -> Self {
        let mut variants = Vec::new();

        for result in results {
            let mut times = result.times.clone();
            times.sort();

            let mean = times.iter().sum::<Duration>() / times.len() as u32;
            let median = times[times.len() / 2];
            let min = times[0];
            let max = times[times.len() - 1];

            let mean_nanos = mean.as_nanos() as f64;
            let variance = times
                .iter()
                .map(|t| {
                    let diff = t.as_nanos() as f64 - mean_nanos;
                    diff * diff
                })
                .sum::<f64>()
                / times.len() as f64;
            let std_dev = Duration::from_nanos(variance.sqrt() as u64);

            variants.push(VariantStats {
                name: result.variant.clone(),
                mean,
                median,
                std_dev,
                min,
                max,
                speedup: 1.0, // Will be updated
            });
        }

        // Find fastest variant
        let fastest_idx = variants
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| v.median)
            .map(|(i, _)| i)
            .unwrap_or(0);

        let fastest = variants[fastest_idx].name.clone();
        let baseline = variants.first().map(|v| v.name.clone()).unwrap_or_default();

        // Calculate speedups relative to baseline
        let baseline_time = variants[0].median.as_nanos() as f64;
        for variant in &mut variants {
            variant.speedup = baseline_time / variant.median.as_nanos() as f64;
        }

        Self {
            name: name.to_string(),
            variants,
            fastest,
            baseline,
        }
    }

    pub fn display(&self) {
        println!("\n=== Benchmark: {} ===\n", self.name);

        for variant in &self.variants {
            println!("{}: ", variant.name);
            println!(
                "  Mean: {:.3}ms ± {:.3}ms",
                variant.mean.as_secs_f64() * 1000.0,
                variant.std_dev.as_secs_f64() * 1000.0
            );
            println!("  Median: {:.3}ms", variant.median.as_secs_f64() * 1000.0);
            println!(
                "  Min: {:.3}ms, Max: {:.3}ms",
                variant.min.as_secs_f64() * 1000.0,
                variant.max.as_secs_f64() * 1000.0
            );

            if variant.name == self.baseline {
                println!("  (baseline)");
            } else {
                println!("  Speedup: {:.2}x", variant.speedup);
            }
            println!();
        }

        println!(
            "Fastest: {} ({:.2}x faster than baseline)",
            self.fastest,
            self.variants
                .iter()
                .find(|v| v.name == self.fastest)
                .map(|v| v.speedup)
                .unwrap_or(1.0)
        );
    }
}

/// Auto-tuning for optimal parameters
pub struct AutoTuner {
    pub name: String,
    pub test_data: Vec<(String, Box<dyn Fn() -> NdimageResult<Duration>>)>,
}

impl AutoTuner {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            test_data: Vec::new(),
        }
    }

    pub fn add_variant<F>(&mut self, name: impl Into<String>, f: F)
    where
        F: Fn() -> NdimageResult<Duration> + 'static,
    {
        self.test_data.push((name.into(), Box::new(f)));
    }

    pub fn find_optimal(&self) -> NdimageResult<String> {
        let mut best_time = Duration::MAX;
        let mut best_variant = String::new();

        for (name, test_fn) in &self.test_data {
            let time = test_fn()?;
            if time < best_time {
                best_time = time;
                best_variant = name.clone();
            }
        }

        Ok(best_variant)
    }
}

/// Performance optimization advisor
///
/// Analyzes profiling data and provides specific optimization recommendations
pub struct OptimizationAdvisor {
    metrics: Vec<OperationMetrics>,
    hardware_info: HardwareInfo,
}

#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub cpu_cores: usize,
    pub simd_support: SimdSupport,
    pub gpu_available: bool,
    pub total_memory: usize,
    pub cache_sizes: CacheSizes,
}

#[derive(Debug, Clone)]
pub struct SimdSupport {
    pub sse: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512: bool,
    pub neon: bool,
}

#[derive(Debug, Clone)]
pub struct CacheSizes {
    pub l1: usize,
    pub l2: usize,
    pub l3: usize,
}

impl OptimizationAdvisor {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            hardware_info: HardwareInfo::detect(),
        }
    }

    pub fn analyze(&mut self, metrics: &[OperationMetrics]) -> OptimizationReport {
        self.metrics = metrics.to_vec();

        let mut recommendations = Vec::new();

        // Analyze memory access patterns
        recommendations.extend(self.analyze_memory_patterns());

        // Analyze computation patterns
        recommendations.extend(self.analyze_computation_patterns());

        // Analyze parallelization opportunities
        recommendations.extend(self.analyze_parallelization());

        // Analyze GPU offloading opportunities
        recommendations.extend(self.analyze_gpu_opportunities());

        let estimated_speedup = self.estimate_speedup(&recommendations);
        let implementation_difficulty = self.assess_difficulty(&recommendations);

        OptimizationReport {
            recommendations,
            estimated_speedup,
            implementation_difficulty,
        }
    }

    fn analyze_memory_patterns(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Group operations by type
        let mut op_groups: HashMap<String, Vec<&OperationMetrics>> = HashMap::new();
        for metric in &self.metrics {
            op_groups
                .entry(metric.name.clone())
                .or_default()
                .push(metric);
        }

        // Check for cache-unfriendly access patterns
        for (op_name, metrics) in op_groups {
            let avg_array_size: usize = metrics
                .iter()
                .map(|m| m.arrayshape.iter().product::<usize>())
                .sum::<usize>()
                / metrics.len().max(1);

            let element_size = std::mem::size_of::<f64>(); // Assume f64
            let working_set_size = avg_array_size * element_size;

            if working_set_size > self.hardware_info.cache_sizes.l3 {
                recommendations.push(OptimizationRecommendation {
                    operation: op_name.clone(),
                    category: OptimizationCategory::Memory,
                    description: "Working set exceeds L3 cache".to_string(),
                    suggestion: "Consider tiling/blocking to improve cache locality".to_string(),
                    estimated_improvement: 1.5,
                });
            }

            // Check for strided access patterns
            if op_name.contains("transpose") || op_name.contains("permute") {
                recommendations.push(OptimizationRecommendation {
                    operation: op_name,
                    category: OptimizationCategory::Memory,
                    description: "Potentially cache-unfriendly access pattern".to_string(),
                    suggestion: "Use blocked/tiled algorithms for better cache usage".to_string(),
                    estimated_improvement: 1.3,
                });
            }
        }

        recommendations
    }

    fn analyze_computation_patterns(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Check for SIMD opportunities
        for metric in &self.metrics {
            let array_size: usize = metric.arrayshape.iter().product();

            if array_size > 1000 && !metric.name.contains("simd") {
                if self.hardware_info.simd_support.avx2 {
                    recommendations.push(OptimizationRecommendation {
                        operation: metric.name.clone(),
                        category: OptimizationCategory::Vectorization,
                        description: "Operation could benefit from SIMD vectorization".to_string(),
                        suggestion: "Implement SIMD version using AVX2 intrinsics".to_string(),
                        estimated_improvement: 2.0,
                    });
                }
            }
        }

        recommendations
    }

    fn analyze_parallelization(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        for metric in &self.metrics {
            let array_size: usize = metric.arrayshape.iter().product();

            // Check if operation is large enough to benefit from parallelization
            if array_size > 50_000 && metric.thread_count == 1 {
                recommendations.push(OptimizationRecommendation {
                    operation: metric.name.clone(),
                    category: OptimizationCategory::Parallelization,
                    description: "Large operation running on single thread".to_string(),
                    suggestion: format!(
                        "Parallelize across {} cores for better performance",
                        self.hardware_info.cpu_cores
                    ),
                    estimated_improvement: (self.hardware_info.cpu_cores as f64).min(4.0),
                });
            }
        }

        recommendations
    }

    fn analyze_gpu_opportunities(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        if !self.hardware_info.gpu_available {
            return recommendations;
        }

        for metric in &self.metrics {
            let array_size: usize = metric.arrayshape.iter().product();

            // GPU is beneficial for large arrays and compute-intensive operations
            if array_size > 1_000_000 && metric.backend == Backend::Cpu {
                recommendations.push(OptimizationRecommendation {
                    operation: metric.name.clone(),
                    category: OptimizationCategory::GpuOffloading,
                    description: "Large array operation suitable for GPU acceleration".to_string(),
                    suggestion: "Offload to GPU for significant speedup".to_string(),
                    estimated_improvement: 10.0,
                });
            }
        }

        recommendations
    }

    fn estimate_speedup(&self, recommendations: &[OptimizationRecommendation]) -> f64 {
        // Estimate overall speedup (simplified model)
        let mut total_improvement = 1.0;

        for rec in recommendations {
            // Apply diminishing returns
            total_improvement *= 1.0 + (rec.estimated_improvement - 1.0) * 0.7;
        }

        total_improvement
    }

    fn assess_difficulty(
        &self,
        recommendations: &[OptimizationRecommendation],
    ) -> ImplementationDifficulty {
        let max_difficulty = recommendations
            .iter()
            .map(|r| match r.category {
                OptimizationCategory::Memory => 2,
                OptimizationCategory::Vectorization => 3,
                OptimizationCategory::Parallelization => 2,
                OptimizationCategory::GpuOffloading => 4,
                OptimizationCategory::Algorithm => 3,
            })
            .max()
            .unwrap_or(1);

        match max_difficulty {
            1 => ImplementationDifficulty::Easy,
            2 => ImplementationDifficulty::Moderate,
            3 => ImplementationDifficulty::Hard,
            _ => ImplementationDifficulty::Expert,
        }
    }
}

#[derive(Debug)]
pub struct OptimizationReport {
    pub recommendations: Vec<OptimizationRecommendation>,
    pub estimated_speedup: f64,
    pub implementation_difficulty: ImplementationDifficulty,
}

#[derive(Debug)]
pub struct OptimizationRecommendation {
    pub operation: String,
    pub category: OptimizationCategory,
    pub description: String,
    pub suggestion: String,
    pub estimated_improvement: f64,
}

#[derive(Debug)]
pub enum OptimizationCategory {
    Memory,
    Vectorization,
    Parallelization,
    GpuOffloading,
    Algorithm,
}

#[derive(Debug)]
pub enum ImplementationDifficulty {
    Easy,
    Moderate,
    Hard,
    Expert,
}

impl HardwareInfo {
    fn detect() -> Self {
        Self {
            cpu_cores: num_cpus::get(),
            simd_support: SimdSupport::detect(),
            gpu_available: cfg!(feature = "cuda") || cfg!(feature = "opencl"),
            total_memory: 16_000_000_000, // 16GB default
            cache_sizes: CacheSizes {
                l1: 32_768,    // 32KB
                l2: 262_144,   // 256KB
                l3: 8_388_608, // 8MB
            },
        }
    }
}

impl SimdSupport {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                sse: is_x86_feature_detected!("sse"),
                avx: is_x86_feature_detected!("avx"),
                avx2: is_x86_feature_detected!("avx2"),
                avx512: false, // Conservative default
                neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                sse: false,
                avx: false,
                avx2: false,
                avx512: false,
                neon: true,
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                sse: false,
                avx: false,
                avx2: false,
                avx512: false,
                neon: false,
            }
        }
    }
}

impl OptimizationReport {
    pub fn display(&self) {
        println!("\n=== Optimization Report ===\n");

        println!("Estimated overall speedup: {:.1}x", self.estimated_speedup);
        println!(
            "Implementation difficulty: {:?}\n",
            self.implementation_difficulty
        );

        println!("Recommendations:");
        for (i, rec) in self.recommendations.iter().enumerate() {
            println!("\n{}. {} - {:?}", i + 1, rec.operation, rec.category);
            println!("   Issue: {}", rec.description);
            println!("   Suggestion: {}", rec.suggestion);
            println!(
                "   Potential improvement: {:.1}x",
                rec.estimated_improvement
            );
        }
    }
}

/// Memory profiler for tracking allocations
pub struct MemoryProfiler {
    allocations: Mutex<HashMap<String, AllocationInfo>>,
    enabled: AtomicBool,
}

#[derive(Debug, Clone)]
struct AllocationInfo {
    total_allocated: usize,
    current_allocated: usize,
    peak_allocated: usize,
    allocation_count: usize,
}

impl MemoryProfiler {
    pub fn new() -> Self {
        Self {
            allocations: Mutex::new(HashMap::new()),
            enabled: AtomicBool::new(false),
        }
    }

    pub fn enable(&self) {
        self.enabled
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn disable(&self) {
        self.enabled
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn track_allocation(&self, operation: &str, size: usize) {
        if !self.enabled.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

        let mut allocations = self
            .allocations
            .lock()
            .expect("Memory allocations mutex should not be poisoned");
        let info = allocations
            .entry(operation.to_string())
            .or_insert(AllocationInfo {
                total_allocated: 0,
                current_allocated: 0,
                peak_allocated: 0,
                allocation_count: 0,
            });

        info.total_allocated += size;
        info.current_allocated += size;
        info.peak_allocated = info.peak_allocated.max(info.current_allocated);
        info.allocation_count += 1;
    }

    pub fn track_deallocation(&self, operation: &str, size: usize) {
        if !self.enabled.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

        let mut allocations = self
            .allocations
            .lock()
            .expect("Memory allocations mutex should not be poisoned");
        if let Some(info) = allocations.get_mut(operation) {
            info.current_allocated = info.current_allocated.saturating_sub(size);
        }
    }

    pub fn report(&self) -> MemoryReport {
        let allocations = self
            .allocations
            .lock()
            .expect("Memory allocations mutex should not be poisoned");

        let mut operations: Vec<_> = allocations
            .iter()
            .map(|(name, info)| (name.clone(), info.clone()))
            .collect();

        operations.sort_by_key(|(_, info)| std::cmp::Reverse(info.peak_allocated));

        MemoryReport { operations }
    }
}

#[derive(Debug)]
pub struct MemoryReport {
    operations: Vec<(String, AllocationInfo)>,
}

impl MemoryReport {
    pub fn display(&self) {
        println!("\n=== Memory Usage Report ===\n");

        for (name, info) in &self.operations {
            println!("{}: ", name);
            println!(
                "  Total allocated: {} MB",
                info.total_allocated / (1024 * 1024)
            );
            println!("  Peak usage: {} MB", info.peak_allocated / (1024 * 1024));
            println!("  Allocations: {}", info.allocation_count);
            println!(
                "  Avg allocation: {} KB",
                (info.total_allocated / info.allocation_count.max(1)) / 1024
            );
        }
    }
}

// Global memory profiler instance
lazy_static::lazy_static! {
    static ref MEMORY_PROFILER: Arc<MemoryProfiler> = Arc::new(MemoryProfiler::new());
}

#[allow(dead_code)]
pub fn enable_memory_profiling() {
    MEMORY_PROFILER.enable();
}

#[allow(dead_code)]
pub fn disable_memory_profiling() {
    MEMORY_PROFILER.disable();
}

#[allow(dead_code)]
pub fn get_memory_report() -> MemoryReport {
    MEMORY_PROFILER.report()
}

use std::sync::atomic::AtomicBool;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_profiling_scope() {
        enable_profiling();
        clear_profiling_data();

        {
            let _scope = ProfilingScope::new("test_op", &[100, 100], Backend::Cpu);
            std::thread::sleep(Duration::from_millis(10));
        }

        let report = get_performance_report();
        assert_eq!(report.operation_breakdown.len(), 1);
        assert!(report.operation_breakdown.contains_key("test_op"));
    }

    #[test]
    #[ignore = "timeout"]
    fn test_benchmark() {
        let mut bench = Benchmark::new("array_operations");

        bench
            .run("baseline", || {
                let a = array![[1.0, 2.0], [3.0, 4.0]];
                Ok(a.sum())
            })
            .expect("benchmark baseline run should succeed");

        bench
            .run("optimized", || {
                let a = array![[1.0, 2.0], [3.0, 4.0]];
                Ok(a.sum())
            })
            .expect("benchmark optimized run should succeed");

        let comparison = bench.compare();
        assert_eq!(comparison.variants.len(), 2);
    }
}
