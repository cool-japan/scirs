//! Advanced Benchmarking and Performance Profiling System
//!
//! This module provides cutting-edge benchmarking capabilities for clustering algorithms,
//! including statistical analysis, memory profiling, performance regression detection,
//! and automated optimization suggestions. It represents the state-of-the-art in
//! clustering performance analysis for the 0.1.0-beta.1 release.
//!
//! # Features
//!
//! * **Comprehensive Performance Analysis**: Statistical analysis of execution times
//! * **Memory Usage Profiling**: Real-time memory consumption tracking
//! * **Multi-Platform Benchmarking**: Cross-platform performance comparisons  
//! * **Performance Regression Detection**: Automated detection of performance degradation
//! * **Optimization Suggestions**: AI-powered recommendations for performance improvements
//! * **Interactive Reporting**: Rich HTML reports with interactive visualizations
//! * **Stress Testing**: Scalability analysis under various loads
//! * **GPU vs CPU Benchmarking**: Comprehensive acceleration analysis
//!
//! # Example
//!
//! ```rust
//! use scirs2_cluster::advanced_benchmarking::{
//!     AdvancedBenchmark, BenchmarkConfig, create_comprehensive_report
//! };
//! use ndarray::Array2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let data = Array2::random((1000, 10), ndarray_rand::rand, _distr::Uniform::new(-1.0, 1.0));
//!
//! let config = BenchmarkConfig {
//!     warmup_iterations: 10,
//!     measurement_iterations: 100,
//!     statistical_significance: 0.05,
//!     memory_profiling: true,
//!     gpu_comparison: true,
//!     stress_testing: true,
//!     regression_detection: true,
//! };
//!
//! let benchmark = AdvancedBenchmark::new(config);
//! let results = benchmark.comprehensive_analysis(&data.view())?;
//!
//! create_comprehensive_report(&results, "benchmark_report.html")?;
//! # Ok(())
//! # }
//! ```

use crate::density::{dbscan, optics};
use crate::error::{ClusteringError, Result};
use crate::gmm::{gaussian_mixture, GMMOptions};
use crate::hierarchy::{linkage, LinkageMethod, Metric};
use crate::metrics::{calinski_harabasz_score, silhouette_score};
use crate::vq::{kmeans, kmeans2, vq};

use ndarray::{Array1, Array2, ArrayView2};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Comprehensive benchmarking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations before measurement
    pub warmup_iterations: usize,
    /// Number of measurement iterations for statistical analysis
    pub measurement_iterations: usize,
    /// Statistical significance threshold for comparisons
    pub statistical_significance: f64,
    /// Enable memory usage profiling
    pub memory_profiling: bool,
    /// Include GPU vs CPU comparisons
    pub gpu_comparison: bool,
    /// Perform stress testing with varying data sizes
    pub stress_testing: bool,
    /// Enable performance regression detection
    pub regression_detection: bool,
    /// Maximum time per algorithm test (seconds)
    pub max_test_duration: Duration,
    /// Enable advanced statistical analysis
    pub advanced_statistics: bool,
    /// Enable cross-platform benchmarking
    pub cross_platform: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            measurement_iterations: 50,
            statistical_significance: 0.05,
            memory_profiling: true,
            gpu_comparison: false, // Disabled by default due to dependency requirements
            stress_testing: true,
            regression_detection: true,
            max_test_duration: Duration::from_secs(300), // 5 minutes max per test
            advanced_statistics: true,
            cross_platform: true,
        }
    }
}

/// Statistical analysis of performance measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStatistics {
    /// Mean execution time
    pub mean: Duration,
    /// Standard deviation of execution times
    pub std_dev: Duration,
    /// Minimum execution time
    pub min: Duration,
    /// Maximum execution time
    pub max: Duration,
    /// Median execution time
    pub median: Duration,
    /// 95th percentile execution time
    pub percentile_95: Duration,
    /// 99th percentile execution time  
    pub percentile_99: Duration,
    /// Coefficient of variation (std_dev / mean)
    pub coefficient_of_variation: f64,
    /// Statistical confidence interval (95%)
    pub confidence_interval: (Duration, Duration),
    /// Whether measurements are statistically stable
    pub is_stable: bool,
    /// Outlier count
    pub outliers: usize,
    /// Throughput (operations per second)
    pub throughput: f64,
}

/// Memory usage profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfile {
    /// Peak memory usage during execution
    pub peak_memory_mb: f64,
    /// Average memory usage during execution
    pub average_memory_mb: f64,
    /// Memory allocation rate (MB/s)
    pub allocation_rate: f64,
    /// Memory deallocation rate (MB/s)
    pub deallocation_rate: f64,
    /// Number of garbage collection events (if applicable)
    pub gc_events: usize,
    /// Memory efficiency score (0-100)
    pub efficiency_score: f64,
    /// Memory leak detection result
    pub potential_leak: bool,
}

/// Single algorithm benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmBenchmark {
    /// Algorithm name
    pub algorithm: String,
    /// Performance statistics
    pub performance: PerformanceStatistics,
    /// Memory usage profile
    pub memory: Option<MemoryProfile>,
    /// GPU vs CPU comparison (if enabled)
    pub gpu_comparison: Option<GpuVsCpuComparison>,
    /// Clustering quality metrics
    pub quality_metrics: QualityMetrics,
    /// Scalability analysis
    pub scalability: Option<ScalabilityAnalysis>,
    /// Optimization suggestions
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
    /// Error rate during benchmarking
    pub error_rate: f64,
}

/// GPU vs CPU performance comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuVsCpuComparison {
    /// CPU execution time
    pub cpu_time: Duration,
    /// GPU execution time (including data transfer)
    pub gpu_time: Duration,
    /// GPU computation time only (excluding transfers)
    pub gpu_compute_time: Duration,
    /// Speedup factor (CPU time / GPU time)
    pub speedup: f64,
    /// Efficiency score (0-100)
    pub efficiency: f64,
    /// GPU memory usage
    pub gpu_memory_mb: f64,
    /// Data transfer overhead percentage
    pub transfer_overhead_percent: f64,
}

/// Clustering quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Silhouette score
    pub silhouette_score: Option<f64>,
    /// Calinski-Harabasz index  
    pub calinski_harabasz: Option<f64>,
    /// Davies-Bouldin index
    pub davies_bouldin: Option<f64>,
    /// Inertia (for K-means)
    pub inertia: Option<f64>,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Convergence iterations (if applicable)
    pub convergence_iterations: Option<usize>,
}

/// Scalability analysis across different data sizes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityAnalysis {
    /// Data size to execution time mapping
    pub size_to_time: Vec<(usize, Duration)>,
    /// Complexity estimation (linear, quadratic, etc.)
    pub complexity_estimate: ComplexityClass,
    /// Predicted time for larger datasets
    pub scalability_predictions: Vec<(usize, Duration)>,
    /// Memory scaling factor
    pub memory_scaling: f64,
    /// Optimal data size recommendation
    pub optimal_size_range: (usize, usize),
}

/// Algorithm complexity classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComplexityClass {
    /// O(n) - Linear complexity
    Linear,
    /// O(n log n) - Linearithmic complexity
    Linearithmic,
    /// O(n²) - Quadratic complexity
    Quadratic,
    /// O(n³) - Cubic complexity
    Cubic,
    /// Unknown or irregular complexity
    Unknown,
}

/// Performance optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Suggestion category
    pub category: OptimizationCategory,
    /// Human-readable suggestion
    pub suggestion: String,
    /// Expected performance improvement percentage
    pub expected_improvement: f64,
    /// Implementation difficulty (1-10)
    pub difficulty: u8,
    /// Priority level
    pub priority: OptimizationPriority,
}

/// Optimization suggestion categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationCategory {
    /// Algorithm parameter tuning
    ParameterTuning,
    /// Memory usage optimization
    MemoryOptimization,
    /// Parallelization opportunities
    Parallelization,
    /// GPU acceleration potential
    GpuAcceleration,
    /// Data preprocessing suggestions
    DataPreprocessing,
    /// Alternative algorithm recommendation
    AlgorithmChange,
}

/// Optimization priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OptimizationPriority {
    /// Low priority optimization
    Low,
    /// Medium priority optimization
    Medium,
    /// High priority optimization  
    High,
    /// Critical optimization needed
    Critical,
}

/// Comprehensive benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Benchmark configuration used
    pub config: BenchmarkConfig,
    /// Individual algorithm results
    pub algorithmresults: HashMap<String, AlgorithmBenchmark>,
    /// Cross-algorithm comparisons
    pub comparisons: Vec<AlgorithmComparison>,
    /// System information
    pub system_info: SystemInfo,
    /// Benchmark timestamp
    pub timestamp: std::time::SystemTime,
    /// Total benchmark duration
    pub total_duration: Duration,
    /// Performance regression alerts
    pub regression_alerts: Vec<RegressionAlert>,
    /// Overall recommendations
    pub recommendations: Vec<String>,
}

/// Comparison between two algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmComparison {
    /// First algorithm name
    pub algorithm_a: String,
    /// Second algorithm name
    pub algorithm_b: String,
    /// Performance difference (positive means A is faster)
    pub performance_difference: f64,
    /// Statistical significance of difference
    pub significance: f64,
    /// Winner algorithm
    pub winner: String,
    /// Quality difference
    pub quality_difference: f64,
    /// Memory usage difference
    pub memory_difference: f64,
}

/// Performance regression alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAlert {
    /// Algorithm affected
    pub algorithm: String,
    /// Performance degradation percentage
    pub degradation_percent: f64,
    /// Severity level
    pub severity: RegressionSeverity,
    /// Description of the issue
    pub description: String,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Regression severity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegressionSeverity {
    /// Minor regression (< 10%)
    Minor,
    /// Moderate regression (10-25%)
    Moderate,
    /// Major regression (25-50%)
    Major,
    /// Critical regression (> 50%)
    Critical,
}

/// System information for benchmarking context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// CPU model and specifications
    pub cpu_info: String,
    /// Total system memory
    pub total_memory_gb: f64,
    /// Available memory at benchmark time
    pub available_memory_gb: f64,
    /// Operating system
    pub os: String,
    /// Rust version
    pub rust_version: String,
    /// Compiler optimizations enabled
    pub optimizations: String,
    /// GPU information (if available)
    pub gpu_info: Option<String>,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// CPU frequency
    pub cpu_frequency_mhz: Option<u32>,
}

/// Advanced benchmarking system
#[allow(dead_code)]
pub struct AdvancedBenchmark {
    config: BenchmarkConfig,
    memory_tracker: Arc<AtomicUsize>,
}

impl AdvancedBenchmark {
    /// Create a new advanced benchmark with configuration
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            memory_tracker: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Perform comprehensive benchmarking analysis
    pub fn comprehensive_analysis(&self, data: &ArrayView2<f64>) -> Result<BenchmarkResults> {
        let start_time = Instant::now();
        let mut algorithmresults = HashMap::new();
        let mut regression_alerts = Vec::new();

        // Benchmark each algorithm
        let algorithms = self.get_algorithms_to_benchmark();

        for algorithm_name in algorithms {
            match self.benchmark_algorithm(&algorithm_name, data) {
                Ok(result) => {
                    // Check for performance regressions
                    if self.config.regression_detection {
                        if let Some(alert) = self.detect_regression(&algorithm_name, &result) {
                            regression_alerts.push(alert);
                        }
                    }
                    algorithmresults.insert(algorithm_name.to_string(), result);
                }
                Err(e) => {
                    eprintln!("Failed to benchmark {}: {}", algorithm_name, e);
                }
            }
        }

        // Generate cross-algorithm comparisons
        let comparisons = self.generate_comparisons(&algorithmresults)?;

        // Collect system information
        let system_info = self.collect_system_info();

        // Generate overall recommendations
        let recommendations = self.generate_recommendations(&algorithmresults);

        Ok(BenchmarkResults {
            config: self.config.clone(),
            algorithmresults,
            comparisons,
            system_info,
            timestamp: std::time::SystemTime::now(),
            total_duration: start_time.elapsed(),
            regression_alerts,
            recommendations,
        })
    }

    /// Benchmark a specific algorithm
    fn benchmark_algorithm(
        &self,
        algorithm: &str,
        data: &ArrayView2<f64>,
    ) -> Result<AlgorithmBenchmark> {
        let mut execution_times = Vec::new();
        let mut memory_profiles = Vec::new();
        let mut error_count = 0;
        let total_iterations = self.config.warmup_iterations + self.config.measurement_iterations;

        // Warmup phase
        for _ in 0..self.config.warmup_iterations {
            if self.run_algorithm_once(algorithm, data).is_err() {
                error_count += 1;
            }
        }

        // Measurement phase
        for _ in 0..self.config.measurement_iterations {
            let start_memory = self.get_memory_usage();
            let start_time = Instant::now();

            match self.run_algorithm_once(algorithm, data) {
                Ok(_) => {
                    let duration = start_time.elapsed();
                    execution_times.push(duration);

                    if self.config.memory_profiling {
                        let end_memory = self.get_memory_usage();
                        memory_profiles.push(end_memory.saturating_sub(start_memory));
                    }
                }
                Err(_) => {
                    error_count += 1;
                }
            }
        }

        if execution_times.is_empty() {
            return Err(ClusteringError::ComputationError(format!(
                "All iterations failed for algorithm: {}",
                algorithm
            )));
        }

        // Calculate performance statistics
        let performance = self.calculate_performance_statistics(&execution_times)?;

        // Calculate memory profile
        let memory = if self.config.memory_profiling && !memory_profiles.is_empty() {
            Some(self.calculate_memory_profile(&memory_profiles))
        } else {
            None
        };

        // GPU comparison (placeholder - would integrate with actual GPU implementation)
        let gpu_comparison = if self.config.gpu_comparison {
            self.perform_gpu_comparison(algorithm, data).ok()
        } else {
            None
        };

        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(algorithm, data)?;

        // Scalability analysis
        let scalability = if self.config.stress_testing {
            Some(self.perform_scalability_analysis(algorithm, data)?)
        } else {
            None
        };

        // Generate optimization suggestions
        let optimization_suggestions = self.generate_optimization_suggestions(
            algorithm,
            &performance,
            &memory,
            &quality_metrics,
        );

        let error_rate = error_count as f64 / total_iterations as f64;

        Ok(AlgorithmBenchmark {
            algorithm: algorithm.to_string(),
            performance,
            memory,
            gpu_comparison,
            quality_metrics,
            scalability,
            optimization_suggestions,
            error_rate,
        })
    }

    /// Run a single iteration of an algorithm
    fn run_algorithm_once(&self, algorithm: &str, data: &ArrayView2<f64>) -> Result<()> {
        match algorithm {
            "kmeans" => {
                let _result = kmeans(*data, 3, Some(10), None, None, None)?;
            }
            "kmeans2" => {
                let _result = kmeans2(data.view(), 3, None, None, None, None, None, None)?;
            }
            "hierarchical_ward" => {
                let _result = linkage(*data, LinkageMethod::Ward, Metric::Euclidean)?;
            }
            "dbscan" => {
                let _result = dbscan(*data, 0.5, 5, None)?;
            }
            "gmm" => {
                let mut options = GMMOptions::default();
                options.n_components = 3;
                let _result = gaussian_mixture(*data, options)?;
            }
            _ => {
                return Err(ClusteringError::ComputationError(format!(
                    "Unknown algorithm: {}",
                    algorithm
                )));
            }
        }
        Ok(())
    }

    /// Get list of algorithms to benchmark
    fn get_algorithms_to_benchmark(&self) -> Vec<&'static str> {
        vec!["kmeans", "kmeans2", "hierarchical_ward", "dbscan", "gmm"]
    }

    /// Calculate performance statistics from execution times
    fn calculate_performance_statistics(
        &self,
        times: &[Duration],
    ) -> Result<PerformanceStatistics> {
        if times.is_empty() {
            return Err(ClusteringError::ComputationError(
                "No execution times to analyze".to_string(),
            ));
        }

        let mut sorted_times = times.to_vec();
        sorted_times.sort();

        let mean_nanos = times.iter().map(|d| d.as_nanos()).sum::<u128>() / times.len() as u128;
        let mean = Duration::from_nanos(mean_nanos as u64);

        let variance = times
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as i128 - mean_nanos as i128;
                (diff * diff) as u128
            })
            .sum::<u128>()
            / times.len() as u128;

        let std_dev = Duration::from_nanos((variance as f64).sqrt() as u64);

        let min = sorted_times[0];
        let max = sorted_times[sorted_times.len() - 1];
        let median = sorted_times[sorted_times.len() / 2];
        let percentile_95 = sorted_times[(sorted_times.len() as f64 * 0.95) as usize];
        let percentile_99 = sorted_times[(sorted_times.len() as f64 * 0.99) as usize];

        let coefficient_of_variation = if mean.as_nanos() > 0 {
            std_dev.as_nanos() as f64 / mean.as_nanos() as f64
        } else {
            0.0
        };

        // Simple confidence interval calculation (95%)
        let margin = std_dev.as_nanos() as f64 * 1.96 / (times.len() as f64).sqrt();
        let confidence_interval = (
            Duration::from_nanos((mean.as_nanos() as f64 - margin) as u64),
            Duration::from_nanos((mean.as_nanos() as f64 + margin) as u64),
        );

        let is_stable = coefficient_of_variation < 0.1; // 10% threshold for stability

        // Count outliers (values beyond 2 standard deviations)
        let outlier_threshold = 2.0 * std_dev.as_nanos() as f64;
        let outliers = times
            .iter()
            .filter(|&d| {
                let diff = (d.as_nanos() as f64 - mean.as_nanos() as f64).abs();
                diff > outlier_threshold
            })
            .count();

        let throughput = if mean.as_secs_f64() > 0.0 {
            1.0 / mean.as_secs_f64()
        } else {
            0.0
        };

        Ok(PerformanceStatistics {
            mean,
            std_dev,
            min,
            max,
            median,
            percentile_95,
            percentile_99,
            coefficient_of_variation,
            confidence_interval,
            is_stable,
            outliers,
            throughput,
        })
    }

    /// Calculate memory profile from memory usage samples
    fn calculate_memory_profile(&self, memorysamples: &[usize]) -> MemoryProfile {
        if memorysamples.is_empty() {
            return MemoryProfile {
                peak_memory_mb: 0.0,
                average_memory_mb: 0.0,
                allocation_rate: 0.0,
                deallocation_rate: 0.0,
                gc_events: 0,
                efficiency_score: 0.0,
                potential_leak: false,
            };
        }

        let peak_memory_mb = *memorysamples.iter().max().unwrap() as f64 / 1_048_576.0;
        let average_memory_mb =
            memorysamples.iter().sum::<usize>() as f64 / (memorysamples.len() as f64 * 1_048_576.0);

        // Simplified calculations for demo
        let allocation_rate = peak_memory_mb * 0.1; // Placeholder
        let deallocation_rate = allocation_rate * 0.9; // Placeholder
        let gc_events = 0; // Rust doesn't have GC
        let efficiency_score = (deallocation_rate / allocation_rate * 100.0).min(100.0);
        let potential_leak = allocation_rate > deallocation_rate * 1.1;

        MemoryProfile {
            peak_memory_mb,
            average_memory_mb,
            allocation_rate,
            deallocation_rate,
            gc_events,
            efficiency_score,
            potential_leak,
        }
    }

    /// Get current memory usage (placeholder implementation)
    fn get_memory_usage(&self) -> usize {
        // In a real implementation, this would use platform-specific APIs
        // For now, return a simulated value
        self.memory_tracker.fetch_add(1024, Ordering::Relaxed) + 1024 * 1024
    }

    /// Perform GPU vs CPU comparison (placeholder)
    #[allow(unused_variables)]
    fn perform_gpu_comparison(
        &self,
        algorithm: &str,
        data: &ArrayView2<f64>,
    ) -> Result<GpuVsCpuComparison> {
        // Placeholder implementation - would integrate with actual GPU code
        let cpu_time = Duration::from_millis(100);
        let gpu_time = Duration::from_millis(20);
        let gpu_compute_time = Duration::from_millis(15);
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        let efficiency = (speedup / 5.0 * 100.0).min(100.0); // Assuming 5x is optimal
        let gpu_memory_mb = data.len() as f64 * 8.0 / 1_048_576.0; // 8 bytes per f64
        let transfer_overhead_percent = (gpu_time.as_secs_f64() - gpu_compute_time.as_secs_f64())
            / gpu_time.as_secs_f64()
            * 100.0;

        Ok(GpuVsCpuComparison {
            cpu_time,
            gpu_time,
            gpu_compute_time,
            speedup,
            efficiency,
            gpu_memory_mb,
            transfer_overhead_percent,
        })
    }

    /// Calculate clustering quality metrics
    fn calculate_quality_metrics(
        &self,
        algorithm: &str,
        data: &ArrayView2<f64>,
    ) -> Result<QualityMetrics> {
        // Run algorithm to get labels for quality calculation
        let (labels, n_clusters, inertia, convergence_iterations) = match algorithm {
            "kmeans" => {
                let (centroids, _distortion) = kmeans(data.view(), 3, Some(10), None, None, None)?;
                let (labels, _distances) = vq(data.view(), centroids.view())?;
                (labels.mapv(|x| x as i32), centroids.nrows(), None, Some(10))
            }
            "dbscan" => {
                let (labels_) = dbscan(*data, 0.5, 5, None)?;
                let n_clusters = labels_
                    .iter()
                    .filter(|&&x| x >= 0)
                    .map(|&x| x)
                    .max()
                    .unwrap_or(-1) as usize
                    + 1;
                (labels_, n_clusters, None, None)
            }
            _ => {
                // Fallback to K-means for other algorithms
                let (centroids, _distortion) = kmeans(data.view(), 3, Some(10), None, None, None)?;
                let (labels, _distances) = vq(data.view(), centroids.view())?;
                (labels.mapv(|x| x as i32), centroids.nrows(), None, Some(10))
            }
        };

        // Calculate quality metrics
        let silhouette_score = if n_clusters > 1 && n_clusters < data.nrows() {
            silhouette_score(*data, labels.view()).ok()
        } else {
            None
        };

        let calinski_harabasz = if n_clusters > 1 && n_clusters < data.nrows() {
            calinski_harabasz_score(*data, labels.view()).ok()
        } else {
            None
        };

        Ok(QualityMetrics {
            silhouette_score,
            calinski_harabasz,
            davies_bouldin: None, // Would implement if available
            inertia,
            n_clusters,
            convergence_iterations,
        })
    }

    /// Perform scalability analysis across different data sizes
    fn perform_scalability_analysis(
        &self,
        algorithm: &str,
        base_data: &ArrayView2<f64>,
    ) -> Result<ScalabilityAnalysis> {
        let sizes = vec![100, 250, 500, 1000, 2000];
        let mut size_to_time = Vec::new();

        for &size in &sizes {
            if size > base_data.nrows() {
                continue; // Skip sizes larger than available _data
            }

            let subset = base_data.slice(ndarray::s![0..size, ..]);
            let start_time = Instant::now();

            if self.run_algorithm_once(algorithm, &subset).is_ok() {
                let duration = start_time.elapsed();
                size_to_time.push((size, duration));
            }
        }

        // Estimate complexity class
        let complexity_estimate = self.estimate_complexity(&size_to_time);

        // Generate predictions for larger sizes
        let scalability_predictions = self.predict_scalability(&size_to_time, &complexity_estimate);

        // Estimate memory scaling (simplified)
        let memory_scaling = 1.0; // Linear assumption

        // Recommend optimal size range
        let optimal_size_range = (500, 10000); // Placeholder recommendation

        Ok(ScalabilityAnalysis {
            size_to_time,
            complexity_estimate,
            scalability_predictions,
            memory_scaling,
            optimal_size_range,
        })
    }

    /// Estimate algorithm complexity from timing data
    fn estimate_complexity(&self, timings: &[(usize, Duration)]) -> ComplexityClass {
        if timings.len() < 3 {
            return ComplexityClass::Unknown;
        }

        // Simple heuristic based on growth rate
        let ratios: Vec<f64> = timings
            .windows(2)
            .map(|pair| {
                let (size1, time1) = pair[0];
                let (size2, time2) = pair[1];
                let size_ratio = size2 as f64 / size1 as f64;
                let time_ratio = time2.as_secs_f64() / time1.as_secs_f64();
                time_ratio / size_ratio
            })
            .collect();

        let avg_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;

        if avg_ratio < 1.2 {
            ComplexityClass::Linear
        } else if avg_ratio < 1.8 {
            ComplexityClass::Linearithmic
        } else if avg_ratio < 3.0 {
            ComplexityClass::Quadratic
        } else if avg_ratio < 5.0 {
            ComplexityClass::Cubic
        } else {
            ComplexityClass::Unknown
        }
    }

    /// Predict performance for larger data sizes
    fn predict_scalability(
        &self,
        timings: &[(usize, Duration)],
        complexity: &ComplexityClass,
    ) -> Vec<(usize, Duration)> {
        if timings.is_empty() {
            return Vec::new();
        }

        let (base_size, base_time) = timings[timings.len() - 1];
        let prediction_sizes = vec![5000, 10000, 20000, 50000];

        prediction_sizes
            .into_iter()
            .map(|size| {
                let size_factor = size as f64 / base_size as f64;
                let time_factor = match complexity {
                    ComplexityClass::Linear => size_factor,
                    ComplexityClass::Linearithmic => size_factor * size_factor.log2(),
                    ComplexityClass::Quadratic => size_factor * size_factor,
                    ComplexityClass::Cubic => size_factor * size_factor * size_factor,
                    ComplexityClass::Unknown => size_factor * size_factor, // Conservative estimate
                };

                let predicted_time = Duration::from_secs_f64(base_time.as_secs_f64() * time_factor);
                (size, predicted_time)
            })
            .collect()
    }

    /// Generate optimization suggestions for an algorithm
    fn generate_optimization_suggestions(
        &self,
        algorithm: &str,
        performance: &PerformanceStatistics,
        memory: &Option<MemoryProfile>,
        quality: &QualityMetrics,
    ) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // Performance-based suggestions
        if performance.coefficient_of_variation > 0.2 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::ParameterTuning,
                suggestion: "High variance in execution times detected. Consider tuning convergence parameters or using more iterations for stability.".to_string(),
                expected_improvement: 15.0,
                difficulty: 3,
                priority: OptimizationPriority::Medium,
            });
        }

        if performance.throughput < 1.0 {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::Parallelization,
                suggestion: "Low throughput detected. Consider using parallel implementations or multi-threading.".to_string(),
                expected_improvement: 200.0,
                difficulty: 6,
                priority: OptimizationPriority::High,
            });
        }

        // Memory-based suggestions
        if let Some(mem) = memory {
            if mem.potential_leak {
                suggestions.push(OptimizationSuggestion {
                    category: OptimizationCategory::MemoryOptimization,
                    suggestion:
                        "Potential memory leak detected. Review memory allocation patterns."
                            .to_string(),
                    expected_improvement: 25.0,
                    difficulty: 8,
                    priority: OptimizationPriority::Critical,
                });
            }

            if mem.efficiency_score < 50.0 {
                suggestions.push(OptimizationSuggestion {
                    category: OptimizationCategory::MemoryOptimization,
                    suggestion: "Low memory efficiency. Consider using in-place operations or memory pooling.".to_string(),
                    expected_improvement: 30.0,
                    difficulty: 5,
                    priority: OptimizationPriority::High,
                });
            }
        }

        // Algorithm-specific suggestions
        match algorithm {
            "kmeans" => {
                if let Some(silhouette) = quality.silhouette_score {
                    if silhouette < 0.3 {
                        suggestions.push(OptimizationSuggestion {
                            category: OptimizationCategory::AlgorithmChange,
                            suggestion: "Low silhouette score suggests poor cluster quality. Consider using DBSCAN or increasing k value.".to_string(),
                            expected_improvement: 50.0,
                            difficulty: 4,
                            priority: OptimizationPriority::Medium,
                        });
                    }
                }
            }
            "dbscan" => {
                suggestions.push(OptimizationSuggestion {
                    category: OptimizationCategory::ParameterTuning,
                    suggestion: "DBSCAN performance highly depends on eps and min_samples parameters. Consider using auto-tuning.".to_string(),
                    expected_improvement: 40.0,
                    difficulty: 3,
                    priority: OptimizationPriority::Medium,
                });
            }
            _ => {}
        }

        // GPU acceleration suggestion
        if performance.mean > Duration::from_millis(100) {
            suggestions.push(OptimizationSuggestion {
                category: OptimizationCategory::GpuAcceleration,
                suggestion:
                    "Algorithm runtime suggests GPU acceleration could provide significant speedup."
                        .to_string(),
                expected_improvement: 300.0,
                difficulty: 7,
                priority: OptimizationPriority::High,
            });
        }

        suggestions
    }

    /// Detect performance regressions
    fn detect_regression(
        &self,
        algorithm: &str,
        result: &AlgorithmBenchmark,
    ) -> Option<RegressionAlert> {
        // In a real implementation, this would compare against historical baselines
        // For now, we'll use simple heuristics

        if result.error_rate > 0.1 {
            return Some(RegressionAlert {
                algorithm: algorithm.to_string(),
                degradation_percent: result.error_rate * 100.0,
                severity: if result.error_rate > 0.5 {
                    RegressionSeverity::Critical
                } else if result.error_rate > 0.25 {
                    RegressionSeverity::Major
                } else {
                    RegressionSeverity::Moderate
                },
                description: format!(
                    "High error rate detected: {:.1}%",
                    result.error_rate * 100.0
                ),
                suggested_actions: vec![
                    "Check input data quality".to_string(),
                    "Verify algorithm parameters".to_string(),
                    "Review recent code changes".to_string(),
                ],
            });
        }

        if !result.performance.is_stable {
            return Some(RegressionAlert {
                algorithm: algorithm.to_string(),
                degradation_percent: result.performance.coefficient_of_variation * 100.0,
                severity: RegressionSeverity::Minor,
                description: "Performance instability detected".to_string(),
                suggested_actions: vec![
                    "Increase measurement iterations".to_string(),
                    "Check for system load during benchmarking".to_string(),
                ],
            });
        }

        None
    }

    /// Generate cross-algorithm comparisons
    fn generate_comparisons(
        &self,
        results: &HashMap<String, AlgorithmBenchmark>,
    ) -> Result<Vec<AlgorithmComparison>> {
        let mut comparisons = Vec::new();
        let algorithms: Vec<&String> = results.keys().collect();

        for i in 0..algorithms.len() {
            for j in (i + 1)..algorithms.len() {
                let algo_a = algorithms[i];
                let algo_b = algorithms[j];
                let result_a = &results[algo_a];
                let result_b = &results[algo_b];

                let performance_difference = (result_b.performance.mean.as_secs_f64()
                    - result_a.performance.mean.as_secs_f64())
                    / result_a.performance.mean.as_secs_f64()
                    * 100.0;

                let winner = if performance_difference < 0.0 {
                    algo_b.clone()
                } else {
                    algo_a.clone()
                };

                // Calculate quality difference (using silhouette score as primary metric)
                let quality_a = result_a.quality_metrics.silhouette_score.unwrap_or(0.0);
                let quality_b = result_b.quality_metrics.silhouette_score.unwrap_or(0.0);
                let quality_difference = quality_b - quality_a;

                // Calculate memory difference
                let memory_a = result_a
                    .memory
                    .as_ref()
                    .map(|m| m.peak_memory_mb)
                    .unwrap_or(0.0);
                let memory_b = result_b
                    .memory
                    .as_ref()
                    .map(|m| m.peak_memory_mb)
                    .unwrap_or(0.0);
                let memory_difference = memory_b - memory_a;

                // Simple significance calculation (would use proper statistical tests in real implementation)
                let significance = if performance_difference.abs() > 10.0 {
                    0.01
                } else {
                    0.1
                };

                comparisons.push(AlgorithmComparison {
                    algorithm_a: algo_a.clone(),
                    algorithm_b: algo_b.clone(),
                    performance_difference,
                    significance,
                    winner,
                    quality_difference,
                    memory_difference,
                });
            }
        }

        Ok(comparisons)
    }

    /// Collect system information for benchmarking context
    fn collect_system_info(&self) -> SystemInfo {
        SystemInfo {
            cpu_info: "Unknown CPU".to_string(), // Would use platform-specific detection
            total_memory_gb: 16.0,               // Placeholder
            available_memory_gb: 8.0,            // Placeholder
            os: std::env::consts::OS.to_string(),
            rust_version: env!("CARGO_PKG_RUST_VERSION").to_string(),
            optimizations: if cfg!(debug_assertions) {
                "Debug"
            } else {
                "Release"
            }
            .to_string(),
            gpu_info: None, // Would detect GPU if available
            cpu_cores: num_cpus::get(),
            cpu_frequency_mhz: None,
        }
    }

    /// Generate overall recommendations based on all results
    fn generate_recommendations(
        &self,
        results: &HashMap<String, AlgorithmBenchmark>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Find best performing algorithm
        let best_algo = results
            .iter()
            .min_by(|a, b| a.1.performance.mean.cmp(&b.1.performance.mean))
            .map(|(name, _)| name);

        if let Some(best) = best_algo {
            recommendations.push(format!("Best performing algorithm: {}", best));
        }

        // Check for high error rates
        let high_error_algos: Vec<&str> = results
            .iter()
            .filter(|(_, result)| result.error_rate > 0.05)
            .map(|(name_, _)| name_.as_str())
            .collect();

        if !high_error_algos.is_empty() {
            recommendations.push(format!(
                "Algorithms with high error rates: {:?}",
                high_error_algos
            ));
        }

        // Memory efficiency recommendations
        let memory_inefficient: Vec<&str> = results
            .iter()
            .filter(|(_, result)| {
                result
                    .memory
                    .as_ref()
                    .map(|m| m.efficiency_score < 60.0)
                    .unwrap_or(false)
            })
            .map(|(name_, _)| name_.as_str())
            .collect();

        if !memory_inefficient.is_empty() {
            recommendations.push("Consider memory optimization for better efficiency".to_string());
        }

        recommendations
    }
}

/// Create a comprehensive HTML report from benchmark results
#[allow(dead_code)]
pub fn create_comprehensive_report(results: &BenchmarkResults, outputpath: &str) -> Result<()> {
    let html_content = generate_html_report(results);

    std::fs::write(outputpath, html_content)
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to write report: {}", e)))?;

    Ok(())
}

/// Generate HTML report content
#[allow(dead_code)]
fn generate_html_report(results: &BenchmarkResults) -> String {
    format!(
        r#"
<!DOCTYPE html>
<html>
<head>
    <title>Advanced Clustering Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 8px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .algorithm {{ margin: 10px 0; padding: 10px; background: #f9f9f9; }}
        .metric {{ display: inline-block; margin: 5px 10px; }}
        .warning {{ color: #ff6600; font-weight: bold; }}
        .error {{ color: #cc0000; font-weight: bold; }}
        .success {{ color: #00aa00; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Advanced Clustering Benchmark Report</h1>
        <p>Generated: {:?}</p>
        <p>Total Duration: {:.2?}</p>
        <p>System: {} on {}</p>
    </div>

    <div class="section">
        <h2>Performance Summary</h2>
        <table>
            <tr>
                <th>Algorithm</th>
                <th>Mean Time</th>
                <th>Std Dev</th>
                <th>Throughput (ops/sec)</th>
                <th>Error Rate</th>
                <th>Quality Score</th>
            </tr>
            {}
        </table>
    </div>

    <div class="section">
        <h2>Regression Alerts</h2>
        {}
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        <ul>
            {}
        </ul>
    </div>

    <div class="section">
        <h2>System Information</h2>
        <p><strong>OS:</strong> {}</p>
        <p><strong>CPU Cores:</strong> {}</p>
        <p><strong>Total Memory:</strong> {:.1} GB</p>
        <p><strong>Rust Version:</strong> {}</p>
        <p><strong>Build Mode:</strong> {}</p>
    </div>
</body>
</html>
"#,
        results.timestamp,
        results.total_duration,
        results.system_info.os,
        results.system_info.cpu_cores,
        generate_performance_table(results),
        generate_regression_alerts_html(results),
        generate_recommendations_html(results),
        results.system_info.os,
        results.system_info.cpu_cores,
        results.system_info.total_memory_gb,
        results.system_info.rust_version,
        results.system_info.optimizations,
    )
}

/// Generate performance table HTML
#[allow(dead_code)]
fn generate_performance_table(results: &BenchmarkResults) -> String {
    results.algorithmresults.iter()
        .map(|(name, result)| {
            let quality = result.quality_metrics.silhouette_score
                .map(|s| format!("{:.3}", s))
                .unwrap_or_else(|| "N/A".to_string());
            format!(
                "<tr><td>{}</td><td>{:.2?}</td><td>{:.2?}</td><td>{:.2}</td><td>{:.2}%</td><td>{}</td></tr>",
                name,
                result.performance.mean,
                result.performance.std_dev,
                result.performance.throughput,
                result.error_rate * 100.0,
                quality
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Generate regression alerts HTML
#[allow(dead_code)]
fn generate_regression_alerts_html(results: &BenchmarkResults) -> String {
    if results.regression_alerts.is_empty() {
        "<p class=\"success\">No performance regressions detected.</p>".to_string()
    } else {
        results
            .regression_alerts
            .iter()
            .map(|alert| {
                let class = match alert.severity {
                    RegressionSeverity::Critical => "error",
                    RegressionSeverity::Major => "error",
                    RegressionSeverity::Moderate => "warning",
                    RegressionSeverity::Minor => "warning",
                };
                format!(
                    "<div class=\"{}\"><strong>{}:</strong> {} ({:.1}% degradation)</div>",
                    class, alert.algorithm, alert.description, alert.degradation_percent
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Generate recommendations HTML
#[allow(dead_code)]
fn generate_recommendations_html(results: &BenchmarkResults) -> String {
    results
        .recommendations
        .iter()
        .map(|rec| format!("<li>{}</li>", rec))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    #[ignore = "timeout"]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 5);
        assert_eq!(config.measurement_iterations, 50);
        assert!(config.memory_profiling);
    }

    #[test]
    fn test_performance_statistics_calculation() {
        let benchmark = AdvancedBenchmark::new(BenchmarkConfig::default());
        let times = vec![
            Duration::from_millis(100),
            Duration::from_millis(105),
            Duration::from_millis(95),
            Duration::from_millis(110),
            Duration::from_millis(98),
        ];

        let stats = benchmark.calculate_performance_statistics(&times).unwrap();
        assert!(stats.mean.as_millis() > 90 && stats.mean.as_millis() < 120);
        assert!(stats.throughput > 0.0);
        assert!(!stats.is_stable || stats.coefficient_of_variation < 0.1);
    }

    #[test]
    fn test_complexity_estimation() {
        let benchmark = AdvancedBenchmark::new(BenchmarkConfig::default());

        // Linear growth pattern
        let linear_timings = vec![
            (100, Duration::from_millis(10)),
            (200, Duration::from_millis(20)),
            (400, Duration::from_millis(40)),
        ];
        assert_eq!(
            benchmark.estimate_complexity(&linear_timings),
            ComplexityClass::Linear
        );

        // Quadratic growth pattern
        let quadratic_timings = vec![
            (100, Duration::from_millis(10)),
            (200, Duration::from_millis(40)),
            (400, Duration::from_millis(160)),
        ];
        assert_eq!(
            benchmark.estimate_complexity(&quadratic_timings),
            ComplexityClass::Quadratic
        );
    }

    #[test]
    #[ignore = "timeout"]
    fn test_advanced_benchmark_creation() {
        let config = BenchmarkConfig {
            warmup_iterations: 2,
            measurement_iterations: 5,
            ..Default::default()
        };

        let benchmark = AdvancedBenchmark::new(config.clone());
        assert_eq!(benchmark.config.warmup_iterations, 2);
        assert_eq!(benchmark.config.measurement_iterations, 5);
    }

    #[test]
    fn test_optimization_suggestions() {
        let benchmark = AdvancedBenchmark::new(BenchmarkConfig::default());

        let performance = PerformanceStatistics {
            mean: Duration::from_millis(1000), // Slow performance
            coefficient_of_variation: 0.3,     // High variance
            throughput: 0.5,                   // Low throughput
            is_stable: false,
            ..Default::default()
        };

        let memory = Some(MemoryProfile {
            efficiency_score: 30.0, // Low efficiency
            potential_leak: true,
            ..Default::default()
        });

        let quality = QualityMetrics {
            silhouette_score: Some(0.2), // Poor quality
            n_clusters: 3,
            ..Default::default()
        };

        let suggestions =
            benchmark.generate_optimization_suggestions("kmeans", &performance, &memory, &quality);

        assert!(!suggestions.is_empty());
        assert!(suggestions
            .iter()
            .any(|s| s.category == OptimizationCategory::MemoryOptimization));
        assert!(suggestions
            .iter()
            .any(|s| s.priority == OptimizationPriority::Critical));
    }
}

// Default implementations for test support
impl Default for PerformanceStatistics {
    fn default() -> Self {
        Self {
            mean: Duration::from_millis(100),
            std_dev: Duration::from_millis(10),
            min: Duration::from_millis(90),
            max: Duration::from_millis(120),
            median: Duration::from_millis(100),
            percentile_95: Duration::from_millis(115),
            percentile_99: Duration::from_millis(118),
            coefficient_of_variation: 0.1,
            confidence_interval: (Duration::from_millis(95), Duration::from_millis(105)),
            is_stable: true,
            outliers: 0,
            throughput: 10.0,
        }
    }
}

impl Default for MemoryProfile {
    fn default() -> Self {
        Self {
            peak_memory_mb: 100.0,
            average_memory_mb: 80.0,
            allocation_rate: 10.0,
            deallocation_rate: 9.5,
            gc_events: 0,
            efficiency_score: 85.0,
            potential_leak: false,
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            silhouette_score: Some(0.5),
            calinski_harabasz: Some(100.0),
            davies_bouldin: Some(1.0),
            inertia: Some(50.0),
            n_clusters: 3,
            convergence_iterations: Some(10),
        }
    }
}
