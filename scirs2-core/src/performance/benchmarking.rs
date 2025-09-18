//! Comprehensive benchmarking framework for performance analysis
//!
//! This module provides advanced benchmarking capabilities for scientific computing operations,
//! including strategy comparison, scalability analysis, and bottleneck identification.
use crate::performance_optimization::{AdaptiveOptimizer, OptimizationStrategy};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warm-up iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Minimum benchmark duration
    pub min_duration: Duration,
    /// Maximum benchmark duration
    pub max_duration: Duration,
    /// Sample size range for testing
    pub sample_sizes: Vec<usize>,
    /// Strategies to benchmark
    pub strategies: Vec<OptimizationStrategy>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            measurement_iterations: 20,
            min_duration: Duration::from_millis(100),
            max_duration: Duration::from_secs(30),
            sample_sizes: vec![100, 1_000, 10_000, 100_000, 1_000_000],
            strategies: vec![
                OptimizationStrategy::Scalar,
                OptimizationStrategy::Simd,
                OptimizationStrategy::Parallel,
            ],
        }
    }
}

/// Benchmark result for a single measurement
#[derive(Debug, Clone)]
pub struct BenchmarkMeasurement {
    /// Strategy used
    pub strategy: OptimizationStrategy,
    /// Input size
    pub input_size: usize,
    /// Duration of measurement
    pub duration: Duration,
    /// Throughput (operations per second)
    pub throughput: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Additional metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Aggregated benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Operation name
    pub operation_name: String,
    /// All measurements
    pub measurements: Vec<BenchmarkMeasurement>,
    /// Performance summary by strategy
    pub strategy_summary: HashMap<OptimizationStrategy, StrategyPerformance>,
    /// Scalability analysis
    pub scalability_analysis: ScalabilityAnalysis,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Total benchmark duration
    pub total_duration: Duration,
}

/// Performance summary for a strategy
#[derive(Debug, Clone)]
pub struct StrategyPerformance {
    /// Average throughput
    pub avg_throughput: f64,
    /// Standard deviation of throughput
    pub throughput_stddev: f64,
    /// Average memory usage
    pub avg_memory_usage: f64,
    /// Best input size for this strategy
    pub optimal_size: usize,
    /// Performance efficiency score (0.saturating_sub(1))
    pub efficiency_score: f64,
}

/// Scalability analysis results
#[derive(Debug, Clone)]
pub struct ScalabilityAnalysis {
    /// Parallel efficiency at different sizes
    pub parallel_efficiency: HashMap<usize, f64>,
    /// Memory scaling behavior
    pub memory_scaling: MemoryScaling,
    /// Performance bottleneck analysis
    pub bottlenecks: Vec<PerformanceBottleneck>,
}

/// Memory scaling characteristics
#[derive(Debug, Clone)]
pub struct MemoryScaling {
    /// Linear coefficient (memory = linear_coeff * size + constant_coeff)
    pub linear_coefficient: f64,
    /// Constant coefficient
    pub constant_coefficient: f64,
    /// R-squared of the fit
    pub r_squared: f64,
}

/// Performance bottleneck identification
#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Input size range where bottleneck occurs
    pub size_range: (usize, usize),
    /// Performance impact (0.saturating_sub(1), higher means more severe)
    pub impact: f64,
    /// Suggested mitigation
    pub mitigation: String,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottleneckType {
    MemoryBandwidth,
    CacheLatency,
    ComputeBound,
    SynchronizationOverhead,
    AlgorithmicComplexity,
}

/// Benchmark runner for comprehensive performance analysis
#[allow(dead_code)]
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    #[allow(dead_code)]
    optimizer: AdaptiveOptimizer,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            optimizer: AdaptiveOptimizer::new(),
        }
    }

    /// Run comprehensive benchmarks for an operation
    pub fn benchmark_operation<F>(&self, operationname: &str, operation: F) -> BenchmarkResults
    where
        F: Fn(&[f64], OptimizationStrategy) -> (Duration, Vec<f64>) + Send + Sync,
    {
        let start_time = Instant::now();
        let mut measurements = Vec::new();

        // Run benchmarks for each size and strategy combination
        for &size in &self.config.sample_sizes {
            let input_data: Vec<f64> = (0..size).map(|i| i as f64).collect();

            for &strategy in &self.config.strategies {
                // Warm-up phase
                for _ in 0..self.config.warmup_iterations {
                    let _ = operation(&input_data, strategy);
                }

                // Measurement phase
                let mut durations = Vec::new();
                for _ in 0..self.config.measurement_iterations {
                    let _duration_result = operation(&input_data, strategy);
                    durations.push(std::time::Duration::from_secs(1));
                }

                // Calculate statistics
                let avg_duration = Duration::from_nanos(
                    (durations.iter().map(|d| d.as_nanos()).sum::<u128>() / durations.len() as u128)
                        as u64,
                );

                let throughput = if avg_duration.as_nanos() > 0 {
                    (size as f64) / (avg_duration.as_secs_f64())
                } else {
                    0.0
                };

                // Estimate memory usage
                let memory_usage = self.estimate_memory_usage(size, strategy);

                measurements.push(BenchmarkMeasurement {
                    strategy,
                    input_size: size,
                    duration: avg_duration,
                    throughput,
                    memory_usage,
                    custom_metrics: HashMap::new(),
                });
            }
        }

        // Analyze results
        let strategy_summary = self.analyze_strategy_performance(&measurements);
        let scalability_analysis = self.analyze_scalability(&measurements);
        let recommendations = self.generate_recommendations(&measurements, &strategy_summary);

        BenchmarkResults {
            operation_name: operationname.to_string(),
            measurements,
            strategy_summary,
            scalability_analysis,
            recommendations,
            total_duration: start_time.elapsed(),
        }
    }

    /// Analyze performance by strategy
    fn analyze_strategy_performance(
        &self,
        measurements: &[BenchmarkMeasurement],
    ) -> HashMap<OptimizationStrategy, StrategyPerformance> {
        let mut strategy_map: HashMap<OptimizationStrategy, Vec<&BenchmarkMeasurement>> =
            HashMap::new();

        for measurement in measurements {
            strategy_map
                .entry(measurement.strategy)
                .or_default()
                .push(measurement);
        }

        let mut summary = HashMap::new();
        for (strategy, strategy_measurements) in strategy_map {
            let throughputs: Vec<f64> =
                strategy_measurements.iter().map(|m| m.throughput).collect();
            let memory_usages: Vec<f64> = strategy_measurements
                .iter()
                .map(|m| m.memory_usage as f64)
                .collect();

            let avg_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
            let throughput_variance = throughputs
                .iter()
                .map(|&x| (x - avg_throughput).powi(2))
                .sum::<f64>()
                / throughputs.len() as f64;
            let throughput_stddev = throughput_variance.sqrt();

            let avg_memory_usage = memory_usages.iter().sum::<f64>() / memory_usages.len() as f64;

            // Find optimal size (highest throughput)
            let optimal_size = strategy_measurements
                .iter()
                .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
                .map(|m| m.input_size)
                .unwrap_or(0);

            // Calculate efficiency score (throughput per memory unit)
            let efficiency_score = if avg_memory_usage > 0.0 {
                (avg_throughput / avg_memory_usage * 1e6).min(1.0)
            } else {
                0.0
            };

            summary.insert(
                strategy,
                StrategyPerformance {
                    avg_throughput,
                    throughput_stddev,
                    avg_memory_usage,
                    optimal_size,
                    efficiency_score,
                },
            );
        }

        summary
    }

    /// Analyze scalability characteristics
    fn analyze_scalability(&self, measurements: &[BenchmarkMeasurement]) -> ScalabilityAnalysis {
        let mut parallel_efficiency = HashMap::new();
        let mut memory_sizes = Vec::new();
        let mut memory_usages = Vec::new();

        // Calculate parallel efficiency
        for &size in &self.config.sample_sizes {
            let scalar_throughput = measurements
                .iter()
                .find(|m| m.input_size == size && m.strategy == OptimizationStrategy::Scalar)
                .map(|m| m.throughput)
                .unwrap_or(0.0);

            let parallel_throughput = measurements
                .iter()
                .find(|m| m.input_size == size && m.strategy == OptimizationStrategy::Parallel)
                .map(|m| m.throughput)
                .unwrap_or(0.0);

            if scalar_throughput > 0.0 {
                let efficiency = parallel_throughput / (scalar_throughput * 4.0); // Assume 4 cores
                parallel_efficiency.insert(size, efficiency.min(1.0));
            }

            memory_sizes.push(size as f64);
            if let Some(measurement) = measurements.iter().find(|m| m.input_size == size) {
                memory_usages.push(measurement.memory_usage as f64);
            }
        }

        // Fit linear model for memory scaling
        let memory_scaling = self.fit_linear_model(&memory_sizes, &memory_usages);

        // Identify bottlenecks
        let bottlenecks = self.identify_bottlenecks(measurements);

        ScalabilityAnalysis {
            parallel_efficiency,
            memory_scaling,
            bottlenecks,
        }
    }

    /// Fit linear model for memory scaling analysis
    fn fit_linear_model(&self, x: &[f64], y: &[f64]) -> MemoryScaling {
        if x.len() != y.len() || x.is_empty() {
            return MemoryScaling {
                linear_coefficient: 0.0,
                constant_coefficient: 0.0,
                r_squared: 0.0,
            };
        }

        let n = x.len() as f64;
        let sum_x = x.iter().sum::<f64>();
        let sum_y = y.iter().sum::<f64>();
        let sum_xy = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum::<f64>();
        let sum_x2 = x.iter().map(|xi| xi * xi).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Calculate R-squared
        let y_mean = sum_y / n;
        let ss_tot = y.iter().map(|yi| (yi - y_mean).powi(2)).sum::<f64>();
        let ss_res = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (yi - (slope * xi + intercept)).powi(2))
            .sum::<f64>();

        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        MemoryScaling {
            linear_coefficient: slope,
            constant_coefficient: intercept,
            r_squared,
        }
    }

    /// Identify performance bottlenecks
    fn identify_bottlenecks(
        &self,
        measurements: &[BenchmarkMeasurement],
    ) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();

        // Group by size
        let mut size_groups: HashMap<usize, Vec<&BenchmarkMeasurement>> = HashMap::new();
        for measurement in measurements {
            size_groups
                .entry(measurement.input_size)
                .or_default()
                .push(measurement);
        }

        for (&size, group) in &size_groups {
            // Check for memory bandwidth bottleneck
            let max_throughput = group.iter().map(|m| m.throughput).fold(0.0f64, f64::max);
            let min_throughput = group
                .iter()
                .map(|m| m.throughput)
                .fold(f64::INFINITY, f64::min);

            if max_throughput > 0.0 && (max_throughput - min_throughput) / max_throughput > 0.5 {
                let impact = (max_throughput - min_throughput) / max_throughput;
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::MemoryBandwidth,
                    size_range: (size, size),
                    impact,
                    mitigation: "Consider cache-friendly data layouts or memory prefetching"
                        .to_string(),
                });
            }

            // Check for synchronization overhead in parallel strategies
            let scalar_perf = group
                .iter()
                .find(|m| m.strategy == OptimizationStrategy::Scalar)
                .map(|m| m.throughput)
                .unwrap_or(0.0);

            let parallel_perf = group
                .iter()
                .find(|m| m.strategy == OptimizationStrategy::Parallel)
                .map(|m| m.throughput)
                .unwrap_or(0.0);

            if scalar_perf > 0.0 && parallel_perf / scalar_perf < 2.0 {
                let impact = 1.0 - (parallel_perf / (scalar_perf * 4.0));
                bottlenecks.push(PerformanceBottleneck {
                    bottleneck_type: BottleneckType::SynchronizationOverhead,
                    size_range: (size, size),
                    impact,
                    mitigation: "Reduce synchronization points or increase work per thread"
                        .to_string(),
                });
            }
        }

        bottlenecks
    }

    /// Generate performance recommendations
    fn generate_recommendations(
        &self,
        measurements: &[BenchmarkMeasurement],
        strategy_summary: &HashMap<OptimizationStrategy, StrategyPerformance>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Find best overall strategy
        let best_strategy = strategy_summary
            .iter()
            .max_by(|(_, a), (_, b)| a.avg_throughput.partial_cmp(&b.avg_throughput).unwrap())
            .map(|(strategy, _)| *strategy);

        if let Some(strategy) = best_strategy {
            recommendations.push(format!("{strategy:?}"));
        }

        // Analyze size-dependent recommendations
        let large_size_threshold = 50_000;
        let large_measurements: Vec<_> = measurements
            .iter()
            .filter(|m| m.input_size >= large_size_threshold)
            .collect();

        if !large_measurements.is_empty() {
            let best_large_strategy = large_measurements
                .iter()
                .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
                .map(|m| m.strategy);

            if let Some(strategy) = best_large_strategy {
                recommendations.push(format!(
                    "For large datasets (>{large_size_threshold}): Use {strategy:?}"
                ));
            }
        }

        // Memory efficiency recommendations
        let most_efficient = strategy_summary
            .iter()
            .max_by(|(_, a), (_, b)| a.efficiency_score.partial_cmp(&b.efficiency_score).unwrap())
            .map(|(strategy, perf)| (*strategy, perf.efficiency_score));

        if let Some((strategy, score)) = most_efficient {
            if score > 0.8 {
                recommendations.push(format!(
                    "Most memory-efficient strategy: {strategy:?} (efficiency: {score:.2})"
                ));
            }
        }

        // Scalability recommendations
        let parallel_measurements: Vec<_> = measurements
            .iter()
            .filter(|m| m.strategy == OptimizationStrategy::Parallel)
            .collect();

        if parallel_measurements.len() >= 2 {
            let throughput_growth = parallel_measurements.last().unwrap().throughput
                / parallel_measurements.first().unwrap().throughput;
            if throughput_growth < 2.0 {
                recommendations.push(
                    "Parallel strategy shows poor scalability - consider algorithmic improvements"
                        .to_string(),
                );
            }
        }

        if recommendations.is_empty() {
            recommendations.push(
                "Performance analysis complete - all strategies show similar characteristics"
                    .to_string(),
            );
        }

        recommendations
    }

    /// Estimate memory usage for a given strategy and size
    fn estimate_memory_usage(&self, size: usize, strategy: OptimizationStrategy) -> usize {
        let base_memory = size * std::mem::size_of::<f64>(); // Input data

        match strategy {
            OptimizationStrategy::Scalar => base_memory,
            OptimizationStrategy::Simd => base_memory + 1024, // Small SIMD overhead
            OptimizationStrategy::Parallel => base_memory + size * std::mem::size_of::<f64>(), // Temporary arrays
            OptimizationStrategy::Gpu => base_memory * 2, // GPU memory transfer overhead
            _ => base_memory,
        }
    }
}

/// Default benchmark configurations for common operations
pub mod presets {
    use super::*;

    /// Configuration for array operations benchmarks
    pub fn array_operations() -> BenchmarkConfig {
        BenchmarkConfig {
            warmup_iterations: 3,
            measurement_iterations: 10,
            min_duration: Duration::from_millis(50),
            max_duration: Duration::from_secs(10),
            sample_sizes: vec![100, 1_000, 10_000, 100_000],
            strategies: {
                let mut set = std::collections::HashSet::new();
                set.insert(OptimizationStrategy::Scalar);
                set.insert(OptimizationStrategy::Simd);
                set.insert(OptimizationStrategy::Parallel);
                set.insert(OptimizationStrategy::ModernArchOptimized);
                set.insert(OptimizationStrategy::VectorOptimized);
                set.insert(OptimizationStrategy::EnergyEfficient);
                set.into_iter().collect::<Vec<_>>()
            },
        }
    }

    /// Configuration for matrix operations benchmarks
    pub fn matrix_operations() -> BenchmarkConfig {
        BenchmarkConfig {
            warmup_iterations: 5,
            measurement_iterations: 15,
            min_duration: Duration::from_millis(100),
            max_duration: Duration::from_secs(30),
            sample_sizes: vec![64, 128, 256, 512, 1024],
            strategies: {
                let mut set = std::collections::HashSet::new();
                set.insert(OptimizationStrategy::Scalar);
                set.insert(OptimizationStrategy::Simd);
                set.insert(OptimizationStrategy::Parallel);
                set.insert(OptimizationStrategy::CacheOptimized);
                set.insert(OptimizationStrategy::ModernArchOptimized);
                set.insert(OptimizationStrategy::VectorOptimized);
                set.insert(OptimizationStrategy::HighThroughput);
                set.into_iter().collect::<Vec<_>>()
            },
        }
    }

    /// Configuration for memory-intensive operations
    pub fn memory_intensive() -> BenchmarkConfig {
        BenchmarkConfig {
            warmup_iterations: 2,
            measurement_iterations: 8,
            min_duration: Duration::from_millis(200),
            max_duration: Duration::from_secs(20),
            sample_sizes: vec![1_000, 10_000, 100_000, 1_000_000, 10_000_000],
            strategies: {
                let mut set = std::collections::HashSet::new();
                set.insert(OptimizationStrategy::Scalar);
                set.insert(OptimizationStrategy::MemoryBound);
                set.insert(OptimizationStrategy::CacheOptimized);
                set.insert(OptimizationStrategy::ModernArchOptimized);
                set.insert(OptimizationStrategy::HighThroughput);
                set.insert(OptimizationStrategy::EnergyEfficient);
                set.into_iter().collect::<Vec<_>>()
            },
        }
    }

    /// Configuration for advanced mode comprehensive benchmarking
    pub fn advanced_comprehensive() -> BenchmarkConfig {
        BenchmarkConfig {
            warmup_iterations: 10,
            measurement_iterations: 25,
            min_duration: Duration::from_millis(100),
            max_duration: Duration::from_secs(60),
            sample_sizes: vec![
                100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000,
            ],
            strategies: {
                let mut set = std::collections::HashSet::new();
                set.insert(OptimizationStrategy::Scalar);
                set.insert(OptimizationStrategy::Simd);
                set.insert(OptimizationStrategy::Parallel);
                set.insert(OptimizationStrategy::Gpu);
                set.insert(OptimizationStrategy::Hybrid);
                set.insert(OptimizationStrategy::CacheOptimized);
                set.insert(OptimizationStrategy::MemoryBound);
                set.insert(OptimizationStrategy::ComputeBound);
                set.insert(OptimizationStrategy::ModernArchOptimized);
                set.insert(OptimizationStrategy::VectorOptimized);
                set.insert(OptimizationStrategy::EnergyEfficient);
                set.insert(OptimizationStrategy::HighThroughput);
                set.into_iter().collect::<Vec<_>>()
            },
        }
    }

    /// Configuration for modern architecture specific benchmarking
    pub fn modern_architectures() -> BenchmarkConfig {
        BenchmarkConfig {
            warmup_iterations: 5,
            measurement_iterations: 15,
            min_duration: Duration::from_millis(50),
            max_duration: Duration::from_secs(30),
            sample_sizes: vec![1_000, 10_000, 100_000, 1_000_000],
            strategies: {
                let mut set = std::collections::HashSet::new();
                set.insert(OptimizationStrategy::ModernArchOptimized);
                set.insert(OptimizationStrategy::VectorOptimized);
                set.insert(OptimizationStrategy::HighThroughput);
                set.insert(OptimizationStrategy::EnergyEfficient);
                set.into_iter().collect::<Vec<_>>()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert!(config.warmup_iterations > 0);
        assert!(config.measurement_iterations > 0);
        assert!(!config.sample_sizes.is_empty());
        assert!(!config.strategies.is_empty());
    }

    #[test]
    fn test_bottleneck_type_enum() {
        let bottleneck_types = [
            BottleneckType::MemoryBandwidth,
            BottleneckType::CacheLatency,
            BottleneckType::ComputeBound,
            BottleneckType::SynchronizationOverhead,
            BottleneckType::AlgorithmicComplexity,
        ];

        for bt in &bottleneck_types {
            // Test debug formatting
            assert!(!format!("{bt:?}").is_empty());
        }

        // Test equality
        assert_eq!(
            BottleneckType::MemoryBandwidth,
            BottleneckType::MemoryBandwidth
        );
        assert_ne!(
            BottleneckType::MemoryBandwidth,
            BottleneckType::CacheLatency
        );
    }

    #[test]
    fn test_benchmark_presets() {
        let configs = [
            presets::array_operations(),
            presets::matrix_operations(),
            presets::memory_intensive(),
            presets::advanced_comprehensive(),
            presets::modern_architectures(),
        ];

        for config in &configs {
            assert!(config.warmup_iterations > 0);
            assert!(config.measurement_iterations > 0);
            assert!(!config.sample_sizes.is_empty());
            assert!(!config.strategies.is_empty());
        }
    }
}
