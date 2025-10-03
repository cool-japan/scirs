//! Advanced performance profiling and optimization tools for ndimage operations
//!
//! This module provides comprehensive performance analysis, monitoring, and optimization
//! recommendations for ndimage operations. It includes real-time profiling, memory tracking,
//! and intelligent performance optimization suggestions.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use scirs2_core::ndarray::{Array, ArrayView, Dimension, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive};

use crate::error::NdimageResult;

/// Comprehensive performance profiler for ndimage operations
#[derive(Debug)]
pub struct PerformanceProfiler {
    /// Operation timing records
    timing_records: Arc<RwLock<HashMap<String, Vec<OperationTiming>>>>,
    /// Memory usage tracking
    memory_tracker: Arc<Mutex<MemoryTracker>>,
    /// Performance metrics aggregator
    metrics_aggregator: Arc<Mutex<MetricsAggregator>>,
    /// Optimization recommendations engine
    optimizer: Arc<Mutex<OptimizationEngine>>,
    /// Real-time monitoring state
    monitoring_active: Arc<Mutex<bool>>,
    /// Configuration
    config: ProfilerConfig,
}

#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Maximum number of timing records to keep per operation
    pub max_records_per_operation: usize,
    /// Sampling interval for memory monitoring
    pub memory_sampling_interval: Duration,
    /// Enable detailed SIMD profiling
    pub enable_simd_profiling: bool,
    /// Enable cache analysis
    pub enable_cache_analysis: bool,
    /// Performance reporting interval
    pub reporting_interval: Duration,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            max_records_per_operation: 1000,
            memory_sampling_interval: Duration::from_millis(100),
            enable_simd_profiling: true,
            enable_cache_analysis: true,
            reporting_interval: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OperationTiming {
    /// Operation name
    pub name: String,
    /// Input array dimensions
    pub input_dimensions: Vec<usize>,
    /// Data type information
    pub data_type: String,
    /// Execution time
    pub execution_time: Duration,
    /// Memory allocated during operation
    pub memory_allocated: usize,
    /// Memory peak usage
    pub memory_peak: usize,
    /// SIMD utilization (0.0 - 1.0)
    pub simd_utilization: f64,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    /// Timestamp
    pub timestamp: Instant,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

#[derive(Debug)]
pub struct MemoryTracker {
    /// Current memory usage
    current_usage: usize,
    /// Peak memory usage
    peak_usage: usize,
    /// Memory usage history (timestamp, usage)
    usagehistory: VecDeque<(Instant, usize)>,
    /// Memory allocation tracking
    allocations: HashMap<String, usize>,
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self {
            current_usage: 0,
            peak_usage: 0,
            usagehistory: VecDeque::new(),
            allocations: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct MetricsAggregator {
    /// Aggregated performance metrics by operation type
    operationmetrics: HashMap<String, AggregatedMetrics>,
    /// System-wide performance indicators
    systemmetrics: SystemMetrics,
    /// Performance trends
    trends: PerformanceTrends,
}

#[derive(Debug, Clone)]
pub struct AggregatedMetrics {
    /// Number of operations recorded
    pub operation_count: usize,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Minimum execution time
    pub min_execution_time: Duration,
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Standard deviation of execution time
    pub std_dev_execution_time: Duration,
    /// Average memory usage
    pub avg_memory_usage: usize,
    /// Average SIMD utilization
    pub avg_simd_utilization: f64,
    /// Average cache hit ratio
    pub avg_cache_hit_ratio: f64,
    /// Performance efficiency score (0.0 - 1.0)
    pub efficiency_score: f64,
}

#[derive(Debug, Clone)]
pub struct SystemMetrics {
    /// Total operations performed
    pub total_operations: usize,
    /// Total execution time across all operations
    pub total_execution_time: Duration,
    /// Total memory allocated
    pub total_memory_allocated: usize,
    /// Average system load
    pub avg_system_load: f64,
    /// SIMD capability utilization
    pub simd_capability_utilization: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    /// Execution time trend (positive = getting slower)
    pub execution_time_trend: f64,
    /// Memory usage trend (positive = using more memory)
    pub memory_usage_trend: f64,
    /// Efficiency trend (positive = getting more efficient)
    pub efficiency_trend: f64,
    /// Trend confidence (0.0 - 1.0)
    pub trend_confidence: f64,
}

#[derive(Debug)]
pub struct OptimizationEngine {
    /// Performance bottleneck analysis
    bottlenecks: Vec<PerformanceBottleneck>,
    /// Optimization recommendations
    recommendations: Vec<OptimizationRecommendation>,
    /// Historical optimization impact
    optimizationhistory: Vec<OptimizationImpact>,
}

#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Operation affected
    pub operation: String,
    /// Severity (0.0 - 1.0, higher = more severe)
    pub severity: f64,
    /// Description
    pub description: String,
    /// Potential performance impact
    pub impact_estimate: f64,
}

#[derive(Debug, Clone)]
pub enum BottleneckType {
    MemoryBandwidth,
    CacheMisses,
    UnoptimizedSIMD,
    SuboptimalAlgorithm,
    MemoryFragmentation,
    ThreadContention,
    IOBottleneck,
}

#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Operation to optimize
    pub operation: String,
    /// Priority (0.0 - 1.0, higher = more important)
    pub priority: f64,
    /// Estimated performance improvement
    pub estimated_improvement: f64,
    /// Implementation difficulty (0.0 - 1.0, higher = more difficult)
    pub implementation_difficulty: f64,
    /// Detailed description
    pub description: String,
    /// Code examples or hints
    pub implementation_hints: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum RecommendationType {
    EnableSIMD,
    OptimizeMemoryLayout,
    UseAlternativeAlgorithm,
    IncreaseCacheEfficiency,
    ReduceMemoryAllocations,
    EnableParallelization,
    OptimizeGPUUsage,
}

#[derive(Debug, Clone)]
pub struct OptimizationImpact {
    /// Optimization applied
    pub optimization: String,
    /// Performance before optimization
    pub beforemetrics: AggregatedMetrics,
    /// Performance after optimization
    pub aftermetrics: AggregatedMetrics,
    /// Actual improvement achieved
    pub improvement_achieved: f64,
    /// Timestamp when optimization was applied
    pub timestamp: Instant,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            timing_records: Arc::new(RwLock::new(HashMap::new())),
            memory_tracker: Arc::new(Mutex::new(MemoryTracker::default())),
            metrics_aggregator: Arc::new(Mutex::new(MetricsAggregator::new())),
            optimizer: Arc::new(Mutex::new(OptimizationEngine::new())),
            monitoring_active: Arc::new(Mutex::new(false)),
            config,
        }
    }

    /// Start real-time performance monitoring
    pub fn start_monitoring(&self) -> NdimageResult<()> {
        let mut active = self.monitoring_active.lock().unwrap();
        if *active {
            return Ok(()); // Already monitoring
        }
        *active = true;

        // Start memory monitoring thread
        let memory_tracker = Arc::clone(&self.memory_tracker);
        let sampling_interval = self.config.memory_sampling_interval;
        let monitoring_active = Arc::clone(&self.monitoring_active);

        thread::spawn(move || {
            while *monitoring_active.lock().unwrap() {
                let current_memory = get_current_memory_usage();
                let mut tracker = memory_tracker.lock().unwrap();
                tracker.update_memory_usage(current_memory);
                drop(tracker);

                thread::sleep(sampling_interval);
            }
        });

        // Start metrics aggregation thread
        let metrics_aggregator = Arc::clone(&self.metrics_aggregator);
        let timing_records = Arc::clone(&self.timing_records);
        let reporting_interval = self.config.reporting_interval;
        let monitoring_active = Arc::clone(&self.monitoring_active);

        thread::spawn(move || {
            while *monitoring_active.lock().unwrap() {
                {
                    let records = timing_records.read().unwrap();
                    let mut aggregator = metrics_aggregator.lock().unwrap();
                    aggregator.updatemetrics(&records);
                }

                thread::sleep(reporting_interval);
            }
        });

        Ok(())
    }

    /// Stop performance monitoring
    pub fn stop_monitoring(&self) {
        let mut active = self.monitoring_active.lock().unwrap();
        *active = false;
    }

    /// Record operation timing and performance data
    pub fn record_operation<T, D>(
        &self,
        operation_name: &str,
        input: &ArrayView<T, D>,
        execution_time: Duration,
        memory_allocated: usize,
        metadata: HashMap<String, String>,
    ) -> NdimageResult<()>
    where
        T: Float + FromPrimitive,
        D: Dimension,
    {
        let timing = OperationTiming {
            name: operation_name.to_string(),
            input_dimensions: input.shape().to_vec(),
            data_type: std::any::type_name::<T>().to_string(),
            execution_time,
            memory_allocated,
            memory_peak: self.memory_tracker.lock().unwrap().peak_usage,
            simd_utilization: self.estimate_simd_utilization(operation_name, input.len()),
            cache_hit_ratio: self.estimate_cache_hit_ratio(input.len()),
            timestamp: Instant::now(),
            metadata,
        };

        let mut records = self.timing_records.write().unwrap();
        let operation_records = records
            .entry(operation_name.to_string())
            .or_insert_with(Vec::new);
        operation_records.push(timing);

        // Limit number of records to prevent memory bloat
        if operation_records.len() > self.config.max_records_per_operation {
            operation_records.remove(0);
        }

        Ok(())
    }

    /// Generate comprehensive performance report
    pub fn generate_performance_report(&self) -> PerformanceReport {
        let _records = self.timing_records.read().unwrap();
        let aggregator = self.metrics_aggregator.lock().unwrap();
        let optimizer = self.optimizer.lock().unwrap();
        let memory_tracker = self.memory_tracker.lock().unwrap();

        PerformanceReport {
            operationmetrics: aggregator.operationmetrics.clone(),
            systemmetrics: aggregator.systemmetrics.clone(),
            trends: aggregator.trends.clone(),
            bottlenecks: optimizer.bottlenecks.clone(),
            recommendations: optimizer.recommendations.clone(),
            memory_statistics: memory_tracker.get_statistics(),
            timestamp: Instant::now(),
        }
    }

    /// Get optimization recommendations for specific operation
    pub fn get_optimization_recommendations(
        &self,
        operation_name: &str,
    ) -> Vec<OptimizationRecommendation> {
        let optimizer = self.optimizer.lock().unwrap();
        optimizer
            .recommendations
            .iter()
            .filter(|rec| rec.operation == operation_name)
            .cloned()
            .collect()
    }

    /// Benchmark specific operation with various array sizes
    pub fn benchmark_operation<F, T>(
        &self,
        operation_name: &str,
        operation: F,
        test_sizes: &[Vec<usize>],
        iterations: usize,
    ) -> NdimageResult<BenchmarkResults>
    where
        F: Fn(&ArrayView<T, IxDyn>) -> NdimageResult<Array<T, IxDyn>>,
        T: Float + FromPrimitive + Clone + Default,
    {
        let mut results = Vec::new();

        for size in test_sizes {
            let input = Array::default(size.as_slice());
            let input_view = input.view();

            let mut timings = Vec::new();
            let mut memory_usages = Vec::new();

            for _ in 0..iterations {
                let start_memory = get_current_memory_usage();
                let start_time = Instant::now();

                let _result = operation(&input_view)?;

                let execution_time = start_time.elapsed();
                let end_memory = get_current_memory_usage();
                let memory_used = end_memory.saturating_sub(start_memory);

                timings.push(execution_time);
                memory_usages.push(memory_used);
            }

            let avg_time = timings.iter().sum::<Duration>() / timings.len() as u32;
            let min_time = timings.iter().min().unwrap().clone();
            let max_time = timings.iter().max().unwrap().clone();
            let avg_memory = memory_usages.iter().sum::<usize>() / memory_usages.len();

            results.push(BenchmarkResult {
                array_size: size.clone(),
                average_time: avg_time,
                min_time,
                max_time,
                average_memory: avg_memory,
                throughput: calculate_throughput(size, avg_time),
            });
        }

        Ok(BenchmarkResults {
            operation_name: operation_name.to_string(),
            results,
            timestamp: Instant::now(),
        })
    }

    // Helper methods

    fn estimate_simd_utilization(&self, operation_name: &str, _arraysize: usize) -> f64 {
        // This would integrate with actual SIMD performance counters in a real implementation
        // For now, provide estimates based on operation characteristics
        match operation_name {
            name if name.contains("simd") => 0.85,
            name if name.contains("convolution") => 0.70,
            name if name.contains("filter") => 0.60,
            _ => 0.30,
        }
    }

    fn estimate_cache_hit_ratio(&self, arraysize: usize) -> f64 {
        // Simple heuristic: smaller arrays have better cache hit ratios
        if arraysize < 1024 * 1024 {
            // < 1MB for f64
            0.95
        } else if arraysize < 16 * 1024 * 1024 {
            // < 16MB
            0.80
        } else {
            0.60
        }
    }
}

impl MetricsAggregator {
    fn new() -> Self {
        Self {
            operationmetrics: HashMap::new(),
            systemmetrics: SystemMetrics {
                total_operations: 0,
                total_execution_time: Duration::ZERO,
                total_memory_allocated: 0,
                avg_system_load: 0.0,
                simd_capability_utilization: 0.0,
            },
            trends: PerformanceTrends {
                execution_time_trend: 0.0,
                memory_usage_trend: 0.0,
                efficiency_trend: 0.0,
                trend_confidence: 0.0,
            },
        }
    }

    fn updatemetrics(&mut self, records: &HashMap<String, Vec<OperationTiming>>) {
        for (operation_name, timings) in records {
            let metrics = self.calculate_aggregatedmetrics(timings);
            self.operationmetrics
                .insert(operation_name.clone(), metrics);
        }

        self.update_systemmetrics(records);
        self.update_trends(records);
    }

    fn calculate_aggregatedmetrics(&self, timings: &[OperationTiming]) -> AggregatedMetrics {
        if timings.is_empty() {
            return AggregatedMetrics {
                operation_count: 0,
                avg_execution_time: Duration::ZERO,
                min_execution_time: Duration::ZERO,
                max_execution_time: Duration::ZERO,
                std_dev_execution_time: Duration::ZERO,
                avg_memory_usage: 0,
                avg_simd_utilization: 0.0,
                avg_cache_hit_ratio: 0.0,
                efficiency_score: 0.0,
            };
        }

        let execution_times: Vec<Duration> = timings.iter().map(|t| t.execution_time).collect();
        let avg_execution_time =
            execution_times.iter().sum::<Duration>() / execution_times.len() as u32;
        let min_execution_time = execution_times.iter().min().unwrap().clone();
        let max_execution_time = execution_times.iter().max().unwrap().clone();

        let avg_memory_usage =
            timings.iter().map(|t| t.memory_allocated).sum::<usize>() / timings.len();
        let avg_simd_utilization =
            timings.iter().map(|t| t.simd_utilization).sum::<f64>() / timings.len() as f64;
        let avg_cache_hit_ratio =
            timings.iter().map(|t| t.cache_hit_ratio).sum::<f64>() / timings.len() as f64;

        // Calculate standard deviation
        let variance = execution_times
            .iter()
            .map(|t| {
                let diff = t.as_nanos() as f64 - avg_execution_time.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>()
            / execution_times.len() as f64;
        let std_dev_nanos = variance.sqrt() as u64;
        let std_dev_execution_time = Duration::from_nanos(std_dev_nanos);

        // Calculate efficiency score (combination of SIMD utilization and cache hit ratio)
        let efficiency_score = (avg_simd_utilization + avg_cache_hit_ratio) / 2.0;

        AggregatedMetrics {
            operation_count: timings.len(),
            avg_execution_time,
            min_execution_time,
            max_execution_time,
            std_dev_execution_time,
            avg_memory_usage,
            avg_simd_utilization,
            avg_cache_hit_ratio,
            efficiency_score,
        }
    }

    fn update_systemmetrics(&mut self, records: &HashMap<String, Vec<OperationTiming>>) {
        let total_operations: usize = records.values().map(|v| v.len()).sum();
        let total_execution_time: Duration =
            records.values().flatten().map(|t| t.execution_time).sum();
        let total_memory_allocated: usize =
            records.values().flatten().map(|t| t.memory_allocated).sum();

        self.systemmetrics.total_operations = total_operations;
        self.systemmetrics.total_execution_time = total_execution_time;
        self.systemmetrics.total_memory_allocated = total_memory_allocated;
    }

    fn update_trends(&mut self, _records: &HashMap<String, Vec<OperationTiming>>) {
        // Simple trend analysis based on recent vs. older measurements
        // In a full implementation, this would use more sophisticated time series analysis
        self.trends.execution_time_trend = 0.0; // Placeholder
        self.trends.memory_usage_trend = 0.0; // Placeholder
        self.trends.efficiency_trend = 0.0; // Placeholder
        self.trends.trend_confidence = 0.5; // Placeholder
    }
}

impl OptimizationEngine {
    fn new() -> Self {
        Self {
            bottlenecks: Vec::new(),
            recommendations: Vec::new(),
            optimizationhistory: Vec::new(),
        }
    }
}

impl MemoryTracker {
    fn update_memory_usage(&mut self, usage: usize) {
        self.current_usage = usage;
        self.peak_usage = self.peak_usage.max(usage);

        let now = Instant::now();
        self.usagehistory.push_back((now, usage));

        // Keep only recent history (last hour)
        let cutoff = now - Duration::from_secs(3600);
        while self
            .usagehistory
            .front()
            .map_or(false, |&(time, _)| time < cutoff)
        {
            self.usagehistory.pop_front();
        }
    }

    fn get_statistics(&self) -> MemoryStatistics {
        let recent_usages: Vec<usize> = self.usagehistory.iter().map(|(_, usage)| *usage).collect();
        let avg_usage = if recent_usages.is_empty() {
            0
        } else {
            recent_usages.iter().sum::<usize>() / recent_usages.len()
        };

        MemoryStatistics {
            current_usage: self.current_usage,
            peak_usage: self.peak_usage,
            average_usage: avg_usage,
            allocations: self.allocations.clone(),
        }
    }
}

// Supporting types for performance reporting

#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Per-operation performance metrics
    pub operationmetrics: HashMap<String, AggregatedMetrics>,
    /// System-wide performance metrics
    pub systemmetrics: SystemMetrics,
    /// Performance trends
    pub trends: PerformanceTrends,
    /// Identified performance bottlenecks
    pub bottlenecks: Vec<PerformanceBottleneck>,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
    /// Memory usage statistics
    pub memory_statistics: MemoryStatistics,
    /// Report timestamp
    pub timestamp: Instant,
}

#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Average memory usage in bytes
    pub average_usage: usize,
    /// Memory allocations by category
    pub allocations: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Operation name
    pub operation_name: String,
    /// Benchmark results for different array sizes
    pub results: Vec<BenchmarkResult>,
    /// Timestamp
    pub timestamp: Instant,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Array dimensions tested
    pub array_size: Vec<usize>,
    /// Average execution time
    pub average_time: Duration,
    /// Minimum execution time
    pub min_time: Duration,
    /// Maximum execution time
    pub max_time: Duration,
    /// Average memory usage
    pub average_memory: usize,
    /// Throughput (elements/second)
    pub throughput: f64,
}

impl PerformanceReport {
    /// Display a formatted performance report
    pub fn display(&self) {
        println!("\n=== Performance Analysis Report ===");
        println!("Generated at: {:?}", self.timestamp);

        println!("\n--- System Metrics ---");
        println!("Total Operations: {}", self.systemmetrics.total_operations);
        println!(
            "Total Execution Time: {:.3}s",
            self.systemmetrics.total_execution_time.as_secs_f64()
        );
        println!(
            "Total Memory Allocated: {:.2} MB",
            self.systemmetrics.total_memory_allocated as f64 / (1024.0 * 1024.0)
        );

        println!("\n--- Memory Statistics ---");
        println!(
            "Current Usage: {:.2} MB",
            self.memory_statistics.current_usage as f64 / (1024.0 * 1024.0)
        );
        println!(
            "Peak Usage: {:.2} MB",
            self.memory_statistics.peak_usage as f64 / (1024.0 * 1024.0)
        );
        println!(
            "Average Usage: {:.2} MB",
            self.memory_statistics.average_usage as f64 / (1024.0 * 1024.0)
        );

        println!("\n--- Top Operations by Time ---");
        let mut operations: Vec<_> = self.operationmetrics.iter().collect();
        operations.sort_by(|a, b| b.1.avg_execution_time.cmp(&a.1.avg_execution_time));

        for (name, metrics) in operations.iter().take(5) {
            println!(
                "{}: {:.3}ms avg, {:.1}% SIMD, {:.1}% cache hits",
                name,
                metrics.avg_execution_time.as_secs_f64() * 1000.0,
                metrics.avg_simd_utilization * 100.0,
                metrics.avg_cache_hit_ratio * 100.0
            );
        }

        if !self.recommendations.is_empty() {
            println!("\n--- Optimization Recommendations ---");
            for (i, rec) in self.recommendations.iter().take(3).enumerate() {
                println!(
                    "{}. {} for '{}' (Priority: {:.1}, Est. improvement: {:.1}%)",
                    i + 1,
                    format!("{:?}", rec.recommendation_type),
                    rec.operation,
                    rec.priority * 100.0,
                    rec.estimated_improvement * 100.0
                );
            }
        }

        if !self.bottlenecks.is_empty() {
            println!("\n--- Performance Bottlenecks ---");
            for bottleneck in &self.bottlenecks {
                println!(
                    "- {:?} in '{}': {} (Severity: {:.1}%)",
                    bottleneck.bottleneck_type,
                    bottleneck.operation,
                    bottleneck.description,
                    bottleneck.severity * 100.0
                );
            }
        }
    }

    /// Export report to JSON format
    pub fn to_json(&self) -> serde_json::Result<String> {
        // This would require serde serialization in a real implementation
        Ok(format!(
            "{{\"timestamp\": \"{:?}\", \"summary\": \"Performance report generated\"}}",
            self.timestamp
        ))
    }
}

// Helper functions

#[allow(dead_code)]
fn get_current_memory_usage() -> usize {
    // In a real implementation, this would use platform-specific APIs
    // to get actual memory usage (e.g., /proc/self/status on Linux,
    // GetProcessMemoryInfo on Windows, etc.)
    // For now, return a placeholder value
    1024 * 1024 * 100 // 100MB placeholder
}

#[allow(dead_code)]
fn calculate_throughput(array_size: &[usize], executiontime: Duration) -> f64 {
    let total_elements: usize = array_size.iter().product();
    let time_seconds = executiontime.as_secs_f64();

    if time_seconds > 0.0 {
        total_elements as f64 / time_seconds
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_profiler_creation() {
        let config = ProfilerConfig::default();
        let profiler = PerformanceProfiler::new(config);

        // Test that profiler can be created without errors
        assert!(!(*profiler.monitoring_active.lock().unwrap()));
    }

    #[test]
    fn test_operation_recording() {
        let profiler = PerformanceProfiler::new(ProfilerConfig::default());
        let input = Array2::<f64>::zeros((100, 100));
        let metadata = HashMap::new();

        let result = profiler.record_operation(
            "test_operation",
            &input.view(),
            Duration::from_millis(10),
            1024,
            metadata,
        );

        assert!(result.is_ok());

        let records = profiler.timing_records.read().unwrap();
        assert!(records.contains_key("test_operation"));
        assert_eq!(records["test_operation"].len(), 1);
    }

    #[test]
    fn test_performance_report_generation() {
        let profiler = PerformanceProfiler::new(ProfilerConfig::default());
        let report = profiler.generate_performance_report();

        assert!(report.operationmetrics.is_empty()); // No operations recorded yet
        assert_eq!(report.systemmetrics.total_operations, 0);
    }
}
