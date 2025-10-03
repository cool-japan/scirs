//! Performance tuning system for interpolation optimization
//!
//! This module provides sophisticated performance optimization capabilities,
//! including adaptive parameter tuning, resource management, and performance-accuracy
//! trade-off optimization for interpolation methods.

use super::types::*;
use crate::error::InterpolateResult;
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::Instant;

/// Performance tuning system for interpolation optimization
#[derive(Debug)]
pub struct PerformanceTuningSystem<F: Float + Debug> {
    /// Current tuning strategy
    strategy: PerformanceTuningStrategy,
    /// Performance targets
    targets: PerformanceTargets,
    /// Adaptive parameters
    adaptive_params: AdaptiveParameters<F>,
    /// Tuning history
    tuning_history: VecDeque<TuningResult>,
    /// Resource monitor
    resource_monitor: ResourceMonitor,
    /// Performance baselines
    performance_baselines: HashMap<InterpolationMethodType, PerformanceBaseline>,
}

/// Performance tuning strategy
#[derive(Debug, Clone)]
pub enum PerformanceTuningStrategy {
    /// Minimize execution time
    MinimizeTime,
    /// Minimize memory usage
    MinimizeMemory,
    /// Balance time and memory
    Balanced,
    /// Adaptive based on system resources
    Adaptive,
    /// Custom weighted optimization
    Custom {
        time_weight: f64,
        memory_weight: f64,
        accuracy_weight: f64,
    },
    /// System resource aware optimization
    ResourceAware {
        cpu_threshold: f64,
        memory_threshold: f64,
    },
}

/// Performance targets for optimization
#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    /// Maximum acceptable execution time (microseconds)
    pub max_execution_time: Option<f64>,
    /// Maximum memory usage (bytes)
    pub max_memory_usage: Option<usize>,
    /// Minimum throughput (operations/second)
    pub min_throughput: Option<f64>,
    /// Maximum latency (microseconds)
    pub max_latency: Option<f64>,
    /// Target CPU utilization (0-1)
    pub target_cpu_utilization: Option<f64>,
    /// Memory efficiency target (operations per MB)
    pub memory_efficiency_target: Option<f64>,
}

/// Resource monitoring system
#[derive(Debug)]
pub struct ResourceMonitor {
    /// Current CPU usage
    cpu_usage: f64,
    /// Current memory usage (bytes)
    memory_usage: usize,
    /// Available memory (bytes)
    available_memory: usize,
    /// System load average
    load_average: f64,
    /// Monitoring interval (milliseconds)
    monitoring_interval: u64,
}

/// Performance baseline for methods
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    /// Baseline execution time (microseconds)
    pub baseline_time: f64,
    /// Baseline memory usage (bytes)
    pub baseline_memory: usize,
    /// Baseline accuracy
    pub baseline_accuracy: f64,
    /// Sample size for baseline
    pub sample_size: usize,
    /// Last update timestamp
    pub last_update: Instant,
}

/// Performance optimization result
#[derive(Debug, Clone)]
pub struct PerformanceOptimizationResult {
    /// Original performance metrics
    pub original_metrics: PerformanceMetrics,
    /// Optimized performance metrics
    pub optimized_metrics: PerformanceMetrics,
    /// Optimization strategy used
    pub strategy_used: PerformanceTuningStrategy,
    /// Parameters that were adjusted
    pub parameter_adjustments: HashMap<String, ParameterAdjustment>,
    /// Overall improvement score
    pub improvement_score: f64,
    /// Optimization success
    pub success: bool,
    /// Optimization time (milliseconds)
    pub optimization_time: f64,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Execution time (microseconds)
    pub execution_time: f64,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Accuracy score
    pub accuracy: f64,
    /// Throughput (operations/second)
    pub throughput: f64,
    /// CPU utilization (0-1)
    pub cpu_utilization: f64,
    /// Cache hit ratio (0-1)
    pub cache_hit_ratio: f64,
}

/// Parameter adjustment details
#[derive(Debug, Clone)]
pub struct ParameterAdjustment {
    /// Original value
    pub original_value: f64,
    /// New value
    pub new_value: f64,
    /// Adjustment type
    pub adjustment_type: AdjustmentType,
    /// Impact on performance
    pub performance_impact: f64,
}

/// Types of parameter adjustments
#[derive(Debug, Clone)]
pub enum AdjustmentType {
    /// Increase parameter value
    Increase,
    /// Decrease parameter value
    Decrease,
    /// Optimize to specific value
    Optimize,
    /// Reset to default
    Reset,
}

impl<F: Float + Debug> PerformanceTuningSystem<F> {
    /// Create a new performance tuning system
    pub fn new() -> InterpolateResult<Self> {
        Ok(Self {
            strategy: PerformanceTuningStrategy::Balanced,
            targets: PerformanceTargets::default(),
            adaptive_params: AdaptiveParameters::default(),
            tuning_history: VecDeque::new(),
            resource_monitor: ResourceMonitor::new()?,
            performance_baselines: HashMap::new(),
        })
    }

    /// Set performance tuning strategy
    pub fn set_strategy(&mut self, strategy: PerformanceTuningStrategy) {
        self.strategy = strategy;
    }

    /// Set performance targets
    pub fn set_targets(&mut self, targets: PerformanceTargets) {
        self.targets = targets;
    }

    /// Optimize performance for given method and data characteristics
    pub fn optimize_performance(
        &mut self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        current_parameters: &HashMap<String, f64>,
        current_metrics: &PerformanceMetrics,
    ) -> InterpolateResult<PerformanceOptimizationResult> {
        let start_time = Instant::now();

        // Update resource monitoring
        self.resource_monitor.update_metrics()?;

        // Check if optimization is needed
        if self.meets_performance_targets(current_metrics) {
            return Ok(PerformanceOptimizationResult {
                original_metrics: current_metrics.clone(),
                optimized_metrics: current_metrics.clone(),
                strategy_used: self.strategy.clone(),
                parameter_adjustments: HashMap::new(),
                improvement_score: 0.0,
                success: true,
                optimization_time: start_time.elapsed().as_millis() as f64,
            });
        }

        // Apply tuning strategy
        let optimized_params = match &self.strategy {
            PerformanceTuningStrategy::MinimizeTime => {
                self.optimize_for_speed(method, data_profile, current_parameters)?
            }
            PerformanceTuningStrategy::MinimizeMemory => {
                self.optimize_for_memory(method, data_profile, current_parameters)?
            }
            PerformanceTuningStrategy::Balanced => {
                self.optimize_balanced(method, data_profile, current_parameters)?
            }
            PerformanceTuningStrategy::Adaptive => {
                self.optimize_adaptive(method, data_profile, current_parameters)?
            }
            PerformanceTuningStrategy::Custom {
                time_weight,
                memory_weight,
                accuracy_weight,
            } => self.optimize_custom(
                method,
                data_profile,
                current_parameters,
                *time_weight,
                *memory_weight,
                *accuracy_weight,
            )?,
            PerformanceTuningStrategy::ResourceAware {
                cpu_threshold,
                memory_threshold,
            } => self.optimize_resource_aware(
                method,
                data_profile,
                current_parameters,
                *cpu_threshold,
                *memory_threshold,
            )?,
        };

        // Calculate parameter adjustments
        let parameter_adjustments =
            self.calculate_parameter_adjustments(current_parameters, &optimized_params);

        // Predict optimized metrics
        let optimized_metrics = self.predict_optimized_metrics(
            method,
            data_profile,
            &optimized_params,
            current_metrics,
        )?;

        // Calculate improvement score
        let improvement_score =
            self.calculate_improvement_score(current_metrics, &optimized_metrics);

        let result = PerformanceOptimizationResult {
            original_metrics: current_metrics.clone(),
            optimized_metrics,
            strategy_used: self.strategy.clone(),
            parameter_adjustments,
            improvement_score,
            success: improvement_score > 0.0,
            optimization_time: start_time.elapsed().as_millis() as f64,
        };

        // Store tuning result
        self.tuning_history.push_back(TuningResult {
            improvement: improvement_score,
            iterations: 1, // Single iteration for this tuning attempt
            final_parameters: optimized_params.values().cloned().collect(),
            converged: result.success,
            time_taken: start_time.elapsed().as_millis() as f64,
        });

        // Limit history size
        if self.tuning_history.len() > 100 {
            self.tuning_history.pop_front();
        }

        // Update baseline if successful
        if result.success {
            self.update_performance_baseline(method, &result.optimized_metrics)?;
        }

        Ok(result)
    }

    /// Optimize for speed (minimize execution time)
    fn optimize_for_speed(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        current_parameters: &HashMap<String, f64>,
    ) -> InterpolateResult<HashMap<String, f64>> {
        let mut optimized = current_parameters.clone();

        match method {
            InterpolationMethodType::CubicSpline => {
                // Use faster algorithms with looser tolerances
                optimized.insert("tolerance".to_string(), 1e-4);
                optimized.insert("max_iterations".to_string(), 50.0);
                optimized.insert("use_fast_algorithm".to_string(), 1.0);
            }
            InterpolationMethodType::BSpline => {
                // Reduce degree for faster computation
                if let Some(degree) = optimized.get_mut("degree") {
                    *degree = (*degree - 1.0).max(1.0);
                }
                optimized.insert("fast_evaluation".to_string(), 1.0);
            }
            InterpolationMethodType::RadialBasisFunction => {
                // Use approximate methods for large datasets
                if data_profile.size > 1000 {
                    optimized.insert("approximation_method".to_string(), 1.0);
                    optimized.insert("subset_size".to_string(), 1000.0);
                }
            }
            InterpolationMethodType::Kriging => {
                // Use simplified variogram models
                optimized.insert("variogram_model".to_string(), 0.0); // 0 = linear
                optimized.insert("nugget_optimization".to_string(), 0.0);
            }
            _ => {
                // Generic speed optimizations
                optimized.insert("parallel_execution".to_string(), 1.0);
                optimized.insert("cache_enabled".to_string(), 1.0);
            }
        }

        Ok(optimized)
    }

    /// Optimize for memory usage
    fn optimize_for_memory(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        current_parameters: &HashMap<String, f64>,
    ) -> InterpolateResult<HashMap<String, f64>> {
        let mut optimized = current_parameters.clone();

        match method {
            InterpolationMethodType::RadialBasisFunction => {
                // Use iterative methods for large problems
                optimized.insert("iterative_solver".to_string(), 1.0);
                optimized.insert("block_size".to_string(), 1000.0);
            }
            InterpolationMethodType::BSpline => {
                // Use sparse representations
                optimized.insert("sparse_representation".to_string(), 1.0);
                optimized.insert("compression_enabled".to_string(), 1.0);
            }
            InterpolationMethodType::Kriging => {
                // Reduce covariance matrix precision
                optimized.insert("reduced_precision".to_string(), 1.0);
                optimized.insert("matrix_approximation".to_string(), 1.0);
            }
            _ => {
                // Generic memory optimizations
                optimized.insert("memory_pool_enabled".to_string(), 1.0);
                optimized.insert("streaming_mode".to_string(), 1.0);

                // Adjust based on data size
                if data_profile.size > 10000 {
                    optimized.insert("chunk_size".to_string(), 1000.0);
                }
            }
        }

        Ok(optimized)
    }

    /// Balanced optimization (time and memory)
    fn optimize_balanced(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        current_parameters: &HashMap<String, f64>,
    ) -> InterpolateResult<HashMap<String, f64>> {
        // Blend speed and memory optimizations
        let speed_params = self.optimize_for_speed(method, data_profile, current_parameters)?;
        let memory_params = self.optimize_for_memory(method, data_profile, current_parameters)?;

        let mut balanced = current_parameters.clone();

        // Take average of speed and memory optimizations where applicable
        for (key, &speed_val) in &speed_params {
            if let Some(&memory_val) = memory_params.get(key) {
                let balanced_val = (speed_val + memory_val) / 2.0;
                balanced.insert(key.clone(), balanced_val);
            } else {
                balanced.insert(key.clone(), speed_val * 0.5); // Reduce impact
            }
        }

        for (key, &memory_val) in &memory_params {
            if !speed_params.contains_key(key) {
                balanced.insert(key.clone(), memory_val * 0.5); // Reduce impact
            }
        }

        Ok(balanced)
    }

    /// Adaptive optimization based on current system state
    fn optimize_adaptive(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        current_parameters: &HashMap<String, f64>,
    ) -> InterpolateResult<HashMap<String, f64>> {
        // Choose strategy based on current system resources
        if self.resource_monitor.memory_usage as f64 / self.resource_monitor.available_memory as f64
            > 0.8
        {
            // High memory usage - optimize for memory
            self.optimize_for_memory(method, data_profile, current_parameters)
        } else if self.resource_monitor.cpu_usage > 0.9 {
            // High CPU usage - optimize for speed
            self.optimize_for_speed(method, data_profile, current_parameters)
        } else {
            // Balanced approach
            self.optimize_balanced(method, data_profile, current_parameters)
        }
    }

    /// Custom weighted optimization
    fn optimize_custom(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        current_parameters: &HashMap<String, f64>,
        time_weight: f64,
        memory_weight: f64,
        accuracy_weight: f64,
    ) -> InterpolateResult<HashMap<String, f64>> {
        let speed_params = self.optimize_for_speed(method, data_profile, current_parameters)?;
        let memory_params = self.optimize_for_memory(method, data_profile, current_parameters)?;
        let accuracy_params = current_parameters.clone(); // Keep original for accuracy

        let mut custom = HashMap::new();

        // Weighted combination of different optimization strategies
        let total_weight = time_weight + memory_weight + accuracy_weight;
        let norm_time = time_weight / total_weight;
        let norm_memory = memory_weight / total_weight;
        let norm_accuracy = accuracy_weight / total_weight;

        // Combine parameters from all strategies
        let all_keys: std::collections::HashSet<_> = speed_params
            .keys()
            .chain(memory_params.keys())
            .chain(accuracy_params.keys())
            .collect();

        for key in all_keys {
            let speed_val = speed_params
                .get(key)
                .copied()
                .unwrap_or_else(|| current_parameters.get(key).copied().unwrap_or(0.0));
            let memory_val = memory_params
                .get(key)
                .copied()
                .unwrap_or_else(|| current_parameters.get(key).copied().unwrap_or(0.0));
            let accuracy_val = accuracy_params.get(key).copied().unwrap_or(0.0);

            let weighted_val =
                norm_time * speed_val + norm_memory * memory_val + norm_accuracy * accuracy_val;
            custom.insert(key.clone(), weighted_val);
        }

        Ok(custom)
    }

    /// Resource-aware optimization
    fn optimize_resource_aware(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        current_parameters: &HashMap<String, f64>,
        cpu_threshold: f64,
        memory_threshold: f64,
    ) -> InterpolateResult<HashMap<String, f64>> {
        let current_cpu = self.resource_monitor.cpu_usage;
        let current_memory_ratio = self.resource_monitor.memory_usage as f64
            / self.resource_monitor.available_memory as f64;

        if current_cpu > cpu_threshold && current_memory_ratio > memory_threshold {
            // Both resources constrained - aggressive optimization
            let mut optimized = current_parameters.clone();
            optimized.insert("aggressive_optimization".to_string(), 1.0);
            optimized.insert("resource_limit_mode".to_string(), 1.0);
            Ok(optimized)
        } else if current_cpu > cpu_threshold {
            // CPU constrained - optimize for speed
            self.optimize_for_speed(method, data_profile, current_parameters)
        } else if current_memory_ratio > memory_threshold {
            // Memory constrained - optimize for memory
            self.optimize_for_memory(method, data_profile, current_parameters)
        } else {
            // Resources available - balanced approach
            self.optimize_balanced(method, data_profile, current_parameters)
        }
    }

    /// Check if current metrics meet performance targets
    fn meets_performance_targets(&self, metrics: &PerformanceMetrics) -> bool {
        if let Some(max_time) = self.targets.max_execution_time {
            if metrics.execution_time > max_time {
                return false;
            }
        }

        if let Some(max_memory) = self.targets.max_memory_usage {
            if metrics.memory_usage > max_memory {
                return false;
            }
        }

        if let Some(min_throughput) = self.targets.min_throughput {
            if metrics.throughput < min_throughput {
                return false;
            }
        }

        if let Some(max_latency) = self.targets.max_latency {
            if metrics.execution_time > max_latency {
                return false;
            }
        }

        true
    }

    /// Calculate parameter adjustments
    fn calculate_parameter_adjustments(
        &self,
        original: &HashMap<String, f64>,
        optimized: &HashMap<String, f64>,
    ) -> HashMap<String, ParameterAdjustment> {
        let mut adjustments = HashMap::new();

        for (key, &opt_val) in optimized {
            if let Some(&orig_val) = original.get(key) {
                if (opt_val - orig_val).abs() > 1e-10 {
                    let adjustment_type = if opt_val > orig_val {
                        AdjustmentType::Increase
                    } else {
                        AdjustmentType::Decrease
                    };

                    let performance_impact = (opt_val - orig_val).abs() / orig_val.max(1e-10);

                    adjustments.insert(
                        key.clone(),
                        ParameterAdjustment {
                            original_value: orig_val,
                            new_value: opt_val,
                            adjustment_type,
                            performance_impact,
                        },
                    );
                }
            }
        }

        adjustments
    }

    /// Predict optimized performance metrics
    fn predict_optimized_metrics(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        optimized_params: &HashMap<String, f64>,
        current_metrics: &PerformanceMetrics,
    ) -> InterpolateResult<PerformanceMetrics> {
        // Simplified prediction model
        let mut predicted = current_metrics.clone();

        // Apply parameter-based adjustments
        if optimized_params.contains_key("fast_evaluation") {
            predicted.execution_time *= 0.7; // 30% faster
        }

        if optimized_params.contains_key("parallel_execution") {
            predicted.execution_time *= 0.5; // 50% faster with parallelism
            predicted.cpu_utilization *= 1.5; // Higher CPU usage
        }

        if optimized_params.contains_key("sparse_representation") {
            predicted.memory_usage = (predicted.memory_usage as f64 * 0.6) as usize;
            // 40% memory reduction
        }

        if optimized_params.contains_key("approximation_method") {
            predicted.execution_time *= 0.4; // Much faster
            predicted.accuracy *= 0.95; // Slight accuracy loss
        }

        // Adjust based on method characteristics
        match method {
            InterpolationMethodType::RadialBasisFunction => {
                if data_profile.size > 5000 {
                    predicted.memory_usage = (predicted.memory_usage as f64 * 1.2) as usize;
                }
            }
            InterpolationMethodType::Kriging => {
                predicted.accuracy *= 1.05; // Kriging typically more accurate
            }
            _ => {}
        }

        // Update throughput based on execution time
        if predicted.execution_time > 0.0 {
            predicted.throughput = 1_000_000.0 / predicted.execution_time; // Operations per second
        }

        Ok(predicted)
    }

    /// Calculate improvement score
    fn calculate_improvement_score(
        &self,
        original: &PerformanceMetrics,
        optimized: &PerformanceMetrics,
    ) -> f64 {
        let time_improvement = if original.execution_time > 0.0 {
            (original.execution_time - optimized.execution_time) / original.execution_time
        } else {
            0.0
        };

        let memory_improvement = if original.memory_usage > 0 {
            (original.memory_usage as f64 - optimized.memory_usage as f64)
                / original.memory_usage as f64
        } else {
            0.0
        };

        let accuracy_change =
            (optimized.accuracy - original.accuracy) / original.accuracy.max(1e-10);

        // Weighted score (prioritize time, then memory, then accuracy)
        0.5 * time_improvement + 0.3 * memory_improvement + 0.2 * accuracy_change
    }

    /// Update performance baseline for a method
    fn update_performance_baseline(
        &mut self,
        method: InterpolationMethodType,
        metrics: &PerformanceMetrics,
    ) -> InterpolateResult<()> {
        let baseline = self
            .performance_baselines
            .entry(method)
            .or_insert(PerformanceBaseline {
                baseline_time: metrics.execution_time,
                baseline_memory: metrics.memory_usage,
                baseline_accuracy: metrics.accuracy,
                sample_size: 0,
                last_update: Instant::now(),
            });

        // Exponential moving average update
        let alpha = 0.1;
        baseline.baseline_time =
            (1.0 - alpha) * baseline.baseline_time + alpha * metrics.execution_time;
        baseline.baseline_memory = ((1.0 - alpha) * baseline.baseline_memory as f64
            + alpha * metrics.memory_usage as f64) as usize;
        baseline.baseline_accuracy =
            (1.0 - alpha) * baseline.baseline_accuracy + alpha * metrics.accuracy;
        baseline.sample_size += 1;
        baseline.last_update = Instant::now();

        Ok(())
    }

    /// Get performance targets
    pub fn get_targets(&self) -> &PerformanceTargets {
        &self.targets
    }

    /// Get tuning history
    pub fn get_tuning_history(&self) -> &VecDeque<TuningResult> {
        &self.tuning_history
    }

    /// Get current resource usage
    pub fn get_resource_usage(&self) -> ResourceUsage {
        ResourceUsage {
            cpu_usage: self.resource_monitor.cpu_usage,
            memory_usage: self.resource_monitor.memory_usage,
            memory_usage_ratio: self.resource_monitor.memory_usage as f64
                / self.resource_monitor.available_memory as f64,
            load_average: self.resource_monitor.load_average,
        }
    }

    /// Get performance baseline for a method
    pub fn get_performance_baseline(
        &self,
        method: InterpolationMethodType,
    ) -> Option<&PerformanceBaseline> {
        self.performance_baselines.get(&method)
    }
}

/// Current resource usage information
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// CPU usage (0-1)
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Memory usage ratio (0-1)
    pub memory_usage_ratio: f64,
    /// System load average
    pub load_average: f64,
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new() -> InterpolateResult<Self> {
        Ok(Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            available_memory: 8_000_000_000, // 8GB default
            load_average: 0.0,
            monitoring_interval: 1000, // 1 second
        })
    }

    /// Update resource metrics
    pub fn update_metrics(&mut self) -> InterpolateResult<()> {
        // Simplified resource monitoring
        // In a real implementation, this would use system APIs

        // Simulate CPU usage (would use actual system monitoring)
        self.cpu_usage = 0.3; // 30% usage

        // Simulate memory usage (would use actual system monitoring)
        self.memory_usage = 2_000_000_000; // 2GB usage

        // Simulate load average
        self.load_average = 1.0;

        Ok(())
    }

    /// Get current CPU usage
    pub fn get_cpu_usage(&self) -> f64 {
        self.cpu_usage
    }

    /// Get current memory usage
    pub fn get_memory_usage(&self) -> usize {
        self.memory_usage
    }

    /// Get memory usage ratio
    pub fn get_memory_ratio(&self) -> f64 {
        self.memory_usage as f64 / self.available_memory as f64
    }
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            max_execution_time: None,
            max_memory_usage: None,
            min_throughput: None,
            max_latency: None,
            target_cpu_utilization: Some(0.8), // 80% CPU utilization
            memory_efficiency_target: Some(1000.0), // 1000 ops per MB
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_tuning_system_creation() {
        let system: PerformanceTuningSystem<f64> = PerformanceTuningSystem::new().unwrap();
        assert!(matches!(
            system.strategy,
            PerformanceTuningStrategy::Balanced
        ));
        assert!(system.tuning_history.is_empty());
    }

    #[test]
    fn test_performance_targets_default() {
        let targets = PerformanceTargets::default();
        assert!(targets.max_execution_time.is_none());
        assert_eq!(targets.target_cpu_utilization, Some(0.8));
    }

    #[test]
    fn test_resource_monitor_creation() {
        let monitor = ResourceMonitor::new().unwrap();
        assert_eq!(monitor.monitoring_interval, 1000);
        assert_eq!(monitor.available_memory, 8_000_000_000);
    }

    #[test]
    fn test_improvement_score_calculation() {
        let system: PerformanceTuningSystem<f64> = PerformanceTuningSystem::new().unwrap();

        let original = PerformanceMetrics {
            execution_time: 1000.0,
            memory_usage: 1000000,
            accuracy: 0.9,
            throughput: 1000.0,
            cpu_utilization: 0.5,
            cache_hit_ratio: 0.8,
        };

        let optimized = PerformanceMetrics {
            execution_time: 500.0, // 50% faster
            memory_usage: 800000,  // 20% less memory
            accuracy: 0.9,
            throughput: 2000.0,
            cpu_utilization: 0.6,
            cache_hit_ratio: 0.9,
        };

        let score = system.calculate_improvement_score(&original, &optimized);
        assert!(score > 0.0); // Should show improvement
        assert!(score > 0.3); // Should be significant improvement
    }
}
