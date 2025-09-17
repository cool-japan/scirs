//! Dynamic batch size control for memory-efficient training
//!
//! This module provides dynamic batch sizing capabilities that adapt to
//! memory pressure and performance characteristics to optimize training
//! efficiency while staying within memory constraints.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::error::{OptimError, Result};

/// Dynamic batch size controller
#[derive(Debug)]
pub struct DynamicBatchController {
    /// Current batch size
    current_batch_size: usize,

    /// Minimum allowed batch size
    min_batch_size: usize,

    /// Maximum allowed batch size
    max_batch_size: usize,

    /// History of batch size changes
    batch_history: VecDeque<BatchSizeEvent>,

    /// Estimated memory usage per sample (bytes)
    memory_per_sample: usize,

    /// Performance metrics for different batch sizes
    performance_metrics: HashMap<usize, PerformanceMetrics>,

    /// Adaptation strategy
    adaptation_strategy: BatchAdaptationStrategy,

    /// Memory pressure threshold for triggering reductions
    pressure_threshold: f32,

    /// Current performance baseline
    current_performance: Option<PerformanceMetrics>,

    /// Batch size adjustment statistics
    stats: BatchControlStats,

    /// Automatic scaling parameters
    scaling_params: ScalingParameters,
}

impl DynamicBatchController {
    /// Create a new dynamic batch controller
    pub fn new(
        initial_batch_size: usize,
        min_batch_size: usize,
        max_batch_size: usize,
    ) -> Self {
        Self {
            current_batch_size: initial_batch_size,
            min_batch_size,
            max_batch_size,
            batch_history: VecDeque::new(),
            memory_per_sample: 0,
            performance_metrics: HashMap::new(),
            adaptation_strategy: BatchAdaptationStrategy::Balanced,
            pressure_threshold: 0.8,
            current_performance: None,
            stats: BatchControlStats::default(),
            scaling_params: ScalingParameters::default(),
        }
    }

    /// Configure batch controller parameters
    pub fn configure(
        &mut self,
        strategy: BatchAdaptationStrategy,
        pressure_threshold: f32,
        memory_per_sample: usize,
        scaling_params: ScalingParameters,
    ) {
        self.adaptation_strategy = strategy;
        self.pressure_threshold = pressure_threshold;
        self.memory_per_sample = memory_per_sample;
        self.scaling_params = scaling_params;
    }

    /// Get current batch size
    pub fn get_batch_size(&self) -> usize {
        self.current_batch_size
    }

    /// Adjust batch size based on memory pressure
    pub fn adjust_for_memory_pressure(&mut self, memory_pressure: f32) -> Result<Option<usize>> {
        if memory_pressure <= self.pressure_threshold {
            return Ok(None);
        }

        let old_size = self.current_batch_size;
        let reduction_factor = self.calculate_reduction_factor(memory_pressure);
        let new_size = ((old_size as f32 * reduction_factor) as usize).max(self.min_batch_size);

        if new_size != old_size {
            self.set_batch_size(new_size, BatchChangeReason::MemoryPressure, memory_pressure)?;
            Ok(Some(new_size))
        } else {
            Ok(None)
        }
    }

    /// Optimize batch size for performance
    pub fn optimize_for_performance(&mut self, current_metrics: PerformanceMetrics) -> Result<Option<usize>> {
        // Record current performance
        self.performance_metrics.insert(self.current_batch_size, current_metrics.clone());
        self.current_performance = Some(current_metrics.clone());

        // Find optimal batch size based on strategy
        let optimal_size = self.find_optimal_batch_size()?;

        if optimal_size != self.current_batch_size {
            self.set_batch_size(optimal_size, BatchChangeReason::PerformanceOptimization, 0.0)?;
            Ok(Some(optimal_size))
        } else {
            Ok(None)
        }
    }

    /// Attempt to scale up batch size when memory allows
    pub fn try_scale_up(&mut self, memory_pressure: f32) -> Result<Option<usize>> {
        if memory_pressure > self.scaling_params.scale_up_threshold {
            return Ok(None);
        }

        if self.current_batch_size >= self.max_batch_size {
            return Ok(None);
        }

        let old_size = self.current_batch_size;
        let scale_factor = self.scaling_params.scale_up_factor;
        let new_size = ((old_size as f32 * scale_factor) as usize).min(self.max_batch_size);

        if new_size > old_size {
            self.set_batch_size(new_size, BatchChangeReason::AutoScale, memory_pressure)?;
            Ok(Some(new_size))
        } else {
            Ok(None)
        }
    }

    /// Set batch size manually
    pub fn set_batch_size_manual(&mut self, batch_size: usize) -> Result<()> {
        if batch_size < self.min_batch_size || batch_size > self.max_batch_size {
            return Err(OptimError::InvalidParameter(
                format!("Batch size {} outside allowed range [{}, {}]",
                       batch_size, self.min_batch_size, self.max_batch_size)
            ));
        }

        self.set_batch_size(batch_size, BatchChangeReason::Manual, 0.0)
    }

    /// Get batch size recommendations
    pub fn get_recommendations(&self, memory_pressure: f32) -> BatchSizeRecommendations {
        let current_size = self.current_batch_size;

        let conservative_size = if memory_pressure > 0.9 {
            self.min_batch_size
        } else {
            ((current_size as f32 * 0.8) as usize).max(self.min_batch_size)
        };

        let aggressive_size = if memory_pressure < 0.6 {
            ((current_size as f32 * 1.5) as usize).min(self.max_batch_size)
        } else {
            current_size
        };

        let optimal_size = self.find_optimal_batch_size().unwrap_or(current_size);

        BatchSizeRecommendations {
            current: current_size,
            conservative: conservative_size,
            aggressive: aggressive_size,
            optimal: optimal_size,
            memory_pressure,
        }
    }

    /// Get batch control statistics
    pub fn get_statistics(&self) -> &BatchControlStats {
        &self.stats
    }

    /// Get performance history
    pub fn get_performance_history(&self) -> &HashMap<usize, PerformanceMetrics> {
        &self.performance_metrics
    }

    /// Reset performance metrics
    pub fn reset_performance_metrics(&mut self) {
        self.performance_metrics.clear();
        self.current_performance = None;
    }

    // Private helper methods

    fn set_batch_size(
        &mut self,
        new_size: usize,
        reason: BatchChangeReason,
        memory_pressure: f32,
    ) -> Result<()> {
        let old_size = self.current_batch_size;

        // Create batch size event
        let event = BatchSizeEvent {
            timestamp: Instant::now(),
            old_size,
            new_size,
            reason: reason.clone(),
            memory_pressure,
        };

        // Update state
        self.current_batch_size = new_size;
        self.batch_history.push_back(event);

        // Limit history size
        if self.batch_history.len() > 1000 {
            self.batch_history.pop_front();
        }

        // Update statistics
        match reason {
            BatchChangeReason::MemoryPressure => self.stats.pressure_triggered_changes += 1,
            BatchChangeReason::PerformanceOptimization => self.stats.performance_optimizations += 1,
            BatchChangeReason::AutoScale => self.stats.auto_scale_events += 1,
            BatchChangeReason::Manual => self.stats.manual_changes += 1,
            BatchChangeReason::ErrorRecovery => self.stats.error_recoveries += 1,
        }

        if new_size > old_size {
            self.stats.total_increases += 1;
        } else if new_size < old_size {
            self.stats.total_decreases += 1;
        }

        self.stats.total_changes += 1;

        Ok(())
    }

    fn calculate_reduction_factor(&self, memory_pressure: f32) -> f32 {
        match self.adaptation_strategy {
            BatchAdaptationStrategy::Conservative => {
                0.5 + 0.4 * (1.0 - memory_pressure).max(0.0)
            }
            BatchAdaptationStrategy::Aggressive => {
                0.3 + 0.6 * (1.0 - memory_pressure).max(0.0)
            }
            BatchAdaptationStrategy::Balanced => {
                0.4 + 0.5 * (1.0 - memory_pressure).max(0.0)
            }
            BatchAdaptationStrategy::Memory => {
                0.2 + 0.3 * (1.0 - memory_pressure).max(0.0)
            }
            BatchAdaptationStrategy::Performance => {
                0.7 + 0.25 * (1.0 - memory_pressure).max(0.0)
            }
        }
    }

    fn find_optimal_batch_size(&self) -> Result<usize> {
        if self.performance_metrics.is_empty() {
            return Ok(self.current_batch_size);
        }

        let mut best_size = self.current_batch_size;
        let mut best_score = 0.0;

        for (&batch_size, metrics) in &self.performance_metrics {
            let score = self.calculate_performance_score(metrics);
            if score > best_score {
                best_score = score;
                best_size = batch_size;
            }
        }

        Ok(best_size)
    }

    fn calculate_performance_score(&self, metrics: &PerformanceMetrics) -> f32 {
        match self.adaptation_strategy {
            BatchAdaptationStrategy::Conservative => {
                0.3 * metrics.throughput + 0.4 * metrics.stability + 0.3 * metrics.memory_efficiency
            }
            BatchAdaptationStrategy::Aggressive => {
                0.7 * metrics.throughput + 0.1 * metrics.stability + 0.2 * metrics.memory_efficiency
            }
            BatchAdaptationStrategy::Balanced => {
                0.4 * metrics.throughput + 0.3 * metrics.stability + 0.3 * metrics.memory_efficiency
            }
            BatchAdaptationStrategy::Performance => {
                0.8 * metrics.throughput + 0.1 * metrics.stability + 0.1 * metrics.memory_efficiency
            }
            BatchAdaptationStrategy::Memory => {
                0.2 * metrics.throughput + 0.3 * metrics.stability + 0.5 * metrics.memory_efficiency
            }
        }
    }
}

/// Batch size change event
#[derive(Debug, Clone)]
pub struct BatchSizeEvent {
    /// When the change occurred
    pub timestamp: Instant,

    /// Previous batch size
    pub old_size: usize,

    /// New batch size
    pub new_size: usize,

    /// Reason for the change
    pub reason: BatchChangeReason,

    /// Memory pressure level at time of change
    pub memory_pressure: f32,
}

impl BatchSizeEvent {
    /// Calculate the change magnitude
    pub fn change_magnitude(&self) -> f32 {
        (self.new_size as f32 - self.old_size as f32) / self.old_size as f32
    }

    /// Check if this was an increase
    pub fn is_increase(&self) -> bool {
        self.new_size > self.old_size
    }

    /// Check if this was a decrease
    pub fn is_decrease(&self) -> bool {
        self.new_size < self.old_size
    }
}

/// Reasons for batch size changes
#[derive(Debug, Clone)]
pub enum BatchChangeReason {
    /// Memory pressure triggered automatic reduction
    MemoryPressure,

    /// Performance optimization based on metrics
    PerformanceOptimization,

    /// Manual adjustment by user
    Manual,

    /// Automatic scaling up when memory allows
    AutoScale,

    /// Recovery from out-of-memory or other errors
    ErrorRecovery,
}

/// Performance metrics for different batch sizes
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Training throughput (samples per second)
    pub throughput: f32,

    /// Memory efficiency (samples per MB of GPU memory)
    pub memory_efficiency: f32,

    /// Computational efficiency (achieved vs theoretical FLOPS)
    pub compute_efficiency: f32,

    /// Training stability (inverse of loss variance)
    pub stability: f32,

    /// Overall performance score
    pub overall_score: f32,

    /// Number of measurements
    pub sample_count: usize,

    /// Timestamp of measurement
    pub timestamp: Instant,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new(
        throughput: f32,
        memory_efficiency: f32,
        compute_efficiency: f32,
        stability: f32,
    ) -> Self {
        let overall_score = 0.4 * throughput + 0.3 * memory_efficiency + 0.2 * compute_efficiency + 0.1 * stability;

        Self {
            throughput,
            memory_efficiency,
            compute_efficiency,
            stability,
            overall_score,
            sample_count: 1,
            timestamp: Instant::now(),
        }
    }

    /// Update metrics with new measurement
    pub fn update(&mut self, other: &PerformanceMetrics) {
        let total_samples = self.sample_count + other.sample_count;
        let weight_self = self.sample_count as f32 / total_samples as f32;
        let weight_other = other.sample_count as f32 / total_samples as f32;

        self.throughput = self.throughput * weight_self + other.throughput * weight_other;
        self.memory_efficiency = self.memory_efficiency * weight_self + other.memory_efficiency * weight_other;
        self.compute_efficiency = self.compute_efficiency * weight_self + other.compute_efficiency * weight_other;
        self.stability = self.stability * weight_self + other.stability * weight_other;

        self.overall_score = 0.4 * self.throughput + 0.3 * self.memory_efficiency +
                           0.2 * self.compute_efficiency + 0.1 * self.stability;

        self.sample_count = total_samples;
        self.timestamp = Instant::now();
    }
}

/// Batch adaptation strategies
#[derive(Debug, Clone, Copy)]
pub enum BatchAdaptationStrategy {
    /// Conservative: Prefer stability and memory safety
    Conservative,

    /// Aggressive: Maximize throughput
    Aggressive,

    /// Balanced: Compromise between performance and stability
    Balanced,

    /// Performance-first: Optimize for maximum training speed
    Performance,

    /// Memory-first: Optimize for memory efficiency
    Memory,
}

/// Scaling parameters for automatic batch size adjustment
#[derive(Debug, Clone)]
pub struct ScalingParameters {
    /// Scale up factor when memory allows
    pub scale_up_factor: f32,

    /// Scale down factor under memory pressure
    pub scale_down_factor: f32,

    /// Memory pressure threshold for scaling up
    pub scale_up_threshold: f32,

    /// Memory pressure threshold for scaling down
    pub scale_down_threshold: f32,

    /// Minimum time between scale operations
    pub min_scale_interval: Duration,

    /// Maximum number of consecutive scale operations
    pub max_consecutive_scales: usize,
}

impl Default for ScalingParameters {
    fn default() -> Self {
        Self {
            scale_up_factor: 1.2,
            scale_down_factor: 0.8,
            scale_up_threshold: 0.6,
            scale_down_threshold: 0.8,
            min_scale_interval: Duration::from_secs(30),
            max_consecutive_scales: 3,
        }
    }
}

/// Batch size recommendations
#[derive(Debug, Clone)]
pub struct BatchSizeRecommendations {
    /// Current batch size
    pub current: usize,

    /// Conservative recommendation (safer memory usage)
    pub conservative: usize,

    /// Aggressive recommendation (higher performance)
    pub aggressive: usize,

    /// Optimal recommendation based on performance history
    pub optimal: usize,

    /// Current memory pressure level
    pub memory_pressure: f32,
}

impl BatchSizeRecommendations {
    /// Get recommendation based on strategy
    pub fn get_for_strategy(&self, strategy: BatchAdaptationStrategy) -> usize {
        match strategy {
            BatchAdaptationStrategy::Conservative | BatchAdaptationStrategy::Memory => self.conservative,
            BatchAdaptationStrategy::Aggressive | BatchAdaptationStrategy::Performance => self.aggressive,
            BatchAdaptationStrategy::Balanced => self.optimal,
        }
    }
}

/// Batch control statistics
#[derive(Debug, Clone, Default)]
pub struct BatchControlStats {
    /// Total number of batch size changes
    pub total_changes: usize,

    /// Number of increases
    pub total_increases: usize,

    /// Number of decreases
    pub total_decreases: usize,

    /// Changes triggered by memory pressure
    pub pressure_triggered_changes: usize,

    /// Changes for performance optimization
    pub performance_optimizations: usize,

    /// Automatic scaling events
    pub auto_scale_events: usize,

    /// Manual changes
    pub manual_changes: usize,

    /// Error recovery changes
    pub error_recoveries: usize,

    /// Average batch size over time
    pub average_batch_size: f32,

    /// Peak batch size achieved
    pub peak_batch_size: usize,

    /// Minimum batch size used
    pub min_batch_size_used: usize,
}

impl BatchControlStats {
    /// Calculate change frequency (changes per hour)
    pub fn change_frequency(&self, uptime: Duration) -> f32 {
        if uptime.as_secs() > 0 {
            self.total_changes as f32 / uptime.as_secs_f32() * 3600.0
        } else {
            0.0
        }
    }

    /// Calculate adaptation efficiency
    pub fn adaptation_efficiency(&self) -> f32 {
        let total_adaptive = self.pressure_triggered_changes + self.performance_optimizations + self.auto_scale_events;
        if self.total_changes > 0 {
            total_adaptive as f32 / self.total_changes as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_batch_controller_creation() {
        let controller = DynamicBatchController::new(32, 8, 128);
        assert_eq!(controller.get_batch_size(), 32);
        assert_eq!(controller.min_batch_size, 8);
        assert_eq!(controller.max_batch_size, 128);
    }

    #[test]
    fn test_memory_pressure_adjustment() {
        let mut controller = DynamicBatchController::new(64, 16, 128);

        // Low pressure should not trigger adjustment
        let result = controller.adjust_for_memory_pressure(0.5).unwrap();
        assert!(result.is_none());

        // High pressure should trigger reduction
        let result = controller.adjust_for_memory_pressure(0.9).unwrap();
        assert!(result.is_some());
        assert!(result.unwrap() < 64);
    }

    #[test]
    fn test_performance_optimization() {
        let mut controller = DynamicBatchController::new(32, 8, 128);

        let metrics = PerformanceMetrics::new(100.0, 0.8, 0.9, 0.7);
        controller.optimize_for_performance(metrics).unwrap();

        // Should record the metrics
        assert!(!controller.performance_metrics.is_empty());
    }

    #[test]
    fn test_batch_size_recommendations() {
        let controller = DynamicBatchController::new(64, 16, 256);

        let recommendations = controller.get_recommendations(0.7);
        assert_eq!(recommendations.current, 64);
        assert!(recommendations.conservative <= recommendations.current);
        assert!(recommendations.aggressive >= recommendations.current);
    }

    #[test]
    fn test_scaling_parameters() {
        let params = ScalingParameters::default();
        assert!(params.scale_up_factor > 1.0);
        assert!(params.scale_down_factor < 1.0);
        assert!(params.scale_up_threshold < params.scale_down_threshold);
    }

    #[test]
    fn test_performance_metrics_update() {
        let mut metrics1 = PerformanceMetrics::new(100.0, 0.8, 0.9, 0.7);
        let metrics2 = PerformanceMetrics::new(120.0, 0.7, 0.8, 0.8);

        metrics1.update(&metrics2);

        // Should be weighted average
        assert!(metrics1.throughput > 100.0 && metrics1.throughput < 120.0);
        assert_eq!(metrics1.sample_count, 2);
    }

    #[test]
    fn test_batch_size_event() {
        let event = BatchSizeEvent {
            timestamp: Instant::now(),
            old_size: 32,
            new_size: 64,
            reason: BatchChangeReason::PerformanceOptimization,
            memory_pressure: 0.5,
        };

        assert!(event.is_increase());
        assert!(!event.is_decrease());
        assert_eq!(event.change_magnitude(), 1.0); // 100% increase
    }

    #[test]
    fn test_adaptation_strategies() {
        let controller = DynamicBatchController::new(64, 16, 256);

        let recommendations = controller.get_recommendations(0.5);

        assert_eq!(
            recommendations.get_for_strategy(BatchAdaptationStrategy::Conservative),
            recommendations.conservative
        );

        assert_eq!(
            recommendations.get_for_strategy(BatchAdaptationStrategy::Aggressive),
            recommendations.aggressive
        );

        assert_eq!(
            recommendations.get_for_strategy(BatchAdaptationStrategy::Balanced),
            recommendations.optimal
        );
    }

    #[test]
    fn test_batch_control_stats() {
        let mut stats = BatchControlStats::default();
        stats.total_changes = 10;
        stats.pressure_triggered_changes = 3;
        stats.performance_optimizations = 4;
        stats.auto_scale_events = 2;
        stats.manual_changes = 1;

        assert_eq!(stats.adaptation_efficiency(), 0.9); // 9/10 adaptive changes
    }
}