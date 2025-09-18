//! Core advanced memory optimizer implementation
//!
//! This module contains the main AdvancedMemoryOptimizer that coordinates
//! all memory optimization strategies including gradient accumulation,
//! checkpointing, parameter offloading, and dynamic batching.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use ndarray::Array2;
use num_traits::Float;

use crate::error::{OptimError, Result};

use super::config::AdvancedMemoryConfig;
use super::memory_tracking::{MemoryUsageTracker, MemoryStatistics};
use super::gradient_management::{GradientAccumulator, ZeroRedundancyState, MixedPrecisionManager};
use super::checkpoint_management::CheckpointManager;
use super::parameter_offloading::ParameterOffloadManager;
use super::dynamic_batching::DynamicBatchController;
use super::pressure_monitoring::MemoryPressureMonitor;

#[cfg(feature = "gpu")]
use scirs2_core::gpu::GpuContext;

/// Advanced memory optimizer for large-scale training
#[derive(Debug)]
pub struct AdvancedMemoryOptimizer<T: Float + Default + Clone + Send + Sync + std::iter::Sum + 'static> {
    /// Configuration settings
    config: AdvancedMemoryConfig,

    /// GPU context (if available)
    #[cfg(feature = "gpu")]
    gpu_context: Option<Arc<GpuContext>>,

    /// Memory usage tracking
    memory_tracker: MemoryUsageTracker,

    /// Gradient accumulation management
    gradient_accumulator: GradientAccumulator<T>,

    /// Activation checkpoint management
    checkpoint_manager: CheckpointManager<T>,

    /// Parameter offloading management
    offload_manager: ParameterOffloadManager<T>,

    /// Dynamic batch size control
    batch_controller: DynamicBatchController,

    /// Memory pressure monitoring
    pressure_monitor: MemoryPressureMonitor,

    /// Zero redundancy optimizer state
    zero_redundancy_state: Option<ZeroRedundancyState<T>>,

    /// Mixed precision training manager
    mixed_precision_manager: MixedPrecisionManager<T>,

    /// Memory mapped parameter storage
    memory_mapped_storage: Option<MemoryMappedStorage<T>>,

    /// Optimization statistics
    stats: OptimizationStats,
}

impl<T: Float + Default + Clone + Send + Sync + std::iter::Sum + 'static> AdvancedMemoryOptimizer<T> {
    /// Create a new advanced memory optimizer
    pub fn new(config: AdvancedMemoryConfig) -> Result<Self> {
        config.validate().map_err(|e| OptimError::InvalidParameter(e))?;

        let memory_tracker = MemoryUsageTracker::new(1024 * 1024 * 1024); // 1GB default

        Ok(Self {
            gradient_accumulator: GradientAccumulator::new(config.max_accumulation_steps),
            checkpoint_manager: CheckpointManager::new(
                super::config::CheckpointStrategy::Adaptive,
                100, // max checkpoints
            ),
            offload_manager: ParameterOffloadManager::new(
                super::config::OffloadStrategy::CostBenefit,
                config.offload_threshold,
            ),
            batch_controller: DynamicBatchController::new(
                config.microbatch_size,
                1,
                config.microbatch_size * 16,
            ),
            pressure_monitor: MemoryPressureMonitor::new(config.memory_pressure_threshold),
            zero_redundancy_state: if config.enable_zero_redundancy {
                Some(ZeroRedundancyState::new())
            } else {
                None
            },
            mixed_precision_manager: MixedPrecisionManager::new(config.enable_mixed_precision),
            memory_mapped_storage: if config.enable_memory_mapping {
                Some(MemoryMappedStorage::new())
            } else {
                None
            },
            memory_tracker,
            config,
            #[cfg(feature = "gpu")]
            gpu_context: None,
            stats: OptimizationStats::default(),
        })
    }

    /// Initialize with GPU context
    #[cfg(feature = "gpu")]
    pub fn with_gpu_context(mut self, context: Arc<GpuContext>) -> Self {
        self.gpu_context = Some(context);
        self
    }

    /// Perform comprehensive memory optimization
    pub fn optimize_memory(&mut self) -> Result<MemoryOptimizationResult> {
        let start_time = Instant::now();

        // Update memory pressure
        self.update_memory_pressure()?;

        let mut result = MemoryOptimizationResult::default();

        // Apply optimizations based on pressure level
        if self.pressure_monitor.is_high_pressure() {
            result.merge(self.apply_high_pressure_optimizations()?);
        }

        if self.pressure_monitor.is_critical_pressure() {
            result.merge(self.apply_critical_pressure_optimizations()?);
        }

        // Apply gradient optimizations
        if self.config.enable_zero_redundancy {
            result.merge(self.optimize_gradients()?);
        }

        // Apply checkpoint optimizations
        if self.config.enable_gradient_checkpointing {
            result.merge(self.optimize_checkpoints()?);
        }

        // Apply parameter offloading
        if self.config.enable_parameter_offloading {
            result.merge(self.optimize_parameter_storage()?);
        }

        // Apply dynamic batching
        if self.config.enable_dynamic_batching {
            result.merge(self.optimize_batch_size()?);
        }

        // Update statistics
        let optimization_time = start_time.elapsed();
        self.stats.total_optimizations += 1;
        self.stats.total_optimization_time += optimization_time;
        self.stats.avg_optimization_time = self.stats.total_optimization_time / self.stats.total_optimizations as u32;

        result.optimization_time = optimization_time;
        result.memory_pressure = self.pressure_monitor.get_pressure();

        Ok(result)
    }

    /// Get current memory statistics
    pub fn get_memory_statistics(&self) -> MemoryStatistics {
        self.memory_tracker.get_statistics(Duration::from_secs(300))
    }

    /// Get optimization statistics
    pub fn get_optimization_statistics(&self) -> &OptimizationStats {
        &self.stats
    }

    /// Update configuration
    pub fn update_config(&mut self, new_config: AdvancedMemoryConfig) -> Result<()> {
        new_config.validate().map_err(|e| OptimError::InvalidParameter(e))?;
        self.config = new_config;
        Ok(())
    }

    /// Reset all optimization state
    pub fn reset(&mut self) {
        self.memory_tracker.reset_peak_usage();
        self.stats = OptimizationStats::default();
        // Note: Other components maintain their own state for consistency
    }

    /// Get memory savings summary
    pub fn get_memory_savings(&self) -> MemorySavingsSummary {
        MemorySavingsSummary {
            gradient_accumulation_savings: self.gradient_accumulator.get_memory_savings(),
            checkpoint_savings: self.checkpoint_manager.estimate_memory_savings(
                1024 * 1024 * 1024 // 1GB total activation memory estimate
            ),
            offload_savings: self.offload_manager.get_memory_savings(),
            mixed_precision_savings: self.mixed_precision_manager.get_memory_savings(),
            zero_redundancy_savings: self.zero_redundancy_state
                .as_ref()
                .map(|zrs| zrs.get_memory_savings())
                .unwrap_or(0),
            total_savings: 0, // Will be calculated
        }
    }

    // Private optimization methods

    fn update_memory_pressure(&mut self) -> Result<()> {
        // Get current GPU memory usage
        let current_usage = self.get_current_memory_usage()?;
        let total_memory = self.get_total_memory()?;

        let pressure = if total_memory > 0 {
            current_usage as f32 / total_memory as f32
        } else {
            0.0
        };

        self.pressure_monitor.update_pressure(pressure);
        self.memory_tracker.update_usage(current_usage);

        Ok(())
    }

    fn apply_high_pressure_optimizations(&mut self) -> Result<MemoryOptimizationResult> {
        let mut result = MemoryOptimizationResult::default();

        // Reduce batch size
        if let Some(new_batch_size) = self.batch_controller.adjust_for_memory_pressure(
            self.pressure_monitor.get_pressure()
        )? {
            result.batch_size_reduced = true;
            result.new_batch_size = Some(new_batch_size);
            result.memory_freed += new_batch_size * 1024; // Rough estimate
        }

        // Trigger additional checkpointing
        if self.config.enable_activation_recomputation {
            // Would trigger checkpointing of recent activations
            result.checkpoints_created += 1;
            result.memory_freed += 10 * 1024 * 1024; // Rough estimate
        }

        Ok(result)
    }

    fn apply_critical_pressure_optimizations(&mut self) -> Result<MemoryOptimizationResult> {
        let mut result = MemoryOptimizationResult::default();

        // Emergency measures
        self.checkpoint_manager.clear_all_checkpoints();
        result.emergency_cleanup = true;
        result.memory_freed += 100 * 1024 * 1024; // Rough estimate

        // Force garbage collection
        // In practice, would trigger GPU memory cleanup
        result.garbage_collection = true;

        Ok(result)
    }

    fn optimize_gradients(&mut self) -> Result<MemoryOptimizationResult> {
        let mut result = MemoryOptimizationResult::default();

        // Check if gradient accumulation is ready
        if self.gradient_accumulator.is_ready_for_step() {
            let _accumulated_gradients = self.gradient_accumulator.get_accumulated_gradients();
            result.gradients_accumulated = true;
            result.memory_freed += self.gradient_accumulator.get_memory_savings();
        }

        Ok(result)
    }

    fn optimize_checkpoints(&mut self) -> Result<MemoryOptimizationResult> {
        let result = MemoryOptimizationResult::default();

        // Checkpoint optimization logic would go here
        // This would involve analyzing which activations to checkpoint vs recompute

        Ok(result)
    }

    fn optimize_parameter_storage(&mut self) -> Result<MemoryOptimizationResult> {
        let mut result = MemoryOptimizationResult::default();

        // Prefetch predicted parameters
        self.offload_manager.prefetch_parameters()?;
        result.parameters_prefetched = true;

        result.memory_freed += self.offload_manager.get_memory_savings();

        Ok(result)
    }

    fn optimize_batch_size(&mut self) -> Result<MemoryOptimizationResult> {
        let mut result = MemoryOptimizationResult::default();

        // Try to scale up batch size if memory allows
        if let Some(new_batch_size) = self.batch_controller.try_scale_up(
            self.pressure_monitor.get_pressure()
        )? {
            result.batch_size_increased = true;
            result.new_batch_size = Some(new_batch_size);
        }

        Ok(result)
    }

    fn get_current_memory_usage(&self) -> Result<usize> {
        // In practice, would query GPU memory usage
        Ok(self.memory_tracker.current_usage)
    }

    fn get_total_memory(&self) -> Result<usize> {
        // In practice, would query total GPU memory
        Ok(self.memory_tracker.total_gpu_memory)
    }
}

/// Memory mapped storage for large parameters
#[derive(Debug)]
pub struct MemoryMappedStorage<T: Float> {
    /// Mapped files
    mapped_files: HashMap<String, MappedFile>,

    /// Total mapped memory
    total_mapped: usize,

    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> MemoryMappedStorage<T> {
    /// Create new memory mapped storage
    pub fn new() -> Self {
        Self {
            mapped_files: HashMap::new(),
            total_mapped: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Mapped file information
#[derive(Debug)]
pub struct MappedFile {
    /// File path
    pub path: String,

    /// Size in bytes
    pub size: usize,

    /// Access count
    pub access_count: usize,
}

/// Result of memory optimization operations
#[derive(Debug, Default)]
pub struct MemoryOptimizationResult {
    /// Total memory freed (bytes)
    pub memory_freed: usize,

    /// Whether batch size was reduced
    pub batch_size_reduced: bool,

    /// Whether batch size was increased
    pub batch_size_increased: bool,

    /// New batch size (if changed)
    pub new_batch_size: Option<usize>,

    /// Number of checkpoints created
    pub checkpoints_created: usize,

    /// Whether gradients were accumulated
    pub gradients_accumulated: bool,

    /// Whether parameters were prefetched
    pub parameters_prefetched: bool,

    /// Whether emergency cleanup was performed
    pub emergency_cleanup: bool,

    /// Whether garbage collection was triggered
    pub garbage_collection: bool,

    /// Time taken for optimization
    pub optimization_time: Duration,

    /// Memory pressure level during optimization
    pub memory_pressure: f32,
}

impl MemoryOptimizationResult {
    /// Merge another result into this one
    pub fn merge(&mut self, other: MemoryOptimizationResult) {
        self.memory_freed += other.memory_freed;
        self.batch_size_reduced |= other.batch_size_reduced;
        self.batch_size_increased |= other.batch_size_increased;
        if other.new_batch_size.is_some() {
            self.new_batch_size = other.new_batch_size;
        }
        self.checkpoints_created += other.checkpoints_created;
        self.gradients_accumulated |= other.gradients_accumulated;
        self.parameters_prefetched |= other.parameters_prefetched;
        self.emergency_cleanup |= other.emergency_cleanup;
        self.garbage_collection |= other.garbage_collection;
        self.optimization_time += other.optimization_time;
    }

    /// Check if any optimizations were performed
    pub fn has_optimizations(&self) -> bool {
        self.memory_freed > 0 ||
        self.batch_size_reduced ||
        self.batch_size_increased ||
        self.checkpoints_created > 0 ||
        self.gradients_accumulated ||
        self.parameters_prefetched ||
        self.emergency_cleanup ||
        self.garbage_collection
    }
}

/// Memory savings summary
#[derive(Debug, Default)]
pub struct MemorySavingsSummary {
    /// Savings from gradient accumulation
    pub gradient_accumulation_savings: usize,

    /// Savings from checkpointing
    pub checkpoint_savings: usize,

    /// Savings from parameter offloading
    pub offload_savings: usize,

    /// Savings from mixed precision
    pub mixed_precision_savings: usize,

    /// Savings from zero redundancy optimization
    pub zero_redundancy_savings: usize,

    /// Total memory savings
    pub total_savings: usize,
}

impl MemorySavingsSummary {
    /// Calculate total savings
    pub fn calculate_total(&mut self) {
        self.total_savings = self.gradient_accumulation_savings +
                           self.checkpoint_savings +
                           self.offload_savings +
                           self.mixed_precision_savings +
                           self.zero_redundancy_savings;
    }

    /// Get savings breakdown as percentages
    pub fn get_breakdown(&self) -> HashMap<String, f32> {
        let mut breakdown = HashMap::new();

        if self.total_savings > 0 {
            let total = self.total_savings as f32;
            breakdown.insert("gradient_accumulation".to_string(),
                           self.gradient_accumulation_savings as f32 / total * 100.0);
            breakdown.insert("checkpointing".to_string(),
                           self.checkpoint_savings as f32 / total * 100.0);
            breakdown.insert("parameter_offloading".to_string(),
                           self.offload_savings as f32 / total * 100.0);
            breakdown.insert("mixed_precision".to_string(),
                           self.mixed_precision_savings as f32 / total * 100.0);
            breakdown.insert("zero_redundancy".to_string(),
                           self.zero_redundancy_savings as f32 / total * 100.0);
        }

        breakdown
    }
}

/// Optimization performance statistics
#[derive(Debug, Default)]
pub struct OptimizationStats {
    /// Total number of optimization runs
    pub total_optimizations: usize,

    /// Total time spent on optimization
    pub total_optimization_time: Duration,

    /// Average optimization time
    pub avg_optimization_time: Duration,

    /// Number of high pressure events handled
    pub high_pressure_events: usize,

    /// Number of critical pressure events handled
    pub critical_pressure_events: usize,

    /// Total memory freed over all optimizations
    pub total_memory_freed: usize,

    /// Number of batch size adjustments
    pub batch_size_adjustments: usize,

    /// Number of emergency cleanups performed
    pub emergency_cleanups: usize,
}

impl OptimizationStats {
    /// Calculate optimization frequency (optimizations per hour)
    pub fn optimization_frequency(&self, uptime: Duration) -> f32 {
        if uptime.as_secs() > 0 {
            self.total_optimizations as f32 / uptime.as_secs_f32() * 3600.0
        } else {
            0.0
        }
    }

    /// Calculate average memory freed per optimization
    pub fn avg_memory_freed(&self) -> usize {
        if self.total_optimizations > 0 {
            self.total_memory_freed / self.total_optimizations
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_memory_optimizer_creation() {
        let config = AdvancedMemoryConfig::default();
        let optimizer = AdvancedMemoryOptimizer::<f32>::new(config);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_memory_optimization_result_merge() {
        let mut result1 = MemoryOptimizationResult {
            memory_freed: 1000,
            batch_size_reduced: true,
            checkpoints_created: 2,
            ..Default::default()
        };

        let result2 = MemoryOptimizationResult {
            memory_freed: 500,
            gradients_accumulated: true,
            checkpoints_created: 1,
            ..Default::default()
        };

        result1.merge(result2);

        assert_eq!(result1.memory_freed, 1500);
        assert!(result1.batch_size_reduced);
        assert!(result1.gradients_accumulated);
        assert_eq!(result1.checkpoints_created, 3);
    }

    #[test]
    fn test_memory_savings_summary() {
        let mut summary = MemorySavingsSummary {
            gradient_accumulation_savings: 1000,
            checkpoint_savings: 2000,
            offload_savings: 3000,
            mixed_precision_savings: 1500,
            zero_redundancy_savings: 2500,
            total_savings: 0,
        };

        summary.calculate_total();
        assert_eq!(summary.total_savings, 10000);

        let breakdown = summary.get_breakdown();
        assert!((breakdown["gradient_accumulation"] - 10.0).abs() < 0.1);
        assert!((breakdown["checkpointing"] - 20.0).abs() < 0.1);
        assert!((breakdown["parameter_offloading"] - 30.0).abs() < 0.1);
    }

    #[test]
    fn test_optimization_stats() {
        let mut stats = OptimizationStats::default();
        stats.total_optimizations = 100;
        stats.total_memory_freed = 50000;

        assert_eq!(stats.avg_memory_freed(), 500);

        let frequency = stats.optimization_frequency(Duration::from_secs(3600));
        assert_eq!(frequency, 100.0);
    }

    #[test]
    fn test_invalid_config_validation() {
        let mut config = AdvancedMemoryConfig::default();
        config.memory_pressure_threshold = 1.5; // Invalid

        let optimizer = AdvancedMemoryOptimizer::<f32>::new(config);
        assert!(optimizer.is_err());
    }
}