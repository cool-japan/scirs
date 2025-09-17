//! Advanced GPU Memory Optimization
//!
//! This module provides comprehensive GPU memory optimization techniques for large-scale
//! neural network training, including:
//!
//! - **Memory-efficient gradient accumulation** with adaptive microbatching
//! - **Activation checkpointing** with intelligent recomputation strategies
//! - **Parameter offloading** to CPU/storage with compression
//! - **Dynamic batch size control** based on memory pressure
//! - **Memory pressure monitoring** with adaptive response
//! - **Mixed precision training** support
//! - **Zero redundancy optimization** for distributed training
//!
//! # Architecture
//!
//! The memory optimizer is built with a modular architecture:
//!
//! - **config**: Configuration structures and optimization strategies
//! - **memory_tracking**: Memory usage tracking, snapshots, and profiling
//! - **gradient_management**: Memory-efficient gradient accumulation
//! - **checkpoint_management**: Activation checkpointing and recomputation
//! - **parameter_offloading**: Parameter offloading with compression
//! - **dynamic_batching**: Adaptive batch size control
//! - **pressure_monitoring**: Memory pressure detection and response
//! - **core**: Main AdvancedMemoryOptimizer implementation
//!
//! # Usage Example
//!
//! ```rust
//! use scirs2_optim::gpu::advanced_memory_optimization::{
//!     AdvancedMemoryOptimizer, AdvancedMemoryConfig
//! };
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create configuration for large models
//! let config = AdvancedMemoryConfig::for_large_models();
//!
//! // Create optimizer
//! let mut optimizer = AdvancedMemoryOptimizer::<f32>::new(config);
//!
//! // Enable memory optimization
//! optimizer.enable_optimization()?;
//!
//! // Monitor memory during training
//! let memory_info = optimizer.get_memory_info();
//! println!("Memory usage: {:.1}%", memory_info.usage_percentage * 100.0);
//! # Ok(())
//! # }
//! ```
//!
//! # Memory Optimization Strategies
//!
//! ## Gradient Checkpointing
//! - Selective activation saving during forward pass
//! - Intelligent recomputation during backward pass
//! - Adaptive checkpoint placement based on memory pressure
//!
//! ## Parameter Offloading
//! - CPU memory offloading for unused parameters
//! - Compressed storage with LZ4/ZSTD
//! - Asynchronous transfer optimization
//!
//! ## Dynamic Batching
//! - Automatic batch size adjustment
//! - Memory pressure-driven scaling
//! - Performance-aware optimization
//!
//! ## Memory Pressure Management
//! - Real-time memory monitoring
//! - Adaptive optimization triggering
//! - Cost-benefit analysis for optimizations

pub mod config;
pub mod memory_tracking;
pub mod gradient_management;
pub mod checkpoint_management;
pub mod parameter_offloading;
pub mod dynamic_batching;
pub mod pressure_monitoring;
pub mod core;

// Re-export core configuration types
pub use config::{
    AdvancedMemoryConfig, AllocationType, PressureAction, CheckpointStrategy,
    EvictionPolicy, StorageLocation, CompressionType, OffloadStrategy,
    TransferCost, CompressionInfo, RecomputationCost, CompressionConfig,
    PerformanceConfig, OptimizationStrategies, CompressionAlgorithm
};

// Re-export memory tracking types
pub use memory_tracking::{
    MemoryUsageTracker, MemorySnapshot, AllocationEvent, PressureEvent,
    MemoryProfiler, MemoryInfo, AllocationInfo, MemoryRegion, ProfilerConfig,
    MemoryOptimizationResult
};

// Re-export gradient management
pub use gradient_management::{
    GradientAccumulator, AccumulationStrategy, MicrobatchConfig,
    GradientCompressionConfig, AccumulationState, GradientSnapshot,
    CompressionMetrics
};

// Re-export checkpoint management
pub use checkpoint_management::{
    CheckpointManager, ActivationCheckpoint, CheckpointMetadata,
    RecomputationGraph, CheckpointCache, CheckpointStats
};

// Re-export parameter offloading
pub use parameter_offloading::{
    ParameterOffloadManager, OffloadedParameter, OffloadBuffer,
    OffloadCache, OffloadMetrics, ParameterLocation, TransferManager
};

// Re-export dynamic batching
pub use dynamic_batching::{
    DynamicBatchController, BatchSizeEvent, BatchChangeReason,
    PerformanceMetrics, BatchAdaptationStrategy, BatchOptimizationConfig,
    BatchHistory, AdaptationMetrics
};

// Re-export pressure monitoring
pub use pressure_monitoring::{
    MemoryPressureMonitor, PressureThresholds, PressureTrend,
    PressureResponse, PressureHistory, MonitoringConfig, PressureMetrics
};

// Re-export main optimizer
pub use core::{
    AdvancedMemoryOptimizer, OptimizationContext, MemoryOptimizationStrategy,
    OptimizationResult, OptimizationMetrics, MemoryState, OptimizationPhase
};

/// Convenience function to create a default memory optimizer
pub fn create_default_optimizer<T: num_traits::Float + Default + Clone + Send + Sync + std::iter::Sum>()
    -> crate::error::Result<AdvancedMemoryOptimizer<T>> {
    let config = AdvancedMemoryConfig::default();
    AdvancedMemoryOptimizer::new(config)
}

/// Convenience function to create an optimizer for large models
pub fn create_large_model_optimizer<T: num_traits::Float + Default + Clone + Send + Sync + std::iter::Sum>()
    -> crate::error::Result<AdvancedMemoryOptimizer<T>> {
    let config = AdvancedMemoryConfig::for_large_models();
    AdvancedMemoryOptimizer::new(config)
}

/// Convenience function to create an optimizer for training efficiency
pub fn create_training_optimizer<T: num_traits::Float + Default + Clone + Send + Sync + std::iter::Sum>()
    -> crate::error::Result<AdvancedMemoryOptimizer<T>> {
    let config = AdvancedMemoryConfig::for_training_efficiency();
    AdvancedMemoryOptimizer::new(config)
}

/// Convenience function to create an optimizer for inference
pub fn create_inference_optimizer<T: num_traits::Float + Default + Clone + Send + Sync + std::iter::Sum>()
    -> crate::error::Result<AdvancedMemoryOptimizer<T>> {
    let config = AdvancedMemoryConfig::for_inference();
    AdvancedMemoryOptimizer::new(config)
}

/// High-level function to optimize memory usage with automatic configuration
pub fn optimize_memory_automatically<T: num_traits::Float + Default + Clone + Send + Sync + std::iter::Sum>(
    available_memory: usize,
    model_size: usize,
    batch_size: usize,
) -> crate::error::Result<AdvancedMemoryOptimizer<T>> {
    let memory_ratio = model_size as f64 / available_memory as f64;

    let config = if memory_ratio > 0.8 {
        // Large model relative to memory - aggressive optimization
        AdvancedMemoryConfig::for_large_models()
    } else if memory_ratio > 0.5 {
        // Moderate memory usage - balanced optimization
        AdvancedMemoryConfig::default()
    } else {
        // Plenty of memory - optimize for performance
        AdvancedMemoryConfig::for_training_efficiency()
    };

    AdvancedMemoryOptimizer::new(config)
}

/// Estimate memory requirements for a given model configuration
pub fn estimate_memory_requirements(
    parameter_count: usize,
    sequence_length: usize,
    batch_size: usize,
    precision_bytes: usize,
) -> MemoryEstimate {
    let parameter_memory = parameter_count * precision_bytes;
    let activation_memory = batch_size * sequence_length * parameter_count * precision_bytes / 100; // Rough estimate
    let gradient_memory = parameter_memory; // Same as parameters
    let optimizer_memory = parameter_memory * 2; // Adam optimizer states

    let total_memory = parameter_memory + activation_memory + gradient_memory + optimizer_memory;
    let overhead = total_memory / 10; // 10% overhead

    MemoryEstimate {
        parameter_memory,
        activation_memory,
        gradient_memory,
        optimizer_memory,
        overhead,
        total_memory: total_memory + overhead,
    }
}

/// Memory requirement estimation result
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Memory required for model parameters
    pub parameter_memory: usize,
    /// Memory required for activations
    pub activation_memory: usize,
    /// Memory required for gradients
    pub gradient_memory: usize,
    /// Memory required for optimizer states
    pub optimizer_memory: usize,
    /// Memory overhead (fragmentation, etc.)
    pub overhead: usize,
    /// Total estimated memory requirement
    pub total_memory: usize,
}

impl MemoryEstimate {
    /// Get memory breakdown as percentages
    pub fn get_breakdown(&self) -> MemoryBreakdown {
        let total = self.total_memory as f32;
        MemoryBreakdown {
            parameters: self.parameter_memory as f32 / total,
            activations: self.activation_memory as f32 / total,
            gradients: self.gradient_memory as f32 / total,
            optimizer: self.optimizer_memory as f32 / total,
            overhead: self.overhead as f32 / total,
        }
    }

    /// Check if memory requirements fit within available memory
    pub fn fits_in_memory(&self, available_memory: usize) -> bool {
        self.total_memory <= available_memory
    }

    /// Get memory efficiency (parameter memory / total memory)
    pub fn efficiency(&self) -> f32 {
        self.parameter_memory as f32 / self.total_memory as f32
    }
}

/// Memory usage breakdown by component
#[derive(Debug, Clone)]
pub struct MemoryBreakdown {
    /// Percentage of memory used by parameters
    pub parameters: f32,
    /// Percentage of memory used by activations
    pub activations: f32,
    /// Percentage of memory used by gradients
    pub gradients: f32,
    /// Percentage of memory used by optimizer states
    pub optimizer: f32,
    /// Percentage of memory used by overhead
    pub overhead: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_default_optimizer() {
        let result = create_default_optimizer::<f32>();
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_large_model_optimizer() {
        let result = create_large_model_optimizer::<f32>();
        assert!(result.is_ok());
    }

    #[test]
    fn test_memory_estimation() {
        let estimate = estimate_memory_requirements(
            1_000_000,  // 1M parameters
            512,        // sequence length
            8,          // batch size
            4,          // 4 bytes per parameter (f32)
        );

        assert!(estimate.total_memory > 0);
        assert!(estimate.parameter_memory > 0);
        assert!(estimate.fits_in_memory(estimate.total_memory));
        assert!(!estimate.fits_in_memory(estimate.total_memory / 2));

        let breakdown = estimate.get_breakdown();
        let total_percentage = breakdown.parameters + breakdown.activations +
                              breakdown.gradients + breakdown.optimizer + breakdown.overhead;
        assert!((total_percentage - 1.0).abs() < 0.01); // Should sum to ~1.0
    }

    #[test]
    fn test_automatic_optimization() {
        // Test different memory scenarios
        let large_model = optimize_memory_automatically::<f32>(
            8 * 1024 * 1024 * 1024, // 8GB available
            7 * 1024 * 1024 * 1024, // 7GB model (87.5% usage)
            16
        );
        assert!(large_model.is_ok());

        let balanced_model = optimize_memory_automatically::<f32>(
            8 * 1024 * 1024 * 1024, // 8GB available
            4 * 1024 * 1024 * 1024, // 4GB model (50% usage)
            16
        );
        assert!(balanced_model.is_ok());

        let small_model = optimize_memory_automatically::<f32>(
            8 * 1024 * 1024 * 1024, // 8GB available
            1 * 1024 * 1024 * 1024, // 1GB model (12.5% usage)
            16
        );
        assert!(small_model.is_ok());
    }

    #[test]
    fn test_memory_estimate_efficiency() {
        let estimate = estimate_memory_requirements(1_000_000, 512, 8, 4);
        let efficiency = estimate.efficiency();
        assert!(efficiency > 0.0 && efficiency <= 1.0);

        // Parameters should be a reasonable fraction of total memory
        assert!(efficiency > 0.1); // At least 10%
    }
}