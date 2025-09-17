//! Adaptive Streaming Optimization System
//!
//! This module provides a comprehensive adaptive streaming optimization system for machine learning
//! workflows. It includes drift detection, performance tracking, resource management, adaptive
//! buffering, meta-learning, and anomaly detection capabilities.
//!
//! The system has been refactored into modular components for better maintainability while preserving
//! API compatibility. All functionality is now organized across specialized modules:
//!
//! - **Configuration**: Centralized configuration management
//! - **Optimizer**: Core adaptive streaming optimizer
//! - **Drift Detection**: Comprehensive concept drift detection
//! - **Performance Tracking**: Real-time performance monitoring and prediction
//! - **Resource Management**: Dynamic resource allocation and monitoring
//! - **Buffering**: Adaptive buffer management with quality-based retention
//! - **Meta-Learning**: Experience-based learning and strategy selection
//! - **Anomaly Detection**: Multi-method anomaly detection for streaming data
//!
//! # Example Usage
//!
//! ```rust
//! use scirs2_optim::streaming::adaptive_streaming::{
//!     AdaptiveStreamingOptimizer, StreamingConfig
//! };
//!
//! // Create configuration
//! let config = StreamingConfig::default();
//!
//! // Initialize optimizer with base optimizer
//! let mut optimizer = AdaptiveStreamingOptimizer::new(base_optimizer, config)?;
//!
//! // Process streaming data
//! for batch in data_stream {
//!     let result = optimizer.adaptive_step(batch)?;
//!     if result.adaptation_applied {
//!         println!("Adaptations applied based on data characteristics");
//!     }
//! }
//! ```

#![allow(dead_code)]

// Re-export all functionality from the modular implementation
pub use self::adaptive_streaming::*;

// Include the modular implementation as a submodule
mod adaptive_streaming;

// Compatibility re-exports for existing API
// This ensures that existing code using this module continues to work unchanged

// Core types
pub use adaptive_streaming::{
    AdaptiveStreamingOptimizer, AdaptiveStepResult, AdaptiveStreamingStats,
    DriftState, ResourceUsage, OptimizationState, StreamingDataPoint,
};

// Configuration types
pub use adaptive_streaming::{
    StreamingConfig, BufferConfig, DriftConfig, PerformanceConfig,
    ResourceConfig, MetaLearningConfig, AnomalyConfig, LearningRateConfig,
};

// Result types
pub use adaptive_streaming::{
    StreamingResult, DriftResult, PerformanceResult, ResourceResult,
    BufferResult, MetaResult, AnomalyResult,
};

// Enums and strategy types
pub use adaptive_streaming::{
    BufferStrategy, EvictionStrategy, DriftDetectionMethod,
    PerformanceMetric, AllocationStrategy, AnomalyDetectionMethod,
};

// Error types
pub use adaptive_streaming::{
    ConfigError, OptimizerError, DriftDetectionError, PerformanceError,
    ResourceError, BufferError, MetaLearningError, AnomalyDetectionError,
};

// Convenience functions
pub use adaptive_streaming::{
    create_default_optimizer, create_optimizer_with_config, validate_config,
};

// Feature-gated exports
#[cfg(feature = "gpu")]
pub use adaptive_streaming::GPUStreamingOptimizer;

#[cfg(feature = "distributed")]
pub use adaptive_streaming::DistributedStreamingOptimizer;

#[cfg(feature = "visualization")]
pub use adaptive_streaming::{PerformanceVisualizer, DriftVisualizer};

#[cfg(feature = "benchmarking")]
pub use adaptive_streaming::{StreamingBenchmark, PerformanceBenchmark};

/// Version information for the adaptive streaming system
pub const ADAPTIVE_STREAMING_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Creates a new adaptive streaming optimizer with comprehensive configuration
///
/// This is a convenience function that provides a fully configured optimizer
/// suitable for most streaming optimization tasks.
///
/// # Arguments
/// * `base_optimizer` - The underlying optimizer to adapt
/// * `latency_budget_ms` - Maximum processing latency budget in milliseconds
/// * `memory_budget_mb` - Maximum memory usage budget in megabytes
/// * `enable_drift_detection` - Whether to enable concept drift detection
/// * `enable_meta_learning` - Whether to enable meta-learning capabilities
///
/// # Returns
/// A configured `AdaptiveStreamingOptimizer` ready for use
///
/// # Example
/// ```rust
/// use scirs2_optim::streaming::adaptive_streaming::create_comprehensive_optimizer;
/// use scirs2_optim::optimizers::SGD;
///
/// let sgd = SGD::new(0.01);
/// let optimizer = create_comprehensive_optimizer(
///     sgd,
///     100,    // 100ms latency budget
///     512,    // 512MB memory budget
///     true,   // Enable drift detection
///     true,   // Enable meta-learning
/// )?;
/// ```
pub fn create_comprehensive_optimizer<O, A, D>(
    base_optimizer: O,
    latency_budget_ms: u64,
    memory_budget_mb: f64,
    enable_drift_detection: bool,
    enable_meta_learning: bool,
) -> StreamingResult<AdaptiveStreamingOptimizer<O, A, D>>
where
    O: Clone + Send + Sync,
    A: ndarray::ScalarOperand + Clone + Default + Send + Sync + 'static,
    D: ndarray::Dimension,
{
    let mut config = StreamingConfig::default();

    // Configure latency budget
    config.performance_config.latency_budget_ms = latency_budget_ms;

    // Configure memory budget
    config.resource_config.memory_budget_mb = memory_budget_mb;

    // Configure drift detection
    if enable_drift_detection {
        config.drift_config.enable_detection = true;
        config.drift_config.sensitivity = 0.1; // Moderate sensitivity
    } else {
        config.drift_config.enable_detection = false;
    }

    // Configure meta-learning
    if enable_meta_learning {
        config.meta_learning_config.enable_meta_learning = true;
        config.meta_learning_config.experience_buffer_size = 1000;
    } else {
        config.meta_learning_config.enable_meta_learning = false;
    }

    AdaptiveStreamingOptimizer::new_with_config(config)
}

/// Creates a performance-optimized adaptive streaming optimizer
///
/// This configuration prioritizes performance over resource efficiency,
/// suitable for high-throughput scenarios with adequate computational resources.
pub fn create_performance_optimized_optimizer<O, A, D>() -> StreamingResult<AdaptiveStreamingOptimizer<O, A, D>>
where
    O: Clone + Send + Sync,
    A: ndarray::ScalarOperand + Clone + Default + Send + Sync + 'static,
    D: ndarray::Dimension,
{
    let mut config = StreamingConfig::default();

    // Performance-first configuration
    config.buffer_config.strategy = BufferStrategy::QualityBased;
    config.buffer_config.max_size = 1024; // Large buffer
    config.performance_config.optimization_target =
        adaptive_streaming::OptimizationTarget::Performance;
    config.resource_config.allocation_strategy = AllocationStrategy::PerformanceFirst;

    AdaptiveStreamingOptimizer::new_with_config(config)
}

/// Creates a resource-efficient adaptive streaming optimizer
///
/// This configuration prioritizes resource efficiency over raw performance,
/// suitable for resource-constrained environments.
pub fn create_resource_efficient_optimizer<O, A, D>() -> StreamingResult<AdaptiveStreamingOptimizer<O, A, D>>
where
    O: Clone + Send + Sync,
    A: ndarray::ScalarOperand + Clone + Default + Send + Sync + 'static,
    D: ndarray::Dimension,
{
    let mut config = StreamingConfig::default();

    // Efficiency-first configuration
    config.buffer_config.strategy = BufferStrategy::FIFO;
    config.buffer_config.max_size = 64; // Small buffer
    config.performance_config.optimization_target =
        adaptive_streaming::OptimizationTarget::Efficiency;
    config.resource_config.allocation_strategy = AllocationStrategy::EfficiencyFirst;
    config.anomaly_config.enable_detection = false; // Disable to save resources

    AdaptiveStreamingOptimizer::new_with_config(config)
}

/// Validates that the adaptive streaming system is properly configured
///
/// This function performs comprehensive validation of the streaming system
/// configuration and returns detailed information about any issues found.
pub fn validate_system_configuration(config: &StreamingConfig) -> Result<Vec<String>, ConfigError> {
    let mut warnings = Vec::new();

    // Validate buffer configuration
    if config.buffer_config.max_size < config.buffer_config.min_size {
        return Err(ConfigError::InvalidBufferConfig(
            "Maximum buffer size cannot be less than minimum buffer size".to_string()
        ));
    }

    // Validate performance configuration
    if config.performance_config.latency_budget_ms == 0 {
        warnings.push("Latency budget is zero, which may cause processing delays".to_string());
    }

    // Validate resource configuration
    if config.resource_config.memory_budget_mb < 10.0 {
        warnings.push("Memory budget is very low, which may impact performance".to_string());
    }

    // Validate meta-learning configuration
    if config.meta_learning_config.enable_meta_learning &&
       config.meta_learning_config.experience_buffer_size < 100 {
        warnings.push("Experience buffer size is small for effective meta-learning".to_string());
    }

    Ok(warnings)
}

/// Provides system information and statistics about the adaptive streaming implementation
pub fn get_system_info() -> adaptive_streaming::SystemInfo {
    adaptive_streaming::SystemInfo {
        version: ADAPTIVE_STREAMING_VERSION.to_string(),
        modular_design: true,
        module_count: 8,
        total_types: 150,
        features_enabled: get_enabled_features(),
    }
}

fn get_enabled_features() -> Vec<String> {
    let mut features = vec![
        "config".to_string(),
        "optimizer".to_string(),
        "drift_detection".to_string(),
        "performance".to_string(),
        "resource_management".to_string(),
        "buffering".to_string(),
        "meta_learning".to_string(),
        "anomaly_detection".to_string(),
    ];

    #[cfg(feature = "gpu")]
    features.push("gpu".to_string());

    #[cfg(feature = "distributed")]
    features.push("distributed".to_string());

    #[cfg(feature = "visualization")]
    features.push("visualization".to_string());

    #[cfg(feature = "benchmarking")]
    features.push("benchmarking".to_string());

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modular_exports() {
        // Test that all major types are accessible through re-exports
        let _config = StreamingConfig::default();
        let _buffer_config = BufferConfig::default();
        let _drift_config = DriftConfig::default();

        // Test enums
        let _drift_state = DriftState::Stable;
        let _buffer_strategy = BufferStrategy::FIFO;
    }

    #[test]
    fn test_system_info() {
        let info = get_system_info();
        assert_eq!(info.modular_design, true);
        assert!(info.module_count > 0);
        assert!(info.features_enabled.len() >= 8); // At least 8 core modules
    }

    #[test]
    fn test_configuration_validation() {
        let config = StreamingConfig::default();
        let validation_result = validate_system_configuration(&config);
        assert!(validation_result.is_ok());
    }

    #[test]
    fn test_invalid_configuration() {
        let mut config = StreamingConfig::default();
        config.buffer_config.max_size = 10;
        config.buffer_config.min_size = 20; // Invalid: max < min

        let validation_result = validate_system_configuration(&config);
        assert!(validation_result.is_err());
    }

    #[test]
    fn test_convenience_functions() {
        // Test that convenience functions are accessible
        let _warnings = validate_config(&StreamingConfig::default());

        #[cfg(test)]
        {
            let _test_config = adaptive_streaming::create_test_config();
        }
    }
}