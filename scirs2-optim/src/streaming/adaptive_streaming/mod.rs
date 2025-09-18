//! Adaptive Streaming Optimization Module
//!
//! This module provides comprehensive adaptive streaming optimization for ML workloads.

pub mod anomaly_detection;
pub mod buffering;
pub mod config;
pub mod drift_detection;
pub mod meta_learning;
pub mod optimizer;
pub mod performance;
pub mod resource_management;

// Simplified exports to avoid import conflicts during refactoring
pub use anomaly_detection::*;
pub use buffering::*;
pub use config::*;
pub use drift_detection::*;
pub use meta_learning::*;
pub use optimizer::*;
pub use performance::*;
pub use resource_management::*;

// Utility functions for common configurations
pub fn create_default_optimizer<O, A, D>() -> StreamingResult<AdaptiveStreamingOptimizer<O, A, D>>
where
    O: Send + Sync + 'static,
    A: ndarray::ScalarOperand + Clone + Default + Send + Sync + 'static + num_traits::Float,
    D: ndarray::Data<Elem = A> + ndarray::Dimension + Send + Sync + 'static,
{
    let config = StreamingConfig::default();
    Ok(AdaptiveStreamingOptimizer::new(config)?)
}

pub fn create_optimizer_with_config<O, A, D>(
    config: StreamingConfig,
) -> StreamingResult<AdaptiveStreamingOptimizer<O, A, D>>
where
    O: Send + Sync + 'static,
    A: ndarray::ScalarOperand + Clone + Default + Send + Sync + 'static + num_traits::Float,
    D: ndarray::Data<Elem = A> + ndarray::Dimension + Send + Sync + 'static,
{
    Ok(AdaptiveStreamingOptimizer::new(config)?)
}

// Result type alias
pub type StreamingResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
