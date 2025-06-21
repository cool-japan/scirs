//! Integration with scirs2-optim for optimization-driven evaluation
//!
//! This module provides utilities for integrating metrics with optimization
//! algorithms and learning rate schedulers without circular dependencies.
//!
//! The integration works by providing configuration types and traits that
//! can be used by external schedulers and optimizers.

mod adapter;
mod hyperparameter;
mod scheduler;

pub use adapter::{MetricOptimizer, OptimizationMode, SchedulerConfig};
pub use hyperparameter::*;
pub use scheduler::{
    MetricLRScheduler, MetricScheduler as MetricSchedulerTrait, SchedulerBridge, SchedulerState,
};
