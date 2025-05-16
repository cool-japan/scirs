//! Integration with scirs2-optim
//!
//! This module provides adapters and utilities for integrating scirs2-metrics
//! with scirs2-optim, allowing metrics to be used for hyperparameter optimization
//! and learning rate scheduling.

mod adapter;
mod hyperparameter;
mod scheduler;

pub use adapter::*;
pub use hyperparameter::*;
pub use scheduler::*;
