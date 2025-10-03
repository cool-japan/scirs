//! Traits for integration with other modules
//!
//! This module defines traits that serve as an abstraction layer for integration
//! with other scirs2 modules without creating direct dependencies. These traits
//! are implemented conditionally based on feature flags.

use crate::error::MetricsError;
use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_core::numeric::Float;
use std::fmt::{Debug, Display};

/// Trait for metrics that can be computed on neural network predictions and targets
pub trait MetricComputation<F: Float + Debug + Display> {
    /// Compute the metric value from predictions and targets
    fn compute(
        &self,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
    ) -> Result<F, MetricsError>;

    /// Get the name of the metric
    fn name(&self) -> &str;
}

/// Trait for callbacks that can track metrics during training
pub trait MetricCallback<F: Float + Debug + Display> {
    /// Initialize the callback at the start of training
    fn on_train_begin(&mut self);

    /// Update with batch results
    fn on_batch_end(
        &mut self,
        batch: usize,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
    );

    /// Finalize metrics at the end of an epoch
    fn on_epoch_end(&mut self, epoch: usize) -> Result<(), MetricsError>;

    /// Clean up at the end of training
    fn on_train_end(&mut self);
}
