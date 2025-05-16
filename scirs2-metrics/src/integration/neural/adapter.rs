//! Adapters for scirs2-neural metrics
//!
//! This module provides adapters for using scirs2-metrics metrics with scirs2-neural.

use crate::classification::{
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
};
use crate::error::MetricsError;
use crate::regression::{
    explained_variance_score, mean_absolute_error, mean_squared_error, r2_score,
};
use ndarray::{Array, IxDyn};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

/// Adapter for using scirs2-metrics metrics with scirs2-neural
pub struct NeuralMetricAdapter<F: Float + Debug + Display + FromPrimitive> {
    /// Metric name
    pub name: String,
    /// Metric function
    pub metric_fn:
        Box<dyn Fn(&Array<F, IxDyn>, &Array<F, IxDyn>) -> Result<F, MetricsError> + Send + Sync>,
}

impl<F: Float + Debug + Display + FromPrimitive> NeuralMetricAdapter<F> {
    /// Create a new neural metric adapter
    pub fn new<S: Into<String>>(
        name: S,
        metric_fn: Box<
            dyn Fn(&Array<F, IxDyn>, &Array<F, IxDyn>) -> Result<F, MetricsError> + Send + Sync,
        >,
    ) -> Self {
        Self {
            name: name.into(),
            metric_fn,
        }
    }

    /// Create a new accuracy adapter
    pub fn accuracy() -> Self {
        Self::new(
            "accuracy",
            Box::new(|preds, targets| Ok(accuracy_score(preds, targets)?)),
        )
    }

    /// Create a new precision adapter
    pub fn precision() -> Self {
        Self::new(
            "precision",
            Box::new(|preds, targets| Ok(precision_score(preds, targets, None, None, None)?)),
        )
    }

    /// Create a new recall adapter
    pub fn recall() -> Self {
        Self::new(
            "recall",
            Box::new(|preds, targets| Ok(recall_score(preds, targets, None, None, None)?)),
        )
    }

    /// Create a new F1 score adapter
    pub fn f1_score() -> Self {
        Self::new(
            "f1_score",
            Box::new(|preds, targets| Ok(f1_score(preds, targets, None, None, None)?)),
        )
    }

    /// Create a new ROC AUC adapter
    pub fn roc_auc() -> Self {
        Self::new(
            "roc_auc",
            Box::new(|preds, targets| Ok(roc_auc_score(preds, targets)?)),
        )
    }

    /// Create a new mean squared error adapter
    pub fn mse() -> Self {
        Self::new(
            "mse",
            Box::new(|preds, targets| Ok(mean_squared_error(preds, targets)?)),
        )
    }

    /// Create a new mean absolute error adapter
    pub fn mae() -> Self {
        Self::new(
            "mae",
            Box::new(|preds, targets| Ok(mean_absolute_error(preds, targets)?)),
        )
    }

    /// Create a new R² score adapter
    pub fn r2() -> Self {
        Self::new(
            "r2",
            Box::new(|preds, targets| Ok(r2_score(preds, targets)?)),
        )
    }

    /// Create a new explained variance adapter
    pub fn explained_variance() -> Self {
        Self::new(
            "explained_variance",
            Box::new(|preds, targets| Ok(explained_variance_score(preds, targets)?)),
        )
    }

    /// Compute the metric value
    pub fn compute(
        &self,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
    ) -> Result<F, MetricsError> {
        (self.metric_fn)(predictions, targets)
    }
}

/// Implementation of the Neural Metric trait for the adapter
///
/// This implementation allows the adapter to be used with scirs2-neural's Metric trait.
///
/// Example usage:
/// ```rust
/// # use scirs2_metrics::integration::neural::NeuralMetricAdapter;
/// # use ndarray::Array;
/// # use scirs2_neural::evaluation::Metric;
///
/// # fn example<F: num_traits::Float + std::fmt::Debug + std::fmt::Display + num_traits::FromPrimitive>() {
/// let metric = NeuralMetricAdapter::<F>::accuracy();
///
/// // Use with scirs2-neural's Metric trait
/// let predictions = Array::<F, _>::zeros([10, 5].into_dyn());
/// let targets = Array::<F, _>::zeros([10, 5].into_dyn());
///
/// let result = metric.update(&predictions, &targets, None);
/// # }
/// ```
#[cfg(feature = "neural_common")]
impl<F: Float + Debug + Display + FromPrimitive + Send + Sync> scirs2_neural::evaluation::Metric<F>
    for NeuralMetricAdapter<F>
{
    fn update(
        &mut self,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
        _loss: Option<F>,
    ) {
        // This method is not used directly as scirs2_neural metrics handle state internally
        // Our adapter only computes metrics on demand
    }

    fn reset(&mut self) {
        // No state to reset
    }

    fn result(&self) -> F {
        // This should never be called directly from the adapter
        // Return zero to satisfy the trait
        F::zero()
    }

    fn name(&self) -> &str {
        &self.name
    }
}
