//! Learning rate schedulers based on metrics
//!
//! This module provides learning rate schedulers that adjust the learning rate
//! based on metric values.

use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};
#[cfg(not(feature = "metrics_integration"))]
use std::fmt::Debug;
#[cfg(feature = "metrics_integration")]
use std::fmt::{Debug, Display};

#[cfg(feature = "metrics_integration")]
use crate::schedulers::LearningRateScheduler;

/// A scheduler that adjusts learning rate based on metrics
#[cfg(feature = "metrics_integration")]
#[derive(Debug, Clone)]
pub struct MetricScheduler<F: Float + Debug + Display + ScalarOperand + FromPrimitive> {
    /// Base scheduler
    scheduler: scirs2_metrics::integration::optim::MetricLRScheduler<F>,
    /// Threshold for considering an improvement
    threshold: F,
}

#[cfg(feature = "metrics_integration")]
impl<F: Float + Debug + Display + ScalarOperand + FromPrimitive> MetricScheduler<F> {
    /// Create a new metric-based scheduler
    pub fn new(
        initial_lr: F,
        factor: F,
        patience: usize,
        min_lr: F,
        metric_name: &str,
        maximize: bool,
    ) -> Self {
        Self {
            scheduler: scirs2_metrics::integration::optim::MetricLRScheduler::new(
                initial_lr,
                factor,
                patience,
                min_lr,
                metric_name,
                maximize,
            ),
            threshold: F::from(1e-4).unwrap(),
        }
    }

    /// Set the threshold for considering an improvement
    pub fn with_threshold(mut self, threshold: F) -> Self {
        self.threshold = threshold;
        self.scheduler.set_threshold(threshold);
        self
    }

    /// Update scheduler with a metric value
    pub fn step_with_metric(&mut self, metric_value: F) -> F {
        self.scheduler.step_with_metric(metric_value)
    }

    /// Get the current learning rate
    pub fn get_lr(&self) -> F {
        self.scheduler.get_learning_rate()
    }

    /// Get the history of learning rates
    pub fn history(&self) -> &[F] {
        self.scheduler.history()
    }

    /// Get the history of metric values
    pub fn metric_history(&self) -> &[F] {
        self.scheduler.metric_history()
    }

    /// Get the best metric value
    pub fn best_metric(&self) -> Option<F> {
        self.scheduler.best_metric()
    }
}

#[cfg(feature = "metrics_integration")]
impl<F: Float + Debug + Display + ScalarOperand + FromPrimitive> LearningRateScheduler<F>
    for MetricScheduler<F>
{
    fn get_learning_rate(&self) -> F {
        self.scheduler.get_learning_rate()
    }

    fn step(&mut self) -> F {
        // Without a metric value, the learning rate remains unchanged
        self.get_learning_rate()
    }

    fn reset(&mut self) {
        self.scheduler.reset();
    }
}

/// A wrapper around ReduceOnPlateau for use with metrics
#[cfg(feature = "metrics_integration")]
#[derive(Debug)]
pub struct MetricBasedReduceOnPlateau<F: Float + Debug + Display + ScalarOperand + FromPrimitive> {
    /// Base scheduler
    scheduler: crate::schedulers::ReduceOnPlateau<F>,
    /// Metric name
    metric_name: String,
    /// Metric history
    metric_history: Vec<F>,
    /// Learning rate history
    lr_history: Vec<F>,
}

#[cfg(feature = "metrics_integration")]
impl<F: Float + Debug + Display + ScalarOperand + FromPrimitive> MetricBasedReduceOnPlateau<F> {
    /// Create a new metric-based ReduceOnPlateau scheduler
    pub fn new(
        initial_lr: F,
        factor: F,
        patience: usize,
        min_lr: F,
        metric_name: &str,
        maximize: bool,
    ) -> Self {
        let mut scheduler =
            crate::schedulers::ReduceOnPlateau::new(initial_lr, factor, patience, min_lr);

        // Set mode based on maximize flag
        if maximize {
            scheduler.mode_max();
        } else {
            scheduler.mode_min();
        }

        Self {
            scheduler,
            metric_name: metric_name.to_string(),
            metric_history: Vec::new(),
            lr_history: Vec::new(),
        }
    }

    /// Update scheduler with a metric value
    pub fn step_with_metric(&mut self, metric_value: F) -> F {
        self.metric_history.push(metric_value);
        let lr = self.scheduler.step_with_metric(metric_value);
        self.lr_history.push(lr);
        lr
    }

    /// Get the metric name
    pub fn metric_name(&self) -> &str {
        &self.metric_name
    }

    /// Get the metric history
    pub fn metric_history(&self) -> &[F] {
        &self.metric_history
    }

    /// Get the learning rate history
    pub fn lr_history(&self) -> &[F] {
        &self.lr_history
    }
}

#[cfg(feature = "metrics_integration")]
impl<F: Float + Debug + Display + ScalarOperand + FromPrimitive> LearningRateScheduler<F>
    for MetricBasedReduceOnPlateau<F>
{
    fn get_learning_rate(&self) -> F {
        self.scheduler.get_learning_rate()
    }

    fn step(&mut self) -> F {
        // Without a metric value, the learning rate remains unchanged
        self.get_learning_rate()
    }

    fn reset(&mut self) {
        self.scheduler.reset();
        self.metric_history.clear();
        self.lr_history.clear();
    }
}

/// Error raised when metrics integration is not enabled
#[cfg(not(feature = "metrics_integration"))]
#[derive(Debug)]
pub struct MetricScheduler<F: Float + Debug> {
    _phantom: std::marker::PhantomData<F>,
}

#[cfg(not(feature = "metrics_integration"))]
impl<F: Float + Debug + ScalarOperand + FromPrimitive> MetricScheduler<F> {
    /// Create a new metric-based scheduler (not implemented)
    pub fn new(
        _initial_lr: F,
        _factor: F,
        _patience: usize,
        _min_lr: F,
        _metric_name: &str,
        _maximize: bool,
    ) -> Self {
        panic!("metrics_integration feature is not enabled - enable it in your Cargo.toml");
    }
}
