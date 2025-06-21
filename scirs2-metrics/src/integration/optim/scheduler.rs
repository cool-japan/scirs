//! Learning rate scheduler integration
//!
//! This module provides utilities for integrating learning rate schedulers
//! with metrics.

#[allow(unused_imports)]
use crate::error::Result;
use crate::integration::optim::OptimizationMode;
use num_traits::{Float, FromPrimitive};
use std::fmt;
#[allow(unused_imports)]
use std::marker::PhantomData;

/// A metric-based learning rate scheduler
#[derive(Debug, Clone)]
pub struct MetricLRScheduler<F: Float + fmt::Debug + fmt::Display + FromPrimitive> {
    /// Current learning rate
    current_lr: F,
    /// Initial learning rate
    initial_lr: F,
    /// Factor by which the learning rate will be reduced
    factor: F,
    /// Number of epochs with no improvement after which learning rate will be reduced
    patience: usize,
    /// Minimum learning rate
    min_lr: F,
    /// Counter for steps with no improvement
    stagnation_count: usize,
    /// Best metric value seen so far
    best_metric: Option<F>,
    /// Threshold for measuring improvement
    threshold: F,
    /// Optimization mode
    mode: OptimizationMode,
    /// Metric name
    metric_name: String,
    /// History of learning rates
    history: Vec<F>,
    /// History of metric values
    metric_history: Vec<F>,
}

impl<F: Float + fmt::Debug + fmt::Display + FromPrimitive> MetricLRScheduler<F> {
    /// Create a new metric-based learning rate scheduler
    pub fn new<S: Into<String>>(
        initial_lr: F,
        factor: F,
        patience: usize,
        min_lr: F,
        metric_name: S,
        maximize: bool,
    ) -> Self {
        Self {
            current_lr: initial_lr,
            initial_lr,
            factor,
            patience,
            min_lr,
            stagnation_count: 0,
            best_metric: None,
            threshold: F::from(1e-4).unwrap(),
            mode: if maximize {
                OptimizationMode::Maximize
            } else {
                OptimizationMode::Minimize
            },
            metric_name: metric_name.into(),
            history: vec![initial_lr],
            metric_history: Vec::new(),
        }
    }

    /// Set the threshold for measuring improvement
    pub fn set_threshold(&mut self, threshold: F) -> &mut Self {
        self.threshold = threshold;
        self
    }

    /// Update the scheduler with a new metric value
    pub fn step_with_metric(&mut self, metric: F) -> F {
        // Record metric
        self.metric_history.push(metric);

        let is_improvement = match self.best_metric {
            None => true, // First metric value is always an improvement
            Some(best) => {
                match self.mode {
                    OptimizationMode::Minimize => {
                        // Mode is 'min', improvement means metric < best * (1 - threshold)
                        metric < best * (F::one() - self.threshold)
                    }
                    OptimizationMode::Maximize => {
                        // Mode is 'max', improvement means metric > best * (1 + threshold)
                        metric > best * (F::one() + self.threshold)
                    }
                }
            }
        };

        if is_improvement {
            self.best_metric = Some(metric);
            self.stagnation_count = 0;
        } else {
            self.stagnation_count += 1;

            if self.stagnation_count >= self.patience {
                // Reduce learning rate
                self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                // Add to history
                self.history.push(self.current_lr);
                // Reset stagnation count
                self.stagnation_count = 0;
            }
        }

        self.current_lr
    }

    /// Get the current learning rate
    pub fn get_learning_rate(&self) -> F {
        self.current_lr
    }

    /// Reset the scheduler
    pub fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.stagnation_count = 0;
        self.best_metric = None;
        self.history = vec![self.initial_lr];
        self.metric_history.clear();
    }

    /// Get the history of learning rates
    pub fn history(&self) -> &[F] {
        &self.history
    }

    /// Get the history of metric values
    pub fn metric_history(&self) -> &[F] {
        &self.metric_history
    }

    /// Get the best metric value
    pub fn best_metric(&self) -> Option<F> {
        self.best_metric
    }

    /// Create a scheduler configuration for use with external optimizers
    ///
    /// This returns the current state as a configuration that can be used
    /// to create or update external schedulers from scirs2-optim.
    pub fn to_scheduler_config(&self) -> crate::integration::optim::SchedulerConfig<F> {
        use crate::integration::optim::SchedulerConfig;

        SchedulerConfig {
            initial_lr: self.initial_lr,
            factor: self.factor,
            patience: self.patience,
            min_lr: self.min_lr,
            mode: self.mode,
            metric_name: self.metric_name.clone(),
        }
    }

    /// Get the current scheduler state for external integration
    pub fn get_state(&self) -> SchedulerState<F> {
        SchedulerState {
            current_lr: self.current_lr,
            best_metric: self.best_metric,
            stagnation_count: self.stagnation_count,
            threshold: self.threshold,
            mode: self.mode,
        }
    }
}

/// Current state of a metric-based scheduler
#[derive(Debug, Clone)]
pub struct SchedulerState<F: Float + fmt::Debug + fmt::Display + FromPrimitive> {
    /// Current learning rate
    pub current_lr: F,
    /// Best metric value seen so far
    pub best_metric: Option<F>,
    /// Counter for steps with no improvement
    pub stagnation_count: usize,
    /// Threshold for measuring improvement
    pub threshold: F,
    /// Optimization mode
    pub mode: OptimizationMode,
}

/// Trait for external scheduler integration
///
/// This trait can be implemented by external schedulers (like those in scirs2-optim)
/// to provide seamless integration with metrics.
pub trait MetricScheduler<F: Float + fmt::Debug + fmt::Display + FromPrimitive> {
    /// Update the scheduler with a new metric value
    fn step_with_metric(&mut self, metric: F) -> F;

    /// Get the current learning rate
    fn get_learning_rate(&self) -> F;

    /// Reset the scheduler to initial state
    fn reset(&mut self);

    /// Set the mode (minimize or maximize)
    fn set_mode(&mut self, mode: OptimizationMode);
}

/// Bridge adapter for external scheduler integration
///
/// This provides a standardized interface for metric-based scheduling
/// without depending on specific external implementations.
pub struct SchedulerBridge<F: Float + fmt::Debug + fmt::Display + FromPrimitive> {
    /// Metric-based scheduler state
    inner: Box<dyn MetricScheduler<F>>,
    /// Metric name
    metric_name: String,
    /// Metric history
    metric_history: Vec<F>,
    /// Learning rate history
    lr_history: Vec<F>,
}
