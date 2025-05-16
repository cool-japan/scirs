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
pub struct MetricScheduler<F: Float + fmt::Debug + fmt::Display + FromPrimitive> {
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

impl<F: Float + fmt::Debug + fmt::Display + FromPrimitive> MetricScheduler<F> {
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

    /// Create a scheduler state for use with scirs2-optim
    #[allow(unexpected_cfgs)]
    #[cfg(feature = "optim_integration")]
    pub fn to_optim_scheduler(&self) -> scirs2_optim::schedulers::ReduceOnPlateau<F> {
        let mut scheduler = scirs2_optim::schedulers::ReduceOnPlateau::new(
            self.current_lr,
            self.factor,
            self.patience,
            self.min_lr,
        );

        // Set mode based on optimization mode
        match self.mode {
            OptimizationMode::Minimize => {
                scheduler.mode_min();
            }
            OptimizationMode::Maximize => {
                scheduler.mode_max();
            }
        }

        // Set threshold
        scheduler.set_threshold(self.threshold);

        scheduler
    }
}

/// Adapter for the ReduceOnPlateau scheduler
#[allow(unexpected_cfgs)]
#[cfg(feature = "optim_integration")]
#[derive(Debug)]
pub struct ReduceOnPlateauAdapter<F: Float + fmt::Debug + fmt::Display + FromPrimitive> {
    /// Scheduler
    scheduler: scirs2_optim::schedulers::ReduceOnPlateau<F>,
    /// Metric name
    metric_name: String,
    /// Metric history
    metric_history: Vec<F>,
    /// Learning rate history
    lr_history: Vec<F>,
    /// Phantom data for F type
    _phantom: PhantomData<F>,
}

#[allow(unexpected_cfgs)]
#[cfg(feature = "optim_integration")]
impl<
        F: Float + fmt::Debug + fmt::Display + FromPrimitive + scirs2_optim::optimizers::ScalarOps,
    > ReduceOnPlateauAdapter<F>
{
    /// Create a new ReduceOnPlateau adapter
    pub fn new<S: Into<String>>(
        initial_lr: F,
        factor: F,
        patience: usize,
        min_lr: F,
        metric_name: S,
        maximize: bool,
    ) -> Self {
        let mut scheduler =
            scirs2_optim::schedulers::ReduceOnPlateau::new(initial_lr, factor, patience, min_lr);

        // Set mode based on maximize flag
        if maximize {
            scheduler.mode_max();
        } else {
            scheduler.mode_min();
        }

        Self {
            scheduler,
            metric_name: metric_name.into(),
            metric_history: Vec::new(),
            lr_history: vec![initial_lr],
            _phantom: PhantomData,
        }
    }

    /// Update the scheduler with a new metric value
    pub fn step_with_metric(&mut self, metric: F) -> F {
        // Record metric
        self.metric_history.push(metric);

        // Update scheduler
        let new_lr = self.scheduler.step_with_metric(metric);

        // Record learning rate if it changed
        if new_lr != self.lr_history.last().unwrap_or(&F::zero()) {
            self.lr_history.push(new_lr);
        }

        new_lr
    }

    /// Get the current learning rate
    pub fn get_learning_rate(&self) -> F {
        self.scheduler.get_learning_rate()
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

    /// Reset the scheduler
    pub fn reset(&mut self) {
        self.scheduler.reset();
        self.metric_history.clear();
        self.lr_history = vec![self.scheduler.get_learning_rate()];
    }

    /// Apply the scheduler to an optimizer
    pub fn apply_to<D, O>(&self, optimizer: &mut O)
    where
        D: scirs2_optim::optimizers::Dimension,
        O: scirs2_optim::optimizers::Optimizer<F, D>,
    {
        self.scheduler.apply_to(optimizer);
    }
}
