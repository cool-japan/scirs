//! Adapters for scirs2-optim integration
//!
//! This module provides adapters for using scirs2-metrics with scirs2-optim.

#[allow(unused_imports)]
use crate::error::MetricsError;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

/// Metric optimization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationMode {
    /// Minimize the metric (lower is better)
    Minimize,
    /// Maximize the metric (higher is better)
    Maximize,
}

impl fmt::Display for OptimizationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizationMode::Minimize => write!(f, "minimize"),
            OptimizationMode::Maximize => write!(f, "maximize"),
        }
    }
}

/// Adapter for using scirs2-metrics with scirs2-optim
#[derive(Debug, Clone)]
pub struct MetricOptimizer<F: Float + fmt::Debug + fmt::Display + FromPrimitive = f64> {
    /// Metric name
    metric_name: String,
    /// Optimization mode
    mode: OptimizationMode,
    /// History of metric values
    history: Vec<F>,
    /// Best metric value seen so far
    best_value: Option<F>,
    /// Additional metrics to track
    additional_metrics: HashMap<String, Vec<F>>,
    /// Phantom data for F type
    _phantom: PhantomData<F>,
}

impl<F: Float + fmt::Debug + fmt::Display + FromPrimitive> MetricOptimizer<F> {
    /// Create a new metric optimizer
    ///
    /// # Arguments
    ///
    /// * `metric_name` - Name of the metric to optimize
    /// * `maximize` - Whether to maximize (true) or minimize (false) the metric
    pub fn new<S: Into<String>>(name: S, maximize: bool) -> Self {
        Self {
            metric_name: name.into(),
            mode: if maximize {
                OptimizationMode::Maximize
            } else {
                OptimizationMode::Minimize
            },
            history: Vec::new(),
            best_value: None,
            additional_metrics: HashMap::new(),
            _phantom: PhantomData,
        }
    }

    /// Get the metric name
    pub fn metric_name(&self) -> &str {
        &self.metric_name
    }

    /// Get the optimization mode
    pub fn mode(&self) -> OptimizationMode {
        self.mode
    }

    /// Get the metric history
    pub fn history(&self) -> &[F] {
        &self.history
    }

    /// Get the best metric value seen so far
    pub fn best_value(&self) -> Option<F> {
        self.best_value
    }

    /// Add a metric value to the history
    pub fn add_value(&mut self, value: F) {
        self.history.push(value);

        // Update best value
        self.best_value = match (self.best_value, self.mode) {
            (None, _) => Some(value),
            (Some(best), OptimizationMode::Maximize) if value > best => Some(value),
            (Some(best), OptimizationMode::Minimize) if value < best => Some(value),
            (Some(best), _) => Some(best),
        };
    }

    /// Add a value for an additional metric to track
    pub fn add_additional_value(&mut self, metricname: &str, value: F) {
        self.additional_metrics
            .entry(metricname.to_string())
            .or_default()
            .push(value);
    }

    /// Get the history of an additional metric
    pub fn additional_metric_history(&self, metricname: &str) -> Option<&[F]> {
        self.additional_metrics
            .get(metricname)
            .map(|v| v.as_slice())
    }

    /// Reset the optimizer state
    pub fn reset(&mut self) {
        self.history.clear();
        self.best_value = None;
        self.additional_metrics.clear();
    }

    /// Check if the current value is better than the previous best
    pub fn is_better(&self, current: F, previous: F) -> bool {
        match self.mode {
            OptimizationMode::Maximize => current > previous,
            OptimizationMode::Minimize => current < previous,
        }
    }

    /// Check if the current metric value is better than the best so far
    pub fn is_improvement(&self, value: F) -> bool {
        match self.best_value {
            None => true,
            Some(best) => self.is_better(value, best),
        }
    }

    /// Create scheduler configuration for this metric
    ///
    /// Returns a configuration that can be used to create an external scheduler.
    /// This provides a bridge to scirs2-optim schedulers without circular dependencies.
    pub fn create_scheduler_config(
        &self,
        initial_lr: F,
        factor: F,
        patience: usize,
        min_lr: F,
    ) -> SchedulerConfig<F> {
        SchedulerConfig {
            initial_lr,
            factor,
            patience,
            min_lr,
            mode: self.mode,
            metric_name: self.metric_name.clone(),
        }
    }
}

/// Configuration for external scheduler creation
#[derive(Debug, Clone)]
pub struct SchedulerConfig<F: Float + fmt::Debug + fmt::Display + FromPrimitive> {
    /// Initial learning rate
    pub initial_lr: F,
    /// Factor by which to reduce learning rate
    pub factor: F,
    /// Number of epochs with no improvement before reduction
    pub patience: usize,
    /// Minimum learning rate
    pub min_lr: F,
    /// Optimization mode (minimize or maximize)
    pub mode: OptimizationMode,
    /// Metric name for tracking
    pub metric_name: String,
}

impl<F: Float + fmt::Debug + fmt::Display + FromPrimitive> SchedulerConfig<F> {
    /// Get the configuration as a tuple for easy destructuring
    pub fn as_tuple(&self) -> (F, F, usize, F, OptimizationMode) {
        (
            self.initial_lr,
            self.factor,
            self.patience,
            self.min_lr,
            self.mode,
        )
    }

    /// Create a new scheduler configuration
    pub fn new(
        initial_lr: F,
        factor: F,
        patience: usize,
        min_lr: F,
        mode: OptimizationMode,
        metric_name: String,
    ) -> Self {
        Self {
            initial_lr,
            factor,
            patience,
            min_lr,
            mode,
            metric_name,
        }
    }
}
