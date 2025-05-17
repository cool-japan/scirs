//! Integration with scirs2-metrics for optimization
//!
//! This module provides the MetricOptimizer which uses metrics to guide optimization.

#[cfg(feature = "metrics_integration")]
use crate::error::Result;
use crate::optimizers::Optimizer;
#[cfg(feature = "metrics_integration")]
use ndarray::{Array, Dimension, ScalarOperand};
#[cfg(not(feature = "metrics_integration"))]
use ndarray::{Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};
#[cfg(feature = "metrics_integration")]
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;

/// An optimizer guided by metric values
#[cfg(feature = "metrics_integration")]
pub struct MetricOptimizer<F, D>
where
    F: Float + Debug + Display + FromPrimitive + ScalarOperand,
    D: Dimension,
{
    /// Base optimizer
    base_optimizer: Box<dyn Optimizer<F, D>>,
    /// Current learning rate
    current_lr: F,
    /// Metric adapter
    metric_adapter: scirs2_metrics::integration::optim::MetricOptimizer<F>,
    /// History of parameter updates
    history: Vec<HashMap<String, Array<F, D>>>,
    /// Best parameters found
    best_params: Option<HashMap<String, Array<F, D>>>,
    /// PhantomData for F and D
    _phantom: PhantomData<(F, D)>,
}

#[cfg(feature = "metrics_integration")]
impl<F, D> MetricOptimizer<F, D>
where
    F: Float + Debug + Display + FromPrimitive + ScalarOperand + 'static,
    D: Dimension + 'static,
{
    /// Create a new MetricOptimizer
    pub fn new<O>(optimizer: O, metric_name: &str, maximize: bool) -> Self
    where
        O: Optimizer<F, D> + 'static,
    {
        let initial_lr = optimizer.get_learning_rate();
        Self {
            base_optimizer: Box::new(optimizer),
            current_lr: initial_lr,
            metric_adapter: scirs2_metrics::integration::optim::MetricOptimizer::new(
                metric_name,
                maximize,
            ),
            history: Vec::new(),
            best_params: None,
            _phantom: PhantomData,
        }
    }

    /// Update the optimizer with a metric value
    pub fn update_metric(&mut self, metric: F) -> Result<()> {
        self.metric_adapter.add_value(metric);
        Ok(())
    }

    /// Update multiple metrics
    pub fn update_metrics(&mut self, metrics: HashMap<String, F>) -> Result<()> {
        // Update the primary metric
        if let Some(value) = metrics.get(self.metric_adapter.metric_name()) {
            self.metric_adapter.add_value(*value);
        }

        // Update additional metrics
        for (name, value) in metrics {
            if name != self.metric_adapter.metric_name() {
                self.metric_adapter.add_additional_value(&name, value);
            }
        }

        Ok(())
    }

    /// Get the metric adapter
    pub fn metric_adapter(&self) -> &scirs2_metrics::integration::optim::MetricOptimizer<F> {
        &self.metric_adapter
    }

    /// Get the metric adapter (mutable)
    pub fn metric_adapter_mut(
        &mut self,
    ) -> &mut scirs2_metrics::integration::optim::MetricOptimizer<F> {
        &mut self.metric_adapter
    }

    /// Get the base optimizer
    pub fn base_optimizer(&self) -> &dyn Optimizer<F, D> {
        &*self.base_optimizer
    }

    /// Get the base optimizer (mutable)
    pub fn base_optimizer_mut(&mut self) -> &mut dyn Optimizer<F, D> {
        &mut *self.base_optimizer
    }

    /// Get the best parameters found
    pub fn best_params(&self) -> Option<&HashMap<String, Array<F, D>>> {
        self.best_params.as_ref()
    }

    /// Get the parameter update history
    pub fn history(&self) -> &[HashMap<String, Array<F, D>>] {
        &self.history
    }

    /// Reset the optimizer
    pub fn reset(&mut self) {
        self.metric_adapter.reset();
        self.history.clear();
        self.best_params = None;
    }

    /// Create a learning rate scheduler for this optimizer
    pub fn create_lr_scheduler(
        &self,
        initial_lr: F,
        factor: F,
        patience: usize,
        min_lr: F,
    ) -> crate::schedulers::ReduceOnPlateau<F> {
        let mut scheduler =
            crate::schedulers::ReduceOnPlateau::new(initial_lr, factor, patience, min_lr);

        // Set mode based on optimization mode
        match self.metric_adapter.mode() {
            scirs2_metrics::integration::optim::OptimizationMode::Minimize => {
                scheduler.mode_min();
            }
            scirs2_metrics::integration::optim::OptimizationMode::Maximize => {
                scheduler.mode_max();
            }
        }

        scheduler
    }
}

#[cfg(feature = "metrics_integration")]
impl<F, D> Optimizer<F, D> for MetricOptimizer<F, D>
where
    F: Float + Debug + Display + FromPrimitive + ScalarOperand + 'static,
    D: Dimension + 'static,
{
    fn step(&mut self, params: &Array<F, D>, gradients: &Array<F, D>) -> Result<Array<F, D>> {
        // Use base optimizer to update parameters
        let updated_params = self.base_optimizer.step(params, gradients)?;

        // Record parameter update in history
        let mut param_update = HashMap::new();
        param_update.insert("params".to_string(), updated_params.clone());
        param_update.insert("gradients".to_string(), gradients.clone());
        self.history.push(param_update);

        // Update best parameters if metric has improved
        if let Some(best_value) = self.metric_adapter.best_value() {
            let is_improvement = match self.metric_adapter.mode() {
                scirs2_metrics::integration::optim::OptimizationMode::Maximize => {
                    // If maximizing, latest metric should be greater than best
                    if let Some(last_value) = self.metric_adapter.history().last() {
                        *last_value > best_value
                    } else {
                        false
                    }
                }
                scirs2_metrics::integration::optim::OptimizationMode::Minimize => {
                    // If minimizing, latest metric should be less than best
                    if let Some(last_value) = self.metric_adapter.history().last() {
                        *last_value < best_value
                    } else {
                        false
                    }
                }
            };

            if is_improvement {
                let mut best_params = HashMap::new();
                best_params.insert("params".to_string(), updated_params.clone());
                self.best_params = Some(best_params);
            }
        }

        Ok(updated_params)
    }

    fn get_learning_rate(&self) -> F {
        self.current_lr
    }

    fn set_learning_rate(&mut self, learning_rate: F) {
        self.current_lr = learning_rate;
    }
}

/// Error raised when metrics integration is not enabled
#[cfg(not(feature = "metrics_integration"))]
#[derive(Debug)]
pub struct MetricOptimizer<F, D>
where
    F: Float + Debug + Display + FromPrimitive + ScalarOperand,
    D: Dimension,
{
    _phantom: PhantomData<(F, D)>,
}

#[cfg(not(feature = "metrics_integration"))]
impl<F, D> MetricOptimizer<F, D>
where
    F: Float + Debug + Display + FromPrimitive + ScalarOperand,
    D: Dimension,
{
    /// Create a new MetricOptimizer (not implemented)
    pub fn new<O>(_optimizer: O, _metric_name: &str, _maximize: bool) -> Self
    where
        O: Optimizer<F, D>,
    {
        panic!("metrics_integration feature is not enabled - enable it in your Cargo.toml");
    }
}
