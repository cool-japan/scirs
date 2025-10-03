//! Callbacks for scirs2-neural training
//!
//! This module provides callbacks that can be used with scirs2-neural's training
//! loops to compute and track metrics from scirs2-metrics.

use crate::error::MetricsError;
use crate::integration::neural::NeuralMetricAdapter;
use crate::integration::traits::MetricComputation;
use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::{Debug, Display};

/// A callback for tracking metrics during neural network training
#[cfg(feature = "neural_common")]
#[derive(Debug)]
pub struct MetricsCallback<
    F: Float + Debug + Display + FromPrimitive + Send + Sync + scirs2_core::simd_ops::SimdUnifiedOps,
> {
    /// Map of metric names to metric adapters
    metrics: Vec<NeuralMetricAdapter<F>>,
    /// Results from the last update
    last_results: HashMap<String, F>,
    /// History of metric values (epoch -> metric_name -> value)
    history: Vec<HashMap<String, F>>,
    /// Whether to log metrics to stderr
    verbose: bool,
}

#[cfg(feature = "neural_common")]
impl<
        F: Float
            + Debug
            + Display
            + FromPrimitive
            + Send
            + Sync
            + scirs2_core::simd_ops::SimdUnifiedOps,
    > MetricsCallback<F>
{
    /// Create a new metrics callback with the specified metrics
    pub fn new(metrics: Vec<NeuralMetricAdapter<F>>, verbose: bool) -> Self {
        Self {
            metrics,
            last_results: HashMap::new(),
            history: Vec::new(),
            verbose,
        }
    }

    /// Get the metric names
    pub fn metric_names(&self) -> Vec<&str> {
        self.metrics.iter().map(|m| m.name.as_str()).collect()
    }

    /// Compute metrics for the given predictions and targets
    pub fn compute_metrics(
        &mut self,
        predictions: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
    ) -> Result<HashMap<String, F>, MetricsError> {
        let mut results = HashMap::new();

        for metric in &self.metrics {
            match metric.compute(predictions, targets) {
                Ok(value) => {
                    results.insert(metric.name.clone(), value);
                }
                Err(err) => {
                    if self.verbose {
                        eprintln!("Error computing metric {}: {}", metric.name, err);
                    }
                    results.insert(metric.name.clone(), F::nan());
                }
            }
        }

        self.last_results = results.clone();
        Ok(results)
    }

    /// Get the last results
    pub fn last_results(&self) -> &HashMap<String, F> {
        &self.last_results
    }

    /// Get the history of metric values
    pub fn history(&self) -> &[HashMap<String, F>] {
        &self.history
    }

    /// Record the current metric values in history
    pub fn record_history(&mut self) {
        self.history.push(self.last_results.clone());
    }
}

#[allow(unexpected_cfgs)]
#[cfg(all(feature = "neural_common", feature = "neural_integration"))]
impl<
        F: Float
            + Debug
            + Display
            + FromPrimitive
            + Send
            + Sync
            + scirs2_core::simd_ops::SimdUnifiedOps,
    > scirs2_neural::callbacks::Callback<F> for MetricsCallback<F>
{
    fn on_event(
        &mut self,
        timing: scirs2,
        neural: callbacks::CallbackTiming,
        context: &mut scirs2_neural::callbacks::CallbackContext<F>,
    ) -> scirs2_neural::error::Result<()> {
        // We only want to compute metrics after each epoch
        if timing != scirs2_neural::callbacks::CallbackTiming::AfterEpoch {
            return Ok(());
        }

        // Currently, we don't have access to predictions and targets in the callback context
        // This would require changes to scirs2-_neural's callback system
        // For now, we'll just log that the callback was invoked

        if self.verbose {
            println!(
                "MetricsCallback: Epoch {}/{}",
                context.epoch + 1,
                context.total_epochs
            );

            // Log current metrics from context
            for (name, value) in &context.metrics {
                if let Some(val) = value {
                    println!("  {}: {:.4}", name, val);
                }
            }
        }

        Ok(())
    }
}
