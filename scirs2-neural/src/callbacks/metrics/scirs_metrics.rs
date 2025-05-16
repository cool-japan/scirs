//! Integration with scirs2-metrics
//!
//! This module provides a callback for using scirs2-metrics with scirs2-neural models.

use crate::callbacks::{Callback, CallbackContext, CallbackTiming};
use crate::error::Result;
use ndarray::{Array, IxDyn};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::{Debug, Display};

/// Feature flag check for scirs2-metrics integration
#[cfg(not(feature = "metrics_integration"))]
compile_error!("The 'metrics_integration' feature must be enabled to use ScirsMetricsCallback");

/// Callback for using scirs2-metrics with neural network training
///
/// This callback integrates with scirs2-metrics to provide advanced metrics
/// during model training and evaluation.
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "metrics_integration")]
/// # {
/// use scirs2_metrics::integration::neural::NeuralMetricAdapter;
/// use scirs2_neural::callbacks::metrics::ScirsMetricsCallback;
///
/// let metrics = vec![
///     NeuralMetricAdapter::<f32>::accuracy(),
///     NeuralMetricAdapter::<f32>::precision(),
///     NeuralMetricAdapter::<f32>::f1_score(),
/// ];
///
/// let callback = ScirsMetricsCallback::new(metrics);
/// # }
/// ```
#[cfg(feature = "metrics_integration")]
pub struct ScirsMetricsCallback<F: Float + Debug + Display + FromPrimitive + Send + Sync> {
    /// Metrics adapters
    metrics: Vec<scirs2_metrics::integration::neural::NeuralMetricAdapter<F>>,
    /// Current batch predictions
    current_predictions: Option<Array<F, IxDyn>>,
    /// Current batch targets
    current_targets: Option<Array<F, IxDyn>>,
    /// Current epoch results
    epoch_results: HashMap<String, F>,
    /// History of results
    history: Vec<HashMap<String, F>>,
    /// Whether to log metric values
    verbose: bool,
}

#[cfg(feature = "metrics_integration")]
impl<F: Float + Debug + Display + FromPrimitive + Send + Sync> ScirsMetricsCallback<F> {
    /// Create a new ScirsMetricsCallback with the given metrics
    pub fn new(
        metrics: Vec<scirs2_metrics::integration::neural::NeuralMetricAdapter<F>>
    ) -> Self {
        Self {
            metrics,
            current_predictions: None,
            current_targets: None,
            epoch_results: HashMap::new(),
            history: Vec::new(),
            verbose: true,
        }
    }

    /// Set whether to log metric values
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Get the history of metric values
    pub fn history(&self) -> &[HashMap<String, F>] {
        &self.history
    }

    /// Get the current epoch results
    pub fn epoch_results(&self) -> &HashMap<String, F> {
        &self.epoch_results
    }
}

#[cfg(feature = "metrics_integration")]
impl<F: Float + Debug + Display + FromPrimitive + Send + Sync> Callback<F> for ScirsMetricsCallback<F> {
    fn on_event(&mut self, timing: CallbackTiming, context: &mut CallbackContext<F>) -> Result<()> {
        match timing {
            CallbackTiming::AfterBatch => {
                // After each batch, we store the predictions and targets
                // We'll need to wait for this feature to be implemented in scirs2-neural
                // as the current CallbackContext doesn't provide access to this data
            },
            CallbackTiming::AfterEpoch => {
                // After each epoch, compute metrics if we have predictions and targets
                if let (Some(preds), Some(targets)) = (&self.current_predictions, &self.current_targets) {
                    // Compute each metric
                    self.epoch_results.clear();

                    for metric in &self.metrics {
                        match metric.compute(preds, targets) {
                            Ok(value) => {
                                let metric_name = metric.name().to_string();
                                if self.verbose {
                                    println!("  {}: {:.4}", metric_name, value);
                                }
                                self.epoch_results.insert(metric_name.clone(), value);
                                
                                // Update context metrics for history tracking
                                context.metrics.push((metric_name, Some(value)));
                            },
                            Err(err) => {
                                if self.verbose {
                                    eprintln!("Error computing {}: {}", metric.name(), err);
                                }
                            }
                        }
                    }

                    // Store this epoch's results in history
                    self.history.push(self.epoch_results.clone());
                }

                // Reset for next epoch
                self.current_predictions = None;
                self.current_targets = None;
            },
            _ => {}
        }

        Ok(())
    }
}