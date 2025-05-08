//! Validation utilities
//!
//! This module provides utilities for model validation during training.

use super::{EvaluationConfig, Evaluator, MetricType};
use crate::data::Dataset;
use crate::error::{Error, Result};
use crate::layers::Layer;

use ndarray::{Array, IxDyn, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::{Debug, Display};

/// Configuration for validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Batch size for validation
    pub batch_size: usize,
    /// Whether to shuffle the validation data
    pub shuffle: bool,
    /// Number of workers for data loading
    pub num_workers: usize,
    /// Validation steps (None for full dataset)
    pub steps: Option<usize>,
    /// Metrics to compute during validation
    pub metrics: Vec<MetricType>,
    /// Verbosity level
    pub verbose: usize,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: false,
            num_workers: 0,
            steps: None,
            metrics: vec![MetricType::Loss],
            verbose: 1,
            early_stopping: None,
        }
    }
}

/// Configuration for early stopping
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Monitor metric (e.g., 'val_loss')
    pub monitor: String,
    /// Minimum change to qualify as improvement
    pub min_delta: f64,
    /// Number of epochs with no improvement after which training will stop
    pub patience: usize,
    /// Whether to restore model weights from the epoch with the best value of the monitored metric
    pub restore_best_weights: bool,
    /// Mode: 'min' or 'max'
    pub mode: EarlyStoppingMode,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            monitor: "val_loss".to_string(),
            min_delta: 0.0001,
            patience: 5,
            restore_best_weights: true,
            mode: EarlyStoppingMode::Min,
        }
    }
}

/// Early stopping mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyStoppingMode {
    /// Minimize metric (e.g., loss)
    Min,
    /// Maximize metric (e.g., accuracy)
    Max,
}

/// Validation handler for model validation during training
#[derive(Debug)]
pub struct ValidationHandler<
    F: Float + Debug + ScalarOperand + Display + FromPrimitive + Send + Sync,
> {
    /// Configuration for validation
    pub config: ValidationConfig,
    /// Evaluator for validation metrics
    evaluator: Evaluator<F>,
    /// Early stopping state
    early_stopping: Option<EarlyStoppingState<F>>,
}

/// Early stopping state
#[derive(Debug)]
pub struct EarlyStoppingState<
    F: Float + Debug + ScalarOperand + Display + FromPrimitive + Send + Sync,
> {
    /// Configuration for early stopping
    config: EarlyStoppingConfig,
    /// Best value of the monitored metric
    best_value: F,
    /// Number of epochs with no improvement
    wait: usize,
    /// Best weights
    best_weights: Option<Vec<Array<F, IxDyn>>>,
    /// Whether early stopping has been triggered
    stopped_epoch: Option<usize>,
}

impl<F: Float + Debug + ScalarOperand + Display + FromPrimitive + Send + Sync>
    ValidationHandler<F>
{
    /// Create a new validation handler
    pub fn new(config: ValidationConfig) -> Result<Self> {
        // Create evaluator
        let eval_config = EvaluationConfig {
            batch_size: config.batch_size,
            shuffle: config.shuffle,
            num_workers: config.num_workers,
            metrics: config.metrics.clone(),
            steps: config.steps,
            verbose: config.verbose,
        };

        let evaluator = Evaluator::new(eval_config)?;

        // Create early stopping state if needed
        let early_stopping = if let Some(es_config) = &config.early_stopping {
            Some(EarlyStoppingState {
                config: es_config.clone(),
                best_value: match es_config.mode {
                    EarlyStoppingMode::Min => F::infinity(),
                    EarlyStoppingMode::Max => F::neg_infinity(),
                },
                wait: 0,
                best_weights: None,
                stopped_epoch: None,
            })
        } else {
            None
        };

        Ok(Self {
            config,
            evaluator,
            early_stopping,
        })
    }

    /// Validate a model on a dataset
    pub fn validate<L: Layer<F>>(
        &mut self,
        model: &mut L,
        dataset: &dyn Dataset<F>,
        loss_fn: Option<&dyn crate::losses::Loss<F>>,
        epoch: usize,
    ) -> Result<(HashMap<String, F>, bool)> {
        // Set model to evaluation mode
        model.set_training(false);

        // Evaluate model
        let metrics = self.evaluator.evaluate(model, dataset, loss_fn)?;

        // Rename metrics with 'val_' prefix
        let mut val_metrics = HashMap::new();
        for (name, value) in metrics {
            val_metrics.insert(format!("val_{}", name), value);
        }

        // Handle early stopping
        let should_stop = if let Some(ref mut es_state) = self.early_stopping {
            let monitor_value = if let Some(value) = val_metrics.get(&es_state.config.monitor) {
                *value
            } else {
                return Err(Error::ValidationError(format!(
                    "Early stopping monitor '{}' not found in validation metrics",
                    es_state.config.monitor
                )));
            };

            // Check if improved
            let improved = match es_state.config.mode {
                EarlyStoppingMode::Min => {
                    monitor_value + F::from(es_state.config.min_delta).unwrap()
                        < es_state.best_value
                }
                EarlyStoppingMode::Max => {
                    monitor_value - F::from(es_state.config.min_delta).unwrap()
                        > es_state.best_value
                }
            };

            if improved {
                if self.config.verbose > 0 {
                    println!(
                        "Epoch {}: {} improved from {:.4} to {:.4}",
                        epoch, es_state.config.monitor, es_state.best_value, monitor_value
                    );
                }

                es_state.best_value = monitor_value;
                es_state.wait = 0;

                // Save best weights if configured
                if es_state.config.restore_best_weights {
                    es_state.best_weights = Some(model.params());
                }

                false
            } else {
                es_state.wait += 1;

                if self.config.verbose > 0 {
                    println!(
                        "Epoch {}: {} did not improve from {:.4}",
                        epoch, es_state.config.monitor, es_state.best_value
                    );
                }

                if es_state.wait >= es_state.config.patience {
                    if self.config.verbose > 0 {
                        println!(
                            "Early stopping triggered: no improvement in {} for {} epochs",
                            es_state.config.monitor, es_state.config.patience
                        );
                    }

                    es_state.stopped_epoch = Some(epoch);

                    // Restore best weights if configured
                    if es_state.config.restore_best_weights {
                        if let Some(ref best_weights) = es_state.best_weights {
                            // Replace model parameters with best weights
                            let mut params = model.params();
                            for (i, best_param) in best_weights.iter().enumerate() {
                                if i < params.len() {
                                    params[i].assign(best_param);
                                }
                            }
                        }
                    }

                    true
                } else {
                    false
                }
            }
        } else {
            false
        };

        // Restore model to training mode
        model.set_training(true);

        Ok((val_metrics, should_stop))
    }

    /// Check if early stopping is enabled
    pub fn has_early_stopping(&self) -> bool {
        self.early_stopping.is_some()
    }

    /// Get the current early stopping state
    pub fn get_early_stopping_state(&self) -> Option<&EarlyStoppingState<F>> {
        self.early_stopping.as_ref()
    }

    /// Reset early stopping state
    pub fn reset_early_stopping(&mut self) {
        if let Some(ref mut es_state) = self.early_stopping {
            es_state.best_value = match es_state.config.mode {
                EarlyStoppingMode::Min => F::infinity(),
                EarlyStoppingMode::Max => F::neg_infinity(),
            };
            es_state.wait = 0;
            es_state.best_weights = None;
            es_state.stopped_epoch = None;
        }
    }
}
