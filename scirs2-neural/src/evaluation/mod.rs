//! Model evaluation framework
//!
//! This module provides utilities for evaluating neural network models,
//! including validation, testing, and performance metrics.

mod cross_validation;
mod metrics;
mod test;
mod validation;

pub use cross_validation::*;
pub use metrics::*;
pub use test::*;
pub use validation::*;

use crate::data::Dataset;
use crate::error::{Error, Result};
use crate::layers::Layer;

use ndarray::{Array, IxDyn, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

/// Configuration for model evaluation
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Batch size for evaluation
    pub batch_size: usize,
    /// Whether to shuffle the data
    pub shuffle: bool,
    /// Number of workers for data loading
    pub num_workers: usize,
    /// Metrics to compute during evaluation
    pub metrics: Vec<MetricType>,
    /// Number of batches to evaluate (None for all batches)
    pub steps: Option<usize>,
    /// Verbosity level (0 = silent, 1 = progress bar, 2 = batch updates)
    pub verbose: usize,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: false,
            num_workers: 0,
            metrics: vec![MetricType::Loss],
            steps: None,
            verbose: 1,
        }
    }
}

/// Trait for building models
pub trait ModelBuilder<F: Float + Debug + ScalarOperand> {
    /// The model type that will be built by this builder
    type Model: Layer<F> + Clone;

    /// Build a new model instance
    fn build(&self) -> Result<Self::Model>;
}

/// Types of evaluation metrics
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Loss metric
    Loss,
    /// Accuracy metric
    Accuracy,
    /// Precision metric
    Precision,
    /// Recall metric
    Recall,
    /// F1 score metric
    F1Score,
    /// Mean squared error
    MeanSquaredError,
    /// Mean absolute error
    MeanAbsoluteError,
    /// R-squared
    RSquared,
    /// Area under ROC curve
    AUC,
    /// Custom metric
    Custom(String),
}

/// Model evaluator for assessing model performance
#[derive(Debug)]
pub struct Evaluator<
    F: Float + Debug + ScalarOperand + FromPrimitive + std::fmt::Display + Send + Sync,
> {
    /// Configuration for evaluation
    pub config: EvaluationConfig,
    /// Metrics to compute
    metrics: HashMap<MetricType, Box<dyn Metric<F>>>,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + std::fmt::Display + Send + Sync>
    Evaluator<F>
{
    /// Create a new evaluator with the given configuration
    pub fn new(config: EvaluationConfig) -> Result<Self> {
        let mut metrics = HashMap::new();

        // Initialize metrics
        for metric_type in &config.metrics {
            let metric: Box<dyn Metric<F>> = match metric_type {
                MetricType::Loss => Box::new(LossMetric::new()),
                MetricType::Accuracy => Box::new(AccuracyMetric::new()),
                MetricType::Precision => Box::new(PrecisionMetric::new()),
                MetricType::Recall => Box::new(RecallMetric::new()),
                MetricType::F1Score => Box::new(F1ScoreMetric::new()),
                MetricType::MeanSquaredError => Box::new(MeanSquaredErrorMetric::new()),
                MetricType::MeanAbsoluteError => Box::new(MeanAbsoluteErrorMetric::new()),
                MetricType::RSquared => Box::new(RSquaredMetric::new()),
                MetricType::AUC => Box::new(AUCMetric::new()),
                MetricType::Custom(name) => {
                    return Err(Error::NotImplemented(format!(
                        "Custom metric '{}' is not yet supported",
                        name
                    )));
                }
            };

            metrics.insert(metric_type.clone(), metric);
        }

        Ok(Self { config, metrics })
    }

    /// Evaluate a model on a dataset
    pub fn evaluate<L: Layer<F> + ?Sized, D: Dataset<F> + ?Sized>(
        &mut self,
        model: &L,
        dataset: &D,
        loss_fn: Option<&dyn crate::losses::Loss<F>>,
    ) -> Result<HashMap<String, F>> {
        // We can't use DataLoader directly with &dyn Dataset<F>
        // Implement a manual batch iteration similar to what we did in test.rs

        // Calculate number of batches
        let num_samples = dataset.len();
        let num_batches = num_samples / self.config.batch_size
            + if num_samples % self.config.batch_size > 0 {
                1
            } else {
                0
            };

        // Reset metrics
        for metric in self.metrics.values_mut() {
            metric.reset();
        }

        // Number of steps to evaluate
        let steps = self.config.steps.unwrap_or(num_batches);

        // Show progress based on verbosity
        if self.config.verbose > 0 {
            println!(
                "Evaluating model on {} samples ({} batches)",
                dataset.len(),
                steps
            );
        }

        // Loop through batches
        let mut batch_count = 0;

        // Generate indices
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if self.config.shuffle {
            use rand::rngs::SmallRng;
            use rand::seq::SliceRandom;
            use rand::SeedableRng;
            let mut rng = SmallRng::seed_from_u64(42);
            indices.shuffle(&mut rng);
        }

        // Process each batch
        for batch_idx in 0..steps.min(num_batches) {
            // Determine batch range
            let start_idx = batch_idx * self.config.batch_size;
            let end_idx = (start_idx + self.config.batch_size).min(dataset.len());
            let batch_indices = &indices[start_idx..end_idx];

            // Skip empty batches
            if batch_indices.is_empty() {
                continue;
            }

            // Load first sample to determine shapes
            let (first_x, first_y) = dataset.get(batch_indices[0])?;

            // Create batch arrays
            let batch_x_shape = [batch_indices.len()]
                .iter()
                .chain(first_x.shape())
                .cloned()
                .collect::<Vec<_>>();
            let batch_y_shape = [batch_indices.len()]
                .iter()
                .chain(first_y.shape())
                .cloned()
                .collect::<Vec<_>>();

            let mut batch_x = Array::zeros(IxDyn(&batch_x_shape));
            let mut batch_y = Array::zeros(IxDyn(&batch_y_shape));

            // Fill batch arrays
            for (i, &idx) in batch_indices.iter().enumerate() {
                let (x, y) = dataset.get(idx)?;

                // Copy data into batch arrays
                let mut batch_x_slice = batch_x.slice_mut(ndarray::s![i, ..]);
                batch_x_slice.assign(&x);

                let mut batch_y_slice = batch_y.slice_mut(ndarray::s![i, ..]);
                batch_y_slice.assign(&y);
            }

            // Forward pass
            let outputs = model.forward(&batch_x)?;

            // Compute loss if needed
            if self.metrics.contains_key(&MetricType::Loss) && loss_fn.is_some() {
                if let Some(loss_fn) = loss_fn {
                    let loss = loss_fn.forward(&outputs, &batch_y)?;
                    self.metrics.get_mut(&MetricType::Loss).unwrap().update(
                        &outputs,
                        &batch_y,
                        Some(loss),
                    );
                }
            }

            // Update other metrics
            for (metric_type, metric) in self.metrics.iter_mut() {
                if *metric_type != MetricType::Loss {
                    metric.update(&outputs, &batch_y, None);
                }
            }

            batch_count += 1;

            // Print progress if verbose
            if self.config.verbose == 2 {
                println!("Batch {}/{}", batch_count, steps);
            }
        }

        // Collect results
        let mut results = HashMap::new();
        for (metric_type, metric) in &self.metrics {
            let value = metric.result();
            let name = match metric_type {
                MetricType::Loss => "loss".to_string(),
                MetricType::Accuracy => "accuracy".to_string(),
                MetricType::Precision => "precision".to_string(),
                MetricType::Recall => "recall".to_string(),
                MetricType::F1Score => "f1_score".to_string(),
                MetricType::MeanSquaredError => "mse".to_string(),
                MetricType::MeanAbsoluteError => "mae".to_string(),
                MetricType::RSquared => "r2".to_string(),
                MetricType::AUC => "auc".to_string(),
                MetricType::Custom(name) => name.clone(),
            };

            results.insert(name, value);
        }

        // Print results if verbose
        if self.config.verbose > 0 {
            println!("Evaluation results:");
            for (name, value) in &results {
                println!("  {}: {:.4}", name, value);
            }
        }

        Ok(results)
    }

    /// Add a custom metric to the evaluator
    pub fn add_metric(&mut self, name: &str, metric: Box<dyn Metric<F>>) {
        self.metrics
            .insert(MetricType::Custom(name.to_string()), metric);
    }
}

/// Metric interface for model evaluation
pub trait Metric<F: Float + Debug + ScalarOperand + FromPrimitive + std::fmt::Display + Send + Sync>:
    Debug
{
    /// Update the metric with new predictions and targets
    fn update(&mut self, predictions: &Array<F, IxDyn>, targets: &Array<F, IxDyn>, loss: Option<F>);

    /// Reset the metric for a new evaluation
    fn reset(&mut self);

    /// Get the result of the metric
    fn result(&self) -> F;

    /// Get the name of the metric
    fn name(&self) -> &str;
}
