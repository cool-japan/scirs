//! Training utilities
//!
//! This module provides utilities for training neural networks,
//! including gradient accumulation, mixed precision training,
//! distributed training, and other advanced training features.

pub mod gradient_accumulation;
pub mod mixed_precision;

pub use gradient_accumulation::*;
pub use mixed_precision::*;

use crate::callbacks::CallbackManager;
use crate::data::{DataLoader, Dataset};
use crate::error::Result;
use crate::evaluation::{EvaluationConfig, Evaluator};
use crate::layers::Layer;
use crate::losses::Loss;
use crate::optimizers::{Optimizer, OptimizerExt};

use num_integer::div_ceil;

use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Batch size for training
    pub batch_size: usize,
    /// Whether to shuffle the training data
    pub shuffle: bool,
    /// Number of workers for data loading
    pub num_workers: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of epochs to train
    pub epochs: usize,
    /// Verbosity level
    pub verbose: usize,
    /// Validation settings
    pub validation: Option<ValidationSettings>,
    /// Gradient accumulation settings
    pub gradient_accumulation: Option<GradientAccumulationConfig>,
    /// Mixed precision settings
    pub mixed_precision: Option<MixedPrecisionConfig>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            num_workers: 0,
            learning_rate: 0.001,
            epochs: 10,
            verbose: 1,
            validation: None,
            gradient_accumulation: None,
            mixed_precision: None,
        }
    }
}

/// Validation settings
#[derive(Debug, Clone)]
pub struct ValidationSettings {
    /// Whether to use validation
    pub enabled: bool,
    /// Validation split (0.0 to 1.0)
    pub validation_split: f64,
    /// Batch size for validation
    pub batch_size: usize,
    /// Number of workers for validation data loading
    pub num_workers: usize,
}

impl Default for ValidationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            validation_split: 0.2,
            batch_size: 32,
            num_workers: 0,
        }
    }
}

/// Training session for tracking training history
#[derive(Debug, Clone)]
pub struct TrainingSession<F: Float + Debug + ScalarOperand> {
    /// Training metrics history
    pub history: HashMap<String, Vec<F>>,
    /// Initial learning rate
    pub initial_learning_rate: F,
    /// Number of epochs trained
    pub epochs_trained: usize,
    /// Number of batches per epoch
    pub batches_per_epoch: usize,
    /// Total number of parameters
    pub total_parameters: usize,
    /// Training configuration
    pub config: TrainingConfig,
}

impl<F: Float + Debug + ScalarOperand> TrainingSession<F> {
    /// Create a new training session
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            history: HashMap::new(),
            initial_learning_rate: F::from(config.learning_rate).unwrap(),
            epochs_trained: 0,
            batches_per_epoch: 0,
            total_parameters: 0,
            config,
        }
    }

    /// Add a metric to history
    pub fn add_metric(&mut self, name: &str, value: F) {
        self.history
            .entry(name.to_string())
            .or_default()
            .push(value);
    }

    /// Get metric history
    pub fn get_metric(&self, name: &str) -> Option<&[F]> {
        self.history.get(name).map(|v| v.as_slice())
    }
}

/// Trainer for a neural network model
pub struct Trainer<
    F: Float + Debug + ScalarOperand + num_traits::FromPrimitive + Send + Sync + std::fmt::Display,
> {
    /// Model to train
    model: Box<dyn Layer<F> + Send + Sync>,
    /// Optimizer for parameter updates
    optimizer: Box<dyn Optimizer<F> + Send + Sync>,
    /// Loss function
    loss_fn: Box<dyn Loss<F> + Send + Sync>,
    /// Callback manager
    callback_manager: CallbackManager<F>,
    /// Gradient accumulator (if enabled)
    gradient_accumulator: Option<GradientAccumulator<F>>,
    /// Mixed precision manager (if enabled)
    #[allow(dead_code)]
    mixed_precision: Option<MixedPrecisionManager<F, F>>,
    /// Training configuration
    config: TrainingConfig,
    /// Current training session
    session: TrainingSession<F>,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + std::fmt::Display + Send + Sync>
    Trainer<F>
{
    /// Create a new trainer
    pub fn new<L, O, LF>(model: L, optimizer: O, loss_fn: LF, config: TrainingConfig) -> Self
    where
        L: Layer<F> + Send + Sync + 'static,
        O: Optimizer<F> + Send + Sync + 'static,
        LF: Loss<F> + Send + Sync + 'static,
    {
        let session = TrainingSession::new(config.clone());

        let gradient_accumulator = config
            .gradient_accumulation
            .as_ref()
            .map(|ga_config| GradientAccumulator::new(ga_config.clone()));

        let mixed_precision = None; // Initialize later if needed

        Self {
            model: Box::new(model),
            optimizer: Box::new(optimizer),
            loss_fn: Box::new(loss_fn),
            callback_manager: CallbackManager::new(),
            gradient_accumulator,
            mixed_precision,
            config,
            session,
        }
    }

    /// Add a callback to the trainer
    pub fn add_callback(&mut self, callback: Box<dyn Fn() -> Result<()> + Send + Sync>) {
        let func_callback = crate::callbacks::FunctionCallback::<F>::new(callback);
        self.callback_manager.add_callback(Box::new(func_callback));
    }

    /// Train the model
    pub fn train<D: Dataset<F> + Clone>(
        &mut self,
        dataset: &D,
        validation_dataset: Option<&D>,
    ) -> Result<TrainingSession<F>> {
        // Initialize gradient accumulation if enabled
        if let Some(ref mut accumulator) = self.gradient_accumulator {
            accumulator.initialize(&*self.model)?;
        }

        // TODO: Initialize mixed precision if enabled

        // Call on_train_begin for callbacks
        self.callback_manager.on_train_begin()?;

        // Count parameters
        let params = self.model.params();
        let total_parameters = params.iter().map(|p| p.len()).sum();
        self.session.total_parameters = total_parameters;

        // Train for specified number of epochs
        for epoch in 0..self.config.epochs {
            // Call on_epoch_begin for callbacks
            self.callback_manager.on_epoch_begin(epoch)?;

            // Create data loader with cloned dataset
            let data_loader = DataLoader::new(
                dataset.clone(),
                self.config.batch_size,
                true,
                false, // Don't drop last batch
            );

            let total_batches = div_ceil(dataset.len(), self.config.batch_size);
            self.session.batches_per_epoch = total_batches;

            let mut total_loss = F::zero();
            let mut batch_count = 0;

            // Initialize epoch metrics
            let mut epoch_metrics = HashMap::new();
            epoch_metrics.insert("loss".to_string(), F::zero());

            // Process batches
            for batch_result in data_loader {
                let batch_idx = batch_count;
                let (inputs, targets) = batch_result?;
                // Call on_batch_begin for callbacks
                self.callback_manager.on_batch_begin(batch_idx)?;

                // Process batch based on training mode
                let batch_loss = if let Some(ref mut accumulator) = self.gradient_accumulator {
                    // Train with gradient accumulation
                    let loss = accumulator.accumulate_gradients(
                        &mut *self.model,
                        &inputs,
                        &targets,
                        &*self.loss_fn,
                    )?;

                    // Apply gradients if it's time to update
                    if accumulator.should_update() || batch_idx == total_batches - 1 {
                        accumulator.apply_gradients(&mut *self.model, &mut *self.optimizer)?;
                    }

                    loss
                } else {
                    // Standard training

                    // Forward pass
                    let outputs = self.model.forward(&inputs)?;

                    // Compute loss and gradients
                    let loss = self.loss_fn.forward(&outputs, &targets)?;
                    let loss_grad = self.loss_fn.backward(&outputs, &targets)?;

                    // Backward pass
                    let _input_grad = self.model.backward(&inputs, &loss_grad)?;

                    // Update parameters
                    self.optimizer.step(&mut *self.model)?;

                    loss
                };

                // Update metrics
                total_loss = total_loss + batch_loss;
                batch_count += 1;
                *epoch_metrics.get_mut("loss").unwrap() =
                    total_loss / F::from(batch_count).unwrap();

                // Call on_batch_end for callbacks
                self.callback_manager
                    .on_batch_end(batch_idx, &epoch_metrics)?;

                // Print progress
                if self.config.verbose > 0 && (batch_idx + 1) % 10 == 0 {
                    println!(
                        "Epoch {}/{} - Batch {}/{} - loss: {:.4}",
                        epoch + 1,
                        self.config.epochs,
                        batch_idx + 1,
                        total_batches,
                        epoch_metrics["loss"]
                    );
                }
            }

            // Compute average loss for the epoch
            let avg_loss = if batch_count > 0 {
                total_loss / F::from(batch_count).unwrap()
            } else {
                F::zero()
            };

            epoch_metrics.insert("loss".to_string(), avg_loss);

            // Validate if a validation dataset is provided
            if let Some(val_dataset) = validation_dataset {
                let val_metrics = self.validate(val_dataset)?;

                // Add validation metrics to epoch metrics
                for (name, value) in &val_metrics {
                    epoch_metrics.insert(format!("val_{}", name), *value);
                }

                if self.config.verbose > 0 {
                    println!(
                        "Epoch {}/{} - loss: {:.4} - val_loss: {:.4}",
                        epoch + 1,
                        self.config.epochs,
                        avg_loss,
                        val_metrics["loss"]
                    );
                }
            } else if self.config.verbose > 0 {
                println!(
                    "Epoch {}/{} - loss: {:.4}",
                    epoch + 1,
                    self.config.epochs,
                    avg_loss
                );
            }

            // Update session history
            for (name, value) in &epoch_metrics {
                self.session.add_metric(name, *value);
            }

            // Call on_epoch_end for callbacks
            let should_stop = self.callback_manager.on_epoch_end(epoch, &epoch_metrics)?;

            // Increment epochs trained
            self.session.epochs_trained += 1;

            // Check if training should stop early
            if should_stop {
                if self.config.verbose > 0 {
                    println!("Early stopping triggered after {} epochs", epoch + 1);
                }
                break;
            }
        }

        // Call on_train_end for callbacks
        self.callback_manager.on_train_end()?;

        Ok(self.session.clone())
    }

    /// Validate the model on a dataset
    pub fn validate<D: Dataset<F>>(&mut self, dataset: &D) -> Result<HashMap<String, F>> {
        // Set model to evaluation mode
        self.model.set_training(false);

        // Create evaluator
        let eval_config = EvaluationConfig {
            batch_size: self.config.batch_size,
            shuffle: false,
            num_workers: self.config.num_workers,
            metrics: vec![],
            steps: None,
            verbose: 0,
        };

        let mut evaluator = Evaluator::new(eval_config)?;

        // Evaluate model
        let metrics = evaluator.evaluate(&*self.model, dataset, Some(&*self.loss_fn))?;

        // Set model back to training mode
        self.model.set_training(true);

        Ok(metrics)
    }

    /// Get the model
    pub fn get_model(&self) -> &dyn Layer<F> {
        &*self.model
    }

    /// Get the model (mutable)
    pub fn get_model_mut(&mut self) -> &mut dyn Layer<F> {
        &mut *self.model
    }

    /// Get the optimizer
    pub fn get_optimizer(&self) -> &dyn Optimizer<F> {
        &*self.optimizer
    }

    /// Get the optimizer (mutable)
    pub fn get_optimizer_mut(&mut self) -> &mut dyn Optimizer<F> {
        &mut *self.optimizer
    }

    /// Get the loss function
    pub fn get_loss_fn(&self) -> &dyn Loss<F> {
        &*self.loss_fn
    }

    /// Get the current training session
    pub fn get_session(&self) -> &TrainingSession<F> {
        &self.session
    }
}
