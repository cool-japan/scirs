//! Gradient Accumulation utilities
//!
//! This module provides utilities for gradient accumulation during training,
//! which allows training with larger effective batch sizes by accumulating
//! gradients over multiple batches before updating model parameters.

use crate::data::{DataLoader, Dataset};
use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use crate::losses::Loss;
use crate::optimizers::{Optimizer, OptimizerExt};

use ndarray::ScalarOperand;
use ndarray::{Array, IxDyn};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

/// Gradient accumulation configuration
#[derive(Debug, Clone)]
pub struct GradientAccumulationConfig {
    /// Number of batches to accumulate gradients over
    pub accumulation_steps: usize,
    /// Whether to average gradients over accumulation steps
    /// If true, the gradients are divided by accumulation_steps
    /// If false, the gradients are summed over accumulation steps
    pub average_gradients: bool,
    /// Whether to zero gradients after each update
    pub zero_gradients_after_update: bool,
    /// Whether to clip gradients
    pub clip_gradients: bool,
    /// Maximum gradient norm for gradient clipping
    pub max_gradient_norm: Option<f64>,
    /// Whether to log gradient statistics during training
    pub log_gradient_stats: bool,
}

impl Default for GradientAccumulationConfig {
    fn default() -> Self {
        Self {
            accumulation_steps: 1,
            average_gradients: true,
            zero_gradients_after_update: true,
            clip_gradients: false,
            max_gradient_norm: None,
            log_gradient_stats: false,
        }
    }
}

/// Gradient statistics
#[derive(Debug, Clone)]
pub struct GradientStats<F: Float + Debug + ScalarOperand + Send + Sync + FromPrimitive> {
    /// Minimum gradient value
    pub min: F,
    /// Maximum gradient value
    pub max: F,
    /// Mean gradient value
    pub mean: F,
    /// Gradient norm
    pub norm: F,
}

/// Gradient accumulator for accumulating gradients over multiple batches
#[derive(Debug)]
pub struct GradientAccumulator<F: Float + Debug + ScalarOperand + Send + Sync + FromPrimitive> {
    /// Configuration for gradient accumulation
    pub config: GradientAccumulationConfig,
    /// Current accumulated gradients
    accumulated_gradients: Vec<Array<F, IxDyn>>,
    /// Current accumulation step
    current_step: usize,
    /// Total number of samples processed
    total_samples: usize,
    /// Gradient statistics
    gradient_stats: Option<GradientStats<F>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + FromPrimitive> GradientAccumulator<F> {
    /// Create a new gradient accumulator
    pub fn new(config: GradientAccumulationConfig) -> Self {
        Self {
            config,
            accumulated_gradients: Vec::new(),
            current_step: 0,
            total_samples: 0,
            gradient_stats: None,
        }
    }

    /// Initialize the accumulator with the model's parameter shapes
    pub fn initialize<L: Layer<F> + ?Sized>(&mut self, model: &L) -> Result<()> {
        // Get parameter shapes from the model
        let params = model.params();

        // Initialize accumulated gradients with zeros
        self.accumulated_gradients = params
            .into_iter()
            .map(|param| Array::<F, _>::zeros(param.raw_dim()))
            .collect();

        self.current_step = 0;
        self.total_samples = 0;
        self.gradient_stats = None;

        Ok(())
    }

    /// Accumulate gradients from a forward and backward pass
    pub fn accumulate_gradients<L: Layer<F> + ?Sized>(
        &mut self,
        model: &mut L,
        inputs: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
        loss_fn: &dyn Loss<F>,
    ) -> Result<F> {
        // Ensure accumulator is initialized
        if self.accumulated_gradients.is_empty() {
            self.initialize(model)?;
        }

        // Forward pass
        let outputs = model.forward(inputs)?;

        // Compute loss and gradients
        let loss = loss_fn.forward(&outputs, targets)?;
        let loss_grad = loss_fn.backward(&outputs, targets)?;

        // Backward pass
        let _input_grad = model.backward(inputs, &loss_grad)?;

        // Accumulate gradients
        let gradients = model.gradients();

        // Ensure we have the same number of gradients
        if gradients.len() != self.accumulated_gradients.len() {
            return Err(NeuralError::ValidationError(format!(
                "Expected {} gradients, got {}",
                self.accumulated_gradients.len(),
                gradients.len()
            )));
        }

        // Accumulate gradients
        for (i, grad) in gradients.into_iter().enumerate() {
            if i < self.accumulated_gradients.len() {
                // Add gradients
                self.accumulated_gradients[i] = &self.accumulated_gradients[i] + &grad;
            }
        }

        // Update counters
        self.current_step += 1;
        self.total_samples += inputs.shape()[0];

        // Compute gradient statistics if enabled
        if self.config.log_gradient_stats {
            self.compute_gradient_stats()?;
        }

        Ok(loss)
    }

    /// Apply accumulated gradients to update model parameters
    pub fn apply_gradients<L: Layer<F> + ?Sized, O: Optimizer<F> + ?Sized>(
        &mut self,
        model: &mut L,
        optimizer: &mut O,
    ) -> Result<()> {
        // Ensure we have accumulated some gradients
        if self.current_step == 0 {
            return Ok(());
        }

        // Scale gradients if averaging is enabled
        if self.config.average_gradients && self.current_step > 1 {
            let scale = F::from(1.0 / self.current_step as f64).unwrap();
            for grad in &mut self.accumulated_gradients {
                *grad = grad.clone() * scale;
            }
        }

        // Clip gradients if enabled
        if self.config.clip_gradients {
            self.clip_gradients()?;
        }

        // Apply gradients to the model
        model.set_gradients(&self.accumulated_gradients)?;

        // Update model parameters using the optimizer
        optimizer.step(model)?;

        // Zero gradients after update if enabled
        if self.config.zero_gradients_after_update {
            self.zero_gradients();
        }

        Ok(())
    }

    /// Zero accumulated gradients
    pub fn zero_gradients(&mut self) {
        self.current_step = 0;
        self.total_samples = 0;

        for grad in &mut self.accumulated_gradients {
            *grad = Array::<F, _>::zeros(grad.raw_dim());
        }

        self.gradient_stats = None;
    }

    /// Compute gradient statistics
    fn compute_gradient_stats(&mut self) -> Result<()> {
        let mut min_val = F::infinity();
        let mut max_val = F::neg_infinity();
        let mut sum_val = F::zero();
        let mut count = 0;
        let mut sum_squares = F::zero();

        for grad in &self.accumulated_gradients {
            for &val in grad.iter() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
                sum_val = sum_val + val;
                sum_squares = sum_squares + val * val;
                count += 1;
            }
        }

        if count > 0 {
            let mean = sum_val / F::from(count).unwrap();
            let norm = sum_squares.sqrt();

            self.gradient_stats = Some(GradientStats {
                min: min_val,
                max: max_val,
                mean,
                norm,
            });
        }

        Ok(())
    }

    /// Clip gradients by global norm
    fn clip_gradients(&mut self) -> Result<()> {
        if let Some(max_norm) = self.config.max_gradient_norm {
            let max_norm = F::from(max_norm).unwrap();

            // Compute global norm
            let mut global_norm_sq = F::zero();
            for grad in &self.accumulated_gradients {
                for &val in grad.iter() {
                    global_norm_sq = global_norm_sq + val * val;
                }
            }

            let global_norm = global_norm_sq.sqrt();

            // Clip if necessary
            if global_norm > max_norm {
                let scale = max_norm / global_norm;

                for grad in &mut self.accumulated_gradients {
                    *grad = grad.clone() * scale;
                }
            }
        }

        Ok(())
    }

    /// Check if it's time to update model parameters
    pub fn should_update(&self) -> bool {
        self.current_step >= self.config.accumulation_steps
    }

    /// Get the current accumulated gradients
    pub fn get_accumulated_gradients(&self) -> &[Array<F, IxDyn>] {
        &self.accumulated_gradients
    }

    /// Get the current accumulation step
    pub fn get_current_step(&self) -> usize {
        self.current_step
    }

    /// Get the total number of samples processed
    pub fn get_total_samples(&self) -> usize {
        self.total_samples
    }

    /// Get the gradient statistics if available
    pub fn get_gradient_stats(&self) -> Option<&GradientStats<F>> {
        self.gradient_stats.as_ref()
    }
}

/// Training step with gradient accumulation
pub struct AccumulationTrainingStep<F: Float + Debug + ScalarOperand + Send + Sync + FromPrimitive>
{
    /// Gradient accumulator
    pub accumulator: GradientAccumulator<F>,
    /// Whether to use the entire epoch or a fixed number of steps
    pub use_epoch: bool,
    /// Number of batches to process per epoch
    pub steps_per_epoch: Option<usize>,
    /// Learning rate scheduler
    pub lr_scheduler: Option<Box<dyn crate::callbacks::LearningRateScheduler<F>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + FromPrimitive> AccumulationTrainingStep<F> {
    /// Create a new accumulation training step
    pub fn new(config: GradientAccumulationConfig) -> Self {
        Self {
            accumulator: GradientAccumulator::new(config),
            use_epoch: true,
            steps_per_epoch: None,
            lr_scheduler: None,
        }
    }

    /// Train for one epoch with gradient accumulation
    pub fn train_epoch<L: Layer<F>, O: Optimizer<F>>(
        &mut self,
        model: &mut L,
        optimizer: &mut O,
        dataset: &dyn Dataset<F>,
        loss_fn: &dyn Loss<F>,
        batch_size: usize,
        shuffle: bool,
    ) -> Result<(F, usize)> {
        // Create data loader
        let data_loader = DataLoader::new(dataset.box_clone(), batch_size, shuffle, false);

        // Determine number of steps
        let total_steps = if self.use_epoch {
            data_loader.len()
        } else if let Some(steps) = self.steps_per_epoch {
            steps.min(data_loader.len())
        } else {
            data_loader.len()
        };

        let mut total_loss = F::zero();
        let mut batch_count = 0;

        // Ensure accumulator is initialized
        self.accumulator.initialize(model)?;

        // Train on batches
        for (batch_idx, batch_result) in data_loader.take(total_steps).enumerate() {
            let (inputs, targets) = batch_result?;
            // Accumulate gradients
            let loss = self
                .accumulator
                .accumulate_gradients(model, &inputs, &targets, loss_fn)?;

            total_loss = total_loss + loss;
            batch_count += 1;

            // If it's time to update parameters or it's the last batch
            if self.accumulator.should_update() || batch_idx + 1 == total_steps {
                // Apply accumulated gradients
                self.accumulator.apply_gradients(model, optimizer)?;

                // Update learning rate if scheduler is configured
                if let Some(ref mut scheduler) = self.lr_scheduler {
                    let current_batch = batch_idx + 1;
                    let progress = current_batch as f64 / total_steps as f64;

                    let lr = scheduler.get_learning_rate(progress)?;
                    optimizer.set_learning_rate(lr);
                }
            }
        }

        // Return average loss
        if batch_count > 0 {
            Ok((total_loss / F::from(batch_count).unwrap(), batch_count))
        } else {
            Ok((F::zero(), 0))
        }
    }

    /// Set the learning rate scheduler
    pub fn set_lr_scheduler(
        &mut self,
        scheduler: Box<dyn crate::callbacks::LearningRateScheduler<F>>,
    ) {
        self.lr_scheduler = Some(scheduler);
    }

    /// Set whether to use the entire epoch or a fixed number of steps
    pub fn set_use_epoch(&mut self, use_epoch: bool) {
        self.use_epoch = use_epoch;
    }

    /// Set the number of steps per epoch
    pub fn set_steps_per_epoch(&mut self, steps: Option<usize>) {
        self.steps_per_epoch = steps;
    }
}

/// Gradient accumulation callbacks for integrating with the training loop
#[derive(Debug)]
pub struct GradientAccumulationCallback<
    F: Float + Debug + ScalarOperand + Send + Sync + FromPrimitive,
> {
    /// Gradient accumulator
    pub accumulator: GradientAccumulator<F>,
    /// Current epoch
    current_epoch: usize,
    /// Current batch
    current_batch: usize,
    /// Total batches in epoch
    total_batches: usize,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + FromPrimitive>
    GradientAccumulationCallback<F>
{
    /// Create a new gradient accumulation callback
    pub fn new(config: GradientAccumulationConfig) -> Self {
        Self {
            accumulator: GradientAccumulator::new(config),
            current_epoch: 0,
            current_batch: 0,
            total_batches: 0,
        }
    }

    /// Initialize the accumulator for a new epoch
    pub fn on_epoch_begin<L: Layer<F>>(
        &mut self,
        model: &L,
        epoch: usize,
        total_batches: usize,
    ) -> Result<()> {
        self.accumulator.initialize(model)?;
        self.current_epoch = epoch;
        self.current_batch = 0;
        self.total_batches = total_batches;

        Ok(())
    }

    /// Process a batch with gradient accumulation
    pub fn on_batch<L: Layer<F>>(
        &mut self,
        model: &mut L,
        inputs: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
        loss_fn: &dyn Loss<F>,
    ) -> Result<F> {
        let loss = self
            .accumulator
            .accumulate_gradients(model, inputs, targets, loss_fn)?;
        self.current_batch += 1;

        Ok(loss)
    }

    /// Apply accumulated gradients if it's time to update
    pub fn on_batch_end<L: Layer<F>, O: Optimizer<F>>(
        &mut self,
        model: &mut L,
        optimizer: &mut O,
    ) -> Result<bool> {
        // Check if it's time to update or if it's the last batch
        let is_last_batch = self.current_batch == self.total_batches;

        if self.accumulator.should_update() || is_last_batch {
            // Apply accumulated gradients
            self.accumulator.apply_gradients(model, optimizer)?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Get gradient statistics
    pub fn get_gradient_stats(&self) -> Option<&GradientStats<F>> {
        self.accumulator.get_gradient_stats()
    }
}

/// Gradient accumulation trainer for simplified training
pub struct GradientAccumulationTrainer<
    F: Float + Debug + ScalarOperand + Send + Sync + FromPrimitive + std::fmt::Display,
> {
    /// Configuration for gradient accumulation
    pub config: GradientAccumulationConfig,
    /// Model to train
    model: Box<dyn Layer<F> + Send + Sync>,
    /// Optimizer for parameter updates
    optimizer: Box<dyn Optimizer<F> + Send + Sync>,
    /// Loss function
    loss_fn: Box<dyn Loss<F> + Send + Sync>,
    /// Gradient accumulator
    accumulator: GradientAccumulator<F>,
    /// Learning rate scheduler
    lr_scheduler: Option<Box<dyn crate::callbacks::LearningRateScheduler<F>>>,
    /// Batch size for training
    batch_size: usize,
    /// Whether to shuffle the data during training
    shuffle: bool,
    /// Verbosity level
    verbose: usize,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + FromPrimitive + std::fmt::Display>
    GradientAccumulationTrainer<F>
{
    /// Create a new gradient accumulation trainer
    pub fn new<L, O, LF>(
        model: L,
        optimizer: O,
        loss_fn: LF,
        config: GradientAccumulationConfig,
        batch_size: usize,
        shuffle: bool,
        verbose: usize,
    ) -> Self
    where
        L: Layer<F> + Send + Sync + 'static,
        O: Optimizer<F> + Send + Sync + 'static,
        LF: Loss<F> + Send + Sync + 'static,
    {
        // Clone the config for the accumulator first
        let config_clone = config.clone();

        Self {
            model: Box::new(model),
            optimizer: Box::new(optimizer),
            loss_fn: Box::new(loss_fn),
            accumulator: GradientAccumulator::new(config_clone),
            lr_scheduler: None,
            config,
            batch_size,
            shuffle,
            verbose,
        }
    }

    /// Set a learning rate scheduler
    pub fn set_lr_scheduler(
        &mut self,
        scheduler: Box<dyn crate::callbacks::LearningRateScheduler<F>>,
    ) {
        self.lr_scheduler = Some(scheduler);
    }

    /// Train the model for a specified number of epochs
    pub fn train(
        &mut self,
        train_dataset: &dyn Dataset<F>,
        validation_dataset: Option<&dyn Dataset<F>>,
        epochs: usize,
    ) -> Result<Vec<F>> {
        // Initialize history
        let mut history = Vec::with_capacity(epochs);

        // Ensure accumulator is initialized
        self.accumulator.initialize(&*self.model)?;

        // Train for specified number of epochs
        for epoch in 0..epochs {
            if self.verbose > 0 {
                println!("Epoch {}/{}", epoch + 1, epochs);
            }

            // Create data loader
            let data_loader = DataLoader::new(
                train_dataset.box_clone(),
                self.batch_size,
                self.shuffle,
                false,
            );
            let total_batches = data_loader.len();

            let mut total_loss = F::zero();
            let mut batch_count = 0;

            // Train on batches
            for (batch_idx, batch_result) in data_loader.enumerate() {
                let (inputs, targets) = batch_result?;
                // Accumulate gradients
                let loss = self.accumulator.accumulate_gradients(
                    &mut *self.model,
                    &inputs,
                    &targets,
                    &*self.loss_fn,
                )?;

                total_loss = total_loss + loss;
                batch_count += 1;

                // If it's time to update parameters or it's the last batch
                if self.accumulator.should_update() || batch_idx + 1 == total_batches {
                    // Apply accumulated gradients
                    self.accumulator
                        .apply_gradients(&mut *self.model, &mut *self.optimizer)?;

                    // Update learning rate if scheduler is configured
                    if let Some(ref mut scheduler) = self.lr_scheduler {
                        let current_batch = batch_idx + 1;
                        let progress = current_batch as f64 / total_batches as f64;

                        let lr = scheduler.get_learning_rate(progress)?;
                        self.optimizer.set_learning_rate(lr);
                    }
                }

                // Print progress
                if self.verbose > 0 && (batch_idx + 1) % 10 == 0 {
                    let avg_loss = if batch_count > 0 {
                        total_loss / F::from(batch_count).unwrap()
                    } else {
                        F::zero()
                    };

                    println!(
                        "Batch {}/{} - loss: {:.4}",
                        batch_idx + 1,
                        total_batches,
                        avg_loss
                    );
                }
            }

            // Compute average loss for the epoch
            let avg_loss = if batch_count > 0 {
                total_loss / F::from(batch_count).unwrap()
            } else {
                F::zero()
            };

            // Validate if a validation dataset is provided
            if let Some(val_dataset) = validation_dataset {
                // Evaluate on validation set
                let val_loss = self.evaluate(val_dataset)?;

                if self.verbose > 0 {
                    println!(
                        "Epoch {}/{} - loss: {:.4} - val_loss: {:.4}",
                        epoch + 1,
                        epochs,
                        avg_loss,
                        val_loss
                    );
                }
            } else if self.verbose > 0 {
                println!("Epoch {}/{} - loss: {:.4}", epoch + 1, epochs, avg_loss);
            }

            // Save loss to history
            history.push(avg_loss);
        }

        Ok(history)
    }

    /// Evaluate the model on a dataset
    pub fn evaluate(&mut self, dataset: &dyn Dataset<F>) -> Result<F> {
        // Create data loader
        let data_loader = DataLoader::new(dataset.box_clone(), self.batch_size, false, false);

        let mut total_loss = F::zero();
        let mut batch_count = 0;

        // Set model to evaluation mode
        self.model.set_training(false);

        // Evaluate on batches
        for batch_result in data_loader {
            let (inputs, targets) = batch_result?;
            // Forward pass
            let outputs = self.model.forward(&inputs)?;

            // Compute loss
            let loss = self.loss_fn.forward(&outputs, &targets)?;

            total_loss = total_loss + loss;
            batch_count += 1;
        }

        // Restore model to training mode
        self.model.set_training(true);

        // Return average loss
        if batch_count > 0 {
            Ok(total_loss / F::from(batch_count).unwrap())
        } else {
            Ok(F::zero())
        }
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
}
