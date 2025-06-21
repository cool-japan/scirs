//! Training utilities and infrastructure
//!
//! This module provides comprehensive utilities for training neural networks,
//! including advanced features like gradient accumulation, mixed precision training,
//! distributed training, and sophisticated training loop management.
//!
//! # Overview
//!
//! The training module consists of several key components:
//!
//! - **Trainer**: High-level training orchestrator that manages the entire training process
//! - **TrainingConfig**: Configuration structure for customizing training behavior
//! - **GradientAccumulator**: For accumulating gradients across multiple batches
//! - **MixedPrecisionManager**: For memory-efficient mixed precision training
//! - **ValidationSettings**: For configuring validation during training
//!
//! # Examples
//!
//! ## Basic Training Loop
//!
//! ```rust
//! use scirs2_neural::training::{Trainer, TrainingConfig, ValidationSettings};
//! use scirs2_neural::data::{DataLoader, Dataset};
//! use scirs2_neural::models::Sequential;
//! use scirs2_neural::layers::Dense;
//! use scirs2_neural::losses::CrossEntropyLoss;
//! use scirs2_neural::optimizers::Adam;
//! use scirs2_neural::callbacks::CallbackManager;
//! use rand::rngs::SmallRng;
//! use rand::SeedableRng;
//!
//! # fn train_model() -> scirs2_neural::error::Result<()> {
//! let mut rng = SmallRng::seed_from_u64(42);
//!
//! // Create a simple model
//! let mut model: Sequential<f32> = Sequential::new();
//! model.add_layer(Dense::new(784, 128, Some("relu"), &mut rng)?);
//! model.add_layer(Dense::new(128, 10, Some("softmax"), &mut rng)?);
//!
//! // Configure training
//! let config = TrainingConfig {
//!     batch_size: 32,
//!     epochs: 10,
//!     learning_rate: 0.001,
//!     shuffle: true,
//!     verbose: 1,
//!     validation: Some(ValidationSettings {
//!         enabled: true,
//!         validation_split: 0.2,
//!         batch_size: 32,
//!         num_workers: 1,
//!     }),
//!     ..Default::default()
//! };
//!
//! // Create trainer
//! // let trainer = Trainer::new(model, optimizer, loss_fn, config);
//!
//! // Set up data, loss, optimizer, and callbacks
//! // let train_loader = DataLoader::new(...);
//! // let loss_fn = CrossEntropyLoss::new();
//! // let optimizer = Adam::new(0.001);
//! // let callbacks = CallbackManager::new();
//!
//! // Train the model
//! // let history = trainer.fit(&mut model, &train_loader, &loss_fn, &mut optimizer, &mut callbacks)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced Training with Gradient Accumulation
//!
//! ```rust
//! use scirs2_neural::training::{TrainingConfig, GradientAccumulationConfig};
//!
//! let config = TrainingConfig {
//!     batch_size: 8,  // Smaller effective batch size
//!     gradient_accumulation: Some(GradientAccumulationConfig {
//!         accumulation_steps: 4,  // Accumulate over 4 steps (effective batch size: 32)
//!         clip_gradients: true,
//!         average_gradients: true,
//!         zero_gradients_after_update: true,
//!         max_gradient_norm: Some(1.0),
//!         log_gradient_stats: false,
//!     }),
//!     ..Default::default()
//! };
//! ```
//!
//! ## Mixed Precision Training
//!
//! ```rust
//! use scirs2_neural::training::{TrainingConfig, MixedPrecisionConfig};
//!
//! let config = TrainingConfig {
//!     mixed_precision: Some(MixedPrecisionConfig {
//!         dynamic_loss_scaling: true,
//!         initial_loss_scale: 1024.0,
//!         scale_factor: 2.0,
//!         scale_window: 2000,
//!         ..Default::default()
//!     }),
//!     ..Default::default()
//! };
//! ```

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

/// Configuration for neural network training
///
/// This structure contains all the parameters needed to configure a training session,
/// including batch size, learning rate, optimization settings, and advanced features
/// like gradient accumulation and mixed precision training.
///
/// # Examples
///
/// ## Basic Configuration
///
/// ```rust
/// use scirs2_neural::training::TrainingConfig;
///
/// let config = TrainingConfig {
///     batch_size: 64,
///     epochs: 20,
///     learning_rate: 0.001,
///     shuffle: true,
///     verbose: 1,
///     ..Default::default()
/// };
/// ```
///
/// ## Configuration for Large Models (with gradient accumulation)
///
/// ```rust
/// use scirs2_neural::training::{TrainingConfig, GradientAccumulationConfig};
///
/// let config = TrainingConfig {
///     batch_size: 8,  // Small batch due to memory constraints
///     gradient_accumulation: Some(GradientAccumulationConfig {
///         accumulation_steps: 8,  // Effective batch size: 64
///         clip_gradients: true,
///         average_gradients: true,
///         zero_gradients_after_update: true,
///         max_gradient_norm: Some(1.0),
///         log_gradient_stats: false,
///     }),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of samples in each training batch
    ///
    /// Larger batch sizes provide more stable gradients but require more memory.
    /// Typical values range from 16 to 512 depending on model size and hardware.
    pub batch_size: usize,

    /// Whether to shuffle the training data between epochs
    ///
    /// Shuffling helps prevent the model from learning the order of data presentation
    /// and generally improves training stability and generalization.
    pub shuffle: bool,

    /// Number of parallel workers for data loading
    ///
    /// Setting this to 0 uses the main thread for data loading.
    /// Higher values can speed up training if data loading is a bottleneck.
    pub num_workers: usize,

    /// Base learning rate for the optimizer
    ///
    /// This is the step size used for parameter updates. Too high values can cause
    /// training instability, while too low values lead to slow convergence.
    /// Typical values range from 1e-5 to 1e-1 depending on the optimizer and model.
    pub learning_rate: f64,

    /// Number of complete passes through the training dataset
    ///
    /// One epoch means seeing each training example exactly once.
    /// More epochs allow for better learning but risk overfitting.
    pub epochs: usize,

    /// Verbosity level for training output
    ///
    /// - 0: Silent mode (no output)
    /// - 1: Progress bar with metrics (default)
    /// - 2: One line per epoch with detailed metrics
    pub verbose: usize,

    /// Validation configuration
    ///
    /// If provided, enables validation during training with the specified settings.
    /// Validation helps monitor overfitting and model generalization.
    pub validation: Option<ValidationSettings>,

    /// Gradient accumulation configuration
    ///
    /// Enables accumulating gradients over multiple batches before applying updates.
    /// Useful for simulating larger batch sizes when memory is limited.
    pub gradient_accumulation: Option<GradientAccumulationConfig>,

    /// Mixed precision training configuration
    ///
    /// Enables training with mixed precision (FP16/FP32) to reduce memory usage
    /// and potentially speed up training on compatible hardware.
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

/// Configuration for validation during training
///
/// Validation helps monitor model performance on unseen data during training,
/// which is crucial for detecting overfitting and ensuring good generalization.
///
/// # Examples
///
/// ```rust
/// use scirs2_neural::training::ValidationSettings;
///
/// let validation = ValidationSettings {
///     enabled: true,
///     validation_split: 0.2,  // Use 20% of data for validation
///     batch_size: 64,
///     num_workers: 2,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct ValidationSettings {
    /// Whether to enable validation during training
    ///
    /// When disabled, no validation will be performed even if this struct is provided.
    pub enabled: bool,

    /// Fraction of training data to use for validation (0.0 to 1.0)
    ///
    /// For example, 0.2 means 20% of the data will be used for validation
    /// and 80% for training. The data is split before training begins.
    pub validation_split: f64,

    /// Batch size for validation
    ///
    /// Can be larger than training batch size since no gradients are computed.
    /// Larger validation batches are more memory efficient and faster.
    pub batch_size: usize,

    /// Number of parallel workers for validation data loading
    ///
    /// Similar to training workers, but for validation data.
    /// Setting to 0 uses the main thread.
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

        // Initialize mixed precision if enabled
        if let Some(ref mp_config) = self.config.mixed_precision {
            if let Some(ref mut mixed_precision) = self.mixed_precision {
                // Initialize the mixed precision manager with the model
                // Since we need a clonable version of the model, we'll try to get it
                if let Some(high_precision_model) = self
                    .model
                    .as_any()
                    .downcast_ref::<crate::layers::Sequential<F>>()
                {
                    // For Sequential models, we can attempt initialization
                    if let Err(e) = mixed_precision.initialize(high_precision_model) {
                        eprintln!("Warning: Failed to initialize mixed precision: {}", e);
                    }
                } else {
                    // For other model types that don't implement Clone, we'll skip mixed precision
                    eprintln!("Warning: Mixed precision requires clonable models. Skipping mixed precision initialization.");
                }
            } else {
                // Create mixed precision manager if it doesn't exist
                let manager = MixedPrecisionManager::<F, F>::new(mp_config.clone());
                self.mixed_precision = Some(manager);

                // Try to initialize it
                if let Some(ref mut mixed_precision) = self.mixed_precision {
                    if let Some(high_precision_model) = self
                        .model
                        .as_any()
                        .downcast_ref::<crate::layers::Sequential<F>>()
                    {
                        if let Err(e) = mixed_precision.initialize(high_precision_model) {
                            eprintln!("Warning: Failed to initialize mixed precision: {}", e);
                        }
                    }
                }
            }
        }

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
