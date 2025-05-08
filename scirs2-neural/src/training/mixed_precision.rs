//! Mixed Precision Training utilities
//!
//! This module provides utilities for mixed precision training,
//! which allows using lower precision floating-point formats (like f16)
//! during forward and backward passes, while keeping master weights in
//! higher precision (like f32 or f64) for stability.

use crate::data::{DataLoader, Dataset};
use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use crate::losses::Loss;
use crate::optimizers::{Optimizer, OptimizerExt};

use ndarray::ScalarOperand;
use ndarray::{Array, IxDyn};
use num_traits::Float;
use std::fmt::Debug;

/// Mixed precision training configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Whether to use dynamic loss scaling
    pub dynamic_loss_scaling: bool,
    /// Initial loss scale (usually a power of 2, like 2^16)
    pub initial_loss_scale: f64,
    /// Scale factor to increase loss scale after successful steps
    pub scale_factor: f64,
    /// Number of consecutive successful steps before increasing loss scale
    pub scale_window: usize,
    /// Minimum loss scale
    pub min_loss_scale: f64,
    /// Maximum loss scale
    pub max_loss_scale: f64,
    /// Whether to enable verbosity during training
    pub verbose: bool,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            dynamic_loss_scaling: true,
            initial_loss_scale: 65536.0, // 2^16
            scale_factor: 2.0,
            scale_window: 2000,
            min_loss_scale: 1.0,
            max_loss_scale: 2_f64.powi(24),
            verbose: false,
        }
    }
}

/// Mixed precision training manager
pub struct MixedPrecisionManager<
    F: Float + Debug + ScalarOperand + num_traits::FromPrimitive + Send + Sync + std::fmt::Display,
    LF: Float + Debug + ScalarOperand + Send + Sync,
> {
    /// Configuration for mixed precision training
    pub config: MixedPrecisionConfig,
    /// Current loss scale
    loss_scale: F,
    /// Count of consecutive successful steps
    success_counter: usize,
    /// Whether the last step was skipped due to overflow
    overflow_occurred: bool,
    /// Master model keeping weights in high precision
    master_model: Option<Box<dyn Layer<F> + Send + Sync>>,
    /// Cache for low precision gradients
    low_precision_gradients: Vec<Array<LF, IxDyn>>,
    /// Cache for high precision gradients
    high_precision_gradients: Vec<Array<F, IxDyn>>,
}

impl<
        F: Float + Debug + ScalarOperand + num_traits::FromPrimitive + Send + Sync + std::fmt::Display,
        LF: Float + Debug + ScalarOperand + Send + Sync,
    > MixedPrecisionManager<F, LF>
{
    /// Create a new mixed precision manager
    pub fn new(config: MixedPrecisionConfig) -> Self {
        // Extract the initial_loss_scale before moving config
        let initial_loss_scale = config.initial_loss_scale;

        Self {
            loss_scale: F::from(initial_loss_scale).unwrap(),
            success_counter: 0,
            overflow_occurred: false,
            master_model: None,
            low_precision_gradients: Vec::new(),
            high_precision_gradients: Vec::new(),
            config,
        }
    }

    /// Initialize the manager with a model
    pub fn initialize<L: Layer<F> + Clone + Send + Sync + 'static + ?Sized>(
        &mut self,
        model: &L,
    ) -> Result<()> {
        // Create a copy of the model as the master model
        self.master_model = Some(Box::new(model.clone()));

        // Initialize gradient caches
        let params = model.params();

        self.high_precision_gradients = params
            .iter()
            .map(|param| Array::<F, _>::zeros(param.raw_dim()))
            .collect();

        self.low_precision_gradients = params
            .iter()
            .map(|param| {
                // Convert shape to low precision
                let low_precision_shape = param.raw_dim();
                Array::<LF, _>::zeros(low_precision_shape)
            })
            .collect();

        Ok(())
    }

    /// Convert a model's parameters to low precision
    pub fn to_low_precision<L: Layer<F> + ?Sized, L2: Layer<LF> + ?Sized>(
        &self,
        high_precision_model: &L,
        low_precision_model: &mut L2,
    ) -> Result<()> {
        // Get master parameters
        let master_params = high_precision_model.params();

        // Convert to low precision
        let low_precision_params: Vec<Array<LF, IxDyn>> = master_params
            .iter()
            .map(|param| {
                // Convert values to low precision
                let mut low_param = Array::<LF, _>::zeros(param.raw_dim());

                for (low_val, &high_val) in low_param.iter_mut().zip(param.iter()) {
                    *low_val = LF::from(high_val).unwrap();
                }

                low_param
            })
            .collect();

        // Set parameters in low precision model
        low_precision_model.set_params(&low_precision_params)?;

        Ok(())
    }

    /// Convert gradients from low to high precision
    pub fn gradients_to_high_precision(
        &mut self,
        low_precision_gradients: &[Array<LF, IxDyn>],
    ) -> Result<Vec<Array<F, IxDyn>>> {
        // Initialize high precision gradients if needed
        if self.high_precision_gradients.is_empty() {
            self.high_precision_gradients = low_precision_gradients
                .iter()
                .map(|grad| {
                    // Convert shape to high precision
                    let high_precision_shape = grad.raw_dim();
                    Array::<F, _>::zeros(high_precision_shape)
                })
                .collect();
        }

        // Check for matching shapes
        if low_precision_gradients.len() != self.high_precision_gradients.len() {
            return Err(NeuralError::ValidationError(format!(
                "Expected {} gradients, got {}",
                self.high_precision_gradients.len(),
                low_precision_gradients.len()
            )));
        }

        // Convert gradients
        for (i, low_grad) in low_precision_gradients.iter().enumerate() {
            let high_grad = &mut self.high_precision_gradients[i];

            // Check for matching shapes
            if low_grad.shape() != high_grad.shape() {
                return Err(NeuralError::ValidationError(format!(
                    "Gradient shape mismatch at index {}: expected {:?}, got {:?}",
                    i,
                    high_grad.shape(),
                    low_grad.shape()
                )));
            }

            // Convert values
            for (high_val, &low_val) in high_grad.iter_mut().zip(low_grad.iter()) {
                *high_val = F::from(low_val).unwrap();
            }
        }

        Ok(self.high_precision_gradients.clone())
    }

    /// Check for overflow in gradients
    fn check_overflow(&self, gradients: &[Array<LF, IxDyn>]) -> bool {
        for grad in gradients {
            for &val in grad.iter() {
                if val.is_nan() || val.is_infinite() {
                    return true;
                }
            }
        }

        false
    }

    /// Update the loss scale based on overflow detection
    fn update_loss_scale(&mut self, overflow: bool) {
        if !self.config.dynamic_loss_scaling {
            return;
        }

        if overflow {
            // Overflow occurred, reduce loss scale
            let new_scale = self.loss_scale / F::from(self.config.scale_factor).unwrap();
            self.loss_scale = new_scale.max(F::from(self.config.min_loss_scale).unwrap());
            self.success_counter = 0;

            if self.config.verbose {
                println!(
                    "Overflow detected! Reducing loss scale to {}",
                    self.loss_scale
                );
            }
        } else {
            // No overflow, increment success counter
            self.success_counter += 1;

            // If enough consecutive successes, increase loss scale
            if self.success_counter >= self.config.scale_window {
                let new_scale = self.loss_scale * F::from(self.config.scale_factor).unwrap();
                self.loss_scale = new_scale.min(F::from(self.config.max_loss_scale).unwrap());
                self.success_counter = 0;

                if self.config.verbose {
                    println!("Increasing loss scale to {}", self.loss_scale);
                }
            }
        }
    }

    /// Train a step with mixed precision
    pub fn train_step<LM: Layer<LF> + ?Sized, HM: Layer<F> + ?Sized, O: Optimizer<F>>(
        &mut self,
        low_precision_model: &mut LM,
        high_precision_model: &mut HM,
        optimizer: &mut O,
        inputs: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
        loss_fn: &dyn Loss<F>,
    ) -> Result<F> {
        // Ensure models are in training mode
        low_precision_model.set_training(true);
        high_precision_model.set_training(true);

        // Sync low precision model from high precision
        self.to_low_precision(high_precision_model, low_precision_model)?;

        // Convert inputs to low precision
        let low_inputs = convert_array::<F, LF>(inputs)?;
        let _low_targets = convert_array::<F, LF>(targets)?;

        // Forward pass in low precision
        let low_outputs = low_precision_model.forward(&low_inputs)?;

        // Convert outputs to high precision for loss computation
        let high_outputs = convert_array::<LF, F>(&low_outputs)?;

        // Compute loss in high precision
        let loss = loss_fn.forward(&high_outputs, targets)?;
        let mut grad_output = loss_fn.backward(&high_outputs, targets)?;

        // Scale gradient for numerical stability
        grad_output = grad_output.clone() * self.loss_scale;

        // Convert grad_output to low precision
        let low_grad_output = convert_array::<F, LF>(&grad_output)?;

        // Backward pass in low precision
        let _low_grad_input = low_precision_model.backward(&low_inputs, &low_grad_output)?;

        // Get low precision gradients
        let low_gradients = low_precision_model.gradients();

        // Check for overflow in gradients
        let overflow = self.check_overflow(&low_gradients);
        self.overflow_occurred = overflow;

        // Update loss scale
        self.update_loss_scale(overflow);

        if overflow {
            // Skip parameter update if overflow occurred
            return Ok(loss);
        }

        // Convert gradients to high precision
        let high_gradients = self.gradients_to_high_precision(&low_gradients)?;

        // Unscale gradients
        let unscaled_gradients: Vec<Array<F, IxDyn>> = high_gradients
            .into_iter()
            .map(|grad| grad / self.loss_scale)
            .collect();

        // Apply gradients to high precision model
        high_precision_model.set_gradients(&unscaled_gradients)?;

        // Update high precision model
        optimizer.step(high_precision_model)?;

        // Sync low precision model
        self.to_low_precision(high_precision_model, low_precision_model)?;

        Ok(loss)
    }

    /// Get the current loss scale
    pub fn get_loss_scale(&self) -> F {
        self.loss_scale
    }

    /// Get the number of consecutive successful steps
    pub fn get_success_counter(&self) -> usize {
        self.success_counter
    }

    /// Check if overflow occurred in the last step
    pub fn overflow_occurred(&self) -> bool {
        self.overflow_occurred
    }

    /// Train for one epoch with mixed precision
    pub fn train_epoch<
        LM: Layer<LF> + ?Sized,
        HM: Layer<F> + ?Sized,
        O: Optimizer<F>,
        D: Dataset<F> + Clone,
    >(
        &mut self,
        low_precision_model: &mut LM,
        high_precision_model: &mut HM,
        optimizer: &mut O,
        dataset: &D,
        loss_fn: &dyn Loss<F>,
        batch_size: usize,
        shuffle: bool,
    ) -> Result<(F, usize)> {
        // Create data loader
        let mut data_loader = DataLoader::new(dataset.clone(), batch_size, shuffle, false);

        let mut total_loss = F::zero();
        let mut batch_count = 0;

        // Train on batches
        while let Some(batch_result) = data_loader.next() {
            let (inputs, targets) = batch_result?;
            // Train step with mixed precision
            let loss = self.train_step(
                low_precision_model,
                high_precision_model,
                optimizer,
                &inputs,
                &targets,
                loss_fn,
            )?;

            // If the step was successful (no overflow), add to total loss
            if !self.overflow_occurred {
                total_loss = total_loss + loss;
                batch_count += 1;
            }
        }

        // Return average loss
        if batch_count > 0 {
            Ok((total_loss / F::from(batch_count).unwrap(), batch_count))
        } else {
            Ok((F::zero(), 0))
        }
    }
}

/// Convert array from one precision to another
fn convert_array<F1: Float + Debug + ScalarOperand, F2: Float + Debug + ScalarOperand>(
    array: &Array<F1, IxDyn>,
) -> Result<Array<F2, IxDyn>> {
    let mut result = Array::<F2, _>::zeros(array.raw_dim());

    for (dest, &src) in result.iter_mut().zip(array.iter()) {
        *dest = F2::from(src).unwrap();
    }

    Ok(result)
}

/// Wrapper for mixed precision model
pub struct MixedPrecisionModel<
    F: Float + Debug + ScalarOperand + num_traits::FromPrimitive + Send + Sync + std::fmt::Display,
    LF: Float + Debug + ScalarOperand + Send + Sync,
> {
    /// High precision model
    pub high_precision_model: Box<dyn Layer<F> + Send + Sync>,
    /// Low precision model
    pub low_precision_model: Box<dyn Layer<LF> + Send + Sync>,
    /// Mixed precision manager
    pub manager: MixedPrecisionManager<F, LF>,
    /// Whether the model is in training mode
    is_training: bool,
}

impl<
        F: Float + Debug + ScalarOperand + num_traits::FromPrimitive + Send + Sync + std::fmt::Display,
        LF: Float + Debug + ScalarOperand + Send + Sync,
    > MixedPrecisionModel<F, LF>
{
    /// Create a new mixed precision model
    pub fn new<HM, LM>(
        high_precision_model: HM,
        mut low_precision_model: LM,
        config: MixedPrecisionConfig,
    ) -> Result<Self>
    where
        HM: Layer<F> + Send + Sync + Clone + 'static,
        LM: Layer<LF> + Send + Sync + 'static,
    {
        let mut manager = MixedPrecisionManager::new(config);

        // Initialize manager
        manager.initialize(&high_precision_model)?;

        // Sync low precision model
        manager.to_low_precision(&high_precision_model, &mut low_precision_model)?;

        Ok(Self {
            high_precision_model: Box::new(high_precision_model),
            low_precision_model: Box::new(low_precision_model),
            manager,
            is_training: true,
        })
    }

    /// Train a step with mixed precision
    pub fn train_step<O: Optimizer<F>>(
        &mut self,
        optimizer: &mut O,
        inputs: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
        loss_fn: &dyn Loss<F>,
    ) -> Result<F> {
        self.manager.train_step(
            &mut *self.low_precision_model,
            &mut *self.high_precision_model,
            optimizer,
            inputs,
            targets,
            loss_fn,
        )
    }

    /// Forward pass that automatically selects model based on training mode
    pub fn forward(&self, inputs: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        if self.is_training {
            // Convert inputs to low precision
            let low_inputs = convert_array::<F, LF>(inputs)?;

            // Forward pass in low precision
            let low_outputs = self.low_precision_model.forward(&low_inputs)?;

            // Convert outputs back to high precision
            convert_array::<LF, F>(&low_outputs)
        } else {
            // In evaluation mode, use high precision
            self.high_precision_model.forward(inputs)
        }
    }

    /// Set training mode for both models
    pub fn set_training(&mut self, training: bool) {
        self.is_training = training;
        self.high_precision_model.set_training(training);
        self.low_precision_model.set_training(training);
    }

    /// Get the high precision model
    pub fn get_high_precision_model(&self) -> &dyn Layer<F> {
        &*self.high_precision_model
    }

    /// Get the high precision model (mutable)
    pub fn get_high_precision_model_mut(&mut self) -> &mut dyn Layer<F> {
        &mut *self.high_precision_model
    }

    /// Get the low precision model
    pub fn get_low_precision_model(&self) -> &dyn Layer<LF> {
        &*self.low_precision_model
    }

    /// Get the low precision model (mutable)
    pub fn get_low_precision_model_mut(&mut self) -> &mut dyn Layer<LF> {
        &mut *self.low_precision_model
    }

    /// Train for one epoch with mixed precision
    pub fn train_epoch<O: Optimizer<F>, D: Dataset<F> + Clone>(
        &mut self,
        optimizer: &mut O,
        dataset: &D,
        loss_fn: &dyn Loss<F>,
        batch_size: usize,
        shuffle: bool,
    ) -> Result<(F, usize)> {
        self.manager.train_epoch(
            &mut *self.low_precision_model,
            &mut *self.high_precision_model,
            optimizer,
            dataset,
            loss_fn,
            batch_size,
            shuffle,
        )
    }
}

/// Mixed precision optimizer that wraps a standard optimizer
pub struct MixedPrecisionOptimizer<
    F: Float + Debug + ScalarOperand + num_traits::FromPrimitive + Send + Sync + std::fmt::Display,
    LF: Float + Debug + ScalarOperand + Send + Sync,
    O: Optimizer<F>,
> {
    /// Base optimizer
    pub optimizer: O,
    /// Mixed precision manager
    pub manager: MixedPrecisionManager<F, LF>,
}

impl<
        F: Float + Debug + ScalarOperand + num_traits::FromPrimitive + Send + Sync + std::fmt::Display,
        LF: Float + Debug + ScalarOperand + Send + Sync,
        O: Optimizer<F>,
    > MixedPrecisionOptimizer<F, LF, O>
{
    /// Create a new mixed precision optimizer
    pub fn new(optimizer: O, config: MixedPrecisionConfig) -> Self {
        Self {
            optimizer,
            manager: MixedPrecisionManager::new(config),
        }
    }

    /// Initialize the mixed precision manager
    pub fn initialize<L: Layer<F> + Clone + Send + Sync + 'static + ?Sized>(
        &mut self,
        model: &L,
    ) -> Result<()> {
        self.manager.initialize(model)
    }

    /// Optimize a step with mixed precision
    pub fn step<LM: Layer<LF>, HM: Layer<F>>(
        &mut self,
        low_precision_model: &mut LM,
        high_precision_model: &mut HM,
        inputs: &Array<F, IxDyn>,
        targets: &Array<F, IxDyn>,
        loss_fn: &dyn Loss<F>,
    ) -> Result<F> {
        self.manager.train_step(
            low_precision_model,
            high_precision_model,
            &mut self.optimizer,
            inputs,
            targets,
            loss_fn,
        )
    }

    /// Get the base optimizer
    pub fn get_optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Get the base optimizer (mutable)
    pub fn get_optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Get the current loss scale
    pub fn get_loss_scale(&self) -> F {
        self.manager.get_loss_scale()
    }

    /// Check if overflow occurred in the last step
    pub fn overflow_occurred(&self) -> bool {
        self.manager.overflow_occurred()
    }
}
