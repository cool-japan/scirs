//! Normalization layers implementation
//!
//! This module provides implementations of various normalization techniques
//! such as Layer Normalization, Batch Normalization, etc.

use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use ndarray::{Array, ArrayView, IxDyn, ScalarOperand};
use num_traits::{Float, One, Zero};
use rand::prelude::*;
use rand::Rng;
use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;

/// Layer Normalization layer
///
/// Implements layer normalization as described in "Layer Normalization"
/// by Ba, Kiros, and Hinton. It normalizes the inputs across the last dimension
/// and applies learnable scale and shift parameters.
///
/// # Examples
///
/// ```
/// use scirs2_neural::layers::{LayerNorm, Layer};
/// use ndarray::{Array, Array3};
/// use rand::rngs::SmallRng;
/// use rand::SeedableRng;
///
/// // Create a layer normalization layer for a 64-dimensional feature space
/// let mut rng = SmallRng::seed_from_u64(42);
/// let layer_norm = LayerNorm::new(64, 1e-5, &mut rng).unwrap();
///
/// // Forward pass with a batch of 2 samples, sequence length 3
/// let batch_size = 2;
/// let seq_len = 3;
/// let d_model = 64;
/// let input = Array3::<f64>::from_elem((batch_size, seq_len, d_model), 0.1).into_dyn();
/// let output = layer_norm.forward(&input).unwrap();
///
/// // Output shape should match input shape
/// assert_eq!(output.shape(), input.shape());
/// ```
pub struct LayerNorm<F: Float + Debug> {
    /// Dimensionality of the input features
    normalized_shape: Vec<usize>,
    /// Learnable scale parameter
    gamma: Array<F, IxDyn>,
    /// Learnable shift parameter
    beta: Array<F, IxDyn>,
    /// Gradient of gamma
    dgamma: Array<F, IxDyn>,
    /// Gradient of beta
    dbeta: Array<F, IxDyn>,
    /// Small constant for numerical stability
    eps: F,
    /// Input cache for backward pass
    input_cache: RefCell<Option<Array<F, IxDyn>>>,
    /// Normalized input cache for backward pass
    norm_cache: RefCell<Option<Array<F, IxDyn>>>,
    /// Mean cache for backward pass
    mean_cache: RefCell<Option<Array<F, IxDyn>>>,
    /// Variance cache for backward pass
    var_cache: RefCell<Option<Array<F, IxDyn>>>,
}

/// 2D Layer Normalization for 2D convolutional networks
pub struct LayerNorm2D<F: Float + Debug> {
    /// Number of channels to normalize
    channels: usize,
    /// Internal layer norm implementation
    layer_norm: LayerNorm<F>,
    /// Name for the layer
    name: Option<String>,
}

impl<F: Float + Debug + ScalarOperand + 'static> LayerNorm2D<F> {
    /// Create a new 2D layer normalization layer
    pub fn new<R: Rng>(channels: usize, eps: f64, name: Option<&str>) -> Result<Self> {
        let layer_norm = LayerNorm::new(channels, eps, &mut rand::rng())?;
        
        Ok(Self {
            channels,
            layer_norm,
            name: name.map(String::from),
        })
    }
    
    /// Get the number of channels
    pub fn channels(&self) -> usize {
        self.channels
    }
    
    /// Get the name of the layer
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> Layer<F> for LayerNorm2D<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // For 2D layer norm, we expect:
        // [batch_size, channels, height, width] format
        
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 4D input [batch_size, channels, height, width], got {:?}",
                input_shape
            )));
        }
        
        let (_batch_size, channels, _height, _width) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        
        if channels != self.channels {
            return Err(NeuralError::InferenceError(format!(
                "Expected {} channels but got {}",
                self.channels, channels
            )));
        }
        
        // Delegate to the internal layer norm
        self.layer_norm.forward(input)
    }
    
    fn backward(&self, input: &Array<F, IxDyn>, grad_output: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Delegate to the internal layer norm
        self.layer_norm.backward(input, grad_output)
    }
    
    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Delegate to the internal layer norm
        self.layer_norm.update(learning_rate)
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> LayerNorm<F> {
    /// Create a new layer normalization layer
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - Shape of the input features to normalize over
    /// * `eps` - Small constant added for numerical stability
    /// * `rng` - Random number generator for initialization
    ///
    /// # Returns
    ///
    /// * A new layer normalization layer
    pub fn new<R: Rng>(normalized_shape: usize, eps: f64, rng: &mut R) -> Result<Self> {
        // Initialize gamma to ones and beta to zeros
        let gamma = Array::<F, _>::from_elem(IxDyn(&[normalized_shape]), F::one());
        let beta = Array::<F, _>::from_elem(IxDyn(&[normalized_shape]), F::zero());

        // Initialize gradient arrays to zeros
        let dgamma = Array::<F, _>::zeros(IxDyn(&[normalized_shape]));
        let dbeta = Array::<F, _>::zeros(IxDyn(&[normalized_shape]));

        // Convert epsilon to F
        let eps = F::from(eps).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert epsilon to type F".to_string())
        })?;

        Ok(Self {
            normalized_shape: vec![normalized_shape],
            gamma,
            beta,
            dgamma,
            dbeta,
            eps,
            input_cache: RefCell::new(None),
            norm_cache: RefCell::new(None),
            mean_cache: RefCell::new(None),
            var_cache: RefCell::new(None),
        })
    }

    /// Helper method to compute mean and variance along the normalization axis
    fn compute_stats(
        &self,
        input: &ArrayView<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        let input_shape = input.shape();
        let ndim = input.ndim();

        if ndim < 1 {
            return Err(NeuralError::InferenceError(
                "Input must have at least 1 dimension".to_string(),
            ));
        }

        // Check if the last dimension matches the normalized shape
        let feat_dim = input_shape[ndim - 1];
        if feat_dim != self.normalized_shape[0] {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Last dimension of input ({}) must match normalized_shape ({})",
                feat_dim, self.normalized_shape[0]
            )));
        }

        // Compute the batch shape (all dimensions except the last one)
        let batch_shape: Vec<usize> = input_shape[..ndim - 1].to_vec();
        let batch_size: usize = batch_shape.iter().product();

        // Reshape input to 2D: [batch_size, features]
        let reshaped = input
            .to_owned()
            .into_shape((batch_size, feat_dim))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape input: {}", e)))?;

        // Initialize mean and variance arrays
        let mut mean = Array::<F, _>::zeros(IxDyn(&[batch_size, 1]));
        let mut var = Array::<F, _>::zeros(IxDyn(&[batch_size, 1]));

        // Compute mean for each sample
        for i in 0..batch_size {
            let mut sum = F::zero();
            for j in 0..feat_dim {
                sum = sum + reshaped[[i, j]];
            }
            mean[[i, 0]] = sum / F::from(feat_dim).unwrap();
        }

        // Compute variance for each sample
        for i in 0..batch_size {
            let mut sum_sq = F::zero();
            for j in 0..feat_dim {
                let diff = reshaped[[i, j]] - mean[[i, 0]];
                sum_sq = sum_sq + diff * diff;
            }
            var[[i, 0]] = sum_sq / F::from(feat_dim).unwrap();
        }

        Ok((mean, var))
    }

    /// Get the normalized shape
    pub fn normalized_shape(&self) -> usize {
        self.normalized_shape[0]
    }

    /// Get the epsilon value
    pub fn eps(&self) -> f64 {
        self.eps.to_f64().unwrap_or(1e-5)
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> Layer<F> for LayerNorm<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        self.input_cache.replace(Some(input.clone()));

        let input_view = input.view();
        let input_shape = input.shape();
        let ndim = input.ndim();

        // Compute mean and variance
        let (mean, var) = self.compute_stats(&input_view)?;

        // Cache mean and variance for backward pass
        self.mean_cache.replace(Some(mean.clone()));
        self.var_cache.replace(Some(var.clone()));

        // Reshape input to 2D: [batch_size, features]
        let feat_dim = input_shape[ndim - 1];
        let batch_shape: Vec<usize> = input_shape[..ndim - 1].to_vec();
        let batch_size: usize = batch_shape.iter().product();

        let reshaped = input
            .to_owned()
            .into_shape((batch_size, feat_dim))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape input: {}", e)))?;

        // Normalize the input
        let mut normalized = Array::<F, _>::zeros((batch_size, feat_dim));
        for i in 0..batch_size {
            for j in 0..feat_dim {
                let x_norm = (reshaped[[i, j]] - mean[[i, 0]]) / (var[[i, 0]] + self.eps).sqrt();
                normalized[[i, j]] = x_norm * self.gamma[[j]] + self.beta[[j]];
            }
        }

        // Cache normalized input for backward pass
        self.norm_cache.replace(Some(normalized.clone()));

        // Reshape back to the original shape
        let output = normalized
            .into_shape(input_shape)
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape output: {}", e)))?;

        Ok(output)
    }

    fn backward(&self, input: &Array<F, IxDyn>, grad_output: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Retrieve cached values
        let input_ref = self.input_cache.borrow();
        let norm_ref = self.norm_cache.borrow();
        let mean_ref = self.mean_cache.borrow();
        let var_ref = self.var_cache.borrow();

        if input_ref.is_none() || norm_ref.is_none() || mean_ref.is_none() || var_ref.is_none() {
            return Err(NeuralError::InferenceError(
                "No cached values for backward pass. Call forward() first.".to_string(),
            ));
        }

        let cached_input = input_ref.as_ref().unwrap();
        let x_norm = norm_ref.as_ref().unwrap();
        let mean = mean_ref.as_ref().unwrap();
        let var = var_ref.as_ref().unwrap();

        // Get dimensions
        let input_shape = input.shape();
        let ndim = input.ndim();
        let feat_dim = input_shape[ndim - 1];
        let batch_shape: Vec<usize> = input_shape[..ndim - 1].to_vec();
        let batch_size: usize = batch_shape.iter().product();

        // Reshape grad_output to 2D: [batch_size, features]
        let grad_output_reshaped = grad_output
            .to_owned()
            .into_shape((batch_size, feat_dim))
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape grad_output: {}", e))
            })?;

        // Reshape input to 2D: [batch_size, features]
        let input_reshaped = input
            .to_owned()
            .into_shape((batch_size, feat_dim))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape input: {}", e)))?;

        // In a real implementation, we would compute gradient updates for gamma and beta
        // For simplicity, this is just a placeholder that returns the gradient of the input

        // Create a placeholder gradient input
        let mut grad_input = Array::<F, _>::zeros((batch_size, feat_dim));

        // Reshape back to the original shape
        let output = grad_input.into_shape(input_shape).map_err(|e| {
            NeuralError::InferenceError(format!("Failed to reshape grad_input: {}", e))
        })?;

        Ok(output)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update parameters using gradients
        // This is a placeholder implementation

        // Apply a small update
        let small_change = F::from(0.001).unwrap();
        let lr = small_change * learning_rate;

        // Update gamma and beta
        for i in 0..self.normalized_shape[0] {
            self.gamma[[i]] = self.gamma[[i]] - lr;
            self.beta[[i]] = self.beta[[i]] - lr;
        }

        Ok(())
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> ParamLayer<F> for LayerNorm<F> {
    fn get_parameters(&self) -> Vec<&Array<F, ndarray::IxDyn>> {
        vec![&self.gamma, &self.beta]
    }

    fn get_gradients(&self) -> Vec<&Array<F, ndarray::IxDyn>> {
        vec![&self.dgamma, &self.dbeta]
    }

    fn set_parameters(&mut self, params: Vec<Array<F, ndarray::IxDyn>>) -> Result<()> {
        if params.len() != 2 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected 2 parameters, got {}",
                params.len()
            )));
        }

        if params[0].shape() != self.gamma.shape() {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Gamma shape mismatch: expected {:?}, got {:?}",
                self.gamma.shape(),
                params[0].shape()
            )));
        }

        if params[1].shape() != self.beta.shape() {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Beta shape mismatch: expected {:?}, got {:?}",
                self.beta.shape(),
                params[1].shape()
            )));
        }

        self.gamma = params[0].clone();
        self.beta = params[1].clone();

        Ok(())
    }
}

/// Batch Normalization layer
///
/// Implements batch normalization as described in "Batch Normalization: Accelerating Deep Network
/// Training by Reducing Internal Covariate Shift" by Ioffe and Szegedy.
///
/// This normalization is applied along the feature dimension (channel dimension for CNNs)
/// over a mini-batch of data.
///
/// # Examples
///
/// ```
/// use scirs2_neural::layers::{BatchNorm, Layer};
/// use ndarray::{Array, Array4};
/// use rand::rngs::SmallRng;
/// use rand::SeedableRng;
///
/// // Create a batch normalization layer for a 64-dimensional feature space
/// let mut rng = SmallRng::seed_from_u64(42);
/// let batch_norm = BatchNorm::new(64, 0.9, 1e-5, &mut rng).unwrap();
///
/// // Forward pass with a batch of 2 samples, 3 channels, 4x4 spatial dimensions
/// let batch_size = 2;
/// let channels = 3;
/// let height = 4;
/// let width = 4;
/// let input = Array4::<f64>::from_elem((batch_size, channels, height, width), 0.1).into_dyn();
/// let output = batch_norm.forward(&input).unwrap();
///
/// // Output shape should match input shape
/// assert_eq!(output.shape(), input.shape());
/// ```
pub struct BatchNorm<F: Float + Debug> {
    /// Number of features (channels)
    num_features: usize,
    /// Learnable scale parameter
    gamma: Array<F, IxDyn>,
    /// Learnable shift parameter
    beta: Array<F, IxDyn>,
    /// Gradient of gamma
    dgamma: Array<F, IxDyn>,
    /// Gradient of beta
    dbeta: Array<F, IxDyn>,
    /// Running mean for inference mode
    running_mean: Array<F, IxDyn>,
    /// Running variance for inference mode
    running_var: Array<F, IxDyn>,
    /// Momentum for running statistics updates
    momentum: F,
    /// Small constant for numerical stability
    eps: F,
    /// Whether we're in training mode
    training: bool,
    /// Input cache for backward pass
    input_cache: RefCell<Option<Array<F, IxDyn>>>,
    /// Input shape cache for backward pass
    input_shape_cache: RefCell<Option<Vec<usize>>>,
    /// Batch mean cache for backward pass
    batch_mean_cache: RefCell<Option<Array<F, IxDyn>>>,
    /// Batch var cache for backward pass
    batch_var_cache: RefCell<Option<Array<F, IxDyn>>>,
    /// Normalized input cache for backward pass
    norm_cache: RefCell<Option<Array<F, IxDyn>>>,
    /// Std deviation cache for backward pass
    std_cache: RefCell<Option<Array<F, IxDyn>>>,
    /// Phantom for type storage
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + ScalarOperand + 'static> BatchNorm<F> {
    /// Create a new batch normalization layer
    ///
    /// # Arguments
    ///
    /// * `num_features` - Number of features/channels to normalize
    /// * `momentum` - Momentum for running mean/variance updates (default: 0.9)
    /// * `eps` - Small constant for numerical stability (default: 1e-5)
    /// * `rng` - Random number generator for initialization
    ///
    /// # Returns
    ///
    /// * A new batch normalization layer
    pub fn new<R: Rng>(num_features: usize, momentum: f64, eps: f64, rng: &mut R) -> Result<Self> {
        // Initialize gamma to ones and beta to zeros
        let gamma = Array::<F, _>::from_elem(IxDyn(&[num_features]), F::one());
        let beta = Array::<F, _>::from_elem(IxDyn(&[num_features]), F::zero());

        // Initialize gradient arrays to zeros
        let dgamma = Array::<F, _>::zeros(IxDyn(&[num_features]));
        let dbeta = Array::<F, _>::zeros(IxDyn(&[num_features]));

        // Initialize running statistics to zeros
        let running_mean = Array::<F, _>::zeros(IxDyn(&[num_features]));
        let running_var = Array::<F, _>::from_elem(IxDyn(&[num_features]), F::one());

        // Convert momentum and epsilon to F
        let momentum = F::from(momentum).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert momentum to type F".to_string())
        })?;
        let eps = F::from(eps).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert epsilon to type F".to_string())
        })?;

        Ok(Self {
            num_features,
            gamma,
            beta,
            dgamma,
            dbeta,
            running_mean,
            running_var,
            momentum,
            eps,
            training: true,
            input_cache: RefCell::new(None),
            input_shape_cache: RefCell::new(None),
            batch_mean_cache: RefCell::new(None),
            batch_var_cache: RefCell::new(None),
            norm_cache: RefCell::new(None),
            std_cache: RefCell::new(None),
            _phantom: PhantomData,
        })
    }

    /// Set the training mode
    ///
    /// In training mode, batch statistics are used for normalization and running statistics are updated.
    /// In inference mode, running statistics are used for normalization and not updated.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Get the number of features
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Get the momentum value
    pub fn momentum(&self) -> f64 {
        self.momentum.to_f64().unwrap_or(0.9)
    }

    /// Get the epsilon value
    pub fn eps(&self) -> f64 {
        self.eps.to_f64().unwrap_or(1e-5)
    }

    /// Get the training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Helper function to reshape input for batch normalization
    /// The input should be reshaped to (N, C, -1) where:
    /// - N is the batch size
    /// - C is the number of channels/features
    /// - -1 flattens all other dimensions
    fn reshape_input(&self, input: &Array<F, IxDyn>) -> Result<(Array<F, IxDyn>, Vec<usize>)> {
        let input_shape = input.shape().to_vec();
        let ndim = input.ndim();

        if ndim < 2 {
            return Err(NeuralError::InvalidArchitecture(
                "Input must have at least 2 dimensions (batch, features, ...)".to_string(),
            ));
        }

        // For 2D inputs, assume shape is (batch_size, features)
        // For 3D+ inputs (e.g. CNN activations), assume shape is (batch_size, channels, dim1, dim2, ...)
        let batch_size = input_shape[0];
        let num_features = input_shape[1];

        if num_features != self.num_features {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected {} features, got {}",
                self.num_features, num_features
            )));
        }

        // Calculate the product of all spatial dimensions (if any)
        let spatial_size: usize = if ndim > 2 {
            input_shape[2..].iter().product()
        } else {
            1
        };

        // Reshape to (batch_size, num_features, spatial_size)
        let reshaped = input
            .clone()
            .into_shape((batch_size, num_features, spatial_size))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape input: {}", e)))?;

        Ok((reshaped, input_shape))
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> Layer<F> for BatchNorm<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        self.input_cache.replace(Some(input.clone()));

        // Reshape input to (batch_size, num_features, spatial_size)
        let (reshaped, input_shape) = self.reshape_input(input)?;
        self.input_shape_cache.replace(Some(input_shape.clone()));

        let batch_size = reshaped.shape()[0];
        let num_features = reshaped.shape()[1];
        let spatial_size = reshaped.shape()[2];

        // Create output with same shape as reshaped input
        let mut normalized = Array::<F, _>::zeros(reshaped.shape());

        if self.training {
            // Calculate batch mean and variance
            let mut batch_mean = Array::<F, _>::zeros(IxDyn(&[num_features]));
            let mut batch_var = Array::<F, _>::zeros(IxDyn(&[num_features]));

            // Compute mean across batch and spatial dimensions for each feature
            for c in 0..num_features {
                let mut sum = F::zero();
                let spatial_elements = batch_size * spatial_size;

                for n in 0..batch_size {
                    for s in 0..spatial_size {
                        sum = sum + reshaped[[n, c, s]];
                    }
                }

                batch_mean[[c]] = sum / F::from(spatial_elements).unwrap();
            }

            // Compute variance across batch and spatial dimensions for each feature
            for c in 0..num_features {
                let mut sum_sq = F::zero();
                let spatial_elements = batch_size * spatial_size;

                for n in 0..batch_size {
                    for s in 0..spatial_size {
                        let diff = reshaped[[n, c, s]] - batch_mean[[c]];
                        sum_sq = sum_sq + diff * diff;
                    }
                }

                batch_var[[c]] = sum_sq / F::from(spatial_elements).unwrap();
            }

            // Cache batch statistics for backward pass
            self.batch_mean_cache.replace(Some(batch_mean.clone()));
            self.batch_var_cache.replace(Some(batch_var.clone()));

            // Compute standard deviation for normalization
            let mut std_dev = batch_var.mapv(|x| (x + self.eps).sqrt());
            self.std_cache.replace(Some(std_dev.clone()));

            // Normalize using batch statistics
            for n in 0..batch_size {
                for c in 0..num_features {
                    for s in 0..spatial_size {
                        let x_norm = (reshaped[[n, c, s]] - batch_mean[[c]]) / std_dev[[c]];
                        normalized[[n, c, s]] = x_norm * self.gamma[[c]] + self.beta[[c]];
                    }
                }
            }

            // Update running statistics
            // running_mean = momentum * running_mean + (1 - momentum) * batch_mean
            // running_var = momentum * running_var + (1 - momentum) * batch_var
            let one = F::one();
            {
                let mut running_mean_mut = RefCell::borrow_mut(&self.running_mean.as_view().cell());
                let mut running_var_mut = RefCell::borrow_mut(&self.running_var.as_view().cell());

                for c in 0..num_features {
                    running_mean_mut[[c]] = self.momentum * running_mean_mut[[c]]
                        + (one - self.momentum) * batch_mean[[c]];
                    running_var_mut[[c]] = self.momentum * running_var_mut[[c]]
                        + (one - self.momentum) * batch_var[[c]];
                }
            }

            // Cache normalized input (pre-gamma/beta) for backward pass
            let mut x_norm = Array::<F, _>::zeros(reshaped.shape());
            for n in 0..batch_size {
                for c in 0..num_features {
                    for s in 0..spatial_size {
                        x_norm[[n, c, s]] = (reshaped[[n, c, s]] - batch_mean[[c]]) / std_dev[[c]];
                    }
                }
            }
            self.norm_cache.replace(Some(x_norm));
        } else {
            // Use running statistics in inference mode
            let std_dev = self.running_var.mapv(|x| (x + self.eps).sqrt());

            // Normalize using running statistics
            for n in 0..batch_size {
                for c in 0..num_features {
                    for s in 0..spatial_size {
                        let x_norm = (reshaped[[n, c, s]] - self.running_mean[[c]]) / std_dev[[c]];
                        normalized[[n, c, s]] = x_norm * self.gamma[[c]] + self.beta[[c]];
                    }
                }
            }
        }

        // Reshape back to the original shape
        let output = normalized
            .into_shape(input_shape)
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape output: {}", e)))?;

        Ok(output)
    }

    fn backward(&self, input: &Array<F, IxDyn>, grad_output: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Retrieve cached values
        let input_ref = self.input_cache.borrow();
        let input_shape_ref = self.input_shape_cache.borrow();
        let batch_mean_ref = self.batch_mean_cache.borrow();
        let batch_var_ref = self.batch_var_cache.borrow();
        let norm_ref = self.norm_cache.borrow();
        let std_ref = self.std_cache.borrow();

        if input_ref.is_none()
            || input_shape_ref.is_none()
            || batch_mean_ref.is_none()
            || batch_var_ref.is_none()
            || norm_ref.is_none()
            || std_ref.is_none()
        {
            return Err(NeuralError::InferenceError(
                "No cached values for backward pass. Call forward() first.".to_string(),
            ));
        }

        let cached_input = input_ref.as_ref().unwrap();
        let input_shape = input_shape_ref.as_ref().unwrap();
        let batch_mean = batch_mean_ref.as_ref().unwrap();
        let batch_var = batch_var_ref.as_ref().unwrap();
        let x_norm = norm_ref.as_ref().unwrap();
        let std_dev = std_ref.as_ref().unwrap();

        // Reshape grad_output to match the cached reshaped input
        let reshaped_grad_output = grad_output
            .clone()
            .into_shape(x_norm.shape())
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape grad_output: {}", e))
            })?;

        let batch_size = x_norm.shape()[0];
        let num_features = x_norm.shape()[1];
        let spatial_size = x_norm.shape()[2];
        let spatial_elements = batch_size * spatial_size;
        let spatial_elements_f = F::from(spatial_elements).unwrap();

        // Calculate gradients for gamma and beta
        let mut dgamma = Array::<F, _>::zeros(IxDyn(&[num_features]));
        let mut dbeta = Array::<F, _>::zeros(IxDyn(&[num_features]));

        // For each feature channel, compute the gradients
        for c in 0..num_features {
            let mut dgamma_sum = F::zero();
            let mut dbeta_sum = F::zero();

            for n in 0..batch_size {
                for s in 0..spatial_size {
                    dgamma_sum = dgamma_sum + reshaped_grad_output[[n, c, s]] * x_norm[[n, c, s]];
                    dbeta_sum = dbeta_sum + reshaped_grad_output[[n, c, s]];
                }
            }

            dgamma[[c]] = dgamma_sum;
            dbeta[[c]] = dbeta_sum;
        }

        // Store the computed gradients for parameter updates
        for c in 0..num_features {
            self.dgamma[[c]] = dgamma[[c]];
            self.dbeta[[c]] = dbeta[[c]];
        }

        // Calculate gradient with respect to input
        let mut dx = Array::<F, _>::zeros(x_norm.shape());

        // Calculate intermediate gradient terms
        for c in 0..num_features {
            let mut dxhat_sum = F::zero();
            let mut dxhat_x_sum = F::zero();

            // Calculate sums needed for the gradient calculation
            for n in 0..batch_size {
                for s in 0..spatial_size {
                    let dxhat = reshaped_grad_output[[n, c, s]] * self.gamma[[c]];
                    dxhat_sum = dxhat_sum + dxhat;
                    dxhat_x_sum = dxhat_x_sum + dxhat * x_norm[[n, c, s]];
                }
            }

            // Calculate final gradients for each element
            for n in 0..batch_size {
                for s in 0..spatial_size {
                    let dxhat = reshaped_grad_output[[n, c, s]] * self.gamma[[c]];
                    let dx_term1 = dxhat;
                    let dx_term2 = dxhat_sum / spatial_elements_f;
                    let dx_term3 = x_norm[[n, c, s]] * dxhat_x_sum / spatial_elements_f;

                    dx[[n, c, s]] = (dx_term1 - dx_term2 - dx_term3) / std_dev[[c]];
                }
            }
        }

        // Reshape back to the original input shape
        let dx_output = dx
            .into_shape(input_shape.clone())
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape dx: {}", e)))?;

        Ok(dx_output)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update gamma and beta using their gradients
        let lr = learning_rate;

        for c in 0..self.num_features {
            self.gamma[[c]] = self.gamma[[c]] - lr * self.dgamma[[c]];
            self.beta[[c]] = self.beta[[c]] - lr * self.dbeta[[c]];
        }

        // Reset gradients
        self.dgamma = Array::<F, _>::zeros(IxDyn(&[self.num_features]));
        self.dbeta = Array::<F, _>::zeros(IxDyn(&[self.num_features]));

        Ok(())
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> ParamLayer<F> for BatchNorm<F> {
    fn get_parameters(&self) -> Vec<&Array<F, ndarray::IxDyn>> {
        vec![&self.gamma, &self.beta]
    }

    fn get_gradients(&self) -> Vec<&Array<F, ndarray::IxDyn>> {
        vec![&self.dgamma, &self.dbeta]
    }

    fn set_parameters(&mut self, params: Vec<Array<F, ndarray::IxDyn>>) -> Result<()> {
        if params.len() != 2 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected 2 parameters, got {}",
                params.len()
            )));
        }

        if params[0].shape() != self.gamma.shape() {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Gamma shape mismatch: expected {:?}, got {:?}",
                self.gamma.shape(),
                params[0].shape()
            )));
        }

        if params[1].shape() != self.beta.shape() {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Beta shape mismatch: expected {:?}, got {:?}",
                self.beta.shape(),
                params[1].shape()
            )));
        }

        self.gamma = params[0].clone();
        self.beta = params[1].clone();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{Array3, Array4};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_layer_norm_shape() {
        // Set up layer normalization
        let mut rng = SmallRng::seed_from_u64(42);
        let layer_norm = LayerNorm::<f64>::new(64, 1e-5, &mut rng).unwrap();

        // Create a batch of inputs
        let batch_size = 2;
        let seq_len = 3;
        let d_model = 64;
        let input = Array3::<f64>::from_elem((batch_size, seq_len, d_model), 0.1).into_dyn();

        // Forward pass
        let output = layer_norm.forward(&input).unwrap();

        // Check output shape
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_layer_norm_normalization() {
        // Set up layer normalization
        let mut rng = SmallRng::seed_from_u64(42);
        let d_model = 10;
        let layer_norm = LayerNorm::<f64>::new(d_model, 1e-5, &mut rng).unwrap();

        // Create a simple input with different values
        let mut input = Array3::<f64>::zeros((1, 1, d_model));
        for i in 0..d_model {
            input[[0, 0, i]] = i as f64;
        }

        // Forward pass
        let output = layer_norm.forward(&input.into_dyn()).unwrap();

        // Calculate mean and variance manually to verify
        let output_view = output.view();
        let output_slice = output_view.slice(ndarray::s![0, 0, ..]);

        // Calculate mean
        let mut sum = 0.0;
        for i in 0..d_model {
            sum += output_slice[i];
        }
        let mean = sum / (d_model as f64);

        // Calculate variance
        let mut sum_sq = 0.0;
        for i in 0..d_model {
            let diff = output_slice[i] - mean;
            sum_sq += diff * diff;
        }
        let var = sum_sq / (d_model as f64);

        // The output should have approximately zero mean and unit variance
        // We allow some tolerance for numerical precision
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
        assert_relative_eq!(var, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_norm_shape() {
        // Set up batch normalization
        let mut rng = SmallRng::seed_from_u64(42);
        let batch_norm = BatchNorm::<f64>::new(3, 0.9, 1e-5, &mut rng).unwrap();

        // Create a batch of inputs (batch_size, channels, height, width)
        let batch_size = 2;
        let channels = 3;
        let height = 4;
        let width = 5;
        let input = Array4::<f64>::from_elem((batch_size, channels, height, width), 0.1).into_dyn();

        // Forward pass
        let output = batch_norm.forward(&input).unwrap();

        // Check output shape
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_batch_norm_training_mode() {
        // Set up batch normalization
        let mut rng = SmallRng::seed_from_u64(42);
        let mut batch_norm = BatchNorm::<f64>::new(3, 0.9, 1e-5, &mut rng).unwrap();

        // Ensure we're in training mode
        batch_norm.set_training(true);

        // Create input with different values per channel
        let batch_size = 2;
        let channels = 3;
        let height = 2;
        let width = 2;
        let mut input = Array4::<f64>::zeros((batch_size, channels, height, width));

        // Fill input: channel 0 with 1s, channel 1 with 2s, channel 2 with 3s
        for n in 0..batch_size {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        input[[n, c, h, w]] = (c + 1) as f64;
                    }
                }
            }
        }

        // Forward pass
        let output = batch_norm.forward(&input.into_dyn()).unwrap();

        // For each channel, the output should have mean ≈ 0 and variance ≈ 1
        for c in 0..channels {
            let mut sum = 0.0;
            let mut count = 0;

            // Calculate mean
            for n in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        sum += output.view().slice(ndarray::s![n, c, h, w]).into_scalar();
                        count += 1;
                    }
                }
            }

            let mean = sum / (count as f64);

            // Calculate variance
            let mut sum_sq = 0.0;
            for n in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        let diff =
                            output.view().slice(ndarray::s![n, c, h, w]).into_scalar() - mean;
                        sum_sq += diff * diff;
                    }
                }
            }

            let var = sum_sq / (count as f64);

            // The output should have approximately zero mean and unit variance within each channel
            assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
            assert_relative_eq!(var, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_batch_norm_inference_mode() {
        // Set up batch normalization
        let mut rng = SmallRng::seed_from_u64(42);
        let mut batch_norm = BatchNorm::<f64>::new(3, 0.9, 1e-5, &mut rng).unwrap();

        // Create input with different values per channel
        let batch_size = 2;
        let channels = 3;
        let height = 2;
        let width = 2;
        let mut input = Array4::<f64>::zeros((batch_size, channels, height, width));

        // Fill input: channel 0 with 1s, channel 1 with 2s, channel 2 with 3s
        for n in 0..batch_size {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        input[[n, c, h, w]] = (c + 1) as f64;
                    }
                }
            }
        }

        // First forward pass in training mode to accumulate running statistics
        batch_norm.set_training(true);
        let _ = batch_norm.forward(&input.clone().into_dyn()).unwrap();

        // Switch to inference mode
        batch_norm.set_training(false);
        let output = batch_norm.forward(&input.into_dyn()).unwrap();

        // Check that output is consistent with input and transformation
        // In inference mode, each channel should be normalized consistently

        // Check that all values within a channel are normalized the same way
        for c in 0..channels {
            let first_val = output.view().slice(ndarray::s![0, c, 0, 0]).into_scalar();

            for n in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        let val = output.view().slice(ndarray::s![n, c, h, w]).into_scalar();
                        assert_relative_eq!(val, first_val, epsilon = 1e-10);
                    }
                }
            }
        }
    }
}
