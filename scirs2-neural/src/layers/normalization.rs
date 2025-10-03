//! Normalization layers implementation
//!
//! This module provides implementations of various normalization techniques
//! such as Layer Normalization, Batch Normalization, etc.

use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::Rng;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Layer Normalization layer
///
/// Implements layer normalization as described in "Layer Normalization"
/// by Ba, Kiros, and Hinton. It normalizes the inputs across the last dimension
/// and applies learnable scale and shift parameters.
#[derive(Debug)]
pub struct LayerNorm<F: Float + Debug + Send + Sync> {
    /// Dimensionality of the input features
    normalizedshape: Vec<usize>,
    /// Learnable scale parameter
    gamma: Array<F, IxDyn>,
    /// Learnable shift parameter
    beta: Array<F, IxDyn>,
    /// Gradient of gamma
    dgamma: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of beta
    dbeta: Arc<RwLock<Array<F, IxDyn>>>,
    /// Small constant for numerical stability
    eps: F,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Normalized input cache for backward pass
    norm_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Mean cache for backward pass
    mean_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Variance cache for backward pass
    var_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Clone for LayerNorm<F> {
    fn clone(&self) -> Self {
        let input_cache_clone = match self.input_cache.read() {
            Ok(guard) => guard.clone(),
            Err(_) => None,
        };
        let norm_cache_clone = match self.norm_cache.read() {
            Ok(guard) => guard.clone(),
            Err(_) => None,
        };
        let mean_cache_clone = match self.mean_cache.read() {
            Ok(guard) => guard.clone(),
            Err(_) => None,
        };
        let var_cache_clone = match self.var_cache.read() {
            Ok(guard) => guard.clone(),
            Err(_) => None,
        };

        Self {
            normalizedshape: self.normalizedshape.clone(),
            gamma: self.gamma.clone(),
            beta: self.beta.clone(),
            dgamma: Arc::new(RwLock::new(self.dgamma.read().unwrap().clone())),
            dbeta: Arc::new(RwLock::new(self.dbeta.read().unwrap().clone())),
            eps: self.eps,
            input_cache: Arc::new(RwLock::new(input_cache_clone)),
            norm_cache: Arc::new(RwLock::new(norm_cache_clone)),
            mean_cache: Arc::new(RwLock::new(mean_cache_clone)),
            var_cache: Arc::new(RwLock::new(var_cache_clone)),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> LayerNorm<F> {
    /// Create a new layer normalization layer
    pub fn new<R: Rng>(normalizedshape: usize, eps: f64, _rng: &mut R) -> Result<Self> {
        let gamma = Array::<F, IxDyn>::from_elem(IxDyn(&[normalizedshape]), F::one());
        let beta = Array::<F, IxDyn>::from_elem(IxDyn(&[normalizedshape]), F::zero());

        let dgamma = Arc::new(RwLock::new(Array::<F, IxDyn>::zeros(IxDyn(&[
            normalizedshape,
        ]))));
        let dbeta = Arc::new(RwLock::new(Array::<F, IxDyn>::zeros(IxDyn(&[
            normalizedshape,
        ]))));

        let eps = F::from(eps).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert epsilon to type F".to_string())
        })?;

        Ok(Self {
            normalizedshape: vec![normalizedshape],
            gamma,
            beta,
            dgamma,
            dbeta,
            eps,
            input_cache: Arc::new(RwLock::new(None)),
            norm_cache: Arc::new(RwLock::new(None)),
            mean_cache: Arc::new(RwLock::new(None)),
            var_cache: Arc::new(RwLock::new(None)),
        })
    }

    /// Get the normalized shape
    pub fn normalizedshape(&self) -> usize {
        self.normalizedshape[0]
    }

    /// Get the epsilon value
    #[allow(dead_code)]
    pub fn eps(&self) -> f64 {
        self.eps.to_f64().unwrap_or(1e-5)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for LayerNorm<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.clone());
        }

        let inputshape = input.shape();
        let ndim = input.ndim();

        if ndim < 1 {
            return Err(NeuralError::InferenceError(
                "Input must have at least 1 dimension".to_string(),
            ));
        }

        let feat_dim = inputshape[ndim - 1];
        if feat_dim != self.normalizedshape[0] {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Last dimension of input ({}) must match normalizedshape ({})",
                feat_dim, self.normalizedshape[0]
            )));
        }

        let batchshape: Vec<usize> = inputshape[..ndim - 1].to_vec();
        let batch_size: usize = batchshape.iter().product();

        // Reshape input to 2D: [batch_size, features]
        let reshaped = input
            .to_owned()
            .into_shape_with_order(IxDyn(&[batch_size, feat_dim]))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape input: {e}")))?;

        // Compute mean and variance for each sample
        let mut mean = Array::<F, IxDyn>::zeros(IxDyn(&[batch_size, 1]));
        let mut var = Array::<F, IxDyn>::zeros(IxDyn(&[batch_size, 1]));

        for i in 0..batch_size {
            let mut sum = F::zero();
            for j in 0..feat_dim {
                sum = sum + reshaped[[i, j]];
            }
            mean[[i, 0]] = sum / F::from(feat_dim).unwrap();

            let mut sum_sq = F::zero();
            for j in 0..feat_dim {
                let diff = reshaped[[i, j]] - mean[[i, 0]];
                sum_sq = sum_sq + diff * diff;
            }
            var[[i, 0]] = sum_sq / F::from(feat_dim).unwrap();
        }

        // Cache mean and variance
        if let Ok(mut cache) = self.mean_cache.write() {
            *cache = Some(mean.clone());
        }
        if let Ok(mut cache) = self.var_cache.write() {
            *cache = Some(var.clone());
        }

        // Normalize and apply gamma/beta
        let mut normalized = Array::<F, IxDyn>::zeros(IxDyn(&[batch_size, feat_dim]));
        for i in 0..batch_size {
            for j in 0..feat_dim {
                let x_norm = (reshaped[[i, j]] - mean[[i, 0]]) / (var[[i, 0]] + self.eps).sqrt();
                normalized[[i, j]] = x_norm * self.gamma[[j]] + self.beta[[j]];
            }
        }

        // Cache normalized input
        if let Ok(mut cache) = self.norm_cache.write() {
            *cache = Some(normalized.clone().into_dimensionality::<IxDyn>().unwrap());
        }

        // Reshape back to original shape
        let output = normalized
            .into_shape_with_order(IxDyn(inputshape))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape output: {e}")))?;

        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Simple implementation - return grad_output as is
        Ok(grad_output.clone())
    }

    fn update(&mut self, _learningrate: F) -> Result<()> {
        // Simple implementation - no-op for now
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "LayerNorm"
    }

    fn parameter_count(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> ParamLayer<F> for LayerNorm<F> {
    fn get_parameters(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn get_gradients(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        vec![]
    }

    fn set_parameters(&mut self, params: Vec<Array<F, scirs2_core::ndarray::IxDyn>>) -> Result<()> {
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
#[derive(Debug, Clone)]
pub struct BatchNorm<F: Float + Debug + Send + Sync> {
    /// Number of features (channels)
    num_features: usize,
    /// Learnable scale parameter
    gamma: Array<F, IxDyn>,
    /// Learnable shift parameter
    beta: Array<F, IxDyn>,
    /// Small constant for numerical stability
    #[allow(dead_code)]
    eps: F,
    /// Momentum for running statistics updates
    #[allow(dead_code)]
    momentum: F,
    /// Whether we're in training mode
    training: bool,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> BatchNorm<F> {
    /// Create a new batch normalization layer
    pub fn new<R: Rng>(
        _num_features: usize,
        momentum: f64,
        eps: f64,
        _rng: &mut R,
    ) -> Result<Self> {
        let gamma = Array::<F, IxDyn>::from_elem(IxDyn(&[_num_features]), F::one());
        let beta = Array::<F, IxDyn>::from_elem(IxDyn(&[_num_features]), F::zero());

        let momentum = F::from(momentum).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert momentum to type F".to_string())
        })?;

        let eps = F::from(eps).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert epsilon to type F".to_string())
        })?;

        Ok(Self {
            num_features: _num_features,
            gamma,
            beta,
            eps,
            momentum,
            training: true,
        })
    }

    /// Set the training mode
    #[allow(dead_code)]
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Get the number of features
    #[allow(dead_code)]
    pub fn num_features(&self) -> usize {
        self.num_features
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for BatchNorm<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Simple implementation - return input as is for now
        Ok(input.clone())
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, _learningrate: F) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "BatchNorm"
    }

    fn parameter_count(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> ParamLayer<F> for BatchNorm<F> {
    fn get_parameters(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn get_gradients(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        vec![]
    }

    fn set_parameters(&mut self, params: Vec<Array<F, scirs2_core::ndarray::IxDyn>>) -> Result<()> {
        if params.len() != 2 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected 2 parameters, got {}",
                params.len()
            )));
        }

        self.gamma = params[0].clone();
        self.beta = params[1].clone();

        Ok(())
    }
}
