//! Dropout layer implementation
//!
//! This module provides implementation of dropout regularization
//! for neural networks as described in "Dropout: A Simple Way to Prevent Neural Networks
//! from Overfitting" by Srivastava et al.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::{Rng, RngCore, SeedableRng};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

/// Dropout layer
///
/// During training, randomly sets input elements to zero with probability `p`.
/// During inference, scales the output by 1/(1-p) to maintain the expected value.
pub struct Dropout<F: Float + Debug + Send + Sync> {
    /// Probability of dropping an element
    p: F,
    /// Random number generator
    rng: Arc<RwLock<Box<dyn RngCore + Send + Sync>>>,
    /// Whether we're in training mode
    training: bool,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Mask cache for backward pass (1 for kept elements, 0 for dropped)
    mask_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Phantom data for type parameter
    _phantom: PhantomData<F>,
}

// Manual implementation of Debug because dyn RngCore doesn't implement Debug
impl<F: Float + Debug + Send + Sync> std::fmt::Debug for Dropout<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dropout")
            .field("p", &self.p)
            .field("rng", &"<dyn RngCore>")
            .field("training", &self.training)
            .finish()
    }
}

// Manual implementation of Clone
impl<F: Float + Debug + Send + Sync> Clone for Dropout<F> {
    fn clone(&self) -> Self {
        let rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
        Self {
            p: self.p,
            rng: Arc::new(RwLock::new(Box::new(rng))),
            training: self.training,
            input_cache: Arc::new(RwLock::new(None)),
            mask_cache: Arc::new(RwLock::new(None)),
            _phantom: PhantomData,
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Dropout<F> {
    /// Create a new dropout layer
    ///
    /// # Arguments
    /// * `p` - Dropout probability (0.0 to 1.0)
    /// * `rng` - Random number generator
    pub fn new<R: Rng + 'static + Clone + Send + Sync>(p: f64, rng: &mut R) -> Result<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(NeuralError::InvalidArchitecture(
                "Dropout probability must be in [0, 1)".to_string(),
            ));
        }

        let p = F::from(p).ok_or_else(|| {
            NeuralError::InvalidArchitecture(
                "Failed to convert dropout probability to type F".to_string(),
            )
        })?;

        Ok(Self {
            p,
            rng: Arc::new(RwLock::new(Box::new(rng.clone()))),
            training: true,
            input_cache: Arc::new(RwLock::new(None)),
            mask_cache: Arc::new(RwLock::new(None)),
            _phantom: PhantomData,
        })
    }

    /// Set the training mode
    /// In training mode, elements are randomly dropped.
    /// In inference mode, all elements are kept but scaled.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Get the dropout probability
    pub fn p(&self) -> f64 {
        self.p.to_f64().unwrap_or(0.0)
    }

    /// Get the training mode
    pub fn is_training(&self) -> bool {
        self.training
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for Dropout<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.clone());
        } else {
            return Err(NeuralError::InferenceError(
                "Failed to acquire write lock on input cache".to_string(),
            ));
        }

        if !self.training || self.p == F::zero() {
            // In inference mode or with p=0, just pass through the input as is
            return Ok(input.clone());
        }

        // In training mode, create a binary mask and apply it
        let mut mask = Array::<F, IxDyn>::from_elem(input.dim(), F::one());
        let one = F::one();
        let zero = F::zero();

        // Apply the dropout mask
        {
            let mut rng_guard = match self.rng.write() {
                Ok(guard) => guard,
                Err(_) => {
                    return Err(NeuralError::InferenceError(
                        "Failed to acquire write lock on RNG".to_string(),
                    ))
                }
            };

            for elem in mask.iter_mut() {
                if F::from((**rng_guard).random::<f64>()).unwrap() < self.p {
                    *elem = zero;
                }
            }
        }

        // Scale by 1/(1-p) to maintain expected value
        let scale = one / (one - self.p);

        // Cache the mask for backward pass
        if let Ok(mut cache) = self.mask_cache.write() {
            *cache = Some(mask.clone());
        } else {
            return Err(NeuralError::InferenceError(
                "Failed to acquire write lock on mask cache".to_string(),
            ));
        }

        // Apply mask and scale
        let output = input * &mask * scale;
        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        if !self.training {
            // In inference mode, gradients pass through unchanged
            return Ok(grad_output.clone());
        }

        // Get cached mask
        let mask = {
            let cache = match self.mask_cache.read() {
                Ok(cache) => cache,
                Err(_) => {
                    return Err(NeuralError::InferenceError(
                        "Failed to acquire read lock on mask cache".to_string(),
                    ))
                }
            };

            match cache.as_ref() {
                Some(mask) => mask.clone(),
                None => {
                    return Err(NeuralError::InferenceError(
                        "No cached mask for backward pass".to_string(),
                    ))
                }
            }
        };

        // Scale factor
        let one = F::one();
        let scale = one / (one - self.p);

        // Apply mask and scale to gradients
        let grad_input = grad_output * &mask * scale;
        Ok(grad_input)
    }

    fn update(&mut self, _learningrate: F) -> Result<()> {
        // Dropout has no parameters to update
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn layer_type(&self) -> &str {
        "Dropout"
    }

    fn parameter_count(&self) -> usize {
        0 // Dropout has no trainable parameters
    }

    fn layer_description(&self) -> String {
        format!("type:Dropout, p:{:.3}", self.p())
    }
}

// Explicit Send + Sync implementations for Dropout layer
unsafe impl<F: Float + Debug + Send + Sync> Send for Dropout<F> {}
unsafe impl<F: Float + Debug + Send + Sync> Sync for Dropout<F> {}
