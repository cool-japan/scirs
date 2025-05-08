//! Dropout layer implementation
//!
//! This module provides implementation of dropout regularization
//! for neural networks as described in "Dropout: A Simple Way to Prevent Neural Networks
//! from Overfitting" by Srivastava et al.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use ndarray::{Array, IxDyn, ScalarOperand};
use num_traits::Float;
use rand::{Rng, RngCore, SeedableRng};
// use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

/// Dropout layer
///
/// During training, randomly sets input elements to zero with probability `p`.
/// During inference, scales the output by 1/(1-p) to maintain the expected value.
///
/// # Examples
///
/// ```
/// use scirs2_neural::layers::{Dropout, Layer};
/// use ndarray::{Array, Array2};
/// use rand::rngs::SmallRng;
/// use rand::SeedableRng;
///
/// // Create a dropout layer with 0.5 dropout probability
/// let mut rng = SmallRng::seed_from_u64(42);
/// let dropout = Dropout::new(0.5, &mut rng).unwrap();
///
/// // Forward pass with a batch of 2 samples, 10 features
/// let batch_size = 2;
/// let features = 10;
/// let input = Array2::<f64>::from_elem((batch_size, features), 1.0).into_dyn();
///
/// // Forward pass in training mode (some values will be dropped)
/// let output = dropout.forward(&input).unwrap();
///
/// // Output shape should match input shape
/// assert_eq!(output.shape(), input.shape());
/// ```
// We need to manually implement Debug because dyn RngCore doesn't implement Debug
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
        let rng = rand::rngs::SmallRng::seed_from_u64(42);
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
    ///
    /// * `p` - Dropout probability (0.0 to 1.0)
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// * A new dropout layer
    pub fn new<R: Rng + 'static + Clone + Send + Sync>(p: f64, rng: &mut R) -> Result<Self> {
        if p < 0.0 || p >= 1.0 {
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
    ///
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

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
        let mut mask = Array::<F, _>::from_elem(input.dim(), F::one());
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
                if (**rng_guard).random::<f64>() < self.p.to_f64().unwrap() {
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
        if !self.training || self.p == F::zero() {
            // In inference mode or with p=0, just pass through the gradient
            return Ok(grad_output.clone());
        }

        // Retrieve cached mask
        let mask_ref = match self.mask_cache.read() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(NeuralError::InferenceError(
                    "Failed to acquire read lock on mask cache".to_string(),
                ))
            }
        };

        if mask_ref.is_none() {
            return Err(NeuralError::InferenceError(
                "No cached mask for backward pass. Call forward() first.".to_string(),
            ));
        }

        let mask = mask_ref.as_ref().unwrap();

        // Scale factor is the same as in forward pass
        let one = F::one();
        let scale = one / (one - self.p);

        // Apply mask and scale to the gradient
        let grad_input = grad_output * mask * scale;

        Ok(grad_input)
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        // Dropout has no parameters to update
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_dropout_shape() {
        // Set up dropout
        let mut rng = SmallRng::seed_from_u64(42);
        let dropout = Dropout::<f64>::new(0.5, &mut rng).unwrap();

        // Create a batch of inputs
        let batch_size = 2;
        let features = 10;
        let input = Array2::<f64>::from_elem((batch_size, features), 1.0).into_dyn();

        // Forward pass
        let output = dropout.forward(&input).unwrap();

        // Check output shape
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_dropout_training_mode() {
        // Set up dropout
        let mut rng = SmallRng::seed_from_u64(42);
        let mut dropout = Dropout::<f64>::new(0.5, &mut rng).unwrap();

        // Ensure training mode
        dropout.set_training(true);

        // Create a batch of inputs
        let batch_size = 100;
        let features = 10;
        let input = Array2::<f64>::from_elem((batch_size, features), 1.0).into_dyn();

        // Forward pass
        let output = dropout.forward(&input).unwrap();

        // Count dropped (zero) elements
        let mut dropped_count = 0;
        for &val in output.iter() {
            if val == 0.0 {
                dropped_count += 1;
            }
        }

        // We expect approximately 50% of elements to be dropped
        // Allow for some statistical variation
        let total_elements = batch_size * features;
        let drop_rate = dropped_count as f64 / total_elements as f64;

        assert!(
            drop_rate > 0.4 && drop_rate < 0.6,
            "Expected drop rate around 0.5, got {}",
            drop_rate
        );
    }

    #[test]
    fn test_dropout_inference_mode() {
        // Set up dropout
        let mut rng = SmallRng::seed_from_u64(42);
        let mut dropout = Dropout::<f64>::new(0.5, &mut rng).unwrap();

        // Set to inference mode
        dropout.set_training(false);

        // Create a batch of inputs
        let batch_size = 2;
        let features = 10;
        let input = Array2::<f64>::from_elem((batch_size, features), 1.0).into_dyn();

        // Forward pass
        let output = dropout.forward(&input).unwrap();

        // In inference mode, all elements should pass through unchanged
        for &val in output.iter() {
            assert_eq!(val, 1.0);
        }
    }

    #[test]
    fn test_dropout_zero_probability() {
        // Set up dropout with p=0 (no dropout)
        let mut rng = SmallRng::seed_from_u64(42);
        let dropout = Dropout::<f64>::new(0.0, &mut rng).unwrap();

        // Create a batch of inputs
        let batch_size = 2;
        let features = 10;
        let input = Array2::<f64>::from_elem((batch_size, features), 1.0).into_dyn();

        // Forward pass
        let output = dropout.forward(&input).unwrap();

        // With p=0, all elements should pass through unchanged
        for &val in output.iter() {
            assert_eq!(val, 1.0);
        }
    }

    #[test]
    fn test_dropout_backward() {
        // Set up dropout
        let mut rng = SmallRng::seed_from_u64(42);
        let dropout = Dropout::<f64>::new(0.5, &mut rng).unwrap();

        // Create a batch of inputs
        let batch_size = 2;
        let features = 10;
        let input = Array2::<f64>::from_elem((batch_size, features), 1.0).into_dyn();

        // Forward pass to create mask
        let output = dropout.forward(&input).unwrap();

        // Create gradient
        let grad_output = Array2::<f64>::from_elem((batch_size, features), 1.0).into_dyn();

        // Backward pass
        let grad_input = dropout.backward(&input, &grad_output).unwrap();

        // Check that grad_input has same shape
        assert_eq!(grad_input.shape(), input.shape());

        // Elements that were set to zero in the forward pass should also be zero in the backward pass
        for (out, grad) in output.iter().zip(grad_input.iter()) {
            if *out == 0.0 {
                assert_eq!(*grad, 0.0);
            } else {
                // Non-zero elements should have the same gradient scale as in forward pass
                assert_eq!(*grad, 2.0); // scale = 1/(1-0.5) = 2.0
            }
        }
    }
}
