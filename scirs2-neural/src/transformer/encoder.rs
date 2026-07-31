//! Transformer encoder implementation
//!
//! This module provides implementation of transformer encoder layers and blocks
//! as described in "Attention Is All You Need" by Vaswani et al.

use crate::error::{NeuralError, Result};
use crate::layers::{AttentionConfig, Layer, LayerNorm, ParamLayer, SelfAttention};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Feed-forward network used in transformer encoder/decoder layers
///
/// Implements the position-wise feed-forward network in transformer models.
/// This consists of two linear transformations with a ReLU activation in between.
/// FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
#[derive(Debug)]
pub struct FeedForward<F: Float + Debug + Send + Sync + SimdUnifiedOps + NumAssign> {
    /// Input/output dimension
    d_model: usize,
    /// Hidden dimension
    d_ff: usize,
    /// First linear transformation weights
    w1: Array<F, IxDyn>,
    /// First linear transformation biases
    b1: Array<F, IxDyn>,
    /// Second linear transformation weights
    w2: Array<F, IxDyn>,
    /// Second linear transformation biases
    b2: Array<F, IxDyn>,
    /// Gradient of w1, written by `backward`
    dw1: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of b1, written by `backward`
    db1: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of w2, written by `backward`
    dw2: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of b2, written by `backward`
    db2: Arc<RwLock<Array<F, IxDyn>>>,
    /// Dropout rate (0 means no dropout)
    dropout: F,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Hidden state cache for backward pass
    hidden_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign> Clone
    for FeedForward<F>
{
    fn clone(&self) -> Self {
        let read_grad = |cell: &Arc<RwLock<Array<F, IxDyn>>>| match cell.read() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        Self {
            d_model: self.d_model,
            d_ff: self.d_ff,
            w1: self.w1.clone(),
            b1: self.b1.clone(),
            w2: self.w2.clone(),
            b2: self.b2.clone(),
            dw1: Arc::new(RwLock::new(read_grad(&self.dw1))),
            db1: Arc::new(RwLock::new(read_grad(&self.db1))),
            dw2: Arc::new(RwLock::new(read_grad(&self.dw2))),
            db2: Arc::new(RwLock::new(read_grad(&self.db2))),
            dropout: self.dropout,
            input_cache: Arc::new(RwLock::new(
                self.input_cache.read().expect("Operation failed").clone(),
            )),
            hidden_cache: Arc::new(RwLock::new(
                self.hidden_cache.read().expect("Operation failed").clone(),
            )),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign>
    FeedForward<F>
{
    /// Create a new feed-forward network for transformers
    ///
    /// # Arguments
    /// * `d_model` - Input/output dimension
    /// * `d_ff` - Hidden dimension
    /// * `dropout` - Dropout rate (0 means no dropout)
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    /// * A new feed-forward network
    pub fn new<R: Rng>(d_model: usize, d_ff: usize, dropout: f64, rng: &mut R) -> Result<Self> {
        // Initialize weights with Xavier/Glorot initialization
        let scale1 = F::from(1.0 / (d_model as f64).sqrt()).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert scale factor".to_string())
        })?;
        let scale2 = F::from(1.0 / (d_ff as f64).sqrt()).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert scale factor".to_string())
        })?;

        // Helper function to create weight matrix
        let create_weight_matrix = |input_size: usize,
                                    output_size: usize,
                                    scale: F,
                                    rng: &mut R|
         -> Result<Array<F, IxDyn>> {
            let weights_vec: Vec<F> = (0..(input_size * output_size))
                .map(|_| {
                    let val = F::from(rng.random_range(-1.0..1.0)).unwrap_or(F::zero());
                    val * scale
                })
                .collect();
            Array::from_shape_vec(IxDyn(&[input_size, output_size]), weights_vec).map_err(|e| {
                NeuralError::InvalidArchitecture(format!("Failed to create weights array: {}", e))
            })
        };

        // Create weight matrices
        let w1 = create_weight_matrix(d_model, d_ff, scale1, rng)?;
        let w2 = create_weight_matrix(d_ff, d_model, scale2, rng)?;

        // Initialize biases with zeros
        let b1 = Array::zeros(IxDyn(&[d_ff]));
        let b2 = Array::zeros(IxDyn(&[d_model]));

        // Initialize gradient arrays with zeros
        let dw1 = Arc::new(RwLock::new(Array::zeros(w1.dim())));
        let dw2 = Arc::new(RwLock::new(Array::zeros(w2.dim())));
        let db1 = Arc::new(RwLock::new(Array::zeros(b1.dim())));
        let db2 = Arc::new(RwLock::new(Array::zeros(b2.dim())));

        // Convert dropout rate
        let dropout = F::from(dropout).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert dropout rate".to_string())
        })?;

        Ok(Self {
            d_model,
            d_ff,
            w1,
            b1,
            w2,
            b2,
            dw1,
            dw2,
            db1,
            db2,
            dropout,
            input_cache: Arc::new(RwLock::new(None)),
            hidden_cache: Arc::new(RwLock::new(None)),
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign> Layer<F>
    for FeedForward<F>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        *self.input_cache.write().expect("Operation failed") = Some(input.clone());

        // Check input shape
        if input.ndim() < 2 {
            return Err(NeuralError::InferenceError(
                "Input must have at least 2 dimensions [batch, ..., features]".to_string(),
            ));
        }

        let input_shape = input.shape();
        let ndim = input.ndim();
        let feat_dim = input_shape[ndim - 1];

        if feat_dim != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Last dimension of input ({}) must match d_model ({})",
                feat_dim, self.d_model
            )));
        }

        // Compute the batch size (all dimensions except the last one)
        let batch_dims: Vec<usize> = input_shape[..ndim - 1].to_vec();
        let batch_size: usize = batch_dims.iter().product();

        // Reshape input to 2D: [batch_size, d_model]
        let reshaped_input = input
            .clone()
            .into_shape_with_order((batch_size, self.d_model))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape input: {}", e)))?;

        // First linear transformation: [batch_size, d_model] x [d_model, d_ff] -> [batch_size, d_ff]
        let mut hidden = Array::<F, _>::zeros((batch_size, self.d_ff));
        for i in 0..batch_size {
            for j in 0..self.d_ff {
                let mut sum = F::zero();
                for k in 0..self.d_model {
                    sum += reshaped_input[[i, k]] * self.w1[[k, j]];
                }
                hidden[[i, j]] = sum + self.b1[[j]];
            }
        }

        // Apply ReLU activation
        for i in 0..batch_size {
            for j in 0..self.d_ff {
                hidden[[i, j]] = if hidden[[i, j]] > F::zero() {
                    hidden[[i, j]]
                } else {
                    F::zero()
                };
            }
        }

        // Cache hidden state for backward pass
        let hidden_dyn = hidden.clone().into_dyn();
        *self.hidden_cache.write().expect("Operation failed") = Some(hidden_dyn);

        // Apply dropout if needed (simplified version)
        if self.dropout > F::zero() {
            let keep_prob = F::one() - self.dropout;
            for i in 0..batch_size {
                for j in 0..self.d_ff {
                    hidden[[i, j]] /= keep_prob;
                }
            }
        }

        // Second linear transformation: [batch_size, d_ff] x [d_ff, d_model] -> [batch_size, d_model]
        let mut output = Array::<F, _>::zeros((batch_size, self.d_model));
        for i in 0..batch_size {
            for j in 0..self.d_model {
                let mut sum = F::zero();
                for k in 0..self.d_ff {
                    sum += hidden[[i, k]] * self.w2[[k, j]];
                }
                output[[i, j]] = sum + self.b2[[j]];
            }
        }

        // Reshape output to match input shape
        let mut output_shape = input_shape.to_vec();
        output_shape[ndim - 1] = self.d_model;
        let output_reshaped = output
            .into_shape_with_order(output_shape)
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape output: {}", e)))?;

        Ok(output_reshaped)
    }

    /// Exact gradient of `FFN(x) = (relu(x W1 + b1) / keep_prob) W2 + b2`.
    ///
    /// Weight and bias gradients are accumulated over every position of the
    /// (flattened) batch and stored for [`Layer::update`]; the returned array is
    /// the gradient with respect to the layer input.
    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Retrieve cached values
        let input_ref = self.input_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on input cache".to_string())
        })?;
        let hidden_ref = self.hidden_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on hidden cache".to_string())
        })?;

        let missing = || {
            NeuralError::InferenceError(
                "No cached values for backward pass. Call forward() first.".to_string(),
            )
        };
        let cached_input = input_ref.as_ref().ok_or_else(missing)?;
        let cached_hidden = hidden_ref.as_ref().ok_or_else(missing)?;

        // Get input dimensions
        let input_shape = input.shape();
        let ndim = input.ndim();
        if ndim < 2 || input_shape[ndim - 1] != self.d_model {
            return Err(NeuralError::ShapeMismatch(format!(
                "FeedForward input must end in d_model ({}), got {:?}",
                self.d_model, input_shape
            )));
        }
        let batch_size: usize = input_shape[..ndim - 1].iter().product();
        if grad_output.shape() != input_shape {
            return Err(NeuralError::ShapeMismatch(format!(
                "Output gradient shape {:?} must match the input shape {:?}",
                grad_output.shape(),
                input_shape
            )));
        }

        // Reshape cached values to 2D
        let cached_input_2d = cached_input
            .clone()
            .into_shape_with_order((batch_size, self.d_model))
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape cached input: {}", e))
            })?;

        let cached_hidden_2d = cached_hidden
            .clone()
            .into_shape_with_order((batch_size, self.d_ff))
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape cached hidden: {}", e))
            })?;

        // Reshape grad_output to 2D
        let grad_output_2d = grad_output
            .clone()
            .into_shape_with_order((batch_size, self.d_model))
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape grad_output: {}", e))
            })?;

        // The forward pass rescales the post-ReLU hidden state by 1/keep_prob,
        // so that same factor appears in the second layer's inputs.
        let keep_prob = if self.dropout > F::zero() {
            F::one() - self.dropout
        } else {
            F::one()
        };
        if keep_prob <= F::zero() {
            return Err(NeuralError::InvalidArchitecture(
                "Dropout rate must be below 1.0".to_string(),
            ));
        }
        let inv_keep = keep_prob.recip();

        let mut dw2 = Array::<F, IxDyn>::zeros(self.w2.dim());
        let mut db2 = Array::<F, IxDyn>::zeros(self.b2.dim());
        let mut dw1 = Array::<F, IxDyn>::zeros(self.w1.dim());
        let mut db1 = Array::<F, IxDyn>::zeros(self.b1.dim());

        // Backward through second linear layer: grad_output -> grad_hidden
        let mut grad_hidden = Array::<F, _>::zeros((batch_size, self.d_ff));
        for i in 0..batch_size {
            for j in 0..self.d_model {
                let g = grad_output_2d[[i, j]];
                db2[[j]] += g;
                if g == F::zero() {
                    continue;
                }
                for k in 0..self.d_ff {
                    dw2[[k, j]] += cached_hidden_2d[[i, k]] * inv_keep * g;
                }
            }
            for k in 0..self.d_ff {
                let mut sum = F::zero();
                for j in 0..self.d_model {
                    sum += grad_output_2d[[i, j]] * self.w2[[k, j]];
                }
                // Undo the 1/keep_prob rescaling, then the ReLU.
                grad_hidden[[i, k]] = if cached_hidden_2d[[i, k]] > F::zero() {
                    sum * inv_keep
                } else {
                    F::zero()
                };
            }
        }

        // Backward through first linear layer: grad_hidden -> grad_input
        let mut grad_input_2d = Array::<F, _>::zeros((batch_size, self.d_model));
        for i in 0..batch_size {
            for j in 0..self.d_ff {
                let g = grad_hidden[[i, j]];
                db1[[j]] += g;
                if g == F::zero() {
                    continue;
                }
                for k in 0..self.d_model {
                    dw1[[k, j]] += cached_input_2d[[i, k]] * g;
                }
            }
            for k in 0..self.d_model {
                let mut sum = F::zero();
                for j in 0..self.d_ff {
                    sum += grad_hidden[[i, j]] * self.w1[[k, j]];
                }
                grad_input_2d[[i, k]] = sum;
            }
        }

        let store = |cell: &Arc<RwLock<Array<F, IxDyn>>>, value: Array<F, IxDyn>| -> Result<()> {
            let mut guard = cell.write().map_err(|_| {
                NeuralError::InferenceError("Failed to acquire write lock on gradients".to_string())
            })?;
            *guard = value;
            Ok(())
        };
        store(&self.dw1, dw1)?;
        store(&self.db1, db1)?;
        store(&self.dw2, dw2)?;
        store(&self.db2, db2)?;

        // Reshape grad_input back to original shape
        let grad_input = grad_input_2d
            .into_shape_with_order(input_shape.to_vec())
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape grad_input: {}", e))
            })?;

        Ok(grad_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        let read = |cell: &Arc<RwLock<Array<F, IxDyn>>>| -> Result<Array<F, IxDyn>> {
            let guard = cell.read().map_err(|_| {
                NeuralError::InferenceError("Failed to acquire read lock on gradients".to_string())
            })?;
            Ok(guard.clone())
        };
        let dw1 = read(&self.dw1)?;
        let db1 = read(&self.db1)?;
        let dw2 = read(&self.dw2)?;
        let db2 = read(&self.db2)?;

        for (param, grad) in [
            (&mut self.w1, &dw1),
            (&mut self.b1, &db1),
            (&mut self.w2, &dw2),
            (&mut self.b2, &db2),
        ] {
            if param.shape() != grad.shape() {
                return Err(NeuralError::ShapeMismatch(format!(
                    "Parameter shape {:?} does not match gradient shape {:?}",
                    param.shape(),
                    grad.shape()
                )));
            }
            scirs2_core::ndarray::Zip::from(&mut *param)
                .and(grad)
                .for_each(|w, &g| *w -= learning_rate * g);
        }

        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        ParamLayer::get_parameters(self)
    }

    fn gradients(&self) -> Vec<Array<F, IxDyn>> {
        ParamLayer::get_gradients(self)
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        ParamLayer::set_parameters(self, params.to_vec())
    }

    fn layer_type(&self) -> &str {
        "FeedForward"
    }

    fn parameter_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign>
    ParamLayer<F> for FeedForward<F>
{
    fn get_parameters(&self) -> Vec<Array<F, IxDyn>> {
        vec![
            self.w1.clone(),
            self.b1.clone(),
            self.w2.clone(),
            self.b2.clone(),
        ]
    }

    /// Gradients of `[w1, b1, w2, b2]`; zero until `backward` has run.
    fn get_gradients(&self) -> Vec<Array<F, IxDyn>> {
        let read = |cell: &Arc<RwLock<Array<F, IxDyn>>>| match cell.read() {
            Ok(guard) => guard.clone(),
            Err(_) => Array::zeros(IxDyn(&[0])),
        };
        vec![
            read(&self.dw1),
            read(&self.db1),
            read(&self.dw2),
            read(&self.db2),
        ]
    }

    fn set_parameters(&mut self, params: Vec<Array<F, IxDyn>>) -> Result<()> {
        if params.len() != 4 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected 4 parameters, got {}",
                params.len()
            )));
        }
        for (existing, replacement) in [
            (&self.w1, &params[0]),
            (&self.b1, &params[1]),
            (&self.w2, &params[2]),
            (&self.b2, &params[3]),
        ] {
            if existing.shape() != replacement.shape() {
                return Err(NeuralError::InvalidArchitecture(format!(
                    "Parameter shape mismatch: expected {:?}, got {:?}",
                    existing.shape(),
                    replacement.shape()
                )));
            }
        }
        self.w1 = params[0].clone();
        self.b1 = params[1].clone();
        self.w2 = params[2].clone();
        self.b2 = params[3].clone();
        Ok(())
    }
}

/// Transformer encoder layer
///
/// Implements a single layer of the transformer encoder as described in
/// "Attention Is All You Need" by Vaswani et al. It consists of multi-head
/// self-attention followed by a position-wise feed-forward network, with
/// residual connections and layer normalization.
#[derive(Debug)]
pub struct TransformerEncoderLayer<F: Float + Debug + Send + Sync + SimdUnifiedOps + NumAssign> {
    /// Multi-head self-attention layer
    self_attn: SelfAttention<F>,
    /// Layer normalization after attention
    norm1: LayerNorm<F>,
    /// Feed-forward network
    feed_forward: FeedForward<F>,
    /// Layer normalization after feed-forward network
    norm2: LayerNorm<F>,
    /// Dropout rate for residual connections
    #[allow(dead_code)]
    dropout: F,
    /// Model embedding dimension
    d_model: usize,
    /// Self-attention output cache for backward pass
    attn_output_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Normalized attention output cache for backward pass
    norm1_output_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// `norm1_output + feed_forward(norm1_output)` cache: the tensor fed into
    /// `norm2`, needed by the backward pass
    norm2_input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign> Clone
    for TransformerEncoderLayer<F>
{
    fn clone(&self) -> Self {
        Self {
            self_attn: self.self_attn.clone(),
            norm1: self.norm1.clone(),
            feed_forward: self.feed_forward.clone(),
            norm2: self.norm2.clone(),
            dropout: self.dropout,
            d_model: self.d_model,
            attn_output_cache: Arc::new(RwLock::new(
                self.attn_output_cache
                    .read()
                    .expect("Operation failed")
                    .clone(),
            )),
            norm1_output_cache: Arc::new(RwLock::new(
                self.norm1_output_cache
                    .read()
                    .expect("Operation failed")
                    .clone(),
            )),
            norm2_input_cache: Arc::new(RwLock::new(
                self.norm2_input_cache
                    .read()
                    .expect("Operation failed")
                    .clone(),
            )),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign>
    TransformerEncoderLayer<F>
{
    /// Create a new transformer encoder layer
    ///
    /// # Arguments
    /// * `d_model` - Model embedding dimension
    /// * `n_heads` - Number of attention heads
    /// * `d_ff` - Feed-forward network hidden dimension
    /// * `dropout` - Dropout rate (0 means no dropout)
    /// * `epsilon` - Small constant for layer normalization
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    /// * A new transformer encoder layer
    pub fn new<R: Rng>(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        dropout: f64,
        epsilon: f64,
        rng: &mut R,
    ) -> Result<Self> {
        // Verify parameters
        if !d_model.is_multiple_of(n_heads) {
            return Err(NeuralError::InvalidArchitecture(format!(
                "d_model ({}) must be divisible by n_heads ({})",
                d_model, n_heads
            )));
        }

        // Calculate head dimension
        let head_dim = d_model / n_heads;

        // Create attention config
        let attn_config = AttentionConfig {
            num_heads: n_heads,
            head_dim,
            dropout_prob: dropout,
            causal: false,
            scale: None,
        };

        // Create components
        let self_attn = SelfAttention::new(d_model, attn_config, rng)?;
        let norm1 = LayerNorm::new(d_model, epsilon, rng)?;
        let feed_forward = FeedForward::new(d_model, d_ff, dropout, rng)?;
        let norm2 = LayerNorm::new(d_model, epsilon, rng)?;

        // Convert dropout rate
        let dropout = F::from(dropout).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert dropout rate".to_string())
        })?;

        Ok(Self {
            self_attn,
            norm1,
            feed_forward,
            norm2,
            dropout,
            d_model,
            attn_output_cache: Arc::new(RwLock::new(None)),
            norm1_output_cache: Arc::new(RwLock::new(None)),
            norm2_input_cache: Arc::new(RwLock::new(None)),
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign> Layer<F>
    for TransformerEncoderLayer<F>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Check input shape
        if input.ndim() < 3 {
            return Err(NeuralError::InferenceError(
                "Input must have at least 3 dimensions [batch, seq_len, features]".to_string(),
            ));
        }

        let input_shape = input.shape();
        let feat_dim = input_shape[input.ndim() - 1];

        if feat_dim != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Last dimension of input ({}) must match d_model ({})",
                feat_dim, self.d_model
            )));
        }

        // 1. Self-attention with residual connection
        let attn_output = self.self_attn.forward(input)?;
        *self.attn_output_cache.write().expect("Operation failed") = Some(attn_output.clone());

        // Add residual connection (x + Sublayer(x))
        let attn_output_residual = input + &attn_output;

        // 2. Layer normalization after attention
        let norm1_output = self.norm1.forward(&attn_output_residual)?;
        *self.norm1_output_cache.write().expect("Operation failed") = Some(norm1_output.clone());

        // 3. Feed-forward network with residual connection
        let ff_output = self.feed_forward.forward(&norm1_output)?;
        let output = &norm1_output + &ff_output;
        *self.norm2_input_cache.write().expect("Operation failed") = Some(output.clone());

        // 4. Layer normalization after feed-forward
        let final_output = self.norm2.forward(&output)?;

        Ok(final_output)
    }

    /// Backpropagate through the whole encoder block.
    ///
    /// The forward pass is
    /// `y = norm2(n1 + FFN(n1))` with `n1 = norm1(x + SelfAttention(x))`,
    /// so the gradient is chained back through `norm2`, the feed-forward
    /// residual branch, `norm1` and finally the attention residual branch.
    /// Every sub-layer records its own parameter gradients along the way.
    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let missing = || {
            NeuralError::InferenceError(
                "No cached values for backward pass. Call forward() first.".to_string(),
            )
        };
        let norm1_output = self
            .norm1_output_cache
            .read()
            .map_err(|_| {
                NeuralError::InferenceError(
                    "Failed to acquire read lock on norm1 cache".to_string(),
                )
            })?
            .clone()
            .ok_or_else(missing)?;
        let norm2_input = self
            .norm2_input_cache
            .read()
            .map_err(|_| {
                NeuralError::InferenceError(
                    "Failed to acquire read lock on norm2 cache".to_string(),
                )
            })?
            .clone()
            .ok_or_else(missing)?;
        let attn_output = self
            .attn_output_cache
            .read()
            .map_err(|_| {
                NeuralError::InferenceError(
                    "Failed to acquire read lock on attention cache".to_string(),
                )
            })?
            .clone()
            .ok_or_else(missing)?;

        // 4. through the second layer normalization
        let grad_norm2_input = self.norm2.backward(&norm2_input, grad_output)?;

        // 3. through the feed-forward residual branch
        let grad_ff = self
            .feed_forward
            .backward(&norm1_output, &grad_norm2_input)?;
        let grad_norm1_output = &grad_norm2_input + &grad_ff;

        // 2. through the first layer normalization
        let residual_input = input + &attn_output;
        let grad_residual = self.norm1.backward(&residual_input, &grad_norm1_output)?;

        // 1. through the self-attention residual branch
        let grad_attn_input = self.self_attn.backward(input, &grad_residual)?;
        Ok(&grad_residual + &grad_attn_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update all components
        self.self_attn.update(learning_rate)?;
        self.norm1.update(learning_rate)?;
        self.feed_forward.update(learning_rate)?;
        self.norm2.update(learning_rate)?;
        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = self.self_attn.params();
        params.extend(self.norm1.params());
        params.extend(self.feed_forward.params());
        params.extend(self.norm2.params());
        params
    }

    fn gradients(&self) -> Vec<Array<F, IxDyn>> {
        let mut grads = self.self_attn.gradients();
        grads.extend(self.norm1.gradients());
        grads.extend(self.feed_forward.gradients());
        grads.extend(self.norm2.gradients());
        grads
    }

    fn layer_type(&self) -> &str {
        "TransformerEncoderLayer"
    }

    fn parameter_count(&self) -> usize {
        self.self_attn.parameter_count()
            + self.norm1.parameter_count()
            + self.feed_forward.parameter_count()
            + self.norm2.parameter_count()
    }
}

/// Transformer encoder
///
/// Stack of transformer encoder layers that processes sequences using
/// self-attention and feed-forward networks.
#[derive(Debug)]
pub struct TransformerEncoder<F: Float + Debug + Send + Sync + SimdUnifiedOps + NumAssign> {
    /// Stack of encoder layers
    layers: Vec<TransformerEncoderLayer<F>>,
    /// Layer outputs cache for backward pass
    layer_outputs: Arc<RwLock<Vec<Array<F, IxDyn>>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign> Clone
    for TransformerEncoder<F>
{
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
            layer_outputs: Arc::new(RwLock::new(
                self.layer_outputs.read().expect("Operation failed").clone(),
            )),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign>
    TransformerEncoder<F>
{
    /// Create a new transformer encoder
    ///
    /// # Arguments
    /// * `d_model` - Model embedding dimension
    /// * `n_layers` - Number of encoder layers
    /// * `n_heads` - Number of attention heads
    /// * `d_ff` - Feed-forward network hidden dimension
    /// * `dropout` - Dropout rate (0 means no dropout)
    /// * `epsilon` - Small constant for layer normalization
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    /// * A new transformer encoder
    pub fn new<R: Rng>(
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
        d_ff: usize,
        dropout: f64,
        epsilon: f64,
        rng: &mut R,
    ) -> Result<Self> {
        // Create encoder layers
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(TransformerEncoderLayer::new(
                d_model, n_heads, d_ff, dropout, epsilon, rng,
            )?);
        }

        Ok(Self {
            layers,
            layer_outputs: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign> Layer<F>
    for TransformerEncoder<F>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Clear layer outputs cache
        *self.layer_outputs.write().expect("Operation failed") = Vec::new();

        // Process input through all encoder layers
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?;
            // Cache layer output for backward pass
            self.layer_outputs
                .write()
                .expect("Operation failed")
                .push(output.clone());
        }

        Ok(output)
    }

    /// Backpropagate through the stack, feeding each layer the input it saw
    /// during the forward pass (the previous layer's cached output).
    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let outputs = self
            .layer_outputs
            .read()
            .map_err(|_| {
                NeuralError::InferenceError(
                    "Failed to acquire read lock on layer outputs".to_string(),
                )
            })?
            .clone();
        if outputs.len() != self.layers.len() {
            return Err(NeuralError::InferenceError(format!(
                "Cached {} layer outputs for {} encoder layers. Call forward() first.",
                outputs.len(),
                self.layers.len()
            )));
        }

        let mut grad = grad_output.clone();
        for (idx, layer) in self.layers.iter().enumerate().rev() {
            let layer_input = if idx == 0 { input } else { &outputs[idx - 1] };
            grad = layer.backward(layer_input, &grad)?;
        }
        Ok(grad)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update all layers
        for layer in &mut self.layers {
            layer.update(learning_rate)?;
        }
        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        self.layers.iter().flat_map(|l| l.params()).collect()
    }

    fn gradients(&self) -> Vec<Array<F, IxDyn>> {
        self.layers.iter().flat_map(|l| l.gradients()).collect()
    }

    fn layer_type(&self) -> &str {
        "TransformerEncoder"
    }

    fn parameter_count(&self) -> usize {
        self.layers.iter().map(|l| l.parameter_count()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_feed_forward_shape() {
        let mut rng = scirs2_core::random::rng();
        let d_model = 64;
        let d_ff = 256;
        let ff = FeedForward::<f64>::new(d_model, d_ff, 0.1, &mut rng).expect("Operation failed");

        let batch_size = 2;
        let seq_len = 10;
        let input = Array3::<f64>::from_elem((batch_size, seq_len, d_model), 0.1).into_dyn();

        let output = ff.forward(&input).expect("Operation failed");
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_encoder_layer_shape() {
        let mut rng = scirs2_core::random::rng();
        let d_model = 64;
        let n_heads = 4;
        let d_ff = 256;
        let dropout = 0.1;
        let epsilon = 1e-5;

        let enc_layer =
            TransformerEncoderLayer::<f64>::new(d_model, n_heads, d_ff, dropout, epsilon, &mut rng)
                .expect("Operation failed");

        let batch_size = 2;
        let seq_len = 10;
        let input = Array3::<f64>::from_elem((batch_size, seq_len, d_model), 0.1).into_dyn();

        let output = enc_layer.forward(&input).expect("Operation failed");
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_encoder_stack_shape() {
        let mut rng = scirs2_core::random::rng();
        let d_model = 64;
        let n_layers = 2;
        let n_heads = 4;
        let d_ff = 256;
        let dropout = 0.1;
        let epsilon = 1e-5;

        let encoder = TransformerEncoder::<f64>::new(
            d_model, n_layers, n_heads, d_ff, dropout, epsilon, &mut rng,
        )
        .expect("Operation failed");

        let batch_size = 2;
        let seq_len = 10;
        let input = Array3::<f64>::from_elem((batch_size, seq_len, d_model), 0.1).into_dyn();

        let output = encoder.forward(&input).expect("Operation failed");
        assert_eq!(output.shape(), input.shape());
        assert_eq!(encoder.num_layers(), n_layers);
    }

    #[test]
    fn test_feed_forward_backward() {
        let mut rng = scirs2_core::random::rng();
        let d_model = 32;
        let d_ff = 64;
        let ff = FeedForward::<f64>::new(d_model, d_ff, 0.0, &mut rng).expect("Operation failed");

        let batch_size = 2;
        let seq_len = 4;
        let input = Array3::<f64>::from_elem((batch_size, seq_len, d_model), 0.1).into_dyn();

        // Forward pass to cache values
        let output = ff.forward(&input).expect("Operation failed");

        // Backward pass
        let grad_output = Array3::<f64>::from_elem((batch_size, seq_len, d_model), 0.01).into_dyn();
        let grad_input = ff.backward(&input, &grad_output).expect("Operation failed");

        assert_eq!(grad_input.shape(), input.shape());
        assert_eq!(output.shape(), input.shape());
    }
}
