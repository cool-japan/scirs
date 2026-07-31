//! Positional Encoding for Transformer Models
//!
//! This module provides various positional encoding strategies used in transformer
//! architectures. Positional encodings inject sequence order information into
//! the model since self-attention is permutation invariant.
//!
//! # Available Encodings
//!
//! - **Sinusoidal**: Classic fixed positional encoding from "Attention Is All You Need"
//! - **Learned**: Trainable position embeddings (like BERT)
//! - **Relative**: Position-relative encodings for better length generalization
//! - **Rotary (RoPE)**: Rotary position embeddings used in modern LLMs
//!
//! # Examples
//!
//! ```rust
//! use scirs2_neural::utils::positional_encoding::{
//!     SinusoidalPositionalEncoding, PositionalEncoding
//! };
//! use scirs2_core::ndarray::Array2;
//!
//! // Create sinusoidal encoding for d_model=64, max_len=100
//! let pe = SinusoidalPositionalEncoding::<f32>::new(64, 100);
//!
//! // Get encoding for sequence of length 10
//! let encoding = pe.encode(10);
//! assert_eq!(encoding.shape(), &[10, 64]);
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{s, Array, Array2, Array3, Axis, IxDyn, Zip};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use std::f64::consts::PI;
use std::fmt::Debug;

/// Trait for positional encoding implementations
pub trait PositionalEncoding<F: Float + Debug + NumAssign> {
    /// Encode positions for a sequence of given length
    ///
    /// # Arguments
    /// * `seq_len` - Length of the sequence
    ///
    /// # Returns
    /// Position encodings of shape [seq_len, d_model]
    fn encode(&self, seq_len: usize) -> Array2<F>;

    /// Apply positional encoding to an input tensor
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, seq_len, d_model]
    ///
    /// # Returns
    /// Tensor with positional encoding added
    fn apply(&self, input: &Array3<F>) -> Result<Array3<F>>;

    /// Forward pass with dynamic array
    ///
    /// # Arguments
    /// * `input` - Input tensor of dynamic shape [batch_size, seq_len, d_model]
    ///
    /// # Returns
    /// Tensor with positional encoding added
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Convert IxDyn to Array3
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected 3D input, got {}D",
                input.ndim()
            )));
        }

        let shape = input.shape();
        let input_3d = input
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix3>()
            .map_err(|e| {
                NeuralError::InvalidArchitecture(format!("Failed to convert to 3D: {}", e))
            })?;

        let output_3d = self.apply(&input_3d.to_owned())?;
        Ok(output_3d.into_dyn())
    }

    /// Backward pass: gradient of the loss with respect to the encoding input
    ///
    /// # Arguments
    /// * `grad_output` - Gradient with respect to the encoded tensor,
    ///   shape `[batch_size, seq_len, d_model]`
    ///
    /// # Returns
    /// Gradient with respect to the tensor passed to [`PositionalEncoding::forward`]
    ///
    /// # Errors
    /// The default implementation reports [`NeuralError::NotImplementedError`]
    /// rather than guessing: the Jacobian depends on how the encoding combines
    /// with the input (additive encodings pass the gradient through unchanged,
    /// rotary encodings apply the transposed rotation). Every encoding shipped
    /// with this crate overrides it.
    fn backward(&self, grad_output: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let _ = grad_output;
        Err(NeuralError::NotImplementedError(
            "this positional encoding does not implement a backward pass".to_string(),
        ))
    }

    /// Accumulate the gradient of any trainable encoding parameters
    ///
    /// # Arguments
    /// * `grad_output` - Gradient with respect to the encoded tensor
    ///
    /// # Returns
    /// Result indicating success or failure. Parameter-free encodings ignore
    /// the gradient, which is why the default implementation is a no-op.
    fn accumulate_gradients(&mut self, grad_output: &Array<F, IxDyn>) -> Result<()> {
        let _ = grad_output;
        Ok(())
    }

    /// Update trainable parameters (for learned encodings)
    ///
    /// # Arguments
    /// * `_learning_rate` - Learning rate for parameter updates
    ///
    /// # Returns
    /// Result indicating success or failure
    fn update(&mut self, _learning_rate: F) -> Result<()> {
        // Default implementation: no-op for non-trainable encodings
        Ok(())
    }

    /// Clone the positional encoding into a boxed trait object
    fn clone_box(&self) -> Box<dyn PositionalEncoding<F> + Send + Sync>
    where
        F: Send + Sync + 'static;

    /// Get the model dimension
    fn d_model(&self) -> usize;

    /// Get the maximum sequence length supported
    fn max_len(&self) -> usize;
}

/// Sinusoidal Positional Encoding
///
/// Implements the classic positional encoding from "Attention Is All You Need":
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
///
/// This encoding has several desirable properties:
/// - Deterministic and parameter-free
/// - Each position gets a unique encoding
/// - Can extrapolate to longer sequences than seen during training
/// - Allows the model to attend to relative positions
#[derive(Debug, Clone)]
pub struct SinusoidalPositionalEncoding<F: Float + Debug + NumAssign> {
    d_model: usize,
    max_len: usize,
    /// Pre-computed positional encodings
    encodings: Array2<F>,
    /// Dropout rate (optional)
    dropout: Option<F>,
}

impl<F: Float + Debug + NumAssign> SinusoidalPositionalEncoding<F> {
    /// Create a new sinusoidal positional encoding
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (must be even)
    /// * `max_len` - Maximum sequence length
    pub fn new(d_model: usize, max_len: usize) -> Self {
        assert!(
            d_model.is_multiple_of(2),
            "d_model must be even for sinusoidal PE"
        );

        let encodings = Self::compute_encodings(d_model, max_len);

        Self {
            d_model,
            max_len,
            encodings,
            dropout: None,
        }
    }

    /// Create with dropout
    pub fn with_dropout(d_model: usize, max_len: usize, dropout: F) -> Self {
        let mut pe = Self::new(d_model, max_len);
        pe.dropout = Some(dropout);
        pe
    }

    /// Compute the sinusoidal encodings
    fn compute_encodings(d_model: usize, max_len: usize) -> Array2<F> {
        let mut encodings = Array2::zeros((max_len, d_model));

        for pos in 0..max_len {
            for i in 0..(d_model / 2) {
                // Compute the divisor: 10000^(2i/d_model)
                let exponent = (2 * i) as f64 / d_model as f64;
                let div_term = (10000.0_f64).powf(exponent);
                let angle = pos as f64 / div_term;

                // sin for even indices, cos for odd indices
                let sin_val = F::from(angle.sin()).unwrap_or(F::zero());
                let cos_val = F::from(angle.cos()).unwrap_or(F::zero());

                encodings[[pos, 2 * i]] = sin_val;
                encodings[[pos, 2 * i + 1]] = cos_val;
            }
        }

        encodings
    }

    /// Get parameters (sinusoidal encoding has no trainable parameters)
    pub fn params(&self) -> Vec<&Array<F, IxDyn>> {
        Vec::new()
    }

    /// Set training mode (no-op for sinusoidal encoding as it has no trainable parameters)
    pub fn set_training(&mut self, _training: bool) {
        // No-op: sinusoidal encoding is not trainable
    }

    /// Backward pass for sinusoidal positional encoding.
    ///
    /// Because sinusoidal PE is parameter-free (output = input + fixed_encoding), the
    /// gradient w.r.t. the input equals the gradient w.r.t. the output.
    /// No accumulation into weight gradients is needed.
    pub fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }
}

impl<F: Float + Debug + NumAssign> PositionalEncoding<F> for SinusoidalPositionalEncoding<F> {
    /// The encoding is added to the input, so the gradient passes through
    /// unchanged.
    fn backward(&self, grad_output: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn encode(&self, seq_len: usize) -> Array2<F> {
        assert!(
            seq_len <= self.max_len,
            "seq_len {} exceeds max_len {}",
            seq_len,
            self.max_len
        );
        self.encodings.slice(s![..seq_len, ..]).to_owned()
    }

    fn apply(&self, input: &Array3<F>) -> Result<Array3<F>> {
        let seq_len = input.shape()[1];
        if seq_len > self.max_len {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Sequence length {} exceeds max_len {}",
                seq_len, self.max_len
            )));
        }

        let encoding = self.encode(seq_len);
        let mut output = input.clone();

        // Add positional encoding to each batch
        for mut batch in output.axis_iter_mut(Axis(0)) {
            Zip::from(&mut batch)
                .and(&encoding)
                .for_each(|b, &e| *b += e);
        }

        Ok(output)
    }

    fn clone_box(&self) -> Box<dyn PositionalEncoding<F> + Send + Sync>
    where
        F: Send + Sync + 'static,
    {
        Box::new(self.clone())
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn max_len(&self) -> usize {
        self.max_len
    }
}

/// Learned Positional Encoding
///
/// Uses trainable embeddings for each position, similar to BERT.
/// More flexible than sinusoidal but requires training data and
/// doesn't extrapolate to longer sequences.
#[derive(Debug, Clone)]
pub struct LearnedPositionalEncoding<F: Float + Debug + NumAssign> {
    d_model: usize,
    max_len: usize,
    /// Learnable position embeddings
    embeddings: Array2<F>,
    /// Gradient of `embeddings`, accumulated by `accumulate_gradients`
    dembeddings: Array2<F>,
}

impl<F: Float + Debug + NumAssign> LearnedPositionalEncoding<F> {
    /// Create a new learned positional encoding with random initialization
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `max_len` - Maximum sequence length
    /// * `rng` - Random number generator
    pub fn new<R: Rng>(d_model: usize, max_len: usize, rng: &mut R) -> Self {
        // Xavier/Glorot initialization
        let std = (2.0 / (max_len + d_model) as f64).sqrt();

        let mut embeddings = Array2::zeros((max_len, d_model));
        for elem in embeddings.iter_mut() {
            // Box-Muller transform for normal distribution
            let u1: f64 = rng.random_range(0.0001..1.0);
            let u2: f64 = rng.random_range(0.0..1.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            *elem = F::from(z * std).unwrap_or(F::zero());
        }

        Self {
            d_model,
            max_len,
            dembeddings: Array2::zeros((max_len, d_model)),
            embeddings,
        }
    }

    /// Create from existing embeddings
    pub fn from_embeddings(embeddings: Array2<F>) -> Self {
        let shape = embeddings.shape();
        let (max_len, d_model) = (shape[0], shape[1]);
        Self {
            d_model,
            max_len,
            embeddings,
            dembeddings: Array2::zeros((max_len, d_model)),
        }
    }

    /// Get mutable reference to embeddings for training
    pub fn embeddings_mut(&mut self) -> &mut Array2<F> {
        &mut self.embeddings
    }

    /// Get reference to embeddings
    pub fn embeddings(&self) -> &Array2<F> {
        &self.embeddings
    }
}

impl<F: Float + Debug + NumAssign> PositionalEncoding<F> for LearnedPositionalEncoding<F> {
    /// The embeddings are added to the input, so the gradient with respect to
    /// the input passes through unchanged.
    fn backward(&self, grad_output: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    /// Accumulate `d(loss)/d(embeddings)`: the output gradient summed over the
    /// batch for every position that was used.
    fn accumulate_gradients(&mut self, grad_output: &Array<F, IxDyn>) -> Result<()> {
        if grad_output.ndim() != 3 {
            return Err(NeuralError::ShapeMismatch(format!(
                "Expected a 3D gradient [batch, seq_len, d_model], got {}D",
                grad_output.ndim()
            )));
        }
        let shape = grad_output.shape();
        let (batch_size, seq_len, d_model) = (shape[0], shape[1], shape[2]);
        if d_model != self.d_model {
            return Err(NeuralError::ShapeMismatch(format!(
                "Gradient feature dimension {} does not match d_model {}",
                d_model, self.d_model
            )));
        }
        if seq_len > self.max_len {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Sequence length {} exceeds max_len {}",
                seq_len, self.max_len
            )));
        }
        self.dembeddings = Array2::zeros((self.max_len, self.d_model));
        for b in 0..batch_size {
            for pos in 0..seq_len {
                for k in 0..d_model {
                    self.dembeddings[[pos, k]] += grad_output[[b, pos, k]];
                }
            }
        }
        Ok(())
    }

    /// Apply the gradients accumulated by
    /// [`PositionalEncoding::accumulate_gradients`].
    fn update(&mut self, learning_rate: F) -> Result<()> {
        Zip::from(&mut self.embeddings)
            .and(&self.dembeddings)
            .for_each(|w, &g| *w -= learning_rate * g);
        Ok(())
    }

    fn encode(&self, seq_len: usize) -> Array2<F> {
        assert!(
            seq_len <= self.max_len,
            "seq_len {} exceeds max_len {}",
            seq_len,
            self.max_len
        );
        self.embeddings.slice(s![..seq_len, ..]).to_owned()
    }

    fn apply(&self, input: &Array3<F>) -> Result<Array3<F>> {
        let seq_len = input.shape()[1];
        if seq_len > self.max_len {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Sequence length {} exceeds max_len {}",
                seq_len, self.max_len
            )));
        }

        let encoding = self.encode(seq_len);
        let mut output = input.clone();

        for mut batch in output.axis_iter_mut(Axis(0)) {
            Zip::from(&mut batch)
                .and(&encoding)
                .for_each(|b, &e| *b += e);
        }

        Ok(output)
    }

    fn clone_box(&self) -> Box<dyn PositionalEncoding<F> + Send + Sync>
    where
        F: Send + Sync + 'static,
    {
        Box::new(self.clone())
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn max_len(&self) -> usize {
        self.max_len
    }
}

/// Rotary Positional Encoding (RoPE)
///
/// Implements rotary position embeddings as described in "RoFormer: Enhanced Transformer
/// with Rotary Position Embedding". RoPE encodes position information by rotating
/// the query and key vectors, which has several advantages:
///
/// - Relative position information is naturally encoded
/// - Better length extrapolation than absolute position encodings
/// - Used in modern LLMs like LLaMA, GPT-NeoX, etc.
#[derive(Debug, Clone)]
pub struct RotaryPositionalEncoding<F: Float + Debug + NumAssign> {
    d_model: usize,
    max_len: usize,
    base: f64,
    /// Pre-computed sin values: [max_len, d_model/2]
    sin_cached: Array2<F>,
    /// Pre-computed cos values: [max_len, d_model/2]
    cos_cached: Array2<F>,
}

impl<F: Float + Debug + NumAssign> RotaryPositionalEncoding<F> {
    /// Create a new RoPE encoding
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (must be even)
    /// * `max_len` - Maximum sequence length
    /// * `base` - Base for frequency computation (default: 10000.0)
    pub fn new(d_model: usize, max_len: usize, base: f64) -> Self {
        assert!(d_model.is_multiple_of(2), "d_model must be even for RoPE");

        let (sin_cached, cos_cached) = Self::compute_rope_cache(d_model, max_len, base);

        Self {
            d_model,
            max_len,
            base,
            sin_cached,
            cos_cached,
        }
    }

    /// Create with default base (10000.0)
    pub fn default_base(d_model: usize, max_len: usize) -> Self {
        Self::new(d_model, max_len, 10000.0)
    }

    /// Compute the RoPE sin/cos cache
    fn compute_rope_cache(d_model: usize, max_len: usize, base: f64) -> (Array2<F>, Array2<F>) {
        let half_dim = d_model / 2;
        let mut sin_cached = Array2::zeros((max_len, half_dim));
        let mut cos_cached = Array2::zeros((max_len, half_dim));

        // Compute inverse frequencies
        for pos in 0..max_len {
            for i in 0..half_dim {
                let freq = 1.0 / base.powf((2 * i) as f64 / d_model as f64);
                let angle = pos as f64 * freq;

                sin_cached[[pos, i]] = F::from(angle.sin()).unwrap_or(F::zero());
                cos_cached[[pos, i]] = F::from(angle.cos()).unwrap_or(F::zero());
            }
        }

        (sin_cached, cos_cached)
    }

    /// Apply rotary embedding to query or key tensor
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, seq_len, d_model]
    /// * `offset` - Position offset (for KV cache during inference)
    ///
    /// # Returns
    /// Rotated tensor with same shape
    pub fn rotate(&self, x: &Array3<F>, offset: usize) -> Result<Array3<F>> {
        let seq_len = x.shape()[1];
        if seq_len + offset > self.max_len {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Position {} exceeds max_len {}",
                seq_len + offset,
                self.max_len
            )));
        }

        let batch_size = x.shape()[0];
        let half_dim = self.d_model / 2;

        let mut output = Array3::zeros(x.raw_dim());

        for b in 0..batch_size {
            for pos in 0..seq_len {
                let abs_pos = pos + offset;
                for i in 0..half_dim {
                    let x1 = x[[b, pos, 2 * i]];
                    let x2 = x[[b, pos, 2 * i + 1]];

                    let cos = self.cos_cached[[abs_pos, i]];
                    let sin = self.sin_cached[[abs_pos, i]];

                    // Apply rotation: [cos, -sin; sin, cos] * [x1; x2]
                    output[[b, pos, 2 * i]] = x1 * cos - x2 * sin;
                    output[[b, pos, 2 * i + 1]] = x1 * sin + x2 * cos;
                }
            }
        }

        Ok(output)
    }

    /// Get sin cache
    pub fn sin_cache(&self) -> &Array2<F> {
        &self.sin_cached
    }

    /// Get cos cache
    pub fn cos_cache(&self) -> &Array2<F> {
        &self.cos_cached
    }
}

impl<F: Float + Debug + NumAssign> PositionalEncoding<F> for RotaryPositionalEncoding<F> {
    /// RoPE multiplies each `(x_{2i}, x_{2i+1})` pair by the rotation matrix
    /// `[[cos, -sin], [sin, cos]]`. Rotations are orthogonal, so the Jacobian
    /// transpose is the rotation by the negated angle.
    fn backward(&self, grad_output: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        if grad_output.ndim() != 3 {
            return Err(NeuralError::ShapeMismatch(format!(
                "Expected a 3D gradient [batch, seq_len, d_model], got {}D",
                grad_output.ndim()
            )));
        }
        let shape = grad_output.shape();
        let (batch_size, seq_len, d_model) = (shape[0], shape[1], shape[2]);
        if d_model != self.d_model {
            return Err(NeuralError::ShapeMismatch(format!(
                "Gradient feature dimension {} does not match d_model {}",
                d_model, self.d_model
            )));
        }
        if seq_len > self.max_len {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Position {} exceeds max_len {}",
                seq_len, self.max_len
            )));
        }

        let half_dim = self.d_model / 2;
        let mut grad_input = Array::<F, IxDyn>::zeros(grad_output.raw_dim());
        for b in 0..batch_size {
            for pos in 0..seq_len {
                for i in 0..half_dim {
                    let g1 = grad_output[[b, pos, 2 * i]];
                    let g2 = grad_output[[b, pos, 2 * i + 1]];
                    let cos = self.cos_cached[[pos, i]];
                    let sin = self.sin_cached[[pos, i]];
                    grad_input[[b, pos, 2 * i]] = g1 * cos + g2 * sin;
                    grad_input[[b, pos, 2 * i + 1]] = -g1 * sin + g2 * cos;
                }
            }
        }
        Ok(grad_input)
    }

    fn encode(&self, seq_len: usize) -> Array2<F> {
        // Return combined sin/cos for compatibility
        // In practice, use rotate() method directly
        let half_dim = self.d_model / 2;
        let mut encoding = Array2::zeros((seq_len, self.d_model));

        for pos in 0..seq_len {
            for i in 0..half_dim {
                encoding[[pos, 2 * i]] = self.sin_cached[[pos, i]];
                encoding[[pos, 2 * i + 1]] = self.cos_cached[[pos, i]];
            }
        }

        encoding
    }

    fn apply(&self, input: &Array3<F>) -> Result<Array3<F>> {
        // For RoPE, apply is the same as rotate with offset 0
        self.rotate(input, 0)
    }

    fn clone_box(&self) -> Box<dyn PositionalEncoding<F> + Send + Sync>
    where
        F: Send + Sync + 'static,
    {
        Box::new(self.clone())
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn max_len(&self) -> usize {
        self.max_len
    }
}

/// Relative Positional Encoding
///
/// Implements relative position encodings that represent the distance between
/// positions rather than absolute positions. This allows better generalization
/// to longer sequences.
#[derive(Debug, Clone)]
pub struct RelativePositionalEncoding<F: Float + Debug + NumAssign> {
    d_model: usize,
    max_len: usize,
    /// Relative position embeddings: [2*max_len-1, d_model]
    /// Index 0 = position -(max_len-1), index max_len-1 = position 0
    rel_embeddings: Array2<F>,
}

impl<F: Float + Debug + NumAssign> RelativePositionalEncoding<F> {
    /// Create a new relative positional encoding
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `max_len` - Maximum sequence length
    /// * `rng` - Random number generator
    pub fn new<R: Rng>(d_model: usize, max_len: usize, rng: &mut R) -> Self {
        let num_positions = 2 * max_len - 1;
        let std = (1.0 / d_model as f64).sqrt();

        let mut rel_embeddings = Array2::zeros((num_positions, d_model));
        for elem in rel_embeddings.iter_mut() {
            let u1: f64 = rng.random_range(0.0001..1.0);
            let u2: f64 = rng.random_range(0.0..1.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            *elem = F::from(z * std).unwrap_or(F::zero());
        }

        Self {
            d_model,
            max_len,
            rel_embeddings,
        }
    }

    /// Get embedding for a relative position
    ///
    /// # Arguments
    /// * `rel_pos` - Relative position (can be negative)
    pub fn get_relative_embedding(&self, rel_pos: i64) -> Option<Array<F, IxDyn>> {
        let max_rel = self.max_len as i64 - 1;
        if rel_pos < -max_rel || rel_pos > max_rel {
            return None;
        }

        let idx = (rel_pos + max_rel) as usize;
        Some(self.rel_embeddings.slice(s![idx, ..]).to_owned().into_dyn())
    }

    /// Get relative position bias matrix for attention
    ///
    /// # Arguments
    /// * `query_len` - Length of query sequence
    /// * `key_len` - Length of key sequence
    ///
    /// # Returns
    /// Relative position bias of shape [query_len, key_len, d_model]
    pub fn get_attention_bias(&self, query_len: usize, key_len: usize) -> Result<Array3<F>> {
        if query_len > self.max_len || key_len > self.max_len {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Sequence length exceeds max_len {}",
                self.max_len
            )));
        }

        let mut bias = Array3::zeros((query_len, key_len, self.d_model));
        let max_rel = self.max_len as i64 - 1;

        for q in 0..query_len {
            for k in 0..key_len {
                let rel_pos = k as i64 - q as i64;
                let idx = (rel_pos + max_rel) as usize;

                for d in 0..self.d_model {
                    bias[[q, k, d]] = self.rel_embeddings[[idx, d]];
                }
            }
        }

        Ok(bias)
    }

    /// Get mutable reference to embeddings
    pub fn embeddings_mut(&mut self) -> &mut Array2<F> {
        &mut self.rel_embeddings
    }
}

impl<F: Float + Debug + NumAssign> PositionalEncoding<F> for RelativePositionalEncoding<F> {
    /// `apply` adds the centre embeddings to the input, so the gradient with
    /// respect to the input passes through unchanged.
    fn backward(&self, grad_output: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn encode(&self, seq_len: usize) -> Array2<F> {
        // For relative PE, return the central positions (around 0 relative position)
        let start = self.max_len - 1;
        self.rel_embeddings
            .slice(s![start..(start + seq_len), ..])
            .to_owned()
    }

    fn apply(&self, input: &Array3<F>) -> Result<Array3<F>> {
        // For relative PE, typically used differently in attention
        // This provides a simple fallback that adds the center embeddings
        let seq_len = input.shape()[1];
        if seq_len > self.max_len {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Sequence length {} exceeds max_len {}",
                seq_len, self.max_len
            )));
        }

        let encoding = self.encode(seq_len);
        let mut output = input.clone();

        for mut batch in output.axis_iter_mut(Axis(0)) {
            Zip::from(&mut batch)
                .and(&encoding)
                .for_each(|b, &e| *b += e);
        }

        Ok(output)
    }

    fn clone_box(&self) -> Box<dyn PositionalEncoding<F> + Send + Sync>
    where
        F: Send + Sync + 'static,
    {
        Box::new(self.clone())
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn max_len(&self) -> usize {
        self.max_len
    }
}

/// Factory for creating positional encodings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionalEncodingType {
    /// Fixed sinusoidal encoding (Transformer)
    Sinusoidal,
    /// Learnable position embeddings (BERT)
    Learned,
    /// Rotary position embeddings (RoPE)
    Rotary,
    /// Relative position encoding
    Relative,
}

/// Factory for creating positional encodings
pub struct PositionalEncodingFactory;

impl PositionalEncodingFactory {
    /// Create a positional encoding of the specified type
    pub fn create<F, R>(
        pe_type: PositionalEncodingType,
        d_model: usize,
        max_len: usize,
        rng: &mut R,
    ) -> Box<dyn PositionalEncoding<F> + Send + Sync>
    where
        F: Float + Debug + NumAssign + Send + Sync + 'static,
        R: Rng,
    {
        match pe_type {
            PositionalEncodingType::Sinusoidal => {
                Box::new(SinusoidalPositionalEncoding::new(d_model, max_len))
            }
            PositionalEncodingType::Learned => {
                Box::new(LearnedPositionalEncoding::new(d_model, max_len, rng))
            }
            PositionalEncodingType::Rotary => {
                Box::new(RotaryPositionalEncoding::default_base(d_model, max_len))
            }
            PositionalEncodingType::Relative => {
                Box::new(RelativePositionalEncoding::new(d_model, max_len, rng))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;
    use scirs2_core::random::SeedableRng;

    #[test]
    fn test_sinusoidal_encoding_shape() {
        let pe = SinusoidalPositionalEncoding::<f32>::new(64, 100);

        let encoding = pe.encode(10);
        assert_eq!(encoding.shape(), &[10, 64]);

        let encoding = pe.encode(50);
        assert_eq!(encoding.shape(), &[50, 64]);
    }

    #[test]
    fn test_sinusoidal_encoding_values() {
        let pe = SinusoidalPositionalEncoding::<f64>::new(4, 10);

        let encoding = pe.encode(3);

        // Position 0 should have sin(0)=0, cos(0)=1 for the first pair
        assert!((encoding[[0, 0]] - 0.0).abs() < 1e-6); // sin(0)
        assert!((encoding[[0, 1]] - 1.0).abs() < 1e-6); // cos(0)

        // Each position should have different values
        assert!((encoding[[0, 0]] - encoding[[1, 0]]).abs() > 1e-10);
    }

    #[test]
    fn test_sinusoidal_apply() {
        let pe = SinusoidalPositionalEncoding::<f32>::new(8, 20);

        let input = Array3::zeros((2, 10, 8)); // batch=2, seq=10, d_model=8
        let output = pe.apply(&input).expect("Operation failed");

        assert_eq!(output.shape(), input.shape());

        // Check that encoding was added
        let encoding = pe.encode(10);
        for b in 0..2 {
            for s in 0..10 {
                for d in 0..8 {
                    assert!((output[[b, s, d]] - encoding[[s, d]]).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_learned_encoding() {
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(42);
        let pe = LearnedPositionalEncoding::<f32>::new(32, 50, &mut rng);

        let encoding = pe.encode(10);
        assert_eq!(encoding.shape(), &[10, 32]);

        // Values should be initialized (not all zero)
        let sum: f32 = encoding.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.1);
    }

    #[test]
    fn test_learned_from_embeddings() {
        let embeddings = Array2::ones((20, 16));
        let pe = LearnedPositionalEncoding::<f32>::from_embeddings(embeddings);

        assert_eq!(pe.d_model(), 16);
        assert_eq!(pe.max_len(), 20);
    }

    #[test]
    fn test_rope_encoding() {
        let pe = RotaryPositionalEncoding::<f32>::default_base(64, 100);

        let encoding = pe.encode(10);
        assert_eq!(encoding.shape(), &[10, 64]);
    }

    #[test]
    fn test_rope_rotate() {
        let pe = RotaryPositionalEncoding::<f64>::default_base(8, 20);

        let input = Array3::ones((1, 5, 8));
        let rotated = pe.rotate(&input, 0).expect("Operation failed");

        assert_eq!(rotated.shape(), input.shape());

        // At position 0, cos(0)=1, sin(0)=0 so rotation is identity
        // Check position 1 or higher where rotation actually occurs
        let mut different = false;
        for pos in 1..5 {
            for i in 0..8 {
                if (rotated[[0, pos, i]] - input[[0, pos, i]]).abs() > 1e-6 {
                    different = true;
                    break;
                }
            }
            if different {
                break;
            }
        }
        assert!(
            different,
            "RoPE should modify input values at non-zero positions"
        );
    }

    #[test]
    fn test_rope_with_offset() {
        let pe = RotaryPositionalEncoding::<f32>::default_base(8, 100);

        let input = Array3::ones((1, 10, 8));

        let rotated_0 = pe.rotate(&input, 0).expect("Operation failed");
        let rotated_5 = pe.rotate(&input, 5).expect("Operation failed");

        // Different offsets should give different results
        let mut different = false;
        for s in 0..10 {
            for d in 0..8 {
                if (rotated_0[[0, s, d]] - rotated_5[[0, s, d]]).abs() > 1e-6 {
                    different = true;
                    break;
                }
            }
        }
        assert!(different, "Different offsets should give different results");
    }

    #[test]
    fn test_relative_encoding() {
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(42);
        let pe = RelativePositionalEncoding::<f32>::new(16, 30, &mut rng);

        // Check relative embedding retrieval
        let rel_0 = pe.get_relative_embedding(0);
        assert!(rel_0.is_some());

        let rel_pos = pe.get_relative_embedding(5);
        assert!(rel_pos.is_some());

        let rel_neg = pe.get_relative_embedding(-5);
        assert!(rel_neg.is_some());

        // Out of range should return None
        let out_of_range = pe.get_relative_embedding(100);
        assert!(out_of_range.is_none());
    }

    #[test]
    fn test_relative_attention_bias() {
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(42);
        let pe = RelativePositionalEncoding::<f32>::new(8, 20, &mut rng);

        let bias = pe.get_attention_bias(10, 10).expect("Operation failed");
        assert_eq!(bias.shape(), &[10, 10, 8]);

        // Diagonal should have same values (relative position 0)
        let rel_0 = pe.get_relative_embedding(0).expect("Operation failed");
        for i in 0..10 {
            for d in 0..8 {
                assert!((bias[[i, i, d]] - rel_0[[d]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_factory() {
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(42);

        let sinusoidal = PositionalEncodingFactory::create::<f32, _>(
            PositionalEncodingType::Sinusoidal,
            32,
            100,
            &mut rng,
        );
        assert_eq!(sinusoidal.d_model(), 32);

        let learned = PositionalEncodingFactory::create::<f32, _>(
            PositionalEncodingType::Learned,
            32,
            100,
            &mut rng,
        );
        assert_eq!(learned.d_model(), 32);

        let rotary = PositionalEncodingFactory::create::<f32, _>(
            PositionalEncodingType::Rotary,
            32,
            100,
            &mut rng,
        );
        assert_eq!(rotary.d_model(), 32);

        let relative = PositionalEncodingFactory::create::<f32, _>(
            PositionalEncodingType::Relative,
            32,
            100,
            &mut rng,
        );
        assert_eq!(relative.d_model(), 32);
    }

    #[test]
    fn test_sinusoidal_properties() {
        let pe = SinusoidalPositionalEncoding::<f64>::new(64, 1000);
        let encoding = pe.encode(100);

        // Each position should be unique
        for i in 0..99 {
            let mut same = true;
            for d in 0..64 {
                if (encoding[[i, d]] - encoding[[i + 1, d]]).abs() > 1e-10 {
                    same = false;
                    break;
                }
            }
            assert!(!same, "Adjacent positions should be different");
        }
    }

    #[test]
    fn test_max_len_error() {
        let pe = SinusoidalPositionalEncoding::<f32>::new(16, 10);

        let input = Array3::zeros((1, 20, 16)); // seq_len > max_len
        let result = pe.apply(&input);

        assert!(result.is_err());
    }
}
