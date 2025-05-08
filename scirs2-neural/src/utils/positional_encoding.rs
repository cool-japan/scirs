//! Positional encoding utilities for transformer models
//!
//! This module provides implementations of positional encoding techniques
//! used in transformer architectures to incorporate sequence order information.

use crate::error::{NeuralError, Result};
use ndarray::{Array, IxDyn};
use num_traits::Float;
use std::fmt::Debug;

/// Types of positional encoding available
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionalEncodingType {
    /// Sinusoidal positional encoding from the "Attention Is All You Need" paper
    Sinusoidal,
    /// Learned positional embeddings, which are trained along with the model
    Learned,
    /// Relative positional encoding that focuses on relative distances
    Relative,
}

/// Factory for creating positional encodings
pub struct PositionalEncodingFactory;

impl PositionalEncodingFactory {
    /// Create a positional encoding
    ///
    /// # Arguments
    ///
    /// * `encoding_type` - Type of positional encoding to create
    /// * `max_len` - Maximum sequence length
    /// * `d_model` - Model embedding dimension
    ///
    /// # Returns
    ///
    /// * Box containing the positional encoding implementation
    pub fn create<F: Float + Debug + 'static>(
        encoding_type: PositionalEncodingType,
        max_len: usize,
        d_model: usize,
    ) -> Result<Box<dyn PositionalEncoding<F>>> {
        match encoding_type {
            PositionalEncodingType::Sinusoidal => Ok(Box::new(SinusoidalPositionalEncoding::new(
                max_len, d_model,
            )?)),
            PositionalEncodingType::Learned => {
                Ok(Box::new(LearnedPositionalEncoding::new(max_len, d_model)))
            }
            PositionalEncodingType::Relative => {
                Ok(Box::new(RelativePositionalEncoding::new(max_len, d_model)))
            }
        }
    }
}

/// Trait for positional encoding implementations
pub trait PositionalEncoding<F: Float + Debug> {
    /// Apply positional encoding to input embeddings
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Input embeddings [batch, seq_len, d_model]
    ///
    /// # Returns
    ///
    /// * Embeddings with positional encoding added
    fn forward(&self, embeddings: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>>;

    /// Get the positional encoding matrix directly
    ///
    /// # Arguments
    ///
    /// * `seq_len` - Sequence length to generate encodings for
    ///
    /// # Returns
    ///
    /// * Positional encoding matrix [seq_len, d_model]
    fn get_encoding(&self, seq_len: usize) -> Result<Array<F, IxDyn>>;

    /// Update learnable parameters if any
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate for the update
    fn update(&mut self, learning_rate: F) -> Result<()>;

    /// Get learnable parameters if any
    ///
    /// # Returns
    ///
    /// * Vector of parameters as arrays
    fn params(&self) -> Vec<Array<F, IxDyn>> {
        Vec::new() // Default implementation returns empty vector
    }

    /// Set training mode (does nothing by default)
    fn set_training(&mut self, _training: bool) {
        // Default implementation does nothing
    }

    /// Get training mode (false by default)
    fn is_training(&self) -> bool {
        false
    }
}

/// Sinusoidal positional encoding from the "Attention Is All You Need" paper
///
/// Uses sine and cosine functions of different frequencies to encode position.
/// The advantage is that this can extrapolate to longer sequences than those
/// seen during training.
#[derive(Debug, Clone)]
pub struct SinusoidalPositionalEncoding<F: Float + Debug> {
    /// Maximum sequence length
    max_len: usize,
    /// Model embedding dimension
    d_model: usize,
    /// Pre-computed encoding matrix [max_len, d_model]
    encoding: Array<F, IxDyn>,
}

impl<F: Float + Debug + 'static> SinusoidalPositionalEncoding<F> {
    /// Create a new sinusoidal positional encoding
    ///
    /// # Arguments
    ///
    /// * `max_len` - Maximum sequence length
    /// * `d_model` - Model embedding dimension
    ///
    /// # Returns
    ///
    /// * A new sinusoidal positional encoding
    pub fn new(max_len: usize, d_model: usize) -> Result<Self> {
        if d_model % 2 != 0 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Model dimension ({}) must be even for sinusoidal positional encoding",
                d_model
            )));
        }

        let mut encoding = Array::<F, _>::zeros((max_len, d_model));

        for pos in 0..max_len {
            for i in 0..d_model / 2 {
                let div_term = F::from(10000.0)
                    .unwrap()
                    .powf(F::from(2.0 * i as f64 / d_model as f64).unwrap());

                // Use sin for even indices
                encoding[[pos, 2 * i]] = F::from(pos as f64).unwrap().sin() / div_term;

                // Use cos for odd indices
                encoding[[pos, 2 * i + 1]] = F::from(pos as f64).unwrap().cos() / div_term;
            }
        }

        Ok(Self {
            max_len,
            d_model,
            encoding: encoding.into_dyn(),
        })
    }
}

impl<F: Float + Debug + 'static> PositionalEncoding<F> for SinusoidalPositionalEncoding<F> {
    fn forward(&self, embeddings: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        if embeddings.ndim() < 2 {
            return Err(NeuralError::InferenceError(
                "Embeddings must have at least 2 dimensions [batch, seq_len, ...]".to_string(),
            ));
        }

        let embed_shape = embeddings.shape();
        let seq_len = embed_shape[1];

        if seq_len > self.max_len {
            return Err(NeuralError::InferenceError(format!(
                "Sequence length ({}) exceeds maximum length ({})",
                seq_len, self.max_len
            )));
        }

        // Get positional encoding for the current sequence length
        let pos_encoding = self.get_encoding(seq_len)?;

        // Create a mutable copy of the embeddings
        let mut output = embeddings.clone();

        // Add positional encoding to each batch element
        for batch_idx in 0..embed_shape[0] {
            let mut batch_slice = output.slice_mut(ndarray::s![batch_idx, .., ..]);

            // Add positional encoding
            for pos in 0..seq_len {
                for dim in 0..self.d_model {
                    batch_slice[[pos, dim]] = batch_slice[[pos, dim]] + pos_encoding[[pos, dim]];
                }
            }
        }

        Ok(output)
    }

    fn get_encoding(&self, seq_len: usize) -> Result<Array<F, IxDyn>> {
        if seq_len > self.max_len {
            return Err(NeuralError::InferenceError(format!(
                "Requested sequence length ({}) exceeds maximum length ({})",
                seq_len, self.max_len
            )));
        }

        // Return a slice of the pre-computed encoding, with proper dimension type
        Ok(self
            .encoding
            .slice(ndarray::s![0..seq_len, ..])
            .to_owned()
            .into_dyn())
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        // Sinusoidal encoding has no learnable parameters
        Ok(())
    }

    /// Set training mode (does nothing for sinusoidal encoding)
    fn set_training(&mut self, _training: bool) {
        // Sinusoidal encoding has no training mode
    }

    /// Get training mode (always false for sinusoidal encoding)
    fn is_training(&self) -> bool {
        false
    }
}

/// Learned positional embeddings
///
/// Uses a lookup table of learned position embeddings. These can potentially
/// capture more complex position patterns but don't extrapolate well to unseen positions.
pub struct LearnedPositionalEncoding<F: Float + Debug> {
    /// Maximum sequence length
    max_len: usize,
    /// Model embedding dimension
    d_model: usize,
    /// Learnable position embeddings [max_len, d_model]
    weights: Array<F, IxDyn>,
    /// Gradient of weights
    dweights: Array<F, IxDyn>,
}

impl<F: Float + Debug + 'static> LearnedPositionalEncoding<F> {
    /// Create a new learned positional encoding
    ///
    /// # Arguments
    ///
    /// * `max_len` - Maximum sequence length
    /// * `d_model` - Model embedding dimension
    ///
    /// # Returns
    ///
    /// * A new learned positional encoding
    pub fn new(max_len: usize, d_model: usize) -> Self {
        // Initialize to small random values
        let init_scale = F::from(0.02).unwrap();
        let weights = Array::<F, _>::from_elem((max_len, d_model), init_scale);
        let dweights = Array::<F, _>::zeros((max_len, d_model));

        Self {
            max_len,
            d_model,
            weights: weights.into_dyn(),
            dweights: dweights.into_dyn(),
        }
    }
}

impl<F: Float + Debug + 'static> PositionalEncoding<F> for LearnedPositionalEncoding<F> {
    fn forward(&self, embeddings: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        if embeddings.ndim() < 2 {
            return Err(NeuralError::InferenceError(
                "Embeddings must have at least 2 dimensions [batch, seq_len, ...]".to_string(),
            ));
        }

        let embed_shape = embeddings.shape();
        let seq_len = embed_shape[1];

        if seq_len > self.max_len {
            return Err(NeuralError::InferenceError(format!(
                "Sequence length ({}) exceeds maximum length ({})",
                seq_len, self.max_len
            )));
        }

        // Get positional encoding for the current sequence length
        let pos_encoding = self.get_encoding(seq_len)?;

        // Create a mutable copy of the embeddings
        let mut output = embeddings.clone();

        // Add positional encoding to each batch element
        for batch_idx in 0..embed_shape[0] {
            let mut batch_slice = output.slice_mut(ndarray::s![batch_idx, .., ..]);

            // Add positional encoding
            for pos in 0..seq_len {
                for dim in 0..self.d_model {
                    batch_slice[[pos, dim]] = batch_slice[[pos, dim]] + pos_encoding[[pos, dim]];
                }
            }
        }

        Ok(output)
    }

    fn get_encoding(&self, seq_len: usize) -> Result<Array<F, IxDyn>> {
        if seq_len > self.max_len {
            return Err(NeuralError::InferenceError(format!(
                "Requested sequence length ({}) exceeds maximum length ({})",
                seq_len, self.max_len
            )));
        }

        // Return a slice of the weights, with proper dimension type
        Ok(self
            .weights
            .slice(ndarray::s![0..seq_len, ..])
            .to_owned()
            .into_dyn())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update weights using the stored gradients
        // This is a simplified implementation - in practice, an optimizer would be used
        let small_change = F::from(0.001).unwrap();
        let lr = learning_rate * small_change;

        for i in 0..self.max_len {
            for j in 0..self.d_model {
                self.weights[[i, j]] = self.weights[[i, j]] - lr * self.dweights[[i, j]];
            }
        }

        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        // Return the learnable weights
        vec![self.weights.clone()]
    }
}

/// Relative positional encoding
///
/// Implements a form of relative positional encoding that can capture
/// pairwise positional relationships efficiently.
pub struct RelativePositionalEncoding<F: Float + Debug> {
    /// Maximum sequence length
    max_len: usize,
    /// Model embedding dimension
    d_model: usize,
    /// Relative position embeddings [2*max_len-1, d_model]
    weights: Array<F, IxDyn>,
    /// Gradient of weights
    dweights: Array<F, IxDyn>,
}

impl<F: Float + Debug + 'static> RelativePositionalEncoding<F> {
    /// Create a new relative positional encoding
    ///
    /// # Arguments
    ///
    /// * `max_len` - Maximum sequence length
    /// * `d_model` - Model embedding dimension
    ///
    /// # Returns
    ///
    /// * A new relative positional encoding
    pub fn new(max_len: usize, d_model: usize) -> Self {
        // For relative positions, we need 2*max_len-1 different embeddings
        // to represent all possible relative positions from -(max_len-1) to +(max_len-1)
        let rel_size = 2 * max_len - 1;

        // Initialize to small random values
        let init_scale = F::from(0.02).unwrap();
        let weights = Array::<F, _>::from_elem((rel_size, d_model), init_scale);
        let dweights = Array::<F, _>::zeros((rel_size, d_model));

        Self {
            max_len,
            d_model,
            weights: weights.into_dyn(),
            dweights: dweights.into_dyn(),
        }
    }

    /// Get the relative position index
    ///
    /// Convert a relative position to the corresponding index in the weights matrix
    #[allow(dead_code)]
    fn rel_pos_to_index(&self, rel_pos: isize) -> usize {
        // Convert relative position to an index
        // rel_pos = -max_len+1 -> index = 0
        // rel_pos = 0 -> index = max_len-1
        // rel_pos = max_len-1 -> index = 2*max_len-2
        (rel_pos + self.max_len as isize - 1) as usize
    }
}

impl<F: Float + Debug + 'static> PositionalEncoding<F> for RelativePositionalEncoding<F> {
    fn forward(&self, embeddings: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        if embeddings.ndim() < 2 {
            return Err(NeuralError::InferenceError(
                "Embeddings must have at least 2 dimensions [batch, seq_len, ...]".to_string(),
            ));
        }

        let embed_shape = embeddings.shape();
        let seq_len = embed_shape[1];

        if seq_len > self.max_len {
            return Err(NeuralError::InferenceError(format!(
                "Sequence length ({}) exceeds maximum length ({})",
                seq_len, self.max_len
            )));
        }

        // For relative positions, we typically don't add them directly to the embeddings
        // but instead use them in the attention mechanism
        // For compatibility with the positional encoding interface, we'll just return the input
        Ok(embeddings.clone())
    }

    fn get_encoding(&self, seq_len: usize) -> Result<Array<F, IxDyn>> {
        if seq_len > self.max_len {
            return Err(NeuralError::InferenceError(format!(
                "Requested sequence length ({}) exceeds maximum length ({})",
                seq_len, self.max_len
            )));
        }

        // For relative positional encoding, we'd need to compute an attention bias
        // Based on all pairwise distances between positions
        // For simplicity, we'll just return a zero matrix matching the interface
        let encoding = Array::<F, _>::zeros((seq_len, self.d_model));
        Ok(encoding.into_dyn())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update weights using the stored gradients
        // This is a simplified implementation - in practice, an optimizer would be used
        let small_change = F::from(0.001).unwrap();
        let lr = learning_rate * small_change;

        let rel_size = 2 * self.max_len - 1;
        for i in 0..rel_size {
            for j in 0..self.d_model {
                self.weights[[i, j]] = self.weights[[i, j]] - lr * self.dweights[[i, j]];
            }
        }

        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        // Return the learnable weights
        vec![self.weights.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array3;

    #[test]
    fn test_sinusoidal_encoding_shape() {
        let max_len = 100;
        let d_model = 64;

        let pos_enc = SinusoidalPositionalEncoding::<f64>::new(max_len, d_model).unwrap();

        // Get encoding for a specific sequence length
        let encoding = pos_enc.get_encoding(50).unwrap();

        // Check shape
        assert_eq!(encoding.shape(), &[50, d_model]);
    }

    #[test]
    fn test_sinusoidal_encoding_properties() {
        let max_len = 100;
        let d_model = 64;

        let pos_enc = SinusoidalPositionalEncoding::<f64>::new(max_len, d_model).unwrap();

        // Get encoding for a specific sequence length
        let encoding = pos_enc.get_encoding(max_len).unwrap();

        // Check that different positions have different encodings
        let pos0 = encoding.slice(ndarray::s![0, ..]).to_owned();
        let pos1 = encoding.slice(ndarray::s![1, ..]).to_owned();

        // At least one element should be different
        let mut all_equal = true;
        for i in 0..d_model {
            if (pos0[i] - pos1[i]).abs() > 1e-10 {
                all_equal = false;
                break;
            }
        }

        assert!(
            !all_equal,
            "Positions 0 and 1 should have different encodings"
        );
    }

    #[test]
    fn test_positional_encoding_factory() {
        let max_len = 100;
        let d_model = 64;

        // Create different types of encodings
        let sinusoidal = PositionalEncodingFactory::create::<f64>(
            PositionalEncodingType::Sinusoidal,
            max_len,
            d_model,
        )
        .unwrap();

        let learned = PositionalEncodingFactory::create::<f64>(
            PositionalEncodingType::Learned,
            max_len,
            d_model,
        )
        .unwrap();

        let relative = PositionalEncodingFactory::create::<f64>(
            PositionalEncodingType::Relative,
            max_len,
            d_model,
        )
        .unwrap();

        // Create a test batch
        let batch_size = 2;
        let seq_len = 10;
        let embeddings = Array3::<f64>::zeros((batch_size, seq_len, d_model)).into_dyn();

        // Verify that all encodings can process the embeddings
        let _ = sinusoidal.forward(&embeddings).unwrap();
        let _ = learned.forward(&embeddings).unwrap();
        let _ = relative.forward(&embeddings).unwrap();
    }

    #[test]
    fn test_sinusoidal_encoding_addition() {
        let max_len = 100;
        let d_model = 64;

        let pos_enc = SinusoidalPositionalEncoding::<f64>::new(max_len, d_model).unwrap();

        // Create a batch with non-zero values
        let batch_size = 2;
        let seq_len = 10;
        let embeddings = Array3::<f64>::from_elem((batch_size, seq_len, d_model), 1.0).into_dyn();

        // Get encoding
        let encoding = pos_enc.get_encoding(seq_len).unwrap();

        // Apply positional encoding
        let output = pos_enc.forward(&embeddings).unwrap();

        // Verify that the embeddings were modified by the encoding
        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..d_model {
                    // output[b,s,d] should be embeddings[b,s,d] + encoding[s,d]
                    assert_relative_eq!(output[[b, s, d]], 1.0 + encoding[[s, d]], epsilon = 1e-10);
                }
            }
        }
    }
}
