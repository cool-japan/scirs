//! Sequence-to-Sequence (Seq2Seq) model architectures
//!
//! This module implements various RNN-based sequence models including:
//! - Encoder-Decoder architectures
//! - Sequence-to-Sequence with attention
//! - Bidirectional RNN encoder with attention
//!
//! These models are useful for machine translation, text summarization,
//! speech recognition, and other sequence generation tasks.

// use crate::activations::Softmax;
use crate::error::{NeuralError, Result};
use crate::layers::{
    Dense,
    Dropout,
    Embedding,
    EmbeddingConfig,
    Layer,
    RNNConfig,
    RecurrentActivation,
    // Thread-safe versions for multi-threading
    ThreadSafeBidirectional,
    ThreadSafeRNN,
    ThreadSafeRecurrentActivation,
};
use rand::SeedableRng;

use ndarray::{Array, Axis, IxDyn, ScalarOperand};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// RNN cell types for sequence models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RNNCellType {
    /// Simple RNN cell
    SimpleRNN,
    /// LSTM (Long Short-Term Memory) cell
    LSTM,
    /// GRU (Gated Recurrent Unit) cell
    GRU,
}

/// Configuration for Seq2Seq models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Seq2SeqConfig {
    /// Vocabulary size for input encoder
    pub input_vocab_size: usize,
    /// Vocabulary size for output decoder
    pub output_vocab_size: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Hidden dimension for RNN cells
    pub hidden_dim: usize,
    /// Number of RNN layers
    pub num_layers: usize,
    /// Cell type for encoder
    pub encoder_cell_type: RNNCellType,
    /// Cell type for decoder
    pub decoder_cell_type: RNNCellType,
    /// Whether to use bidirectional encoder
    pub bidirectional_encoder: bool,
    /// Whether to use attention
    pub use_attention: bool,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl Default for Seq2SeqConfig {
    fn default() -> Self {
        Self {
            input_vocab_size: 10000,
            output_vocab_size: 10000,
            embedding_dim: 256,
            hidden_dim: 512,
            num_layers: 2,
            encoder_cell_type: RNNCellType::LSTM,
            decoder_cell_type: RNNCellType::LSTM,
            bidirectional_encoder: true,
            use_attention: true,
            dropout_rate: 0.1,
            max_seq_len: 100,
        }
    }
}

/// Attention mechanism for sequence models
#[derive(Debug, Clone)]
pub struct Attention<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Attention projection for decoder state
    pub decoder_projection: Dense<F>,
    /// Attention projection for encoder outputs
    pub encoder_projection: Option<Dense<F>>,
    /// Combined projection
    pub combined_projection: Dense<F>,
    /// Output projection
    pub output_projection: Dense<F>,
    /// Attention type
    pub attention_type: AttentionType,
    /// Whether encoder outputs are bidirectional
    pub bidirectional_encoder: bool,
}

/// Types of attention mechanisms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionType {
    /// Additive attention (Bahdanau)
    Additive,
    /// Multiplicative attention (Luong)
    Multiplicative,
    /// General attention (learned projection)
    General,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Attention<F> {
    /// Create a new Attention module
    pub fn new(
        decoder_dim: usize,
        encoder_dim: usize,
        attention_dim: usize,
        attention_type: AttentionType,
        bidirectional_encoder: bool,
    ) -> Result<Self> {
        // Create a random number generator for initialization
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        // Create projections based on attention type
        let decoder_projection = Dense::<F>::new(decoder_dim, attention_dim, None, &mut rng)?;

        // For additive attention, we need to project encoder outputs
        let encoder_projection = if attention_type == AttentionType::Additive {
            Some(Dense::<F>::new(encoder_dim, attention_dim, None, &mut rng)?)
        } else {
            None
        };

        // For multiplicative and general attention
        let combined_dim = match attention_type {
            AttentionType::Additive => attention_dim,
            AttentionType::Multiplicative => 1,
            AttentionType::General => encoder_dim,
        };

        let combined_projection = Dense::<F>::new(combined_dim, 1, None, &mut rng)?;

        // Project context vector and decoder state for output
        let output_projection =
            Dense::<F>::new(encoder_dim + decoder_dim, decoder_dim, None, &mut rng)?;

        Ok(Self {
            decoder_projection,
            encoder_projection,
            combined_projection,
            output_projection,
            attention_type,
            bidirectional_encoder,
        })
    }

    /// Compute attention weights and context vector
    pub fn forward(
        &self,
        decoder_state: &Array<F, IxDyn>,
        encoder_outputs: &Array<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        // Get shapes
        let batch_size = decoder_state.shape()[0];
        let seq_len = encoder_outputs.shape()[1];
        let encoder_dim = encoder_outputs.shape()[2];

        // Project decoder state
        let decoder_projected = self.decoder_projection.forward(decoder_state)?;

        // Compute attention scores based on attention type
        let attention_scores = match self.attention_type {
            AttentionType::Additive => {
                // Project encoder outputs if needed
                let encoder_projected = if let Some(ref proj) = self.encoder_projection {
                    // Reshape for projection
                    let flat_encoder = encoder_outputs
                        .to_owned()
                        .into_shape_with_order((batch_size * seq_len, encoder_dim))?;
                    let projected = proj.forward(&flat_encoder.into_dyn())?;
                    let proj_shape = projected.shape()[1];
                    projected
                        .into_shape_with_order((batch_size, seq_len, proj_shape))?
                        .into_dyn()
                } else {
                    return Err(NeuralError::InferenceError(
                        "Encoder projection missing for additive attention".to_string(),
                    ));
                };

                // Expand decoder state for broadcasting
                let expanded_decoder = decoder_projected.to_owned().into_shape_with_order((
                    batch_size,
                    1,
                    decoder_projected.shape()[1],
                ))?;
                let expanded = expanded_decoder
                    .broadcast((batch_size, seq_len, expanded_decoder.shape()[2]))
                    .unwrap();

                // Add encoder and decoder projections
                let combined = &expanded + &encoder_projected;

                // Apply tanh and project to get scores
                let tanh = combined.mapv(|x| x.tanh());
                let flat_tanh = tanh
                    .to_owned()
                    .into_shape_with_order((batch_size * seq_len, tanh.shape()[2]))?;
                let scores = self.combined_projection.forward(&flat_tanh.into_dyn())?;
                scores
                    .into_shape_with_order((batch_size, seq_len))?
                    .into_dyn()
            }
            AttentionType::Multiplicative => {
                // Expand decoder state for each encoder position
                let expanded_decoder = decoder_projected.to_owned().into_shape_with_order((
                    batch_size,
                    1,
                    decoder_projected.shape()[1],
                ))?;

                // Batched dot product
                let mut scores = Array::<F, _>::zeros((batch_size, seq_len));

                for b in 0..batch_size {
                    let decoder_slice = expanded_decoder.slice(ndarray::s![b, 0, ..]);

                    for s in 0..seq_len {
                        let encoder_slice = encoder_outputs.slice(ndarray::s![b, s, ..]);
                        // Manually calculate dot product to avoid ambiguity
                        let mut dot_product = F::zero();
                        for i in 0..decoder_slice.len() {
                            dot_product = dot_product + decoder_slice[i] * encoder_slice[i];
                        }
                        scores[[b, s]] = dot_product;
                    }
                }

                scores.into_dyn()
            }
            AttentionType::General => {
                // Project decoder state once (used as a weight matrix)
                let weight_matrix = decoder_projected.to_owned();

                // Batched matrix multiply
                let mut scores = Array::<F, _>::zeros((batch_size, seq_len));

                for b in 0..batch_size {
                    let weight = weight_matrix.slice(ndarray::s![b, ..]);

                    for s in 0..seq_len {
                        let encoder_slice = encoder_outputs.slice(ndarray::s![b, s, ..]);
                        // Manually calculate dot product to avoid ambiguity
                        let mut dot_product = F::zero();
                        for i in 0..weight.len() {
                            dot_product = dot_product + weight[i] * encoder_slice[i];
                        }
                        scores[[b, s]] = dot_product;
                    }
                }

                scores.into_dyn()
            }
        };

        // Apply softmax to get attention weights
        let mut attention_weights = Array::<F, _>::zeros(attention_scores.raw_dim());

        // Manual softmax implementation
        for b in 0..batch_size {
            let mut row = attention_scores.slice(ndarray::s![b, ..]).to_owned();

            // Find max for numerical stability
            let max_val = row.fold(F::neg_infinity(), |m, &v| m.max(v));

            // Compute exp and sum
            let mut exp_sum = F::zero();
            for i in 0..seq_len {
                let exp_val = (row[i] - max_val).exp();
                row[i] = exp_val;
                exp_sum = exp_sum + exp_val;
            }

            // Normalize
            if exp_sum > F::zero() {
                for i in 0..seq_len {
                    row[i] = row[i] / exp_sum;
                }
            }

            // Copy normalized weights
            for i in 0..seq_len {
                attention_weights[[b, i]] = row[i];
            }
        }

        // Compute context vector
        let attention_weights_expanded = attention_weights
            .to_owned()
            .into_shape_with_order((batch_size, seq_len, 1))?;
        let broadcast_weights = attention_weights_expanded
            .broadcast((batch_size, seq_len, encoder_dim))
            .unwrap();

        // Element-wise multiply and sum over sequence dimension
        let weighted_encoder = encoder_outputs * &broadcast_weights;
        let context = weighted_encoder.sum_axis(Axis(1));

        // Concatenate context and decoder state - ensure both are in the same format for stacking
        let decoder_state_dyn = decoder_state.to_owned().into_dyn();
        let decoder_and_context =
            ndarray::stack(Axis(1), &[context.view(), decoder_state_dyn.view()])?;
        let flattened = decoder_and_context
            .into_shape_with_order((batch_size, context.shape()[1] + decoder_state.shape()[1]))?;

        // Project combined vector - convert to IxDyn for Layer trait
        let flattened_dyn = flattened.to_owned().into_dyn();
        let output = self.output_projection.forward(&flattened_dyn)?;

        Ok((output, attention_weights))
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for Attention<F> {
    fn forward(&self, _input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        Err(NeuralError::InvalidArchitecture("Attention layer requires separate decoder state and encoder outputs. Use the dedicated forward method.".to_string()))
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Err(NeuralError::NotImplemented(
            "Backward pass for Attention is not implemented yet".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Err(NeuralError::NotImplemented(
            "Update for Attention is not implemented yet".to_string(),
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        params.extend(self.decoder_projection.params());

        if let Some(ref proj) = self.encoder_projection {
            params.extend(proj.params());
        }

        params.extend(self.combined_projection.params());
        params.extend(self.output_projection.params());

        params
    }

    fn set_training(&mut self, training: bool) {
        self.decoder_projection.set_training(training);

        if let Some(ref mut proj) = self.encoder_projection {
            proj.set_training(training);
        }

        self.combined_projection.set_training(training);
        self.output_projection.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.decoder_projection.is_training()
    }
}

/// Encoder for Seq2Seq models
// TODO: Implement Debug and Clone manually once the contained types support it
pub struct Seq2SeqEncoder<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Input embedding layer
    pub embedding: Embedding<F>,
    /// RNN layers
    pub rnn_layers: Vec<Box<dyn Layer<F> + Send + Sync>>,
    /// Dropout layer
    pub dropout: Option<Dropout<F>>,
    /// Whether the encoder is bidirectional
    pub bidirectional: bool,
    /// RNN cell type
    pub cell_type: RNNCellType,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of layers
    pub num_layers: usize,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Seq2SeqEncoder<F> {
    /// Create a new Seq2SeqEncoder
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        cell_type: RNNCellType,
        bidirectional: bool,
        dropout_rate: Option<f64>,
    ) -> Result<Self> {
        // Create embedding layer with config
        let embedding_config = EmbeddingConfig {
            num_embeddings: vocab_size,
            embedding_dim,
            padding_idx: None,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            sparse: false,
        };
        let embedding = Embedding::<F>::new(embedding_config)?;

        // Create RNN layers
        let mut rnn_layers: Vec<Box<dyn Layer<F> + Send + Sync>> = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let input_size = if i == 0 {
                embedding_dim
            } else if bidirectional && i > 0 {
                hidden_dim * 2
            } else {
                hidden_dim
            };

            // Create the appropriate RNN layer based on cell type
            let rnn: Box<dyn Layer<F> + Send + Sync> = match cell_type {
                RNNCellType::SimpleRNN => {
                    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
                    let config = RNNConfig {
                        input_size,
                        hidden_size: hidden_dim,
                        activation: RecurrentActivation::Tanh,
                    };

                    // Use thread-safe RNN implementation to ensure multi-threading compatibility
                    let rnn = ThreadSafeRNN::<F>::new(
                        config.input_size,
                        config.hidden_size,
                        ThreadSafeRecurrentActivation::Tanh, // Convert activation
                        &mut rng,
                    )?;

                    if bidirectional {
                        // Use thread-safe bidirectional wrapper for multi-threading
                        let brnn = ThreadSafeBidirectional::new(Box::new(rnn), None)?;
                        Box::new(brnn)
                    } else {
                        Box::new(rnn)
                    }
                }
                RNNCellType::LSTM => {
                    // Use thread-safe RNN as a replacement for LSTM until LSTM is made thread-safe
                    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

                    // For true thread safety, we'll use our ThreadSafeRNN with tanh activation
                    // as a temporary replacement for LSTM
                    let rnn = ThreadSafeRNN::<F>::new(
                        input_size,
                        hidden_dim,
                        ThreadSafeRecurrentActivation::Tanh,
                        &mut rng,
                    )?;

                    if bidirectional {
                        // Use thread-safe bidirectional wrapper
                        let brnn = ThreadSafeBidirectional::new(Box::new(rnn), None)?;
                        Box::new(brnn)
                    } else {
                        Box::new(rnn)
                    }
                }
                RNNCellType::GRU => {
                    // Use thread-safe RNN as a replacement for GRU until GRU is made thread-safe
                    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

                    // For true thread safety, we'll use our ThreadSafeRNN with tanh activation
                    // as a temporary replacement for GRU
                    let rnn = ThreadSafeRNN::<F>::new(
                        input_size,
                        hidden_dim,
                        ThreadSafeRecurrentActivation::Tanh,
                        &mut rng,
                    )?;

                    if bidirectional {
                        // Use thread-safe bidirectional wrapper
                        let brnn = ThreadSafeBidirectional::new(Box::new(rnn), None)?;
                        Box::new(brnn)
                    } else {
                        Box::new(rnn)
                    }
                }
            };

            rnn_layers.push(rnn);
        }

        // Create dropout layer if needed
        let dropout = if let Some(rate) = dropout_rate {
            if rate > 0.0 {
                let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
                Some(Dropout::<F>::new(rate, &mut rng)?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            embedding,
            rnn_layers,
            dropout,
            bidirectional,
            cell_type,
            hidden_dim,
            num_layers,
        })
    }

    /// Forward pass through the encoder
    pub fn forward(
        &self,
        input_seq: &Array<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Vec<Array<F, IxDyn>>)> {
        // Apply embedding
        let mut x = self.embedding.forward(input_seq)?;

        // Apply dropout if available
        if let Some(ref dropout) = self.dropout {
            x = dropout.forward(&x)?;
        }

        // Process through RNN layers
        let mut states = Vec::new();

        for layer in &self.rnn_layers {
            // Each RNN layer returns sequences and final state
            let output = layer.forward(&x)?;

            // For bidirectional layers, we need to concatenate forward and backward states
            if self.bidirectional {
                // Extract sequences (first element) and states
                let sequences = output
                    .slice_axis(Axis(1), ndarray::Slice::from(0..1))
                    .into_shape_with_order((
                        output.shape()[0],
                        output.shape()[2],
                        output.shape()[3],
                    ))?
                    .to_owned(); // Convert to owned array

                let state = output
                    .slice_axis(Axis(1), ndarray::Slice::from(1..2))
                    .into_shape_with_order((output.shape()[0], output.shape()[3]))?
                    .to_owned(); // Convert to owned array

                x = sequences.into_dyn();
                states.push(state.into_dyn());
            } else {
                // Extract sequences (first element) and state (second element)
                let sequences = output
                    .slice_axis(Axis(1), ndarray::Slice::from(0..1))
                    .into_shape_with_order((
                        output.shape()[0],
                        output.shape()[2],
                        output.shape()[3],
                    ))?
                    .to_owned(); // Convert to owned array

                let state = output
                    .slice_axis(Axis(1), ndarray::Slice::from(1..2))
                    .into_shape_with_order((output.shape()[0], output.shape()[3]))?
                    .to_owned(); // Convert to owned array

                x = sequences.into_dyn();
                states.push(state.into_dyn());
            }
        }

        Ok((x, states))
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for Seq2SeqEncoder<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // This simplified version only returns the output sequences
        // For the full state, use the dedicated forward method
        let (output, _) = self.forward(input)?;
        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Err(NeuralError::NotImplemented(
            "Not implemented yet".to_string(),
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Err(NeuralError::NotImplemented(
            "Not implemented yet".to_string(),
        ))
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        params.extend(self.embedding.params());

        for layer in &self.rnn_layers {
            params.extend(layer.params());
        }

        if let Some(ref dropout) = self.dropout {
            params.extend(dropout.params());
        }

        params
    }

    fn set_training(&mut self, training: bool) {
        self.embedding.set_training(training);

        for layer in &mut self.rnn_layers {
            layer.set_training(training);
        }

        if let Some(ref mut dropout) = self.dropout {
            dropout.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.embedding.is_training()
    }
}

/// Decoder for Seq2Seq models
// TODO: Implement Debug and Clone manually once the contained types support it
pub struct Seq2SeqDecoder<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Output embedding layer
    pub embedding: Embedding<F>,
    /// RNN layers
    pub rnn_layers: Vec<Box<dyn Layer<F> + Send + Sync>>,
    /// Attention mechanism (optional)
    pub attention: Option<Attention<F>>,
    /// Output projection
    pub output_projection: Dense<F>,
    /// Dropout layer
    pub dropout: Option<Dropout<F>>,
    /// RNN cell type
    pub cell_type: RNNCellType,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Output vocabulary size
    pub vocab_size: usize,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Seq2SeqDecoder<F> {
    /// Create a new Seq2SeqDecoder
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        cell_type: RNNCellType,
        use_attention: bool,
        encoder_bidirectional: bool,
        dropout_rate: Option<f64>,
    ) -> Result<Self> {
        // Create embedding layer with config
        let embedding_config = EmbeddingConfig {
            num_embeddings: vocab_size,
            embedding_dim,
            padding_idx: None,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            sparse: false,
        };
        let embedding = Embedding::<F>::new(embedding_config)?;

        // Create RNN layers
        let mut rnn_layers: Vec<Box<dyn Layer<F> + Send + Sync>> = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let input_size = if i == 0 { embedding_dim } else { hidden_dim };

            // Create the appropriate RNN layer based on cell type
            let rnn: Box<dyn Layer<F> + Send + Sync> = match cell_type {
                RNNCellType::SimpleRNN => {
                    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
                    let config = RNNConfig {
                        input_size,
                        hidden_size: hidden_dim,
                        activation: RecurrentActivation::Tanh,
                    };

                    // Use thread-safe RNN implementation to ensure multi-threading compatibility
                    let rnn = ThreadSafeRNN::<F>::new(
                        config.input_size,
                        config.hidden_size,
                        ThreadSafeRecurrentActivation::Tanh, // Convert activation
                        &mut rng,
                    )?;
                    Box::new(rnn)
                }
                RNNCellType::LSTM => {
                    // Use thread-safe RNN as a replacement for LSTM until LSTM is made thread-safe
                    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

                    // For true thread safety, we'll use our ThreadSafeRNN with tanh activation
                    // as a temporary replacement for LSTM
                    let rnn = ThreadSafeRNN::<F>::new(
                        input_size,
                        hidden_dim,
                        ThreadSafeRecurrentActivation::Tanh,
                        &mut rng,
                    )?;
                    Box::new(rnn)
                }
                RNNCellType::GRU => {
                    // Use thread-safe RNN as a replacement for GRU until GRU is made thread-safe
                    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

                    // For true thread safety, we'll use our ThreadSafeRNN with tanh activation
                    // as a temporary replacement for GRU
                    let rnn = ThreadSafeRNN::<F>::new(
                        input_size,
                        hidden_dim,
                        ThreadSafeRecurrentActivation::Tanh,
                        &mut rng,
                    )?;
                    Box::new(rnn)
                }
            };

            rnn_layers.push(rnn);
        }

        // Create attention mechanism if needed
        let attention = if use_attention {
            let encoder_dim = if encoder_bidirectional {
                hidden_dim * 2
            } else {
                hidden_dim
            };

            Some(Attention::<F>::new(
                hidden_dim,
                encoder_dim,
                hidden_dim,
                AttentionType::Additive,
                encoder_bidirectional,
            )?)
        } else {
            None
        };

        // Create output projection with activation function
        let mut rng_clone = rand::rngs::SmallRng::seed_from_u64(42);
        let output_projection = Dense::<F>::new(
            hidden_dim,
            vocab_size,
            None, // No custom activation function
            &mut rng_clone,
        )?;

        // Create dropout layer if needed
        let dropout = if let Some(rate) = dropout_rate {
            if rate > 0.0 {
                let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
                Some(Dropout::<F>::new(rate, &mut rng)?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            embedding,
            rnn_layers,
            attention,
            output_projection,
            dropout,
            cell_type,
            hidden_dim,
            num_layers,
            vocab_size,
        })
    }

    /// Forward pass through the decoder (single step)
    pub fn forward_step(
        &self,
        input_tokens: &Array<F, IxDyn>,
        prev_states: &[Array<F, IxDyn>],
        encoder_outputs: Option<&Array<F, IxDyn>>,
    ) -> Result<(Array<F, IxDyn>, Vec<Array<F, IxDyn>>)> {
        // Apply embedding
        let mut x = self.embedding.forward(input_tokens)?;

        // Apply dropout if available
        if let Some(ref dropout) = self.dropout {
            x = dropout.forward(&x)?;
        }

        // Process through RNN layers with initial states
        let mut states_out = Vec::new();

        for (i, layer) in self.rnn_layers.iter().enumerate() {
            // Each RNN layer returns sequences and final state
            let prev_state = if i < prev_states.len() {
                Some(&prev_states[i])
            } else {
                None
            };

            // Forward pass with initial state
            let output = if let Some(state) = prev_state {
                // Prepare initial state format expected by the RNN layer
                let initial_state = state.to_owned().into_shape_with_order((
                    state.shape()[0],
                    1,
                    state.shape()[1],
                ))?;

                let x_dyn = x.to_owned().into_dyn();
                let initial_state_dyn = initial_state.to_owned().into_dyn();
                let combined_input =
                    ndarray::stack(Axis(1), &[x_dyn.view(), initial_state_dyn.view()])?;
                layer.forward(&combined_input.to_owned().into_dyn())?
            } else {
                layer.forward(&x.to_owned().into_dyn())?
            };

            // Extract sequences (first element) and state (second element)
            let sequences = output
                .slice_axis(Axis(1), ndarray::Slice::from(0..1))
                .into_shape_with_order((output.shape()[0], output.shape()[2], output.shape()[3]))?
                .to_owned();

            let state = output
                .slice_axis(Axis(1), ndarray::Slice::from(1..2))
                .into_shape_with_order((output.shape()[0], output.shape()[3]))?
                .to_owned();

            x = sequences.into_dyn();
            states_out.push(state.into_dyn());
        }

        // Apply attention if available
        let final_output = if let Some(ref attention) = self.attention {
            if let Some(encoder_out) = encoder_outputs {
                // Get the last RNN layer's output
                let batch_size = x.shape()[0];
                let hidden_size = x.shape()[2];

                // Reshape to (batch_size, hidden_size)
                let last_hidden = x.into_shape_with_order((batch_size, hidden_size))?;

                // Apply attention
                // Convert to IxDyn for compatibility with Layer trait
                let dyn_last_hidden = last_hidden.to_owned().into_dyn();
                let (attentional_hidden, _) = attention.forward(&dyn_last_hidden, encoder_out)?;

                // Project to vocabulary size
                self.output_projection.forward(&attentional_hidden)?
            } else {
                return Err(NeuralError::InvalidArchitecture(
                    "Attention requires encoder outputs".to_string(),
                ));
            }
        } else {
            // Without attention, just project the last hidden state
            let batch_size = x.shape()[0];
            let hidden_size = x.shape()[2];

            // Reshape to (batch_size, hidden_size)
            let last_hidden = x.into_shape_with_order((batch_size, hidden_size))?;

            // Project to vocabulary size - convert to IxDyn first
            let dyn_last_hidden = last_hidden.to_owned().into_dyn();
            self.output_projection.forward(&dyn_last_hidden)?
        };

        Ok((final_output, states_out))
    }

    /// Forward pass for decoding a complete sequence
    pub fn forward_sequence(
        &self,
        input_tokens: &Array<F, IxDyn>,
        initial_states: &[Array<F, IxDyn>],
        encoder_outputs: Option<&Array<F, IxDyn>>,
    ) -> Result<Array<F, IxDyn>> {
        let batch_size = input_tokens.shape()[0];
        let seq_len = input_tokens.shape()[1];

        // Prepare output buffer
        let mut outputs = Array::<F, _>::zeros((batch_size, seq_len, self.vocab_size));
        let mut states = initial_states.to_vec();

        // Process each time step
        for t in 0..seq_len {
            // Extract tokens for this time step
            let tokens_t = input_tokens
                .slice_axis(Axis(1), ndarray::Slice::from(t..t + 1))
                .to_owned();

            // Decode one step
            let (output_t, new_states) = self.forward_step(&tokens_t, &states, encoder_outputs)?;

            // Store output
            for b in 0..batch_size {
                for v in 0..self.vocab_size {
                    outputs[[b, t, v]] = output_t[[b, v]];
                }
            }

            // Update states
            states = new_states;
        }

        Ok(outputs.into_dyn())
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for Seq2SeqDecoder<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // This simplified version assumes:
        // 1. Input contains decoder inputs only
        // 2. Initial states are zero-initialized
        // 3. No encoder outputs/attention

        let batch_size = input.shape()[0];
        let _seq_len = input.shape()[1]; // Not directly used but kept for clarity

        // Initialize empty states
        let mut initial_states = Vec::new();
        for _ in 0..self.rnn_layers.len() {
            let state = Array::<F, _>::zeros((batch_size, self.hidden_dim)).into_dyn();
            initial_states.push(state);
        }

        // Forward sequence without encoder outputs
        self.forward_sequence(input, &initial_states, None)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Err(NeuralError::NotImplemented(
            "Not implemented yet".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Err(NeuralError::NotImplemented(
            "Not implemented yet".to_string(),
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        params.extend(self.embedding.params());

        for layer in &self.rnn_layers {
            params.extend(layer.params());
        }

        if let Some(ref attention) = self.attention {
            params.extend(attention.params());
        }

        params.extend(self.output_projection.params());

        if let Some(ref dropout) = self.dropout {
            params.extend(dropout.params());
        }

        params
    }

    fn set_training(&mut self, training: bool) {
        self.embedding.set_training(training);

        for layer in &mut self.rnn_layers {
            layer.set_training(training);
        }

        if let Some(ref mut attention) = self.attention {
            attention.set_training(training);
        }

        self.output_projection.set_training(training);

        if let Some(ref mut dropout) = self.dropout {
            dropout.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.embedding.is_training()
    }
}

/// Sequence-to-Sequence (Seq2Seq) model
// TODO: Implement Debug and Clone manually once the contained types support it
pub struct Seq2Seq<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// Encoder component
    pub encoder: Seq2SeqEncoder<F>,
    /// Decoder component
    pub decoder: Seq2SeqDecoder<F>,
    /// Model configuration
    pub config: Seq2SeqConfig,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Seq2Seq<F> {
    /// Create a new Seq2Seq model
    pub fn new(config: Seq2SeqConfig) -> Result<Self> {
        // Create encoder
        let encoder = Seq2SeqEncoder::<F>::new(
            config.input_vocab_size,
            config.embedding_dim,
            config.hidden_dim,
            config.num_layers,
            config.encoder_cell_type,
            config.bidirectional_encoder,
            Some(config.dropout_rate),
        )?;

        // Create decoder
        let decoder = Seq2SeqDecoder::<F>::new(
            config.output_vocab_size,
            config.embedding_dim,
            config.hidden_dim,
            config.num_layers,
            config.decoder_cell_type,
            config.use_attention,
            config.bidirectional_encoder,
            Some(config.dropout_rate),
        )?;

        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    /// Forward pass for training (teacher forcing)
    pub fn forward_train(
        &self,
        input_seq: &Array<F, IxDyn>,
        target_seq: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Encode input sequence
        let (encoder_outputs, encoder_states) = self.encoder.forward(input_seq)?;

        // Prepare decoder initial states (use last encoder states)
        let decoder_initial_states = if self.config.encoder_cell_type
            == self.config.decoder_cell_type
        {
            // If cell types match, we can directly use encoder states
            encoder_states
        } else {
            // If cell types don't match, we need to project encoder states
            // For simplicity, we'll just zero-initialize decoder states
            let batch_size = input_seq.shape()[0];
            let mut initial_states = Vec::new();

            for _ in 0..self.config.num_layers {
                let state = Array::<F, _>::zeros((batch_size, self.config.hidden_dim)).into_dyn();
                initial_states.push(state);
            }

            initial_states
        };

        // Decode target sequence with teacher forcing
        let decoder_output = self.decoder.forward_sequence(
            target_seq,
            &decoder_initial_states,
            Some(&encoder_outputs),
        )?;

        Ok(decoder_output)
    }

    /// Generate sequences for inference
    pub fn generate(
        &self,
        input_seq: &Array<F, IxDyn>,
        max_length: Option<usize>,
        start_token_id: usize,
        end_token_id: Option<usize>,
    ) -> Result<Array<F, IxDyn>> {
        let batch_size = input_seq.shape()[0];
        let max_len = max_length.unwrap_or(self.config.max_seq_len);

        // Encode input sequence
        let (encoder_outputs, encoder_states) = self.encoder.forward(input_seq)?;

        // Prepare decoder initial states
        let decoder_states = if self.config.encoder_cell_type == self.config.decoder_cell_type {
            encoder_states
        } else {
            let mut initial_states = Vec::new();
            for _ in 0..self.config.num_layers {
                let state = Array::<F, _>::zeros((batch_size, self.config.hidden_dim)).into_dyn();
                initial_states.push(state);
            }
            initial_states
        };

        // Initialize first decoder input with start tokens
        let mut decoder_input = Array::<F, _>::zeros((batch_size, 1));
        for b in 0..batch_size {
            decoder_input[[b, 0]] = F::from(start_token_id as f64).unwrap();
        }
        let mut decoder_input = decoder_input.into_dyn();

        // Prepare output buffer
        let mut output_ids = Array::<F, _>::zeros((batch_size, max_len));
        let mut states = decoder_states;

        // Keep track of completed sequences
        let mut completed = vec![false; batch_size];

        // Generate sequence
        for t in 0..max_len {
            // Decode one step
            let (output_t, new_states) =
                self.decoder
                    .forward_step(&decoder_input, &states, Some(&encoder_outputs))?;

            // Get most probable token
            let mut next_tokens = Array::<F, _>::zeros((batch_size, 1));
            for b in 0..batch_size {
                if completed[b] {
                    continue;
                }

                // Find max probability token
                let mut max_prob = F::neg_infinity();
                let mut max_idx = 0;

                for v in 0..self.config.output_vocab_size {
                    if output_t[[b, v]] > max_prob {
                        max_prob = output_t[[b, v]];
                        max_idx = v;
                    }
                }

                // Store predicted token
                output_ids[[b, t]] = F::from(max_idx as f64).unwrap();
                next_tokens[[b, 0]] = F::from(max_idx as f64).unwrap();

                // Check if sequence is completed
                if let Some(eos_id) = end_token_id {
                    if max_idx == eos_id {
                        completed[b] = true;
                    }
                }
            }

            // Early stopping if all sequences are completed
            if completed.iter().all(|&c| c) {
                break;
            }

            // Update decoder input for next step
            decoder_input = next_tokens.into_dyn();

            // Update states
            states = new_states;
        }

        Ok(output_ids.into_dyn())
    }

    /// Create a basic Seq2Seq model for machine translation
    pub fn create_translation_model(
        src_vocab_size: usize,
        tgt_vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<Self> {
        let config = Seq2SeqConfig {
            input_vocab_size: src_vocab_size,
            output_vocab_size: tgt_vocab_size,
            embedding_dim: hidden_dim,
            hidden_dim,
            num_layers: 2,
            encoder_cell_type: RNNCellType::LSTM,
            decoder_cell_type: RNNCellType::LSTM,
            bidirectional_encoder: true,
            use_attention: true,
            dropout_rate: 0.1,
            max_seq_len: 100,
        };

        Self::new(config)
    }

    /// Create a small and fast Seq2Seq model
    pub fn create_small_model(src_vocab_size: usize, tgt_vocab_size: usize) -> Result<Self> {
        let config = Seq2SeqConfig {
            input_vocab_size: src_vocab_size,
            output_vocab_size: tgt_vocab_size,
            embedding_dim: 128,
            hidden_dim: 256,
            num_layers: 1,
            encoder_cell_type: RNNCellType::GRU,
            decoder_cell_type: RNNCellType::GRU,
            bidirectional_encoder: false,
            use_attention: true,
            dropout_rate: 0.1,
            max_seq_len: 50,
        };

        Self::new(config)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for Seq2Seq<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // This simplified forward assumes the input is only the encoder inputs
        // and automatically generates decoder outputs without teacher forcing

        self.generate(
            input,
            Some(self.config.max_seq_len),
            0, // Assuming 0 is the start token
            None,
        )
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Err(NeuralError::NotImplemented(
            "Not implemented yet".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Err(NeuralError::NotImplemented(
            "Not implemented yet".to_string(),
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        params.extend(self.encoder.params());
        params.extend(self.decoder.params());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.encoder.set_training(training);
        self.decoder.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.encoder.is_training()
    }
}
