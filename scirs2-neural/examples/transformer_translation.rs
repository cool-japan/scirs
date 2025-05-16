use ndarray::{s, Array1, Array2, Array3, Array4};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32;

// Embedding layer (similar to previous examples)
#[derive(Debug, Serialize, Deserialize)]
struct Embedding {
    vocab_size: usize,
    embedding_dim: usize,
    weight: Array2<f32>,
}

impl Embedding {
    fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        // Xavier/Glorot initialization
        let bound = (3.0 / embedding_dim as f32).sqrt();

        // Create a random number generator
        let mut rng = rand::rng();

        // Initialize with random values
        let mut weight = Array2::<f32>::zeros((vocab_size, embedding_dim));
        for elem in weight.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }

        Embedding {
            vocab_size,
            embedding_dim,
            weight,
        }
    }

    fn forward(&self, x: &Array2<usize>) -> Array3<f32> {
        // x: [batch_size, seq_len] - Input token IDs
        // Returns: [batch_size, seq_len, embedding_dim] - Embedded vectors

        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut output = Array3::<f32>::zeros((batch_size, seq_len, self.embedding_dim));

        for b in 0..batch_size {
            for t in 0..seq_len {
                let token_id = x[[b, t]];
                if token_id < self.vocab_size {
                    for e in 0..self.embedding_dim {
                        output[[b, t, e]] = self.weight[[token_id, e]];
                    }
                }
            }
        }

        output
    }
}

// Positional Encoding
#[derive(Debug, Serialize, Deserialize)]
struct PositionalEncoding {
    max_seq_len: usize,
    d_model: usize,
    encoding: Array2<f32>,
}

impl PositionalEncoding {
    fn new(max_seq_len: usize, d_model: usize) -> Self {
        // Create positional encoding matrix
        let mut encoding = Array2::<f32>::zeros((max_seq_len, d_model));

        for pos in 0..max_seq_len {
            for i in 0..d_model {
                let div_term = 10000.0_f32.powf(2.0 * (i / 2) as f32 / d_model as f32);

                if i % 2 == 0 {
                    // Sine for even indices
                    encoding[[pos, i]] = (pos as f32 / div_term).sin();
                } else {
                    // Cosine for odd indices
                    encoding[[pos, i]] = (pos as f32 / div_term).cos();
                }
            }
        }

        PositionalEncoding {
            max_seq_len,
            d_model,
            encoding,
        }
    }

    fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        // x: [batch_size, seq_len, embedding_dim]
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        // Add positional encoding
        let mut output = x.clone();

        for b in 0..batch_size {
            for t in 0..seq_len.min(self.max_seq_len) {
                for d in 0..self.d_model {
                    output[[b, t, d]] += self.encoding[[t, d]];
                }
            }
        }

        output
    }
}

// Layer Normalization
#[derive(Debug, Serialize, Deserialize)]
struct LayerNorm {
    normalized_shape: usize,
    epsilon: f32,
    gamma: Array1<f32>,
    beta: Array1<f32>,
}

impl LayerNorm {
    fn new(normalized_shape: usize, epsilon: f32) -> Self {
        // Initialize parameters
        let gamma = Array1::<f32>::ones(normalized_shape);
        let beta = Array1::<f32>::zeros(normalized_shape);

        LayerNorm {
            normalized_shape,
            epsilon,
            gamma,
            beta,
        }
    }

    fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        // x: [batch_size, seq_len, d_model]
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut output = Array3::<f32>::zeros(x.raw_dim());

        // Apply normalization per sequence item
        for b in 0..batch_size {
            for t in 0..seq_len {
                // Get sequence item
                let x_i = x.slice(s![b, t, ..]).to_owned();

                // Calculate mean
                let mean = x_i.mean().unwrap_or(0.0);

                // Calculate variance
                let mut variance = 0.0;
                for &val in x_i.iter() {
                    variance += (val - mean).powi(2);
                }
                variance /= self.normalized_shape as f32;

                // Normalize
                for d in 0..self.normalized_shape {
                    output[[b, t, d]] = (x_i[d] - mean) / (variance + self.epsilon).sqrt();

                    // Scale and shift
                    output[[b, t, d]] = self.gamma[d] * output[[b, t, d]] + self.beta[d];
                }
            }
        }

        output
    }
}

// Multi-Head Attention
#[derive(Debug, Serialize, Deserialize)]
struct MultiHeadAttention {
    d_model: usize,
    num_heads: usize,
    d_k: usize, // d_model / num_heads

    // Projection matrices
    w_q: Array2<f32>, // [d_model, d_model]
    w_k: Array2<f32>, // [d_model, d_model]
    w_v: Array2<f32>, // [d_model, d_model]
    w_o: Array2<f32>, // [d_model, d_model]
}

impl MultiHeadAttention {
    fn new(d_model: usize, num_heads: usize) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model must be divisible by num_heads"
        );

        let d_k = d_model / num_heads;

        // Xavier initialization
        let bound = (6.0 / (d_model + d_model) as f32).sqrt();

        // Create a random number generator
        let mut rng = rand::rng();

        // Initialize with random values
        let mut w_q = Array2::<f32>::zeros((d_model, d_model));
        let mut w_k = Array2::<f32>::zeros((d_model, d_model));
        let mut w_v = Array2::<f32>::zeros((d_model, d_model));
        let mut w_o = Array2::<f32>::zeros((d_model, d_model));

        for elem in w_q.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }

        for elem in w_k.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }

        for elem in w_v.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }

        for elem in w_o.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }

        MultiHeadAttention {
            d_model,
            num_heads,
            d_k,
            w_q,
            w_k,
            w_v,
            w_o,
        }
    }

    fn forward(
        &self,
        q: &Array3<f32>,
        k: &Array3<f32>,
        v: &Array3<f32>,
        mask: Option<&Array3<f32>>,
    ) -> Array3<f32> {
        // q, k, v: [batch_size, seq_len, d_model]
        let batch_size = q.shape()[0];
        let q_len = q.shape()[1];
        let k_len = k.shape()[1];

        // Linear projections
        let q_proj = self.project(q, &self.w_q); // [batch_size, q_len, d_model]
        let k_proj = self.project(k, &self.w_k); // [batch_size, k_len, d_model]
        let v_proj = self.project(v, &self.w_v); // [batch_size, k_len, d_model]

        // Reshape for multi-head attention
        // [batch_size, num_heads, seq_len, d_k]
        let q_heads = self.reshape_for_multihead(&q_proj);
        let k_heads = self.reshape_for_multihead(&k_proj);
        let v_heads = self.reshape_for_multihead(&v_proj);

        // Calculate attention scores
        // [batch_size, num_heads, q_len, k_len]
        let mut attention_scores = Array4::<f32>::zeros((batch_size, self.num_heads, q_len, k_len));

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..q_len {
                    let q_vec = q_heads.slice(s![b, h, i, ..]).to_owned();

                    for j in 0..k_len {
                        let k_vec = k_heads.slice(s![b, h, j, ..]).to_owned();

                        // Calculate dot product
                        let mut dot_product = 0.0;
                        for d in 0..self.d_k {
                            dot_product += q_vec[d] * k_vec[d];
                        }

                        // Scale
                        attention_scores[[b, h, i, j]] = dot_product / (self.d_k as f32).sqrt();
                    }
                }
            }
        }

        // Apply mask if provided
        if let Some(m) = mask {
            for b in 0..batch_size {
                for h in 0..self.num_heads {
                    for i in 0..q_len {
                        for j in 0..k_len {
                            if m[[b, i, j]] == 0.0 {
                                attention_scores[[b, h, i, j]] = f32::NEG_INFINITY;
                            }
                        }
                    }
                }
            }
        }

        // Apply softmax
        let attention_weights = self.softmax_attention(&attention_scores);

        // Apply attention weights to values
        // [batch_size, num_heads, q_len, d_k]
        let mut context = Array4::<f32>::zeros((batch_size, self.num_heads, q_len, self.d_k));

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..q_len {
                    for j in 0..k_len {
                        let weight = attention_weights[[b, h, i, j]];

                        for d in 0..self.d_k {
                            context[[b, h, i, d]] += weight * v_heads[[b, h, j, d]];
                        }
                    }
                }
            }
        }

        // Reshape back
        // [batch_size, q_len, d_model]
        let context_combined = self.reshape_from_multihead(&context);

        // Final projection
        self.project(&context_combined, &self.w_o)
    }

    fn project(&self, x: &Array3<f32>, weight: &Array2<f32>) -> Array3<f32> {
        // x: [batch_size, seq_len, d_model]
        // weight: [d_model, d_model]
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut output = Array3::<f32>::zeros((batch_size, seq_len, self.d_model));

        for b in 0..batch_size {
            for t in 0..seq_len {
                let x_vec = x.slice(s![b, t, ..]).to_owned();
                let result = x_vec.dot(weight);

                for d in 0..self.d_model {
                    output[[b, t, d]] = result[d];
                }
            }
        }

        output
    }

    fn reshape_for_multihead(&self, x: &Array3<f32>) -> Array4<f32> {
        // x: [batch_size, seq_len, d_model]
        // Returns: [batch_size, num_heads, seq_len, d_k]
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut output = Array4::<f32>::zeros((batch_size, self.num_heads, seq_len, self.d_k));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for h in 0..self.num_heads {
                    for d in 0..self.d_k {
                        output[[b, h, t, d]] = x[[b, t, h * self.d_k + d]];
                    }
                }
            }
        }

        output
    }

    fn reshape_from_multihead(&self, x: &Array4<f32>) -> Array3<f32> {
        // x: [batch_size, num_heads, seq_len, d_k]
        // Returns: [batch_size, seq_len, d_model]
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[2];

        let mut output = Array3::<f32>::zeros((batch_size, seq_len, self.d_model));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for h in 0..self.num_heads {
                    for d in 0..self.d_k {
                        output[[b, t, h * self.d_k + d]] = x[[b, h, t, d]];
                    }
                }
            }
        }

        output
    }

    fn softmax_attention(&self, x: &Array4<f32>) -> Array4<f32> {
        // x: [batch_size, num_heads, q_len, k_len]
        let batch_size = x.shape()[0];
        let num_heads = x.shape()[1];
        let q_len = x.shape()[2];
        let k_len = x.shape()[3];

        let mut output = Array4::<f32>::zeros(x.raw_dim());

        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..q_len {
                    // Get row
                    let row = x.slice(s![b, h, i, ..]).to_owned();

                    // Find max value for numerical stability
                    let max_val = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                    // Check if all values are -inf (can happen with masking)
                    if max_val == f32::NEG_INFINITY {
                        // Handle the edge case (uniform distribution)
                        let uniform_val = 1.0 / k_len as f32;
                        for j in 0..k_len {
                            output[[b, h, i, j]] = uniform_val;
                        }
                        continue;
                    }

                    // Calculate exp and sum
                    let mut sum = 0.0;
                    let mut exp_vals = vec![0.0; k_len];

                    for j in 0..k_len {
                        let exp_val = (row[j] - max_val).exp();
                        exp_vals[j] = exp_val;
                        sum += exp_val;
                    }

                    // Check for numerical issues (sum too small or NaN)
                    if sum < 1e-10 || sum.is_nan() {
                        // Fall back to uniform distribution
                        let uniform_val = 1.0 / k_len as f32;
                        for j in 0..k_len {
                            output[[b, h, i, j]] = uniform_val;
                        }
                    } else {
                        // Normalize
                        for j in 0..k_len {
                            output[[b, h, i, j]] = exp_vals[j] / sum;
                        }
                    }
                }
            }
        }

        output
    }
}

// Position-wise Feed-Forward Network
#[derive(Debug, Serialize, Deserialize)]
struct FeedForward {
    d_model: usize,
    d_ff: usize,

    // Parameters
    w1: Array2<f32>, // [d_model, d_ff]
    b1: Array1<f32>, // [d_ff]
    w2: Array2<f32>, // [d_ff, d_model]
    b2: Array1<f32>, // [d_model]
}

impl FeedForward {
    fn new(d_model: usize, d_ff: usize) -> Self {
        // Xavier initialization
        let bound1 = (6.0 / (d_model + d_ff) as f32).sqrt();
        let bound2 = (6.0 / (d_ff + d_model) as f32).sqrt();

        // Create a random number generator
        let mut rng = rand::rng();

        // Initialize with random values
        let mut w1 = Array2::<f32>::zeros((d_model, d_ff));
        let b1 = Array1::<f32>::zeros(d_ff);
        let mut w2 = Array2::<f32>::zeros((d_ff, d_model));
        let b2 = Array1::<f32>::zeros(d_model);

        for elem in w1.iter_mut() {
            *elem = rng.random_range(-bound1..bound1);
        }

        for elem in w2.iter_mut() {
            *elem = rng.random_range(-bound2..bound2);
        }

        FeedForward {
            d_model,
            d_ff,
            w1,
            b1,
            w2,
            b2,
        }
    }

    fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        // x: [batch_size, seq_len, d_model]
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        let mut hidden = Array3::<f32>::zeros((batch_size, seq_len, self.d_ff));
        let mut output = Array3::<f32>::zeros((batch_size, seq_len, self.d_model));

        // First layer
        for b in 0..batch_size {
            for t in 0..seq_len {
                let x_vec = x.slice(s![b, t, ..]).to_owned();

                // Linear transformation
                let mut h = x_vec.dot(&self.w1);

                // Add bias
                for d in 0..self.d_ff {
                    h[d] += self.b1[d];
                }

                // ReLU activation
                for d in 0..self.d_ff {
                    h[d] = h[d].max(0.0);
                    hidden[[b, t, d]] = h[d];
                }
            }
        }

        // Second layer
        for b in 0..batch_size {
            for t in 0..seq_len {
                let h_vec = hidden.slice(s![b, t, ..]).to_owned();

                // Linear transformation
                let mut o = h_vec.dot(&self.w2);

                // Add bias
                for d in 0..self.d_model {
                    o[d] += self.b2[d];
                    output[[b, t, d]] = o[d];
                }
            }
        }

        output
    }
}

// Encoder Layer
#[derive(Debug, Serialize, Deserialize)]
struct EncoderLayer {
    d_model: usize,

    // Multi-head attention
    self_attn: MultiHeadAttention,

    // Feed-forward network
    feed_forward: FeedForward,

    // Layer normalization
    norm1: LayerNorm,
    norm2: LayerNorm,

    // Dropout rate
    dropout_rate: f32,
}

impl EncoderLayer {
    fn new(d_model: usize, num_heads: usize, d_ff: usize, dropout_rate: f32) -> Self {
        let self_attn = MultiHeadAttention::new(d_model, num_heads);
        let feed_forward = FeedForward::new(d_model, d_ff);
        let norm1 = LayerNorm::new(d_model, 1e-6);
        let norm2 = LayerNorm::new(d_model, 1e-6);

        EncoderLayer {
            d_model,
            self_attn,
            feed_forward,
            norm1,
            norm2,
            dropout_rate,
        }
    }

    fn forward(&self, x: &Array3<f32>, mask: Option<&Array3<f32>>) -> Array3<f32> {
        // x: [batch_size, seq_len, d_model]

        // Self-attention with residual connection and layer norm
        let attn_output = self.self_attn.forward(x, x, x, mask);

        // Apply dropout (in a real implementation)
        // For simplicity, we're skipping actual dropout here

        // Add residual connection
        let mut residual1 = Array3::<f32>::zeros(x.raw_dim());
        for b in 0..x.shape()[0] {
            for t in 0..x.shape()[1] {
                for d in 0..self.d_model {
                    residual1[[b, t, d]] = x[[b, t, d]] + attn_output[[b, t, d]];
                }
            }
        }

        // Apply layer normalization
        let norm1_output = self.norm1.forward(&residual1);

        // Feed-forward network
        let ff_output = self.feed_forward.forward(&norm1_output);

        // Apply dropout (in a real implementation)

        // Add residual connection
        let mut residual2 = Array3::<f32>::zeros(x.raw_dim());
        for b in 0..x.shape()[0] {
            for t in 0..x.shape()[1] {
                for d in 0..self.d_model {
                    residual2[[b, t, d]] = norm1_output[[b, t, d]] + ff_output[[b, t, d]];
                }
            }
        }

        // Apply layer normalization
        self.norm2.forward(&residual2)
    }
}

// Decoder Layer
#[derive(Debug, Serialize, Deserialize)]
struct DecoderLayer {
    d_model: usize,

    // Multi-head attentions
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,

    // Feed-forward network
    feed_forward: FeedForward,

    // Layer normalization
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,

    // Dropout rate
    dropout_rate: f32,
}

impl DecoderLayer {
    fn new(d_model: usize, num_heads: usize, d_ff: usize, dropout_rate: f32) -> Self {
        let self_attn = MultiHeadAttention::new(d_model, num_heads);
        let cross_attn = MultiHeadAttention::new(d_model, num_heads);
        let feed_forward = FeedForward::new(d_model, d_ff);
        let norm1 = LayerNorm::new(d_model, 1e-6);
        let norm2 = LayerNorm::new(d_model, 1e-6);
        let norm3 = LayerNorm::new(d_model, 1e-6);

        DecoderLayer {
            d_model,
            self_attn,
            cross_attn,
            feed_forward,
            norm1,
            norm2,
            norm3,
            dropout_rate,
        }
    }

    fn forward(
        &self,
        x: &Array3<f32>,
        enc_output: &Array3<f32>,
        self_mask: Option<&Array3<f32>>,
        cross_mask: Option<&Array3<f32>>,
    ) -> Array3<f32> {
        // x: [batch_size, tgt_len, d_model]
        // enc_output: [batch_size, src_len, d_model]

        // Self-attention with residual connection and layer norm
        let self_attn_output = self.self_attn.forward(x, x, x, self_mask);

        // Apply dropout (in a real implementation)

        // Add residual connection
        let mut residual1 = Array3::<f32>::zeros(x.raw_dim());
        for b in 0..x.shape()[0] {
            for t in 0..x.shape()[1] {
                for d in 0..self.d_model {
                    residual1[[b, t, d]] = x[[b, t, d]] + self_attn_output[[b, t, d]];
                }
            }
        }

        // Apply layer normalization
        let norm1_output = self.norm1.forward(&residual1);

        // Cross-attention with encoder outputs
        let cross_attn_output =
            self.cross_attn
                .forward(&norm1_output, enc_output, enc_output, cross_mask);

        // Apply dropout (in a real implementation)

        // Add residual connection
        let mut residual2 = Array3::<f32>::zeros(x.raw_dim());
        for b in 0..x.shape()[0] {
            for t in 0..x.shape()[1] {
                for d in 0..self.d_model {
                    residual2[[b, t, d]] = norm1_output[[b, t, d]] + cross_attn_output[[b, t, d]];
                }
            }
        }

        // Apply layer normalization
        let norm2_output = self.norm2.forward(&residual2);

        // Feed-forward network
        let ff_output = self.feed_forward.forward(&norm2_output);

        // Apply dropout (in a real implementation)

        // Add residual connection
        let mut residual3 = Array3::<f32>::zeros(x.raw_dim());
        for b in 0..x.shape()[0] {
            for t in 0..x.shape()[1] {
                for d in 0..self.d_model {
                    residual3[[b, t, d]] = norm2_output[[b, t, d]] + ff_output[[b, t, d]];
                }
            }
        }

        // Apply layer normalization
        self.norm3.forward(&residual3)
    }
}

// Transformer Encoder
#[derive(Debug, Serialize, Deserialize)]
struct Encoder {
    d_model: usize,
    layers: Vec<EncoderLayer>,
    norm: LayerNorm,
}

impl Encoder {
    fn new(
        d_model: usize,
        num_layers: usize,
        num_heads: usize,
        d_ff: usize,
        dropout_rate: f32,
    ) -> Self {
        let mut layers = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            layers.push(EncoderLayer::new(d_model, num_heads, d_ff, dropout_rate));
        }

        let norm = LayerNorm::new(d_model, 1e-6);

        Encoder {
            d_model,
            layers,
            norm,
        }
    }

    fn forward(&self, x: &Array3<f32>, mask: Option<&Array3<f32>>) -> Array3<f32> {
        // x: [batch_size, src_len, d_model]
        let mut output = x.clone();

        // Process through each encoder layer
        for layer in &self.layers {
            output = layer.forward(&output, mask);
        }

        // Apply final layer normalization
        self.norm.forward(&output)
    }
}

// Transformer Decoder
#[derive(Debug, Serialize, Deserialize)]
struct Decoder {
    d_model: usize,
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
}

impl Decoder {
    fn new(
        d_model: usize,
        num_layers: usize,
        num_heads: usize,
        d_ff: usize,
        dropout_rate: f32,
    ) -> Self {
        let mut layers = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            layers.push(DecoderLayer::new(d_model, num_heads, d_ff, dropout_rate));
        }

        let norm = LayerNorm::new(d_model, 1e-6);

        Decoder {
            d_model,
            layers,
            norm,
        }
    }

    fn forward(
        &self,
        x: &Array3<f32>,
        enc_output: &Array3<f32>,
        self_mask: Option<&Array3<f32>>,
        cross_mask: Option<&Array3<f32>>,
    ) -> Array3<f32> {
        // x: [batch_size, tgt_len, d_model]
        // enc_output: [batch_size, src_len, d_model]
        let mut output = x.clone();

        // Process through each decoder layer
        for layer in &self.layers {
            output = layer.forward(&output, enc_output, self_mask, cross_mask);
        }

        // Apply final layer normalization
        self.norm.forward(&output)
    }
}

// Complete Transformer model
#[derive(Debug, Serialize, Deserialize)]
struct Transformer {
    // Embedding layers
    src_embedding: Embedding,
    tgt_embedding: Embedding,

    // Positional encoding
    positional_encoding: PositionalEncoding,

    // Encoder and decoder
    encoder: Encoder,
    decoder: Decoder,

    // Output projection
    output_projection: Array2<f32>,

    // Model dimensions
    d_model: usize,
    src_vocab_size: usize,
    tgt_vocab_size: usize,
}

impl Transformer {
    fn new(
        src_vocab_size: usize,
        tgt_vocab_size: usize,
        d_model: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        num_heads: usize,
        d_ff: usize,
        max_seq_len: usize,
        dropout_rate: f32,
    ) -> Self {
        // Create embeddings
        let src_embedding = Embedding::new(src_vocab_size, d_model);
        let tgt_embedding = Embedding::new(tgt_vocab_size, d_model);

        // Create positional encoding
        let positional_encoding = PositionalEncoding::new(max_seq_len, d_model);

        // Create encoder and decoder
        let encoder = Encoder::new(d_model, num_encoder_layers, num_heads, d_ff, dropout_rate);
        let decoder = Decoder::new(d_model, num_decoder_layers, num_heads, d_ff, dropout_rate);

        // Create output projection
        let output_bound = (3.0 / (d_model + tgt_vocab_size) as f32).sqrt();

        // Create a random number generator
        let mut rng = rand::rng();

        // Initialize with random values
        let mut output_projection = Array2::<f32>::zeros((tgt_vocab_size, d_model));
        for elem in output_projection.iter_mut() {
            *elem = rng.random_range(-output_bound..output_bound);
        }

        Transformer {
            src_embedding,
            tgt_embedding,
            positional_encoding,
            encoder,
            decoder,
            output_projection,
            d_model,
            src_vocab_size,
            tgt_vocab_size,
        }
    }

    fn create_padding_mask(&self, seq: &Array2<usize>, pad_idx: usize) -> Array3<f32> {
        // seq: [batch_size, seq_len]
        // Creates mask where pad_idx positions are masked (0.0) and other positions are valid (1.0)
        let batch_size = seq.shape()[0];
        let seq_len = seq.shape()[1];

        let mut mask = Array3::<f32>::ones((batch_size, seq_len, seq_len));

        for b in 0..batch_size {
            for t in 0..seq_len {
                if seq[[b, t]] == pad_idx {
                    // Mask this position in both dimensions
                    for i in 0..seq_len {
                        mask[[b, i, t]] = 0.0; // Mask column
                        mask[[b, t, i]] = 0.0; // Mask row
                    }
                }
            }
        }

        mask
    }

    fn create_look_ahead_mask(&self, size: usize) -> Array3<f32> {
        // Creates lower triangular mask of shape [1, size, size]
        let mut mask = Array3::<f32>::zeros((1, size, size));

        for i in 0..size {
            for j in 0..size {
                if j <= i {
                    mask[[0, i, j]] = 1.0;
                }
            }
        }

        mask
    }

    fn combine_masks(&self, mask1: &Array3<f32>, mask2: &Array3<f32>) -> Array3<f32> {
        // Combines two masks elementwise
        // mask1, mask2: [batch_size, seq_len, seq_len] or [1, seq_len, seq_len]
        let batch_size = mask1.shape()[0].max(mask2.shape()[0]);
        let seq_len = mask1.shape()[1];

        let mut combined = Array3::<f32>::zeros((batch_size, seq_len, seq_len));

        for b in 0..batch_size {
            let b1 = b.min(mask1.shape()[0] - 1);
            let b2 = b.min(mask2.shape()[0] - 1);

            for i in 0..seq_len {
                for j in 0..seq_len {
                    combined[[b, i, j]] = mask1[[b1, i, j]] * mask2[[b2, i, j]];
                }
            }
        }

        combined
    }

    fn forward(
        &self,
        src: &Array2<usize>,
        tgt: &Array2<usize>,
        src_pad_idx: usize,
        tgt_pad_idx: usize,
    ) -> Array3<f32> {
        // src: [batch_size, src_len] - Source token IDs
        // tgt: [batch_size, tgt_len] - Target token IDs
        let batch_size = src.shape()[0];
        let _src_len = src.shape()[1];
        let tgt_len = tgt.shape()[1];

        // Create masks
        let src_mask = self.create_padding_mask(src, src_pad_idx);
        let tgt_padding_mask = self.create_padding_mask(tgt, tgt_pad_idx);
        let tgt_look_ahead_mask = self.create_look_ahead_mask(tgt_len);
        let tgt_mask = self.combine_masks(&tgt_padding_mask, &tgt_look_ahead_mask);

        // Encoder
        // Embed source tokens
        let src_embedded = self.src_embedding.forward(src);

        // Add positional encoding
        let src_pos_encoded = self.positional_encoding.forward(&src_embedded);

        // Encode
        let enc_output = self.encoder.forward(&src_pos_encoded, Some(&src_mask));

        // Decoder
        // Embed target tokens
        let tgt_embedded = self.tgt_embedding.forward(tgt);

        // Add positional encoding
        let tgt_pos_encoded = self.positional_encoding.forward(&tgt_embedded);

        // Decode
        let dec_output = self.decoder.forward(
            &tgt_pos_encoded,
            &enc_output,
            Some(&tgt_mask),
            Some(&src_mask),
        );

        // Project to vocabulary
        let mut logits = Array3::<f32>::zeros((batch_size, tgt_len, self.tgt_vocab_size));

        for b in 0..batch_size {
            for t in 0..tgt_len {
                let dec_features = dec_output.slice(s![b, t, ..]).to_owned();
                let result = dec_features.dot(&self.output_projection.t());

                for v in 0..self.tgt_vocab_size {
                    logits[[b, t, v]] = result[v];
                }
            }
        }

        logits
    }

    fn greedy_decode(
        &self,
        src: &Array2<usize>,
        src_pad_idx: usize,
        sos_idx: usize,
        eos_idx: usize,
        max_len: usize,
    ) -> Array2<usize> {
        // src: [batch_size, src_len] - Source token IDs
        let batch_size = src.shape()[0];
        let _src_len = src.shape()[1];

        // Create source mask
        let src_mask = self.create_padding_mask(src, src_pad_idx);

        // Encode source sequence
        let src_embedded = self.src_embedding.forward(src);
        let src_pos_encoded = self.positional_encoding.forward(&src_embedded);
        let enc_output = self.encoder.forward(&src_pos_encoded, Some(&src_mask));

        // Initialize target sequence with SOS token
        let mut decoded_tokens = Array2::<usize>::from_elem((batch_size, max_len), 0);
        for b in 0..batch_size {
            decoded_tokens[[b, 0]] = sos_idx;
        }

        // Decode step by step
        for t in 0..max_len - 1 {
            // Current target sequence
            let tgt = decoded_tokens.slice(s![.., 0..t + 1]).to_owned();

            // Create target mask
            let tgt_len = t + 1;
            let tgt_padding_mask = self.create_padding_mask(&tgt, 0); // Assuming 0 is used for padding in the target
            let tgt_look_ahead_mask = self.create_look_ahead_mask(tgt_len);
            let tgt_mask = self.combine_masks(&tgt_padding_mask, &tgt_look_ahead_mask);

            // Embed target sequence
            let tgt_embedded = self.tgt_embedding.forward(&tgt);
            let tgt_pos_encoded = self.positional_encoding.forward(&tgt_embedded);

            // Decode
            let dec_output = self.decoder.forward(
                &tgt_pos_encoded,
                &enc_output,
                Some(&tgt_mask),
                Some(&src_mask),
            );

            // Project to vocabulary (for the last position only)
            let mut logits = Array2::<f32>::zeros((batch_size, self.tgt_vocab_size));

            for b in 0..batch_size {
                let dec_features = dec_output.slice(s![b, t, ..]).to_owned();
                let result = dec_features.dot(&self.output_projection.t());

                for v in 0..self.tgt_vocab_size {
                    logits[[b, v]] = result[v];
                }
            }

            // Get next token (argmax)
            for b in 0..batch_size {
                // Skip if the sequence has already ended with EOS
                let mut has_eos = false;
                for i in 0..t + 1 {
                    if decoded_tokens[[b, i]] == eos_idx {
                        has_eos = true;
                        break;
                    }
                }

                if !has_eos {
                    // Find max probability token
                    let mut max_idx = 0;
                    let mut max_val = logits[[b, 0]];

                    for v in 1..self.tgt_vocab_size {
                        if logits[[b, v]] > max_val {
                            max_idx = v;
                            max_val = logits[[b, v]];
                        }
                    }

                    // Add token to sequence
                    decoded_tokens[[b, t + 1]] = max_idx;
                }
            }

            // Check if all sequences have ended with EOS
            let mut all_done = true;
            for b in 0..batch_size {
                let mut has_eos = false;
                for i in 0..t + 2 {
                    if decoded_tokens[[b, i]] == eos_idx {
                        has_eos = true;
                        break;
                    }
                }

                if !has_eos {
                    all_done = false;
                    break;
                }
            }

            if all_done {
                break;
            }
        }

        decoded_tokens
    }
}

// Helper functions for creating translation dataset and processing
fn create_toy_translation_dataset() -> (Vec<Vec<String>>, Vec<Vec<String>>) {
    // Simple English-French pairs (similar to previous examples but expanded)
    let pairs = vec![
        ("Hello world", "Bonjour le monde"),
        ("How are you", "Comment allez-vous"),
        ("I love programming", "J'aime la programmation"),
        ("This is a test", "C'est un test"),
        ("Good morning", "Bonjour"),
        ("What is your name", "Comment t'appelles-tu"),
        ("The weather is nice today", "Le temps est beau aujourd'hui"),
        ("I am learning Rust", "J'apprends Rust"),
        ("Thank you very much", "Merci beaucoup"),
        ("See you tomorrow", "À demain"),
        ("I like to read books", "J'aime lire des livres"),
        ("Where is the library", "Où est la bibliothèque"),
        ("The restaurant is closed", "Le restaurant est fermé"),
        (
            "Can you help me please",
            "Pouvez-vous m'aider s'il vous plaît",
        ),
        ("I don't understand", "Je ne comprends pas"),
        ("What time is it", "Quelle heure est-il"),
        ("The car is red", "La voiture est rouge"),
        ("I have a cat", "J'ai un chat"),
        ("She speaks French", "Elle parle français"),
        ("We are going to the beach", "Nous allons à la plage"),
    ];

    let mut english = Vec::new();
    let mut french = Vec::new();

    for (en, fr) in pairs {
        english.push(en.split_whitespace().map(|s| s.to_string()).collect());
        french.push(fr.split_whitespace().map(|s| s.to_string()).collect());
    }

    (english, french)
}

// Create vocabularies for source and target languages
fn create_vocabulary(
    sentences: &[Vec<String>],
) -> (HashMap<String, usize>, HashMap<usize, String>) {
    let mut word_to_idx = HashMap::new();
    let mut idx_to_word = HashMap::new();

    // Special tokens
    let special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"];

    // Add special tokens
    for (i, token) in special_tokens.iter().enumerate() {
        word_to_idx.insert(token.to_string(), i);
        idx_to_word.insert(i, token.to_string());
    }

    // Add words from sentences
    let mut idx = special_tokens.len();
    for sentence in sentences {
        for word in sentence {
            if !word_to_idx.contains_key(word) {
                word_to_idx.insert(word.clone(), idx);
                idx_to_word.insert(idx, word.clone());
                idx += 1;
            }
        }
    }

    (word_to_idx, idx_to_word)
}

// Convert sentences to token IDs with padding
fn tokenize_sentences(
    sentences: &[Vec<String>],
    word_to_idx: &HashMap<String, usize>,
    max_len: usize,
    add_sos_eos: bool,
) -> Array2<usize> {
    let pad_idx = *word_to_idx.get("<pad>").unwrap_or(&0);
    let sos_idx = *word_to_idx.get("<sos>").unwrap_or(&1);
    let eos_idx = *word_to_idx.get("<eos>").unwrap_or(&2);
    let unk_idx = *word_to_idx.get("<unk>").unwrap_or(&3);

    let seq_len = if add_sos_eos {
        // Need space for SOS, tokens, EOS
        max_len + 2
    } else {
        max_len
    };

    let mut tokens = Array2::<usize>::from_elem((sentences.len(), seq_len), pad_idx);

    for (i, sentence) in sentences.iter().enumerate() {
        let mut pos = 0;

        // Add SOS token
        if add_sos_eos {
            tokens[[i, pos]] = sos_idx;
            pos += 1;
        }

        // Add tokens
        for word in sentence.iter().take(max_len) {
            tokens[[i, pos]] = *word_to_idx.get(word).unwrap_or(&unk_idx);
            pos += 1;
        }

        // Add EOS token
        if add_sos_eos && pos < seq_len {
            tokens[[i, pos]] = eos_idx;
        }
    }

    tokens
}

// Simplified training function (without actual parameter updates)
fn train_transformer(
    model: &mut Transformer,
    src_tokens: &Array2<usize>,
    tgt_tokens: &Array2<usize>,
    num_epochs: usize,
    src_pad_idx: usize,
    tgt_pad_idx: usize,
) {
    println!("Training Transformer model...");

    // Training loop
    for epoch in 1..=num_epochs {
        // Forward pass
        let logits = model.forward(
            src_tokens,
            &tgt_tokens
                .slice(s![.., ..tgt_tokens.shape()[1] - 1])
                .to_owned(),
            src_pad_idx,
            tgt_pad_idx,
        );

        // Target tokens (shifted right) for loss calculation
        let target_tokens = tgt_tokens.slice(s![.., 1..]).to_owned();
        let batch_size = src_tokens.shape()[0];
        let tgt_len = target_tokens.shape()[1];

        // Calculate cross-entropy loss
        let mut total_loss = 0.0;
        let mut token_count = 0;

        for b in 0..batch_size {
            for t in 0..tgt_len {
                let target = target_tokens[[b, t]];

                if target != tgt_pad_idx {
                    // Get logits for this position
                    let logits_t = logits.slice(s![b, t, ..]).to_owned();

                    // Calculate loss (negative log-likelihood)
                    // Find max for numerical stability
                    let max_val = logits_t.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                    // Calculate softmax denominator
                    let mut sum_exp = 0.0;
                    for l in 0..logits_t.len() {
                        sum_exp += (logits_t[l] - max_val).exp();
                    }

                    // Calculate log-probability of target token
                    if target < model.tgt_vocab_size {
                        let log_prob = logits_t[target] - max_val - sum_exp.ln();
                        total_loss -= log_prob;
                        token_count += 1;
                    }
                }
            }
        }

        let avg_loss = if token_count > 0 {
            total_loss / token_count as f32
        } else {
            0.0
        };

        println!("Epoch {}/{} - Loss: {:.4}", epoch, num_epochs, avg_loss);

        // In a real implementation, we would:
        // 1. Calculate gradients through backpropagation
        // 2. Update parameters using an optimizer
        // 3. Validate on a separate dataset
    }
}

// Evaluate the model
fn evaluate_transformer(
    model: &mut Transformer,
    src_tokens: &Array2<usize>,
    tgt_tokens: &Array2<usize>,
    src_idx_to_word: &HashMap<usize, String>,
    tgt_idx_to_word: &HashMap<usize, String>,
    src_pad_idx: usize,
    tgt_pad_idx: usize,
    sos_idx: usize,
    eos_idx: usize,
) {
    println!("\nEvaluating model...");

    // Greedy decoding
    let decoded_tokens = model.greedy_decode(src_tokens, src_pad_idx, sos_idx, eos_idx, 20);

    // Calculate BLEU score (simplified version)
    let batch_size = src_tokens.shape()[0];
    let mut total_match_1gram = 0;
    let mut total_candidate_1gram = 0;
    let mut total_reference_1gram = 0;

    // Print some examples
    println!("\nTranslation examples:");

    for b in 0..batch_size.min(5) {
        // Source sentence
        let mut src_sentence = String::new();
        for t in 0..src_tokens.shape()[1] {
            let token_id = src_tokens[[b, t]];
            if token_id != src_pad_idx {
                if let Some(word) = src_idx_to_word.get(&token_id) {
                    if word != "<pad>" && word != "<sos>" && word != "<eos>" {
                        src_sentence.push_str(word);
                        src_sentence.push(' ');
                    }
                }
            }
        }

        // Reference translation
        let mut ref_sentence = String::new();
        let mut ref_tokens = Vec::new();

        for t in 0..tgt_tokens.shape()[1] {
            let token_id = tgt_tokens[[b, t]];
            if token_id != tgt_pad_idx {
                if let Some(word) = tgt_idx_to_word.get(&token_id) {
                    if word != "<pad>" && word != "<sos>" && word != "<eos>" {
                        ref_sentence.push_str(word);
                        ref_sentence.push(' ');
                        ref_tokens.push(token_id);
                    }
                }
            }
        }

        // Model translation
        let mut model_sentence = String::new();
        let mut model_tokens = Vec::new();

        for t in 0..decoded_tokens.shape()[1] {
            let token_id = decoded_tokens[[b, t]];
            if token_id == eos_idx {
                break;
            }

            if token_id != tgt_pad_idx && token_id != sos_idx {
                if let Some(word) = tgt_idx_to_word.get(&token_id) {
                    if word != "<pad>" && word != "<sos>" && word != "<eos>" {
                        model_sentence.push_str(word);
                        model_sentence.push(' ');
                        model_tokens.push(token_id);
                    }
                }
            }
        }

        // Calculate 1-gram matches
        let mut matches = 0;
        for &model_token in &model_tokens {
            for &ref_token in &ref_tokens {
                if model_token == ref_token {
                    matches += 1;
                    break;
                }
            }
        }

        total_match_1gram += matches;
        total_candidate_1gram += model_tokens.len();
        total_reference_1gram += ref_tokens.len();

        // Print results
        println!("Source:     {}", src_sentence.trim());
        println!("Reference:  {}", ref_sentence.trim());
        println!("Generated:  {}", model_sentence.trim());
        println!();
    }

    // Calculate precision and recall
    let precision = if total_candidate_1gram > 0 {
        total_match_1gram as f32 / total_candidate_1gram as f32
    } else {
        0.0
    };

    let recall = if total_reference_1gram > 0 {
        total_match_1gram as f32 / total_reference_1gram as f32
    } else {
        0.0
    };

    // Calculate F1 score
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    println!("Evaluation metrics:");
    println!("Precision: {:.4}", precision);
    println!("Recall:    {:.4}", recall);
    println!("F1 Score:  {:.4}", f1);
}

fn main() {
    println!("Neural Machine Translation with Transformer");
    println!("==========================================");

    // Create a toy translation dataset
    let (english, french) = create_toy_translation_dataset();

    // Create vocabularies
    let (en_word_to_idx, en_idx_to_word) = create_vocabulary(&english);
    let (fr_word_to_idx, fr_idx_to_word) = create_vocabulary(&french);

    println!("English vocabulary size: {}", en_word_to_idx.len());
    println!("French vocabulary size: {}", fr_word_to_idx.len());

    // Tokenize sentences
    let en_max_len = english.iter().map(|s| s.len()).max().unwrap_or(10);
    let fr_max_len = french.iter().map(|s| s.len()).max().unwrap_or(10);
    let max_len = en_max_len.max(fr_max_len);

    let src_tokens = tokenize_sentences(&english, &en_word_to_idx, max_len, false);
    let tgt_tokens = tokenize_sentences(&french, &fr_word_to_idx, max_len, true);

    // Model parameters
    let d_model = 32; // Embedding dimension
    let num_heads = 4; // Number of attention heads
    let num_encoder_layers = 2;
    let num_decoder_layers = 2;
    let d_ff = 64; // Feed-forward dimension
    let _batch_size = english.len();
    let max_seq_len = max_len + 2; // Add space for SOS and EOS
    let dropout_rate = 0.1;

    // Create Transformer model
    let mut model = Transformer::new(
        en_word_to_idx.len(),
        fr_word_to_idx.len(),
        d_model,
        num_encoder_layers,
        num_decoder_layers,
        num_heads,
        d_ff,
        max_seq_len,
        dropout_rate,
    );

    // Special token indices
    let src_pad_idx = *en_word_to_idx.get("<pad>").unwrap_or(&0);
    let tgt_pad_idx = *fr_word_to_idx.get("<pad>").unwrap_or(&0);
    let sos_idx = *fr_word_to_idx.get("<sos>").unwrap_or(&1);
    let eos_idx = *fr_word_to_idx.get("<eos>").unwrap_or(&2);

    // Train model
    train_transformer(
        &mut model,
        &src_tokens,
        &tgt_tokens,
        10,
        src_pad_idx,
        tgt_pad_idx,
    );

    // Evaluate model
    evaluate_transformer(
        &mut model,
        &src_tokens,
        &tgt_tokens,
        &en_idx_to_word,
        &fr_idx_to_word,
        src_pad_idx,
        tgt_pad_idx,
        sos_idx,
        eos_idx,
    );

    println!("\nTransformer NMT model implementation completed!");
}
