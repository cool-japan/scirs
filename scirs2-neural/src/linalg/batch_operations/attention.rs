//! Attention mechanism operations for neural networks
//!
//! This module contains functions for implementing various attention mechanisms
//! used in transformer models and other sequence processing neural networks.

use ndarray::{s, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::{NeuralError, Result};

/// Type alias for feed forward return values
type FeedForwardReturn<F> = (Array3<F>, Array2<F>, Array1<F>, Array2<F>, Array1<F>);

/// Computes scaled dot-product attention for transformer models.
///
/// Calculates attention values based on query, key and value matrices.
/// Formula: softmax(Q K^T / sqrt(d_k)) V
///
/// # Arguments
///
/// * `query` - Query matrix with shape [batch_size, seq_len_q, d_k]
/// * `key` - Key matrix with shape [batch_size, seq_len_k, d_k]
/// * `value` - Value matrix with shape [batch_size, seq_len_k, d_v]
/// * `mask` - Optional mask with shape [batch_size, seq_len_q, seq_len_k]
///   Used to mask certain positions (e.g., future positions in decoder)
///
/// # Returns
///
/// * Tuple of (attention_output, attention_weights) where:
///   - attention_output has shape [batch_size, seq_len_q, d_v]
///   - attention_weights has shape [batch_size, seq_len_q, seq_len_k]
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array3};
/// use scirs2_neural::linalg::scaled_dot_product_attention;
///
/// // Create sample inputs
/// let batch_size = 2;
/// let seq_len_q = 3;
/// let seq_len_k = 4;
/// let d_k = 2;
/// let d_v = 2;
///
/// // Initialize random query, key, value matrices
/// let query = Array::from_shape_vec(
///     (batch_size, seq_len_q, d_k),
///     vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
/// ).unwrap();
///
/// let key = Array::from_shape_vec(
///     (batch_size, seq_len_k, d_k),
///     vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
/// ).unwrap();
///
/// let value = Array::from_shape_vec(
///     (batch_size, seq_len_k, d_v),
///     vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
/// ).unwrap();
///
/// // Compute attention (without mask)
/// let (output, weights) = scaled_dot_product_attention(
///     &query.view(),
///     &key.view(),
///     &value.view(),
///     None
/// ).unwrap();
///
/// assert_eq!(output.shape(), &[batch_size, seq_len_q, d_v]);
/// assert_eq!(weights.shape(), &[batch_size, seq_len_q, seq_len_k]);
/// ```
pub fn scaled_dot_product_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    mask: Option<&ArrayView3<F>>,
) -> Result<(Array3<F>, Array3<F>)>
where
    F: Float + Debug,
{
    // Get dimensions
    let batch_size = query.shape()[0];
    let seq_len_q = query.shape()[1];
    let d_k = query.shape()[2];

    let seq_len_k = key.shape()[1];
    let d_v = value.shape()[2];

    // Validate shapes
    if key.shape()[0] != batch_size || key.shape()[2] != d_k {
        return Err(NeuralError::ShapeMismatch(format!(
            "Key shape mismatch in scaled_dot_product_attention: query shape {:?}, key shape {:?}",
            query.shape(),
            key.shape()
        )));
    }

    if value.shape()[0] != batch_size || value.shape()[1] != seq_len_k {
        return Err(NeuralError::ShapeMismatch(
            format!("Value shape mismatch in scaled_dot_product_attention: key shape {:?}, value shape {:?}",
                   key.shape(), value.shape())
        ));
    }

    if let Some(m) = mask {
        if m.shape()[0] != batch_size || m.shape()[1] != seq_len_q || m.shape()[2] != seq_len_k {
            return Err(NeuralError::ShapeMismatch(format!(
                "Mask shape mismatch in scaled_dot_product_attention: expected {:?}, got {:?}",
                [batch_size, seq_len_q, seq_len_k],
                m.shape()
            )));
        }
    }

    // Initialize attention scores: Q * K^T / sqrt(d_k)
    let scale = F::one() / F::from(d_k).unwrap().sqrt();
    let mut attention_scores = Array3::<F>::zeros((batch_size, seq_len_q, seq_len_k));

    // Compute Q * K^T
    for b in 0..batch_size {
        for i in 0..seq_len_q {
            for j in 0..seq_len_k {
                let mut sum = F::zero();

                for k in 0..d_k {
                    sum = sum + query[[b, i, k]] * key[[b, j, k]];
                }

                attention_scores[[b, i, j]] = sum * scale;
            }
        }
    }

    // Apply mask if provided
    if let Some(m) = mask {
        for b in 0..batch_size {
            for i in 0..seq_len_q {
                for j in 0..seq_len_k {
                    if m[[b, i, j]] == F::zero() {
                        // Set masked positions to large negative value for softmax
                        attention_scores[[b, i, j]] = F::from(-1e9).unwrap();
                    }
                }
            }
        }
    }

    // Apply softmax along the key dimension
    let mut attention_weights = Array3::<F>::zeros((batch_size, seq_len_q, seq_len_k));

    for b in 0..batch_size {
        for i in 0..seq_len_q {
            // Find max for numerical stability
            let mut max_val = attention_scores[[b, i, 0]];
            for j in 1..seq_len_k {
                if attention_scores[[b, i, j]] > max_val {
                    max_val = attention_scores[[b, i, j]];
                }
            }

            // Compute exp(x - max_val) for each score
            let mut sum_exp = F::zero();
            for j in 0..seq_len_k {
                let exp_val = (attention_scores[[b, i, j]] - max_val).exp();
                attention_weights[[b, i, j]] = exp_val;
                sum_exp = sum_exp + exp_val;
            }

            // Normalize to get softmax values
            for j in 0..seq_len_k {
                attention_weights[[b, i, j]] = attention_weights[[b, i, j]] / sum_exp;
            }
        }
    }

    // Compute attention output (attention_weights * V)
    let mut output = Array3::<F>::zeros((batch_size, seq_len_q, d_v));

    for b in 0..batch_size {
        for i in 0..seq_len_q {
            for j in 0..d_v {
                let mut sum = F::zero();

                for k in 0..seq_len_k {
                    sum = sum + attention_weights[[b, i, k]] * value[[b, k, j]];
                }

                output[[b, i, j]] = sum;
            }
        }
    }

    Ok((output, attention_weights))
}

/// Implements multi-head attention mechanism for transformer models.
///
/// Splits query, key, and value into multiple heads, computes scaled dot-product
/// attention for each head, and concatenates the results.
///
/// # Arguments
///
/// * `query` - Query matrix with shape [batch_size, seq_len_q, d_model]
/// * `key` - Key matrix with shape [batch_size, seq_len_k, d_model]
/// * `value` - Value matrix with shape [batch_size, seq_len_k, d_model]
/// * `wq` - Query weight matrix with shape [d_model, d_model]
/// * `wk` - Key weight matrix with shape [d_model, d_model]
/// * `wv` - Value weight matrix with shape [d_model, d_model]
/// * `wo` - Output weight matrix with shape [d_model, d_model]
/// * `num_heads` - Number of attention heads
/// * `mask` - Optional mask with shape [batch_size, seq_len_q, seq_len_k]
///
/// # Returns
///
/// * Multi-head attention output with shape [batch_size, seq_len_q, d_model]
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array2, Array3};
/// use scirs2_neural::linalg::multi_head_attention;
///
/// // Create sample inputs
/// let batch_size = 2;
/// let seq_len_q = 3;
/// let seq_len_k = 4;
/// let d_model = 8;
/// let num_heads = 2;
///
/// // Initialize inputs with placeholder values
/// let query = Array::from_shape_fn(
///     (batch_size, seq_len_q, d_model),
///     |_| 0.1
/// );
///
/// let key = Array::from_shape_fn(
///     (batch_size, seq_len_k, d_model),
///     |_| 0.1
/// );
///
/// let value = Array::from_shape_fn(
///     (batch_size, seq_len_k, d_model),
///     |_| 0.1
/// );
///
/// // Initialize weight matrices
/// let wq = Array::from_shape_fn((d_model, d_model), |_| 0.1);
/// let wk = Array::from_shape_fn((d_model, d_model), |_| 0.1);
/// let wv = Array::from_shape_fn((d_model, d_model), |_| 0.1);
/// let wo = Array::from_shape_fn((d_model, d_model), |_| 0.1);
///
/// // Compute multi-head attention
/// let output = multi_head_attention(
///     &query.view(),
///     &key.view(),
///     &value.view(),
///     &wq.view(),
///     &wk.view(),
///     &wv.view(),
///     &wo.view(),
///     num_heads,
///     None
/// ).unwrap();
///
/// assert_eq!(output.shape(), &[batch_size, seq_len_q, d_model]);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    wq: &ArrayView2<F>,
    wk: &ArrayView2<F>,
    wv: &ArrayView2<F>,
    wo: &ArrayView2<F>,
    num_heads: usize,
    mask: Option<&ArrayView3<F>>,
) -> Result<Array3<F>>
where
    F: Float + Debug,
{
    // Get dimensions
    let batch_size = query.shape()[0];
    let seq_len_q = query.shape()[1];
    let seq_len_k = key.shape()[1];
    let d_model = query.shape()[2];

    // Validate shapes
    if key.shape()[0] != batch_size || key.shape()[2] != d_model {
        return Err(NeuralError::ShapeMismatch(format!(
            "Key shape mismatch in multi_head_attention: query shape {:?}, key shape {:?}",
            query.shape(),
            key.shape()
        )));
    }

    if value.shape()[0] != batch_size
        || value.shape()[1] != seq_len_k
        || value.shape()[2] != d_model
    {
        return Err(NeuralError::ShapeMismatch(format!(
            "Value shape mismatch in multi_head_attention: key shape {:?}, value shape {:?}",
            key.shape(),
            value.shape()
        )));
    }

    if wq.shape() != [d_model, d_model]
        || wk.shape() != [d_model, d_model]
        || wv.shape() != [d_model, d_model]
        || wo.shape() != [d_model, d_model]
    {
        return Err(NeuralError::ShapeMismatch(
            "Weight matrix shape mismatch in multi_head_attention".to_string(),
        ));
    }

    if d_model % num_heads != 0 {
        return Err(NeuralError::ShapeMismatch(format!(
            "d_model ({}) must be divisible by num_heads ({})",
            d_model, num_heads
        )));
    }

    let depth = d_model / num_heads;

    // Linear projections
    let mut q_proj = Array3::<F>::zeros((batch_size, seq_len_q, d_model));
    let mut k_proj = Array3::<F>::zeros((batch_size, seq_len_k, d_model));
    let mut v_proj = Array3::<F>::zeros((batch_size, seq_len_k, d_model));

    // Apply weight matrices
    for b in 0..batch_size {
        for i in 0..seq_len_q {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum = sum + query[[b, i, k]] * wq[[k, j]];
                }
                q_proj[[b, i, j]] = sum;
            }
        }

        for i in 0..seq_len_k {
            for j in 0..d_model {
                let mut sum_k = F::zero();
                let mut sum_v = F::zero();

                for k in 0..d_model {
                    sum_k = sum_k + key[[b, i, k]] * wk[[k, j]];
                    sum_v = sum_v + value[[b, i, k]] * wv[[k, j]];
                }

                k_proj[[b, i, j]] = sum_k;
                v_proj[[b, i, j]] = sum_v;
            }
        }
    }

    // Reshape to split into heads
    let mut q_split = Array4::<F>::zeros((batch_size, seq_len_q, num_heads, depth));
    let mut k_split = Array4::<F>::zeros((batch_size, seq_len_k, num_heads, depth));
    let mut v_split = Array4::<F>::zeros((batch_size, seq_len_k, num_heads, depth));

    // Split projections into multiple heads
    for b in 0..batch_size {
        for i in 0..seq_len_q {
            for h in 0..num_heads {
                for d in 0..depth {
                    q_split[[b, i, h, d]] = q_proj[[b, i, h * depth + d]];
                }
            }
        }

        for i in 0..seq_len_k {
            for h in 0..num_heads {
                for d in 0..depth {
                    k_split[[b, i, h, d]] = k_proj[[b, i, h * depth + d]];
                    v_split[[b, i, h, d]] = v_proj[[b, i, h * depth + d]];
                }
            }
        }
    }

    // Transpose to get [batch_size, num_heads, seq_len, depth]
    let mut q_heads = Array4::<F>::zeros((batch_size, num_heads, seq_len_q, depth));
    let mut k_heads = Array4::<F>::zeros((batch_size, num_heads, seq_len_k, depth));
    let mut v_heads = Array4::<F>::zeros((batch_size, num_heads, seq_len_k, depth));

    for b in 0..batch_size {
        for h in 0..num_heads {
            for i in 0..seq_len_q {
                for d in 0..depth {
                    q_heads[[b, h, i, d]] = q_split[[b, i, h, d]];
                }
            }

            for i in 0..seq_len_k {
                for d in 0..depth {
                    k_heads[[b, h, i, d]] = k_split[[b, i, h, d]];
                    v_heads[[b, h, i, d]] = v_split[[b, i, h, d]];
                }
            }
        }
    }

    // Prepare scaled dot-product attention for each head
    let mut head_outputs = Array4::<F>::zeros((batch_size, num_heads, seq_len_q, depth));

    // Apply scaled dot-product attention to each head
    for h in 0..num_heads {
        // Extract query, key, value for current head
        let mut q_head = Array3::<F>::zeros((batch_size, seq_len_q, depth));
        let mut k_head = Array3::<F>::zeros((batch_size, seq_len_k, depth));
        let mut v_head = Array3::<F>::zeros((batch_size, seq_len_k, depth));

        for b in 0..batch_size {
            for i in 0..seq_len_q {
                for d in 0..depth {
                    q_head[[b, i, d]] = q_heads[[b, h, i, d]];
                }
            }

            for i in 0..seq_len_k {
                for d in 0..depth {
                    k_head[[b, i, d]] = k_heads[[b, h, i, d]];
                    v_head[[b, i, d]] = v_heads[[b, h, i, d]];
                }
            }
        }

        // Compute attention for this head
        let (head_output, _) =
            scaled_dot_product_attention(&q_head.view(), &k_head.view(), &v_head.view(), mask)?;

        // Store head output
        for b in 0..batch_size {
            for i in 0..seq_len_q {
                for d in 0..depth {
                    head_outputs[[b, h, i, d]] = head_output[[b, i, d]];
                }
            }
        }
    }

    // Reshape back: transpose and concatenate heads
    let mut concat_heads = Array3::<F>::zeros((batch_size, seq_len_q, d_model));

    for b in 0..batch_size {
        for i in 0..seq_len_q {
            for h in 0..num_heads {
                for d in 0..depth {
                    concat_heads[[b, i, h * depth + d]] = head_outputs[[b, h, i, d]];
                }
            }
        }
    }

    // Final linear projection
    let mut output = Array3::<F>::zeros((batch_size, seq_len_q, d_model));

    for b in 0..batch_size {
        for i in 0..seq_len_q {
            for j in 0..d_model {
                let mut sum = F::zero();

                for k in 0..d_model {
                    sum = sum + concat_heads[[b, i, k]] * wo[[k, j]];
                }

                output[[b, i, j]] = sum;
            }
        }
    }

    Ok(output)
}

/// Generates positional encoding for transformer models.
///
/// Adds information about the position of tokens in a sequence, allowing
/// the model to understand the order of tokens without recurrence.
///
/// # Arguments
///
/// * `seq_len` - Length of the sequence
/// * `d_model` - Dimensionality of the model
/// * `max_seq_len` - Maximum sequence length for precomputation (optional)
///
/// # Returns
///
/// * Positional encoding matrix with shape [seq_len, d_model]
///
/// # Examples
///
/// ```
/// use scirs2_neural::linalg::positional_encoding;
///
/// // Generate positional encoding for a sequence of length 10
/// // with embedding dimension 16
/// let pos_encoding = positional_encoding::<f64>(10, 16, None).unwrap();
/// assert_eq!(pos_encoding.shape(), &[10, 16]);
/// ```
pub fn positional_encoding<F: Float + Debug>(
    seq_len: usize,
    d_model: usize,
    max_seq_len: Option<usize>,
) -> Result<Array2<F>> {
    // Use provided max_seq_len or default to seq_len
    let max_len = max_seq_len.unwrap_or(seq_len);

    if max_len < seq_len {
        return Err(NeuralError::InvalidArgument(format!(
            "max_seq_len ({}) must be at least as large as seq_len ({})",
            max_len, seq_len
        )));
    }

    if d_model % 2 != 0 {
        return Err(NeuralError::InvalidArgument(format!(
            "d_model ({}) must be even",
            d_model
        )));
    }

    // Allocate position encoding matrix
    let mut pos_encoding = Array2::<F>::zeros((max_len, d_model));

    // Calculate position encoding
    for pos in 0..max_len {
        for i in 0..d_model / 2 {
            let div_term =
                (F::from(2 * i).unwrap() / F::from(d_model).unwrap()).exp() * F::from(1e4).unwrap();
            let pos_f = F::from(pos).unwrap();

            // Even indices: sin
            pos_encoding[[pos, 2 * i]] = (pos_f / div_term).sin();

            // Odd indices: cos
            pos_encoding[[pos, 2 * i + 1]] = (pos_f / div_term).cos();
        }
    }

    // Return the requested sequence length portion
    let result = pos_encoding.slice(s![0..seq_len, ..]).to_owned();
    Ok(result)
}

/// Implements the feed-forward network component of a transformer model.
///
/// The feed-forward network consists of two linear transformations with a
/// ReLU activation in between.
///
/// # Arguments
///
/// * `x` - Input tensor with shape [batch_size, seq_len, d_model]
/// * `w1` - First weight matrix with shape [d_model, d_ff]
/// * `b1` - First bias vector with shape [d_ff]
/// * `w2` - Second weight matrix with shape [d_ff, d_model]
/// * `b2` - Second bias vector with shape [d_model]
///
/// # Returns
///
/// * Output tensor with shape [batch_size, seq_len, d_model]
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array1, Array2, Array3};
/// use scirs2_neural::linalg::transformer_ffn;
///
/// // Create sample inputs
/// let batch_size = 2;
/// let seq_len = 3;
/// let d_model = 4;
/// let d_ff = 8;
///
/// // Initialize inputs and weights
/// let x = Array::from_shape_fn(
///     (batch_size, seq_len, d_model),
///     |_| 0.1
/// );
///
/// let w1 = Array::from_shape_fn((d_model, d_ff), |_| 0.1);
/// let b1 = Array::from_shape_fn(d_ff, |_| 0.1);
/// let w2 = Array::from_shape_fn((d_ff, d_model), |_| 0.1);
/// let b2 = Array::from_shape_fn(d_model, |_| 0.1);
///
/// // Apply transformer feed-forward network
/// let output = transformer_ffn(
///     &x.view(),
///     &w1.view(),
///     &b1.view(),
///     &w2.view(),
///     &b2.view()
/// ).unwrap();
///
/// assert_eq!(output.shape(), x.shape());
/// ```
pub fn transformer_ffn<F>(
    x: &ArrayView3<F>,
    w1: &ArrayView2<F>,
    b1: &ArrayView1<F>,
    w2: &ArrayView2<F>,
    b2: &ArrayView1<F>,
) -> Result<Array3<F>>
where
    F: Float + Debug,
{
    // Get dimensions
    let batch_size = x.shape()[0];
    let seq_len = x.shape()[1];
    let d_model = x.shape()[2];
    let d_ff = w1.shape()[1];

    // Validate shapes
    if w1.shape()[0] != d_model {
        return Err(NeuralError::ShapeMismatch(format!(
            "w1 shape mismatch in transformer_ffn: x shape {:?}, w1 shape {:?}",
            x.shape(),
            w1.shape()
        )));
    }

    if b1.shape()[0] != d_ff {
        return Err(NeuralError::ShapeMismatch(format!(
            "b1 shape mismatch in transformer_ffn: b1 shape {:?}, expected [{:?}]",
            b1.shape(),
            d_ff
        )));
    }

    if w2.shape()[0] != d_ff || w2.shape()[1] != d_model {
        return Err(NeuralError::ShapeMismatch(format!(
            "w2 shape mismatch in transformer_ffn: w2 shape {:?}, expected [{:?}, {:?}]",
            w2.shape(),
            d_ff,
            d_model
        )));
    }

    if b2.shape()[0] != d_model {
        return Err(NeuralError::ShapeMismatch(format!(
            "b2 shape mismatch in transformer_ffn: b2 shape {:?}, expected [{:?}]",
            b2.shape(),
            d_model
        )));
    }

    // First layer: linear + ReLU
    let mut ffn_mid = Array3::<F>::zeros((batch_size, seq_len, d_ff));

    for b in 0..batch_size {
        for s in 0..seq_len {
            for j in 0..d_ff {
                let mut sum = b1[j]; // Add bias

                for k in 0..d_model {
                    sum = sum + x[[b, s, k]] * w1[[k, j]];
                }

                // Apply ReLU: max(0, x)
                ffn_mid[[b, s, j]] = if sum > F::zero() { sum } else { F::zero() };
            }
        }
    }

    // Second layer: linear
    let mut output = Array3::<F>::zeros((batch_size, seq_len, d_model));

    for b in 0..batch_size {
        for s in 0..seq_len {
            for j in 0..d_model {
                let mut sum = b2[j]; // Add bias

                for k in 0..d_ff {
                    sum = sum + ffn_mid[[b, s, k]] * w2[[k, j]];
                }

                output[[b, s, j]] = sum;
            }
        }
    }

    Ok(output)
}

/// Computes the backward pass for transformer feed-forward network.
///
/// # Arguments
///
/// * `dout` - Gradient of loss with respect to FFN output, shape [batch_size, seq_len, d_model]
/// * `x` - Original input to FFN, shape [batch_size, seq_len, d_model]
/// * `w1` - First weight matrix, shape [d_model, d_ff]
/// * `b1` - First bias vector, shape [d_ff]
/// * `w2` - Second weight matrix, shape [d_ff, d_model]
/// * `b2` - Second bias vector, shape [d_model]
///
/// # Returns
///
/// * Tuple of (dx, dw1, db1, dw2, db2) containing gradients for all inputs
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Array1, Array2, Array3};
/// use scirs2_neural::linalg::{transformer_ffn, transformer_ffn_backward};
///
/// // Setup (same as forward example)
/// let batch_size = 2;
/// let seq_len = 3;
/// let d_model = 4;
/// let d_ff = 8;
///
/// let x = Array::from_shape_fn((batch_size, seq_len, d_model), |_| 0.1);
/// let w1 = Array::from_shape_fn((d_model, d_ff), |_| 0.1);
/// let b1 = Array::from_shape_fn(d_ff, |_| 0.1);
/// let w2 = Array::from_shape_fn((d_ff, d_model), |_| 0.1);
/// let b2 = Array::from_shape_fn(d_model, |_| 0.1);
///
/// // Forward pass
/// let out = transformer_ffn(&x.view(), &w1.view(), &b1.view(), &w2.view(), &b2.view()).unwrap();
///
/// // Assume gradient of loss with respect to output
/// let dout = Array::from_shape_fn(out.raw_dim(), |_| 0.01);
///
/// // Backward pass
/// let (dx, dw1, db1, dw2, db2) = transformer_ffn_backward(
///     &dout.view(), &x.view(), &w1.view(), &b1.view(), &w2.view(), &b2.view()
/// ).unwrap();
///
/// assert_eq!(dx.shape(), x.shape());
/// assert_eq!(dw1.shape(), w1.shape());
/// assert_eq!(db1.shape(), b1.shape());
/// assert_eq!(dw2.shape(), w2.shape());
/// assert_eq!(db2.shape(), b2.shape());
/// ```
pub fn transformer_ffn_backward<F>(
    dout: &ArrayView3<F>,
    x: &ArrayView3<F>,
    w1: &ArrayView2<F>,
    b1: &ArrayView1<F>,
    w2: &ArrayView2<F>,
    b2: &ArrayView1<F>,
) -> Result<FeedForwardReturn<F>>
where
    F: Float + Debug,
{
    // Get dimensions
    let batch_size = x.shape()[0];
    let seq_len = x.shape()[1];
    let d_model = x.shape()[2];
    let d_ff = w1.shape()[1];

    // Validate shapes
    if dout.shape() != x.shape() {
        return Err(NeuralError::ShapeMismatch(format!(
            "dout shape mismatch in transformer_ffn_backward: dout shape {:?}, x shape {:?}",
            dout.shape(),
            x.shape()
        )));
    }

    // Forward pass to get intermediate activations
    // First layer: linear
    let mut ffn_pre_relu = Array3::<F>::zeros((batch_size, seq_len, d_ff));

    for b in 0..batch_size {
        for s in 0..seq_len {
            for j in 0..d_ff {
                let mut sum = b1[j]; // Add bias

                for k in 0..d_model {
                    sum = sum + x[[b, s, k]] * w1[[k, j]];
                }

                ffn_pre_relu[[b, s, j]] = sum;
            }
        }
    }

    // Apply ReLU
    let mut ffn_mid = Array3::<F>::zeros((batch_size, seq_len, d_ff));

    for b in 0..batch_size {
        for s in 0..seq_len {
            for j in 0..d_ff {
                ffn_mid[[b, s, j]] = if ffn_pre_relu[[b, s, j]] > F::zero() {
                    ffn_pre_relu[[b, s, j]]
                } else {
                    F::zero()
                };
            }
        }
    }

    // Initialize gradients
    let mut dx = Array3::<F>::zeros(x.raw_dim());
    let mut dw1 = Array2::<F>::zeros(w1.raw_dim());
    let mut db1 = Array1::<F>::zeros(b1.raw_dim());
    let mut dw2 = Array2::<F>::zeros(w2.raw_dim());
    let mut db2 = Array1::<F>::zeros(b2.raw_dim());

    // Gradient with respect to second layer output
    let mut dffn_mid = Array3::<F>::zeros(ffn_mid.raw_dim());

    // Backpropagate through second linear layer
    for b in 0..batch_size {
        for s in 0..seq_len {
            for j in 0..d_model {
                // Gradient with respect to bias
                db2[j] = db2[j] + dout[[b, s, j]];

                // Gradient with respect to weights
                for k in 0..d_ff {
                    dw2[[k, j]] = dw2[[k, j]] + ffn_mid[[b, s, k]] * dout[[b, s, j]];
                    dffn_mid[[b, s, k]] = dffn_mid[[b, s, k]] + w2[[k, j]] * dout[[b, s, j]];
                }
            }
        }
    }

    // Backpropagate through ReLU
    let mut dffn_pre_relu = Array3::<F>::zeros(ffn_pre_relu.raw_dim());

    for b in 0..batch_size {
        for s in 0..seq_len {
            for j in 0..d_ff {
                // ReLU gradient is 1 for positive inputs, 0 otherwise
                if ffn_pre_relu[[b, s, j]] > F::zero() {
                    dffn_pre_relu[[b, s, j]] = dffn_mid[[b, s, j]];
                }
            }
        }
    }

    // Backpropagate through first linear layer
    for b in 0..batch_size {
        for s in 0..seq_len {
            for j in 0..d_ff {
                // Gradient with respect to bias
                db1[j] = db1[j] + dffn_pre_relu[[b, s, j]];

                // Gradient with respect to weights and inputs
                for k in 0..d_model {
                    dw1[[k, j]] = dw1[[k, j]] + x[[b, s, k]] * dffn_pre_relu[[b, s, j]];
                    dx[[b, s, k]] = dx[[b, s, k]] + w1[[k, j]] * dffn_pre_relu[[b, s, j]];
                }
            }
        }
    }

    Ok((dx, dw1, db1, dw2, db2))
}
