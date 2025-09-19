//! Optimized attention mechanisms for transformer models
//!
//! This module provides efficient implementations of various attention mechanisms
//! commonly used in transformer-based neural networks. It includes standard
//! scaled dot-product attention, multi-head attention, flash attention for memory
//! efficiency, sparse attention patterns, and various position-aware attention variants.
//!
//! ## Overview
//!
//! * Basic attention - scaled dot-product attention, masked attention
//! * Memory-efficient implementations - flash attention
//! * Sparsity-aware implementations - sparse attention patterns
//! * Position-aware variants - relative position encoding, rotary embeddings, ALiBi
//! * Advanced patterns - grouped query attention, linear attention
//!
//! ## Examples
//!
//! Basic scaled dot-product attention:
//!
//! ```
//! use ndarray::{Array2, Array3};
//! use scirs2_linalg::attention::{scaled_dot_product_attention, AttentionMask};
//!
//! // Create query, key, value matrices
//! let batchsize = 2;
//! let seq_len = 4;
//! let d_model = 8;
//!
//! // Random matrices for demonstration
//! let query = Array3::<f32>::ones((batchsize, seq_len, d_model));
//! let key = Array3::<f32>::ones((batchsize, seq_len, d_model));
//! let value = Array3::<f32>::ones((batchsize, seq_len, d_model));
//!
//! // Compute attention
//! let output = scaled_dot_product_attention(
//!     &query.view(),
//!     &key.view(),
//!     &value.view(),
//!     None,
//!     1.0 / (d_model as f32).sqrt()
//! ).unwrap();
//!
//! assert_eq!(output.shape(), &[batchsize, seq_len, d_model]);
//! ```
//!
//! Multi-head attention:
//!
//! ```
//! use ndarray::{Array2, Array3};
//! use scirs2_linalg::attention::{multi_head_attention, AttentionConfig};
//!
//! // Create query, key, value matrices
//! let batchsize = 2;
//! let seq_len = 4;
//! let d_model = 64;
//! let num_heads = 8;
//! let head_dim = d_model / num_heads;
//!
//! // Random matrices for demonstration
//! let query = Array3::<f32>::ones((batchsize, seq_len, d_model));
//! let key = Array3::<f32>::ones((batchsize, seq_len, d_model));
//! let value = Array3::<f32>::ones((batchsize, seq_len, d_model));
//!
//! // Linear projection weights
//! let wq = Array2::<f32>::ones((d_model, d_model));
//! let wk = Array2::<f32>::ones((d_model, d_model));
//! let wv = Array2::<f32>::ones((d_model, d_model));
//! let wo = Array2::<f32>::ones((d_model, d_model));
//!
//! // Configure attention
//! let config = AttentionConfig {
//!     num_heads,
//!     head_dim,
//!     dropout_prob: 0.0,
//!     causal: false,
//!     scale: Some(1.0 / (head_dim as f32).sqrt()),
//! };
//!
//! // Compute multi-head attention
//! let output = multi_head_attention(
//!     &query.view(),
//!     &key.view(),
//!     &value.view(),
//!     &wq.view(),
//!     &wk.view(),
//!     &wv.view(),
//!     &wo.view(),
//!     None,
//!     &config
//! ).unwrap();
//!
//! assert_eq!(output.shape(), &[batchsize, seq_len, d_model]);
//! ```

// Sub-modules
mod cross_attention;
mod multi_head;
mod scaled_dot_product;
mod utils;

// Re-export all public items from sub-modules
pub use cross_attention::{
    attention_with_alibi, attention_with_rpe, flash_attention, linear_attention,
    relative_position_attention, rotary_embedding, sparse_attention,
};
pub use multi_head::{grouped_query_attention, multi_head_attention};
pub use scaled_dot_product::{causal_attention, masked_attention, scaled_dot_product_attention};
pub use utils::{apply_mask, attention, AttentionConfig, AttentionMask};

// For backward compatibility, keep these imports available
use crate::error::LinalgResult;
use ndarray::{Array3, ArrayView3};
use num_traits::{Float, NumAssignOps, Zero};
use std::ops::{Add, Div, Mul, Sub};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_basic_attention() {
        // Simple 2x2x2 test case - consistent inputs for reproducible tests
        let query = array![[[1.0, 1.0], [1.0, 1.0]]]
            .into_shape_with_order((1, 2, 2))
            .unwrap();
        let key = array![[[1.0, 1.0], [1.0, 1.0]]]
            .into_shape_with_order((1, 2, 2))
            .unwrap();
        let value = array![[[5.0, 6.0], [7.0, 8.0]]]
            .into_shape_with_order((1, 2, 2))
            .unwrap();

        // Scale factor (1/sqrt(d_k))
        let scale = 1.0 / (2.0_f64).sqrt();

        let result = attention(&query.view(), &key.view(), &value.view(), None, scale).unwrap();

        // The shape should be the same as the query
        assert_eq!(result.shape(), &[1, 2, 2]);

        // With identical query and key vectors, we expect the attention weights to be equal
        // This means each position should get the average of the value vectors
        let expected_first_pos = [(5.0 + 7.0) / 2.0, (6.0 + 8.0) / 2.0];
        let expected_second_pos = [(5.0 + 7.0) / 2.0, (6.0 + 8.0) / 2.0];

        // Check approximate equality with a more generous tolerance
        assert!((result[[0, 0, 0]] - expected_first_pos[0]).abs() < 1e-5);
        assert!((result[[0, 0, 1]] - expected_first_pos[1]).abs() < 1e-5);
        assert!((result[[0, 1, 0]] - expected_second_pos[0]).abs() < 1e-5);
        assert!((result[[0, 1, 1]] - expected_second_pos[1]).abs() < 1e-5);
    }

    #[test]
    fn test_causal_attention() {
        // Create a simple test case
        let query = array![[[1.0, 1.0], [1.0, 1.0]]]
            .into_shape_with_order((1, 2, 2))
            .unwrap();
        let key = array![[[1.0, 1.0], [1.0, 1.0]]]
            .into_shape_with_order((1, 2, 2))
            .unwrap();
        let value = array![[[1.0, 2.0], [3.0, 4.0]]]
            .into_shape_with_order((1, 2, 2))
            .unwrap();

        let scale = 1.0 / (2.0_f64).sqrt();

        let result = causal_attention(&query.view(), &key.view(), &value.view(), scale).unwrap();

        // First position can only attend to itself
        assert!((result[[0, 0, 0]] - 1.0).abs() < 1e-6);
        assert!((result[[0, 0, 1]] - 2.0).abs() < 1e-6);

        // Second position can attend to both positions
        // Since the attention weights are equal due to identical query and key,
        // the result should be the average of the two value vectors
        assert!((result[[0, 1, 0]] - 2.0).abs() < 1e-6);
        assert!((result[[0, 1, 1]] - 3.0).abs() < 1e-6);
    }
}
