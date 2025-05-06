//! Vision Transformer (ViT) implementation
//!
//! Vision Transformer (ViT) is a transformer-based model for image classification
//! that divides an image into fixed-size patches, linearly embeds them, adds position
//! embeddings, and processes them using a standard Transformer encoder.
//!
//! Reference: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", Dosovitskiy et al. (2020)
//! https://arxiv.org/abs/2010.11929

use crate::error::{NeuralError, Result};
use crate::layers::{
    Dense, Dropout, Layer, LayerNorm, MultiHeadAttention, PatchEmbedding,
};
use ndarray::{Array, IxDyn, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

/// Configuration for a Vision Transformer model
#[derive(Debug, Clone)]
pub struct ViTConfig {
    /// Image size (height, width)
    pub image_size: (usize, usize),
    /// Patch size (height, width)
    pub patch_size: (usize, usize),
    /// Number of input channels (e.g., 3 for RGB)
    pub in_channels: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// MLP hidden dimension
    pub mlp_dim: usize,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Attention dropout rate
    pub attention_dropout_rate: f64,
}

impl ViTConfig {
    /// Create a ViT-Base configuration
    pub fn vit_base(
        image_size: (usize, usize),
        patch_size: (usize, usize),
        in_channels: usize,
        num_classes: usize,
    ) -> Self {
        Self {
            image_size,
            patch_size,
            in_channels,
            num_classes,
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_dim: 3072,
            dropout_rate: 0.1,
            attention_dropout_rate: 0.0,
        }
    }

    /// Create a ViT-Large configuration
    pub fn vit_large(
        image_size: (usize, usize),
        patch_size: (usize, usize),
        in_channels: usize,
        num_classes: usize,
    ) -> Self {
        Self {
            image_size,
            patch_size,
            in_channels,
            num_classes,
            embed_dim: 1024,
            num_layers: 24,
            num_heads: 16,
            mlp_dim: 4096,
            dropout_rate: 0.1,
            attention_dropout_rate: 0.0,
        }
    }

    /// Create a ViT-Huge configuration
    pub fn vit_huge(
        image_size: (usize, usize),
        patch_size: (usize, usize),
        in_channels: usize,
        num_classes: usize,
    ) -> Self {
        Self {
            image_size,
            patch_size,
            in_channels,
            num_classes,
            embed_dim: 1280,
            num_layers: 32,
            num_heads: 16,
            mlp_dim: 5120,
            dropout_rate: 0.1,
            attention_dropout_rate: 0.0,
        }
    }
}

/// Transformer encoder block for ViT
struct TransformerEncoderBlock<F: Float + Debug + ScalarOperand> {
    /// Layer normalization 1
    norm1: LayerNorm<F>,
    /// Multi-head attention
    attention: MultiHeadAttention<F>,
    /// Layer normalization 2
    norm2: LayerNorm<F>,
    /// MLP layers
    mlp: Box<dyn Layer<F>>,
    /// Dropout for attention
    attn_dropout: Dropout<F>,
    /// Dropout for MLP
    mlp_dropout: Dropout<F>,
}

impl<F: Float + Debug + ScalarOperand> TransformerEncoderBlock<F> {
    /// Create a new transformer encoder block
    pub fn new(
        dim: usize,
        num_heads: usize,
        mlp_dim: usize,
        dropout_rate: F,
        attention_dropout_rate: F,
    ) -> Result<Self> {
        // Layer normalization for attention
        let norm1 = LayerNorm::new(dim, 1e-6)?;

        // Multi-head attention
        let attention = MultiHeadAttention::new(dim, num_heads, attention_dropout_rate)?;

        // Layer normalization for MLP
        let norm2 = LayerNorm::new(dim, 1e-6)?;

        // MLP with GELU activation
        // Note: We're creating a simple 2-layer MLP with GELU activation
        struct MLP<F: Float + Debug + ScalarOperand> {
            dense1: Dense<F>,
            dense2: Dense<F>,
            act_fn: Box<dyn Fn(F) -> F>,
        }

        impl<F: Float + Debug + ScalarOperand> Layer<F> for MLP<F> {
            fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
                let mut x = self.dense1.forward(input)?;

                // Apply GELU activation
                x = x.mapv(|v| (*self.act_fn)(v));

                x = self.dense2.forward(&x)?;
                Ok(x)
            }

            fn backward(&self, input: &Array<F, IxDyn>, grad_output: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
                Ok(grad_output.clone())
            }

            fn update(&mut self, learning_rate: F) -> Result<()> {
                self.dense1.update(learning_rate)?;
                self.dense2.update(learning_rate)?;
                Ok(())
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                self
            }
        }

        // GELU activation function
        let gelu = Box::new(|x: F| {
            // Approximation of GELU
            let x3 = x * x * x;
            x * F::from(0.5).unwrap() * (F::one() + (x + F::from(0.044715).unwrap() * x3).tanh())
        });

        let mlp = Box::new(MLP {
            dense1: Dense::new(dim, mlp_dim, None, true)?,
            dense2: Dense::new(mlp_dim, dim, None, true)?,
            act_fn: gelu,
        });

        // Dropouts
        let attn_dropout = Dropout::new(dropout_rate);
        let mlp_dropout = Dropout::new(dropout_rate);

        Ok(Self {
            norm1,
            attention,
            norm2,
            mlp,
            attn_dropout,
            mlp_dropout,
        })
    }
}

impl<F: Float + Debug + ScalarOperand> Layer<F> for TransformerEncoderBlock<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Norm -> Attention -> Dropout -> Add
        let norm1 = self.norm1.forward(input)?;
        let attn = self.attention.forward(&norm1)?;
        let attn_drop = self.attn_dropout.forward(&attn)?;

        // Add residual connection
        let mut residual1 = input.clone();
        for i in 0..residual1.len() {
            residual1[i] = residual1[i] + attn_drop[i];
        }

        // Norm -> MLP -> Dropout -> Add
        let norm2 = self.norm2.forward(&residual1)?;
        let mlp = self.mlp.forward(&norm2)?;
        let mlp_drop = self.mlp_dropout.forward(&mlp)?;

        // Add residual connection
        let mut residual2 = residual1.clone();
        for i in 0..residual2.len() {
            residual2[i] = residual2[i] + mlp_drop[i];
        }

        Ok(residual2)
    }

    fn backward(&self, input: &Array<F, IxDyn>, grad_output: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.norm1.update(learning_rate)?;
        self.attention.update(learning_rate)?;
        self.norm2.update(learning_rate)?;
        self.mlp.update(learning_rate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Vision Transformer implementation
pub struct VisionTransformer<F: Float + Debug + ScalarOperand> {
    /// Patch embedding layer
    patch_embed: PatchEmbedding<F>,
    /// Class token embedding
    cls_token: Array<F, IxDyn>,
    /// Position embedding
    pos_embed: Array<F, IxDyn>,
    /// Dropout layer
    dropout: Dropout<F>,
    /// Transformer encoder blocks
    encoder_blocks: Vec<TransformerEncoderBlock<F>>,
    /// Layer normalization
    norm: LayerNorm<F>,
    /// Final classification head
    classifier: Dense<F>,
    /// Model configuration
    config: ViTConfig,
}

impl<F: Float + Debug + ScalarOperand> VisionTransformer<F> {
    /// Create a new Vision Transformer model
    pub fn new(config: ViTConfig) -> Result<Self> {
        // Calculate number of patches
        let h_patches = config.image_size.0 / config.patch_size.0;
        let w_patches = config.image_size.1 / config.patch_size.1;
        let num_patches = h_patches * w_patches;

        // Create patch embedding layer
        let patch_embed = PatchEmbedding::new(
            config.image_size,
            config.patch_size,
            config.in_channels,
            config.embed_dim,
            true,
        )?;

        // Create class token
        let cls_token = Array::zeros(IxDyn(&[1, 1, config.embed_dim]));

        // Create position embedding (include class token)
        let pos_embed = Array::zeros(IxDyn(&[1, num_patches + 1, config.embed_dim]));

        // Create dropout
        let dropout_rate = F::from(config.dropout_rate).unwrap();
        let dropout = Dropout::new(dropout_rate);

        // Create transformer encoder blocks
        let mut encoder_blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let block = TransformerEncoderBlock::new(
                config.embed_dim,
                config.num_heads,
                config.mlp_dim,
                F::from(config.dropout_rate).unwrap(),
                F::from(config.attention_dropout_rate).unwrap(),
            )?;
            encoder_blocks.push(block);
        }

        // Layer normalization
        let norm = LayerNorm::new(config.embed_dim, 1e-6)?;

        // Classification head
        let classifier = Dense::new(
            config.embed_dim,
            config.num_classes,
            None, // No activation for final layer
            true,
        )?;

        Ok(Self {
            patch_embed,
            cls_token,
            pos_embed,
            dropout,
            encoder_blocks,
            norm,
            classifier,
            config,
        })
    }

    /// Create a ViT-Base model
    pub fn vit_base(
        image_size: (usize, usize),
        patch_size: (usize, usize),
        in_channels: usize,
        num_classes: usize,
    ) -> Result<Self> {
        let config = ViTConfig::vit_base(image_size, patch_size, in_channels, num_classes);
        Self::new(config)
    }

    /// Create a ViT-Large model
    pub fn vit_large(
        image_size: (usize, usize),
        patch_size: (usize, usize),
        in_channels: usize,
        num_classes: usize,
    ) -> Result<Self> {
        let config = ViTConfig::vit_large(image_size, patch_size, in_channels, num_classes);
        Self::new(config)
    }

    /// Create a ViT-Huge model
    pub fn vit_huge(
        image_size: (usize, usize),
        patch_size: (usize, usize),
        in_channels: usize,
        num_classes: usize,
    ) -> Result<Self> {
        let config = ViTConfig::vit_huge(image_size, patch_size, in_channels, num_classes);
        Self::new(config)
    }
}

impl<F: Float + Debug + ScalarOperand> Layer<F> for VisionTransformer<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Check input shape
        let shape = input.shape();
        if shape.len() != 4
            || shape[1] != self.config.in_channels
            || shape[2] != self.config.image_size.0
            || shape[3] != self.config.image_size.1
        {
            return Err(NeuralError::InferenceError(format!(
                "Expected input shape [batch_size, {}, {}, {}], got {:?}",
                self.config.in_channels, self.config.image_size.0, self.config.image_size.1, shape
            )));
        }

        let batch_size = shape[0];

        // Extract patch embeddings
        let mut x = self.patch_embed.forward(input)?;

        // Reshape to [batch_size, num_patches, embed_dim]
        let h_patches = self.config.image_size.0 / self.config.patch_size.0;
        let w_patches = self.config.image_size.1 / self.config.patch_size.1;
        let num_patches = h_patches * w_patches;

        // Prepend class token
        let mut cls_tokens = Array::zeros(IxDyn(&[batch_size, 1, self.config.embed_dim]));
        for b in 0..batch_size {
            for i in 0..self.config.embed_dim {
                cls_tokens[[b, 0, i]] = self.cls_token[[0, 0, i]];
            }
        }

        // Concatenate class token with patch embeddings
        let mut x_with_cls =
            Array::zeros(IxDyn(&[batch_size, num_patches + 1, self.config.embed_dim]));

        // Copy class token
        for b in 0..batch_size {
            for i in 0..self.config.embed_dim {
                x_with_cls[[b, 0, i]] = cls_tokens[[b, 0, i]];
            }
        }

        // Copy patch embeddings
        for b in 0..batch_size {
            for p in 0..num_patches {
                for i in 0..self.config.embed_dim {
                    x_with_cls[[b, p + 1, i]] = x[[b, p, i]];
                }
            }
        }

        // Add position embeddings
        for b in 0..batch_size {
            for p in 0..num_patches + 1 {
                for i in 0..self.config.embed_dim {
                    x_with_cls[[b, p, i]] = x_with_cls[[b, p, i]] + self.pos_embed[[0, p, i]];
                }
            }
        }

        // Apply dropout
        x = self.dropout.forward(&x_with_cls)?;

        // Apply transformer encoder blocks
        for block in &self.encoder_blocks {
            x = block.forward(&x)?;
        }

        // Apply layer normalization
        x = self.norm.forward(&x)?;

        // Use only the class token for classification
        let mut cls_token_final = Array::zeros(IxDyn(&[batch_size, self.config.embed_dim]));
        for b in 0..batch_size {
            for i in 0..self.config.embed_dim {
                cls_token_final[[b, i]] = x[[b, 0, i]];
            }
        }

        // Apply classifier head
        let logits = self.classifier.forward(&cls_token_final)?;

        Ok(logits)
    }

    fn backward(&self, input: &Array<F, IxDyn>, grad_output: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.patch_embed.update(learning_rate)?;

        for block in &mut self.encoder_blocks {
            block.update(learning_rate)?;
        }

        self.norm.update(learning_rate)?;
        self.classifier.update(learning_rate)?;

        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
