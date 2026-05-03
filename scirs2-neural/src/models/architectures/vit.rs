//! Vision Transformer (ViT) implementation
//!
//! Vision Transformer (ViT) is a transformer-based model for image classification
//! that divides an image into fixed-size patches, linearly embeds them, adds position
//! embeddings, and processes them using a standard Transformer encoder.
//! Reference: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", Dosovitskiy et al. (2020)
//! <https://arxiv.org/abs/2010.11929>

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Dropout, Layer, LayerNorm, MultiHeadAttention, PatchEmbedding};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::SeedableRng;
use scirs2_core::simd_ops::SimdUnifiedOps;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Configuration for a Vision Transformer model
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// MLP with GELU activation for transformer blocks
#[derive(Clone, Debug)]
struct TransformerMlp<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> {
    dense1: Dense<F>,
    dense2: Dense<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for TransformerMlp<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut x = self.dense1.forward(input)?;
        // Apply GELU activation inline
        x = x.mapv(|v| {
            // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let x3 = v * v * v;
            v * F::from(0.5).expect("Failed to convert constant to float")
                * (F::one()
                    + (v + F::from(0.044715).expect("Failed to convert constant to float") * x3)
                        .tanh())
        });
        x = self.dense2.forward(&x)?;
        Ok(x)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
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

/// Transformer encoder block for ViT
struct TransformerEncoderBlock<
    F: Float + Debug + ScalarOperand + Clone + Send + Sync + SimdUnifiedOps + NumAssign,
> {
    /// Layer normalization 1
    norm1: LayerNorm<F>,
    /// Multi-head attention
    attention: MultiHeadAttention<F>,
    /// Layer normalization 2
    norm2: LayerNorm<F>,
    /// MLP layers
    mlp: TransformerMlp<F>,
    /// Dropout for attention
    attn_dropout: Dropout<F>,
    /// Dropout for MLP
    mlp_dropout: Dropout<F>,
}

impl<
        F: Float + Debug + ScalarOperand + Clone + Send + Sync + SimdUnifiedOps + NumAssign + 'static,
    > Clone for TransformerEncoderBlock<F>
{
    fn clone(&self) -> Self {
        Self {
            norm1: self.norm1.clone(),
            attention: self.attention.clone(),
            norm2: self.norm2.clone(),
            mlp: self.mlp.clone(),
            attn_dropout: self.attn_dropout.clone(),
            mlp_dropout: self.mlp_dropout.clone(),
        }
    }
}

impl<
        F: Float + Debug + ScalarOperand + Clone + Send + Sync + SimdUnifiedOps + NumAssign + 'static,
    > TransformerEncoderBlock<F>
{
    /// Create a new transformer encoder block
    pub fn new(
        dim: usize,
        num_heads: usize,
        mlp_dim: usize,
        dropout_rate: F,
        attention_dropout_rate: F,
    ) -> Result<Self> {
        // Layer normalization for attention
        let mut ln_rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
        let norm1 = LayerNorm::new(dim, 1e-6, &mut ln_rng)?;

        // Multi-head attention
        let attn_config = crate::layers::AttentionConfig {
            num_heads,
            head_dim: dim / num_heads,
            dropout_prob: attention_dropout_rate.to_f64().expect("Operation failed"),
            causal: false,
            scale: None,
        };
        let mut attn_rng = scirs2_core::random::rngs::SmallRng::from_seed([43; 32]);
        let attention = MultiHeadAttention::new(dim, attn_config, &mut attn_rng)?;

        // Layer normalization for MLP
        let mut ln2_rng = scirs2_core::random::rngs::SmallRng::from_seed([44; 32]);
        let norm2 = LayerNorm::new(dim, 1e-6, &mut ln2_rng)?;

        // MLP
        let mut mlp_rng1 = scirs2_core::random::rngs::SmallRng::from_seed([45; 32]);
        let mut mlp_rng2 = scirs2_core::random::rngs::SmallRng::from_seed([46; 32]);
        let mlp = TransformerMlp {
            dense1: Dense::new(dim, mlp_dim, None, &mut mlp_rng1)?,
            dense2: Dense::new(mlp_dim, dim, None, &mut mlp_rng2)?,
        };

        // Dropouts
        let dropout_rate_f64 = dropout_rate.to_f64().expect("Operation failed");
        let mut dropout_rng1 = scirs2_core::random::rngs::SmallRng::from_seed([47; 32]);
        let mut dropout_rng2 = scirs2_core::random::rngs::SmallRng::from_seed([48; 32]);
        let attn_dropout = Dropout::new(dropout_rate_f64, &mut dropout_rng1)?;
        let mlp_dropout = Dropout::new(dropout_rate_f64, &mut dropout_rng2)?;

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

impl<
        F: Float + Debug + ScalarOperand + Clone + Send + Sync + SimdUnifiedOps + NumAssign + 'static,
    > Layer<F> for TransformerEncoderBlock<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Norm -> Attention -> Dropout -> Add
        let norm1_out = self.norm1.forward(input)?;
        let attn = self.attention.forward(&norm1_out)?;
        let attn_drop = self.attn_dropout.forward(&attn)?;

        // Add residual connection
        let residual1 = input + &attn_drop;

        // Norm -> MLP -> Dropout -> Add
        let norm2_out = self.norm2.forward(&residual1)?;
        let mlp_out = self.mlp.forward(&norm2_out)?;
        let mlp_drop = self.mlp_dropout.forward(&mlp_out)?;

        // Add residual connection
        let residual2 = &residual1 + &mlp_drop;

        Ok(residual2)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
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

/// Vision Transformer (ViT) model.
///
/// Divides images into patches, embeds them, prepends a learnable class token,
/// adds position embeddings, and processes through a stack of Transformer encoder blocks.
/// The class-token output is then fed to a linear classification head.
pub struct VisionTransformer<
    F: Float + Debug + ScalarOperand + Clone + Send + Sync + SimdUnifiedOps + NumAssign,
> {
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

impl<
        F: Float + Debug + ScalarOperand + Clone + Send + Sync + SimdUnifiedOps + NumAssign + 'static,
    > std::fmt::Debug for VisionTransformer<F>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionTransformer")
            .field("patch_embed", &self.patch_embed)
            .field("cls_token", &self.cls_token)
            .field("pos_embed", &self.pos_embed)
            .field("dropout", &self.dropout)
            .field(
                "encoder_blocks",
                &format!("<{} blocks>", self.encoder_blocks.len()),
            )
            .field("norm", &self.norm)
            .field("classifier", &self.classifier)
            .field("config", &self.config)
            .finish()
    }
}

impl<
        F: Float + Debug + ScalarOperand + Clone + Send + Sync + SimdUnifiedOps + NumAssign + 'static,
    > Clone for VisionTransformer<F>
{
    fn clone(&self) -> Self {
        Self {
            patch_embed: self.patch_embed.clone(),
            cls_token: self.cls_token.clone(),
            pos_embed: self.pos_embed.clone(),
            dropout: self.dropout.clone(),
            encoder_blocks: self.encoder_blocks.clone(),
            norm: self.norm.clone(),
            classifier: self.classifier.clone(),
            config: self.config.clone(),
        }
    }
}

impl<
        F: Float + Debug + ScalarOperand + Clone + Send + Sync + SimdUnifiedOps + NumAssign + 'static,
    > VisionTransformer<F>
{
    /// Create a new Vision Transformer model
    pub fn new(config: ViTConfig) -> Result<Self> {
        // Calculate number of patches
        let h_patches = config.image_size.0 / config.patch_size.0;
        let w_patches = config.image_size.1 / config.patch_size.1;
        let num_patches = h_patches * w_patches;

        // Create patch embedding layer
        let mut pe_rng = scirs2_core::random::rngs::SmallRng::from_seed([48; 32]);
        let patch_embed = PatchEmbedding::new(
            config.image_size,
            config.patch_size,
            config.in_channels,
            config.embed_dim,
            true,
            &mut pe_rng,
        )?;

        // Create class token
        let cls_token = Array::zeros(IxDyn(&[1, 1, config.embed_dim]));

        // Create position embedding (include class token)
        let pos_embed = Array::zeros(IxDyn(&[1, num_patches + 1, config.embed_dim]));

        // Create dropout
        let mut dropout_rng = scirs2_core::random::rngs::SmallRng::from_seed([49; 32]);
        let dropout = Dropout::new(config.dropout_rate, &mut dropout_rng)?;

        // Create transformer encoder blocks
        let mut encoder_blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let block = TransformerEncoderBlock::new(
                config.embed_dim,
                config.num_heads,
                config.mlp_dim,
                F::from(config.dropout_rate).ok_or_else(|| {
                    NeuralError::InvalidArchitecture(
                        "Failed to convert dropout_rate to float".to_string(),
                    )
                })?,
                F::from(config.attention_dropout_rate).ok_or_else(|| {
                    NeuralError::InvalidArchitecture(
                        "Failed to convert attention_dropout_rate to float".to_string(),
                    )
                })?,
            )?;
            encoder_blocks.push(block);
            let _ = i;
        }

        // Layer normalization
        let mut norm_rng = scirs2_core::random::rngs::SmallRng::from_seed([50; 32]);
        let norm = LayerNorm::new(config.embed_dim, 1e-6, &mut norm_rng)?;

        // Classification head
        let mut classifier_rng = scirs2_core::random::rngs::SmallRng::from_seed([51; 32]);
        let classifier = Dense::new(
            config.embed_dim,
            config.num_classes,
            None,
            &mut classifier_rng,
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

    /// Get the model configuration
    pub fn config(&self) -> &ViTConfig {
        &self.config
    }
}

impl<
        F: Float + Debug + ScalarOperand + Clone + Send + Sync + SimdUnifiedOps + NumAssign + 'static,
    > Layer<F> for VisionTransformer<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
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

        // Extract patch embeddings: [batch, num_patches, embed_dim]
        let x = self.patch_embed.forward(input)?;

        let h_patches = self.config.image_size.0 / self.config.patch_size.0;
        let w_patches = self.config.image_size.1 / self.config.patch_size.1;
        let num_patches = h_patches * w_patches;

        // Concatenate class token + patch embeddings: [batch, num_patches+1, embed_dim]
        let mut x_with_cls =
            Array::zeros(IxDyn(&[batch_size, num_patches + 1, self.config.embed_dim]));

        // Copy class token (broadcast from [1,1,embed_dim])
        for b in 0..batch_size {
            for i in 0..self.config.embed_dim {
                x_with_cls[[b, 0, i]] = self.cls_token[[0, 0, i]];
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

        // Add position embeddings (broadcast from [1, num_patches+1, embed_dim])
        for b in 0..batch_size {
            for p in 0..num_patches + 1 {
                for i in 0..self.config.embed_dim {
                    x_with_cls[[b, p, i]] += self.pos_embed[[0, p, i]];
                }
            }
        }

        // Dropout → encoder blocks → layer norm
        let mut x = self.dropout.forward(&x_with_cls)?;
        for block in &self.encoder_blocks {
            x = block.forward(&x)?;
        }
        x = self.norm.forward(&x)?;

        // Extract class token: [batch, embed_dim]
        let mut cls_token_final = Array::zeros(IxDyn(&[batch_size, self.config.embed_dim]));
        for b in 0..batch_size {
            for i in 0..self.config.embed_dim {
                cls_token_final[[b, i]] = x[[b, 0, i]];
            }
        }

        // Classification head
        self.classifier.forward(&cls_token_final)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
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

    fn layer_type(&self) -> &str {
        "VisionTransformer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vit_config_base() {
        let config = ViTConfig::vit_base((224, 224), (16, 16), 3, 1000);
        assert_eq!(config.embed_dim, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 12);
    }

    #[test]
    fn test_vit_config_large() {
        let config = ViTConfig::vit_large((224, 224), (16, 16), 3, 1000);
        assert_eq!(config.embed_dim, 1024);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_heads, 16);
    }

    #[test]
    fn test_vit_config_huge() {
        let config = ViTConfig::vit_huge((224, 224), (16, 16), 3, 1000);
        assert_eq!(config.embed_dim, 1280);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 16);
    }
}
