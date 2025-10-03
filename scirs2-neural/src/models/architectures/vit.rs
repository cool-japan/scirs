//! Vision Transformer (ViT) implementation
//!
//! Vision Transformer (ViT) is a transformer-based model for image classification
//! that divides an image into fixed-size patches, linearly embeds them, adds position
//! embeddings, and processes them using a standard Transformer encoder.
//! Reference: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", Dosovitskiy et al. (2020)
//! https://arxiv.org/abs/2010.11929

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Dropout, Layer, LayerNorm, MultiHeadAttention, PatchEmbedding};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::SeedableRng;
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
            embed_dim: 1024,
            num_layers: 24,
            num_heads: 16,
            mlp_dim: 4096,
    /// Create a ViT-Huge configuration
    pub fn vit_huge(
            embed_dim: 1280,
            num_layers: 32,
            mlp_dim: 5120,
/// MLP with GELU activation for transformer blocks
#[derive(Clone, Debug)]
struct TransformerMlp<F: Float + Debug + ScalarOperand + Send + Sync> {
    dense1: Dense<F>,
    dense2: Dense<F>,
impl<F: Float + Debug + ScalarOperand + Send + Sync> Layer<F> for TransformerMlp<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut x = self.dense1.forward(input)?;
        // Apply GELU activation inline
        x = x.mapv(|v| {
            // GELU approximation: x * 0.5 * (1 + tanh(x + 0.044715 * x^3))
            let x3 = v * v * v;
            v * F::from(0.5).unwrap() * (F::one() + (v + F::from(0.044715).unwrap() * x3).tanh())
        });
        x = self.dense2.forward(&x)?;
        Ok(x)
    fn backward(
        &mut self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    fn update(&mut self, learningrate: F) -> Result<()> {
        self.dense1.update(learning_rate)?;
        self.dense2.update(learning_rate)?;
        Ok(())
    fn as_any(&self) -> &dyn std::any::Any {
        self
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
/// Transformer encoder block for ViT
struct TransformerEncoderBlock<
    F: Float + Debug + ScalarOperand + Clone + Send + Sync + scirs2_core::simd_ops::SimdUnifiedOps,
> {
    /// Layer normalization 1
    norm1: LayerNorm<F>,
    /// Multi-head attention
    attention: MultiHeadAttention<F>,
    /// Layer normalization 2
    norm2: LayerNorm<F>,
    /// MLP layers - now using concrete type for clonability
    mlp: TransformerMlp<F>,
    /// Dropout for attention
    attn_dropout: Dropout<F>,
    /// Dropout for MLP
    mlp_dropout: Dropout<F>,
impl<
        F: Float + Debug + ScalarOperand + Clone + Send + Sync + scirs2_core::simd_ops::SimdUnifiedOps,
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
        // Create config for attention
        let attn_config = crate::layers::AttentionConfig {
            num_heads,
            head_dim: dim / num_heads,
            dropout_prob: attention_dropout_rate.to_f64().unwrap(),
            causal: false,
            scale: None, // Use default scaling
        };
        let mut attn_rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
        let attention = MultiHeadAttention::new(dim, attn_config, &mut attn_rng)?;
        // Layer normalization for MLP
        let norm2 = LayerNorm::new(dim, 1e-6, &mut ln_rng)?;
        let mlp = TransformerMlp {
            dense1: {
                let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
                Dense::new(dim, mlp_dim, None, &mut rng)?
            },
            dense2: {
                Dense::new(mlp_dim, dim, None, &mut rng)?
        // Dropouts
        let dropout_rate_f64 = dropout_rate.to_f64().unwrap();
        let mut dropout_rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
        let attn_dropout = Dropout::new(dropout_rate_f64, &mut dropout_rng)?;
        let mlp_dropout = Dropout::new(dropout_rate_f64, &mut dropout_rng)?;
        Ok(Self {
            norm1,
            attention,
            norm2,
            mlp,
            attn_dropout,
            mlp_dropout,
        })
    > Layer<F> for TransformerEncoderBlock<F>
        // Norm -> Attention -> Dropout -> Add
        let norm1 = self.norm1.forward(input)?;
        let attn = self.attention.forward(&norm1)?;
        let attn_drop = self.attn_dropout.forward(&attn)?;
        // Add residual connection
        let mut residual1 = input.clone();
        for i in 0..residual1.len() {
            residual1[i] = residual1[i] + attn_drop[i];
        // Norm -> MLP -> Dropout -> Add
        let norm2 = self.norm2.forward(&residual1)?;
        let mlp = self.mlp.forward(&norm2)?;
        let mlp_drop = self.mlp_dropout.forward(&mlp)?;
        let mut residual2 = residual1.clone();
        for i in 0..residual2.len() {
            residual2[i] = residual2[i] + mlp_drop[i];
        Ok(residual2)
        self.norm1.update(learning_rate)?;
        self.attention.update(learning_rate)?;
        self.norm2.update(learning_rate)?;
        self.mlp.update(learning_rate)?;
/// Vision Transformer implementation
pub struct VisionTransformer<
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
// Custom Debug implementation for VisionTransformer
    > std::fmt::Debug for VisionTransformer<F>
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
// VisionTransformer can now be cloned since TransformerEncoderBlock is cloneable
    > Clone for VisionTransformer<F>
    fn clone(&self) -> Self {
            patch_embed: self.patch_embed.clone(),
            cls_token: self.cls_token.clone(),
            pos_embed: self.pos_embed.clone(),
            dropout: self.dropout.clone(),
            encoder_blocks: self.encoder_blocks.clone(),
            norm: self.norm.clone(),
            classifier: self.classifier.clone(),
            config: self.config.clone(),
    > VisionTransformer<F>
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
        let dropout_rate = config.dropout_rate; // Use directly as f64
        let dropout = Dropout::new(dropout_rate, &mut dropout_rng)?;
        // Create transformer encoder blocks
        let mut encoder_blocks = Vec::with_capacity(_config.num_layers);
        for _ in 0.._config.num_layers {
            let block = TransformerEncoderBlock::new(
                config.embed_dim,
                config.num_heads,
                config.mlp_dim,
                F::from(_config.dropout_rate).unwrap(),
                F::from(_config.attention_dropout_rate).unwrap(),
            )?;
            encoder_blocks.push(block);
        // Layer normalization
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
        let norm = LayerNorm::new(_config.embed_dim, 1e-6, &mut rng)?;
        // Classification head
        let classifier = Dense::new(
            config.num_classes,
            None, // No activation for final layer
            &mut rng,
            patch_embed,
            cls_token,
            pos_embed,
            dropout,
            encoder_blocks,
            norm,
            classifier,
            config,
    /// Create a ViT-Base model
        let _config = ViTConfig::vit_base(image_size, patch_size, in_channels, num_classes);
        Self::new(_config)
    /// Create a ViT-Large model
        let _config = ViTConfig::vit_large(image_size, patch_size, in_channels, num_classes);
    /// Create a ViT-Huge model
        let _config = ViTConfig::vit_huge(image_size, patch_size, in_channels, num_classes);
    > Layer<F> for VisionTransformer<F>
        // Check input shape
        let shape = input.shape();
        if shape.len() != 4
            || shape[1] != self._config.in_channels
            || shape[2] != self._config.image_size.0
            || shape[3] != self._config.image_size.1
        {
            return Err(NeuralError::InferenceError(format!(
                "Expected input shape [batch_size, {}, {}, {}], got {:?}",
                self.config.in_channels, self.config.image_size.0, self.config.image_size.1, shape
            )));
        let batch_size = shape[0];
        // Extract patch embeddings
        let mut x = self.patch_embed.forward(input)?;
        // Reshape to [batch_size, num_patches, embed_dim]
        let h_patches = self.config.image_size.0 / self.config.patch_size.0;
        let w_patches = self.config.image_size.1 / self.config.patch_size.1;
        // Prepend class token
        let mut cls_tokens = Array::zeros(IxDyn(&[batch_size, 1, self.config.embed_dim]));
        for b in 0..batch_size {
            for i in 0..self.config.embed_dim {
                cls_tokens[[b, 0, i]] = self.cls_token[[0, 0, i]];
            }
        // Concatenate class token with patch embeddings
        let mut x_with_cls =
            Array::zeros(IxDyn(&[batch_size, num_patches + 1, self.config.embed_dim]));
        // Copy class token
                x_with_cls[[b, 0, i]] = cls_tokens[[b, 0, i]];
        // Copy patch embeddings
            for p in 0..num_patches {
                for i in 0..self.config.embed_dim {
                    x_with_cls[[b, p + 1, i]] = x[[b, p, i]];
                }
        // Add position embeddings
            for p in 0..num_patches + 1 {
                    x_with_cls[[b, p, i]] = x_with_cls[[b, p, i]] + self.pos_embed[[0, p, i]];
        // Apply dropout
        x = self.dropout.forward(&x_with_cls)?;
        // Apply transformer encoder blocks
        for block in &self.encoder_blocks {
            x = block.forward(&x)?;
        // Apply layer normalization
        x = self.norm.forward(&x)?;
        // Use only the class token for classification
        let mut cls_token_final = Array::zeros(IxDyn(&[batch_size, self.config.embed_dim]));
                cls_token_final[[b, i]] = x[[b, 0, i]];
        // Apply classifier head
        let logits = self.classifier.forward(&cls_token_final)?;
        Ok(logits)
        self.patch_embed.update(learning_rate)?;
        for block in &mut self.encoder_blocks {
            block.update(learning_rate)?;
        self.norm.update(learning_rate)?;
        self.classifier.update(learning_rate)?;
