//! CLIP (Contrastive Language-Image Pre-training) Architecture
//!
//! This module implements a CLIP-like architecture as described in
//! "Learning Transferable Visual Models From Natural Language Supervision"
//! (<https://arxiv.org/abs/2103.00020>)
//! CLIP is a multi-modal model that learns visual concepts from natural language supervision,
//! enabling zero-shot transfer to various visual classification tasks.

use crate::error::Result;
use crate::layers::{Dense, Layer, LayerNorm, Sequential};
use crate::models::architectures::{ViTConfig, VisionTransformer};
use crate::transformer::TransformerEncoderLayer;
use crate::utils::positional_encoding::{PositionalEncoding, SinusoidalPositionalEncoding};
use scirs2_core::ndarray::{Array, Axis, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{rngs::SmallRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
/// Type alias for CLIP output (image embeddings, text embeddings, logit scale)
type ClipOutput<F> = (Array<F, IxDyn>, Array<F, IxDyn>, Array<F, IxDyn>);
/// Configuration for the text encoder in CLIP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CLIPTextConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden size for embeddings and transformer
    pub hidden_size: usize,
    /// Intermediate size in feed-forward layers
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
}
/// Configuration for the CLIP model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CLIPConfig {
    /// Text encoder configuration
    pub text_config: CLIPTextConfig,
    /// Vision encoder configuration
    pub vision_config: ViTConfig,
    /// Projection dimension for both text and vision encoders
    pub projection_dim: usize,
    /// Whether to include the classifier
    pub include_head: bool,
    /// Number of classes for the classifier (if include_head is true)
    pub num_classes: usize,
}

impl Default for CLIPTextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 512,
            intermediate_size: 2048,
            num_layers: 12,
            num_heads: 8,
            max_position_embeddings: 77,
            dropout_rate: 0.1,
            layer_norm_eps: 1e-5,
        }
    }
}

/// Text encoder for CLIP model
#[derive(Debug, Clone)]
pub struct CLIPTextEncoder<
    F: Float
        + Debug
        + ScalarOperand
        + Send
        + Sync
        + 'static
        + scirs2_core::simd_ops::SimdUnifiedOps
        + NumAssign,
> {
    /// Token embedding
    pub token_embedding: Sequential<F>,
    /// Position embedding
    pub position_embedding: SinusoidalPositionalEncoding<F>,
    /// Transformer encoder layers
    pub encoder_layers: Vec<TransformerEncoderLayer<F>>,
    /// Layer normalization
    pub layer_norm: LayerNorm<F>,
    /// Final projection layer
    pub projection: Dense<F>,
    /// Text configuration
    pub config: CLIPTextConfig,
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Send
            + Sync
            + scirs2_core::simd_ops::SimdUnifiedOps
            + NumAssign,
    > CLIPTextEncoder<F>
{
    /// Create a new CLIPTextEncoder
    pub fn new(_config: CLIPTextConfig, projection_dim: usize) -> Result<Self> {
        // Token embedding
        let mut token_embedding = Sequential::new();
        let mut rng = SmallRng::from_seed([42; 32]);
        token_embedding.add(Dense::<F>::new(
            _config.vocab_size,
            _config.hidden_size,
            None,
            &mut rng,
        )?);
        // Position embedding
        let position_embedding = SinusoidalPositionalEncoding::<F>::new(
            _config.hidden_size,
            _config.max_position_embeddings,
        );
        // Transformer encoder layers
        let mut encoder_layers = Vec::with_capacity(_config.num_layers);
        for _i in 0.._config.num_layers {
            encoder_layers.push(TransformerEncoderLayer::<F>::new(
                _config.hidden_size,
                _config.num_heads,
                _config.intermediate_size,
                _config.dropout_rate,
                _config.layer_norm_eps,
                &mut rng,
            )?);
        }
        // Layer normalization
        let layer_norm =
            LayerNorm::<F>::new(_config.hidden_size, _config.layer_norm_eps, &mut rng)?;
        // Projection
        let projection = Dense::<F>::new(_config.hidden_size, projection_dim, None, &mut rng)?;
        Ok(Self {
            token_embedding,
            position_embedding,
            encoder_layers,
            layer_norm,
            projection,
            config: _config,
        })
    }
}
impl<
        F: Float
            + Debug
            + ScalarOperand
            + Send
            + Sync
            + scirs2_core::simd_ops::SimdUnifiedOps
            + 'static
            + NumAssign,
    > Layer<F> for CLIPTextEncoder<F>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Apply token embedding
        let mut x = self.token_embedding.forward(input)?;
        // Apply position embedding
        x = self.position_embedding.forward(&x)?;
        // Apply transformer encoder layers
        for layer in &self.encoder_layers {
            x = layer.forward(&x)?;
        }
        // Apply layer normalization
        x = self.layer_norm.forward(&x)?;
        // Extract the [CLS] token embedding (assuming it's the first token)
        let batch_size = x.shape()[0];
        let hidden_size = x.shape()[2];
        let cls_token = x
            .slice_axis(Axis(1), scirs2_core::ndarray::Slice::from(0..1))
            .into_shape_with_order((batch_size, hidden_size))?;
        // Apply projection - convert to owned array to fix the reference type
        let cls_token_owned = cls_token.to_owned().into_dyn();
        let output = self.projection.forward(&cls_token_owned)?;
        Ok(output)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // CLIPTextEncoder backward: reverse the forward pass
        // Backward through projection
        let grad_after_proj = self.projection.backward(grad_output, grad_output)?;
        // Expand CLS token gradient back to full sequence
        // Note: This is simplified - in reality we need to handle the slicing properly
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let hidden_size = grad_after_proj.shape()[1];
        // Create gradient for full sequence (most gradients go to CLS token position)
        let mut grad_full_seq =
            Array::<F, IxDyn>::zeros(IxDyn(&[batch_size, seq_len, hidden_size]));
        // Put the gradient at the CLS token position (index 0)
        for i in 0..batch_size {
            for j in 0..hidden_size {
                grad_full_seq[[i, 0, j]] = grad_after_proj[[i, j]];
            }
        }
        let grad_full_seq = grad_full_seq.into_dyn();
        // Backward through layer normalization
        let mut grad = self.layer_norm.backward(&grad_full_seq, &grad_full_seq)?;
        // Backward through transformer encoder layers in reverse order
        for layer in self.encoder_layers.iter().rev() {
            grad = layer.backward(&grad, &grad)?;
        }
        // Backward through position embedding.
        // SinusoidalPositionalEncoding is parameter-free; its backward is a pass-through
        // (∂L/∂input = ∂L/∂output since output = input + fixed_encoding).
        grad = self.position_embedding.backward(&grad, &grad)?;
        // Backward through token embedding
        let grad_input = self.token_embedding.backward(input, &grad)?;
        Ok(grad_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update all components
        // Update token embedding
        self.token_embedding.update(learning_rate)?;
        // Update position embedding
        self.position_embedding.update(learning_rate)?;
        // Update all encoder layers
        for layer in &mut self.encoder_layers {
            layer.update(learning_rate)?;
        }
        // Update layer normalization
        self.layer_norm.update(learning_rate)?;
        // Update projection layer
        self.projection.update(learning_rate)?;
        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        params.extend(self.token_embedding.params());
        // position_embedding has no trainable parameters, so we skip it
        for layer in &self.encoder_layers {
            params.extend(layer.params());
        }
        params.extend(self.layer_norm.params());
        params.extend(self.projection.params());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.token_embedding.set_training(training);
        self.position_embedding.set_training(training);
        for layer in &mut self.encoder_layers {
            layer.set_training(training);
        }
        self.layer_norm.set_training(training);
        self.projection.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.token_embedding.is_training()
    }
}
/// Vision encoder for CLIP model — wraps a `VisionTransformer` followed by a linear projection.
pub struct CLIPVisionEncoder<
    F: Float
        + Debug
        + ScalarOperand
        + Send
        + Sync
        + Clone
        + 'static
        + scirs2_core::simd_ops::SimdUnifiedOps
        + NumAssign,
> {
    /// Vision Transformer backbone
    pub vision_transformer: VisionTransformer<F>,
    /// Linear projection into shared embedding space
    pub projection: Dense<F>,
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Send
            + Sync
            + Clone
            + 'static
            + scirs2_core::simd_ops::SimdUnifiedOps
            + NumAssign,
    > CLIPVisionEncoder<F>
{
    /// Create a new CLIPVisionEncoder.
    pub fn new(config: ViTConfig, projection_dim: usize) -> Result<Self> {
        let embed_dim = config.embed_dim;
        let vision_transformer = VisionTransformer::<F>::new(config)?;
        let mut rng_proj = SmallRng::from_seed([42; 32]);
        let projection = Dense::<F>::new(embed_dim, projection_dim, None, &mut rng_proj)?;
        Ok(Self {
            vision_transformer,
            projection,
        })
    }
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Send
            + Sync
            + Clone
            + scirs2_core::simd_ops::SimdUnifiedOps
            + 'static
            + NumAssign,
    > Layer<F> for CLIPVisionEncoder<F>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let x = self.vision_transformer.forward(input)?;
        self.projection.forward(&x)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let grad_after_proj = self.projection.backward(grad_output, grad_output)?;
        self.vision_transformer.backward(input, &grad_after_proj)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.projection.update(learning_rate)?;
        self.vision_transformer.update(learning_rate)?;
        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        params.extend(self.projection.params());
        params.extend(self.vision_transformer.params());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.projection.set_training(training);
        self.vision_transformer.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.vision_transformer.is_training()
    }

    fn layer_type(&self) -> &str {
        "CLIPVisionEncoder"
    }
}

/// CLIP — Contrastive Language-Image Pre-training model.
///
/// Combines a vision encoder (ViT) and a text encoder (Transformer) in a shared
/// embedding space, optimised with a contrastive objective.
pub struct CLIP<
    F: Float
        + Debug
        + ScalarOperand
        + Send
        + Sync
        + Clone
        + 'static
        + scirs2_core::simd_ops::SimdUnifiedOps
        + NumAssign,
> {
    /// Vision encoder
    pub vision_encoder: CLIPVisionEncoder<F>,
    /// Text encoder
    pub text_encoder: CLIPTextEncoder<F>,
    /// Optional classifier for downstream classification
    pub classifier: Option<Dense<F>>,
    /// Model configuration
    pub _config: CLIPConfig,
    /// Log temperature parameter for contrastive loss
    pub logit_scale: F,
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Send
            + Sync
            + Clone
            + 'static
            + scirs2_core::simd_ops::SimdUnifiedOps
            + NumAssign,
    > CLIP<F>
{
    /// Create a new CLIP model from config.
    pub fn new(config: CLIPConfig) -> Result<Self> {
        let vision_encoder =
            CLIPVisionEncoder::<F>::new(config.vision_config.clone(), config.projection_dim)?;
        let text_encoder =
            CLIPTextEncoder::<F>::new(config.text_config.clone(), config.projection_dim)?;
        let classifier = if config.include_head {
            let mut rng_cls = SmallRng::from_seed([42; 32]);
            Some(Dense::<F>::new(
                config.projection_dim,
                config.num_classes,
                None,
                &mut rng_cls,
            )?)
        } else {
            None
        };
        // ln(1/0.07) ≈ 2.6592
        let logit_scale = F::from(2.6592_f64).ok_or_else(|| {
            crate::error::NeuralError::InvalidArchitecture(
                "CLIP: failed to convert logit_scale to float".to_string(),
            )
        })?;
        Ok(Self {
            vision_encoder,
            text_encoder,
            classifier,
            _config: config,
            logit_scale,
        })
    }

    /// Forward pass for image-text contrastive learning.
    /// Returns `(image_embeddings, text_embeddings, similarity_matrix)`.
    pub fn forward_contrastive(
        &self,
        image_input: &Array<F, IxDyn>,
        text_input: &Array<F, IxDyn>,
    ) -> Result<ClipOutput<F>> {
        let image_features = self.vision_encoder.forward(image_input)?;
        let text_features = self.text_encoder.forward(text_input)?;
        let image_features_norm = normalize_features(&image_features)?;
        let text_features_norm = normalize_features(&text_features)?;
        let logits_per_image =
            compute_similarity(&image_features_norm, &text_features_norm, self.logit_scale)?;
        Ok((image_features, text_features, logits_per_image))
    }

    /// Forward pass for zero-shot image classification using pre-computed text embeddings.
    pub fn forward_classification(
        &self,
        image_input: &Array<F, IxDyn>,
        text_embeddings: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let image_features = self.vision_encoder.forward(image_input)?;
        let image_features_norm = normalize_features(&image_features)?;
        compute_similarity(&image_features_norm, text_embeddings, self.logit_scale)
    }

    /// Create a CLIP-Base model (ViT-B/16 + 12-layer text encoder).
    pub fn clip_base(num_classes: usize, include_head: bool) -> Result<Self> {
        let vision_config = ViTConfig {
            image_size: (224, 224),
            patch_size: (16, 16),
            in_channels: 3,
            num_classes,
            embed_dim: 768,
            num_heads: 12,
            mlp_dim: 3072,
            num_layers: 12,
            dropout_rate: 0.1,
            attention_dropout_rate: 0.1,
        };
        Self::new(CLIPConfig {
            text_config: CLIPTextConfig::default(),
            vision_config,
            projection_dim: 512,
            include_head,
            num_classes,
        })
    }

    /// Create a small CLIP model.
    pub fn clip_small(num_classes: usize, include_head: bool) -> Result<Self> {
        let vision_config = ViTConfig {
            image_size: (224, 224),
            patch_size: (16, 16),
            in_channels: 3,
            num_classes,
            embed_dim: 512,
            num_heads: 6,
            mlp_dim: 2048,
            num_layers: 8,
            dropout_rate: 0.1,
            attention_dropout_rate: 0.1,
        };
        let text_config = CLIPTextConfig {
            vocab_size: 49408,
            hidden_size: 384,
            intermediate_size: 1536,
            num_layers: 8,
            num_heads: 6,
            max_position_embeddings: 77,
            dropout_rate: 0.1,
            layer_norm_eps: 1e-5,
        };
        Self::new(CLIPConfig {
            text_config,
            vision_config,
            projection_dim: 256,
            include_head,
            num_classes,
        })
    }
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + Send
            + Sync
            + Clone
            + scirs2_core::simd_ops::SimdUnifiedOps
            + 'static
            + NumAssign,
    > Layer<F> for CLIP<F>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let image_features = self.vision_encoder.forward(input)?;
        if let Some(ref classifier) = self.classifier {
            return classifier.forward(&image_features);
        }
        Ok(image_features)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let mut grad = grad_output.clone();
        if let Some(ref classifier) = self.classifier {
            grad = classifier.backward(&grad, &grad)?;
        }
        self.vision_encoder.backward(input, &grad)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.vision_encoder.update(learning_rate)?;
        self.text_encoder.update(learning_rate)?;
        if let Some(ref mut classifier) = self.classifier {
            classifier.update(learning_rate)?;
        }
        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        params.extend(self.vision_encoder.params());
        params.extend(self.text_encoder.params());
        if let Some(ref classifier) = self.classifier {
            params.extend(classifier.params());
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.vision_encoder.set_training(training);
        self.text_encoder.set_training(training);
        if let Some(ref mut classifier) = self.classifier {
            classifier.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.vision_encoder.is_training()
    }

    fn layer_type(&self) -> &str {
        "CLIP"
    }
}
/// Normalize feature vectors (L2 normalization)
#[allow(dead_code)]
fn normalize_features<F: Float + Debug + ScalarOperand>(
    features: &Array<F, IxDyn>,
) -> Result<Array<F, IxDyn>> {
    let shape = features.shape();
    let batch_size = shape[0];
    let feature_dim = shape[1];
    // Reshape to 2D for easier computation
    let features_2d = features
        .clone()
        .into_shape_with_order((batch_size, feature_dim))?;
    // Compute L2 norm along the feature dimension
    let norm = features_2d.map_axis(Axis(1), |x| {
        let sum_squares = x.iter().fold(F::zero(), |acc, &val| acc + val * val);
        let norm = sum_squares.sqrt();
        // Avoid division by zero
        if norm > F::from(1e-12).expect("Failed to convert constant to float") {
            norm
        } else {
            F::one()
        }
    });
    // Expand norm to match feature dims for broadcasting
    let norm_expanded = norm.insert_axis(Axis(1));
    // Normalize features
    let normalized = features_2d.clone() / norm_expanded;
    // Reshape back to original shape
    Ok(normalized.into_shape_with_order(shape)?)
}

/// Compute similarity matrix between two sets of features
#[allow(dead_code)]
fn compute_similarity<F: Float + Debug + ScalarOperand>(
    features_a: &Array<F, IxDyn>,
    features_b: &Array<F, IxDyn>,
    temperature: F,
) -> Result<Array<F, IxDyn>> {
    // Get shapes
    let shape_a = features_a.shape();
    let shape_b = features_b.shape();
    let batch_a = shape_a[0];
    let batch_b = shape_b[0];
    // Reshape features to 2D matrices
    let features_a_2d = features_a
        .clone()
        .into_shape_with_order((batch_a, shape_a[1]))?;
    let features_b_2d = features_b
        .clone()
        .into_shape_with_order((batch_b, shape_b[1]))?;
    // Compute dot product (similarity matrix)
    let similarity = features_a_2d.dot(&features_b_2d.t());
    // Apply temperature scaling
    let scaled_similarity = similarity * temperature;
    Ok(scaled_similarity.into_dyn())
}
