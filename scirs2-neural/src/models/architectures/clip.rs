//! CLIP (Contrastive Language-Image Pre-training) Architecture
//!
//! This module implements a CLIP-like architecture as described in
//! "Learning Transferable Visual Models From Natural Language Supervision"
//! (https://arxiv.org/abs/2103.00020)
//!
//! CLIP is a multi-modal model that learns visual concepts from natural language supervision,
//! enabling zero-shot transfer to various visual classification tasks.

use crate::error::{Error, Result};
use crate::layers::{Dense, Layer, LayerNorm, Sequential};
use crate::models::architectures::{ViTConfig, VisionTransformer};
use crate::transformer::TransformerEncoderLayer;
use crate::utils::positional_encoding::{PositionalEncoding, SinusoidalPositionalEncoding};

use ndarray::{Array, Axis, IxDyn, ScalarOperand};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

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
pub struct CLIPTextEncoder<F: Float + Debug + ScalarOperand + Send + Sync + 'static> {
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

impl<F: Float + Debug + ScalarOperand + Send + Sync> CLIPTextEncoder<F> {
    /// Create a new CLIPTextEncoder
    pub fn new(config: CLIPTextConfig, projection_dim: usize) -> Result<Self> {
        // Token embedding
        let mut token_embedding = Sequential::new();
        let mut rng = rand::rng();
        token_embedding.add(Dense::<F>::new(
            config.vocab_size,
            config.hidden_size,
            None,
            &mut rng,
        )?);

        // Position embedding
        let position_embedding = SinusoidalPositionalEncoding::<F>::new(
            config.max_position_embeddings,
            config.hidden_size,
        )?;

        // Transformer encoder layers
        let mut encoder_layers = Vec::with_capacity(config.num_layers);
        for _i in 0..config.num_layers {
            encoder_layers.push(TransformerEncoderLayer::<F>::new(
                config.hidden_size,
                config.num_heads,
                config.intermediate_size,
                config.dropout_rate,
                config.layer_norm_eps,
                &mut rng,
            )?);
        }

        // Layer normalization
        let layer_norm = LayerNorm::<F>::new(config.hidden_size, config.layer_norm_eps, &mut rng)?;

        // Projection
        let mut rng_proj = rand::rng();
        let projection = Dense::<F>::new(config.hidden_size, projection_dim, None, &mut rng_proj)?;

        Ok(Self {
            token_embedding,
            position_embedding,
            encoder_layers,
            layer_norm,
            projection,
            config,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for CLIPTextEncoder<F> {
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
            .slice_axis(Axis(1), ndarray::Slice::from(0..1))
            .into_shape_with_order((batch_size, hidden_size))?;

        // Apply projection - convert to owned array to fix the reference type
        let cls_token_owned = cls_token.to_owned().into_dyn();
        let output = self.projection.forward(&cls_token_owned)?;

        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Err(Error::NotImplemented(
            "CLIPTextEncoder backward not implemented".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Err(Error::NotImplemented(
            "CLIPTextEncoder update not implemented".to_string(),
        ))
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        params.extend(self.token_embedding.params());
        params.extend(self.position_embedding.params());

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

/// Vision encoder for CLIP model (uses Vision Transformer)
#[derive(Debug, Clone)]
pub struct CLIPVisionEncoder<F: Float + Debug + ScalarOperand + Send + Sync + 'static> {
    /// Vision Transformer
    pub vision_transformer: VisionTransformer<F>,
    /// Final projection layer
    pub projection: Dense<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> CLIPVisionEncoder<F> {
    /// Create a new CLIPVisionEncoder
    pub fn new(config: ViTConfig, projection_dim: usize) -> Result<Self> {
        // Create ViT with a clone of the config to avoid ownership issues
        let vision_transformer = VisionTransformer::<F>::new(config.clone())?;

        // Projection layer
        let mut rng_proj = rand::rng();
        let projection = Dense::<F>::new(config.embed_dim, projection_dim, None, &mut rng_proj)?;

        Ok(Self {
            vision_transformer,
            projection,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for CLIPVisionEncoder<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Apply vision transformer
        let x = self.vision_transformer.forward(input)?;

        // Apply projection
        let output = self.projection.forward(&x)?;

        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Err(Error::NotImplemented(
            "CLIPVisionEncoder backward not implemented".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Err(Error::NotImplemented(
            "CLIPVisionEncoder update not implemented".to_string(),
        ))
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut params = Vec::new();
        params.extend(self.vision_transformer.params());
        params.extend(self.projection.params());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.vision_transformer.set_training(training);
        self.projection.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.vision_transformer.is_training()
    }
}

/// CLIP model implementation
#[derive(Debug, Clone)]
pub struct CLIP<F: Float + Debug + ScalarOperand + Send + Sync + 'static> {
    /// Vision encoder
    pub vision_encoder: CLIPVisionEncoder<F>,
    /// Text encoder
    pub text_encoder: CLIPTextEncoder<F>,
    /// Optional classifier for zero-shot classification
    pub classifier: Option<Dense<F>>,
    /// Model configuration
    pub config: CLIPConfig,
    /// Temperature parameter for contrastive loss
    pub logit_scale: F,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> CLIP<F> {
    /// Create a new CLIP model
    pub fn new(config: CLIPConfig) -> Result<Self> {
        // Create vision encoder
        let vision_encoder =
            CLIPVisionEncoder::<F>::new(config.vision_config.clone(), config.projection_dim)?;

        // Create text encoder
        let text_encoder =
            CLIPTextEncoder::<F>::new(config.text_config.clone(), config.projection_dim)?;

        // Create classifier if needed
        let classifier = if config.include_head {
            let mut rng_cls = rand::rng();
            Some(Dense::<F>::new(
                config.projection_dim,
                config.num_classes,
                None,
                &mut rng_cls,
            )?)
        } else {
            None
        };

        // Initialize logit scale (typically ln(1/0.07))
        let logit_scale = F::from(2.6592).unwrap();

        Ok(Self {
            vision_encoder,
            text_encoder,
            classifier,
            config,
            logit_scale,
        })
    }

    /// Forward pass for image-text contrastive learning
    pub fn forward_contrastive(
        &self,
        image_input: &Array<F, IxDyn>,
        text_input: &Array<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>, Array<F, IxDyn>)> {
        // Get image and text embeddings
        let image_features = self.vision_encoder.forward(image_input)?;
        let text_features = self.text_encoder.forward(text_input)?;

        // Normalize embeddings
        let image_features_norm = normalize_features(&image_features)?;
        let text_features_norm = normalize_features(&text_features)?;

        // Compute similarity matrix (batch_size x batch_size)
        let logits_per_image =
            compute_similarity(&image_features_norm, &text_features_norm, self.logit_scale)?;

        // Transpose to get logits_per_text (currently unused but kept for API consistency)
        let _logits_per_text = logits_per_image.t().into_dyn();

        Ok((image_features, text_features, logits_per_image))
    }

    /// Forward pass for zero-shot image classification using a text encoder
    pub fn forward_classification(
        &self,
        image_input: &Array<F, IxDyn>,
        text_embeddings: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Get image embeddings
        let image_features = self.vision_encoder.forward(image_input)?;

        // Normalize image embeddings
        let image_features_norm = normalize_features(&image_features)?;

        // Compute similarity with text embeddings
        let logits = compute_similarity(&image_features_norm, text_embeddings, self.logit_scale)?;

        Ok(logits)
    }

    /// Create a CLIP model with default settings
    pub fn clip_base(num_classes: usize, include_head: bool) -> Result<Self> {
        let vision_config = ViTConfig {
            image_size: (224, 224),
            patch_size: (16, 16),
            in_channels: 3,
            num_classes,
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_dim: 3072,
            dropout_rate: 0.1,
            attention_dropout_rate: 0.1,
        };

        let text_config = CLIPTextConfig {
            vocab_size: 49408,
            hidden_size: 512,
            intermediate_size: 2048,
            num_layers: 12,
            num_heads: 8,
            max_position_embeddings: 77,
            dropout_rate: 0.1,
            layer_norm_eps: 1e-5,
        };

        let config = CLIPConfig {
            text_config,
            vision_config,
            projection_dim: 512,
            include_head,
            num_classes,
        };

        Self::new(config)
    }

    /// Create a small CLIP model
    pub fn clip_small(num_classes: usize, include_head: bool) -> Result<Self> {
        let vision_config = ViTConfig {
            image_size: (224, 224),
            patch_size: (16, 16),
            in_channels: 3,
            num_classes,
            embed_dim: 512,
            num_layers: 8,
            num_heads: 8,
            mlp_dim: 2048,
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

        let config = CLIPConfig {
            text_config,
            vision_config,
            projection_dim: 256,
            include_head,
            num_classes,
        };

        Self::new(config)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for CLIP<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // In a typical scenario, input would be an image
        // For classification tasks, we would need pre-computed text embeddings

        // Get image features
        let image_features = self.vision_encoder.forward(input)?;

        // If classifier is present, use it for direct classification
        if let Some(ref classifier) = self.classifier {
            return classifier.forward(&image_features);
        }

        Ok(image_features)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Err(Error::NotImplemented(
            "CLIP backward not implemented".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Err(Error::NotImplemented(
            "CLIP update not implemented".to_string(),
        ))
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
}

/// Normalize feature vectors (L2 normalization)
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
        if norm > F::from(1e-12).unwrap() {
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
