//! Configuration validation utilities
//!
//! This module provides functions for validating model configurations
//! against schema and parameter constraints.

use super::ModelConfig;
use crate::error::{Error, Result};
use crate::models::architectures::*;

/// Validate a model configuration
pub fn validate_model_config(config: &ModelConfig) -> Result<()> {
    match config {
        ModelConfig::ResNet(config) => validate_resnet_config(config),
        ModelConfig::ViT(config) => validate_vit_config(config),
        ModelConfig::Bert(config) => validate_bert_config(config),
        ModelConfig::GPT(config) => validate_gpt_config(config),
        ModelConfig::EfficientNet(config) => validate_efficientnet_config(config),
        ModelConfig::MobileNet(config) => validate_mobilenet_config(config),
        ModelConfig::ConvNeXt(config) => validate_convnext_config(config),
        ModelConfig::CLIP(config) => validate_clip_config(config),
        ModelConfig::FeatureFusion(config) => validate_feature_fusion_config(config),
        ModelConfig::Seq2Seq(config) => validate_seq2seq_config(config),
    }
}

/// Validate a ResNet configuration
fn validate_resnet_config(config: &ResNetConfig) -> Result<()> {
    // Validate number of layers
    if !vec![18, 34, 50, 101, 152].contains(&config.num_layers) {
        return Err(Error::ValidationError(format!(
            "Invalid ResNet depth: {}. Expected one of: 18, 34, 50, 101, 152",
            config.num_layers
        )));
    }

    // Validate input channels
    if config.in_channels == 0 {
        return Err(Error::ValidationError(
            "Invalid input channels: must be greater than 0".to_string(),
        ));
    }

    // Validate number of classes
    if config.num_classes == 0 {
        return Err(Error::ValidationError(
            "Invalid number of classes: must be greater than 0".to_string(),
        ));
    }

    Ok(())
}

/// Validate a Vision Transformer configuration
fn validate_vit_config(config: &ViTConfig) -> Result<()> {
    // Validate image size
    if config.image_size % config.patch_size != 0 {
        return Err(Error::ValidationError(format!(
            "Image size ({}) must be divisible by patch size ({})",
            config.image_size, config.patch_size
        )));
    }

    // Validate patch size
    if config.patch_size == 0 {
        return Err(Error::ValidationError(
            "Invalid patch size: must be greater than 0".to_string(),
        ));
    }

    // Validate hidden size
    if config.hidden_size % config.num_heads != 0 {
        return Err(Error::ValidationError(format!(
            "Hidden size ({}) must be divisible by number of heads ({})",
            config.hidden_size, config.num_heads
        )));
    }

    // Validate dropout rates
    if config.dropout_rate < 0.0 || config.dropout_rate > 1.0 {
        return Err(Error::ValidationError(format!(
            "Invalid dropout rate: {}. Must be between 0.0 and 1.0",
            config.dropout_rate
        )));
    }

    if config.attention_dropout_rate < 0.0 || config.attention_dropout_rate > 1.0 {
        return Err(Error::ValidationError(format!(
            "Invalid attention dropout rate: {}. Must be between 0.0 and 1.0",
            config.attention_dropout_rate
        )));
    }

    // Validate classifier type
    if config.classifier != "token" && config.classifier != "gap" {
        return Err(Error::ValidationError(format!(
            "Invalid classifier type: {}. Expected 'token' or 'gap'",
            config.classifier
        )));
    }

    Ok(())
}

/// Validate a BERT configuration
fn validate_bert_config(config: &BertConfig) -> Result<()> {
    // Validate vocab size
    if config.vocab_size == 0 {
        return Err(Error::ValidationError(
            "Invalid vocabulary size: must be greater than 0".to_string(),
        ));
    }

    // Validate hidden size
    if config.hidden_size % config.num_attention_heads != 0 {
        return Err(Error::ValidationError(format!(
            "Hidden size ({}) must be divisible by number of attention heads ({})",
            config.hidden_size, config.num_attention_heads
        )));
    }

    // Validate intermediate size
    if config.intermediate_size < config.hidden_size {
        return Err(Error::ValidationError(format!(
            "Intermediate size ({}) should not be smaller than hidden size ({})",
            config.intermediate_size, config.hidden_size
        )));
    }

    // Validate max position embeddings
    if config.max_position_embeddings < 64 {
        return Err(Error::ValidationError(format!(
            "Max position embeddings ({}) is too small, should be at least 64",
            config.max_position_embeddings
        )));
    }

    // Validate dropout
    if config.hidden_dropout_prob < 0.0 || config.hidden_dropout_prob > 1.0 {
        return Err(Error::ValidationError(format!(
            "Invalid hidden dropout probability: {}. Must be between 0.0 and 1.0",
            config.hidden_dropout_prob
        )));
    }

    if config.attention_probs_dropout_prob < 0.0 || config.attention_probs_dropout_prob > 1.0 {
        return Err(Error::ValidationError(format!(
            "Invalid attention dropout probability: {}. Must be between 0.0 and 1.0",
            config.attention_probs_dropout_prob
        )));
    }

    Ok(())
}

/// Validate a GPT configuration
fn validate_gpt_config(config: &GPTConfig) -> Result<()> {
    // Validate vocab size
    if config.vocab_size == 0 {
        return Err(Error::ValidationError(
            "Invalid vocabulary size: must be greater than 0".to_string(),
        ));
    }

    // Validate hidden size
    if config.n_embd % config.n_head != 0 {
        return Err(Error::ValidationError(format!(
            "Embedding dimension ({}) must be divisible by number of heads ({})",
            config.n_embd, config.n_head
        )));
    }

    // Validate context length
    if config.n_ctx < 64 {
        return Err(Error::ValidationError(format!(
            "Context length ({}) is too small, should be at least 64",
            config.n_ctx
        )));
    }

    // Validate dropout
    if config.embd_pdrop < 0.0 || config.embd_pdrop > 1.0 {
        return Err(Error::ValidationError(format!(
            "Invalid embedding dropout probability: {}. Must be between 0.0 and 1.0",
            config.embd_pdrop
        )));
    }

    if config.attn_pdrop < 0.0 || config.attn_pdrop > 1.0 {
        return Err(Error::ValidationError(format!(
            "Invalid attention dropout probability: {}. Must be between 0.0 and 1.0",
            config.attn_pdrop
        )));
    }

    if config.resid_pdrop < 0.0 || config.resid_pdrop > 1.0 {
        return Err(Error::ValidationError(format!(
            "Invalid residual dropout probability: {}. Must be between 0.0 and 1.0",
            config.resid_pdrop
        )));
    }

    Ok(())
}

/// Validate an EfficientNet configuration
fn validate_efficientnet_config(config: &EfficientNetConfig) -> Result<()> {
    // Validate width multiplier
    if config.width_multiplier <= 0.0 {
        return Err(Error::ValidationError(format!(
            "Invalid width multiplier: {}. Must be greater than 0.0",
            config.width_multiplier
        )));
    }

    // Validate depth multiplier
    if config.depth_multiplier <= 0.0 {
        return Err(Error::ValidationError(format!(
            "Invalid depth multiplier: {}. Must be greater than 0.0",
            config.depth_multiplier
        )));
    }

    // Validate input channels
    if config.input_channels == 0 {
        return Err(Error::ValidationError(
            "Invalid input channels: must be greater than 0".to_string(),
        ));
    }

    // Validate number of classes
    if config.num_classes == 0 && config.include_top {
        return Err(Error::ValidationError(
            "Invalid number of classes: must be greater than 0 when include_top is true"
                .to_string(),
        ));
    }

    // Validate dropout rate
    if let Some(rate) = config.dropout_rate {
        if rate < 0.0 || rate > 1.0 {
            return Err(Error::ValidationError(format!(
                "Invalid dropout rate: {}. Must be between 0.0 and 1.0",
                rate
            )));
        }
    }

    Ok(())
}

/// Validate a MobileNet configuration
fn validate_mobilenet_config(config: &MobileNetConfig) -> Result<()> {
    // Validate width multiplier
    if config.width_multiplier <= 0.0 {
        return Err(Error::ValidationError(format!(
            "Invalid width multiplier: {}. Must be greater than 0.0",
            config.width_multiplier
        )));
    }

    // Validate input channels
    if config.input_channels == 0 {
        return Err(Error::ValidationError(
            "Invalid input channels: must be greater than 0".to_string(),
        ));
    }

    // Validate number of classes
    if config.num_classes == 0 && config.include_top {
        return Err(Error::ValidationError(
            "Invalid number of classes: must be greater than 0 when include_top is true"
                .to_string(),
        ));
    }

    // Validate dropout rate
    if let Some(rate) = config.dropout_rate {
        if rate < 0.0 || rate > 1.0 {
            return Err(Error::ValidationError(format!(
                "Invalid dropout rate: {}. Must be between 0.0 and 1.0",
                rate
            )));
        }
    }

    Ok(())
}

/// Validate a ConvNeXt configuration
fn validate_convnext_config(config: &ConvNeXtConfig) -> Result<()> {
    // Validate depths
    if config.depths.is_empty() {
        return Err(Error::ValidationError(
            "Invalid depths: must have at least one stage".to_string(),
        ));
    }

    // Validate dims
    if config.dims.is_empty() {
        return Err(Error::ValidationError(
            "Invalid dims: must have at least one stage".to_string(),
        ));
    }

    // Validate depths and dims have same length
    if config.depths.len() != config.dims.len() {
        return Err(Error::ValidationError(format!(
            "Depths array length ({}) must match dims array length ({})",
            config.depths.len(),
            config.dims.len()
        )));
    }

    // Validate input channels
    if config.input_channels == 0 {
        return Err(Error::ValidationError(
            "Invalid input channels: must be greater than 0".to_string(),
        ));
    }

    // Validate number of classes
    if config.num_classes == 0 && config.include_top {
        return Err(Error::ValidationError(
            "Invalid number of classes: must be greater than 0 when include_top is true"
                .to_string(),
        ));
    }

    // Validate dropout rate
    if let Some(rate) = config.dropout_rate {
        if rate < 0.0 || rate > 1.0 {
            return Err(Error::ValidationError(format!(
                "Invalid dropout rate: {}. Must be between 0.0 and 1.0",
                rate
            )));
        }
    }

    // Validate layer scale initialization value
    if config.layer_scale_init_value < 0.0 {
        return Err(Error::ValidationError(format!(
            "Invalid layer scale initialization value: {}. Must be non-negative",
            config.layer_scale_init_value
        )));
    }

    Ok(())
}

/// Validate a CLIP configuration
fn validate_clip_config(config: &CLIPConfig) -> Result<()> {
    // Validate vision config
    validate_vit_config(&config.vision_config)?;

    // Validate text config
    if config.text_config.vocab_size == 0 {
        return Err(Error::ValidationError(
            "Invalid vocabulary size: must be greater than 0".to_string(),
        ));
    }

    if config.text_config.hidden_size % config.text_config.num_heads != 0 {
        return Err(Error::ValidationError(format!(
            "Text hidden size ({}) must be divisible by number of heads ({})",
            config.text_config.hidden_size, config.text_config.num_heads
        )));
    }

    if config.text_config.dropout_rate < 0.0 || config.text_config.dropout_rate > 1.0 {
        return Err(Error::ValidationError(format!(
            "Invalid text dropout rate: {}. Must be between 0.0 and 1.0",
            config.text_config.dropout_rate
        )));
    }

    // Validate projection dimension
    if config.projection_dim == 0 {
        return Err(Error::ValidationError(
            "Invalid projection dimension: must be greater than 0".to_string(),
        ));
    }

    // Validate number of classes
    if config.include_head && config.num_classes == 0 {
        return Err(Error::ValidationError(
            "Invalid number of classes: must be greater than 0 when include_head is true"
                .to_string(),
        ));
    }

    Ok(())
}

/// Validate a Feature Fusion configuration
fn validate_feature_fusion_config(config: &FeatureFusionConfig) -> Result<()> {
    // Validate input dimensions
    if config.input_dims.is_empty() {
        return Err(Error::ValidationError(
            "Invalid input dimensions: must have at least one modality".to_string(),
        ));
    }

    for (i, &dim) in config.input_dims.iter().enumerate() {
        if dim == 0 {
            return Err(Error::ValidationError(format!(
                "Invalid input dimension for modality {}: must be greater than 0",
                i
            )));
        }
    }

    // Validate hidden dimension
    if config.hidden_dim == 0 {
        return Err(Error::ValidationError(
            "Invalid hidden dimension: must be greater than 0".to_string(),
        ));
    }

    // Validate fusion method specific constraints
    match config.fusion_method {
        FusionMethod::Attention | FusionMethod::FiLM | FusionMethod::Bilinear => {
            if config.input_dims.len() != 2 {
                return Err(Error::ValidationError(format!(
                    "{:?} fusion requires exactly 2 modalities, got {}",
                    config.fusion_method,
                    config.input_dims.len()
                )));
            }
        }
        _ => {}
    }

    // Validate dropout rate
    if config.dropout_rate < 0.0 || config.dropout_rate > 1.0 {
        return Err(Error::ValidationError(format!(
            "Invalid dropout rate: {}. Must be between 0.0 and 1.0",
            config.dropout_rate
        )));
    }

    // Validate number of classes
    if config.include_head && config.num_classes == 0 {
        return Err(Error::ValidationError(
            "Invalid number of classes: must be greater than 0 when include_head is true"
                .to_string(),
        ));
    }

    Ok(())
}

/// Validate a Seq2Seq configuration
fn validate_seq2seq_config(config: &Seq2SeqConfig) -> Result<()> {
    // Validate vocabulary sizes
    if config.input_vocab_size == 0 {
        return Err(Error::ValidationError(
            "Invalid input vocabulary size: must be greater than 0".to_string(),
        ));
    }

    if config.output_vocab_size == 0 {
        return Err(Error::ValidationError(
            "Invalid output vocabulary size: must be greater than 0".to_string(),
        ));
    }

    // Validate embedding dimension
    if config.embedding_dim == 0 {
        return Err(Error::ValidationError(
            "Invalid embedding dimension: must be greater than 0".to_string(),
        ));
    }

    // Validate hidden dimension
    if config.hidden_dim == 0 {
        return Err(Error::ValidationError(
            "Invalid hidden dimension: must be greater than 0".to_string(),
        ));
    }

    // Validate number of layers
    if config.num_layers == 0 {
        return Err(Error::ValidationError(
            "Invalid number of layers: must be greater than 0".to_string(),
        ));
    }

    // Validate dropout rate
    if config.dropout_rate < 0.0 || config.dropout_rate > 1.0 {
        return Err(Error::ValidationError(format!(
            "Invalid dropout rate: {}. Must be between 0.0 and 1.0",
            config.dropout_rate
        )));
    }

    // Validate maximum sequence length
    if config.max_seq_len == 0 {
        return Err(Error::ValidationError(
            "Invalid maximum sequence length: must be greater than 0".to_string(),
        ));
    }

    Ok(())
}
