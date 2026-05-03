//! Configuration validation utilities
//!
//! This module provides functions for validating model configurations
//! against schema and parameter constraints.

use super::ModelConfig;
use crate::error::{Error, Result};

/// Validate a model configuration
pub fn validate_model_config(config: &ModelConfig) -> Result<()> {
    match config {
        ModelConfig::ResNet(c) => {
            if c.input_channels == 0 {
                return Err(Error::ValidationError(
                    "ResNet: input_channels must be > 0".to_string(),
                ));
            }
            if c.num_classes == 0 {
                return Err(Error::ValidationError(
                    "ResNet: num_classes must be > 0".to_string(),
                ));
            }
            Ok(())
        }
        ModelConfig::ViT(c) => {
            if c.in_channels == 0 {
                return Err(Error::ValidationError(
                    "ViT: in_channels must be > 0".to_string(),
                ));
            }
            if c.embed_dim % c.num_heads != 0 {
                return Err(Error::ValidationError(format!(
                    "ViT: embed_dim ({}) must be divisible by num_heads ({})",
                    c.embed_dim, c.num_heads
                )));
            }
            if c.dropout_rate < 0.0 || c.dropout_rate > 1.0 {
                return Err(Error::ValidationError(format!(
                    "ViT: dropout_rate ({}) must be in [0, 1]",
                    c.dropout_rate
                )));
            }
            Ok(())
        }
        ModelConfig::Bert(c) => {
            if c.vocab_size == 0 {
                return Err(Error::ValidationError(
                    "Bert: vocab_size must be > 0".to_string(),
                ));
            }
            if c.hidden_size % c.num_attention_heads != 0 {
                return Err(Error::ValidationError(format!(
                    "Bert: hidden_size ({}) must be divisible by num_attention_heads ({})",
                    c.hidden_size, c.num_attention_heads
                )));
            }
            Ok(())
        }
        ModelConfig::GPT(c) => {
            if c.vocab_size == 0 {
                return Err(Error::ValidationError(
                    "GPT: vocab_size must be > 0".to_string(),
                ));
            }
            if c.hidden_size % c.num_attention_heads != 0 {
                return Err(Error::ValidationError(format!(
                    "GPT: hidden_size ({}) must be divisible by num_attention_heads ({})",
                    c.hidden_size, c.num_attention_heads
                )));
            }
            Ok(())
        }
        ModelConfig::EfficientNet(c) => {
            if c.width_coefficient <= 0.0 {
                return Err(Error::ValidationError(format!(
                    "EfficientNet: width_coefficient ({}) must be > 0",
                    c.width_coefficient
                )));
            }
            if c.depth_coefficient <= 0.0 {
                return Err(Error::ValidationError(format!(
                    "EfficientNet: depth_coefficient ({}) must be > 0",
                    c.depth_coefficient
                )));
            }
            Ok(())
        }
        ModelConfig::MobileNet(_c) => Ok(()),
        ModelConfig::ConvNeXt(c) => {
            if c.depths.is_empty() {
                return Err(Error::ValidationError(
                    "ConvNeXt: depths must not be empty".to_string(),
                ));
            }
            if c.depths.len() != c.dims.len() {
                return Err(Error::ValidationError(format!(
                    "ConvNeXt: depths len ({}) must equal dims len ({})",
                    c.depths.len(),
                    c.dims.len()
                )));
            }
            Ok(())
        }
        ModelConfig::CLIP(c) => {
            if c.text_config.vocab_size == 0 {
                return Err(Error::ValidationError(
                    "CLIP: text vocab_size must be > 0".to_string(),
                ));
            }
            if c.projection_dim == 0 {
                return Err(Error::ValidationError(
                    "CLIP: projection_dim must be > 0".to_string(),
                ));
            }
            Ok(())
        }
        ModelConfig::FeatureFusion(c) => {
            if c.input_dims.is_empty() {
                return Err(Error::ValidationError(
                    "FeatureFusion: input_dims must not be empty".to_string(),
                ));
            }
            if c.hidden_dim == 0 {
                return Err(Error::ValidationError(
                    "FeatureFusion: hidden_dim must be > 0".to_string(),
                ));
            }
            Ok(())
        }
        ModelConfig::Seq2Seq(c) => {
            if c.input_vocab_size == 0 {
                return Err(Error::ValidationError(
                    "Seq2Seq: input_vocab_size must be > 0".to_string(),
                ));
            }
            if c.output_vocab_size == 0 {
                return Err(Error::ValidationError(
                    "Seq2Seq: output_vocab_size must be > 0".to_string(),
                ));
            }
            Ok(())
        }
    }
}
