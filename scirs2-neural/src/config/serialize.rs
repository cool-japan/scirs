//! Configuration serialization and deserialization utilities
//!
//! This module provides utilities for serializing and deserializing model configurations
//! to and from various formats (JSON, YAML, etc.)

use super::ModelConfig;
use crate::error::{NeuralError, Result};

/// Configuration serialization utilities
pub struct ConfigSerializer;

impl ConfigSerializer {
    /// Serialize a model configuration to JSON
    pub fn to_json(config: &ModelConfig, pretty: bool) -> Result<String> {
        if pretty {
            serde_json::to_string_pretty(config).map_err(|e| {
                NeuralError::SerializationError(format!("Failed to serialize to JSON: {}", e))
            })
        } else {
            serde_json::to_string(config).map_err(|e| {
                NeuralError::SerializationError(format!("Failed to serialize to JSON: {}", e))
            })
        }
    }

    /// Serialize a model configuration to YAML
    pub fn to_yaml(config: &ModelConfig) -> Result<String> {
        serde_yaml::to_string(config).map_err(|e| {
            NeuralError::SerializationError(format!("Failed to serialize to YAML: {}", e))
        })
    }

    /// Deserialize a model configuration from JSON
    pub fn from_json(json: &str) -> Result<ModelConfig> {
        serde_json::from_str(json).map_err(|e| {
            NeuralError::DeserializationError(format!("Failed to deserialize from JSON: {}", e))
        })
    }

    /// Deserialize a model configuration from YAML
    pub fn from_yaml(yaml: &str) -> Result<ModelConfig> {
        serde_yaml::from_str(yaml).map_err(|e| {
            NeuralError::DeserializationError(format!("Failed to deserialize from YAML: {}", e))
        })
    }
}

/// Helper for creating model configurations programmatically
pub struct ConfigBuilder;

impl ConfigBuilder {
    /// Create a ResNet-50 configuration
    pub fn resnet50(input_channels: usize, num_classes: usize) -> ModelConfig {
        use crate::models::architectures::ResNetConfig;
        ModelConfig::ResNet(ResNetConfig::resnet50(input_channels, num_classes))
    }

    /// Create a BERT-base-uncased configuration
    pub fn bert_base() -> ModelConfig {
        use crate::models::architectures::BertConfig;
        ModelConfig::Bert(BertConfig::bert_base_uncased())
    }

    /// Create an EfficientNet-B0 configuration
    pub fn efficientnet_b0(input_channels: usize, num_classes: usize) -> ModelConfig {
        use crate::models::architectures::EfficientNetConfig;
        ModelConfig::EfficientNet(EfficientNetConfig::efficientnet_b0(
            input_channels,
            num_classes,
        ))
    }
}
