//! Neural network model configuration system
//!
//! This module provides utilities for loading, saving, and validating model
//! configurations using JSON and YAML formats. It enables flexible model
//! creation and reproducibility.

mod schema;
mod serialize;
mod validation;
pub use schema::*;
pub use serialize::*;
pub use validation::*;

use crate::error::{Error, Result};
use crate::models::architectures::{
    BertConfig, CLIPConfig, ConvNeXtConfig, EfficientNetConfig, FeatureFusionConfig, GPTConfig,
    MobileNetConfig, ResNetConfig, Seq2SeqConfig, ViTConfig,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

/// Model configuration container
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "model_type")]
pub enum ModelConfig {
    /// ResNet configuration
    #[serde(rename = "resnet")]
    ResNet(ResNetConfig),
    /// Vision Transformer configuration
    #[serde(rename = "vit")]
    ViT(ViTConfig),
    /// BERT configuration
    #[serde(rename = "bert")]
    Bert(BertConfig),
    /// GPT configuration
    #[serde(rename = "gpt")]
    GPT(GPTConfig),
    /// EfficientNet configuration
    #[serde(rename = "efficientnet")]
    EfficientNet(EfficientNetConfig),
    /// MobileNet configuration
    #[serde(rename = "mobilenet")]
    MobileNet(MobileNetConfig),
    /// ConvNeXt configuration
    #[serde(rename = "convnext")]
    ConvNeXt(ConvNeXtConfig),
    /// CLIP configuration
    #[serde(rename = "clip")]
    CLIP(CLIPConfig),
    /// Feature Fusion configuration
    #[serde(rename = "feature_fusion")]
    FeatureFusion(FeatureFusionConfig),
    /// Seq2Seq configuration
    #[serde(rename = "seq2seq")]
    Seq2Seq(Seq2SeqConfig),
}

/// Format for configuration files
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFormat {
    /// JSON format
    JSON,
    /// YAML format
    YAML,
}

impl ModelConfig {
    /// Load a model configuration from a file
    pub fn from_file<P: AsRef<Path>>(path: P, format: Option<ConfigFormat>) -> Result<Self> {
        let path = path.as_ref();
        // Determine format from extension if not specified
        let format = if let Some(fmt) = format {
            fmt
        } else if let Some(ext) = path.extension() {
            if ext == "json" {
                ConfigFormat::JSON
            } else if ext == "yaml" || ext == "yml" {
                ConfigFormat::YAML
            } else {
                return Err(Error::InvalidArgument(format!(
                    "Unsupported file extension: {:?}. Expected .json, .yaml, or .yml",
                    ext
                )));
            }
        } else {
            return Err(Error::InvalidArgument("File has no extension".to_string()));
        };
        // Read file content
        let mut file = fs::File::open(path)
            .map_err(|e| Error::IOError(format!("Failed to open config file: {}", e)))?;
        let mut content = String::new();
        file.read_to_string(&mut content)
            .map_err(|e| Error::IOError(format!("Failed to read config file: {}", e)))?;
        // Parse based on format
        match format {
            ConfigFormat::JSON => serde_json::from_str(&content)
                .map_err(|e| Error::DeserializationError(format!("Failed to parse JSON: {}", e))),
            ConfigFormat::YAML => serde_yaml::from_str(&content)
                .map_err(|e| Error::DeserializationError(format!("Failed to parse YAML: {}", e))),
        }
    }

    /// Save a model configuration to a file
    pub fn to_file<P: AsRef<Path>>(&self, path: P, format: Option<ConfigFormat>) -> Result<()> {
        let path = path.as_ref();
        // Create directory if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| Error::IOError(format!("Failed to create directory: {}", e)))?;
        }
        // Determine format from extension if not specified
        let format = if let Some(fmt) = format {
            fmt
        } else if let Some(ext) = path.extension() {
            if ext == "json" {
                ConfigFormat::JSON
            } else if ext == "yaml" || ext == "yml" {
                ConfigFormat::YAML
            } else {
                ConfigFormat::JSON
            }
        } else {
            ConfigFormat::JSON
        };
        // Create file
        let mut file = fs::File::create(path)
            .map_err(|e| Error::IOError(format!("Failed to create config file: {}", e)))?;
        // Serialize based on format
        match format {
            ConfigFormat::JSON => {
                let content = serde_json::to_string_pretty(self).map_err(|e| {
                    Error::SerializationError(format!("Failed to serialize to JSON: {}", e))
                })?;
                file.write_all(content.as_bytes())
                    .map_err(|e| Error::IOError(format!("Failed to write config file: {}", e)))?;
            }
            ConfigFormat::YAML => {
                let content = serde_yaml::to_string(self).map_err(|e| {
                    Error::SerializationError(format!("Failed to serialize to YAML: {}", e))
                })?;
                file.write_all(content.as_bytes())
                    .map_err(|e| Error::IOError(format!("Failed to write config file: {}", e)))?;
            }
        }
        Ok(())
    }

    /// Convert configuration to JSON string
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| Error::SerializationError(format!("Failed to serialize to JSON: {}", e)))
    }

    /// Convert configuration to YAML string
    pub fn to_yaml(&self) -> Result<String> {
        serde_yaml::to_string(self)
            .map_err(|e| Error::SerializationError(format!("Failed to serialize to YAML: {}", e)))
    }

    /// Parse configuration from JSON string
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| Error::DeserializationError(format!("Failed to parse JSON: {}", e)))
    }

    /// Parse configuration from YAML string
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        serde_yaml::from_str(yaml)
            .map_err(|e| Error::DeserializationError(format!("Failed to parse YAML: {}", e)))
    }

    /// Validate the configuration against schema and parameter constraints
    pub fn validate(&self) -> Result<()> {
        validation::validate_model_config(self)
    }

    /// Return the model type name as a string
    pub fn model_type(&self) -> &'static str {
        match self {
            ModelConfig::ResNet(_) => "resnet",
            ModelConfig::ViT(_) => "vit",
            ModelConfig::Bert(_) => "bert",
            ModelConfig::GPT(_) => "gpt",
            ModelConfig::EfficientNet(_) => "efficientnet",
            ModelConfig::MobileNet(_) => "mobilenet",
            ModelConfig::ConvNeXt(_) => "convnext",
            ModelConfig::CLIP(_) => "clip",
            ModelConfig::FeatureFusion(_) => "feature_fusion",
            ModelConfig::Seq2Seq(_) => "seq2seq",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::{ResNetBlock, ResNetLayer};

    fn make_resnet_config() -> ModelConfig {
        ModelConfig::ResNet(ResNetConfig {
            block: ResNetBlock::Basic,
            layers: vec![ResNetLayer {
                blocks: 2,
                channels: 64,
                stride: 1,
            }],
            input_channels: 3,
            num_classes: 10,
            dropout_rate: 0.0,
        })
    }

    #[test]
    fn test_config_json_roundtrip() {
        let config = make_resnet_config();
        let json = config.to_json().expect("serialization failed");
        let restored: ModelConfig = ModelConfig::from_json(&json).expect("deserialization failed");
        assert_eq!(restored.model_type(), "resnet");
    }

    #[test]
    fn test_config_yaml_roundtrip() {
        let config = make_resnet_config();
        let yaml = config.to_yaml().expect("yaml serialization failed");
        let restored: ModelConfig =
            ModelConfig::from_yaml(&yaml).expect("yaml deserialization failed");
        assert_eq!(restored.model_type(), "resnet");
    }

    #[test]
    fn test_config_validation_resnet_valid() {
        let config = make_resnet_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_resnet_invalid_channels() {
        let config = ModelConfig::ResNet(ResNetConfig {
            block: ResNetBlock::Basic,
            layers: vec![],
            input_channels: 0, // invalid
            num_classes: 10,
            dropout_rate: 0.0,
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_file_roundtrip() {
        let config = make_resnet_config();
        let tmp = std::env::temp_dir().join("scirs2_test_config.json");
        config
            .to_file(&tmp, Some(ConfigFormat::JSON))
            .expect("write failed");
        let loaded = ModelConfig::from_file(&tmp, Some(ConfigFormat::JSON)).expect("read failed");
        assert_eq!(loaded.model_type(), "resnet");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_model_type_names() {
        assert_eq!(make_resnet_config().model_type(), "resnet");
    }
}
