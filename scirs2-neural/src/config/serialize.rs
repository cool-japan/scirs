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
        serde_yaml::to_string(config)
            .map_err(|e| NeuralError::SerializationError(format!("Failed to serialize to YAML: {}", e)))
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
    /// Create a ResNet configuration
    pub fn resnet(num_layers: usize, num_classes: usize, in_channels: usize) -> ModelConfig {
        use crate::models::architectures::ResNetConfig;

        ModelConfig::ResNet(ResNetConfig {
            num_layers,
            in_channels,
            num_classes,
            zero_init_residual: false,
        })
    }

    /// Create a Vision Transformer configuration
    pub fn vit(image_size: usize, patch_size: usize, num_classes: usize) -> ModelConfig {
        use crate::models::architectures::ViTConfig;

        ModelConfig::ViT(ViTConfig {
            image_size,
            patch_size,
            in_channels: 3,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_dim: 3072,
            dropout_rate: 0.1,
            attention_dropout_rate: 0.0,
            classifier: "token".to_string(),
            num_classes,
            include_top: true,
        })
    }

    /// Create a BERT configuration
    pub fn bert(vocab_size: usize, hidden_size: usize, num_layers: usize) -> ModelConfig {
        use crate::models::architectures::BertConfig;

        ModelConfig::Bert(BertConfig {
            vocab_size,
            hidden_size,
            num_hidden_layers: num_layers,
            num_attention_heads: hidden_size / 64,
            intermediate_size: hidden_size * 4,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
        })
    }

    /// Create a GPT configuration
    pub fn gpt(vocab_size: usize, n_embd: usize, n_layer: usize) -> ModelConfig {
        use crate::models::architectures::GPTConfig;

        ModelConfig::GPT(GPTConfig {
            vocab_size,
            n_ctx: 1024,
            n_embd,
            n_layer,
            n_head: n_embd / 64,
            embd_pdrop: 0.1,
            attn_pdrop: 0.1,
            resid_pdrop: 0.1,
            layer_norm_epsilon: 1e-5,
        })
    }

    /// Create an EfficientNet configuration
    pub fn efficientnet(variant: char, num_classes: usize) -> ModelConfig {
        use crate::models::architectures::EfficientNetConfig;

        // Set width and depth multipliers based on variant
        let (width_multiplier, depth_multiplier) = match variant {
            'B' => (1.0, 1.0), // B0
            '0' => (1.0, 1.0), // B0
            '1' => (1.0, 1.1), // B1
            '2' => (1.1, 1.2), // B2
            '3' => (1.2, 1.4), // B3
            '4' => (1.4, 1.8), // B4
            '5' => (1.6, 2.2), // B5
            '6' => (1.8, 2.6), // B6
            '7' => (2.0, 3.1), // B7
            _ => (1.0, 1.0),   // Default to B0
        };

        ModelConfig::EfficientNet(EfficientNetConfig {
            input_channels: 3,
            width_multiplier,
            depth_multiplier,
            dropout_rate: Some(0.2),
            num_classes,
            include_top: true,
        })
    }

    /// Create a MobileNet configuration
    pub fn mobilenet(version: usize, num_classes: usize) -> ModelConfig {
        use crate::models::architectures::{MobileNetConfig, MobileNetVersion};

        let version_enum = match version {
            1 => MobileNetVersion::V1,
            2 => MobileNetVersion::V2,
            3 => MobileNetVersion::V3Large,
            _ => MobileNetVersion::V2, // Default to V2
        };

        ModelConfig::MobileNet(MobileNetConfig {
            version: version_enum,
            input_channels: 3,
            width_multiplier: 1.0,
            num_classes,
            dropout_rate: Some(0.2),
            include_top: true,
        })
    }

    /// Create a ConvNeXt configuration
    pub fn convnext(variant: &str, num_classes: usize) -> ModelConfig {
        use crate::models::architectures::{ConvNeXtConfig, ConvNeXtVariant};

        let (variant_enum, depths, dims) = match variant.to_lowercase().as_str() {
            "tiny" => (
                ConvNeXtVariant::Tiny,
                vec![3, 3, 9, 3],
                vec![96, 192, 384, 768],
            ),
            "small" => (
                ConvNeXtVariant::Small,
                vec![3, 3, 27, 3],
                vec![96, 192, 384, 768],
            ),
            "base" => (
                ConvNeXtVariant::Base,
                vec![3, 3, 27, 3],
                vec![128, 256, 512, 1024],
            ),
            "large" => (
                ConvNeXtVariant::Large,
                vec![3, 3, 27, 3],
                vec![192, 384, 768, 1536],
            ),
            "xlarge" => (
                ConvNeXtVariant::XLarge,
                vec![3, 3, 27, 3],
                vec![256, 512, 1024, 2048],
            ),
            _ => (
                ConvNeXtVariant::Base,
                vec![3, 3, 27, 3],
                vec![128, 256, 512, 1024],
            ),
        };

        ModelConfig::ConvNeXt(ConvNeXtConfig {
            variant: variant_enum,
            input_channels: 3,
            depths,
            dims,
            num_classes,
            dropout_rate: Some(0.1),
            layer_scale_init_value: 1e-6,
            include_top: true,
        })
    }

    /// Create a CLIP configuration
    pub fn clip(image_size: usize, vocab_size: usize, num_classes: usize) -> ModelConfig {
        use crate::models::architectures::{CLIPConfig, CLIPTextConfig, ViTConfig};

        let vision_config = ViTConfig {
            image_size,
            patch_size: 16,
            in_channels: 3,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_dim: 3072,
            dropout_rate: 0.1,
            attention_dropout_rate: 0.0,
            classifier: "token".to_string(),
            num_classes,
            include_top: false,
        };

        let text_config = CLIPTextConfig {
            vocab_size,
            hidden_size: 512,
            intermediate_size: 2048,
            num_layers: 12,
            num_heads: 8,
            max_position_embeddings: 77,
            dropout_rate: 0.1,
            layer_norm_eps: 1e-5,
        };

        ModelConfig::CLIP(CLIPConfig {
            text_config,
            vision_config,
            projection_dim: 512,
            include_head: true,
            num_classes,
        })
    }

    /// Create a Seq2Seq configuration
    pub fn seq2seq(
        input_vocab_size: usize,
        output_vocab_size: usize,
        hidden_dim: usize,
    ) -> ModelConfig {
        use crate::models::architectures::{RNNCellType, Seq2SeqConfig};

        ModelConfig::Seq2Seq(Seq2SeqConfig {
            input_vocab_size,
            output_vocab_size,
            embedding_dim: hidden_dim,
            hidden_dim,
            num_layers: 2,
            encoder_cell_type: RNNCellType::LSTM,
            decoder_cell_type: RNNCellType::LSTM,
            bidirectional_encoder: true,
            use_attention: true,
            dropout_rate: 0.1,
            max_seq_len: 100,
        })
    }
}
