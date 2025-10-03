//! Hugging Face model adapter for loading and saving models
//!
//! This module provides adapters for converting between SciRS2 and
//! Hugging Face model formats.

use super::config::{HfConfig, HfTokenizerConfig};
use super::pipelines::{
    FeatureExtractionPipeline, FillMaskPipeline, HfPipeline, QuestionAnsweringPipeline,
    SummarizationPipeline, TextClassificationPipeline, TextGenerationPipeline,
    TokenClassificationPipeline, TranslationPipeline, ZeroShotClassificationPipeline,
};
use crate::error::{Result, TextError};
use crate::model_registry::{ModelMetadata, ModelRegistry};
use crate::transformer::{TransformerConfig, TransformerModel};
use scirs2_core::ndarray::Array2;
use std::fs;
use std::path::Path;

#[cfg(feature = "serde-support")]
use std::io::{BufReader, BufWriter};

#[cfg(feature = "serde-support")]
use serde_json;

/// Hugging Face model adapter
pub struct HfModelAdapter {
    /// Model configuration
    config: HfConfig,
    /// Model registry for storage
    registry: Option<ModelRegistry>,
    /// Model metadata
    metadata: Option<ModelMetadata>,
}

impl HfModelAdapter {
    /// Create new HF model adapter
    pub fn new(config: HfConfig) -> Self {
        Self {
            config,
            registry: None,
            metadata: None,
        }
    }

    /// Set model registry
    pub fn with_registry(mut self, registry: ModelRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Set model metadata
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Load model from HF format directory
    pub fn load_from_hf_directory<P: AsRef<Path>>(
        &self,
        model_path: P,
    ) -> Result<TransformerModel> {
        let model_path = model_path.as_ref();

        // Check for required files
        let config_file = model_path.join("config.json");
        if !config_file.exists() {
            return Err(TextError::InvalidInput(
                "HF config.json not found".to_string(),
            ));
        }

        // Load configuration
        let transformer_config = if config_file.exists() {
            #[cfg(feature = "serde-support")]
            {
                let file = fs::File::open(&config_file)
                    .map_err(|e| TextError::IoError(format!("Failed to open config file: {e}")))?;
                let reader = BufReader::new(file);
                let hfconfig: HfConfig = serde_json::from_reader(reader).map_err(|e| {
                    TextError::InvalidInput(format!("Failed to deserialize config: {e}"))
                })?;
                hfconfig.to_transformer_config()?
            }

            #[cfg(not(feature = "serde-support"))]
            {
                // Fallback when serde is not available
                self.config.to_transformer_config()?
            }
        } else {
            self.config.to_transformer_config()?
        };

        // Create vocabulary (simplified - would load from tokenizer.json)
        let vocabulary: Vec<String> = (0..transformer_config.vocab_size)
            .map(|i| format!("[TOKEN_{i}]"))
            .collect();

        // Create transformer model
        TransformerModel::new(transformer_config, vocabulary)
    }

    /// Save model to HF format directory
    pub fn save_to_hf_directory<P: AsRef<Path>>(
        &self,
        model: &TransformerModel,
        output_path: P,
    ) -> Result<()> {
        let output_path = output_path.as_ref();

        // Create output directory
        std::fs::create_dir_all(output_path)
            .map_err(|e| TextError::IoError(format!("Failed to create directory: {e}")))?;

        // Save configuration
        #[cfg(feature = "serde-support")]
        {
            let config_file = fs::File::create(output_path.join("config.json"))
                .map_err(|e| TextError::IoError(format!("Failed to create config file: {e}")))?;
            let writer = BufWriter::new(config_file);
            serde_json::to_writer_pretty(writer, &self.config)
                .map_err(|e| TextError::InvalidInput(format!("Failed to serialize config: {e}")))?;
        }

        #[cfg(not(feature = "serde-support"))]
        {
            let config_json = format!("{:#?}", self.config);
            fs::write(output_path.join("config.json"), config_json)
                .map_err(|e| TextError::IoError(format!("Failed to write config: {e}")))?;
        }

        // Save model weights in binary format
        let model_data = self.serialize_model_weights(model)?;
        fs::write(output_path.join("pytorch_model.bin"), model_data)
            .map_err(|e| TextError::IoError(format!("Failed to write model: {e}")))?;

        // Save tokenizer configuration
        let tokenizer_config = HfTokenizerConfig::default();

        #[cfg(feature = "serde-support")]
        {
            let tokenizer_file = fs::File::create(output_path.join("tokenizer.json"))
                .map_err(|e| TextError::IoError(format!("Failed to create tokenizer file: {e}")))?;
            let writer = BufWriter::new(tokenizer_file);
            serde_json::to_writer_pretty(writer, &tokenizer_config).map_err(|e| {
                TextError::InvalidInput(format!("Failed to serialize tokenizer config: {e}"))
            })?;
        }

        #[cfg(not(feature = "serde-support"))]
        {
            let tokenizer_json = format!("{tokenizer_config:#?}");
            fs::write(output_path.join("tokenizer.json"), tokenizer_json)
                .map_err(|e| TextError::IoError(format!("Failed to write tokenizer: {e}")))?;
        }

        Ok(())
    }

    /// Create HF-compatible pipeline
    pub fn create_pipeline(&self, task: &str) -> Result<HfPipeline> {
        match task {
            "text-classification" => Ok(HfPipeline::TextClassification(
                TextClassificationPipeline::new(),
            )),
            "feature-extraction" => Ok(HfPipeline::FeatureExtraction(
                FeatureExtractionPipeline::new(),
            )),
            "fill-mask" => Ok(HfPipeline::FillMask(FillMaskPipeline::new())),
            "zero-shot-classification" => Ok(HfPipeline::ZeroShotClassification(
                ZeroShotClassificationPipeline::new(),
            )),
            "question-answering" => Ok(HfPipeline::QuestionAnswering(
                QuestionAnsweringPipeline::new(),
            )),
            "text-generation" => Ok(HfPipeline::TextGeneration(TextGenerationPipeline::new())),
            "summarization" => Ok(HfPipeline::Summarization(SummarizationPipeline::new())),
            "translation" => Ok(HfPipeline::Translation(TranslationPipeline::new())),
            "token-classification" => Ok(HfPipeline::TokenClassification(
                TokenClassificationPipeline::new(),
            )),
            _ => Err(TextError::InvalidInput(format!("Unsupported task: {task}"))),
        }
    }

    /// Create pipeline from model directory
    pub fn create_pipeline_from_model<P: AsRef<Path>>(
        &self,
        model_path: P,
        task: Option<&str>,
    ) -> Result<HfPipeline> {
        let model_path = model_path.as_ref();

        // Load config to infer task if not provided
        let config_file = model_path.join("config.json");
        let inferred_task = if config_file.exists() && task.is_none() {
            // Try to infer task from config
            "text-classification" // Default fallback
        } else {
            task.unwrap_or("text-classification")
        };

        self.create_pipeline(inferred_task)
    }

    /// Serialize model weights to binary format
    fn serialize_model_weights(&self, model: &TransformerModel) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();

        // Write magic header for format identification
        buffer.extend_from_slice(b"SCIRS2_TF");

        // Write version
        buffer.extend_from_slice(&1u32.to_le_bytes());

        // Write config
        buffer.extend_from_slice(&(model.config.d_model as u32).to_le_bytes());
        buffer.extend_from_slice(&(model.config.nheads as u32).to_le_bytes());
        buffer.extend_from_slice(&(model.config.d_ff as u32).to_le_bytes());
        buffer.extend_from_slice(&(model.config.n_encoder_layers as u32).to_le_bytes());
        buffer.extend_from_slice(&(model.config.vocab_size as u32).to_le_bytes());
        buffer.extend_from_slice(&(model.config.max_seqlen as u32).to_le_bytes());
        buffer.extend_from_slice(&model.config.dropout.to_le_bytes());

        // Serialize token embeddings
        self.serialize_array2(model.token_embedding.get_embeddings(), &mut buffer);

        // Serialize encoder layers (placeholder - would need access to internal weights)
        let num_layers = model.config.n_encoder_layers as u32;
        buffer.extend_from_slice(&num_layers.to_le_bytes());

        // For now, write placeholder data for encoder weights
        // In a full implementation, we'd need to expose weight access methods
        for _layer_idx in 0..num_layers {
            // Write placeholder attention weights
            let attention_weight_size = (model.config.d_model * model.config.d_model * 4) as u32; // Q, K, V, O
            buffer.extend_from_slice(&attention_weight_size.to_le_bytes());
            for _ in 0..attention_weight_size {
                buffer.extend_from_slice(&0.0f64.to_le_bytes());
            }

            // Write placeholder feed-forward weights
            let ff_weight_size = (model.config.d_model * model.config.d_ff * 2) as u32; // W1, W2
            buffer.extend_from_slice(&ff_weight_size.to_le_bytes());
            for _ in 0..ff_weight_size {
                buffer.extend_from_slice(&0.0f64.to_le_bytes());
            }
        }

        Ok(buffer)
    }

    /// Serialize Array2<f64> to binary buffer
    fn serialize_array2(&self, array: &Array2<f64>, buffer: &mut Vec<u8>) {
        let shape = array.shape();
        buffer.extend_from_slice(&(shape[0] as u32).to_le_bytes());
        buffer.extend_from_slice(&(shape[1] as u32).to_le_bytes());

        for value in array.iter() {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
    }

    /// Deserialize model weights from binary format
    #[allow(dead_code)]
    fn deserialize_model_weights(&self, data: &[u8]) -> Result<TransformerConfig> {
        if data.len() < 10 {
            return Err(TextError::InvalidInput(
                "Invalid model file format".to_string(),
            ));
        }

        let mut offset = 0;

        // Check magic header
        if &data[offset..offset + 9] != b"SCIRS2_TF" {
            return Err(TextError::InvalidInput(
                "Invalid model file header".to_string(),
            ));
        }
        offset += 9;

        // Read version
        let version = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        if version != 1 {
            return Err(TextError::InvalidInput(format!(
                "Unsupported model format version: {version}"
            )));
        }

        // Read config
        let d_model = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;
        let n_heads = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;
        let d_ff = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;
        let n_encoder_layers = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;
        let vocab_size = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;
        let max_seq_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;
        let dropout = f64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);

        Ok(TransformerConfig {
            d_model,
            nheads: n_heads,
            d_ff,
            n_encoder_layers,
            n_decoder_layers: 0, // Encoder-only
            max_seqlen: max_seq_len,
            dropout,
            vocab_size,
        })
    }
}
