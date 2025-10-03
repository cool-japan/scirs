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
impl ModelConfig {
    /// Load a model configuration from a file
    pub fn from_file<P: AsRef<Path>>(path: P, format: Option<ConfigFormat>) -> Result<Self> {
        let _path = path.as_ref();
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
        let mut file = fs::File::open(_path)
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
        // Create directory if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| Error::IOError(format!("Failed to create directory: {}", e)))?;
        // Create file
        let mut file = fs::File::create(path)
            .map_err(|e| Error::IOError(format!("Failed to create config file: {}", e)))?;
        // Serialize based on format
            ConfigFormat::JSON => {
                let content = serde_json::to_string_pretty(self).map_err(|e| {
                    Error::SerializationError(format!("Failed to serialize to JSON: {}", e))
                })?;
                file.write_all(content.as_bytes())
                    .map_err(|e| Error::IOError(format!("Failed to write config file: {}", e)))?;
            ConfigFormat::YAML => {
                let content = serde_yaml::to_string(self).map_err(|e| {
                    Error::SerializationError(format!("Failed to serialize to YAML: {}", e))
        Ok(())
    /// Convert configuration to JSON string
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| Error::SerializationError(format!("Failed to serialize to JSON: {}", e)))
    /// Convert configuration to YAML string
    pub fn to_yaml(&self) -> Result<String> {
        serde_yaml::to_string(self)
            .map_err(|e| Error::SerializationError(format!("Failed to serialize to YAML: {}", e)))
    /// Parse configuration from JSON string
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(_json)
            .map_err(|e| Error::DeserializationError(format!("Failed to parse JSON: {}", e)))
    /// Parse configuration from YAML string
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        serde_yaml::from_str(_yaml)
            .map_err(|e| Error::DeserializationError(format!("Failed to parse YAML: {}", e)))
    /// Validate the configuration against schema and parameter constraints
    pub fn validate(&self) -> Result<()> {
        validation::validate_model_config(self)
    /// Create a model from this configuration
    pub fn create_model<
        F: num_traits:: Float
            + std::fmt::Debug
            + scirs2_core::numeric::NumAssign
            + scirs2_core::ndarray::ScalarOperand
            + Send
            + Sync
            + 'static
            + scirs2_core::simd_ops::SimdUnifiedOps,
    >(
        &self,
    ) -> Result<Box<dyn crate::layers::Layer<F> + Send + Sync>> {
        use crate::models::architectures::*;
        match self {
            ModelConfig::ResNet(config) => {
                let model = ResNet::<F>::new(config.clone())?;
                Ok(Box::new(model)), ModelConfig::ViT(config) => {
                let model = VisionTransformer::<F>::new(config.clone())?;
            ModelConfig::Bert(config) => {
                let model = BertModel::<F>::new(config.clone())?;
            ModelConfig::GPT(config) => {
                let model = GPTModel::<F>::new(config.clone())?;
            ModelConfig::EfficientNet(config) => {
                let model = EfficientNet::<F>::new(config.clone())?;
            ModelConfig::MobileNet(config) => {
                let model = MobileNet::<F>::new(config.clone())?;
            ModelConfig::ConvNeXt(config) => {
                let model = ConvNeXt::<F>::new(config.clone())?;
            ModelConfig::CLIP(config) => {
                let model = CLIP::<F>::new(config.clone())?;
            ModelConfig::FeatureFusion(config) => {
                let model = FeatureFusion::<F>::new(config.clone())?;
            ModelConfig::Seq2Seq(config) => {
                let model = Seq2Seq::<F>::new(config.clone())?;
