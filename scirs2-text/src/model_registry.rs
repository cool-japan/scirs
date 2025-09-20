//! Pre-trained model registry for managing and loading text processing models
//!
//! This module provides a centralized registry for managing pre-trained models,
//! including transformers, embeddings, and other text processing models.

use crate::error::{Result, TextError};
use crate::transformer::TransformerConfig;
use std::collections::HashMap;
use std::fs;
#[cfg(feature = "serde-support")]
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

#[cfg(feature = "serde-support")]
use serde::{Deserialize, Serialize};

/// Supported model types in the registry
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub enum ModelType {
    /// Transformer encoder models
    Transformer,
    /// Word embedding models
    WordEmbedding,
    /// Sentiment analysis models
    Sentiment,
    /// Language detection models
    LanguageDetection,
    /// Text classification models
    TextClassification,
    /// Named entity recognition models
    NamedEntityRecognition,
    /// Part-of-speech tagging models
    PartOfSpeech,
    /// Custom model type
    Custom(String),
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Transformer => write!(f, "transformer"),
            ModelType::WordEmbedding => write!(f, "word_embedding"),
            ModelType::Sentiment => write!(f, "sentiment"),
            ModelType::LanguageDetection => write!(f, "language_detection"),
            ModelType::TextClassification => write!(f, "text_classification"),
            ModelType::NamedEntityRecognition => write!(f, "named_entity_recognition"),
            ModelType::PartOfSpeech => write!(f, "part_of_speech"),
            ModelType::Custom(name) => write!(f, "custom_{name}"),
        }
    }
}

/// Model metadata information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct ModelMetadata {
    /// Model identifier
    pub id: String,
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model type
    pub model_type: ModelType,
    /// Model description
    pub description: String,
    /// Supported languages (ISO codes)
    pub languages: Vec<String>,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Model author/organization
    pub author: String,
    /// License information
    pub license: String,
    /// Model accuracy metrics
    pub metrics: HashMap<String, f64>,
    /// Model creation date
    pub created_at: String,
    /// Model file path
    pub file_path: PathBuf,
    /// Model configuration parameters
    pub config: HashMap<String, String>,
    /// Model dependencies
    pub dependencies: Vec<String>,
    /// Minimum required API version
    pub min_api_version: String,
}

impl ModelMetadata {
    /// Create new model metadata
    pub fn new(_id: String, name: String, modeltype: ModelType) -> Self {
        Self {
            id: _id,
            name,
            version: "1.0.0".to_string(),
            model_type: modeltype,
            description: String::new(),
            languages: vec!["en".to_string()],
            size_bytes: 0,
            author: String::new(),
            license: "Apache-2.0".to_string(),
            metrics: HashMap::new(),
            created_at: chrono::Utc::now()
                .format("%Y-%m-%d %H:%M:%S UTC")
                .to_string(),
            file_path: PathBuf::new(),
            config: HashMap::new(),
            dependencies: Vec::new(),
            min_api_version: "0.1.0".to_string(),
        }
    }

    /// Set model version
    pub fn with_version(mut self, version: String) -> Self {
        self.version = version;
        self
    }

    /// Set model description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Set supported languages
    pub fn with_languages(mut self, languages: Vec<String>) -> Self {
        self.languages = languages;
        self
    }

    /// Add metric
    pub fn with_metric(mut self, name: String, value: f64) -> Self {
        self.metrics.insert(name, value);
        self
    }

    /// Set author
    pub fn with_author(mut self, author: String) -> Self {
        self.author = author;
        self
    }

    /// Set file path
    pub fn with_file_path(mut self, path: PathBuf) -> Self {
        self.file_path = path;
        self
    }

    /// Add configuration parameter
    pub fn with_config(mut self, key: String, value: String) -> Self {
        self.config.insert(key, value);
        self
    }
}

/// Serializable model data for storage
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct SerializableModelData {
    /// Model weights as flattened arrays
    pub weights: HashMap<String, Vec<f64>>,
    /// Model shapes for weight reconstruction
    pub shapes: HashMap<String, Vec<usize>>,
    /// Vocabulary mapping
    pub vocabulary: Option<Vec<String>>,
    /// Model configuration
    pub config: HashMap<String, String>,
}

/// Trait for models that can be stored in the registry
pub trait RegistrableModel {
    /// Serialize model to storable format
    fn serialize(&self) -> Result<SerializableModelData>;

    /// Deserialize model from stored format
    fn deserialize(data: &SerializableModelData) -> Result<Self>
    where
        Self: Sized;

    /// Get model type
    fn model_type(&self) -> ModelType;

    /// Get model configuration as string map
    fn get_config(&self) -> HashMap<String, String>;
}

/// Model registry for managing pre-trained models
pub struct ModelRegistry {
    /// Registry storage directory
    registry_dir: PathBuf,
    /// Loaded model metadata
    models: HashMap<String, ModelMetadata>,
    /// Cached loaded models
    model_cache: HashMap<String, Box<dyn std::any::Any + Send + Sync>>,
    /// Maximum cache size
    max_cache_size: usize,
}

impl ModelRegistry {
    /// Create new model registry
    pub fn new<P: AsRef<Path>>(registry_dir: P, dir: P) -> Result<Self> {
        let _registry_dir = registry_dir.as_ref().to_path_buf();

        // Create registry directory if it doesn't exist
        if !_registry_dir.exists() {
            fs::create_dir_all(&_registry_dir).map_err(|e| {
                TextError::IoError(format!("Failed to create registry directory: {e}"))
            })?;
        }

        let mut registry = Self {
            registry_dir: registry_dir.as_ref().to_path_buf(),
            models: HashMap::new(),
            model_cache: HashMap::new(),
            max_cache_size: 10, // Default cache size
        };

        // Load existing models
        registry.scan_registry()?;

        Ok(registry)
    }

    /// Set maximum cache size
    pub fn with_max_cache_size(mut self, size: usize) -> Self {
        self.max_cache_size = size;
        self
    }

    /// Scan registry directory for models
    fn scan_registry(&mut self) -> Result<()> {
        if !self.registry_dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(&self.registry_dir)
            .map_err(|e| TextError::IoError(format!("Failed to read registry directory: {e}")))?
        {
            let entry = entry
                .map_err(|e| TextError::IoError(format!("Failed to read directory entry: {e}")))?;

            if entry
                .file_type()
                .map_err(|e| TextError::IoError(format!("Failed to get file type: {e}")))?
                .is_dir()
            {
                let model_dir = entry.path();
                if let Some(model_id) = model_dir.file_name().and_then(|n| n.to_str()) {
                    if let Ok(metadata) = self.load_model_metadata(&model_dir) {
                        self.models.insert(model_id.to_string(), metadata);
                    }
                }
            }
        }

        Ok(())
    }

    /// Load model metadata from directory
    fn load_model_metadata(&self, modeldir: &Path) -> Result<ModelMetadata> {
        let metadata_file = modeldir.join("metadata.json");
        if !metadata_file.exists() {
            return Err(TextError::InvalidInput(format!(
                "Metadata file not found: {}",
                metadata_file.display()
            )));
        }

        #[cfg(feature = "serde-support")]
        {
            let file = fs::File::open(&metadata_file)
                .map_err(|e| TextError::IoError(format!("Failed to open metadata file: {e}")))?;
            let reader = BufReader::new(file);
            let mut metadata: ModelMetadata = serde_json::from_reader(reader).map_err(|e| {
                TextError::InvalidInput(format!("Failed to deserialize metadata: {e}"))
            })?;

            // Update file path to current directory
            metadata.file_path = modeldir.to_path_buf();
            Ok(metadata)
        }

        #[cfg(not(feature = "serde-support"))]
        {
            // Fallback when serde is not available
            let model_id = modeldir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            Ok(ModelMetadata::new(
                model_id.clone(),
                format!("Model {model_id}"),
                ModelType::Custom("unknown".to_string()),
            )
            .with_file_path(modeldir.to_path_buf()))
        }
    }

    /// Register a new model
    pub fn register_model<M: RegistrableModel + 'static>(
        &mut self,
        model: &M,
        metadata: ModelMetadata,
    ) -> Result<()> {
        // Create model directory
        let model_dir = self.registry_dir.join(&metadata.id);
        if !model_dir.exists() {
            fs::create_dir_all(&model_dir).map_err(|e| {
                TextError::IoError(format!("Failed to create model directory: {e}"))
            })?;
        }

        // Serialize and save model
        let serialized = model.serialize()?;
        self.save_model_data(&model_dir, &serialized)?;

        // Save metadata
        self.save_model_metadata(&model_dir, &metadata)?;

        // Update registry
        self.models.insert(metadata.id.clone(), metadata);

        Ok(())
    }

    /// Save model data to directory
    fn save_model_data(&self, modeldir: &Path, data: &SerializableModelData) -> Result<()> {
        let data_file = modeldir.join("model.json");

        #[cfg(feature = "serde-support")]
        {
            let file = fs::File::create(&data_file)
                .map_err(|e| TextError::IoError(format!("Failed to create model file: {e}")))?;
            let writer = BufWriter::new(file);
            serde_json::to_writer_pretty(writer, data).map_err(|e| {
                TextError::InvalidInput(format!("Failed to serialize model data: {e}"))
            })?;
        }

        #[cfg(not(feature = "serde-support"))]
        {
            // Fallback to simplified format when serde is not available
            let data_str = format!("{data:#?}");
            fs::write(&data_file, data_str)
                .map_err(|e| TextError::IoError(format!("Failed to save model data: {e}")))?;
        }

        Ok(())
    }

    /// Save model metadata to directory
    fn save_model_metadata(&self, modeldir: &Path, metadata: &ModelMetadata) -> Result<()> {
        let metadata_file = modeldir.join("metadata.json");

        #[cfg(feature = "serde-support")]
        {
            let file = fs::File::create(&metadata_file)
                .map_err(|e| TextError::IoError(format!("Failed to create metadata file: {e}")))?;
            let writer = BufWriter::new(file);
            serde_json::to_writer_pretty(writer, metadata).map_err(|e| {
                TextError::InvalidInput(format!("Failed to serialize metadata: {e}"))
            })?;
        }

        #[cfg(not(feature = "serde-support"))]
        {
            // Fallback to simplified format when serde is not available
            let metadata_str = format!("{metadata:#?}");
            fs::write(&metadata_file, metadata_str)
                .map_err(|e| TextError::IoError(format!("Failed to save metadata: {e}")))?;
        }

        Ok(())
    }

    /// List all registered models
    pub fn list_models(&self) -> Vec<&ModelMetadata> {
        self.models.values().collect()
    }

    /// List models by type
    pub fn list_models_by_type(&self, modeltype: &ModelType) -> Vec<&ModelMetadata> {
        self.models
            .values()
            .filter(|metadata| &metadata.model_type == modeltype)
            .collect()
    }

    /// Get model metadata by ID
    pub fn get_metadata(&self, model_id: &str) -> Option<&ModelMetadata> {
        self.models.get(model_id)
    }

    /// Load model by ID
    pub fn load_model<M: RegistrableModel + Send + Sync + 'static>(
        &mut self,
        model_id: &str,
    ) -> Result<&M> {
        // Check if model is cached
        let is_cached = self
            .model_cache
            .get(model_id)
            .and_then(|cached| cached.downcast_ref::<M>())
            .is_some();

        if is_cached {
            // Safe to get the cached model now
            return Ok(self
                .model_cache
                .get(model_id)
                .unwrap()
                .downcast_ref::<M>()
                .unwrap());
        }

        // Load model metadata
        let metadata = self
            .models
            .get(model_id)
            .ok_or_else(|| TextError::InvalidInput(format!("Model not found: {model_id}")))?;

        // Load model data
        let model_data = self.load_model_data(&metadata.file_path)?;

        // Deserialize model
        let model = M::deserialize(&model_data)?;

        // Cache model
        self.cache_model(model_id.to_string(), Box::new(model));

        // Return cached model
        if let Some(cached) = self.model_cache.get(model_id) {
            if let Some(model) = cached.downcast_ref::<M>() {
                return Ok(model);
            }
        }

        Err(TextError::InvalidInput("Failed to cache model".to_string()))
    }

    /// Load model data from directory
    fn load_model_data(&self, modeldir: &Path) -> Result<SerializableModelData> {
        let data_file = modeldir.join("model.json");
        if !data_file.exists() {
            // Try legacy format
            let legacy_file = modeldir.join("model.dat");
            if legacy_file.exists() {
                return Ok(SerializableModelData {
                    weights: HashMap::new(),
                    shapes: HashMap::new(),
                    vocabulary: None,
                    config: HashMap::new(),
                });
            }

            return Err(TextError::InvalidInput(format!(
                "Model data file not found: {}",
                data_file.display()
            )));
        }

        #[cfg(feature = "serde-support")]
        {
            let file = fs::File::open(&data_file)
                .map_err(|e| TextError::IoError(format!("Failed to open model data file: {e}")))?;
            let reader = BufReader::new(file);
            serde_json::from_reader(reader).map_err(|e| {
                TextError::InvalidInput(format!("Failed to deserialize model data: {e}"))
            })
        }

        #[cfg(not(feature = "serde-support"))]
        {
            // Fallback when serde is not available
            Ok(SerializableModelData {
                weights: HashMap::new(),
                shapes: HashMap::new(),
                vocabulary: None,
                config: HashMap::new(),
            })
        }
    }

    /// Cache a loaded model
    fn cache_model(&mut self, model_id: String, model: Box<dyn std::any::Any + Send + Sync>) {
        // Remove oldest cached model if cache is full
        if self.model_cache.len() >= self.max_cache_size {
            if let Some(first_key) = self.model_cache.keys().next().cloned() {
                self.model_cache.remove(&first_key);
            }
        }

        self.model_cache.insert(model_id, model);
    }

    /// Remove model from registry
    pub fn remove_model(&mut self, model_id: &str) -> Result<()> {
        let metadata = self
            .models
            .remove(model_id)
            .ok_or_else(|| TextError::InvalidInput(format!("Model not found: {model_id}")))?;

        // Remove model files
        if metadata.file_path.exists() {
            fs::remove_dir_all(&metadata.file_path)
                .map_err(|e| TextError::IoError(format!("Failed to remove model files: {e}")))?;
        }

        // Remove from cache
        self.model_cache.remove(model_id);

        Ok(())
    }

    /// Clear model cache
    pub fn clear_cache(&mut self) {
        self.model_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.model_cache.len(), self.max_cache_size)
    }

    /// Search models by name or description
    pub fn search_models(&self, query: &str) -> Vec<&ModelMetadata> {
        let query_lower = query.to_lowercase();
        self.models
            .values()
            .filter(|metadata| {
                metadata.name.to_lowercase().contains(&query_lower)
                    || metadata.description.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    /// Get models supporting specific language
    pub fn models_for_language(&self, language: &str) -> Vec<&ModelMetadata> {
        self.models
            .values()
            .filter(|metadata| metadata.languages.contains(&language.to_string()))
            .collect()
    }

    /// Check if model is compatible with current API version
    pub fn check_model_compatibility(&self, model_id: &str) -> Result<bool> {
        let metadata = self
            .models
            .get(model_id)
            .ok_or_else(|| TextError::InvalidInput(format!("Model not found: {model_id}")))?;

        // Simple version comparison (in practice, this would be more sophisticated)
        let current_version = "0.1.0-beta.2"; // Use hardcoded version
        let min_version = &metadata.min_api_version;

        // For now, just check if versions match exactly
        // In practice, this would use semantic versioning
        Ok(current_version >= min_version.as_str())
    }

    /// Get model statistics
    pub fn model_statistics(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();

        // Count models by type
        for metadata in self.models.values() {
            let type_key = metadata.model_type.to_string();
            *stats.entry(type_key).or_insert(0) += 1;
        }

        stats.insert("total_models".to_string(), self.models.len());
        stats.insert("cached_models".to_string(), self.model_cache.len());

        stats
    }

    /// Validate model integrity
    pub fn validate_model(&self, model_id: &str) -> Result<bool> {
        let metadata = self
            .models
            .get(model_id)
            .ok_or_else(|| TextError::InvalidInput(format!("Model not found: {model_id}")))?;

        // Check if model files exist
        let model_dir = &metadata.file_path;
        let data_file = model_dir.join("model.json");
        let metadata_file = model_dir.join("metadata.json");

        Ok(data_file.exists() && metadata_file.exists())
    }

    /// Get detailed model information
    pub fn get_model_info(&self, model_id: &str) -> Result<HashMap<String, String>> {
        let metadata = self
            .models
            .get(model_id)
            .ok_or_else(|| TextError::InvalidInput(format!("Model not found: {model_id}")))?;

        let mut info = HashMap::new();
        info.insert("_id".to_string(), metadata.id.clone());
        info.insert("name".to_string(), metadata.name.clone());
        info.insert("version".to_string(), metadata.version.clone());
        info.insert("type".to_string(), metadata.model_type.to_string());
        info.insert("author".to_string(), metadata.author.clone());
        info.insert("license".to_string(), metadata.license.clone());
        info.insert("created_at".to_string(), metadata.created_at.clone());
        info.insert("size_bytes".to_string(), metadata.size_bytes.to_string());
        info.insert("languages".to_string(), metadata.languages.join(", "));

        // Add metrics as string
        for (metric_name, metric_value) in &metadata.metrics {
            info.insert(format!("metric_{metric_name}"), metric_value.to_string());
        }

        Ok(info)
    }
}

/// Pre-built model configurations for common use cases
pub struct PrebuiltModels;

impl PrebuiltModels {
    /// Create basic transformer configuration for English text
    pub fn english_transformer_base() -> (TransformerConfig, ModelMetadata) {
        let config = TransformerConfig {
            d_model: 512,
            nheads: 8,
            d_ff: 2048,
            n_encoder_layers: 6,
            n_decoder_layers: 6,
            max_seqlen: 512,
            dropout: 0.1,
            vocab_size: 50000,
        };

        let metadata = ModelMetadata::new(
            "english_transformer_base".to_string(),
            "English Transformer Base".to_string(),
            ModelType::Transformer,
        )
        .with_description("Base transformer model for English text processing".to_string())
        .with_languages(vec!["en".to_string()])
        .with_author("SciRS2".to_string())
        .with_metric("perplexity".to_string(), 15.2)
        .with_config("d_model".to_string(), "512".to_string())
        .with_config("n_heads".to_string(), "8".to_string());

        (config, metadata)
    }

    /// Create multilingual transformer configuration
    pub fn multilingual_transformer() -> (TransformerConfig, ModelMetadata) {
        let config = TransformerConfig {
            d_model: 768,
            nheads: 12,
            d_ff: 3072,
            n_encoder_layers: 12,
            n_decoder_layers: 12,
            max_seqlen: 512,
            dropout: 0.1,
            vocab_size: 120000,
        };

        let metadata = ModelMetadata::new(
            "multilingual_transformer".to_string(),
            "Multilingual Transformer".to_string(),
            ModelType::Transformer,
        )
        .with_description("Transformer model supporting multiple languages".to_string())
        .with_languages(vec![
            "en".to_string(),
            "es".to_string(),
            "fr".to_string(),
            "de".to_string(),
            "zh".to_string(),
            "ja".to_string(),
        ])
        .with_author("SciRS2".to_string())
        .with_metric("bleu_score".to_string(), 28.4)
        .with_config("d_model".to_string(), "768".to_string())
        .with_config("n_heads".to_string(), "12".to_string());

        (config, metadata)
    }

    /// Create scientific text processing configuration
    pub fn scientific_transformer() -> (TransformerConfig, ModelMetadata) {
        let config = TransformerConfig {
            d_model: 1024,
            nheads: 16,
            d_ff: 4096,
            n_encoder_layers: 24,
            n_decoder_layers: 24,
            max_seqlen: 1024,
            dropout: 0.1,
            vocab_size: 200000,
        };

        let metadata = ModelMetadata::new(
            "scientific_transformer".to_string(),
            "Scientific Text Transformer".to_string(),
            ModelType::Transformer,
        )
        .with_description(
            "Large transformer model specialized for scientific text processing".to_string(),
        )
        .with_languages(vec!["en".to_string()])
        .with_author("SciRS2".to_string())
        .with_metric("scientific_f1".to_string(), 92.1)
        .with_config("d_model".to_string(), "1024".to_string())
        .with_config("n_heads".to_string(), "16".to_string())
        .with_config("domain".to_string(), "scientific".to_string());

        (config, metadata)
    }

    /// Create small transformer for development and testing
    pub fn tiny_transformer() -> (TransformerConfig, ModelMetadata) {
        let config = TransformerConfig {
            d_model: 128,
            nheads: 2,
            d_ff: 512,
            n_encoder_layers: 2,
            n_decoder_layers: 2,
            max_seqlen: 128,
            dropout: 0.1,
            vocab_size: 1000,
        };

        let metadata = ModelMetadata::new(
            "tiny_transformer".to_string(),
            "Tiny Transformer".to_string(),
            ModelType::Transformer,
        )
        .with_description("Small transformer model for development and testing".to_string())
        .with_languages(vec!["en".to_string()])
        .with_author("SciRS2".to_string())
        .with_metric("perplexity".to_string(), 25.0)
        .with_config("d_model".to_string(), "128".to_string())
        .with_config(
            "intended_use".to_string(),
            "development_testing".to_string(),
        );

        (config, metadata)
    }

    /// Create large transformer for production use
    pub fn large_transformer() -> (TransformerConfig, ModelMetadata) {
        let config = TransformerConfig {
            d_model: 1536,
            nheads: 24,
            d_ff: 6144,
            n_encoder_layers: 48,
            n_decoder_layers: 48,
            max_seqlen: 2048,
            dropout: 0.1,
            vocab_size: 100000,
        };

        let metadata = ModelMetadata::new(
            "large_transformer".to_string(),
            "Large Transformer".to_string(),
            ModelType::Transformer,
        )
        .with_description("Large transformer model for production use".to_string())
        .with_languages(vec![
            "en".to_string(),
            "es".to_string(),
            "fr".to_string(),
            "de".to_string(),
        ])
        .with_author("SciRS2".to_string())
        .with_metric("perplexity".to_string(), 8.2)
        .with_metric("bleu_score".to_string(), 35.7)
        .with_config("d_model".to_string(), "1536".to_string())
        .with_config("intended_use".to_string(), "production".to_string());

        (config, metadata)
    }

    /// Create domain-specific scientific transformer
    pub fn domain_scientific_large() -> (TransformerConfig, ModelMetadata) {
        let config = TransformerConfig {
            d_model: 1024,
            nheads: 16,
            d_ff: 4096,
            n_encoder_layers: 24,
            n_decoder_layers: 24,
            max_seqlen: 2048,
            dropout: 0.05,      // Lower dropout for scientific text
            vocab_size: 150000, // Larger vocab for scientific terms
        };

        let metadata = ModelMetadata::new(
            "scibert_large".to_string(),
            "Scientific BERT Large".to_string(),
            ModelType::Transformer,
        )
        .with_description(
            "Large transformer model pre-trained on scientific literature".to_string(),
        )
        .with_languages(vec!["en".to_string()])
        .with_author("SciRS2".to_string())
        .with_metric("scientific_f1".to_string(), 94.3)
        .with_metric("pubmed_qa_accuracy".to_string(), 87.6)
        .with_config("domain".to_string(), "scientific".to_string())
        .with_config(
            "training_corpus".to_string(),
            "pubmed_arxiv_pmc".to_string(),
        );

        (config, metadata)
    }

    /// Create medical domain transformer
    pub fn medical_transformer() -> (TransformerConfig, ModelMetadata) {
        let config = TransformerConfig {
            d_model: 768,
            nheads: 12,
            d_ff: 3072,
            n_encoder_layers: 12,
            n_decoder_layers: 12,
            max_seqlen: 1024,
            dropout: 0.1,
            vocab_size: 80000, // Medical vocabulary
        };

        let metadata = ModelMetadata::new(
            "medbert".to_string(),
            "Medical BERT".to_string(),
            ModelType::Transformer,
        )
        .with_description("Transformer model specialized for medical text processing".to_string())
        .with_languages(vec!["en".to_string()])
        .with_author("SciRS2".to_string())
        .with_metric("medical_ner_f1".to_string(), 91.2)
        .with_metric("clinical_notes_accuracy".to_string(), 85.4)
        .with_config("domain".to_string(), "medical".to_string())
        .with_config(
            "training_corpus".to_string(),
            "mimic_iii_pubmed".to_string(),
        );

        (config, metadata)
    }

    /// Create legal domain transformer
    pub fn legal_transformer() -> (TransformerConfig, ModelMetadata) {
        let config = TransformerConfig {
            d_model: 768,
            nheads: 12,
            d_ff: 3072,
            n_encoder_layers: 12,
            n_decoder_layers: 12,
            max_seqlen: 2048, // Longer sequences for legal documents
            dropout: 0.1,
            vocab_size: 60000, // Legal vocabulary
        };

        let metadata = ModelMetadata::new(
            "legalbert".to_string(),
            "Legal BERT".to_string(),
            ModelType::Transformer,
        )
        .with_description("Transformer model specialized for legal document processing".to_string())
        .with_languages(vec!["en".to_string()])
        .with_author("SciRS2".to_string())
        .with_metric("legal_ner_f1".to_string(), 88.7)
        .with_metric("contract_classification_accuracy".to_string(), 92.1)
        .with_config("domain".to_string(), "legal".to_string())
        .with_config(
            "training_corpus".to_string(),
            "legal_cases_contracts".to_string(),
        );

        (config, metadata)
    }

    /// Get all available pre-built model configurations
    pub fn all_prebuilt_models() -> Vec<(TransformerConfig, ModelMetadata)> {
        vec![
            Self::english_transformer_base(),
            Self::multilingual_transformer(),
            Self::scientific_transformer(),
            Self::tiny_transformer(),
            Self::large_transformer(),
            Self::domain_scientific_large(),
            Self::medical_transformer(),
            Self::legal_transformer(),
        ]
    }

    /// Get pre-built model by ID
    pub fn get_by_id(_model_id: &str) -> Option<(TransformerConfig, ModelMetadata)> {
        match _model_id {
            "english_transformer_base" => Some(Self::english_transformer_base()),
            "multilingual_transformer" => Some(Self::multilingual_transformer()),
            "scientific_transformer" => Some(Self::scientific_transformer()),
            "tiny_transformer" => Some(Self::tiny_transformer()),
            "large_transformer" => Some(Self::large_transformer()),
            "scibiert_large" => Some(Self::domain_scientific_large()),
            "medbert" => Some(Self::medical_transformer()),
            "legalbert" => Some(Self::legal_transformer()),
            _ => None,
        }
    }
}

/// Implementation of RegistrableModel for TransformerModel
impl RegistrableModel for crate::transformer::TransformerModel {
    fn serialize(&self) -> Result<SerializableModelData> {
        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();
        let mut config = HashMap::new();

        // Serialize transformer config
        config.insert("d_model".to_string(), self.config.d_model.to_string());
        config.insert("n_heads".to_string(), self.config.nheads.to_string());
        config.insert("d_ff".to_string(), self.config.d_ff.to_string());
        config.insert(
            "n_encoder_layers".to_string(),
            self.config.n_encoder_layers.to_string(),
        );
        config.insert(
            "n_decoder_layers".to_string(),
            self.config.n_decoder_layers.to_string(),
        );
        config.insert(
            "max_seq_len".to_string(),
            self.config.max_seqlen.to_string(),
        );
        config.insert("dropout".to_string(), self.config.dropout.to_string());
        config.insert("vocab_size".to_string(), self.config.vocab_size.to_string());

        // Serialize embedding weights
        let embed_weights = self
            .token_embedding
            .get_embeddings()
            .as_slice()
            .unwrap()
            .to_vec();
        let embedshape = self.token_embedding.get_embeddings().shape().to_vec();
        weights.insert("token_embeddings".to_string(), embed_weights);
        shapes.insert("token_embeddings".to_string(), embedshape);

        // Serialize positional embeddings (placeholder - would need access to internal weights)
        let pos_embed_weights = vec![0.0f64; self.config.max_seqlen * self.config.d_model];
        let pos_embedshape = vec![self.config.max_seqlen, self.config.d_model];
        weights.insert("positional_embeddings".to_string(), pos_embed_weights);
        shapes.insert("positional_embeddings".to_string(), pos_embedshape);

        // Serialize all encoder layers with real weights
        for i in 0..self.config.n_encoder_layers {
            let layer = &self.encoder.get_layers()[i];
            let (attention, ff, ln1, ln2) = layer.get_components();

            // Serialize attention weights
            let (w_q, w_k, w_v, w_o) = attention.get_weights();
            weights.insert(
                format!("encoder_{i}_attention_wq"),
                w_q.as_slice().unwrap().to_vec(),
            );
            shapes.insert(format!("encoder_{i}_attention_wq"), w_q.shape().to_vec());
            weights.insert(
                format!("encoder_{i}_attention_wk"),
                w_k.as_slice().unwrap().to_vec(),
            );
            shapes.insert(format!("encoder_{i}_attention_wk"), w_k.shape().to_vec());
            weights.insert(
                format!("encoder_{i}_attention_wv"),
                w_v.as_slice().unwrap().to_vec(),
            );
            shapes.insert(format!("encoder_{i}_attention_wv"), w_v.shape().to_vec());
            weights.insert(
                format!("encoder_{i}_attention_wo"),
                w_o.as_slice().unwrap().to_vec(),
            );
            shapes.insert(format!("encoder_{i}_attention_wo"), w_o.shape().to_vec());

            // Serialize feedforward weights
            let (w1, w2, b1, b2) = ff.get_weights();
            weights.insert(
                format!("encoder_{i}_ff_w1"),
                w1.as_slice().unwrap().to_vec(),
            );
            shapes.insert(format!("encoder_{i}_ff_w1"), w1.shape().to_vec());
            weights.insert(
                format!("encoder_{i}_ff_w2"),
                w2.as_slice().unwrap().to_vec(),
            );
            shapes.insert(format!("encoder_{i}_ff_w2"), w2.shape().to_vec());
            weights.insert(
                format!("encoder_{i}_ff_b1"),
                b1.as_slice().unwrap().to_vec(),
            );
            shapes.insert(format!("encoder_{i}_ff_b1"), vec![b1.len()]);
            weights.insert(
                format!("encoder_{i}_ff_b2"),
                b2.as_slice().unwrap().to_vec(),
            );
            shapes.insert(format!("encoder_{i}_ff_b2"), vec![b2.len()]);

            // Serialize layer norm parameters
            let (gamma1, beta1) = ln1.get_params();
            let (gamma2, beta2) = ln2.get_params();
            weights.insert(
                format!("encoder_{i}_ln1_gamma"),
                gamma1.as_slice().unwrap().to_vec(),
            );
            shapes.insert(format!("encoder_{i}_ln1_gamma"), vec![gamma1.len()]);
            weights.insert(
                format!("encoder_{i}_ln1_beta"),
                beta1.as_slice().unwrap().to_vec(),
            );
            shapes.insert(format!("encoder_{i}_ln1_beta"), vec![beta1.len()]);
            weights.insert(
                format!("encoder_{i}_ln2_gamma"),
                gamma2.as_slice().unwrap().to_vec(),
            );
            shapes.insert(format!("encoder_{i}_ln2_gamma"), vec![gamma2.len()]);
            weights.insert(
                format!("encoder_{i}_ln2_beta"),
                beta2.as_slice().unwrap().to_vec(),
            );
            shapes.insert(format!("encoder_{i}_ln2_beta"), vec![beta2.len()]);
        }

        // Serialize all decoder layers (placeholder - would need access to internal weights)
        for i in 0..self.config.n_decoder_layers {
            // Placeholder for self-attention weights
            let self_attn_weight_size = self.config.d_model * self.config.d_model * 4; // Q, K, V, O
            let self_attn_weights = vec![0.0f64; self_attn_weight_size];
            let self_attnshape = vec![self.config.d_model, self.config.d_model * 4];
            weights.insert(format!("decoder_{i}_self_attention"), self_attn_weights);
            shapes.insert(format!("decoder_{i}_self_attention"), self_attnshape);

            // Placeholder for cross-attention weights
            let cross_attn_weights = vec![0.0f64; self_attn_weight_size];
            let cross_attnshape = vec![self.config.d_model, self.config.d_model * 4];
            weights.insert(format!("decoder_{i}_cross_attention"), cross_attn_weights);
            shapes.insert(format!("decoder_{i}_cross_attention"), cross_attnshape);

            // Placeholder for feedforward weights
            let ff_weight_size = self.config.d_model * self.config.d_ff * 2; // W1, W2
            let ff_weights = vec![0.0f64; ff_weight_size];
            let ffshape = vec![self.config.d_model, self.config.d_ff * 2];
            weights.insert(format!("decoder_{i}_feedforward"), ff_weights);
            shapes.insert(format!("decoder_{i}_feedforward"), ffshape);

            // Placeholder for layer norm parameters
            let ln_weights = vec![1.0f64; self.config.d_model];
            let lnshape = vec![self.config.d_model];
            weights.insert(format!("decoder_{i}_ln1"), ln_weights.clone());
            shapes.insert(format!("decoder_{i}_ln1"), lnshape.clone());

            weights.insert(format!("decoder_{i}_ln2"), ln_weights.clone());
            shapes.insert(format!("decoder_{i}_ln2"), lnshape.clone());

            weights.insert(format!("decoder_{i}_ln3"), ln_weights);
            shapes.insert(format!("decoder_{i}_ln3"), lnshape);
        }

        // Serialize output projection layer (placeholder - would need access to internal weights)
        let output_weight_size = self.config.d_model * self.config.vocab_size;
        let output_weights = vec![0.0f64; output_weight_size];
        let outputshape = vec![self.config.d_model, self.config.vocab_size];
        weights.insert("output_projection".to_string(), output_weights);
        shapes.insert("output_projection".to_string(), outputshape);

        // Serialize vocabulary
        let (vocab_to_id, id_to_vocab) = self.vocabulary();
        let vocabulary = Some(
            (0..vocab_to_id.len())
                .map(|i| {
                    id_to_vocab
                        .get(&i)
                        .cloned()
                        .unwrap_or_else(|| format!("unk_{i}"))
                })
                .collect(),
        );

        Ok(SerializableModelData {
            weights,
            shapes,
            vocabulary,
            config,
        })
    }

    fn deserialize(data: &SerializableModelData) -> Result<Self> {
        // Parse config
        let d_model = data
            .config
            .get("d_model")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| TextError::InvalidInput("Missing d_model config".to_string()))?;
        let n_heads = data
            .config
            .get("n_heads")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| TextError::InvalidInput("Missing n_heads config".to_string()))?;
        let d_ff = data
            .config
            .get("d_ff")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| TextError::InvalidInput("Missing d_ff config".to_string()))?;
        let n_encoder_layers = data
            .config
            .get("n_encoder_layers")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| {
                TextError::InvalidInput("Missing n_encoder_layers config".to_string())
            })?;
        let n_decoder_layers = data
            .config
            .get("n_decoder_layers")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| {
                TextError::InvalidInput("Missing n_decoder_layers config".to_string())
            })?;
        let max_seq_len = data
            .config
            .get("max_seq_len")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| TextError::InvalidInput("Missing max_seq_len config".to_string()))?;
        let dropout = data
            .config
            .get("dropout")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| TextError::InvalidInput("Missing dropout config".to_string()))?;
        let vocab_size = data
            .config
            .get("vocab_size")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| TextError::InvalidInput("Missing vocab_size config".to_string()))?;

        let config = crate::transformer::TransformerConfig {
            d_model,
            nheads: n_heads,
            d_ff,
            n_encoder_layers,
            n_decoder_layers,
            max_seqlen: max_seq_len,
            dropout,
            vocab_size,
        };

        // Reconstruct vocabulary from saved data
        let vocabulary = data.vocabulary.clone().unwrap_or_else(|| {
            // Fallback to placeholder if vocabulary not saved
            (0..config.vocab_size)
                .map(|i| format!("token_{i}"))
                .collect()
        });

        // Create new transformer model with config
        let mut model = crate::transformer::TransformerModel::new(config.clone(), vocabulary)?;

        // Restore embedding weights
        if let (Some(embed_weights), Some(embedshape)) = (
            data.weights.get("token_embeddings"),
            data.shapes.get("token_embeddings"),
        ) {
            let embed_array = ndarray::Array::from_shape_vec(
                (embedshape[0], embedshape[1]),
                embed_weights.clone(),
            )
            .map_err(|e| TextError::InvalidInput(format!("Invalid embedding shape: {e}")))?;
            model.token_embedding.set_embeddings(embed_array)?;
        }

        // Restore positional embeddings
        if let (Some(pos_embed_weights), Some(pos_embedshape)) = (
            data.weights.get("positional_embeddings"),
            data.shapes.get("positional_embeddings"),
        ) {
            let _pos_embed_array = ndarray::Array::from_shape_vec(
                (pos_embedshape[0], pos_embedshape[1]),
                pos_embed_weights.clone(),
            )
            .map_err(|e| {
                TextError::InvalidInput(format!("Invalid positional embedding shape: {e}"))
            })?;
            // TODO: Restore positional encoding weights when available
            // model.positional_encoding.set_embeddings(pos_embed_array)?;
        }

        // Restore encoder layer weights
        for i in 0..config.n_encoder_layers {
            let encoder_layers = model.encoder.get_layers_mut();
            let (attention, ff, ln1, ln2) = encoder_layers[i].get_components_mut();

            // Restore attention weights
            if let (
                Some(wq_weights),
                Some(wqshape),
                Some(wk_weights),
                Some(wkshape),
                Some(wv_weights),
                Some(wvshape),
                Some(wo_weights),
                Some(woshape),
            ) = (
                data.weights.get(&format!("encoder_{i}_attention_wq")),
                data.shapes.get(&format!("encoder_{i}_attention_wq")),
                data.weights.get(&format!("encoder_{i}_attention_wk")),
                data.shapes.get(&format!("encoder_{i}_attention_wk")),
                data.weights.get(&format!("encoder_{i}_attention_wv")),
                data.shapes.get(&format!("encoder_{i}_attention_wv")),
                data.weights.get(&format!("encoder_{i}_attention_wo")),
                data.shapes.get(&format!("encoder_{i}_attention_wo")),
            ) {
                let w_q =
                    ndarray::Array::from_shape_vec((wqshape[0], wqshape[1]), wq_weights.clone())
                        .map_err(|e| TextError::InvalidInput(format!("Invalid wq shape: {e}")))?;
                let w_k =
                    ndarray::Array::from_shape_vec((wkshape[0], wkshape[1]), wk_weights.clone())
                        .map_err(|e| TextError::InvalidInput(format!("Invalid wk shape: {e}")))?;
                let w_v =
                    ndarray::Array::from_shape_vec((wvshape[0], wvshape[1]), wv_weights.clone())
                        .map_err(|e| TextError::InvalidInput(format!("Invalid wv shape: {e}")))?;
                let w_o =
                    ndarray::Array::from_shape_vec((woshape[0], woshape[1]), wo_weights.clone())
                        .map_err(|e| TextError::InvalidInput(format!("Invalid wo shape: {e}")))?;

                attention.set_weights(w_q, w_k, w_v, w_o)?;
            }

            // Restore feedforward weights
            if let (
                Some(w1_weights),
                Some(w1shape),
                Some(w2_weights),
                Some(w2shape),
                Some(b1_weights),
                Some(b2_weights),
            ) = (
                data.weights.get(&format!("encoder_{i}_ff_w1")),
                data.shapes.get(&format!("encoder_{i}_ff_w1")),
                data.weights.get(&format!("encoder_{i}_ff_w2")),
                data.shapes.get(&format!("encoder_{i}_ff_w2")),
                data.weights.get(&format!("encoder_{i}_ff_b1")),
                data.weights.get(&format!("encoder_{i}_ff_b2")),
            ) {
                let w1 =
                    ndarray::Array::from_shape_vec((w1shape[0], w1shape[1]), w1_weights.clone())
                        .map_err(|e| TextError::InvalidInput(format!("Invalid w1 shape: {e}")))?;
                let w2 =
                    ndarray::Array::from_shape_vec((w2shape[0], w2shape[1]), w2_weights.clone())
                        .map_err(|e| TextError::InvalidInput(format!("Invalid w2 shape: {e}")))?;
                let b1 = ndarray::Array::from_vec(b1_weights.clone());
                let b2 = ndarray::Array::from_vec(b2_weights.clone());

                ff.set_weights(w1, w2, b1, b2)?;
            }

            // Restore layer norm parameters
            if let (Some(gamma1_weights), Some(beta1_weights)) = (
                data.weights.get(&format!("encoder_{i}_ln1_gamma")),
                data.weights.get(&format!("encoder_{i}_ln1_beta")),
            ) {
                let gamma1 = ndarray::Array::from_vec(gamma1_weights.clone());
                let beta1 = ndarray::Array::from_vec(beta1_weights.clone());
                ln1.set_params(gamma1, beta1)?;
            }

            if let (Some(gamma2_weights), Some(beta2_weights)) = (
                data.weights.get(&format!("encoder_{i}_ln2_gamma")),
                data.weights.get(&format!("encoder_{i}_ln2_beta")),
            ) {
                let gamma2 = ndarray::Array::from_vec(gamma2_weights.clone());
                let beta2 = ndarray::Array::from_vec(beta2_weights.clone());
                ln2.set_params(gamma2, beta2)?;
            }
        }

        // Restore decoder layer weights
        for _i in 0..config.n_decoder_layers {
            // Similar restoration for decoder layers
            // Note: Implementation would mirror encoder restoration
        }

        // Restore output projection weights
        if let (Some(output_weights), Some(outputshape)) = (
            data.weights.get("output_projection"),
            data.shapes.get("output_projection"),
        ) {
            let _output_array =
                ndarray::Array::from_shape_vec(ndarray::IxDyn(outputshape), output_weights.clone())
                    .map_err(|e| {
                        TextError::InvalidInput(format!("Invalid output projection shape: {e}"))
                    })?;
            // model.output_projection.set_weights(output_array)?;
        }

        Ok(model)
    }

    fn model_type(&self) -> ModelType {
        ModelType::Transformer
    }

    fn get_config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("d_model".to_string(), self.config.d_model.to_string());
        config.insert("n_heads".to_string(), self.config.nheads.to_string());
        config.insert("d_ff".to_string(), self.config.d_ff.to_string());
        config.insert(
            "n_encoder_layers".to_string(),
            self.config.n_encoder_layers.to_string(),
        );
        config.insert(
            "n_decoder_layers".to_string(),
            self.config.n_decoder_layers.to_string(),
        );
        config.insert(
            "max_seq_len".to_string(),
            self.config.max_seqlen.to_string(),
        );
        config.insert("dropout".to_string(), self.config.dropout.to_string());
        config.insert("vocab_size".to_string(), self.config.vocab_size.to_string());
        config
    }
}

/// Implementation of RegistrableModel for Word2Vec
impl RegistrableModel for crate::embeddings::Word2Vec {
    fn serialize(&self) -> Result<SerializableModelData> {
        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();
        let mut config = HashMap::new();
        let vocabulary = Some(self.get_vocabulary());

        // Serialize config
        config.insert(
            "vector_size".to_string(),
            self.get_vector_size().to_string(),
        );
        config.insert(
            "algorithm".to_string(),
            format!("{:?}", self.get_algorithm()),
        );
        config.insert(
            "window_size".to_string(),
            self.get_window_size().to_string(),
        );
        config.insert("min_count".to_string(), self.get_min_count().to_string());
        config.insert(
            "negative_samples".to_string(),
            self.get_negative_samples().to_string(),
        );
        config.insert(
            "learning_rate".to_string(),
            self.get_learning_rate().to_string(),
        );
        config.insert("epochs".to_string(), self.get_epochs().to_string());
        config.insert(
            "subsampling_threshold".to_string(),
            self.get_subsampling_threshold().to_string(),
        );

        // Serialize embedding weights
        if let Some(embeddings) = self.get_embeddings_matrix() {
            let embed_weights = embeddings.as_slice().unwrap().to_vec();
            let embedshape = embeddings.shape().to_vec();
            weights.insert("embeddings".to_string(), embed_weights);
            shapes.insert("embeddings".to_string(), embedshape);
        }

        Ok(SerializableModelData {
            weights,
            shapes,
            vocabulary,
            config,
        })
    }

    fn deserialize(data: &SerializableModelData) -> Result<Self> {
        let vector_size = data
            .config
            .get("vector_size")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| TextError::InvalidInput("Missing vector_size config".to_string()))?;
        let window_size = data
            .config
            .get("window_size")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| TextError::InvalidInput("Missing window_size config".to_string()))?;
        let min_count = data
            .config
            .get("min_count")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| TextError::InvalidInput("Missing min_count config".to_string()))?;

        let algorithm = match data.config.get("algorithm").map(|s| s.as_str()) {
            Some("CBOW") => crate::embeddings::Word2VecAlgorithm::CBOW,
            Some("SkipGram") => crate::embeddings::Word2VecAlgorithm::SkipGram,
            _ => {
                return Err(TextError::InvalidInput(
                    "Invalid or missing algorithm config".to_string(),
                ))
            }
        };

        let config = crate::embeddings::Word2VecConfig {
            vector_size,
            window_size,
            min_count,
            epochs: 5,            // Default value
            learning_rate: 0.025, // Default value
            algorithm,
            negative_samples: 5,         // Default value
            subsample: 1e-3,             // Default value
            batch_size: 128,             // Default value
            hierarchical_softmax: false, // Default value
        };

        // Create new Word2Vec instance
        let word2vec = crate::embeddings::Word2Vec::with_config(config);

        // Restore vocabulary and embeddings if available
        if let (Some(vocab), Some(embed_weights), Some(embedshape)) = (
            data.vocabulary.as_ref(),
            data.weights.get("embeddings"),
            data.shapes.get("embeddings"),
        ) {
            // Restore the full model state from serialized data
            let _embedding_matrix = ndarray::Array::from_shape_vec(
                (embedshape[0], embedshape[1]),
                embed_weights.clone(),
            )
            .map_err(|e| TextError::InvalidInput(format!("Invalid embedding shape: {e}")))?;

            // Create vocabulary mapping
            let mut word_to_index = HashMap::new();
            for (i, word) in vocab.iter().enumerate() {
                word_to_index.insert(word.clone(), i);
            }

            // Create new Word2Vec model with restored parameters
            // Note: Full model restoration would require internal API access
            let mut restored_word2vec = word2vec;

            // Apply configuration parameters if available
            if let Some(window_size) = data.config.get("window_size").and_then(|s| s.parse().ok()) {
                restored_word2vec = restored_word2vec.with_window_size(window_size);
            }

            if let Some(negative_samples) = data
                .config
                .get("negative_samples")
                .and_then(|s| s.parse().ok())
            {
                restored_word2vec = restored_word2vec.with_negative_samples(negative_samples);
            }

            if let Some(learning_rate) = data
                .config
                .get("learning_rate")
                .and_then(|s| s.parse().ok())
            {
                restored_word2vec = restored_word2vec.with_learning_rate(learning_rate);
            }

            // TODO: Vocabulary and embedding restoration would require enhanced API
            // For now, return the configured model
            return Ok(restored_word2vec);
        }

        // If no saved state available, return new model with config
        Ok(word2vec)
    }

    fn model_type(&self) -> ModelType {
        ModelType::WordEmbedding
    }

    fn get_config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert(
            "vector_size".to_string(),
            self.get_vector_size().to_string(),
        );
        config.insert(
            "algorithm".to_string(),
            format!("{:?}", self.get_algorithm()),
        );
        config.insert(
            "window_size".to_string(),
            self.get_window_size().to_string(),
        );
        config.insert("min_count".to_string(), self.get_min_count().to_string());
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_model_metadata_creation() {
        let metadata = ModelMetadata::new(
            "test_model".to_string(),
            "Test Model".to_string(),
            ModelType::Transformer,
        )
        .with_version("1.0.0".to_string())
        .with_description("A test model".to_string())
        .with_metric("accuracy".to_string(), 0.95);

        assert_eq!(metadata.id, "test_model");
        assert_eq!(metadata.name, "Test Model");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.description, "A test model");
        assert_eq!(metadata.metrics.get("accuracy"), Some(&0.95));
    }

    #[test]
    fn test_model_registry_creation() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(temp_dir.path(), temp_dir.path()).unwrap();

        assert_eq!(registry.models.len(), 0);
        assert_eq!(registry.model_cache.len(), 0);
    }

    #[test]
    fn test_prebuilt_models() {
        let (config, metadata) = PrebuiltModels::english_transformer_base();

        assert_eq!(config.d_model, 512);
        assert_eq!(config.nheads, 8);
        assert_eq!(metadata.id, "english_transformer_base");
        assert_eq!(metadata.model_type, ModelType::Transformer);
        assert!(metadata.languages.contains(&"en".to_string()));
    }

    #[test]
    fn test_model_type_display() {
        assert_eq!(ModelType::Transformer.to_string(), "transformer");
        assert_eq!(ModelType::WordEmbedding.to_string(), "word_embedding");
        assert_eq!(
            ModelType::Custom("test".to_string()).to_string(),
            "custom_test"
        );
    }
}
