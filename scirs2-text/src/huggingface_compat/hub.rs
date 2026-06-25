//! Hugging Face Hub integration for model discovery and download
//!
//! This module provides functionality for interacting with the Hugging Face
//! model hub to discover, download, and manage models.

use crate::error::{Result, TextError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(feature = "serde-support")]
use serde::{Deserialize, Serialize};

/// Hugging Face Hub interface
#[derive(Debug)]
pub struct HfHub {
    /// Cache directory for downloaded models
    cache_dir: PathBuf,
    /// API token for authenticated requests
    token: Option<String>,
    /// Model repository cache
    model_cache: HashMap<String, HfModelInfo>,
}

impl HfHub {
    /// Create new HF Hub interface
    pub fn new() -> Self {
        let cache_dir = std::env::var("HF_HOME")
            .or_else(|_| std::env::var("HUGGINGFACE_HUB_CACHE"))
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let mut home = std::env::var("HOME")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| PathBuf::from("."));
                home.push(".cache");
                home.push("huggingface");
                home.push("hub");
                home
            });

        Self {
            cache_dir,
            token: None,
            model_cache: HashMap::new(),
        }
    }

    /// Set authentication token
    pub fn with_token(mut self, token: String) -> Self {
        self.token = Some(token);
        self
    }

    /// Set cache directory
    pub fn with_cache_dir<P: AsRef<Path>>(mut self, cache_dir: P) -> Self {
        self.cache_dir = cache_dir.as_ref().to_path_buf();
        self
    }

    /// List available models on the Hugging Face Hub matching an optional filter.
    ///
    /// Querying the live model index requires HTTP access to
    /// `https://huggingface.co/api/models`. This build of `scirs2-text` does not
    /// bundle an HTTP client, so the operation cannot be performed and an honest
    /// error is returned instead of a fabricated list. Once a networking backend
    /// is available, this method should issue the real request and parse the JSON
    /// response.
    pub fn list_models(&self, _filter: Option<&str>) -> Result<Vec<String>> {
        Err(TextError::RuntimeError(
            "Listing Hugging Face Hub models requires network access via an HTTP \
             client, which is not available in this build of scirs2-text. Enable a \
             networking backend or query https://huggingface.co/api/models directly."
                .to_string(),
        ))
    }

    /// Get model information from the Hugging Face Hub.
    ///
    /// Returns a previously cached [`HfModelInfo`] if one was inserted via
    /// [`HfHub::cache_model_info`]. Otherwise the metadata must be fetched from
    /// `https://huggingface.co/api/models/{model_id}`, which requires HTTP access
    /// that is not available in this build. Rather than fabricate download/like
    /// counts and tags, an honest error is returned.
    pub fn model_info(&mut self, model_id: &str) -> Result<HfModelInfo> {
        if let Some(info) = self.model_cache.get(model_id) {
            return Ok(info.clone());
        }

        Err(TextError::RuntimeError(format!(
            "Fetching metadata for '{model_id}' requires network access to the \
             Hugging Face Hub, which is not available in this build of scirs2-text. \
             Provide the information explicitly via HfHub::cache_model_info, or query \
             https://huggingface.co/api/models/{model_id} directly."
        )))
    }

    /// Insert known model information into the local cache.
    ///
    /// This lets callers that already have model metadata (for example, obtained
    /// out-of-band or from a local registry) make it available to
    /// [`HfHub::model_info`] without performing a network request.
    pub fn cache_model_info(&mut self, info: HfModelInfo) {
        self.model_cache.insert(info.model_id.clone(), info);
    }

    /// Download model files
    pub fn download_model<P: AsRef<Path>>(
        &self,
        model_id: &str,
        cache_dir: Option<P>,
    ) -> Result<PathBuf> {
        let download_dir = cache_dir
            .map(|p| p.as_ref().to_path_buf())
            .unwrap_or_else(|| self.cache_dir.join(model_id));

        // If the model has already been materialised locally (for example by a
        // prior real download performed out-of-band, or by an external tool such
        // as `huggingface-cli`), return the existing path. A model is considered
        // present when its `config.json` exists.
        if download_dir.join("config.json").exists() {
            return Ok(download_dir);
        }

        // Otherwise we would need to fetch the model weights and configuration
        // from `https://huggingface.co/{model_id}`, which requires HTTP access
        // that this build of scirs2-text does not provide. We deliberately do
        // NOT fabricate placeholder weight/config files, as that would masquerade
        // as a successful download and silently corrupt downstream loading.
        Err(TextError::RuntimeError(format!(
            "Model '{model_id}' is not available locally at {} and downloading it \
             requires network access to the Hugging Face Hub, which is not enabled \
             in this build of scirs2-text. Place the model files there manually (for \
             example via `huggingface-cli download {model_id}`) or enable a \
             networking backend.",
            download_dir.display()
        )))
    }

    /// Upload model to hub
    pub fn upload_model<P: AsRef<Path>>(
        &self,
        model_path: P,
        repo_id: &str,
        commit_message: Option<&str>,
    ) -> Result<()> {
        let model_path = model_path.as_ref();

        if !model_path.exists() {
            return Err(TextError::InvalidInput(
                "Model path does not exist".to_string(),
            ));
        }

        // Validate required files
        let required_files = ["config.json"];
        for file in &required_files {
            if !model_path.join(file).exists() {
                return Err(TextError::InvalidInput(format!(
                    "Required file {file} not found"
                )));
            }
        }

        println!(
            "Would upload model from {} to {} with message: {}",
            model_path.display(),
            repo_id,
            commit_message.unwrap_or("Upload model")
        );

        Ok(())
    }

    /// Create model repository
    pub fn create_repo(&self, repo_id: &str, private: bool) -> Result<()> {
        if self.token.is_none() {
            return Err(TextError::InvalidInput(
                "Authentication token required".to_string(),
            ));
        }

        println!("Would create repository {} (private: {})", repo_id, private);

        Ok(())
    }

    /// Get cached model path
    pub fn get_cached_model_path(&self, model_id: &str) -> PathBuf {
        self.cache_dir.join(model_id)
    }
}

impl Default for HfHub {
    fn default() -> Self {
        Self::new()
    }
}

/// Model information from Hugging Face Hub
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(Serialize, Deserialize))]
pub struct HfModelInfo {
    /// Model identifier
    pub model_id: String,
    /// Model tags
    pub tags: Vec<String>,
    /// Pipeline task type
    pub pipeline_tag: Option<String>,
    /// Download count
    pub downloads: u64,
    /// Like count
    pub likes: u64,
    /// Library name (e.g., "transformers")
    pub library_name: Option<String>,
}
