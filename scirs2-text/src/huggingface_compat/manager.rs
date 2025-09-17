//! Model management for Hugging Face compatible models
//!
//! This module provides high-level model management functionality
//! including loading, caching, and lifecycle management.

use super::config::HfConfig;
use super::hub::HfHub;
use crate::error::Result;
use std::collections::HashMap;
use std::path::Path;

/// Hugging Face model manager
#[derive(Debug)]
pub struct HfModelManager {
    /// Model cache
    models: HashMap<String, HfConfig>,
    /// Hub interface
    hub: HfHub,
}

impl HfModelManager {
    /// Create new model manager
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            hub: HfHub::new(),
        }
    }

    /// Load model by ID
    pub fn load_model(&mut self, model_id: &str) -> Result<HfConfig> {
        if let Some(config) = self.models.get(model_id) {
            return Ok(config.clone());
        }

        // Download model if not cached
        let model_path = self.hub.download_model(model_id, None::<&Path>)?;

        // Load configuration
        let config_path = model_path.join("config.json");
        let config = if config_path.exists() {
            #[cfg(feature = "serde-support")]
            {
                use std::fs::File;
                use std::io::BufReader;
                let file = File::open(config_path)?;
                let reader = BufReader::new(file);
                serde_json::from_reader(reader)?
            }

            #[cfg(not(feature = "serde-support"))]
            {
                HfConfig::default()
            }
        } else {
            HfConfig::default()
        };

        self.models.insert(model_id.to_string(), config.clone());
        Ok(config)
    }

    /// Unload model from cache
    pub fn unload_model(&mut self, model_id: &str) -> bool {
        self.models.remove(model_id).is_some()
    }

    /// List cached models
    pub fn list_cached_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// Clear model cache
    pub fn clear_cache(&mut self) {
        self.models.clear();
    }

    /// Get model configuration
    pub fn get_model_config(&self, model_id: &str) -> Option<&HfConfig> {
        self.models.get(model_id)
    }

    /// Set hub token
    pub fn set_hub_token(&mut self, token: String) {
        self.hub = std::mem::take(&mut self.hub).with_token(token);
    }
}

impl Default for HfModelManager {
    fn default() -> Self {
        Self::new()
    }
}
