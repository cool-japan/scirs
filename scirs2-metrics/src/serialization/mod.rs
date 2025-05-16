//! Metric serialization utilities
//!
//! This module provides tools for saving and loading metric calculations,
//! comparing results between runs, and versioning metric results.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::error::{MetricsError, Result};

// Re-export submodules
pub mod comparison;
pub mod format;

/// Metric result with metadata
///
/// This struct represents a metric result with associated metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    /// Name of the metric
    pub name: String,
    /// Value of the metric
    pub value: f64,
    /// Optional additional values (e.g., for metrics with multiple outputs)
    pub additional_values: Option<HashMap<String, f64>>,
    /// When the metric was calculated
    pub timestamp: DateTime<Utc>,
    /// Optional metadata
    pub metadata: Option<MetricMetadata>,
}

/// Metadata for metric results
///
/// This struct represents metadata associated with metric results.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricMetadata {
    /// ID of the dataset
    pub dataset_id: Option<String>,
    /// ID of the model
    pub model_id: Option<String>,
    /// Parameters used for the metric calculation
    pub parameters: Option<HashMap<String, String>>,
    /// Additional metadata
    pub additional_metadata: Option<HashMap<String, String>>,
}

/// Named metric collection
///
/// This struct represents a collection of metric results with a common name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricCollection {
    /// Name of the collection
    pub name: String,
    /// Description of the collection
    pub description: Option<String>,
    /// Metrics in the collection
    pub metrics: Vec<MetricResult>,
    /// When the collection was created
    pub created_at: DateTime<Utc>,
    /// When the collection was last updated
    pub updated_at: DateTime<Utc>,
    /// Version of the collection
    pub version: String,
}

impl MetricCollection {
    /// Create a new metric collection
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the collection
    /// * `description` - Optional description of the collection
    ///
    /// # Returns
    ///
    /// * A new MetricCollection
    pub fn new(name: &str, description: Option<&str>) -> Self {
        let now = Utc::now();

        MetricCollection {
            name: name.to_string(),
            description: description.map(|s| s.to_string()),
            metrics: Vec::new(),
            created_at: now,
            updated_at: now,
            version: "1.0.0".to_string(),
        }
    }

    /// Add a metric result to the collection
    ///
    /// # Arguments
    ///
    /// * `metric` - Metric result to add
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn add_metric(&mut self, metric: MetricResult) -> &mut Self {
        self.metrics.push(metric);
        self.updated_at = Utc::now();
        self
    }

    /// Set the version of the collection
    ///
    /// # Arguments
    ///
    /// * `version` - Version of the collection
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_version(&mut self, version: &str) -> &mut Self {
        self.version = version.to_string();
        self
    }

    /// Save the collection to a file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the collection to
    /// * `format` - Format to save the collection in
    ///
    /// # Returns
    ///
    /// * Result indicating success or error
    pub fn save<P: AsRef<Path>>(&self, path: P, format: SerializationFormat) -> Result<()> {
        let serialized = match format {
            SerializationFormat::Json => serde_json::to_string_pretty(self)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?,
            SerializationFormat::Yaml => serde_yaml::to_string(self)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?,
            SerializationFormat::Toml => toml::to_string_pretty(self)
                .map_err(|e| MetricsError::SerializationError(e.to_string()))?,
            SerializationFormat::Cbor => {
                let mut bytes = Vec::new();
                ciborium::ser::into_writer(self, &mut bytes)
                    .map_err(|e| MetricsError::SerializationError(e.to_string()))?;
                return save_binary(path, &bytes);
            }
        };

        save_text(path, &serialized)
    }

    /// Load a collection from a file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to load the collection from
    /// * `format` - Format to load the collection from
    ///
    /// # Returns
    ///
    /// * Result containing the loaded collection
    pub fn load<P: AsRef<Path>>(path: P, format: SerializationFormat) -> Result<Self> {
        match format {
            SerializationFormat::Json => {
                let text = load_text(path)?;
                serde_json::from_str(&text)
                    .map_err(|e| MetricsError::SerializationError(e.to_string()))
            }
            SerializationFormat::Yaml => {
                let text = load_text(path)?;
                serde_yaml::from_str(&text)
                    .map_err(|e| MetricsError::SerializationError(e.to_string()))
            }
            SerializationFormat::Toml => {
                let text = load_text(path)?;
                toml::from_str(&text).map_err(|e| MetricsError::SerializationError(e.to_string()))
            }
            SerializationFormat::Cbor => {
                let bytes = load_binary(path)?;
                ciborium::de::from_reader(&bytes[..])
                    .map_err(|e| MetricsError::SerializationError(e.to_string()))
            }
        }
    }
}

/// Create a new metric result
///
/// # Arguments
///
/// * `name` - Name of the metric
/// * `value` - Value of the metric
/// * `additional_values` - Optional additional values
/// * `metadata` - Optional metadata
///
/// # Returns
///
/// * A new MetricResult
pub fn create_metric_result(
    name: &str,
    value: f64,
    additional_values: Option<HashMap<String, f64>>,
    metadata: Option<MetricMetadata>,
) -> MetricResult {
    MetricResult {
        name: name.to_string(),
        value,
        additional_values,
        timestamp: Utc::now(),
        metadata,
    }
}

/// Supported serialization formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// JSON format
    Json,
    /// YAML format
    Yaml,
    /// TOML format
    Toml,
    /// CBOR format (binary)
    Cbor,
}

/// Save text to a file
///
/// # Arguments
///
/// * `path` - Path to save to
/// * `text` - Text to save
///
/// # Returns
///
/// * Result indicating success or error
fn save_text<P: AsRef<Path>>(path: P, text: &str) -> Result<()> {
    let mut file = File::create(path).map_err(|e| MetricsError::IOError(e.to_string()))?;

    file.write_all(text.as_bytes())
        .map_err(|e| MetricsError::IOError(e.to_string()))?;

    Ok(())
}

/// Load text from a file
///
/// # Arguments
///
/// * `path` - Path to load from
///
/// # Returns
///
/// * Result containing the loaded text
fn load_text<P: AsRef<Path>>(path: P) -> Result<String> {
    let mut file = File::open(path).map_err(|e| MetricsError::IOError(e.to_string()))?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|e| MetricsError::IOError(e.to_string()))?;

    Ok(contents)
}

/// Save binary data to a file
///
/// # Arguments
///
/// * `path` - Path to save to
/// * `data` - Data to save
///
/// # Returns
///
/// * Result indicating success or error
fn save_binary<P: AsRef<Path>>(path: P, data: &[u8]) -> Result<()> {
    let mut file = File::create(path).map_err(|e| MetricsError::IOError(e.to_string()))?;

    file.write_all(data)
        .map_err(|e| MetricsError::IOError(e.to_string()))?;

    Ok(())
}

/// Load binary data from a file
///
/// # Arguments
///
/// * `path` - Path to load from
///
/// # Returns
///
/// * Result containing the loaded data
fn load_binary<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
    let mut file = File::open(path).map_err(|e| MetricsError::IOError(e.to_string()))?;

    let mut contents = Vec::new();
    file.read_to_end(&mut contents)
        .map_err(|e| MetricsError::IOError(e.to_string()))?;

    Ok(contents)
}
