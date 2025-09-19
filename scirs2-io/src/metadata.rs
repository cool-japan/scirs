//! Advanced metadata management for scientific data
//!
//! Provides comprehensive metadata handling across different file formats with
//! unified interfaces for storing, retrieving, and transforming metadata.

#![allow(dead_code)]
#![allow(missing_docs)]

use crate::error::{IoError, Result};
use chrono::{DateTime, Utc};
use indexmap::{indexmap, IndexMap};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Standard metadata keys commonly used across scientific data formats
pub mod standard_keys {
    pub const TITLE: &str = "title";
    pub const AUTHOR: &str = "author";
    pub const DESCRIPTION: &str = "description";
    pub const CREATION_DATE: &str = "creation_date";
    pub const MODIFICATION_DATE: &str = "modification_date";
    pub const VERSION: &str = "version";
    pub const LICENSE: &str = "license";
    pub const KEYWORDS: &str = "keywords";
    pub const UNITS: &str = "units";
    pub const DIMENSIONS: &str = "dimensions";
    pub const COORDINATE_SYSTEM: &str = "coordinate_system";
    pub const INSTRUMENT: &str = "instrument";
    pub const EXPERIMENT: &str = "experiment";
    pub const PROCESSING_HISTORY: &str = "processing_history";
    pub const REFERENCES: &str = "references";
    pub const PROVENANCE: &str = "provenance";
}

/// Metadata value types supporting various data formats
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetadataValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Floating point value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// Date/time value
    DateTime(DateTime<Utc>),
    /// Array of values
    Array(Vec<MetadataValue>),
    /// Nested metadata object
    Object(IndexMap<String, MetadataValue>),
    /// Binary data
    Binary(Vec<u8>),
}

impl fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(s) => write!(f, "{}", s),
            Self::Integer(i) => write!(f, "{}", i),
            Self::Float(fl) => write!(f, "{}", fl),
            Self::Boolean(b) => write!(f, "{}", b),
            Self::DateTime(dt) => write!(f, "{}", dt.to_rfc3339()),
            Self::Array(arr) => write!(
                f,
                "[{}]",
                arr.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Object(_) => write!(f, "[object]"),
            Self::Binary(b) => write!(f, "[binary: {} bytes]", b.len()),
        }
    }
}

/// Advanced metadata container with rich functionality
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    /// Core metadata stored in insertion order
    data: IndexMap<String, MetadataValue>,
    /// Format-specific metadata extensions
    extensions: HashMap<String, IndexMap<String, MetadataValue>>,
    /// Metadata schema version
    schema_version: String,
}

impl Metadata {
    /// Create a new empty metadata container
    pub fn new() -> Self {
        Self {
            data: IndexMap::new(),
            extensions: HashMap::new(),
            schema_version: "1.0".to_string(),
        }
    }

    /// Create metadata with a specific schema version
    pub fn with_schema(version: &str) -> Self {
        Self {
            data: IndexMap::new(),
            extensions: HashMap::new(),
            schema_version: version.to_string(),
        }
    }

    /// Set a metadata value
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<MetadataValue>) {
        self.data.insert(key.into(), value.into());
    }

    /// Get a metadata value
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.data.get(key)
    }

    /// Get a typed metadata value
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.get(key)? {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get integer metadata value
    pub fn get_integer(&self, key: &str) -> Option<i64> {
        match self.get(key)? {
            MetadataValue::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Get float metadata value
    pub fn get_float(&self, key: &str) -> Option<f64> {
        match self.get(key)? {
            MetadataValue::Float(f) => Some(*f),
            MetadataValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Set format-specific extension metadata
    pub fn set_extension(
        &mut self,
        format: &str,
        key: impl Into<String>,
        value: impl Into<MetadataValue>,
    ) {
        self.extensions
            .entry(format.to_string())
            .or_default()
            .insert(key.into(), value.into());
    }

    /// Get format-specific extension metadata
    pub fn get_extension(&self, format: &str) -> Option<&IndexMap<String, MetadataValue>> {
        self.extensions.get(format)
    }

    /// Merge metadata from another container
    pub fn merge(&mut self, other: &Metadata) {
        for (key, value) in &other.data {
            self.data.insert(key.clone(), value.clone());
        }
        for (format, ext_data) in &other.extensions {
            let ext = self.extensions.entry(format.clone()).or_default();
            for (key, value) in ext_data {
                ext.insert(key.clone(), value.clone());
            }
        }
    }

    /// Validate metadata against a schema
    pub fn validate(&self, schema: &MetadataSchema) -> Result<()> {
        schema.validate(self)
    }

    /// Convert metadata to a specific format
    pub fn to_format(&self, format: MetadataFormat) -> Result<String> {
        match format {
            MetadataFormat::Json => serde_json::to_string_pretty(self)
                .map_err(|e| IoError::SerializationError(e.to_string())),
            MetadataFormat::Yaml => {
                serde_yaml::to_string(self).map_err(|e| IoError::SerializationError(e.to_string()))
            }
            MetadataFormat::Toml => {
                toml::to_string_pretty(self).map_err(|e| IoError::SerializationError(e.to_string()))
            }
        }
    }

    /// Load metadata from a file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let _path = path.as_ref();
        let content = std::fs::read_to_string(_path).map_err(IoError::Io)?;

        let extension = _path.extension().and_then(|ext| ext.to_str()).unwrap_or("");

        match extension {
            "json" => serde_json::from_str(&content)
                .map_err(|e| IoError::SerializationError(e.to_string())),
            "yaml" | "yml" => serde_yaml::from_str(&content)
                .map_err(|e| IoError::SerializationError(e.to_string())),
            "toml" => {
                toml::from_str(&content).map_err(|e| IoError::SerializationError(e.to_string()))
            }
            _ => Err(IoError::UnsupportedFormat(format!(
                "Unknown metadata format: {}",
                extension
            ))),
        }
    }

    /// Save metadata to a file
    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");

        let format = match extension {
            "json" => MetadataFormat::Json,
            "yaml" | "yml" => MetadataFormat::Yaml,
            "toml" => MetadataFormat::Toml,
            _ => {
                return Err(IoError::UnsupportedFormat(format!(
                    "Unknown metadata format: {}",
                    extension
                )))
            }
        };

        let content = self.to_format(format)?;
        std::fs::write(path, content).map_err(IoError::Io)
    }

    /// Add processing history entry
    pub fn add_processing_history(&mut self, entry: ProcessingHistoryEntry) -> Result<()> {
        let history = match self.data.get_mut(standard_keys::PROCESSING_HISTORY) {
            Some(MetadataValue::Array(arr)) => arr,
            _ => {
                self.data.insert(
                    standard_keys::PROCESSING_HISTORY.to_string(),
                    MetadataValue::Array(Vec::new()),
                );
                match self.data.get_mut(standard_keys::PROCESSING_HISTORY) {
                    Some(MetadataValue::Array(arr)) => arr,
                    _ => {
                        return Err(IoError::Other(
                            "Failed to create processing history array".to_string(),
                        ))
                    }
                }
            }
        };

        let entry_obj = indexmap! {
            "timestamp".to_string() => MetadataValue::DateTime(entry.timestamp),
            "operation".to_string() => MetadataValue::String(entry.operation),
            "parameters".to_string() => MetadataValue::Object(entry.parameters),
            "user".to_string() => MetadataValue::String(entry.user.unwrap_or_else(|| "unknown".to_string())),
        };

        history.push(MetadataValue::Object(entry_obj));
        Ok(())
    }

    /// Update modification timestamp
    pub fn update_modification_date(&mut self) {
        self.set(
            standard_keys::MODIFICATION_DATE,
            MetadataValue::DateTime(Utc::now()),
        );
    }
}

/// Processing history entry for tracking data transformations
#[derive(Debug, Clone)]
pub struct ProcessingHistoryEntry {
    pub timestamp: DateTime<Utc>,
    pub operation: String,
    pub parameters: IndexMap<String, MetadataValue>,
    pub user: Option<String>,
}

impl ProcessingHistoryEntry {
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            timestamp: Utc::now(),
            operation: operation.into(),
            parameters: IndexMap::new(),
            user: std::env::var("USER").ok(),
        }
    }

    pub fn with_parameter(
        mut self,
        key: impl Into<String>,
        value: impl Into<MetadataValue>,
    ) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }
}

/// Metadata output formats
#[derive(Debug, Clone, Copy)]
pub enum MetadataFormat {
    Json,
    Yaml,
    Toml,
}

/// Metadata schema for validation
#[derive(Debug, Clone)]
pub struct MetadataSchema {
    required_fields: Vec<String>,
    field_types: HashMap<String, MetadataFieldType>,
    constraints: Vec<MetadataConstraint>,
}

#[derive(Debug, Clone)]
pub enum MetadataFieldType {
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Array(Box<MetadataFieldType>),
    Object,
}

#[derive(Debug, Clone)]
pub enum MetadataConstraint {
    MinValue(String, f64),
    MaxValue(String, f64),
    Pattern(String, String),
    OneOf(String, Vec<MetadataValue>),
}

impl Default for MetadataSchema {
    fn default() -> Self {
        Self::new()
    }
}

impl MetadataSchema {
    pub fn new() -> Self {
        Self {
            required_fields: Vec::new(),
            field_types: HashMap::new(),
            constraints: Vec::new(),
        }
    }

    pub fn require(mut self, field: impl Into<String>) -> Self {
        self.required_fields.push(field.into());
        self
    }

    pub fn field_type(mut self, field: impl Into<String>, fieldtype: MetadataFieldType) -> Self {
        self.field_types.insert(field.into(), fieldtype);
        self
    }

    pub fn constraint(mut self, constraint: MetadataConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    pub fn validate(&self, metadata: &Metadata) -> Result<()> {
        // Check required fields
        for field in &self.required_fields {
            if metadata.get(field).is_none() {
                return Err(IoError::ValidationError(format!(
                    "Required field '{}' is missing",
                    field
                )));
            }
        }

        // Validate field types
        for (field, expected_type) in &self.field_types {
            if let Some(value) = metadata.get(field) {
                if !self.validate_type(value, expected_type) {
                    return Err(IoError::ValidationError(format!(
                        "Field '{}' has incorrect type",
                        field
                    )));
                }
            }
        }

        // Apply constraints
        for constraint in &self.constraints {
            self.apply_constraint(metadata, constraint)?;
        }

        Ok(())
    }

    #[allow(clippy::only_used_in_recursion)]
    fn validate_type(&self, value: &MetadataValue, expected: &MetadataFieldType) -> bool {
        match (value, expected) {
            (MetadataValue::String(_), MetadataFieldType::String) => true,
            (MetadataValue::Integer(_), MetadataFieldType::Integer) => true,
            (MetadataValue::Float(_), MetadataFieldType::Float) => true,
            (MetadataValue::Boolean(_), MetadataFieldType::Boolean) => true,
            (MetadataValue::DateTime(_), MetadataFieldType::DateTime) => true,
            (MetadataValue::Array(arr), MetadataFieldType::Array(elem_type)) => {
                arr.iter().all(|v| self.validate_type(v, elem_type))
            }
            (MetadataValue::Object(_), MetadataFieldType::Object) => true,
            _ => false,
        }
    }

    fn apply_constraint(&self, metadata: &Metadata, constraint: &MetadataConstraint) -> Result<()> {
        match constraint {
            MetadataConstraint::MinValue(field, min) => {
                if let Some(val) = metadata.get_float(field) {
                    if val < *min {
                        return Err(IoError::ValidationError(format!(
                            "Field '{}' value {} is less than minimum {}",
                            field, val, min
                        )));
                    }
                }
            }
            MetadataConstraint::MaxValue(field, max) => {
                if let Some(val) = metadata.get_float(field) {
                    if val > *max {
                        return Err(IoError::ValidationError(format!(
                            "Field '{}' value {} is greater than maximum {}",
                            field, val, max
                        )));
                    }
                }
            }
            MetadataConstraint::Pattern(field, pattern) => {
                if let Some(val) = metadata.get_string(field) {
                    let re = regex::Regex::new(pattern).map_err(|e| {
                        IoError::ValidationError(format!("Invalid regex pattern: {e}"))
                    })?;
                    if !re.is_match(val) {
                        return Err(IoError::ValidationError(format!(
                            "Field '{}' value '{}' does not match pattern '{}'",
                            field, val, pattern
                        )));
                    }
                }
            }
            MetadataConstraint::OneOf(field, allowed) => {
                if let Some(val) = metadata.get(field) {
                    if !allowed.contains(val) {
                        return Err(IoError::ValidationError(format!(
                            "Field '{}' value is not in allowed set",
                            field
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Metadata transformer for converting between different metadata formats
pub struct MetadataTransformer {
    mappings: HashMap<String, String>,
    transformations: HashMap<String, Box<dyn Fn(&MetadataValue) -> MetadataValue>>,
}

impl Default for MetadataTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl MetadataTransformer {
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
            transformations: HashMap::new(),
        }
    }

    /// Add a field mapping
    pub fn map_field(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.mappings.insert(from.into(), to.into());
        self
    }

    /// Transform metadata using configured mappings
    pub fn transform(&self, input: &Metadata) -> Metadata {
        let mut output = Metadata::new();
        output.schema_version = input.schema_version.clone();

        for (key, value) in &input.data {
            let new_key = self
                .mappings
                .get(key)
                .cloned()
                .unwrap_or_else(|| key.clone());
            let new_value = if let Some(transform) = self.transformations.get(key) {
                transform(value)
            } else {
                value.clone()
            };
            output.set(new_key, new_value);
        }

        output.extensions = input.extensions.clone();
        output
    }
}

/// Predefined metadata schemas for common scientific data types
pub mod schemas {
    use super::*;

    /// Schema for image metadata
    pub fn image_schema() -> MetadataSchema {
        MetadataSchema::new()
            .require("width")
            .require("height")
            .field_type("width", MetadataFieldType::Integer)
            .field_type("height", MetadataFieldType::Integer)
            .field_type("channels", MetadataFieldType::Integer)
            .field_type("bit_depth", MetadataFieldType::Integer)
            .constraint(MetadataConstraint::MinValue("width".to_string(), 1.0))
            .constraint(MetadataConstraint::MinValue("height".to_string(), 1.0))
    }

    /// Schema for time series metadata
    pub fn time_series_schema() -> MetadataSchema {
        MetadataSchema::new()
            .require("start_time")
            .require("sampling_rate")
            .field_type("start_time", MetadataFieldType::DateTime)
            .field_type("sampling_rate", MetadataFieldType::Float)
            .field_type("units", MetadataFieldType::String)
            .constraint(MetadataConstraint::MinValue(
                "sampling_rate".to_string(),
                0.0,
            ))
    }

    /// Schema for geospatial metadata
    pub fn geospatial_schema() -> MetadataSchema {
        MetadataSchema::new()
            .require("coordinate_system")
            .field_type("coordinate_system", MetadataFieldType::String)
            .field_type(
                "bounds",
                MetadataFieldType::Array(Box::new(MetadataFieldType::Float)),
            )
            .field_type("projection", MetadataFieldType::String)
    }
}

impl From<String> for MetadataValue {
    fn from(s: String) -> Self {
        MetadataValue::String(s)
    }
}

impl From<&str> for MetadataValue {
    fn from(s: &str) -> Self {
        MetadataValue::String(s.to_string())
    }
}

impl From<i64> for MetadataValue {
    fn from(i: i64) -> Self {
        MetadataValue::Integer(i)
    }
}

impl From<f64> for MetadataValue {
    fn from(f: f64) -> Self {
        MetadataValue::Float(f)
    }
}

impl From<bool> for MetadataValue {
    fn from(b: bool) -> Self {
        MetadataValue::Boolean(b)
    }
}

impl From<DateTime<Utc>> for MetadataValue {
    fn from(dt: DateTime<Utc>) -> Self {
        MetadataValue::DateTime(dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_basic_operations() {
        let mut metadata = Metadata::new();
        metadata.set("title", "Test Dataset");
        metadata.set("version", 1i64);
        metadata.set("temperature", 25.5f64);

        assert_eq!(metadata.get_string("title"), Some("Test Dataset"));
        assert_eq!(metadata.get_integer("version"), Some(1));
        assert_eq!(metadata.get_float("temperature"), Some(25.5));
    }

    #[test]
    fn test_metadata_schema_validation() {
        let schema = MetadataSchema::new()
            .require("title")
            .require("version")
            .field_type("version", MetadataFieldType::Integer)
            .constraint(MetadataConstraint::MinValue("version".to_string(), 1.0));

        let mut metadata = Metadata::new();
        metadata.set("title", "Test");
        metadata.set("version", 2i64);

        assert!(schema.validate(&metadata).is_ok());
    }

    #[test]
    fn test_processing_history() {
        let mut metadata = Metadata::new();

        let entry = ProcessingHistoryEntry::new("normalize")
            .with_parameter("method", "z-score")
            .with_parameter("mean", 0.0)
            .with_parameter("std", 1.0);

        metadata.add_processing_history(entry).unwrap();

        let history = metadata.get(standard_keys::PROCESSING_HISTORY);
        assert!(matches!(history, Some(MetadataValue::Array(_))));
    }
}

/// Advanced metadata index for fast searching and querying
#[derive(Debug, Clone)]
pub struct MetadataIndex {
    /// Inverted index for text search
    text_index: HashMap<String, HashSet<String>>,
    /// Numeric index for range queries
    numeric_index: HashMap<String, Vec<(String, f64)>>,
    /// Date index for temporal queries
    date_index: HashMap<String, Vec<(String, DateTime<Utc>)>>,
}

impl Default for MetadataIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl MetadataIndex {
    pub fn new() -> Self {
        Self {
            text_index: HashMap::new(),
            numeric_index: HashMap::new(),
            date_index: HashMap::new(),
        }
    }

    /// Index a metadata collection
    pub fn index_metadata(&mut self, id: &str, metadata: &Metadata) {
        for (key, value) in &metadata.data {
            self.index_value(id, key, value);
        }
    }

    fn index_value(&mut self, id: &str, key: &str, value: &MetadataValue) {
        match value {
            MetadataValue::String(s) => {
                // Tokenize and index text
                for token in s.to_lowercase().split_whitespace() {
                    self.text_index
                        .entry(format!("{key}:{token}"))
                        .or_default()
                        .insert(id.to_string());
                }
            }
            MetadataValue::Integer(i) => {
                self.numeric_index
                    .entry(key.to_string())
                    .or_default()
                    .push((id.to_string(), *i as f64));
            }
            MetadataValue::Float(f) => {
                self.numeric_index
                    .entry(key.to_string())
                    .or_default()
                    .push((id.to_string(), *f));
            }
            MetadataValue::DateTime(dt) => {
                self.date_index
                    .entry(key.to_string())
                    .or_default()
                    .push((id.to_string(), *dt));
            }
            MetadataValue::Array(arr) => {
                for (i, item) in arr.iter().enumerate() {
                    self.index_value(id, &format!("{key}[{i}]"), item);
                }
            }
            MetadataValue::Object(obj) => {
                for (sub_key, sub_value) in obj {
                    self.index_value(id, &format!("{key}.{sub_key}"), sub_value);
                }
            }
            _ => {}
        }
    }

    /// Search for metadata by text query
    pub fn searchtext(&self, field: &str, query: &str) -> HashSet<String> {
        let key = format!("{field}:{}", query.to_lowercase());
        self.text_index.get(&key).cloned().unwrap_or_default()
    }

    /// Search for metadata by numeric range
    pub fn search_range(&self, field: &str, min: f64, max: f64) -> HashSet<String> {
        self.numeric_index
            .get(field)
            .map(|values| {
                values
                    .iter()
                    .filter(|(_, v)| *v >= min && *v <= max)
                    .map(|(id_, _)| id_.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Search for metadata by date range
    pub fn search_date_range(
        &self,
        field: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> HashSet<String> {
        self.date_index
            .get(field)
            .map(|values| {
                values
                    .iter()
                    .filter(|(_, dt)| *dt >= start && *dt <= end)
                    .map(|(id_, _)| id_.clone())
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Metadata version control system
#[derive(Debug, Clone)]
pub struct MetadataVersionControl {
    /// Current version
    current: Arc<RwLock<Metadata>>,
    /// Version history
    history: Arc<RwLock<Vec<MetadataVersion>>>,
    /// Maximum number of versions to keep
    max_versions: usize,
}

#[derive(Debug, Clone)]
pub struct MetadataVersion {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: Metadata,
    pub parent_id: Option<String>,
    pub message: String,
    pub author: Option<String>,
    pub hash: String,
}

impl MetadataVersionControl {
    pub fn new(initial: Metadata) -> Self {
        let version = MetadataVersion {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            metadata: initial.clone(),
            parent_id: None,
            message: "Initial version".to_string(),
            author: std::env::var("USER").ok(),
            hash: Self::compute_hash(&initial),
        };

        Self {
            current: Arc::new(RwLock::new(initial)),
            history: Arc::new(RwLock::new(vec![version])),
            max_versions: 100,
        }
    }

    /// Commit a new version
    pub fn commit(&self, metadata: Metadata, message: impl Into<String>) -> Result<String> {
        let mut history = self.history.write().unwrap();
        let parent_id = history.last().map(|v| v.id.clone());

        let version = MetadataVersion {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            metadata: metadata.clone(),
            parent_id,
            message: message.into(),
            author: std::env::var("USER").ok(),
            hash: Self::compute_hash(&metadata),
        };

        let versionid = version.id.clone();
        history.push(version);

        // Prune old versions if necessary
        if history.len() > self.max_versions {
            let keep_count = self.max_versions;
            let remove_count = history.len() - keep_count;
            history.drain(0..remove_count);
        }

        *self.current.write().unwrap() = metadata;

        Ok(versionid)
    }

    /// Get a specific version
    pub fn get_version(&self, versionid: &str) -> Option<MetadataVersion> {
        self.history
            .read()
            .unwrap()
            .iter()
            .find(|v| v.id == versionid)
            .cloned()
    }

    /// Get version history
    pub fn get_history(&self) -> Vec<MetadataVersion> {
        self.history.read().unwrap().clone()
    }

    /// Compute diff between two versions
    pub fn diff(&self, version1: &str, version2: &str) -> Option<MetadataDiff> {
        let history = self.history.read().unwrap();
        let v1 = history.iter().find(|v| v.id == version1)?;
        let v2 = history.iter().find(|v| v.id == version2)?;

        Some(MetadataDiff::compute(&v1.metadata, &v2.metadata))
    }

    /// Rollback to a specific version
    pub fn rollback(&self, versionid: &str) -> Result<()> {
        let version = self
            .get_version(versionid)
            .ok_or_else(|| IoError::NotFound(format!("Version {versionid} not found")))?;

        self.commit(version.metadata.clone(), format!("Rollback to {versionid}"))?;
        Ok(())
    }

    fn compute_hash(metadata: &Metadata) -> String {
        let json = serde_json::to_string(metadata).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Represents differences between two metadata objects
#[derive(Debug, Clone)]
pub struct MetadataDiff {
    pub added: IndexMap<String, MetadataValue>,
    pub removed: IndexMap<String, MetadataValue>,
    pub modified: IndexMap<String, (MetadataValue, MetadataValue)>,
}

impl MetadataDiff {
    pub fn compute(old: &Metadata, new: &Metadata) -> Self {
        let mut added = IndexMap::new();
        let mut removed = IndexMap::new();
        let mut modified = IndexMap::new();

        // Find removed and modified fields
        for (key, old_value) in &old.data {
            match new.data.get(key) {
                None => {
                    removed.insert(key.clone(), old_value.clone());
                }
                Some(new_value) if new_value != old_value => {
                    modified.insert(key.clone(), (old_value.clone(), new_value.clone()));
                }
                _ => {}
            }
        }

        // Find added fields
        for (key, new_value) in &new.data {
            if !old.data.contains_key(key) {
                added.insert(key.clone(), new_value.clone());
            }
        }

        Self {
            added,
            removed,
            modified,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.modified.is_empty()
    }
}

/// Metadata inheritance and composition system
#[derive(Debug, Clone)]
pub struct MetadataTemplate {
    /// Base metadata to inherit from
    base: Metadata,
    /// Fields that can be overridden
    overridable: HashSet<String>,
    /// Fields that must be provided
    required: HashSet<String>,
    /// Default values for optional fields
    defaults: IndexMap<String, MetadataValue>,
}

impl MetadataTemplate {
    pub fn new(base: Metadata) -> Self {
        Self {
            base,
            overridable: HashSet::new(),
            required: HashSet::new(),
            defaults: IndexMap::new(),
        }
    }

    pub fn allow_override(mut self, field: impl Into<String>) -> Self {
        self.overridable.insert(field.into());
        self
    }

    pub fn require_field(mut self, field: impl Into<String>) -> Self {
        self.required.insert(field.into());
        self
    }

    pub fn default_value(
        mut self,
        field: impl Into<String>,
        value: impl Into<MetadataValue>,
    ) -> Self {
        self.defaults.insert(field.into(), value.into());
        self
    }

    /// Create a new metadata instance from this template
    pub fn instantiate(&self, overrides: IndexMap<String, MetadataValue>) -> Result<Metadata> {
        let mut metadata = self.base.clone();

        // Apply defaults
        for (key, value) in &self.defaults {
            if !metadata.data.contains_key(key) {
                metadata.set(key.clone(), value.clone());
            }
        }

        // Apply overrides
        for (key, value) in overrides {
            if !self.overridable.contains(&key) && self.base.data.contains_key(&key) {
                return Err(IoError::ValidationError(format!(
                    "Field '{}' cannot be overridden",
                    key
                )));
            }
            metadata.set(key, value);
        }

        // Check required fields
        for field in &self.required {
            if !metadata.data.contains_key(field) {
                return Err(IoError::ValidationError(format!(
                    "Required field '{}' is missing",
                    field
                )));
            }
        }

        Ok(metadata)
    }
}

/// Cross-reference resolution system
#[derive(Debug, Clone)]
pub struct MetadataReferenceResolver {
    /// Registry of metadata objects by ID
    registry: Arc<RwLock<HashMap<String, Metadata>>>,
    /// Reference graph for dependency tracking
    references: Arc<RwLock<HashMap<String, HashSet<String>>>>,
}

impl Default for MetadataReferenceResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl MetadataReferenceResolver {
    pub fn new() -> Self {
        Self {
            registry: Arc::new(RwLock::new(HashMap::new())),
            references: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a metadata object
    pub fn register(&self, id: impl Into<String>, metadata: Metadata) -> Result<()> {
        let id = id.into();
        let refs = self.extract_references(&metadata);

        self.registry.write().unwrap().insert(id.clone(), metadata);
        self.references.write().unwrap().insert(id, refs);

        Ok(())
    }

    /// Resolve all references in a metadata object
    pub fn resolve(&self, metadata: &mut Metadata) -> Result<()> {
        self.resolve_value(&mut metadata.data)?;
        Ok(())
    }

    fn resolve_value(&self, data: &mut IndexMap<String, MetadataValue>) -> Result<()> {
        for value in data.values_mut() {
            match value {
                MetadataValue::String(s) if s.starts_with("ref:") => {
                    let ref_id = s.strip_prefix("ref:").unwrap();
                    let registry = self.registry.read().unwrap();
                    if let Some(referenced) = registry.get(ref_id) {
                        *value = MetadataValue::Object(referenced.data.clone());
                    } else {
                        return Err(IoError::NotFound(format!(
                            "Reference '{}' not found",
                            ref_id
                        )));
                    }
                }
                MetadataValue::Object(obj) => {
                    self.resolve_value(obj)?;
                }
                MetadataValue::Array(arr) => {
                    for item in arr {
                        if let MetadataValue::Object(obj) = item {
                            self.resolve_value(obj)?;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn extract_references(&self, metadata: &Metadata) -> HashSet<String> {
        let mut refs = HashSet::new();
        self.extract_refs_from_value(&metadata.data, &mut refs);
        refs
    }

    #[allow(clippy::only_used_in_recursion)]
    fn extract_refs_from_value(
        &self,
        data: &IndexMap<String, MetadataValue>,
        refs: &mut HashSet<String>,
    ) {
        for value in data.values() {
            match value {
                MetadataValue::String(s) if s.starts_with("ref:") => {
                    if let Some(ref_id) = s.strip_prefix("ref:") {
                        refs.insert(ref_id.to_string());
                    }
                }
                MetadataValue::Object(obj) => {
                    self.extract_refs_from_value(obj, refs);
                }
                MetadataValue::Array(arr) => {
                    for item in arr {
                        if let MetadataValue::Object(obj) = item {
                            self.extract_refs_from_value(obj, refs);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Get all objects that reference a given ID
    pub fn get_referencing(&self, id: &str) -> Vec<String> {
        let references = self.references.read().unwrap();
        references
            .iter()
            .filter(|(_, refs)| refs.contains(id))
            .map(|(referencing_id_, _)| referencing_id_.clone())
            .collect()
    }
}

/// Metadata provenance tracking with cryptographic verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataProvenance {
    /// Chain of custody
    chain: Vec<ProvenanceEntry>,
    /// Cryptographic signatures
    signatures: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceEntry {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub action: String,
    pub agent: String,
    pub previous_hash: Option<String>,
    pub data_hash: String,
    pub metadata_snapshot: Option<Metadata>,
}

impl Default for MetadataProvenance {
    fn default() -> Self {
        Self::new()
    }
}

impl MetadataProvenance {
    pub fn new() -> Self {
        Self {
            chain: Vec::new(),
            signatures: HashMap::new(),
        }
    }

    /// Add a provenance entry
    pub fn add_entry(
        &mut self,
        action: impl Into<String>,
        agent: impl Into<String>,
        metadata: &Metadata,
    ) {
        let previous_hash = self.chain.last().map(|e| e.data_hash.clone());

        let entry = ProvenanceEntry {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            action: action.into(),
            agent: agent.into(),
            previous_hash: previous_hash.clone(),
            data_hash: self.compute_hash(metadata, previous_hash.as_deref()),
            metadata_snapshot: Some(metadata.clone()),
        };

        self.chain.push(entry);
    }

    /// Verify the integrity of the provenance chain
    pub fn verify_chain(&self) -> Result<()> {
        let mut previous_hash: Option<String> = None;

        for entry in &self.chain {
            if entry.previous_hash != previous_hash {
                return Err(IoError::ValidationError(format!(
                    "Provenance chain broken at entry {}",
                    entry.id
                )));
            }

            if let Some(metadata) = &entry.metadata_snapshot {
                let expected_hash = self.compute_hash(metadata, previous_hash.as_deref());
                if entry.data_hash != expected_hash {
                    return Err(IoError::ValidationError(format!(
                        "Data hash mismatch at entry {}",
                        entry.id
                    )));
                }
            }

            previous_hash = Some(entry.data_hash.clone());
        }

        Ok(())
    }

    fn compute_hash(&self, metadata: &Metadata, previous: Option<&str>) -> String {
        let mut hasher = Sha256::new();
        if let Some(prev) = previous {
            hasher.update(prev.as_bytes());
        }
        let json = serde_json::to_string(metadata).unwrap_or_default();
        hasher.update(json.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Export provenance as a verifiable certificate
    pub fn export_certificate(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| IoError::SerializationError(e.to_string()))
    }
}

/// External metadata repository integration
#[cfg(feature = "reqwest")]
#[derive(Debug, Clone)]
pub struct MetadataRepository {
    /// Repository URL
    url: String,
    /// Local cache
    cache: Arc<RwLock<HashMap<String, Metadata>>>,
    /// HTTP client
    client: reqwest::blocking::Client,
}

#[cfg(feature = "reqwest")]
impl MetadataRepository {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            cache: Arc::new(RwLock::new(HashMap::new())),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Fetch metadata from repository
    pub fn fetch(&self, id: &str) -> Result<Metadata> {
        // Check cache first
        if let Some(metadata) = self.cache.read().unwrap().get(id) {
            return Ok(metadata.clone());
        }

        // Fetch from repository
        let url = format!("{}/metadata/{id}", self.url);
        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| IoError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(IoError::NetworkError(format!(
                "Failed to fetch metadata: {}",
                response.status()
            )));
        }

        let metadata: Metadata = response
            .json()
            .map_err(|e| IoError::SerializationError(e.to_string()))?;

        // Update cache
        self.cache
            .write()
            .unwrap()
            .insert(id.to_string(), metadata.clone());

        Ok(metadata)
    }

    /// Push metadata to repository
    pub fn push(&self, id: &str, metadata: &Metadata) -> Result<()> {
        let url = format!("{}/metadata/{id}", self.url);
        let response = self
            .client
            .put(&url)
            .json(metadata)
            .send()
            .map_err(|e| IoError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(IoError::NetworkError(format!(
                "Failed to push metadata: {}",
                response.status()
            )));
        }

        // Update cache
        self.cache
            .write()
            .unwrap()
            .insert(id.to_string(), metadata.clone());

        Ok(())
    }

    /// Search repository
    pub fn search(&self, query: &str) -> Result<Vec<String>> {
        let url = format!("{}/search?q={}", self.url, urlencoding::encode(query));
        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| IoError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(IoError::NetworkError(format!(
                "Search failed: {}",
                response.status()
            )));
        }

        let results: Vec<String> = response
            .json()
            .map_err(|e| IoError::SerializationError(e.to_string()))?;

        Ok(results)
    }
}

/// Automatic metadata extraction from files
pub struct MetadataExtractor {
    extractors: HashMap<String, Box<dyn Fn(&Path) -> Result<Metadata> + Send + Sync>>,
}

impl Default for MetadataExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl MetadataExtractor {
    pub fn new() -> Self {
        let mut extractor = Self {
            extractors: HashMap::new(),
        };

        // Register default extractors
        extractor.register_defaults();
        extractor
    }

    fn register_defaults(&mut self) {
        // Image metadata extractor
        self.register(
            "image",
            Box::new(|path| {
                let mut metadata = Metadata::new();

                // Use image crate to extract metadata
                if let Ok(img) = image::open(path) {
                    metadata.set("width", img.width() as i64);
                    metadata.set("height", img.height() as i64);
                    metadata.set("color_type", format!("{:?}", img.color()));
                }

                // Extract EXIF data if available
                if let Ok(file) = std::fs::File::open(path) {
                    let exif_reader = exif::Reader::new();
                    if let Ok(exif) =
                        exif_reader.read_from_container(&mut std::io::BufReader::new(file))
                    {
                        for field in exif.fields() {
                            let key = format!("exif.{}", field.tag);
                            let value = field.display_value().to_string();
                            metadata.set_extension("exif", key, value);
                        }
                    }
                }

                Ok(metadata)
            }),
        );

        // Audio metadata extractor
        self.register(
            "audio",
            Box::new(|path| {
                let mut metadata = Metadata::new();

                // Basic audio file info
                if let Ok(meta) = std::fs::metadata(path) {
                    metadata.set("file_size", meta.len() as i64);
                    if let Ok(modified) = meta.modified() {
                        metadata.set("modified", MetadataValue::DateTime(modified.into()));
                    }
                }

                // Extract audio-specific metadata
                // This would use audio-specific libraries in a real implementation

                Ok(metadata)
            }),
        );

        // NetCDF metadata extractor
        self.register(
            "netcdf",
            Box::new(|_path| {
                let metadata = Metadata::new();

                // Extract NetCDF global attributes
                // This would use the netcdf module in a real implementation

                Ok(metadata)
            }),
        );
    }

    /// Register a custom extractor
    pub fn register(
        &mut self,
        format: &str,
        extractor: Box<dyn Fn(&Path) -> Result<Metadata> + Send + Sync>,
    ) {
        self.extractors.insert(format.to_string(), extractor);
    }

    /// Extract metadata from a file
    pub fn extract(&self, path: impl AsRef<Path>) -> Result<Metadata> {
        let path = path.as_ref();
        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");

        // Determine format from extension
        let format = match extension {
            "png" | "jpg" | "jpeg" | "gif" | "bmp" | "tiff" => "image",
            "wav" | "mp3" | "flac" | "ogg" => "audio",
            "nc" | "nc4" => "netcdf",
            "h5" | "hdf5" => "hdf5",
            _ => {
                return Err(IoError::UnsupportedFormat(format!(
                    "No extractor for format: {}",
                    extension
                )))
            }
        };

        if let Some(extractor) = self.extractors.get(format) {
            extractor(path)
        } else {
            Err(IoError::UnsupportedFormat(format!(
                "No extractor for format: {}",
                format
            )))
        }
    }

    /// Extract and merge metadata from multiple sources
    pub fn extract_composite(&self, paths: &[impl AsRef<Path>]) -> Result<Metadata> {
        let mut composite = Metadata::new();

        for (i, path) in paths.iter().enumerate() {
            let metadata = self.extract(path)?;

            // Store each file's metadata under a numbered key
            let key = format!("file_{}", i);
            composite.set(key, MetadataValue::Object(metadata.data));
        }

        composite.set("file_count", paths.len() as i64);
        composite.update_modification_date();

        Ok(composite)
    }
}
