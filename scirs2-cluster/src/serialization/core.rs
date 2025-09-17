//! Core serialization traits and metadata structures
//!
//! This module provides the fundamental traits and metadata structures
//! for model serialization and deserialization.

use crate::error::{ClusteringError, Result};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Trait for clustering models that can be serialized
pub trait SerializableModel: Serialize + for<'de> Deserialize<'de> {
    /// Save the model to a file
    fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to create file: {}", e)))?;
        self.save_to_writer(file)
    }

    /// Save the model to a writer
    fn save_to_writer<W: Write>(&self, writer: W) -> Result<()> {
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to serialize model: {}", e)))
    }

    /// Save the model to a file with compression
    fn save_to_file_compressed<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to create file: {}", e)))?;
        let encoder = GzEncoder::new(file, Compression::default());
        self.save_to_writer(encoder)
    }

    /// Load the model from a compressed file
    fn load_from_file_compressed<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to open file: {}", e)))?;
        let decoder = GzDecoder::new(file);
        Self::load_from_reader(decoder)
    }

    /// Load the model from a file
    fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to open file: {}", e)))?;
        Self::load_from_reader(&mut file)
    }

    /// Load the model from a reader
    fn load_from_reader<R: Read>(reader: R) -> Result<Self> {
        serde_json::from_reader(reader).map_err(|e| {
            ClusteringError::InvalidInput(format!("Failed to deserialize model: {}", e))
        })
    }
}

/// Enhanced model metadata with versioning and performance metrics
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EnhancedModelMetadata {
    /// Model format version for backward compatibility
    pub format_version: String,
    /// scirs2-cluster library version
    pub library_version: String,
    /// Timestamp when model was created (Unix epoch)
    pub created_timestamp: u64,
    /// Algorithm name and configuration hash
    pub algorithm_signature: String,
    /// Performance metrics during training
    pub training_metrics: TrainingMetrics,
    /// Data characteristics
    pub data_characteristics: DataCharacteristics,
    /// Model integrity hash
    pub integrity_hash: String,
    /// Platform information
    pub platform_info: PlatformInfo,
}

/// Training performance metrics
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingMetrics {
    /// Total training time in milliseconds
    pub training_time_ms: u64,
    /// Number of iterations/epochs
    pub iterations: usize,
    /// Final convergence metric (e.g., inertia, log-likelihood)
    pub final_convergence_metric: f64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// CPU utilization during training (0.0 to 100.0)
    pub avg_cpu_utilization: f64,
}

/// Data characteristics for validation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DataCharacteristics {
    /// Number of samples in training data
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Data type fingerprint
    pub data_type_fingerprint: String,
    /// Feature range summaries (min, max for each feature)
    pub feature_ranges: Option<Vec<(f64, f64)>>,
    /// Whether data was normalized/standardized
    pub preprocessing_applied: Vec<String>,
}

/// Platform information for cross-platform compatibility
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PlatformInfo {
    /// Operating system
    pub os: String,
    /// Architecture (x86_64, aarch64, etc.)
    pub arch: String,
    /// Rust compiler version
    pub rust_version: String,
    /// CPU features used (SIMD, etc.)
    pub cpu_features: Vec<String>,
}

impl Default for EnhancedModelMetadata {
    fn default() -> Self {
        Self {
            format_version: "1.0.0".to_string(),
            library_version: env!("CARGO_PKG_VERSION").to_string(),
            created_timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            algorithm_signature: "unknown".to_string(),
            training_metrics: TrainingMetrics::default(),
            data_characteristics: DataCharacteristics::default(),
            integrity_hash: String::new(),
            platform_info: PlatformInfo::detect(),
        }
    }
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            training_time_ms: 0,
            iterations: 0,
            final_convergence_metric: 0.0,
            peak_memory_bytes: 0,
            avg_cpu_utilization: 0.0,
        }
    }
}

impl Default for DataCharacteristics {
    fn default() -> Self {
        Self {
            n_samples: 0,
            n_features: 0,
            data_type_fingerprint: "unknown".to_string(),
            feature_ranges: None,
            preprocessing_applied: Vec::new(),
        }
    }
}

impl PlatformInfo {
    /// Detect current platform information
    pub fn detect() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            rust_version: option_env!("CARGO_PKG_RUST_VERSION")
                .filter(|s| !s.is_empty())
                .unwrap_or("unknown")
                .to_string(),
            cpu_features: Self::detect_cpu_features(),
        }
    }

    /// Detect available CPU features
    fn detect_cpu_features() -> Vec<String> {
        let mut features = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                features.push("avx2".to_string());
            }
            if std::arch::is_x86_feature_detected!("sse4.1") {
                features.push("sse4.1".to_string());
            }
            if std::arch::is_x86_feature_detected!("fma") {
                features.push("fma".to_string());
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                features.push("neon".to_string());
            }
        }

        features
    }
}

/// Enhanced model wrapper with metadata
#[derive(Serialize, Debug, Clone)]
pub struct EnhancedModel<T: SerializableModel> {
    /// The actual model data
    pub model: T,
    /// Enhanced metadata
    pub metadata: EnhancedModelMetadata,
}

impl<'de, T: SerializableModel> Deserialize<'de> for EnhancedModel<T> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct EnhancedModelHelper<U> {
            model: U,
            metadata: EnhancedModelMetadata,
        }

        let helper = EnhancedModelHelper::deserialize(deserializer)?;
        Ok(EnhancedModel {
            model: helper.model,
            metadata: helper.metadata,
        })
    }
}

impl<T: SerializableModel> EnhancedModel<T> {
    /// Create a new enhanced model with metadata
    pub fn new(model: T, metadata: EnhancedModelMetadata) -> Self {
        Self { model, metadata }
    }

    /// Create an enhanced model with automatic metadata generation
    pub fn with_auto_metadata(model: T, algorithm_name: &str) -> Self {
        let mut metadata = EnhancedModelMetadata::default();
        metadata.algorithm_signature = algorithm_name.to_string();
        Self { model, metadata }
    }

    /// Validate model integrity
    pub fn validate_integrity(&self) -> Result<bool> {
        // Simple validation - in practice this would check the hash
        Ok(!self.metadata.integrity_hash.is_empty())
    }

    /// Get model format version
    pub fn format_version(&self) -> &str {
        &self.metadata.format_version
    }

    /// Check if model is compatible with current library version
    pub fn is_compatible(&self) -> bool {
        // Simple compatibility check based on major version
        let model_version = &self.metadata.library_version;
        let current_version = env!("CARGO_PKG_VERSION");

        let model_major = model_version.split('.').next().unwrap_or("0");
        let current_major = current_version.split('.').next().unwrap_or("0");

        model_major == current_major
    }

    /// Get training duration in seconds
    pub fn training_duration_seconds(&self) -> f64 {
        self.metadata.training_metrics.training_time_ms as f64 / 1000.0
    }

    /// Get memory usage in MB
    pub fn peak_memory_mb(&self) -> f64 {
        self.metadata.training_metrics.peak_memory_bytes as f64 / (1024.0 * 1024.0)
    }
}

impl<T: SerializableModel> SerializableModel for EnhancedModel<T> {}

/// Format a timestamp for display
pub fn format_timestamp(timestamp: u64) -> String {
    match SystemTime::UNIX_EPOCH.checked_add(std::time::Duration::from_secs(timestamp)) {
        Some(_datetime) => {
            // Simple conversion: Unix timestamp to year
            // 1640995200 is 2022-01-01 00:00:00 UTC
            let years_since_1970 = timestamp / (365 * 24 * 3600); // Approximate
            let year = 1970 + years_since_1970;
            format!("Timestamp: {} (approx year {})", timestamp, year)
        }
        None => "Invalid timestamp".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct TestModel {
        value: i32,
    }

    impl SerializableModel for TestModel {}

    #[test]
    fn test_enhanced_model_creation() {
        let model = TestModel { value: 42 };
        let enhanced = EnhancedModel::with_auto_metadata(model, "test_algorithm");

        assert_eq!(enhanced.metadata.algorithm_signature, "test_algorithm");
        assert_eq!(enhanced.model.value, 42);
    }

    #[test]
    fn test_platform_info_detection() {
        let platform = PlatformInfo::detect();
        assert!(!platform.os.is_empty());
        assert!(!platform.arch.is_empty());
    }

    #[test]
    fn test_format_timestamp() {
        let timestamp = 1640995200; // 2022-01-01 00:00:00 UTC
        let formatted = format_timestamp(timestamp);
        assert!(formatted.contains("2022"));
    }
}
