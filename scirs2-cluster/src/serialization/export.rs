//! Advanced export functionality for clustering models
//!
//! This module provides sophisticated export capabilities including
//! multiple formats, metadata enrichment, and cross-platform compatibility.

use crate::error::{ClusteringError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "yaml")]
use serde_yaml;

use super::core::{EnhancedModelMetadata, PlatformInfo, SerializableModel};
use super::models::*;

/// Export formats supported by the serialization system
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format with full metadata
    Json,
    /// Compressed JSON format
    JsonGz,
    /// Binary format (MessagePack)
    Binary,
    /// CSV format (for simple models)
    Csv,
    /// YAML format
    Yaml,
    /// XML format
    Xml,
    /// HDF5 format (for large datasets)
    Hdf5,
    /// Custom format with user-defined structure
    Custom(String),
}

/// Trait for advanced export capabilities
pub trait AdvancedExport {
    /// Export model to specified format with metadata
    fn export_with_metadata(
        &self,
        format: ExportFormat,
        metadata: Option<ModelMetadata>,
    ) -> Result<Vec<u8>>;

    /// Export to file with automatic format detection
    fn export_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()>;

    /// Export model summary for quick inspection
    fn export_summary(&self) -> Result<String>;

    /// Export model in a format compatible with other libraries
    fn export_compatible(&self, target_library: &str) -> Result<Value>;

    /// Validate model before export
    fn validate_for_export(&self) -> Result<()>;
}

/// Comprehensive model metadata for exports
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelMetadata {
    /// Model name and version
    pub model_info: ModelInfo,
    /// Algorithm configuration
    pub algorithm_config: AlgorithmConfig,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Data characteristics
    pub data_characteristics: ModelDataCharacteristics,
    /// Export settings
    pub export_settings: ExportSettings,
}

/// Model information
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Creation timestamp
    pub created_at: String,
    /// Author/creator
    pub author: Option<String>,
    /// Description
    pub description: Option<String>,
}

/// Algorithm configuration details
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AlgorithmConfig {
    /// Algorithm name
    pub algorithm: String,
    /// Hyperparameters used
    pub hyperparameters: HashMap<String, Value>,
    /// Preprocessing steps applied
    pub preprocessing: Vec<String>,
    /// Random seed used
    pub random_seed: Option<u64>,
    /// Convergence criteria
    pub convergence_criteria: Option<HashMap<String, f64>>,
}

/// Performance metrics collected during training
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PerformanceMetrics {
    /// Training time in seconds
    pub training_time_seconds: f64,
    /// Memory usage in MB
    pub peak_memory_mb: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Model quality metrics
    pub quality_metrics: HashMap<String, f64>,
    /// Convergence information
    pub convergence_info: Option<ConvergenceInfo>,
}

/// Convergence information
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConvergenceInfo {
    /// Whether algorithm converged
    pub converged: bool,
    /// Number of iterations to convergence
    pub iterations: usize,
    /// Final objective value
    pub final_objective: f64,
    /// Convergence tolerance achieved
    pub tolerance_achieved: f64,
}

/// Data characteristics for model validation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelDataCharacteristics {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Feature names (if available)
    pub feature_names: Option<Vec<String>>,
    /// Data types for each feature
    pub feature_types: Option<Vec<String>>,
    /// Statistical summaries
    pub feature_statistics: Option<HashMap<String, FeatureStats>>,
}

/// Statistical summary for a feature
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FeatureStats {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Missing value count
    pub missing_count: usize,
}

/// Export settings and options
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExportSettings {
    /// Include raw model data
    pub include_raw_data: bool,
    /// Include training data
    pub include_training_data: bool,
    /// Compression level (0-9)
    pub compression_level: Option<u8>,
    /// Precision for floating point values
    pub float_precision: Option<usize>,
    /// Custom export options
    pub custom_options: HashMap<String, Value>,
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            include_raw_data: true,
            include_training_data: false,
            compression_level: None,
            float_precision: Some(6),
            custom_options: HashMap::new(),
        }
    }
}

/// Implementation of AdvancedExport for KMeansModel
impl AdvancedExport for KMeansModel {
    fn export_with_metadata(
        &self,
        format: ExportFormat,
        metadata: Option<ModelMetadata>,
    ) -> Result<Vec<u8>> {
        let export_data = KMeansExportData {
            model: self.clone(),
            metadata,
            format_version: "1.0".to_string(),
            export_timestamp: chrono::Utc::now().to_rfc3339(),
        };

        match format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(&export_data)
                    .map_err(|e| ClusteringError::InvalidInput(e.to_string()))?;
                Ok(json.into_bytes())
            }
            ExportFormat::JsonGz => {
                let json = serde_json::to_string(&export_data)
                    .map_err(|e| ClusteringError::InvalidInput(e.to_string()))?;

                use flate2::write::GzEncoder;
                use flate2::Compression;
                use std::io::Write;

                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                encoder
                    .write_all(json.as_bytes())
                    .map_err(|e| ClusteringError::InvalidInput(e.to_string()))?;
                encoder
                    .finish()
                    .map_err(|e| ClusteringError::InvalidInput(e.to_string()))
            }
            #[cfg(feature = "yaml")]
            ExportFormat::Yaml => {
                let yaml = serde_yaml::to_string(&export_data)
                    .map_err(|e| ClusteringError::InvalidInput(e.to_string()))?;
                Ok(yaml.into_bytes())
            }
            #[cfg(not(feature = "yaml"))]
            ExportFormat::Yaml => Err(ClusteringError::InvalidInput(
                "YAML support not enabled. Enable the 'yaml' feature".to_string(),
            )),
            ExportFormat::Csv => self.export_csv(),
            _ => Err(ClusteringError::InvalidInput(format!(
                "Unsupported export format: {:?}",
                format
            ))),
        }
    }

    fn export_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let format = detect_format_from_extension(path)?;
        let data = self.export_with_metadata(format, None)?;

        std::fs::write(path, data)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to write file: {}", e)))
    }

    fn export_summary(&self) -> Result<String> {
        let summary = KMeansSummary {
            algorithm: "K-Means".to_string(),
            n_clusters: self.n_clusters,
            n_features: self.centroids.ncols(),
            n_iterations: self.n_iter,
            inertia: self.inertia,
            has_labels: self.labels.is_some(),
        };

        serde_json::to_string_pretty(&summary)
            .map_err(|e| ClusteringError::InvalidInput(e.to_string()))
    }

    fn export_compatible(&self, target_library: &str) -> Result<Value> {
        match target_library.to_lowercase().as_str() {
            "sklearn" | "scikit-learn" => self.to_sklearn_format(),
            "tensorflow" | "tf" => self.to_tensorflow_format(),
            "pytorch" => self.to_pytorch_format(),
            _ => Err(ClusteringError::InvalidInput(format!(
                "Unsupported target library: {}",
                target_library
            ))),
        }
    }

    fn validate_for_export(&self) -> Result<()> {
        if self.centroids.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "Cannot export model with empty centroids".to_string(),
            ));
        }

        if self.n_clusters == 0 {
            return Err(ClusteringError::InvalidInput(
                "Cannot export model with zero clusters".to_string(),
            ));
        }

        if self.centroids.nrows() != self.n_clusters {
            return Err(ClusteringError::InvalidInput(
                "Centroids shape inconsistent with n_clusters".to_string(),
            ));
        }

        Ok(())
    }
}

/// Export data structure for K-Means
#[derive(Serialize, Deserialize, Debug, Clone)]
struct KMeansExportData {
    model: KMeansModel,
    metadata: Option<ModelMetadata>,
    format_version: String,
    export_timestamp: String,
}

/// Summary structure for K-Means
#[derive(Serialize, Deserialize, Debug, Clone)]
struct KMeansSummary {
    algorithm: String,
    n_clusters: usize,
    n_features: usize,
    n_iterations: usize,
    inertia: f64,
    has_labels: bool,
}

impl KMeansModel {
    /// Export as CSV format
    fn export_csv(&self) -> Result<Vec<u8>> {
        let mut csv_content = String::new();

        // Header
        csv_content.push_str("cluster_id");
        for i in 0..self.centroids.ncols() {
            csv_content.push_str(&format!(",feature_{}", i));
        }
        csv_content.push('\n');

        // Centroids data
        for (cluster_id, centroid) in self.centroids.rows().into_iter().enumerate() {
            csv_content.push_str(&cluster_id.to_string());
            for value in centroid {
                csv_content.push_str(&format!(",{:.6}", value));
            }
            csv_content.push('\n');
        }

        Ok(csv_content.into_bytes())
    }

    /// Convert to scikit-learn compatible format
    fn to_sklearn_format(&self) -> Result<Value> {
        use serde_json::json;

        Ok(json!({
            "cluster_centers_": self.centroids.as_slice().unwrap(),
            "labels_": self.labels.as_ref().map(|l| l.as_slice().unwrap()),
            "inertia_": self.inertia,
            "n_iter_": self.n_iter,
            "n_clusters": self.n_clusters,
            "_sklearn_version": "1.0.0"
        }))
    }

    /// Convert to TensorFlow compatible format
    fn to_tensorflow_format(&self) -> Result<Value> {
        use serde_json::json;

        Ok(json!({
            "centroids": {
                "data": self.centroids.as_slice().unwrap(),
                "shape": [self.centroids.nrows(), self.centroids.ncols()],
                "dtype": "float64"
            },
            "metadata": {
                "n_clusters": self.n_clusters,
                "inertia": self.inertia,
                "iterations": self.n_iter
            }
        }))
    }

    /// Convert to PyTorch compatible format
    fn to_pytorch_format(&self) -> Result<Value> {
        use serde_json::json;

        Ok(json!({
            "state_dict": {
                "centroids": self.centroids.as_slice().unwrap()
            },
            "hyperparameters": {
                "n_clusters": self.n_clusters
            },
            "metrics": {
                "inertia": self.inertia,
                "n_iter": self.n_iter
            }
        }))
    }
}

/// Detect export format from file extension
fn detect_format_from_extension<P: AsRef<Path>>(path: P) -> Result<ExportFormat> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase();

    match extension.as_str() {
        "json" => Ok(ExportFormat::Json),
        "gz" | "json.gz" => Ok(ExportFormat::JsonGz),
        "yaml" | "yml" => Ok(ExportFormat::Yaml),
        "csv" => Ok(ExportFormat::Csv),
        "xml" => Ok(ExportFormat::Xml),
        "h5" | "hdf5" => Ok(ExportFormat::Hdf5),
        _ => Err(ClusteringError::InvalidInput(format!(
            "Unknown file extension: {}",
            extension
        ))),
    }
}

/// Export utility functions
pub mod utils {
    use super::*;

    /// Create default metadata for a model
    pub fn create_default_metadata(algorithm_name: &str) -> ModelMetadata {
        ModelMetadata {
            model_info: ModelInfo {
                name: format!("{}_model", algorithm_name),
                version: "1.0.0".to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
                author: None,
                description: None,
            },
            algorithm_config: AlgorithmConfig {
                algorithm: algorithm_name.to_string(),
                hyperparameters: HashMap::new(),
                preprocessing: Vec::new(),
                random_seed: None,
                convergence_criteria: None,
            },
            performance_metrics: PerformanceMetrics {
                training_time_seconds: 0.0,
                peak_memory_mb: 0.0,
                cpu_utilization: 0.0,
                quality_metrics: HashMap::new(),
                convergence_info: None,
            },
            data_characteristics: ModelDataCharacteristics {
                n_samples: 0,
                n_features: 0,
                feature_names: None,
                feature_types: None,
                feature_statistics: None,
            },
            export_settings: ExportSettings::default(),
        }
    }

    /// Compress data using gzip
    pub fn compress_data(data: &[u8]) -> Result<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(data)
            .map_err(|e| ClusteringError::InvalidInput(e.to_string()))?;
        encoder
            .finish()
            .map_err(|e| ClusteringError::InvalidInput(e.to_string()))
    }

    /// Decompress gzip data
    pub fn decompress_data(compressed: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(compressed);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .map_err(|e| ClusteringError::InvalidInput(e.to_string()))?;
        Ok(decompressed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_kmeans_export_summary() {
        let centroids = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let model = KMeansModel::new(centroids, 2, 10, 0.5, None);

        let summary = model.export_summary().unwrap();
        assert!(summary.contains("K-Means"));
        assert!(summary.contains("\"n_clusters\": 2"));
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(
            detect_format_from_extension("model.json").unwrap(),
            ExportFormat::Json
        );
        assert_eq!(
            detect_format_from_extension("model.yaml").unwrap(),
            ExportFormat::Yaml
        );
        assert_eq!(
            detect_format_from_extension("model.csv").unwrap(),
            ExportFormat::Csv
        );
    }

    #[test]
    fn test_sklearn_compatibility() {
        let centroids = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let model = KMeansModel::new(centroids, 2, 10, 0.5, None);

        let sklearn_format = model.export_compatible("sklearn").unwrap();
        assert!(sklearn_format.get("cluster_centers_").is_some());
        assert!(sklearn_format.get("_sklearn_version").is_some());
    }
}
