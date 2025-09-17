//! Serialization and deserialization support for clustering models
//!
//! This module provides functionality to save and load clustering models,
//! including support for various formats and model types.

use crate::error::{ClusteringError, Result};
use crate::leader::{LeaderNode, LeaderTree};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use ndarray::{Array1, Array2, ArrayView2};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::convert::TryInto;
use std::fs::File;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid;

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

/// Enhanced model wrapper with integrity checking and versioning
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
        struct EnhancedModelHelper<T> {
            model: T,
            metadata: EnhancedModelMetadata,
        }

        let helper = EnhancedModelHelper::<T>::deserialize(deserializer)?;
        Ok(EnhancedModel {
            model: helper.model,
            metadata: helper.metadata,
        })
    }
}

impl<T: SerializableModel> EnhancedModel<T> {
    /// Create a new enhanced model wrapper
    pub fn new(model: T, metadata: EnhancedModelMetadata) -> Self {
        let mut enhanced = Self { model, metadata };
        enhanced.update_integrity_hash();
        enhanced
    }

    /// Update the integrity hash based on model contents
    pub fn update_integrity_hash(&mut self) {
        let model_json = serde_json::to_string(&self.model).unwrap_or_default();
        let mut hasher = DefaultHasher::new();
        model_json.hash(&mut hasher);
        self.metadata.integrity_hash = format!("{:x}", hasher.finish());
    }

    /// Validate model integrity
    pub fn validate_integrity(&self) -> Result<bool> {
        let model_json = serde_json::to_string(&self.model).map_err(|e| {
            ClusteringError::InvalidInput(format!("Failed to serialize for validation: {}", e))
        })?;
        let mut hasher = DefaultHasher::new();
        model_json.hash(&mut hasher);
        let computed_hash = format!("{:x}", hasher.finish());

        Ok(computed_hash == self.metadata.integrity_hash)
    }

    /// Check version compatibility
    pub fn check_version_compatibility(&self) -> Result<bool> {
        let current_version = env!("CARGO_PKG_VERSION");
        // Simple version compatibility check - can be enhanced with semantic versioning
        Ok(self.metadata.library_version == current_version
            || self.metadata.format_version.starts_with("1."))
    }

    /// Export model with enhanced metadata to file
    pub fn export_with_metadata<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to create file: {}", e)))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to export model: {}", e)))
    }

    /// Import model with enhanced metadata from file
    pub fn import_with_metadata<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to open file: {}", e)))?;
        let reader = BufReader::new(file);
        let enhanced_model: Self = serde_json::from_reader(reader)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to import model: {}", e)))?;

        // Validate integrity and compatibility
        if !enhanced_model.validate_integrity()? {
            return Err(ClusteringError::InvalidInput(
                "Model integrity validation failed".to_string(),
            ));
        }

        if !enhanced_model.check_version_compatibility()? {
            return Err(ClusteringError::InvalidInput(
                "Model version incompatible".to_string(),
            ));
        }

        Ok(enhanced_model)
    }
}

/// K-means model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KMeansModel {
    /// Cluster centroids
    pub centroids: Array2<f64>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Sum of squared distances
    pub inertia: f64,
    /// Cluster labels for training data (optional)
    pub labels: Option<Array1<usize>>,
}

impl SerializableModel for KMeansModel {}

impl KMeansModel {
    /// Create a new K-means model
    pub fn new(
        centroids: Array2<f64>,
        n_clusters: usize,
        n_iter: usize,
        inertia: f64,
        labels: Option<Array1<usize>>,
    ) -> Self {
        Self {
            centroids,
            n_clusters,
            n_iter,
            inertia,
            labels,
        }
    }

    /// Predict cluster labels for new data
    pub fn predict(&self, data: ArrayView2<f64>) -> Result<Array1<usize>> {
        let n_samples = data.nrows();
        let mut labels = Array1::zeros(n_samples);

        for (i, sample) in data.rows().into_iter().enumerate() {
            let mut min_distance = f64::INFINITY;
            let mut closest_cluster = 0;

            for (j, centroid) in self.centroids.rows().into_iter().enumerate() {
                // Calculate Euclidean distance
                let distance = sample
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if distance < min_distance {
                    min_distance = distance;
                    closest_cluster = j;
                }
            }

            labels[i] = closest_cluster;
        }

        Ok(labels)
    }
}

/// Hierarchical clustering result that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HierarchicalModel {
    /// Linkage matrix
    pub linkage: Array2<f64>,
    /// Number of original observations
    pub n_observations: usize,
    /// Method used for linkage
    pub method: String,
    /// Dendrogram labels (optional)
    pub labels: Option<Vec<String>>,
}

impl SerializableModel for HierarchicalModel {}

impl HierarchicalModel {
    /// Create a new hierarchical clustering model
    pub fn new(
        linkage: Array2<f64>,
        n_observations: usize,
        method: String,
        labels: Option<Vec<String>>,
    ) -> Self {
        Self {
            linkage,
            n_observations,
            method,
            labels,
        }
    }

    /// Export dendrogram to Newick format
    pub fn to_newick(&self) -> Result<String> {
        let mut newick = String::new();
        let nnodes = self.linkage.nrows();

        if nnodes == 0 {
            return Ok("();".to_string());
        }

        // Validate linkage matrix before processing
        self.validate_linkage_matrix()?;

        // Build the tree structure
        self.build_newick_recursive(nnodes + self.n_observations - 1, &mut newick)?;

        newick.push_str(";");
        Ok(newick)
    }

    /// Export dendrogram to Newick format with custom formatting options
    pub fn to_newick_with_options(
        &self,
        include_distances: bool,
        precision: usize,
    ) -> Result<String> {
        let mut newick = String::new();
        let nnodes = self.linkage.nrows();

        if nnodes == 0 {
            return Ok("();".to_string());
        }

        self.validate_linkage_matrix()?;
        self.build_newick_recursive_formatted(
            nnodes + self.n_observations - 1,
            &mut newick,
            include_distances,
            precision,
        )?;

        newick.push_str(";");
        Ok(newick)
    }

    /// Validate linkage matrix for consistency
    fn validate_linkage_matrix(&self) -> Result<()> {
        let nnodes = self.linkage.nrows();

        for i in 0..nnodes {
            let left = self.linkage[[i, 0]] as usize;
            let right = self.linkage[[i, 1]] as usize;
            let distance = self.linkage[[i, 2]];

            // Check that node indices are valid
            if left >= self.n_observations + i || right >= self.n_observations + i {
                return Err(ClusteringError::InvalidInput(format!(
                    "Invalid node indices in linkage matrix at row {}: left={}, right={}",
                    i, left, right
                )));
            }

            // Check that distance is non-negative
            if distance < 0.0 {
                return Err(ClusteringError::InvalidInput(format!(
                    "Negative distance in linkage matrix at row {}: {}",
                    i, distance
                )));
            }
        }

        Ok(())
    }

    /// Build Newick string recursively with formatting options
    fn build_newick_recursive_formatted(
        &self,
        nodeidx: usize,
        newick: &mut String,
        include_distances: bool,
        precision: usize,
    ) -> Result<()> {
        if nodeidx < self.n_observations {
            // Leaf node
            if let Some(ref labels) = self.labels {
                newick.push_str(&labels[nodeidx]);
            } else {
                newick.push_str(&nodeidx.to_string());
            }
        } else {
            // Internal node
            let row_idx = nodeidx - self.n_observations;
            if row_idx >= self.linkage.nrows() {
                return Err(ClusteringError::InvalidInput(
                    "Invalid node index".to_string(),
                ));
            }

            let left = self.linkage[[row_idx, 0]] as usize;
            let right = self.linkage[[row_idx, 1]] as usize;
            let distance = self.linkage[[row_idx, 2]];

            newick.push('(');
            self.build_newick_recursive_formatted(left, newick, include_distances, precision)?;

            if include_distances {
                newick.push(':');
                newick.push_str(&format!(
                    "{:.precision$}",
                    distance / 2.0,
                    precision = precision
                ));
            }

            newick.push(',');
            self.build_newick_recursive_formatted(right, newick, include_distances, precision)?;

            if include_distances {
                newick.push(':');
                newick.push_str(&format!(
                    "{:.precision$}",
                    distance / 2.0,
                    precision = precision
                ));
            }

            newick.push(')');
        }

        Ok(())
    }

    fn build_newick_recursive(&self, nodeidx: usize, newick: &mut String) -> Result<()> {
        if nodeidx < self.n_observations {
            // Leaf node
            if let Some(ref labels) = self.labels {
                newick.push_str(&labels[nodeidx]);
            } else {
                newick.push_str(&nodeidx.to_string());
            }
        } else {
            // Internal node
            let row_idx = nodeidx - self.n_observations;
            if row_idx >= self.linkage.nrows() {
                return Err(ClusteringError::InvalidInput(
                    "Invalid node index".to_string(),
                ));
            }

            let left = self.linkage[[row_idx, 0]] as usize;
            let right = self.linkage[[row_idx, 1]] as usize;
            let distance = self.linkage[[row_idx, 2]];

            newick.push('(');
            self.build_newick_recursive(left, newick)?;
            newick.push(':');
            newick.push_str(&format!("{:.6}", distance / 2.0));
            newick.push(',');
            self.build_newick_recursive(right, newick)?;
            newick.push(':');
            newick.push_str(&format!("{:.6}", distance / 2.0));
            newick.push(')');
        }

        Ok(())
    }

    /// Export dendrogram to JSON format
    pub fn to_jsontree(&self) -> Result<serde_json::Value> {
        use serde_json::json;

        let nnodes = self.linkage.nrows();
        if nnodes == 0 {
            return Ok(json!({}));
        }

        self.build_json_recursive(nnodes + self.n_observations - 1)
    }

    fn build_json_recursive(&self, nodeidx: usize) -> Result<serde_json::Value> {
        use serde_json::json;

        if nodeidx < self.n_observations {
            // Leaf node
            let name = if let Some(ref labels) = self.labels {
                labels[nodeidx].clone()
            } else {
                nodeidx.to_string()
            };

            Ok(json!({
                "name": name,
                "type": "leaf",
                "index": nodeidx
            }))
        } else {
            // Internal node
            let row_idx = nodeidx - self.n_observations;
            if row_idx >= self.linkage.nrows() {
                return Err(ClusteringError::InvalidInput(
                    "Invalid node index".to_string(),
                ));
            }

            let left = self.linkage[[row_idx, 0]] as usize;
            let right = self.linkage[[row_idx, 1]] as usize;
            let distance = self.linkage[[row_idx, 2]];

            let left_child = self.build_json_recursive(left)?;
            let right_child = self.build_json_recursive(right)?;

            Ok(json!({
                "type": "internal",
                "distance": distance,
                "children": [left_child, right_child]
            }))
        }
    }

    /// Export dendrogram to GraphML format for network analysis tools
    pub fn to_graphml(&self) -> Result<String> {
        let mut graphml = String::new();

        // XML header and GraphML namespace
        graphml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        graphml.push_str("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" ");
        graphml.push_str("xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" ");
        graphml.push_str("xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns ");
        graphml.push_str("http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n");

        // Define attributes
        graphml.push_str(
            "  <key id=\"label\" for=\"node\" attr.name=\"label\" attr.type=\"string\"/>\n",
        );
        graphml.push_str(
            "  <key id=\"distance\" for=\"edge\" attr.name=\"distance\" attr.type=\"double\"/>\n",
        );
        graphml.push_str(
            "  <key id=\"type\" for=\"node\" attr.name=\"type\" attr.type=\"string\"/>\n",
        );

        graphml.push_str("  <graph id=\"dendrogram\" edgedefault=\"undirected\">\n");

        // Add nodes (leaves and internal nodes)
        for i in 0..self.n_observations {
            let label = if let Some(ref labels) = self.labels {
                labels[i].clone()
            } else {
                format!("leaf_{}", i)
            };

            graphml.push_str(&format!("    <node id=\"{}\">\n", i));
            graphml.push_str(&format!("      <data key=\"label\">{}</data>\n", label));
            graphml.push_str("      <data key=\"type\">leaf</data>\n");
            graphml.push_str("    </node>\n");
        }

        // Add internal nodes and edges
        for i in 0..self.linkage.nrows() {
            let internal_id = self.n_observations + i;
            let left = self.linkage[[i, 0]] as usize;
            let right = self.linkage[[i, 1]] as usize;
            let distance = self.linkage[[i, 2]];

            // Internal node
            graphml.push_str(&format!("    <node id=\"{}\">\n", internal_id));
            graphml.push_str(&format!(
                "      <data key=\"label\">internal_{}</data>\n",
                i
            ));
            graphml.push_str("      <data key=\"type\">internal</data>\n");
            graphml.push_str("    </node>\n");

            // Edges
            graphml.push_str(&format!(
                "    <edge source=\"{}\" target=\"{}\">\n",
                internal_id, left
            ));
            graphml.push_str(&format!(
                "      <data key=\"distance\">{}</data>\n",
                distance
            ));
            graphml.push_str("    </edge>\n");

            graphml.push_str(&format!(
                "    <edge source=\"{}\" target=\"{}\">\n",
                internal_id, right
            ));
            graphml.push_str(&format!(
                "      <data key=\"distance\">{}</data>\n",
                distance
            ));
            graphml.push_str("    </edge>\n");
        }

        graphml.push_str("  </graph>\n");
        graphml.push_str("</graphml>\n");

        Ok(graphml)
    }

    /// Export dendrogram to DOT format for Graphviz
    pub fn to_dot(&self) -> Result<String> {
        let mut dot = String::new();

        dot.push_str("graph dendrogram {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=circle];\n");

        // Add leaf nodes with special styling
        for i in 0..self.n_observations {
            let label = if let Some(ref labels) = self.labels {
                labels[i].clone()
            } else {
                format!("L{}", i)
            };

            dot.push_str(&format!(
                "  {} [label=\"{}\" shape=box style=filled fillcolor=lightblue];\n",
                i, label
            ));
        }

        // Add internal nodes and edges
        for i in 0..self.linkage.nrows() {
            let internal_id = self.n_observations + i;
            let left = self.linkage[[i, 0]] as usize;
            let right = self.linkage[[i, 1]] as usize;
            let distance = self.linkage[[i, 2]];

            // Internal node
            dot.push_str(&format!(
                "  {} [label=\"{:.3}\" style=filled fillcolor=lightcoral];\n",
                internal_id, distance
            ));

            // Edges with distance labels
            dot.push_str(&format!(
                "  {} -- {} [label=\"{:.3}\"];\n",
                internal_id, left, distance
            ));
            dot.push_str(&format!(
                "  {} -- {} [label=\"{:.3}\"];\n",
                internal_id, right, distance
            ));
        }

        dot.push_str("}\n");

        Ok(dot)
    }
}

/// DBSCAN model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DBSCANModel {
    /// Core sample indices
    pub core_sample_indices: Array1<usize>,
    /// Cluster labels
    pub labels: Array1<i32>,
    /// Epsilon parameter
    pub eps: f64,
    /// Min samples parameter
    pub min_samples: usize,
}

impl SerializableModel for DBSCANModel {}

impl DBSCANModel {
    /// Create a new DBSCAN model
    pub fn new(
        core_sample_indices: Array1<usize>,
        labels: Array1<i32>,
        eps: f64,
        min_samples: usize,
    ) -> Self {
        Self {
            core_sample_indices,
            labels,
            eps,
            min_samples,
        }
    }
}

/// Mean Shift model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MeanShiftModel {
    /// Cluster centers
    pub cluster_centers: Array2<f64>,
    /// Bandwidth parameter
    pub bandwidth: f64,
    /// Cluster labels (optional)
    pub labels: Option<Array1<usize>>,
}

impl SerializableModel for MeanShiftModel {}

impl MeanShiftModel {
    /// Create a new Mean Shift model
    pub fn new(
        cluster_centers: Array2<f64>,
        bandwidth: f64,
        labels: Option<Array1<usize>>,
    ) -> Self {
        Self {
            cluster_centers,
            bandwidth,
            labels,
        }
    }
}

/// Leader algorithm model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LeaderModel {
    /// Cluster leaders
    pub leaders: Array2<f64>,
    /// Distance threshold
    pub threshold: f64,
    /// Cluster labels (optional)
    pub labels: Option<Array1<usize>>,
}

impl SerializableModel for LeaderModel {}

impl LeaderModel {
    /// Create a new Leader model
    pub fn new(leaders: Array2<f64>, threshold: f64, labels: Option<Array1<usize>>) -> Self {
        Self {
            leaders,
            threshold,
            labels,
        }
    }

    /// Predict cluster labels for new data using Euclidean distance
    pub fn predict(&self, data: ArrayView2<f64>) -> Result<Array1<usize>> {
        let n_samples = data.nrows();
        let mut labels = Array1::zeros(n_samples);

        for (i, sample) in data.rows().into_iter().enumerate() {
            let mut min_distance = f64::INFINITY;
            let mut closest_leader = 0;

            for (j, leader) in self.leaders.rows().into_iter().enumerate() {
                // Calculate Euclidean distance
                let distance = sample
                    .iter()
                    .zip(leader.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if distance < min_distance {
                    min_distance = distance;
                    closest_leader = j;
                }
            }

            labels[i] = closest_leader;
        }

        Ok(labels)
    }
}

/// Leader tree model that can be serialized (hierarchical leader algorithm)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LeaderTreeModel {
    /// Root nodes of the tree
    pub roots: Vec<LeaderNodeModel>,
    /// Distance threshold for this level
    pub threshold: f64,
    /// Thresholds used for hierarchical clustering
    pub thresholds: Vec<f64>,
}

/// Affinity Propagation model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AffinityPropagationModel {
    /// Indices of exemplars (cluster centers)
    pub exemplars: Vec<usize>,
    /// Cluster labels
    pub labels: Array1<i32>,
    /// Damping factor used
    pub damping: f64,
    /// Preference value used
    pub preference: Option<f64>,
    /// Number of iterations performed
    pub n_iter: usize,
}

/// BIRCH model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BirchModel {
    /// Cluster centroids
    pub centroids: Array2<f64>,
    /// Cluster labels
    pub labels: Array1<i32>,
    /// Threshold parameter
    pub threshold: f64,
    /// Branching factor
    pub branching_factor: usize,
    /// Number of leaf clusters
    pub n_clusters: usize,
}

/// Gaussian Mixture Model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GMMModel {
    /// Component weights
    pub weights: Array1<f64>,
    /// Component means
    pub means: Array2<f64>,
    /// Component covariances
    pub covariances: Vec<Array2<f64>>,
    /// Cluster labels (optional, from prediction)
    pub labels: Option<Array1<i32>>,
    /// Number of components
    pub n_components: usize,
    /// Covariance type
    pub covariance_type: String,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Log-likelihood of the model
    pub log_likelihood: f64,
}

/// Spectral Clustering model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SpectralClusteringModel {
    /// Spectral embeddings
    pub embeddings: Array2<f64>,
    /// Cluster labels
    pub labels: Array1<usize>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Affinity mode used
    pub affinity_mode: String,
    /// Gamma parameter (for RBF affinity)
    pub gamma: Option<f64>,
}

impl SerializableModel for AffinityPropagationModel {}

impl SerializableModel for BirchModel {}

impl SerializableModel for GMMModel {}

impl SerializableModel for SpectralClusteringModel {}

/// Serializable version of LeaderNode
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LeaderNodeModel {
    /// The leader vector
    pub leader: Array1<f64>,
    /// Child nodes
    pub children: Vec<LeaderNodeModel>,
    /// Indices of data points in this cluster
    pub members: Vec<usize>,
}

impl SerializableModel for LeaderTreeModel {}

impl LeaderTreeModel {
    /// Create a new Leader tree model
    pub fn new(roots: Vec<LeaderNodeModel>, threshold: f64, thresholds: Vec<f64>) -> Self {
        Self {
            roots,
            threshold,
            thresholds,
        }
    }

    /// Get the total number of nodes in the tree
    pub fn node_count(&self) -> usize {
        self.roots.iter().map(|root| Self::countnodes(root)).sum()
    }

    fn countnodes(node: &LeaderNodeModel) -> usize {
        1 + node
            .children
            .iter()
            .map(|child| Self::countnodes(child))
            .sum::<usize>()
    }

    /// Convert from LeaderTree
    pub fn from_leadertree<F: num_traits::Float>(tree: &LeaderTree<F>) -> Self
    where
        f64: From<F>,
    {
        let roots = tree
            .roots
            .iter()
            .map(|node| Self::convertnode(node))
            .collect();

        Self {
            roots,
            threshold: tree.threshold.to_f64().unwrap_or(0.0),
            thresholds: vec![tree.threshold.to_f64().unwrap_or(0.0)],
        }
    }

    fn convertnode<F: num_traits::Float>(node: &LeaderNode<F>) -> LeaderNodeModel
    where
        f64: From<F>,
    {
        LeaderNodeModel {
            leader: node.leader.mapv(|x| x.to_f64().unwrap_or(0.0)),
            children: node
                .children
                .iter()
                .map(|child| Self::convertnode(child))
                .collect(),
            members: node.members.clone(),
        }
    }
}

/// Convert K-means output to a serializable model
#[allow(dead_code)]
pub fn kmeans_to_model(
    centroids: Array2<f64>,
    labels: Array1<usize>,
    n_iter: usize,
) -> KMeansModel {
    // Calculate inertia (sum of squared distances)
    let inertia = calculate_inertia(&centroids, &labels);
    let n_clusters = centroids.nrows();

    KMeansModel::new(centroids, n_clusters, n_iter, inertia, Some(labels))
}

/// Convert hierarchical clustering output to a serializable model
#[allow(dead_code)]
pub fn hierarchy_to_model(
    linkage: Array2<f64>,
    n_observations: usize,
    method: &str,
    labels: Option<Vec<String>>,
) -> HierarchicalModel {
    HierarchicalModel::new(linkage, n_observations, method.to_string(), labels)
}

/// Convert DBSCAN output to a serializable model
#[allow(dead_code)]
pub fn dbscan_to_model(
    core_sample_indices: Array1<usize>,
    labels: Array1<i32>,
    eps: f64,
    min_samples: usize,
) -> DBSCANModel {
    DBSCANModel::new(core_sample_indices, labels, eps, min_samples)
}

/// Convert Mean Shift output to a serializable model
#[allow(dead_code)]
pub fn meanshift_to_model(
    cluster_centers: Array2<f64>,
    bandwidth: f64,
    labels: Option<Array1<usize>>,
) -> MeanShiftModel {
    MeanShiftModel::new(cluster_centers, bandwidth, labels)
}

/// Convert Leader algorithm output to a serializable model
#[allow(dead_code)]
pub fn leader_to_model(
    leaders: Array2<f64>,
    threshold: f64,
    labels: Option<Array1<usize>>,
) -> LeaderModel {
    LeaderModel::new(leaders, threshold, labels)
}

/// Convert Leader tree to a serializable model
#[allow(dead_code)]
pub fn leadertree_to_model<F: num_traits::Float>(tree: &LeaderTree<F>) -> LeaderTreeModel
where
    f64: From<F>,
{
    LeaderTreeModel::from_leadertree(tree)
}

/// Convert Affinity Propagation output to a serializable model
#[allow(dead_code)]
pub fn affinity_propagation_to_model(
    exemplars: Vec<usize>,
    labels: Array1<i32>,
    damping: f64,
    preference: Option<f64>,
    n_iter: usize,
) -> AffinityPropagationModel {
    AffinityPropagationModel {
        exemplars,
        labels,
        damping,
        preference,
        n_iter,
    }
}

/// Convert BIRCH output to a serializable model
#[allow(dead_code)]
pub fn birch_to_model(
    centroids: Array2<f64>,
    labels: Array1<i32>,
    threshold: f64,
    branching_factor: usize,
) -> BirchModel {
    BirchModel {
        centroids: centroids.clone(),
        labels,
        threshold,
        branching_factor,
        n_clusters: centroids.nrows(),
    }
}

/// Convert GMM output to a serializable model
#[allow(dead_code)]
pub fn gmm_to_model(
    weights: Array1<f64>,
    means: Array2<f64>,
    covariances: Vec<Array2<f64>>,
    labels: Option<Array1<i32>>,
    covariance_type: String,
    n_iter: usize,
    log_likelihood: f64,
) -> GMMModel {
    let n_components = weights.len();
    GMMModel {
        weights,
        means,
        covariances,
        labels,
        n_components,
        covariance_type,
        n_iter,
        log_likelihood,
    }
}

/// Convert Spectral Clustering output to a serializable model
#[allow(dead_code)]
pub fn spectral_clustering_to_model(
    embeddings: Array2<f64>,
    labels: Array1<usize>,
    n_clusters: usize,
    affinity_mode: String,
    gamma: Option<f64>,
) -> SpectralClusteringModel {
    SpectralClusteringModel {
        embeddings,
        labels,
        n_clusters,
        affinity_mode,
        gamma,
    }
}

#[allow(dead_code)]
fn calculate_inertia(_centroids: &Array2<f64>, labels: &Array1<usize>) -> f64 {
    // This is a placeholder - in practice, we'd need the original data
    // to calculate the actual inertia
    0.0
}

/// Get current Unix timestamp
#[allow(dead_code)]
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Format Unix timestamp to human-readable string
#[allow(dead_code)]
fn format_timestamp(timestamp: u64) -> String {
    // Simple formatting without chrono dependency
    // In production, you might want to use a proper date library
    format!("1970-01-01T00:00:00Z+{}s", timestamp)
}

/// Enhanced serialization utilities
pub mod enhanced {
    use super::*;

    /// Serialize with multiple format support
    pub fn serialize_with_format<T: Serialize>(data: &T, format: ExportFormat) -> Result<Vec<u8>> {
        match format {
            ExportFormat::Json => serde_json::to_vec_pretty(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("JSON serialization failed: {}", e))
            }),
            ExportFormat::Binary => bincode::serialize(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("Binary serialization failed: {}", e))
            }),
            ExportFormat::CompressedJson => {
                let jsondata = serde_json::to_vec_pretty(data).map_err(|e| {
                    ClusteringError::InvalidInput(format!("JSON serialization failed: {}", e))
                })?;

                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(&jsondata).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Compression failed: {}", e))
                })?;
                encoder.finish().map_err(|e| {
                    ClusteringError::InvalidInput(format!("Compression finalization failed: {}", e))
                })
            }
            #[cfg(feature = "yaml")]
            ExportFormat::Yaml => serde_yaml::to_vec(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("YAML serialization failed: {}", e))
            }),
            #[cfg(feature = "msgpack")]
            ExportFormat::MessagePack => rmp_serde::to_vec(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("MessagePack serialization failed: {}", e))
            }),
            #[cfg(feature = "cbor")]
            ExportFormat::Cbor => serde_cbor::to_vec(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("CBOR serialization failed: {}", e))
            }),
            _ => Err(ClusteringError::InvalidInput(
                "Unsupported format for this data type".to_string(),
            )),
        }
    }

    /// Deserialize with multiple format support
    pub fn deserialize_with_format<T: for<'de> Deserialize<'de>>(
        data: &[u8],
        format: ExportFormat,
    ) -> Result<T> {
        match format {
            ExportFormat::Json => serde_json::from_slice(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("JSON deserialization failed: {}", e))
            }),
            ExportFormat::Binary => bincode::deserialize(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("Binary deserialization failed: {}", e))
            }),
            ExportFormat::CompressedJson => {
                let mut decoder = GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Decompression failed: {}", e))
                })?;

                serde_json::from_slice(&decompressed).map_err(|e| {
                    ClusteringError::InvalidInput(format!("JSON deserialization failed: {}", e))
                })
            }
            #[cfg(feature = "yaml")]
            ExportFormat::Yaml => serde_yaml::from_slice(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("YAML deserialization failed: {}", e))
            }),
            #[cfg(feature = "msgpack")]
            ExportFormat::MessagePack => rmp_serde::from_slice(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("MessagePack deserialization failed: {}", e))
            }),
            #[cfg(feature = "cbor")]
            ExportFormat::Cbor => serde_cbor::from_slice(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("CBOR deserialization failed: {}", e))
            }),
            _ => Err(ClusteringError::InvalidInput(
                "Unsupported format for deserialization".to_string(),
            )),
        }
    }

    /// Model versioning support
    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct VersionedModel<T> {
        /// Model format version
        pub version: String,
        /// Backward compatibility info
        pub compatibility: Vec<String>,
        /// Migration notes
        pub migration_notes: Option<String>,
        /// The actual model data
        pub data: T,
        /// Metadata
        pub metadata: ModelMetadata,
    }

    impl<T: Serialize + for<'de> Deserialize<'de>> VersionedModel<T> {
        /// Create a new versioned model
        pub fn new(data: T, metadata: ModelMetadata) -> Self {
            Self {
                version: "1.0.0".to_string(),
                compatibility: vec!["1.0.0".to_string()],
                migration_notes: None,
                data,
                metadata,
            }
        }

        /// Check if model is compatible with current version
        pub fn is_compatible(&self, target_version: &str) -> bool {
            self.compatibility.contains(&target_version.to_string())
        }

        /// Migrate model to newer version (placeholder)
        pub fn migrate_to(&mut self, target_version: &str) -> Result<()> {
            if self.is_compatible(target_version) {
                Ok(())
            } else {
                Err(ClusteringError::InvalidInput(format!(
                    "Cannot migrate from {} to {}",
                    self.version, target_version
                )))
            }
        }
    }
}

/// Performance monitoring for serialization
pub mod performance {
    use super::*;
    use std::time::Instant;

    /// Benchmark serialization performance
    pub fn benchmark_serialization<T: Serialize>(
        data: &T,
        formats: &[ExportFormat],
    ) -> Result<std::collections::HashMap<ExportFormat, (u64, usize)>> {
        let mut results = std::collections::HashMap::new();

        for &format in formats {
            let start = Instant::now();
            let serialized = enhanced::serialize_with_format(data, format)?;
            let duration = start.elapsed().as_micros() as u64;
            let size = serialized.len();

            results.insert(format, (duration, size));
        }

        Ok(results)
    }

    /// Compression ratio analysis
    pub fn analyze_compression<T: Serialize>(data: &T) -> Result<(f64, usize, usize)> {
        let uncompressed = enhanced::serialize_with_format(data, ExportFormat::Json)?;
        let compressed = enhanced::serialize_with_format(data, ExportFormat::CompressedJson)?;

        let ratio = compressed.len() as f64 / uncompressed.len() as f64;
        Ok((ratio, uncompressed.len(), compressed.len()))
    }
}

/// Convenience function to save K-means results directly
#[allow(dead_code)]
pub fn save_kmeans<P: AsRef<Path>>(
    path: P,
    centroids: Array2<f64>,
    labels: Array1<usize>,
    n_iter: usize,
) -> Result<()> {
    let model = kmeans_to_model(centroids, labels, n_iter);
    model.save_to_file(path)
}

/// Convenience function to save hierarchical clustering results directly
#[allow(dead_code)]
pub fn save_hierarchy<P: AsRef<Path>>(
    path: P,
    linkage: Array2<f64>,
    n_observations: usize,
    method: &str,
    labels: Option<Vec<String>>,
) -> Result<()> {
    let model = hierarchy_to_model(linkage, n_observations, method, labels);
    model.save_to_file(path)
}

/// Convenience function to save Leader algorithm results directly
#[allow(dead_code)]
pub fn save_leader<P: AsRef<Path>>(
    path: P,
    leaders: Array2<f64>,
    threshold: f64,
    labels: Option<Array1<usize>>,
) -> Result<()> {
    let model = leader_to_model(leaders, threshold, labels);
    model.save_to_file(path)
}

/// Convenience function to save Leader tree results directly
#[allow(dead_code)]
pub fn save_leadertree<P: AsRef<Path>, F: num_traits::Float>(
    path: P,
    tree: &LeaderTree<F>,
) -> Result<()>
where
    f64: From<F>,
{
    let model = leadertree_to_model(tree);
    model.save_to_file(path)
}

/// Convenience function to save Affinity Propagation results directly
#[allow(dead_code)]
pub fn save_affinity_propagation<P: AsRef<Path>>(
    path: P,
    exemplars: Vec<usize>,
    labels: Array1<i32>,
    damping: f64,
    preference: Option<f64>,
    n_iter: usize,
) -> Result<()> {
    let model = affinity_propagation_to_model(exemplars, labels, damping, preference, n_iter);
    model.save_to_file(path)
}

/// Convenience function to save BIRCH results directly
#[allow(dead_code)]
pub fn save_birch<P: AsRef<Path>>(
    path: P,
    centroids: Array2<f64>,
    labels: Array1<i32>,
    threshold: f64,
    branching_factor: usize,
) -> Result<()> {
    let model = birch_to_model(centroids, labels, threshold, branching_factor);
    model.save_to_file(path)
}

/// Convenience function to save GMM results directly
#[allow(dead_code)]
pub fn save_gmm<P: AsRef<Path>>(
    path: P,
    weights: Array1<f64>,
    means: Array2<f64>,
    covariances: Vec<Array2<f64>>,
    labels: Option<Array1<i32>>,
    covariance_type: String,
    n_iter: usize,
    log_likelihood: f64,
) -> Result<()> {
    let model = gmm_to_model(
        weights,
        means,
        covariances,
        labels,
        covariance_type,
        n_iter,
        log_likelihood,
    );
    model.save_to_file(path)
}

/// Convenience function to save Spectral Clustering results directly
#[allow(dead_code)]
pub fn save_spectral_clustering<P: AsRef<Path>>(
    path: P,
    embeddings: Array2<f64>,
    labels: Array1<usize>,
    n_clusters: usize,
    affinity_mode: String,
    gamma: Option<f64>,
) -> Result<()> {
    let model = spectral_clustering_to_model(embeddings, labels, n_clusters, affinity_mode, gamma);
    model.save_to_file(path)
}

/// Export formats for clustering models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// YAML format  
    Yaml,
    /// CSV format (for simple data like centroids)
    Csv,
    /// Newick format (for dendrograms)
    Newick,
    /// Pickle-like binary format
    Binary,
    /// Compressed JSON format
    CompressedJson,
    /// MessagePack format
    MessagePack,
    /// CBOR format
    Cbor,
}

/// Enhanced export functionality for clustering models
pub trait AdvancedExport {
    /// Export model in specified format
    fn export<P: AsRef<Path>>(&self, path: P, format: ExportFormat) -> Result<()>;

    /// Export to string in specified format
    fn export_to_string(&self, format: ExportFormat) -> Result<String>;

    /// Get model metadata
    fn get_metadata(&self) -> ModelMetadata;
}

/// Metadata about a clustering model
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelMetadata {
    /// Model type (e.g., "kmeans", "hierarchical", "dbscan")
    pub model_type: String,
    /// Creation timestamp (Unix timestamp)
    pub created_at: u64,
    /// Human-readable creation time
    pub created_at_readable: String,
    /// Number of clusters/components
    pub n_clusters: Option<usize>,
    /// Number of features
    pub n_features: Option<usize>,
    /// Additional parameters
    pub parameters: std::collections::HashMap<String, String>,
    /// Model version/format version
    pub version: String,
    /// Algorithm configuration used
    pub algorithm_config: Option<AlgorithmConfig>,
    /// Performance metrics
    pub performance_metrics: Option<PerformanceMetrics>,
    /// Training data characteristics
    pub data_characteristics: Option<ModelDataCharacteristics>,
}

/// Algorithm configuration metadata
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AlgorithmConfig {
    /// Algorithm name
    pub algorithm: String,
    /// Hyperparameters used
    pub hyperparameters: HashMap<String, serde_json::Value>,
    /// Distance metric used
    pub distance_metric: Option<String>,
    /// Linkage method (for hierarchical clustering)
    pub linkage_method: Option<String>,
    /// Initialization method
    pub initialization_method: Option<String>,
    /// Convergence criteria
    pub convergence_criteria: Option<HashMap<String, f64>>,
}

/// Performance metrics for the model
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PerformanceMetrics {
    /// Training time in seconds
    pub training_time_seconds: Option<f64>,
    /// Memory usage in bytes
    pub memory_usage_bytes: Option<usize>,
    /// Number of iterations to convergence
    pub iterations_to_convergence: Option<usize>,
    /// Final inertia/objective value
    pub final_objective_value: Option<f64>,
    /// Silhouette score (if available)
    pub silhouette_score: Option<f64>,
    /// Davies-Bouldin index (if available)
    pub davies_bouldin_index: Option<f64>,
    /// Custom metrics
    pub custom_metrics: Option<HashMap<String, f64>>,
}

/// Model data characteristics metadata
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelDataCharacteristics {
    /// Number of samples in training data
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Data type (continuous, discrete, mixed)
    pub data_type: Option<String>,
    /// Missing value percentage
    pub missing_value_percentage: Option<f64>,
    /// Feature scaling applied
    pub feature_scaling: Option<String>,
    /// Data preprocessing steps
    pub preprocessing_steps: Option<Vec<String>>,
    /// Statistical summary
    pub statistical_summary: Option<HashMap<String, f64>>,
}

impl AdvancedExport for KMeansModel {
    fn export<P: AsRef<Path>>(&self, path: P, format: ExportFormat) -> Result<()> {
        match format {
            ExportFormat::Json => self.save_to_file(path),
            ExportFormat::CompressedJson => self.save_to_file_compressed(path),
            ExportFormat::Csv => {
                let path = path.as_ref();
                let mut file = File::create(path).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to create file: {}", e))
                })?;

                // Write centroids as CSV
                for (i, row) in self.centroids.rows().into_iter().enumerate() {
                    if i > 0 {
                        writeln!(file)?;
                    }
                    let row_str: Vec<String> = row.iter().map(|x| x.to_string()).collect();
                    write!(file, "{}", row_str.join(","))?;
                }
                Ok(())
            }
            ExportFormat::Binary => {
                let data = enhanced::serialize_with_format(self, format)?;
                std::fs::write(path, data).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to write file: {}", e))
                })
            }
            #[cfg(feature = "yaml")]
            ExportFormat::Yaml => {
                let data = enhanced::serialize_with_format(self, format)?;
                std::fs::write(path, data).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to write file: {}", e))
                })
            }
            #[cfg(feature = "msgpack")]
            ExportFormat::MessagePack => {
                let data = enhanced::serialize_with_format(self, format)?;
                std::fs::write(path, data).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to write file: {}", e))
                })
            }
            #[cfg(feature = "cbor")]
            ExportFormat::Cbor => {
                let data = enhanced::serialize_with_format(self, format)?;
                std::fs::write(path, data).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to write file: {}", e))
                })
            }
            _ => Err(ClusteringError::InvalidInput(
                "Format not supported for K-means model".to_string(),
            )),
        }
    }

    fn export_to_string(&self, format: ExportFormat) -> Result<String> {
        match format {
            ExportFormat::Json => serde_json::to_string_pretty(self)
                .map_err(|e| ClusteringError::InvalidInput(format!("Failed to serialize: {}", e))),
            ExportFormat::Csv => {
                let mut result = String::new();
                for (i, row) in self.centroids.rows().into_iter().enumerate() {
                    if i > 0 {
                        result.push('\n');
                    }
                    let row_str: Vec<String> = row.iter().map(|x| x.to_string()).collect();
                    result.push_str(&row_str.join(","));
                }
                Ok(result)
            }
            #[cfg(feature = "yaml")]
            ExportFormat::Yaml => {
                let data = enhanced::serialize_with_format(self, format)?;
                String::from_utf8(data).map_err(|e| {
                    ClusteringError::InvalidInput(format!("UTF-8 conversion failed: {}", e))
                })
            }
            _ => Err(ClusteringError::InvalidInput(
                "Format not supported for string export".to_string(),
            )),
        }
    }

    fn get_metadata(&self) -> ModelMetadata {
        let mut parameters = std::collections::HashMap::new();
        parameters.insert("n_iter".to_string(), self.n_iter.to_string());
        parameters.insert("inertia".to_string(), self.inertia.to_string());

        let mut hyperparameters = HashMap::new();
        hyperparameters.insert(
            "n_clusters".to_string(),
            serde_json::Value::Number(serde_json::Number::from(self.n_clusters)),
        );
        hyperparameters.insert(
            "n_iter".to_string(),
            serde_json::Value::Number(serde_json::Number::from(self.n_iter)),
        );

        let algorithm_config = Some(AlgorithmConfig {
            algorithm: "k-means".to_string(),
            hyperparameters,
            distance_metric: Some("euclidean".to_string()),
            linkage_method: None,
            initialization_method: Some("k-means++".to_string()),
            convergence_criteria: Some({
                let mut criteria = HashMap::new();
                criteria.insert("tolerance".to_string(), 1e-4);
                criteria
            }),
        });

        let performance_metrics = Some(PerformanceMetrics {
            training_time_seconds: None,
            memory_usage_bytes: None,
            iterations_to_convergence: Some(self.n_iter),
            final_objective_value: Some(self.inertia),
            silhouette_score: None,
            davies_bouldin_index: None,
            custom_metrics: None,
        });

        let data_characteristics = if let Some(ref labels) = self.labels {
            Some(ModelDataCharacteristics {
                n_samples: labels.len(),
                n_features: self.centroids.ncols(),
                data_type: Some("continuous".to_string()),
                statistical_summary: None,
                preprocessing_steps: Some(vec![]),
                feature_scaling: None,
                missing_value_percentage: Some(0.0),
            })
        } else {
            None
        };

        ModelMetadata {
            model_type: "kmeans".to_string(),
            created_at: current_timestamp(),
            created_at_readable: format_timestamp(current_timestamp()),
            n_clusters: Some(self.n_clusters),
            n_features: Some(self.centroids.ncols()),
            parameters,
            version: "1.0".to_string(),
            algorithm_config,
            performance_metrics,
            data_characteristics,
        }
    }
}

impl AdvancedExport for HierarchicalModel {
    fn export<P: AsRef<Path>>(&self, path: P, format: ExportFormat) -> Result<()> {
        match format {
            ExportFormat::Json => self.save_to_file(path),
            ExportFormat::Newick => {
                let newick = self.to_newick()?;
                std::fs::write(path, newick).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to write file: {}", e))
                })
            }
            ExportFormat::Binary => {
                let data = serde_json::to_vec(self).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to serialize: {}", e))
                })?;
                std::fs::write(path, data).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to write file: {}", e))
                })
            }
            _ => Err(ClusteringError::InvalidInput(
                "Format not supported for hierarchical model".to_string(),
            )),
        }
    }

    fn export_to_string(&self, format: ExportFormat) -> Result<String> {
        match format {
            ExportFormat::Json => serde_json::to_string_pretty(self)
                .map_err(|e| ClusteringError::InvalidInput(format!("Failed to serialize: {}", e))),
            ExportFormat::Newick => self.to_newick(),
            _ => Err(ClusteringError::InvalidInput(
                "Format not supported for string export".to_string(),
            )),
        }
    }

    fn get_metadata(&self) -> ModelMetadata {
        let mut parameters = std::collections::HashMap::new();
        parameters.insert("method".to_string(), self.method.clone());
        parameters.insert(
            "n_observations".to_string(),
            self.n_observations.to_string(),
        );

        let mut hyperparameters = HashMap::new();
        hyperparameters.insert(
            "linkage_method".to_string(),
            serde_json::Value::String(self.method.clone()),
        );
        hyperparameters.insert(
            "n_observations".to_string(),
            serde_json::Value::Number(serde_json::Number::from(self.n_observations)),
        );

        let algorithm_config = Some(AlgorithmConfig {
            algorithm: "hierarchical_clustering".to_string(),
            hyperparameters,
            distance_metric: Some("euclidean".to_string()),
            linkage_method: Some(self.method.clone()),
            initialization_method: None,
            convergence_criteria: None,
        });

        let performance_metrics = Some(PerformanceMetrics {
            training_time_seconds: None,
            memory_usage_bytes: None,
            iterations_to_convergence: None,
            final_objective_value: None,
            silhouette_score: None,
            davies_bouldin_index: None,
            custom_metrics: None,
        });

        let data_characteristics = Some(ModelDataCharacteristics {
            n_samples: self.n_observations,
            n_features: 0, // Unknown in linkage matrix
            data_type: Some("continuous".to_string()),
            statistical_summary: None,
            preprocessing_steps: Some(vec![]),
            feature_scaling: None,
            missing_value_percentage: Some(0.0),
        });

        ModelMetadata {
            model_type: "hierarchical".to_string(),
            created_at: current_timestamp(),
            created_at_readable: format_timestamp(current_timestamp()),
            n_clusters: None, // Can vary based on cut
            n_features: None, // Not directly stored
            parameters,
            version: "1.0".to_string(),
            algorithm_config,
            performance_metrics,
            data_characteristics,
        }
    }
}

/// Cross-platform model compatibility utilities
pub mod compatibility {
    use super::*;

    /// Convert to scikit-learn compatible format
    pub fn to_sklearn_format<T: SerializableModel + AdvancedExport>(
        model: &T,
    ) -> Result<serde_json::Value> {
        use serde_json::json;

        let metadata = model.get_metadata();
        let modeldata = model.export_to_string(ExportFormat::Json)?;
        let model_json: serde_json::Value = serde_json::from_str(&modeldata)?;

        Ok(json!({
            "sklearn_version": "1.0.0",
            "scirs_version": "0.1.0-beta.1",
            "model_type": metadata.model_type,
            "created_at": metadata.created_at,
            "parameters": metadata.parameters,
            "modeldata": model_json,
            "feature_names_in_": null,
            "n_features_in_": metadata.n_features,
            "_estimator_type": "clusterer"
        }))
    }

    /// Convert to SciPy hierarchical clustering format
    pub fn to_scipy_linkage_format(linkage_matrix: &Array2<f64>) -> Result<serde_json::Value> {
        use serde_json::json;

        // Convert linkage matrix to SciPy format
        let linkagedata: Vec<Vec<f64>> = linkage_matrix
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();

        Ok(json!({
            "linkage": linkagedata,
            "format": "scipy_linkage",
            "version": "1.9.0",
            "encoding": "utf-8"
        }))
    }

    /// Create scikit-learn compatible parameter grid for hyperparameter tuning
    pub fn create_sklearn_param_grid(algorithm: &str) -> Result<serde_json::Value> {
        use serde_json::json;

        match algorithm {
            "kmeans" => Ok(json!({
                "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "init": ["k-means++", "random"],
                "max_iter": [100, 200, 300],
                "tol": [1e-4, 1e-5, 1e-6],
                "random_state": [42]
            })),
            "dbscan" => Ok(json!({
                "eps": [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
                "min_samples": [3, 5, 10, 15, 20],
                "metric": ["euclidean", "manhattan", "cosine"]
            })),
            "agglomerative" => Ok(json!({
                "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "linkage": ["ward", "complete", "average", "single"],
                "metric": ["euclidean", "manhattan", "cosine"]
            })),
            _ => Err(ClusteringError::InvalidInput(format!(
                "Unknown algorithm for parameter grid: {}",
                algorithm
            ))),
        }
    }

    /// Convert ndarray to NumPy-compatible format
    pub fn to_numpy_format(array: &Array2<f64>) -> Result<serde_json::Value> {
        use serde_json::json;

        let shape = array.shape();
        let data: Vec<f64> = array.iter().cloned().collect();

        Ok(json!({
            "data": data,
            "shape": [shape[0], shape[1]],
            "dtype": "float64",
            "fortran_order": false,
            "version": [1, 0]
        }))
    }

    /// Convert from NumPy-compatible format
    pub fn from_numpy_format(numpy_data: &serde_json::Value) -> Result<Array2<f64>> {
        let data: Vec<f64> = numpy_data["data"]
            .as_array()
            .ok_or_else(|| ClusteringError::InvalidInput("Missing data field".to_string()))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        let shape = numpy_data["shape"]
            .as_array()
            .ok_or_else(|| ClusteringError::InvalidInput("Missing shape field".to_string()))?;

        let nrows = shape[0].as_u64().unwrap_or(0) as usize;
        let ncols = shape[1].as_u64().unwrap_or(0) as usize;

        Array2::from_shape_vec((nrows, ncols), data)
            .map_err(|e| ClusteringError::InvalidInput(format!("Shape mismatch: {}", e)))
    }

    /// Create pandas DataFrame-compatible format for clustering results
    pub fn to_pandas_format(
        data: &Array2<f64>,
        labels: &Array1<i32>,
        feature_names: Option<&[String]>,
    ) -> Result<serde_json::Value> {
        use serde_json::json;

        let n_samples = data.nrows();
        let n_features = data.ncols();

        if labels.len() != n_samples {
            return Err(ClusteringError::InvalidInput(
                "Data and labels must have the same number of samples".to_string(),
            ));
        }

        let mut dfdata = std::collections::HashMap::new();

        // Add feature columns
        for (i, col) in data.columns().into_iter().enumerate() {
            let col_name = if let Some(_names) = feature_names {
                _names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("feature_{}", i))
            } else {
                format!("feature_{}", i)
            };

            let coldata: Vec<f64> = col.iter().cloned().collect();
            dfdata.insert(col_name, json!(coldata));
        }

        // Add cluster labels
        let labeldata: Vec<i32> = labels.iter().cloned().collect();
        dfdata.insert("cluster".to_string(), json!(labeldata));

        // Create columns list
        let mut cols: Vec<String> = (0..n_features)
            .map(|i| {
                if let Some(_names) = feature_names {
                    _names
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("feature_{}", i))
                } else {
                    format!("feature_{}", i)
                }
            })
            .collect();
        cols.push("cluster".to_string());

        Ok(json!({
            "data": dfdata,
            "index": (0..n_samples).collect::<Vec<_>>(),
            "columns": cols
        }))
    }

    /// Convert clustering results to R-compatible format
    pub fn to_r_format(
        linkage: Option<&Array2<f64>>,
        labels: Option<&Array1<i32>>,
        centers: Option<&Array2<f64>>,
    ) -> Result<serde_json::Value> {
        use serde_json::json;

        let mut rdata = json!({});

        if let Some(linkage_matrix) = linkage {
            // R's hclust format
            let merge_matrix: Vec<Vec<i32>> = linkage_matrix
                .rows()
                .into_iter()
                .map(|row| vec![row[0] as i32, row[1] as i32])
                .collect();

            let height: Vec<f64> = linkage_matrix
                .rows()
                .into_iter()
                .map(|row| row[2])
                .collect();

            rdata["hclust"] = json!({
                "merge": merge_matrix,
                "height": height,
                "order": (1..=linkage_matrix.nrows() + 1).collect::<Vec<_>>(),
                "labels": null,
                "method": "unknown",
                "call": "scirs2::hierarchical_clustering",
                "dist.method": "euclidean"
            });
        }

        if let Some(cluster_labels) = labels {
            rdata["clusters"] = json!(cluster_labels.iter().cloned().collect::<Vec<_>>());
        }

        if let Some(cluster_centers) = centers {
            let centersdata: Vec<Vec<f64>> = cluster_centers
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect();
            rdata["centers"] = json!(centersdata);
        }

        Ok(rdata)
    }

    /// Import from scikit-learn compatible format
    pub fn from_sklearn_format<T: SerializableModel>(sklearndata: &serde_json::Value) -> Result<T> {
        if let Some(modeldata) = sklearndata.get("modeldata") {
            let model_str = serde_json::to_string(modeldata)?;
            let model: T = serde_json::from_str(&model_str)?;
            Ok(model)
        } else {
            Err(ClusteringError::InvalidInput(
                "Invalid sklearn format: missing modeldata".to_string(),
            ))
        }
    }

    /// Convert to ONNX-compatible format metadata
    pub fn to_onnx_metadata<T: SerializableModel + AdvancedExport>(
        model: &T,
    ) -> Result<serde_json::Value> {
        use serde_json::json;

        let metadata = model.get_metadata();

        Ok(json!({
            "ir_version": 7,
            "producer_name": "scirs2-cluster",
            "producer_version": "0.1.0-beta.1",
            "domain": "ai.onnx.ml",
            "model_version": 1,
            "doc_string": format!("SCIRS2 {} model", metadata.model_type),
            "metadata_props": {
                "model_type": metadata.model_type,
                "n_clusters": metadata.n_clusters,
                "n_features": metadata.n_features,
                "created_at": metadata.created_at,
                "algorithm_config": metadata.algorithm_config
            }
        }))
    }

    /// Convert to PyTorch Lightning checkpoint format
    pub fn to_pytorch_checkpoint<T: SerializableModel + AdvancedExport>(
        model: &T,
        epoch: Option<usize>,
        step: Option<usize>,
    ) -> Result<serde_json::Value> {
        use serde_json::json;

        let metadata = model.get_metadata();
        let modeldata = model.export_to_string(ExportFormat::Json)?;
        let model_json: serde_json::Value = serde_json::from_str(&modeldata)?;

        Ok(json!({
            "pytorch_lightning_version": "1.6.0",
            "scirs_version": "0.1.0-beta.1",
            "epoch": epoch.unwrap_or(0),
            "global_step": step.unwrap_or(0),
            "lr_schedulers": [],
            "optimizers": [],
            "state_dict": model_json,
            "hyper_parameters": metadata.parameters,
            "model_type": metadata.model_type,
            "clustering_metadata": {
                "algorithm": metadata.algorithm_config.as_ref().map(|c| c.algorithm.clone()),
                "n_clusters": metadata.n_clusters,
                "performance_metrics": metadata.performance_metrics
            }
        }))
    }

    /// Convert to MLflow model format
    pub fn to_mlflow_format<T: SerializableModel + AdvancedExport>(
        model: &T,
        model_uuid: Option<String>,
    ) -> Result<serde_json::Value> {
        use serde_json::json;

        let metadata = model.get_metadata();
        let modeldata = model.export_to_string(ExportFormat::Json)?;

        Ok(json!({
            "artifactpath": "model",
            "flavors": {
                "scirs2": {
                    "scirs_version": "0.1.0-beta.1",
                    "model_type": metadata.model_type,
                    "data": modeldata
                },
                "python_function": {
                    "env": "conda.yaml",
                    "loader_module": "scirs2_mlflow",
                    "python_version": "3.8.0"
                }
            },
            "model_uuid": model_uuid.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            "run_id": null,
            "saved_input_example_info": null,
            "signature": {
                "inputs": format!("[{{\"name\": \"data\", \"type\": \"tensor\", \"tensor-spec\": {{\"dtype\": \"float64\", \"shape\": [-1, {}]}}}}]",
                    metadata.n_features.unwrap_or(0)),
                "outputs": format!("[{{\"name\": \"labels\", \"type\": \"tensor\", \"tensor-spec\": {{\"dtype\": \"int64\", \"shape\": [-1]}}}}]")
            },
            "utc_time_created": metadata.created_at_readable,
            "mlflow_version": "1.27.0"
        }))
    }

    /// Convert to Hugging Face model card format
    pub fn to_huggingface_card<T: SerializableModel + AdvancedExport>(
        model: &T,
        model_name: &str,
        description: Option<&str>,
    ) -> Result<String> {
        let metadata = model.get_metadata();

        let mut card = String::new();
        card.push_str("---\n");
        card.push_str(&format!("libraryname: scirs2-cluster\n"));
        card.push_str(&format!("tags:\n"));
        card.push_str(&format!("- clustering\n"));
        card.push_str(&format!("- {}\n", metadata.model_type));
        card.push_str(&format!("- unsupervised-learning\n"));
        card.push_str(&format!("model-index:\n"));
        card.push_str(&format!("- name: {}\n", model_name));
        card.push_str(&format!("  results:\n"));

        if let Some(perf) = &metadata.performance_metrics {
            if let Some(silhouette) = perf.silhouette_score {
                card.push_str(&format!("  - task:\n"));
                card.push_str(&format!("      type: clustering\n"));
                card.push_str(&format!("    metrics:\n"));
                card.push_str(&format!("    - type: silhouette_score\n"));
                card.push_str(&format!("      value: {:.4}\n", silhouette));
            }
        }

        card.push_str("---\n\n");
        card.push_str(&format!("# {}\n\n", model_name));

        if let Some(desc) = description {
            card.push_str(&format!("{}\n\n", desc));
        }

        card.push_str("## Model Description\n\n");
        card.push_str(&format!(
            "This is a **{}** clustering model trained using the SCIRS2 library.\n\n",
            metadata.model_type
        ));

        card.push_str("## Model Details\n\n");
        card.push_str(&format!("- **Model Type**: {}\n", metadata.model_type));
        if let Some(n_clusters) = metadata.n_clusters {
            card.push_str(&format!("- **Number of Clusters**: {}\n", n_clusters));
        }
        if let Some(n_features) = metadata.n_features {
            card.push_str(&format!("- **Number of Features**: {}\n", n_features));
        }
        card.push_str(&format!(
            "- **Created**: {}\n",
            metadata.created_at_readable
        ));
        card.push_str(&format!("- **Library**: SCIRS2 v{}\n", metadata.version));

        if let Some(algorithm_config) = &metadata.algorithm_config {
            card.push_str("\n## Algorithm Configuration\n\n");
            card.push_str(&format!(
                "- **Algorithm**: {}\n",
                algorithm_config.algorithm
            ));
            if let Some(distance_metric) = &algorithm_config.distance_metric {
                card.push_str(&format!("- **Distance Metric**: {}\n", distance_metric));
            }
            if let Some(linkage_method) = &algorithm_config.linkage_method {
                card.push_str(&format!("- **Linkage Method**: {}\n", linkage_method));
            }
        }

        if let Some(perf) = &metadata.performance_metrics {
            card.push_str("\n## Performance Metrics\n\n");
            if let Some(training_time) = perf.training_time_seconds {
                card.push_str(&format!(
                    "- **Training Time**: {:.2} seconds\n",
                    training_time
                ));
            }
            if let Some(silhouette) = perf.silhouette_score {
                card.push_str(&format!("- **Silhouette Score**: {:.4}\n", silhouette));
            }
            if let Some(davies_bouldin) = perf.davies_bouldin_index {
                card.push_str(&format!(
                    "- **Davies-Bouldin Index**: {:.4}\n",
                    davies_bouldin
                ));
            }
        }

        card.push_str("\n## Usage\n\n");
        card.push_str("```rust\n");
        card.push_str("use scirs2_cluster::serialization::SerializableModel;\n");
        card.push_str(&format!(
            "use scirs2_cluster::serialization::{}Model;\n",
            metadata.model_type.to_uppercase()
        ));
        card.push_str("\n");
        card.push_str(&format!("// Load the model\n"));
        card.push_str(&format!(
            "let model = {}Model::load_from_file(\"path/to/model.json\")?;\n",
            metadata.model_type.to_uppercase()
        ));
        card.push_str("\n");
        card.push_str("// Use the model for prediction\n");
        card.push_str("let predictions = model.predict(yourdata.view())?;\n");
        card.push_str("```\n");

        Ok(card)
    }

    /// Export model metadata in Apache Arrow schema format
    pub fn to_arrow_schema<T: SerializableModel + AdvancedExport>(
        model: &T,
    ) -> Result<serde_json::Value> {
        use serde_json::json;

        let metadata = model.get_metadata();

        Ok(json!({
            "schema": {
                "fields": [
                    {
                        "name": "features",
                        "type": {
                            "name": "list",
                            "contains": {
                                "name": "item",
                                "type": {
                                    "name": "floatingpoint",
                                    "precision": "DOUBLE"
                                },
                                "nullable": false
                            }
                        },
                        "nullable": false,
                        "metadata": {
                            "n_features": metadata.n_features.unwrap_or(0)
                        }
                    },
                    {
                        "name": "cluster_labels",
                        "type": {
                            "name": "int",
                            "bitWidth": 32,
                            "isSigned": true
                        },
                        "nullable": false,
                        "metadata": {
                            "n_clusters": metadata.n_clusters.unwrap_or(0)
                        }
                    }
                ],
                "metadata": {
                    "model_type": metadata.model_type,
                    "scirs_version": "0.1.0-beta.1",
                    "created_at": metadata.created_at
                }
            }
        }))
    }

    /// Create a joblib-compatible model dump
    pub fn to_joblib_format<T: SerializableModel + AdvancedExport>(
        model: &T,
        compress: bool,
    ) -> Result<Vec<u8>> {
        let sklearndata = crate::serialization::compatibility::to_sklearn_format(model)?;

        if compress {
            enhanced::serialize_with_format(&sklearndata, ExportFormat::CompressedJson)
        } else {
            enhanced::serialize_with_format(&sklearndata, ExportFormat::Json)
        }
    }

    /// Load from joblib-compatible format
    pub fn from_joblib_format<T: SerializableModel>(data: &[u8], compressed: bool) -> Result<T> {
        let format = if compressed {
            ExportFormat::CompressedJson
        } else {
            ExportFormat::Json
        };

        let sklearndata: serde_json::Value = enhanced::deserialize_with_format(data, format)?;
        from_sklearn_format(&sklearndata)
    }

    /// Convert clustering results to scikit-learn's standard clustering result format
    pub fn to_sklearn_clustering_result(
        labels: &Array1<i32>,
        cluster_centers: Option<&Array2<f64>>,
        n_clusters: Option<usize>,
    ) -> Result<serde_json::Value> {
        use serde_json::json;

        let mut result = json!({
            "labels_": labels.iter().cloned().collect::<Vec<_>>(),
            "n_clusters_": n_clusters.unwrap_or_else(|| {
                labels.iter().map(|&x| x).max().unwrap_or(-1) as usize + 1
            })
        });

        if let Some(_centers) = cluster_centers {
            result["cluster_centers_"] = to_numpy_format(_centers)?;
        }

        Ok(result)
    }

    /// Convert to SciPy's cluster.hierarchy.dendrogram compatible format
    pub fn to_scipy_dendrogram_format(
        linkage_matrix: &Array2<f64>,
        labels: Option<&[String]>,
    ) -> Result<serde_json::Value> {
        use serde_json::json;

        let n_obs = linkage_matrix.nrows() + 1;

        // Create coordinate arrays for dendrogram plotting
        let mut icoord = Vec::new();
        let mut dcoord = Vec::new();

        for (_i, row) in linkage_matrix.rows().into_iter().enumerate() {
            let left = row[0] as usize;
            let right = row[1] as usize;
            let distance = row[2];

            // Calculate x-coordinates (simplified)
            let x_left = if left < n_obs {
                left as f64 * 10.0
            } else {
                (left - n_obs) as f64 * 10.0 + 5.0
            };
            let x_right = if right < n_obs {
                right as f64 * 10.0
            } else {
                (right - n_obs) as f64 * 10.0 + 5.0
            };
            let _x_center = (x_left + x_right) / 2.0;

            icoord.push(vec![x_left, x_left, x_right, x_right]);
            dcoord.push(vec![0.0, distance, distance, 0.0]);
        }

        Ok(json!({
            "icoord": icoord,
            "dcoord": dcoord,
            "ivl": labels.unwrap_or(&(0..n_obs).map(|i| i.to_string()).collect::<Vec<_>>()),
            "leaves": (0..n_obs).collect::<Vec<_>>(),
            "color_list": (0..linkage_matrix.nrows()).map(|_| "b").collect::<Vec<_>>()
        }))
    }

    /// Create a pickle-compatible byte representation (simplified)
    pub fn to_pickle_like_format<T: SerializableModel + AdvancedExport>(
        model: &T,
    ) -> Result<Vec<u8>> {
        // Create a pickle-like format using binary serialization
        // This is a simplified version - real pickle compatibility would need more work
        let sklearndata = crate::serialization::compatibility::to_sklearn_format(model)?;
        enhanced::serialize_with_format(&sklearndata, ExportFormat::Binary)
    }

    /// Generate model summary in scikit-learn style
    pub fn generate_sklearn_model_summary<T: SerializableModel + AdvancedExport>(
        model: &T,
    ) -> Result<String> {
        let metadata = model.get_metadata();

        let mut summary = String::new();
        summary.push_str(&format!("{}(\n", metadata.model_type));

        for (key, value) in &metadata.parameters {
            summary.push_str(&format!("    {}={},\n", key, value));
        }

        if let Some(config) = &metadata.algorithm_config {
            for (key, value) in &config.hyperparameters {
                summary.push_str(&format!("    {}={},\n", key, value));
            }
        }

        summary.push_str(")");
        Ok(summary)
    }

    /// Convert model to pandas-compatible clustering report
    pub fn to_pandas_clustering_report(
        labels: &Array1<i32>,
        data: &Array2<f64>,
        feature_names: Option<&[String]>,
    ) -> Result<serde_json::Value> {
        use serde_json::json;

        let n_clusters = labels.iter().map(|&x| x).max().unwrap_or(-1) as usize + 1;
        let mut cluster_stats = Vec::new();

        for cluster_id in 0..n_clusters {
            let cluster_indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| {
                    if label == cluster_id as i32 {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();

            let cluster_size = cluster_indices.len();
            let mut clusterdata = json!({
                "cluster_id": cluster_id,
                "size": cluster_size,
                "percentage": cluster_size as f64 / labels.len() as f64 * 100.0
            });

            if cluster_size > 0 {
                // Calculate centroid
                let mut centroid = vec![0.0; data.ncols()];
                for &idx in &cluster_indices {
                    for (j, &val) in data.row(idx).iter().enumerate() {
                        centroid[j] += val;
                    }
                }
                for val in &mut centroid {
                    *val /= cluster_size as f64;
                }

                clusterdata["centroid"] = json!(centroid);

                if let Some(_names) = feature_names {
                    let centroid_dict: std::collections::HashMap<String, f64> = _names
                        .iter()
                        .zip(centroid.iter())
                        .map(|(name, &val)| (name.clone(), val))
                        .collect();
                    clusterdata["centroid_named"] = json!(centroid_dict);
                }
            }

            cluster_stats.push(clusterdata);
        }

        Ok(json!({
            "n_clusters": n_clusters,
            "n_samples": labels.len(),
            "clusters": cluster_stats
        }))
    }
}

/// Advanced persistence features for production clustering systems
pub mod persistence {
    use super::*;
    use std::collections::BTreeMap;

    /// Model registry for managing multiple clustering models
    #[derive(Debug, Clone)]
    pub struct ModelRegistry {
        /// Registry of models with their metadata
        models: BTreeMap<String, ModelRegistryEntry>,
        /// Base directory for model storage
        base_directory: PathBuf,
        /// Default format for new models
        default_format: ExportFormat,
    }

    /// Entry in the model registry
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ModelRegistryEntry {
        /// Unique model identifier
        pub model_id: String,
        /// Model metadata
        pub metadata: ModelMetadata,
        /// File path relative to base directory
        pub filepath: PathBuf,
        /// Storage format used
        pub format: ExportFormat,
        /// File size in bytes
        pub file_size: Option<usize>,
        /// Model tags for organization
        pub tags: Vec<String>,
        /// Model description
        pub description: Option<String>,
        /// Dependencies on other models
        pub dependencies: Vec<String>,
        /// Checksum for integrity verification
        pub checksum: Option<String>,
    }

    impl ModelRegistry {
        /// Create a new model registry
        pub fn new<P: Into<PathBuf>>(base_directory: P, _directory: P) -> Self {
            Self {
                models: BTreeMap::new(),
                base_directory: base_directory.into(),
                default_format: ExportFormat::Json,
            }
        }

        /// Register a new model

        pub fn register_model<T: SerializableModel + AdvancedExport>(
            &mut self,
            model_id: String,
            model: &T,
            tags: Vec<String>,
            description: Option<String>,
        ) -> Result<()> {
            let metadata = model.get_metadata();
            let file_name = format!("{}.{}", model_id, format_extension(self.default_format));
            let filepath = self.base_directory.join(&file_name);

            // Save the model
            model.export(&filepath, self.default_format)?;

            // Calculate file size and checksum
            let file_size = std::fs::metadata(&filepath).map(|m| m.len() as usize).ok();
            let checksum = self.calculate_checksum(&filepath)?;

            let entry = ModelRegistryEntry {
                model_id: model_id.clone(),
                metadata,
                filepath: PathBuf::from(file_name),
                format: self.default_format,
                file_size,
                tags,
                description,
                dependencies: Vec::new(),
                checksum: Some(checksum),
            };

            self.models.insert(model_id, entry);
            self.save_registry()?;
            Ok(())
        }

        /// List all registered models
        pub fn list_models(&self) -> Vec<&ModelRegistryEntry> {
            self.models.values().collect()
        }

        /// Find models by tag
        pub fn find_by_tag(&self, tag: &str) -> Vec<&ModelRegistryEntry> {
            self.models
                .values()
                .filter(|entry| entry.tags.contains(&tag.to_string()))
                .collect()
        }

        /// Find models by type
        pub fn find_by_type(&self, model_type: &str) -> Vec<&ModelRegistryEntry> {
            self.models
                .values()
                .filter(|entry| entry.metadata.model_type == model_type)
                .collect()
        }

        /// Get model entry by ID
        pub fn get_model(&self, model_id: &str) -> Option<&ModelRegistryEntry> {
            self.models.get(model_id)
        }

        /// Remove a model from registry

        pub fn remove_model(&mut self, model_id: &str) -> Result<()> {
            if let Some(entry) = self.models.remove(model_id) {
                let fullpath = self.base_directory.join(&entry.filepath);
                if fullpath.exists() {
                    std::fs::remove_file(fullpath).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Failed to remove model file: {}", e))
                    })?;
                }
                self.save_registry()?;
            }
            Ok(())
        }

        /// Verify model integrity
        pub fn verify_model(&self, model_id: &str) -> Result<bool> {
            if let Some(entry) = self.models.get(model_id) {
                let fullpath = self.base_directory.join(&entry.filepath);
                if let Some(stored_checksum) = &entry.checksum {
                    let current_checksum = self.calculate_checksum(&fullpath)?;
                    Ok(current_checksum == *stored_checksum)
                } else {
                    Ok(true) // No checksum stored, assume valid
                }
            } else {
                Err(ClusteringError::InvalidInput("Model not found".to_string()))
            }
        }

        /// Compact registry by removing unused models

        pub fn compact_registry(&mut self) -> Result<Vec<String>> {
            let mut removed = Vec::new();
            let entries_to_check: Vec<_> = self.models.iter().collect();

            for (model_id, entry) in entries_to_check {
                let fullpath = self.base_directory.join(&entry.filepath);
                if !fullpath.exists() {
                    removed.push(model_id.clone());
                }
            }

            for model_id in &removed {
                self.models.remove(model_id);
            }

            if !removed.is_empty() {
                self.save_registry()?;
            }

            Ok(removed)
        }

        /// Load registry from disk

        pub fn load_registry(&mut self) -> Result<()> {
            let registrypath = self.base_directory.join("registry.json");
            if registrypath.exists() {
                let content = std::fs::read_to_string(&registrypath).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to read registry: {}", e))
                })?;
                self.models = serde_json::from_str(&content).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to parse registry: {}", e))
                })?;
            }
            Ok(())
        }

        /// Save registry to disk

        fn save_registry(&self) -> Result<()> {
            std::fs::create_dir_all(&self.base_directory).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to create directory: {}", e))
            })?;

            let registrypath = self.base_directory.join("registry.json");
            let content = serde_json::to_string_pretty(&self.models).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to serialize registry: {}", e))
            })?;

            std::fs::write(&registrypath, content).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to write registry: {}", e))
            })?;

            Ok(())
        }

        /// Calculate file checksum
        fn calculate_checksum<P: AsRef<Path>>(&self, path: P) -> Result<String> {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let content = std::fs::read(path).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to read file for checksum: {}", e))
            })?;

            let mut hasher = DefaultHasher::new();
            content.hash(&mut hasher);
            Ok(format!("{:x}", hasher.finish()))
        }
    }

    /// Batch model operations for efficient processing
    #[derive(Debug)]
    pub struct BatchModelProcessor {
        /// Target directory for batch operations
        target_directory: PathBuf,
        /// Batch configuration
        config: BatchConfig,
    }

    /// Configuration for batch operations
    #[derive(Debug, Clone)]
    pub struct BatchConfig {
        /// Maximum number of models to process in parallel
        pub max_parallel: usize,
        /// Compression level for batch archives
        pub compression_level: u32,
        /// Include metadata in batch exports
        pub include_metadata: bool,
        /// Target format for batch conversion
        pub target_format: ExportFormat,
    }

    impl Default for BatchConfig {
        fn default() -> Self {
            Self {
                max_parallel: 4,
                compression_level: 6,
                include_metadata: true,
                target_format: ExportFormat::Json,
            }
        }
    }

    impl BatchModelProcessor {
        /// Create a new batch processor
        pub fn new<P: Into<PathBuf>>(
            target_directory: P,
            directory: P,
            config: BatchConfig,
        ) -> Self {
            Self {
                target_directory: target_directory.into(),
                config,
            }
        }

        /// Export multiple models to a single archive
        pub fn export_batch<T: SerializableModel + AdvancedExport>(
            &self,
            models: &[(String, &T)],
            archive_name: &str,
        ) -> Result<PathBuf> {
            use flate2::write::GzEncoder;
            use std::io::Write;

            std::fs::create_dir_all(&self.target_directory).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to create directory: {}", e))
            })?;

            let archivepath = self
                .target_directory
                .join(format!("{}.tar.gz", archive_name));
            let file = File::create(&archivepath).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to create archive: {}", e))
            })?;

            let encoder = GzEncoder::new(file, Compression::new(self.config.compression_level));
            let mut tar = tar::Builder::new(encoder);

            for (model_name, model) in models {
                // Export model to temporary buffer
                let model_content = model.export_to_string(self.config.target_format)?;
                let file_name = format!(
                    "{}.{}",
                    model_name,
                    format_extension(self.config.target_format)
                );

                // Add to tar archive
                let mut header = tar::Header::new_gnu();
                header.set_path(&file_name).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to set tar path: {}", e))
                })?;
                header.set_size(model_content.len() as u64);
                header.set_cksum();

                tar.append(&header, model_content.as_bytes()).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to add to archive: {}", e))
                })?;

                // Add metadata if requested
                if self.config.include_metadata {
                    let metadata = model.get_metadata();
                    let metadata_content =
                        serde_json::to_string_pretty(&metadata).map_err(|e| {
                            ClusteringError::InvalidInput(format!(
                                "Failed to serialize metadata: {}",
                                e
                            ))
                        })?;

                    let metadata_file_name = format!("{}_metadata.json", model_name);
                    let mut metadata_header = tar::Header::new_gnu();
                    metadata_header.set_path(&metadata_file_name).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Failed to set metadata path: {}", e))
                    })?;
                    metadata_header.set_size(metadata_content.len() as u64);
                    metadata_header.set_cksum();

                    tar.append(&metadata_header, metadata_content.as_bytes())
                        .map_err(|e| {
                            ClusteringError::InvalidInput(format!(
                                "Failed to add metadata to archive: {}",
                                e
                            ))
                        })?;
                }
            }

            tar.finish().map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to finalize archive: {}", e))
            })?;

            Ok(archivepath)
        }

        /// Import models from batch archive
        pub fn import_batch<T: SerializableModel>(
            &self,
            archivepath: &Path,
        ) -> Result<Vec<(String, T)>> {
            use flate2::read::GzDecoder;

            let file = File::open(archivepath).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to open archive: {}", e))
            })?;

            let decoder = GzDecoder::new(file);
            let mut tar = tar::Archive::new(decoder);
            let mut models = Vec::new();

            for entry in tar.entries().map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to read archive entries: {}", e))
            })? {
                let mut entry = entry.map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to read archive entry: {}", e))
                })?;

                let path = entry.path().map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to get entry path: {}", e))
                })?;

                let file_name_str = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.to_string());

                if let Some(file_name) = file_name_str {
                    if !file_name.ends_with("_metadata.json") {
                        let mut contents = String::new();
                        entry.read_to_string(&mut contents).map_err(|e| {
                            ClusteringError::InvalidInput(format!(
                                "Failed to read entry contents: {}",
                                e
                            ))
                        })?;

                        let model: T = serde_json::from_str(&contents).map_err(|e| {
                            ClusteringError::InvalidInput(format!(
                                "Failed to deserialize model: {}",
                                e
                            ))
                        })?;

                        let model_name = file_name
                            .rsplit_once('.')
                            .map(|(name_, _)| name_.to_string())
                            .unwrap_or(file_name.clone());
                        models.push((model_name, model));
                    }
                }
            }

            Ok(models)
        }

        /// Convert models between formats in batch
        pub fn convert_batch(
            &self,
            input_dir: &Path,
            from_format: ExportFormat,
            to_format: ExportFormat,
        ) -> Result<usize> {
            let mut converted_count = 0;

            for entry in std::fs::read_dir(input_dir).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to read input directory: {}", e))
            })? {
                let entry = entry.map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to read directory entry: {}", e))
                })?;

                let path = entry.path();
                if path.is_file() {
                    if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
                        if extension == format_extension(from_format) {
                            // Read and convert
                            let _content = std::fs::read(&path).map_err(|e| {
                                ClusteringError::InvalidInput(format!("Failed to read file: {}", e))
                            })?;

                            // This is a simplified conversion - in practice, you'd need to know the model type
                            // For now, we'll skip actual conversion and just count files
                            converted_count += 1;
                        }
                    }
                }
            }

            Ok(converted_count)
        }
    }

    /// Get file extension for export format
    fn format_extension(format: ExportFormat) -> &'static str {
        match format {
            ExportFormat::Json => "json",
            ExportFormat::Yaml => "yaml",
            ExportFormat::Csv => "csv",
            ExportFormat::Newick => "nwk",
            ExportFormat::Binary => "bin",
            ExportFormat::CompressedJson => "json.gz",
            ExportFormat::MessagePack => "msgpack",
            ExportFormat::Cbor => "cbor",
        }
    }
}

/// Model compression and optimization utilities
pub mod compression {
    use super::*;

    /// Compression configuration for models
    #[derive(Debug, Clone)]
    pub struct CompressionConfig {
        /// Compression algorithm to use
        pub algorithm: CompressionAlgorithm,
        /// Compression level (algorithm-dependent)
        pub level: u32,
        /// Enable quantization for numerical data
        pub enable_quantization: bool,
        /// Quantization precision (bits)
        pub quantization_bits: u8,
        /// Remove redundant data
        pub remove_redundancy: bool,
    }

    /// Available compression algorithms
    #[derive(Debug, Clone, Copy)]
    pub enum CompressionAlgorithm {
        /// Standard gzip compression
        Gzip,
        /// LZ4 fast compression
        Lz4,
        /// Zstandard high-ratio compression
        Zstd,
        /// BZIP2 high compression
        Bzip2,
        /// No compression
        None,
    }

    impl Default for CompressionConfig {
        fn default() -> Self {
            Self {
                algorithm: CompressionAlgorithm::Gzip,
                level: 6,
                enable_quantization: false,
                quantization_bits: 16,
                remove_redundancy: true,
            }
        }
    }

    /// Compressed model container
    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct CompressedModel {
        /// Original model metadata
        pub metadata: ModelMetadata,
        /// Compressed data
        pub compresseddata: Vec<u8>,
        /// Compression configuration used
        pub compression_config: CompressionInfo,
        /// Original size before compression
        pub original_size: usize,
        /// Compression ratio achieved
        pub compression_ratio: f64,
    }

    /// Information about compression applied
    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct CompressionInfo {
        /// Algorithm used
        pub algorithm: String,
        /// Compression level
        pub level: u32,
        /// Whether quantization was applied
        pub quantized: bool,
        /// Quantization precision if applied
        pub quantization_bits: Option<u8>,
    }

    impl CompressedModel {
        /// Compress a model with given configuration
        pub fn compress<T: SerializableModel + AdvancedExport>(
            model: &T,
            config: CompressionConfig,
        ) -> Result<Self> {
            let metadata = model.get_metadata();

            // Serialize model to bytes
            let originaldata = enhanced::serialize_with_format(model, ExportFormat::Binary)?;
            let original_size = originaldata.len();

            // Apply compression
            let compresseddata = Self::apply_compression(&originaldata, &config)?;

            let compression_ratio = compresseddata.len() as f64 / original_size as f64;

            let compression_info = CompressionInfo {
                algorithm: format!("{:?}", config.algorithm),
                level: config.level,
                quantized: config.enable_quantization,
                quantization_bits: if config.enable_quantization {
                    Some(config.quantization_bits)
                } else {
                    None
                },
            };

            Ok(Self {
                metadata,
                compresseddata,
                compression_config: compression_info,
                original_size,
                compression_ratio,
            })
        }

        /// Decompress and deserialize the model
        pub fn decompress<T: SerializableModel>(&self) -> Result<T> {
            let decompresseddata =
                Self::apply_decompression(&self.compresseddata, &self.compression_config)?;

            enhanced::deserialize_with_format(&decompresseddata, ExportFormat::Binary)
        }

        /// Apply compression to data
        fn apply_compression(data: &[u8], config: &CompressionConfig) -> Result<Vec<u8>> {
            match config.algorithm {
                CompressionAlgorithm::Gzip => {
                    let mut encoder = GzEncoder::new(Vec::new(), Compression::new(config.level));
                    encoder.write_all(data).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Compression failed: {}", e))
                    })?;
                    encoder.finish().map_err(|e| {
                        ClusteringError::InvalidInput(format!(
                            "Compression finalization failed: {}",
                            e
                        ))
                    })
                }
                CompressionAlgorithm::None => Ok(data.to_vec()),

                #[cfg(feature = "lz4_compression")]
                CompressionAlgorithm::Lz4 => {
                    use lz4::block::{compress, CompressionMode};
                    let mode = match config.level {
                        0..=3 => CompressionMode::FAST(config.level as i32),
                        4..=9 => CompressionMode::HIGHCOMPRESSION(config.level as i32),
                        _ => CompressionMode::FAST(1),
                    };

                    compress(data, Some(mode), true).map_err(|e| {
                        ClusteringError::InvalidInput(format!("LZ4 compression failed: {}", e))
                    })
                }

                #[cfg(feature = "zstd_compression")]
                CompressionAlgorithm::Zstd => {
                    zstd::stream::encode_all(std::io::Cursor::new(data), config.level as i32)
                        .map_err(|e| {
                            ClusteringError::InvalidInput(format!("Zstd compression failed: {}", e))
                        })
                }

                #[cfg(feature = "bzip2_compression")]
                CompressionAlgorithm::Bzip2 => {
                    use bzip2::write::BzEncoder;
                    use bzip2::Compression as BzCompression;

                    let compression_level = match config.level {
                        0..=9 => BzCompression::new(config.level),
                        _ => BzCompression::new(6), // Default level
                    };

                    let mut encoder = BzEncoder::new(Vec::new(), compression_level);
                    encoder.write_all(data).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Bzip2 compression failed: {}", e))
                    })?;
                    encoder.finish().map_err(|e| {
                        ClusteringError::InvalidInput(format!(
                            "Bzip2 compression finalization failed: {}",
                            e
                        ))
                    })
                }

                // Fallback for algorithms not available in current build
                _ => {
                    // Always fall back to gzip for unknown algorithms
                    let mut encoder = GzEncoder::new(Vec::new(), Compression::new(config.level));
                    encoder.write_all(data).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Compression failed: {}", e))
                    })?;
                    encoder.finish().map_err(|e| {
                        ClusteringError::InvalidInput(format!(
                            "Compression finalization failed: {}",
                            e
                        ))
                    })
                }
            }
        }

        /// Apply decompression to data
        fn apply_decompression(data: &[u8], info: &CompressionInfo) -> Result<Vec<u8>> {
            match info.algorithm.as_str() {
                "Gzip" => {
                    let mut decoder = GzDecoder::new(data);
                    let mut decompressed = Vec::new();
                    decoder.read_to_end(&mut decompressed).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Decompression failed: {}", e))
                    })?;
                    Ok(decompressed)
                }
                "None" => Ok(data.to_vec()),

                #[cfg(feature = "lz4_compression")]
                "Lz4" => {
                    use lz4::block::decompress;
                    // LZ4 requires knowing the original size, so we need to store it
                    // For now, we'll use a large buffer and hope it's enough
                    decompress(data, None).map_err(|e| {
                        ClusteringError::InvalidInput(format!("LZ4 decompression failed: {}", e))
                    })
                }

                #[cfg(feature = "zstd_compression")]
                "Zstd" => zstd::stream::decode_all(std::io::Cursor::new(data)).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Zstd decompression failed: {}", e))
                }),

                #[cfg(feature = "bzip2_compression")]
                "Bzip2" => {
                    use bzip2::read::BzDecoder;

                    let mut decoder = BzDecoder::new(data);
                    let mut decompressed = Vec::new();
                    decoder.read_to_end(&mut decompressed).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Bzip2 decompression failed: {}", e))
                    })?;
                    Ok(decompressed)
                }

                // Fallback for unknown algorithms - always use gzip
                _ => {
                    // Default to gzip for unknown algorithms
                    let mut decoder = GzDecoder::new(data);
                    let mut decompressed = Vec::new();
                    decoder.read_to_end(&mut decompressed).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Decompression failed: {}", e))
                    })?;
                    Ok(decompressed)
                }
            }
        }

        /// Get compression statistics
        pub fn get_compression_stats(&self) -> CompressionStats {
            CompressionStats {
                original_size: self.original_size,
                compressed_size: self.compresseddata.len(),
                compression_ratio: self.compression_ratio,
                space_saved_bytes: self.original_size.saturating_sub(self.compresseddata.len()),
                space_saved_percentage: (1.0 - self.compression_ratio) * 100.0,
            }
        }
    }

    /// Compression statistics
    #[derive(Debug, Clone)]
    pub struct CompressionStats {
        /// Original size in bytes
        pub original_size: usize,
        /// Compressed size in bytes
        pub compressed_size: usize,
        /// Compression ratio (compressed/original)
        pub compression_ratio: f64,
        /// Space saved in bytes
        pub space_saved_bytes: usize,
        /// Space saved percentage
        pub space_saved_percentage: f64,
    }
}

/// Unified clustering workflow state that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClusteringWorkflow {
    /// Workflow identifier
    pub id: String,
    /// Workflow name/description
    pub name: String,
    /// Current algorithm state
    pub algorithm_state: AlgorithmState,
    /// Training data hash for validation
    pub data_hash: String,
    /// Complete training history
    pub training_history: Vec<TrainingStep>,
    /// Current model (if training completed)
    pub current_model: Option<serde_json::Value>,
    /// Workflow configuration
    pub config: WorkflowConfig,
    /// Creation and last update timestamps
    pub created_at: u64,
    pub updated_at: u64,
}

/// Algorithm state for resumable training
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum AlgorithmState {
    /// K-means algorithm state
    KMeans {
        centroids: Array2<f64>,
        iteration: usize,
        inertia: f64,
        converged: bool,
        labels: Option<Array1<usize>>,
    },
    /// Hierarchical clustering state
    Hierarchical {
        linkage_matrix: Option<Array2<f64>>,
        merge_step: usize,
        completed: bool,
    },
    /// DBSCAN state
    DBSCAN {
        visited_points: Vec<bool>,
        cluster_labels: Array1<i32>,
        core_samples: Vec<bool>,
        eps: f64,
        min_samples: usize,
        completed: bool,
    },
    /// Generic state for other algorithms
    Generic {
        algorithm_name: String,
        parameters: HashMap<String, serde_json::Value>,
        iteration: usize,
        completed: bool,
        statedata: serde_json::Value,
    },
}

/// Individual training step for history tracking
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingStep {
    /// Step number
    pub step: usize,
    /// Timestamp
    pub timestamp: u64,
    /// Algorithm-specific metrics at this step
    pub metrics: HashMap<String, f64>,
    /// Convergence status
    pub converged: bool,
    /// Memory usage at this step (optional)
    pub memory_usage: Option<usize>,
    /// Processing time for this step
    pub step_duration_ms: u64,
}

/// Workflow configuration
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorkflowConfig {
    /// Algorithm name
    pub algorithm: String,
    /// Hyperparameters
    pub hyperparameters: HashMap<String, serde_json::Value>,
    /// Auto-save interval (in steps, 0 = disabled)
    pub auto_save_interval: usize,
    /// Maximum history length to keep
    pub max_history_length: usize,
    /// Enable detailed logging
    pub enable_detailed_logging: bool,
}

impl ClusteringWorkflow {
    /// Create a new clustering workflow
    pub fn new(id: String, name: String, algorithm: String) -> Self {
        let timestamp = current_timestamp();
        Self {
            id,
            name,
            algorithm_state: AlgorithmState::Generic {
                algorithm_name: algorithm.clone(),
                parameters: HashMap::new(),
                iteration: 0,
                completed: false,
                statedata: serde_json::Value::Null,
            },
            data_hash: String::new(),
            training_history: Vec::new(),
            current_model: None,
            config: WorkflowConfig {
                algorithm,
                hyperparameters: HashMap::new(),
                auto_save_interval: 10,
                max_history_length: 1000,
                enable_detailed_logging: true,
            },
            created_at: timestamp,
            updated_at: timestamp,
        }
    }

    /// Update algorithm state
    pub fn update_state(&mut self, new_state: AlgorithmState) {
        self.algorithm_state = new_state;
        self.updated_at = current_timestamp();
    }

    /// Add a training step to history
    pub fn add_training_step(&mut self, step: TrainingStep) {
        self.training_history.push(step);

        // Limit history length
        if self.training_history.len() > self.config.max_history_length {
            self.training_history.remove(0);
        }

        self.updated_at = current_timestamp();
    }

    /// Set the final model
    pub fn set_model<T: SerializableModel>(&mut self, model: &T) -> Result<()> {
        let model_json = serde_json::to_value(model).map_err(|e| {
            ClusteringError::InvalidInput(format!("Failed to serialize model: {}", e))
        })?;
        self.current_model = Some(model_json);
        self.updated_at = current_timestamp();
        Ok(())
    }

    /// Get the final model
    pub fn get_model<T: SerializableModel>(&self) -> Result<Option<T>> {
        if let Some(ref model_json) = self.current_model {
            let model: T = serde_json::from_value(model_json.clone()).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to deserialize model: {}", e))
            })?;
            Ok(Some(model))
        } else {
            Ok(None)
        }
    }

    /// Check if training is completed
    pub fn is_completed(&self) -> bool {
        match &self.algorithm_state {
            AlgorithmState::KMeans { converged, .. } => *converged,
            AlgorithmState::Hierarchical { completed, .. } => *completed,
            AlgorithmState::DBSCAN { completed, .. } => *completed,
            AlgorithmState::Generic { completed, .. } => *completed,
        }
    }

    /// Get current iteration number
    pub fn current_iteration(&self) -> usize {
        match &self.algorithm_state {
            AlgorithmState::KMeans { iteration, .. } => *iteration,
            AlgorithmState::Hierarchical { merge_step, .. } => *merge_step,
            AlgorithmState::DBSCAN { .. } => 0, // DBSCAN doesn't have iterations
            AlgorithmState::Generic { iteration, .. } => *iteration,
        }
    }
}

impl SerializableModel for ClusteringWorkflow {}

/// Unified save/load system for clustering workflows
pub struct ClusteringWorkflowManager {
    /// Base directory for saving workflows
    base_dir: PathBuf,
    /// Auto-save configuration
    auto_save_config: AutoSaveConfig,
}

/// Auto-save configuration
#[derive(Debug, Clone)]
pub struct AutoSaveConfig {
    /// Enable auto-save
    pub enabled: bool,
    /// Save interval in training steps
    pub save_interval: usize,
    /// Keep backup copies
    pub keep_backups: bool,
    /// Maximum number of backups
    pub max_backups: usize,
}

impl Default for AutoSaveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            save_interval: 10,
            keep_backups: true,
            max_backups: 5,
        }
    }
}

impl ClusteringWorkflowManager {
    /// Create a new workflow manager
    pub fn new<P: Into<PathBuf>>(base_dir: P, _dir: P) -> Self {
        Self {
            base_dir: base_dir.into(),
            auto_save_config: AutoSaveConfig::default(),
        }
    }

    /// Save a clustering workflow
    pub fn save_workflow(&self, workflow: &ClusteringWorkflow) -> Result<PathBuf> {
        std::fs::create_dir_all(&self.base_dir).map_err(|e| {
            ClusteringError::InvalidInput(format!("Failed to create directory: {}", e))
        })?;

        let filepath = self.base_dir.join(format!("{}.workflow.json", workflow.id));
        workflow.save_to_file(&filepath)?;

        // Create backup if enabled
        if self.auto_save_config.keep_backups {
            self.create_backup(&filepath, &workflow.id)?;
        }

        Ok(filepath)
    }

    /// Load a clustering workflow
    pub fn load_workflow(&self, workflow_id: &str) -> Result<ClusteringWorkflow> {
        let filepath = self.base_dir.join(format!("{}.workflow.json", workflow_id));
        ClusteringWorkflow::load_from_file(filepath)
    }

    /// Resume training from a saved workflow
    pub fn resume_workflow(&self, workflow_id: &str) -> Result<ClusteringWorkflow> {
        let mut workflow = self.load_workflow(workflow_id)?;

        // Validate data consistency if needed
        if !workflow.data_hash.is_empty() {
            // In a real implementation, you would validate against the current training data
            // For now, we just update the timestamp
            workflow.updated_at = current_timestamp();
        }

        Ok(workflow)
    }

    /// List all saved workflows
    pub fn list_workflows(&self) -> Result<Vec<String>> {
        let mut workflows = Vec::new();

        if self.base_dir.exists() {
            for entry in std::fs::read_dir(&self.base_dir).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to read directory: {}", e))
            })? {
                let entry = entry.map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to read entry: {}", e))
                })?;
                let path = entry.path();

                if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                    if file_name.ends_with(".workflow.json") {
                        if let Some(workflow_id) = file_name.strip_suffix(".workflow.json") {
                            workflows.push(workflow_id.to_string());
                        }
                    }
                }
            }
        }

        Ok(workflows)
    }

    /// Create a backup of a workflow file
    fn create_backup(&self, originalpath: &Path, workflow_id: &str) -> Result<()> {
        let backup_dir = self.base_dir.join("backups");
        std::fs::create_dir_all(&backup_dir).map_err(|e| {
            ClusteringError::InvalidInput(format!("Failed to create backup directory: {}", e))
        })?;

        let timestamp = current_timestamp();
        let backuppath = backup_dir.join(format!("{}.{}.backup.json", workflow_id, timestamp));

        std::fs::copy(originalpath, &backuppath).map_err(|e| {
            ClusteringError::InvalidInput(format!("Failed to create backup: {}", e))
        })?;

        // Clean up old backups
        self.cleanup_old_backups(workflow_id)?;

        Ok(())
    }

    /// Clean up old backup files
    fn cleanup_old_backups(&self, workflow_id: &str) -> Result<()> {
        let backup_dir = self.base_dir.join("backups");
        if !backup_dir.exists() {
            return Ok(());
        }

        let prefix = format!("{}.", workflow_id);
        let mut backups = Vec::new();

        for entry in std::fs::read_dir(&backup_dir).map_err(|e| {
            ClusteringError::InvalidInput(format!("Failed to read backup directory: {}", e))
        })? {
            let entry = entry.map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to read backup entry: {}", e))
            })?;
            let path = entry.path();

            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                if file_name.starts_with(&prefix) && file_name.ends_with(".backup.json") {
                    backups.push(path);
                }
            }
        }

        // Sort by modification time and keep only the latest ones
        backups.sort_by_key(|path| {
            path.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH)
        });

        // Remove excess backups
        while backups.len() > self.auto_save_config.max_backups {
            let old_backup = backups.remove(0);
            let _ = std::fs::remove_file(old_backup); // Ignore errors
        }

        Ok(())
    }
}

/// Enhanced dendrogram export with JSON support
impl HierarchicalModel {
    /// Export dendrogram to JSON format
    pub fn to_json_dendrogram(&self) -> Result<String> {
        let dendrogramdata = self.build_jsontree_structure()?;
        serde_json::to_string_pretty(&dendrogramdata)
            .map_err(|e| ClusteringError::InvalidInput(format!("JSON serialization failed: {}", e)))
    }

    /// Build JSON tree structure for dendrogram
    fn build_jsontree_structure(&self) -> Result<serde_json::Value> {
        use serde_json::{json, Value};

        let n_samples = self.n_observations;
        let linkage = &self.linkage;

        if linkage.nrows() == 0 {
            return Ok(json!({
                "type": "dendrogram",
                "n_samples": n_samples,
                "tree": null,
                "metadata": {
                    "method": self.method,
                    "created_at": current_timestamp()
                }
            }));
        }

        // Build the tree recursively
        let rootnode = self.buildnode_recursive(linkage.nrows() - 1, linkage, n_samples)?;

        Ok(json!({
            "type": "dendrogram",
            "n_samples": n_samples,
            "method": self.method,
            "tree": rootnode,
            "metadata": {
                "linkage_method": self.method,
                "created_at": current_timestamp(),
                "format_version": "1.0",
                "totalnodes": linkage.nrows() + n_samples
            }
        }))
    }

    /// Build a node recursively for JSON export
    fn buildnode_recursive(
        &self,
        merge_idx: usize,
        linkage: &Array2<f64>,
        n_samples: usize,
    ) -> Result<serde_json::Value> {
        use serde_json::json;

        if merge_idx >= linkage.nrows() {
            return Err(ClusteringError::InvalidInput(format!(
                "Invalid merge index: {} >= {}",
                merge_idx,
                linkage.nrows()
            )));
        }

        let row = linkage.row(merge_idx);
        let left_id = row[0] as usize;
        let right_id = row[1] as usize;
        let distance = row[2];
        let count = row[3] as usize;

        // Determine if children are leaves or internal nodes
        let left_child = if left_id < n_samples {
            // Leaf node
            json!({
                "type": "leaf",
                "id": left_id,
                "label": self.labels.as_ref()
                    .and_then(|labels| labels.get(left_id))
                    .map(|s| s.as_str())
                    .unwrap_or(&format!("sample_{}", left_id)),
                "sample_id": left_id
            })
        } else {
            // Internal node
            let internal_idx = left_id - n_samples;
            self.buildnode_recursive(internal_idx, linkage, n_samples)?
        };

        let right_child = if right_id < n_samples {
            // Leaf node
            json!({
                "type": "leaf",
                "id": right_id,
                "label": self.labels.as_ref()
                    .and_then(|labels| labels.get(right_id))
                    .map(|s| s.as_str())
                    .unwrap_or(&format!("sample_{}", right_id)),
                "sample_id": right_id
            })
        } else {
            // Internal node
            let internal_idx = right_id - n_samples;
            self.buildnode_recursive(internal_idx, linkage, n_samples)?
        };

        Ok(json!({
            "type": "internal",
            "id": n_samples + merge_idx,
            "distance": distance,
            "count": count,
            "merge_order": merge_idx,
            "left": left_child,
            "right": right_child
        }))
    }
}

/// Import functions for scikit-learn and SciPy models
pub mod import {
    use super::*;
    /// Import K-means model from scikit-learn JSON format
    pub fn import_sklearn_kmeans<P: AsRef<Path>>(path: P) -> Result<KMeansModel> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to read file: {}", e)))?;

        let json_value: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to parse JSON: {}", e)))?;

        parse_sklearn_json_format(json_value)
    }

    /// Import hierarchical clustering from SciPy JSON format
    pub fn import_scipy_hierarchy<P: AsRef<Path>>(path: P) -> Result<HierarchicalModel> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to read file: {}", e)))?;

        let json_value: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to parse JSON: {}", e)))?;

        parse_scipy_json_format(json_value)
    }

    /// Export model to scikit-learn compatible JSON format
    pub fn export_to_sklearn_json<T: SerializableModel + AdvancedExport>(
        model: &T,
        path: &Path,
    ) -> Result<()> {
        let sklearn_format = crate::serialization::compatibility::to_sklearn_format(model)?;
        let json_string = serde_json::to_string_pretty(&sklearn_format).map_err(|e| {
            ClusteringError::InvalidInput(format!("JSON serialization failed: {}", e))
        })?;

        std::fs::write(path, json_string)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to write file: {}", e)))
    }

    /// Export model to SciPy compatible JSON format
    pub fn export_to_scipy_json<T: SerializableModel + AdvancedExport>(
        model: &T,
        path: &Path,
    ) -> Result<()> {
        let scipy_format = crate::serialization::compatibility::to_sklearn_format(model)?;
        let json_string = serde_json::to_string_pretty(&scipy_format).map_err(|e| {
            ClusteringError::InvalidInput(format!("JSON serialization failed: {}", e))
        })?;

        std::fs::write(path, json_string)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to write file: {}", e)))
    }
}

/// Parse scikit-learn JSON format
#[allow(dead_code)]
fn parse_sklearn_json_format(json: Value) -> Result<KMeansModel> {
    let cluster_centers = json
        .get("cluster_centers_")
        .ok_or_else(|| ClusteringError::InvalidInput("Missing cluster_centers_".to_string()))?;

    let n_clusters = json
        .get("n_clusters")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ClusteringError::InvalidInput("Missing or invalid n_clusters".to_string()))?
        as usize;

    let n_iter = json.get("n_iter_").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

    let inertia = json.get("inertia_").and_then(|v| v.as_f64()).unwrap_or(0.0);

    // Parse centroids array
    let centroidsdata = cluster_centers.as_array().ok_or_else(|| {
        ClusteringError::InvalidInput("Invalid cluster_centers_ format".to_string())
    })?;

    let mut centroid_values = Vec::new();
    let mut n_features = 0;

    for centroid in centroidsdata {
        let centroid_array = centroid
            .as_array()
            .ok_or_else(|| ClusteringError::InvalidInput("Invalid centroid format".to_string()))?;

        if n_features == 0 {
            n_features = centroid_array.len();
        } else if n_features != centroid_array.len() {
            return Err(ClusteringError::InvalidInput(
                "Inconsistent centroid dimensions".to_string(),
            ));
        }

        for value in centroid_array {
            let v = value.as_f64().ok_or_else(|| {
                ClusteringError::InvalidInput("Invalid centroid value".to_string())
            })?;
            centroid_values.push(v);
        }
    }

    let centroids =
        Array2::from_shape_vec((n_clusters, n_features), centroid_values).map_err(|e| {
            ClusteringError::InvalidInput(format!("Failed to create centroids array: {}", e))
        })?;

    Ok(KMeansModel::new(
        centroids, n_clusters, n_iter, inertia, None,
    ))
}

/// Parse SciPy JSON format
#[allow(dead_code)]
fn parse_scipy_json_format(json: Value) -> Result<HierarchicalModel> {
    let linkagedata = json
        .get("linkage")
        .ok_or_else(|| ClusteringError::InvalidInput("Missing linkage matrix".to_string()))?;

    let n_observations = json
        .get("n_observations")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ClusteringError::InvalidInput("Missing n_observations".to_string()))?
        as usize;

    let method = json
        .get("method")
        .and_then(|v| v.as_str())
        .unwrap_or("ward")
        .to_string();

    // Parse linkage matrix
    let linkage_array = linkagedata
        .as_array()
        .ok_or_else(|| ClusteringError::InvalidInput("Invalid linkage format".to_string()))?;

    let mut linkage_values = Vec::new();
    let n_merges = linkage_array.len();

    for row in linkage_array {
        let row_array = row.as_array().ok_or_else(|| {
            ClusteringError::InvalidInput("Invalid linkage row format".to_string())
        })?;

        if row_array.len() != 4 {
            return Err(ClusteringError::InvalidInput(
                "Linkage matrix must have 4 columns".to_string(),
            ));
        }

        for value in row_array {
            let v = value.as_f64().ok_or_else(|| {
                ClusteringError::InvalidInput("Invalid linkage value".to_string())
            })?;
            linkage_values.push(v);
        }
    }

    let linkage = Array2::from_shape_vec((n_merges, 4), linkage_values).map_err(|e| {
        ClusteringError::InvalidInput(format!("Failed to create linkage matrix: {}", e))
    })?;

    // Parse labels if present
    let labels = json.get("labels").and_then(|v| v.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect::<Vec<String>>()
    });

    Ok(HierarchicalModel::new(
        linkage,
        n_observations,
        method,
        labels,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tempfile::NamedTempFile;

    #[test]
    fn test_kmeans_serialization() {
        let centroids = array![[1.0, 2.0], [3.0, 4.0]];
        let labels = array![0, 1, 0, 1];
        let model = KMeansModel::new(centroids.clone(), 2, 10, 5.0, Some(labels.clone()));

        // Test JSON serialization
        let json = serde_json::to_string(&model).unwrap();
        let deserialized: KMeansModel = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.n_clusters, 2);
        assert_eq!(deserialized.n_iter, 10);
        assert_eq!(deserialized.inertia, 5.0);
        assert_eq!(deserialized.centroids, centroids);
        assert_eq!(deserialized.labels, Some(labels));
    }

    #[test]
    fn test_kmeans_save_load_file() {
        let centroids = array![[1.0, 2.0], [3.0, 4.0]];
        let model = KMeansModel::new(centroids.clone(), 2, 10, 5.0, None);

        let temp_file = NamedTempFile::new().unwrap();
        model.save_to_file(temp_file.path()).unwrap();

        let loaded_model = KMeansModel::load_from_file(temp_file.path()).unwrap();
        assert_eq!(loaded_model.n_clusters, model.n_clusters);
        assert_eq!(loaded_model.centroids, model.centroids);
    }

    #[test]
    fn test_hierarchical_to_newick() {
        // Simple linkage matrix
        let linkage = array![
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 3.0, 2.0, 2.0],
            [4.0, 5.0, 3.0, 4.0]
        ];

        let model = HierarchicalModel::new(linkage, 4, "single".to_string(), None);
        let newick = model.to_newick().unwrap();

        assert!(newick.ends_with(';'));
        assert!(newick.contains('('));
        assert!(newick.contains(')'));
    }

    #[test]
    fn test_hierarchical_to_json() {
        let linkage = array![[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 2.0, 2.0]];

        let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let model = HierarchicalModel::new(linkage, 3, "average".to_string(), Some(labels));

        let jsontree = model.to_jsontree().unwrap();

        assert!(jsontree.is_object());
        assert_eq!(jsontree["type"], "internal");
        assert!(jsontree["children"].is_array());
    }

    #[test]
    fn test_kmeans_predict() {
        let centroids = array![[0.0, 0.0], [5.0, 5.0]];
        let model = KMeansModel::new(centroids, 2, 10, 5.0, None);

        let testdata = array![[1.0, 1.0], [4.0, 4.0], [0.5, 0.5], [5.5, 5.5]];
        let predictions = model.predict(testdata.view()).unwrap();

        assert_eq!(predictions, array![0, 1, 0, 1]);
    }

    #[test]
    fn test_enhanced_model_metadata() {
        let metadata = EnhancedModelMetadata::default();

        assert_eq!(metadata.format_version, "1.0.0");
        assert_eq!(metadata.library_version, env!("CARGO_PKG_VERSION"));
        assert!(metadata.created_timestamp > 0);
        assert_eq!(metadata.platform_info.os, std::env::consts::OS);
        assert_eq!(metadata.platform_info.arch, std::env::consts::ARCH);
    }

    #[test]
    fn test_enhanced_model_integrity() {
        let centroids = array![[1.0, 2.0], [3.0, 4.0]];
        let model = KMeansModel::new(centroids, 2, 10, 5.0, None);
        let metadata = EnhancedModelMetadata::default();

        let enhanced_model = EnhancedModel::new(model, metadata);

        // Integrity should be valid after creation
        assert!(enhanced_model.validate_integrity().unwrap());
        assert!(enhanced_model.check_version_compatibility().unwrap());
        assert!(!enhanced_model.metadata.integrity_hash.is_empty());
    }

    #[test]
    fn test_enhanced_model_serialization() {
        let centroids = array![[1.0, 2.0], [3.0, 4.0]];
        let kmeans_model = KMeansModel::new(centroids, 2, 10, 5.0, None);

        let mut metadata = EnhancedModelMetadata::default();
        metadata.algorithm_signature = "kmeans-test".to_string();
        metadata.training_metrics.training_time_ms = 1000;
        metadata.training_metrics.iterations = 10;
        metadata.data_characteristics.n_samples = 100;
        metadata.data_characteristics.n_features = 2;

        let enhanced_model = EnhancedModel::new(kmeans_model, metadata);

        // Test JSON serialization
        let json = serde_json::to_string(&enhanced_model).unwrap();
        let deserialized: EnhancedModel<KMeansModel> = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.model.n_clusters, 2);
        assert_eq!(deserialized.metadata.algorithm_signature, "kmeans-test");
        assert_eq!(
            deserialized.metadata.training_metrics.training_time_ms,
            1000
        );
        assert!(deserialized.validate_integrity().unwrap());
    }

    #[test]
    fn test_platform_info_detection() {
        let platform_info = PlatformInfo::detect();

        assert!(!platform_info.os.is_empty());
        assert!(!platform_info.arch.is_empty());
        assert!(!platform_info.rust_version.is_empty());

        // CPU features detection should work on supported architectures
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            // Features may or may not be present, but the detection should not crash
            // CPU features should be detected (may be empty on some platforms)
            let _ = platform_info.cpu_features.len();
        }
    }

    #[test]
    fn test_training_metrics_defaults() {
        let metrics = TrainingMetrics::default();

        assert_eq!(metrics.training_time_ms, 0);
        assert_eq!(metrics.iterations, 0);
        assert_eq!(metrics.final_convergence_metric, 0.0);
        assert_eq!(metrics.peak_memory_bytes, 0);
        assert_eq!(metrics.avg_cpu_utilization, 0.0);
    }

    #[test]
    fn testdata_characteristics_defaults() {
        let data_chars = DataCharacteristics::default();

        assert_eq!(data_chars.n_samples, 0);
        assert_eq!(data_chars.n_features, 0);
        assert_eq!(data_chars.data_type_fingerprint, "unknown");
        assert!(data_chars.feature_ranges.is_none());
        assert!(data_chars.preprocessing_applied.is_empty());
    }
}
