//! Model serialization and deserialization
//!
//! This module provides comprehensive serialization capabilities for clustering models,
//! including metadata enrichment, cross-platform compatibility, and workflow management.
//!
//! # Examples
//!
//! ## Basic Model Serialization
//!
//! ```rust
//! use scirs2_cluster::serialization::{SerializableModel, EnhancedModel};
//! use scirs2_core::ndarray::Array2;
//!
//! // Pretend we trained a KMeans and have centroids
//! let centroids = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
//! let model = scirs2_cluster::KMeansModel::new(centroids, 3, 10, 0.5, None);
//!
//! // Save model with enhanced metadata
//! let enhanced_model = EnhancedModel::with_auto_metadata(model, "kmeans");
//! enhanced_model.save_to_file("model.json").unwrap();
//!
//! // Load model back
//! let loaded_model: EnhancedModel<scirs2_cluster::KMeansModel> = EnhancedModel::load_from_file("model.json").unwrap();
//! ```
//!
//! ## Advanced Export with Multiple Formats
//!
//! ```rust
//! use scirs2_cluster::serialization::{AdvancedExport, ExportFormat, ModelMetadata};
//! use scirs2_core::ndarray::Array2;
//! use scirs2_cluster::serialization::utils::create_default_metadata;
//!
//! // Export model in different formats
//! let centroids = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
//! let model = scirs2_cluster::KMeansModel::new(centroids, 2, 10, 0.1, None);
//! let metadata = create_default_metadata("kmeans");
//! let json_data = model.export_with_metadata(ExportFormat::Json, Some(metadata)).unwrap();
//! // YAML export requires enabling the `yaml` feature
//!
//! // Export for compatibility with other libraries
//! let sklearn_format = model.export_compatible("sklearn").unwrap();
//! let pytorch_format = model.export_compatible("pytorch").unwrap();
//! ```
//!
//! ## Workflow Management
//!
//! ```rust
//! use scirs2_cluster::serialization::{ClusteringWorkflow, WorkflowConfig, TrainingStep};
//!
//! // Create and manage clustering workflows
//! let config = WorkflowConfig::default();
//! let mut workflow = ClusteringWorkflow::new("my_experiment".to_string(), config);
//!
//! // Add and execute a simple step
//! workflow.add_step(TrainingStep {
//!     name: "data_preprocessing".to_string(),
//!     algorithm: "kmeans".to_string(),
//!     parameters: Default::default(),
//!     dependencies: vec![],
//!     completed: false,
//!     execution_time: None,
//!     results: None,
//! });
//! workflow.execute().unwrap();
//!
//! // Save and resume workflow state
//! // Save and load checkpoint using default path in config
//! // Note: to actually create a checkpoint file, set `checkpoint_dir` in `WorkflowConfig`.
//! ```

pub mod compatibility;
pub mod core;
pub mod export;
pub mod models;
pub mod workflow;

use scirs2_core::ndarray::Array1;

// Re-export main types for convenience
pub use core::{
    format_timestamp, DataCharacteristics, EnhancedModel, EnhancedModelMetadata, PlatformInfo,
    SerializableModel, TrainingMetrics,
};

pub use export::{
    AdvancedExport, AlgorithmConfig, ConvergenceInfo, ExportFormat, ExportSettings, FeatureStats,
    ModelDataCharacteristics, ModelInfo, ModelMetadata, PerformanceMetrics,
};

pub use models::{
    // Conversion functions
    affinity_propagation_to_model,
    birch_to_model,
    dbscan_to_model,
    gmm_to_model,
    hierarchy_to_model,
    kmeans_to_model,
    leader_to_model,
    leadertree_to_model,
    meanshift_to_model,
    // Save functions
    save_affinity_propagation,
    save_birch,
    save_dbscan,
    save_gmm,
    save_hierarchy,
    save_kmeans,
    save_leader,
    save_leadertree,
    save_meanshift,
    save_spectral_clustering,
    spectral_clustering_to_model,
    AffinityPropagationModel,
    BirchModel,
    ClusteringModel,
    DBSCANModel,
    GMMModel,
    HierarchicalModel,
    KMeansModel,
    LeaderModel,
    LeaderTreeModel,
    MeanShiftModel,
    SpectralClusteringModel,
    SpectralModel,
};

pub use workflow::{
    AlgorithmState, AutoSaveConfig, ClusteringWorkflow, ClusteringWorkflowManager, ExecutionRecord,
    StepResult, TrainingStep, WorkflowConfig, WorkflowState, WorkflowStep,
};

// Re-export utility modules
pub use export::utils;

// Re-export compatibility functions
pub use compatibility::*;

/// Convenience function to create a serializable K-means model
pub fn create_kmeans_model(
    centroids: scirs2_core::ndarray::Array2<f64>,
    n_clusters: usize,
    n_iter: usize,
    inertia: f64,
    labels: Option<scirs2_core::ndarray::Array1<usize>>,
) -> KMeansModel {
    KMeansModel::new(centroids, n_clusters, n_iter, inertia, labels)
}

/// Convenience function to create a serializable DBSCAN model
pub fn create_dbscan_model(
    core_sample_indices: Vec<usize>,
    components: scirs2_core::ndarray::Array2<f64>,
    labels: scirs2_core::ndarray::Array1<i32>,
    eps: f64,
    min_samples: usize,
) -> DBSCANModel {
    DBSCANModel::new(
        Array1::from_vec(core_sample_indices),
        labels,
        eps,
        min_samples,
    )
}

/// Convenience function to create a serializable hierarchical clustering model
pub fn create_hierarchical_model(
    n_clusters: usize,
    labels: scirs2_core::ndarray::Array1<usize>,
    linkage_matrix: scirs2_core::ndarray::Array2<f64>,
    distances: Vec<f64>,
) -> HierarchicalModel {
    HierarchicalModel::new(linkage_matrix, n_clusters, "ward".to_string(), None)
}

/// Convenience function to create enhanced model metadata
pub fn create_enhanced_metadata(algorithm_name: &str) -> EnhancedModelMetadata {
    let mut metadata = EnhancedModelMetadata::default();
    metadata.algorithm_signature = algorithm_name.to_string();
    metadata
}

/// Convenience function to create default export settings
pub fn default_export_settings() -> ExportSettings {
    ExportSettings::default()
}

/// Convenience function to export model to multiple formats
pub fn export_model_multi_format<T: AdvancedExport>(
    model: &T,
    base_path: &str,
    formats: &[ExportFormat],
    metadata: Option<ModelMetadata>,
) -> crate::error::Result<()> {
    for format in formats {
        let extension = match format {
            ExportFormat::Json => "json",
            ExportFormat::JsonGz => "json.gz",
            ExportFormat::Yaml => "yaml",
            ExportFormat::Csv => "csv",
            ExportFormat::Xml => "xml",
            ExportFormat::Hdf5 => "h5",
            ExportFormat::Binary => "bin",
            ExportFormat::Custom(ext) => ext,
        };

        let file_path = format!("{}.{}", base_path, extension);
        let data = model.export_with_metadata(format.clone(), metadata.clone())?;
        std::fs::write(file_path, data).map_err(|e| {
            crate::error::ClusteringError::InvalidInput(format!("Failed to write file: {}", e))
        })?;
    }
    Ok(())
}

/// Convenience function to validate model before serialization
pub fn validate_model_for_serialization<T: AdvancedExport>(model: &T) -> crate::error::Result<()> {
    model.validate_for_export()
}

/// Convenience function to create a workflow with default configuration
pub fn create_default_workflow(name: String) -> ClusteringWorkflow {
    ClusteringWorkflow::new(name, WorkflowConfig::default())
}

/// Batch export multiple models with different formats
pub fn batch_export_models<T: AdvancedExport>(
    models: &[(String, &T)],
    base_directory: &str,
    format: ExportFormat,
    metadata_fn: Option<fn(&str) -> ModelMetadata>,
) -> crate::error::Result<()> {
    std::fs::create_dir_all(base_directory).map_err(|e| {
        crate::error::ClusteringError::InvalidInput(format!("Failed to create directory: {}", e))
    })?;

    for (name, model) in models {
        let metadata = metadata_fn.map(|f| f(name));
        let file_path = std::path::Path::new(base_directory).join(name);
        let data = model.export_with_metadata(format.clone(), metadata)?;
        std::fs::write(file_path, data).map_err(|e| {
            crate::error::ClusteringError::InvalidInput(format!(
                "Failed to write model {}: {}",
                name, e
            ))
        })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_create_kmeans_model() {
        let centroids = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let model = create_kmeans_model(centroids, 2, 10, 0.5, None);
        assert_eq!(model.n_clusters, 2);
        assert_eq!(model.n_iter, 10);
    }

    #[test]
    fn test_create_dbscan_model() {
        let core_samples = vec![0, 1, 2];
        let components =
            Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        let labels = scirs2_core::ndarray::Array1::from_vec(vec![0, 0, 1]);
        let model = create_dbscan_model(core_samples, components, labels, 0.5, 2);
        assert_eq!(model.eps, 0.5);
        assert_eq!(model.min_samples, 2);
    }

    #[test]
    fn test_enhanced_metadata_creation() {
        let metadata = create_enhanced_metadata("test_algorithm");
        assert_eq!(metadata.algorithm_signature, "test_algorithm");
        assert!(!metadata.platform_info.os.is_empty());
    }

    #[test]
    fn test_default_export_settings() {
        let settings = default_export_settings();
        assert!(settings.include_raw_data);
        assert!(!settings.include_training_data);
        assert_eq!(settings.float_precision, Some(6));
    }

    #[test]
    fn test_create_default_workflow() {
        let workflow = create_default_workflow("test_workflow".to_string());
        assert_eq!(workflow.workflow_id, "test_workflow");
        assert!(matches!(workflow.current_state, AlgorithmState::NotStarted));
    }
}
