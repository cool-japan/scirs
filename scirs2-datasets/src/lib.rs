#![allow(deprecated)]
//! Datasets module for SciRS2
//!
//! This module provides dataset loading utilities similar to scikit-learn's datasets module.
//! It includes toy datasets, sample datasets, time series datasets, data generators,
//! and utilities for loading and processing datasets.
//!
//! # Features
//!
//! - **Toy datasets**: Classic datasets like Iris, Boston Housing, Breast Cancer, and Digits
//! - **Data generators**: Create synthetic datasets for classification, regression, clustering, and time series
//! - **Cross-validation utilities**: K-fold, stratified, and time series cross-validation
//! - **Dataset utilities**: Train/test splitting, normalization, and metadata handling
//! - **Caching**: Efficient caching system for downloaded datasets
//! - **Registry**: Centralized registry for dataset metadata and locations
//!
//! # Examples
//!
//! ## Loading toy datasets
//!
//! ```rust
//! use scirs2__datasets::{load_iris, load_boston};
//!
//! // Load the classic Iris dataset
//! let iris = load_iris().unwrap();
//! println!("Iris dataset: {} samples, {} features", iris.n_samples(), iris.n_features());
//!
//! // Load the Boston housing dataset
//! let boston = load_boston().unwrap();
//! println!("Boston dataset: {} samples, {} features", boston.n_samples(), boston.n_features());
//! ```
//!
//! ## Generating synthetic datasets
//!
//! ```rust
//! use scirs2__datasets::{make_classification, make_regression, make_blobs, make_spirals, make_moons};
//!
//! // Generate a classification dataset
//! let classification = make_classification(100, 5, 3, 2, 4, Some(42)).unwrap();
//! println!("Classification dataset: {} samples, {} features, {} classes",
//!          classification.n_samples(), classification.n_features(), 3);
//!
//! // Generate a regression dataset
//! let regression = make_regression(50, 4, 3, 0.1, Some(42)).unwrap();
//! println!("Regression dataset: {} samples, {} features",
//!          regression.n_samples(), regression.n_features());
//!
//! // Generate a clustering dataset
//! let blobs = make_blobs(80, 3, 4, 1.0, Some(42)).unwrap();
//! println!("Blobs dataset: {} samples, {} features, {} clusters",
//!          blobs.n_samples(), blobs.n_features(), 4);
//!
//! // Generate non-linear patterns
//! let spirals = make_spirals(200, 2, 0.1, Some(42)).unwrap();
//! let moons = make_moons(150, 0.05, Some(42)).unwrap();
//! ```
//!
//! ## Cross-validation
//!
//! ```rust
//! use scirs2__datasets::{load_iris, k_fold_split, stratified_k_fold_split};
//!
//! let iris = load_iris().unwrap();
//!
//! // K-fold cross-validation
//! let k_folds = k_fold_split(iris.n_samples(), 5, true, Some(42)).unwrap();
//! println!("Created {} folds for K-fold CV", k_folds.len());
//!
//! // Stratified K-fold cross-validation
//! if let Some(target) = &iris.target {
//!     let stratified_folds = stratified_k_fold_split(target, 5, true, Some(42)).unwrap();
//!     println!("Created {} stratified folds", stratified_folds.len());
//! }
//! ```
//!
//! ## Dataset manipulation
//!
//! ```rust
//! use scirs2__datasets::{load_iris, Dataset};
//!
//! let iris = load_iris().unwrap();
//!
//! // Access dataset properties
//! println!("Dataset: {} samples, {} features", iris.n_samples(), iris.n_features());
//! if let Some(featurenames) = iris.featurenames() {
//!     println!("Features: {:?}", featurenames);
//! }
//! ```

#![warn(missing_docs)]

pub mod advanced_generators;
pub mod benchmarks;
pub mod cache;
pub mod cloud;
pub mod distributed;
pub mod domain_specific;
pub mod error;
pub mod explore;
pub mod external;
pub mod generators;
pub mod gpu;
pub mod gpu_optimization;
pub mod loaders;
pub mod ml_integration;
pub mod real_world;
pub mod registry;
pub mod sample;
pub mod streaming;
pub mod time_series;
pub mod toy;
/// Core utilities for working with datasets
///
/// This module provides the Dataset struct and helper functions for
/// manipulating and transforming datasets.
pub mod utils;

/// API stability guarantees and compatibility documentation
///
/// This module defines the API stability levels and compatibility guarantees
/// for the scirs2-datasets crate.
pub mod stability;

// Temporary module to test method resolution conflict
mod method_resolution_test;

pub mod adaptive_streaming_engine;
pub mod neuromorphic_data_processor;
pub mod quantum_enhanced_generators;
pub mod quantum_neuromorphic_fusion;

// Re-export commonly used functionality
pub use adaptive_streaming_engine::{
    create_adaptive_engine, create_adaptive_engine_with_config, AdaptiveStreamConfig,
    AdaptiveStreamingEngine, AlertSeverity, AlertType, ChunkMetadata, DataCharacteristics,
    MemoryStrategy, PatternType, PerformanceMetrics, QualityAlert, QualityMetrics,
    StatisticalMoments, StreamChunk, TrendDirection, TrendIndicators,
};
pub use advanced_generators::{
    make_adversarial_examples, make_anomaly_dataset, make_continual_learning_dataset,
    make_domain_adaptation_dataset, make_few_shot_dataset, make_multitask_dataset,
    AdversarialConfig, AnomalyConfig, AnomalyType, AttackMethod, ContinualLearningDataset,
    DomainAdaptationConfig, DomainAdaptationDataset, FewShotDataset, MultiTaskConfig,
    MultiTaskDataset, TaskType,
};
pub use benchmarks::{BenchmarkResult, BenchmarkRunner, BenchmarkSuite, PerformanceComparison};
pub use cloud::{
    presets::{azure_client, gcs_client, public_s3_client, s3_client, s3_compatible_client},
    public_datasets::{AWSOpenData, AzureOpenData, GCPPublicData},
    CloudClient, CloudConfig, CloudCredentials, CloudProvider,
};
pub use distributed::{DistributedConfig, DistributedProcessor, ScalingMethod, ScalingParameters};
pub use domain_specific::{
    astronomy::StellarDatasets,
    climate::ClimateDatasets,
    convenience::{
        list_domain_datasets, load_atmospheric_chemistry, load_climate_data, load_exoplanets,
        load_gene_expression, load_stellar_classification,
    },
    genomics::GenomicsDatasets,
    DomainConfig, QualityFilters,
};
pub use explore::{
    convenience::{explore, export_summary, info, quick_summary},
    DatasetExplorer, DatasetSummary, ExploreConfig, FeatureStatistics, InferredDataType,
    OutputFormat, QualityAssessment,
};
#[cfg(not(feature = "download"))]
pub use external::convenience::{load_github_dataset_sync, load_uci_dataset_sync};
pub use external::{
    convenience::{list_uci_datasets, load_from_url_sync},
    repositories::{GitHubRepository, KaggleRepository, UCIRepository},
    ExternalClient, ExternalConfig, ProgressCallback,
};
pub use ml_integration::{
    convenience::{create_experiment, cv_split, prepare_for_ml, train_test_split},
    CrossValidationResults, DataSplit, MLExperiment, MLPipeline, MLPipelineConfig,
    ScalingMethod as MLScalingMethod,
};

pub use cache::{
    get_cachedir, BatchOperations, BatchResult, CacheFileInfo, CacheManager, CacheStats,
    DatasetCache, DetailedCacheStats,
};
#[cfg(feature = "download")]
pub use external::convenience::{load_from_url, load_github_dataset, load_uci_dataset};
pub use generators::{
    add_time_series_noise, benchmark_gpu_vs_cpu, get_gpu_info, gpu_is_available,
    inject_missing_data, inject_outliers, make_anisotropic_blobs, make_blobs, make_blobs_gpu,
    make_circles, make_classification, make_classification_gpu, make_corrupted_dataset, make_helix,
    make_hierarchical_clusters, make_intersecting_manifolds, make_manifold, make_moons,
    make_regression, make_regression_gpu, make_s_curve, make_severed_sphere, make_spirals,
    make_swiss_roll, make_swiss_roll_advanced, make_time_series, make_torus, make_twin_peaks,
    ManifoldConfig, ManifoldType, MissingPattern, OutlierType,
};
pub use gpu::{
    get_optimal_gpu_config, is_cuda_available, is_opencl_available, list_gpu_devices,
    make_blobs_auto_gpu, make_classification_auto_gpu, make_regression_auto_gpu, GpuBackend,
    GpuBenchmark, GpuBenchmarkResults, GpuConfig, GpuContext, GpuDeviceInfo, GpuMemoryConfig,
};
pub use gpu_optimization::{
    benchmark_advanced_performance, generate_advanced_matrix, AdvancedGpuOptimizer,
    AdvancedKernelConfig, BenchmarkResult as AdvancedBenchmarkResult, DataLayout,
    LoadBalancingMethod, MemoryAccessPattern, PerformanceBenchmarkResults, SpecializationLevel,
    VectorizationStrategy,
};
pub use loaders::{
    load_csv, load_csv_legacy, load_csv_parallel, load_csv_streaming, load_json, load_raw,
    save_json, CsvConfig, DatasetChunkIterator, StreamingConfig,
};
pub use neuromorphic_data_processor::{
    create_neuromorphic_processor, create_neuromorphic_processor_with_topology, NetworkTopology,
    NeuromorphicProcessor, NeuromorphicTransform, SynapticPlasticity,
};
pub use quantum_enhanced_generators::{
    make_quantum_blobs, make_quantum_classification, make_quantum_regression,
    QuantumDatasetGenerator,
};
pub use quantum_neuromorphic_fusion::{
    create_fusion_with_params, create_quantum_neuromorphic_fusion, QuantumBioFusionResult,
    QuantumInterference, QuantumNeuromorphicFusion,
};
pub use real_world::{
    list_real_world_datasets, load_adult, load_california_housing, load_heart_disease,
    load_red_wine_quality, load_titanic, RealWorldConfig, RealWorldDatasets,
};
pub use registry::{get_registry, load_dataset_byname, DatasetMetadata, DatasetRegistry};
pub use sample::*;
pub use streaming::{
    stream_classification, stream_csv, stream_regression, DataChunk, StreamConfig, StreamProcessor,
    StreamStats, StreamTransformer, StreamingIterator,
};
pub use toy::*;
pub use utils::{
    analyze_dataset_advanced, create_balanced_dataset, create_binned_features,
    generate_synthetic_samples, importance_sample, k_fold_split, min_max_scale,
    polynomial_features, quick_quality_assessment, random_oversample, random_sample,
    random_undersample, robust_scale, statistical_features, stratified_k_fold_split,
    stratified_sample, time_series_split, AdvancedDatasetAnalyzer, AdvancedQualityMetrics,
    BalancingStrategy, BinningStrategy, CorrelationInsights, CrossValidationFolds, Dataset,
    NormalityAssessment,
};
