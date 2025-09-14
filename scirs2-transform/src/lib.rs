#![allow(deprecated)]
//! Data transformation module for SciRS2
//!
//! This module provides utilities for transforming data in ways that are useful
//! for machine learning and data analysis. The main functionalities include:
//!
//! - Data normalization and standardization
//! - Feature engineering
//! - Dimensionality reduction

#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]

/// Error handling for the transformation module
pub mod error;

/// Basic normalization methods for data
pub mod normalize;

/// Feature engineering techniques
pub mod features;

/// Dimensionality reduction algorithms
pub mod reduction;

/// Matrix decomposition techniques
pub mod decomposition;

/// Advanced scaling and transformation methods
pub mod scaling;

/// Missing value imputation utilities
pub mod impute;

/// Categorical data encoding utilities
pub mod encoding;

/// Feature selection utilities
pub mod selection;

/// Time series feature extraction
pub mod time_series;

/// Pipeline API for chaining transformations
pub mod pipeline;

/// SIMD-accelerated normalization operations
#[cfg(feature = "simd")]
pub mod normalize_simd;

/// SIMD-accelerated feature engineering operations
#[cfg(feature = "simd")]
pub mod features_simd;

/// SIMD-accelerated scaling operations
#[cfg(feature = "simd")]
pub mod scaling_simd;

/// Out-of-core processing for large datasets
pub mod out_of_core;

/// Streaming transformations for continuous data
pub mod streaming;

/// Text processing transformers
pub mod text;

/// Image processing transformers
pub mod image;

/// Utility functions and helpers for data transformation
pub mod utils;

/// Test module for advanced implementations
#[cfg(test)]
mod advanced_test;
/// Performance optimizations and enhanced implementations
pub mod performance;

/// Optimization configuration and auto-tuning system
pub mod optimization_config;

/// Graph embedding transformers
pub mod graph;

/// GPU-accelerated transformations
#[cfg(feature = "gpu")]
pub mod gpu;

/// Distributed processing for multi-node transformations
#[cfg(feature = "distributed")]
pub mod distributed;

/// Automated feature engineering with meta-learning
pub mod auto_feature_engineering;

/// Quantum-inspired optimization for data transformations
pub mod quantum_optimization;

/// Neuromorphic computing integration for real-time adaptation
pub mod neuromorphic_adaptation;

/// Production monitoring with drift detection
#[cfg(feature = "monitoring")]
pub mod monitoring;

// Re-export important types and functions
pub use decomposition::{DictionaryLearning, NMF};
pub use encoding::{
    BinaryEncoder, EncodedOutput, FrequencyEncoder, OneHotEncoder, OrdinalEncoder, SparseMatrix,
    TargetEncoder, WOEEncoder,
};
pub use error::{Result, TransformError};
pub use features::{
    binarize, discretize_equal_frequency, discretize_equal_width, log_transform, power_transform,
    PolynomialFeatures, PowerTransformer,
};
pub use impute::{
    DistanceMetric, ImputeStrategy, IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer,
    WeightingScheme,
};
pub use normalize::{normalize_array, normalize_vector, NormalizationMethod, Normalizer};
pub use pipeline::{
    make_column_transformer, make_pipeline, ColumnTransformer, Pipeline, RemainderOption,
    Transformer,
};
pub use reduction::{
    trustworthiness, AffinityMethod, Isomap, SpectralEmbedding, TruncatedSVD, LDA, LLE, PCA, TSNE,
    UMAP,
};
pub use scaling::{MaxAbsScaler, QuantileTransformer};
pub use selection::{MutualInfoSelector, RecursiveFeatureElimination, VarianceThreshold};
pub use time_series::{FourierFeatures, LagFeatures, TimeSeriesFeatures, WaveletFeatures};

#[cfg(feature = "simd")]
pub use normalize__simd::{
    simd_l2_normalize_1d, simd_maxabs_normalize_1d, simd_minmax_normalize_1d,
    simd_normalize_adaptive, simd_normalize_array, simd_normalize_batch, simd_zscore_normalize_1d,
    AdaptiveBlockSizer,
};

#[cfg(feature = "simd")]
pub use features__simd::{
    simd_binarize, simd_polynomial_features_optimized, simd_power_transform, SimdPolynomialFeatures,
};

#[cfg(feature = "simd")]
pub use scaling__simd::{SimdMaxAbsScaler, SimdRobustScaler, SimdStandardScaler};

pub use graph::{
    adjacency_to_edge_list, edge_list_to_adjacency, ActivationType, DeepWalk, GraphAutoencoder,
    LaplacianType, Node2Vec,
};
pub use image::{
    resize_images, rgb_to_grayscale, BlockNorm, HOGDescriptor, ImageNormMethod, ImageNormalizer,
    PatchExtractor,
};
pub use optimization_config::{
    AdaptiveParameterTuner, AdvancedConfigOptimizer, AutoTuner, ConfigurationPredictor,
    DataCharacteristics, OptimizationConfig, OptimizationReport, PerformanceMetric, SystemMonitor,
    SystemResources, TransformationRecommendation,
};
pub use out_of_core::{
    csv_chunks, ChunkedArrayReader, ChunkedArrayWriter, OutOfCoreConfig, OutOfCoreNormalizer,
    OutOfCoreTransformer,
};
pub use performance::{EnhancedPCA, EnhancedStandardScaler};
pub use streaming::{
    OutlierMethod, StreamingFeatureSelector, StreamingMinMaxScaler, StreamingOutlierDetector,
    StreamingPCA, StreamingQuantileTracker, StreamingStandardScaler, StreamingTransformer,
    WindowedStreamingTransformer,
};
pub use text::{CountVectorizer, HashingVectorizer, StreamingCountVectorizer, TfidfVectorizer};
pub use utils::{
    ArrayMemoryPool, DataChunker, PerfUtils, ProcessingStrategy, StatUtils, TypeConverter,
    ValidationUtils,
};

// GPU acceleration exports
#[cfg(feature = "gpu")]
pub use gpu::{GpuMatrixOps, GpuPCA, GpuTSNE};

// Distributed processing exports
#[cfg(feature = "distributed")]
pub use distributed::{
    AutoScalingConfig, CircuitBreaker, ClusterHealthSummary, DistributedConfig,
    DistributedCoordinator, DistributedPCA, EnhancedDistributedCoordinator, NodeHealth, NodeInfo,
    NodeStatus, PartitioningStrategy,
};

// Automated feature engineering exports
pub use auto_feature_engineering::{
    AdvancedMetaLearningSystem, AutoFeatureEngineer, DatasetMetaFeatures, EnhancedMetaFeatures,
    MultiObjectiveRecommendation, TransformationConfig, TransformationType,
};

// Quantum optimization exports
pub use quantum_optimization::{
    AdvancedQuantumMetrics, AdvancedQuantumOptimizer, AdvancedQuantumParams,
    QuantumHyperparameterTuner, QuantumInspiredOptimizer, QuantumParticle,
    QuantumTransformationOptimizer,
};

// Neuromorphic computing exports
pub use neuromorphic_adaptation::{
    AdvancedNeuromorphicMetrics, AdvancedNeuromorphicProcessor, NeuromorphicAdaptationNetwork,
    NeuromorphicMemorySystem, NeuromorphicTransformationSystem, SpikingNeuron, SystemState,
    TransformationEpisode,
};

// Production monitoring exports
#[cfg(feature = "monitoring")]
pub use monitoring::{
    AlertConfig, AlertType, DriftDetectionResult, DriftMethod, PerformanceMetrics,
    TransformationMonitor,
};
