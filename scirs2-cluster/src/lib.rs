#![allow(deprecated)]
#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(missing_docs)]
#![allow(clippy::for_loops_over_fallibles)]
#![allow(unused_parens)]
#![allow(unexpected_cfgs)]
#![allow(unused_attributes)]
#![allow(dead_code)]
//! Clustering algorithms module for SciRS2
//!
//! This module provides implementations of various clustering algorithms such as:
//! - Vector quantization (k-means, etc.)
//! - Hierarchical clustering
//! - Density-based clustering (DBSCAN, OPTICS, etc.)
//! - Mean Shift clustering
//! - Spectral clustering
//! - Affinity Propagation
//!
//! ## Features
//!
//! * **Vector Quantization**: K-means and K-means++ for partitioning data
//! * **Hierarchical Clustering**: Agglomerative clustering with various linkage methods
//! * **Density-based Clustering**: DBSCAN and OPTICS for finding clusters of arbitrary shape
//! * **Mean Shift**: Non-parametric clustering based on density estimation
//! * **Spectral Clustering**: Graph-based clustering using eigenvectors of the graph Laplacian
//! * **Affinity Propagation**: Message-passing based clustering that identifies exemplars
//! * **Evaluation Metrics**: Silhouette coefficient, Davies-Bouldin index, and other measures to evaluate clustering quality
//! * **Data Preprocessing**: Utilities for normalizing, standardizing, and whitening data before clustering
//!
//! ## Examples
//!
//! ```
//! use ndarray::{Array2, ArrayView2};
//! use scirs2_cluster::vq::kmeans;
//! use scirs2_cluster::preprocess::standardize;
//!
//! // Example data with two clusters
//! let data = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0,
//!     1.2, 1.8,
//!     0.8, 1.9,
//!     3.7, 4.2,
//!     3.9, 3.9,
//!     4.2, 4.1,
//! ]).unwrap();
//!
//! // Standardize the data
//! let standardized = standardize(data.view(), true).unwrap();
//!
//! // Run k-means with k=2
//! let (centroids, labels) = kmeans(standardized.view(), 2, None, None, None, None).unwrap();
//!
//! // Print the results
//! println!("Centroids: {:?}", centroids);
//! println!("Cluster assignments: {:?}", labels);
//! ```

/// Cutting-edge clustering algorithms including quantum-inspired methods and advanced online learning.
///
/// This module provides state-of-the-art clustering algorithms that push the boundaries
/// of traditional clustering methods. It includes quantum-inspired algorithms that leverage
/// quantum computing principles and advanced online learning variants with concept drift detection.
///
/// # Features
///
/// * **Quantum K-means**: Uses quantum superposition principles for potentially better optimization
/// * **Adaptive Online Clustering**: Automatically adapts to changing data distributions
/// * **Concept Drift Detection**: Detects and adapts to changes in streaming data
/// * **Dynamic Cluster Management**: Creates, merges, and removes clusters automatically
/// * **Quantum Annealing**: Simulated quantum annealing for global optimization
pub mod advanced;
/// Advanced benchmarking and performance profiling system.
///
/// This module provides cutting-edge benchmarking capabilities for clustering algorithms,
/// including comprehensive performance analysis, memory profiling, scalability analysis,
/// performance regression detection, and AI-powered optimization suggestions.
///
/// # Features
///
/// * **Statistical Performance Analysis**: Comprehensive timing statistics with confidence intervals
/// * **Memory Usage Profiling**: Real-time memory consumption tracking and leak detection
/// * **Scalability Analysis**: Algorithm complexity estimation and performance predictions
/// * **Regression Detection**: Automated detection of performance degradation
/// * **Optimization Suggestions**: AI-powered recommendations for performance improvements
/// * **Interactive Reporting**: Rich HTML reports with detailed analytics
/// * **Cross-Platform Benchmarking**: Performance comparisons across different systems
/// * **GPU vs CPU Analysis**: Comprehensive acceleration analysis
pub mod advanced_benchmarking;
/// Advanced Clustering - AI-Driven Quantum-Neuromorphic Clustering
///
/// This module represents the pinnacle of clustering intelligence, combining
/// AI-driven algorithm selection with quantum-neuromorphic fusion algorithms
/// to achieve unprecedented clustering performance. It leverages meta-learning,
/// neural architecture search, and bio-quantum computing paradigms.
///
/// # Revolutionary Advanced Features
///
/// * **AI-Driven Clustering Selection** - Automatically select optimal clustering algorithms
/// * **Quantum-Neuromorphic Clustering** - Fusion of quantum and spiking neural networks
/// * **Meta-Learning Optimization** - Learn optimal hyperparameters from experience
/// * **Adaptive Resource Allocation** - Dynamic GPU/CPU/QPU resource management
/// * **Multi-Objective Clustering** - Optimize for accuracy, speed, and interpretability
/// * **Continual Learning** - Adapt to changing data distributions in real-time
/// * **Bio-Quantum Clustering** - Nature-inspired quantum clustering algorithms
pub mod advanced_clustering;
/// Enhanced visualization specifically for advanced clustering results.
///
/// This module provides specialized visualization capabilities for advanced clustering,
/// including quantum state visualization, neuromorphic adaptation plots, and AI algorithm
/// selection insights with real-time interactive capabilities.
///
/// # Features
///
/// * **Quantum State Visualization**: Real-time coherence and entanglement plots
/// * **Neuromorphic Adaptation**: Spiking neuron activity and plasticity visualization
/// * **AI Algorithm Selection**: Performance predictions and selection insights
/// * **Performance Dashboard**: Comprehensive metrics comparison with classical methods
/// * **Export Capabilities**: Multiple formats including interactive HTML and JSON
pub mod advanced_visualization;
pub mod affinity;
pub mod birch;
pub mod density;
/// Distributed clustering algorithms for large-scale datasets.
///
/// This module provides distributed implementations of clustering algorithms that can
/// handle datasets too large to fit in memory on a single machine. It supports
/// distributed K-means, hierarchical clustering, and various data partitioning strategies.
///
/// # Features
///
/// * **Distributed K-means**: Multi-node K-means with coordination rounds
/// * **Distributed Hierarchical Clustering**: Large-scale hierarchical clustering
/// * **Data Partitioning**: Multiple strategies for distributing data across workers
/// * **Load Balancing**: Dynamic and static load balancing strategies
/// * **Memory Management**: Configurable memory limits and optimization
/// * **Fault Tolerance**: Worker failure detection and recovery mechanisms
pub mod distributed;
/// Enhanced Advanced Features - Advanced AI-Driven Clustering Extensions
///
/// This module extends the advanced clustering capabilities with cutting-edge
/// features including deep learning integration, quantum-inspired algorithms,
/// and advanced ensemble methods for superior clustering performance.
///
/// # Revolutionary Deep Learning Features
///
/// * **Transformer-Based Embeddings** - Deep representations using attention mechanisms
/// * **Graph Neural Networks** - Complex relationship modeling through graph convolutions
/// * **Reinforcement Learning** - Adaptive clustering strategy optimization
/// * **Neural Architecture Search** - Automatic design of optimal clustering networks
/// * **Deep Ensemble Methods** - Robust clustering through uncertainty quantification
/// * **Advanced Uncertainty Estimation** - Confidence intervals and reliability metrics
pub mod enhanced_clustering_features;
/// Ensemble clustering methods for improved robustness.
///
/// This module provides ensemble clustering techniques that combine multiple
/// clustering algorithms or multiple runs of the same algorithm to achieve
/// more robust and stable clustering results.
pub mod ensemble;
pub mod error;
pub mod gmm;
/// GPU acceleration module for clustering algorithms.
///
/// This module provides GPU acceleration interfaces and implementations for clustering
/// algorithms. It supports multiple GPU backends including CUDA, OpenCL, ROCm, and others.
/// When GPU acceleration is not available or disabled, algorithms automatically fall back
/// to optimized CPU implementations.
///
/// # Features
///
/// * **Multiple GPU Backends**: Support for CUDA, OpenCL, ROCm, Intel OneAPI, and Metal
/// * **Automatic Fallback**: Seamless fallback to CPU when GPU is not available
/// * **Memory Management**: Efficient GPU memory allocation and pooling
/// * **Performance Monitoring**: Built-in benchmarking and performance statistics
/// * **Device Selection**: Automatic or manual GPU device selection strategies
#[cfg(feature = "gpu")]
pub mod gpu;
/// Advanced GPU and Distributed Computing Extensions
///
/// This module provides GPU acceleration and distributed computing capabilities
/// for advanced clustering, enabling massive scalability and performance
/// improvements for large-scale clustering tasks.
///
/// # High-Performance Computing Features
///
/// * **GPU Acceleration** - CUDA/OpenCL/ROCm GPU acceleration with automatic fallback
/// * **Distributed Computing** - Multi-node clustering with fault tolerance
/// * **Hybrid GPU-Distributed** - Combined GPU and distributed processing
/// * **Advanced Memory Management** - Optimized GPU memory allocation and transfer
/// * **Load Balancing** - Dynamic workload distribution across nodes
/// * **Fault Tolerance** - Automatic recovery from worker node failures
pub mod gpu_distributed_clustering;
/// Graph clustering and community detection algorithms.
///
/// This module provides implementations of various graph clustering algorithms for
/// detecting communities and clusters in network data. These algorithms work with
/// graph representations where nodes represent data points and edges represent
/// similarities or connections between them.
///
/// # Features
///
/// * **Community Detection**: Louvain algorithm for modularity optimization
/// * **Label Propagation**: Fast algorithm for community detection
/// * **Hierarchical Methods**: Girvan-Newman algorithm for hierarchical communities
/// * **Graph Construction**: k-NN graphs, adjacency matrix support
/// * **Quality Metrics**: Modularity calculation and community evaluation
pub mod graph;
pub mod hierarchy;
pub mod input_validation;
pub mod leader;
/// Mean Shift clustering implementation.
///
/// This module provides the Mean Shift clustering algorithm, which is a centroid-based
/// algorithm that works by updating candidates for centroids to be the mean of the points
/// within a given region. These candidates are then filtered in a post-processing stage to
/// eliminate near-duplicates, forming the final set of centroids.
///
/// Mean Shift is a non-parametric clustering technique that doesn't require specifying the
/// number of clusters in advance and can find clusters of arbitrary shapes.
pub mod meanshift;
pub mod metrics;
pub mod neighbor_search;
/// Native plotting capabilities for clustering results.
///
/// This module provides native plotting implementations using popular Rust visualization
/// libraries like plotters and egui. It bridges the visualization data structures with
/// actual plotting backends to create publication-ready plots.
///
/// # Features
///
/// * **Static Plots**: PNG, SVG, PDF output using plotters
/// * **Interactive Plots**: Real-time visualization using egui
/// * **Publication Ready**: High-quality plots with customizable styling
/// * **Multiple Backends**: Support for different rendering backends
/// * **Performance Optimized**: Efficient rendering for large datasets
#[cfg(any(feature = "plotters", feature = "egui"))]
pub mod plotting;
pub mod preprocess;
/// Python bindings for scirs2-cluster using PyO3.
///
/// This module provides Python bindings that make scirs2-cluster algorithms
/// accessible from Python with scikit-learn compatible APIs. The bindings
/// include all major clustering algorithms and evaluation metrics.
///
/// # Features
///
/// * **Scikit-learn Compatible**: Drop-in replacements for scikit-learn clustering algorithms
/// * **K-means**: Python binding for K-means clustering with multiple initialization methods
/// * **DBSCAN**: Python binding for DBSCAN density-based clustering
/// * **Hierarchical**: Python binding for agglomerative clustering with various linkage methods
/// * **Evaluation Metrics**: Silhouette score, Calinski-Harabasz score, and Davies-Bouldin score
/// * **Numpy Integration**: Seamless integration with NumPy arrays
#[cfg(feature = "pyo3")]
pub mod python_bindings;
pub mod quantum_clustering;
pub mod serialization;
pub mod sparse;
pub mod spectral;
pub mod stability;
pub mod streaming;
/// Text clustering algorithms with semantic similarity support.
///
/// This module provides specialized clustering algorithms for text data that leverage
/// semantic similarity measures rather than traditional distance metrics. It includes
/// algorithms optimized for document clustering, sentence clustering, and topic modeling.
///
/// # Features
///
/// * **Semantic K-means**: K-means clustering with semantic similarity metrics
/// * **Hierarchical Text Clustering**: Agglomerative clustering for text data
/// * **Topic-based Clustering**: Clustering based on topic modeling approaches
/// * **Multiple Text Representations**: Support for TF-IDF, word embeddings, contextualized embeddings
/// * **Semantic Similarity Metrics**: Cosine, Jaccard, Jensen-Shannon, and other text-specific metrics
pub mod text_clustering;
/// Time series clustering algorithms with specialized distance metrics.
///
/// This module provides clustering algorithms specifically designed for time series data,
/// including dynamic time warping (DTW) distance and other temporal similarity measures.
/// These algorithms can handle time series of different lengths and temporal alignments.
///
/// # Features
///
/// * **Dynamic Time Warping**: DTW distance with optional constraints
/// * **Soft DTW**: Differentiable variant for gradient-based optimization
/// * **Time Series K-means**: Clustering with DTW barycenter averaging
/// * **Time Series K-medoids**: Robust clustering using actual time series as centers
/// * **Hierarchical Clustering**: Agglomerative clustering with DTW distance
pub mod time_series;
/// Automatic hyperparameter tuning for clustering algorithms.
///
/// This module provides comprehensive hyperparameter optimization capabilities
/// for all clustering algorithms in the scirs2-cluster crate. It supports
/// grid search, random search, Bayesian optimization, and adaptive strategies.
pub mod tuning;
/// Enhanced visualization capabilities for clustering results.
///
/// This module provides comprehensive visualization tools for clustering algorithms,
/// including scatter plots, 3D visualizations, dimensionality reduction plots,
/// and interactive exploration tools for high-dimensional data.
///
/// # Features
///
/// * **2D/3D Scatter Plots**: Create scatter plot visualizations of clustering results
/// * **Dimensionality Reduction**: Support for PCA, t-SNE, UMAP, and MDS for high-dimensional data
/// * **Color Schemes**: Multiple color palettes including colorblind-friendly options
/// * **Interactive Features**: Zoom, pan, and selection capabilities
/// * **Animation Support**: Animate iterative algorithms and streaming data
/// * **Export Capabilities**: Export to various formats (JSON, HTML, images)
pub mod visualization;
pub mod vq;

// Re-exports
pub use advanced::{
    adaptive_online_clustering, deep_embedded_clustering, qaoa_clustering, quantum_kmeans,
    rl_clustering, transfer_learning_clustering, variational_deep_embedding, vqe_clustering,
    AdaptiveOnlineClustering, AdaptiveOnlineConfig, DeepClusteringConfig, DeepEmbeddedClustering,
    FeatureAlignment, QAOAClustering, QAOAConfig, QAOACostFunction, QuantumConfig, QuantumKMeans,
    RLClustering, RLClusteringConfig, RewardFunction, TransferLearningClustering,
    TransferLearningConfig, VQEAnsatz, VQEClustering, VQEConfig, VariationalDeepEmbedding,
};

// Re-export quantum clustering from quantum_clustering module
pub use quantum_clustering::{
    quantum_annealing_clustering, CoolingSchedule, QuantumAnnealingClustering,
    QuantumAnnealingConfig,
};

// Re-export advanced clustering capabilities
pub use advanced_clustering::{
    AdvancedClusterer, AdvancedClusteringResult, AdvancedConfig, AdvancedPerformanceMetrics,
};

// Re-export advanced visualization capabilities
pub use advanced_visualization::{
    create_advanced_visualization_report, visualize_advanced_results, AISelectionPlot,
    AdvancedVisualizationConfig, AdvancedVisualizationOutput, AdvancedVisualizer, ClusterPlot,
    NeuromorphicAdaptationPlot, PerformanceDashboard, QuantumCoherencePlot, QuantumColorScheme,
    VisualizationExportFormat,
};

// Re-export enhanced advanced features
pub use enhanced_clustering_features::{
    DeepAdvancedClusterer, DeepAdvancedResult, DeepEnsembleCoordinator, EnsembleConsensus,
    GraphNeuralNetworkProcessor, GraphStructureInsights, NeuralArchitectureSearchEngine,
    OptimalArchitecture, ReinforcementLearningAgent, SpectralProperties,
    TransformerClusterEmbedder,
};

// Re-export GPU and distributed advanced features
pub use gpu_distributed_clustering::{
    CommunicationOverhead, CoordinationStrategy, DistributedAdvancedClusterer,
    DistributedAdvancedResult, DistributedProcessingMetrics, GpuAccelerationConfig,
    GpuAccelerationMetrics, GpuAdvancedClusterer, GpuAdvancedResult, GpuDeviceSelection,
    GpuMemoryStrategy, GpuOptimizationLevel, HybridGpuDistributedClusterer,
    HybridGpuDistributedResult, LoadBalancingStats, WorkerNodeConfig, WorkerPerformanceStats,
};

// Re-export advanced benchmarking capabilities
pub use advanced_benchmarking::{
    create_comprehensive_report, AdvancedBenchmark, AlgorithmBenchmark, AlgorithmComparison,
    BenchmarkConfig, BenchmarkResults, ComplexityClass, GpuVsCpuComparison, MemoryProfile,
    OptimizationCategory, OptimizationPriority, OptimizationSuggestion, PerformanceStatistics,
    QualityMetrics, RegressionAlert, RegressionSeverity, ScalabilityAnalysis, SystemInfo,
};

pub use affinity::{affinity_propagation, AffinityPropagationOptions};
pub use birch::{birch, Birch, BirchOptions, BirchStatistics};
pub use density::hdbscan::{
    dbscan_clustering, hdbscan, ClusterSelectionMethod, HDBSCANOptions, HDBSCANResult, StoreCenter,
};
pub use density::optics::{extract_dbscan_clustering, extract_xi_clusters, OPTICSResult};
pub use density::*;
pub use ensemble::convenience::{
    bootstrap_ensemble, ensemble_clustering, multi_algorithm_ensemble,
};
pub use ensemble::{
    ClusteringAlgorithm, ClusteringResult, ConsensusMethod, ConsensusStatistics, DiversityMetrics,
    DiversityStrategy, EnsembleClusterer, EnsembleConfig, EnsembleResult, NoiseType,
    ParameterRange, SamplingStrategy,
};
pub use gmm::{gaussian_mixture, CovarianceType, GMMInit, GMMOptions, GaussianMixture};
pub use graph::{
    girvan_newman, graph_clustering, label_propagation, louvain, Graph, GraphClusteringAlgorithm,
    GraphClusteringConfig,
};
pub use hierarchy::*;
pub use input_validation::{
    check_duplicate_points, suggest_clustering_algorithm, validate_clustering_data,
    validate_convergence_parameters, validate_distance_parameter, validate_integer_parameter,
    validate_n_clusters, validate_sample_weights, ValidationConfig,
};
pub use leader::{
    euclidean_distance, leader_clustering, manhattan_distance, LeaderClustering, LeaderNode,
    LeaderTree,
};
pub use meanshift::{estimate_bandwidth, get_bin_seeds, mean_shift, MeanShift, MeanShiftOptions};
pub use metrics::{
    adjusted_rand_index, calinski_harabasz_score, davies_bouldin_score,
    homogeneity_completeness_v_measure, normalized_mutual_info, silhouette_samples,
    silhouette_score,
};

// Re-export ensemble validation methods
pub use metrics::ensemble::{
    bootstrap_confidence_interval, consensus_clustering_score, cross_validation_score,
    multi_criterion_validation, robust_validation,
};

// Re-export information-theoretic methods
pub use metrics::information_theoretic::{
    jensen_shannon_divergence, normalized_variation_of_information,
};

// Re-export stability-based methods
pub use metrics::stability::{cluster_stability_bootstrap, optimal_clusters_stability};

// Re-export advanced metrics
pub use metrics::advanced::{bic_score, dunn_index};
pub use neighbor_search::{
    create_neighbor_searcher, BallTree, BruteForceSearch, KDTree, NeighborResult,
    NeighborSearchAlgorithm, NeighborSearchConfig, NeighborSearcher,
};
pub use preprocess::{min_max_scale, normalize, standardize, whiten, NormType};
pub use serialization::{
    affinity_propagation_to_model,
    birch_to_model,
    compatibility,
    dbscan_to_model,
    gmm_to_model,
    hierarchy_to_model,
    kmeans_to_model,
    leader_to_model,
    leadertree_to_model,
    meanshift_to_model,
    save_affinity_propagation,
    save_birch,
    save_gmm,
    save_hierarchy,
    save_kmeans,
    save_leader,
    save_leadertree,
    save_spectral_clustering,
    spectral_clustering_to_model,
    AdvancedExport,
    AffinityPropagationModel,
    AlgorithmState,
    AutoSaveConfig,
    BirchModel,
    // Unified workflow management
    ClusteringWorkflow,
    ClusteringWorkflowManager,
    DBSCANModel,
    DataCharacteristics,
    // Enhanced serialization with metadata and versioning
    EnhancedModel,
    EnhancedModelMetadata,
    ExportFormat,
    GMMModel,
    HierarchicalModel,
    KMeansModel,
    LeaderModel,
    LeaderNodeModel,
    LeaderTreeModel,
    MeanShiftModel,
    ModelMetadata,
    PlatformInfo,
    SerializableModel,
    SpectralClusteringModel,
    TrainingMetrics,
    TrainingStep,
    WorkflowConfig,
};

// Re-export compatibility utilities for scikit-learn and SciPy integration
pub use serialization::compatibility::{
    create_sklearn_param_grid,
    // TODO: Fix these function imports (they may be methods, not functions)
    // export_to_scipy_json,
    // export_to_sklearn_json,
    from_joblib_format,
    from_numpy_format,
    from_sklearn_format,
    generate_sklearn_model_summary,
    // import_scipy_hierarchy,
    // Import functions for external model formats
    // import_sklearn_kmeans,
    to_arrow_schema,
    to_huggingface_card,
    to_joblib_format,
    to_mlflow_format,
    to_numpy_format,
    to_onnx_metadata,
    to_pandas_clustering_report,
    to_pandas_format,
    to_pickle_like_format,
    to_pytorch_checkpoint,
    to_r_format,
    to_scipy_dendrogram_format,
    to_scipy_linkage_format,
    to_sklearn_clustering_result,
    to_sklearn_format,
};
pub use sparse::{
    sparse_epsilon_graph, sparse_knn_graph, SparseDistanceMatrix, SparseHierarchicalClustering,
};
pub use spectral::{
    spectral_bipartition, spectral_clustering, AffinityMode, SpectralClusteringOptions,
};
pub use stability::{
    BootstrapValidator, ConsensusClusterer, OptimalKSelector, StabilityConfig, StabilityResult,
};
pub use streaming::{
    ChunkedDistanceMatrix, ProgressiveHierarchical, StreamingConfig, StreamingKMeans,
};
pub use text_clustering::{
    semantic_hierarchical, semantic_kmeans, topic_clustering, SemanticClusteringConfig,
    SemanticHierarchical, SemanticKMeans, SemanticSimilarity, TextPreprocessing,
    TextRepresentation, TopicBasedClustering,
};
pub use time_series::{
    dtw_barycenter_averaging, dtw_distance, dtw_distance_custom, dtw_hierarchical_clustering,
    dtw_k_means, dtw_k_medoids, soft_dtw_distance, time_series_clustering, TimeSeriesAlgorithm,
    TimeSeriesClusteringConfig,
};
pub use tuning::{
    auto_select_clustering_algorithm,
    quick_algorithm_selection,
    AcquisitionFunction,
    AlgorithmSelectionResult,
    // Auto-selection functionality
    AutoClusteringSelector,
    AutoTuner,
    BayesianState,
    CVStrategy,
    ConvergenceInfo,
    CrossValidationConfig,
    EarlyStoppingConfig,
    EnsembleResults,
    EvaluationMetric,
    EvaluationResult,
    ExplorationStats,
    HyperParameter,
    KernelType,
    LoadBalancingStrategy,
    ParallelConfig,
    ResourceConstraints,
    SearchSpace,
    SearchStrategy,
    StandardSearchSpaces,
    StoppingReason,
    SurrogateModel,
    TuningConfig,
    TuningResult,
};

// Re-export visualization and animation capabilities
pub use visualization::{
    create_scatter_plot_2d, create_scatter_plot_3d, AnimationConfig, BoundaryType, ClusterBoundary,
    ColorScheme, DimensionalityReduction, EasingFunction, LegendEntry, ScatterPlot2D,
    ScatterPlot3D, VisualizationConfig,
};

// Re-export animation features
pub use visualization::animation::{
    AnimationFrame, IterativeAnimationConfig, IterativeAnimationRecorder, StreamingVisualizer,
};

// Re-export interactive visualization features
pub use visualization::interactive::{
    ClusterStats, InteractiveConfig, InteractiveState, InteractiveVisualizer,
};

// Re-export export capabilities
pub use visualization::export::{
    export_scatter_2d_to_html, export_scatter_2d_to_json, export_scatter_3d_to_html,
    export_scatter_3d_to_json, save_visualization_to_file,
};

// Re-export native plotting capabilities (when plotting features are enabled)
#[cfg(feature = "plotters")]
pub use plotting::{
    plot_dendrogram, plot_scatter_2d, save_clustering_plot, save_dendrogram_plot, PlotFormat,
    PlotOutput,
};

#[cfg(feature = "egui")]
pub use plotting::{launch_interactive_visualization, InteractiveClusteringApp};

// Re-export distributed clustering capabilities
pub use distributed::{
    DataPartition, DistributedKMeans, DistributedKMeansConfig, PartitioningStrategy, WorkerStatus,
};

// Re-export distributed utilities - not available in current implementation
// pub use distributed::utils::{estimate_optimal_workers, generate_large_dataset};
pub use vq::*;

// GPU acceleration re-exports (when GPU feature is enabled)
#[cfg(feature = "gpu")]
pub use gpu::{
    DeviceSelection, DistanceMetric as GpuDistanceMetric, GpuBackend, GpuConfig, GpuContext,
    GpuDevice, GpuDistanceMatrix, GpuKMeans, GpuKMeansConfig, GpuMemoryManager, GpuStats,
    MemoryStats, MemoryStrategy,
};

#[cfg(feature = "gpu")]
/// GPU acceleration benchmark utilities
pub mod gpu_benchmark {
    pub use crate::gpu::benchmark::*;
}

#[cfg(feature = "gpu")]
/// High-level GPU-accelerated clustering with automatic fallback
pub mod accelerated {
    pub use crate::gpu::accelerated::*;
}

// GPU acceleration interface (when GPU feature is enabled)
#[cfg(feature = "gpu")]
/// GPU-accelerated clustering with automatic CPU fallback
///
/// This module provides high-level clustering algorithms that automatically
/// use GPU acceleration when available, falling back to CPU implementations
/// when GPU is not available or optimal.
pub mod gpu_accelerated {
    pub use crate::gpu::accelerated::*;
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod property_tests;
