#![allow(deprecated)]
//! Computer vision module for SciRS2
//!
//! This module provides computer vision functionality that builds on top of the
//! scirs2-ndimage module, including image processing, feature detection, and segmentation.
//!
//! # Thread Safety
//!
//! All functions in this module are thread-safe and can be called concurrently on different images.
//! When the `parallel` feature is enabled (via scirs2-core), many algorithms will automatically
//! utilize multiple CPU cores for improved performance on large images. Key parallel optimizations include:
//!
//! - Gradient computations in edge detection algorithms
//! - Pixel-wise operations in preprocessing functions
//! - Feature detection algorithms that process image regions independently
//! - Segmentation algorithms with parallelizable clustering steps
//!
//! Note that while functions are thread-safe, mutable image data should not be shared between threads
//! without proper synchronization.

#![warn(missing_docs)]

// Re-export image crate with the expected name
// (using standard import instead of deprecated extern crate)

pub mod color;
pub mod error;
pub mod feature;
pub mod gpu_ops;
/// Image preprocessing functionality
///
/// Includes operations like filtering, histogram manipulation,
/// and morphological operations.
pub mod preprocessing;
pub mod quality;
pub mod registration;
pub mod segmentation;
pub mod simd_ops;
pub mod streaming;

// Advanced mode enhancements - cutting-edge computer vision
pub mod ai_optimization;
pub mod neuromorphic_streaming;
pub mod quantum_inspired_streaming;
pub mod transform;

// Advanced Advanced-mode modules - future development features
pub mod activity_recognition;
pub mod scene_understanding;
pub mod visual_reasoning;
pub mod visual_slam;

// Cross-module Advanced coordination
/// Advanced Integration - Cross-Module AI Coordination
///
/// This module provides the highest level of AI integration across all SciRS2 modules,
/// combining quantum-inspired processing, neuromorphic computing, advanced AI optimization,
/// and cross-module coordination into a unified Advanced processing framework.
///
/// # Features
///
/// * **Cross-Module Coordination** - Unified Advanced across vision, clustering, spatial, neural
/// * **Global Optimization** - Multi-objective optimization across all modules
/// * **Unified Meta-Learning** - Cross-module transfer learning and adaptation
/// * **Resource Management** - Optimal allocation of computational resources
/// * **Performance Tracking** - Comprehensive monitoring and optimization
pub mod integration;

/// Advanced Performance Benchmarking for Advanced Mode
///
/// This module provides comprehensive performance benchmarking capabilities
/// for all Advanced mode features, including quantum-inspired processing,
/// neuromorphic computing, AI optimization, and cross-module coordination.
///
/// # Features
///
/// * **Comprehensive Benchmarking** - Full performance analysis across all Advanced features
/// * **Statistical Analysis** - Advanced statistical metrics and trend analysis
/// * **Resource Monitoring** - Detailed resource usage tracking and optimization
/// * **Quality Assessment** - Accuracy, consistency, and quality metrics
/// * **Scalability Analysis** - Performance scaling with different workloads
/// * **Comparative Analysis** - Speedup and advantage measurements vs baseline
pub mod performance_benchmark;

// Comment out problematic modules during tests to focus on fixing other issues
#[cfg(not(test))]
/// Private transform module for compatibility
///
/// Contains placeholder modules that help maintain compatibility
/// with external code that might reference these modules directly.
pub mod _transform {
    /// Non-rigid transformation compatibility module
    pub mod non_rigid {}
    /// Perspective transformation compatibility module
    pub mod perspective {}
}

// Re-export commonly used items
pub use error::{Result, VisionError};

// Re-export feature functionality (select items to avoid conflicts)
pub use feature::{
    array_to_image,
    descriptor::{detect_and_compute, match_descriptors, Descriptor, KeyPoint},
    harris_corners,
    image_to_array,
    laplacian::{laplacian_edges, laplacian_of_gaussian},
    log_blob::{log_blob_detect, log_blobs_to_image, LogBlob, LogBlobConfig},
    orb::{detect_and_compute_orb, match_orb_descriptors, OrbConfig, OrbDescriptor},
    prewitt::prewitt_edges,
    sobel::sobel_edges_simd,
    sobel_edges,
    AdvancedDenoiser,
    AppearanceExtractor,
    AttentionFeatureMatcher,
    DeepSORT,
    DenoisingMethod,
    Detection,
    // Advanced enhancement features
    HDRProcessor,
    KalmanFilter,
    LearnedSIFT,
    NeuralFeatureConfig,
    NeuralFeatureMatcher,
    SIFTConfig,
    // Neural features
    SuperPointNet,
    SuperResolutionMethod,
    SuperResolutionProcessor,
    ToneMappingMethod,
    Track,
    TrackState,
    // Advanced tracking features
    TrackingBoundingBox,
    TrackingMetrics,
};
// Re-export with unique name to avoid ambiguity
pub use feature::homography::warp_perspective as feature_warp_perspective;

// Re-export segmentation functionality
pub use segmentation::*;

// Re-export preprocessing functionality
pub use preprocessing::*;

// Re-export color functionality
pub use color::*;

// Re-export transform functionality (select items to avoid conflicts)
pub use transform::{
    affine::{estimate_affine_transform, warp_affine, AffineTransform, BorderMode},
    non_rigid::{
        warp_elastic, warp_non_rigid, warp_thin_plate_spline, ElasticDeformation, ThinPlateSpline,
    },
    perspective::{correct_perspective, BorderMode as PerspectiveBorderMode, PerspectiveTransform},
};
// Re-export with unique name to avoid ambiguity
pub use transform::perspective::warp_perspective as transform_warp_perspective;

// Re-export SIMD operations
pub use simd_ops::{
    check_simd_support, simd_convolve_2d, simd_gaussian_blur, simd_histogram_equalization,
    simd_normalize_image, simd_sobel_gradients, SimdPerformanceStats,
};

// Re-export GPU operations
pub use gpu_ops::{
    gpu_batch_process, gpu_convolve_2d, gpu_gaussian_blur, gpu_harris_corners, gpu_sobel_gradients,
    GpuBenchmark, GpuMemoryStats, GpuVisionContext,
};

// Re-export streaming operations
pub use streaming::{
    BatchProcessor, BlurStage, EdgeDetectionStage, Frame, FrameMetadata, GrayscaleStage,
    MotionDetectionStage, PerformanceMonitor, PipelineMetrics, ProcessingStage, StreamPipeline,
    StreamProcessor, VideoStreamReader,
};

// Re-export Advanced mode enhancements
pub use quantum_inspired_streaming::{
    ProcessingDecision, QuantumAdaptiveStreamPipeline, QuantumAmplitude, QuantumAnnealingStage,
    QuantumEntanglementStage, QuantumProcessingState, QuantumStreamProcessor,
    QuantumSuperpositionStage,
};

pub use neuromorphic_streaming::{
    AdaptiveNeuromorphicPipeline, EfficiencyMetrics, EventDrivenProcessor, EventStats,
    NeuromorphicEdgeDetector, NeuromorphicMode, NeuromorphicProcessingStats, PlasticSynapse,
    SpikingNeuralNetwork, SpikingNeuron,
};

pub use ai_optimization::{
    ArchitecturePerformance, GeneticPipelineOptimizer, NeuralArchitectureSearch, PerformanceMetric,
    PipelineGenome, PredictiveScaler, ProcessingArchitecture, RLParameterOptimizer,
    ScalingRecommendation, SearchStrategy,
};

// Re-export advanced Advanced-mode features
pub use scene_understanding::{
    analyze_scene_with_reasoning, ContextualReasoningEngine, DetectedObject as SceneObject,
    SceneAnalysisResult, SceneGraph, SceneUnderstandingEngine, SpatialRelation,
    SpatialRelationType, TemporalInfo,
};

pub use visual_reasoning::{
    perform_advanced_visual_reasoning, QueryType, ReasoningAnswer, ReasoningStep,
    UncertaintyQuantification, VisualReasoningEngine, VisualReasoningQuery, VisualReasoningResult,
};

pub use activity_recognition::{
    monitor_activities_realtime, recognize_activities_comprehensive, ActivityRecognitionEngine,
    ActivityRecognitionResult, ActivitySequence, ActivitySummary as ActivitySceneSummary,
    DetectedActivity, MotionCharacteristics, PersonInteraction, TemporalActivityModeler,
};

pub use visual_slam::{
    process_visual_slam, process_visual_slam_realtime, CameraPose, CameraTrajectory, LoopClosure,
    Map3D, SLAMResult, SLAMSystemState, SemanticMap, VisualSLAMSystem,
};

// Re-export Advanced integration functionality
pub use integration::{
    batch_process_advanced, process_with_advanced_mode, realtime_advanced_stream,
    AdvancedProcessingResult, CrossModuleAdvancedProcessingResult, EmergentBehavior, FusionQuality,
    NeuralQuantumHybridProcessor, PerformanceMetrics,
    UncertaintyQuantification as AdvancedUncertaintyQuantification,
};

// Re-export performance benchmarking functionality
pub use performance_benchmark::{
    AdvancedBenchmarkSuite, BenchmarkConfig, BenchmarkResult, ComparisonMetrics,
    PerformanceMetrics as BenchmarkPerformanceMetrics, QualityMetrics, ResourceUsage,
    ScalabilityMetrics, StatisticalSummary,
};
