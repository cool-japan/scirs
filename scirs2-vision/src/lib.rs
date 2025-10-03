#![allow(deprecated)]
//! # SciRS2 Computer Vision
//!
//! **scirs2-vision** provides comprehensive computer vision and image processing capabilities
//! built on SciRS2's scientific computing infrastructure, offering torchvision/OpenCV-compatible
//! APIs with Rust's performance and safety.
//!
//! ## 🎯 Key Features
//!
//! - **Feature Detection**: SIFT, ORB, Harris corners, LoG blob detection
//! - **Image Segmentation**: Watershed, region growing, graph cuts, semantic segmentation
//! - **Edge Detection**: Sobel, Canny, Prewitt, Laplacian
//! - **Image Registration**: Homography, affine, perspective transformations
//! - **Color Processing**: Color space conversions, histogram operations
//! - **Advanced Processing**: Super-resolution, HDR, denoising
//! - **Object Tracking**: DeepSORT, Kalman filtering
//! - **Performance**: SIMD acceleration, parallel processing, GPU support
//!
//! ## 📦 Module Overview
//!
//! | Module | Description | Python Equivalent |
//! |--------|-------------|-------------------|
//! | [`feature`] | Feature detection and matching (SIFT, ORB, Harris) | OpenCV features2d |
//! | [`segmentation`] | Image segmentation algorithms | scikit-image.segmentation |
//! | [`preprocessing`] | Image filtering and enhancement | torchvision.transforms |
//! | [`color`] | Color space conversions | OpenCV color |
//! | [`transform`] | Geometric transformations | torchvision.transforms |
//! | [`registration`] | Image registration and alignment | scikit-image.registration |
//! | [`streaming`] | Real-time video processing | OpenCV VideoCapture |
//!
//! ## 🚀 Quick Start
//!
//! ### Installation
//!
//! Add to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! scirs2-vision = "0.1.0-rc.1"
//! ```
//!
//! ### Feature Detection (Harris Corners)
//!
//! ```rust,no_run
//! use scirs2_vision::harris_corners;
//! use image::open;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Load image
//!     let img = open("image.jpg")?;
//!
//!     // Detect Harris corners
//!     let corners = harris_corners(&img, 3, 0.04, 100.0)?;
//!     println!("Detected Harris corners");
//!
//!     Ok(())
//! }
//! ```
//!
//! ### SIFT Feature Detection
//!
//! ```rust,no_run
//! use scirs2_vision::detect_and_compute;
//! use image::open;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let img = open("image.jpg")?;
//!
//!     // Detect SIFT keypoints and compute descriptors
//!     let descriptors = detect_and_compute(&img, 500, 0.03)?;
//!     println!("Detected {} SIFT features", descriptors.len());
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Edge Detection (Sobel)
//!
//! ```rust,no_run
//! use scirs2_vision::sobel_edges;
//! use image::open;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let img = open("image.jpg")?;
//!
//!     // Detect edges using Sobel operator
//!     let edges = sobel_edges(&img, 0.1)?;
//!     println!("Edge map computed");
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Image Segmentation (Watershed)
//!
//! ```rust,no_run
//! use scirs2_vision::watershed;
//! use image::open;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let img = open("image.jpg")?;
//!
//!     // Perform watershed segmentation
//!     let segments = watershed(&img, None, 8)?;
//!     println!("Segmented into regions");
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Color Space Conversion
//!
//! ```rust,no_run
//! use scirs2_vision::{rgb_to_grayscale, rgb_to_hsv};
//! use image::open;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let img = open("image.jpg")?;
//!
//!     // Convert to grayscale
//!     let gray = rgb_to_grayscale(&img, None)?;
//!
//!     // Convert to HSV
//!     let hsv = rgb_to_hsv(&img)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Image Registration (Homography)
//!
//! ```rust,no_run
//! use scirs2_vision::find_homography;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Source and destination points
//!     let src_points = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
//!     let dst_points = vec![(10.0, 10.0), (110.0, 5.0), (105.0, 105.0), (5.0, 110.0)];
//!
//!     // Find homography matrix
//!     let (_h, _inliers) = find_homography(&src_points, &dst_points, 3.0, 0.99)?;
//!
//!     // Warp image using homography
//!     // let warped = warp_perspective(&img, &h, (width, height))?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Super-Resolution
//!
//! ```rust,no_run
//! use scirs2_vision::{SuperResolutionProcessor, SuperResolutionMethod};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let processor = SuperResolutionProcessor::new(
//!         2, // scale factor
//!         SuperResolutionMethod::ESRCNN
//!     )?;
//!
//!     // Upscale image
//!     // let upscaled = processor.process(&low_res_image)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Object Tracking (DeepSORT)
//!
//! ```rust,no_run
//! use scirs2_vision::{DeepSORT, Detection, TrackingBoundingBox};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut tracker = DeepSORT::new();
//!
//!     // For each frame
//!     let bbox = TrackingBoundingBox::new(10.0, 10.0, 50.0, 50.0, 0.9, 0);
//!     let detections = vec![Detection::new(bbox)];
//!
//!     let tracks = tracker.update(detections)?;
//!     for track in tracks {
//!         println!("Track ID: {}, Position: {:?}", track.id, track.get_bbox());
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## 🧠 Feature Detection Methods
//!
//! ### Classical Features
//!
//! - **Harris Corners**: Fast corner detection algorithm
//! - **SIFT**: Scale-Invariant Feature Transform (rotation and scale invariant)
//! - **ORB**: Oriented FAST and Rotated BRIEF (efficient alternative to SIFT)
//! - **LoG Blobs**: Laplacian of Gaussian blob detection
//!
//! ### Neural Features
//!
//! - **SuperPoint**: Learned keypoint detector and descriptor
//! - **Learned SIFT**: Neural network-enhanced SIFT
//!
//! ### Feature Matching
//!
//! - **Brute-force matching**: Exhaustive descriptor comparison
//! - **Neural matching**: Attention-based feature matching
//!
//! ## 🎨 Segmentation Methods
//!
//! ### Classical Segmentation
//!
//! - **Watershed**: Marker-based segmentation
//! - **Region Growing**: Similarity-based region merging
//! - **Graph Cuts**: Energy minimization segmentation
//! - **Threshold**: Otsu, adaptive thresholding
//!
//! ### Advanced Segmentation
//!
//! - **Semantic Segmentation**: Deep learning-based pixel classification
//! - **Instance Segmentation**: Object detection + segmentation
//!
//! ## 🚄 Performance
//!
//! scirs2-vision leverages multiple optimization strategies:
//!
//! - **SIMD**: Vectorized operations for pixel processing
//! - **Parallel**: Multi-threaded execution for large images
//! - **GPU**: CUDA/OpenCL support for accelerated operations
//! - **Memory Efficient**: Zero-copy views, chunked processing
//!
//! ### Thread Safety
//!
//! All functions are thread-safe and can be called concurrently on different images.
//! When the `parallel` feature is enabled, many algorithms automatically utilize
//! multiple CPU cores:
//!
//! - Gradient computations in edge detection
//! - Pixel-wise operations in preprocessing
//! - Feature detection across image regions
//! - Segmentation clustering steps
//!
//! Note: Mutable image data should not be shared between threads without proper synchronization.
//!
//! ## 📊 Comparison with Other Libraries
//!
//! | Feature | OpenCV | torchvision | scirs2-vision |
//! |---------|--------|-------------|---------------|
//! | Feature Detection | ✅ | ❌ | ✅ |
//! | Segmentation | ✅ | ⚠️ (limited) | ✅ |
//! | Geometric Transforms | ✅ | ✅ | ✅ |
//! | GPU Support | ✅ | ✅ | ✅ (limited) |
//! | Type Safety | ❌ | ❌ | ✅ |
//! | Memory Safety | ❌ | ⚠️ | ✅ |
//! | Pure Rust | ❌ | ❌ | ✅ |
//!
//! ## 🔗 Integration with SciRS2 Ecosystem
//!
//! - **scirs2-ndimage**: Low-level image operations
//! - **scirs2-linalg**: Matrix operations for geometric transforms
//! - **scirs2-neural**: Deep learning models for advanced tasks
//! - **scirs2-stats**: Statistical analysis of image features
//! - **scirs2-cluster**: Clustering for segmentation
//!
//! ## 🔒 Version
//!
//! Current version: **0.1.0-rc.1** (Released October 03, 2025)

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
    find_homography,
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
    AdvancedStreamPipeline, AdvancedStreamProcessor, BlurStage, EdgeDetectionStage, Frame,
    FrameMetadata, GrayscaleStage, PipelineMetrics, ProcessingStage, SimplePerformanceMonitor,
    StreamPipeline, StreamProcessor, VideoStreamReader,
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
    AdvancedProcessingResult, CrossModuleAdvancedProcessingResult, EmergentBehaviorDetection,
    FusionQualityIndicators, NeuralQuantumHybridProcessor, PerformanceMetrics,
    UncertaintyQuantification as AdvancedUncertaintyQuantification,
};

// Re-export performance benchmarking functionality
pub use performance_benchmark::{
    AdvancedBenchmarkSuite, BenchmarkConfig, BenchmarkResult, ComparisonMetrics,
    PerformanceMetrics as BenchmarkPerformanceMetrics, QualityMetrics, ResourceUsage,
    ScalabilityMetrics, StatisticalSummary,
};
