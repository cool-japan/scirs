#![allow(deprecated)]
#![allow(clippy::new_without_default)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_question_mark)]
#![allow(clippy::inconsistent_digit_grouping)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::unwrap_or_default)]
#![allow(clippy::single_char_add_str)]
#![allow(clippy::map_flatten)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::useless_vec)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::option_as_ref_deref)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::unnecessary_lazy_evaluations)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::needless_return)]
#![allow(clippy::let_and_return)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::bool_comparison)]
#![allow(clippy::vec_init_then_push)]
#![allow(clippy::useless_format)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::map_entry)]
#![allow(clippy::map_clone)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_map)]
#![allow(clippy::manual_dangling_ptr)]
#![allow(clippy::manual_contains)]
#![allow(clippy::manual_c_str_literals)]
#![allow(clippy::implicit_saturating_sub)]
#![allow(clippy::format_in_format_args)]
#![allow(clippy::for_kv_map)]
#![allow(clippy::extra_unused_type_parameters)]
#![allow(clippy::explicit_counter_loop)]
#![allow(clippy::explicit_auto_deref)]
#![allow(clippy::empty_line_after_outer_attr)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(missing_docs)]
#![allow(for_loops_over_fallibles)]
#![allow(unused_parens)]
#![allow(unexpected_cfgs)]
#![allow(unused_attributes)]
#![allow(dead_code)]
//! # SciRS2 NDImage - N-Dimensional Image Processing
//!
//! **scirs2-ndimage** provides comprehensive N-dimensional image processing modeled after SciPy's
//! `ndimage` module, offering filtering, morphology, interpolation, measurements, segmentation,
//! and feature detection with GPU acceleration and chunked processing for large images.
//!
//! ## 🎯 Key Features
//!
//! - **SciPy Compatibility**: Drop-in replacement for `scipy.ndimage` functions
//! - **Comprehensive Filters**: Gaussian, median, Sobel, bilateral, anisotropic diffusion
//! - **Morphological Operations**: Erosion, dilation, opening, closing, skeletonization
//! - **Image Measurements**: Center of mass, moments, label statistics
//! - **Segmentation**: Watershed, region growing, active contours
//! - **Feature Detection**: Edge detection (Canny, Sobel), corner detection (Harris)
//! - **GPU Acceleration**: CUDA/ROCm for large-scale image processing
//! - **Memory Efficient**: Chunked processing for images larger than RAM
//!
//! ## 📦 Module Overview
//!
//! | SciRS2 Module | SciPy Equivalent | Description |
//! |---------------|------------------|-------------|
//! | `filters` | `scipy.ndimage.filters` | Convolution, Gaussian, median filters |
//! | `morphology` | `scipy.ndimage.morphology` | Erosion, dilation, binary operations |
//! | `interpolation` | `scipy.ndimage.interpolation` | Rotation, zoom, shift, affine transforms |
//! | `measurements` | `scipy.ndimage.measurements` | Label, center of mass, histograms |
//! | `segmentation` | - | Watershed, region growing, active contours |
//! | `features` | - | Edge detection, corner detection, keypoints |
//!
//! ## 🚀 Quick Start
//!
//! ```toml
//! [dependencies]
//! scirs2-ndimage = "0.1.0-rc.1"
//! ```
//!
//! ```rust,no_run
//! use scirs2_ndimage::filters;
//! use scirs2_core::ndarray::Array2;
//!
//! // Gaussian filter
//! let image = Array2::<f64>::zeros((256, 256));
//! let filtered = filters::gaussian_filter(&image, 2.0, None, None).unwrap();
//! ```
//!
//! ## 🔒 Version: 0.1.0-rc.1 (October 03, 2025)

// Public modules
pub mod adaptive_image_optimizer;
pub mod advanced_fusion_algorithms;
pub mod advanced_streaming_ai;
pub mod ai_driven_adaptive_processing;
pub mod analysis;
pub mod api_compatibility_verification;
pub mod backend;
pub mod biological_vision_inspired;
pub mod chunked;
pub mod chunked_v2;
pub mod comprehensive_examples;
pub mod comprehensive_scipy_benchmarks;
pub mod comprehensive_scipy_validation;
pub mod comprehensive_validation;
pub mod domain_specific;
pub mod error;
pub mod features;
pub mod filters;
pub mod gpu_operations;
pub mod hyperdimensional_computing;
pub mod interpolation;
pub mod measurements;
pub mod memory_management;
pub mod meta_learning_algorithms;
pub mod mmap_io;
pub mod morphology;
pub mod neuromorphic_computing;
pub mod performance_profiler;
pub mod profiling;
pub mod python_interop;
pub mod quantum_ai_consciousness;
pub mod quantum_enhanced_gpu;
pub mod quantum_inspired;
pub mod quantum_neuromorphic_fusion;
pub mod scipy_compat;
pub mod scipy_compat_layer;
pub mod scipy_migration_layer;
pub mod scipy_performance_comparison;
#[cfg(test)]
mod scipy_validation_tests;
pub mod segmentation;
pub mod streaming;
pub mod threading;
pub mod utils;
pub mod visualization;

// Re-exports
pub use self::error::*;

// Feature detection module exports
pub use self::features::{
    canny,
    edge_detector,
    edge_detector_simple,
    fast_corners,
    gradient_edges,
    harris_corners,
    laplacian_edges,
    sobel_edges,
    BatchNormParams,
    EdgeDetectionAlgorithm,
    EdgeDetectionConfig,
    FeatureDetectorWeights,
    GradientMethod,
    // Machine learning-based detection
    LearnedEdgeDetector,
    LearnedKeypointDescriptor,
    MLDetectorConfig,
    ObjectProposal,
    ObjectProposalGenerator,
    SemanticFeatureExtractor,
};

// Filters module exports
pub use self::filters::{
    // Advanced filters
    adaptive_bilateral_filter,
    adaptive_wiener_filter,
    anisotropic_diffusion,
    bilateral_filter,
    bilateral_gradient_filter,
    coherence_enhancing_diffusion,
    convolve,
    // Wavelets
    dwt_1d,
    dwt_2d,
    filter_functions,
    gabor_filter,
    gabor_filter_bank,
    gaussian_filter,
    gaussian_filter_chunked,
    gaussian_filter_f32,
    gaussian_filter_f64,
    generic_filter,
    idwt_1d,
    idwt_2d,
    laplace,
    log_gabor_filter,
    maximum_filter,
    median_filter,
    median_filter_chunked,
    minimum_filter,
    multi_scale_bilateral_filter,
    non_local_means,
    percentile_filter,
    rank_filter,
    shock_filter,
    sobel,
    steerable_filter,
    uniform_filter,
    uniform_filter_chunked,
    wavelet_decompose,
    wavelet_denoise,
    wavelet_reconstruct,
    BorderMode,
    GaborParams,
    MultiScaleBilateralConfig,
    WaveletFamily,
    WaveletFilter,
};

#[cfg(feature = "simd")]
pub use self::filters::{bilateral_filter_simd_f32, bilateral_filter_simd_f64};

// Segmentation module exports
pub use self::segmentation::{
    active_contour,
    adaptive_threshold,
    chan_vese,
    chan_vese_multiphase,
    checkerboard_level_set,
    create_circle_contour,
    create_ellipse_contour,
    // Advanced segmentation algorithms
    graph_cuts,
    marker_watershed,
    mask_to_contour,
    mask_to_level_set,
    otsu_threshold,
    smooth_contour,
    threshold_binary,
    watershed,
    ActiveContourParams,
    AdaptiveMethod,
    ChanVeseParams,
    GraphCutsParams,
    InteractiveGraphCuts,
};

// Interpolation module exports
pub use self::interpolation::{
    affine_transform, bspline, geometric_transform, map_coordinates, rotate, shift, spline_filter,
    spline_filter1d, value_at_coordinates, zoom, BoundaryMode, InterpolationOrder,
};

// Measurements module exports
pub use self::measurements::{
    center_of_mass, count_labels, extrema, find_objects, local_extrema, mean_labels, moments,
    moments_inertia_tensor, peak_prominences, peak_widths, region_properties, sum_labels,
    variance_labels, RegionProperties,
};

// Morphology module exports
pub use self::morphology::{
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    binary_hit_or_miss,
    binary_opening,
    black_tophat,
    box_structure,
    disk_structure,
    distance_transform_bf,
    distance_transform_cdt,
    // Distance transform functions
    distance_transform_edt,
    find_boundaries,
    generate_binary_structure,
    geodesic_dilation_2d,
    geodesic_erosion_2d,
    granulometry_2d,
    grey_closing,
    grey_dilation,
    grey_erosion,
    grey_opening,
    iterate_structure,
    label,
    morphological_gradient,
    morphological_laplace,
    morphological_reconstruction_2d,
    multi_scale_morphology_2d,
    remove_small_holes,
    remove_small_objects,
    white_tophat,
    Connectivity,
    DistanceMetric,
    MorphBorderMode,
    MorphOperation,
    MultiScaleMorphConfig,
    StructureType,
};

// Memory management exports
pub use self::memory_management::{
    check_memory_limit, create_output_array, estimate_memory_usage, BufferPool, InPlaceOp,
    MemoryConfig, MemoryEfficientOp, MemoryStrategy,
};

// Chunked processing exports
pub use self::chunked::{process_chunked, ChunkConfig, ChunkProcessor, GaussianChunkProcessor};

// Backend exports
pub use self::backend::{
    auto_backend, Backend, BackendBuilder, BackendConfig, BackendExecutor, BackendOp,
};

// Threading exports
pub use self::threading::{
    configure_parallel_ops, get_thread_pool_config, init_thread_pool, update_thread_pool_config,
    AdaptiveThreadPool, ThreadPoolConfig, ThreadPoolContext, WorkStealingQueue, WorkerInfo,
};

// Streaming exports
pub use self::streaming::{
    stream_process_file, StreamConfig, StreamProcessor, StreamableOp, StreamingGaussianFilter,
};

// Domain-specific imaging exports
pub use self::domain_specific::{
    medical::{
        detect_lung_nodules, enhance_bone_structure, frangi_vesselness, Nodule,
        VesselEnhancementParams,
    },
    microscopy::{
        colocalization_analysis, detect_nuclei, segment_cells, CellInfo, CellSegmentationParams,
        ColocalizationMetrics, ThresholdMethod,
    },
    satellite::{compute_ndvi, detect_clouds, detect_water_bodies, pan_sharpen, PanSharpenMethod},
};

// Analysis module exports
pub use self::analysis::{
    batch_quality_assessment, compute_local_variance, contrast_to_noise_ratio,
    estimate_fractal_dimension, image_entropy, image_quality_assessment, image_sharpness,
    local_feature_analysis, mean_absolute_error, mean_squared_error, multi_scale_analysis,
    peak_signal_to_noise_ratio, signal_to_noise_ratio, structural_similarity_index,
    texture_analysis, ImageQualityMetrics, MultiScaleConfig, TextureMetrics,
};

// SIMD-optimized analysis functions
#[cfg(feature = "simd")]
pub use self::analysis::{compute_moments_simd_f32, image_quality_assessment_simd_f32};

// Parallel analysis functions
#[cfg(feature = "parallel")]
pub use self::analysis::image_entropy_parallel;

// Visualization module exports
pub use self::visualization::{
    create_colormap, createimage_montage, generate_report, plot_contour, plot_heatmap,
    plot_histogram, plot_profile, plot_statistical_comparison, plot_surface, visualize_gradient,
    ColorMap, PlotConfig, ReportConfig, ReportFormat,
};

// SciPy performance comparison exports
pub use self::scipy_performance_comparison::{
    calculate_accuracy_metrics, validate_api_compatibility, AccuracyResult, BenchmarkConfig,
    CompatibilityResult, PerformanceResult, SciPyBenchmarkSuite,
};

// API compatibility verification exports
pub use self::api_compatibility_verification::{
    ApiCompatibilityResult, ApiCompatibilityTester, CompatibilityConfig, ParameterTest,
};

// Comprehensive SciPy validation exports
pub use self::comprehensive_scipy_validation::{
    SciPyValidationSuite, ValidationConfig as SciPyValidationConfig, ValidationResult,
};

// Comprehensive examples exports
pub use self::comprehensive_examples::{validate_all_examples, ExampleTutorial, TutorialStep};

// Quantum-inspired processing exports
pub use self::quantum_inspired::{
    quantum_amplitude_amplification, quantum_annealing_segmentation,
    quantum_entanglement_correlation, quantum_error_correction, quantum_fourier_enhancement,
    quantum_machine_learning_classifier, quantum_superposition_filter,
    quantum_tensor_network_processing, quantum_variational_enhancement,
    quantum_walk_edge_detection, QuantumConfig, QuantumState,
};

// Neuromorphic computing exports
pub use self::neuromorphic_computing::{
    event_driven_processing, homeostatic_adaptive_filter, liquidstate_machine,
    spiking_neural_network_filter, stdp_unsupervised_learning, temporal_coding_feature_extraction,
    Event, NeuromorphicConfig, PlasticSynapse, SpikingNeuron,
};

// Advanced fusion core exports
pub use self::advanced_fusion_algorithms::{
    enhanced_meta_learning_with_temporal_fusion, enhanced_quantum_consciousness_evolution,
    fusion_processing, quantum_aware_resource_scheduling_optimization, AdaptiveMemoryConsolidation,
    AdvancedConfig, AdvancedState, CoherenceStrategy, ConsciousnessComplexity, ConsciousnessState,
    EnhancedMetaLearningSystem, HierarchicalLearner, MetaLearningTracker,
    QuantumAwareResourceScheduler, QuantumCoherenceOptimizer, QuantumConsciousnessEvolution,
    ResourceSchedulingDecision, StrategyEvolution, TemporalMemoryFusion, WorkloadCharacteristics,
};

// Enhanced validation exports
pub use self::comprehensive_validation::{
    validated_advanced_processing, ComprehensiveValidator, PerformanceBenchmark,
    PerformanceSummary, ValidationConfig, ValidationError, ValidationReport,
};

// Utils exports
pub use self::utils::{safe_f64_to_float, safe_float_to_f64, safe_usize_to_float};
