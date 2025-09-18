#![recursion_limit = "512"]
// TODO: Address deprecated code usage and remove this allow
#![allow(deprecated)]
// TODO: Remove dead code or justify why it's kept
#![allow(dead_code)]

//! # ``SciRS2`` Core (Beta 1)
//!
//! Core utilities and common functionality for the ``SciRS2`` library.
//!
//! This crate provides shared utilities, error types, and common traits
//! used across the ``SciRS2`` ecosystem of crates.
//!
//! ## Beta 1 Features
//!
//! - **Stable APIs**: Core functionality with API stability guarantees for production use
//! - **Advanced Error Diagnostics**: ML-inspired error pattern recognition and domain-specific recovery strategies
//! - **Performance Optimizations**: Enhanced SIMD operations, adaptive chunking, and intelligent load balancing
//! - **GPU Acceleration**: CUDA, Metal MPS, and other backend support for accelerated computing
//! - **Memory Management**: Efficient memory-mapped arrays and adaptive chunking for large datasets
//!
//! ## Overview
//!
//! * Common error types and traits
//! * High-performance numerical operations
//!   * SIMD-accelerated computations
//!   * Parallel processing for multi-core systems
//!   * Memory-efficient algorithms
//!   * GPU acceleration abstractions
//! * Caching and memoization for optimized performance
//! * Type definitions and conversions
//! * Physical and mathematical constants
//! * Configuration system
//! * Input/output utilities
//! * Validation utilities
//! * Numeric traits and conversions
//! * Memory management utilities
//! * Logging and diagnostics
//! * Profiling tools
//! * Random number generation
//!
//! ## Performance Optimizations
//!
//! The library provides several performance optimization features:
//!
//! * **SIMD Operations**: Uses CPU vector instructions for faster array operations
//! * **Parallel Processing**: Leverages multi-core systems for improved performance
//! * **GPU Acceleration**: Provides abstractions for GPU computation (CUDA, WebGPU, Metal)
//! * **Memory-Efficient Algorithms**: Optimizes memory usage for large-scale computations
//! * **Caching and Memoization**: Avoids redundant computations
//! * **Profiling and Instrumentation**: Identifies performance bottlenecks
//! * **Memory Management**: Efficient memory utilization and pooling
//!
//! ## Additional Utilities
//!
//! * **Logging**: Structured logging for scientific applications
//! * **Random Number Generation**: Consistent interface for random sampling
//! * **Type Conversions**: Safe numeric and complex number conversions
//!
//! ## Feature Flags
//!
//! These features can be controlled via feature flags:
//!
//! * `simd`: Enable SIMD acceleration
//! * `parallel`: Enable parallel processing
//! * `cache`: Enable caching and memoization functionality
//! * `validation`: Enable validation utilities
//! * `logging`: Enable structured logging and diagnostics
//! * `gpu`: Enable GPU acceleration abstractions
//! * `memory_management`: Enable advanced memory management
//! * `memory_efficient`: Enable memory-efficient array operations and views
//! * `array`: Enable scientific array types (``MaskedArray``, ``RecordArray``)
//! * `profiling`: Enable performance profiling tools
//! * `random`: Enable random number generation utilities
//! * `types`: Enable type conversion utilities
//! * `linalg`: Enable linear algebra with BLAS/LAPACK bindings
//! * `cloud`: Enable cloud storage integration (S3, GCS, Azure)
//! * `jit`: Enable just-in-time compilation with LLVM
//! * `ml_pipeline`: Enable ML pipeline integration and real-time processing
//! * `all`: Enable all features except backend-specific ones

// Re-export modules
pub mod api_freeze;
pub mod apiversioning;
#[cfg(feature = "array")]
pub mod array;
pub mod array_protocol;
#[cfg(feature = "types")]
pub mod batch_conversions;
#[cfg(feature = "cache")]
pub mod cache;
#[cfg(feature = "cloud")]
pub mod cloud;
pub mod config;
pub mod constants;
pub mod distributed;
pub mod ecosystem;
pub mod error;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub mod gpu_registry;
pub mod io;
#[cfg(feature = "jit")]
pub mod jit;
#[cfg(feature = "logging")]
pub mod logging;
#[cfg(feature = "memory_management")]
pub mod memory;
#[cfg(feature = "memory_efficient")]
pub mod memory_efficient;
pub mod metrics;
#[cfg(feature = "ml_pipeline")]
pub mod ml_pipeline;
pub mod ndarray_ext;
pub mod numeric;
#[cfg(feature = "parallel")]
pub mod parallel;
#[cfg(feature = "parallel")]
pub mod parallel_ops;
pub mod performance;
pub mod performance_optimization;
#[cfg(feature = "profiling")]
pub mod profiling;
#[cfg(feature = "random")]
pub mod random;
pub mod resource;
#[cfg(feature = "simd")]
pub mod simd;
pub mod simd_ops;
#[cfg(feature = "testing")]
pub mod testing;
// Universal Functions (ufuncs) module
pub mod error_templates;
pub mod safe_ops;
#[cfg(feature = "types")]
pub mod types;
#[cfg(feature = "ufuncs")]
pub mod ufuncs;
pub mod units;
pub mod utils;
pub mod validation;

// Production-level features for enterprise deployments
pub mod observability;
pub mod stability;
pub mod versioning;

// Advanced optimization and AI features
pub mod neural_architecture_search;
pub mod quantum_optimization;

// Advanced Mode Ecosystem Integration
pub mod advanced_ecosystem_integration;

// Advanced JIT Compilation Framework
pub mod advanced_jit_compilation;

// Advanced Distributed Computing Framework
pub mod advanced_distributed_computing;

// Advanced Cloud Storage Framework
// pub mod distributed_storage; // Module not implemented yet

// Advanced Tensor Cores and Automatic Kernel Tuning Framework
pub mod advanced_tensor_cores;

// Tensor cores optimization modules
#[cfg(feature = "gpu")]
pub mod tensor_cores;

// Benchmarking module
#[cfg(feature = "benchmarking")]
pub mod benchmarking;

// Re-exports
#[cfg(feature = "cache")]
pub use crate::cache::*;
#[cfg(feature = "cloud")]
pub use crate::cloud::{
    CloudConfig, CloudCredentials, CloudError, CloudObjectMetadata, CloudProvider,
    CloudStorageClient, EncryptionConfig, EncryptionMethod, HttpMethod, ListResult,
    TransferOptions,
};
pub use crate::config::production as config_production;
pub use crate::config::{
    get_config, get_config_value, set_config_value, set_global_config, Config, ConfigValue,
};
pub use crate::constants::{math, physical, prefixes};
pub use crate::error::*;
#[cfg(feature = "gpu")]
pub use crate::gpu::*;
pub use crate::io::*;
#[cfg(feature = "jit")]
pub use crate::jit::DataType as JitDataType;
#[cfg(feature = "jit")]
pub use crate::jit::{
    CompiledKernel, ExecutionProfile, JitBackend, JitCompiler, JitConfig, JitError, KernelLanguage,
    KernelSource, OptimizationLevel, TargetArchitecture,
};
#[cfg(feature = "logging")]
pub use crate::logging::*;
#[cfg(feature = "memory_management")]
pub use crate::memory::{
    format_memory_report, generate_memory_report, global_buffer_pool, track_allocation,
    track_deallocation, track_resize, BufferPool, ChunkProcessor, ChunkProcessor2D,
    GlobalBufferPool, ZeroCopyView,
};

#[cfg(feature = "leak_detection")]
pub use crate::memory::{
    LeakCheckGuard, LeakDetectionConfig, LeakDetector, LeakReport, LeakType, MemoryCheckpoint,
    MemoryLeak, ProfilerTool,
};

#[cfg(feature = "memory_efficient")]
pub use crate::memory_efficient::{
    chunk_wise_binary_op, chunk_wise_op, chunk_wise_reduce, create_disk_array, create_mmap,
    create_temp_mmap, diagonal_view, evaluate, load_chunks, open_mmap, register_fusion,
    transpose_view, view_as, view_mut_as, AccessMode, AdaptiveChunking, AdaptiveChunkingBuilder,
    AdaptiveChunkingParams, AdaptiveChunkingResult, ArithmeticOps, ArrayView, BroadcastOps,
    ChunkIter, ChunkedArray, ChunkingStrategy, DiskBackedArray, FusedOp, LazyArray, LazyOp,
    LazyOpKind, MemoryMappedArray, MemoryMappedChunkIter, MemoryMappedChunks, MemoryMappedSlice,
    MemoryMappedSlicing, OpFusion, OutOfCoreArray, ViewMut, ZeroCopyOps,
};

// Compression-related types are only available with the memory_compression feature
#[cfg(feature = "memory_compression")]
pub use crate::memory_efficient::{
    CompressedMemMapBuilder, CompressedMemMappedArray, CompressionAlgorithm,
};

// Re-export the parallel memory-mapped array capabilities
#[cfg(all(feature = "memory_efficient", feature = "parallel"))]
pub use crate::memory_efficient::MemoryMappedChunksParallel;

#[cfg(feature = "array")]
pub use crate::array::{
    is_masked, mask_array, masked_equal, masked_greater, masked_inside, masked_invalid,
    masked_less, masked_outside, masked_where, record_array_from_typed_arrays,
    record_array_fromrecords, ArrayError, FieldValue, MaskedArray, Record, RecordArray, NOMASK,
};

#[cfg(feature = "memory_metrics")]
pub use crate::memory::metrics::{
    clear_snapshots,
    compare_snapshots,
    // Utility functions
    format_bytes,
    format_duration,
    take_snapshot,
    MemoryEvent,
    MemoryEventType,
    // Core metrics types
    MemoryMetricsCollector,
    MemoryMetricsConfig,
    // Memory snapshots and leak detection
    MemorySnapshot,
    SnapshotDiff,
    // Tracked memory components
    TrackedBufferPool,
    TrackedChunkProcessor,
    TrackedChunkProcessor2D,
};

#[cfg(feature = "types")]
pub use crate::batch_conversions::{
    utils as batch_utils, BatchConversionConfig, BatchConversionResult, BatchConverter,
    ElementConversionError,
};
#[cfg(all(feature = "memory_metrics", feature = "gpu"))]
pub use crate::memory::metrics::{setup_gpu_memory_tracking, TrackedGpuBuffer, TrackedGpuContext};
pub use crate::metrics::{
    global_healthmonitor, global_metrics_registry, Counter, Gauge, HealthCheck, HealthMonitor,
    HealthStatus, Histogram, MetricPoint, MetricType, MetricValue, Timer,
};
#[cfg(feature = "ml_pipeline")]
pub use crate::ml_pipeline::DataType as MLDataType;
#[cfg(feature = "ml_pipeline")]
pub use crate::ml_pipeline::{
    DataBatch, DataSample, FeatureConstraint, FeatureSchema, FeatureTransformer, FeatureValue,
    MLPipeline, MLPipelineError, ModelPredictor, ModelType, PipelineConfig, PipelineMetrics,
    PipelineNode, TransformType,
};
pub use crate::numeric::*;
#[cfg(feature = "parallel")]
pub use crate::parallel::*;
#[cfg(feature = "parallel")]
pub use crate::parallel_ops::{
    is_parallel_enabled, num_threads, par_chunks, par_chunks_mut, par_join, par_scope,
};
// Re-export all parallel traits and types
#[cfg(feature = "parallel")]
pub use crate::parallel_ops::*;
#[cfg(feature = "profiling")]
pub use crate::profiling::{profiling_memory_tracker, Profiler};
#[cfg(feature = "random")]
pub use crate::random::*;
pub use crate::resource::{
    get_available_memory, get_performance_tier, get_recommended_chunk_size,
    get_recommended_thread_count, get_system_resources, get_total_memory, is_gpu_available,
    is_simd_supported, DiscoveryConfig, PerformanceTier, ResourceDiscovery, SystemResources,
};
#[cfg(feature = "simd")]
pub use crate::simd::*;
#[cfg(feature = "testing")]
pub use crate::testing::{TestConfig, TestResult, TestRunner, TestSuite};
#[cfg(feature = "types")]
pub use crate::types::{convert, ComplexConversionError, ComplexExt, ComplexOps};
pub use crate::units::{
    convert, global_unit_registry, unit_value, Dimension, UnitDefinition, UnitRegistry, UnitSystem,
    UnitValue,
};
pub use crate::utils::*;
pub use crate::validation::production as validation_production;
pub use crate::validation::{
    check_finite, check_in_bounds, check_positive, checkarray_finite, checkshape,
};

#[cfg(feature = "data_validation")]
pub use crate::validation::data::DataType as ValidationDataType;
#[cfg(feature = "data_validation")]
pub use crate::validation::data::{
    Constraint, FieldDefinition, ValidationConfig, ValidationError, ValidationResult,
    ValidationRule, ValidationSchema, Validator,
};

// Production-level feature re-exports
pub use crate::observability::{audit, tracing};
pub use crate::stability::{
    global_stability_manager, ApiContract, BreakingChange, BreakingChangeType, ConcurrencyContract,
    MemoryContract, NumericalContract, PerformanceContract, StabilityGuaranteeManager,
    StabilityLevel, UsageContext,
};
pub use crate::versioning::{
    compatibility, deprecation, migration, negotiation, semantic, ApiVersion, CompatibilityLevel,
    SupportStatus, Version, VersionManager,
};

// Advanced optimization and AI feature re-exports
pub use crate::neural_architecture_search::{
    ActivationType, Architecture, ArchitecturePerformance, ConnectionType, HardwareConstraints,
    LayerType, NASStrategy, NeuralArchitectureSearch, OptimizationObjectives, OptimizerType,
    SearchResults, SearchSpace,
};
pub use crate::quantum_optimization::{
    OptimizationResult, QuantumOptimizer, QuantumParameters, QuantumState, QuantumStrategy,
};

// Advanced JIT Compilation re-exports
// pub use crate::advanced_jit_compilation::{
//     AdaptiveCodeGenerator, CompilationStatistics, JitAnalytics, JitCompilerConfig, JitProfiler,
//     KernelCache, KernelMetadata, KernelPerformance, LlvmCompilationEngine, OptimizationResults,
//     PerformanceImprovement, RuntimeOptimizer, advancedJitCompiler,
// }; // Missing module

// Advanced Cloud Storage re-exports
// pub use crate::distributed_storage::{
//     AdaptiveStreamingEngine, CloudPerformanceAnalytics, CloudProviderConfig, CloudProviderId,
//     CloudProviderType, CloudSecurityManager, CloudStorageMonitoring, CloudStorageProvider,
//     DataOptimizationEngine, DownloadRequest, DownloadResponse, IntelligentCacheSystem,
//     ParallelTransferManager, StreamRequest, advancedCloudConfig,
//     advancedCloudStorageCoordinator, UploadRequest, UploadResponse,
// };

// Benchmarking re-exports
#[cfg(feature = "benchmarking")]
pub use crate::benchmarking::{
    BenchmarkConfig, BenchmarkMeasurement, BenchmarkResult, BenchmarkRunner, BenchmarkStatistics,
    BenchmarkSuite,
};

/// ``SciRS2`` core version information
pub const fn _version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Initialize the library (called automatically)
#[doc(hidden)]
#[allow(dead_code)]
pub fn __init() {
    use std::sync::Once;
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        // Initialize API freeze registry
        crate::api_freeze::initialize_api_freeze();
    });
}

// Ensure initialization happens
#[doc(hidden)]
#[used]
#[cfg_attr(target_os = "linux", link_section = ".init_array")]
#[cfg_attr(target_os = "macos", link_section = "__DATA,__mod_init_func")]
#[cfg_attr(target_os = "windows", link_section = ".CRT$XCU")]
static INIT: extern "C" fn() = {
    extern "C" fn __init_wrapper() {
        __init();
    }
    __init_wrapper
};
