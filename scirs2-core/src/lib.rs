#![recursion_limit = "512"]
// TODO: Address deprecated code usage and remove this allow
#![allow(deprecated)]
// TODO: Remove dead code or justify why it's kept
#![allow(dead_code)]
// Clippy allow attributes for non-critical warnings
#![allow(clippy::empty_line_after_outer_attribute)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::useless_format)]
#![allow(clippy::result_large_err)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::enumerate_and_ignore)]
#![allow(clippy::redundant_pattern_matching)]
#![allow(clippy::or_fun_call)]
#![allow(clippy::unnecessary_lazy_evaluations)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::new_without_default)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::get_first)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::needless_return)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::empty_line_after_outer_attribute_doc)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::empty_line_after_outer_attr)]
#![allow(clippy::unused_enumerate_index)]
#![allow(clippy::unwrap_or_default)]

//! # SciRS2 Core - Foundation for Scientific Computing in Rust
//!
//! **scirs2-core** is the foundational crate for the SciRS2 scientific computing ecosystem,
//! providing essential utilities, abstractions, and optimizations used by all SciRS2 modules.
//!
//! ## 🎯 Design Philosophy
//!
//! - **Zero-Cost Abstractions**: Performance without compromising safety
//! - **Layered Architecture**: Clear separation between interface and implementation
//! - **Policy Compliance**: Enforce [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md) - only scirs2-core uses external dependencies directly
//! - **Production Ready**: Enterprise-grade error handling, diagnostics, and stability guarantees
//!
//! ## 🚀 Key Features
//!
//! ### Performance Acceleration
//!
//! - **SIMD Operations**: CPU vector instructions (SSE, AVX, NEON) for array operations
//! - **Parallel Processing**: Multi-threaded execution with intelligent load balancing
//! - **GPU Acceleration**: Unified interface for CUDA, Metal, OpenCL, and WebGPU
//! - **Memory Efficiency**: Zero-copy views, memory-mapped arrays, adaptive chunking
//!
//! ### Core Utilities
//!
//! - **Error Handling**: ML-inspired diagnostics with recovery strategies
//! - **Validation**: Input checking with informative error messages
//! - **Caching**: Memoization and result caching for expensive computations
//! - **Profiling**: Performance monitoring and bottleneck detection
//!
//! ### Scientific Infrastructure
//!
//! - **Constants**: Physical and mathematical constants (via [`constants`] module)
//! - **Random Number Generation**: Consistent RNG interface across the ecosystem
//! - **Complex Numbers**: Type-safe complex arithmetic
//! - **Array Protocol**: Unified array interface for interoperability
//!
//! ## 📦 Module Overview
//!
//! ### Performance & Optimization
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`simd_ops`] | SIMD-accelerated operations with platform detection |
//! | [`parallel_ops`] | Parallel processing primitives |
//! | [`gpu`] | GPU acceleration abstractions |
//! | [`memory`] | Memory management (buffer pools, zero-copy views) |
//! | [`memory_efficient`] | Memory-mapped arrays and lazy evaluation |
//!
//! ### Error Handling & Diagnostics
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`error`] | Error types and traits |
//! | [`validation`] | Input validation utilities |
//! | [`logging`] | Structured logging for diagnostics |
//! | [`profiling`] | Performance profiling tools |
//!
//! ### Scientific Computing Basics
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`ndarray`] | Unified ndarray interface (re-exports with SciRS2 extensions) |
//! | [`numeric`] | Generic numerical operations |
//! | [`random`] | Random number generation |
//! | [`constants`] | Mathematical and physical constants |
//!
//! ### Infrastructure
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`config`] | Configuration management |
//! | [`cache`] | Result caching and memoization |
//! | [`io`] | I/O utilities |
//! | [`cloud`] | Cloud storage integration (S3, GCS, Azure) |
//!
//! ## 🚀 Quick Start
//!
//! ### Installation
//!
//! ```toml
//! [dependencies]
//! scirs2-core = { version = "0.1.0-rc.1", features = ["simd", "parallel"] }
//! ```
//!
//! ### SIMD Operations
//!
//! ```rust
//! use scirs2_core::simd_ops::SimdUnifiedOps;
//! use ndarray::array;
//!
//! // Automatic SIMD acceleration based on CPU capabilities
//! let a = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let b = array![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
//!
//! // SIMD-accelerated element-wise addition
//! let result = f64::simd_add(&a.view(), &b.view());
//! ```
//!
//! ### Parallel Processing
//!
//! ```rust
//! # #[cfg(feature = "parallel")]
//! # {
//! use scirs2_core::parallel_ops::*;
//!
//! // Parallel iteration over chunks
//! let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
//! par_chunks(&data, 3).for_each(|chunk| {
//!     // Process each chunk in parallel
//!     let sum: i32 = chunk.iter().sum();
//!     println!("Chunk sum: {}", sum);
//! });
//! # }
//! ```
//!
//! ### Input Validation
//!
//! ```rust
//! use scirs2_core::validation::*;
//! use scirs2_core::error::CoreResult;
//! use ndarray::Array2;
//!
//! fn process_matrix(data: &Array2<f64>, k: usize) -> CoreResult<()> {
//!     // Validate inputs
//!     check_positive(k, "k")?;
//!     checkarray_finite(data, "data")?;
//!     checkshape(data, &[2, 3], "data")?;
//!
//!     // Process data...
//!     Ok(())
//! }
//! # let data = Array2::<f64>::zeros((2, 3));
//! # let _ = process_matrix(&data, 5);
//! ```
//!
//! ### Constants
//!
//! ```rust
//! use scirs2_core::constants::{math, physical};
//!
//! // Mathematical constants
//! let pi = math::PI;
//! let e = math::E;
//!
//! // Physical constants
//! let c = physical::SPEED_OF_LIGHT;
//! let h = physical::PLANCK;
//! ```
//!
//! ### Random Number Generation
//!
//! ```rust
//! # #[cfg(feature = "random")]
//! # {
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use scirs2_core::random::*;
//! use rand::Rng;
//!
//! // Standard distributions
//! let mut rng = rand::rng();
//! let normal = Normal::new(0.0, 1.0)?;
//! let samples: Vec<f64> = (0..1000).map(|_| normal.sample(&mut rng)).collect();
//!
//! let uniform = Uniform::new(0.0, 1.0)?;
//! let samples: Vec<f64> = (0..1000).map(|_| uniform.sample(&mut rng)).collect();
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! ### Memory-Efficient Operations
//!
//! ```rust,no_run
//! # #[cfg(feature = "memory_efficient")]
//! # {
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use scirs2_core::memory_efficient::*;
//! use std::path::Path;
//! use ndarray::Array2;
//!
//! // Memory-mapped array for large datasets
//! let data = Array2::<f64>::zeros((1000, 1000));
//! let path = Path::new("/path/to/large_file.dat");
//! let mmap = create_mmap(&data, path, AccessMode::ReadWrite, 0)?;
//!
//! // Chunked operations for out-of-core processing
//! let result = chunk_wise_op(&data, |chunk| {
//!     chunk.mapv(|x| x * 2.0)
//! }, ChunkingStrategy::Fixed(10000))?;
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! ## 🏗️ Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │     SciRS2 Ecosystem Modules            │
//! │  (linalg, stats, neural, vision, etc.)  │
//! └─────────────────────────────────────────┘
//!              ▼
//! ┌─────────────────────────────────────────┐
//! │         scirs2-core (This Crate)        │
//! │  ┌────────────────────────────────────┐ │
//! │  │  High-Level Abstractions           │ │
//! │  │  - SIMD Operations                 │ │
//! │  │  - Parallel Processing             │ │
//! │  │  - GPU Acceleration                │ │
//! │  │  - Error Handling                  │ │
//! │  └────────────────────────────────────┘ │
//! │  ┌────────────────────────────────────┐ │
//! │  │  Core Utilities                    │ │
//! │  │  - Array Protocol                  │ │
//! │  │  - Validation                      │ │
//! │  │  - Memory Management               │ │
//! │  │  - Configuration                   │ │
//! │  └────────────────────────────────────┘ │
//! └─────────────────────────────────────────┘
//!              ▼
//! ┌─────────────────────────────────────────┐
//! │     External Dependencies               │
//! │  (ndarray, rayon, BLAS, etc.)           │
//! │  ⚠️  Only scirs2-core depends directly  │
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## 🎨 Feature Flags
//!
//! ### Performance Features
//!
//! - `simd` - SIMD acceleration (SSE, AVX, NEON)
//! - `parallel` - Multi-threaded execution via Rayon
//! - `gpu` - GPU acceleration (CUDA, Metal, OpenCL)
//!
//! ### Memory Management
//!
//! - `memory_management` - Advanced memory management (buffer pools, tracking)
//! - `memory_efficient` - Memory-mapped arrays and lazy evaluation
//! - `memory_metrics` - Memory usage tracking and profiling
//!
//! ### Scientific Computing
//!
//! - `array` - Scientific array types (MaskedArray, RecordArray)
//! - `random` - Random number generation
//! - `linalg` - Linear algebra with BLAS/LAPACK
//!
//! ### Development & Debugging
//!
//! - `validation` - Input validation (recommended for development)
//! - `logging` - Structured logging
//! - `profiling` - Performance profiling
//! - `testing` - Testing utilities
//!
//! ### Advanced Features
//!
//! - `cloud` - Cloud storage (S3, GCS, Azure)
//! - `jit` - Just-in-time compilation with LLVM
//! - `ml_pipeline` - ML pipeline integration
//!
//! ### Convenience
//!
//! - `default` - Commonly used features (parallel, validation)
//! - `all` - All features except backend-specific ones
//!
//! ## 🔒 SciRS2 Policy Compliance
//!
//! **Important**: scirs2-core is the **only** crate in the SciRS2 ecosystem that directly
//! depends on external crates like `ndarray`, `rand`, `rayon`, etc.
//!
//! All other SciRS2 crates **must** use abstractions provided by scirs2-core:
//!
//! ```rust,ignore
//! // ✅ CORRECT: Use scirs2-core abstractions
//! use scirs2_core::ndarray::Array2;
//! use scirs2_core::random::Normal;
//! use scirs2_core::parallel_ops::*;
//!
//! // ❌ WRONG: Don't import external deps directly in other crates
//! // use ndarray::Array2;       // NO!
//! // use rand_distr::Normal;    // NO!
//! // use rayon::prelude::*;     // NO!
//! ```
//!
//! This policy ensures:
//! - Consistent APIs across the ecosystem
//! - Centralized version management
//! - Easy addition of SciRS2-specific extensions
//! - Better compile times through reduced duplication
//!
//! ## 📊 Performance
//!
//! scirs2-core provides multiple optimization levels:
//!
//! | Feature | Speedup | Use Case |
//! |---------|---------|----------|
//! | SIMD | 2-8x | Array operations, numerical computations |
//! | Parallel | 2-16x | Large datasets, independent operations |
//! | GPU | 10-100x | Massive parallelism, deep learning |
//! | Memory-mapped | ∞ | Out-of-core processing, datasets larger than RAM |
//!
//! ### Platform Detection
//!
//! Automatic CPU feature detection for optimal SIMD usage:
//!
//! ```rust
//! use scirs2_core::simd_ops::PlatformCapabilities;
//!
//! let caps = PlatformCapabilities::detect();
//! println!("SIMD available: {}", caps.simd_available);
//! println!("AVX2 available: {}", caps.avx2_available);
//! println!("GPU available: {}", caps.gpu_available);
//! ```
//!
//! ## 🔗 Integration
//!
//! scirs2-core integrates seamlessly with the Rust ecosystem:
//!
//! - **ndarray**: Core array operations
//! - **num-traits**: Generic numeric operations
//! - **rayon**: Parallel processing
//! - **BLAS/LAPACK**: Optimized linear algebra
//!
//! ## 🔒 Version
//!
//! Current version: **0.1.0-rc.1** (Released October 03, 2025)
//!
//! ## 📚 Examples
//!
//! See the [examples directory](https://github.com/cool-japan/scirs/tree/master/scirs2-core/examples)
//! for more detailed usage examples.

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
pub mod chunking;
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
pub mod ndarray;
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
pub mod simd_aligned;
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
#[allow(ambiguous_glob_reexports)]
pub use crate::error::*;

// Re-export the array! macro for convenient array creation
// This addresses the common pain point where users expect array! to be available
// directly from scirs2_core instead of requiring import from scirs2_autograd
//
// # Example
//
// ```rust
// use scirs2_core::array;
//
// let matrix = array![[1, 2, 3], [4, 5, 6]];
// assert_eq!(matrix.shape(), &[2, 3]);
// ```
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
// Legacy re-export from ndarray_ext (kept for backward compatibility)
pub use crate::ndarray_ext::array as array_legacy;

// Complete ndarray functionality through the unified module
pub use crate::ndarray::{
    arr1,
    arr2,
    // Essential macros - now available at crate root
    array,
    s,
    // Common types for convenience
    Array,
    Array1,
    Array2,
    ArrayD,
    ArrayView,
    ArrayView1,
    ArrayView2,
    ArrayViewMut,
    Axis,
    Ix1,
    Ix2,
    IxDyn,
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
    AdaptiveChunkingParams, AdaptiveChunkingResult, ArithmeticOps, BroadcastOps, ChunkIter,
    ChunkedArray, ChunkingStrategy, DiskBackedArray, FusedOp, LazyArray, LazyOp, LazyOpKind,
    MemoryMappedArray, MemoryMappedChunkIter, MemoryMappedChunks, MemoryMappedSlice,
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
#[allow(ambiguous_glob_reexports)]
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

// Re-export complex number types for SCIRS2 POLICY compliance
pub use num_complex::{Complex, Complex32, Complex64};

// Re-export RNG types for SCIRS2 POLICY compliance
pub use crate::units::{
    convert, global_unit_registry, unit_value, Dimension, UnitDefinition, UnitRegistry, UnitSystem,
    UnitValue,
};
pub use crate::utils::*;
pub use crate::validation::production as validation_production;
pub use crate::validation::{
    check_finite, check_in_bounds, check_positive, checkarray_finite, checkshape,
};
pub use rand_chacha::{ChaCha12Rng, ChaCha20Rng, ChaCha8Rng};

// ================================
// Prelude Module
// ================================

/// Convenient re-exports of commonly used items
///
/// Import this module to get quick access to the most frequently used
/// types, traits, and functions in the SciRS2 ecosystem without needing
/// to remember specific import paths.
///
/// # Example
///
/// ```rust
/// use scirs2_core::prelude::*;
///
/// let data = array![[1.0, 2.0], [3.0, 4.0]];
/// ```
pub mod prelude;

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
