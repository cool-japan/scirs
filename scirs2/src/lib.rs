#![allow(deprecated)]
//! # SciRS2 - Scientific Computing for Rust
//!
//! **SciRS2** is a comprehensive scientific computing and AI/ML infrastructure for Rust,
//! providing SciPy-compatible APIs with Rust's performance, safety, and concurrency features.
//!
//! ## 🎯 Key Features
//!
//! - **SciPy-Compatible APIs**: Familiar function signatures for easy migration from Python
//! - **High Performance**: Rust's zero-cost abstractions with SIMD, parallel, and GPU acceleration
//! - **Type Safety**: Compile-time guarantees preventing runtime errors
//! - **Modular Design**: Use only what you need via feature flags
//! - **Production Ready**: Comprehensive error handling, validation, and stability guarantees
//!
//! ## 📦 Module Overview
//!
//! ### Core Scientific Computing
//!
//! | Module | Description | SciPy Equivalent |
//! |--------|-------------|------------------|
//! | [`linalg`] | Linear algebra (decompositions, eigenvalues, solvers) | `scipy.linalg` |
//! | [`stats`] | Statistical functions and distributions | `scipy.stats` |
//! | [`optimize`] | Optimization algorithms (unconstrained, constrained) | `scipy.optimize` |
//! | [`integrate`] | Numerical integration and ODEs | `scipy.integrate` |
//! | [`interpolate`] | Interpolation methods | `scipy.interpolate` |
//! | [`fft`] | Fast Fourier Transform | `scipy.fft` |
//! | [`signal`] | Signal processing | `scipy.signal` |
//! | [`special`] | Special mathematical functions | `scipy.special` |
//! | [`sparse`] | Sparse matrix operations | `scipy.sparse` |
//! | [`spatial`] | Spatial algorithms (KD-trees, distance metrics) | `scipy.spatial` |
//! | `ndimage` | N-dimensional image processing | `scipy.ndimage` |
//!
//! ### Machine Learning & AI
//!
//! | Module | Description | Python Equivalent |
//! |--------|-------------|-------------------|
//! | `neural` | Neural network building blocks | PyTorch/TensorFlow |
//! | `autograd` | Automatic differentiation | PyTorch autograd |
//! | [`cluster`] | Clustering algorithms (K-means, DBSCAN, etc.) | scikit-learn.cluster |
//! | `metrics` | ML evaluation metrics | scikit-learn.metrics |
//! | `text` | Text processing and NLP | NLTK/spaCy basics |
//! | `vision` | Computer vision utilities | torchvision basics |
//!
//! ### Data & I/O
//!
//! | Module | Description |
//! |--------|-------------|
//! | `datasets` | Sample datasets for testing and learning |
//! | `io` | Input/output utilities (CSV, HDF5, Parquet) |
//! | `transform` | Data transformation pipelines |
//! | `series` | Time series analysis |
//! | `graph` | Graph processing algorithms |
//!
//! ### Utilities
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`constants`] | Physical and mathematical constants |
//! | `error` (module) | Error types and handling |
//! | [`validation`] | Input validation utilities |
//!
//! **Note**: ML optimization algorithms have been moved to the independent
//! [OptiRS](https://github.com/cool-japan/optirs) project from v0.1.0-beta.2.
//!
//! ## 🚀 Quick Start
//!
//! ### Installation
//!
//! Add SciRS2 to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! scirs2 = { version = "0.1.0-rc.1", features = ["linalg", "stats"] }
//! ```
//!
//! Or install all features:
//!
//! ```toml
//! [dependencies]
//! scirs2 = { version = "0.1.0-rc.1", features = ["full"] }
//! ```
//!
//! ### Linear Algebra Example
//!
//! ```rust,no_run
//! # use scirs2_core::ndarray::array;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Matrix operations
//! let a = array![[1.0, 2.0], [3.0, 4.0]];
//!
//! // Determinant
//! let det = scirs2::linalg::det(&a.view(), None)?;
//! println!("Determinant: {}", det);
//!
//! // Matrix inverse
//! let inv = scirs2::linalg::inv(&a.view(), None)?;
//! println!("Inverse:\n{:?}", inv);
//!
//! // SVD decomposition
//! let (u, s, vt) = scirs2::linalg::svd(&a.view(), true, None)?;
//! println!("Singular values: {:?}", s);
//! # Ok(())
//! # }
//! ```
//!
//! ### Statistics Example
//!
//! ```rust,no_run
//! # use scirs2_core::ndarray::array;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
//!
//! // Descriptive statistics
//! let mean = scirs2::stats::mean(&data.view())?;
//! let std = scirs2::stats::std(&data.view(), 0, None)?;
//! let median = scirs2::stats::median(&data.view())?;
//!
//! println!("Mean: {}, Std: {}, Median: {}", mean, std, median);
//!
//! // Statistical distributions
//! use scirs2::stats::distributions::Normal;
//! let normal = Normal::new(0.0, 1.0)?;
//! let samples = normal.rvs(1000)?;
//! println!("Generated {} samples", samples.len());
//! # Ok(())
//! # }
//! ```
//!
//! ### Neural Network Example
//!
//! ```rust,ignore
//! use scirs2::neural::{Sequential, Dense, ReLU};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Build a simple neural network
//!     let mut model = Sequential::new();
//!     model.add(Dense::new(784, 128)?);
//!     model.add(ReLU::new());
//!     model.add(Dense::new(128, 10)?);
//!
//!     // Forward pass
//!     // let output = model.forward(&input)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## 🎨 Feature Flags
//!
//! Control which modules to include:
//!
//! ### Core Modules
//!
//! - `linalg` - Linear algebra operations
//! - `stats` - Statistical functions
//! - `optimize` - Optimization algorithms
//! - `integrate` - Numerical integration
//! - `interpolate` - Interpolation methods
//! - `fft` - Fast Fourier Transform
//! - `special` - Special functions
//! - `signal` - Signal processing
//! - `sparse` - Sparse matrices
//! - `spatial` - Spatial algorithms
//!
//! ### ML/AI Modules
//!
//! - `neural` - Neural networks
//! - `autograd` - Automatic differentiation
//! - `cluster` - Clustering algorithms
//! - `metrics` - ML metrics
//! - `text` - Text processing
//! - `vision` - Computer vision
//!
//! ### Data Modules
//!
//! - `datasets` - Sample datasets
//! - `io` - I/O utilities
//! - `transform` - Data transformation
//! - `series` - Time series
//! - `graph` - Graph processing
//! - `ndimage` - Image processing
//!
//! ### Convenience Features
//!
//! - `full` - Enable all features
//! - `default` - Enable commonly used features
//!
//! ## 🏗️ Architecture
//!
//! SciRS2 follows a layered architecture:
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │         User Applications               │
//! └─────────────────────────────────────────┘
//!              ▼
//! ┌─────────────────────────────────────────┐
//! │  scirs2 (Unified Interface)             │
//! │  - Feature-gated re-exports             │
//! │  - Unified prelude                      │
//! └─────────────────────────────────────────┘
//!              ▼
//! ┌─────────────────────────────────────────┐
//! │  Domain Modules                         │
//! │  linalg, stats, neural, etc.            │
//! └─────────────────────────────────────────┘
//!              ▼
//! ┌─────────────────────────────────────────┐
//! │  scirs2-core (Foundation)               │
//! │  - Error handling                       │
//! │  - SIMD/Parallel/GPU abstractions       │
//! │  - Memory management                    │
//! │  - Validation utilities                 │
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## 📊 Performance
//!
//! SciRS2 leverages multiple optimization strategies:
//!
//! - **SIMD**: Automatic vectorization for array operations
//! - **Parallel**: Multi-threaded execution via Rayon
//! - **GPU**: CUDA/Metal/OpenCL support for accelerated computing
//! - **BLAS/LAPACK**: Native library bindings for optimal performance
//!
//! Benchmark comparisons with SciPy show 2-10x speedups for many operations
//! (see `benchmarks/` directory for details).
//!
//! ## 🔒 Stability & Versioning
//!
//! SciRS2 follows semantic versioning and provides:
//!
//! - **API Stability**: Beta releases maintain API compatibility
//! - **Deprecation Policy**: 2-release deprecation cycle
//! - **Production Features**: Enterprise-grade error handling and diagnostics
//!
//! Current version: **0.1.0-rc.1** (Released October 03, 2025)
//!
//! ## 🤝 Ecosystem Integration
//!
//! SciRS2 integrates with the Rust scientific ecosystem:
//!
//! - **ndarray**: Core array operations
//! - **num-traits**: Generic numeric operations
//! - **OptiRS**: Advanced optimization (formerly scirs2-optim)
//! - **nalgebra**: Alternative linear algebra (interoperable)
//!
//! ## 📚 Additional Resources
//!
//! - [GitHub Repository](https://github.com/cool-japan/scirs)
//! - [API Documentation](https://docs.rs/scirs2)
//! - [Examples](https://github.com/cool-japan/scirs/tree/master/examples)
//! - [Migration Guide from SciPy](https://github.com/cool-japan/scirs/blob/master/docs/migration.md)
//!
//! ## 📜 License
//!
//! Licensed under either of Apache License, Version 2.0 or MIT license at your option.

// Re-export from scirs2-core
#[cfg(feature = "cache")]
pub use scirs2_core::cache;
#[cfg(feature = "logging")]
pub use scirs2_core::logging;
#[cfg(feature = "memory_management")]
pub use scirs2_core::memory;
#[cfg(feature = "profiling")]
pub use scirs2_core::profiling;
pub use scirs2_core::{constants, error, utils, validation};

// Optional modules (enabled via features)
#[cfg(feature = "linalg")]
pub use scirs2_linalg as linalg;

#[cfg(feature = "stats")]
pub use scirs2_stats as stats;

#[cfg(feature = "integrate")]
pub use scirs2_integrate as integrate;

#[cfg(feature = "interpolate")]
pub use scirs2_interpolate as interpolate;

#[cfg(feature = "optimize")]
pub use scirs2_optimize as optimize;

#[cfg(feature = "fft")]
pub use scirs2_fft as fft;

#[cfg(feature = "special")]
pub use scirs2_special as special;

#[cfg(feature = "signal")]
pub use scirs2_signal as signal;

#[cfg(feature = "sparse")]
pub use scirs2_sparse as sparse;

#[cfg(feature = "spatial")]
pub use scirs2_spatial as spatial;

#[cfg(feature = "cluster")]
pub use scirs2_cluster as cluster;

#[cfg(feature = "ndimage")]
pub use scirs2_ndimage as ndimage;

#[cfg(feature = "io")]
pub use scirs2_io as io;

#[cfg(feature = "datasets")]
pub use scirs2_datasets as datasets;

#[cfg(feature = "neural")]
pub use scirs2_neural as neural;

// optim module moved to independent OptiRS project from v0.1.0-beta.2

#[cfg(feature = "graph")]
pub use scirs2_graph as graph;

#[cfg(feature = "transform")]
pub use scirs2_transform as transform;

#[cfg(feature = "metrics")]
pub use scirs2_metrics as metrics;

#[cfg(feature = "text")]
pub use scirs2_text as text;

#[cfg(feature = "vision")]
pub use scirs2_vision as vision;

#[cfg(feature = "series")]
pub use scirs2_series as series;

#[cfg(feature = "autograd")]
pub use scirs2_autograd as autograd;

/// Version information
pub mod version {
    /// Current SciRS2 version
    pub const VERSION: &str = env!("CARGO_PKG_VERSION");
}

/// Re-export of common utilities and types
pub mod prelude {
    pub use scirs2_core::validation;
    // Use the Error type directly from thiserror
    pub use thiserror::Error;

    // Core numeric utilities (SCIRS2 POLICY: use scirs2_core re-exports)
    pub use scirs2_core::ndarray::{Array, Array1, Array2, ArrayD};

    // Re-export common type conversions (SCIRS2 POLICY: use scirs2_core re-exports)
    pub use scirs2_core::numeric::{Float, One, Zero};

    // Various modules with feature gates
    #[cfg(feature = "linalg")]
    pub use crate::linalg;

    #[cfg(feature = "stats")]
    pub use crate::stats;

    #[cfg(feature = "special")]
    pub use crate::special;

    #[cfg(feature = "optimize")]
    pub use crate::optimize;

    #[cfg(feature = "neural")]
    pub use crate::neural;
}

// Public API
/// SciRS2 version information
#[allow(dead_code)]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
