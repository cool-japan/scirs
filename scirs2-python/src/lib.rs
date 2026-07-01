//! SciRS2 Python Bindings
//!
//! This crate provides Python bindings for the SciRS2 scientific computing library,
//! offering a high-performance SciPy alternative with a familiar Python API.
//!
//! # Modules
//!
//! - `cluster`: Clustering algorithms (K-Means, DBSCAN, Hierarchical, etc.)
//! - `series`: Time series analysis and ARIMA models
//! - `linalg`: Linear algebra operations
//! - `stats`: Statistical distributions and tests
//! - `fft`: Fast Fourier Transforms
//! - `optimize`: Optimization algorithms
//! - `special`: Special mathematical functions
//! - `integrate`: Numerical integration
//! - `interpolate`: Interpolation functions
//! - `signal`: Signal processing
//! - `spatial`: Spatial algorithms
//! - `sparse`: Sparse matrix operations
//! - `ndimage`: N-dimensional image processing
//! - `graph`: Graph algorithms
//! - `metrics`: ML evaluation metrics
//! - `io`: File I/O operations
//! - `datasets`: Dataset loading and generation
//! - `transform`: Data transformation and preprocessing
//! - `text`: Text processing and NLP
//! - `vision`: Computer vision algorithms
//! - `autograd`: Automatic differentiation (placeholder)
//! - `neural`: Neural network layers (placeholder)
//!
//! # Architecture
//!
//! This crate uses:
//! - **scirs2-numpy** - SciRS2 fork of rust-numpy with native ndarray 0.17 support
//! - **PyO3** for Python-Rust interop
//! - **NumPy** for seamless array compatibility
//!
//! Using scirs2-numpy provides direct ndarray 0.17 compatibility with all
//! internal SciRS2 crates, eliminating version mismatches and enabling
//! zero-copy operations where possible.

use pyo3::prelude::*;

// Core modules (always available)
pub mod async_ops;
pub mod dlpack;
pub mod error;
pub mod gpu_ops;
pub mod pandas_compat;
pub mod parallel;

// Submodules
#[cfg(feature = "cluster")]
pub mod cluster;

#[cfg(feature = "series")]
pub mod series;

#[cfg(feature = "linalg")]
pub mod linalg;

#[cfg(feature = "stats")]
pub mod stats;

#[cfg(feature = "fft")]
pub mod fft;

#[cfg(feature = "optimize")]
pub mod optimize;

#[cfg(feature = "special")]
pub mod special;

#[cfg(feature = "integrate")]
pub mod integrate;

#[cfg(feature = "interpolate")]
pub mod interpolate;

#[cfg(feature = "signal")]
pub mod signal;

#[cfg(feature = "spatial")]
pub mod spatial;

#[cfg(feature = "sparse")]
pub mod sparse;

#[cfg(feature = "ndimage")]
pub mod ndimage;

#[cfg(feature = "graph")]
#[allow(unused_must_use)] // For graph.add_edge() Result
pub mod graph;

#[cfg(feature = "metrics")]
pub mod metrics;

#[cfg(feature = "io")]
pub mod io;

#[cfg(feature = "datasets")]
pub mod datasets;

#[cfg(feature = "transform")]
pub mod transform;

#[cfg(feature = "text")]
pub mod text;

#[cfg(feature = "vision")]
pub mod vision;

#[cfg(feature = "autograd")]
pub mod autograd;

#[cfg(feature = "neural")]
pub mod neural;

// Symbolic mathematics — always available (additive, no feature gate).
pub mod symbolic;

/// SciRS2 Python module
///
/// A comprehensive scientific computing library in Rust with Python bindings.
/// Provides SciPy-compatible APIs with Rust-level performance.
#[pymodule]
fn scirs2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Package metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__author__",
        "COOLJAPAN OU (Team KitaSan) <contact@cooljapan.tech>",
    )?;

    // Register core modules (async, pandas support, DLPack interop, parallel batch, GPU ops)
    async_ops::register_async_module(m)?;
    pandas_compat::register_pandas_module(m)?;
    dlpack::register_dlpack_module(m)?;
    parallel::register_parallel_module(m)?;
    gpu_ops::register_gpu_module(m)?;

    // Register classes and functions directly in main module
    #[cfg(feature = "cluster")]
    cluster::register_module(m)?;

    #[cfg(feature = "series")]
    series::register_module(m)?;

    #[cfg(feature = "linalg")]
    linalg::register_module(m)?;

    #[cfg(feature = "stats")]
    stats::register_module(m)?;

    #[cfg(feature = "fft")]
    fft::register_module(m)?;

    #[cfg(feature = "optimize")]
    optimize::register_module(m)?;

    #[cfg(feature = "special")]
    special::register_module(m)?;

    #[cfg(feature = "integrate")]
    integrate::register_module(m)?;

    #[cfg(feature = "interpolate")]
    interpolate::register_module(m)?;

    #[cfg(feature = "signal")]
    signal::register_module(m)?;

    #[cfg(feature = "spatial")]
    spatial::register_module(m)?;

    #[cfg(feature = "sparse")]
    sparse::register_module(m)?;

    #[cfg(feature = "ndimage")]
    ndimage::register_module(m)?;

    #[cfg(feature = "graph")]
    graph::register_module(m)?;

    #[cfg(feature = "metrics")]
    metrics::register_module(m)?;

    #[cfg(feature = "io")]
    io::register_module(m)?;

    #[cfg(feature = "datasets")]
    datasets::register_module(m)?;

    #[cfg(feature = "transform")]
    transform::register_module(m)?;

    #[cfg(feature = "text")]
    text::register_module(m)?;

    #[cfg(feature = "vision")]
    vision::register_module(m)?;

    #[cfg(feature = "autograd")]
    autograd::register_module(m)?;

    #[cfg(feature = "neural")]
    neural::register_module(m)?;

    // Symbolic mathematics sub-namespace (always registered).
    symbolic::register_module(m)?;

    Ok(())
}
