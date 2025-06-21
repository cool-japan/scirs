//! # SciRS2
//!
//! SciRS2 is a comprehensive scientific computing library for Rust,
//! aiming to provide functionality similar to SciPy in Python.
//!
//! ## Overview
//!
//! SciRS2 provides a collection of modules for various scientific computing tasks:
//!
//! * Linear algebra operations - [`linalg`](linalg)
//! * Statistical functions - [`stats`](stats)
//! * Optimization algorithms - [`optimize`](optimize)
//! * Numerical integration - [`integrate`](integrate)
//! * Fast Fourier Transform - [`fft`](fft)
//! * Signal processing - [`signal`](signal)
//! * Special functions - [`special`](special)
//! * Sparse matrix operations - [`sparse`](sparse)
//! * Spatial algorithms - [`spatial`](spatial)
//! * Clustering algorithms - [`cluster`](cluster)
//! * N-dimensional image processing - [`ndimage`](ndimage)
//! * Neural network building blocks - [`neural`](neural)
//! * Automatic differentiation - [`autograd`](autograd)
//! * Physical and mathematical constants - [`constants`](constants)
//! * Input/output utilities - [`io`](io)
//! * Text processing - [`text`](text)
//! * Computer vision - [`vision`](vision)
//! * Time series analysis - [`series`](series)
//! * Graph processing - [`graph`](graph)
//! * Data transformation - [`transform`](transform)
//! * ML evaluation metrics - [`metrics`](metrics)
//! * ML optimization algorithms - [`optim`](optim)
//! * Dataset utilities - [`datasets`](datasets)
//!
//! ## Examples
//!
//! ```rust,no_run
//! use scirs2::prelude::*;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Example using linear algebra module (when enabled)
//!     #[cfg(feature = "linalg")]
//!     {
//!         use ndarray::array;
//!         
//!         // Create a matrix
//!         let a = array![[1.0, 2.0], [3.0, 4.0]];
//!         
//!         // Calculate determinant
//!         if let Ok(det_val) = scirs2::linalg::det(&a.view(), None) {
//!             println!("Determinant: {}", det_val);
//!         }
//!         
//!         // Calculate matrix inverse
//!         if let Ok(inv_a) = scirs2::linalg::inv(&a.view(), None) {
//!             println!("Inverse matrix: {:?}", inv_a);
//!         }
//!     }
//!     
//!     // Example using stats module (when enabled)
//!     #[cfg(feature = "stats")]
//!     {
//!         use ndarray::array;
//!         
//!         // Create a data array
//!         let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
//!         
//!         // Calculate mean
//!         if let Ok(mean_val) = scirs2::stats::mean(&data.view()) {
//!             println!("Mean: {}", mean_val);
//!         }
//!         
//!         // Calculate standard deviation
//!         if let Ok(std_val) = scirs2::stats::std(&data.view(), 0) {
//!             println!("Standard deviation: {}", std_val);
//!         }
//!     }
//!     
//!     Ok(())
//! }
//! ```

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

#[cfg(feature = "optim")]
pub use scirs2_optim as optim;

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

    // Core numeric utilities
    pub use ndarray::{Array, Array1, Array2, ArrayD};

    // Re-export common type conversions
    pub use num_traits::{Float, One, Zero};

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
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
