//! Out-of-core Kriging (simple kriging with Gaussian covariance).
//!
//! [`OutOfCoreKriging`] is a thin wrapper around [`OutOfCoreRbf`] configured
//! with a Gaussian kernel and a nugget term.  This corresponds to **simple
//! kriging** with a squared-exponential (Gaussian) covariance function:
//!
//! ```text
//! C(x, x') = exp(−ε ‖x − x'‖²) + δ(x, x') · nugget
//! ```
//!
//! where `nugget` absorbs measurement noise (analogous to Tikhonov
//! regularization in RBF nomenclature).
//!
//! # Out-of-core behaviour
//! All disk-backed coefficient storage is handled by the inner
//! [`OutOfCoreRbf`].  No additional files are written.
//!
//! # Example
//! ```rust
//! use std::env::temp_dir;
//! use scirs2_core::ndarray::{Array1, Array2};
//! use scirs2_interpolate::outofcore::{OutOfCoreKriging, OutOfCoreKrigingConfig};
//!
//! let n = 40usize;
//! let centers = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);
//! let values  = centers.column(0).mapv(|x| x * x);
//! let query   = Array2::from_shape_fn((5, 1), |(i, _)| i as f64 / 5.0);
//!
//! let scratch = temp_dir().join("doctest_ooc_kriging");
//! std::fs::create_dir_all(&scratch).ok();
//!
//! let mut ok = OutOfCoreKriging::new(OutOfCoreKrigingConfig {
//!     scratch_dir: scratch.clone(),
//!     chunk_size: 20,
//!     ..Default::default()
//! });
//! ok.fit(&centers, &values).expect("fit");
//! let preds = ok.predict(&query).expect("predict");
//! assert_eq!(preds.len(), 5);
//! ok.cleanup().ok();
//! std::fs::remove_dir_all(&scratch).ok();
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use std::path::PathBuf;

use super::rbf::{OocRbfKernel, OutOfCoreRbf, OutOfCoreRbfConfig};
use crate::error::InterpolateError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`OutOfCoreKriging`].
#[derive(Debug, Clone)]
pub struct OutOfCoreKrigingConfig {
    /// Number of rows per prediction chunk (forwarded to the inner RBF).
    pub chunk_size: usize,
    /// Soft cap on in-memory data (informational).
    pub cache_size_mb: f64,
    /// Directory for disk-backed coefficient storage.
    pub scratch_dir: PathBuf,
    /// Gaussian length-scale parameter `ε` in `exp(−ε ‖x−x'‖²)`.
    ///
    /// Larger values → narrower covariance → more local interpolation.
    pub length_scale: f64,
    /// Nugget variance (diagonal regularisation, analogous to noise variance).
    ///
    /// Keeps the covariance matrix numerically positive-definite.
    pub nugget: f64,
}

impl Default for OutOfCoreKrigingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            cache_size_mb: 100.0,
            scratch_dir: std::env::temp_dir(),
            length_scale: 1.0,
            nugget: 1e-6,
        }
    }
}

// ---------------------------------------------------------------------------
// OutOfCoreKriging
// ---------------------------------------------------------------------------

/// Out-of-core simple kriging interpolator backed by a Gaussian covariance.
///
/// Internally delegates to [`OutOfCoreRbf`] with:
/// - `kernel = OocRbfKernel::Gaussian`
/// - `epsilon = length_scale`
/// - `regularization = nugget`
pub struct OutOfCoreKriging {
    inner: OutOfCoreRbf,
    config: OutOfCoreKrigingConfig,
}

impl OutOfCoreKriging {
    /// Construct an unfitted kriging interpolator.
    pub fn new(config: OutOfCoreKrigingConfig) -> Self {
        let rbf_cfg = OutOfCoreRbfConfig {
            chunk_size: config.chunk_size,
            cache_size_mb: config.cache_size_mb,
            scratch_dir: config.scratch_dir.clone(),
            kernel: OocRbfKernel::Gaussian,
            epsilon: config.length_scale,
            regularization: config.nugget,
        };
        Self {
            inner: OutOfCoreRbf::new(rbf_cfg),
            config,
        }
    }

    /// Fit the kriging model.
    ///
    /// # Errors
    /// See [`OutOfCoreRbf::fit`].
    pub fn fit(
        &mut self,
        centers: &Array2<f64>,
        values: &Array1<f64>,
    ) -> Result<(), InterpolateError> {
        self.inner.fit(centers, values)
    }

    /// Predict at `query_points`.
    ///
    /// # Errors
    /// See [`OutOfCoreRbf::predict`].
    pub fn predict(&self, query_points: &Array2<f64>) -> Result<Array1<f64>, InterpolateError> {
        self.inner.predict(query_points)
    }

    /// Delete the disk-backed coefficient file.
    pub fn cleanup(&self) -> Result<(), InterpolateError> {
        self.inner.cleanup()
    }

    /// Number of training centers.
    pub fn n_centers(&self) -> usize {
        self.inner.n_centers()
    }

    /// Input dimensionality.
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Reference to the kriging configuration.
    pub fn config(&self) -> &OutOfCoreKrigingConfig {
        &self.config
    }
}
