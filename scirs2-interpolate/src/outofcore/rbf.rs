//! Out-of-core Radial Basis Function interpolation.
//!
//! [`OutOfCoreRbf`] fits an RBF model on datasets too large to hold entirely in
//! RAM by storing the computed coefficients on disk via [`DiskStorage`].  During
//! prediction, coefficient chunks are loaded from disk as needed.
//!
//! # Fit strategy
//! For `n ≤ chunk_size × 10` ("small"), the full `n × n` kernel matrix is built
//! in memory and solved directly with `scirs2_linalg::solve`.
//!
//! For `n > chunk_size × 10` ("large"), a Nyström-style landmark approximation
//! is used: `m = chunk_size` landmark points are chosen uniformly, and the
//! coefficients are projected back to `n` positions so that prediction still
//! works with a single dot-product against all centers.
//!
//! # Out-of-core prediction
//! The coefficient vector is read back from disk in one pass; the kernel row for
//! each query point is computed on-the-fly against all centers (which must fit
//! in RAM, as they are needed for every query).
//!
//! # Example
//! ```rust
//! use std::env::temp_dir;
//! use scirs2_core::ndarray::{Array1, Array2};
//! use scirs2_interpolate::outofcore::{OutOfCoreRbf, OutOfCoreRbfConfig, OocRbfKernel};
//!
//! let n = 50usize;
//! let centers = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64);
//! let values  = centers.column(0).mapv(|x| (x * std::f64::consts::PI).sin());
//! let query   = Array2::from_shape_fn((10, 1), |(i, _)| i as f64 / 10.0);
//!
//! let scratch = temp_dir().join("doctest_ooc_rbf");
//! std::fs::create_dir_all(&scratch).ok();
//!
//! let mut ooc = OutOfCoreRbf::new(OutOfCoreRbfConfig {
//!     scratch_dir: scratch.clone(),
//!     kernel: OocRbfKernel::Gaussian,
//!     chunk_size: 25,
//!     ..Default::default()
//! });
//! ooc.fit(&centers, &values).expect("fit");
//! let preds = ooc.predict(&query).expect("predict");
//! assert_eq!(preds.len(), 10);
//! ooc.cleanup().ok();
//! std::fs::remove_dir_all(&scratch).ok();
//! ```

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_linalg::solve;
use std::path::PathBuf;

use super::storage::DiskStorage;
use crate::error::InterpolateError;

// ---------------------------------------------------------------------------
// RBF kernel enum
// ---------------------------------------------------------------------------

/// Kernel functions available for out-of-core RBF interpolation.
///
/// Named `OocRbfKernel` to avoid collision with the existing `rbf::RbfKernel`
/// in the parent crate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OocRbfKernel {
    /// Gaussian: `exp(−ε r²)`.  Requires positive `epsilon`.
    Gaussian,
    /// Thin-plate spline: `r² ln(r)` (0 when r = 0).
    ThinPlate,
    /// Multiquadric: `sqrt(r² + ε²)`.  Requires positive `epsilon`.
    Multiquadric,
    /// Inverse multiquadric: `1 / sqrt(r² + ε²)`.  Requires positive `epsilon`.
    InverseMultiquadric,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`OutOfCoreRbf`].
#[derive(Debug, Clone)]
pub struct OutOfCoreRbfConfig {
    /// Number of rows processed per iteration during prediction (default: 1000).
    pub chunk_size: usize,
    /// Soft cap on in-memory data (informational; not enforced by current impl).
    pub cache_size_mb: f64,
    /// Directory where the coefficient file is written.
    pub scratch_dir: PathBuf,
    /// RBF kernel to use.
    pub kernel: OocRbfKernel,
    /// Shape parameter `ε` for Gaussian, Multiquadric, InverseMultiquadric.
    pub epsilon: f64,
    /// Tikhonov regularisation added to the kernel matrix diagonal.
    pub regularization: f64,
}

impl Default for OutOfCoreRbfConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            cache_size_mb: 100.0,
            scratch_dir: std::env::temp_dir(),
            kernel: OocRbfKernel::ThinPlate,
            epsilon: 1.0,
            regularization: 1e-10,
        }
    }
}

// ---------------------------------------------------------------------------
// OutOfCoreRbf
// ---------------------------------------------------------------------------

/// Out-of-core Radial Basis Function interpolator.
///
/// Centers are kept in memory (needed for every prediction).  Coefficients are
/// stored on disk and loaded on demand.
pub struct OutOfCoreRbf {
    config: OutOfCoreRbfConfig,
    /// Training centers kept in RAM (needed for kernel evaluation during predict).
    centers: Array2<f64>,
    /// Backing store for the coefficient vector (length = `n_centers`).
    coeff_storage: Option<DiskStorage>,
    /// Absolute path to the coefficient file.
    coeff_path: PathBuf,
    /// Dimensionality of input space.
    d_in: usize,
    /// Number of training centers.
    n: usize,
}

impl OutOfCoreRbf {
    /// Construct an unfitted interpolator with the given configuration.
    pub fn new(config: OutOfCoreRbfConfig) -> Self {
        let coeff_path = config.scratch_dir.join("outofcore_rbf_coeffs.bin");
        Self {
            config,
            centers: Array2::zeros((0, 0)),
            coeff_storage: None,
            coeff_path,
            d_in: 0,
            n: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Kernel evaluation
    // -----------------------------------------------------------------------

    fn eval_kernel(&self, r: f64) -> f64 {
        let eps = self.config.epsilon;
        match self.config.kernel {
            OocRbfKernel::Gaussian => (-eps * r * r).exp(),
            OocRbfKernel::ThinPlate => {
                if r == 0.0 {
                    0.0
                } else {
                    r * r * r.ln()
                }
            }
            OocRbfKernel::Multiquadric => (r * r + eps * eps).sqrt(),
            OocRbfKernel::InverseMultiquadric => 1.0 / (r * r + eps * eps).sqrt(),
        }
    }

    /// Compute the kernel row `k(xi, xj)` for `j = 0..centers.nrows()`.
    fn kernel_row(&self, xi: ArrayView1<f64>, centers: &Array2<f64>) -> Array1<f64> {
        let n = centers.nrows();
        let d = centers.ncols();
        let mut row = Array1::zeros(n);
        for j in 0..n {
            let mut sq = 0.0_f64;
            for k in 0..d {
                let diff = xi[k] - centers[[j, k]];
                sq += diff * diff;
            }
            row[j] = self.eval_kernel(sq.sqrt());
        }
        row
    }

    // -----------------------------------------------------------------------
    // Internal solve helper
    // -----------------------------------------------------------------------

    /// Solve the linear system `A x = b` using `scirs2_linalg::solve`.
    fn solve_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, InterpolateError> {
        solve(&a.view(), &b.view(), None)
            .map_err(|e| InterpolateError::LinalgError(format!("OOC RBF solve: {e}")))
    }

    // -----------------------------------------------------------------------
    // Fit: direct (small n)
    // -----------------------------------------------------------------------

    fn fit_direct(
        &self,
        centers: &Array2<f64>,
        values: &Array1<f64>,
    ) -> Result<Array1<f64>, InterpolateError> {
        let n = centers.nrows();
        let reg = self.config.regularization;

        // Build the full n×n kernel matrix
        let mut k = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let row = self.kernel_row(centers.row(i), centers);
            for j in 0..n {
                k[[i, j]] = row[j];
            }
            k[[i, i]] += reg;
        }

        Self::solve_system(&k, values)
    }

    // -----------------------------------------------------------------------
    // Fit: landmark approximation (large n)
    // -----------------------------------------------------------------------

    /// Nyström-style approximation using `m = chunk_size` landmark points.
    ///
    /// We solve `K_mm α_m = K_mn y` (least-squares projection), then expand
    /// `α_m` back to an `n`-length vector so prediction uses a single
    /// `k(x, centers) · α` dot-product.
    fn fit_landmark(
        &self,
        centers: &Array2<f64>,
        values: &Array1<f64>,
    ) -> Result<Array1<f64>, InterpolateError> {
        let n = centers.nrows();
        let m = self.config.chunk_size.min(n);
        let step = if m > 0 { n / m } else { 1 };

        let landmark_idx: Vec<usize> = (0..m).map(|k| (k * step).min(n - 1)).collect();
        let landmarks = centers.select(Axis(0), &landmark_idx);

        // Build K_mm [m × m]
        let reg = self.config.regularization;
        let mut k_mm = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            let row = self.kernel_row(landmarks.row(i), &landmarks);
            for j in 0..m {
                k_mm[[i, j]] = row[j];
            }
            k_mm[[i, i]] += reg;
        }

        // Compute K_mn y  (m-vector: each entry is the dot of row i of K_mn with y)
        // K_mn[i, j] = k(landmark_i, center_j), so K_mn y = sum_j k(landmark_i, cj) * y_j
        let mut k_mn_y = Array1::<f64>::zeros(m);
        for i in 0..m {
            let row = self.kernel_row(landmarks.row(i), centers);
            k_mn_y[i] = row.dot(values);
        }

        // Solve K_mm alpha_m = K_mn y
        let alpha_m = Self::solve_system(&k_mm, &k_mn_y)?;

        // Expand to n-length vector: place alpha_m[i] at landmark_idx[i]
        let mut full_alpha = Array1::<f64>::zeros(n);
        for (li, &idx) in landmark_idx.iter().enumerate() {
            full_alpha[idx] = alpha_m[li];
        }
        Ok(full_alpha)
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Fit the model on the given centres and target values.
    ///
    /// Stores the resulting coefficient vector in a disk-backed file under
    /// `config.scratch_dir`.
    ///
    /// # Errors
    /// Returns [`InterpolateError`] on shape mismatch, singular kernel matrix,
    /// or I/O failure when writing the coefficient file.
    pub fn fit(
        &mut self,
        centers: &Array2<f64>,
        values: &Array1<f64>,
    ) -> Result<(), InterpolateError> {
        let n = centers.nrows();
        let d = centers.ncols();

        if n == 0 {
            return Err(InterpolateError::InvalidInput {
                message: "OutOfCoreRbf::fit: empty center set".into(),
            });
        }
        if values.len() != n {
            return Err(InterpolateError::ShapeMismatch {
                expected: format!("{n} values"),
                actual: format!("{} values", values.len()),
                object: "OutOfCoreRbf::fit values vs centers".into(),
            });
        }

        self.n = n;
        self.d_in = d;
        self.centers = centers.to_owned();

        // Choose direct or landmark path
        let use_direct = n <= self.config.chunk_size.saturating_mul(10);
        let coefficients = if use_direct {
            self.fit_direct(centers, values)?
        } else {
            self.fit_landmark(centers, values)?
        };

        // Ensure scratch dir exists and write coefficients to disk
        std::fs::create_dir_all(&self.config.scratch_dir).map_err(|e| {
            InterpolateError::IoError(format!(
                "OutOfCoreRbf: cannot create scratch_dir '{}': {e}",
                self.config.scratch_dir.display()
            ))
        })?;

        let storage = DiskStorage::create(&self.coeff_path, n, 1)?;
        let coeff_slice = coefficients.as_slice().ok_or_else(|| {
            InterpolateError::ComputationError(
                "OutOfCoreRbf: coefficient array is non-contiguous".into(),
            )
        })?;
        storage.write_rows(0, coeff_slice)?;
        self.coeff_storage = Some(storage);
        Ok(())
    }

    /// Predict at `query_points` (shape `[n_query, d]`).
    ///
    /// Coefficients are loaded from disk; the kernel row for each query is
    /// computed in `chunk_size` batches.
    ///
    /// # Errors
    /// Returns [`InterpolateError::ComputationError`] if the model has not been
    /// fitted yet, or on I/O / shape errors.
    pub fn predict(&self, query_points: &Array2<f64>) -> Result<Array1<f64>, InterpolateError> {
        let storage = self.coeff_storage.as_ref().ok_or_else(|| {
            InterpolateError::ComputationError("OutOfCoreRbf: model not fitted".into())
        })?;

        if query_points.ncols() != self.d_in {
            return Err(InterpolateError::DimensionMismatch(format!(
                "query dim {} != training dim {}",
                query_points.ncols(),
                self.d_in
            )));
        }

        // Load full coefficient vector from disk
        let coeff_raw = storage.read_rows(0, self.n)?;
        let coefficients = Array1::from_vec(coeff_raw);

        let n_query = query_points.nrows();
        let mut predictions = Array1::<f64>::zeros(n_query);

        for qi in 0..n_query {
            let k_row = self.kernel_row(query_points.row(qi), &self.centers);
            predictions[qi] = k_row.dot(&coefficients);
        }

        Ok(predictions)
    }

    /// Delete the backing coefficient file.
    ///
    /// Idempotent: no error if the file no longer exists.
    pub fn cleanup(&self) -> Result<(), InterpolateError> {
        if self.coeff_path.exists() {
            std::fs::remove_file(&self.coeff_path)
                .map_err(|e| InterpolateError::IoError(format!("OutOfCoreRbf::cleanup: {e}")))?;
        }
        Ok(())
    }

    /// Return the number of training centers.
    pub fn n_centers(&self) -> usize {
        self.n
    }

    /// Return the input dimensionality.
    pub fn dim(&self) -> usize {
        self.d_in
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &OutOfCoreRbfConfig {
        &self.config
    }
}
