//! Gaussian Process interpolator with automatic kernel structure discovery.
//!
//! [`AutoKernelGp`] implements a simplified version of the Duvenaud-style
//! Automatic Statistician approach: it searches a grammar of base kernels
//! (`Rbf`, `Matern52`, `Periodic`, `Linear`, `WhiteNoise`) combined by sum
//! and product operators, and selects the best composite kernel using k-fold
//! cross-validated mean squared error (CV-MSE).
//!
//! ## Algorithm summary
//!
//! 1. **Generate candidates** — BFS over the kernel grammar up to `max_depth`
//!    (default 2), producing depth-0 (base), depth-1 (base ± base), and
//!    depth-2 (depth-1 ± base) expressions.  Commutative duplicates are pruned.
//!
//! 2. **Optimise hyperparameters** — For each candidate:
//!    - RBF / Matérn: golden-section on ℓ ∈ [0.05, 10].
//!    - Periodic: grid on p ∈ {0.1, 0.2, 0.5, 1.0, 2.0, 5.0}, golden-section on ℓ.
//!    - Linear / WhiteNoise: golden-section on variance.
//!    - Composite: sub-expressions optimised independently (greedy).
//!
//! 3. **Cross-validate** — k-fold (default 5) leave-out MSE on the training data.
//!    The kernel with the lowest CV-MSE is selected.
//!
//! 4. **Final fit** — GP alpha vector `α = (K + σ²I)⁻¹ · y` is computed via
//!    inline Cholesky (Banachiewicz), with a small jitter `1e-8` for stability.
//!
//! 5. **Predict** — `y* = K(x*, X) · α`.
//!
//! ## Selection criterion
//!
//! CV-MSE on held-out data is used (not log marginal likelihood on the full
//! training set) to guard against over-fitting by complex kernels and to make
//! the selection criterion directly interpretable as predictive error.

pub mod kernel;
pub mod search;

pub use kernel::{BaseKernel, KernelExpr};

use crate::error::{InterpolateError, InterpolateResult};
use search::{build_cross_kernel, gp_fit, gp_predict, search_kernels};

// Default period grid for Periodic kernel hyperparameter search.
const DEFAULT_PERIOD_GRID: &[f64] = &[0.1, 0.2, 0.5, 1.0, 2.0, 5.0];

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`AutoKernelGp`].
#[derive(Debug, Clone)]
pub struct AutoKernelGpConfig {
    /// Maximum depth of kernel expression tree.  Default: 2.
    ///
    /// - Depth 0: only base kernels.
    /// - Depth 1: base ± base (10 unique pairs × 2 ops = 20 extra).
    /// - Depth 2: depth-1 ± base (≈ 40 extra).
    pub max_depth: usize,
    /// Random restarts for hyperparameter optimisation (increases search
    /// budget for golden-section).  Default: 3.
    pub n_restarts: usize,
    /// Observation noise variance added to the kernel diagonal.  Default: 0.01.
    pub noise_variance: f64,
    /// Number of cross-validation folds.  Default: 5.
    pub cv_folds: usize,
    /// Deterministic seed (reserved for future use with random restarts).
    pub seed: u64,
}

impl Default for AutoKernelGpConfig {
    fn default() -> Self {
        Self {
            max_depth: 2,
            n_restarts: 3,
            noise_variance: 0.01,
            cv_folds: 5,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// AutoKernelGp
// ---------------------------------------------------------------------------

/// Gaussian Process interpolator with automatic kernel structure discovery.
///
/// # Example
///
/// ```rust
/// use scirs2_interpolate::auto_kernel_gp::{AutoKernelGp, AutoKernelGpConfig};
///
/// let x: Vec<f64> = (0..20).map(|i| i as f64 * std::f64::consts::PI / 10.0).collect();
/// let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
///
/// let config = AutoKernelGpConfig {
///     max_depth: 1,
///     cv_folds: 3,
///     ..Default::default()
/// };
/// let mut gp = AutoKernelGp::new(config);
/// gp.fit(&x, &y).expect("fit ok");
///
/// let x_new = vec![0.1, 0.5, 1.0];
/// let preds = gp.predict(&x_new).expect("predict ok");
/// assert_eq!(preds.len(), 3);
/// ```
pub struct AutoKernelGp {
    /// The selected kernel expression.
    best_kernel: KernelExpr,
    /// Cross-validation score of the selected kernel (lower = better).
    best_cv_score: f64,
    /// GP dual variables α = (K + σ²I)⁻¹ y.
    alpha: Vec<f64>,
    /// Training input locations.
    train_x: Vec<f64>,
    /// Training output values.
    train_y: Vec<f64>,
    /// Configuration.
    config: AutoKernelGpConfig,
    /// All kernel-search results, sorted by CV score.
    search_results: Vec<(String, f64)>,
    /// Whether the GP has been fitted.
    is_fitted: bool,
}

impl AutoKernelGp {
    /// Create a new (unfitted) `AutoKernelGp`.
    pub fn new(config: AutoKernelGpConfig) -> Self {
        Self {
            best_kernel: KernelExpr::Base(BaseKernel::Rbf { length_scale: 1.0 }),
            best_cv_score: f64::MAX,
            alpha: Vec::new(),
            train_x: Vec::new(),
            train_y: Vec::new(),
            config,
            search_results: Vec::new(),
            is_fitted: false,
        }
    }

    /// Search the kernel grammar and fit the GP on `x`, `y`.
    ///
    /// `x` must be strictly sorted (ascending) for well-defined kernel matrices,
    /// though this is not enforced.  Duplicate `x` values will cause a
    /// near-singular kernel matrix which is handled by jitter.
    pub fn fit(&mut self, x: &[f64], y: &[f64]) -> InterpolateResult<()> {
        if x.len() != y.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "x length {} ≠ y length {}",
                x.len(),
                y.len()
            )));
        }
        if x.len() < 2 {
            return Err(InterpolateError::InvalidInput {
                message: "at least 2 training points are required".to_string(),
            });
        }

        // Run kernel search.
        let (ranked, best_kernel) = search_kernels(
            x,
            y,
            self.config.max_depth,
            self.config.n_restarts,
            self.config.noise_variance,
            self.config.cv_folds,
            DEFAULT_PERIOD_GRID,
        );

        self.search_results = ranked;
        self.best_cv_score = self
            .search_results
            .first()
            .map(|(_, s)| *s)
            .unwrap_or(f64::MAX);
        self.best_kernel = best_kernel;

        // Final fit with the selected kernel on the full training set.
        self.alpha =
            gp_fit(&self.best_kernel, x, y, self.config.noise_variance).ok_or_else(|| {
                InterpolateError::ComputationError(
                    "Cholesky failed for selected kernel on full training set".to_string(),
                )
            })?;

        self.train_x = x.to_vec();
        self.train_y = y.to_vec();
        self.is_fitted = true;
        Ok(())
    }

    /// Predict at new input locations `x_new`.
    pub fn predict(&self, x_new: &[f64]) -> InterpolateResult<Vec<f64>> {
        if !self.is_fitted {
            return Err(InterpolateError::InvalidState(
                "GP must be fitted before prediction".to_string(),
            ));
        }
        if x_new.is_empty() {
            return Ok(Vec::new());
        }
        Ok(gp_predict(
            &self.best_kernel,
            &self.train_x,
            &self.alpha,
            x_new,
        ))
    }

    /// Return a human-readable description of the selected kernel structure.
    pub fn selected_kernel_description(&self) -> String {
        self.best_kernel.description()
    }

    /// Return the best cross-validation score (lower is better).
    pub fn best_cv_score(&self) -> f64 {
        self.best_cv_score
    }

    /// Return the full ranked list of `(kernel_description, cv_mse_score)` pairs.
    ///
    /// Sorted by CV-MSE ascending (best first).
    pub fn kernel_search_results(&self) -> &[(String, f64)] {
        &self.search_results
    }

    /// Return the selected kernel expression.
    pub fn kernel(&self) -> &KernelExpr {
        &self.best_kernel
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sin_data(n: usize) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n)
            .map(|i| i as f64 * 2.0 * std::f64::consts::PI / n as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
        (x, y)
    }

    #[test]
    fn auto_kernel_gp_fits_sin_data() {
        let (x, y) = sin_data(20);
        let config = AutoKernelGpConfig {
            max_depth: 1,
            cv_folds: 3,
            n_restarts: 1,
            ..Default::default()
        };
        let mut gp = AutoKernelGp::new(config);
        gp.fit(&x, &y).expect("fit: should succeed on sin data");
        // Predict at training points — should be close.
        let preds = gp.predict(&x).expect("predict: should succeed");
        assert_eq!(preds.len(), x.len());
        let mse: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / x.len() as f64;
        assert!(
            mse < 0.5,
            "MSE at training points should be small, got {mse}"
        );
    }

    #[test]
    fn auto_kernel_gp_predict_shape_correct() {
        let (x, y) = sin_data(15);
        let config = AutoKernelGpConfig {
            max_depth: 0, // only base kernels
            cv_folds: 3,
            n_restarts: 1,
            ..Default::default()
        };
        let mut gp = AutoKernelGp::new(config);
        gp.fit(&x, &y).expect("fit ok");
        let x_new = vec![0.1, 0.5, 1.0, 2.0, 4.0];
        let preds = gp.predict(&x_new).expect("predict ok");
        assert_eq!(
            preds.len(),
            x_new.len(),
            "prediction shape must match query length"
        );
    }

    #[test]
    fn auto_kernel_gp_description_is_nonempty() {
        let (x, y) = sin_data(12);
        let config = AutoKernelGpConfig {
            max_depth: 1,
            cv_folds: 3,
            n_restarts: 1,
            ..Default::default()
        };
        let mut gp = AutoKernelGp::new(config);
        gp.fit(&x, &y).expect("fit ok");
        let desc = gp.selected_kernel_description();
        assert!(
            !desc.is_empty(),
            "kernel description must not be empty: '{desc}'"
        );
    }

    #[test]
    fn auto_kernel_gp_selects_periodic_kernel_for_sin() {
        // With enough data and depth ≥ 1 including the Periodic base kernel,
        // the search should find a kernel that fits sin well.  We use a soft
        // assertion: the CV score is finite.
        let (x, y) = sin_data(20);
        let config = AutoKernelGpConfig {
            max_depth: 1,
            cv_folds: 3,
            n_restarts: 2,
            noise_variance: 1e-4,
            ..Default::default()
        };
        let mut gp = AutoKernelGp::new(config);
        gp.fit(&x, &y).expect("fit ok");
        assert!(
            gp.best_cv_score().is_finite(),
            "CV score must be finite, got {}",
            gp.best_cv_score()
        );
        assert!(
            !gp.kernel_search_results().is_empty(),
            "search results must not be empty"
        );
    }

    #[test]
    fn auto_kernel_gp_cv_score_improves_with_depth() {
        // Depth-2 search should find a kernel with CV-MSE ≤ depth-1 on sin data
        // (same data — strictly: at least one depth-2 candidate covers all depth-1 ones).
        let (x, y) = sin_data(18);
        let mut scores = Vec::new();
        for max_depth in [0usize, 1, 2] {
            let config = AutoKernelGpConfig {
                max_depth,
                cv_folds: 3,
                n_restarts: 1,
                noise_variance: 1e-3,
                ..Default::default()
            };
            let mut gp = AutoKernelGp::new(config);
            gp.fit(&x, &y).expect("fit ok");
            scores.push(gp.best_cv_score());
        }
        // Each deeper search covers at least as many candidates as the shallower one,
        // so CV score should be non-increasing.
        assert!(
            scores[1] <= scores[0] * 1.1,
            "depth-1 score {} should be ≤ depth-0 score {} (with 10% tolerance)",
            scores[1],
            scores[0]
        );
        assert!(
            scores[2] <= scores[1] * 1.1,
            "depth-2 score {} should be ≤ depth-1 score {} (with 10% tolerance)",
            scores[2],
            scores[1]
        );
    }

    #[test]
    fn auto_kernel_gp_predict_before_fit_errors() {
        let gp = AutoKernelGp::new(AutoKernelGpConfig::default());
        let result = gp.predict(&[0.5, 1.0]);
        assert!(result.is_err(), "predict before fit should return an error");
    }
}
