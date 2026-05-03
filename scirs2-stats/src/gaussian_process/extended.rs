//! Extended Gaussian Process API
//!
//! This module provides an alternative, slice-based API for Gaussian Process
//! kernels and regression, complementing the existing `Kernel` trait which
//! uses `ArrayView1`. The `KernelFunction` trait accepts raw `&[f64]` slices
//! and is more ergonomic for many use cases.
//!
//! # Kernels
//!
//! - [`RBFKernel`] – Radial Basis Function (squared-exponential)
//! - [`MaternKernel`] – Matérn ν = 3/2 or ν = 5/2
//! - [`PeriodicKernel`] – Periodic / stationary periodic
//! - [`CompositeKernel`] – Sum or product of two kernels
//!
//! # Regressor
//!
//! [`GpRegressor`] is a self-contained Gaussian Process regressor that exposes:
//!
//! - `fit(X, y)` – train on 2-D input matrix and 1-D target
//! - `predict(X_star)` → `(mean, variance)` – predictive distribution
//! - `log_marginal_likelihood()` – model fit criterion

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// KernelFunction trait
// ---------------------------------------------------------------------------

/// Trait for covariance functions operating on raw `&[f64]` slices.
///
/// Implementors define how similarity between two input vectors is measured.
pub trait KernelFunction: Clone + Send + Sync {
    /// Evaluate the kernel between two input vectors.
    fn call(&self, x1: &[f64], x2: &[f64]) -> f64;
}

// ---------------------------------------------------------------------------
// RBF (Squared-Exponential) Kernel
// ---------------------------------------------------------------------------

/// Radial Basis Function (squared-exponential) kernel.
///
/// k(x, x') = variance · exp(−‖x − x'‖² / (2 · length_scale²))
#[derive(Debug, Clone)]
pub struct RBFKernel {
    /// Length scale (controls how quickly correlation decays with distance).
    pub length_scale: f64,
    /// Output variance (signal variance).
    pub variance: f64,
}

impl RBFKernel {
    /// Create a new RBF kernel.
    pub fn new(length_scale: f64, variance: f64) -> Self {
        Self {
            length_scale,
            variance,
        }
    }
}

impl Default for RBFKernel {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            variance: 1.0,
        }
    }
}

impl KernelFunction for RBFKernel {
    fn call(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let sq_dist: f64 = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        self.variance * (-0.5 * sq_dist / (self.length_scale * self.length_scale)).exp()
    }
}

// ---------------------------------------------------------------------------
// Matérn Kernel
// ---------------------------------------------------------------------------

/// Smoothness parameter for the Matérn kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaternNu {
    /// ν = 1.5 → once differentiable.
    Half, // ν = 1/2 (exponential)
    /// ν = 3/2 → once mean-square differentiable.
    ThreeHalves,
    /// ν = 5/2 → twice mean-square differentiable.
    FiveHalves,
}

/// Matérn covariance kernel.
///
/// Supports ν ∈ {1/2, 3/2, 5/2}.
#[derive(Debug, Clone)]
pub struct MaternKernel {
    /// Length scale parameter.
    pub length_scale: f64,
    /// Smoothness: ν = 1/2, 3/2, or 5/2.
    pub nu: MaternNu,
}

impl MaternKernel {
    /// Create a new Matérn kernel.
    pub fn new(length_scale: f64, nu: MaternNu) -> Self {
        Self { length_scale, nu }
    }

    /// Create a Matérn-3/2 kernel (default).
    pub fn matern32(length_scale: f64) -> Self {
        Self::new(length_scale, MaternNu::ThreeHalves)
    }

    /// Create a Matérn-5/2 kernel.
    pub fn matern52(length_scale: f64) -> Self {
        Self::new(length_scale, MaternNu::FiveHalves)
    }
}

impl Default for MaternKernel {
    fn default() -> Self {
        Self::matern32(1.0)
    }
}

impl KernelFunction for MaternKernel {
    fn call(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let dist: f64 = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();

        let r = dist / self.length_scale;

        match self.nu {
            MaternNu::Half => (-r).exp(),
            MaternNu::ThreeHalves => {
                let sqrt3_r = 3.0_f64.sqrt() * r;
                (1.0 + sqrt3_r) * (-sqrt3_r).exp()
            }
            MaternNu::FiveHalves => {
                let sqrt5_r = 5.0_f64.sqrt() * r;
                (1.0 + sqrt5_r + sqrt5_r * sqrt5_r / 3.0) * (-sqrt5_r).exp()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Periodic Kernel
// ---------------------------------------------------------------------------

/// Periodic (stationary) kernel.
///
/// k(x, x') = exp(−2 sin²(π ‖x−x'‖ / period) / length_scale²)
#[derive(Debug, Clone)]
pub struct PeriodicKernel {
    /// Length scale (controls the smoothness of periodic patterns).
    pub length_scale: f64,
    /// Period of the pattern.
    pub period: f64,
}

impl PeriodicKernel {
    /// Create a new Periodic kernel.
    pub fn new(length_scale: f64, period: f64) -> Self {
        Self {
            length_scale,
            period,
        }
    }
}

impl Default for PeriodicKernel {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            period: 1.0,
        }
    }
}

impl KernelFunction for PeriodicKernel {
    fn call(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let dist: f64 = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();

        let sin_term = (std::f64::consts::PI * dist / self.period).sin();
        (-2.0 * sin_term * sin_term / (self.length_scale * self.length_scale)).exp()
    }
}

// ---------------------------------------------------------------------------
// Composite Kernel
// ---------------------------------------------------------------------------

/// Composition mode for combining two kernels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompositeMode {
    /// k(x, x') = k₁(x, x') + k₂(x, x')
    Sum,
    /// k(x, x') = k₁(x, x') · k₂(x, x')
    Product,
}

/// A kernel formed by composing two kernels via sum or product.
#[derive(Debug, Clone)]
pub struct CompositeKernel<K1, K2>
where
    K1: KernelFunction,
    K2: KernelFunction,
{
    kernel1: K1,
    kernel2: K2,
    mode: CompositeMode,
}

impl<K1: KernelFunction, K2: KernelFunction> CompositeKernel<K1, K2> {
    /// Create a sum kernel: k = k1 + k2.
    pub fn sum(kernel1: K1, kernel2: K2) -> Self {
        Self {
            kernel1,
            kernel2,
            mode: CompositeMode::Sum,
        }
    }

    /// Create a product kernel: k = k1 * k2.
    pub fn product(kernel1: K1, kernel2: K2) -> Self {
        Self {
            kernel1,
            kernel2,
            mode: CompositeMode::Product,
        }
    }
}

impl<K1: KernelFunction, K2: KernelFunction> KernelFunction for CompositeKernel<K1, K2> {
    fn call(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let v1 = self.kernel1.call(x1, x2);
        let v2 = self.kernel2.call(x1, x2);
        match self.mode {
            CompositeMode::Sum => v1 + v2,
            CompositeMode::Product => v1 * v2,
        }
    }
}

// ---------------------------------------------------------------------------
// Cholesky helpers
// ---------------------------------------------------------------------------

/// Cholesky decomposition (lower triangular).  Returns L such that A = L Lᵀ.
fn cholesky(a: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0_f64;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let diag = a[[i, i]] - sum;
                if diag < 0.0 {
                    return Err(StatsError::ComputationError(format!(
                        "Cholesky failed at ({i},{i}): negative diagonal {diag}"
                    )));
                }
                l[[i, i]] = diag.sqrt();
            } else {
                let ljj = l[[j, j]];
                if ljj == 0.0 {
                    return Err(StatsError::ComputationError(
                        "Cholesky failed: zero diagonal element".to_string(),
                    ));
                }
                l[[i, j]] = (a[[i, j]] - sum) / ljj;
            }
        }
    }
    Ok(l)
}

/// Solve L x = b (forward substitution, L lower triangular).
fn solve_lower(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut x = Array1::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

/// Solve Lᵀ x = b (back substitution, L lower triangular so Lᵀ is upper).
fn solve_upper_transpose(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

// ---------------------------------------------------------------------------
// GpRegressor
// ---------------------------------------------------------------------------

/// A self-contained Gaussian Process regressor using a `KernelFunction`.
///
/// Uses Cholesky-based inference for numerical stability.
///
/// # Example
///
/// ```
/// use scirs2_stats::gaussian_process::extended::{GpRegressor, RBFKernel};
///
/// let kernel = RBFKernel::default();
/// let mut gpr = GpRegressor::new(kernel, 1e-6);
///
/// let x_train = vec![vec![0.0], vec![1.0], vec![2.0]];
/// let y_train = vec![0.0, 1.0, 0.0];
///
/// gpr.fit(&x_train, &y_train).expect("fit failed");
///
/// let x_test = vec![vec![1.5_f64]];
/// let (mean, var) = gpr.predict(&x_test).expect("predict failed");
/// println!("mean={}, var={}", mean[0], var[0]);
/// ```
pub struct GpRegressor<K: KernelFunction> {
    kernel: K,
    /// Noise variance added to the diagonal.
    noise: f64,
    /// Training inputs as row vectors.
    x_train: Option<Vec<Vec<f64>>>,
    /// Cholesky factor L where K + σ²I = LLᵀ.
    chol: Option<Array2<f64>>,
    /// α = (K + σ²I)⁻¹ y.
    alpha: Option<Array1<f64>>,
    /// Cached log-marginal likelihood.
    lml: Option<f64>,
    /// Training targets.
    y_train: Option<Vec<f64>>,
}

impl<K: KernelFunction> GpRegressor<K> {
    /// Create a new `GpRegressor`.
    ///
    /// # Arguments
    /// * `kernel` – covariance function
    /// * `noise` – noise variance (added to diagonal)
    pub fn new(kernel: K, noise: f64) -> Self {
        Self {
            kernel,
            noise: noise.max(1e-12),
            x_train: None,
            chol: None,
            alpha: None,
            lml: None,
            y_train: None,
        }
    }

    /// Compute the n×n kernel matrix for a set of points.
    fn kernel_matrix(&self, xs: &[Vec<f64>]) -> Array2<f64> {
        let n = xs.len();
        let mut k = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let v = self.kernel.call(&xs[i], &xs[j]);
                k[[i, j]] = v;
                k[[j, i]] = v;
            }
        }
        k
    }

    /// Compute the n×m cross-kernel matrix K(X, X*).
    fn cross_kernel_matrix(&self, xs: &[Vec<f64>], xs_star: &[Vec<f64>]) -> Array2<f64> {
        let n = xs.len();
        let m = xs_star.len();
        let mut k = Array2::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                k[[i, j]] = self.kernel.call(&xs[i], &xs_star[j]);
            }
        }
        k
    }

    /// Fit the GP to training data.
    ///
    /// # Arguments
    /// * `x` – `n` training inputs (each a `Vec<f64>` feature vector)
    /// * `y` – `n` training targets
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> StatsResult<()> {
        if x.len() != y.len() {
            return Err(StatsError::InvalidArgument(
                "x and y must have the same length".to_string(),
            ));
        }
        if x.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Cannot fit with empty training data".to_string(),
            ));
        }

        let n = x.len();
        let mut kmat = self.kernel_matrix(x);

        // Add noise to diagonal
        for i in 0..n {
            kmat[[i, i]] += self.noise;
        }

        // Try Cholesky; if it fails add extra jitter
        let l = match cholesky(&kmat) {
            Ok(l) => l,
            Err(_) => {
                for i in 0..n {
                    kmat[[i, i]] += 1e-6;
                }
                cholesky(&kmat)?
            }
        };

        let y_arr = Array1::from_vec(y.to_vec());
        let v = solve_lower(&l, &y_arr);
        let alpha = solve_upper_transpose(&l, &v);

        // Log-marginal likelihood: -0.5 yᵀ α - Σ ln L_ii - n/2 ln(2π)
        let data_fit = y_arr
            .iter()
            .zip(alpha.iter())
            .map(|(&yi, &ai)| yi * ai)
            .sum::<f64>();
        let log_det: f64 = (0..n).map(|i| l[[i, i]].ln()).sum::<f64>() * 2.0;
        let lml =
            -0.5 * data_fit - 0.5 * log_det - 0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln();

        self.x_train = Some(x.to_vec());
        self.chol = Some(l);
        self.alpha = Some(alpha);
        self.lml = Some(lml);
        self.y_train = Some(y.to_vec());

        Ok(())
    }

    /// Predict mean and variance at test points.
    ///
    /// # Returns
    /// `(mean, variance)` – both `Vec<f64>` with length `m = x_star.len()`.
    pub fn predict(&self, x_star: &[Vec<f64>]) -> StatsResult<(Vec<f64>, Vec<f64>)> {
        let x_train = self
            .x_train
            .as_ref()
            .ok_or_else(|| StatsError::InvalidArgument("Model not fitted yet".to_string()))?;
        let chol = self
            .chol
            .as_ref()
            .ok_or_else(|| StatsError::InvalidArgument("Model not fitted yet".to_string()))?;
        let alpha = self
            .alpha
            .as_ref()
            .ok_or_else(|| StatsError::InvalidArgument("Model not fitted yet".to_string()))?;

        let n = x_train.len();
        let m = x_star.len();

        // K_star: (n, m) cross-covariance
        let k_star = self.cross_kernel_matrix(x_train, x_star);

        // Mean: μ* = K_star.T α
        let mut mean = vec![0.0_f64; m];
        for j in 0..m {
            for i in 0..n {
                mean[j] += k_star[[i, j]] * alpha[i];
            }
        }

        // Variance: σ²* = k** - v.T v  where v = L⁻¹ K_star
        let mut variance = vec![0.0_f64; m];
        for j in 0..m {
            // k(x_j*, x_j*)
            let k_diag = self.kernel.call(&x_star[j], &x_star[j]);

            // v = L⁻¹ K_star[:, j]
            let col = Array1::from_vec((0..n).map(|i| k_star[[i, j]]).collect());
            let v = solve_lower(chol, &col);
            let v_sq: f64 = v.iter().map(|&vi| vi * vi).sum();
            variance[j] = (k_diag - v_sq).max(0.0);
        }

        Ok((mean, variance))
    }

    /// Return the log-marginal likelihood (computed during fitting).
    ///
    /// Returns `None` if the model has not been fitted.
    pub fn log_marginal_likelihood(&self) -> Option<f64> {
        self.lml
    }
}

// ---------------------------------------------------------------------------
// Array2-based adapter
// ---------------------------------------------------------------------------

/// Adapter: convert `Array2<f64>` rows into `Vec<Vec<f64>>`.
fn array2_to_rows(arr: &Array2<f64>) -> Vec<Vec<f64>> {
    arr.rows()
        .into_iter()
        .map(|row| row.iter().copied().collect())
        .collect()
}

/// Adapt `GpRegressor` to accept `Array2<f64>` / `Array1<f64>`.
impl<K: KernelFunction> GpRegressor<K> {
    /// Fit from ndarray types.
    pub fn fit_arrays(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> StatsResult<()> {
        let rows = array2_to_rows(x);
        let y_vec: Vec<f64> = y.iter().copied().collect();
        self.fit(&rows, &y_vec)
    }

    /// Predict from ndarray types, returning `(mean_array, variance_array)`.
    pub fn predict_arrays(&self, x_star: &Array2<f64>) -> StatsResult<(Array1<f64>, Array1<f64>)> {
        let rows = array2_to_rows(x_star);
        let (mean, var) = self.predict(&rows)?;
        Ok((Array1::from_vec(mean), Array1::from_vec(var)))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----- KernelFunction trait & kernel tests -----

    #[test]
    fn test_rbf_kernel_diagonal() {
        let k = RBFKernel::new(1.0, 2.0);
        // k(x, x) = variance
        assert!(
            (k.call(&[1.0, 2.0], &[1.0, 2.0]) - 2.0).abs() < 1e-12,
            "RBF diagonal should equal variance"
        );
    }

    #[test]
    fn test_rbf_kernel_symmetry() {
        let k = RBFKernel::default();
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        assert!(
            (k.call(&a, &b) - k.call(&b, &a)).abs() < 1e-12,
            "RBF should be symmetric"
        );
    }

    #[test]
    fn test_rbf_kernel_decays() {
        let k = RBFKernel::default();
        let origin = vec![0.0];
        let near = vec![0.5];
        let far = vec![5.0];
        assert!(
            k.call(&origin, &near) > k.call(&origin, &far),
            "RBF should decay with distance"
        );
    }

    #[test]
    fn test_matern_kernel_32_diagonal_one() {
        let k = MaternKernel::matern32(1.0);
        // k(x, x) = 1 for Matérn (dist=0)
        assert!(
            (k.call(&[1.0, 2.0], &[1.0, 2.0]) - 1.0).abs() < 1e-12,
            "Matérn-3/2 diagonal should be 1"
        );
    }

    #[test]
    fn test_matern_kernel_52_symmetry() {
        let k = MaternKernel::matern52(1.5);
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(
            (k.call(&a, &b) - k.call(&b, &a)).abs() < 1e-12,
            "Matérn-5/2 should be symmetric"
        );
    }

    #[test]
    fn test_periodic_kernel_periodicity() {
        let k = PeriodicKernel::new(1.0, 2.0);
        let x = vec![0.0];
        let x_period = vec![2.0]; // shifted by one full period
                                  // Should be close to k(x, x) = 1.0 (same phase)
        let at_period = k.call(&x, &x_period);
        let at_self = k.call(&x, &x);
        assert!(
            (at_period - at_self).abs() < 1e-10,
            "Periodic kernel should repeat at period: {at_self} vs {at_period}"
        );
    }

    #[test]
    fn test_composite_sum_kernel() {
        let k1 = RBFKernel::new(1.0, 1.0);
        let k2 = RBFKernel::new(2.0, 0.5);
        let k_sum = CompositeKernel::sum(k1.clone(), k2.clone());
        let x = vec![0.0];
        let y = vec![1.0];
        let expected = k1.call(&x, &y) + k2.call(&x, &y);
        assert!((k_sum.call(&x, &y) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_composite_product_kernel() {
        let k1 = RBFKernel::new(1.0, 1.0);
        let k2 = MaternKernel::matern32(1.0);
        let k_prod = CompositeKernel::product(k1.clone(), k2.clone());
        let x = vec![0.0];
        let y = vec![1.0];
        let expected = k1.call(&x, &y) * k2.call(&x, &y);
        assert!((k_prod.call(&x, &y) - expected).abs() < 1e-12);
    }

    // ----- GpRegressor tests -----

    #[test]
    fn test_gp_fit_predict_mean_at_training_points() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let mut gpr = GpRegressor::new(kernel, 1e-8);
        let x = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y = vec![0.0, 1.0, 0.0];
        gpr.fit(&x, &y).expect("fit failed");
        let (mean, _var) = gpr.predict(&x).expect("predict failed");
        for (i, (&m, &yi)) in mean.iter().zip(y.iter()).enumerate() {
            assert!(
                (m - yi).abs() < 1e-4,
                "mean[{i}] = {m} should be close to y[{i}] = {yi}"
            );
        }
    }

    #[test]
    fn test_gp_variance_positive() {
        let kernel = RBFKernel::default();
        let mut gpr = GpRegressor::new(kernel, 1e-6);
        let x = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y = vec![0.0, 1.0, 0.0];
        gpr.fit(&x, &y).expect("fit failed");
        let x_test = vec![vec![0.5], vec![1.5]];
        let (_mean, var) = gpr.predict(&x_test).expect("predict failed");
        for &v in &var {
            assert!(v >= 0.0, "variance should be non-negative, got {v}");
        }
    }

    #[test]
    fn test_gp_log_marginal_likelihood_finite() {
        let kernel = RBFKernel::default();
        let mut gpr = GpRegressor::new(kernel, 1e-6);
        let x = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y = vec![0.0, 1.0, 0.0];
        gpr.fit(&x, &y).expect("fit failed");
        let lml = gpr
            .log_marginal_likelihood()
            .expect("lml should be set after fit");
        assert!(lml.is_finite(), "LML should be finite, got {lml}");
    }

    #[test]
    fn test_gp_predict_before_fit_returns_error() {
        let kernel = RBFKernel::default();
        let gpr: GpRegressor<RBFKernel> = GpRegressor::new(kernel, 1e-6);
        let x_test = vec![vec![0.0]];
        assert!(gpr.predict(&x_test).is_err());
    }

    #[test]
    fn test_gp_matern_kernel() {
        let kernel = MaternKernel::matern52(1.0);
        let mut gpr = GpRegressor::new(kernel, 1e-6);
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![1.0, 0.0, 1.0, 0.0];
        gpr.fit(&x, &y).expect("fit with Matérn-5/2 failed");
        let (_m, _v) = gpr.predict(&vec![vec![1.5]]).expect("predict failed");
    }

    #[test]
    fn test_gp_composite_kernel() {
        let rbf = RBFKernel::new(1.0, 1.0);
        let mat = MaternKernel::matern32(1.0);
        let kernel = CompositeKernel::sum(rbf, mat);
        let mut gpr = GpRegressor::new(kernel, 1e-6);
        let x = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y = vec![0.0, 1.0, 0.0];
        gpr.fit(&x, &y).expect("fit with composite kernel failed");
        let (mean, _v) = gpr.predict(&vec![vec![1.0]]).expect("predict failed");
        assert!(mean[0].is_finite());
    }
}
