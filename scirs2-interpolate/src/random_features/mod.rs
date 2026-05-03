//! Random Feature RBF interpolation using Rahimi-Recht (2007) random Fourier features.
//!
//! Approximates shift-invariant kernels via random Fourier features:
//!
//! ```text
//! k(x, y) ≈ z(x)^T z(y)
//! z(x) = sqrt(2/D) * [cos(ω_1^T x + b_1), ..., cos(ω_D^T x + b_D)]
//! ```
//!
//! where ω_i are sampled from the spectral density of the kernel.
//!
//! # References
//! - Rahimi, A. & Recht, B. (2007). Random features for large-scale kernel machines. NIPS.
//! - Yu, F. X. et al. (2016). Orthogonal Random Features. NeurIPS.

// Sub-modules — new ndarray-based API
pub mod feature_map;
pub(crate) mod mod_internal;
pub mod orthogonal;
pub mod regressor;

// Re-export key public types at the module level for convenience.
pub use feature_map::{FourierFeatureMap, RffKernel};
pub use orthogonal::OrthogonalFourierFeatureMap;
pub use regressor::RandomFeaturesRegressor;

use crate::error::InterpolateError;

// ─── Linear Congruential Generator ─────────────────────────────────────────

/// A seeded LCG pseudo-random number generator (64-bit Knuth multiplicative).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Uniform float in [0, 1)
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal sample via Box-Muller transform.
    fn next_normal(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 1e-300 {
                return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            }
        }
    }

    /// Cauchy sample: ratio of two standard normals (a/b where b != 0).
    fn next_cauchy(&mut self) -> f64 {
        loop {
            let b = self.next_normal();
            if b.abs() > 1e-15 {
                return self.next_normal() / b;
            }
        }
    }
}

// ─── KernelType ─────────────────────────────────────────────────────────────

/// Type of shift-invariant kernel to approximate.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum KernelType {
    /// Gaussian (RBF) kernel: k(x, y) = exp(-||x-y||² / (2 bw²))
    Gaussian,
    /// Laplacian kernel: k(x, y) = exp(-||x-y||₁ / bw)
    Laplacian,
    /// Cauchy kernel: k(x, y) = 1 / (1 + ||x-y||² / bw²)
    Cauchy,
    /// Matérn 3/2 kernel
    Matern32,
    /// Matérn 5/2 kernel
    Matern52,
}

// ─── RandomFeatureConfig ────────────────────────────────────────────────────

/// Configuration for the random feature map.
#[derive(Debug, Clone)]
pub struct RandomFeatureConfig {
    /// Number of random features D (approximation quality grows with D).
    pub n_features: usize,
    /// Kernel type to approximate.
    pub kernel: KernelType,
    /// Bandwidth / length-scale parameter.
    pub bandwidth: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for RandomFeatureConfig {
    fn default() -> Self {
        Self {
            n_features: 500,
            kernel: KernelType::Gaussian,
            bandwidth: 1.0,
            seed: 42,
        }
    }
}

// ─── RandomFeatureMap ────────────────────────────────────────────────────────

/// Random Fourier feature map z: R^d → R^(2D).
///
/// Approximates a shift-invariant kernel via:
/// ```text
/// z(x) = sqrt(2/D) * [cos(ω_1^T x + b_1), ..., cos(ω_D^T x + b_D)]
/// ```
#[derive(Debug, Clone)]
pub struct RandomFeatureMap {
    /// Frequency vectors, shape `[n_features][n_dims]`.
    pub weights: Vec<Vec<f64>>,
    /// Phase offsets, shape `[n_features]`.
    pub biases: Vec<f64>,
    /// Configuration used to create this map.
    pub config: RandomFeatureConfig,
    /// Number of input dimensions.
    n_dims: usize,
}

impl RandomFeatureMap {
    /// Create a new random feature map for the given number of input dimensions.
    pub fn new(n_dims: usize, config: RandomFeatureConfig) -> Result<Self, InterpolateError> {
        if n_dims == 0 {
            return Err(InterpolateError::InvalidInput {
                message: "n_dims must be > 0".to_string(),
            });
        }
        if config.n_features == 0 {
            return Err(InterpolateError::InvalidInput {
                message: "n_features must be > 0".to_string(),
            });
        }
        if config.bandwidth <= 0.0 {
            return Err(InterpolateError::InvalidInput {
                message: "bandwidth must be positive".to_string(),
            });
        }

        let mut rng = Lcg::new(config.seed);
        let d = config.n_features;
        let bw = config.bandwidth;

        let weights: Vec<Vec<f64>> = (0..d)
            .map(|_| {
                (0..n_dims)
                    .map(|_| match config.kernel {
                        KernelType::Gaussian => rng.next_normal() / bw,
                        KernelType::Laplacian | KernelType::Cauchy => rng.next_cauchy() / bw,
                        KernelType::Matern32 => {
                            // Matérn 3/2 spectral density ~ Student-t ν=3: sample as
                            // Normal / sqrt(Chi²(3)/3). Approximate via ratio method.
                            let g = rng.next_normal() / bw;
                            // Scale by chi factor for ν=3
                            let chi = {
                                let s: f64 = (0..3).map(|_| rng.next_normal().powi(2)).sum();
                                (s / 3.0).sqrt()
                            };
                            g / chi.max(1e-12)
                        }
                        KernelType::Matern52 => {
                            // Matérn 5/2 spectral density ~ Student-t ν=5
                            let g = rng.next_normal() / bw;
                            let chi = {
                                let s: f64 = (0..5).map(|_| rng.next_normal().powi(2)).sum();
                                (s / 5.0).sqrt()
                            };
                            g / chi.max(1e-12)
                        }
                    })
                    .collect()
            })
            .collect();

        let biases: Vec<f64> = (0..d)
            .map(|_| rng.next_f64() * 2.0 * std::f64::consts::PI)
            .collect();

        Ok(Self {
            weights,
            biases,
            config,
            n_dims,
        })
    }

    /// Map input points `x` (shape `[n_samples][n_dims]`) to features
    /// of shape `[n_samples][n_features]`.
    ///
    /// Feature formula: `z_i(x) = sqrt(2/D) * cos(ω_i^T x + b_i)`.
    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, InterpolateError> {
        if x.is_empty() {
            return Ok(Vec::new());
        }
        let n = x.len();
        let d = self.config.n_features;
        let scale = (2.0 / d as f64).sqrt();

        let mut z = vec![vec![0.0f64; d]; n];

        for (i, xi) in x.iter().enumerate() {
            if xi.len() != self.n_dims {
                return Err(InterpolateError::DimensionMismatch(format!(
                    "Expected {} dimensions, got {}",
                    self.n_dims,
                    xi.len()
                )));
            }
            for j in 0..d {
                let dot: f64 = self.weights[j]
                    .iter()
                    .zip(xi.iter())
                    .map(|(w, xv)| w * xv)
                    .sum();
                z[i][j] = scale * (dot + self.biases[j]).cos();
            }
        }

        Ok(z)
    }

    /// Approximate kernel value k(x1, x2) ≈ z(x1)^T z(x2).
    pub fn kernel_approx(&self, x1: &[f64], x2: &[f64]) -> Result<f64, InterpolateError> {
        if x1.len() != self.n_dims || x2.len() != self.n_dims {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Expected {} dimensions",
                self.n_dims
            )));
        }
        let scale = 2.0 / self.config.n_features as f64;
        let mut result = 0.0f64;
        for j in 0..self.config.n_features {
            let dot1: f64 = self.weights[j]
                .iter()
                .zip(x1.iter())
                .map(|(w, xv)| w * xv)
                .sum();
            let dot2: f64 = self.weights[j]
                .iter()
                .zip(x2.iter())
                .map(|(w, xv)| w * xv)
                .sum();
            result += (dot1 + self.biases[j]).cos() * (dot2 + self.biases[j]).cos();
        }
        Ok(scale * result)
    }
}

// ─── RandomFeatureInterpolator ───────────────────────────────────────────────

/// Kernel ridge regression using random Fourier features.
///
/// Solves: `(Z^T Z + λ I) w = Z^T y` where Z is the random feature matrix.
/// Prediction: `f(x*) = z(x*)^T w`.
#[derive(Debug, Clone)]
pub struct RandomFeatureInterpolator {
    /// The underlying random feature map.
    pub feature_map: RandomFeatureMap,
    /// Solved weight vector, shape `[n_features]`.
    pub weights: Vec<f64>,
    /// Ridge regularization parameter λ.
    pub regularization: f64,
    /// Whether the model has been fitted.
    fitted: bool,
}

impl RandomFeatureInterpolator {
    /// Create a new (unfitted) interpolator.
    pub fn new(
        n_dims: usize,
        config: RandomFeatureConfig,
        regularization: f64,
    ) -> Result<Self, InterpolateError> {
        let feature_map = RandomFeatureMap::new(n_dims, config)?;
        Ok(Self {
            feature_map,
            weights: Vec::new(),
            regularization,
            fitted: false,
        })
    }

    /// Fit to training data `(x, y)`.
    ///
    /// Solves `(Z^T Z + λ I) w = Z^T y` via Cholesky decomposition.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<(), InterpolateError> {
        if x.is_empty() {
            return Err(InterpolateError::InsufficientData(
                "Training data is empty".to_string(),
            ));
        }
        if x.len() != y.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "x has {} rows but y has {} elements",
                x.len(),
                y.len()
            )));
        }

        let z = self.feature_map.transform(x)?;
        let n = z.len();
        let d = self.feature_map.config.n_features;

        // Build Z^T Z  (d × d)
        let mut ztzt = vec![vec![0.0f64; d]; d];
        for i in 0..d {
            for j in 0..=i {
                let val: f64 = (0..n).map(|k| z[k][i] * z[k][j]).sum();
                ztzt[i][j] = val;
                ztzt[j][i] = val;
            }
            ztzt[i][i] += self.regularization;
        }

        // Build Z^T y  (d)
        let mut zty = vec![0.0f64; d];
        for j in 0..d {
            zty[j] = (0..n).map(|k| z[k][j] * y[k]).sum();
        }

        // Solve via Cholesky: (Z^T Z + λI) w = Z^T y
        self.weights = cholesky_solve(&ztzt, &zty)?;
        self.fitted = true;
        Ok(())
    }

    /// Predict at new points.
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, InterpolateError> {
        if !self.fitted {
            return Err(InterpolateError::InvalidState(
                "Model not fitted yet, call fit() first".to_string(),
            ));
        }
        let z = self.feature_map.transform(x)?;
        let preds: Vec<f64> = z
            .iter()
            .map(|zi| zi.iter().zip(self.weights.iter()).map(|(a, b)| a * b).sum())
            .collect();
        Ok(preds)
    }

    /// Compute mean absolute error between the random feature kernel approximation
    /// and the true kernel evaluated on all pairs in `x`.
    pub fn kernel_error(
        &self,
        x: &[Vec<f64>],
        true_kernel_fn: impl Fn(&[f64], &[f64]) -> f64,
    ) -> Result<f64, InterpolateError> {
        let n = x.len();
        if n == 0 {
            return Ok(0.0);
        }
        let mut total_err = 0.0f64;
        let mut count = 0usize;
        for i in 0..n {
            for j in 0..n {
                let approx = self.feature_map.kernel_approx(&x[i], &x[j])?;
                let exact = true_kernel_fn(&x[i], &x[j]);
                total_err += (approx - exact).abs();
                count += 1;
            }
        }
        if count == 0 {
            Ok(0.0)
        } else {
            Ok(total_err / count as f64)
        }
    }
}

// ─── Cholesky solver ─────────────────────────────────────────────────────────

/// Solve `A x = b` for a symmetric positive definite `A` via Cholesky decomposition.
/// Uses in-place lower triangular factoring.
pub(crate) fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, InterpolateError> {
    let n = a.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Compute lower Cholesky factor L in-place
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s: f64 = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s < 0.0 {
                    // Fallback: use diagonal regularization and retry
                    return cholesky_solve_fallback(a, b);
                }
                l[i][j] = s.sqrt().max(1e-300);
            } else {
                l[i][j] = s / l[j][j].max(1e-300);
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i].max(1e-300);
    }

    // Backward substitution: L^T x = y
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i].max(1e-300);
    }

    Ok(x)
}

/// Fallback Cholesky using conjugate gradient for near-singular systems.
fn cholesky_solve_fallback(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, InterpolateError> {
    // Use conjugate gradient method
    let n = a.len();
    let mut x = vec![0.0f64; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs_old: f64 = r.iter().map(|v| v * v).sum();

    for _ in 0..(n * 10) {
        // ap = A * p
        let ap: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| a[i][j] * p[j]).sum())
            .collect();
        let pap: f64 = p.iter().zip(ap.iter()).map(|(a, b)| a * b).sum();
        if pap.abs() < 1e-300 {
            break;
        }
        let alpha = rs_old / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new: f64 = r.iter().map(|v| v * v).sum();
        if rs_new.sqrt() < 1e-10 {
            break;
        }
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    Ok(x)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_kernel(bw: f64, x1: &[f64], x2: &[f64]) -> f64 {
        let sq: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        (-sq / (2.0 * bw * bw)).exp()
    }

    #[test]
    fn test_gaussian_kernel_approximation_error() {
        let config = RandomFeatureConfig {
            n_features: 200,
            kernel: KernelType::Gaussian,
            bandwidth: 1.0,
            seed: 12345,
        };
        let map = RandomFeatureMap::new(2, config).expect("should create map");

        // Test on a small grid
        let xs: Vec<Vec<f64>> = (0..5)
            .flat_map(|i| (0..5).map(move |j| vec![i as f64 * 0.5, j as f64 * 0.5]))
            .collect();

        let mut total_err = 0.0f64;
        let mut count = 0usize;
        let bw = 1.0;
        for x1 in &xs {
            for x2 in &xs {
                let approx = map.kernel_approx(x1, x2).expect("kernel approx");
                let exact = gaussian_kernel(bw, x1, x2);
                total_err += (approx - exact).abs();
                count += 1;
            }
        }
        let mean_err = total_err / count as f64;
        assert!(
            mean_err < 0.15,
            "Mean kernel approximation error too large: {mean_err}"
        );
    }

    #[test]
    fn test_fit_predict_sin() {
        // 1D sin function: wrap in Vec<Vec<f64>>
        let x_train: Vec<Vec<f64>> = (0..30)
            .map(|i| vec![i as f64 * 2.0 * std::f64::consts::PI / 30.0])
            .collect();
        let y_train: Vec<f64> = x_train.iter().map(|xi| xi[0].sin()).collect();

        let config = RandomFeatureConfig {
            n_features: 100,
            kernel: KernelType::Gaussian,
            bandwidth: 1.0,
            seed: 999,
        };
        let mut interp =
            RandomFeatureInterpolator::new(1, config, 1e-4).expect("create interpolator");
        interp.fit(&x_train, &y_train).expect("fit");

        let x_test: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64 * 2.0 * std::f64::consts::PI / 10.0])
            .collect();
        let preds = interp.predict(&x_test).expect("predict");
        assert_eq!(preds.len(), 10, "Prediction count mismatch");
    }

    #[test]
    fn test_predict_shape_correct() {
        let n = 15;
        let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 * 0.1]).collect();
        let y: Vec<f64> = x.iter().map(|xi| xi[0] * 2.0 + 1.0).collect();

        let config = RandomFeatureConfig {
            n_features: 50,
            kernel: KernelType::Laplacian,
            bandwidth: 0.5,
            seed: 7,
        };
        let mut interp = RandomFeatureInterpolator::new(1, config, 1e-3).expect("create");
        interp.fit(&x, &y).expect("fit");

        let x_new: Vec<Vec<f64>> = (0..7).map(|i| vec![i as f64 * 0.15]).collect();
        let preds = interp.predict(&x_new).expect("predict");
        assert_eq!(preds.len(), 7);
    }

    #[test]
    fn test_kernel_types() {
        for kernel in [
            KernelType::Gaussian,
            KernelType::Laplacian,
            KernelType::Cauchy,
            KernelType::Matern32,
            KernelType::Matern52,
        ] {
            let config = RandomFeatureConfig {
                n_features: 50,
                kernel,
                bandwidth: 1.0,
                seed: 1,
            };
            let map = RandomFeatureMap::new(2, config).expect("create");
            let x1 = vec![0.0, 0.0];
            let x2 = vec![1.0, 1.0];
            let k = map.kernel_approx(&x1, &x2).expect("kernel approx");
            assert!(k.is_finite(), "Kernel value should be finite");
        }
    }

    #[test]
    fn test_kernel_error_gaussian() {
        let config = RandomFeatureConfig {
            n_features: 300,
            kernel: KernelType::Gaussian,
            bandwidth: 1.0,
            seed: 42,
        };
        let x_train: Vec<Vec<f64>> = (0..5)
            .map(|i| vec![i as f64 * 0.4, i as f64 * 0.2])
            .collect();
        let y_train: Vec<f64> = x_train.iter().map(|xi| xi[0].sin()).collect();

        let mut interp = RandomFeatureInterpolator::new(2, config, 1e-4).expect("create");
        interp.fit(&x_train, &y_train).expect("fit");

        let bw = 1.0f64;
        let err = interp
            .kernel_error(&x_train, |x1, x2| {
                let sq: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                (-sq / (2.0 * bw * bw)).exp()
            })
            .expect("kernel error");
        assert!(err < 0.15, "Kernel error too large: {err}");
    }
}
