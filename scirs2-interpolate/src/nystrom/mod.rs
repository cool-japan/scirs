//! Nyström approximation for large-scale Kriging.
//!
//! Large-scale kernel methods are made tractable by approximating the full
//! n×n kernel matrix K with a low-rank approximation built from m « n
//! landmark points:
//!
//! ```text
//! K ≈ K_nm · K_mm^{-1} · K_nm^T
//! ```
//!
//! where `K_nm[i,j] = k(x_i, x_j^*)` with x^* being the m landmark points.
//!
//! # Module structure
//!
//! | Sub-module | Contents |
//! |---|---|
//! | [`landmarks`] | Landmark selection strategies (`UniformRandom`, `KMeansCenters`, `LeverageScore`) |
//! | [`predictor`] | `NystromKriging` struct and fit/predict/predict_variance |
//!
//! # References
//! - Williams, C. K. I. & Seeger, M. (2001). Using the Nyström method to speed up
//!   kernel machines. NIPS.
//! - Drineas, P. & Mahoney, M. W. (2005). On the Nyström method for approximating
//!   a Gram matrix for improved kernel-based learning. JMLR.

pub mod landmarks;
pub mod predictor;

// ─── Primary public API ───────────────────────────────────────────────────────

pub use landmarks::{select_landmarks, LandmarkStrategy};
pub use predictor::{kernel_eval, NystromKriging};

use crate::error::InterpolateError;
use crate::random_features::KernelType;

// ─── NystromConfig ───────────────────────────────────────────────────────────

/// Configuration for Nyström Kriging (legacy convenience API).
///
/// For new code prefer `NystromKriging::new(kernel, m, strategy, noise_variance,
/// bandwidth, seed)` directly.
#[derive(Debug, Clone)]
pub struct NystromConfig {
    /// Number of landmark (inducing) points.
    pub n_landmarks: usize,
    /// Kernel type.
    pub kernel: KernelType,
    /// Kernel bandwidth / length-scale.
    pub bandwidth: f64,
    /// Nugget / regularization added to the diagonal.
    pub regularization: f64,
    /// Seed for landmark selection.
    pub seed: u64,
}

impl Default for NystromConfig {
    fn default() -> Self {
        Self {
            n_landmarks: 100,
            kernel: KernelType::Gaussian,
            bandwidth: 1.0,
            regularization: 1e-6,
            seed: 0,
        }
    }
}

/// Backward-compatible alias: `apply_kernel` delegates to [`kernel_eval`].
///
/// `kernel_eval` is the canonical function; `apply_kernel` is kept for source
/// compatibility with any downstream code that uses the old name.
#[inline]
pub fn apply_kernel(kernel: &KernelType, bandwidth: f64, x1: &[f64], x2: &[f64]) -> f64 {
    kernel_eval(kernel, bandwidth, x1, x2)
}

// ─── NystromKrigingLegacy ────────────────────────────────────────────────────

/// Vec-based legacy wrapper around [`NystromKriging`].
///
/// Built from a [`NystromConfig`].  For new code prefer [`NystromKriging`].
#[derive(Debug, Clone)]
pub struct NystromKrigingLegacy {
    inner: NystromKriging,
    /// Echo the config for inspection.
    pub config: NystromConfig,
}

impl NystromKrigingLegacy {
    /// Create a new unfitted model from a [`NystromConfig`].
    pub fn new(config: NystromConfig) -> Self {
        let inner = NystromKriging::new(
            config.kernel.clone(),
            config.n_landmarks,
            LandmarkStrategy::KMeansCenters { n_iter: 20 },
            config.regularization,
            config.bandwidth,
            config.seed,
        );
        Self { inner, config }
    }

    /// Fit to training data (Vec-of-Vec API).
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<(), InterpolateError> {
        let n = x.len();
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "Training data is empty".to_string(),
            ));
        }
        if n != y.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "x has {} rows but y has {} elements",
                n,
                y.len()
            )));
        }
        let d = x[0].len();
        let x_arr = vec_of_vecs_to_array2(x, n, d)?;
        let y_arr = scirs2_core::ndarray::Array1::from_vec(y.to_vec());
        self.inner.fit(&x_arr, &y_arr)
    }

    /// Predict at test points (Vec-of-Vec API).
    pub fn predict(&self, x_test: &[Vec<f64>]) -> Result<Vec<f64>, InterpolateError> {
        let n_test = x_test.len();
        if n_test == 0 {
            return Ok(Vec::new());
        }
        let d = x_test[0].len();
        let x_arr = vec_of_vecs_to_array2(x_test, n_test, d)?;
        Ok(self.inner.predict(&x_arr)?.into_raw_vec_and_offset().0)
    }

    /// Predict with posterior variance (Vec-of-Vec API).
    pub fn predict_with_variance(
        &self,
        x_test: &[Vec<f64>],
    ) -> Result<(Vec<f64>, Vec<f64>), InterpolateError> {
        let n_test = x_test.len();
        if n_test == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let d = x_test[0].len();
        let x_arr = vec_of_vecs_to_array2(x_test, n_test, d)?;
        let means = self.inner.predict(&x_arr)?.into_raw_vec_and_offset().0;
        let vars = self
            .inner
            .predict_variance(&x_arr)?
            .into_raw_vec_and_offset()
            .0;
        Ok((means, vars))
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

fn vec_of_vecs_to_array2(
    x: &[Vec<f64>],
    n: usize,
    d: usize,
) -> Result<scirs2_core::ndarray::Array2<f64>, InterpolateError> {
    let flat: Vec<f64> = x.iter().flat_map(|row| row.iter().copied()).collect();
    scirs2_core::ndarray::Array2::from_shape_vec((n, d), flat).map_err(|e| {
        InterpolateError::ShapeError(format!("Failed to convert Vec<Vec<f64>> to Array2: {e}"))
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::StdRng, RngExt, SeedableRng};

    fn make_gaussian_data(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                vec![
                    rng.random::<f64>() * 4.0 - 2.0,
                    rng.random::<f64>() * 4.0 - 2.0,
                ]
            })
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|xi| (-xi[0] * xi[0] - xi[1] * xi[1]).exp())
            .collect();
        (x, y)
    }

    #[test]
    fn test_nystrom_fit_and_predict() {
        let (x, y) = make_gaussian_data(50, 42);
        let config = NystromConfig {
            n_landmarks: 15,
            kernel: KernelType::Gaussian,
            bandwidth: 1.0,
            regularization: 1e-4,
            seed: 0,
        };
        let mut model = NystromKrigingLegacy::new(config);
        model.fit(&x, &y).expect("fit");

        let preds = model.predict(&x).expect("predict");
        assert_eq!(preds.len(), x.len());

        let mse: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(p, yi)| (p - yi).powi(2))
            .sum::<f64>()
            / preds.len() as f64;
        assert!(mse < 0.5, "MSE too large: {mse}");
    }

    #[test]
    fn test_predict_variance_non_negative() {
        let (x, y) = make_gaussian_data(30, 77);
        let config = NystromConfig {
            n_landmarks: 10,
            kernel: KernelType::Gaussian,
            bandwidth: 1.0,
            regularization: 1e-4,
            seed: 1,
        };
        let mut model = NystromKrigingLegacy::new(config);
        model.fit(&x, &y).expect("fit");

        let x_test: Vec<Vec<f64>> = (0..5)
            .map(|i| vec![i as f64 * 0.3, i as f64 * 0.3])
            .collect();
        let (means, vars) = model
            .predict_with_variance(&x_test)
            .expect("predict_with_variance");
        assert_eq!(means.len(), 5);
        assert_eq!(vars.len(), 5);
        for v in &vars {
            assert!(*v >= 0.0, "Variance must be non-negative, got {v}");
        }
    }

    #[test]
    fn test_apply_kernel_values() {
        let x1 = [0.0f64, 0.0];
        let x2 = [1.0f64, 0.0];
        let bw = 1.0;

        let g = apply_kernel(&KernelType::Gaussian, bw, &x1, &x2);
        let expected = (-0.5f64).exp();
        assert!((g - expected).abs() < 1e-10, "Gaussian kernel mismatch");

        let l = apply_kernel(&KernelType::Laplacian, bw, &x1, &x2);
        let exp_l = (-1.0f64).exp();
        assert!((l - exp_l).abs() < 1e-10, "Laplacian kernel mismatch");

        let m32 = apply_kernel(&KernelType::Matern32, bw, &x1, &x2);
        let v = 3.0_f64.sqrt();
        let expected_m32 = (1.0 + v) * (-v).exp();
        assert!(
            (m32 - expected_m32).abs() < 1e-10,
            "Matern32 kernel mismatch"
        );
    }

    #[test]
    fn test_nystrom_matern_kernel() {
        let (x, y) = make_gaussian_data(20, 100);
        let config = NystromConfig {
            n_landmarks: 8,
            kernel: KernelType::Matern52,
            bandwidth: 1.5,
            regularization: 1e-4,
            seed: 5,
        };
        let mut model = NystromKrigingLegacy::new(config);
        model.fit(&x, &y).expect("fit matern52");
        let preds = model.predict(&x[..5]).expect("predict");
        assert_eq!(preds.len(), 5);
        for p in &preds {
            assert!(p.is_finite(), "Prediction should be finite");
        }
    }
}
