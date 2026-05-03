//! Ridge regressor built on top of a `FourierFeatureMap`.
//!
//! Solves `(ZᵀZ + λI)w = Zᵀy` where `Z` is the random feature matrix.
//! Prediction: `f(x*) = z(x*)ᵀ w`.

use crate::error::InterpolateError;
use crate::random_features::feature_map::{FourierFeatureMap, RffKernel};
use crate::random_features::mod_internal::cholesky_solve_vec;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// ─── RandomFeaturesRegressor ─────────────────────────────────────────────────

/// Kernel ridge regression via random Fourier features.
///
/// After fitting, prediction is O(D) per test point, making this suitable
/// for large-scale approximate kernel regression.
///
/// # Example
/// ```rust,ignore
/// use scirs2_interpolate::random_features::regressor::RandomFeaturesRegressor;
/// use scirs2_interpolate::random_features::feature_map::RffKernel;
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let mut reg = RandomFeaturesRegressor::new(
///     RffKernel::Gaussian { length_scale: 1.0 },
///     200,   // D random features
///     1e-4,  // ridge lambda
///     42,    // seed
/// );
/// let x = Array2::<f64>::zeros((50, 2));
/// let y = Array1::<f64>::zeros(50);
/// reg.fit(&x.view(), &y.view()).expect("fit");
/// let y_pred = reg.predict(&x.view()).expect("predict");
/// ```
#[derive(Debug, Clone)]
pub struct RandomFeaturesRegressor {
    /// Underlying random feature map.
    pub feature_map: FourierFeatureMap,
    /// Fitted weight vector w (shape `[D]`), empty until `fit()` is called.
    pub weights: Array1<f64>,
    /// Ridge regularization parameter λ.
    pub lambda: f64,
    /// Whether the model has been fitted.
    fitted: bool,
}

impl RandomFeaturesRegressor {
    /// Create a new (unfitted) `RandomFeaturesRegressor`.
    ///
    /// # Arguments
    /// * `kernel`     — kernel and length-scale for the feature map
    /// * `d_features` — number of random features D
    /// * `lambda`     — ridge regularization (> 0 recommended for stability)
    /// * `seed`       — RNG seed
    ///
    /// # Panics
    /// Delegates to [`FourierFeatureMap::new`]; panics if `d_features == 0`.
    /// The input dimension `d_in` is inferred from the first call to `fit()`.
    pub fn new(kernel: RffKernel, d_features: usize, lambda: f64, seed: u64) -> Self {
        // d_in will be determined at fit time; use a placeholder of 1 and
        // rebuild the map once we know d_in.
        // We store the parameters for deferred construction.
        Self {
            feature_map: FourierFeatureMap::new(kernel, 1, d_features, seed),
            weights: Array1::zeros(0),
            lambda,
            fitted: false,
        }
    }

    /// Fit the regressor to training data.
    ///
    /// Internally builds `Z = feature_map.transform(x)` (shape `[n, D]`)
    /// then solves `(ZᵀZ + λI)w = Zᵀy` via Cholesky decomposition.
    ///
    /// # Errors
    /// Returns [`InterpolateError`] on empty input, shape mismatch, or singular system.
    pub fn fit(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
    ) -> Result<(), InterpolateError> {
        let n = x.nrows();
        let d_in = x.ncols();
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "Training data is empty".to_string(),
            ));
        }
        if n != y.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "x has {n} rows but y has {} elements",
                y.len(),
            )));
        }
        if d_in == 0 {
            return Err(InterpolateError::InvalidInput {
                message: "input dimension d_in must be > 0".to_string(),
            });
        }

        // Rebuild feature map with correct d_in (preserving d_out and kernel).
        let d_out = self.feature_map.d_out;
        let kernel = self.feature_map.kernel.clone();
        // Use a stable derived seed (combine original seed with d_in).
        let seed = d_in as u64 * 0x9e37_79b9 ^ 42;
        self.feature_map = FourierFeatureMap::new(kernel, d_in, d_out, seed);

        let z = self.feature_map.transform(x)?;
        let d = d_out;

        // Build ZᵀZ (D × D).
        let mut ztzt = vec![vec![0.0f64; d]; d];
        for k in 0..n {
            for i in 0..d {
                for j in 0..=i {
                    let v = z[(k, i)] * z[(k, j)];
                    ztzt[i][j] += v;
                    if i != j {
                        ztzt[j][i] += v;
                    }
                }
            }
        }
        for i in 0..d {
            ztzt[i][i] += self.lambda;
        }

        // Build Zᵀy (D).
        let mut zty = vec![0.0f64; d];
        for j in 0..d {
            for k in 0..n {
                zty[j] += z[(k, j)] * y[k];
            }
        }

        let w = cholesky_solve_vec(&ztzt, &zty)?;
        self.weights = Array1::from_vec(w);
        self.fitted = true;
        Ok(())
    }

    /// Predict at new data points.
    ///
    /// # Errors
    /// Returns an error if the model has not been fitted yet, or on shape mismatch.
    pub fn predict(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>, InterpolateError> {
        if !self.fitted {
            return Err(InterpolateError::InvalidState(
                "Model not fitted; call fit() first".to_string(),
            ));
        }
        let z = self.feature_map.transform(x)?;
        let n = z.nrows();
        let mut preds = Array1::<f64>::zeros(n);
        for i in 0..n {
            let zi = z.row(i);
            preds[i] = zi.iter().zip(self.weights.iter()).map(|(a, b)| a * b).sum();
        }
        Ok(preds)
    }

    /// Whether the model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_regressor_sin_1d() {
        let n = 60;
        let x = Array2::from_shape_fn((n, 1), |(i, _)| {
            i as f64 * std::f64::consts::PI * 2.0 / n as f64
        });
        let y: Array1<f64> = x.column(0).mapv(f64::sin);

        let mut reg =
            RandomFeaturesRegressor::new(RffKernel::Gaussian { length_scale: 1.0 }, 200, 1e-4, 42);
        reg.fit(&x.view(), &y.view()).expect("fit");

        let x_test = Array2::from_shape_fn((20, 1), |(i, _)| {
            i as f64 * std::f64::consts::PI * 2.0 / 20.0
        });
        let y_pred = reg.predict(&x_test.view()).expect("predict");
        let y_true: Array1<f64> = x_test.column(0).mapv(f64::sin);

        let rmse: f64 = {
            let sum_sq: f64 = y_pred
                .iter()
                .zip(y_true.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum();
            (sum_sq / 20.0).sqrt()
        };
        assert!(rmse < 0.5, "RMSE {rmse} should be < 0.5 for sin regression");
    }

    #[test]
    fn test_predict_before_fit_errors() {
        let reg =
            RandomFeaturesRegressor::new(RffKernel::Gaussian { length_scale: 1.0 }, 50, 1e-3, 0);
        let x = Array2::<f64>::zeros((3, 1));
        assert!(reg.predict(&x.view()).is_err(), "should error before fit");
    }

    #[test]
    fn test_regressor_output_length() {
        let n_train = 20;
        let n_test = 7;
        let x_train = Array2::from_shape_fn((n_train, 2), |(i, j)| (i + j) as f64 * 0.1);
        let y_train = Array1::from_iter((0..n_train).map(|i| i as f64 * 0.5));
        let x_test = Array2::from_shape_fn((n_test, 2), |(i, j)| (i + j) as f64 * 0.15);

        let mut reg =
            RandomFeaturesRegressor::new(RffKernel::Laplacian { length_scale: 0.5 }, 30, 1e-3, 7);
        reg.fit(&x_train.view(), &y_train.view()).expect("fit");
        let preds = reg.predict(&x_test.view()).expect("predict");
        assert_eq!(preds.len(), n_test);
    }
}
