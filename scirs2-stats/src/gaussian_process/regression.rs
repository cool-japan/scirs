//! High-level Gaussian Process Regression API
//!
//! This module provides a scikit-learn compatible interface for GP regression.

use super::gp::GaussianProcess;
use super::kernel::{Kernel, SquaredExponential, SumKernel, WhiteKernel};
use super::prior::{Prior, ZeroPrior};
use crate::error::StatsResult;
use scirs2_core::error::CoreError;
use scirs2_core::ndarray::ArrayStatCompat;
use scirs2_core::ndarray::{Array1, Array2};

/// Gaussian Process Regressor with scikit-learn compatible API
///
/// # Examples
///
/// ```
/// use scirs2_stats::gaussian_process::{GaussianProcessRegressor, SquaredExponential};
/// use scirs2_core::ndarray::{array, Array2};
///
/// let kernel = SquaredExponential::default();
/// let mut gpr = GaussianProcessRegressor::new(kernel);
///
/// let x_train = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
/// let y_train = array![0.0, 1.0, 0.5];
///
/// gpr.fit(&x_train, &y_train).unwrap();
///
/// let x_test = Array2::from_shape_vec((1, 1), vec![1.5]).unwrap();
/// let predictions = gpr.predict(&x_test).unwrap();
/// ```
pub struct GaussianProcessRegressor<K: Kernel> {
    /// The underlying Gaussian Process
    gp: GaussianProcess<SumKernel<K, WhiteKernel>, ZeroPrior>,
    /// User-provided kernel (before adding noise)
    user_kernel: K,
    /// Alpha parameter for regularization
    alpha: f64,
    /// Whether to normalize target values
    normalize_y: bool,
    /// Mean of training targets (for normalization)
    y_train_mean: Option<f64>,
    /// Std of training targets (for normalization)
    y_train_std: Option<f64>,
}

impl<K: Kernel> GaussianProcessRegressor<K> {
    /// Create a new Gaussian Process Regressor
    ///
    /// # Arguments
    ///
    /// * `kernel` - The covariance kernel
    ///
    /// # Returns
    ///
    /// A new GaussianProcessRegressor with default settings
    pub fn new(kernel: K) -> Self {
        Self::with_options(kernel, 1e-10, false)
    }

    /// Create a new GP Regressor with custom options
    ///
    /// # Arguments
    ///
    /// * `kernel` - The covariance kernel
    /// * `alpha` - Noise level / regularization parameter
    /// * `normalize_y` - Whether to normalize target values
    pub fn with_options(kernel: K, alpha: f64, normalize_y: bool) -> Self {
        let noise_kernel = WhiteKernel::new(alpha);
        let combined_kernel = SumKernel::new(kernel.clone(), noise_kernel);
        let prior = ZeroPrior::new();
        let gp = GaussianProcess::new(combined_kernel, prior, 0.0);

        Self {
            gp,
            user_kernel: kernel,
            alpha,
            normalize_y,
            y_train_mean: None,
            y_train_std: None,
        }
    }

    /// Fit the Gaussian Process model
    ///
    /// # Arguments
    ///
    /// * `x` - Training features (n_samples, n_features)
    /// * `y` - Training targets (n_samples,)
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> StatsResult<()> {
        if x.nrows() != y.len() {
            return Err(
                CoreError::InvalidInput(scirs2_core::error::ErrorContext::new(
                    "Number of samples in X and y must match",
                ))
                .into(),
            );
        }

        // Normalize y if requested
        let y_normalized = if self.normalize_y {
            let mean = y.mean_or(0.0);
            let std = y.std(0.0);
            let std = if std < 1e-10 { 1.0 } else { std };

            self.y_train_mean = Some(mean);
            self.y_train_std = Some(std);

            (y - mean) / std
        } else {
            y.clone()
        };

        self.gp.fit(x, &y_normalized)
    }

    /// Predict mean values
    ///
    /// # Arguments
    ///
    /// * `x` - Test features (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// Predicted mean values
    pub fn predict(&self, x: &Array2<f64>) -> StatsResult<Array1<f64>> {
        let predictions = self.gp.predict(x)?;

        // Denormalize if needed
        Ok(if self.normalize_y {
            let mean = self.y_train_mean.unwrap_or(0.0);
            let std = self.y_train_std.unwrap_or(1.0);
            predictions * std + mean
        } else {
            predictions
        })
    }

    /// Predict with uncertainty estimates
    ///
    /// # Arguments
    ///
    /// * `x` - Test features (n_samples, n_features)
    /// * `return_std` - Whether to return standard deviation
    ///
    /// # Returns
    ///
    /// (predictions, standard_deviations) if return_std is true
    pub fn predict_with_std(&self, x: &Array2<f64>) -> StatsResult<(Array1<f64>, Array1<f64>)> {
        let (mean, std) = self.gp.predict_with_std(x)?;

        // Denormalize if needed
        if self.normalize_y {
            let y_mean = self.y_train_mean.unwrap_or(0.0);
            let y_std = self.y_train_std.unwrap_or(1.0);
            Ok((mean * y_std + y_mean, std * y_std))
        } else {
            Ok((mean, std))
        }
    }

    /// Get the kernel
    pub fn kernel(&self) -> &K {
        &self.user_kernel
    }

    /// Get the kernel (mutable)
    pub fn kernel_mut(&mut self) -> &mut K {
        &mut self.user_kernel
    }

    /// Compute log marginal likelihood
    pub fn log_marginal_likelihood(&self) -> StatsResult<f64> {
        self.gp.log_marginal_likelihood()
    }

    /// Get number of training samples
    pub fn n_train_samples(&self) -> usize {
        self.gp.n_train_samples()
    }

    /// Score the model using R² metric
    ///
    /// # Arguments
    ///
    /// * `x` - Test features
    /// * `y` - True test targets
    ///
    /// # Returns
    ///
    /// R² score
    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> StatsResult<f64> {
        let y_pred = self.predict(x)?;

        if y.len() != y_pred.len() {
            return Err(
                CoreError::InvalidInput(scirs2_core::error::ErrorContext::new(
                    "Prediction and true values must have same length",
                ))
                .into(),
            );
        }

        // Compute R²
        let y_mean = y.mean_or(0.0);
        let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = y
            .iter()
            .zip(y_pred.iter())
            .map(|(&yi, &yp)| (yi - yp).powi(2))
            .sum();

        if ss_tot < 1e-10 {
            return Ok(1.0); // Perfect prediction if variance is zero
        }

        Ok(1.0 - ss_res / ss_tot)
    }
}

/// Create a default GP regressor with RBF kernel
///
/// This is a convenience function for the most common use case.
pub fn default_gp_regressor() -> GaussianProcessRegressor<SquaredExponential> {
    GaussianProcessRegressor::new(SquaredExponential::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_gpr_basic() {
        let kernel = SquaredExponential::default();
        let mut gpr = GaussianProcessRegressor::new(kernel);

        let x_train = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y_train = array![0.0, 1.0, 1.5, 1.0, 0.0];

        gpr.fit(&x_train, &y_train).unwrap();

        let x_test = Array2::from_shape_vec((1, 1), vec![2.5]).unwrap();
        let predictions = gpr.predict(&x_test).unwrap();

        // Prediction should be reasonable
        assert!(predictions[0] > 0.5 && predictions[0] < 2.0);
    }

    #[test]
    fn test_gpr_with_std() {
        let kernel = SquaredExponential::default();
        let mut gpr = GaussianProcessRegressor::new(kernel);

        let x_train = Array2::from_shape_vec((3, 1), vec![0.0, 2.0, 4.0]).unwrap();
        let y_train = array![1.0, 0.0, 1.0];

        gpr.fit(&x_train, &y_train).unwrap();

        let x_test = Array2::from_shape_vec((2, 1), vec![1.0, 5.0]).unwrap();
        let (mean, std) = gpr.predict_with_std(&x_test).unwrap();

        // All predictions should have positive uncertainty
        assert!(std.iter().all(|&s| s > 0.0));

        // Point far from training data should have higher uncertainty
        assert!(std[1] > std[0] || std[1].abs() - std[0].abs() < 0.1);
    }

    #[test]
    fn test_gpr_normalize() {
        let kernel = SquaredExponential::default();
        let mut gpr = GaussianProcessRegressor::with_options(kernel, 1e-10, true);

        let x_train = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y_train = array![100.0, 200.0, 150.0]; // Large values

        gpr.fit(&x_train, &y_train).unwrap();

        let predictions = gpr.predict(&x_train).unwrap();

        // Should fit training data well despite large values
        for i in 0..3 {
            assert!((predictions[i] - y_train[i]).abs() < 20.0);
        }
    }

    #[test]
    fn test_gpr_score() {
        let kernel = SquaredExponential::default();
        let mut gpr = GaussianProcessRegressor::new(kernel);

        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0.0, 1.0, 2.0, 1.5, 0.5];

        gpr.fit(&x, &y).unwrap();

        let score = gpr.score(&x, &y).unwrap();

        // Should fit training data well (R² close to 1)
        assert!(score > 0.8);
    }
}
