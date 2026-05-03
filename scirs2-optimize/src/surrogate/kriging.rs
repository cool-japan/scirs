//! Kriging (Gaussian Process) Surrogate Model
//!
//! Kriging is a spatial interpolation method that provides not only predictions
//! but also prediction uncertainty (variance). It is the foundation of
//! Bayesian optimization and is well-suited for expensive black-box optimization.
//!
//! ## Features
//!
//! - Multiple correlation functions (Gaussian, Matern, exponential)
//! - Nugget parameter for handling noisy evaluations
//! - Maximum Likelihood Estimation (MLE) of hyperparameters
//! - Analytical prediction variance
//!
//! ## References
//!
//! - Sacks, J., Welch, W.J., Mitchell, T.J., Wynn, H.P. (1989).
//!   Design and Analysis of Computer Experiments.
//! - Rasmussen, C.E. & Williams, C.K.I. (2006).
//!   Gaussian Processes for Machine Learning.

use super::{solve_general, SurrogateModel};
use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::RngExt;

/// Correlation function for Kriging
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CorrelationFunction {
    /// Squared exponential (Gaussian): exp(-sum(theta_k * |x_k - y_k|^2))
    SquaredExponential,
    /// Matern 3/2: (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
    Matern32,
    /// Matern 5/2: (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
    Matern52,
    /// Exponential (Matern 1/2): exp(-r)
    Exponential,
    /// Power exponential: exp(-sum(theta_k * |x_k - y_k|^p))
    PowerExponential {
        /// Smoothness parameter (1 = exponential, 2 = Gaussian)
        p: f64,
    },
}

impl Default for CorrelationFunction {
    fn default() -> Self {
        CorrelationFunction::SquaredExponential
    }
}

/// Options for Kriging surrogate
#[derive(Debug, Clone)]
pub struct KrigingOptions {
    /// Correlation function to use
    pub correlation: CorrelationFunction,
    /// Nugget parameter (regularization / noise variance)
    /// If None, will be estimated from data
    pub nugget: Option<f64>,
    /// Whether to estimate hyperparameters via MLE
    pub optimize_hyperparams: bool,
    /// Number of random restarts for hyperparameter optimization
    pub n_restarts: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Initial length-scale parameters (theta). If None, uses heuristic.
    pub initial_theta: Option<Vec<f64>>,
    /// Lower bound for theta
    pub theta_lower: f64,
    /// Upper bound for theta
    pub theta_upper: f64,
}

impl Default for KrigingOptions {
    fn default() -> Self {
        Self {
            correlation: CorrelationFunction::default(),
            nugget: Some(1e-6),
            optimize_hyperparams: true,
            n_restarts: 5,
            seed: None,
            initial_theta: None,
            theta_lower: 1e-3,
            theta_upper: 1e3,
        }
    }
}

/// Kriging (Gaussian Process) Surrogate Model
pub struct KrigingSurrogate {
    options: KrigingOptions,
    /// Training points (normalized)
    x_train: Option<Array2<f64>>,
    /// Training values (normalized)
    y_train: Option<Array1<f64>>,
    /// Estimated length-scale parameters
    theta: Option<Vec<f64>>,
    /// Estimated nugget
    nugget: f64,
    /// Kriging weights (alpha = R^{-1} * (y - mu))
    alpha: Option<Array1<f64>>,
    /// Estimated mean (trend)
    mu: f64,
    /// Estimated process variance
    sigma_sq: f64,
    /// Correlation matrix (R)
    corr_matrix: Option<Array2<f64>>,
    /// Cholesky factor of R (lower triangular)
    chol_factor: Option<Array2<f64>>,
    /// Normalization parameters
    x_min: Option<Array1<f64>>,
    x_range: Option<Array1<f64>>,
    y_mean: f64,
    y_std: f64,
}

impl KrigingSurrogate {
    /// Create a new Kriging surrogate
    pub fn new(options: KrigingOptions) -> Self {
        let nugget = options.nugget.unwrap_or(1e-6);
        Self {
            options,
            x_train: None,
            y_train: None,
            theta: None,
            nugget,
            alpha: None,
            mu: 0.0,
            sigma_sq: 1.0,
            corr_matrix: None,
            chol_factor: None,
            x_min: None,
            x_range: None,
            y_mean: 0.0,
            y_std: 1.0,
        }
    }

    /// Compute correlation between two points given theta
    fn correlation(&self, x1: &[f64], x2: &[f64], theta: &[f64]) -> f64 {
        let d = x1.len();
        match self.options.correlation {
            CorrelationFunction::SquaredExponential => {
                let mut sum = 0.0;
                for k in 0..d {
                    let diff = x1[k] - x2[k];
                    sum += theta[k.min(theta.len() - 1)] * diff * diff;
                }
                (-sum).exp()
            }
            CorrelationFunction::Matern32 => {
                let mut weighted_sq_sum = 0.0;
                for k in 0..d {
                    let diff = x1[k] - x2[k];
                    weighted_sq_sum += theta[k.min(theta.len() - 1)] * diff * diff;
                }
                let r = (3.0 * weighted_sq_sum).sqrt();
                (1.0 + r) * (-r).exp()
            }
            CorrelationFunction::Matern52 => {
                let mut weighted_sq_sum = 0.0;
                for k in 0..d {
                    let diff = x1[k] - x2[k];
                    weighted_sq_sum += theta[k.min(theta.len() - 1)] * diff * diff;
                }
                let r = (5.0 * weighted_sq_sum).sqrt();
                (1.0 + r + r * r / 3.0) * (-r).exp()
            }
            CorrelationFunction::Exponential => {
                let mut sum = 0.0;
                for k in 0..d {
                    let diff = (x1[k] - x2[k]).abs();
                    sum += theta[k.min(theta.len() - 1)] * diff;
                }
                (-sum).exp()
            }
            CorrelationFunction::PowerExponential { p } => {
                let mut sum = 0.0;
                for k in 0..d {
                    let diff = (x1[k] - x2[k]).abs();
                    sum += theta[k.min(theta.len() - 1)] * diff.powf(p);
                }
                (-sum).exp()
            }
        }
    }

    /// Compute the correlation matrix for given points
    fn compute_correlation_matrix(
        &self,
        x: &Array2<f64>,
        theta: &[f64],
        nugget: f64,
    ) -> Array2<f64> {
        let n = x.nrows();
        let mut r = Array2::zeros((n, n));
        for i in 0..n {
            r[[i, i]] = 1.0 + nugget;
            let x_i: Vec<f64> = (0..x.ncols()).map(|k| x[[i, k]]).collect();
            for j in (i + 1)..n {
                let x_j: Vec<f64> = (0..x.ncols()).map(|k| x[[j, k]]).collect();
                let c = self.correlation(&x_i, &x_j, theta);
                r[[i, j]] = c;
                r[[j, i]] = c;
            }
        }
        r
    }

    /// Compute correlation vector between a point and training data
    fn compute_correlation_vector(
        &self,
        x: &[f64],
        x_train: &Array2<f64>,
        theta: &[f64],
    ) -> Array1<f64> {
        let n = x_train.nrows();
        let mut r = Array1::zeros(n);
        for i in 0..n {
            let x_i: Vec<f64> = (0..x_train.ncols()).map(|k| x_train[[i, k]]).collect();
            r[i] = self.correlation(x, &x_i, theta);
        }
        r
    }

    /// Cholesky decomposition
    fn cholesky(&self, a: &Array2<f64>) -> OptimizeResult<Array2<f64>> {
        let n = a.nrows();
        let mut l = Array2::zeros((n, n));
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[j, k]] * l[[j, k]];
            }
            let diag = a[[j, j]] - sum;
            if diag <= 0.0 {
                return Err(OptimizeError::ComputationError(
                    "Correlation matrix is not positive definite".to_string(),
                ));
            }
            l[[j, j]] = diag.sqrt();
            for i in (j + 1)..n {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
        Ok(l)
    }

    /// Solve L * x = b (forward substitution)
    fn solve_lower(&self, l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = b.len();
        let mut x = Array1::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[[i, j]] * x[j];
            }
            x[i] = (b[i] - sum) / l[[i, i]];
        }
        x
    }

    /// Solve L^T * x = b (back substitution)
    fn solve_upper(&self, l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = b.len();
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += l[[j, i]] * x[j];
            }
            x[i] = (b[i] - sum) / l[[i, i]];
        }
        x
    }

    /// Compute concentrated log-likelihood for given theta
    fn log_likelihood(
        &self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        theta: &[f64],
        nugget: f64,
    ) -> f64 {
        let n = x_train.nrows();
        let r = self.compute_correlation_matrix(x_train, theta, nugget);

        let chol = match self.cholesky(&r) {
            Ok(l) => l,
            Err(_) => return f64::NEG_INFINITY,
        };

        // Compute log determinant
        let log_det: f64 = (0..n).map(|i| chol[[i, i]].ln()).sum::<f64>() * 2.0;

        // Solve R * ones = r_ones  to get mu
        let ones = Array1::ones(n);
        let z = self.solve_lower(&chol, &ones);
        let r_inv_ones = self.solve_upper(&chol, &z);
        let ones_r_inv_ones: f64 = ones.dot(&r_inv_ones);

        if ones_r_inv_ones.abs() < 1e-30 {
            return f64::NEG_INFINITY;
        }

        // Solve R * y_solve = y
        let z_y = self.solve_lower(&chol, y_train);
        let r_inv_y = self.solve_upper(&chol, &z_y);

        let mu_hat = ones.dot(&r_inv_y) / ones_r_inv_ones;

        // Compute sigma^2
        let residual: Array1<f64> = y_train - mu_hat;
        let z_res = self.solve_lower(&chol, &residual);
        let r_inv_res = self.solve_upper(&chol, &z_res);
        let sigma_sq = residual.dot(&r_inv_res) / n as f64;

        if sigma_sq <= 0.0 {
            return f64::NEG_INFINITY;
        }

        // Concentrated log-likelihood
        -0.5 * (n as f64 * sigma_sq.ln() + log_det)
    }

    /// Optimize hyperparameters using random search with local refinement
    fn optimize_hyperparameters(
        &self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
    ) -> (Vec<f64>, f64) {
        let d = x_train.ncols();
        let seed = self
            .options
            .seed
            .unwrap_or_else(|| scirs2_core::random::rng().random_range(0..u64::MAX));
        let mut rng = StdRng::seed_from_u64(seed);

        let theta_lo = self.options.theta_lower;
        let theta_hi = self.options.theta_upper;
        let log_lo = theta_lo.ln();
        let log_hi = theta_hi.ln();

        let nugget = self.nugget;

        // Initial theta
        let mut best_theta: Vec<f64> = self
            .options
            .initial_theta
            .clone()
            .unwrap_or_else(|| vec![1.0; d]);

        let mut best_ll = self.log_likelihood(x_train, y_train, &best_theta, nugget);

        // Random restarts
        for _ in 0..self.options.n_restarts {
            let theta: Vec<f64> = (0..d)
                .map(|_| rng.random_range(log_lo..log_hi).exp())
                .collect();

            let ll = self.log_likelihood(x_train, y_train, &theta, nugget);
            if ll > best_ll {
                best_ll = ll;
                best_theta = theta;
            }
        }

        // Local refinement via coordinate-wise line search
        for _ in 0..3 {
            for k in 0..d {
                let mut best_tk = best_theta[k];
                let mut best_ll_k = best_ll;

                for &factor in &[0.1, 0.3, 0.5, 0.7, 1.5, 2.0, 3.0, 10.0] {
                    let mut trial = best_theta.clone();
                    trial[k] = (best_theta[k] * factor).clamp(theta_lo, theta_hi);
                    let ll = self.log_likelihood(x_train, y_train, &trial, nugget);
                    if ll > best_ll_k {
                        best_ll_k = ll;
                        best_tk = trial[k];
                    }
                }

                best_theta[k] = best_tk;
                best_ll = best_ll_k;
            }
        }

        (best_theta, nugget)
    }

    /// Normalize X to [0, 1]
    fn normalize_x(&self, x: &Array2<f64>) -> Array2<f64> {
        if let (Some(ref x_min), Some(ref x_range)) = (&self.x_min, &self.x_range) {
            let mut normalized = x.clone();
            for i in 0..x.nrows() {
                for j in 0..x.ncols() {
                    let r = if x_range[j] > 1e-30 { x_range[j] } else { 1.0 };
                    normalized[[i, j]] = (x[[i, j]] - x_min[j]) / r;
                }
            }
            normalized
        } else {
            x.clone()
        }
    }

    /// Normalize a single x point
    fn normalize_x_point(&self, x: &Array1<f64>) -> Vec<f64> {
        if let (Some(ref x_min), Some(ref x_range)) = (&self.x_min, &self.x_range) {
            x.iter()
                .enumerate()
                .map(|(j, &xj)| {
                    let r = if x_range[j] > 1e-30 { x_range[j] } else { 1.0 };
                    (xj - x_min[j]) / r
                })
                .collect()
        } else {
            x.to_vec()
        }
    }
}

impl SurrogateModel for KrigingSurrogate {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> OptimizeResult<()> {
        let n = x.nrows();
        let d = x.ncols();

        if n < 2 {
            return Err(OptimizeError::InvalidInput(
                "Need at least 2 data points for Kriging".to_string(),
            ));
        }

        // Compute normalization
        let mut x_min = Array1::zeros(d);
        let mut x_max = Array1::zeros(d);
        for j in 0..d {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for i in 0..n {
                if x[[i, j]] < lo {
                    lo = x[[i, j]];
                }
                if x[[i, j]] > hi {
                    hi = x[[i, j]];
                }
            }
            x_min[j] = lo;
            x_max[j] = hi;
        }
        let x_range = &x_max - &x_min;
        self.x_min = Some(x_min);
        self.x_range = Some(x_range);

        let y_sum: f64 = y.iter().sum();
        self.y_mean = y_sum / n as f64;
        let y_var: f64 = y.iter().map(|yi| (yi - self.y_mean).powi(2)).sum::<f64>() / n as f64;
        self.y_std = y_var.sqrt().max(1e-30);

        // Normalize
        let x_norm = self.normalize_x(x);
        let y_norm: Array1<f64> = y.mapv(|yi| (yi - self.y_mean) / self.y_std);

        // Optimize hyperparameters
        let (theta, nugget) = if self.options.optimize_hyperparams {
            self.optimize_hyperparameters(&x_norm, &y_norm)
        } else {
            let theta = self
                .options
                .initial_theta
                .clone()
                .unwrap_or_else(|| vec![1.0; d]);
            (theta, self.nugget)
        };
        self.theta = Some(theta.clone());
        self.nugget = nugget;

        // Build correlation matrix
        let r = self.compute_correlation_matrix(&x_norm, &theta, nugget);
        let chol = self.cholesky(&r)?;

        // Estimate mu
        let ones = Array1::ones(n);
        let z = self.solve_lower(&chol, &ones);
        let r_inv_ones = self.solve_upper(&chol, &z);
        let ones_r_inv_ones = ones.dot(&r_inv_ones);

        let z_y = self.solve_lower(&chol, &y_norm);
        let r_inv_y = self.solve_upper(&chol, &z_y);

        self.mu = if ones_r_inv_ones.abs() > 1e-30 {
            ones.dot(&r_inv_y) / ones_r_inv_ones
        } else {
            y_norm.mean().unwrap_or(0.0)
        };

        // Compute alpha = R^{-1} * (y - mu)
        let residual: Array1<f64> = &y_norm - self.mu;
        let z_res = self.solve_lower(&chol, &residual);
        let alpha = self.solve_upper(&chol, &z_res);

        // Estimate sigma^2
        self.sigma_sq = (residual.dot(&alpha) / n as f64).max(1e-20);

        self.alpha = Some(alpha);
        self.corr_matrix = Some(r);
        self.chol_factor = Some(chol);
        self.x_train = Some(x_norm);
        self.y_train = Some(y_norm);

        Ok(())
    }

    fn predict(&self, x: &Array1<f64>) -> OptimizeResult<f64> {
        let x_train = self
            .x_train
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("Model not fitted".to_string()))?;
        let alpha = self
            .alpha
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("Model not fitted".to_string()))?;
        let theta = self
            .theta
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("Model not fitted".to_string()))?;

        let x_norm = self.normalize_x_point(x);
        let r = self.compute_correlation_vector(&x_norm, x_train, theta);

        let prediction_norm = self.mu + r.dot(alpha);

        // Denormalize
        Ok(prediction_norm * self.y_std + self.y_mean)
    }

    fn predict_with_uncertainty(&self, x: &Array1<f64>) -> OptimizeResult<(f64, f64)> {
        let x_train = self
            .x_train
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("Model not fitted".to_string()))?;
        let alpha = self
            .alpha
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("Model not fitted".to_string()))?;
        let theta = self
            .theta
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("Model not fitted".to_string()))?;
        let chol = self
            .chol_factor
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("Model not fitted".to_string()))?;

        let n = x_train.nrows();
        let x_norm = self.normalize_x_point(x);
        let r = self.compute_correlation_vector(&x_norm, x_train, theta);

        // Mean prediction
        let prediction_norm = self.mu + r.dot(alpha);
        let mean = prediction_norm * self.y_std + self.y_mean;

        // Prediction variance via Kriging equations
        // s^2(x) = sigma^2 * (1 - r^T R^{-1} r + (1 - 1^T R^{-1} r)^2 / (1^T R^{-1} 1))
        let z = self.solve_lower(chol, &r);
        let rt_r_inv_r = z.dot(&z);

        let ones = Array1::ones(n);
        let z_ones = self.solve_lower(chol, &ones);
        let ones_r_inv_r: f64 = z_ones.dot(&z);
        let ones_r_inv_ones: f64 = z_ones.dot(&z_ones);

        let numerator = (1.0 - ones_r_inv_r).powi(2);
        let denominator = ones_r_inv_ones.max(1e-30);

        let mse_norm = self.sigma_sq * (1.0 - rt_r_inv_r + numerator / denominator).max(0.0);
        let std = (mse_norm * self.y_std * self.y_std).sqrt().max(1e-10);

        Ok((mean, std))
    }

    fn n_samples(&self) -> usize {
        self.x_train.as_ref().map_or(0, |x| x.nrows())
    }

    fn n_features(&self) -> usize {
        self.x_train.as_ref().map_or(0, |x| x.ncols())
    }

    fn update(&mut self, x: &Array1<f64>, y: f64) -> OptimizeResult<()> {
        // Refit with new data point
        let (new_x, new_y) =
            if let (Some(ref x_train), Some(ref y_train)) = (&self.x_train, &self.y_train) {
                let d = x_train.ncols();
                let n = x_train.nrows();

                // Denormalize
                let mut x_denorm = Array2::zeros((n, d));
                for i in 0..n {
                    for j in 0..d {
                        let r = self.x_range.as_ref().map_or(1.0, |xr| {
                            if xr[j] > 1e-30 {
                                xr[j]
                            } else {
                                1.0
                            }
                        });
                        let m = self.x_min.as_ref().map_or(0.0, |xm| xm[j]);
                        x_denorm[[i, j]] = x_train[[i, j]] * r + m;
                    }
                }
                let y_denorm: Array1<f64> = y_train.mapv(|yi| yi * self.y_std + self.y_mean);

                let mut new_x = Array2::zeros((n + 1, d));
                for i in 0..n {
                    for j in 0..d {
                        new_x[[i, j]] = x_denorm[[i, j]];
                    }
                }
                for j in 0..d {
                    new_x[[n, j]] = x[j];
                }

                let mut new_y = Array1::zeros(n + 1);
                for i in 0..n {
                    new_y[i] = y_denorm[i];
                }
                new_y[n] = y;

                (new_x, new_y)
            } else {
                let d = x.len();
                let mut new_x = Array2::zeros((1, d));
                for j in 0..d {
                    new_x[[0, j]] = x[j];
                }
                (new_x, Array1::from_vec(vec![y]))
            };

        self.fit(&new_x, &new_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kriging_basic_interpolation() {
        let x_train = Array2::from_shape_vec((5, 1), vec![0.0, 0.25, 0.5, 0.75, 1.0])
            .expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 0.25, 1.0, 0.75, 0.0]);

        let mut kriging = KrigingSurrogate::new(KrigingOptions {
            nugget: Some(1e-4),
            optimize_hyperparams: false,
            initial_theta: Some(vec![10.0]),
            ..Default::default()
        });

        let result = kriging.fit(&x_train, &y_train);
        assert!(result.is_ok(), "Kriging fit failed: {:?}", result.err());

        // Predict at training points (should approximate closely)
        for i in 0..5 {
            let x = Array1::from_vec(vec![x_train[[i, 0]]]);
            let pred = kriging.predict(&x).expect("Prediction failed");
            assert!(
                (pred - y_train[i]).abs() < 0.2,
                "Kriging interpolation error at {}: pred={}, actual={}",
                i,
                pred,
                y_train[i]
            );
        }
    }

    #[test]
    fn test_kriging_uncertainty() {
        let x_train = Array2::from_shape_vec((4, 1), vec![0.0, 0.33, 0.66, 1.0])
            .expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

        let mut kriging = KrigingSurrogate::new(KrigingOptions {
            nugget: Some(1e-4),
            optimize_hyperparams: false,
            initial_theta: Some(vec![5.0]),
            ..Default::default()
        });
        kriging.fit(&x_train, &y_train).expect("Fit failed");

        // Uncertainty at a training point should be lower
        let (_, unc_near) = kriging
            .predict_with_uncertainty(&Array1::from_vec(vec![0.33]))
            .expect("Prediction failed");
        let (_, unc_far) = kriging
            .predict_with_uncertainty(&Array1::from_vec(vec![2.0]))
            .expect("Prediction failed");

        assert!(
            unc_far > unc_near,
            "Far uncertainty ({}) should exceed near uncertainty ({})",
            unc_far,
            unc_near
        );
    }

    #[test]
    fn test_kriging_2d() {
        let x_train = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);

        let mut kriging = KrigingSurrogate::new(KrigingOptions {
            nugget: Some(1e-4),
            n_restarts: 2,
            ..Default::default()
        });
        assert!(kriging.fit(&x_train, &y_train).is_ok());

        let pred = kriging.predict(&Array1::from_vec(vec![0.5, 0.5]));
        assert!(pred.is_ok());
        let val = pred.expect("2D prediction failed");
        assert!(val > -1.0 && val < 3.0, "Kriging 2D prediction: {}", val);
    }

    #[test]
    fn test_kriging_matern32() {
        let x_train =
            Array2::from_shape_vec((3, 1), vec![0.0, 0.5, 1.0]).expect("Array creation failed");
        let y_train = Array1::from_vec(vec![1.0, 2.0, 1.0]);

        let mut kriging = KrigingSurrogate::new(KrigingOptions {
            correlation: CorrelationFunction::Matern32,
            nugget: Some(1e-4),
            optimize_hyperparams: false,
            initial_theta: Some(vec![5.0]),
            ..Default::default()
        });
        assert!(kriging.fit(&x_train, &y_train).is_ok());
        let pred = kriging.predict(&Array1::from_vec(vec![0.25]));
        assert!(pred.is_ok());
    }

    #[test]
    fn test_kriging_matern52() {
        let x_train =
            Array2::from_shape_vec((3, 1), vec![0.0, 0.5, 1.0]).expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 1.0, 0.0]);

        let mut kriging = KrigingSurrogate::new(KrigingOptions {
            correlation: CorrelationFunction::Matern52,
            nugget: Some(1e-4),
            optimize_hyperparams: false,
            initial_theta: Some(vec![5.0]),
            ..Default::default()
        });
        assert!(kriging.fit(&x_train, &y_train).is_ok());
    }

    #[test]
    fn test_kriging_exponential() {
        let x_train =
            Array2::from_shape_vec((3, 1), vec![0.0, 0.5, 1.0]).expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 1.0, 0.0]);

        let mut kriging = KrigingSurrogate::new(KrigingOptions {
            correlation: CorrelationFunction::Exponential,
            nugget: Some(1e-3),
            optimize_hyperparams: false,
            initial_theta: Some(vec![5.0]),
            ..Default::default()
        });
        assert!(kriging.fit(&x_train, &y_train).is_ok());
    }

    #[test]
    fn test_kriging_update() {
        let x_train =
            Array2::from_shape_vec((3, 1), vec![0.0, 0.5, 1.0]).expect("Array creation failed");
        let y_train = Array1::from_vec(vec![0.0, 1.0, 0.0]);

        let mut kriging = KrigingSurrogate::new(KrigingOptions {
            nugget: Some(1e-4),
            optimize_hyperparams: false,
            initial_theta: Some(vec![5.0]),
            ..Default::default()
        });
        kriging.fit(&x_train, &y_train).expect("Fit failed");
        assert_eq!(kriging.n_samples(), 3);

        kriging
            .update(&Array1::from_vec(vec![0.25]), 0.5)
            .expect("Update failed");
        assert_eq!(kriging.n_samples(), 4);
    }

    #[test]
    fn test_kriging_power_exponential() {
        let x_train =
            Array2::from_shape_vec((3, 1), vec![0.0, 0.5, 1.0]).expect("Array creation failed");
        let y_train = Array1::from_vec(vec![1.0, 0.5, 1.0]);

        let mut kriging = KrigingSurrogate::new(KrigingOptions {
            correlation: CorrelationFunction::PowerExponential { p: 1.5 },
            nugget: Some(1e-3),
            optimize_hyperparams: false,
            initial_theta: Some(vec![5.0]),
            ..Default::default()
        });
        assert!(kriging.fit(&x_train, &y_train).is_ok());
    }
}
