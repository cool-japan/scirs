//! Density ratio estimation algorithms.
//!
//! Density ratio estimation estimates the ratio w(x) = p(x) / q(x) between
//! two probability densities from finite samples. These estimates are used in
//! covariate shift adaptation, importance weighting, two-sample testing, and
//! change-point detection.
//!
//! # Algorithms
//!
//! * [`RuLSIF`] - Relative Unconstrained Least-Squares Importance Fitting
//! * [`KLIEP`] - KL Importance Estimation Procedure
//! * [`KullbackLeibler`] - Plug-in KL divergence ratio estimator
//!
//! # Example
//!
//! ```rust
//! use scirs2_cluster::density_ratio::{DensityRatioEstimator, KLIEP, importance_weights, ImportanceMethod};
//! use scirs2_core::ndarray::Array2;
//!
//! // Source and target samples
//! let source = Array2::from_shape_vec((6, 1), vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5]).expect("operation should succeed");
//! let target = Array2::from_shape_vec((6, 1), vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).expect("operation should succeed");
//!
//! let weights = importance_weights(source.view(), target.view(), ImportanceMethod::Kliep).expect("operation should succeed");
//! assert_eq!(weights.len(), 6);
//! ```

use std::f64::consts::PI;

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{ClusteringError, Result};

// ─── Trait ──────────────────────────────────────────────────────────────────

/// Trait for density ratio estimators: w(x) ≈ p_numerator(x) / p_denominator(x).
pub trait DensityRatioEstimator {
    /// Fit the estimator using numerator (`p`) and denominator (`q`) samples.
    ///
    /// Both arrays have shape `(n_samples, n_features)`.
    fn fit(&mut self, numerator: ArrayView2<f64>, denominator: ArrayView2<f64>) -> Result<()>;

    /// Predict importance weights w(x) ≈ p(x) / q(x) for new `query` points.
    ///
    /// Returns a vector of length `query.shape()[0]`.
    fn predict(&self, query: ArrayView2<f64>) -> Result<Vec<f64>>;
}

// ─── Kernel helpers ─────────────────────────────────────────────────────────

/// Compute an RBF (Gaussian) kernel matrix between rows of `a` and rows of `b`.
///
/// `K[i, j] = exp(-||a_i - b_j||^2 / (2 * sigma^2))`
fn rbf_kernel_matrix(a: ArrayView2<f64>, b: ArrayView2<f64>, sigma: f64) -> Array2<f64> {
    let na = a.shape()[0];
    let nb = b.shape()[0];
    let d = a.shape()[1];
    let two_s2 = 2.0 * sigma * sigma;

    let mut k = Array2::<f64>::zeros((na, nb));
    for i in 0..na {
        for j in 0..nb {
            let mut sq = 0.0_f64;
            for f in 0..d {
                let diff = a[[i, f]] - b[[j, f]];
                sq += diff * diff;
            }
            k[[i, j]] = (-sq / two_s2).exp();
        }
    }
    k
}

/// Compute squared Euclidean distance between two row slices.
fn sq_dist(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi) * (ai - bi))
        .sum()
}

/// Median heuristic for the RBF bandwidth: sigma = median(pairwise distances) / sqrt(2).
fn median_bandwidth(data: ArrayView2<f64>) -> f64 {
    let n = data.shape()[0];
    if n <= 1 {
        return 1.0;
    }
    let mut dists = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            dists.push(sq_dist(data.row(i), data.row(j)).sqrt());
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = dists[dists.len() / 2];
    if med < 1e-10 {
        1.0
    } else {
        med / (2.0_f64).sqrt()
    }
}

/// Simple Cholesky-based solver for small positive-definite systems.
/// Solves `A x = b` where `A` is `(m×m)` symmetric positive-definite.
fn solve_pd_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let m = a.shape()[0];
    if m == 0 {
        return Err(ClusteringError::ComputationError(
            "Empty system".to_string(),
        ));
    }
    // Cholesky: L s.t. A = L L^T
    let mut l = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s < 0.0 {
                    s = 1e-12; // numerical fallback
                }
                l[[i, j]] = s.sqrt();
            } else if l[[j, j]].abs() < 1e-15 {
                l[[i, j]] = 0.0;
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    // Forward solve: L y = b
    let mut y = Array1::<f64>::zeros(m);
    for i in 0..m {
        let mut s = b[i];
        for k in 0..i {
            s -= l[[i, k]] * y[k];
        }
        if l[[i, i]].abs() < 1e-15 {
            y[i] = 0.0;
        } else {
            y[i] = s / l[[i, i]];
        }
    }
    // Backward solve: L^T x = y
    let mut x = Array1::<f64>::zeros(m);
    for i in (0..m).rev() {
        let mut s = y[i];
        for k in (i + 1)..m {
            s -= l[[k, i]] * x[k];
        }
        if l[[i, i]].abs() < 1e-15 {
            x[i] = 0.0;
        } else {
            x[i] = s / l[[i, i]];
        }
    }
    Ok(x)
}

// ─── KullbackLeibler ─────────────────────────────────────────────────────────

/// Plug-in KL divergence density ratio estimator.
///
/// Estimates w(x) = p(x)/q(x) as a ratio of kernel density estimates.
/// Uses Gaussian kernels with a common bandwidth selected by the median heuristic.
///
/// This estimator is simple and intuitive but can suffer from the curse of
/// dimensionality in high dimensions; prefer [`RuLSIF`] or [`KLIEP`] in
/// production usage.
#[derive(Debug, Clone)]
pub struct KullbackLeibler {
    /// Bandwidth for the kernel density estimates.
    pub sigma: Option<f64>,
    /// Numerator samples stored after `fit`.
    numerator_samples: Option<Array2<f64>>,
    /// Denominator samples stored after `fit`.
    denominator_samples: Option<Array2<f64>>,
    /// Resolved bandwidth.
    resolved_sigma: f64,
}

impl KullbackLeibler {
    /// Create a new estimator.  Pass `None` to use the median heuristic.
    pub fn new(sigma: Option<f64>) -> Self {
        Self {
            sigma,
            numerator_samples: None,
            denominator_samples: None,
            resolved_sigma: 1.0,
        }
    }
}

impl DensityRatioEstimator for KullbackLeibler {
    fn fit(&mut self, numerator: ArrayView2<f64>, denominator: ArrayView2<f64>) -> Result<()> {
        if numerator.shape()[1] != denominator.shape()[1] {
            return Err(ClusteringError::InvalidInput(
                "numerator and denominator must have the same number of features".to_string(),
            ));
        }
        self.resolved_sigma = self.sigma.unwrap_or_else(|| {
            let s1 = median_bandwidth(numerator);
            let s2 = median_bandwidth(denominator);
            (s1 + s2) / 2.0
        });
        self.numerator_samples = Some(numerator.to_owned());
        self.denominator_samples = Some(denominator.to_owned());
        Ok(())
    }

    fn predict(&self, query: ArrayView2<f64>) -> Result<Vec<f64>> {
        let num_samples = self.numerator_samples.as_ref().ok_or_else(|| {
            ClusteringError::InvalidState("Call fit() before predict()".to_string())
        })?;
        let den_samples = self.denominator_samples.as_ref().ok_or_else(|| {
            ClusteringError::InvalidState("Call fit() before predict()".to_string())
        })?;

        let sigma = self.resolved_sigma;
        let n_num = num_samples.shape()[0] as f64;
        let n_den = den_samples.shape()[0] as f64;
        let n_query = query.shape()[0];

        // Two-pi normalization factor (same for both, cancels in ratio)
        let d = query.shape()[1] as f64;
        let norm = (2.0 * PI * sigma * sigma).powf(d / 2.0);

        let mut weights = Vec::with_capacity(n_query);
        for qi in 0..n_query {
            let xq = query.row(qi);

            // KDE for numerator
            let p_num: f64 = num_samples
                .rows()
                .into_iter()
                .map(|row| (-sq_dist(xq, row) / (2.0 * sigma * sigma)).exp())
                .sum::<f64>()
                / (n_num * norm);

            // KDE for denominator
            let p_den: f64 = den_samples
                .rows()
                .into_iter()
                .map(|row| (-sq_dist(xq, row) / (2.0 * sigma * sigma)).exp())
                .sum::<f64>()
                / (n_den * norm);

            let ratio = if p_den < 1e-300 { 0.0 } else { p_num / p_den };
            weights.push(ratio.max(0.0));
        }
        Ok(weights)
    }
}

// ─── KLIEP ───────────────────────────────────────────────────────────────────

/// KL Importance Estimation Procedure (KLIEP).
///
/// Estimates w(x) = p(x)/q(x) by directly minimising the KL divergence from
/// p to the model distribution ĝ = w·q.  The model is a non-negative linear
/// combination of RBF basis functions centred on numerator samples (or a
/// randomly selected subset).
///
/// Reference: Sugiyama et al. (2008), "Direct importance estimation with model
/// selection and its application to covariate shift adaptation", NIPS.
#[derive(Debug, Clone)]
pub struct KLIEP {
    /// RBF bandwidth. `None` selects via median heuristic.
    pub sigma: Option<f64>,
    /// L2 regularisation on the coefficient vector.
    pub lambda: f64,
    /// Maximum gradient ascent iterations.
    pub max_iter: usize,
    /// Learning rate for gradient ascent.
    pub learning_rate: f64,
    /// Number of basis centres (subsampled from numerator).  `None` = all.
    pub n_centers: Option<usize>,

    // fitted state
    alpha: Option<Array1<f64>>,
    centers: Option<Array2<f64>>,
    resolved_sigma: f64,
}

impl KLIEP {
    /// Create a new KLIEP estimator with default hyper-parameters.
    pub fn new() -> Self {
        Self {
            sigma: None,
            lambda: 1e-3,
            max_iter: 1000,
            learning_rate: 1e-3,
            n_centers: None,
            alpha: None,
            centers: None,
            resolved_sigma: 1.0,
        }
    }

    /// Builder: set bandwidth.
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = Some(sigma);
        self
    }

    /// Builder: set regularisation.
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Builder: set number of RBF centres.
    pub fn with_n_centers(mut self, n: usize) -> Self {
        self.n_centers = Some(n);
        self
    }
}

impl Default for KLIEP {
    fn default() -> Self {
        Self::new()
    }
}

impl DensityRatioEstimator for KLIEP {
    fn fit(&mut self, numerator: ArrayView2<f64>, denominator: ArrayView2<f64>) -> Result<()> {
        let n_num = numerator.shape()[0];
        let n_den = denominator.shape()[0];
        let n_feat = numerator.shape()[1];

        if n_feat != denominator.shape()[1] {
            return Err(ClusteringError::InvalidInput(
                "Feature dimension mismatch".to_string(),
            ));
        }
        if n_num == 0 || n_den == 0 {
            return Err(ClusteringError::InvalidInput(
                "Empty sample arrays".to_string(),
            ));
        }

        // Resolve bandwidth
        let sigma = self.sigma.unwrap_or_else(|| {
            let s1 = median_bandwidth(numerator);
            let s2 = median_bandwidth(denominator);
            (s1 + s2) / 2.0
        });
        self.resolved_sigma = sigma;

        // Select basis centres (subset of numerator)
        let n_centers = self.n_centers.unwrap_or(n_num).min(n_num);
        let centers: Array2<f64> = if n_centers == n_num {
            numerator.to_owned()
        } else {
            // Evenly-spaced subsample (deterministic, no randomness needed here)
            let step = n_num / n_centers;
            let mut c = Array2::<f64>::zeros((n_centers, n_feat));
            for (ci, ni) in (0..n_num).step_by(step.max(1)).take(n_centers).enumerate() {
                c.row_mut(ci).assign(&numerator.row(ni));
            }
            c
        };

        // Kernel matrices: phi_num (n_num × n_centers), phi_den (n_den × n_centers)
        let phi_num = rbf_kernel_matrix(numerator, centers.view(), sigma);
        let phi_den = rbf_kernel_matrix(denominator, centers.view(), sigma);

        // Initialise alpha uniformly
        let mut alpha = Array1::<f64>::from_elem(n_centers, 1.0 / n_centers as f64);

        let lr = self.learning_rate;
        let lambda = self.lambda;

        // Gradient ascent on KL objective:
        //   J(alpha) = E_p[log w(x)] - log E_q[w(x)]
        // with constraint that w >= 0 and is normalised on q.
        for _iter in 0..self.max_iter {
            // w_num[i] = phi_num[i,:] . alpha
            let w_num: Vec<f64> = (0..n_num)
                .map(|i| {
                    (0..n_centers)
                        .map(|j| phi_num[[i, j]] * alpha[j])
                        .sum::<f64>()
                        .max(1e-15)
                })
                .collect();

            // w_den[i] = phi_den[i,:] . alpha
            let w_den: Vec<f64> = (0..n_den)
                .map(|i| {
                    (0..n_centers)
                        .map(|j| phi_den[[i, j]] * alpha[j])
                        .sum::<f64>()
                        .max(1e-15)
                })
                .collect();

            let z_den: f64 = w_den.iter().sum::<f64>() / n_den as f64;
            if z_den < 1e-300 {
                break;
            }

            // Gradient w.r.t. alpha
            let mut grad = Array1::<f64>::zeros(n_centers);
            // Positive part: E_p[phi / w]
            for i in 0..n_num {
                for j in 0..n_centers {
                    grad[j] += phi_num[[i, j]] / w_num[i];
                }
            }
            for j in 0..n_centers {
                grad[j] /= n_num as f64;
            }
            // Negative part: E_q[phi] / z_den
            for i in 0..n_den {
                for j in 0..n_centers {
                    grad[j] -= phi_den[[i, j]] / (n_den as f64 * z_den);
                }
            }
            // L2 regularisation gradient
            for j in 0..n_centers {
                grad[j] -= lambda * alpha[j];
            }

            // Update + project non-negative
            for j in 0..n_centers {
                alpha[j] = (alpha[j] + lr * grad[j]).max(0.0);
            }

            // Re-normalise so that mean w on denominator = 1
            let z: f64 = (0..n_den)
                .map(|i| {
                    (0..n_centers)
                        .map(|j| phi_den[[i, j]] * alpha[j])
                        .sum::<f64>()
                        .max(0.0)
                })
                .sum::<f64>()
                / n_den as f64;
            if z > 1e-15 {
                for j in 0..n_centers {
                    alpha[j] /= z;
                }
            }
        }

        self.alpha = Some(alpha);
        self.centers = Some(centers);
        Ok(())
    }

    fn predict(&self, query: ArrayView2<f64>) -> Result<Vec<f64>> {
        let alpha = self.alpha.as_ref().ok_or_else(|| {
            ClusteringError::InvalidState("Call fit() before predict()".to_string())
        })?;
        let centers = self.centers.as_ref().ok_or_else(|| {
            ClusteringError::InvalidState("Call fit() before predict()".to_string())
        })?;

        let phi = rbf_kernel_matrix(query, centers.view(), self.resolved_sigma);
        let n_query = query.shape()[0];
        let n_centers = centers.shape()[0];

        let mut weights = Vec::with_capacity(n_query);
        for i in 0..n_query {
            let w: f64 = (0..n_centers)
                .map(|j| phi[[i, j]] * alpha[j])
                .sum::<f64>()
                .max(0.0);
            weights.push(w);
        }
        Ok(weights)
    }
}

// ─── RuLSIF ──────────────────────────────────────────────────────────────────

/// Relative Unconstrained Least-Squares Importance Fitting (RuLSIF).
///
/// Minimises a least-squares criterion with L2 regularisation to estimate
/// the *relative* density ratio:
///
///   r_alpha(x) = p(x) / (alpha * p(x) + (1 - alpha) * q(x))
///
/// Setting `alpha = 0` recovers plain uLSIF.
///
/// Reference: Yamada et al. (2013), "Relative Density-Ratio Estimation for
/// Robust Distribution Comparison", Neural Computation.
#[derive(Debug, Clone)]
pub struct RuLSIF {
    /// Relative mixture parameter alpha ∈ [0, 1).  0 = uLSIF.
    pub alpha: f64,
    /// RBF bandwidth.  `None` = median heuristic.
    pub sigma: Option<f64>,
    /// L2 regularisation.
    pub lambda: f64,
    /// Number of RBF centres.  `None` = use all denominator samples.
    pub n_centers: Option<usize>,

    // fitted state
    theta: Option<Array1<f64>>,
    centers: Option<Array2<f64>>,
    resolved_sigma: f64,
}

impl RuLSIF {
    /// Create a new RuLSIF estimator.
    pub fn new(alpha: f64, lambda: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0 - 1e-6),
            sigma: None,
            lambda,
            n_centers: None,
            theta: None,
            centers: None,
            resolved_sigma: 1.0,
        }
    }

    /// Builder: set bandwidth.
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = Some(sigma);
        self
    }

    /// Builder: set number of RBF centres.
    pub fn with_n_centers(mut self, n: usize) -> Self {
        self.n_centers = Some(n);
        self
    }
}

impl Default for RuLSIF {
    fn default() -> Self {
        Self::new(0.1, 1e-3)
    }
}

impl DensityRatioEstimator for RuLSIF {
    fn fit(&mut self, numerator: ArrayView2<f64>, denominator: ArrayView2<f64>) -> Result<()> {
        let n_num = numerator.shape()[0];
        let n_den = denominator.shape()[0];
        let n_feat = numerator.shape()[1];

        if n_feat != denominator.shape()[1] {
            return Err(ClusteringError::InvalidInput(
                "Feature dimension mismatch".to_string(),
            ));
        }
        if n_num == 0 || n_den == 0 {
            return Err(ClusteringError::InvalidInput(
                "Empty sample arrays".to_string(),
            ));
        }

        // Resolve bandwidth
        let sigma = self.sigma.unwrap_or_else(|| {
            let s1 = median_bandwidth(numerator);
            let s2 = median_bandwidth(denominator);
            (s1 + s2) / 2.0
        });
        self.resolved_sigma = sigma;

        // Select centres from denominator (or subsample)
        let n_centers = self.n_centers.unwrap_or(n_den).min(n_den);
        let centers: Array2<f64> = if n_centers == n_den {
            denominator.to_owned()
        } else {
            let step = (n_den / n_centers).max(1);
            let mut c = Array2::<f64>::zeros((n_centers, n_feat));
            for (ci, di) in (0..n_den).step_by(step).take(n_centers).enumerate() {
                c.row_mut(ci).assign(&denominator.row(di));
            }
            c
        };

        // Kernel matrices
        let phi_num = rbf_kernel_matrix(numerator, centers.view(), sigma); // n_num × n_centers
        let phi_den = rbf_kernel_matrix(denominator, centers.view(), sigma); // n_den × n_centers

        let m = n_centers;

        // hat_H = (alpha/n_num) * Phi_num^T Phi_num + ((1-alpha)/n_den) * Phi_den^T Phi_den + lambda * I
        let mut h2 = Array2::<f64>::zeros((m, m));
        for i in 0..n_num {
            for j in 0..m {
                for k in j..m {
                    let v = phi_num[[i, j]] * phi_num[[i, k]];
                    h2[[j, k]] += v;
                    if k != j {
                        h2[[k, j]] += v;
                    }
                }
            }
        }
        let mut h3 = Array2::<f64>::zeros((m, m));
        for i in 0..n_den {
            for j in 0..m {
                for k in j..m {
                    let v = phi_den[[i, j]] * phi_den[[i, k]];
                    h3[[j, k]] += v;
                    if k != j {
                        h3[[k, j]] += v;
                    }
                }
            }
        }
        let mut hat_h = Array2::<f64>::zeros((m, m));
        for j in 0..m {
            for k in 0..m {
                hat_h[[j, k]] = self.alpha * h2[[j, k]] / n_num as f64
                    + (1.0 - self.alpha) * h3[[j, k]] / n_den as f64;
            }
            hat_h[[j, j]] += self.lambda;
        }

        // h_vec = (1/n_num) * sum_i phi_num[i,:]
        let mut h_vec = Array1::<f64>::zeros(m);
        for i in 0..n_num {
            for j in 0..m {
                h_vec[j] += phi_num[[i, j]];
            }
        }
        for j in 0..m {
            h_vec[j] /= n_num as f64;
        }

        // Solve hat_h * theta = h_vec
        let theta = solve_pd_system(&hat_h, &h_vec)?;
        // Project non-negative
        let theta = theta.mapv(|v: f64| v.max(0.0));

        self.theta = Some(theta);
        self.centers = Some(centers);
        Ok(())
    }

    fn predict(&self, query: ArrayView2<f64>) -> Result<Vec<f64>> {
        let theta = self.theta.as_ref().ok_or_else(|| {
            ClusteringError::InvalidState("Call fit() before predict()".to_string())
        })?;
        let centers = self.centers.as_ref().ok_or_else(|| {
            ClusteringError::InvalidState("Call fit() before predict()".to_string())
        })?;

        let phi = rbf_kernel_matrix(query, centers.view(), self.resolved_sigma);
        let n_query = query.shape()[0];
        let n_centers = centers.shape()[0];

        let mut weights = Vec::with_capacity(n_query);
        for i in 0..n_query {
            let w: f64 = (0..n_centers)
                .map(|j| phi[[i, j]] * theta[j])
                .sum::<f64>()
                .max(0.0);
            weights.push(w);
        }
        Ok(weights)
    }
}

// ─── Convenience API ─────────────────────────────────────────────────────────

/// Method selector for [`importance_weights`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportanceMethod {
    /// KL Importance Estimation Procedure.
    Kliep,
    /// Relative Unconstrained Least-Squares Importance Fitting.
    RuLSIF,
    /// Plug-in KDE ratio.
    KullbackLeibler,
}

/// Compute importance weights w(x_i) ≈ p_target(x_i) / p_source(x_i) for
/// each source sample x_i.
///
/// # Arguments
///
/// * `source_samples` - Samples from the source (denominator) distribution.
/// * `target_samples` - Samples from the target (numerator) distribution.
/// * `method`         - Which estimator to use.
///
/// # Returns
///
/// A vector of non-negative importance weights, one per source sample.
pub fn importance_weights(
    source_samples: ArrayView2<f64>,
    target_samples: ArrayView2<f64>,
    method: ImportanceMethod,
) -> Result<Vec<f64>> {
    match method {
        ImportanceMethod::Kliep => {
            let mut est = KLIEP::new();
            est.fit(target_samples, source_samples)?;
            est.predict(source_samples)
        }
        ImportanceMethod::RuLSIF => {
            let mut est = RuLSIF::default();
            est.fit(target_samples, source_samples)?;
            est.predict(source_samples)
        }
        ImportanceMethod::KullbackLeibler => {
            let mut est = KullbackLeibler::new(None);
            est.fit(target_samples, source_samples)?;
            est.predict(source_samples)
        }
    }
}

/// Direct KLIEP density ratio estimation.
///
/// # Arguments
///
/// * `numerator_samples`   - Samples from p.
/// * `denominator_samples` - Samples from q.
/// * `kernel_centers`      - RBF centres (if `None`, uses numerator samples).
/// * `lambda`              - L2 regularisation.
///
/// # Returns
///
/// Importance weight w(x_i) for each numerator sample.
pub fn density_ratio_kliep(
    numerator_samples: ArrayView2<f64>,
    denominator_samples: ArrayView2<f64>,
    kernel_centers: Option<ArrayView2<f64>>,
    lambda: f64,
) -> Result<Vec<f64>> {
    let n_centers = kernel_centers.map(|c: ArrayView2<f64>| c.shape()[0]);
    let mut est = KLIEP {
        sigma: None,
        lambda,
        max_iter: 1000,
        learning_rate: 1e-3,
        n_centers,
        alpha: None,
        centers: None,
        resolved_sigma: 1.0,
    };
    est.fit(numerator_samples, denominator_samples)?;
    est.predict(numerator_samples)
}

// ─── Two-sample testing / covariate shift ────────────────────────────────────

/// Compute the covariate shift score between source and target distributions.
///
/// The score is defined as the mean log-importance-weight estimated by KLIEP:
///
///   score = (1/n) Σ_i log(w(x_i) + ε)
///
/// A score near zero indicates negligible shift; larger positive values indicate
/// that the source and target distributions differ substantially.
///
/// # Arguments
///
/// * `source` - Source domain samples.
/// * `target` - Target domain samples.
///
/// # Returns
///
/// A non-negative scalar; larger values indicate greater covariate shift.
pub fn covariate_shift_score(source: ArrayView2<f64>, target: ArrayView2<f64>) -> Result<f64> {
    let weights = importance_weights(source, target, ImportanceMethod::Kliep)?;
    if weights.is_empty() {
        return Ok(0.0);
    }
    let eps = 1e-15_f64;
    let mean_log_w =
        weights.iter().map(|&w: &f64| (w + eps).ln()).sum::<f64>() / weights.len() as f64;
    Ok(mean_log_w.abs())
}

/// Two-sample test statistic based on the RuLSIF density ratio.
///
/// Returns the Pearson divergence PE(p ‖ α p + (1−α) q) estimated from finite
/// samples, which is zero iff p = q and positive otherwise.
pub fn two_sample_test_statistic(
    samples_p: ArrayView2<f64>,
    samples_q: ArrayView2<f64>,
    alpha: f64,
    lambda: f64,
) -> Result<f64> {
    let mut est = RuLSIF::new(alpha, lambda);
    est.fit(samples_p, samples_q)?;
    let w_p = est.predict(samples_p)?;
    let w_q = est.predict(samples_q)?;

    // PE divergence = 0.5 * E_q[w^2] - E_p[w] + 0.5
    let mean_w2_q = w_q.iter().map(|&w| w * w).sum::<f64>() / w_q.len() as f64;
    let mean_w_p = w_p.iter().sum::<f64>() / w_p.len() as f64;
    let pe = 0.5 * mean_w2_q - mean_w_p + 0.5_f64;
    Ok(pe.max(0.0_f64))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_gaussian_samples(mean: f64, std: f64, n: usize, seed: u64) -> Array2<f64> {
        // Deterministic Box-Muller to avoid rand dependency
        let mut out = vec![0.0_f64; n];
        let mut state = seed;
        let lcg_next = |s: u64| {
            s.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)
        };
        for i in 0..n {
            state = lcg_next(state);
            let u1 = (state as f64) / u64::MAX as f64 * (1.0 - 1e-10) + 1e-10;
            state = lcg_next(state);
            let u2 = (state as f64) / u64::MAX as f64;
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            out[i] = mean + std * z;
        }
        let data: Vec<f64> = out;
        Array2::from_shape_vec((n, 1), data).expect("shape")
    }

    #[test]
    fn test_kliep_same_distribution() {
        let src = make_gaussian_samples(0.0, 1.0, 40, 1);
        let tgt = make_gaussian_samples(0.0, 1.0, 40, 2);
        let w = importance_weights(src.view(), tgt.view(), ImportanceMethod::Kliep)
            .expect("kliep failed");
        assert_eq!(w.len(), 40);
        // All weights should be non-negative
        assert!(w.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_kliep_shifted_distribution() {
        let src = make_gaussian_samples(0.0, 1.0, 40, 3);
        let tgt = make_gaussian_samples(3.0, 1.0, 40, 4);
        let w = importance_weights(src.view(), tgt.view(), ImportanceMethod::Kliep)
            .expect("kliep shifted");
        assert!(w.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_rulsif_basic() {
        let src = make_gaussian_samples(0.0, 1.0, 30, 5);
        let tgt = make_gaussian_samples(1.0, 1.0, 30, 6);
        let w =
            importance_weights(src.view(), tgt.view(), ImportanceMethod::RuLSIF).expect("rulsif");
        assert_eq!(w.len(), 30);
        assert!(w.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_kde_ratio_basic() {
        let src = make_gaussian_samples(0.0, 1.0, 30, 7);
        let tgt = make_gaussian_samples(0.0, 1.0, 30, 8);
        let w = importance_weights(src.view(), tgt.view(), ImportanceMethod::KullbackLeibler)
            .expect("kde ratio");
        assert_eq!(w.len(), 30);
        assert!(w.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_covariate_shift_score_zero_shift() {
        let src = make_gaussian_samples(0.0, 1.0, 30, 9);
        let tgt = make_gaussian_samples(0.0, 1.0, 30, 10);
        let score = covariate_shift_score(src.view(), tgt.view()).expect("score");
        // Should be a finite non-negative number
        assert!(score.is_finite());
        assert!(score >= 0.0);
    }

    #[test]
    fn test_two_sample_test_nonneg() {
        let p = make_gaussian_samples(0.0, 1.0, 30, 11);
        let q = make_gaussian_samples(2.0, 1.0, 30, 12);
        let stat = two_sample_test_statistic(p.view(), q.view(), 0.1, 1e-3).expect("test stat");
        assert!(stat >= 0.0);
        assert!(stat.is_finite());
    }

    #[test]
    fn test_density_ratio_kliep_fn() {
        let num = make_gaussian_samples(1.0, 0.5, 20, 13);
        let den = make_gaussian_samples(0.0, 1.0, 20, 14);
        let w = density_ratio_kliep(num.view(), den.view(), None, 1e-3).expect("kliep fn");
        assert_eq!(w.len(), 20);
    }

    #[test]
    fn test_feature_mismatch_error() {
        let a = Array2::from_shape_vec((5, 2), vec![0.0; 10]).expect("a");
        let b = Array2::from_shape_vec((5, 3), vec![0.0; 15]).expect("b");
        let mut est = KLIEP::new();
        let result = est.fit(a.view(), b.view());
        assert!(result.is_err());
    }
}
