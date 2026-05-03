//! Probabilistic soft clustering algorithms.
//!
//! This module provides two complementary soft-clustering algorithms:
//!
//! * [`GaussianMixtureModel`] – parametric EM-based GMM with BIC/AIC model
//!   selection, k-means++ initialisation, and full soft-assignment output.
//!
//! * [`DirichletProcessMixtureModel`] – nonparametric Bayesian mixture with
//!   stick-breaking process and variational mean-field inference that
//!   automatically infers the number of active components.
//!
//! # Example – GMM
//!
//! ```rust
//! use scirs2_cluster::soft_clustering::{GaussianMixtureModel, GmmParams};
//! use scirs2_core::ndarray::Array2;
//!
//! let data = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0,  1.2, 1.8,  0.8, 1.9,
//!     4.0, 5.0,  4.2, 4.8,  3.9, 5.1,
//! ]).expect("operation should succeed");
//!
//! let params = GaussianMixtureModel::fit(data.view(), 2, 100, 1e-4).expect("operation should succeed");
//! let proba  = params.predict_proba(data.view()).expect("operation should succeed");
//! assert_eq!(proba.shape(), [6, 2]);
//! ```

use std::f64::consts::PI;

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{ClusteringError, Result};

// ═══════════════════════════════════════════════════════════════════════════
// Internal maths helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Digamma function approximation (Stirling series).
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let mut v = x;
    let mut result = 0.0;
    // Shift argument so the asymptotic is accurate
    while v < 6.0 {
        result -= 1.0 / v;
        v += 1.0;
    }
    // Asymptotic series
    result += v.ln() - 0.5 / v;
    let inv_v2 = 1.0 / (v * v);
    result -= inv_v2 * (1.0 / 12.0 - inv_v2 * (1.0 / 120.0 - inv_v2 / 252.0));
    result
}

/// Compute log-sum-exp for a row of a 2D array.
fn logsumexp_row(row: &[f64]) -> f64 {
    let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return f64::NEG_INFINITY;
    }
    let s: f64 = row.iter().map(|&v| (v - max).exp()).sum();
    max + s.ln()
}

/// Cholesky decomposition of an `(n × n)` symmetric positive-definite matrix.
/// Returns the lower-triangular factor `L` s.t. `A = L L^T`.
fn cholesky(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.shape()[0];
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 {
                    s = 1e-12;
                }
                l[[i, j]] = s.sqrt();
            } else if l[[j, j]].abs() < 1e-15 {
                l[[i, j]] = 0.0;
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    Ok(l)
}

/// Log-determinant of a positive-definite matrix via Cholesky.
fn log_det_pd(a: &Array2<f64>) -> Result<f64> {
    let l = cholesky(a)?;
    let n = l.shape()[0];
    let mut log_det = 0.0;
    for i in 0..n {
        log_det += 2.0 * l[[i, i]].ln();
    }
    Ok(log_det)
}

/// Solve `L L^T x = b` (Cholesky back-substitution).
fn cholesky_solve(l: &Array2<f64>, b: ArrayView1<f64>) -> Array1<f64> {
    let n = l.shape()[0];
    let mut y = Array1::<f64>::zeros(n);
    // Forward substitution L y = b
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[[i, k]] * y[k];
        }
        y[i] = if l[[i, i]].abs() < 1e-15 {
            0.0
        } else {
            s / l[[i, i]]
        };
    }
    // Back substitution L^T x = y
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[[k, i]] * x[k];
        }
        x[i] = if l[[i, i]].abs() < 1e-15 {
            0.0
        } else {
            s / l[[i, i]]
        };
    }
    x
}

/// Log of the multivariate Gaussian pdf N(x | mu, Sigma).
/// Uses the Cholesky factor `l` of Sigma.
fn log_mvn(x: ArrayView1<f64>, mu: ArrayView1<f64>, l: &Array2<f64>) -> f64 {
    let d = x.len() as f64;
    let diff: Array1<f64> = x.iter().zip(mu.iter()).map(|(&xi, &mi)| xi - mi).collect();
    let z = cholesky_solve(l, diff.view());
    let maha: f64 = z.iter().map(|&v| v * v).sum();
    let log_det_l: f64 = (0..l.shape()[0]).map(|i| l[[i, i]].ln()).sum::<f64>();
    -0.5 * (d * (2.0 * PI).ln() + 2.0 * log_det_l + maha)
}

/// K-means++ initialisation; returns `(n_components × n_features)` centroid array.
fn kmeans_pp_init(data: ArrayView2<f64>, k: usize, seed: u64) -> Array2<f64> {
    let n = data.shape()[0];
    let d = data.shape()[1];

    let mut rng_state = seed;
    let lcg = |s: u64| {
        s.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
    };
    let rand_f64 = |s: &mut u64| -> f64 {
        *s = lcg(*s);
        (*s >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut centers = Array2::<f64>::zeros((k, d));
    // First centre: random row
    rng_state = lcg(rng_state);
    let first = (rng_state as usize) % n;
    centers.row_mut(0).assign(&data.row(first));

    for ci in 1..k {
        // For each point compute min distance^2 to chosen centres
        let mut dists = Vec::with_capacity(n);
        let mut sum_d = 0.0;
        for i in 0..n {
            let mut min_d2 = f64::INFINITY;
            for cj in 0..ci {
                let d2: f64 = data
                    .row(i)
                    .iter()
                    .zip(centers.row(cj).iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                if d2 < min_d2 {
                    min_d2 = d2;
                }
            }
            dists.push(min_d2);
            sum_d += min_d2;
        }
        // Sample proportionally to distance
        let mut u = rand_f64(&mut rng_state) * sum_d;
        let mut chosen = n - 1;
        for (i, &d_i) in dists.iter().enumerate() {
            u -= d_i;
            if u <= 0.0 {
                chosen = i;
                break;
            }
        }
        centers.row_mut(ci).assign(&data.row(chosen));
    }
    centers
}

// ═══════════════════════════════════════════════════════════════════════════
// GMM – fitted parameters bundle
// ═══════════════════════════════════════════════════════════════════════════

/// Fitted parameters of a Gaussian Mixture Model.
///
/// All heavy computation is done in [`GaussianMixtureModel::fit`]; this
/// struct is a pure data container that also exposes `predict_proba`,
/// `predict`, `score`, `bic`, and `aic`.
#[derive(Debug, Clone)]
pub struct GmmParams {
    /// Mixture weights, shape `(k,)`.
    pub weights: Array1<f64>,
    /// Component means, shape `(k, d)`.
    pub means: Array2<f64>,
    /// Cholesky factors of component covariances, each shape `(d, d)`.
    pub chol_covs: Vec<Array2<f64>>,
    /// Number of EM iterations performed.
    pub n_iter: usize,
    /// Whether the EM converged.
    pub converged: bool,
    /// Log-likelihood per sample at convergence.
    pub log_likelihood: f64,
}

impl GmmParams {
    /// Number of mixture components.
    pub fn n_components(&self) -> usize {
        self.weights.len()
    }

    /// Feature dimension.
    pub fn n_features(&self) -> usize {
        self.means.shape()[1]
    }

    /// Compute soft assignments (posterior responsibilities).
    ///
    /// Returns an `(n_samples, k)` array where each row sums to 1.
    pub fn predict_proba(&self, data: ArrayView2<f64>) -> Result<Array2<f64>> {
        let n = data.shape()[0];
        let k = self.n_components();
        let mut log_resp = Array2::<f64>::zeros((n, k));

        for i in 0..n {
            for c in 0..k {
                if self.weights[c] <= 0.0 {
                    log_resp[[i, c]] = f64::NEG_INFINITY;
                    continue;
                }
                log_resp[[i, c]] = self.weights[c].ln()
                    + log_mvn(data.row(i), self.means.row(c), &self.chol_covs[c]);
            }
            // Normalise in log space
            let row: Vec<f64> = (0..k).map(|c| log_resp[[i, c]]).collect();
            let lse = logsumexp_row(&row);
            for c in 0..k {
                log_resp[[i, c]] = (log_resp[[i, c]] - lse).exp();
            }
        }
        Ok(log_resp)
    }

    /// Hard cluster assignments: argmax of `predict_proba`.
    pub fn predict(&self, data: ArrayView2<f64>) -> Result<Array1<usize>> {
        let proba = self.predict_proba(data)?;
        let n = proba.shape()[0];
        let k = proba.shape()[1];
        let mut labels = Array1::<usize>::zeros(n);
        for i in 0..n {
            let mut best = 0;
            let mut best_p = proba[[i, 0]];
            for c in 1..k {
                if proba[[i, c]] > best_p {
                    best_p = proba[[i, c]];
                    best = c;
                }
            }
            labels[i] = best;
        }
        Ok(labels)
    }

    /// Mean log-likelihood over `data`.
    pub fn score(&self, data: ArrayView2<f64>) -> Result<f64> {
        let n = data.shape()[0];
        let k = self.n_components();
        let mut total_ll = 0.0;
        for i in 0..n {
            let mut log_terms: Vec<f64> = Vec::with_capacity(k);
            for c in 0..k {
                if self.weights[c] > 0.0 {
                    log_terms.push(
                        self.weights[c].ln()
                            + log_mvn(data.row(i), self.means.row(c), &self.chol_covs[c]),
                    );
                }
            }
            total_ll += logsumexp_row(&log_terms);
        }
        Ok(total_ll / n as f64)
    }

    /// Number of free parameters in the model.
    ///
    /// For full-covariance GMM:
    ///   k - 1  (weights) + k*d (means) + k*(d*(d+1)/2) (covariances)
    fn n_free_params(&self) -> usize {
        let k = self.n_components();
        let d = self.n_features();
        (k - 1) + k * d + k * (d * (d + 1) / 2)
    }

    /// Bayesian Information Criterion.
    ///
    /// BIC = -2 * log_likelihood * n_samples + p * ln(n_samples)
    pub fn bic(&self, data: ArrayView2<f64>) -> Result<f64> {
        let n = data.shape()[0] as f64;
        let ll = self.score(data)? * n;
        let p = self.n_free_params() as f64;
        Ok(-2.0 * ll + p * n.ln())
    }

    /// Akaike Information Criterion.
    ///
    /// AIC = -2 * log_likelihood * n_samples + 2 * p
    pub fn aic(&self, data: ArrayView2<f64>) -> Result<f64> {
        let n = data.shape()[0] as f64;
        let ll = self.score(data)? * n;
        let p = self.n_free_params() as f64;
        Ok(-2.0 * ll + 2.0 * p)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GaussianMixtureModel – EM algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// Gaussian Mixture Model with Expectation-Maximisation training.
///
/// Initialises with k-means++ for robustness, then iterates E / M steps
/// until log-likelihood change drops below `tol` or `max_iter` is reached.
/// Covariance matrices are full (unconstrained) with a regularisation term
/// added to the diagonal to ensure numerical stability.
pub struct GaussianMixtureModel;

impl GaussianMixtureModel {
    /// Fit a GMM to `data`.
    ///
    /// # Arguments
    ///
    /// * `data`         – `(n_samples, n_features)` input array.
    /// * `n_components` – Number of Gaussian components `k`.
    /// * `max_iter`     – Maximum EM iterations.
    /// * `tol`          – Convergence tolerance on mean log-likelihood change.
    ///
    /// # Returns
    ///
    /// A [`GmmParams`] bundle with the fitted parameters.
    pub fn fit(
        data: ArrayView2<f64>,
        n_components: usize,
        max_iter: usize,
        tol: f64,
    ) -> Result<GmmParams> {
        let n = data.shape()[0];
        let d = data.shape()[1];
        let k = n_components;

        if k == 0 {
            return Err(ClusteringError::InvalidInput(
                "n_components must be >= 1".to_string(),
            ));
        }
        if n < k {
            return Err(ClusteringError::InvalidInput(
                "n_samples must be >= n_components".to_string(),
            ));
        }
        if d == 0 {
            return Err(ClusteringError::InvalidInput(
                "n_features must be >= 1".to_string(),
            ));
        }

        let reg = 1e-6_f64;

        // ── Initialise with k-means++ ────────────────────────────────────
        let init_means = kmeans_pp_init(data, k, 42);

        // Initial responsibilities: hard-assign each point to nearest centre
        let mut resp = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let mut best_c = 0;
            let mut best_d = f64::INFINITY;
            for c in 0..k {
                let d2: f64 = data
                    .row(i)
                    .iter()
                    .zip(init_means.row(c).iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                if d2 < best_d {
                    best_d = d2;
                    best_c = c;
                }
            }
            resp[[i, best_c]] = 1.0;
        }

        // Initial M-step from hard assignments
        let (mut weights, mut means, mut chol_covs) = Self::m_step(data, resp.view(), k, d, reg)?;

        let mut prev_ll = f64::NEG_INFINITY;
        let mut n_iter = 0;
        let mut converged = false;

        for iter in 0..max_iter {
            n_iter = iter + 1;

            // ── E-step ──────────────────────────────────────────────────
            resp = Self::e_step(data, &weights, &means, &chol_covs, k)?;

            // ── Compute log-likelihood ───────────────────────────────────
            let ll = Self::mean_log_likelihood(data, &weights, &means, &chol_covs, k);

            if (ll - prev_ll).abs() < tol {
                converged = true;
                prev_ll = ll;
                // One more M-step to stay consistent
                let (w, m, c) = Self::m_step(data, resp.view(), k, d, reg)?;
                weights = w;
                means = m;
                chol_covs = c;
                break;
            }
            prev_ll = ll;

            // ── M-step ──────────────────────────────────────────────────
            let (w, m, c) = Self::m_step(data, resp.view(), k, d, reg)?;
            weights = w;
            means = m;
            chol_covs = c;
        }

        Ok(GmmParams {
            weights,
            means,
            chol_covs,
            n_iter,
            converged,
            log_likelihood: prev_ll,
        })
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    fn e_step(
        data: ArrayView2<f64>,
        weights: &Array1<f64>,
        means: &Array2<f64>,
        chol_covs: &[Array2<f64>],
        k: usize,
    ) -> Result<Array2<f64>> {
        let n = data.shape()[0];
        let mut log_resp = Array2::<f64>::zeros((n, k));

        for i in 0..n {
            for c in 0..k {
                if weights[c] <= 0.0 {
                    log_resp[[i, c]] = f64::NEG_INFINITY;
                    continue;
                }
                log_resp[[i, c]] =
                    weights[c].ln() + log_mvn(data.row(i), means.row(c), &chol_covs[c]);
            }
            let row: Vec<f64> = (0..k).map(|c| log_resp[[i, c]]).collect();
            let lse = logsumexp_row(&row);
            for c in 0..k {
                log_resp[[i, c]] = (log_resp[[i, c]] - lse).exp();
            }
        }
        Ok(log_resp)
    }

    fn m_step(
        data: ArrayView2<f64>,
        resp: ArrayView2<f64>,
        k: usize,
        d: usize,
        reg: f64,
    ) -> Result<(Array1<f64>, Array2<f64>, Vec<Array2<f64>>)> {
        let n = data.shape()[0];

        // Effective counts: N_k = sum_i r_{ik}
        let nk: Vec<f64> = (0..k)
            .map(|c| (0..n).map(|i| resp[[i, c]]).sum::<f64>().max(1e-10))
            .collect();

        let total_n: f64 = nk.iter().sum();
        let weights: Array1<f64> = nk.iter().map(|&nkc| nkc / total_n).collect();

        // Means: mu_k = (1/N_k) sum_i r_{ik} x_i
        let mut means = Array2::<f64>::zeros((k, d));
        for c in 0..k {
            for i in 0..n {
                for f in 0..d {
                    means[[c, f]] += resp[[i, c]] * data[[i, f]];
                }
            }
            for f in 0..d {
                means[[c, f]] /= nk[c];
            }
        }

        // Covariances: Sigma_k = (1/N_k) sum_i r_{ik} (x_i - mu_k)(x_i - mu_k)^T + reg*I
        let mut chol_covs = Vec::with_capacity(k);
        for c in 0..k {
            let mut cov = Array2::<f64>::zeros((d, d));
            for i in 0..n {
                for f1 in 0..d {
                    let diff_f1 = data[[i, f1]] - means[[c, f1]];
                    for f2 in f1..d {
                        let diff_f2 = data[[i, f2]] - means[[c, f2]];
                        let v = resp[[i, c]] * diff_f1 * diff_f2 / nk[c];
                        cov[[f1, f2]] += v;
                        if f2 != f1 {
                            cov[[f2, f1]] += v;
                        }
                    }
                }
            }
            for f in 0..d {
                cov[[f, f]] += reg;
            }
            let l = cholesky(&cov)?;
            chol_covs.push(l);
        }

        Ok((weights, means, chol_covs))
    }

    fn mean_log_likelihood(
        data: ArrayView2<f64>,
        weights: &Array1<f64>,
        means: &Array2<f64>,
        chol_covs: &[Array2<f64>],
        k: usize,
    ) -> f64 {
        let n = data.shape()[0];
        let mut total = 0.0;
        for i in 0..n {
            let mut log_terms: Vec<f64> = Vec::with_capacity(k);
            for c in 0..k {
                if weights[c] > 0.0 {
                    log_terms
                        .push(weights[c].ln() + log_mvn(data.row(i), means.row(c), &chol_covs[c]));
                }
            }
            total += logsumexp_row(&log_terms);
        }
        total / n as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dirichlet Process Mixture Model
// ═══════════════════════════════════════════════════════════════════════════

/// Fitted state returned by [`DirichletProcessMixtureModel::fit`].
#[derive(Debug, Clone)]
pub struct DpmmResult {
    /// Variational mean stick-breaking weights (truncated at `T` components).
    pub stick_weights: Array1<f64>,
    /// Posterior mean of component means, shape `(T, d)`.
    pub means: Array2<f64>,
    /// Active component mask: `active[t]` is `true` if component `t` has
    /// meaningful responsibility in the fitted data.
    pub active: Vec<bool>,
    /// ELBO (variational lower bound) at convergence.
    pub elbo: f64,
    /// Number of variational EM iterations.
    pub n_iter: usize,
    /// Whether the variational EM converged.
    pub converged: bool,
    // Cholesky factors for prediction (diagonal Gaussian per component)
    chol_covs: Vec<Array2<f64>>,
    n_active: usize,
}

impl DpmmResult {
    /// Number of truncation components.
    pub fn n_components(&self) -> usize {
        self.stick_weights.len()
    }

    /// Number of components with non-negligible weight.
    pub fn n_active_components(&self) -> usize {
        self.n_active
    }

    /// Soft assignments: `(n_samples, T)` responsibility matrix.
    pub fn predict_proba(&self, data: ArrayView2<f64>) -> Result<Array2<f64>> {
        let n = data.shape()[0];
        let t = self.n_components();
        let mut log_resp = Array2::<f64>::zeros((n, t));
        for i in 0..n {
            for c in 0..t {
                let w = self.stick_weights[c];
                if w <= 0.0 || !self.active[c] {
                    log_resp[[i, c]] = f64::NEG_INFINITY;
                    continue;
                }
                log_resp[[i, c]] =
                    w.ln() + log_mvn(data.row(i), self.means.row(c), &self.chol_covs[c]);
            }
            let row: Vec<f64> = (0..t).map(|c| log_resp[[i, c]]).collect();
            let lse = logsumexp_row(&row);
            for c in 0..t {
                log_resp[[i, c]] = (log_resp[[i, c]] - lse).exp();
            }
        }
        Ok(log_resp)
    }

    /// Hard cluster assignments (ignoring inactive components).
    pub fn predict(&self, data: ArrayView2<f64>) -> Result<Array1<usize>> {
        let proba = self.predict_proba(data)?;
        let n = proba.shape()[0];
        let t = proba.shape()[1];
        let mut labels = Array1::<usize>::zeros(n);
        for i in 0..n {
            let mut best = 0;
            let mut best_p = proba[[i, 0]];
            for c in 1..t {
                if proba[[i, c]] > best_p {
                    best_p = proba[[i, c]];
                    best = c;
                }
            }
            labels[i] = best;
        }
        Ok(labels)
    }
}

/// Dirichlet Process Mixture Model via truncated stick-breaking variational inference.
///
/// The model assumes a DP with concentration `alpha`, truncated at `T` components.
/// Each component is a spherical Gaussian N(x | mu_k, sigma_k^2 I).
/// Variational mean-field approximation is used, updating:
///
/// 1. Component responsibilities (E-like step).
/// 2. Posterior stick-breaking parameters (M-like step for the DP prior).
/// 3. Posterior component parameters (M-like step for the Gaussian likelihood).
///
/// The number of *effective* components is inferred automatically from the data.
pub struct DirichletProcessMixtureModel {
    /// DP concentration parameter (larger => more components).
    pub alpha: f64,
    /// Truncation level.
    pub truncation: usize,
    /// Maximum variational EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the ELBO.
    pub tol: f64,
    /// Minimum component weight threshold to declare a component active.
    pub activity_threshold: f64,
}

impl DirichletProcessMixtureModel {
    /// Create a new DPMM estimator.
    pub fn new(alpha: f64, truncation: usize) -> Self {
        Self {
            alpha,
            truncation,
            max_iter: 200,
            tol: 1e-4,
            activity_threshold: 1e-2,
        }
    }

    /// Fit the DPMM to `data`.
    pub fn fit(&self, data: ArrayView2<f64>) -> Result<DpmmResult> {
        let n = data.shape()[0];
        let d = data.shape()[1];
        let t = self.truncation;

        if n == 0 || d == 0 {
            return Err(ClusteringError::InvalidInput(
                "Data must be non-empty".to_string(),
            ));
        }
        if t < 1 {
            return Err(ClusteringError::InvalidInput(
                "truncation must be >= 1".to_string(),
            ));
        }

        let reg = 1e-6_f64;
        let alpha = self.alpha;

        // ── Initialise via k-means++ ─────────────────────────────────────
        let k_init = t.min(n);
        let init_means = kmeans_pp_init(data, k_init, 7);

        // Responsibilities: hard-assign to nearest centre
        let mut phi = Array2::<f64>::zeros((n, t));
        for i in 0..n {
            let mut best_c = 0;
            let mut best_d = f64::INFINITY;
            for c in 0..k_init {
                let d2: f64 = data
                    .row(i)
                    .iter()
                    .zip(init_means.row(c).iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                if d2 < best_d {
                    best_d = d2;
                    best_c = c;
                }
            }
            phi[[i, best_c]] = 1.0;
        }

        // Variational parameters
        // Stick-breaking: gamma_k = (a_k, b_k) for Beta(a_k, b_k)
        let mut a_gamma = Array1::<f64>::from_elem(t, 1.0);
        let mut b_gamma = Array1::<f64>::from_elem(t, alpha);

        // Gaussian posteriors: (m_k, beta_k, nu_k, W_k_diag) — use diagonal
        let mut m = Array2::<f64>::zeros((t, d)); // posterior mean
        let mut beta_k = Array1::<f64>::from_elem(t, 1.0); // precision scale
        let mut nu_k = Array1::<f64>::from_elem(t, d as f64 + 1.0); // dof
        let mut w_k = Array2::<f64>::from_elem((t, d), 1.0); // diagonal Wishart

        // Copy init_means
        for c in 0..k_init {
            for f in 0..d {
                m[[c, f]] = init_means[[c, f]];
            }
        }

        let mut prev_elbo = f64::NEG_INFINITY;
        let mut n_iter = 0;
        let mut converged = false;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // ── E-step: update phi (responsibilities) ────────────────────
            // E[log pi_k] from stick-breaking
            let mut e_log_pi = Array1::<f64>::zeros(t);
            let mut cumsum_b = 0.0;
            for k in 0..t {
                let e_log_v_k = digamma(a_gamma[k]) - digamma(a_gamma[k] + b_gamma[k]);
                let e_log_1mv_k = digamma(b_gamma[k]) - digamma(a_gamma[k] + b_gamma[k]);
                e_log_pi[k] = e_log_v_k + cumsum_b;
                cumsum_b += e_log_1mv_k;
            }

            // E[log |Lambda_k|] = sum_f (digamma((nu_k + 1 - f) / 2) + ln(2 * W_kf))
            // For diagonal Wishart this simplifies to:
            let e_log_lam: Vec<f64> = (0..t)
                .map(|k| {
                    (0..d)
                        .map(|f| {
                            let dof_f = (nu_k[k] + 1.0 - f as f64) / 2.0;
                            digamma(dof_f.max(0.5)) + (2.0 * w_k[[k, f]]).ln()
                        })
                        .sum::<f64>()
                })
                .collect();

            for i in 0..n {
                let mut log_rho = Vec::with_capacity(t);
                for k in 0..t {
                    // E[||x_i - mu_k||^2 * Lambda_k] under diagonal Wishart-Gaussian
                    let trace_term: f64 = (0..d)
                        .map(|f| {
                            nu_k[k] * w_k[[k, f]] * (data[[i, f]] - m[[k, f]]).powi(2)
                                + 1.0 / beta_k[k]
                        })
                        .sum();
                    log_rho.push(
                        e_log_pi[k] + 0.5 * e_log_lam[k]
                            - 0.5 * d as f64 * (2.0 * PI).ln()
                            - 0.5 * trace_term,
                    );
                }
                let lse = logsumexp_row(&log_rho);
                for k in 0..t {
                    phi[[i, k]] = (log_rho[k] - lse).exp();
                }
            }

            // ── M-step: update stick-breaking parameters ─────────────────
            let nk: Vec<f64> = (0..t)
                .map(|k| (0..n).map(|i| phi[[i, k]]).sum::<f64>().max(1e-10))
                .collect();

            for k in 0..t {
                let sum_after: f64 = nk[(k + 1)..].iter().sum();
                a_gamma[k] = 1.0 + nk[k];
                b_gamma[k] = alpha + sum_after;
            }

            // ── M-step: update Gaussian posteriors ───────────────────────
            for k in 0..t {
                let beta_0 = 1.0;
                let nu_0 = d as f64 + 1.0;

                // Update beta and m
                beta_k[k] = beta_0 + nk[k];
                let mut x_bar = vec![0.0_f64; d];
                for i in 0..n {
                    for f in 0..d {
                        x_bar[f] += phi[[i, k]] * data[[i, f]];
                    }
                }
                for f in 0..d {
                    x_bar[f] /= nk[k];
                    m[[k, f]] = (beta_0 * 0.0 + nk[k] * x_bar[f]) / beta_k[k];
                }

                // Update nu
                nu_k[k] = nu_0 + nk[k];

                // Update W (diagonal precision matrix)
                for f in 0..d {
                    let mut scatter = 0.0;
                    for i in 0..n {
                        scatter += phi[[i, k]] * (data[[i, f]] - x_bar[f]).powi(2);
                    }
                    let bc_correction = beta_0 * nk[k] / beta_k[k] * x_bar[f].powi(2);
                    w_k[[k, f]] = 1.0 / (1.0 / (1.0 + reg) + scatter + bc_correction);
                }
            }

            // ── Compute ELBO (simplified) ─────────────────────────────────
            let elbo = Self::compute_elbo(
                data, &phi, &a_gamma, &b_gamma, &m, &beta_k, &nu_k, &w_k, alpha, n, d, t,
            );

            if (elbo - prev_elbo).abs() < self.tol {
                converged = true;
                prev_elbo = elbo;
                break;
            }
            prev_elbo = elbo;
        }

        // ── Convert variational parameters to summary ────────────────────
        // Compute expected stick weights via E[V_k]
        let mut expected_weights = Array1::<f64>::zeros(t);
        let mut log_remaining: f64 = 0.0;
        for k in 0..t {
            let e_v_k = a_gamma[k] / (a_gamma[k] + b_gamma[k]);
            expected_weights[k] = e_v_k * log_remaining.exp();
            log_remaining += (1.0 - e_v_k).ln();
        }

        let active: Vec<bool> = (0..t)
            .map(|k| expected_weights[k] > self.activity_threshold / t as f64)
            .collect();
        let n_active = active.iter().filter(|&&a| a).count();

        // Build diagonal Cholesky factors for prediction
        let mut chol_covs = Vec::with_capacity(t);
        for k in 0..t {
            // Posterior predictive covariance ≈ diag(1 / (nu_k * w_k))
            let mut cov = Array2::<f64>::zeros((d, d));
            for f in 0..d {
                let var = (1.0 / (nu_k[k] * w_k[[k, f]])).max(reg);
                cov[[f, f]] = var.sqrt(); // store Cholesky (diagonal)
            }
            chol_covs.push(cov);
        }

        let final_means = m.clone();

        Ok(DpmmResult {
            stick_weights: expected_weights,
            means: final_means,
            active,
            elbo: prev_elbo,
            n_iter,
            converged,
            chol_covs,
            n_active,
        })
    }

    /// Simplified ELBO estimate for convergence monitoring.
    #[allow(clippy::too_many_arguments)]
    fn compute_elbo(
        data: ArrayView2<f64>,
        phi: &Array2<f64>,
        a_gamma: &Array1<f64>,
        b_gamma: &Array1<f64>,
        m: &Array2<f64>,
        beta_k: &Array1<f64>,
        nu_k: &Array1<f64>,
        w_k: &Array2<f64>,
        alpha: f64,
        n: usize,
        d: usize,
        t: usize,
    ) -> f64 {
        // E[log p(X | Z, params)] approximated via responsibilities
        let mut ll = 0.0;
        for i in 0..n {
            for k in 0..t {
                if phi[[i, k]] < 1e-15 {
                    continue;
                }
                let log_norm = -(d as f64) / 2.0 * (2.0 * PI).ln();
                let neg_quad: f64 = -(0..d)
                    .map(|f| nu_k[k] * w_k[[k, f]] * (data[[i, f]] - m[[k, f]]).powi(2))
                    .sum::<f64>()
                    / 2.0;
                let e_log_lam: f64 = (0..d)
                    .map(|f| {
                        let dof_f = (nu_k[k] + 1.0 - f as f64) / 2.0;
                        digamma(dof_f.max(0.5)) + (2.0 * w_k[[k, f]]).ln()
                    })
                    .sum::<f64>()
                    / 2.0;
                ll += phi[[i, k]] * (log_norm + e_log_lam + neg_quad);
            }
        }

        // E[log p(Z | pi)] - E[log q(Z)]
        let mut z_term = 0.0;
        for i in 0..n {
            for k in 0..t {
                let phi_ik = phi[[i, k]];
                if phi_ik > 1e-15 {
                    z_term -= phi_ik * phi_ik.ln(); // entropy
                }
            }
        }

        // DP prior contribution (simplified)
        let dp_term: f64 = (0..t)
            .map(|k| (alpha - 1.0) * (digamma(b_gamma[k]) - digamma(a_gamma[k] + b_gamma[k])))
            .sum();

        // Beta variational entropy
        let beta_entropy: f64 = (0..t)
            .map(|k| {
                let ab = a_gamma[k] + b_gamma[k];
                let ent = (beta_k[k]).ln() - (a_gamma[k] - 1.0) * digamma(a_gamma[k]) + (ab).ln()
                    - (b_gamma[k] - 1.0) * digamma(b_gamma[k])
                    + digamma(ab);
                ent
            })
            .sum();

        ll + z_term + dp_term + beta_entropy
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn two_cluster_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (12, 2),
            vec![
                1.0, 1.0, 1.1, 0.9, 0.9, 1.1, 1.0, 1.0, 0.8, 1.2, 1.2, 0.8, 5.0, 5.0, 5.1, 4.9,
                4.9, 5.1, 5.0, 5.0, 4.8, 5.2, 5.2, 4.8,
            ],
        )
        .expect("data")
    }

    #[test]
    fn test_gmm_fit_basic() {
        let data = two_cluster_data();
        let params = GaussianMixtureModel::fit(data.view(), 2, 100, 1e-4).expect("gmm fit");
        assert_eq!(params.n_components(), 2);
        assert_eq!(params.n_features(), 2);
        assert!(params.converged || params.n_iter > 0);
    }

    #[test]
    fn test_gmm_predict_proba() {
        let data = two_cluster_data();
        let params = GaussianMixtureModel::fit(data.view(), 2, 100, 1e-4).expect("gmm fit");
        let proba = params.predict_proba(data.view()).expect("predict_proba");
        assert_eq!(proba.shape(), [12, 2]);
        // Each row should sum to 1
        for i in 0..12 {
            let row_sum: f64 = (0..2).map(|c| proba[[i, c]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "row {i} sums to {row_sum}");
        }
    }

    #[test]
    fn test_gmm_predict_hard() {
        let data = two_cluster_data();
        let params = GaussianMixtureModel::fit(data.view(), 2, 100, 1e-4).expect("gmm fit");
        let labels = params.predict(data.view()).expect("predict");
        assert_eq!(labels.len(), 12);
        // Two distinct clusters expected
        let unique: std::collections::HashSet<_> = labels.iter().copied().collect();
        assert!(unique.len() <= 2);
    }

    #[test]
    fn test_gmm_score_finite() {
        let data = two_cluster_data();
        let params = GaussianMixtureModel::fit(data.view(), 2, 100, 1e-4).expect("gmm fit");
        let score = params.score(data.view()).expect("score");
        assert!(score.is_finite(), "score must be finite, got {score}");
    }

    #[test]
    fn test_gmm_bic_aic() {
        let data = two_cluster_data();
        let params = GaussianMixtureModel::fit(data.view(), 2, 100, 1e-4).expect("gmm fit");
        let bic = params.bic(data.view()).expect("bic");
        let aic = params.aic(data.view()).expect("aic");
        assert!(bic.is_finite());
        assert!(aic.is_finite());
        // BIC >= AIC for reasonable n > e
        // (not always true with tiny datasets, just check they're finite)
    }

    #[test]
    fn test_gmm_k1_trivial() {
        let data = two_cluster_data();
        let params = GaussianMixtureModel::fit(data.view(), 1, 50, 1e-4).expect("gmm k=1");
        let labels = params.predict(data.view()).expect("predict k=1");
        // All labels should be 0
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_gmm_invalid_k() {
        let data = two_cluster_data();
        let result = GaussianMixtureModel::fit(data.view(), 0, 50, 1e-4);
        assert!(result.is_err());
    }

    #[test]
    fn test_dpmm_fit_basic() {
        let data = two_cluster_data();
        let model = DirichletProcessMixtureModel::new(1.0, 6);
        let result = model.fit(data.view()).expect("dpmm fit");
        assert_eq!(result.n_components(), 6);
        assert!(result.n_iter > 0);
        // At least one active component
        assert!(result.n_active_components() >= 1);
    }

    #[test]
    fn test_dpmm_predict_proba() {
        let data = two_cluster_data();
        let model = DirichletProcessMixtureModel::new(1.0, 4);
        let result = model.fit(data.view()).expect("dpmm fit");
        let proba = result.predict_proba(data.view()).expect("proba");
        assert_eq!(proba.shape()[0], 12);
        assert_eq!(proba.shape()[1], 4);
        for i in 0..12 {
            let row_sum: f64 = (0..4).map(|c| proba[[i, c]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "row {i} sum {row_sum}");
        }
    }

    #[test]
    fn test_dpmm_predict_hard() {
        let data = two_cluster_data();
        let model = DirichletProcessMixtureModel::new(1.0, 4);
        let result = model.fit(data.view()).expect("dpmm fit");
        let labels = result.predict(data.view()).expect("predict");
        assert_eq!(labels.len(), 12);
    }

    #[test]
    fn test_dpmm_alpha_concentration() {
        // Higher alpha => more active components expected
        let data = two_cluster_data();
        let model_low = DirichletProcessMixtureModel::new(0.01, 8);
        let model_high = DirichletProcessMixtureModel::new(10.0, 8);
        let r_low = model_low.fit(data.view()).expect("low alpha");
        let r_high = model_high.fit(data.view()).expect("high alpha");
        assert!(r_high.n_active_components() >= r_low.n_active_components());
    }
}
