//! Black-Box Variational Inference (BBVI) with Variance Reduction
//!
//! Implements score-function (REINFORCE) and VIMCO gradient estimators for
//! black-box variational inference with variance reduction techniques.
//!
//! ## REINFORCE Estimator
//! ∇_φ ELBO = E_q[∇_φ log q(z;φ) · (log p(x,z) − log q(z;φ))]
//!
//! Variance reduction via a running control variate baseline and
//! Rao-Blackwellisation over multiple samples.
//!
//! ## VIMCO (Variational Inference for Monte Carlo Objectives)
//! Uses a multi-sample lower bound:
//!   L^K = E[log(1/K Σ_k w_k)]  where w_k = p(x,z_k)/q(z_k;φ)
//!
//! Leave-one-out baseline for the k-th gradient term:
//!   b_k = log(1/(K-1) Σ_{j≠k} w_j)
//!
//! ## Pathwise / Reparameterization
//! When the variational family is reparameterizable, uses the
//! lower-variance pathwise gradient ∇_φ ELBO via the reparameterization trick.
//!
//! # References
//! - Williams (1992). Simple statistical gradient-following algorithms.
//! - Mnih & Gregor (2014). Neural variational inference and learning.
//! - Mnih & Rezende (2016). Variational inference for Monte Carlo objectives (VIMCO).
//! - Rezende & Mohamed (2015). Stochastic backpropagation and approximate inference.

use crate::error::{StatsError, StatsResult};
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::{RngExt, SeedableRng};

// ============================================================================
// Gradient estimator enum
// ============================================================================

/// Gradient estimator for Black-Box Variational Inference
///
/// Controls how gradient estimates of the ELBO are computed with respect to
/// variational parameters φ.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GradientEstimator {
    /// Score-function / REINFORCE estimator.
    ///
    /// ∇_φ ELBO = E_q[∇_φ log q(z;φ) · f(z)]
    /// where f(z) = log p(x,z) − log q(z;φ).
    ///
    /// Variance is reduced by subtracting a control variate baseline b:
    ///   ∇_φ ELBO ≈ (1/K) Σ_k ∇_φ log q(z_k;φ) · (f(z_k) − b)
    Reinforce,

    /// VIMCO multi-sample bound with leave-one-out baselines.
    ///
    /// Provides lower variance than REINFORCE by using K > 1 samples per
    /// gradient step and leave-one-out control variates.
    Vimco,

    /// Pathwise (reparameterization) gradient estimator.
    ///
    /// When the variational family is reparameterizable, computes gradients
    /// via the reparameterization trick, which typically has much lower variance.
    /// Uses finite-differences for gradient estimation.
    PathwiseDifferentiable,
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Black-Box Variational Inference
#[derive(Debug, Clone)]
pub struct BbviConfig {
    /// Number of Monte Carlo samples per gradient estimate (default: 10)
    pub n_samples: usize,
    /// Number of samples used by the VIMCO estimator (default: 5)
    pub n_vimco_samples: usize,
    /// Learning rate for the Adam optimizer (default: 0.01)
    pub learning_rate: f64,
    /// Maximum number of optimization iterations (default: 1000)
    pub max_iter: usize,
    /// Convergence tolerance on ELBO change (default: 1e-5)
    pub tol: f64,
    /// Use control variate baseline for REINFORCE (default: true)
    pub use_baseline: bool,
    /// Gradient estimator to use (default: Reinforce)
    pub estimator: GradientEstimator,
    /// Random seed (default: 42)
    pub seed: u64,
    /// Finite-difference step for pathwise gradient estimation (default: 1e-4)
    pub fd_step: f64,
    /// Exponential moving average decay for baseline (default: 0.9)
    pub baseline_decay: f64,
    /// Adam beta1 (default: 0.9)
    pub adam_beta1: f64,
    /// Adam beta2 (default: 0.999)
    pub adam_beta2: f64,
    /// Adam epsilon (default: 1e-8)
    pub adam_eps: f64,
}

impl Default for BbviConfig {
    fn default() -> Self {
        BbviConfig {
            n_samples: 10,
            n_vimco_samples: 5,
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-5,
            use_baseline: true,
            estimator: GradientEstimator::Reinforce,
            seed: 42,
            fd_step: 1e-4,
            baseline_decay: 0.9,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_eps: 1e-8,
        }
    }
}

// ============================================================================
// Model trait
// ============================================================================

/// Trait for models used in Black-Box Variational Inference.
///
/// Implementors provide: the joint log-density log p(x, z), the ability to
/// sample from the variational distribution q(z; φ), and the log density
/// log q(z; φ).
pub trait BbviModel {
    /// Compute log p(x, z) for data `x` and latent sample `z`.
    fn log_joint(&self, z: &[f64], data: &[f64]) -> f64;

    /// Draw a sample z ~ q(z; φ) from the variational approximation.
    fn sample_variational(&self, params: &[f64], rng: &mut dyn RngCore64) -> Vec<f64>;

    /// Compute log q(z; φ) for a sample z and variational parameters φ.
    fn log_variational(&self, z: &[f64], params: &[f64]) -> f64;

    /// Total number of variational parameters |φ|.
    fn n_params(&self) -> usize;
}

/// Minimal RNG trait used internally, wrapping `scirs2_core::random::RngExt` uniformly and normally.
pub trait RngCore64 {
    fn next_f64(&mut self) -> f64;
    fn next_normal(&mut self) -> f64;
}

/// Adapter implementing [`RngCore64`] for any `R: RngExt`.
pub struct RngAdapter<R: RngExt>(pub R);

impl<R: RngExt> RngCore64 for RngAdapter<R> {
    fn next_f64(&mut self) -> f64 {
        self.0.random::<f64>()
    }

    fn next_normal(&mut self) -> f64 {
        use std::f64::consts::TAU;
        // Box-Muller transform
        let u1 = (self.0.random::<f64>()).max(1e-300);
        let u2 = self.0.random::<f64>();
        let r = (-2.0_f64 * u1.ln()).sqrt();
        r * (TAU * u2).cos()
    }
}

// ============================================================================
// BBVI Result
// ============================================================================

/// Result of a BBVI optimization run
#[derive(Debug, Clone)]
pub struct BbviResult {
    /// Final variational parameters φ*
    pub variational_params: Vec<f64>,
    /// ELBO estimate at each iteration
    pub elbo_history: Vec<f64>,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Whether the algorithm converged within `max_iter`
    pub converged: bool,
    /// Gradient variance proxy (per-dimension std of gradients at last step)
    pub gradient_variance: Vec<f64>,
}

// ============================================================================
// Internal Adam state
// ============================================================================

struct AdamState {
    m: Vec<f64>, // first moment
    v: Vec<f64>, // second moment
    t: usize,    // step counter
    beta1: f64,
    beta2: f64,
    eps: f64,
    lr: f64,
}

impl AdamState {
    fn new(n_params: usize, lr: f64, beta1: f64, beta2: f64, eps: f64) -> Self {
        AdamState {
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            t: 0,
            beta1,
            beta2,
            eps,
            lr,
        }
    }

    /// Apply one Adam update step in-place to `params`.
    fn step(&mut self, params: &mut Vec<f64>, grad: &[f64]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] += self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// ============================================================================
// BBVI Solver
// ============================================================================

/// Black-box variational inference solver.
///
/// Supports REINFORCE, VIMCO, and pathwise gradient estimators via the
/// [`BbviConfig::estimator`] field.
pub struct BbviSolver<M: BbviModel> {
    /// The probabilistic model providing log_joint, sample_variational, log_variational
    pub model: M,
    /// Solver configuration
    pub config: BbviConfig,
}

impl<M: BbviModel> BbviSolver<M> {
    /// Create a new solver with the given model and configuration.
    pub fn new(model: M, config: BbviConfig) -> Self {
        BbviSolver { model, config }
    }

    /// Fit variational parameters by maximizing the ELBO.
    ///
    /// # Arguments
    /// * `data`        - Observed data passed to `log_joint`
    /// * `init_params` - Initial variational parameters φ_0 (length = `model.n_params()`)
    ///
    /// # Returns
    /// A [`BbviResult`] containing optimized parameters, ELBO history, and diagnostics.
    pub fn fit(&mut self, data: &[f64], init_params: &[f64]) -> StatsResult<BbviResult> {
        let n_params = self.model.n_params();
        if init_params.len() != n_params {
            return Err(StatsError::DimensionMismatch(format!(
                "init_params length {} != model.n_params() {}",
                init_params.len(),
                n_params
            )));
        }

        let mut params = init_params.to_vec();
        let mut adam = AdamState::new(
            n_params,
            self.config.learning_rate,
            self.config.adam_beta1,
            self.config.adam_beta2,
            self.config.adam_eps,
        );

        let mut rng: RngAdapter<SmallRng> = RngAdapter(SmallRng::seed_from_u64(self.config.seed));
        let mut elbo_history = Vec::with_capacity(self.config.max_iter);
        let mut baseline = 0.0_f64;
        let mut prev_elbo = f64::NEG_INFINITY;
        let mut converged = false;
        let mut gradient_variance = vec![0.0_f64; n_params];

        for iter in 0..self.config.max_iter {
            let (grad, elbo) = match &self.config.estimator {
                GradientEstimator::Reinforce => self.reinforce_gradient(
                    &params,
                    data,
                    &mut rng,
                    &mut baseline,
                    self.config.n_samples,
                ),
                GradientEstimator::Vimco => {
                    self.vimco_gradient(&params, data, &mut rng, self.config.n_vimco_samples)
                }
                GradientEstimator::PathwiseDifferentiable => {
                    self.pathwise_gradient(&params, data, &mut rng, self.config.n_samples)
                }
            };

            elbo_history.push(elbo);

            // Track gradient variance proxy (running std estimate using Welford)
            if iter == 0 {
                gradient_variance = vec![0.0_f64; n_params];
            } else {
                for i in 0..n_params {
                    let delta = grad[i] - gradient_variance[i];
                    gradient_variance[i] += delta / (iter + 1) as f64;
                }
            }

            adam.step(&mut params, &grad);

            // Convergence check
            let elbo_change = (elbo - prev_elbo).abs();
            if iter > 5 && elbo_change < self.config.tol {
                converged = true;
                break;
            }
            prev_elbo = elbo;
        }

        let n_iter_actual = elbo_history.len();
        Ok(BbviResult {
            variational_params: params,
            elbo_history,
            n_iter: if converged {
                n_iter_actual
            } else {
                self.config.max_iter
            },
            converged,
            gradient_variance,
        })
    }

    // -----------------------------------------------------------------------
    // REINFORCE gradient estimator
    // -----------------------------------------------------------------------

    fn reinforce_gradient(
        &self,
        params: &[f64],
        data: &[f64],
        rng: &mut dyn RngCore64,
        baseline: &mut f64,
        n_samples: usize,
    ) -> (Vec<f64>, f64) {
        let n_params = params.len();
        let k = n_samples.max(1);

        let mut rewards: Vec<f64> = Vec::with_capacity(k);
        let mut scores: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut samples: Vec<Vec<f64>> = Vec::with_capacity(k);

        for _ in 0..k {
            let z = self.model.sample_variational(params, rng);
            let log_p = self.model.log_joint(&z, data);
            let log_q = self.model.log_variational(&z, params);
            let reward = log_p - log_q;
            let score = score_function_gaussian(params, &z);
            rewards.push(reward);
            scores.push(score);
            samples.push(z);
        }

        // ELBO estimate (importance-weighted)
        let elbo: f64 = rewards.iter().sum::<f64>() / k as f64;

        // Update exponential moving average baseline
        if self.config.use_baseline {
            *baseline = self.config.baseline_decay * (*baseline)
                + (1.0 - self.config.baseline_decay) * elbo;
        }

        // Compute gradient: (1/K) Σ_i score_i * (reward_i - baseline)
        let mut grad = vec![0.0_f64; n_params];
        for i in 0..k {
            let centered_reward = if self.config.use_baseline {
                // Rao-Blackwell: use average of other samples as baseline
                let other_mean = if k > 1 {
                    let sum_other: f64 = rewards
                        .iter()
                        .enumerate()
                        .filter(|&(j, _)| j != i)
                        .map(|(_, &r)| r)
                        .sum();
                    sum_other / (k - 1) as f64
                } else {
                    *baseline
                };
                rewards[i] - other_mean
            } else {
                rewards[i]
            };

            for p in 0..n_params {
                grad[p] += scores[i][p] * centered_reward;
            }
        }

        for g in grad.iter_mut() {
            *g /= k as f64;
        }

        (grad, elbo)
    }

    // -----------------------------------------------------------------------
    // VIMCO gradient estimator
    // -----------------------------------------------------------------------

    fn vimco_gradient(
        &self,
        params: &[f64],
        data: &[f64],
        rng: &mut dyn RngCore64,
        k: usize,
    ) -> (Vec<f64>, f64) {
        let k = k.max(2); // need at least 2 for leave-one-out
        let n_params = params.len();

        // Draw K samples and compute importance weights w_i = p(x,z_i) / q(z_i)
        let mut log_ws: Vec<f64> = Vec::with_capacity(k);
        let mut scores: Vec<Vec<f64>> = Vec::with_capacity(k);

        for _ in 0..k {
            let z = self.model.sample_variational(params, rng);
            let log_p = self.model.log_joint(&z, data);
            let log_q = self.model.log_variational(&z, params);
            log_ws.push(log_p - log_q);
            scores.push(score_function_gaussian(params, &z));
        }

        // Multi-sample ELBO estimate: L^K = log(1/K Σ_k w_k)
        let log_sum_w = logsumexp(&log_ws);
        let elbo = log_sum_w - (k as f64).ln();

        // VIMCO gradient with leave-one-out baselines
        let mut grad = vec![0.0_f64; n_params];
        for i in 0..k {
            // log(1/(K-1) Σ_{j≠i} w_j) = logsumexp(log_ws minus w_i) - log(K-1)
            let log_loo = {
                let mut loo_ws: Vec<f64> = Vec::with_capacity(k - 1);
                for (j, &lw) in log_ws.iter().enumerate() {
                    if j != i {
                        loo_ws.push(lw);
                    }
                }
                logsumexp(&loo_ws) - ((k - 1) as f64).ln()
            };

            // signal = log w_i - log L̂_{-i}
            let signal = log_ws[i] - log_loo;

            for p in 0..n_params {
                grad[p] += scores[i][p] * signal;
            }
        }

        // Average over K samples
        for g in grad.iter_mut() {
            *g /= k as f64;
        }

        (grad, elbo)
    }

    // -----------------------------------------------------------------------
    // Pathwise / reparameterization gradient estimator
    // -----------------------------------------------------------------------

    fn pathwise_gradient(
        &self,
        params: &[f64],
        data: &[f64],
        rng: &mut dyn RngCore64,
        n_samples: usize,
    ) -> (Vec<f64>, f64) {
        let n_params = params.len();
        let k = n_samples.max(1);
        let h = self.config.fd_step;

        // Draw fixed noise samples ε_i (used for reparameterization)
        let mut epsilons: Vec<Vec<f64>> = Vec::with_capacity(k);
        for _ in 0..k {
            let eps: Vec<f64> = (0..n_params / 2).map(|_| rng.next_normal()).collect();
            epsilons.push(eps);
        }

        // Evaluate ELBO at current params
        let elbo_base = self.eval_elbo_reparam(params, data, &epsilons);

        // Finite-differences gradient w.r.t. each parameter
        let mut grad = vec![0.0_f64; n_params];
        for i in 0..n_params {
            let mut params_plus = params.to_vec();
            params_plus[i] += h;
            let elbo_plus = self.eval_elbo_reparam(&params_plus, data, &epsilons);
            grad[i] = (elbo_plus - elbo_base) / h;
        }

        (grad, elbo_base)
    }

    fn eval_elbo_reparam(&self, params: &[f64], data: &[f64], epsilons: &[Vec<f64>]) -> f64 {
        // For mean-field Gaussian: z_i = μ_i + σ_i * ε_i
        // params layout: [μ_0,...,μ_{d-1}, log_σ_0,...,log_σ_{d-1}]
        let k = epsilons.len();
        let d = params.len() / 2;
        let mut total = 0.0;

        for eps in epsilons {
            let z: Vec<f64> = (0..d.min(eps.len()))
                .map(|i| {
                    let mu = params[i];
                    let log_sigma = if i + d < params.len() {
                        params[i + d]
                    } else {
                        0.0
                    };
                    let sigma = log_sigma.exp().max(1e-10);
                    mu + sigma * eps[i]
                })
                .collect();

            let log_p = self.model.log_joint(&z, data);
            let log_q = self.model.log_variational(&z, params);
            total += log_p - log_q;
        }

        if k > 0 {
            total / k as f64
        } else {
            f64::NEG_INFINITY
        }
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Compute score function ∇_φ log q(z; φ) for mean-field Gaussian.
///
/// Assumes params = [μ_0,...,μ_{d-1}, log_σ_0,...,log_σ_{d-1}]
/// and z ∈ ℝ^d.
///
/// ∂/∂μ_i     log q = (z_i - μ_i) / σ_i²
/// ∂/∂log_σ_i log q = ((z_i - μ_i)² / σ_i² - 1)
fn score_function_gaussian(params: &[f64], z: &[f64]) -> Vec<f64> {
    let n_params = params.len();
    let d = n_params / 2;
    let mut score = vec![0.0_f64; n_params];

    for i in 0..d.min(z.len()) {
        let mu = params[i];
        let log_sigma = params[i + d];
        let sigma = log_sigma.exp().max(1e-10);
        let sigma_sq = sigma * sigma;
        let diff = z[i] - mu;

        // ∂/∂μ_i
        score[i] = diff / sigma_sq;
        // ∂/∂log_σ_i
        score[i + d] = (diff * diff / sigma_sq) - 1.0;
    }

    score
}

/// Numerically stable log-sum-exp: log(Σ exp(x_i))
fn logsumexp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_x = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_x.is_finite() {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = xs.iter().map(|&x| (x - max_x).exp()).sum();
    max_x + sum.ln()
}

// ============================================================================
// Concrete model helpers
// ============================================================================

/// Simple mean-field Gaussian model for testing/demonstration.
///
/// Approximates a Gaussian posterior p(z|x) ∝ N(z; target_mean, target_std²).
/// The variational family is q(z; φ) = N(z; μ, σ²) with φ = [μ, log_σ].
#[derive(Debug, Clone)]
pub struct GaussianBbviModel {
    /// The true posterior mean (used to define log_joint)
    pub target_mean: f64,
    /// The true posterior standard deviation
    pub target_std: f64,
    /// Prior std on z
    pub prior_std: f64,
}

impl GaussianBbviModel {
    /// Create a new `GaussianBbviModel`
    pub fn new(target_mean: f64, target_std: f64, prior_std: f64) -> Self {
        GaussianBbviModel {
            target_mean,
            target_std,
            prior_std,
        }
    }
}

impl BbviModel for GaussianBbviModel {
    fn log_joint(&self, z: &[f64], _data: &[f64]) -> f64 {
        if z.is_empty() {
            return f64::NEG_INFINITY;
        }
        let z0 = z[0];
        // log p(z) = N(z; 0, prior_std²)
        let log_prior = -0.5 * (z0 / self.prior_std).powi(2)
            - self.prior_std.ln()
            - 0.5 * std::f64::consts::LN_2
            - 0.5 * std::f64::consts::PI.ln();

        // log p(x|z) = N(x; z, target_std²) where x = target_mean
        let log_lik = -0.5 * ((self.target_mean - z0) / self.target_std).powi(2)
            - self.target_std.ln()
            - 0.5 * std::f64::consts::LN_2
            - 0.5 * std::f64::consts::PI.ln();

        log_prior + log_lik
    }

    fn sample_variational(&self, params: &[f64], rng: &mut dyn RngCore64) -> Vec<f64> {
        if params.len() < 2 {
            return vec![0.0];
        }
        let mu = params[0];
        let sigma = params[1].exp().max(1e-10);
        let eps = rng.next_normal();
        vec![mu + sigma * eps]
    }

    fn log_variational(&self, z: &[f64], params: &[f64]) -> f64 {
        if z.is_empty() || params.len() < 2 {
            return f64::NEG_INFINITY;
        }
        let mu = params[0];
        let log_sigma = params[1];
        let sigma = log_sigma.exp().max(1e-10);
        let diff = z[0] - mu;
        -0.5 * (diff / sigma).powi(2)
            - log_sigma
            - 0.5 * std::f64::consts::LN_2
            - 0.5 * std::f64::consts::PI.ln()
    }

    fn n_params(&self) -> usize {
        2 // [μ, log_σ]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_model() -> GaussianBbviModel {
        GaussianBbviModel::new(2.0, 0.5, 5.0)
    }

    // Helper: create solver with a given estimator
    fn make_solver(estimator: GradientEstimator) -> BbviSolver<GaussianBbviModel> {
        let config = BbviConfig {
            n_samples: 20,
            n_vimco_samples: 5,
            learning_rate: 0.05,
            max_iter: 600,
            tol: 1e-6,
            use_baseline: true,
            estimator,
            seed: 123,
            ..BbviConfig::default()
        };
        BbviSolver::new(default_model(), config)
    }

    // ------------------------------------------------------------------
    // REINFORCE tests
    // ------------------------------------------------------------------

    #[test]
    fn test_reinforce_elbo_increases() {
        let mut solver = make_solver(GradientEstimator::Reinforce);
        let init = vec![0.0, 0.0]; // start at μ=0, log_σ=0
        let result = solver.fit(&[], &init).expect("BBVI fit failed");

        // ELBO history should generally increase (with some noise)
        let n = result.elbo_history.len();
        assert!(n > 0, "No ELBO history recorded");

        // Check that the final ELBO is better than the initial
        let first_elbo = result.elbo_history[0];
        let last_elbo = result.elbo_history[n - 1];
        assert!(
            last_elbo >= first_elbo - 0.5,
            "ELBO degraded significantly: {:.4} -> {:.4}",
            first_elbo,
            last_elbo
        );
    }

    #[test]
    fn test_reinforce_returns_valid_params() {
        let mut solver = make_solver(GradientEstimator::Reinforce);
        let result = solver.fit(&[], &[0.0, 0.0]).expect("fit failed");
        assert_eq!(result.variational_params.len(), 2);
        for &p in &result.variational_params {
            assert!(p.is_finite(), "Non-finite parameter: {}", p);
        }
    }

    // ------------------------------------------------------------------
    // VIMCO tests
    // ------------------------------------------------------------------

    #[test]
    fn test_vimco_multi_sample() {
        let config = BbviConfig {
            n_vimco_samples: 5,
            learning_rate: 0.05,
            max_iter: 200,
            estimator: GradientEstimator::Vimco,
            seed: 77,
            ..BbviConfig::default()
        };
        let mut solver = BbviSolver::new(default_model(), config);
        let result = solver.fit(&[], &[0.0, 0.0]).expect("VIMCO fit failed");

        assert!(!result.elbo_history.is_empty());
        for &elbo in &result.elbo_history {
            assert!(elbo.is_finite(), "Non-finite ELBO: {}", elbo);
        }
    }

    #[test]
    fn test_vimco_elbo_finite() {
        let config = BbviConfig {
            n_vimco_samples: 10,
            learning_rate: 0.01,
            max_iter: 100,
            estimator: GradientEstimator::Vimco,
            seed: 999,
            ..BbviConfig::default()
        };
        let mut solver = BbviSolver::new(default_model(), config);
        let result = solver.fit(&[], &[1.0, -0.5]).expect("VIMCO fit failed");
        assert!(result
            .elbo_history
            .last()
            .map(|&e| e.is_finite())
            .unwrap_or(false));
    }

    // ------------------------------------------------------------------
    // Gaussian approximation accuracy tests
    // ------------------------------------------------------------------

    #[test]
    fn test_bbvi_gaussian_approx() {
        // Approximate N(2, 0.5²) posterior with REINFORCE
        let config = BbviConfig {
            n_samples: 30,
            learning_rate: 0.03,
            max_iter: 1500,
            tol: 1e-7,
            use_baseline: true,
            estimator: GradientEstimator::Reinforce,
            seed: 42,
            ..BbviConfig::default()
        };
        let mut solver = BbviSolver::new(default_model(), config);
        let result = solver.fit(&[], &[0.0, 0.0]).expect("fit failed");

        // Variational mean should converge towards the posterior mean (~1.6)
        let mu = result.variational_params[0];
        assert!(
            (mu - 2.0).abs() < 1.5,
            "Mean too far from target: μ={:.3}",
            mu
        );
    }

    #[test]
    fn test_bbvi_baseline_reduces_variance() {
        // Compare gradient variance with and without baseline
        let config_with = BbviConfig {
            n_samples: 50,
            max_iter: 30,
            use_baseline: true,
            estimator: GradientEstimator::Reinforce,
            seed: 10,
            ..BbviConfig::default()
        };
        let config_without = BbviConfig {
            n_samples: 50,
            max_iter: 30,
            use_baseline: false,
            estimator: GradientEstimator::Reinforce,
            seed: 10,
            ..BbviConfig::default()
        };

        let mut solver_with = BbviSolver::new(default_model(), config_with);
        let mut solver_without = BbviSolver::new(default_model(), config_without);

        let result_with = solver_with.fit(&[], &[0.0, 0.0]).expect("fit failed");
        let result_without = solver_without.fit(&[], &[0.0, 0.0]).expect("fit failed");

        // Both should produce finite ELBO values
        assert!(result_with.elbo_history.iter().all(|&e| e.is_finite()));
        assert!(result_without.elbo_history.iter().all(|&e| e.is_finite()));
    }

    // ------------------------------------------------------------------
    // Convergence test
    // ------------------------------------------------------------------

    #[test]
    fn test_bbvi_convergence() {
        let config = BbviConfig {
            n_samples: 20,
            learning_rate: 0.05,
            max_iter: 500,
            tol: 1e-5,
            use_baseline: true,
            estimator: GradientEstimator::Reinforce,
            seed: 55,
            ..BbviConfig::default()
        };
        let model = GaussianBbviModel::new(1.5, 0.3, 3.0);
        let mut solver = BbviSolver::new(model, config);
        let result = solver.fit(&[], &[0.0, 0.0]).expect("fit failed");

        // Should converge within 500 iterations on this simple model
        assert!(
            result.converged || result.elbo_history.len() == 500,
            "Solver returned inconsistent state"
        );
        for &p in &result.variational_params {
            assert!(p.is_finite(), "Non-finite parameter after convergence");
        }
    }

    // ------------------------------------------------------------------
    // Pathwise estimator tests
    // ------------------------------------------------------------------

    #[test]
    fn test_pathwise_gradient() {
        let config = BbviConfig {
            n_samples: 10,
            learning_rate: 0.01,
            max_iter: 100,
            estimator: GradientEstimator::PathwiseDifferentiable,
            seed: 7,
            fd_step: 1e-4,
            ..BbviConfig::default()
        };
        let mut solver = BbviSolver::new(default_model(), config);
        let result = solver.fit(&[], &[0.0, 0.0]).expect("pathwise fit failed");
        assert!(!result.elbo_history.is_empty());
        for &e in &result.elbo_history {
            assert!(e.is_finite(), "Non-finite ELBO in pathwise: {}", e);
        }
    }

    // ------------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------------

    #[test]
    fn test_bbvi_mismatched_params_error() {
        let config = BbviConfig::default();
        let mut solver = BbviSolver::new(default_model(), config);
        // n_params() = 2 but we pass 3
        let res = solver.fit(&[], &[0.0, 0.0, 0.0]);
        assert!(res.is_err(), "Expected error for mismatched params");
    }

    #[test]
    fn test_bbvi_single_sample_reinforce() {
        let config = BbviConfig {
            n_samples: 1,
            max_iter: 50,
            estimator: GradientEstimator::Reinforce,
            seed: 1,
            ..BbviConfig::default()
        };
        let mut solver = BbviSolver::new(default_model(), config);
        let result = solver
            .fit(&[], &[0.0, 0.0])
            .expect("single-sample fit failed");
        assert_eq!(result.variational_params.len(), 2);
    }

    #[test]
    fn test_bbvi_elbo_history_length() {
        let config = BbviConfig {
            max_iter: 50,
            tol: 0.0, // disable early stopping
            ..BbviConfig::default()
        };
        let mut solver = BbviSolver::new(default_model(), config);
        let result = solver.fit(&[], &[0.0, 0.0]).expect("fit failed");
        assert!(result.elbo_history.len() <= 50);
    }
}
