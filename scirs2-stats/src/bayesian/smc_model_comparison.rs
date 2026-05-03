//! Sequential Monte Carlo (SMC) for Bayesian Model Comparison
//!
//! Implements SMC samplers (Del Moral, Doucet & Jasra, 2006) for computing
//! marginal likelihoods and Bayes factors, enabling sequential Bayesian model
//! comparison.
//!
//! Algorithm:
//! 1. Initialize N particles from the prior.
//! 2. Temper the likelihood via an annealing schedule β_0=0 → β_T=1.
//! 3. At each temperature step:
//!    a. Re-weight particles by exp(Δβ · log_likelihood(θ)).
//!    b. Estimate ESS; if ESS < α·N, resample (stratified).
//!    c. Rejuvenate via MCMC (random-walk MH) to maintain diversity.
//! 4. Accumulate incremental normalizing constant log Z = Σ log(mean weights).
//!
//! # References
//! - Del Moral, Doucet & Jasra (2006). "Sequential Monte Carlo samplers."
//! - Chopin & Robert (2010). "Properties of nested sampling."
//! - Zhou, Johansen & Aston (2016). "Towards automatic model comparison:
//!   an adaptive sequential Monte Carlo approach."

use crate::error::{StatsError, StatsResult as Result};

// ---------------------------------------------------------------------------
// Minimal LCG
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = self.state >> 11;
        (bits as f64) * (1.0 / (1u64 << 53) as f64)
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Numerically stable log-sum-exp: log(Σ exp(x_i)).
pub fn logsumexp(log_w: &[f64]) -> f64 {
    if log_w.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_w = log_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_w.is_finite() {
        return f64::NEG_INFINITY;
    }
    let sum_exp: f64 = log_w.iter().map(|&w| (w - max_w).exp()).sum();
    max_w + sum_exp.ln()
}

/// Effective sample size from log-weights: ESS = exp(2·lse(w) - lse(2w)).
pub fn ess(log_weights: &[f64]) -> f64 {
    if log_weights.is_empty() {
        return 0.0;
    }
    let lse_w = logsumexp(log_weights);
    let log_w2: Vec<f64> = log_weights.iter().map(|&w| 2.0 * w).collect();
    let lse_2w = logsumexp(&log_w2);
    (2.0 * lse_w - lse_2w).exp()
}

/// Normalize log-weights to sum to 1.
fn normalize_log_weights(log_w: &[f64]) -> Vec<f64> {
    let lse = logsumexp(log_w);
    log_w.iter().map(|&w| w - lse).collect()
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the SMC model comparison sampler.
#[derive(Debug, Clone)]
pub struct SmcConfig {
    /// Number of SMC particles.
    pub n_particles: usize,
    /// Resample when ESS / n_particles drops below this threshold.
    pub ess_threshold: f64,
    /// Number of MCMC rejuvenation steps per particle per temperature.
    pub mcmc_steps: usize,
    /// Maximum number of temperature increments (annealing steps).
    pub max_temperatures: usize,
    /// MCMC random-walk step size (adaptive if 0.0).
    pub step_size: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for SmcConfig {
    fn default() -> Self {
        Self {
            n_particles: 1000,
            ess_threshold: 0.5,
            mcmc_steps: 3,
            max_temperatures: 100,
            step_size: 0.1,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// SMC Particles
// ---------------------------------------------------------------------------

/// Collection of SMC particles at a given temperature.
#[derive(Debug, Clone)]
pub struct SmcParticles {
    /// Parameter vectors for each particle (n_particles × d).
    pub thetas: Vec<Vec<f64>>,
    /// Log-weights (unnormalized).
    pub log_weights: Vec<f64>,
    /// Current annealing temperature β ∈ [0, 1].
    pub temperature: f64,
}

impl SmcParticles {
    /// Number of particles.
    pub fn n_particles(&self) -> usize {
        self.thetas.len()
    }

    /// Compute normalized weights.
    pub fn normalized_weights(&self) -> Vec<f64> {
        let lse = logsumexp(&self.log_weights);
        self.log_weights.iter().map(|&w| (w - lse).exp()).collect()
    }

    /// Effective sample size.
    pub fn ess(&self) -> f64 {
        ess(&self.log_weights)
    }

    /// Weighted mean of particle parameters.
    pub fn weighted_mean(&self) -> Vec<f64> {
        if self.thetas.is_empty() {
            return vec![];
        }
        let d = self.thetas[0].len();
        let weights = self.normalized_weights();
        let mut mean = vec![0.0f64; d];
        for (theta, &w) in self.thetas.iter().zip(weights.iter()) {
            for (m, &t) in mean.iter_mut().zip(theta.iter()) {
                *m += w * t;
            }
        }
        mean
    }
}

// ---------------------------------------------------------------------------
// SMC Result
// ---------------------------------------------------------------------------

/// Result of an SMC run.
#[derive(Debug, Clone)]
pub struct SmcResult {
    /// Log marginal likelihood estimate: log p(data | model).
    pub log_marginal_likelihood: f64,
    /// Final particle collection.
    pub particles: SmcParticles,
    /// Number of temperature steps taken.
    pub n_steps: usize,
    /// Temperature schedule used.
    pub temperatures: Vec<f64>,
}

impl SmcResult {
    /// Compute the Bayes factor: p(data | M1) / p(data | M2).
    ///
    /// Returns the log Bayes factor (log BF = log Z_1 - log Z_2).
    pub fn log_bayes_factor(&self, other: &SmcResult) -> f64 {
        self.log_marginal_likelihood - other.log_marginal_likelihood
    }

    /// Bayes factor: p(data | M1) / p(data | M2) = exp(log BF).
    pub fn bayes_factor(&self, other: &SmcResult) -> f64 {
        self.log_bayes_factor(other).exp()
    }
}

// ---------------------------------------------------------------------------
// SMC Model Comparison
// ---------------------------------------------------------------------------

/// Sequential Monte Carlo sampler for Bayesian model comparison.
pub struct SmcModelComparison {
    config: SmcConfig,
}

impl SmcModelComparison {
    /// Create a new SMC sampler with the given configuration.
    pub fn new(config: SmcConfig) -> Result<Self> {
        if config.n_particles == 0 {
            return Err(StatsError::InvalidInput(
                "SmcModelComparison: n_particles must be > 0".into(),
            ));
        }
        if config.ess_threshold <= 0.0 || config.ess_threshold > 1.0 {
            return Err(StatsError::DomainError(format!(
                "SmcModelComparison: ess_threshold must be in (0,1], got {}",
                config.ess_threshold
            )));
        }
        Ok(Self { config })
    }

    /// Initialize particles by sampling from the prior.
    pub fn initialize(
        &self,
        prior_sampler: &dyn Fn(&mut dyn FnMut() -> f64) -> Vec<f64>,
        rng: &mut Lcg,
    ) -> SmcParticles {
        let n = self.config.n_particles;
        let log_w0 = -(n as f64).ln();
        let mut uniform = || rng.next_f64();
        let thetas: Vec<Vec<f64>> = (0..n).map(|_| prior_sampler(&mut uniform)).collect();
        SmcParticles {
            thetas,
            log_weights: vec![log_w0; n],
            temperature: 0.0,
        }
    }

    /// Re-weight particles by `exp(delta_beta * log_likelihood(theta))`.
    ///
    /// Returns the log incremental normalizing constant.
    pub fn reweight(
        &self,
        particles: &mut SmcParticles,
        log_likelihood: &dyn Fn(&[f64]) -> f64,
        delta_beta: f64,
    ) -> f64 {
        let n = particles.n_particles();
        let mut log_increments = vec![0.0f64; n];
        for (i, theta) in particles.thetas.iter().enumerate() {
            let ll = log_likelihood(theta);
            if ll.is_finite() {
                log_increments[i] = delta_beta * ll;
            } else {
                log_increments[i] = f64::NEG_INFINITY;
            }
        }

        // Log incremental normalizing constant: log(mean_i exp(delta_beta * ll_i))
        // = logsumexp(log_w + delta_beta * ll) - logsumexp(log_w)
        let log_w_before = logsumexp(&particles.log_weights);
        for (i, &inc) in log_increments.iter().enumerate() {
            particles.log_weights[i] += inc;
        }
        let log_w_after = logsumexp(&particles.log_weights);
        particles.temperature += delta_beta;

        log_w_after - log_w_before
    }

    /// Find the next temperature increment via bisection.
    ///
    /// Finds the largest `delta_beta ≤ 1 - current_temp` such that
    /// `ESS(new weights) ≥ target_ess_ratio * N`.
    ///
    /// Returns the chosen `delta_beta`.
    pub fn find_next_temperature(
        &self,
        particles: &SmcParticles,
        log_likelihood: &dyn Fn(&[f64]) -> f64,
        target_ess_ratio: f64,
    ) -> f64 {
        let n = particles.n_particles() as f64;
        let remaining = 1.0 - particles.temperature;
        if remaining <= 1e-10 {
            return 0.0;
        }

        // Compute log-likelihood for each particle
        let log_liks: Vec<f64> = particles
            .thetas
            .iter()
            .map(|theta| {
                let ll = log_likelihood(theta);
                if ll.is_finite() {
                    ll
                } else {
                    -1e300
                }
            })
            .collect();

        // Binary search on delta_beta in [0, remaining]
        let target_ess = target_ess_ratio * n;
        let mut lo = 0.0_f64;
        let mut hi = remaining;

        // Quick check: if full step gives sufficient ESS, take it
        let ess_full = compute_ess_for_delta(&particles.log_weights, &log_liks, remaining);
        if ess_full >= target_ess {
            return remaining;
        }

        // Bisection
        for _ in 0..50 {
            let mid = 0.5 * (lo + hi);
            let ess_mid = compute_ess_for_delta(&particles.log_weights, &log_liks, mid);
            if ess_mid >= target_ess {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo < 1e-10 {
                break;
            }
        }

        lo.max(1e-8).min(remaining)
    }

    /// Stratified resampling of particles.
    ///
    /// Draws N indices proportional to exp(log_weights) using stratified
    /// sampling to reduce variance compared to multinomial resampling.
    pub fn resample(&self, particles: &mut SmcParticles, rng: &mut Lcg) {
        let n = particles.n_particles();
        if n == 0 {
            return;
        }

        let weights = particles.normalized_weights();
        let indices = stratified_resample(&weights, n, rng);

        let new_thetas: Vec<Vec<f64>> = indices
            .iter()
            .map(|&i| particles.thetas[i].clone())
            .collect();

        let log_w_uniform = -(n as f64).ln();
        particles.thetas = new_thetas;
        particles.log_weights = vec![log_w_uniform; n];
    }

    /// Metropolis-Hastings rejuvenation using random-walk proposal.
    ///
    /// Runs `config.mcmc_steps` MH steps per particle targeting the
    /// tempered posterior `π_β(θ) ∝ prior(θ) * likelihood(θ)^β`.
    pub fn mcmc_rejuvenate(
        &self,
        particles: &mut SmcParticles,
        log_posterior: &dyn Fn(&[f64]) -> f64,
        step_size: f64,
        rng: &mut Lcg,
    ) {
        let n = particles.n_particles();
        if n == 0 || particles.thetas.is_empty() {
            return;
        }
        let d = particles.thetas[0].len();

        for i in 0..n {
            let mut theta = particles.thetas[i].clone();
            let mut log_p = log_posterior(&theta);

            for _ in 0..self.config.mcmc_steps {
                // Random-walk proposal
                let proposal: Vec<f64> = theta
                    .iter()
                    .map(|&t| t + step_size * rng.next_normal())
                    .collect();

                let log_p_prop = log_posterior(&proposal);
                let log_accept = log_p_prop - log_p;

                if rng.next_f64().ln() < log_accept {
                    theta = proposal;
                    log_p = log_p_prop;
                }
            }

            particles.thetas[i] = theta;
            // log_weights unchanged (MH preserves target distribution)
        }

        let _ = d; // suppress warning
    }

    /// Run the complete SMC algorithm.
    ///
    /// # Parameters
    /// - `log_likelihood`: log p(data | θ)
    /// - `prior_sampler`: samples θ from the prior (takes a uniform RNG callback)
    /// - `prior_log_density`: log p(θ)
    ///
    /// # Returns
    /// [`SmcResult`] with the estimated log marginal likelihood.
    pub fn run(
        &self,
        log_likelihood: &dyn Fn(&[f64]) -> f64,
        prior_sampler: &dyn Fn(&mut dyn FnMut() -> f64) -> Vec<f64>,
        prior_log_density: &dyn Fn(&[f64]) -> f64,
    ) -> Result<SmcResult> {
        let mut rng = Lcg::new(self.config.seed);
        let mut particles = self.initialize(prior_sampler, &mut rng);

        let mut log_z = 0.0_f64;
        let mut temperatures = vec![0.0_f64];
        let mut n_steps = 0usize;

        let step_size = if self.config.step_size > 0.0 {
            self.config.step_size
        } else {
            0.1
        };

        for _step in 0..self.config.max_temperatures {
            if (1.0 - particles.temperature) < 1e-8 {
                break;
            }

            // Find optimal temperature increment
            let delta_beta =
                self.find_next_temperature(&particles, log_likelihood, self.config.ess_threshold);

            if delta_beta < 1e-12 {
                // Tiny step: take remaining temperature and finish
                let remaining = 1.0 - particles.temperature;
                log_z += self.reweight(&mut particles, log_likelihood, remaining);
                temperatures.push(particles.temperature);
                n_steps += 1;
                break;
            }

            // Reweight
            log_z += self.reweight(&mut particles, log_likelihood, delta_beta);
            temperatures.push(particles.temperature);
            n_steps += 1;

            // Resample if ESS is low
            let ess_val = particles.ess();
            if ess_val < self.config.ess_threshold * particles.n_particles() as f64 {
                self.resample(&mut particles, &mut rng);
            }

            // MCMC rejuvenation: target is tempered posterior
            let beta = particles.temperature;
            let log_post = move |theta: &[f64]| {
                let ll = log_likelihood(theta);
                let lp = prior_log_density(theta);
                if ll.is_finite() && lp.is_finite() {
                    beta * ll + lp
                } else {
                    f64::NEG_INFINITY
                }
            };
            self.mcmc_rejuvenate(&mut particles, &log_post, step_size, &mut rng);
        }

        // Final temperature = 1.0
        if (1.0 - particles.temperature).abs() > 1e-6 {
            let remaining = 1.0 - particles.temperature;
            log_z += self.reweight(&mut particles, log_likelihood, remaining);
            temperatures.push(1.0);
            n_steps += 1;
        }

        Ok(SmcResult {
            log_marginal_likelihood: log_z,
            particles,
            n_steps,
            temperatures,
        })
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Compute ESS for a hypothetical reweighting by delta_beta.
fn compute_ess_for_delta(log_weights: &[f64], log_liks: &[f64], delta_beta: f64) -> f64 {
    let incremented: Vec<f64> = log_weights
        .iter()
        .zip(log_liks.iter())
        .map(|(&w, &ll)| w + delta_beta * ll)
        .collect();
    ess(&incremented)
}

/// Stratified resampling: draw N indices ~ Categorical(weights).
///
/// Stratified resampling divides [0,1) into N strata of width 1/N,
/// draws one uniform from each stratum, and finds the corresponding
/// weight index via the CDF.
fn stratified_resample(weights: &[f64], n: usize, rng: &mut Lcg) -> Vec<usize> {
    let mut indices = Vec::with_capacity(n);
    if weights.is_empty() {
        return indices;
    }

    let n_f = n as f64;
    let u_start = rng.next_f64() / n_f;

    let mut cumsum = 0.0_f64;
    let mut j = 0usize;

    for i in 0..n {
        let u_i = u_start + i as f64 / n_f;
        while j < weights.len() - 1 && cumsum + weights[j] < u_i {
            cumsum += weights[j];
            j += 1;
        }
        indices.push(j);
    }

    indices
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ess_equal_weights() {
        let n = 100usize;
        let log_w = vec![-(n as f64).ln(); n];
        let ess_val = ess(&log_w);
        let tol = 1.0;
        assert!(
            (ess_val - n as f64).abs() < tol,
            "ESS of equal weights should be N={n}, got {ess_val:.3}"
        );
    }

    #[test]
    fn test_ess_one_hot() {
        // One particle has all the weight → ESS = 1
        let n = 100usize;
        let mut log_w = vec![-1e15_f64; n];
        log_w[0] = 0.0;
        let ess_val = ess(&log_w);
        assert!(
            (ess_val - 1.0).abs() < 0.01,
            "ESS of one-hot weight should be ~1, got {ess_val:.3}"
        );
    }

    #[test]
    fn test_logsumexp_stable_large() {
        let big = vec![1000.0, 1000.1, 999.9];
        let result = logsumexp(&big);
        // Should not overflow
        assert!(
            result.is_finite(),
            "logsumexp of large values should be finite"
        );
        // Expected ≈ 1000 + ln(3) ≈ 1001.099
        assert!((result - 1001.099).abs() < 0.01, "got {result}");
    }

    #[test]
    fn test_logsumexp_stable_small() {
        let small = vec![-1000.0, -1000.1, -999.9];
        let result = logsumexp(&small);
        assert!(
            result.is_finite(),
            "logsumexp of small values should be finite"
        );
        assert!(
            (result - (-999.0 + (3.0_f64).ln() - 1.0)).abs() < 0.1,
            "got {result}"
        );
    }

    #[test]
    fn test_find_next_temperature_range() {
        let config = SmcConfig {
            n_particles: 200,
            ess_threshold: 0.5,
            mcmc_steps: 1,
            max_temperatures: 50,
            step_size: 0.1,
            seed: 42,
        };
        let smc = SmcModelComparison::new(config).expect("ok");
        let mut rng = Lcg::new(42);

        // Simple Gaussian model
        let prior_sampler = |uniform: &mut dyn FnMut() -> f64| vec![2.0 * uniform() - 1.0];
        let particles = smc.initialize(&prior_sampler, &mut rng);

        // Log-likelihood: N(0, 1) data
        let log_lik = |theta: &[f64]| {
            let mu = theta[0];
            -0.5 * mu * mu // simplified: log p(0 | mu, sigma=1)
        };

        let delta = smc.find_next_temperature(&particles, &log_lik, 0.5);
        assert!(delta > 0.0, "delta_beta should be positive, got {delta}");
        assert!(delta <= 1.0, "delta_beta should be <= 1, got {delta}");
    }

    #[test]
    fn test_run_finite_log_ml() {
        let config = SmcConfig {
            n_particles: 200,
            ess_threshold: 0.5,
            mcmc_steps: 2,
            max_temperatures: 20,
            step_size: 0.3,
            seed: 99,
        };
        let smc = SmcModelComparison::new(config).expect("ok");

        // Model: theta ~ N(0, 1), data = [1.0, 0.5, -0.5] from N(theta, 1)
        let data = vec![1.0_f64, 0.5, -0.5];
        let prior_sampler = |u: &mut dyn FnMut() -> f64| {
            let u1 = u().max(1e-300);
            let u2 = u();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            vec![z]
        };
        let prior_log_density =
            |theta: &[f64]| -0.5 * theta[0] * theta[0] - 0.5 * (2.0 * std::f64::consts::PI).ln();
        let data_clone = data.clone();
        let log_likelihood = move |theta: &[f64]| {
            let mu = theta[0];
            data_clone
                .iter()
                .map(|&x| -0.5 * (x - mu).powi(2) - 0.5 * (2.0 * std::f64::consts::PI).ln())
                .sum::<f64>()
        };

        let result = smc
            .run(&log_likelihood, &prior_sampler, &prior_log_density)
            .expect("run ok");

        assert!(
            result.log_marginal_likelihood.is_finite(),
            "log ML should be finite, got {}",
            result.log_marginal_likelihood
        );
    }

    #[test]
    fn test_bayes_factor_correct_vs_misspecified() {
        // Correct model: theta ~ N(0,1), data ~ N(theta, 1)
        // Misspecified model: theta ~ N(10, 1), data ~ N(theta, 1) (wrong prior)
        let data: Vec<f64> = vec![0.1, -0.2, 0.3, 0.0, 0.2];

        let run_smc = |prior_mean: f64, seed: u64| -> f64 {
            let config = SmcConfig {
                n_particles: 500,
                ess_threshold: 0.5,
                mcmc_steps: 3,
                max_temperatures: 30,
                step_size: 0.3,
                seed,
            };
            let smc = SmcModelComparison::new(config).expect("ok");
            let pm = prior_mean;
            let prior_sampler = move |u: &mut dyn FnMut() -> f64| {
                let u1 = u().max(1e-300);
                let u2 = u();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                vec![pm + z]
            };
            let prior_log_density = move |theta: &[f64]| {
                -0.5 * (theta[0] - pm).powi(2) - 0.5 * (2.0 * std::f64::consts::PI).ln()
            };
            let data_c = data.clone();
            let log_likelihood = move |theta: &[f64]| {
                let mu = theta[0];
                data_c
                    .iter()
                    .map(|&x| -0.5 * (x - mu).powi(2) - 0.5 * (2.0 * std::f64::consts::PI).ln())
                    .sum::<f64>()
            };
            smc.run(&log_likelihood, &prior_sampler, &prior_log_density)
                .map(|r| r.log_marginal_likelihood)
                .unwrap_or(f64::NEG_INFINITY)
        };

        let log_ml_correct = run_smc(0.0, 42);
        let log_ml_wrong = run_smc(10.0, 43);
        let log_bf = log_ml_correct - log_ml_wrong;

        assert!(
            log_bf > 0.0,
            "Bayes factor should favor correct model (log BF = {log_bf:.3})"
        );
    }

    #[test]
    fn test_resample_preserves_particle_count() {
        let config = SmcConfig::default();
        let smc = SmcModelComparison::new(config).expect("ok");
        let mut rng = Lcg::new(1);

        let n = 50;
        let thetas: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let log_weights = vec![-(n as f64).ln(); n];
        let mut particles = SmcParticles {
            thetas,
            log_weights,
            temperature: 0.5,
        };

        smc.resample(&mut particles, &mut rng);
        assert_eq!(
            particles.thetas.len(),
            n,
            "resample should preserve particle count"
        );
        assert_eq!(particles.log_weights.len(), n);
    }

    #[test]
    fn test_smc_result_bayes_factor() {
        let r1 = SmcResult {
            log_marginal_likelihood: -5.0,
            particles: SmcParticles {
                thetas: vec![],
                log_weights: vec![],
                temperature: 1.0,
            },
            n_steps: 10,
            temperatures: vec![0.0, 1.0],
        };
        let r2 = SmcResult {
            log_marginal_likelihood: -8.0,
            particles: SmcParticles {
                thetas: vec![],
                log_weights: vec![],
                temperature: 1.0,
            },
            n_steps: 10,
            temperatures: vec![0.0, 1.0],
        };

        let bf = r1.bayes_factor(&r2);
        assert!((bf - 3.0_f64.exp()).abs() < 0.01, "BF = {bf:.3}");
        let log_bf = r1.log_bayes_factor(&r2);
        assert!((log_bf - 3.0).abs() < 1e-10);
    }
}
