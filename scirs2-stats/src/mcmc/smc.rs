//! Sequential Monte Carlo (SMC) methods
//!
//! This module provides implementations of Sequential Monte Carlo algorithms
//! including particle filters, SMC samplers for static models, and evidence
//! estimation via marginal likelihood.
//!
//! # Overview
//!
//! Sequential Monte Carlo methods maintain a population of weighted particles
//! that approximate a sequence of target distributions. As the target evolves
//! (either through time or a temperature schedule), particles are propagated,
//! reweighted, and resampled to maintain an accurate approximation.
//!
//! ## Key Algorithms
//!
//! - **Particle filter**: Bayesian filtering for state-space models
//! - **SMC sampler**: Static model inference via annealing
//! - **Resampling**: Systematic, multinomial, stratified, and residual methods
//! - **Evidence estimation**: Marginal likelihood via log-sum-exp accumulation

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::validation::*;
use scirs2_core::Rng;
use scirs2_core::RngExt;

// ──────────────────────────────────────────────────────────────────────────────
// Resampling strategies
// ──────────────────────────────────────────────────────────────────────────────

/// Resampling strategy for particle filters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResamplingStrategy {
    /// Systematic resampling: single random number, O(N)
    Systematic,
    /// Multinomial resampling: N random numbers, O(N log N)
    Multinomial,
    /// Stratified resampling: N random numbers in stratified intervals, O(N)
    Stratified,
    /// Residual resampling: deterministic + multinomial residual, O(N)
    Residual,
}

impl Default for ResamplingStrategy {
    fn default() -> Self {
        ResamplingStrategy::Systematic
    }
}

/// Compute normalised weights from log-weights using the log-sum-exp trick.
///
/// Returns `(normalised_weights, log_evidence_increment)` where the increment
/// is `log sum_i exp(log_w_i)`.
pub fn normalize_log_weights(log_weights: &Array1<f64>) -> (Array1<f64>, f64) {
    let max_lw = log_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    if max_lw.is_infinite() && max_lw < 0.0 {
        // All weights are −∞; return uniform weights and −∞ evidence
        let n = log_weights.len();
        let uniform = Array1::from_elem(n, 1.0 / n as f64);
        return (uniform, f64::NEG_INFINITY);
    }

    let shifted: Array1<f64> = log_weights.mapv(|lw| (lw - max_lw).exp());
    let sum_shifted = shifted.sum();
    let log_evidence = max_lw + sum_shifted.ln();
    let weights = shifted / sum_shifted;
    (weights, log_evidence)
}

/// Compute the effective sample size (ESS) from normalised weights.
///
/// ESS = 1 / sum(w_i^2), which equals N when all weights are equal.
pub fn effective_sample_size(weights: &Array1<f64>) -> f64 {
    let sum_sq: f64 = weights.iter().map(|&w| w * w).sum();
    if sum_sq <= 0.0 {
        0.0
    } else {
        1.0 / sum_sq
    }
}

/// Perform resampling according to the given strategy.
///
/// Returns a vector of parent indices (length N) selecting which particles to
/// keep. The selected particles should be used to create a new equally-weighted
/// particle set.
pub fn resample<R: Rng + ?Sized>(
    weights: &Array1<f64>,
    strategy: ResamplingStrategy,
    rng: &mut R,
) -> Result<Vec<usize>> {
    let n = weights.len();
    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "Weights array must be non-empty".to_string(),
        ));
    }

    // Verify weights sum approximately to 1
    let weight_sum: f64 = weights.iter().sum();
    if (weight_sum - 1.0).abs() > 1e-6 {
        return Err(StatsError::ComputationError(format!(
            "Weights must sum to 1, got {}",
            weight_sum
        )));
    }

    match strategy {
        ResamplingStrategy::Systematic => systematic_resample(weights, n, rng),
        ResamplingStrategy::Multinomial => multinomial_resample(weights, n, rng),
        ResamplingStrategy::Stratified => stratified_resample(weights, n, rng),
        ResamplingStrategy::Residual => residual_resample(weights, n, rng),
    }
}

fn systematic_resample<R: Rng + ?Sized>(
    weights: &Array1<f64>,
    n: usize,
    rng: &mut R,
) -> Result<Vec<usize>> {
    let u0: f64 = rng.random::<f64>() / n as f64;
    let mut indices = Vec::with_capacity(n);
    let mut cumsum = 0.0_f64;
    let mut j = 0_usize;

    for i in 0..n {
        let threshold = u0 + i as f64 / n as f64;
        while cumsum < threshold && j < n {
            cumsum += weights[j];
            if cumsum >= threshold {
                break;
            }
            j += 1;
        }
        indices.push(j.min(n - 1));
    }
    Ok(indices)
}

fn multinomial_resample<R: Rng + ?Sized>(
    weights: &Array1<f64>,
    n: usize,
    rng: &mut R,
) -> Result<Vec<usize>> {
    // Build CDF
    let mut cdf = Vec::with_capacity(n);
    let mut cumsum = 0.0_f64;
    for &w in weights.iter() {
        cumsum += w;
        cdf.push(cumsum);
    }

    let mut indices = Vec::with_capacity(n);
    for _ in 0..n {
        let u: f64 = rng.random();
        // Binary search for the index
        let idx = cdf.partition_point(|&c| c < u);
        indices.push(idx.min(n - 1));
    }
    Ok(indices)
}

fn stratified_resample<R: Rng + ?Sized>(
    weights: &Array1<f64>,
    n: usize,
    rng: &mut R,
) -> Result<Vec<usize>> {
    let mut cdf = Vec::with_capacity(n);
    let mut cumsum = 0.0_f64;
    for &w in weights.iter() {
        cumsum += w;
        cdf.push(cumsum);
    }

    let mut indices = Vec::with_capacity(n);
    for i in 0..n {
        let u: f64 = (i as f64 + rng.random::<f64>()) / n as f64;
        let idx = cdf.partition_point(|&c| c < u);
        indices.push(idx.min(n - 1));
    }
    Ok(indices)
}

fn residual_resample<R: Rng + ?Sized>(
    weights: &Array1<f64>,
    n: usize,
    rng: &mut R,
) -> Result<Vec<usize>> {
    // Deterministic floor copies
    let mut indices = Vec::with_capacity(n);
    let mut residuals = Vec::with_capacity(n);
    let mut n_residual = n;

    for (i, &w) in weights.iter().enumerate() {
        let floor_count = (w * n as f64).floor() as usize;
        for _ in 0..floor_count {
            indices.push(i);
        }
        let residual = w * n as f64 - floor_count as f64;
        residuals.push(residual);
        n_residual = n_residual.saturating_sub(floor_count);
    }

    // Multinomial resample the residuals
    let residual_sum: f64 = residuals.iter().sum();
    if n_residual > 0 && residual_sum > 0.0 {
        let residual_weights: Array1<f64> =
            Array1::from_vec(residuals.iter().map(|&r| r / residual_sum).collect());
        let residual_indices = multinomial_resample(&residual_weights, n_residual, rng)?;
        indices.extend_from_slice(&residual_indices);
    }

    Ok(indices)
}

// ──────────────────────────────────────────────────────────────────────────────
// Trait definitions
// ──────────────────────────────────────────────────────────────────────────────

/// Log-likelihood for the observation at time t given the state.
pub trait ObservationModel: Send + Sync {
    /// Compute log p(y_t | x_t)
    fn log_likelihood(&self, state: &Array1<f64>, observation: &Array1<f64>) -> f64;
}

/// Transition kernel for the state-space model.
pub trait TransitionKernel: Send + Sync {
    /// Sample x_{t+1} ~ p(· | x_t)
    fn sample<R: Rng + ?Sized>(&self, state: &Array1<f64>, rng: &mut R) -> Array1<f64>;
}

/// Prior distribution for particle filter initialisation.
pub trait PriorDistribution: Send + Sync {
    /// Sample from the prior p(x_0)
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64>;

    /// Evaluate log p(x_0)
    fn log_density(&self, x: &Array1<f64>) -> f64;
}

/// Target distribution for SMC sampler (unnormalised log density).
pub trait SmcTargetDistribution: Send + Sync {
    /// Evaluate the unnormalised log density at temperature `beta` in [0, 1].
    ///
    /// A common choice is:
    ///   log π_β(x) = log prior(x) + β · log likelihood(x)
    fn log_density(&self, x: &Array1<f64>, beta: f64) -> f64;

    /// Dimensionality of the state space.
    fn dim(&self) -> usize;
}

/// Markov kernel used within the SMC sampler to refresh particles.
pub trait SmcKernel: Send + Sync {
    /// Apply the MCMC kernel targeting π_β to move a particle.
    fn step<R: Rng + ?Sized>(
        &self,
        state: &Array1<f64>,
        beta: f64,
        target: &dyn SmcTargetDistribution,
        rng: &mut R,
    ) -> Array1<f64>;
}

// ──────────────────────────────────────────────────────────────────────────────
// Particle filter
// ──────────────────────────────────────────────────────────────────────────────

/// Bootstrap particle filter (sequential importance resampling)
///
/// Implements the classic bootstrap particle filter which:
/// 1. Propagates particles through the transition kernel
/// 2. Weights by the observation likelihood
/// 3. Resamples when the ESS falls below a threshold
///
/// The filter tracks the log marginal likelihood as the sum of incremental
/// log-evidence at each step.
pub struct ParticleFilter<O, T, P>
where
    O: ObservationModel,
    T: TransitionKernel,
    P: PriorDistribution,
{
    /// Observation model
    pub observation_model: O,
    /// Transition kernel
    pub transition_kernel: T,
    /// Prior distribution
    pub prior: P,
    /// Current particle states
    pub particles: Array2<f64>,
    /// Current normalised log-weights
    pub log_weights: Array1<f64>,
    /// Accumulated log marginal likelihood
    pub log_marginal_likelihood: f64,
    /// Number of particles
    pub n_particles: usize,
    /// State dimension
    pub state_dim: usize,
    /// ESS threshold for triggering resampling (fraction of N)
    pub ess_threshold: f64,
    /// Resampling strategy
    pub resampling_strategy: ResamplingStrategy,
    /// Current time step
    pub t: usize,
}

impl<O, T, P> ParticleFilter<O, T, P>
where
    O: ObservationModel,
    T: TransitionKernel,
    P: PriorDistribution,
{
    /// Create a new bootstrap particle filter and initialise from the prior.
    ///
    /// # Arguments
    ///
    /// * `n_particles` - Number of particles
    /// * `state_dim` - Dimensionality of the state
    /// * `observation_model` - p(y | x) model
    /// * `transition_kernel` - p(x_t | x_{t-1}) kernel
    /// * `prior` - p(x_0) prior
    /// * `ess_threshold` - Fraction of N below which to resample (default: 0.5)
    /// * `strategy` - Resampling strategy
    /// * `rng` - Random number generator
    pub fn new<R: Rng + ?Sized>(
        n_particles: usize,
        state_dim: usize,
        observation_model: O,
        transition_kernel: T,
        prior: P,
        ess_threshold: f64,
        strategy: ResamplingStrategy,
        rng: &mut R,
    ) -> Result<Self> {
        check_positive(n_particles, "n_particles")?;
        check_positive(state_dim, "state_dim")?;

        if !(0.0..=1.0).contains(&ess_threshold) {
            return Err(StatsError::InvalidArgument(
                "ess_threshold must be in [0, 1]".to_string(),
            ));
        }

        // Sample initial particles from prior
        let mut particles = Array2::zeros((n_particles, state_dim));
        for i in 0..n_particles {
            let sample = prior.sample(rng);
            if sample.len() != state_dim {
                return Err(StatsError::DimensionMismatch(format!(
                    "Prior sample has dimension {} but state_dim is {}",
                    sample.len(),
                    state_dim
                )));
            }
            particles.row_mut(i).assign(&sample);
        }

        // Uniform log-weights
        let log_w0 = -(n_particles as f64).ln();
        let log_weights = Array1::from_elem(n_particles, log_w0);

        Ok(Self {
            observation_model,
            transition_kernel,
            prior,
            particles,
            log_weights,
            log_marginal_likelihood: 0.0,
            n_particles,
            state_dim,
            ess_threshold,
            resampling_strategy: strategy,
            t: 0,
        })
    }

    /// Update the filter with a new observation.
    ///
    /// Performs one step of the bootstrap particle filter:
    /// 1. Propagate particles: x_t ~ p(· | x_{t-1})
    /// 2. Update weights: w_t ∝ w_{t-1} · p(y_t | x_t)
    /// 3. Resample if ESS < threshold
    ///
    /// Returns the current log marginal likelihood estimate.
    pub fn update<R: Rng + ?Sized>(
        &mut self,
        observation: &Array1<f64>,
        rng: &mut R,
    ) -> Result<f64> {
        // --- Step 1: Propagate ---
        let mut new_particles = Array2::zeros((self.n_particles, self.state_dim));
        for i in 0..self.n_particles {
            let x_prev = self.particles.row(i).to_owned();
            let x_new = self.transition_kernel.sample(&x_prev, rng);
            new_particles.row_mut(i).assign(&x_new);
        }
        self.particles = new_particles;

        // --- Step 2: Reweight ---
        let mut new_log_weights = Array1::zeros(self.n_particles);
        for i in 0..self.n_particles {
            let x = self.particles.row(i).to_owned();
            let log_lik = self.observation_model.log_likelihood(&x, observation);
            new_log_weights[i] = self.log_weights[i] + log_lik;
        }

        // Normalise and accumulate evidence
        let (normalised_weights, log_evidence_increment) = normalize_log_weights(&new_log_weights);
        self.log_marginal_likelihood += log_evidence_increment;
        self.log_weights = new_log_weights;

        // --- Step 3: Resample if needed ---
        let ess = effective_sample_size(&normalised_weights);
        if ess < self.ess_threshold * self.n_particles as f64 {
            let indices = resample(&normalised_weights, self.resampling_strategy, rng)?;

            let old_particles = self.particles.clone();
            for (i, &parent) in indices.iter().enumerate() {
                let row = old_particles.row(parent).to_owned();
                self.particles.row_mut(i).assign(&row);
            }

            // Reset to uniform log-weights
            let log_w_uniform = -(self.n_particles as f64).ln();
            self.log_weights = Array1::from_elem(self.n_particles, log_w_uniform);
        }

        self.t += 1;
        Ok(self.log_marginal_likelihood)
    }

    /// Run the particle filter over a sequence of observations.
    ///
    /// Returns `(filtered_means, log_marginal_likelihood)` where
    /// `filtered_means` has shape (T, state_dim).
    pub fn filter<R: Rng + ?Sized>(
        &mut self,
        observations: &Array2<f64>,
        rng: &mut R,
    ) -> Result<(Array2<f64>, f64)> {
        let n_steps = observations.nrows();
        let mut filtered_means = Array2::zeros((n_steps, self.state_dim));

        for t in 0..n_steps {
            let obs = observations.row(t).to_owned();
            self.update(&obs, rng)?;

            // Compute weighted mean
            let (norm_w, _) = normalize_log_weights(&self.log_weights);
            let mut mean = Array1::zeros(self.state_dim);
            for i in 0..self.n_particles {
                let row = self.particles.row(i);
                mean = mean + row.to_owned() * norm_w[i];
            }
            filtered_means.row_mut(t).assign(&mean);
        }

        Ok((filtered_means, self.log_marginal_likelihood))
    }

    /// Return the current particle approximation as (particles, normalised_weights).
    pub fn current_approximation(&self) -> (Array2<f64>, Array1<f64>) {
        let (weights, _) = normalize_log_weights(&self.log_weights);
        (self.particles.clone(), weights)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Random-walk Metropolis kernel for SMC sampler
// ──────────────────────────────────────────────────────────────────────────────

/// Random-walk Metropolis kernel suitable for use within an SMC sampler.
///
/// At each call the kernel attempts a single Metropolis-Hastings step with
/// a Gaussian proposal of standard deviation `step_size`.
pub struct RandomWalkKernel {
    /// Proposal standard deviation
    pub step_size: f64,
}

impl RandomWalkKernel {
    /// Create a new random-walk Metropolis kernel.
    pub fn new(step_size: f64) -> Result<Self> {
        if step_size <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "step_size must be positive".to_string(),
            ));
        }
        Ok(Self { step_size })
    }
}

impl SmcKernel for RandomWalkKernel {
    fn step<R: Rng + ?Sized>(
        &self,
        state: &Array1<f64>,
        beta: f64,
        target: &dyn SmcTargetDistribution,
        rng: &mut R,
    ) -> Array1<f64> {
        let dim = state.len();
        let mut proposal = state.clone();
        for i in 0..dim {
            // Box-Muller: one call for simplicity
            let u1: f64 = rng.random::<f64>().max(f64::EPSILON);
            let u2: f64 = rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            proposal[i] = state[i] + self.step_size * z;
        }

        let log_accept = target.log_density(&proposal, beta) - target.log_density(state, beta);
        let u: f64 = rng.random();
        if u.ln() < log_accept {
            proposal
        } else {
            state.clone()
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SMC sampler for static models
// ──────────────────────────────────────────────────────────────────────────────

/// SMC sampler for static models using annealing.
///
/// The SMC sampler bridges a prior (β=0) to the posterior (β=1) through a
/// geometric temperature schedule. At each temperature level:
///
/// 1. Reweight particles by the ratio π_{β_{t+1}} / π_{β_t}
/// 2. Resample if ESS is too low
/// 3. Apply MCMC refreshment steps to diversify the population
///
/// The log evidence estimate (log Z) is accumulated as the sum of
/// incremental log-normalising constants at each temperature step.
pub struct SmcSampler<T, K>
where
    T: SmcTargetDistribution,
    K: SmcKernel,
{
    /// Target distribution
    pub target: T,
    /// MCMC kernel for refreshment
    pub kernel: K,
    /// Current particle states (n_particles × dim)
    pub particles: Array2<f64>,
    /// Current log-weights (unnormalised)
    pub log_weights: Array1<f64>,
    /// Accumulated log-evidence
    pub log_evidence: f64,
    /// Temperature schedule
    pub betas: Vec<f64>,
    /// Number of MCMC refreshment steps per temperature level
    pub n_mcmc_steps: usize,
    /// Resampling strategy
    pub resampling_strategy: ResamplingStrategy,
    /// ESS threshold (fraction of N)
    pub ess_threshold: f64,
    /// Number of particles
    pub n_particles: usize,
}

impl<T: SmcTargetDistribution, K: SmcKernel> SmcSampler<T, K> {
    /// Construct an SMC sampler.
    ///
    /// # Arguments
    ///
    /// * `n_particles` - Number of particles
    /// * `target` - Target distribution implementing [`SmcTargetDistribution`]
    /// * `kernel` - MCMC kernel for particle refreshment
    /// * `betas` - Temperature schedule (must start at 0 and end at 1,
    ///   strictly increasing). If `None`, uses a geometric schedule with
    ///   10 steps.
    /// * `n_mcmc_steps` - MCMC steps per temperature level
    /// * `ess_threshold` - ESS fraction below which to resample
    /// * `strategy` - Resampling strategy
    pub fn new(
        n_particles: usize,
        target: T,
        kernel: K,
        betas: Option<Vec<f64>>,
        n_mcmc_steps: usize,
        ess_threshold: f64,
        strategy: ResamplingStrategy,
    ) -> Result<Self> {
        check_positive(n_particles, "n_particles")?;

        let betas = match betas {
            Some(b) => b,
            None => {
                // Default geometric schedule
                let n_steps = 10_usize;
                (0..=n_steps)
                    .map(|i| (i as f64 / n_steps as f64).powi(2))
                    .collect()
            }
        };

        // Validate temperature schedule
        if betas.is_empty() || (betas[0] - 0.0).abs() > 1e-12 {
            return Err(StatsError::InvalidArgument(
                "betas must be non-empty and start at 0".to_string(),
            ));
        }
        if (betas[betas.len() - 1] - 1.0).abs() > 1e-12 {
            return Err(StatsError::InvalidArgument(
                "betas must end at 1".to_string(),
            ));
        }
        for w in betas.windows(2) {
            if w[1] <= w[0] {
                return Err(StatsError::InvalidArgument(
                    "betas must be strictly increasing".to_string(),
                ));
            }
        }

        if !(0.0..=1.0).contains(&ess_threshold) {
            return Err(StatsError::InvalidArgument(
                "ess_threshold must be in [0, 1]".to_string(),
            ));
        }

        let dim = target.dim();
        // Particles will be initialised by calling `initialise`
        let particles = Array2::zeros((n_particles, dim));
        let log_w0 = -(n_particles as f64).ln();
        let log_weights = Array1::from_elem(n_particles, log_w0);

        Ok(Self {
            target,
            kernel,
            particles,
            log_weights,
            log_evidence: 0.0,
            betas,
            n_mcmc_steps,
            resampling_strategy: strategy,
            ess_threshold,
            n_particles,
        })
    }

    /// Initialise particles from a sampler at β=0 (the prior).
    ///
    /// The `prior_sampler` closure should return a sample from the prior.
    pub fn initialise<R: Rng + ?Sized, F>(&mut self, prior_sampler: F, rng: &mut R) -> Result<()>
    where
        F: Fn(&mut R) -> Array1<f64>,
    {
        let dim = self.target.dim();
        for i in 0..self.n_particles {
            let sample = prior_sampler(rng);
            if sample.len() != dim {
                return Err(StatsError::DimensionMismatch(format!(
                    "Prior sample has dimension {} but target.dim() is {}",
                    sample.len(),
                    dim
                )));
            }
            self.particles.row_mut(i).assign(&sample);
        }
        // Uniform log-weights at β=0
        let log_w0 = -(self.n_particles as f64).ln();
        self.log_weights = Array1::from_elem(self.n_particles, log_w0);
        self.log_evidence = 0.0;
        Ok(())
    }

    /// Run the full SMC sampler from β=0 to β=1.
    ///
    /// Returns `(samples, log_evidence)` where `samples` has shape
    /// (n_particles, dim) and contains the final particle set (at β=1),
    /// and `log_evidence` is the log marginal likelihood estimate.
    pub fn run<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<(Array2<f64>, f64)> {
        let n_betas = self.betas.len();

        for level in 1..n_betas {
            let beta_prev = self.betas[level - 1];
            let beta_curr = self.betas[level];

            // --- Reweight ---
            let mut new_log_weights = Array1::zeros(self.n_particles);
            for i in 0..self.n_particles {
                let x = self.particles.row(i).to_owned();
                let log_w_increment =
                    self.target.log_density(&x, beta_curr) - self.target.log_density(&x, beta_prev);
                new_log_weights[i] = self.log_weights[i] + log_w_increment;
            }

            let (normalised_weights, log_z_increment) = normalize_log_weights(&new_log_weights);
            self.log_evidence += log_z_increment;
            self.log_weights = new_log_weights;

            // --- Resample if needed ---
            let ess = effective_sample_size(&normalised_weights);
            if ess < self.ess_threshold * self.n_particles as f64 {
                let indices = resample(&normalised_weights, self.resampling_strategy, rng)?;

                let old_particles = self.particles.clone();
                for (i, &parent) in indices.iter().enumerate() {
                    let row = old_particles.row(parent).to_owned();
                    self.particles.row_mut(i).assign(&row);
                }

                let log_w_uniform = -(self.n_particles as f64).ln();
                self.log_weights = Array1::from_elem(self.n_particles, log_w_uniform);
            }

            // --- MCMC refreshment ---
            for i in 0..self.n_particles {
                let mut state = self.particles.row(i).to_owned();
                for _ in 0..self.n_mcmc_steps {
                    state = self.kernel.step(&state, beta_curr, &self.target, rng);
                }
                self.particles.row_mut(i).assign(&state);
            }
        }

        Ok((self.particles.clone(), self.log_evidence))
    }

    /// Return the current weighted sample approximation.
    pub fn current_approximation(&self) -> (Array2<f64>, Array1<f64>) {
        let (weights, _) = normalize_log_weights(&self.log_weights);
        (self.particles.clone(), weights)
    }

    /// Compute the (normalised) weighted mean of the current particle set.
    pub fn weighted_mean(&self) -> Array1<f64> {
        let (weights, _) = normalize_log_weights(&self.log_weights);
        let dim = self.target.dim();
        let mut mean = Array1::zeros(dim);
        for i in 0..self.n_particles {
            let row = self.particles.row(i);
            mean = mean + row.to_owned() * weights[i];
        }
        mean
    }

    /// Compute the (normalised) weighted variance of each dimension.
    pub fn weighted_variance(&self) -> Array1<f64> {
        let (weights, _) = normalize_log_weights(&self.log_weights);
        let mean = self.weighted_mean();
        let dim = self.target.dim();
        let mut variance = Array1::zeros(dim);
        for i in 0..self.n_particles {
            let row = self.particles.row(i);
            let diff = row.to_owned() - &mean;
            variance = variance + diff.mapv(|v| v * v) * weights[i];
        }
        variance
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Convenience: Gaussian prior/target for SMC
// ──────────────────────────────────────────────────────────────────────────────

/// Simple isotropic Gaussian prior for SMC sampler initialisation.
#[derive(Debug, Clone)]
pub struct GaussianPrior {
    /// Prior mean
    pub mean: Array1<f64>,
    /// Prior standard deviation (isotropic)
    pub std: f64,
}

impl GaussianPrior {
    /// Create a new Gaussian prior.
    pub fn new(mean: Array1<f64>, std: f64) -> Result<Self> {
        if std <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "std must be positive".to_string(),
            ));
        }
        Ok(Self { mean, std })
    }
}

impl PriorDistribution for GaussianPrior {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
        let dim = self.mean.len();
        let mut out = self.mean.clone();
        for i in 0..dim {
            let u1: f64 = rng.random::<f64>().max(f64::EPSILON);
            let u2: f64 = rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            out[i] = self.mean[i] + self.std * z;
        }
        out
    }

    fn log_density(&self, x: &Array1<f64>) -> f64 {
        let dim = self.mean.len() as f64;
        let inv_var = 1.0 / (self.std * self.std);
        let diff = x - &self.mean;
        let quad = diff.iter().map(|&v| v * v).sum::<f64>() * inv_var;
        -0.5 * quad - 0.5 * dim * (2.0 * std::f64::consts::PI * self.std * self.std).ln()
    }
}

/// Gaussian target distribution for SMC sampler testing.
///
/// Anneals from the prior (β=0) to the posterior (β=1):
///   log π_β(x) = (1-β)·log prior(x) + β·log likelihood(x)
pub struct GaussianSmcTarget {
    /// Prior mean
    pub prior_mean: Array1<f64>,
    /// Prior variance (isotropic)
    pub prior_var: f64,
    /// Likelihood mean
    pub likelihood_mean: Array1<f64>,
    /// Likelihood variance (isotropic)
    pub likelihood_var: f64,
}

impl GaussianSmcTarget {
    /// Create a new Gaussian SMC target.
    pub fn new(
        prior_mean: Array1<f64>,
        prior_var: f64,
        likelihood_mean: Array1<f64>,
        likelihood_var: f64,
    ) -> Result<Self> {
        if prior_mean.len() != likelihood_mean.len() {
            return Err(StatsError::DimensionMismatch(
                "prior_mean and likelihood_mean must have the same length".to_string(),
            ));
        }
        if prior_var <= 0.0 || likelihood_var <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "Variances must be positive".to_string(),
            ));
        }
        Ok(Self {
            prior_mean,
            prior_var,
            likelihood_mean,
            likelihood_var,
        })
    }
}

impl SmcTargetDistribution for GaussianSmcTarget {
    fn log_density(&self, x: &Array1<f64>, beta: f64) -> f64 {
        let dim = self.prior_mean.len() as f64;

        let prior_diff = x - &self.prior_mean;
        let log_prior = -0.5 * prior_diff.iter().map(|&v| v * v).sum::<f64>() / self.prior_var
            - 0.5 * dim * (2.0 * std::f64::consts::PI * self.prior_var).ln();

        let lik_diff = x - &self.likelihood_mean;
        let log_lik = -0.5 * lik_diff.iter().map(|&v| v * v).sum::<f64>() / self.likelihood_var
            - 0.5 * dim * (2.0 * std::f64::consts::PI * self.likelihood_var).ln();

        (1.0 - beta) * log_prior + beta * log_lik
    }

    fn dim(&self) -> usize {
        self.prior_mean.len()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_normalize_log_weights() {
        let log_w = array![0.0_f64, 0.0, 0.0, 0.0];
        let (w, log_z) = normalize_log_weights(&log_w);
        assert!((w.sum() - 1.0).abs() < 1e-12);
        assert!((log_z - (4.0_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_effective_sample_size() {
        // Uniform weights → ESS = N
        let n = 100_usize;
        let weights = Array1::from_elem(n, 1.0 / n as f64);
        let ess = effective_sample_size(&weights);
        assert!((ess - n as f64).abs() < 1e-10);

        // One particle with all weight → ESS = 1
        let mut degenerate = Array1::zeros(n);
        degenerate[0] = 1.0;
        let ess_deg = effective_sample_size(&degenerate);
        assert!((ess_deg - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_systematic_resample() {
        let mut rng = SmallRng::seed_from_u64(42);
        let weights = array![0.5_f64, 0.3, 0.1, 0.1];
        let indices = resample(&weights, ResamplingStrategy::Systematic, &mut rng)
            .expect("resample should succeed");
        assert_eq!(indices.len(), 4);
        assert!(indices.iter().all(|&i| i < 4));
    }

    #[test]
    fn test_stratified_resample() {
        let mut rng = SmallRng::seed_from_u64(99);
        let weights = array![0.25_f64, 0.25, 0.25, 0.25];
        let indices = resample(&weights, ResamplingStrategy::Stratified, &mut rng)
            .expect("resample should succeed");
        assert_eq!(indices.len(), 4);
    }

    #[test]
    fn test_multinomial_resample() {
        let mut rng = SmallRng::seed_from_u64(7);
        let weights = array![0.5_f64, 0.25, 0.25];
        let indices = resample(&weights, ResamplingStrategy::Multinomial, &mut rng)
            .expect("resample should succeed");
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_smc_sampler_gaussian() {
        let mut rng = SmallRng::seed_from_u64(12345);
        let dim = 2_usize;

        let prior_mean = Array1::zeros(dim);
        let likelihood_mean = Array1::from_elem(dim, 2.0);

        let target = GaussianSmcTarget::new(prior_mean.clone(), 1.0, likelihood_mean.clone(), 0.5)
            .expect("target creation should succeed");

        let kernel = RandomWalkKernel::new(0.3).expect("kernel creation should succeed");

        let betas: Vec<f64> = (0..=5).map(|i| i as f64 / 5.0).collect();

        let mut sampler = SmcSampler::new(
            200,
            target,
            kernel,
            Some(betas),
            3,
            0.5,
            ResamplingStrategy::Systematic,
        )
        .expect("sampler creation should succeed");

        let prior = GaussianPrior::new(prior_mean, 1.0).expect("prior creation should succeed");
        sampler
            .initialise(|rng| prior.sample(rng), &mut rng)
            .expect("initialise should succeed");

        let (_samples, log_z) = sampler.run(&mut rng).expect("run should succeed");
        assert!(log_z.is_finite());

        let mean = sampler.weighted_mean();
        // Posterior mean should be between prior (0) and likelihood (2) means
        for &m in mean.iter() {
            assert!(m > -1.0 && m < 3.5, "mean {} out of expected range", m);
        }
    }

    #[test]
    fn test_gaussian_prior_density() {
        let prior = GaussianPrior::new(Array1::zeros(2), 1.0).expect("should succeed");
        let x = Array1::zeros(2);
        let log_p = prior.log_density(&x);
        // log N(0|0,1) in 2D = -ln(2π) ≈ -1.8379
        let expected = -(2.0 * std::f64::consts::PI).ln();
        assert!(
            (log_p - expected).abs() < 0.1,
            "log_p={log_p}, expected={expected}"
        );
    }
}
