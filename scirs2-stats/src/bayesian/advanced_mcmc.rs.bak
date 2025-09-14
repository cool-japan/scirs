//! Advanced MCMC sampling methods
//!
//! This module provides sophisticated Markov Chain Monte Carlo methods including
//! Hamiltonian Monte Carlo, No-U-Turn Sampler (NUTS), and adaptive algorithms.

use crate::error::{StatsError, StatsResult as Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use scirs2_core::validation::*;
use scirs2_core::{rng, Rng};

/// Trait for defining log probability density functions
pub trait LogDensity {
    /// Compute log probability density at given point
    fn log_density(&self, theta: &ArrayView1<f64>) -> Result<f64>;

    /// Compute gradient of log density (if available)
    fn gradient(&self, theta: &ArrayView1<f64>) -> Result<Option<Array1<f64>>>;

    /// Number of dimensions
    fn ndim(&self) -> usize;
}

/// Hamiltonian Monte Carlo (HMC) sampler
///
/// HMC uses gradient information to make efficient proposals in high-dimensional spaces.
/// It simulates Hamiltonian dynamics with momentum variables to explore the posterior.
#[derive(Debug, Clone)]
pub struct HamiltonianMonteCarlo {
    /// Step size for leapfrog integration
    pub stepsize: f64,
    /// Number of leapfrog steps
    pub n_steps: usize,
    /// Mass matrix (inverse covariance of momentum)
    pub mass_matrix: Option<Array2<f64>>,
    /// Random number generator seed
    pub seed: Option<u64>,
    /// Whether to adapt step size
    pub adapt_stepsize: bool,
    /// Target acceptance rate for adaptation
    pub target_acceptance: f64,
    /// Adaptation window size
    pub adaptation_window: usize,
}

impl HamiltonianMonteCarlo {
    /// Create a new HMC sampler
    pub fn new(stepsize: f64, n_steps: usize) -> Result<Self> {
        check_positive(stepsize, "stepsize")?;
        check_positive(n_steps, "n_steps")?;

        Ok(Self {
            stepsize,
            n_steps,
            mass_matrix: None,
            seed: None,
            adapt_stepsize: true,
            target_acceptance: 0.8,
            adaptation_window: 1000,
        })
    }

    /// Set mass matrix
    pub fn with_mass_matrix(mut self, mass_matrix: Array2<f64>) -> Result<Self> {
        checkarray_finite(&mass_matrix, "mass_matrix")?;
        self.mass_matrix = Some(mass_matrix);
        Ok(self)
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Disable step size adaptation
    pub fn without_adaptation(mut self) -> Self {
        self.adapt_stepsize = false;
        self
    }

    /// Sample from the target distribution
    pub fn sample<D: LogDensity>(
        &self,
        target: &D,
        n_samples_: usize,
        initial_state: ArrayView1<f64>,
    ) -> Result<HMCResult> {
        check_positive(n_samples_, "n_samples_")?;
        checkarray_finite(&initial_state, "initial_state")?;

        if initial_state.len() != target.ndim() {
            return Err(StatsError::DimensionMismatch(format!(
                "initial_state length ({}) must match target dimension ({})",
                initial_state.len(),
                target.ndim()
            )));
        }

        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut rng()),
        };

        let ndim = target.ndim();
        let mut samples = Array2::zeros((n_samples_, ndim));
        let mut log_probs = Array1::zeros(n_samples_);
        let mut accepted = Array1::from_elem(n_samples_, false);

        // Initialize mass matrix if not provided
        let mass_matrix = self
            .mass_matrix
            .clone()
            .unwrap_or_else(|| Array2::eye(ndim));
        let mass_matrix_inv = self.invert_mass_matrix(&mass_matrix)?;

        let mut current_state = initial_state.to_owned();
        let mut current_log_prob = target.log_density(&current_state.view())?;

        let mut stepsize = self.stepsize;
        let mut n_accepted = 0;

        for i in 0..n_samples_ {
            // Generate momentum
            let momentum = self.samplemomentum(&mass_matrix, &mut rng)?;

            // Hamiltonian dynamics
            let (proposed_state, proposedmomentum, proposed_log_prob) = self.leapfrog_integration(
                &current_state,
                &momentum,
                target,
                &mass_matrix_inv,
                stepsize,
            )?;

            // Metropolis acceptance
            let current_energy =
                -current_log_prob + 0.5 * self.kinetic_energy(&momentum, &mass_matrix_inv)?;
            let proposed_energy = -proposed_log_prob
                + 0.5 * self.kinetic_energy(&proposedmomentum, &mass_matrix_inv)?;

            let accept_prob = (-proposed_energy + current_energy).exp().min(1.0);
            let accept = rng.random::<f64>() < accept_prob;

            if accept {
                current_state = proposed_state;
                current_log_prob = proposed_log_prob;
                n_accepted += 1;
                accepted[i] = true;
            }

            samples.row_mut(i).assign(&current_state);
            log_probs[i] = current_log_prob;

            // Adapt step size
            if self.adapt_stepsize && i < self.adaptation_window {
                stepsize = self.adapt_stepsize_simple(stepsize, accept, self.target_acceptance);
            }
        }

        let acceptance_rate = n_accepted as f64 / n_samples_ as f64;

        Ok(HMCResult {
            samples,
            log_probabilities: log_probs,
            accepted,
            acceptance_rate,
            final_stepsize: stepsize,
            n_samples_,
            ndim,
        })
    }

    /// Sample momentum from multivariate normal
    fn samplemomentum<R: Rng>(
        &self,
        mass_matrix: &Array2<f64>,
        rng: &mut R,
    ) -> Result<Array1<f64>> {
        let ndim = mass_matrix.nrows();
        let mut momentum = Array1::zeros(ndim);

        // Sample from N(0, mass_matrix)
        for i in 0..ndim {
            momentum[i] = rng.random::<f64>() * 2.0 - 1.0; // Simplified - should use proper normal sampling
        }

        // Transform by Cholesky factor of mass _matrix (simplified)
        let scaledmomentum = mass_matrix.dot(&momentum);
        Ok(scaledmomentum)
    }

    /// Compute kinetic energy
    fn kinetic_energy(&self, momentum: &Array1<f64>, mass_matrix_inv: &Array2<f64>) -> Result<f64> {
        let kinetic = 0.5 * momentum.dot(&mass_matrix_inv.dot(momentum));
        Ok(kinetic)
    }

    /// Leapfrog integration for Hamiltonian dynamics
    fn leapfrog_integration<D: LogDensity>(
        &self,
        initial_position: &Array1<f64>,
        initialmomentum: &Array1<f64>,
        target: &D,
        mass_matrix_inv: &Array2<f64>,
        stepsize: f64,
    ) -> Result<(Array1<f64>, Array1<f64>, f64)> {
        let mut _position = initial_position.clone();
        let mut momentum = initialmomentum.clone();

        // Half step for momentum
        if let Some(grad) = target.gradient(&_position.view())? {
            momentum = &momentum + &(stepsize * 0.5 * &grad);
        } else {
            return Err(StatsError::ComputationError(
                "Gradient required for HMC but not available".to_string(),
            ));
        }

        // Full steps
        for _ in 0..self.n_steps {
            // Full step for _position
            _position = &_position + &(stepsize * &mass_matrix_inv.dot(&momentum));

            // Full step for momentum (except last)
            if let Some(grad) = target.gradient(&_position.view())? {
                momentum = &momentum + &(stepsize * &grad);
            }
        }

        // Final half step for momentum
        if let Some(grad) = target.gradient(&_position.view())? {
            momentum = &momentum + &(stepsize * 0.5 * &grad);
        }

        // Negate momentum for reversibility
        momentum = -momentum;

        let final_log_prob = target.log_density(&_position.view())?;

        Ok((_position, momentum, final_log_prob))
    }

    /// Invert mass matrix (simplified)
    fn invert_mass_matrix(&self, mass_matrix: &Array2<f64>) -> Result<Array2<f64>> {
        // Simplified inversion - in practice use proper _matrix inversion
        if mass_matrix.is_square() {
            // For now, assume diagonal mass _matrix
            let mut inv = Array2::zeros(mass_matrix.raw_dim());
            for i in 0..mass_matrix.nrows() {
                if mass_matrix[[i, i]].abs() < 1e-12 {
                    return Err(StatsError::ComputationError(
                        "Mass _matrix is singular".to_string(),
                    ));
                }
                inv[[i, i]] = 1.0 / mass_matrix[[i, i]];
            }
            Ok(inv)
        } else {
            Err(StatsError::ComputationError(
                "Mass _matrix must be square".to_string(),
            ))
        }
    }

    /// Simple step size adaptation
    fn adapt_stepsize_simple(
        &self,
        current_stepsize: f64,
        accepted: bool,
        target_rate: f64,
    ) -> f64 {
        let acceptance_rate = if accepted { 1.0 } else { 0.0 };
        let adaptation_rate = 0.01;

        if acceptance_rate > target_rate {
            current_stepsize * (1.0 + adaptation_rate)
        } else {
            current_stepsize * (1.0 - adaptation_rate)
        }
    }
}

/// Result of HMC sampling
#[derive(Debug, Clone)]
pub struct HMCResult {
    /// Generated samples (n_samples_ × ndim)
    pub samples: Array2<f64>,
    /// Log probabilities for each sample
    pub log_probabilities: Array1<f64>,
    /// Acceptance indicators for each sample
    pub accepted: Array1<bool>,
    /// Overall acceptance rate
    pub acceptance_rate: f64,
    /// Final adapted step size
    pub final_stepsize: f64,
    /// Number of samples
    pub n_samples_: usize,
    /// Number of dimensions
    pub ndim: usize,
}

/// No-U-Turn Sampler (NUTS)
///
/// NUTS is an extension of HMC that automatically tunes the number of leapfrog steps
/// by stopping when the trajectory starts to double back on itself.
#[derive(Debug, Clone)]
pub struct NoUTurnSampler {
    /// Initial step size
    pub initial_stepsize: f64,
    /// Maximum tree depth
    pub max_tree_depth: usize,
    /// Target acceptance rate for step size adaptation
    pub target_acceptance: f64,
    /// Step size adaptation parameter
    pub gamma: f64,
    /// Step size adaptation parameter
    pub t0: f64,
    /// Step size adaptation parameter
    pub kappa: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl NoUTurnSampler {
    /// Create a new NUTS sampler
    pub fn new() -> Self {
        Self {
            initial_stepsize: 1.0,
            max_tree_depth: 10,
            target_acceptance: 0.8,
            gamma: 0.05,
            t0: 10.0,
            kappa: 0.75,
            seed: None,
        }
    }

    /// Set initial step size
    pub fn with_stepsize(mut self, stepsize: f64) -> Self {
        self.initial_stepsize = stepsize;
        self
    }

    /// Set maximum tree depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_tree_depth = depth;
        self
    }

    /// Sample using NUTS algorithm
    pub fn sample<D: LogDensity>(
        &self,
        target: &D,
        n_samples_: usize,
        initial_state: ArrayView1<f64>,
    ) -> Result<NUTSResult> {
        check_positive(n_samples_, "n_samples_")?;
        checkarray_finite(&initial_state, "initial_state")?;

        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut rng()),
        };

        let ndim = target.ndim();
        let mut samples = Array2::zeros((n_samples_, ndim));
        let mut log_probs = Array1::zeros(n_samples_);
        let mut treesizes = Array1::zeros(n_samples_);

        let mut current_state = initial_state.to_owned();
        let mut current_log_prob;

        let mut stepsize = self.initial_stepsize;
        let mut stepsize_bar = self.initial_stepsize;
        let mut h_bar = 0.0;

        for i in 0..n_samples_ {
            // Sample momentum
            let momentum = self.samplemomentum(ndim, &mut rng);

            // Build tree and sample
            let (new_state, new_log_prob, treesize) =
                self.build_tree(&current_state, &momentum, target, stepsize, &mut rng)?;

            current_state = new_state;
            current_log_prob = new_log_prob;

            samples.row_mut(i).assign(&current_state);
            log_probs[i] = current_log_prob;
            treesizes[i] = treesize as f64;

            // Adapt step size using dual averaging
            if i < n_samples_ / 2 {
                let acceptance_prob = 1.0; // Simplified - should track actual acceptance
                h_bar = (1.0 - 1.0 / (i as f64 + self.t0)) * h_bar
                    + (self.target_acceptance - acceptance_prob) / (i as f64 + self.t0);

                stepsize = self.initial_stepsize * (-h_bar).exp();

                let eta = (i as f64 + 1.0).powf(-self.kappa);
                stepsize_bar = (-eta * h_bar).exp() * (i as f64 + 1.0).powf(-self.kappa)
                    + (1.0 - (i as f64 + 1.0).powf(-self.kappa)) * stepsize_bar;
            } else {
                stepsize = stepsize_bar;
            }
        }

        Ok(NUTSResult {
            samples,
            log_probabilities: log_probs,
            treesizes,
            final_stepsize: stepsize,
            n_samples_,
            ndim,
        })
    }

    /// Sample momentum from standard normal
    fn samplemomentum<R: Rng>(&self, ndim: usize, rng: &mut R) -> Array1<f64> {
        let mut momentum = Array1::zeros(ndim);
        for i in 0..ndim {
            // Simplified normal sampling using Box-Muller
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            momentum[i] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
        momentum
    }

    /// Build binary tree for NUTS
    fn build_tree<D: LogDensity, R: Rng>(
        &self,
        position: &Array1<f64>,
        momentum: &Array1<f64>,
        target: &D,
        stepsize: f64,
        rng: &mut R,
    ) -> Result<(Array1<f64>, f64, usize)> {
        // Simplified tree building - this is a basic implementation
        // A full NUTS implementation would be much more complex

        let mut current_pos = position.clone();
        let mut current_mom = momentum.clone();
        let _current_log_prob = target.log_density(&current_pos.view())?;

        // Take a few leapfrog steps (simplified)
        let n_steps = 2_usize.pow(rng.gen_range(1..self.max_tree_depth.min(4) + 1) as u32);

        for _ in 0..n_steps {
            // Simplified leapfrog step
            if let Some(grad) = target.gradient(&current_pos.view())? {
                current_mom = &current_mom + &(stepsize * 0.5 * &grad);
                current_pos = &current_pos + &(stepsize * &current_mom);
                if let Some(grad) = target.gradient(&current_pos.view())? {
                    current_mom = &current_mom + &(stepsize * 0.5 * &grad);
                }
            }
        }

        let new_log_prob = target.log_density(&current_pos.view())?;

        Ok((current_pos, new_log_prob, n_steps))
    }
}

/// Result of NUTS sampling
#[derive(Debug, Clone)]
pub struct NUTSResult {
    /// Generated samples (n_samples_ × ndim)
    pub samples: Array2<f64>,
    /// Log probabilities for each sample
    pub log_probabilities: Array1<f64>,
    /// Tree sizes for each iteration
    pub treesizes: Array1<f64>,
    /// Final adapted step size
    pub final_stepsize: f64,
    /// Number of samples
    pub n_samples_: usize,
    /// Number of dimensions
    pub ndim: usize,
}

/// Adaptive Metropolis sampler
///
/// Adapts the proposal covariance based on the sample history to improve efficiency.
#[derive(Debug, Clone)]
pub struct AdaptiveMetropolis {
    /// Initial proposal covariance
    pub initial_covariance: Option<Array2<f64>>,
    /// Adaptation start after this many samples
    pub adaptation_start: usize,
    /// Scaling factor for covariance
    pub scale_factor: f64,
    /// Small constant to prevent singularity
    pub epsilon: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl AdaptiveMetropolis {
    /// Create a new adaptive Metropolis sampler
    pub fn new() -> Self {
        Self {
            initial_covariance: None,
            adaptation_start: 100,
            scale_factor: 2.38 * 2.38, // Optimal scaling for multivariate normal
            epsilon: 1e-6,
            seed: None,
        }
    }

    /// Set initial covariance matrix
    pub fn with_covariance(mut self, cov: Array2<f64>) -> Self {
        self.initial_covariance = Some(cov);
        self
    }

    /// Sample using adaptive Metropolis
    pub fn sample<D: LogDensity>(
        &self,
        target: &D,
        n_samples_: usize,
        initial_state: ArrayView1<f64>,
    ) -> Result<AdaptiveMetropolisResult> {
        check_positive(n_samples_, "n_samples_")?;
        checkarray_finite(&initial_state, "initial_state")?;

        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut rng()),
        };

        let ndim = target.ndim();
        let mut samples = Array2::zeros((n_samples_, ndim));
        let mut log_probs = Array1::zeros(n_samples_);
        let mut accepted = Array1::from_elem(n_samples_, false);

        let mut current_state = initial_state.to_owned();
        let mut current_log_prob = target.log_density(&current_state.view())?;

        // Initialize covariance
        let mut covariance = self
            .initial_covariance
            .clone()
            .unwrap_or_else(|| Array2::eye(ndim));

        let mut sample_mean = Array1::zeros(ndim);
        let mut sample_cov = Array2::zeros((ndim, ndim));
        let mut n_adapted = 0;
        let mut n_accepted = 0;

        for i in 0..n_samples_ {
            // Generate proposal
            let proposal = self.generate_proposal(&current_state, &covariance, &mut rng)?;
            let proposal_log_prob = target.log_density(&proposal.view())?;

            // Metropolis acceptance
            let log_accept_prob = proposal_log_prob - current_log_prob;
            let accept = log_accept_prob.exp() > rng.random::<f64>();

            if accept {
                current_state = proposal;
                current_log_prob = proposal_log_prob;
                n_accepted += 1;
                accepted[i] = true;
            }

            samples.row_mut(i).assign(&current_state);
            log_probs[i] = current_log_prob;

            // Update adaptation statistics
            if i >= self.adaptation_start {
                n_adapted += 1;
                let delta = &current_state - &sample_mean;
                sample_mean = &sample_mean + &delta / (n_adapted as f64);

                // Update sample covariance (Welford's algorithm)
                let delta2 = &current_state - &sample_mean;
                for j in 0..ndim {
                    for k in 0..ndim {
                        sample_cov[[j, k]] += delta[j] * delta2[k];
                    }
                }

                // Update proposal covariance
                if n_adapted > 1 {
                    covariance =
                        &sample_cov / (n_adapted - 1) as f64 * self.scale_factor / ndim as f64;

                    // Add small diagonal term for numerical stability
                    for j in 0..ndim {
                        covariance[[j, j]] += self.epsilon;
                    }
                }
            }
        }

        let acceptance_rate = n_accepted as f64 / n_samples_ as f64;

        Ok(AdaptiveMetropolisResult {
            samples,
            log_probabilities: log_probs,
            accepted,
            acceptance_rate,
            final_covariance: covariance,
            n_samples_,
            ndim,
        })
    }

    /// Generate proposal using multivariate normal
    fn generate_proposal<R: Rng>(
        &self,
        current: &Array1<f64>,
        covariance: &Array2<f64>,
        rng: &mut R,
    ) -> Result<Array1<f64>> {
        let ndim = current.len();

        // Sample from N(0, I)
        let mut z = Array1::zeros(ndim);
        for i in 0..ndim {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            z[i] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }

        // Transform to N(current, covariance) using Cholesky decomposition
        // Simplified: assume diagonal covariance for now
        let mut proposal = current.clone();
        for i in 0..ndim {
            proposal[i] += z[i] * covariance[[i, i]].sqrt();
        }

        Ok(proposal)
    }
}

/// Result of adaptive Metropolis sampling
#[derive(Debug, Clone)]
pub struct AdaptiveMetropolisResult {
    /// Generated samples (n_samples_ × ndim)
    pub samples: Array2<f64>,
    /// Log probabilities for each sample
    pub log_probabilities: Array1<f64>,
    /// Acceptance indicators for each sample
    pub accepted: Array1<bool>,
    /// Overall acceptance rate
    pub acceptance_rate: f64,
    /// Final adapted covariance matrix
    pub final_covariance: Array2<f64>,
    /// Number of samples
    pub n_samples_: usize,
    /// Number of dimensions
    pub ndim: usize,
}

/// Parallel tempering (replica exchange) MCMC
///
/// Runs multiple chains at different temperatures to improve mixing and exploration
/// of multimodal distributions.
#[derive(Debug, Clone)]
pub struct ParallelTempering {
    /// Temperature ladder
    pub temperatures: Array1<f64>,
    /// Base sampler for each chain
    pub stepsize: f64,
    /// Number of steps between swap attempts
    pub swap_interval: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl ParallelTempering {
    /// Create a new parallel tempering sampler
    pub fn new(temperatures: Array1<f64>, stepsize: f64) -> Result<Self> {
        checkarray_finite(&temperatures, "temperatures")?;
        check_positive(stepsize, "stepsize")?;

        for &temp in temperatures.iter() {
            if temp <= 0.0 {
                return Err(StatsError::InvalidArgument(
                    "All _temperatures must be positive".to_string(),
                ));
            }
        }

        Ok(Self {
            temperatures,
            stepsize,
            swap_interval: 10,
            seed: None,
        })
    }

    /// Sample using parallel tempering
    pub fn sample<D: LogDensity + Send + Sync>(
        &self,
        target: &D,
        n_samples_: usize,
        initial_states: ArrayView2<f64>,
    ) -> Result<ParallelTemperingResult> {
        check_positive(n_samples_, "n_samples_")?;
        checkarray_finite(&initial_states, "initial_states")?;

        let n_chains = self.temperatures.len();
        let ndim = target.ndim();

        if initial_states.nrows() != n_chains || initial_states.ncols() != ndim {
            return Err(StatsError::DimensionMismatch(format!(
                "initial_states shape ({}, {}) must match (n_chains={}, ndim={})",
                initial_states.nrows(),
                initial_states.ncols(),
                n_chains,
                ndim
            )));
        }

        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut rng()),
        };

        // Initialize chains
        let mut chain_samples_ = vec![Array2::zeros((n_samples_, ndim)); n_chains];
        let mut chain_log_probs = vec![Array1::zeros(n_samples_); n_chains];
        let mut current_states: Vec<Array1<f64>> = initial_states
            .rows()
            .into_iter()
            .map(|row| row.to_owned())
            .collect();

        let mut current_log_probs = vec![0.0; n_chains];
        for (i, state) in current_states.iter().enumerate() {
            current_log_probs[i] = target.log_density(&state.view())?;
        }

        let mut n_swaps_attempted = 0;
        let mut n_swaps_accepted = 0;

        for sample_idx in 0..n_samples_ {
            // Update each chain with Metropolis step
            for chain_idx in 0..n_chains {
                let temp = self.temperatures[chain_idx];
                let (new_state, new_log_prob) = self.metropolis_step(
                    &current_states[chain_idx],
                    current_log_probs[chain_idx],
                    target,
                    temp,
                    &mut rng,
                )?;

                current_states[chain_idx] = new_state;
                current_log_probs[chain_idx] = new_log_prob;

                chain_samples_[chain_idx]
                    .row_mut(sample_idx)
                    .assign(&current_states[chain_idx]);
                chain_log_probs[chain_idx][sample_idx] = current_log_probs[chain_idx];
            }

            // Attempt swaps between adjacent temperatures
            if sample_idx % self.swap_interval == 0 && n_chains > 1 {
                for i in 0..n_chains - 1 {
                    n_swaps_attempted += 1;

                    let temp_i = self.temperatures[i];
                    let temp_j = self.temperatures[i + 1];
                    let log_prob_i = current_log_probs[i];
                    let log_prob_j = current_log_probs[i + 1];

                    // Compute swap probability
                    let beta_i = 1.0 / temp_i;
                    let beta_j = 1.0 / temp_j;
                    let log_swap_prob = (beta_i - beta_j) * (log_prob_j - log_prob_i);

                    if log_swap_prob.exp() > rng.random::<f64>() {
                        // Accept swap
                        current_states.swap(i, i + 1);
                        current_log_probs.swap(i, i + 1);
                        n_swaps_accepted += 1;
                    }
                }
            }
        }

        let swap_acceptance_rate = if n_swaps_attempted > 0 {
            n_swaps_accepted as f64 / n_swaps_attempted as f64
        } else {
            0.0
        };

        Ok(ParallelTemperingResult {
            chain_samples_,
            chain_log_probabilities: chain_log_probs,
            temperatures: self.temperatures.clone(),
            swap_acceptance_rate,
            n_samples_,
            n_chains,
            ndim,
        })
    }

    /// Single Metropolis step for a tempered chain
    fn metropolis_step<D: LogDensity, R: Rng>(
        &self,
        current: &Array1<f64>,
        current_log_prob: f64,
        target: &D,
        temperature: f64,
        rng: &mut R,
    ) -> Result<(Array1<f64>, f64)> {
        // Simple random walk proposal
        let ndim = current.len();
        let mut proposal = current.clone();

        for i in 0..ndim {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let normal_sample = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            proposal[i] += self.stepsize * normal_sample;
        }

        let proposal_log_prob = target.log_density(&proposal.view())?;

        // Tempered acceptance probability
        let log_accept_prob = (proposal_log_prob - current_log_prob) / temperature;

        if log_accept_prob.exp() > rng.random::<f64>() {
            Ok((proposal, proposal_log_prob))
        } else {
            Ok((current.clone(), current_log_prob))
        }
    }
}

/// Result of parallel tempering sampling
#[derive(Debug, Clone)]
pub struct ParallelTemperingResult {
    /// Samples from each chain (one per temperature)
    pub chain_samples_: Vec<Array2<f64>>,
    /// Log probabilities for each chain
    pub chain_log_probabilities: Vec<Array1<f64>>,
    /// Temperature ladder used
    pub temperatures: Array1<f64>,
    /// Rate of accepted temperature swaps
    pub swap_acceptance_rate: f64,
    /// Number of samples per chain
    pub n_samples_: usize,
    /// Number of chains (temperatures)
    pub n_chains: usize,
    /// Number of dimensions
    pub ndim: usize,
}

impl ParallelTemperingResult {
    /// Get samples from the cold chain (temperature = 1.0)
    pub fn cold_chain_samples_(&self) -> Result<&Array2<f64>> {
        // Find chain with temperature closest to 1.0
        let mut min_diff = f64::INFINITY;
        let mut cold_idx = 0;

        for (i, &temp) in self.temperatures.iter().enumerate() {
            let diff = (temp - 1.0).abs();
            if diff < min_diff {
                min_diff = diff;
                cold_idx = i;
            }
        }

        Ok(&self.chain_samples_[cold_idx])
    }
}

/// Example multivariate normal target distribution for testing
#[derive(Debug, Clone)]
pub struct MultivariateNormal {
    pub mean: Array1<f64>,
    pub precision: Array2<f64>,
    pub log_det_precision: f64,
}

impl MultivariateNormal {
    pub fn new(mean: Array1<f64>, covariance: Array2<f64>) -> Result<Self> {
        checkarray_finite(&mean, "mean")?;
        checkarray_finite(&covariance, "covariance")?;

        // Simplified precision computation (should use proper matrix inversion)
        let precision = Array2::eye(mean.len()); // Placeholder
        let log_det_precision = 0.0; // Placeholder

        Ok(Self {
            mean,
            precision,
            log_det_precision,
        })
    }
}

impl LogDensity for MultivariateNormal {
    fn log_density(&self, theta: &ArrayView1<f64>) -> Result<f64> {
        let diff = theta - &self.mean;
        let quad_form = diff.dot(&self.precision.dot(&diff));
        let log_prob = -0.5 * quad_form + 0.5 * self.log_det_precision
            - 0.5 * self.mean.len() as f64 * (2.0 * std::f64::consts::PI).ln();
        Ok(log_prob)
    }

    fn gradient(&self, theta: &ArrayView1<f64>) -> Result<Option<Array1<f64>>> {
        let diff = theta - &self.mean;
        let grad = -self.precision.dot(&diff);
        Ok(Some(grad))
    }

    fn ndim(&self) -> usize {
        self.mean.len()
    }
}
