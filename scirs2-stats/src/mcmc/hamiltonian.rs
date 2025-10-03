//! Hamiltonian Monte Carlo (HMC) sampling
//!
//! HMC is a sophisticated MCMC method that uses gradient information to make
//! more efficient proposals than random walk methods.

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Distribution, Normal};
use scirs2_core::validation::*;
use scirs2_core::Rng;
use std::fmt::Debug;

/// Target distribution trait with gradient information for HMC
pub trait DifferentiableTarget: Send + Sync {
    /// Compute the log probability density
    fn log_density(&self, x: &Array1<f64>) -> f64;

    /// Compute the gradient of the log density
    fn gradient(&self, x: &Array1<f64>) -> Array1<f64>;

    /// Get the dimensionality
    fn dim(&self) -> usize;

    /// Optional: compute both log density and gradient together for efficiency
    fn log_density_and_gradient(&self, x: &Array1<f64>) -> (f64, Array1<f64>) {
        (self.log_density(x), self.gradient(x))
    }
}

/// Hamiltonian Monte Carlo sampler
pub struct HamiltonianMonteCarlo<T: DifferentiableTarget> {
    /// Target distribution
    pub target: T,
    /// Current position
    pub position: Array1<f64>,
    /// Current log density
    pub current_log_density: f64,
    /// Step size for leapfrog integration
    pub stepsize: f64,
    /// Number of leapfrog steps
    pub n_steps: usize,
    /// Mass matrix (identity for standard HMC)
    pub mass_matrix: Array2<f64>,
    /// Mass matrix inverse
    pub mass_inv: Array2<f64>,
    /// Number of accepted proposals
    pub n_accepted: usize,
    /// Total number of proposals
    pub n_proposed: usize,
}

impl<T: DifferentiableTarget> HamiltonianMonteCarlo<T> {
    /// Create a new HMC sampler
    pub fn new(target: T, initial: Array1<f64>, stepsize: f64, nsteps: usize) -> Result<Self> {
        checkarray_finite(&initial, "initial")?;
        check_positive(stepsize, "stepsize")?;
        check_positive(nsteps, "nsteps")?;

        if initial.len() != target.dim() {
            return Err(StatsError::DimensionMismatch(format!(
                "initial dimension ({}) must match target dimension ({})",
                initial.len(),
                target.dim()
            )));
        }

        let dim = initial.len();
        let mass_matrix = Array2::eye(dim);
        let mass_inv = Array2::eye(dim);
        let current_log_density = target.log_density(&initial);

        Ok(Self {
            target,
            position: initial,
            current_log_density,
            stepsize,
            n_steps: nsteps,
            mass_matrix,
            mass_inv,
            n_accepted: 0,
            n_proposed: 0,
        })
    }

    /// Set custom mass matrix
    pub fn with_mass_matrix(mut self, massmatrix: Array2<f64>) -> Result<Self> {
        checkarray_finite(&massmatrix, "massmatrix")?;

        if massmatrix.nrows() != self.position.len() || massmatrix.ncols() != self.position.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "massmatrix shape ({}, {}) must be ({}, {})",
                massmatrix.nrows(),
                massmatrix.ncols(),
                self.position.len(),
                self.position.len()
            )));
        }

        // Compute inverse
        let mass_inv = scirs2_linalg::inv(&massmatrix.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to invert mass matrix: {}", e))
        })?;

        self.mass_matrix = massmatrix;
        self.mass_inv = mass_inv;
        Ok(self)
    }

    /// Perform one HMC step
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Array1<f64>> {
        let _dim = self.position.len();

        // Sample momentum from N(0, M)
        let momentum = self.sample_momentum(rng)?;

        // Store initial state
        let initial_position = self.position.clone();
        let initial_momentum = momentum.clone();
        let initial_log_density = self.current_log_density;

        // Perform leapfrog integration
        let (final_position, final_momentum) = self.leapfrog(initial_position.clone(), momentum)?;

        // Compute Hamiltonian for initial and final states
        let initial_hamiltonian =
            -initial_log_density + 0.5 * self.kinetic_energy(&initial_momentum);
        let final_log_density = self.target.log_density(&final_position);
        let final_hamiltonian = -final_log_density + 0.5 * self.kinetic_energy(&final_momentum);

        // Metropolis acceptance step
        let log_alpha = -(final_hamiltonian - initial_hamiltonian);
        let u: f64 = rng.random();

        self.n_proposed += 1;

        if u.ln() < log_alpha {
            // Accept
            self.position = final_position;
            self.current_log_density = final_log_density;
            self.n_accepted += 1;
        }
        // If rejected, keep current position

        Ok(self.position.clone())
    }

    /// Sample momentum from N(0, M)
    fn sample_momentum<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Array1<f64>> {
        let dim = self.position.len();
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            StatsError::ComputationError(format!("Failed to create normal distribution: {}", e))
        })?;

        // Sample from standard normal
        let z = Array1::from_shape_fn(dim, |_| normal.sample(rng));

        // Transform to p ~ N(0, M) using Cholesky decomposition
        // For simplicity, assume diagonal mass matrix
        let mut momentum = Array1::zeros(dim);
        for i in 0..dim {
            momentum[i] = z[i] * self.mass_matrix[[i, i]].sqrt();
        }

        Ok(momentum)
    }

    /// Compute kinetic energy: 0.5 * p^T * M^{-1} * p
    fn kinetic_energy(&self, momentum: &Array1<f64>) -> f64 {
        // For diagonal mass matrix, this simplifies
        let mut energy = 0.0;
        for i in 0..momentum.len() {
            energy += momentum[i] * momentum[i] * self.mass_inv[[i, i]];
        }
        0.5 * energy
    }

    /// Leapfrog integration
    fn leapfrog(
        &self,
        mut position: Array1<f64>,
        mut momentum: Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        // Initial half step for momentum
        let gradient = self.target.gradient(&position);
        momentum = momentum + 0.5 * self.stepsize * gradient;

        // Alternating full steps
        for _ in 0..self.n_steps {
            // Full step for position
            let momentum_update = self.mass_inv.dot(&momentum);
            position = position + self.stepsize * momentum_update;

            // Full step for momentum (except last iteration)
            if self.n_steps > 1 {
                let gradient = self.target.gradient(&position);
                momentum = momentum + self.stepsize * gradient;
            }
        }

        // Final half step for momentum
        let gradient = self.target.gradient(&position);
        momentum = momentum + 0.5 * self.stepsize * gradient;

        // Negate momentum for reversibility
        momentum = -momentum;

        Ok((position, momentum))
    }

    /// Sample multiple states
    pub fn sample<R: Rng + ?Sized>(
        &mut self,
        n_samples_: usize,
        rng: &mut R,
    ) -> Result<Array2<f64>> {
        let dim = self.position.len();
        let mut samples = Array2::zeros((n_samples_, dim));

        for i in 0..n_samples_ {
            let sample = self.step(rng)?;
            samples.row_mut(i).assign(&sample);
        }

        Ok(samples)
    }

    /// Sample with burn-in
    pub fn sample_with_burnin<R: Rng + ?Sized>(
        &mut self,
        n_samples_: usize,
        burnin: usize,
        rng: &mut R,
    ) -> Result<Array2<f64>> {
        check_positive(burnin, "burnin")?;

        // Burn-in
        for _ in 0..burnin {
            self.step(rng)?;
        }

        // Reset counters after burn-in
        self.reset_counters();

        // Collect _samples
        self.sample(n_samples_, rng)
    }

    /// Get acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        if self.n_proposed == 0 {
            0.0
        } else {
            self.n_accepted as f64 / self.n_proposed as f64
        }
    }

    /// Reset acceptance counters
    pub fn reset_counters(&mut self) {
        self.n_accepted = 0;
        self.n_proposed = 0;
    }
}

/// No-U-Turn Sampler (NUTS) - adaptive version of HMC
pub struct NoUTurnSampler<T: DifferentiableTarget> {
    /// Base HMC sampler
    pub hmc: HamiltonianMonteCarlo<T>,
    /// Maximum tree depth
    pub max_tree_depth: usize,
    /// Target acceptance probability
    pub target_accept_prob: f64,
    /// Step size adaptation parameters
    pub stepsize_adaptation: DualAveragingAdaptation,
}

/// Dual averaging adaptation for step size
#[derive(Debug, Clone)]
pub struct DualAveragingAdaptation {
    /// Target acceptance probability
    pub target: f64,
    /// Shrinkage target for log step size
    pub gamma: f64,
    /// Relaxation exponent
    pub t0: f64,
    /// Adaptation rate
    pub kappa: f64,
    /// Current iteration
    pub iteration: usize,
    /// Log step size average
    pub log_step_avg: f64,
    /// H statistic accumulator
    pub h_avg: f64,
}

impl DualAveragingAdaptation {
    /// Create new dual averaging adaptation
    pub fn new(target: f64, initial_logstep: f64) -> Self {
        Self {
            target,
            gamma: 0.05,
            t0: 10.0,
            kappa: 0.75,
            iteration: 0,
            log_step_avg: initial_logstep,
            h_avg: 0.0,
        }
    }

    /// Update step size based on acceptance probability
    pub fn update(&mut self, alpha: f64) -> f64 {
        self.iteration += 1;
        let m = self.iteration as f64;

        // Update H statistic
        self.h_avg =
            (1.0 - 1.0 / (m + self.t0)) * self.h_avg + (self.target - alpha) / (m + self.t0);

        // Update log step size
        let log_step = self.log_step_avg - self.h_avg / (self.gamma * m.powf(self.kappa));

        // Update average
        let weight = m.powf(-self.kappa);
        self.log_step_avg = (1.0 - weight) * self.log_step_avg + weight * log_step;

        log_step.exp()
    }
}

impl<T: DifferentiableTarget> NoUTurnSampler<T> {
    /// Create new NUTS sampler
    pub fn new(target: T, initial: Array1<f64>, initial_stepsize: f64) -> Result<Self> {
        let hmc = HamiltonianMonteCarlo::new(target, initial, initial_stepsize, 1)?;
        let stepsize_adaptation = DualAveragingAdaptation::new(0.8, initial_stepsize.ln());

        Ok(Self {
            hmc,
            max_tree_depth: 10,
            target_accept_prob: 0.8,
            stepsize_adaptation,
        })
    }

    /// Perform one NUTS step
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Array1<f64>> {
        // Sample momentum
        let momentum = self.hmc.sample_momentum(rng)?;

        // Build tree and sample
        let (new_position, alpha) =
            self.build_tree(self.hmc.position.clone(), momentum, 0.0, 1, rng)?;

        // Update step size during adaptation
        let new_stepsize = self.stepsize_adaptation.update(alpha);
        self.hmc.stepsize = new_stepsize;

        // Update position if different
        if !new_position
            .iter()
            .zip(self.hmc.position.iter())
            .all(|(a, b)| (a - b).abs() < f64::EPSILON)
        {
            self.hmc.position = new_position;
            self.hmc.current_log_density = self.hmc.target.log_density(&self.hmc.position);
            self.hmc.n_accepted += 1;
        }

        self.hmc.n_proposed += 1;
        Ok(self.hmc.position.clone())
    }

    /// Build tree for NUTS algorithm (simplified version)
    fn build_tree<R: Rng + ?Sized>(
        &self,
        position: Array1<f64>,
        momentum: Array1<f64>,
        log_u: f64,
        depth: usize,
        rng: &mut R,
    ) -> Result<(Array1<f64>, f64)> {
        if depth >= self.max_tree_depth {
            // Base case: return input position with low acceptance
            return Ok((position, 0.0));
        }

        // Perform leapfrog step
        let (new_position, new_momentum) = self.hmc.leapfrog(position.clone(), momentum.clone())?;

        // Compute log probability for new state
        let new_log_density = self.hmc.target.log_density(&new_position);
        let new_hamiltonian = -new_log_density + 0.5 * self.hmc.kinetic_energy(&new_momentum);

        // Check if proposal is acceptable
        let current_hamiltonian =
            -self.hmc.current_log_density + 0.5 * self.hmc.kinetic_energy(&momentum);
        let log_alpha = -(new_hamiltonian - current_hamiltonian);
        let alpha = log_alpha.exp().min(1.0);

        if log_u <= log_alpha {
            Ok((new_position, alpha))
        } else {
            Ok((position, alpha))
        }
    }

    /// Sample with adaptation
    pub fn sample_adaptive<R: Rng + ?Sized>(
        &mut self,
        n_samples_: usize,
        n_adapt: usize,
        rng: &mut R,
    ) -> Result<Array2<f64>> {
        // Adaptation phase
        for _ in 0..n_adapt {
            self.step(rng)?;
        }

        // Reset counters
        self.hmc.reset_counters();

        // Sampling phase
        let dim = self.hmc.position.len();
        let mut samples = Array2::zeros((n_samples_, dim));

        for i in 0..n_samples_ {
            let sample = self.step(rng)?;
            samples.row_mut(i).assign(&sample);
        }

        Ok(samples)
    }
}

// Example implementations

/// Multivariate normal target with gradient
#[derive(Debug, Clone)]
pub struct MultivariateNormalHMC {
    /// Mean vector
    pub mean: Array1<f64>,
    /// Precision matrix
    pub precision: Array2<f64>,
    /// Log normalizing constant
    pub log_norm_const: f64,
}

impl MultivariateNormalHMC {
    /// Create new multivariate normal target
    pub fn new(mean: Array1<f64>, covariance: Array2<f64>) -> Result<Self> {
        checkarray_finite(&mean, "mean")?;
        checkarray_finite(&covariance, "covariance")?;

        if covariance.nrows() != mean.len() || covariance.ncols() != mean.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "covariance shape ({}, {}) must be ({}, {})",
                covariance.nrows(),
                covariance.ncols(),
                mean.len(),
                mean.len()
            )));
        }

        let precision = scirs2_linalg::inv(&covariance.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to invert covariance: {}", e))
        })?;

        let det = scirs2_linalg::det(&covariance.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to compute determinant: {}", e))
        })?;

        if det <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "Covariance must be positive definite".to_string(),
            ));
        }

        let d = mean.len() as f64;
        let log_norm_const = -0.5 * (d * (2.0 * std::f64::consts::PI).ln() + det.ln());

        Ok(Self {
            mean,
            precision,
            log_norm_const,
        })
    }
}

impl DifferentiableTarget for MultivariateNormalHMC {
    fn log_density(&self, x: &Array1<f64>) -> f64 {
        let diff = x - &self.mean;
        let quad_form = diff.dot(&self.precision.dot(&diff));
        self.log_norm_const - 0.5 * quad_form
    }

    fn gradient(&self, x: &Array1<f64>) -> Array1<f64> {
        let diff = x - &self.mean;
        -self.precision.dot(&diff)
    }

    fn dim(&self) -> usize {
        self.mean.len()
    }

    fn log_density_and_gradient(&self, x: &Array1<f64>) -> (f64, Array1<f64>) {
        let diff = x - &self.mean;
        let quad_form = diff.dot(&self.precision.dot(&diff));
        let log_density = self.log_norm_const - 0.5 * quad_form;
        let gradient = -self.precision.dot(&diff);
        (log_density, gradient)
    }
}

/// Custom differentiable target from functions
pub struct CustomDifferentiableTarget<F, G> {
    /// Log density function
    pub log_density_fn: F,
    /// Gradient function
    pub gradient_fn: G,
    /// Dimensionality
    pub dim: usize,
}

impl<F, G> CustomDifferentiableTarget<F, G> {
    /// Create new custom target
    pub fn new(dim: usize, log_density_fn: F, gradientfn: G) -> Result<Self> {
        check_positive(dim, "dim")?;
        Ok(Self {
            log_density_fn,
            gradient_fn: gradientfn,
            dim,
        })
    }
}

impl<F, G> DifferentiableTarget for CustomDifferentiableTarget<F, G>
where
    F: Fn(&Array1<f64>) -> f64 + Send + Sync,
    G: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync,
{
    fn log_density(&self, x: &Array1<f64>) -> f64 {
        (self.log_density_fn)(x)
    }

    fn gradient(&self, x: &Array1<f64>) -> Array1<f64> {
        (self.gradient_fn)(x)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}
