//! Metropolis-Hastings algorithm for MCMC sampling

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Distribution, Uniform};
use scirs2_core::validation::*;
use scirs2_linalg::{det, inv};
use std::fmt::Debug;

/// Target distribution trait for MCMC sampling
pub trait TargetDistribution: Send + Sync {
    /// Compute the log probability density at a given point
    fn log_density(&self, x: &Array1<f64>) -> f64;

    /// Get the dimensionality of the distribution
    fn dim(&self) -> usize;
}

/// Proposal distribution trait for Metropolis-Hastings
pub trait ProposalDistribution: Send + Sync {
    /// Sample a new proposal given the current state
    fn sample<R: scirs2_core::random::Rng + ?Sized>(
        &self,
        current: &Array1<f64>,
        rng: &mut R,
    ) -> Array1<f64>;

    /// Compute the log density ratio q(x|y) / q(y|x) for asymmetric proposals
    fn log_ratio(from: &Array1<f64>, to: &Array1<f64>) -> f64 {
        0.0 // Default _to symmetric proposal
    }
}

/// Random walk proposal with normal distribution
#[derive(Debug, Clone)]
pub struct RandomWalkProposal {
    /// Step size (standard deviation)
    pub stepsize: f64,
}

impl RandomWalkProposal {
    /// Create a new random walk proposal
    pub fn new(stepsize: f64) -> Result<Self> {
        check_positive(stepsize, "stepsize")?;
        Ok(Self { stepsize })
    }
}

impl ProposalDistribution for RandomWalkProposal {
    fn sample<R: scirs2_core::random::Rng + ?Sized>(
        &self,
        current: &Array1<f64>,
        rng: &mut R,
    ) -> Array1<f64> {
        use scirs2_core::random::Normal;
        let normal = Normal::new(0.0, self.stepsize).unwrap();
        current + Array1::from_shape_fn(current.len(), |_| normal.sample(rng))
    }
}

/// Metropolis-Hastings sampler
pub struct MetropolisHastings<T: TargetDistribution, P: ProposalDistribution> {
    /// Target distribution to sample from
    pub target: T,
    /// Proposal distribution
    pub proposal: P,
    /// Current state
    pub current: Array1<f64>,
    /// Current log density
    pub current_log_density: f64,
    /// Number of accepted proposals
    pub n_accepted: usize,
    /// Total number of proposals
    pub n_proposed: usize,
}

impl<T: TargetDistribution, P: ProposalDistribution> MetropolisHastings<T, P> {
    /// Create a new Metropolis-Hastings sampler
    pub fn new(target: T, proposal: P, initial: Array1<f64>) -> Result<Self> {
        checkarray_finite(&initial, "initial")?;
        if initial.len() != target.dim() {
            return Err(StatsError::DimensionMismatch(format!(
                "initial dimension ({}) must match _target dimension ({})",
                initial.len(),
                target.dim()
            )));
        }

        let current_log_density = target.log_density(&initial);

        Ok(Self {
            target,
            proposal,
            current: initial,
            current_log_density,
            n_accepted: 0,
            n_proposed: 0,
        })
    }

    /// Perform one step of the Metropolis-Hastings algorithm
    pub fn step<R: scirs2_core::random::Rng + ?Sized>(&mut self, rng: &mut R) -> Array1<f64> {
        // Propose new state
        let proposed = self.proposal.sample(&self.current, rng);
        let proposed_log_density = self.target.log_density(&proposed);

        // Compute acceptance ratio
        let log_ratio = proposed_log_density - self.current_log_density
            + P::log_ratio(&self.current, &proposed);

        // Accept or reject
        self.n_proposed += 1;
        let u: f64 = Uniform::new(0.0, 1.0).unwrap().sample(rng);
        if u.ln() < log_ratio {
            self.current = proposed;
            self.current_log_density = proposed_log_density;
            self.n_accepted += 1;
        }

        self.current.clone()
    }

    /// Sample multiple states from the distribution
    pub fn sample<R: scirs2_core::random::Rng + ?Sized>(
        &mut self,
        nsamples_: usize,
        rng: &mut R,
    ) -> Array2<f64> {
        let dim = self.current.len();
        let mut samples = Array2::zeros((nsamples_, dim));

        for i in 0..nsamples_ {
            let sample = self.step(rng);
            samples.row_mut(i).assign(&sample);
        }

        samples
    }

    /// Sample with thinning to reduce autocorrelation
    pub fn sample_thinned<R: scirs2_core::random::Rng + ?Sized>(
        &mut self,
        n_samples_: usize,
        thin: usize,
        rng: &mut R,
    ) -> Result<Array2<f64>> {
        check_positive(thin, "thin")?;

        let dim = self.current.len();
        let mut samples = Array2::zeros((n_samples_, dim));

        for i in 0..n_samples_ {
            // Take thin steps but only keep the last one
            for _ in 0..thin {
                self.step(rng);
            }
            samples.row_mut(i).assign(&self.current);
        }

        Ok(samples)
    }

    /// Get the acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        if self.n_proposed == 0 {
            0.0
        } else {
            self.n_accepted as f64 / self.n_proposed as f64
        }
    }

    /// Reset counters
    pub fn reset_counters(&mut self) {
        self.n_accepted = 0;
        self.n_proposed = 0;
    }
}

/// Adaptive Metropolis-Hastings that adjusts proposal step size
pub struct AdaptiveMetropolisHastings<T: TargetDistribution> {
    /// Base sampler
    pub sampler: MetropolisHastings<T, RandomWalkProposal>,
    /// Target acceptance rate
    pub target_rate: f64,
    /// Adaptation rate
    pub adaptation_rate: f64,
    /// Minimum step size
    pub min_stepsize: f64,
    /// Maximum step size
    pub max_stepsize: f64,
}

impl<T: TargetDistribution> AdaptiveMetropolisHastings<T> {
    /// Create a new adaptive Metropolis-Hastings sampler
    pub fn new(
        target: T,
        initial: Array1<f64>,
        initial_stepsize: f64,
        target_rate: f64,
    ) -> Result<Self> {
        check_probability(target_rate, "target_rate")?;
        check_positive(initial_stepsize, "initial_stepsize")?;

        let proposal = RandomWalkProposal::new(initial_stepsize)?;
        let sampler = MetropolisHastings::new(target, proposal, initial)?;

        Ok(Self {
            sampler,
            target_rate,
            adaptation_rate: 0.05,
            min_stepsize: 1e-6,
            max_stepsize: 10.0,
        })
    }

    /// Perform one adaptive step
    pub fn step<R: scirs2_core::random::Rng + ?Sized>(&mut self, rng: &mut R) -> Array1<f64> {
        let sample = self.sampler.step(rng);

        // Adapt step size based on acceptance rate
        if self.sampler.n_proposed.is_multiple_of(100) && self.sampler.n_proposed > 0 {
            let current_rate = self.sampler.acceptance_rate();
            let adjustment = 1.0 + self.adaptation_rate * (current_rate - self.target_rate);

            let new_stepsize = (self.sampler.proposal.stepsize * adjustment)
                .max(self.min_stepsize)
                .min(self.max_stepsize);

            self.sampler.proposal.stepsize = new_stepsize;
        }

        sample
    }

    /// Run adaptation phase
    pub fn adapt<R: scirs2_core::random::Rng + ?Sized>(
        &mut self,
        nsteps: usize,
        rng: &mut R,
    ) -> Result<()> {
        check_positive(nsteps, "n_steps")?;

        for _ in 0..nsteps {
            self.step(rng);
        }

        // Reset counters after adaptation
        self.sampler.reset_counters();
        Ok(())
    }
}

// Example implementations for common distributions

/// Multivariate normal target distribution
#[derive(Debug, Clone)]
pub struct MultivariateNormalTarget {
    /// Mean vector
    pub mean: Array1<f64>,
    /// Precision matrix (inverse covariance)
    pub precision: Array2<f64>,
    /// Log normalizing constant
    pub log_norm_const: f64,
}

impl MultivariateNormalTarget {
    /// Create a new multivariate normal target
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

        // Compute precision matrix (inverse of covariance)
        let precision = inv(&covariance.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to invert covariance matrix: {}", e))
        })?;

        // Compute determinant
        let det_value = det(&covariance.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to compute determinant: {}", e))
        })?;

        if det_value <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "Covariance matrix must be positive definite".to_string(),
            ));
        }

        let d = mean.len() as f64;
        let log_norm_const = -0.5 * (d * (2.0 * std::f64::consts::PI).ln() + det_value.ln());

        Ok(Self {
            mean,
            precision,
            log_norm_const,
        })
    }
}

impl TargetDistribution for MultivariateNormalTarget {
    fn log_density(&self, x: &Array1<f64>) -> f64 {
        let diff = x - &self.mean;
        let quad_form = diff.dot(&self.precision.dot(&diff));
        self.log_norm_const - 0.5 * quad_form
    }

    fn dim(&self) -> usize {
        self.mean.len()
    }
}

/// Custom target distribution from a log density function
pub struct CustomTarget<F> {
    /// Log density function
    pub log_density_fn: F,
    /// Dimensionality
    pub dim: usize,
}

impl<F> CustomTarget<F> {
    /// Create a new custom target distribution
    pub fn new(dim: usize, log_densityfn: F) -> Result<Self> {
        check_positive(dim, "dim")?;
        Ok(Self {
            log_density_fn: log_densityfn,
            dim,
        })
    }
}

impl<F> TargetDistribution for CustomTarget<F>
where
    F: Fn(&Array1<f64>) -> f64 + Send + Sync,
{
    fn log_density(&self, x: &Array1<f64>) -> f64 {
        (self.log_density_fn)(x)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}
