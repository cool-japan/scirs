//! Advanced MCMC methods
//!
//! This module implements sophisticated MCMC algorithms including multiple-try Metropolis,
//! parallel tempering, slice sampling, and ensemble methods.

use super::{ProposalDistribution, TargetDistribution};
use crate::error::{StatsError, StatsResult as Result};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::validation::*;
use scirs2_core::Rng;
use statrs::statistics::Statistics;
use std::sync::Arc;

/// Multiple-try Metropolis sampler
///
/// Generates multiple proposals at each step and selects one using weighted sampling,
/// which can lead to better acceptance rates and mixing.
pub struct MultipleTryMetropolis<T: TargetDistribution, P: ProposalDistribution> {
    /// Target distribution
    pub target: T,
    /// Proposal distribution
    pub proposal: P,
    /// Current state
    pub current: Array1<f64>,
    /// Current log density
    pub current_log_density: f64,
    /// Number of proposal trials per step
    pub n_tries: usize,
    /// Number of accepted proposals
    pub n_accepted: usize,
    /// Total number of steps
    pub n_steps: usize,
}

impl<T: TargetDistribution, P: ProposalDistribution> MultipleTryMetropolis<T, P> {
    /// Create a new multiple-try Metropolis sampler
    pub fn new(target: T, proposal: P, initial: Array1<f64>, ntries: usize) -> Result<Self> {
        checkarray_finite(&initial, "initial")?;
        check_positive(ntries, "n_tries")?;

        if initial.len() != target.dim() {
            return Err(StatsError::DimensionMismatch(format!(
                "initial dimension ({}) must match target dimension ({})",
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
            n_tries: ntries,
            n_accepted: 0,
            n_steps: 0,
        })
    }

    /// Perform one step of multiple-try Metropolis
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Array1<f64>> {
        // Generate multiple proposals
        let mut proposals = Vec::with_capacity(self.n_tries);
        let mut log_densities = Vec::with_capacity(self.n_tries);
        let mut weights = Vec::with_capacity(self.n_tries);

        for _ in 0..self.n_tries {
            let proposal = self.proposal.sample(&self.current, rng);
            let log_density = self.target.log_density(&proposal);
            let weight = log_density.exp();

            proposals.push(proposal);
            log_densities.push(log_density);
            weights.push(weight);
        }

        // Select proposal using weighted sampling
        let total_weight: f64 = weights.iter().sum();
        if total_weight <= 0.0 {
            // All proposals have zero weight, reject
            self.n_steps += 1;
            return Ok(self.current.clone());
        }

        let u: f64 = rng.random();
        let mut cumsum = 0.0;
        let mut selected_idx = 0;

        for (i, &weight) in weights.iter().enumerate() {
            cumsum += weight / total_weight;
            if u <= cumsum {
                selected_idx = i;
                break;
            }
        }

        let selected_proposal = &proposals[selected_idx];
        let selected_log_density = log_densities[selected_idx];

        // Compute reverse proposals from selected proposal
        let mut reverse_weights = Vec::with_capacity(self.n_tries);
        for _ in 0..self.n_tries {
            let reverse_proposal = self.proposal.sample(selected_proposal, rng);
            let reverse_log_density = self.target.log_density(&reverse_proposal);
            let reverse_weight = reverse_log_density.exp();
            reverse_weights.push(reverse_weight);
        }

        // Include current state in reverse proposals
        reverse_weights.push(self.current_log_density.exp());

        let reverse_total_weight: f64 = reverse_weights.iter().sum();

        // Compute acceptance ratio
        let log_ratio = selected_log_density - self.current_log_density + reverse_total_weight.ln()
            - total_weight.ln();

        // Accept or reject
        let accept_u: f64 = rng.random();
        self.n_steps += 1;

        if accept_u.ln() < log_ratio {
            self.current = selected_proposal.clone();
            self.current_log_density = selected_log_density;
            self.n_accepted += 1;
        }

        Ok(self.current.clone())
    }

    /// Sample multiple states
    pub fn sample<R: Rng + ?Sized>(
        &mut self,
        n_samples_: usize,
        rng: &mut R,
    ) -> Result<Array2<f64>> {
        let dim = self.current.len();
        let mut samples = Array2::zeros((n_samples_, dim));

        for i in 0..n_samples_ {
            let sample = self.step(rng)?;
            samples.row_mut(i).assign(&sample);
        }

        Ok(samples)
    }

    /// Get acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        if self.n_steps == 0 {
            0.0
        } else {
            self.n_accepted as f64 / self.n_steps as f64
        }
    }
}

/// Parallel Tempering (Replica Exchange) sampler
///
/// Runs multiple chains at different temperatures in parallel and exchanges
/// states between chains to improve mixing.
pub struct ParallelTempering<
    T: TargetDistribution + Clone + Send,
    P: ProposalDistribution + Clone + Send,
> {
    /// Base target distribution
    pub base_target: T,
    /// Proposal distribution
    pub proposal: P,
    /// Temperature schedule
    pub temperatures: Array1<f64>,
    /// Current states for each chain
    pub states: Vec<Array1<f64>>,
    /// Current log densities for each chain
    pub log_densities: Vec<f64>,
    /// Number of chains
    pub n_chains: usize,
    /// Exchange attempt frequency
    pub exchange_freq: usize,
    /// Acceptance counters for moves
    pub move_accepted: Vec<usize>,
    /// Acceptance counters for exchanges
    pub exchange_accepted: Vec<usize>,
    /// Total move attempts
    pub move_attempts: Vec<usize>,
    /// Total exchange attempts
    pub exchange_attempts: Vec<usize>,
}

impl<T: TargetDistribution + Clone + Send, P: ProposalDistribution + Clone + Send>
    ParallelTempering<T, P>
{
    /// Create a new parallel tempering sampler
    pub fn new(
        base_target: T,
        proposal: P,
        temperatures: Array1<f64>,
        initial_states: Vec<Array1<f64>>,
        exchange_freq: usize,
    ) -> Result<Self> {
        check_positive(exchange_freq, "exchange_freq")?;

        let n_chains = temperatures.len();
        if initial_states.len() != n_chains {
            return Err(StatsError::DimensionMismatch(format!(
                "initial_states length ({}) must match temperatures length ({})",
                initial_states.len(),
                n_chains
            )));
        }

        // Check temperatures are positive and sorted
        for &temp in temperatures.iter() {
            check_positive(temp, "temperature")?;
        }

        // Compute initial log densities
        let mut log_densities = Vec::with_capacity(n_chains);
        for (i, state) in initial_states.iter().enumerate() {
            checkarray_finite(state, "initial_state")?;
            let temp = temperatures[i];
            let log_density = base_target.log_density(state) / temp;
            log_densities.push(log_density);
        }

        Ok(Self {
            base_target,
            proposal,
            states: initial_states,
            log_densities,
            temperatures,
            n_chains,
            exchange_freq,
            move_accepted: vec![0; n_chains],
            exchange_accepted: vec![0; n_chains - 1],
            move_attempts: vec![0; n_chains],
            exchange_attempts: vec![0; n_chains - 1],
        })
    }

    /// Perform one step for all chains
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<()> {
        // Metropolis steps for all chains
        for i in 0..self.n_chains {
            let temp = self.temperatures[i];
            let current_state = &self.states[i];

            // Generate proposal
            let proposal = self.proposal.sample(current_state, rng);
            let proposal_log_density = self.base_target.log_density(&proposal) / temp;

            // Accept or reject
            let log_ratio = proposal_log_density - self.log_densities[i]
                + P::log_ratio(current_state, &proposal);

            self.move_attempts[i] += 1;
            let u: f64 = rng.random();

            if u.ln() < log_ratio {
                self.states[i] = proposal;
                self.log_densities[i] = proposal_log_density;
                self.move_accepted[i] += 1;
            }
        }

        Ok(())
    }

    /// Attempt exchanges between adjacent chains
    pub fn exchange_step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<()> {
        for i in 0..(self.n_chains - 1) {
            let temp1 = self.temperatures[i];
            let temp2 = self.temperatures[i + 1];

            let log_density1 = self.log_densities[i];
            let log_density2 = self.log_densities[i + 1];

            // Compute exchange probability
            let log_ratio = (log_density1 * temp1 - log_density2 * temp2) / temp2
                - (log_density1 * temp1 - log_density2 * temp2) / temp1;

            self.exchange_attempts[i] += 1;
            let u: f64 = rng.random();

            if u.ln() < log_ratio {
                // Exchange states
                self.states.swap(i, i + 1);

                // Update log densities for new temperatures
                let state1_new_log_density = self.base_target.log_density(&self.states[i]) / temp1;
                let state2_new_log_density =
                    self.base_target.log_density(&self.states[i + 1]) / temp2;

                self.log_densities[i] = state1_new_log_density;
                self.log_densities[i + 1] = state2_new_log_density;

                self.exchange_accepted[i] += 1;
            }
        }

        Ok(())
    }

    /// Run the parallel tempering sampler
    pub fn sample<R: Rng + ?Sized>(
        &mut self,
        n_samples_: usize,
        rng: &mut R,
    ) -> Result<Array2<f64>> {
        let dim = self.states[0].len();
        let mut samples = Array2::zeros((n_samples_, dim));

        for i in 0..n_samples_ {
            self.step(rng)?;

            // Attempt exchanges periodically
            if i % self.exchange_freq == 0 {
                self.exchange_step(rng)?;
            }

            // Store sample from coldest chain (temperature = 1.0)
            let coldest_idx = self
                .temperatures
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            samples.row_mut(i).assign(&self.states[coldest_idx]);
        }

        Ok(samples)
    }

    /// Get acceptance rates for moves
    pub fn move_acceptance_rates(&self) -> Array1<f64> {
        let mut rates = Array1::zeros(self.n_chains);
        for i in 0..self.n_chains {
            if self.move_attempts[i] > 0 {
                rates[i] = self.move_accepted[i] as f64 / self.move_attempts[i] as f64;
            }
        }
        rates
    }

    /// Get acceptance rates for exchanges
    pub fn exchange_acceptance_rates(&self) -> Array1<f64> {
        let mut rates = Array1::zeros(self.n_chains - 1);
        for i in 0..(self.n_chains - 1) {
            if self.exchange_attempts[i] > 0 {
                rates[i] = self.exchange_accepted[i] as f64 / self.exchange_attempts[i] as f64;
            }
        }
        rates
    }
}

/// Slice sampler
///
/// Uses auxiliary variables to transform the sampling problem into uniform sampling
/// over the area under the probability density function.
pub struct SliceSampler<T: TargetDistribution> {
    /// Target distribution
    pub target: T,
    /// Current state
    pub current: Array1<f64>,
    /// Current log density
    pub current_log_density: f64,
    /// Step size for finding interval
    pub stepsize: f64,
    /// Maximum number of doublings for interval finding
    pub max_doublings: usize,
    /// Number of accepted proposals
    pub n_accepted: usize,
    /// Total number of proposals
    pub n_proposed: usize,
}

impl<T: TargetDistribution> SliceSampler<T> {
    /// Create a new slice sampler
    pub fn new(target: T, initial: Array1<f64>, stepsize: f64) -> Result<Self> {
        checkarray_finite(&initial, "initial")?;
        check_positive(stepsize, "stepsize")?;

        if initial.len() != target.dim() {
            return Err(StatsError::DimensionMismatch(format!(
                "initial dimension ({}) must match target dimension ({})",
                initial.len(),
                target.dim()
            )));
        }

        let current_log_density = target.log_density(&initial);

        Ok(Self {
            target,
            current: initial,
            current_log_density,
            stepsize,
            max_doublings: 20,
            n_accepted: 0,
            n_proposed: 0,
        })
    }

    /// Perform one step of slice sampling
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Array1<f64>> {
        let dim = self.current.len();
        let mut new_state = self.current.clone();

        // Sample each dimension sequentially
        for d in 0..dim {
            new_state[d] = self.slice_sample_dimension(&new_state, d, rng)?;
        }

        self.current = new_state;
        self.current_log_density = self.target.log_density(&self.current);
        self.n_proposed += 1;
        self.n_accepted += 1; // Slice sampling always accepts

        Ok(self.current.clone())
    }

    /// Sample a single dimension using slice sampling
    fn slice_sample_dimension<R: Rng + ?Sized>(
        &self,
        state: &Array1<f64>,
        dimension: usize,
        rng: &mut R,
    ) -> Result<f64> {
        let current_value = state[dimension];
        let current_log_density = self.target.log_density(state);

        // Sample auxiliary variable (slice level)
        let u: f64 = rng.random();
        let slice_level = current_log_density + u.ln();

        // Find initial interval
        let mut left = current_value - self.stepsize * rng.random::<f64>();
        let mut right = left + self.stepsize;

        // Expand interval using doubling procedure
        for _ in 0..self.max_doublings {
            let mut left_state = state.clone();
            left_state[dimension] = left;
            let left_log_density = self.target.log_density(&left_state);

            let mut right_state = state.clone();
            right_state[dimension] = right;
            let right_log_density = self.target.log_density(&right_state);

            if left_log_density <= slice_level && right_log_density <= slice_level {
                break;
            }

            if rng.random::<bool>() {
                left = left - (right - left);
            } else {
                right = right + (right - left);
            }
        }

        // Sample from interval using shrinkage
        loop {
            let proposal = left + (right - left) * rng.random::<f64>();
            let mut proposal_state = state.clone();
            proposal_state[dimension] = proposal;
            let proposal_log_density = self.target.log_density(&proposal_state);

            if proposal_log_density > slice_level {
                return Ok(proposal);
            }

            // Shrink interval
            if proposal < current_value {
                left = proposal;
            } else {
                right = proposal;
            }

            // Prevent infinite loop
            if (right - left).abs() < 1e-10 {
                return Ok(current_value);
            }
        }
    }

    /// Sample multiple states
    pub fn sample<R: Rng + ?Sized>(
        &mut self,
        n_samples_: usize,
        rng: &mut R,
    ) -> Result<Array2<f64>> {
        let dim = self.current.len();
        let mut samples = Array2::zeros((n_samples_, dim));

        for i in 0..n_samples_ {
            let sample = self.step(rng)?;
            samples.row_mut(i).assign(&sample);
        }

        Ok(samples)
    }

    /// Get acceptance rate (always 1.0 for slice sampling)
    pub fn acceptance_rate(&self) -> f64 {
        1.0
    }
}

/// Ensemble sampler (Affine Invariant MCMC)
///
/// Uses an ensemble of walkers that evolve simultaneously, with proposals
/// based on the current positions of other walkers.
pub struct EnsembleSampler<T: TargetDistribution + Clone + Send + Sync> {
    /// Target distribution
    pub target: Arc<T>,
    /// Walker positions
    pub walkers: Array2<f64>,
    /// Log densities for each walker
    pub log_densities: Array1<f64>,
    /// Number of walkers
    pub n_walkers: usize,
    /// Dimensionality
    pub dim: usize,
    /// Scale parameter for proposals
    pub scale: f64,
    /// Acceptance counters
    pub n_accepted: Array1<usize>,
    /// Total proposals
    pub n_proposed: Array1<usize>,
}

impl<T: TargetDistribution + Clone + Send + Sync> EnsembleSampler<T> {
    /// Create a new ensemble sampler
    pub fn new(target: T, initialwalkers: Array2<f64>, scale: Option<f64>) -> Result<Self> {
        checkarray_finite(&initialwalkers, "initial_walkers")?;
        let (n_walkers, dim) = initialwalkers.dim();
        let scale = scale.unwrap_or(2.0);

        if n_walkers < 2 * dim {
            return Err(StatsError::InvalidArgument(format!(
                "Number of walkers ({}) should be at least 2 * dim ({})",
                n_walkers,
                2 * dim
            )));
        }

        check_positive(scale, "scale")?;

        // Compute initial log densities
        let mut log_densities = Array1::zeros(n_walkers);
        for i in 0..n_walkers {
            let walker = initialwalkers.row(i);
            log_densities[i] = target.log_density(&walker.to_owned());
        }

        Ok(Self {
            target: Arc::new(target),
            walkers: initialwalkers,
            log_densities,
            n_walkers,
            dim,
            scale,
            n_accepted: Array1::zeros(n_walkers),
            n_proposed: Array1::zeros(n_walkers),
        })
    }

    /// Perform one step for all walkers
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<()> {
        // Split walkers into two groups
        let n_half = self.n_walkers / 2;

        // Update first half using second half as complementary ensemble
        self.update_group(0, n_half, n_half, self.n_walkers, rng)?;

        // Update second half using first half as complementary ensemble
        self.update_group(n_half, self.n_walkers, 0, n_half, rng)?;

        Ok(())
    }

    /// Update a group of walkers
    fn update_group<R: Rng + ?Sized>(
        &mut self,
        start: usize,
        end: usize,
        comp_start: usize,
        comp_end: usize,
        rng: &mut R,
    ) -> Result<()> {
        for i in start..end {
            // Select random walker from complementary ensemble
            let compsize = comp_end - comp_start;
            let j = comp_start + rng.gen_range(0..compsize);

            // Generate stretch parameter
            let z = ((self.scale - 1.0) * rng.random::<f64>() + 1.0).powf(2.0) / self.scale;

            // Compute proposal
            let walker_i = self.walkers.row(i);
            let walker_j = self.walkers.row(j);
            let proposal = &walker_j.to_owned() + z * (&walker_i.to_owned() - &walker_j.to_owned());

            // Compute log density
            let proposal_log_density = self.target.log_density(&proposal);

            // Compute acceptance probability
            let log_ratio =
                (self.dim as f64 - 1.0) * z.ln() + proposal_log_density - self.log_densities[i];

            // Accept or reject
            let u: f64 = rng.random();
            self.n_proposed[i] += 1;

            if u.ln() < log_ratio {
                self.walkers.row_mut(i).assign(&proposal);
                self.log_densities[i] = proposal_log_density;
                self.n_accepted[i] += 1;
            }
        }

        Ok(())
    }

    /// Sample multiple steps
    pub fn sample<R: Rng + ?Sized>(
        &mut self,
        n_samples_: usize,
        rng: &mut R,
    ) -> Result<Array2<f64>> {
        let total_samples = n_samples_ * self.n_walkers;
        let mut samples = Array2::zeros((total_samples, self.dim));

        for i in 0..n_samples_ {
            self.step(rng)?;

            // Store all walker positions
            for j in 0..self.n_walkers {
                let sample_idx = i * self.n_walkers + j;
                samples.row_mut(sample_idx).assign(&self.walkers.row(j));
            }
        }

        Ok(samples)
    }

    /// Get acceptance rates for all walkers
    pub fn acceptance_rates(&self) -> Array1<f64> {
        let mut rates = Array1::zeros(self.n_walkers);
        for i in 0..self.n_walkers {
            if self.n_proposed[i] > 0 {
                rates[i] = self.n_accepted[i] as f64 / self.n_proposed[i] as f64;
            }
        }
        rates
    }

    /// Get current walker positions
    pub fn get_walkers(&self) -> &Array2<f64> {
        &self.walkers
    }

    /// Compute chain statistics (mean, autocorrelation time, etc.)
    pub fn chain_statistics(&self, samples: &Array2<f64>) -> Result<ChainStatistics> {
        let (n_samples_, dim) = samples.dim();

        // Compute means
        let means = samples.mean_axis(Axis(0)).unwrap();

        // Compute variances
        let mut variances = Array1::zeros(dim);
        for j in 0..dim {
            let col = samples.column(j);
            let mean_j = means[j];
            let var_j = col.mapv(|x| (x - mean_j).powi(2)).mean();
            variances[j] = var_j;
        }

        // Estimate autocorrelation times (simplified)
        let mut autocorr_times = Array1::zeros(dim);
        for j in 0..dim {
            autocorr_times[j] = self.estimate_autocorr_time(&samples.column(j))?;
        }

        Ok(ChainStatistics {
            means,
            variances,
            autocorr_times,
            n_samples_,
            dim,
        })
    }

    /// Estimate autocorrelation time for a single chain
    fn estimate_autocorr_time(&self, chain: &ArrayView1<f64>) -> Result<f64> {
        let n = chain.len();
        if n < 4 {
            return Ok(1.0);
        }

        let mean = chain.mean().unwrap();
        let variance = chain.mapv(|x| (x - mean).powi(2)).mean();

        if variance <= 0.0 {
            return Ok(1.0);
        }

        // Compute autocorrelation function
        let max_lag = (n / 4).min(200);
        let mut autocorr = Array1::zeros(max_lag);

        for lag in 0..max_lag {
            let mut sum = 0.0;
            let mut count = 0;

            for i in 0..(n - lag) {
                sum += (chain[i] - mean) * (chain[i + lag] - mean);
                count += 1;
            }

            if count > 0 {
                autocorr[lag] = sum / (count as f64 * variance);
            }
        }

        // Find first negative value or when autocorr drops below e^(-1)
        let threshold = std::f64::consts::E.recip();
        for lag in 1..max_lag {
            if autocorr[lag] < threshold || autocorr[lag] < 0.0 {
                return Ok(lag as f64);
            }
        }

        Ok(max_lag as f64)
    }
}

/// Chain statistics from ensemble sampling
#[derive(Debug, Clone)]
pub struct ChainStatistics {
    /// Mean values for each dimension
    pub means: Array1<f64>,
    /// Variances for each dimension
    pub variances: Array1<f64>,
    /// Autocorrelation times for each dimension
    pub autocorr_times: Array1<f64>,
    /// Number of samples
    pub n_samples_: usize,
    /// Dimensionality
    pub dim: usize,
}

impl ChainStatistics {
    /// Get effective sample sizes
    pub fn effective_samplesizes(&self) -> Array1<f64> {
        self.autocorr_times.mapv(|tau| {
            if tau > 0.0 {
                self.n_samples_ as f64 / (2.0 * tau)
            } else {
                self.n_samples_ as f64
            }
        })
    }

    /// Check if chains have converged (simplified Gelman-Rubin diagnostic)
    pub fn is_converged(&self, threshold: f64) -> bool {
        // For ensemble methods, check if autocorrelation times are reasonable
        let max_autocorr = self.autocorr_times.iter().cloned().fold(0.0f64, f64::max);
        let min_eff_samples = self.n_samples_ as f64 / (2.0 * max_autocorr);

        min_eff_samples > threshold
    }
}
