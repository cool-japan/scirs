//! Gibbs sampling for MCMC
//!
//! Gibbs sampling is a MCMC method for sampling from multivariate distributions
//! when direct sampling is difficult but conditional distributions are available.

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::validation::*;
use scirs2_core::Rng;
use std::fmt::Debug;

/// Conditional distribution trait for Gibbs sampling
pub trait ConditionalDistribution: Send + Sync {
    /// Sample from the conditional distribution P(X_i | X_{-i})
    ///
    /// # Arguments
    /// * `current_state` - Current values of all variables
    /// * `variable_index` - Index of the variable to sample
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// New value for the variable at `variable_index`
    fn sample_conditional<R: Rng + ?Sized>(
        &self,
        current_state: &Array1<f64>,
        variable_index: usize,
        rng: &mut R,
    ) -> Result<f64>;

    /// Get the dimensionality of the distribution
    fn dim(&self) -> usize;

    /// Optionally compute log density for monitoring
    fn log_density(&self, x: &Array1<f64>) -> Option<f64> {
        None
    }
}

/// Gibbs sampler
pub struct GibbsSampler<C: ConditionalDistribution> {
    /// Conditional distributions
    pub conditionals: C,
    /// Current state
    pub current: Array1<f64>,
    /// Number of samples generated
    pub n_samples_: usize,
    /// Variable update order (None for sequential, Some for custom order)
    pub update_order: Option<Vec<usize>>,
}

impl<C: ConditionalDistribution> GibbsSampler<C> {
    /// Create a new Gibbs sampler
    pub fn new(conditionals: C, initial: Array1<f64>) -> Result<Self> {
        checkarray_finite(&initial, "initial")?;
        if initial.len() != conditionals.dim() {
            return Err(StatsError::DimensionMismatch(format!(
                "initial dimension ({}) must match conditionals dimension ({})",
                initial.len(),
                conditionals.dim()
            )));
        }

        Ok(Self {
            conditionals,
            current: initial,
            n_samples_: 0,
            update_order: None,
        })
    }

    /// Set custom variable update order
    pub fn with_update_order(mut self, order: Vec<usize>) -> Result<Self> {
        if order.len() != self.conditionals.dim() {
            return Err(StatsError::InvalidArgument(
                "Update order length must match dimension".to_string(),
            ));
        }

        // Check that all indices are valid and unique
        let mut sorted_order = order.clone();
        sorted_order.sort_unstable();
        for (i, &idx) in sorted_order.iter().enumerate() {
            if idx != i {
                return Err(StatsError::InvalidArgument(
                    "Update order must contain each index exactly once".to_string(),
                ));
            }
        }

        self.update_order = Some(order);
        Ok(self)
    }

    /// Perform one full sweep of Gibbs sampling
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Array1<f64>> {
        let dim = self.current.len();

        // Determine update order
        let order = match &self.update_order {
            Some(order) => order.clone(),
            None => (0..dim).collect(),
        };

        // Update each variable in order
        for &var_idx in &order {
            let new_value = self
                .conditionals
                .sample_conditional(&self.current, var_idx, rng)?;
            self.current[var_idx] = new_value;
        }

        self.n_samples_ += 1;
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

    /// Sample with burn-in period
    pub fn sample_with_burnin<R: Rng + ?Sized>(
        &mut self,
        n_samples_: usize,
        burnin: usize,
        rng: &mut R,
    ) -> Result<Array2<f64>> {
        check_positive(burnin, "burnin")?;

        // Burn-in period
        for _ in 0..burnin {
            self.step(rng)?;
        }

        // Collect _samples
        self.sample(n_samples_, rng)
    }

    /// Sample with thinning to reduce autocorrelation
    pub fn sample_thinned<R: Rng + ?Sized>(
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
                self.step(rng)?;
            }
            samples.row_mut(i).assign(&self.current);
        }

        Ok(samples)
    }
}

/// Multivariate normal Gibbs sampler
///
/// For sampling from a multivariate normal distribution where each variable
/// given all others follows a normal distribution.
#[derive(Debug, Clone)]
pub struct MultivariateNormalGibbs {
    /// Mean vector
    pub mean: Array1<f64>,
    /// Precision matrix (inverse covariance)
    pub precision: Array2<f64>,
}

impl MultivariateNormalGibbs {
    /// Create a new multivariate normal Gibbs sampler
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

        // Compute precision matrix
        let precision = scirs2_linalg::inv(&covariance.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to invert covariance: {}", e))
        })?;

        Ok(Self { mean, precision })
    }

    /// Create from precision matrix directly
    pub fn from_precision(mean: Array1<f64>, precision: Array2<f64>) -> Result<Self> {
        checkarray_finite(&mean, "mean")?;
        checkarray_finite(&precision, "precision")?;

        if precision.nrows() != mean.len() || precision.ncols() != mean.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "precision shape ({}, {}) must be ({}, {})",
                precision.nrows(),
                precision.ncols(),
                mean.len(),
                mean.len()
            )));
        }

        Ok(Self { mean, precision })
    }
}

impl ConditionalDistribution for MultivariateNormalGibbs {
    fn sample_conditional<R: Rng + ?Sized>(
        &self,
        current_state: &Array1<f64>,
        variable_index: usize,
        rng: &mut R,
    ) -> Result<f64> {
        let dim = self.mean.len();
        if variable_index >= dim {
            return Err(StatsError::InvalidArgument(format!(
                "variable_index ({}) must be less than dimension ({})",
                variable_index, dim
            )));
        }

        // For multivariate normal, conditional distribution is:
        // X_i | X_{-i} ~ Normal(mu_i + Sigma_{i,-i} * Sigma_{-i,-i}^{-1} * (X_{-i} - mu_{-i}), Sigma_{ii|{-i}})
        // Where Sigma_{ii|{-i}} = 1 / Precision_{ii}

        let precision_ii = self.precision[[variable_index, variable_index]];
        if precision_ii.abs() < f64::EPSILON {
            return Err(StatsError::ComputationError(
                "Precision matrix must have positive diagonal elements".to_string(),
            ));
        }

        // Conditional variance
        let conditional_variance = 1.0 / precision_ii;
        let conditional_std = conditional_variance.sqrt();

        // Conditional mean
        let mut sum = 0.0;
        for j in 0..dim {
            if j != variable_index {
                sum += self.precision[[variable_index, j]] * (current_state[j] - self.mean[j]);
            }
        }
        let conditional_mean = self.mean[variable_index] - sum / precision_ii;

        // Sample from normal distribution
        use scirs2_core::random::{Distribution, Normal};
        let normal = Normal::new(conditional_mean, conditional_std).map_err(|e| {
            StatsError::ComputationError(format!("Failed to create normal distribution: {}", e))
        })?;

        Ok(normal.sample(rng))
    }

    fn dim(&self) -> usize {
        self.mean.len()
    }

    fn log_density(&self, x: &Array1<f64>) -> Option<f64> {
        let diff = x - &self.mean;
        let quad_form = diff.dot(&self.precision.dot(&diff));

        // Compute log determinant of precision
        let det = scirs2_linalg::det(&self.precision.view(), None).ok()?;
        if det <= 0.0 {
            return None;
        }

        let d = self.mean.len() as f64;
        let log_norm_const = 0.5 * (det.ln() - d * (2.0 * std::f64::consts::PI).ln());

        Some(log_norm_const - 0.5 * quad_form)
    }
}

/// Gaussian mixture model Gibbs sampler
///
/// Samples component assignments and parameters for a Gaussian mixture model
#[derive(Debug, Clone)]
pub struct GaussianMixtureGibbs {
    /// Current component means
    pub means: Array2<f64>,
    /// Current component precisions
    pub precisions: Vec<Array2<f64>>,
    /// Current mixing weights
    pub weights: Array1<f64>,
    /// Data points
    pub data: Array2<f64>,
    /// Current component assignments
    pub assignments: Array1<usize>,
    /// Number of components
    pub n_components: usize,
    /// Hyperparameters for priors
    pub prior_mean: Array1<f64>,
    pub prior_precision: Array2<f64>,
    pub prior_alpha: Array1<f64>, // Dirichlet prior for weights
}

impl GaussianMixtureGibbs {
    /// Create a new Gaussian mixture Gibbs sampler
    pub fn new(
        data: Array2<f64>,
        n_components: usize,
        prior_mean: Array1<f64>,
        prior_precision: Array2<f64>,
        prior_alpha: Array1<f64>,
    ) -> Result<Self> {
        checkarray_finite(&data, "data")?;
        check_positive(n_components, "n_components")?;
        checkarray_finite(&prior_mean, "prior_mean")?;
        checkarray_finite(&prior_precision, "prior_precision")?;
        checkarray_finite(&prior_alpha, "prior_alpha")?;

        let (n_samples_, dim) = data.dim();

        if prior_alpha.len() != n_components {
            return Err(StatsError::DimensionMismatch(format!(
                "prior_alpha length ({}) must equal n_components ({})",
                prior_alpha.len(),
                n_components
            )));
        }

        // Initialize parameters
        let means = Array2::zeros((n_components, dim));
        let precisions = vec![Array2::eye(dim); n_components];
        let weights = Array1::from_elem(n_components, 1.0 / n_components as f64);
        let assignments = Array1::zeros(n_samples_);

        Ok(Self {
            means,
            precisions,
            weights,
            data,
            assignments,
            n_components,
            prior_mean,
            prior_precision,
            prior_alpha,
        })
    }

    /// Perform one step of Gibbs sampling for GMM
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<()> {
        // 1. Sample component assignments
        self.sample_assignments(rng)?;

        // 2. Sample component parameters given assignments
        self.sample_parameters(rng)?;

        // 3. Sample mixing weights
        self.sample_weights(rng)?;

        Ok(())
    }

    /// Sample component assignments for each data point
    fn sample_assignments<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<()> {
        for i in 0..self.data.nrows() {
            let data_point = self.data.row(i);
            let mut log_probs = Array1::zeros(self.n_components);

            // Compute log probabilities for each component
            for k in 0..self.n_components {
                let mean_k = self.means.row(k);
                let precision_k = &self.precisions[k];

                let diff = &data_point.to_owned() - &mean_k.to_owned();
                let quad_form = diff.dot(&precision_k.dot(&diff));

                // Log determinant of precision
                let det = scirs2_linalg::det(&precision_k.view(), None).map_err(|e| {
                    StatsError::ComputationError(format!("Failed to compute determinant: {}", e))
                })?;

                if det <= 0.0 {
                    return Err(StatsError::ComputationError(
                        "Precision matrix must be positive definite".to_string(),
                    ));
                }

                let log_likelihood = 0.5 * det.ln() - 0.5 * quad_form;
                log_probs[k] = self.weights[k].ln() + log_likelihood;
            }

            // Convert to probabilities and sample
            let max_log_prob = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut probs = log_probs.mapv(|x| (x - max_log_prob).exp());
            let prob_sum = probs.sum();
            probs /= prob_sum;

            // Sample from categorical distribution
            let u: f64 = rng.random();
            let mut cumsum = 0.0;
            let mut selected = 0;

            for (k, &p) in probs.iter().enumerate() {
                cumsum += p;
                if u <= cumsum {
                    selected = k;
                    break;
                }
            }

            self.assignments[i] = selected;
        }

        Ok(())
    }

    /// Sample component parameters given assignments
    fn sample_parameters<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<()> {
        for k in 0..self.n_components {
            // Find data points assigned to component k
            let assigned_indices: Vec<usize> = self
                .assignments
                .iter()
                .enumerate()
                .filter_map(|(i, &assignment)| if assignment == k { Some(i) } else { None })
                .collect();

            if assigned_indices.is_empty() {
                // No data assigned to this component, sample from prior
                self.sample_from_prior(k, rng)?;
            } else {
                // Sample posterior given assigned data
                self.sample_posterior(k, &assigned_indices, rng)?;
            }
        }

        Ok(())
    }

    /// Sample parameters from prior when no data is assigned
    fn sample_from_prior<R: Rng + ?Sized>(&mut self, component: usize, rng: &mut R) -> Result<()> {
        // Sample mean from prior
        use scirs2_core::random::{Distribution, Normal};

        let dim = self.prior_mean.len();
        let mut new_mean = Array1::zeros(dim);

        // For simplicity, assume diagonal prior precision
        for i in 0..dim {
            let variance = 1.0 / self.prior_precision[[i, i]];
            let std = variance.sqrt();
            let normal = Normal::new(self.prior_mean[i], std).map_err(|e| {
                StatsError::ComputationError(format!("Failed to create normal: {}", e))
            })?;
            new_mean[i] = normal.sample(rng);
        }

        self.means.row_mut(component).assign(&new_mean);

        // For precision, use prior (simplified - in practice would sample from Wishart)
        self.precisions[component] = self.prior_precision.clone();

        Ok(())
    }

    /// Sample posterior parameters given assigned data
    fn sample_posterior<R: Rng + ?Sized>(
        &mut self,
        component: usize,
        assigned_indices: &[usize],
        rng: &mut R,
    ) -> Result<()> {
        let n_assigned = assigned_indices.len();
        let dim = self.prior_mean.len();

        // Compute sample mean
        let mut sample_mean = Array1::zeros(dim);
        for &i in assigned_indices {
            sample_mean = sample_mean + self.data.row(i);
        }
        sample_mean /= n_assigned as f64;

        // Posterior parameters for mean (assuming identity precision for simplicity)
        let posterior_precision = &self.prior_precision + Array2::eye(dim) * n_assigned as f64;
        let posterior_mean = {
            let prior_contrib = self.prior_precision.dot(&self.prior_mean);
            let data_contrib = Array1::from_elem(dim, n_assigned as f64) * &sample_mean;
            let precision_inv =
                scirs2_linalg::inv(&posterior_precision.view(), None).map_err(|e| {
                    StatsError::ComputationError(format!("Failed to invert precision: {}", e))
                })?;
            precision_inv.dot(&(prior_contrib + data_contrib))
        };

        // Sample new mean
        use scirs2_core::random::{Distribution, Normal};
        let mut new_mean = Array1::zeros(dim);

        for i in 0..dim {
            let variance = 1.0 / posterior_precision[[i, i]];
            let std = variance.sqrt();
            let normal = Normal::new(posterior_mean[i], std).map_err(|e| {
                StatsError::ComputationError(format!("Failed to create normal: {}", e))
            })?;
            new_mean[i] = normal.sample(rng);
        }

        self.means.row_mut(component).assign(&new_mean);

        // Update precision (simplified)
        self.precisions[component] = posterior_precision;

        Ok(())
    }

    /// Sample mixing weights from Dirichlet posterior
    fn sample_weights<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<()> {
        // Count assignments to each component
        let mut counts = Array1::<f64>::zeros(self.n_components);
        for &assignment in self.assignments.iter() {
            counts[assignment] += 1.0;
        }

        // Posterior parameters for Dirichlet
        let posterior_alpha = &self.prior_alpha + &counts;

        // Sample from Dirichlet (using Gamma sampling)
        use scirs2_core::random::{Distribution, Gamma};
        let mut gamma_samples = Array1::zeros(self.n_components);

        for k in 0..self.n_components {
            let gamma = Gamma::new(posterior_alpha[k], 1.0).map_err(|e| {
                StatsError::ComputationError(format!("Failed to create Gamma: {}", e))
            })?;
            gamma_samples[k] = gamma.sample(rng);
        }

        // Normalize to get Dirichlet sample
        let sum = gamma_samples.sum();
        self.weights = gamma_samples / sum;

        Ok(())
    }
}

/// Blocked Gibbs sampler for improved efficiency
///
/// Updates blocks of variables together rather than one at a time
pub struct BlockedGibbsSampler<C: ConditionalDistribution> {
    /// Base Gibbs sampler
    pub sampler: GibbsSampler<C>,
    /// Variable blocks (each inner vec contains indices of variables to update together)
    pub blocks: Vec<Vec<usize>>,
}

impl<C: ConditionalDistribution> BlockedGibbsSampler<C> {
    /// Create a new blocked Gibbs sampler
    pub fn new(conditionals: C, initial: Array1<f64>, blocks: Vec<Vec<usize>>) -> Result<Self> {
        let sampler = GibbsSampler::new(conditionals, initial)?;

        // Validate blocks
        let dim = sampler.conditionals.dim();
        let mut all_indices = Vec::new();
        for block in &blocks {
            for &idx in block {
                if idx >= dim {
                    return Err(StatsError::InvalidArgument(format!(
                        "Block index {} exceeds dimension {}",
                        idx, dim
                    )));
                }
                all_indices.push(idx);
            }
        }

        all_indices.sort_unstable();
        all_indices.dedup();
        if all_indices.len() != dim {
            return Err(StatsError::InvalidArgument(
                "Blocks must cover all variables exactly once".to_string(),
            ));
        }

        Ok(Self { sampler, blocks })
    }

    /// Perform one step of blocked Gibbs sampling
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Result<Array1<f64>> {
        // Update each block
        for block in &self.blocks {
            for &var_idx in block {
                let new_value = self.sampler.conditionals.sample_conditional(
                    &self.sampler.current,
                    var_idx,
                    rng,
                )?;
                self.sampler.current[var_idx] = new_value;
            }
        }

        self.sampler.n_samples_ += 1;
        Ok(self.sampler.current.clone())
    }

    /// Sample multiple states
    pub fn sample<R: Rng + ?Sized>(
        &mut self,
        n_samples_: usize,
        rng: &mut R,
    ) -> Result<Array2<f64>> {
        let dim = self.sampler.current.len();
        let mut samples = Array2::zeros((n_samples_, dim));

        for i in 0..n_samples_ {
            let sample = self.step(rng)?;
            samples.row_mut(i).assign(&sample);
        }

        Ok(samples)
    }
}
