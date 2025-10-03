//! Hierarchical Bayesian models
//!
//! This module implements hierarchical (multi-level) Bayesian models that allow
//! for group-level variation and borrowing of strength across groups.

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{Distribution, Gamma, Normal};
use scirs2_core::validation::*;
use scirs2_core::Rng;
use statrs::statistics::Statistics;

/// Hierarchical linear model with random intercepts and slopes
///
/// Model structure:
/// Level 1: y_ij = β₀j + β₁j * x_ij + ε_ij,  ε_ij ~ N(0, σ²)
/// Level 2: β₀j = γ₀₀ + γ₀₁ * w_j + u₀j,     u₀j ~ N(0, τ₀₀)
///          β₁j = γ₁₀ + γ₁₁ * w_j + u₁j,     u₁j ~ N(0, τ₁₁)
#[derive(Debug, Clone)]
pub struct HierarchicalLinearModel {
    /// Fixed effects parameters
    pub fixed_effects: Array2<f64>,
    /// Random effects covariance matrix
    pub random_effects_cov: Array2<f64>,
    /// Residual variance
    pub residual_variance: f64,
    /// Group identifiers
    pub groups: Array1<usize>,
    /// Number of groups
    pub n_groups: usize,
    /// Number of level-1 predictors
    pub n_level1_predictors: usize,
    /// Number of level-2 predictors
    pub n_level2_predictors: usize,
    /// Whether to include random slopes
    pub random_slopes: bool,
}

impl HierarchicalLinearModel {
    /// Create a new hierarchical linear model
    pub fn new(
        n_groups: usize,
        n_level1_predictors: usize,
        n_level2_predictors: usize,
        random_slopes: bool,
    ) -> Result<Self> {
        check_positive(n_groups, "n_groups")?;
        check_positive(n_level1_predictors, "n_level1_predictors")?;

        let n_random_effects = if random_slopes {
            n_level1_predictors + 1
        } else {
            1
        };
        let fixed_effects = Array2::zeros((n_random_effects, n_level2_predictors + 1));
        let random_effects_cov = Array2::eye(n_random_effects);

        Ok(Self {
            fixed_effects,
            random_effects_cov,
            residual_variance: 1.0,
            groups: Array1::zeros(0),
            n_groups,
            n_level1_predictors,
            n_level2_predictors,
            random_slopes,
        })
    }

    /// Fit the hierarchical model using MCMC
    pub fn fit_mcmc<R: Rng + ?Sized>(
        &mut self,
        y: ArrayView1<f64>,
        x_level1: ArrayView2<f64>,
        x_level2: ArrayView2<f64>,
        groups: ArrayView1<usize>,
        n_iter: usize,
        burnin: usize,
        rng: &mut R,
    ) -> Result<HierarchicalModelResults> {
        checkarray_finite(&y, "y")?;
        checkarray_finite(&x_level1, "x_level1")?;
        checkarray_finite(&x_level2, "x_level2")?;
        check_positive(n_iter, "n_iter")?;

        let n_obs = y.len();
        if x_level1.nrows() != n_obs {
            return Err(StatsError::DimensionMismatch(format!(
                "x_level1 rows ({}) must match y length ({})",
                x_level1.nrows(),
                n_obs
            )));
        }

        if groups.len() != n_obs {
            return Err(StatsError::DimensionMismatch(format!(
                "groups length ({}) must match y length ({})",
                groups.len(),
                n_obs
            )));
        }

        self.groups = groups.to_owned();

        // Initialize storage for MCMC samples
        let n_random_effects = if self.random_slopes {
            self.n_level1_predictors + 1
        } else {
            1
        };
        let n_fixed = (self.n_level2_predictors + 1) * n_random_effects;

        let mut fixed_effects_samples = Array2::zeros((n_iter - burnin, n_fixed));
        let mut random_effects_samples =
            Array2::zeros((n_iter - burnin, self.n_groups * n_random_effects));
        let mut variance_samples = Array1::zeros(n_iter - burnin);
        let mut tau_samples = Array2::zeros((n_iter - burnin, n_random_effects * n_random_effects));

        // Initialize random effects for each group
        let mut random_effects = Array2::zeros((self.n_groups, n_random_effects));

        // MCMC iterations
        for _iter in 0..n_iter {
            // 1. Update random effects for each group
            self.update_random_effects(&y, &x_level1, &x_level2, &mut random_effects, rng)?;

            // 2. Update fixed effects
            self.update_fixed_effects(&random_effects, &x_level2, rng)?;

            // 3. Update residual variance
            self.update_residual_variance(&y, &x_level1, &random_effects, rng)?;

            // 4. Update random effects covariance
            self.update_random_effects_covariance(&random_effects, rng)?;

            // Store samples after burnin
            if _iter >= burnin {
                let sample_idx = _iter - burnin;

                // Store fixed effects
                let mut fixed_flat = Array1::zeros(n_fixed);
                let mut idx = 0;
                for i in 0..self.fixed_effects.nrows() {
                    for j in 0..self.fixed_effects.ncols() {
                        fixed_flat[idx] = self.fixed_effects[[i, j]];
                        idx += 1;
                    }
                }
                fixed_effects_samples
                    .row_mut(sample_idx)
                    .assign(&fixed_flat);

                // Store random effects
                let mut random_flat = Array1::zeros(self.n_groups * n_random_effects);
                let mut idx = 0;
                for group in 0..self.n_groups {
                    for effect in 0..n_random_effects {
                        random_flat[idx] = random_effects[[group, effect]];
                        idx += 1;
                    }
                }
                random_effects_samples
                    .row_mut(sample_idx)
                    .assign(&random_flat);

                // Store variances
                variance_samples[sample_idx] = self.residual_variance;

                // Store tau (covariance matrix flattened)
                let mut tau_flat = Array1::zeros(n_random_effects * n_random_effects);
                let mut idx = 0;
                for i in 0..n_random_effects {
                    for j in 0..n_random_effects {
                        tau_flat[idx] = self.random_effects_cov[[i, j]];
                        idx += 1;
                    }
                }
                tau_samples.row_mut(sample_idx).assign(&tau_flat);
            }
        }

        Ok(HierarchicalModelResults {
            fixed_effects_samples,
            random_effects_samples,
            variance_samples,
            tau_samples,
            n_groups: self.n_groups,
            n_random_effects,
            n_iter: n_iter - burnin,
        })
    }

    /// Update random effects for each group using Gibbs sampling
    fn update_random_effects<R: scirs2_core::random::Rng + ?Sized>(
        &self,
        y: &ArrayView1<f64>,
        x_level1: &ArrayView2<f64>,
        x_level2: &ArrayView2<f64>,
        random_effects: &mut Array2<f64>,
        rng: &mut R,
    ) -> Result<()> {
        let n_random_effects = random_effects.ncols();

        for group in 0..self.n_groups {
            // Find observations for this group
            let group_indices: Vec<usize> = self
                .groups
                .iter()
                .enumerate()
                .filter_map(|(i, &g)| if g == group { Some(i) } else { None })
                .collect();

            if group_indices.is_empty() {
                continue;
            }

            // Extract group data
            let n_group_obs = group_indices.len();
            let mut y_group = Array1::zeros(n_group_obs);
            let mut x_group = Array2::zeros((n_group_obs, self.n_level1_predictors));

            for (i, &obs_idx) in group_indices.iter().enumerate() {
                y_group[i] = y[obs_idx];
                x_group.row_mut(i).assign(&x_level1.row(obs_idx));
            }

            // Compute posterior parameters for random _effects
            let precision_prior = scirs2_linalg::inv(&self.random_effects_cov.view(), None)
                .map_err(|e| {
                    StatsError::ComputationError(format!("Failed to invert covariance: {}", e))
                })?;

            // Design matrix for random _effects (intercept + slopes if enabled)
            let mut z_group = Array2::zeros((n_group_obs, n_random_effects));
            z_group.column_mut(0).fill(1.0); // Intercept
            if self.random_slopes && n_random_effects > 1 {
                for i in 1..n_random_effects {
                    z_group.column_mut(i).assign(&x_group.column(i - 1));
                }
            }

            let zt_z = z_group.t().dot(&z_group);
            let precision_posterior = precision_prior.clone() + zt_z / self.residual_variance;

            let covariance_posterior = scirs2_linalg::inv(&precision_posterior.view(), None)
                .map_err(|e| {
                    StatsError::ComputationError(format!(
                        "Failed to invert posterior precision: {}",
                        e
                    ))
                })?;

            // Compute prior mean for this group
            let group_level2 = if group < x_level2.nrows() {
                x_level2.row(group).to_owned()
            } else {
                Array1::zeros(x_level2.ncols())
            };

            let mut prior_mean = Array1::zeros(n_random_effects);
            for i in 0..n_random_effects {
                prior_mean[i] = self.fixed_effects.row(i).dot(&group_level2);
            }

            let data_contrib = z_group.t().dot(&y_group) / self.residual_variance;
            let prior_contrib = precision_prior.dot(&prior_mean);
            let posterior_mean = covariance_posterior.dot(&(data_contrib + prior_contrib));

            // Sample from multivariate normal
            let mvn_sample =
                sample_multivariate_normal(&posterior_mean, &covariance_posterior, rng)?;
            random_effects.row_mut(group).assign(&mvn_sample);
        }

        Ok(())
    }

    /// Update fixed effects using Gibbs sampling
    fn update_fixed_effects<R: Rng + ?Sized>(
        &mut self,
        random_effects: &Array2<f64>,
        x_level2: &ArrayView2<f64>,
        rng: &mut R,
    ) -> Result<()> {
        let n_random_effects = self.fixed_effects.nrows();
        let n_level2_predictors = self.fixed_effects.ncols();

        for i in 0..n_random_effects {
            // Extract dependent variable (random effect i for all groups)
            let y_i = random_effects.column(i);

            // Prior parameters (weak priors)
            let prior_precision = 1e-6;
            let prior_mean = 0.0;

            // Likelihood precision
            let tau_ii = self.random_effects_cov[[i, i]];
            let likelihood_precision = 1.0 / tau_ii;

            // Posterior parameters
            let xtx = x_level2.t().dot(x_level2);
            let precision_posterior =
                Array2::eye(n_level2_predictors) * prior_precision + xtx * likelihood_precision;
            let covariance_posterior = scirs2_linalg::inv(&precision_posterior.view(), None)
                .map_err(|e| {
                    StatsError::ComputationError(format!("Failed to invert precision: {}", e))
                })?;

            let xty = x_level2.t().dot(&y_i);
            let data_contrib = xty * likelihood_precision;
            let prior_contrib =
                Array1::from_elem(n_level2_predictors, prior_mean * prior_precision);
            let mean_posterior = covariance_posterior.dot(&(data_contrib + prior_contrib));

            // Sample from multivariate normal
            let sample = sample_multivariate_normal(&mean_posterior, &covariance_posterior, rng)?;
            self.fixed_effects.row_mut(i).assign(&sample);
        }

        Ok(())
    }

    /// Update residual variance using Gibbs sampling
    fn update_residual_variance<R: Rng + ?Sized>(
        &mut self,
        y: &ArrayView1<f64>,
        x_level1: &ArrayView2<f64>,
        random_effects: &Array2<f64>,
        rng: &mut R,
    ) -> Result<()> {
        let n_obs = y.len();

        // Compute residuals
        let mut residuals_sum_sq = 0.0;
        for (obs_idx, &group) in self.groups.iter().enumerate() {
            let y_obs = y[obs_idx];
            let x_obs = x_level1.row(obs_idx);

            // Predicted value
            let intercept = random_effects[[group, 0]];
            let mut y_pred = intercept;

            if self.random_slopes && random_effects.ncols() > 1 {
                for j in 0..self.n_level1_predictors {
                    y_pred += random_effects[[group, j + 1]] * x_obs[j];
                }
            }

            let residual = y_obs - y_pred;
            residuals_sum_sq += residual * residual;
        }

        // Inverse gamma prior parameters
        let alpha_prior = 1e-3;
        let beta_prior = 1e-3;

        // Posterior parameters
        let alpha_posterior = alpha_prior + n_obs as f64 / 2.0;
        let beta_posterior = beta_prior + residuals_sum_sq / 2.0;

        // Sample from inverse gamma (via gamma)
        let gamma_dist = Gamma::new(alpha_posterior, 1.0 / beta_posterior).map_err(|e| {
            StatsError::ComputationError(format!("Failed to create Gamma distribution: {}", e))
        })?;
        let precision_sample = gamma_dist.sample(rng);
        self.residual_variance = 1.0 / precision_sample;

        Ok(())
    }

    /// Update random effects covariance matrix using inverse Wishart
    fn update_random_effects_covariance<R: scirs2_core::random::Rng + ?Sized>(
        &mut self,
        random_effects: &Array2<f64>,
        rng: &mut R,
    ) -> Result<()> {
        let n_random_effects = random_effects.ncols();
        let n_groups = random_effects.nrows();

        // Compute sample covariance of random _effects
        let mut sum_outer_products = Array2::<f64>::zeros((n_random_effects, n_random_effects));

        for group in 0..n_groups {
            let _effects = random_effects.row(group);
            let outer = outer_product(&_effects.to_owned());
            sum_outer_products = sum_outer_products + outer;
        }

        // Inverse Wishart prior parameters
        let nu_prior = n_random_effects as f64 + 2.0; // Degrees of freedom
        let psi_prior = Array2::<f64>::eye(n_random_effects) * 0.1; // Scale matrix

        // Posterior parameters
        let nu_posterior = nu_prior + n_groups as f64;
        let psi_posterior = psi_prior + sum_outer_products;

        // Sample from inverse Wishart (simplified using independent gamma for diagonal)
        // In full implementation, would use proper inverse Wishart sampling
        let mut new_cov = Array2::<f64>::zeros((n_random_effects, n_random_effects));

        for i in 0..n_random_effects {
            // Sample diagonal elements from inverse gamma
            let alpha = nu_posterior / 2.0;
            let beta = psi_posterior[[i, i]] / 2.0;

            let gamma_dist = Gamma::new(alpha, 1.0 / beta).map_err(|e| {
                StatsError::ComputationError(format!("Failed to create Gamma distribution: {}", e))
            })?;
            let precision = gamma_dist.sample(rng);
            new_cov[[i, i]] = 1.0 / precision;
        }

        // For off-diagonal elements, use simplified approach
        for i in 0..n_random_effects {
            for j in (i + 1)..n_random_effects {
                let val1: f64 = psi_posterior[[i, i]];
                let val2: f64 = psi_posterior[[j, j]];
                let denom: f64 = (val1 * val2).sqrt();
                let correlation: f64 = psi_posterior[[i, j]] / denom;
                let covariance = correlation * (new_cov[[i, i]] * new_cov[[j, j]]).sqrt();
                new_cov[[i, j]] = covariance * 0.1; // Shrink off-diagonal
                new_cov[[j, i]] = new_cov[[i, j]];
            }
        }

        self.random_effects_cov = new_cov;
        Ok(())
    }

    /// Predict for new data
    pub fn predict(
        &self,
        x_level1: ArrayView2<f64>,
        x_level2: ArrayView2<f64>,
        groups: ArrayView1<usize>,
    ) -> Result<Array1<f64>> {
        checkarray_finite(&x_level1, "x_level1")?;
        checkarray_finite(&x_level2, "x_level2")?;

        let n_obs = x_level1.nrows();
        let mut predictions = Array1::zeros(n_obs);

        for (obs_idx, &group) in groups.iter().enumerate() {
            if group >= self.n_groups {
                return Err(StatsError::InvalidArgument(format!(
                    "Group {} exceeds number of groups {}",
                    group, self.n_groups
                )));
            }

            let x_obs = x_level1.row(obs_idx);

            // Compute group-level predictors
            let zeros_array = Array1::zeros(x_level2.ncols());
            let group_level2 = if group < x_level2.nrows() {
                x_level2.row(group)
            } else {
                // Handle new groups by using population mean
                zeros_array.view()
            };

            // Compute random intercept
            let intercept = self.fixed_effects.row(0).dot(&group_level2);
            let mut y_pred = intercept;

            // Add slope effects if enabled
            if self.random_slopes && self.fixed_effects.nrows() > 1 {
                for j in 0..self.n_level1_predictors {
                    let slope = self.fixed_effects.row(j + 1).dot(&group_level2);
                    y_pred += slope * x_obs[j];
                }
            }

            predictions[obs_idx] = y_pred;
        }

        Ok(predictions)
    }
}

/// Results from hierarchical model fitting
#[derive(Debug, Clone)]
pub struct HierarchicalModelResults {
    /// MCMC samples of fixed effects (flattened)
    pub fixed_effects_samples: Array2<f64>,
    /// MCMC samples of random effects (flattened)
    pub random_effects_samples: Array2<f64>,
    /// MCMC samples of residual variance
    pub variance_samples: Array1<f64>,
    /// MCMC samples of random effects covariance (flattened)
    pub tau_samples: Array2<f64>,
    /// Number of groups
    pub n_groups: usize,
    /// Number of random effects per group
    pub n_random_effects: usize,
    /// Number of MCMC samples
    pub n_iter: usize,
}

impl HierarchicalModelResults {
    /// Compute posterior summaries for fixed effects
    pub fn fixed_effects_summary(&self) -> Result<Array2<f64>> {
        let n_params = self.fixed_effects_samples.ncols();
        let mut summary = Array2::zeros((n_params, 4)); // mean, std, 2.5%, 97.5%

        for param in 0..n_params {
            let samples = self.fixed_effects_samples.column(param);
            let mean = samples.mean();
            let std = samples.variance().sqrt();

            let mut sorted_samples = samples.to_vec();
            sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let q025_idx = (0.025 * sorted_samples.len() as f64) as usize;
            let q975_idx = (0.975 * sorted_samples.len() as f64) as usize;
            let q025 = sorted_samples[q025_idx];
            let q975 = sorted_samples[q975_idx.min(sorted_samples.len() - 1)];

            summary[[param, 0]] = mean;
            summary[[param, 1]] = std;
            summary[[param, 2]] = q025;
            summary[[param, 3]] = q975;
        }

        Ok(summary)
    }

    /// Compute posterior summaries for random effects variances
    pub fn random_effects_variance_summary(&self) -> Result<Array2<f64>> {
        let n_params = self.n_random_effects * self.n_random_effects;
        let mut summary = Array2::zeros((n_params, 4));

        for param in 0..n_params {
            let samples = self.tau_samples.column(param);
            let mean = samples.mean();
            let std = samples.variance().sqrt();

            let mut sorted_samples = samples.to_vec();
            sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let q025_idx = (0.025 * sorted_samples.len() as f64) as usize;
            let q975_idx = (0.975 * sorted_samples.len() as f64) as usize;
            let q025 = sorted_samples[q025_idx];
            let q975 = sorted_samples[q975_idx.min(sorted_samples.len() - 1)];

            summary[[param, 0]] = mean;
            summary[[param, 1]] = std;
            summary[[param, 2]] = q025;
            summary[[param, 3]] = q975;
        }

        Ok(summary)
    }
}

/// Bayesian ANOVA with hierarchical structure
#[derive(Debug, Clone)]
pub struct HierarchicalANOVA {
    /// Group means
    pub group_means: Array1<f64>,
    /// Overall mean
    pub overall_mean: f64,
    /// Between-group variance
    pub between_variance: f64,
    /// Within-group variance
    pub within_variance: f64,
    /// Group assignments
    pub groups: Array1<usize>,
    /// Number of groups
    pub n_groups: usize,
}

impl HierarchicalANOVA {
    /// Create new hierarchical ANOVA
    pub fn new(n_groups: usize) -> Result<Self> {
        check_positive(n_groups, "n_groups")?;

        Ok(Self {
            group_means: Array1::zeros(n_groups),
            overall_mean: 0.0,
            between_variance: 1.0,
            within_variance: 1.0,
            groups: Array1::zeros(0),
            n_groups,
        })
    }

    /// Fit hierarchical ANOVA using MCMC
    pub fn fit_mcmc<R: Rng + ?Sized>(
        &mut self,
        y: ArrayView1<f64>,
        groups: ArrayView1<usize>,
        n_iter: usize,
        burnin: usize,
        rng: &mut R,
    ) -> Result<HierarchicalANOVAResults> {
        checkarray_finite(&y, "y")?;
        check_positive(n_iter, "n_iter")?;

        if y.len() != groups.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "y length ({}) must match groups length ({})",
                y.len(),
                groups.len()
            )));
        }

        self.groups = groups.to_owned();

        // Initialize storage
        let mut group_means_samples = Array2::zeros((n_iter - burnin, self.n_groups));
        let mut overall_mean_samples_ = Array1::zeros(n_iter - burnin);
        let mut between_var_samples = Array1::zeros(n_iter - burnin);
        let mut within_var_samples = Array1::zeros(n_iter - burnin);

        // Group statistics
        let mut group_counts = vec![0; self.n_groups];
        let mut group_sums = vec![0.0; self.n_groups];

        for (&obs_group, &obs_y) in groups.iter().zip(y.iter()) {
            if obs_group >= self.n_groups {
                return Err(StatsError::InvalidArgument(format!(
                    "Group {} exceeds n_groups {}",
                    obs_group, self.n_groups
                )));
            }
            group_counts[obs_group] += 1;
            group_sums[obs_group] += obs_y;
        }

        // MCMC iterations
        for _iter in 0..n_iter {
            // 1. Update group means
            for group in 0..self.n_groups {
                if group_counts[group] > 0 {
                    // Posterior parameters
                    let prior_precision = 1.0 / self.between_variance;
                    let likelihood_precision = group_counts[group] as f64 / self.within_variance;
                    let posterior_precision = prior_precision + likelihood_precision;
                    let posterior_variance = 1.0 / posterior_precision;

                    let prior_mean_contribution = self.overall_mean * prior_precision;
                    let likelihood_mean_contribution = group_sums[group] * likelihood_precision;
                    let posterior_mean = (prior_mean_contribution + likelihood_mean_contribution)
                        / posterior_precision;

                    // Sample from normal
                    let normal =
                        Normal::new(posterior_mean, posterior_variance.sqrt()).map_err(|e| {
                            StatsError::ComputationError(format!("Failed to create normal: {}", e))
                        })?;
                    self.group_means[group] = normal.sample(rng);
                } else {
                    // No observations in this group, sample from prior
                    let normal = Normal::new(self.overall_mean, self.between_variance.sqrt())
                        .map_err(|e| {
                            StatsError::ComputationError(format!("Failed to create normal: {}", e))
                        })?;
                    self.group_means[group] = normal.sample(rng);
                }
            }

            // 2. Update overall mean
            let group_mean_avg = self.group_means.clone().mean();
            let prior_variance = 10.0; // Weak prior
            let likelihood_variance = self.between_variance / self.n_groups as f64;
            let posterior_variance = 1.0 / (1.0 / prior_variance + 1.0 / likelihood_variance);
            let posterior_mean =
                (0.0 / prior_variance + group_mean_avg / likelihood_variance) * posterior_variance;

            let normal = Normal::new(posterior_mean, posterior_variance.sqrt()).map_err(|e| {
                StatsError::ComputationError(format!("Failed to create normal: {}", e))
            })?;
            self.overall_mean = normal.sample(rng);

            // 3. Update between-group variance
            let sum_sq_deviations: f64 = self
                .group_means
                .iter()
                .map(|&mean| (mean - self.overall_mean).powi(2))
                .sum();

            let alpha_prior = 1e-3;
            let beta_prior = 1e-3;
            let alpha_posterior = alpha_prior + self.n_groups as f64 / 2.0;
            let beta_posterior = beta_prior + sum_sq_deviations / 2.0;

            let gamma_dist = Gamma::new(alpha_posterior, 1.0 / beta_posterior).map_err(|e| {
                StatsError::ComputationError(format!("Failed to create Gamma: {}", e))
            })?;
            let precision = gamma_dist.sample(rng);
            self.between_variance = 1.0 / precision;

            // 4. Update within-group variance
            let mut within_sum_sq = 0.0;
            let mut total_obs = 0;

            for (&obs_group, &obs_y) in groups.iter().zip(y.iter()) {
                let residual = obs_y - self.group_means[obs_group];
                within_sum_sq += residual * residual;
                total_obs += 1;
            }

            let alpha_posterior = alpha_prior + total_obs as f64 / 2.0;
            let beta_posterior = beta_prior + within_sum_sq / 2.0;

            let gamma_dist = Gamma::new(alpha_posterior, 1.0 / beta_posterior).map_err(|e| {
                StatsError::ComputationError(format!("Failed to create Gamma: {}", e))
            })?;
            let precision = gamma_dist.sample(rng);
            self.within_variance = 1.0 / precision;

            // Store samples after burnin
            if _iter >= burnin {
                let sample_idx = _iter - burnin;
                group_means_samples
                    .row_mut(sample_idx)
                    .assign(&self.group_means);
                overall_mean_samples_[sample_idx] = self.overall_mean;
                between_var_samples[sample_idx] = self.between_variance;
                within_var_samples[sample_idx] = self.within_variance;
            }
        }

        Ok(HierarchicalANOVAResults {
            group_means_samples,
            overall_mean_samples_,
            between_variance_samples: between_var_samples,
            within_variance_samples: within_var_samples,
            n_groups: self.n_groups,
            n_iter: n_iter - burnin,
        })
    }
}

/// Results from hierarchical ANOVA
#[derive(Debug, Clone)]
pub struct HierarchicalANOVAResults {
    /// MCMC samples of group means
    pub group_means_samples: Array2<f64>,
    /// MCMC samples of overall mean
    pub overall_mean_samples_: Array1<f64>,
    /// MCMC samples of between-group variance
    pub between_variance_samples: Array1<f64>,
    /// MCMC samples of within-group variance
    pub within_variance_samples: Array1<f64>,
    /// Number of groups
    pub n_groups: usize,
    /// Number of MCMC samples
    pub n_iter: usize,
}

impl HierarchicalANOVAResults {
    /// Compute intraclass correlation coefficient (ICC)
    pub fn icc_samples(&self) -> Array1<f64> {
        let mut icc = Array1::zeros(self.n_iter);
        for i in 0..self.n_iter {
            let between_var = self.between_variance_samples[i];
            let within_var = self.within_variance_samples[i];
            icc[i] = between_var / (between_var + within_var);
        }
        icc
    }

    /// Compute posterior probability that group i has higher mean than group j
    pub fn prob_group_higher(&self, group_i: usize, group_j: usize) -> Result<f64> {
        if group_i >= self.n_groups || group_j >= self.n_groups {
            return Err(StatsError::InvalidArgument(
                "Group indices out of bounds".to_string(),
            ));
        }

        let mut count = 0;
        for iter in 0..self.n_iter {
            if self.group_means_samples[[iter, group_i]] > self.group_means_samples[[iter, group_j]]
            {
                count += 1;
            }
        }

        Ok(count as f64 / self.n_iter as f64)
    }
}

// Helper functions

/// Sample from multivariate normal distribution
#[allow(dead_code)]
fn sample_multivariate_normal<R: Rng + ?Sized>(
    mean: &Array1<f64>,
    covariance: &Array2<f64>,
    rng: &mut R,
) -> Result<Array1<f64>> {
    let dim = mean.len();
    let normal = Normal::new(0.0, 1.0)
        .map_err(|e| StatsError::ComputationError(format!("Failed to create normal: {}", e)))?;

    // Sample from standard normal
    let z = Array1::from_shape_fn(dim, |_| normal.sample(rng));

    // Cholesky decomposition (simplified - use diagonal for now)
    let mut sample = Array1::zeros(dim);
    for i in 0..dim {
        sample[i] = mean[i] + z[i] * covariance[[i, i]].sqrt();
    }

    Ok(sample)
}

/// Compute outer product of a vector
#[allow(dead_code)]
fn outer_product(v: &Array1<f64>) -> Array2<f64> {
    let n = v.len();
    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = v[i] * v[j];
        }
    }
    result
}
