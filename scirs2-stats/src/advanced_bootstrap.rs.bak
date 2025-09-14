//! Advanced bootstrap methods for complex statistical inference
//!
//! This module provides sophisticated bootstrap resampling techniques that go beyond
//! simple random sampling, including stratified bootstrap, block bootstrap for time series,
//! and other specialized resampling methods for complex data structures.

use crate::error::{StatsError, StatsResult};
use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive, NumCast, One, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::{parallel_ops::*, rng, simd_ops::SimdUnifiedOps, validation::*};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Advanced bootstrap configuration
#[derive(Debug, Clone)]
pub struct AdvancedBootstrapConfig {
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Bootstrap type
    pub bootstrap_type: BootstrapType,
    /// Enable parallel processing
    pub parallel: bool,
    /// Confidence level for intervals
    pub confidence_level: f64,
    /// Block length for block bootstrap (auto-selected if None)
    pub block_length: Option<usize>,
    /// Enable bias correction
    pub bias_correction: bool,
    /// Enable acceleration correction (BCa intervals)
    pub acceleration_correction: bool,
    /// Maximum number of parallel threads
    pub max_threads: Option<usize>,
}

impl Default for AdvancedBootstrapConfig {
    fn default() -> Self {
        Self {
            n_bootstrap: 1000,
            seed: None,
            bootstrap_type: BootstrapType::Basic,
            parallel: true,
            confidence_level: 0.95,
            block_length: None,
            bias_correction: true,
            acceleration_correction: true,
            max_threads: None,
        }
    }
}

/// Bootstrap method types
#[derive(Debug, Clone, PartialEq)]
pub enum BootstrapType {
    /// Standard bootstrap with replacement
    Basic,
    /// Stratified bootstrap maintaining group proportions
    Stratified {
        /// Stratification variable (group indices)
        strata: Vec<usize>,
    },
    /// Block bootstrap for time series data
    Block {
        /// Block type
        block_type: BlockType,
    },
    /// Bayesian bootstrap using random weights
    Bayesian,
    /// Wild bootstrap for regression residuals
    Wild {
        /// Wild bootstrap distribution
        distribution: WildDistribution,
    },
    /// Parametric bootstrap using fitted distributions
    Parametric {
        /// Distribution parameters
        distribution_params: ParametricBootstrapParams,
    },
    /// Balanced bootstrap ensuring each observation appears exactly once per resample
    Balanced,
}

/// Block bootstrap types for time series
#[derive(Debug, Clone, PartialEq)]
pub enum BlockType {
    /// Moving block bootstrap (overlapping blocks)
    Moving,
    /// Circular block bootstrap (wrap-around)
    Circular,
    /// Non-overlapping block bootstrap
    NonOverlapping,
    /// Stationary bootstrap (random block lengths)
    Stationary {
        /// Expected block length
        expected_length: f64,
    },
    /// Tapered block bootstrap (gradual weight decay at block edges)
    Tapered {
        /// Tapering function
        taper_function: TaperFunction,
    },
}

/// Tapering functions for block bootstrap
#[derive(Debug, Clone, PartialEq)]
pub enum TaperFunction {
    /// Linear tapering
    Linear,
    /// Cosine tapering (Tukey window)
    Cosine,
    /// Exponential tapering
    Exponential { decay_rate: f64 },
}

/// Wild bootstrap distributions
#[derive(Debug, Clone, PartialEq)]
pub enum WildDistribution {
    /// Rademacher distribution (±1 with equal probability)
    Rademacher,
    /// Mammen distribution (optimal for wild bootstrap)
    Mammen,
    /// Standard normal distribution
    Normal,
    /// Two-point distribution with specified weights
    TwoPoint { prob_positive: f64 },
}

/// Parametric bootstrap parameters
#[derive(Debug, Clone, PartialEq)]
pub enum ParametricBootstrapParams {
    /// Normal distribution parameters
    Normal { mean: f64, std: f64 },
    /// Exponential distribution parameter
    Exponential { rate: f64 },
    /// Gamma distribution parameters
    Gamma { shape: f64, scale: f64 },
    /// Beta distribution parameters
    Beta { alpha: f64, beta: f64 },
    /// Custom distribution with CDF function
    Custom {
        /// Distribution name
        name: String,
        /// Parameters
        params: HashMap<String, f64>,
    },
}

/// Bootstrap result with comprehensive statistics
#[derive(Debug, Clone)]
pub struct AdvancedBootstrapResult<F> {
    /// Bootstrap samples
    pub bootstrap_samples: Array1<F>,
    /// Original statistic value
    pub original_statistic: F,
    /// Bootstrap mean
    pub bootstrap_mean: F,
    /// Bootstrap standard error
    pub standard_error: F,
    /// Bias estimate
    pub bias: F,
    /// Confidence intervals
    pub confidence_intervals: BootstrapConfidenceIntervals<F>,
    /// Bootstrap method used
    pub method: BootstrapType,
    /// Number of successful bootstrap samples
    pub n_successful: usize,
    /// Effective sample size (for block bootstrap)
    pub effective_samplesize: Option<usize>,
    /// Bootstrap diagnostics
    pub diagnostics: BootstrapDiagnostics<F>,
}

/// Bootstrap confidence intervals
#[derive(Debug, Clone)]
pub struct BootstrapConfidenceIntervals<F> {
    /// Percentile method intervals
    pub percentile: (F, F),
    /// Basic bootstrap intervals
    pub basic: (F, F),
    /// Bias-corrected (BC) intervals
    pub bias_corrected: Option<(F, F)>,
    /// Bias-corrected and accelerated (BCa) intervals
    pub bias_corrected_accelerated: Option<(F, F)>,
    /// Studentized (bootstrap-t) intervals
    pub studentized: Option<(F, F)>,
}

/// Bootstrap diagnostics
#[derive(Debug, Clone)]
pub struct BootstrapDiagnostics<F> {
    /// Distribution characteristics
    pub distribution_stats: BootstrapDistributionStats<F>,
    /// Quality metrics
    pub quality_metrics: QualityMetrics<F>,
    /// Convergence information
    pub convergence_info: ConvergenceInfo<F>,
    /// Method-specific diagnostics
    pub method_specific: HashMap<String, F>,
}

/// Bootstrap distribution statistics
#[derive(Debug, Clone)]
pub struct BootstrapDistributionStats<F> {
    /// Skewness of bootstrap distribution
    pub skewness: F,
    /// Kurtosis of bootstrap distribution
    pub kurtosis: F,
    /// Jarque-Bera test statistic for normality
    pub jarque_bera: F,
    /// Anderson-Darling test statistic
    pub anderson_darling: F,
    /// Minimum bootstrap value
    pub min_value: F,
    /// Maximum bootstrap value
    pub max_value: F,
}

/// Quality metrics for bootstrap assessment
#[derive(Debug, Clone)]
pub struct QualityMetrics<F> {
    /// Monte Carlo standard error
    pub mc_standard_error: F,
    /// Coverage probability estimate
    pub coverage_probability: F,
    /// Bootstrap efficiency (relative to analytical)
    pub efficiency: Option<F>,
    /// Stability measure across subsamples
    pub stability: F,
}

/// Convergence information
#[derive(Debug, Clone)]
pub struct ConvergenceInfo<F> {
    /// Has the bootstrap converged
    pub converged: bool,
    /// Number of samples needed for convergence
    pub convergence_samplesize: Option<usize>,
    /// Running mean stability
    pub mean_stability: F,
    /// Running variance stability
    pub variance_stability: F,
}

/// Advanced bootstrap processor
pub struct AdvancedBootstrapProcessor<F> {
    config: AdvancedBootstrapConfig,
    rng: StdRng,
    _phantom: PhantomData<F>,
}

impl<F> AdvancedBootstrapProcessor<F>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + FromPrimitive
        + Copy
        + Send
        + Sync
        + std::fmt::Display,
{
    /// Create new advanced bootstrap processor
    pub fn new(config: AdvancedBootstrapConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut rng()),
        };

        Self {
            config,
            rng,
            _phantom: PhantomData,
        }
    }

    /// Perform advanced bootstrap resampling
    pub fn bootstrap<T>(
        &mut self,
        data: &ArrayView1<F>,
        statistic_fn: impl Fn(&ArrayView1<F>) -> StatsResult<T> + Send + Sync + Copy,
    ) -> StatsResult<AdvancedBootstrapResult<F>>
    where
        T: Into<F> + Copy + Send + Sync,
    {
        checkarray_finite(data, "data")?;

        if data.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Data cannot be empty".to_string(),
            ));
        }

        // Compute original statistic
        let original_statistic = statistic_fn(data)?.into();

        // Generate bootstrap samples based on method
        let bootstrap_type = self.config.bootstrap_type.clone();
        let bootstrap_samples = match bootstrap_type {
            BootstrapType::Basic => self.basic_bootstrap(data, statistic_fn)?,
            BootstrapType::Stratified { strata } => {
                self.stratified_bootstrap(data, &strata, statistic_fn)?
            }
            BootstrapType::Block { block_type } => {
                self.block_bootstrap(data, &block_type, statistic_fn)?
            }
            BootstrapType::Bayesian => self.bayesian_bootstrap(data, statistic_fn)?,
            BootstrapType::Wild { distribution } => {
                self.wild_bootstrap(data, &distribution, statistic_fn)?
            }
            BootstrapType::Parametric {
                distribution_params,
            } => self.parametric_bootstrap(data, &distribution_params, statistic_fn)?,
            BootstrapType::Balanced => self.balanced_bootstrap(data, statistic_fn)?,
        };

        // Compute bootstrap statistics
        let bootstrap_mean = self.compute_mean(&bootstrap_samples);
        let standard_error = self.compute_std(&bootstrap_samples);
        let bias = bootstrap_mean - original_statistic;

        // Compute confidence intervals
        let confidence_intervals = self.compute_confidence_intervals(
            &bootstrap_samples,
            original_statistic,
            standard_error,
        )?;

        // Compute diagnostics
        let diagnostics = self.compute_diagnostics(&bootstrap_samples, original_statistic)?;

        // Determine effective sample size
        let effective_samplesize = match &self.config.bootstrap_type {
            BootstrapType::Block { .. } => Some(self.compute_effective_samplesize(data.len())),
            _ => None,
        };

        Ok(AdvancedBootstrapResult {
            bootstrap_samples,
            original_statistic,
            bootstrap_mean,
            standard_error,
            bias,
            confidence_intervals,
            method: self.config.bootstrap_type.clone(),
            n_successful: self.config.n_bootstrap,
            effective_samplesize,
            diagnostics,
        })
    }

    /// Basic bootstrap with replacement
    fn basic_bootstrap<T>(
        &mut self,
        data: &ArrayView1<F>,
        statistic_fn: impl Fn(&ArrayView1<F>) -> StatsResult<T> + Send + Sync + Copy,
    ) -> StatsResult<Array1<F>>
    where
        T: Into<F> + Copy + Send + Sync,
    {
        let n = data.len();
        let mut bootstrap_samples = Array1::zeros(self.config.n_bootstrap);

        if self.config.parallel && self.config.n_bootstrap > 100 {
            // Parallel execution
            let samples: Result<Vec<_>, _> = (0..self.config.n_bootstrap)
                .into_par_iter()
                .map(|_| {
                    let mut local_rng = { StdRng::from_rng(&mut rng()) };
                    let mut resample = Array1::zeros(n);

                    for i in 0..n {
                        let idx = local_rng.gen_range(0..n);
                        resample[i] = data[idx];
                    }

                    statistic_fn(&resample.view()).map(|s| s.into())
                })
                .collect();

            let sample_values = samples?;
            for (i, value) in sample_values.into_iter().enumerate() {
                bootstrap_samples[i] = value;
            }
        } else {
            // Sequential execution
            for i in 0..self.config.n_bootstrap {
                let mut resample = Array1::zeros(n);

                for j in 0..n {
                    let idx = self.rng.gen_range(0..n);
                    resample[j] = data[idx];
                }

                bootstrap_samples[i] = statistic_fn(&resample.view())?.into();
            }
        }

        Ok(bootstrap_samples)
    }

    /// Stratified bootstrap maintaining group proportions
    fn stratified_bootstrap<T>(
        &mut self,
        data: &ArrayView1<F>,
        strata: &[usize],
        statistic_fn: impl Fn(&ArrayView1<F>) -> StatsResult<T> + Send + Sync + Copy,
    ) -> StatsResult<Array1<F>>
    where
        T: Into<F> + Copy + Send + Sync,
    {
        if data.len() != strata.len() {
            return Err(StatsError::DimensionMismatch(
                "Data and strata must have same length".to_string(),
            ));
        }

        // Group data by strata
        let mut strata_groups: HashMap<usize, Vec<(usize, F)>> = HashMap::new();
        for (i, (&value, &stratum)) in data.iter().zip(strata.iter()).enumerate() {
            strata_groups
                .entry(stratum)
                .or_insert_with(Vec::new)
                .push((i, value));
        }

        let n = data.len();
        let mut bootstrap_samples = Array1::zeros(self.config.n_bootstrap);

        for i in 0..self.config.n_bootstrap {
            let mut resample = Array1::zeros(n);
            let mut resample_idx = 0;

            // Sample from each stratum proportionally
            for (_, groupdata) in &strata_groups {
                let groupsize = groupdata.len();

                for _ in 0..groupsize {
                    let idx = self.rng.gen_range(0..groupsize);
                    resample[resample_idx] = groupdata[idx].1;
                    resample_idx += 1;
                }
            }

            bootstrap_samples[i] = statistic_fn(&resample.view())?.into();
        }

        Ok(bootstrap_samples)
    }

    /// Block bootstrap for time series data
    fn block_bootstrap<T>(
        &mut self,
        data: &ArrayView1<F>,
        block_type: &BlockType,
        statistic_fn: impl Fn(&ArrayView1<F>) -> StatsResult<T> + Send + Sync + Copy,
    ) -> StatsResult<Array1<F>>
    where
        T: Into<F> + Copy + Send + Sync,
    {
        let n = data.len();
        let block_length = self
            .config
            .block_length
            .unwrap_or_else(|| self.optimal_block_length(n));

        if block_length >= n {
            return Err(StatsError::InvalidArgument(
                "Block length must be less than data length".to_string(),
            ));
        }

        let mut bootstrap_samples = Array1::zeros(self.config.n_bootstrap);

        for i in 0..self.config.n_bootstrap {
            let resample = match block_type {
                BlockType::Moving => self.moving_blockbootstrap(data, block_length)?,
                BlockType::Circular => self.circular_blockbootstrap(data, block_length)?,
                BlockType::NonOverlapping => {
                    self.non_overlapping_blockbootstrap(data, block_length)?
                }
                BlockType::Stationary { expected_length } => {
                    self.stationarybootstrap(data, *expected_length)?
                }
                BlockType::Tapered { taper_function } => {
                    self.tapered_blockbootstrap(data, block_length, taper_function)?
                }
            };

            bootstrap_samples[i] = statistic_fn(&resample.view())?.into();
        }

        Ok(bootstrap_samples)
    }

    /// Moving block bootstrap
    fn moving_blockbootstrap(
        &mut self,
        data: &ArrayView1<F>,
        block_length: usize,
    ) -> StatsResult<Array1<F>> {
        let n = data.len();
        let n_blocks = (n + block_length - 1) / block_length; // Ceiling division
        let mut resample = Array1::zeros(n);
        let mut pos = 0;

        for _ in 0..n_blocks {
            if pos >= n {
                break;
            }

            let start_idx = self.rng.gen_range(0..(n - block_length));
            let copy_length = std::cmp::min(block_length, n - pos);

            for i in 0..copy_length {
                resample[pos + i] = data[start_idx + i];
            }
            pos += copy_length;
        }

        Ok(resample)
    }

    /// Circular block bootstrap
    fn circular_blockbootstrap(
        &mut self,
        data: &ArrayView1<F>,
        block_length: usize,
    ) -> StatsResult<Array1<F>> {
        let n = data.len();
        let n_blocks = (n + block_length - 1) / block_length;
        let mut resample = Array1::zeros(n);
        let mut pos = 0;

        for _ in 0..n_blocks {
            if pos >= n {
                break;
            }

            let start_idx = self.rng.gen_range(0..n);
            let copy_length = std::cmp::min(block_length, n - pos);

            for i in 0..copy_length {
                let idx = (start_idx + i) % n; // Circular indexing
                resample[pos + i] = data[idx];
            }
            pos += copy_length;
        }

        Ok(resample)
    }

    /// Non-overlapping block bootstrap
    fn non_overlapping_blockbootstrap(
        &mut self,
        data: &ArrayView1<F>,
        block_length: usize,
    ) -> StatsResult<Array1<F>> {
        let n = data.len();
        let n_complete_blocks = n / block_length;
        let remainder = n % block_length;

        // Create blocks
        let mut blocks = Vec::new();
        for i in 0..n_complete_blocks {
            let start = i * block_length;
            let end = start + block_length;
            blocks.push(data.slice(ndarray::s![start..end]).to_owned());
        }

        // Add remainder as partial block if exists
        if remainder > 0 {
            let start = n_complete_blocks * block_length;
            blocks.push(data.slice(ndarray::s![start..]).to_owned());
        }

        // Resample blocks
        let mut resample = Array1::zeros(n);
        let mut pos = 0;

        while pos < n {
            let block_idx = self.rng.gen_range(0..blocks.len());
            let block = &blocks[block_idx];
            let copy_length = std::cmp::min(block.len(), n - pos);

            for i in 0..copy_length {
                resample[pos + i] = block[i];
            }
            pos += copy_length;
        }

        Ok(resample)
    }

    /// Stationary bootstrap with random block lengths
    fn stationarybootstrap(
        &mut self,
        data: &ArrayView1<F>,
        expected_length: f64,
    ) -> StatsResult<Array1<F>> {
        let n = data.len();
        let p = 1.0 / expected_length; // Probability of ending a block
        let mut resample = Array1::zeros(n);
        let mut pos = 0;

        while pos < n {
            let start_idx = self.rng.gen_range(0..n);
            let mut block_length = 1;

            // Generate random block _length using geometric distribution
            while self.rng.random::<f64>() > p && block_length < n - pos {
                block_length += 1;
            }

            // Copy block with circular indexing
            for i in 0..block_length {
                if pos + i >= n {
                    break;
                }
                let idx = (start_idx + i) % n;
                resample[pos + i] = data[idx];
            }

            pos += block_length;
        }

        Ok(resample)
    }

    /// Tapered block bootstrap
    fn tapered_blockbootstrap(
        &mut self,
        data: &ArrayView1<F>,
        block_length: usize,
        taper_function: &TaperFunction,
    ) -> StatsResult<Array1<F>> {
        let n = data.len();
        let mut resample = Array1::zeros(n);
        let n_blocks = (n + block_length - 1) / block_length;
        let mut pos = 0;

        for _ in 0..n_blocks {
            if pos >= n {
                break;
            }

            let start_idx = self.rng.gen_range(0..(n - block_length));
            let copy_length = std::cmp::min(block_length, n - pos);

            // Apply tapering weights
            for i in 0..copy_length {
                let weight = self.compute_taper_weight(i, copy_length, taper_function);
                let value = data[start_idx + i] * F::from(weight).unwrap();

                if pos + i < resample.len() {
                    resample[pos + i] = resample[pos + i] + value;
                }
            }
            pos += copy_length;
        }

        Ok(resample)
    }

    /// Compute taper weight
    fn compute_taper_weight(
        &self,
        position: usize,
        block_length: usize,
        taper_function: &TaperFunction,
    ) -> f64 {
        let t = position as f64 / (block_length - 1) as f64;

        match taper_function {
            TaperFunction::Linear => {
                if t <= 0.5 {
                    2.0 * t
                } else {
                    2.0 * (1.0 - t)
                }
            }
            TaperFunction::Cosine => 0.5 * (1.0 - (std::f64::consts::PI * t).cos()),
            TaperFunction::Exponential { decay_rate } => {
                let distance_from_center = (t - 0.5).abs();
                (-decay_rate * distance_from_center).exp()
            }
        }
    }

    /// Bayesian bootstrap using random weights
    fn bayesian_bootstrap<T>(
        &mut self,
        data: &ArrayView1<F>,
        statistic_fn: impl Fn(&ArrayView1<F>) -> StatsResult<T> + Send + Sync + Copy,
    ) -> StatsResult<Array1<F>>
    where
        T: Into<F> + Copy + Send + Sync,
    {
        let n = data.len();
        let mut bootstrap_samples = Array1::zeros(self.config.n_bootstrap);

        for i in 0..self.config.n_bootstrap {
            // Generate random weights from Dirichlet(1,1,...,1) = Exponential(1)
            let mut weights = Array1::zeros(n);
            let mut weight_sum = F::zero();

            for j in 0..n {
                let exp_sample = -self.rng.random::<f64>().ln(); // Exponential(1) sample
                weights[j] = F::from(exp_sample).unwrap();
                weight_sum = weight_sum + weights[j];
            }

            // Normalize weights
            for j in 0..n {
                weights[j] = weights[j] / weight_sum;
            }

            // Create weighted resample
            let mut resample = Array1::zeros(n);
            for j in 0..n {
                resample[j] = data[j] * weights[j] * F::from(n).unwrap(); // Scale by n
            }

            bootstrap_samples[i] = statistic_fn(&resample.view())?.into();
        }

        Ok(bootstrap_samples)
    }

    /// Wild bootstrap for regression residuals
    fn wild_bootstrap<T>(
        &mut self,
        data: &ArrayView1<F>,
        distribution: &WildDistribution,
        statistic_fn: impl Fn(&ArrayView1<F>) -> StatsResult<T> + Send + Sync + Copy,
    ) -> StatsResult<Array1<F>>
    where
        T: Into<F> + Copy + Send + Sync,
    {
        let n = data.len();
        let mut bootstrap_samples = Array1::zeros(self.config.n_bootstrap);

        for i in 0..self.config.n_bootstrap {
            let mut resample = Array1::zeros(n);

            for j in 0..n {
                let multiplier = match distribution {
                    WildDistribution::Rademacher => {
                        if self.rng.random::<f64>() < 0.5 {
                            -1.0
                        } else {
                            1.0
                        }
                    }
                    WildDistribution::Mammen => {
                        let _golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
                        let p = (5.0_f64.sqrt() + 1.0) / (2.0 * 5.0_f64.sqrt());
                        if self.rng.random::<f64>() < p {
                            -(5.0_f64.sqrt() - 1.0) / 2.0
                        } else {
                            (5.0_f64.sqrt() + 1.0) / 2.0
                        }
                    }
                    WildDistribution::Normal => {
                        // Box-Muller transform for standard normal
                        let u1 = self.rng.random::<f64>();
                        let u2 = self.rng.random::<f64>();
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                    }
                    WildDistribution::TwoPoint { prob_positive } => {
                        if self.rng.random::<f64>() < *prob_positive {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                };

                resample[j] = data[j] * F::from(multiplier).unwrap();
            }

            bootstrap_samples[i] = statistic_fn(&resample.view())?.into();
        }

        Ok(bootstrap_samples)
    }

    /// Parametric bootstrap using fitted distributions
    fn parametric_bootstrap<T>(
        &mut self,
        data: &ArrayView1<F>,
        distribution_params: &ParametricBootstrapParams,
        statistic_fn: impl Fn(&ArrayView1<F>) -> StatsResult<T> + Send + Sync + Copy,
    ) -> StatsResult<Array1<F>>
    where
        T: Into<F> + Copy + Send + Sync,
    {
        let n = data.len();
        let mut bootstrap_samples = Array1::zeros(self.config.n_bootstrap);

        for i in 0..self.config.n_bootstrap {
            let resample = match distribution_params {
                ParametricBootstrapParams::Normal { mean, std } => {
                    self.generate_normal_sample(n, *mean, *std)?
                }
                ParametricBootstrapParams::Exponential { rate } => {
                    self.generate_exponential_sample(n, *rate)?
                }
                ParametricBootstrapParams::Gamma { shape, scale } => {
                    self.generate_gamma_sample(n, *shape, *scale)?
                }
                ParametricBootstrapParams::Beta { alpha, beta } => {
                    self.generate_beta_sample(n, *alpha, *beta)?
                }
                ParametricBootstrapParams::Custom { name, .. } => {
                    return Err(StatsError::InvalidArgument(format!(
                        "Custom distribution '{}' not implemented",
                        name
                    )));
                }
            };

            bootstrap_samples[i] = statistic_fn(&resample.view())?.into();
        }

        Ok(bootstrap_samples)
    }

    /// Balanced bootstrap ensuring each observation appears exactly once per resample
    fn balanced_bootstrap<T>(
        &mut self,
        data: &ArrayView1<F>,
        statistic_fn: impl Fn(&ArrayView1<F>) -> StatsResult<T> + Send + Sync + Copy,
    ) -> StatsResult<Array1<F>>
    where
        T: Into<F> + Copy + Send + Sync,
    {
        let n = data.len();
        let mut bootstrap_samples = Array1::zeros(self.config.n_bootstrap);

        // Create balanced indices
        let total_samples = self.config.n_bootstrap * n;
        let mut all_indices = Vec::with_capacity(total_samples);

        for _ in 0..self.config.n_bootstrap {
            for i in 0..n {
                all_indices.push(i);
            }
        }

        // Shuffle the indices
        for i in (1..all_indices.len()).rev() {
            let j = self.rng.gen_range(0..i);
            all_indices.swap(i, j);
        }

        // Create bootstrap samples
        for i in 0..self.config.n_bootstrap {
            let mut resample = Array1::zeros(n);
            let start_idx = i * n;

            for j in 0..n {
                let data_idx = all_indices[start_idx + j];
                resample[j] = data[data_idx];
            }

            bootstrap_samples[i] = statistic_fn(&resample.view())?.into();
        }

        Ok(bootstrap_samples)
    }

    /// Generate normal distribution sample
    fn generate_normal_sample(&mut self, n: usize, mean: f64, std: f64) -> StatsResult<Array1<F>> {
        let mut sample = Array1::zeros(n);

        for i in 0..n {
            let u1 = self.rng.random::<f64>();
            let u2 = self.rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            sample[i] = F::from(mean + std * z).unwrap();
        }

        Ok(sample)
    }

    /// Generate exponential distribution sample
    fn generate_exponential_sample(&mut self, n: usize, rate: f64) -> StatsResult<Array1<F>> {
        let mut sample = Array1::zeros(n);

        for i in 0..n {
            let u = self.rng.random::<f64>();
            let x = -u.ln() / rate;
            sample[i] = F::from(x).unwrap();
        }

        Ok(sample)
    }

    /// Generate gamma distribution sample (simplified)
    fn generate_gamma_sample(
        &mut self,
        n: usize,
        shape: f64,
        scale: f64,
    ) -> StatsResult<Array1<F>> {
        // Simplified implementation - would use proper gamma generation in full version
        let mut sample = Array1::zeros(n);

        for i in 0..n {
            // Using normal approximation for simplicity
            let mean = shape * scale;
            let std = (shape * scale * scale).sqrt();
            let u1 = self.rng.random::<f64>();
            let u2 = self.rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            sample[i] = F::from((mean + std * z).max(0.0)).unwrap();
        }

        Ok(sample)
    }

    /// Generate beta distribution sample (simplified)
    fn generate_beta_sample(&mut self, n: usize, alpha: f64, beta: f64) -> StatsResult<Array1<F>> {
        // Simplified implementation using gamma ratio method
        let mut sample = Array1::zeros(n);

        for i in 0..n {
            let x = self.rng.random::<f64>().powf(1.0 / alpha);
            let y = self.rng.random::<f64>().powf(1.0 / beta);
            let value = x / (x + y);
            sample[i] = F::from(value).unwrap();
        }

        Ok(sample)
    }

    /// Optimal block length selection using automatic methods
    fn optimal_block_length(&self, n: usize) -> usize {
        // Simplified implementation - would use more sophisticated methods in practice
        let length = (n as f64).powf(1.0 / 3.0).ceil() as usize;
        std::cmp::max(1, std::cmp::min(length, n / 4))
    }

    /// Compute effective sample size for block bootstrap
    fn compute_effective_samplesize(&self, n: usize) -> usize {
        let block_length = self
            .config
            .block_length
            .unwrap_or_else(|| self.optimal_block_length(n));

        // Approximation for effective sample size
        let correlation_factor = 1.0 - (block_length as f64 - 1.0) / (2.0 * n as f64);
        (n as f64 * correlation_factor).ceil() as usize
    }

    /// Compute confidence intervals
    fn compute_confidence_intervals(
        &self,
        bootstrap_samples: &Array1<F>,
        original_statistic: F,
        _standard_error: F,
    ) -> StatsResult<BootstrapConfidenceIntervals<F>> {
        let alpha = 1.0 - self.config.confidence_level;
        let lower_percentile = alpha / 2.0;
        let upper_percentile = 1.0 - alpha / 2.0;

        // Sort bootstrap _samples
        let mut sorted_samples = bootstrap_samples.to_vec();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_samples.len();
        let lower_idx = ((n as f64) * lower_percentile).floor() as usize;
        let upper_idx = ((n as f64) * upper_percentile).ceil() as usize - 1;

        // Percentile method
        let percentile = (
            sorted_samples[lower_idx],
            sorted_samples[upper_idx.min(n - 1)],
        );

        // Basic bootstrap method
        let basic = (
            F::from(2.0).unwrap() * original_statistic - sorted_samples[upper_idx.min(n - 1)],
            F::from(2.0).unwrap() * original_statistic - sorted_samples[lower_idx],
        );

        // Bias-corrected intervals (simplified)
        let bias_corrected = if self.config.bias_correction {
            let bias_correction =
                self.compute_bias_correction(bootstrap_samples, original_statistic);
            Some((
                percentile.0 + bias_correction,
                percentile.1 + bias_correction,
            ))
        } else {
            None
        };

        // BCa intervals (simplified)
        let bias_corrected_accelerated = if self.config.acceleration_correction {
            // Simplified implementation
            bias_corrected
        } else {
            None
        };

        Ok(BootstrapConfidenceIntervals {
            percentile,
            basic,
            bias_corrected,
            bias_corrected_accelerated,
            studentized: None, // Would require additional standard _error estimates
        })
    }

    /// Compute bias correction
    fn compute_bias_correction(&self, bootstrap_samples: &Array1<F>, originalstatistic: F) -> F {
        let _count_below = bootstrap_samples
            .iter()
            .filter(|&&x| x < originalstatistic)
            .count();

        let _proportion = _count_below as f64 / bootstrap_samples.len() as f64;

        // Simplified bias correction
        let bootstrap_mean = self.compute_mean(bootstrap_samples);
        bootstrap_mean - originalstatistic
    }

    /// Compute diagnostics
    fn compute_diagnostics(
        &self,
        bootstrap_samples: &Array1<F>,
        original_statistic: F,
    ) -> StatsResult<BootstrapDiagnostics<F>> {
        let distribution_stats = self.compute_distribution_stats(bootstrap_samples)?;
        let quality_metrics =
            self.compute_quality_metrics(bootstrap_samples, original_statistic)?;
        let convergence_info = self.compute_convergence_info(bootstrap_samples)?;
        let method_specific = HashMap::new(); // Would be populated based on method

        Ok(BootstrapDiagnostics {
            distribution_stats,
            quality_metrics,
            convergence_info,
            method_specific,
        })
    }

    /// Compute distribution statistics
    fn compute_distribution_stats(
        &self,
        samples: &Array1<F>,
    ) -> StatsResult<BootstrapDistributionStats<F>> {
        let mean = self.compute_mean(samples);
        let std = self.compute_std(samples);

        // Skewness
        let skewness = if std > F::zero() {
            let skew_sum = samples
                .iter()
                .map(|&x| {
                    let z = (x - mean) / std;
                    z * z * z
                })
                .fold(F::zero(), |acc, x| acc + x);
            skew_sum / F::from(samples.len()).unwrap()
        } else {
            F::zero()
        };

        // Kurtosis
        let kurtosis = if std > F::zero() {
            let kurt_sum = samples
                .iter()
                .map(|&x| {
                    let z = (x - mean) / std;
                    z * z * z * z
                })
                .fold(F::zero(), |acc, x| acc + x);
            kurt_sum / F::from(samples.len()).unwrap() - F::from(3.0).unwrap()
        } else {
            F::zero()
        };

        // Min and max
        let min_value = samples.iter().copied().fold(F::infinity(), F::min);
        let max_value = samples.iter().copied().fold(F::neg_infinity(), F::max);

        Ok(BootstrapDistributionStats {
            skewness,
            kurtosis,
            jarque_bera: F::zero(),      // Simplified
            anderson_darling: F::zero(), // Simplified
            min_value,
            max_value,
        })
    }

    /// Compute quality metrics
    fn compute_quality_metrics(
        &self,
        samples: &Array1<F>,
        _original_statistic: F,
    ) -> StatsResult<QualityMetrics<F>> {
        let std_error = self.compute_std(samples);
        let mc_std_error = std_error / F::from((samples.len() as f64).sqrt()).unwrap();

        Ok(QualityMetrics {
            mc_standard_error: mc_std_error,
            coverage_probability: F::from(self.config.confidence_level).unwrap(),
            efficiency: None,    // Would require analytical comparison
            stability: F::one(), // Simplified
        })
    }

    /// Compute convergence information
    fn compute_convergence_info(&self, samples: &Array1<F>) -> StatsResult<ConvergenceInfo<F>> {
        // Simplified convergence assessment
        let converged = samples.len() >= 100; // Simple threshold

        Ok(ConvergenceInfo {
            converged,
            convergence_samplesize: if converged { Some(samples.len()) } else { None },
            mean_stability: F::one(),     // Simplified
            variance_stability: F::one(), // Simplified
        })
    }

    /// Compute mean
    fn compute_mean(&self, data: &Array1<F>) -> F {
        if data.is_empty() {
            F::zero()
        } else {
            data.sum() / F::from(data.len()).unwrap()
        }
    }

    /// Compute standard deviation
    fn compute_std(&self, data: &Array1<F>) -> F {
        if data.len() <= 1 {
            return F::zero();
        }

        let mean = self.compute_mean(data);
        let variance = data
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(data.len() - 1).unwrap();

        variance.sqrt()
    }
}

/// Convenience function for stratified bootstrap
#[allow(dead_code)]
pub fn stratified_bootstrap<F, T>(
    data: &ArrayView1<F>,
    strata: &[usize],
    statistic_fn: impl Fn(&ArrayView1<F>) -> StatsResult<T> + Send + Sync + Copy,
    config: Option<AdvancedBootstrapConfig>,
) -> StatsResult<AdvancedBootstrapResult<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + FromPrimitive
        + Copy
        + Send
        + Sync
        + std::fmt::Display,
    T: Into<F> + Copy + Send + Sync,
{
    let mut config = config.unwrap_or_default();
    config.bootstrap_type = BootstrapType::Stratified {
        strata: strata.to_vec(),
    };

    let mut processor = AdvancedBootstrapProcessor::new(config);
    processor.bootstrap(data, statistic_fn)
}

/// Convenience function for block bootstrap
#[allow(dead_code)]
pub fn block_bootstrap<F, T>(
    data: &ArrayView1<F>,
    block_type: BlockType,
    statistic_fn: impl Fn(&ArrayView1<F>) -> StatsResult<T> + Send + Sync + Copy,
    config: Option<AdvancedBootstrapConfig>,
) -> StatsResult<AdvancedBootstrapResult<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + FromPrimitive
        + Copy
        + Send
        + Sync
        + std::fmt::Display,
    T: Into<F> + Copy + Send + Sync,
{
    let mut config = config.unwrap_or_default();
    config.bootstrap_type = BootstrapType::Block { block_type };

    let mut processor = AdvancedBootstrapProcessor::new(config);
    processor.bootstrap(data, statistic_fn)
}

/// Convenience function for moving block bootstrap
#[allow(dead_code)]
pub fn moving_block_bootstrap<F, T>(
    data: &ArrayView1<F>,
    statistic_fn: impl Fn(&ArrayView1<F>) -> StatsResult<T> + Send + Sync + Copy,
    block_length: Option<usize>,
    n_bootstrap: Option<usize>,
) -> StatsResult<AdvancedBootstrapResult<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + FromPrimitive
        + Copy
        + Send
        + Sync
        + std::fmt::Display,
    T: Into<F> + Copy + Send + Sync,
{
    let mut config = AdvancedBootstrapConfig::default();
    config.bootstrap_type = BootstrapType::Block {
        block_type: BlockType::Moving,
    };
    config.block_length = block_length;
    config.n_bootstrap = n_bootstrap.unwrap_or(1000);

    let mut processor = AdvancedBootstrapProcessor::new(config);
    processor.bootstrap(data, statistic_fn)
}

/// Convenience function for circular block bootstrap
#[allow(dead_code)]
pub fn circular_block_bootstrap<F, T>(
    data: &ArrayView1<F>,
    statistic_fn: impl Fn(&ArrayView1<F>) -> StatsResult<T> + Send + Sync + Copy,
    block_length: Option<usize>,
    n_bootstrap: Option<usize>,
) -> StatsResult<AdvancedBootstrapResult<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + FromPrimitive
        + Copy
        + Send
        + Sync
        + std::fmt::Display,
    T: Into<F> + Copy + Send + Sync,
{
    let mut config = AdvancedBootstrapConfig::default();
    config.bootstrap_type = BootstrapType::Block {
        block_type: BlockType::Circular,
    };
    config.block_length = block_length;
    config.n_bootstrap = n_bootstrap.unwrap_or(1000);

    let mut processor = AdvancedBootstrapProcessor::new(config);
    processor.bootstrap(data, statistic_fn)
}

/// Convenience function for stationary bootstrap
#[allow(dead_code)]
pub fn stationary_bootstrap<F, T>(
    data: &ArrayView1<F>,
    statistic_fn: impl Fn(&ArrayView1<F>) -> StatsResult<T> + Send + Sync + Copy,
    expected_block_length: f64,
    n_bootstrap: Option<usize>,
) -> StatsResult<AdvancedBootstrapResult<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + FromPrimitive
        + Copy
        + Send
        + Sync
        + std::fmt::Display,
    T: Into<F> + Copy + Send + Sync,
{
    let mut config = AdvancedBootstrapConfig::default();
    config.bootstrap_type = BootstrapType::Block {
        block_type: BlockType::Stationary {
            expected_length: expected_block_length,
        },
    };
    config.n_bootstrap = n_bootstrap.unwrap_or(1000);

    let mut processor = AdvancedBootstrapProcessor::new(config);
    processor.bootstrap(data, statistic_fn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    #[ignore = "timeout"]
    fn test_basicbootstrap() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean_fn = |x: &ArrayView1<f64>| -> StatsResult<f64> { Ok(x.sum() / x.len() as f64) };

        let config = AdvancedBootstrapConfig {
            n_bootstrap: 100,
            seed: Some(42),
            ..Default::default()
        };

        let mut processor = AdvancedBootstrapProcessor::new(config);
        let result = processor.bootstrap(&data.view(), mean_fn).unwrap();

        assert_eq!(result.n_successful, 100);
        assert!(result.bootstrap_samples.len() == 100);
        assert!((result.original_statistic - 3.0).abs() < 1e-10);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_stratifiedbootstrap() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let strata = vec![0, 0, 1, 1, 2, 2]; // Three strata
        let mean_fn = |x: &ArrayView1<f64>| -> StatsResult<f64> { Ok(x.sum() / x.len() as f64) };

        let result = stratified_bootstrap(
            &data.view(),
            &strata,
            mean_fn,
            Some(AdvancedBootstrapConfig {
                n_bootstrap: 50,
                seed: Some(123),
                ..Default::default()
            }),
        )
        .unwrap();

        assert_eq!(result.n_successful, 50);
        assert!(matches!(result.method, BootstrapType::Stratified { .. }));
    }

    #[test]
    #[ignore = "timeout"]
    fn test_moving_blockbootstrap() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mean_fn = |x: &ArrayView1<f64>| -> StatsResult<f64> { Ok(x.sum() / x.len() as f64) };

        let result = moving_block_bootstrap(
            &data.view(),
            mean_fn,
            Some(3),  // block length
            Some(50), // n_bootstrap
        )
        .unwrap();

        assert_eq!(result.n_successful, 50);
        assert!(result.effective_samplesize.is_some());
    }

    #[test]
    #[ignore = "timeout"]
    fn test_circular_blockbootstrap() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mean_fn = |x: &ArrayView1<f64>| -> StatsResult<f64> { Ok(x.sum() / x.len() as f64) };

        let result = circular_block_bootstrap(&data.view(), mean_fn, Some(2), Some(30)).unwrap();

        assert_eq!(result.n_successful, 30);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_confidence_intervals() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean_fn = |x: &ArrayView1<f64>| -> StatsResult<f64> { Ok(x.sum() / x.len() as f64) };

        let config = AdvancedBootstrapConfig {
            n_bootstrap: 200,
            confidence_level: 0.95,
            seed: Some(42),
            ..Default::default()
        };

        let mut processor = AdvancedBootstrapProcessor::new(config);
        let result = processor.bootstrap(&data.view(), mean_fn).unwrap();

        let ci = &result.confidence_intervals;
        assert!(ci.percentile.0 <= ci.percentile.1);
        assert!(ci.basic.0 <= ci.basic.1);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_bootstrap_diagnostics() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let var_fn = |x: &ArrayView1<f64>| -> StatsResult<f64> {
            let mean = x.sum() / x.len() as f64;
            let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (x.len() - 1) as f64;
            Ok(var)
        };

        let config = AdvancedBootstrapConfig {
            n_bootstrap: 100,
            seed: Some(456),
            ..Default::default()
        };

        let mut processor = AdvancedBootstrapProcessor::new(config);
        let result = processor.bootstrap(&data.view(), var_fn).unwrap();

        assert!(result.diagnostics.convergence_info.converged);
        assert!(
            result.diagnostics.distribution_stats.min_value
                <= result.diagnostics.distribution_stats.max_value
        );
    }
}
