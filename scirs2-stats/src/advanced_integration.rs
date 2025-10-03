//! Advanced Statistical Analysis Integration
//!
//! This module provides high-level interfaces that integrate multiple advanced
//! statistical methods for comprehensive data analysis workflows.

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::validation::*;

use crate::bayesian::{BayesianLinearRegression, BayesianRegressionResult};
use crate::mcmc::{GibbsSampler, MultivariateNormalGibbs};
use crate::multivariate::{FactorAnalysis, FactorAnalysisResult, PCAResult, PCA};
use crate::qmc::{halton, latin_hypercube, sobol};
use crate::survival::{CoxPHModel, KaplanMeierEstimator};

/// Comprehensive Bayesian analysis workflow
#[derive(Debug, Clone)]
pub struct BayesianAnalysisWorkflow {
    /// Enable MCMC sampling
    pub use_mcmc: bool,
    /// Number of MCMC samples
    pub n_mcmc_samples: usize,
    /// MCMC burn-in period
    pub mcmc_burnin: usize,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl Default for BayesianAnalysisWorkflow {
    fn default() -> Self {
        Self {
            use_mcmc: true,
            n_mcmc_samples: 1000,
            mcmc_burnin: 100,
            random_seed: None,
        }
    }
}

/// Results of comprehensive Bayesian analysis
#[derive(Debug, Clone)]
pub struct BayesianAnalysisResult {
    /// Bayesian regression results
    pub regression: BayesianRegressionResult,
    /// MCMC samples (if requested)
    pub mcmc_samples: Option<Array2<f64>>,
    /// Posterior predictive samples
    pub predictive_samples: Option<Array2<f64>>,
    /// Model comparison metrics
    pub model_metrics: BayesianModelMetrics,
}

/// Bayesian model comparison metrics
#[derive(Debug, Clone)]
pub struct BayesianModelMetrics {
    /// Log marginal likelihood
    pub log_marginal_likelihood: f64,
    /// Deviance Information Criterion
    pub dic: f64,
    /// Widely Applicable Information Criterion
    pub waic: f64,
    /// Leave-One-Out Cross-Validation
    pub loo_ic: f64,
}

impl BayesianAnalysisWorkflow {
    /// Create new Bayesian analysis workflow
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure MCMC settings
    pub fn with_mcmc(mut self, n_samples_: usize, burnin: usize) -> Self {
        self.use_mcmc = true;
        self.n_mcmc_samples = n_samples_;
        self.mcmc_burnin = burnin;
        self
    }

    /// Disable MCMC sampling
    pub fn without_mcmc(mut self) -> Self {
        self.use_mcmc = false;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Perform comprehensive Bayesian analysis
    pub fn analyze(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<BayesianAnalysisResult> {
        checkarray_finite(&x, "x")?;
        checkarray_finite(&y, "y")?;

        let (n_samples_, n_features) = x.dim();
        if y.len() != n_samples_ {
            return Err(StatsError::DimensionMismatch(format!(
                "y length ({}) must match x rows ({})",
                y.len(),
                n_samples_
            )));
        }

        // Perform Bayesian linear regression
        let bayesian_reg = BayesianLinearRegression::new(n_features, true)?;
        let regression = bayesian_reg.fit(x, y)?;

        // MCMC sampling if requested
        let mcmc_samples = if self.use_mcmc {
            Some(self.perform_mcmc_sampling(&regression, n_features)?)
        } else {
            None
        };

        // Generate predictive samples
        let predictive_samples = if self.use_mcmc {
            Some(self.generate_predictive_samples(&bayesian_reg, &regression, x)?)
        } else {
            None
        };

        // Compute model metrics
        let model_metrics = self.compute_model_metrics(&regression, x, y)?;

        Ok(BayesianAnalysisResult {
            regression,
            mcmc_samples,
            predictive_samples,
            model_metrics,
        })
    }

    /// Perform MCMC sampling from posterior
    fn perform_mcmc_sampling(
        &self,
        regression: &BayesianRegressionResult,
        _n_features: usize,
    ) -> Result<Array2<f64>> {
        use scirs2_core::random::{rngs::StdRng, SeedableRng};

        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use std::time::{SystemTime, UNIX_EPOCH};
                let seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                StdRng::seed_from_u64(seed)
            }
        };

        // Use Gibbs sampling for multivariate normal posterior
        let gibbs_sampler = MultivariateNormalGibbs::from_precision(
            regression.posterior_mean.clone(),
            regression.posterior_covariance.clone(),
        )?;

        let mut sampler = GibbsSampler::new(gibbs_sampler, regression.posterior_mean.clone())?;

        // Burn-in
        for _ in 0..self.mcmc_burnin {
            sampler.step(&mut rng)?;
        }

        // Collect samples
        let samples = sampler.sample(self.n_mcmc_samples, &mut rng)?;
        Ok(samples)
    }

    /// Generate posterior predictive samples
    fn generate_predictive_samples(
        &self,
        bayesian_reg: &BayesianLinearRegression,
        regression: &BayesianRegressionResult,
        x_test: ArrayView2<f64>,
    ) -> Result<Array2<f64>> {
        use scirs2_core::random::{rngs::StdRng, SeedableRng};
        use scirs2_core::random::{Distribution, Normal};

        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use std::time::{SystemTime, UNIX_EPOCH};
                let seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                StdRng::seed_from_u64(seed)
            }
        };

        let n_test = x_test.nrows();
        let mut predictive_samples = Array2::zeros((self.n_mcmc_samples, n_test));

        // Generate predictive samples
        for i in 0..self.n_mcmc_samples {
            // Sample from posterior parameter distribution
            let mut beta_sample = Array1::zeros(regression.posterior_mean.len());
            for j in 0..beta_sample.len() {
                let var = regression.posterior_covariance[[j, j]];
                let normal =
                    Normal::new(regression.posterior_mean[j], var.sqrt()).map_err(|e| {
                        StatsError::ComputationError(format!("Failed to create normal: {}", e))
                    })?;
                beta_sample[j] = normal.sample(&mut rng);
            }

            // Generate predictions with this parameter sample
            let pred_result = bayesian_reg.predict(x_test, regression)?;

            // Add noise
            let noise_std = (regression.posterior_beta / regression.posterior_alpha).sqrt();
            let noise_normal = Normal::new(0.0, noise_std).map_err(|e| {
                StatsError::ComputationError(format!("Failed to create noise normal: {}", e))
            })?;

            for j in 0..n_test {
                let noise = noise_normal.sample(&mut rng);
                predictive_samples[[i, j]] = pred_result.mean[j] + noise;
            }
        }

        Ok(predictive_samples)
    }

    /// Compute Bayesian model comparison metrics
    fn compute_model_metrics(
        &self,
        regression: &BayesianRegressionResult,
        x: ArrayView2<f64>,
        _y: ArrayView1<f64>,
    ) -> Result<BayesianModelMetrics> {
        let n_samples_ = x.nrows() as f64;
        let n_params = regression.posterior_mean.len() as f64;

        // Log marginal likelihood (already computed)
        let log_marginal_likelihood = regression.log_marginal_likelihood;

        // Simplified DIC calculation
        let deviance = -2.0 * log_marginal_likelihood;
        let effective_params = n_params; // Simplified
        let dic = deviance + 2.0 * effective_params;

        // Simplified WAIC (Watanabe-Akaike Information Criterion)
        let waic = -2.0 * log_marginal_likelihood + 2.0 * effective_params;

        // Simplified LOO-IC (Leave-One-Out Information Criterion)
        let loo_ic = -2.0 * log_marginal_likelihood
            + 2.0 * effective_params * n_samples_ / (n_samples_ - n_params - 1.0);

        Ok(BayesianModelMetrics {
            log_marginal_likelihood,
            dic,
            waic,
            loo_ic,
        })
    }
}

/// Dimensionality reduction and analysis workflow
#[derive(Debug, Clone)]
pub struct DimensionalityAnalysisWorkflow {
    /// Number of PCA components
    pub n_pca_components: Option<usize>,
    /// Number of factors for factor analysis
    pub n_factors: Option<usize>,
    /// Whether to use incremental PCA for large datasets
    pub use_incremental_pca: bool,
    /// PCA batch size (for incremental)
    pub pca_batchsize: usize,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl Default for DimensionalityAnalysisWorkflow {
    fn default() -> Self {
        Self {
            n_pca_components: None,
            n_factors: None,
            use_incremental_pca: false,
            pca_batchsize: 1000,
            random_seed: None,
        }
    }
}

/// Results of dimensionality analysis
#[derive(Debug, Clone)]
pub struct DimensionalityAnalysisResult {
    /// PCA results
    pub pca: Option<PCAResult>,
    /// Factor analysis results
    pub factor_analysis: Option<FactorAnalysisResult>,
    /// Recommended number of components/factors
    pub recommendations: DimensionalityRecommendations,
    /// Comparison metrics
    pub comparison_metrics: DimensionalityMetrics,
}

/// Recommendations for dimensionality reduction
#[derive(Debug, Clone)]
pub struct DimensionalityRecommendations {
    /// Optimal number of PCA components (Kaiser criterion)
    pub optimal_pca_components: usize,
    /// Optimal number of factors (parallel analysis)
    pub optimal_factors: usize,
    /// Variance explained by recommended components
    pub explained_variance_ratio: f64,
}

/// Comparison metrics for dimensionality reduction methods
#[derive(Debug, Clone)]
pub struct DimensionalityMetrics {
    /// Scree plot data (eigenvalues)
    pub eigenvalues: Array1<f64>,
    /// Cumulative explained variance
    pub cumulative_variance: Array1<f64>,
    /// Kaiser-Meyer-Olkin measure
    pub kmo_measure: f64,
    /// Bartlett's test statistic and p-value
    pub bartlett_test: (f64, f64),
}

impl DimensionalityAnalysisWorkflow {
    /// Create new dimensionality analysis workflow
    pub fn new() -> Self {
        Self::default()
    }

    /// Set PCA configuration
    pub fn with_pca(
        mut self,
        n_components: Option<usize>,
        incremental: bool,
        batchsize: usize,
    ) -> Self {
        self.n_pca_components = n_components;
        self.use_incremental_pca = incremental;
        self.pca_batchsize = batchsize;
        self
    }

    /// Set factor analysis configuration
    pub fn with_factor_analysis(mut self, n_factors: Option<usize>) -> Self {
        self.n_factors = n_factors;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Perform comprehensive dimensionality analysis
    pub fn analyze(&self, data: ArrayView2<f64>) -> Result<DimensionalityAnalysisResult> {
        checkarray_finite(&data, "data")?;
        let (n_samples_, n_features) = data.dim();

        if n_samples_ < 3 {
            return Err(StatsError::InvalidArgument(
                "Need at least 3 samples for analysis".to_string(),
            ));
        }

        // Perform PCA analysis
        let pca = if self.use_incremental_pca && n_samples_ > self.pca_batchsize {
            Some(self.perform_incremental_pca(data)?)
        } else {
            Some(self.perform_standard_pca(data)?)
        };

        // Perform factor analysis if requested
        let factor_analysis = if self.n_factors.is_some() {
            Some(self.perform_factor_analysis(data)?)
        } else {
            None
        };

        // Generate recommendations
        let recommendations = self.generate_recommendations(data, &pca)?;

        // Compute comparison metrics
        let comparison_metrics = self.compute_metrics(data)?;

        Ok(DimensionalityAnalysisResult {
            pca,
            factor_analysis,
            recommendations,
            comparison_metrics,
        })
    }

    /// Perform standard PCA
    fn perform_standard_pca(&self, data: ArrayView2<f64>) -> Result<PCAResult> {
        let n_components = self
            .n_pca_components
            .unwrap_or(data.ncols().min(data.nrows()));

        let pca = PCA::new()
            .with_n_components(n_components)
            .with_center(true)
            .with_scale(false);

        if let Some(seed) = self.random_seed {
            pca.with_random_state(seed).fit(data)
        } else {
            pca.fit(data)
        }
    }

    /// Perform incremental PCA for large datasets
    fn perform_incremental_pca(&self, data: ArrayView2<f64>) -> Result<PCAResult> {
        // For now, fall back to standard PCA since IncrementalPCA fields are private
        // This would need to be implemented with public accessors in the actual IncrementalPCA
        self.perform_standard_pca(data)
    }

    /// Perform factor analysis
    fn perform_factor_analysis(&self, data: ArrayView2<f64>) -> Result<FactorAnalysisResult> {
        use crate::multivariate::RotationType;

        let n_factors = self.n_factors.unwrap_or(2);

        let mut fa = FactorAnalysis::new(n_factors)?
            .with_rotation(RotationType::Varimax)
            .with_max_iter(1000)
            .with_tolerance(1e-6);

        if let Some(seed) = self.random_seed {
            fa = fa.with_random_state(seed);
        }

        fa.fit(data)
    }

    /// Generate dimensionality recommendations
    fn generate_recommendations(
        &self,
        data: ArrayView2<f64>,
        pca: &Option<PCAResult>,
    ) -> Result<DimensionalityRecommendations> {
        use crate::multivariate::{efa::parallel_analysis, mle_components};

        // Kaiser criterion for PCA (eigenvalues > 1)
        let optimal_pca_components = if let Some(ref pca_result) = pca {
            pca_result
                .explained_variance
                .iter()
                .position(|&ev| ev < 1.0)
                .unwrap_or(pca_result.explained_variance.len())
        } else {
            mle_components(data, None)?
        };

        // Parallel analysis for factor analysis
        let optimal_factors = parallel_analysis(data, 100, 95.0, self.random_seed)?;

        // Explained variance ratio
        let explained_variance_ratio = if let Some(ref pca_result) = pca {
            pca_result
                .explained_variance_ratio
                .slice(scirs2_core::ndarray::s![..optimal_pca_components])
                .sum()
        } else {
            0.0
        };

        Ok(DimensionalityRecommendations {
            optimal_pca_components,
            optimal_factors,
            explained_variance_ratio,
        })
    }

    /// Compute comparison metrics
    fn compute_metrics(&self, data: ArrayView2<f64>) -> Result<DimensionalityMetrics> {
        use crate::multivariate::efa::{bartlett_test, kmo_test};

        // Compute covariance matrix for eigenvalues
        let mean = data.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
        let mut centered = data.to_owned();
        for mut row in centered.rows_mut() {
            row -= &mean;
        }

        let cov = centered.t().dot(&centered) / (data.nrows() - 1) as f64;

        // Compute eigenvalues
        use scirs2_core::ndarray::ndarray_linalg::Eigh;
        let eigenvalues = cov
            .eigh(scirs2_core::ndarray::ndarray_linalg::UPLO::Upper)
            .map_err(|e| {
                StatsError::ComputationError(format!("Eigenvalue decomposition failed: {}", e))
            })?
            .0;

        // Sort eigenvalues in descending order
        let mut sorted_eigenvalues = eigenvalues.to_vec();
        sorted_eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let eigenvalues = Array1::from_vec(sorted_eigenvalues);

        // Cumulative variance
        let total_variance = eigenvalues.sum();
        let mut cumulative_variance = Array1::zeros(eigenvalues.len());
        let mut cumsum = 0.0;
        for i in 0..eigenvalues.len() {
            cumsum += eigenvalues[i];
            cumulative_variance[i] = cumsum / total_variance;
        }

        // KMO measure
        let kmo_measure = kmo_test(data)?;

        // Bartlett's test
        let bartlett_test = bartlett_test(data)?;

        Ok(DimensionalityMetrics {
            eigenvalues,
            cumulative_variance,
            kmo_measure,
            bartlett_test,
        })
    }
}

/// Quasi-Monte Carlo integration and optimization workflow
#[derive(Debug, Clone)]
pub struct QMCWorkflow {
    /// Sequence type
    pub sequence_type: QMCSequenceType,
    /// Whether to use scrambling
    pub scrambling: bool,
    /// Number of dimensions
    pub dimensions: usize,
    /// Number of samples
    pub n_samples_: usize,
    /// Random seed
    pub random_seed: Option<u64>,
}

/// QMC sequence types
#[derive(Debug, Clone, Copy)]
pub enum QMCSequenceType {
    /// Sobol sequence
    Sobol,
    /// Halton sequence
    Halton,
    /// Latin Hypercube sampling
    LatinHypercube,
}

/// QMC analysis results
#[derive(Debug, Clone)]
pub struct QMCResult {
    /// Generated samples
    pub samples: Array2<f64>,
    /// Sequence type used
    pub sequence_type: QMCSequenceType,
    /// Quality metrics
    pub quality_metrics: QMCQualityMetrics,
}

/// Quality metrics for QMC sequences
#[derive(Debug, Clone)]
pub struct QMCQualityMetrics {
    /// Star discrepancy
    pub star_discrepancy: f64,
    /// Uniformity measure
    pub uniformity: f64,
    /// Coverage efficiency
    pub coverage_efficiency: f64,
}

impl Default for QMCWorkflow {
    fn default() -> Self {
        Self {
            sequence_type: QMCSequenceType::Sobol,
            scrambling: true,
            dimensions: 2,
            n_samples_: 1000,
            random_seed: None,
        }
    }
}

impl QMCWorkflow {
    /// Create new QMC workflow
    pub fn new(dimensions: usize, n_samples_: usize) -> Self {
        Self {
            dimensions,
            n_samples_,
            ..Default::default()
        }
    }

    /// Set sequence type
    pub fn with_sequence_type(mut self, sequence_type: QMCSequenceType) -> Self {
        self.sequence_type = sequence_type;
        self
    }

    /// Enable or disable scrambling
    pub fn with_scrambling(mut self, scrambling: bool) -> Self {
        self.scrambling = scrambling;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Generate QMC samples with quality assessment
    pub fn generate(&self) -> Result<QMCResult> {
        check_positive(self.dimensions, "dimensions")?;
        check_positive(self.n_samples_, "n_samples_")?;

        // Generate samples based on sequence type
        let samples = match self.sequence_type {
            QMCSequenceType::Sobol => sobol(
                self.n_samples_,
                self.dimensions,
                self.scrambling,
                self.random_seed,
            )?,
            QMCSequenceType::Halton => halton(
                self.n_samples_,
                self.dimensions,
                self.scrambling,
                self.random_seed,
            )?,
            QMCSequenceType::LatinHypercube => {
                latin_hypercube(self.n_samples_, self.dimensions, self.random_seed)?
            }
        };

        // Compute quality metrics
        let quality_metrics = self.compute_quality_metrics(&samples)?;

        Ok(QMCResult {
            samples,
            sequence_type: self.sequence_type,
            quality_metrics,
        })
    }

    /// Compute quality metrics for the sequence
    fn compute_quality_metrics(&self, samples: &Array2<f64>) -> Result<QMCQualityMetrics> {
        use crate::qmc::star_discrepancy;

        // Convert to format expected by star_discrepancy
        let sample_points: Vec<Array1<f64>> = samples
            .rows()
            .into_iter()
            .map(|row| row.to_owned())
            .collect();

        let samples_view = Array1::from_vec(sample_points);
        let star_discrepancy = star_discrepancy(&samples_view.view())?;

        // Compute uniformity measure (coefficient of variation of nearest neighbor distances)
        let uniformity = self.compute_uniformity(samples)?;

        // Compute coverage efficiency
        let coverage_efficiency = self.compute_coverage_efficiency(samples)?;

        Ok(QMCQualityMetrics {
            star_discrepancy,
            uniformity,
            coverage_efficiency,
        })
    }

    /// Compute uniformity measure
    fn compute_uniformity(&self, samples: &Array2<f64>) -> Result<f64> {
        let n_samples_ = samples.nrows();
        let mut min_distances = Array1::zeros(n_samples_);

        // Compute minimum distance to other points for each sample
        for i in 0..n_samples_ {
            let mut min_dist = f64::INFINITY;
            for j in 0..n_samples_ {
                if i != j {
                    let mut dist = 0.0;
                    for k in 0..self.dimensions {
                        let diff = samples[[i, k]] - samples[[j, k]];
                        dist += diff * diff;
                    }
                    dist = dist.sqrt();
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
            }
            min_distances[i] = min_dist;
        }

        // Coefficient of variation of minimum distances
        let mean_dist = min_distances.mean().unwrap();
        let var_dist = min_distances.var(1.0);
        let uniformity = 1.0 / (var_dist.sqrt() / mean_dist); // Inverse CV

        Ok(uniformity)
    }

    /// Compute coverage efficiency
    fn compute_coverage_efficiency(&self, samples: &Array2<f64>) -> Result<f64> {
        // Simple approximation: ratio of actual coverage to expected coverage
        let n_bins = (self.n_samples_ as f64)
            .powf(1.0 / self.dimensions as f64)
            .ceil() as usize;
        let mut occupied_bins = std::collections::HashSet::new();

        for i in 0..samples.nrows() {
            let mut bin_id = Vec::new();
            for j in 0..self.dimensions {
                let bin = (samples[[i, j]] * n_bins as f64).floor() as usize;
                bin_id.push(bin.min(n_bins - 1));
            }
            occupied_bins.insert(bin_id);
        }

        let total_bins = n_bins.pow(self.dimensions as u32);
        let coverage_efficiency = occupied_bins.len() as f64 / total_bins as f64;

        Ok(coverage_efficiency)
    }
}

/// Comprehensive survival analysis workflow
#[derive(Debug, Clone)]
pub struct SurvivalAnalysisWorkflow {
    /// Confidence level for intervals
    pub confidence_level: f64,
    /// Whether to fit Cox model
    pub fit_cox_model: bool,
    /// Maximum iterations for Cox model
    pub cox_max_iter: usize,
    /// Convergence tolerance for Cox model
    pub cox_tolerance: f64,
}

impl Default for SurvivalAnalysisWorkflow {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            fit_cox_model: true,
            cox_max_iter: 100,
            cox_tolerance: 1e-6,
        }
    }
}

/// Comprehensive survival analysis results
#[derive(Debug, Clone)]
pub struct SurvivalAnalysisResult {
    /// Kaplan-Meier estimator
    pub kaplan_meier: crate::survival::KaplanMeierEstimator,
    /// Cox proportional hazards model (if requested and covariates provided)
    pub cox_model: Option<crate::survival::CoxPHModel>,
    /// Survival summary statistics
    pub summary_stats: SurvivalSummaryStats,
}

/// Summary statistics for survival analysis
#[derive(Debug, Clone)]
pub struct SurvivalSummaryStats {
    /// Median survival time
    pub median_survival: Option<f64>,
    /// 25th percentile survival time
    pub q25_survival: Option<f64>,
    /// 75th percentile survival time
    pub q75_survival: Option<f64>,
    /// Event rate
    pub event_rate: f64,
    /// Censoring rate
    pub censoring_rate: f64,
}

impl SurvivalAnalysisWorkflow {
    /// Create new survival analysis workflow
    pub fn new() -> Self {
        Self::default()
    }

    /// Set confidence level
    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Configure Cox model fitting
    pub fn with_cox_model(mut self, max_iter: usize, tolerance: f64) -> Self {
        self.fit_cox_model = true;
        self.cox_max_iter = max_iter;
        self.cox_tolerance = tolerance;
        self
    }

    /// Disable Cox model fitting
    pub fn without_cox_model(mut self) -> Self {
        self.fit_cox_model = false;
        self
    }

    /// Perform comprehensive survival analysis
    pub fn analyze(
        &self,
        durations: ArrayView1<f64>,
        events: ArrayView1<bool>,
        covariates: Option<ArrayView2<f64>>,
    ) -> Result<SurvivalAnalysisResult> {
        checkarray_finite(&durations, "durations")?;

        if durations.len() != events.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "durations length ({}) must match events length ({})",
                durations.len(),
                events.len()
            )));
        }

        // Fit Kaplan-Meier estimator
        let kaplan_meier =
            KaplanMeierEstimator::fit(durations, events, Some(self.confidence_level))?;

        // Fit Cox model if requested and covariates provided
        let cox_model = if self.fit_cox_model {
            if let Some(cov) = covariates {
                Some(CoxPHModel::fit(
                    durations,
                    events,
                    cov,
                    Some(self.cox_max_iter),
                    Some(self.cox_tolerance),
                )?)
            } else {
                None
            }
        } else {
            None
        };

        // Compute summary statistics
        let summary_stats = self.compute_summary_stats(&durations, &events, &kaplan_meier)?;

        Ok(SurvivalAnalysisResult {
            kaplan_meier,
            cox_model,
            summary_stats,
        })
    }

    /// Compute survival summary statistics
    fn compute_summary_stats(
        &self,
        _durations: &ArrayView1<f64>,
        events: &ArrayView1<bool>,
        km: &KaplanMeierEstimator,
    ) -> Result<SurvivalSummaryStats> {
        // Event and censoring rates
        let total_events: usize = events.iter().map(|&e| if e { 1 } else { 0 }).sum();
        let total_observations = events.len();
        let event_rate = total_events as f64 / total_observations as f64;
        let censoring_rate = 1.0 - event_rate;

        // Median survival time (already computed in KM estimator)
        let median_survival = km.median_survival_time;

        // Percentile survival times
        let q25_survival = self.find_survival_percentile(km, 0.75)?; // 75% survival = 25th percentile time
        let q75_survival = self.find_survival_percentile(km, 0.25)?; // 25% survival = 75th percentile time

        Ok(SurvivalSummaryStats {
            median_survival,
            q25_survival,
            q75_survival,
            event_rate,
            censoring_rate,
        })
    }

    /// Find time at which survival probability equals target
    fn find_survival_percentile(
        &self,
        km: &KaplanMeierEstimator,
        target_survival: f64,
    ) -> Result<Option<f64>> {
        for i in 0..km.survival_function.len() {
            if km.survival_function[i] <= target_survival {
                return Ok(Some(km.event_times[i]));
            }
        }
        Ok(None) // Target survival not reached
    }
}
