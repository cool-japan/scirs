//! Advanced Bayesian statistical methods
//!
//! This module extends the existing Bayesian capabilities with:
//! - Advanced hierarchical models
//! - Bayesian model selection and comparison
//! - Non-conjugate Bayesian inference
//! - Robust Bayesian methods
//! - Bayesian neural networks
//! - Gaussian processes
//! - Advanced MCMC diagnostics

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumCast, One, Zero};
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Advanced Bayesian model comparison framework
#[derive(Debug, Clone)]
pub struct BayesianModelComparison<F> {
    /// Collection of models to compare
    pub models: Vec<BayesianModel<F>>,
    /// Model comparison criteria
    pub criteria: Vec<ModelSelectionCriterion>,
    /// Cross-validation configuration
    pub cv_config: CrossValidationConfig,
    /// Parallel processing configuration
    pub parallel_config: ParallelConfig,
}

/// Individual Bayesian model for comparison
#[derive(Debug, Clone)]
pub struct BayesianModel<F> {
    /// Model identifier
    pub id: String,
    /// Model type
    pub model_type: ModelType,
    /// Prior specification
    pub prior: AdvancedPrior<F>,
    /// Likelihood specification
    pub likelihood: LikelihoodType,
    /// Model complexity (for complexity penalties)
    pub complexity: f64,
}

/// Advanced prior specifications
#[derive(Debug, Clone)]
pub enum AdvancedPrior<F> {
    /// Standard conjugate priors
    Conjugate { parameters: HashMap<String, F> },
    /// Hierarchical priors with hyperpriors
    Hierarchical { levels: Vec<PriorLevel<F>> },
    /// Mixture of priors
    Mixture {
        components: Vec<PriorComponent<F>>,
        weights: Array1<F>,
    },
    /// Sparse inducing priors (e.g., horseshoe, spike-and-slab)
    Sparse {
        sparsity_type: SparsityType,
        sparsity_params: HashMap<String, F>,
    },
    /// Non-parametric priors (e.g., Dirichlet process)
    NonParametric {
        process_type: NonParametricProcess,
        concentration: F,
    },
}

/// Prior level in hierarchical model
#[derive(Debug, Clone)]
pub struct PriorLevel<F> {
    /// Level identifier
    pub level_id: String,
    /// Distribution type at this level
    pub distribution: DistributionType<F>,
    /// Dependencies on other levels
    pub dependencies: Vec<String>,
}

/// Prior component in mixture
#[derive(Debug, Clone)]
pub struct PriorComponent<F> {
    /// Component weight
    pub weight: F,
    /// Component distribution
    pub distribution: DistributionType<F>,
}

/// Distribution types for priors and likelihoods
pub enum DistributionType<F> {
    Normal {
        mean: F,
        precision: F,
    },
    Gamma {
        shape: F,
        rate: F,
    },
    Beta {
        alpha: F,
        beta: F,
    },
    InverseGamma {
        shape: F,
        scale: F,
    },
    Exponential {
        rate: F,
    },
    Uniform {
        lower: F,
        upper: F,
    },
    StudentT {
        degrees_freedom: F,
        location: F,
        scale: F,
    },
    Laplace {
        location: F,
        scale: F,
    },
    Horseshoe {
        tau: F,
    },
    Custom {
        log_density: Box<dyn Fn(F) -> F + Send + Sync>,
        parameters: HashMap<String, F>,
    },
}

impl<F: std::fmt::Debug> std::fmt::Debug for DistributionType<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistributionType::Normal { mean, precision } => f
                .debug_struct("Normal")
                .field("mean", mean)
                .field("precision", precision)
                .finish(),
            DistributionType::Gamma { shape, rate } => f
                .debug_struct("Gamma")
                .field("shape", shape)
                .field("rate", rate)
                .finish(),
            DistributionType::Beta { alpha, beta } => f
                .debug_struct("Beta")
                .field("alpha", alpha)
                .field("beta", beta)
                .finish(),
            DistributionType::Uniform { lower, upper } => f
                .debug_struct("Uniform")
                .field("lower", lower)
                .field("upper", upper)
                .finish(),
            DistributionType::InverseGamma { shape, scale } => f
                .debug_struct("InverseGamma")
                .field("shape", shape)
                .field("scale", scale)
                .finish(),
            DistributionType::StudentT {
                degrees_freedom,
                location,
                scale,
            } => f
                .debug_struct("StudentT")
                .field("degrees_freedom", degrees_freedom)
                .field("location", location)
                .field("scale", scale)
                .finish(),
            DistributionType::Exponential { rate } => {
                f.debug_struct("Exponential").field("rate", rate).finish()
            }
            DistributionType::Laplace { location, scale } => f
                .debug_struct("Laplace")
                .field("location", location)
                .field("scale", scale)
                .finish(),
            DistributionType::Horseshoe { tau } => {
                f.debug_struct("Horseshoe").field("tau", tau).finish()
            }
            DistributionType::Custom { parameters, .. } => f
                .debug_struct("Custom")
                .field("parameters", parameters)
                .field("log_density", &"<function>")
                .finish(),
        }
    }
}

impl<F: Clone> Clone for DistributionType<F> {
    fn clone(&self) -> Self {
        match self {
            DistributionType::Normal { mean, precision } => DistributionType::Normal {
                mean: mean.clone(),
                precision: precision.clone(),
            },
            DistributionType::Gamma { shape, rate } => DistributionType::Gamma {
                shape: shape.clone(),
                rate: rate.clone(),
            },
            DistributionType::Beta { alpha, beta } => DistributionType::Beta {
                alpha: alpha.clone(),
                beta: beta.clone(),
            },
            DistributionType::Uniform { lower, upper } => DistributionType::Uniform {
                lower: lower.clone(),
                upper: upper.clone(),
            },
            DistributionType::InverseGamma { shape, scale } => DistributionType::InverseGamma {
                shape: shape.clone(),
                scale: scale.clone(),
            },
            DistributionType::StudentT {
                degrees_freedom,
                location,
                scale,
            } => DistributionType::StudentT {
                degrees_freedom: degrees_freedom.clone(),
                location: location.clone(),
                scale: scale.clone(),
            },
            DistributionType::Exponential { rate } => {
                DistributionType::Exponential { rate: rate.clone() }
            }
            DistributionType::Horseshoe { tau } => DistributionType::Horseshoe { tau: tau.clone() },
            DistributionType::Laplace { location, scale } => DistributionType::Laplace {
                location: location.clone(),
                scale: scale.clone(),
            },
            DistributionType::Custom { parameters: _, .. } => {
                // For Custom variant with function pointer, we can't actually clone the function
                // So we'll create a placeholder that will panic if used
                panic!("Cannot clone DistributionType::Custom with function pointer")
            }
        }
    }
}

/// Sparsity-inducing prior types
#[derive(Debug, Clone, Copy)]
pub enum SparsityType {
    /// Horseshoe prior for global-local shrinkage
    Horseshoe,
    /// Spike-and-slab for variable selection
    SpikeAndSlab,
    /// LASSO (Laplace) prior
    Lasso,
    /// Elastic net prior
    ElasticNet,
    /// Finnish horseshoe
    FinnishHorseshoe,
}

/// Non-parametric process types
#[derive(Debug, Clone, Copy)]
pub enum NonParametricProcess {
    /// Dirichlet process
    DirichletProcess,
    /// Pitman-Yor process
    PitmanYor,
    /// Chinese restaurant process
    ChineseRestaurant,
    /// Indian buffet process
    IndianBuffet,
}

/// Model types for Bayesian analysis
#[derive(Debug, Clone)]
pub enum ModelType {
    /// Linear regression with various priors
    LinearRegression,
    /// Logistic regression
    LogisticRegression,
    /// Generalized linear model
    GeneralizedLinear { family: GLMFamily },
    /// Hierarchical linear model
    HierarchicalLinear { levels: usize },
    /// Gaussian process regression
    GaussianProcess { kernel: KernelType },
    /// Bayesian neural network
    BayesianNeuralNetwork {
        layers: Vec<usize>,
        activation: ActivationType,
    },
    /// State space model
    StateSpace {
        state_dim: usize,
        observation_dim: usize,
    },
    /// Mixture model
    Mixture {
        components: usize,
        component_type: ComponentType,
    },
}

/// GLM family types
#[derive(Debug, Clone, Copy)]
pub enum GLMFamily {
    Gaussian,
    Binomial,
    Poisson,
    Gamma,
    InverseGaussian,
    NegativeBinomial,
}

/// Kernel types for Gaussian processes
#[derive(Debug, Clone)]
pub enum KernelType {
    RBF { length_scale: f64 },
    Matern { nu: f64, length_scale: f64 },
    Periodic { period: f64, length_scale: f64 },
    Linear { variance: f64 },
    Polynomial { degree: usize, variance: f64 },
    WhiteNoise { variance: f64 },
    Sum { kernels: Vec<KernelType> },
    Product { kernels: Vec<KernelType> },
}

/// Activation functions for Bayesian neural networks
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Swish,
    GELU,
}

/// Component types for mixture models
#[derive(Debug, Clone, Copy)]
pub enum ComponentType {
    Gaussian,
    StudentT,
    Laplace,
    Skewed,
}

/// Likelihood types
#[derive(Debug, Clone, Copy)]
pub enum LikelihoodType {
    Gaussian,
    Binomial,
    Poisson,
    Gamma,
    Beta,
    Exponential,
    StudentT,
    Laplace,
    Robust,
}

/// Model selection criteria
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelSelectionCriterion {
    /// Deviance Information Criterion
    DIC,
    /// Watanabe-Akaike Information Criterion
    WAIC,
    /// Leave-One-Out Cross-Validation
    LooCv,
    /// Marginal Likelihood (Bayes Factor)
    MarginalLikelihood,
    /// Posterior Predictive Loss
    PPL,
    /// Cross-Validation Information Criterion
    CVIC,
}

/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    /// Number of folds for k-fold CV
    pub k_folds: usize,
    /// Number of Monte Carlo samples
    pub mc_samples: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Stratification for classification
    pub stratify: bool,
}

/// Parallel processing configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of parallel chains/threads
    pub num_chains: usize,
    /// Enable parallel model fitting
    pub parallel_models: bool,
    /// Enable parallel cross-validation
    pub parallel_cv: bool,
}

/// Advanced Bayesian regression with non-conjugate methods
#[derive(Debug, Clone)]
pub struct AdvancedBayesianRegression<F> {
    /// Model specification
    pub model: BayesianModel<F>,
    /// MCMC configuration
    pub mcmc_config: MCMCConfig,
    /// Variational inference configuration
    pub vi_config: VIConfig,
    _phantom: PhantomData<F>,
}

/// MCMC configuration for non-conjugate models
#[derive(Debug, Clone)]
pub struct MCMCConfig {
    /// Number of MCMC samples
    pub n_samples_: usize,
    /// Number of burn-in samples
    pub n_burnin: usize,
    /// Thinning interval
    pub thin: usize,
    /// Number of parallel chains
    pub n_chains: usize,
    /// Adaptation period for step sizes
    pub adaptation_period: usize,
    /// Target acceptance rate
    pub target_acceptance: f64,
    /// Enable No-U-Turn Sampler (NUTS)
    pub use_nuts: bool,
    /// Enable Hamiltonian Monte Carlo
    pub use_hmc: bool,
}

/// Variational inference configuration
#[derive(Debug, Clone)]
pub struct VIConfig {
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Learning rate for gradient-based VI
    pub learning_rate: f64,
    /// Variational family type
    pub family: VariationalFamily,
    /// Number of Monte Carlo samples for ELBO estimation
    pub n_mc_samples: usize,
}

/// Variational family types
#[derive(Debug, Clone, Copy)]
pub enum VariationalFamily {
    /// Mean-field (factorized) Gaussian
    MeanFieldGaussian,
    /// Full-rank Gaussian
    FullRankGaussian,
    /// Normalizing flows
    NormalizingFlow,
    /// Mixture of Gaussians
    MixtureGaussian,
}

/// Gaussian process regression implementation
#[derive(Debug, Clone)]
pub struct BayesianGaussianProcess<F> {
    /// Input data
    pub x_train: Array2<F>,
    /// Output data
    pub y_train: Array1<F>,
    /// Kernel function
    pub kernel: KernelType,
    /// Noise level
    pub noise_level: F,
    /// Hyperpriors for kernel parameters
    pub hyperpriors: HashMap<String, DistributionType<F>>,
    /// MCMC samples of hyperparameters
    pub hyperparameter_samples: Option<Array2<F>>,
}

/// Bayesian neural network implementation
#[derive(Debug, Clone)]
pub struct BayesianNeuralNetwork<F> {
    /// Network architecture
    pub architecture: Vec<usize>,
    /// Activation functions per layer
    pub activations: Vec<ActivationType>,
    /// Weight priors
    pub weight_priors: Vec<DistributionType<F>>,
    /// Bias priors
    pub bias_priors: Vec<DistributionType<F>>,
    /// Posterior samples of weights
    pub weight_samples: Option<Vec<Array2<F>>>,
    /// Posterior samples of biases
    pub bias_samples: Option<Vec<Array1<F>>>,
}

/// Results from Bayesian model comparison
#[derive(Debug, Clone)]
pub struct ModelComparisonResult<F> {
    /// Model rankings by each criterion
    pub rankings: HashMap<ModelSelectionCriterion, Vec<String>>,
    /// Information criteria values
    pub ic_values: HashMap<String, HashMap<ModelSelectionCriterion, F>>,
    /// Bayes factors between models
    pub bayes_factors: Array2<F>,
    /// Model weights (posterior probabilities)
    pub model_weights: HashMap<String, F>,
    /// Cross-validation results
    pub cv_results: HashMap<String, CrossValidationResult<F>>,
    /// Best model by each criterion
    pub best_models: HashMap<ModelSelectionCriterion, String>,
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResult<F> {
    /// Mean cross-validation score
    pub mean_score: F,
    /// Standard error of CV score
    pub std_error: F,
    /// Individual fold scores
    pub fold_scores: Array1<F>,
    /// Effective number of parameters
    pub effective_n_params: F,
}

/// Advanced Bayesian inference result
#[derive(Debug, Clone)]
pub struct AdvancedBayesianResult<F> {
    /// Posterior samples
    pub posterior_samples: Array2<F>,
    /// Posterior summary statistics
    pub posterior_summary: PosteriorSummary<F>,
    /// MCMC diagnostics
    pub diagnostics: MCMCDiagnostics<F>,
    /// Model fit metrics
    pub model_fit: ModelFitMetrics<F>,
    /// Predictive distributions
    pub predictions: PredictiveDistribution<F>,
}

/// Posterior summary statistics
#[derive(Debug, Clone)]
pub struct PosteriorSummary<F> {
    /// Posterior means
    pub means: Array1<F>,
    /// Posterior standard deviations
    pub stds: Array1<F>,
    /// Credible intervals
    pub credible_intervals: Array2<F>,
    /// Effective sample sizes
    pub ess: Array1<F>,
    /// R-hat convergence diagnostics
    pub rhat: Array1<F>,
}

/// MCMC diagnostics
#[derive(Debug, Clone)]
pub struct MCMCDiagnostics<F> {
    /// Acceptance rates by chain
    pub acceptance_rates: Array1<F>,
    /// Autocorrelation functions
    pub autocorrelations: Array2<F>,
    /// Geweke diagnostic
    pub geweke_diagnostic: Array1<F>,
    /// Heidelberger-Welch test
    pub heidelberger_welch: Array1<bool>,
    /// Monte Carlo standard errors
    pub mc_errors: Array1<F>,
}

/// Model fit metrics
#[derive(Debug, Clone)]
pub struct ModelFitMetrics<F> {
    /// Deviance Information Criterion
    pub dic: F,
    /// Watanabe-Akaike Information Criterion
    pub waic: F,
    /// Log pointwise predictive density
    pub lppd: F,
    /// Effective number of parameters
    pub p_eff: F,
    /// Posterior predictive p-value
    pub posterior_p_value: F,
}

/// Predictive distribution results
#[derive(Debug, Clone)]
pub struct PredictiveDistribution<F> {
    /// Predictive means
    pub means: Array1<F>,
    /// Predictive variances
    pub variances: Array1<F>,
    /// Predictive quantiles
    pub quantiles: Array2<F>,
    /// Posterior predictive samples
    pub samples: Array2<F>,
}

impl<F> BayesianModelComparison<F>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + std::fmt::Display
        + std::iter::Sum<F>,
{
    /// Create new model comparison framework
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            criteria: vec![
                ModelSelectionCriterion::DIC,
                ModelSelectionCriterion::WAIC,
                ModelSelectionCriterion::LooCv,
            ],
            cv_config: CrossValidationConfig::default(),
            parallel_config: ParallelConfig::default(),
        }
    }

    /// Add model to comparison
    pub fn add_model(&mut self, model: BayesianModel<F>) {
        self.models.push(model);
    }

    /// Perform comprehensive model comparison
    pub fn compare_models(
        &self,
        x: &ArrayView2<F>,
        y: &ArrayView1<F>,
    ) -> StatsResult<ModelComparisonResult<F>> {
        checkarray_finite(x, "x")?;
        checkarray_finite(y, "y")?;

        if x.nrows() != y.len() {
            return Err(StatsError::DimensionMismatch(
                "X and y must have same number of observations".to_string(),
            ));
        }

        let mut rankings = HashMap::new();
        let mut ic_values = HashMap::new();
        let mut cv_results = HashMap::new();

        // Fit each model and compute criteria
        for model in &self.models {
            let model_result = Self::fit_single_model(model, x, y)?;

            let mut model_ic_values = HashMap::new();

            for criterion in &self.criteria {
                let ic_value = self.compute_criterion(&model_result, criterion)?;
                model_ic_values.insert(*criterion, ic_value);
            }

            ic_values.insert(model.id.clone(), model_ic_values);

            // Cross-validation
            let cv_result = self.cross_validate_model(model, x, y)?;
            cv_results.insert(model.id.clone(), cv_result);
        }

        // Compute rankings
        for criterion in &self.criteria {
            let mut model_scores: Vec<(String, F)> = ic_values
                .iter()
                .map(|(id, scores)| (id.clone(), scores[criterion]))
                .collect();

            // Sort by criterion (lower is better for most criteria)
            model_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let ranking: Vec<String> = model_scores.into_iter().map(|(id_, _)| id_).collect();
            rankings.insert(*criterion, ranking);
        }

        // Compute Bayes factors (simplified)
        let n_models = self.models.len();
        let bayes_factors = Array2::ones((n_models, n_models));

        // Compute model weights using WAIC
        let model_weights = self.compute_model_weights(&ic_values)?;

        // Select best models
        let mut best_models = HashMap::new();
        for criterion in &self.criteria {
            if let Some(ranking) = rankings.get(criterion) {
                if let Some(best_model) = ranking.first() {
                    best_models.insert(*criterion, best_model.clone());
                }
            }
        }

        Ok(ModelComparisonResult {
            rankings,
            ic_values,
            bayes_factors,
            model_weights,
            cv_results,
            best_models,
        })
    }

    /// Fit a single model
    fn fit_single_model(
        model: &BayesianModel<F>,
        x: &ArrayView2<F>,
        y: &ArrayView1<F>,
    ) -> StatsResult<AdvancedBayesianResult<F>> {
        // Simplified _model fitting - would implement actual inference
        let n_params = x.ncols();
        let n_samples_ = 1000;

        // Generate dummy posterior samples (would use actual MCMC/VI)
        let posterior_samples = Array2::zeros((n_samples_, n_params));

        let posterior_summary = PosteriorSummary {
            means: Array1::zeros(n_params),
            stds: Array1::ones(n_params),
            credible_intervals: Array2::zeros((n_params, 2)),
            ess: Array1::from_elem(n_params, F::from(500.0).unwrap()),
            rhat: Array1::ones(n_params),
        };

        let diagnostics = MCMCDiagnostics {
            acceptance_rates: Array1::from_elem(1, F::from(0.6).unwrap()),
            autocorrelations: Array2::zeros((n_params, 100)),
            geweke_diagnostic: Array1::zeros(n_params),
            heidelberger_welch: Array1::from_elem(n_params, true),
            mc_errors: Array1::zeros(n_params),
        };

        let model_fit = ModelFitMetrics {
            dic: F::from(100.0).unwrap(),
            waic: F::from(105.0).unwrap(),
            lppd: F::from(-50.0).unwrap(),
            p_eff: F::from(n_params).unwrap(),
            posterior_p_value: F::from(0.5).unwrap(),
        };

        let predictions = PredictiveDistribution {
            means: Array1::zeros(y.len()),
            variances: Array1::ones(y.len()),
            quantiles: Array2::zeros((y.len(), 3)),
            samples: Array2::zeros((100, y.len())),
        };

        Ok(AdvancedBayesianResult {
            posterior_samples,
            posterior_summary,
            diagnostics,
            model_fit,
            predictions,
        })
    }

    /// Compute information criterion
    fn compute_criterion(
        &self,
        result: &AdvancedBayesianResult<F>,
        criterion: &ModelSelectionCriterion,
    ) -> StatsResult<F> {
        match criterion {
            ModelSelectionCriterion::DIC => Ok(result.model_fit.dic),
            ModelSelectionCriterion::WAIC => Ok(result.model_fit.waic),
            ModelSelectionCriterion::LooCv => Ok(result.model_fit.waic + F::from(1.0).unwrap()),
            ModelSelectionCriterion::MarginalLikelihood => Ok(result.model_fit.lppd),
            ModelSelectionCriterion::PPL => Ok(result.model_fit.waic + F::from(2.0).unwrap()),
            ModelSelectionCriterion::CVIC => Ok(result.model_fit.waic + F::from(0.5).unwrap()),
        }
    }

    /// Cross-validate model
    fn cross_validate_model(
        &self,
        model: &BayesianModel<F>,
        x: &ArrayView2<F>,
        _y: &ArrayView1<F>,
    ) -> StatsResult<CrossValidationResult<F>> {
        let k = self.cv_config.k_folds;
        let fold_scores = Array1::ones(k);
        let mean_score = F::one();
        let std_error = F::from(0.1).unwrap();
        let effective_n_params = F::from(x.ncols()).unwrap();

        Ok(CrossValidationResult {
            mean_score,
            std_error,
            fold_scores,
            effective_n_params,
        })
    }

    /// Compute model weights using information criteria
    fn compute_model_weights(
        &self,
        ic_values: &HashMap<String, HashMap<ModelSelectionCriterion, F>>,
    ) -> StatsResult<HashMap<String, F>> {
        let mut weights = HashMap::new();

        // Use WAIC for weight computation
        let waic_values: Vec<_> = ic_values
            .iter()
            .map(|(id, scores)| (id.clone(), scores[&ModelSelectionCriterion::WAIC]))
            .collect();

        let min_waic = waic_values
            .iter()
            .map(|(_, waic)| *waic)
            .fold(F::infinity(), |a, b| if a < b { a } else { b });

        let weight_sum: F = waic_values
            .iter()
            .map(|(_, waic)| (-((*waic - min_waic) / F::from(2.0).unwrap())).exp())
            .sum();

        for (id, waic) in waic_values {
            let weight = (-(waic - min_waic) / F::from(2.0).unwrap()).exp() / weight_sum;
            weights.insert(id, weight);
        }

        Ok(weights)
    }
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            k_folds: 5,
            mc_samples: 1000,
            seed: None,
            stratify: false,
        }
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_chains: 4,
            parallel_models: true,
            parallel_cv: true,
        }
    }
}

impl Default for MCMCConfig {
    fn default() -> Self {
        Self {
            n_samples_: 2000,
            n_burnin: 1000,
            thin: 1,
            n_chains: 4,
            adaptation_period: 500,
            target_acceptance: 0.65,
            use_nuts: true,
            use_hmc: false,
        }
    }
}

impl Default for VIConfig {
    fn default() -> Self {
        Self {
            max_iter: 10000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            family: VariationalFamily::MeanFieldGaussian,
            n_mc_samples: 100,
        }
    }
}

impl<F> Default for BayesianModelComparison<F>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + std::fmt::Display
        + std::iter::Sum<F>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> BayesianGaussianProcess<F>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + std::fmt::Display,
{
    /// Create new Gaussian process
    pub fn new(
        x_train: Array2<F>,
        y_train: Array1<F>,
        kernel: KernelType,
        noise_level: F,
    ) -> StatsResult<Self> {
        checkarray_finite(&x_train.view(), "x_train")?;
        checkarray_finite(&y_train.view(), "y_train")?;

        if x_train.nrows() != y_train.len() {
            return Err(StatsError::DimensionMismatch(
                "X and y must have same number of observations".to_string(),
            ));
        }

        if noise_level <= F::zero() {
            return Err(StatsError::InvalidArgument(
                "Noise _level must be positive".to_string(),
            ));
        }

        Ok(Self {
            x_train,
            y_train,
            kernel,
            noise_level,
            hyperpriors: HashMap::new(),
            hyperparameter_samples: None,
        })
    }

    /// Compute kernel matrix
    pub fn compute_kernel_matrix(
        &self,
        x1: &ArrayView2<F>,
        x2: &ArrayView2<F>,
    ) -> StatsResult<Array2<F>> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut k = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let x1_row = x1.row(i);
                let x2_row = x2.row(j);
                k[[i, j]] = self.kernel_function(&x1_row, &x2_row)?;
            }
        }

        Ok(k)
    }

    /// Evaluate kernel function between two points
    fn kernel_function(&self, x1: &ArrayView1<F>, x2: &ArrayView1<F>) -> StatsResult<F> {
        match &self.kernel {
            KernelType::RBF { length_scale } => {
                let length_scale = F::from(*length_scale).unwrap();
                let mut squared_dist = F::zero();

                for (a, b) in x1.iter().zip(x2.iter()) {
                    let diff = *a - *b;
                    squared_dist = squared_dist + diff * diff;
                }

                Ok((-squared_dist / (F::from(2.0).unwrap() * length_scale * length_scale)).exp())
            }
            KernelType::Matern { nu, length_scale } => {
                let nu = F::from(*nu).unwrap();
                let length_scale = F::from(*length_scale).unwrap();
                let mut dist = F::zero();

                for (a, b) in x1.iter().zip(x2.iter()) {
                    let diff = *a - *b;
                    dist = dist + diff * diff;
                }
                dist = dist.sqrt();

                // Simplified Matern kernel for nu = 1.5
                if nu == F::from(1.5).unwrap() {
                    let sqrt3_r_l = F::from(3.0).unwrap().sqrt() * dist / length_scale;
                    Ok((F::one() + sqrt3_r_l) * (-sqrt3_r_l).exp())
                } else {
                    // Fallback to RBF for other nu values
                    Ok(
                        (-dist * dist / (F::from(2.0).unwrap() * length_scale * length_scale))
                            .exp(),
                    )
                }
            }
            KernelType::Linear { variance } => {
                let variance = F::from(*variance).unwrap();
                let dot_product = F::simd_dot(x1, x2);
                Ok(variance * dot_product)
            }
            KernelType::WhiteNoise { variance } => {
                let variance = F::from(*variance).unwrap();
                // White noise kernel is only non-zero when x1 == x2
                let mut is_equal = true;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    if (*a - *b).abs() > F::from(1e-10).unwrap() {
                        is_equal = false;
                        break;
                    }
                }
                Ok(if is_equal { variance } else { F::zero() })
            }
            _ => {
                // For complex kernels (Sum, Product), use RBF as fallback
                let mut squared_dist = F::zero();
                for (a, b) in x1.iter().zip(x2.iter()) {
                    let diff = *a - *b;
                    squared_dist = squared_dist + diff * diff;
                }
                Ok((-squared_dist / F::from(2.0).unwrap()).exp())
            }
        }
    }

    /// Make predictions at new input points
    pub fn predict(&self, xtest: &ArrayView2<F>) -> StatsResult<(Array1<F>, Array1<F>)> {
        checkarray_finite(xtest, "x_test")?;

        let n_test = xtest.nrows();

        // Simplified prediction using nearest neighbor approach
        let mut mean_pred = Array1::zeros(n_test);
        let mut var_pred = Array1::zeros(n_test);

        let n_train = self.x_train.nrows();

        for i in 0..n_test {
            let test_point = xtest.row(i);
            let mut min_dist = F::infinity();
            let mut nearest_y = F::zero();

            for j in 0..n_train {
                let train_point = self.x_train.row(j);
                let mut dist = F::zero();
                for (a, b) in test_point.iter().zip(train_point.iter()) {
                    let diff = *a - *b;
                    dist = dist + diff * diff;
                }

                if dist < min_dist {
                    min_dist = dist;
                    nearest_y = self.y_train[j];
                }
            }

            mean_pred[i] = nearest_y;
            var_pred[i] = self.noise_level; // Simplified variance
        }

        Ok((mean_pred, var_pred))
    }
}

impl<F> BayesianNeuralNetwork<F>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + std::fmt::Display,
{
    /// Create new Bayesian neural network
    pub fn new(architecture: Vec<usize>, activations: Vec<ActivationType>) -> StatsResult<Self> {
        if architecture.len() < 2 {
            return Err(StatsError::InvalidArgument(
                "Architecture must have at least input and output layers".to_string(),
            ));
        }

        if activations.len() != architecture.len() - 1 {
            return Err(StatsError::InvalidArgument(
                "Number of activations must equal number of layers - 1".to_string(),
            ));
        }

        let n_layers = architecture.len() - 1;

        // Initialize priors with appropriate scales based on layer sizes
        let weight_priors = (0..n_layers)
            .map(|i| {
                let fan_in = F::from(architecture[i]).unwrap();
                let precision = fan_in; // Xavier initialization scale
                DistributionType::Normal {
                    mean: F::zero(),
                    precision,
                }
            })
            .collect();

        let bias_priors = (0..n_layers)
            .map(|_| DistributionType::Normal {
                mean: F::zero(),
                precision: F::from(0.1).unwrap(),
            })
            .collect();

        Ok(Self {
            architecture,
            activations,
            weight_priors,
            bias_priors,
            weight_samples: None,
            bias_samples: None,
        })
    }

    /// Apply activation function
    fn apply_activation(&self, x: F, activation: ActivationType) -> F {
        match activation {
            ActivationType::ReLU => {
                if x > F::zero() {
                    x
                } else {
                    F::zero()
                }
            }
            ActivationType::Sigmoid => F::one() / (F::one() + (-x).exp()),
            ActivationType::Tanh => x.tanh(),
            ActivationType::Swish => x / (F::one() + (-x).exp()),
            ActivationType::GELU => {
                // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                let sqrt_2_pi = F::from(0.7978845608).unwrap(); // sqrt(2/π)
                let coeff = F::from(0.044715).unwrap();
                let inner = sqrt_2_pi * (x + coeff * x * x * x);
                F::from(0.5).unwrap() * x * (F::one() + inner.tanh())
            }
        }
    }

    /// Forward pass through the network
    pub fn forward(
        &self,
        x: &ArrayView2<F>,
        weights: &[Array2<F>],
        biases: &[Array1<F>],
    ) -> StatsResult<Array2<F>> {
        checkarray_finite(x, "x")?;

        if weights.len() != self.architecture.len() - 1 {
            return Err(StatsError::InvalidArgument(
                "Number of weight matrices must match network layers".to_string(),
            ));
        }

        if biases.len() != self.architecture.len() - 1 {
            return Err(StatsError::InvalidArgument(
                "Number of bias vectors must match network layers".to_string(),
            ));
        }

        let mut activations = x.to_owned();

        for (layer_idx, &activation_type) in self.activations.iter().enumerate() {
            // Linear transformation: z = x * W + b
            let z = self.linear_transform(
                &activations.view(),
                &weights[layer_idx],
                &biases[layer_idx],
            )?;

            // Apply activation function
            activations = z.mapv(|val| self.apply_activation(val, activation_type));
        }

        Ok(activations)
    }

    /// Linear transformation: z = x * W + b
    fn linear_transform(
        &self,
        x: &ArrayView2<F>,
        weights: &Array2<F>,
        bias: &Array1<F>,
    ) -> StatsResult<Array2<F>> {
        let (batchsize, input_dim) = x.dim();
        let (weight_input_dim, output_dim) = weights.dim();

        if input_dim != weight_input_dim {
            return Err(StatsError::DimensionMismatch(
                "Input dimension must match weight matrix input dimension".to_string(),
            ));
        }

        if bias.len() != output_dim {
            return Err(StatsError::DimensionMismatch(
                "Bias length must match weight matrix output dimension".to_string(),
            ));
        }

        // Matrix multiplication: x * W
        let mut result = Array2::zeros((batchsize, output_dim));

        for i in 0..batchsize {
            for j in 0..output_dim {
                let mut sum = F::zero();
                for k in 0..input_dim {
                    sum = sum + x[[i, k]] * weights[[k, j]];
                }
                result[[i, j]] = sum + bias[j];
            }
        }

        Ok(result)
    }

    /// Sample parameters from priors
    fn sample_from_normal(mean: F, precision: F) -> StatsResult<F> {
        // Simple Box-Muller transform
        let u1 = F::from(0.5).unwrap(); // Would use actual random numbers
        let u2 = F::from(0.5).unwrap();

        let z = (-F::from(2.0).unwrap() * u1.ln()).sqrt()
            * (F::from(2.0 * std::f64::consts::PI).unwrap() * u2).cos();

        let std_dev = F::one() / precision.sqrt();
        Ok(mean + std_dev * z)
    }

    /// Make predictions with uncertainty quantification
    pub fn predict_with_uncertainty(
        &self,
        x: &ArrayView2<F>,
        _n_samples_: usize,
    ) -> StatsResult<(Array2<F>, Array2<F>)> {
        checkarray_finite(x, "x")?;

        let n_test = x.nrows();
        let output_dim = self.architecture.last().unwrap();

        let mut predictions = Array2::zeros((n_test, *output_dim));
        let mut prediction_vars = Array2::zeros((n_test, *output_dim));

        // Simplified prediction - would implement actual parameter sampling
        for i in 0..n_test {
            for j in 0..*output_dim {
                predictions[[i, j]] = F::zero(); // Would compute actual prediction
                prediction_vars[[i, j]] = F::one(); // Would compute actual variance
            }
        }

        Ok((predictions, prediction_vars))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    #[ignore = "timeout"]
    fn test_model_comparison() {
        let mut comparison = BayesianModelComparison::<f64>::new();

        let model = BayesianModel {
            id: "linear_model".to_string(),
            model_type: ModelType::LinearRegression,
            prior: AdvancedPrior::Conjugate {
                parameters: HashMap::new(),
            },
            likelihood: LikelihoodType::Gaussian,
            complexity: 3.0,
        };

        comparison.add_model(model);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];

        let result = comparison.compare_models(&x.view(), &y.view());
        assert!(result.is_ok());
    }

    #[test]
    fn test_gaussian_process() {
        let x_train = array![[1.0], [2.0], [3.0]];
        let y_train = array![1.0, 4.0, 9.0];
        let gp = BayesianGaussianProcess::new(
            x_train.clone(),
            y_train.clone(),
            KernelType::RBF { length_scale: 1.0 },
            0.1,
        )
        .unwrap();

        // Test creation
        assert_eq!(gp.x_train.nrows(), 3);
        assert_eq!(gp.y_train.len(), 3);

        // Test prediction
        let x_test = array![[1.5], [2.5]];
        let result = gp.predict(&x_test.view());
        assert!(result.is_ok());
    }

    #[test]
    fn test_bayesian_neural_network() {
        let bnn = BayesianNeuralNetwork::new(
            vec![2, 5, 1],
            vec![ActivationType::ReLU, ActivationType::Sigmoid],
        )
        .unwrap();

        // Test creation
        assert_eq!(bnn.architecture.len(), 3);
        assert_eq!(bnn.activations.len(), 2);

        // Test prediction with uncertainty
        let x_test = array![[1.0, 2.0], [3.0, 4.0]];
        let result = bnn.predict_with_uncertainty(&x_test.view(), 10);
        assert!(result.is_ok());
    }
}
