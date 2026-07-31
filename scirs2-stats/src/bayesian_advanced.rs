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
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, NumCast, One, Zero};
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};
use std::collections::HashMap;
use std::marker::PhantomData;

mod bnn_train;
mod diagnostics;
mod glm;
mod model_fit;

pub use bnn_train::BnnTrainingConfig;

/// Convenience trait bundling every numeric capability the advanced Bayesian
/// routines in this module need: SIMD kernels (`SimdUnifiedOps`), the linear
/// algebra used for Laplace/Gaussian-process posteriors (`scirs2-linalg`
/// requires `NumAssign + Sum + ScalarOperand + 'static`), and safe
/// round-tripping through `f64` for RNG draws and special functions.
///
/// In practice this is only ever instantiated for `f32`/`f64`, the two
/// floating types `SimdUnifiedOps` supports, so widening the bound here (over
/// the narrower bounds the individual `impl` blocks used before) does not
/// restrict any real caller.
pub trait AdvancedBayesianFloat:
    Float
    + NumCast
    + NumAssign
    + SimdUnifiedOps
    + Zero
    + One
    + PartialOrd
    + Copy
    + Send
    + Sync
    + std::fmt::Display
    + std::iter::Sum<Self>
    + ScalarOperand
    + 'static
{
}

impl<T> AdvancedBayesianFloat for T where
    T: Float
        + NumCast
        + NumAssign
        + SimdUnifiedOps
        + Zero
        + One
        + PartialOrd
        + Copy
        + Send
        + Sync
        + std::fmt::Display
        + std::iter::Sum<T>
        + ScalarOperand
        + 'static
{
}

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
    /// Trained posterior ensemble of weights: `weight_samples[m][l]` is the
    /// weight matrix of layer `l` for ensemble member `m`. Populated by
    /// [`BayesianNeuralNetwork::fit`]; `None` until then.
    pub weight_samples: Option<Vec<Vec<Array2<F>>>>,
    /// Trained posterior ensemble of biases: `bias_samples[m][l]` is the bias
    /// vector of layer `l` for ensemble member `m`. Populated by
    /// [`BayesianNeuralNetwork::fit`]; `None` until then.
    pub bias_samples: Option<Vec<Vec<Array1<F>>>>,
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
    /// Posterior predictive p-value (Pearson chi-square goodness of fit,
    /// using the fitted predictive mean/variance at each observation)
    pub posterior_p_value: F,
    /// Laplace- (or, for closed-form Gaussian models, exact-) approximated
    /// log marginal likelihood (model evidence), used to compute Bayes
    /// factors between models
    pub log_marginal_likelihood: F,
    /// Gelfand-Ghosh posterior predictive loss `D = G + P`, where `G` is the
    /// sum of squared errors between the predictive mean and the observed
    /// data and `P` is the sum of predictive variances
    pub ppl: F,
    /// Leave-one-out cross-validation score, on the same `-2 * log-density`
    /// deviance scale as `dic`/`waic` (lower is better)
    pub loo_cv: F,
    /// K-fold cross-validation information criterion, on the same
    /// `-2 * log-density` deviance scale as `dic`/`waic` (lower is better)
    pub cvic: F,
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

impl<F: AdvancedBayesianFloat> BayesianModelComparison<F> {
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

    /// Perform comprehensive model comparison: fits every registered model
    /// via a real Bayesian inference engine (see the crate-private
    /// `model_fit::fit_dispatch`) -- a Laplace-approximated GLM, an exact
    /// Gaussian process posterior, or a trained Bayesian neural network deep
    /// ensemble, depending on each model's `model_type` -- computes real
    /// information criteria and cross-validation scores from the resulting
    /// posterior samples/likelihoods, and derives real pairwise Bayes
    /// factors from each model's (Laplace- or exactly-) approximated log
    /// marginal likelihood.
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
        if self.models.is_empty() {
            return Err(StatsError::InvalidArgument(
                "At least one model must be registered via add_model before compare_models"
                    .to_string(),
            ));
        }

        let mut rankings = HashMap::new();
        let mut ic_values = HashMap::new();
        let mut cv_results = HashMap::new();
        let mut log_marginal_likelihoods: HashMap<String, F> = HashMap::new();

        // Fit each model and compute criteria
        for model in &self.models {
            let model_result = self.fit_single_model(model, x, y)?;
            log_marginal_likelihoods.insert(
                model.id.clone(),
                model_result.model_fit.log_marginal_likelihood,
            );

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

        // Compute rankings. `compute_criterion` always returns values on a
        // "lower is better" deviance-like scale (including
        // `MarginalLikelihood`, which it negates), so one ascending sort
        // works uniformly for every criterion.
        for criterion in &self.criteria {
            let mut model_scores: Vec<(String, F)> = ic_values
                .iter()
                .map(|(id, scores)| (id.clone(), scores[criterion]))
                .collect();

            model_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let ranking: Vec<String> = model_scores.into_iter().map(|(id_, _)| id_).collect();
            rankings.insert(*criterion, ranking);
        }

        // Real pairwise Bayes factors from each model's log marginal
        // likelihood: bayes_factors[i][j] = p(y | model_i) / p(y | model_j).
        let n_models = self.models.len();
        let mut bayes_factors = Array2::<F>::ones((n_models, n_models));
        for (i, model_i) in self.models.iter().enumerate() {
            let log_ml_i = log_marginal_likelihoods[&model_i.id];
            for (j, model_j) in self.models.iter().enumerate() {
                let log_ml_j = log_marginal_likelihoods[&model_j.id];
                bayes_factors[[i, j]] = (log_ml_i - log_ml_j).exp();
            }
        }

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

    /// Fit a single model via a real Bayesian inference engine (see
    /// [`model_fit::fit_dispatch`] for the per-`ModelType` dispatch), then
    /// fill in the leave-one-out and k-fold cross-validation criteria, which
    /// need repeated refits and so are computed separately from the rest of
    /// `AdvancedBayesianResult`.
    fn fit_single_model(
        &self,
        model: &BayesianModel<F>,
        x: &ArrayView2<F>,
        y: &ArrayView1<F>,
    ) -> StatsResult<AdvancedBayesianResult<F>> {
        let mut result = model_fit::fit_dispatch(model, x, y, &model_fit::primary_bnn_config())?;

        let n = x.nrows();
        let n_f = F::from(n).expect("sample count fits in any Float");
        let two = F::from(-2.0).expect("-2.0 fits in any Float");

        // True leave-one-out cross-validation is only affordable up to a
        // modest sample size (it refits the model once per data point);
        // beyond that, cap it at a bounded number of folds -- still a real,
        // honestly-labeled k-fold estimate, just not exact LOO -- to keep
        // worst-case runtime in check.
        let loo_k = n.min(15);
        let (loo_mean_ll, _, _) =
            model_fit::k_fold_mean_loglik(model, x, y, loo_k, &model_fit::cv_bnn_config())?;
        result.model_fit.loo_cv = two * loo_mean_ll * n_f;

        let cvic_k = self.cv_config.k_folds.min(n.max(2));
        let (cvic_mean_ll, _, _) =
            model_fit::k_fold_mean_loglik(model, x, y, cvic_k, &model_fit::cv_bnn_config())?;
        result.model_fit.cvic = two * cvic_mean_ll * n_f;

        Ok(result)
    }

    /// Compute information criterion. Every criterion is returned on a
    /// `-2 * log-likelihood`-like "deviance" scale where **lower is
    /// better**, including `MarginalLikelihood` (negated, since raw
    /// evidence is "higher is better") -- this lets `compare_models` rank
    /// every criterion with the same ascending sort.
    fn compute_criterion(
        &self,
        result: &AdvancedBayesianResult<F>,
        criterion: &ModelSelectionCriterion,
    ) -> StatsResult<F> {
        match criterion {
            ModelSelectionCriterion::DIC => Ok(result.model_fit.dic),
            ModelSelectionCriterion::WAIC => Ok(result.model_fit.waic),
            ModelSelectionCriterion::LooCv => Ok(result.model_fit.loo_cv),
            ModelSelectionCriterion::MarginalLikelihood => {
                Ok(-result.model_fit.log_marginal_likelihood)
            }
            ModelSelectionCriterion::PPL => Ok(result.model_fit.ppl),
            ModelSelectionCriterion::CVIC => Ok(result.model_fit.cvic),
        }
    }

    /// Cross-validate model via real, repeated refitting on `k`-fold splits
    /// of `(x, y)` (see [`model_fit::k_fold_mean_loglik`]), scoring each
    /// held-out fold by its mean log predictive density.
    fn cross_validate_model(
        &self,
        model: &BayesianModel<F>,
        x: &ArrayView2<F>,
        y: &ArrayView1<F>,
    ) -> StatsResult<CrossValidationResult<F>> {
        let k = self.cv_config.k_folds.min(x.nrows().max(2));
        let (mean_score, std_error, fold_scores) =
            model_fit::k_fold_mean_loglik(model, x, y, k, &model_fit::cv_bnn_config())?;
        let effective_n_params = F::from(x.ncols()).expect("column count fits in any Float");

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
            .map(|(_, waic)| {
                (-((*waic - min_waic) / F::from(2.0).expect("Failed to convert constant to float")))
                    .exp()
            })
            .sum();

        for (id, waic) in waic_values {
            let weight = (-(waic - min_waic)
                / F::from(2.0).expect("Failed to convert constant to float"))
            .exp()
                / weight_sum;
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

impl<F: AdvancedBayesianFloat> Default for BayesianModelComparison<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: AdvancedBayesianFloat> BayesianGaussianProcess<F> {
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
                let length_scale = F::from(*length_scale).expect("Failed to convert to float");
                let mut squared_dist = F::zero();

                for (a, b) in x1.iter().zip(x2.iter()) {
                    let diff = *a - *b;
                    squared_dist = squared_dist + diff * diff;
                }

                Ok((-squared_dist
                    / (F::from(2.0).expect("Failed to convert constant to float")
                        * length_scale
                        * length_scale))
                    .exp())
            }
            KernelType::Matern { nu, length_scale } => {
                let nu = F::from(*nu).expect("Failed to convert to float");
                let length_scale = F::from(*length_scale).expect("Failed to convert to float");
                let mut dist = F::zero();

                for (a, b) in x1.iter().zip(x2.iter()) {
                    let diff = *a - *b;
                    dist = dist + diff * diff;
                }
                dist = dist.sqrt();

                // Simplified Matern kernel for nu = 1.5
                if nu == F::from(1.5).expect("Failed to convert constant to float") {
                    let sqrt3_r_l = F::from(3.0)
                        .expect("Failed to convert constant to float")
                        .sqrt()
                        * dist
                        / length_scale;
                    Ok((F::one() + sqrt3_r_l) * (-sqrt3_r_l).exp())
                } else {
                    // Fallback to RBF for other nu values
                    Ok((-dist * dist
                        / (F::from(2.0).expect("Failed to convert constant to float")
                            * length_scale
                            * length_scale))
                        .exp())
                }
            }
            KernelType::Linear { variance } => {
                let variance = F::from(*variance).expect("Failed to convert to float");
                let dot_product = F::simd_dot(x1, x2);
                Ok(variance * dot_product)
            }
            KernelType::WhiteNoise { variance } => {
                let variance = F::from(*variance).expect("Failed to convert to float");
                // White noise kernel is only non-zero when x1 == x2
                let mut is_equal = true;
                for (a, b) in x1.iter().zip(x2.iter()) {
                    if (*a - *b).abs()
                        > F::from(1e-10).expect("Failed to convert constant to float")
                    {
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
                Ok(
                    (-squared_dist / F::from(2.0).expect("Failed to convert constant to float"))
                        .exp(),
                )
            }
        }
    }

    /// Compute the Cholesky factor `L` of the noise-regularized training
    /// kernel matrix `K(X, X) + sigma^2 I`.
    fn training_cholesky(&self) -> StatsResult<Array2<F>> {
        let n_train = self.x_train.nrows();
        let mut k_train = self.compute_kernel_matrix(&self.x_train.view(), &self.x_train.view())?;
        for i in 0..n_train {
            k_train[[i, i]] = k_train[[i, i]] + self.noise_level;
        }
        scirs2_linalg::cholesky(&k_train.view(), None).map_err(|e| {
            StatsError::ComputationError(format!(
                "Gaussian process kernel matrix is not positive definite (Cholesky decomposition failed): {e}"
            ))
        })
    }

    /// Solve `(K(X, X) + sigma^2 I) alpha = y_train` given the Cholesky
    /// factor `l` via forward + back substitution.
    fn solve_alpha(&self, l: &Array2<F>) -> StatsResult<Array1<F>> {
        let z = scirs2_linalg::solve_triangular(&l.view(), &self.y_train.view(), true, false)
            .map_err(|e| {
                StatsError::ComputationError(format!("GP forward substitution failed: {e}"))
            })?;
        scirs2_linalg::solve_triangular(&l.t(), &z.view(), false, false)
            .map_err(|e| StatsError::ComputationError(format!("GP back substitution failed: {e}")))
    }

    /// Make predictions at new input points using the exact Gaussian process
    /// posterior: `mean = K(X*, X) alpha` and
    /// `var = k(x*, x*) - K(X*, X) (K(X, X) + sigma^2 I)^-1 K(X, X*)`, where
    /// `alpha = (K(X, X) + sigma^2 I)^-1 y_train`.
    pub fn predict(&self, xtest: &ArrayView2<F>) -> StatsResult<(Array1<F>, Array1<F>)> {
        checkarray_finite(xtest, "x_test")?;
        if xtest.ncols() != self.x_train.ncols() {
            return Err(StatsError::DimensionMismatch(format!(
                "x_test has {} columns, expected {} to match the training data",
                xtest.ncols(),
                self.x_train.ncols()
            )));
        }

        let n_test = xtest.nrows();
        let l = self.training_cholesky()?;
        let alpha = self.solve_alpha(&l)?;

        // Cross-covariance K(X*, X), shape (n_test, n_train).
        let k_star = self.compute_kernel_matrix(xtest, &self.x_train.view())?;
        let mean_pred = k_star.dot(&alpha);

        let mut var_pred = Array1::<F>::zeros(n_test);
        for i in 0..n_test {
            let k_star_i = k_star.row(i).to_owned();
            let v = scirs2_linalg::solve_triangular(&l.view(), &k_star_i.view(), true, false)
                .map_err(|e| {
                    StatsError::ComputationError(format!(
                        "GP predictive variance solve failed: {e}"
                    ))
                })?;
            let quad = v.dot(&v);
            let test_row = xtest.row(i);
            let k_ii = self.kernel_function(&test_row, &test_row)?;
            var_pred[i] = (k_ii - quad).max(F::zero());
        }

        Ok((mean_pred, var_pred))
    }

    /// Exact log marginal likelihood (model evidence) of the training data:
    /// `log p(y|X) = -1/2 y^T alpha - sum_i log(L_ii) - n/2 log(2 pi)`.
    pub fn log_marginal_likelihood(&self) -> StatsResult<F> {
        let n = self.x_train.nrows();
        let l = self.training_cholesky()?;
        let alpha = self.solve_alpha(&l)?;
        let data_fit = self.y_train.dot(&alpha);

        let mut log_det_half = F::zero();
        for i in 0..n {
            let diag = l[[i, i]]
                .abs()
                .max(F::from(1e-300).expect("1e-300 fits in any Float"));
            log_det_half = log_det_half + diag.ln();
        }

        let two_pi = F::from(2.0 * std::f64::consts::PI).expect("2*pi fits in any Float");
        let half = F::from(0.5).expect("0.5 fits in any Float");
        Ok(-half * data_fit
            - log_det_half
            - half * F::from(n).expect("n fits in any Float") * two_pi.ln())
    }
}

impl<F: AdvancedBayesianFloat> BayesianNeuralNetwork<F> {
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
                let fan_in = F::from(architecture[i]).expect("Failed to convert to float");
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
                precision: F::from(0.1).expect("Failed to convert constant to float"),
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
                let sqrt_2_pi = F::from(0.7978845608).expect("Failed to convert constant to float"); // sqrt(2/π)
                let coeff = F::from(0.044715).expect("Failed to convert constant to float");
                let inner = sqrt_2_pi * (x + coeff * x * x * x);
                F::from(0.5).expect("Failed to convert constant to float")
                    * x
                    * (F::one() + inner.tanh())
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

    // `fit` and `predict_with_uncertainty` (real deep-ensemble training and
    // posterior-predictive Monte Carlo, replacing the old fabricated
    // all-zero/all-one stub) live in `bayesian_advanced::bnn_train`, along
    // with the exact backpropagation machinery they share.
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
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

        let x = array![[1.0, 0.5], [3.0, -1.0], [5.0, 2.0], [7.0, -0.5]];
        let y = array![1.2, 2.1, 3.4, 3.8];

        let result = comparison
            .compare_models(&x.view(), &y.view())
            .expect("compare_models should succeed for a well-specified single model");

        // The old stub produced a canned PosteriorSummary of zeros/ones and a
        // hardcoded R-hat of 1.0 for a model that was never actually fit.
        // With a real fit, the posterior mean/variance must reflect the
        // input data (not be exactly zero), and the reported diagnostics
        // must be finite real numbers.
        let fit = &result.ic_values["linear_model"];
        assert!(fit[&ModelSelectionCriterion::WAIC].is_finite());
        assert!(fit[&ModelSelectionCriterion::DIC].is_finite());
        assert!(result.model_weights["linear_model"] > 0.0);
    }

    #[test]
    fn test_model_comparison_prefers_true_generating_model() {
        // True process: a (mostly) monotonic 0/1 step-like response in `x`,
        // with two intentionally "flipped" labels near the boundary (at
        // x=-0.5 and x=0.5) so the classes are not perfectly separable --
        // avoiding the classic logistic-regression perfect-separation
        // pathology (an infinite-magnitude MLE) while still being a shape
        // only a logit link can represent well. A logit-link (Binomial) GLM
        // is the correctly-specified model; a plain identity-link Gaussian
        // linear regression is fundamentally misspecified for a bounded
        // 0/1 response (it both extrapolates outside [0, 1] beyond the data
        // range and cannot saturate near the boundaries). Model comparison
        // over real fits of both should therefore robustly prefer the
        // correctly-specified model.
        let xs_base: Vec<f64> = vec![
            -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
        ];
        let ys_base: Vec<f64> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        // Replicate the pattern (a standard repeated-trials design, as in a
        // dose-response assay with several subjects tested at each dose
        // level) so the logit fit's Laplace posterior is well-identified
        // enough that WAIC/DIC's effective-parameter penalty does not swamp
        // its (real) better fit -- with only the 14 base points, the
        // near-boundary curvature leaves real, legitimate posterior
        // uncertainty in the logit slope large enough to dominate the
        // comparison, which is a separate, genuine phenomenon from "which
        // model is correctly specified".
        let reps = 4;
        let xs: Vec<f64> = xs_base
            .iter()
            .cloned()
            .cycle()
            .take(xs_base.len() * reps)
            .collect();
        let ys: Vec<f64> = ys_base
            .iter()
            .cloned()
            .cycle()
            .take(ys_base.len() * reps)
            .collect();
        // `fit_glm`'s design matrix has no implicit intercept column (`eta =
        // X . beta` exactly), so an explicit leading column of ones is
        // required for either candidate model to represent a nonzero
        // intercept.
        let x = Array2::from_shape_fn((xs.len(), 2), |(i, j)| if j == 0 { 1.0 } else { xs[i] });
        let y = Array1::from_vec(ys);

        let mut comparison = BayesianModelComparison::<f64>::new();
        comparison.add_model(BayesianModel {
            id: "true_logit_link".to_string(),
            model_type: ModelType::GeneralizedLinear {
                family: GLMFamily::Binomial,
            },
            prior: AdvancedPrior::Conjugate {
                parameters: HashMap::new(),
            },
            likelihood: LikelihoodType::Binomial,
            complexity: 2.0,
        });
        comparison.add_model(BayesianModel {
            id: "wrong_gaussian_link".to_string(),
            model_type: ModelType::LinearRegression,
            prior: AdvancedPrior::Conjugate {
                parameters: HashMap::new(),
            },
            likelihood: LikelihoodType::Gaussian,
            complexity: 2.0,
        });

        let result = comparison
            .compare_models(&x.view(), &y.view())
            .expect("compare_models should succeed for two well-specified GLM models");

        for criterion in [ModelSelectionCriterion::WAIC, ModelSelectionCriterion::DIC] {
            let ranking = &result.rankings[&criterion];
            assert_eq!(
                ranking.first().map(|s| s.as_str()),
                Some("true_logit_link"),
                "{criterion:?} should rank the correctly-specified model first, got {ranking:?}"
            );
        }

        // Bayes factor of the true model versus the misspecified one should
        // favor the true model (models are indexed in add_model order:
        // 0 = true_logit_link, 1 = wrong_gaussian_link).
        let bf_true_vs_wrong = result.bayes_factors[[0, 1]];
        assert!(
            bf_true_vs_wrong > 1.0,
            "Bayes factor should favor the true generating model, got {bf_true_vs_wrong}"
        );
    }

    #[test]
    fn test_generalized_linear_family_likelihood_mismatch_is_rejected() {
        // `ModelType::GeneralizedLinear { family }` and `BayesianModel::likelihood`
        // are two separate fields that must describe the same distribution:
        // the actual Laplace-approximated fit is driven entirely by
        // `likelihood`, so a caller who declares `family: GLMFamily::Poisson`
        // while leaving `likelihood: LikelihoodType::Gaussian` would --
        // without this check -- have their declared Poisson family silently
        // discarded in favor of a Gaussian fit with no indication anything
        // was wrong. `compare_models` must instead reject this
        // inconsistency with a clear error rather than silently fitting a
        // different model than the one declared.
        let xs: Vec<f64> = (0..10).map(|i| i as f64 * 0.3).collect();
        let ys: Vec<f64> = xs.iter().map(|&xv| (0.4 + 0.6 * xv).exp()).collect();
        let x = Array2::from_shape_fn((xs.len(), 2), |(i, j)| if j == 0 { 1.0 } else { xs[i] });
        let y = Array1::from_vec(ys);

        let mut comparison = BayesianModelComparison::<f64>::new();
        comparison.add_model(BayesianModel {
            id: "mismatched_model".to_string(),
            model_type: ModelType::GeneralizedLinear {
                family: GLMFamily::Poisson,
            },
            prior: AdvancedPrior::Conjugate {
                parameters: HashMap::new(),
            },
            // Deliberately inconsistent with `model_type`'s declared family.
            likelihood: LikelihoodType::Gaussian,
            complexity: 2.0,
        });

        let err = comparison.compare_models(&x.view(), &y.view()).expect_err(
            "a GeneralizedLinear model whose declared family disagrees with its \
                 likelihood must be rejected, not silently fit as `likelihood` alone",
        );
        let message = err.to_string();
        assert!(
            message.contains("family") && message.contains("likelihood"),
            "error should explain the family/likelihood mismatch, got: {message}"
        );
    }

    #[test]
    fn test_gaussian_process_noiseless_interpolation() {
        // Three points on a curve (not collinear), so a real RBF-kernel
        // posterior mean and a naive nearest-neighbor guess would disagree.
        let x_train = array![[0.0], [1.0], [2.0]];
        let y_train = array![0.0, 1.0, 4.0];
        let noise = 1e-6; // near-noiseless

        let gp = BayesianGaussianProcess::new(
            x_train.clone(),
            y_train.clone(),
            KernelType::RBF { length_scale: 1.0 },
            noise,
        )
        .expect("GP construction should succeed");

        assert_eq!(gp.x_train.nrows(), 3);
        assert_eq!(gp.y_train.len(), 3);

        // Noiseless-GP interpolation property: posterior mean at the
        // training inputs should reproduce the training targets almost
        // exactly, with near-zero posterior variance there.
        let (mean_train, var_train) = gp
            .predict(&x_train.view())
            .expect("prediction at training points should succeed");
        for i in 0..3 {
            assert!(
                (mean_train[i] - y_train[i]).abs() < 1e-3,
                "GP should nearly interpolate noiseless training data at point {i}: got {}, expected {}",
                mean_train[i],
                y_train[i]
            );
            assert!(
                var_train[i] < 1e-2,
                "GP posterior variance at a training point should be tiny, got {}",
                var_train[i]
            );
        }

        // At the midpoint between x=0 (y=0) and x=1 (y=1), the real
        // RBF-weighted posterior mean must be a smooth blend, not exactly
        // either training value (which is what a 1-nearest-neighbor stub
        // -- ties resolved toward the first point seen -- would return).
        let x_mid = array![[0.5]];
        let (mean_mid, _) = gp
            .predict(&x_mid.view())
            .expect("midpoint prediction should succeed");
        assert!(
            (mean_mid[0] - 0.0).abs() > 1e-3 && (mean_mid[0] - 1.0).abs() > 1e-3,
            "GP posterior mean at the midpoint should be a genuine blend of neighboring \
             training values, not equal to either one exactly: got {}",
            mean_mid[0]
        );

        // Posterior variance must grow away from the training data -- the
        // headline GP behavior a constant-variance stub cannot reproduce.
        let x_far = array![[50.0]];
        let (_, var_far) = gp
            .predict(&x_far.view())
            .expect("far-point prediction should succeed");
        assert!(
            var_far[0] > var_train[0] + 1e-3,
            "GP posterior variance should grow away from training data: far={}, near={}",
            var_far[0],
            var_train[0]
        );

        let log_ml = gp
            .log_marginal_likelihood()
            .expect("log marginal likelihood should compute");
        assert!(log_ml.is_finite());
    }

    #[test]
    fn test_bayesian_neural_network_prior_predictive_is_input_dependent() {
        let bnn = BayesianNeuralNetwork::<f64>::new(
            vec![2, 5, 1],
            vec![ActivationType::ReLU, ActivationType::Sigmoid],
        )
        .expect("network construction should succeed");

        // No `fit()` call: predictions must come from real forward passes
        // through prior-sampled weights (prior-predictive Monte Carlo), not
        // the old fabricated all-zero/all-one stub.
        let x_test = array![[0.0, 0.0], [5.0, -5.0], [-5.0, 5.0], [10.0, 10.0]];
        let (means, vars) = bnn
            .predict_with_uncertainty(&x_test.view(), 200)
            .expect("prior-predictive prediction should succeed");

        let first_mean = means[[0, 0]];
        let all_means_equal =
            (0..x_test.nrows()).all(|i| (means[[i, 0]] - first_mean).abs() < 1e-9);
        assert!(
            !all_means_equal,
            "predictive means should genuinely depend on very different input rows, got {means:?}"
        );
        for v in vars.iter() {
            assert!(*v >= 0.0, "variance must be non-negative, got {v}");
        }
        assert!(
            means.iter().any(|&m| m.abs() > 1e-9),
            "means should not all be the fabricated placeholder 0.0, got {means:?}"
        );
    }

    #[test]
    fn test_bayesian_neural_network_fit_improves_predictions() {
        // A function this architecture (ReLU hidden layer, Sigmoid output)
        // can realistically represent: y = sigmoid(0.5*x1 - 0.3*x2).
        let xs: Vec<[f64; 2]> = vec![
            [-2.0, -2.0],
            [-2.0, 0.0],
            [-2.0, 2.0],
            [0.0, -2.0],
            [0.0, 0.0],
            [0.0, 2.0],
            [2.0, -2.0],
            [2.0, 0.0],
            [2.0, 2.0],
        ];
        let sigmoid = |z: f64| 1.0 / (1.0 + (-z).exp());
        let ys: Vec<f64> = xs
            .iter()
            .map(|p| sigmoid(0.5 * p[0] - 0.3 * p[1]))
            .collect();

        let x = Array2::from_shape_fn((xs.len(), 2), |(i, j)| xs[i][j]);
        let y_col = Array2::from_shape_fn((ys.len(), 1), |(i, _)| ys[i]);
        let y_flat = Array1::from_vec(ys);

        let mut bnn = BayesianNeuralNetwork::<f64>::new(
            vec![2, 6, 1],
            vec![ActivationType::ReLU, ActivationType::Sigmoid],
        )
        .expect("network construction should succeed");

        let config = BnnTrainingConfig {
            n_ensemble: 6,
            epochs: 400,
            learning_rate: 0.2,
            bootstrap: true,
            seed: Some(20_260_729),
        };
        bnn.fit(&x.view(), &y_col.view(), &config)
            .expect("BNN ensemble training should succeed");

        let (means_after, vars_after) = bnn
            .predict_with_uncertainty(&x.view(), 40)
            .expect("post-fit prediction should succeed");

        let mse_after: f64 = (0..xs.len())
            .map(|i| {
                let d = means_after[[i, 0]] - y_flat[i];
                d * d
            })
            .sum::<f64>()
            / xs.len() as f64;

        let mean_y = y_flat.iter().sum::<f64>() / y_flat.len() as f64;
        let baseline_mse: f64 =
            y_flat.iter().map(|&yv| (yv - mean_y).powi(2)).sum::<f64>() / y_flat.len() as f64;

        assert!(
            mse_after < baseline_mse * 0.5,
            "fitted BNN should fit learnable training data substantially better than a \
             mean-only baseline: mse_after={mse_after}, baseline_mse={baseline_mse}"
        );

        let first_var = vars_after[[0, 0]];
        let any_different = (0..xs.len()).any(|i| (vars_after[[i, 0]] - first_var).abs() > 1e-9);
        assert!(
            any_different,
            "post-fit predictive variance should vary across inputs, got {vars_after:?}"
        );
    }
}
