//! Uncertainty Quantification and Performance Metrics
//!
//! This module contains structures for uncertainty quantification, performance tracking,
//! and various methods for estimating prediction uncertainty.

use scirs2_core::ndarray::{Array1, Array2};

/// Model performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Training metrics
    pub training_metrics: Vec<EpochMetrics>,
    /// Validation metrics
    pub validation_metrics: Vec<EpochMetrics>,
    /// Test metrics
    pub test_metrics: Option<TestMetrics>,
    /// Confusion matrix (for classification)
    pub confusion_matrix: Option<Array2<usize>>,
    /// Feature importance scores
    pub feature_importance: Option<Array1<f64>>,
}

/// Metrics for each training epoch
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    /// Epoch number
    pub epoch: usize,
    /// Loss value
    pub loss: f64,
    /// Accuracy (for classification)
    pub accuracy: Option<f64>,
    /// Precision scores per class
    pub precision: Option<Vec<f64>>,
    /// Recall scores per class
    pub recall: Option<Vec<f64>>,
    /// F1 scores per class
    pub f1_score: Option<Vec<f64>>,
    /// Learning rate used
    pub learning_rate: f64,
}

/// Test set evaluation metrics
#[derive(Debug, Clone)]
pub struct TestMetrics {
    /// Overall accuracy
    pub accuracy: f64,
    /// Precision per class
    pub precision: Vec<f64>,
    /// Recall per class
    pub recall: Vec<f64>,
    /// F1 score per class
    pub f1_score: Vec<f64>,
    /// Area under ROC curve
    pub auc_roc: f64,
    /// Area under precision-recall curve
    pub auc_pr: f64,
    /// Matthews correlation coefficient
    pub mcc: f64,
}

/// Uncertainty quantification for predictions
#[derive(Debug, Clone, Default)]
pub struct UncertaintyQuantification {
    /// Bayesian neural network configuration
    pub bayesian_config: Option<BayesianConfig>,
    /// Monte Carlo dropout configuration
    pub mc_dropout_config: Option<MCDropoutConfig>,
    /// Ensemble configuration
    pub ensemble_config: Option<EnsembleConfig>,
    /// Conformal prediction configuration
    pub conformal_config: Option<ConformalConfig>,
}

/// Bayesian neural network configuration
#[derive(Debug, Clone)]
pub struct BayesianConfig {
    /// Prior distribution parameters
    pub prior_params: PriorParams,
    /// Variational inference method
    pub variational_method: VariationalMethod,
    /// Number of Monte Carlo samples
    pub mc_samples: usize,
    /// KL divergence weight
    pub kl_weight: f64,
}

/// Prior distribution parameters
#[derive(Debug, Clone)]
pub struct PriorParams {
    /// Weight prior mean
    pub weight_mean: f64,
    /// Weight prior standard deviation
    pub weight_std: f64,
    /// Bias prior mean
    pub bias_mean: f64,
    /// Bias prior standard deviation
    pub bias_std: f64,
}

/// Variational inference methods
#[derive(Debug, Clone, Copy)]
pub enum VariationalMethod {
    /// Mean-field variational inference
    MeanField,
    /// Matrix-variate Gaussian
    MatrixVariate,
    /// Normalizing flows
    NormalizingFlows,
}

/// Monte Carlo dropout configuration
#[derive(Debug, Clone)]
pub struct MCDropoutConfig {
    /// Dropout rate during inference
    pub dropoutrate: f64,
    /// Number of forward passes
    pub num_samples: usize,
    /// Use different dropout masks
    pub stochastic_masks: bool,
}

/// Ensemble configuration
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    /// Number of models in ensemble
    pub num_models: usize,
    /// Ensemble aggregation method
    pub aggregation_method: EnsembleAggregation,
    /// Diversity encouragement method
    pub diversity_method: DiversityMethod,
}

/// Ensemble aggregation methods
#[derive(Debug, Clone, Copy)]
pub enum EnsembleAggregation {
    /// Simple averaging
    Average,
    /// Weighted averaging
    WeightedAverage,
    /// Voting (for classification)
    Voting,
    /// Stacking with meta-learner
    Stacking,
}

/// Methods to encourage diversity in ensemble
#[derive(Debug, Clone, Copy)]
pub enum DiversityMethod {
    /// Bootstrap aggregating
    Bagging,
    /// Different random initializations
    RandomInit,
    /// Different architectures
    DifferentArchitectures,
    /// Adversarial training
    AdversarialTraining,
}

/// Conformal prediction configuration
#[derive(Debug, Clone)]
pub struct ConformalConfig {
    /// Confidence level (e.g., 0.95 for 95% confidence)
    pub confidence_level: f64,
    /// Conformity score function
    pub score_function: ConformityScore,
    /// Calibration set size
    pub calibration_size: usize,
}

/// Conformity score functions
#[derive(Debug, Clone, Copy)]
pub enum ConformityScore {
    /// Absolute residuals (for regression)
    AbsoluteResiduals,
    /// Normalized residuals
    NormalizedResiduals,
    /// Softmax scores (for classification)
    SoftmaxScores,
    /// Margin scores
    MarginScores,
}
