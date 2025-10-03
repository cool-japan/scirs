//! Ensemble Learning for Bifurcation Classification
//!
//! This module contains ensemble learning methods, feature selection,
//! and cross-validation configurations for bifurcation prediction.

use scirs2_core::ndarray::Array1;

/// Advanced ensemble learning for bifurcation classification
#[derive(Debug, Clone)]
pub struct BifurcationEnsembleClassifier {
    /// Individual classifiers in the ensemble
    pub base_classifiers: Vec<BaseClassifier>,
    /// Meta-learner for ensemble combination
    pub meta_learner: Option<MetaLearner>,
    /// Ensemble training strategy
    pub training_strategy: EnsembleTrainingStrategy,
    /// Cross-validation configuration
    pub cross_validation: CrossValidationConfig,
    /// Feature selection methods
    pub feature_selection: FeatureSelectionConfig,
}

/// Base classifier types for ensemble
#[derive(Debug, Clone)]
pub enum BaseClassifier {
    /// Neural network classifier
    NeuralNetwork(Box<super::neural_network::BifurcationPredictionNetwork>),
    /// Random forest classifier
    RandomForest {
        n_trees: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
    },
    /// Support Vector Machine
    SVM {
        kernel: SVMKernel,
        c_parameter: f64,
        gamma: Option<f64>,
    },
    /// Gradient boosting classifier
    GradientBoosting {
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        subsample: f64,
    },
    /// K-Nearest Neighbors
    KNN {
        n_neighbors: usize,
        weights: KNNWeights,
        distance_metric: DistanceMetric,
    },
}

/// SVM kernel types
#[derive(Debug, Clone, Copy)]
pub enum SVMKernel {
    Linear,
    RBF,
    Polynomial(usize), // degree
    Sigmoid,
}

/// KNN weight functions
#[derive(Debug, Clone, Copy)]
pub enum KNNWeights {
    Uniform,
    Distance,
}

/// Distance metrics for KNN
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Minkowski(f64), // p parameter
    Cosine,
    Hamming,
}

/// Meta-learner for ensemble combination
#[derive(Debug, Clone)]
pub enum MetaLearner {
    /// Linear combination
    LinearCombination { weights: Array1<f64> },
    /// Logistic regression meta-learner
    LogisticRegression { regularization: f64 },
    /// Neural network meta-learner
    NeuralNetwork { hidden_layers: Vec<usize> },
    /// Decision tree meta-learner
    DecisionTree { max_depth: Option<usize> },
}

/// Ensemble training strategies
#[derive(Debug, Clone)]
pub enum EnsembleTrainingStrategy {
    /// Train all models on full dataset
    FullDataset,
    /// Bootstrap aggregating (bagging)
    Bagging { n_samples: usize, replacement: bool },
    /// Cross-validation based training
    CrossValidation { n_folds: usize, stratified: bool },
    /// Stacking with holdout validation
    Stacking { holdout_ratio: f64 },
}

/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    /// Number of folds
    pub n_folds: usize,
    /// Use stratified CV
    pub stratified: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Shuffle data before splitting
    pub shuffle: bool,
}

/// Feature selection configuration
#[derive(Debug, Clone)]
pub struct FeatureSelectionConfig {
    /// Feature selection methods to apply
    pub methods: Vec<FeatureSelectionMethod>,
    /// Number of features to select
    pub n_features: Option<usize>,
    /// Selection threshold
    pub threshold: Option<f64>,
    /// Cross-validation for feature selection
    pub cross_validate: bool,
}

/// Feature selection methods
#[derive(Debug, Clone)]
pub enum FeatureSelectionMethod {
    /// Univariate statistical tests
    UnivariateSelection { score_func: ScoreFunction },
    /// Recursive feature elimination
    RecursiveElimination {
        estimator: String, // estimator type
    },
    /// L1-based selection (Lasso)
    L1BasedSelection { alpha: f64 },
    /// Tree-based feature importance
    TreeBasedSelection { importance_threshold: f64 },
    /// Mutual information
    MutualInformation,
    /// Principal component analysis
    PCA { n_components: usize },
}

/// Statistical score functions for feature selection
#[derive(Debug, Clone, Copy)]
pub enum ScoreFunction {
    /// F-statistic for classification
    FClassif,
    /// Chi-squared test
    Chi2,
    /// Mutual information for classification
    MutualInfoClassif,
    /// F-statistic for regression
    FRegression,
    /// Mutual information for regression
    MutualInfoRegression,
}
