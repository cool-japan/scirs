//! Training Configuration for Neural Networks
//!
//! This module contains training configuration, optimization algorithms,
//! learning rate schedules, and regularization techniques.

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfiguration {
    /// Learning rate schedule
    pub learning_rate: LearningRateSchedule,
    /// Optimization algorithm
    pub optimizer: Optimizer,
    /// Loss function
    pub loss_function: LossFunction,
    /// Regularization techniques
    pub regularization: RegularizationConfig,
    /// Training batch size
    pub batch_size: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// Validation split ratio
    pub validation_split: f64,
    /// Early stopping configuration
    pub early_stopping: EarlyStoppingConfig,
}

/// Learning rate scheduling strategies
#[derive(Debug, Clone)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant(f64),
    /// Exponential decay
    ExponentialDecay {
        initial_lr: f64,
        decay_rate: f64,
        decay_steps: usize,
    },
    /// Cosine annealing
    CosineAnnealing {
        initial_lr: f64,
        min_lr: f64,
        cycle_length: usize,
    },
    /// Step decay
    StepDecay {
        initial_lr: f64,
        drop_rate: f64,
        epochs_drop: usize,
    },
    /// Adaptive learning rate
    Adaptive {
        initial_lr: f64,
        patience: usize,
        factor: f64,
    },
}

/// Optimization algorithms
#[derive(Debug, Clone)]
pub enum Optimizer {
    /// Stochastic Gradient Descent
    SGD { momentum: f64, nesterov: bool },
    /// Adam optimizer
    Adam {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    /// AdamW (Adam with weight decay)
    AdamW {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    },
    /// RMSprop optimizer
    RMSprop { alpha: f64, epsilon: f64 },
    /// AdaGrad optimizer
    AdaGrad { epsilon: f64 },
}

/// Loss function types
#[derive(Debug, Clone, Copy)]
pub enum LossFunction {
    /// Mean Squared Error (for regression)
    MSE,
    /// Cross-entropy (for classification)
    CrossEntropy,
    /// Focal loss (for imbalanced classification)
    FocalLoss(f64, f64), // alpha, gamma
    /// Huber loss (robust regression)
    HuberLoss(f64), // delta
    /// Custom weighted loss
    WeightedMSE,
}

/// Regularization configuration
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
    /// L1 regularization strength
    pub l1_lambda: f64,
    /// L2 regularization strength
    pub l2_lambda: f64,
    /// Dropout probability
    pub dropout_prob: f64,
    /// Data augmentation techniques
    pub data_augmentation: Vec<DataAugmentation>,
    /// Label smoothing factor
    pub label_smoothing: f64,
}

/// Data augmentation techniques
#[derive(Debug, Clone)]
pub enum DataAugmentation {
    /// Add Gaussian noise
    GaussianNoise(f64), // standard deviation
    /// Time shift augmentation
    TimeShift(f64), // maximum shift ratio
    /// Scaling augmentation
    Scaling(f64, f64), // min_scale, max_scale
    /// Feature permutation
    FeaturePermutation,
    /// Mixup augmentation
    Mixup(f64), // alpha parameter
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Enable early stopping
    pub enabled: bool,
    /// Metric to monitor
    pub monitor: String,
    /// Minimum change to qualify as improvement
    pub min_delta: f64,
    /// Number of epochs with no improvement to stop
    pub patience: usize,
    /// Whether higher metric values are better
    pub maximize: bool,
}

impl Default for TrainingConfiguration {
    fn default() -> Self {
        Self {
            learning_rate: LearningRateSchedule::Constant(0.001),
            optimizer: Optimizer::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            loss_function: LossFunction::CrossEntropy,
            regularization: RegularizationConfig::default(),
            batch_size: 32,
            epochs: 100,
            validation_split: 0.2,
            early_stopping: EarlyStoppingConfig::default(),
        }
    }
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            l1_lambda: 0.0,
            l2_lambda: 0.001,
            dropout_prob: 0.1,
            data_augmentation: Vec::new(),
            label_smoothing: 0.0,
        }
    }
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitor: "val_loss".to_string(),
            min_delta: 1e-4,
            patience: 10,
            maximize: false,
        }
    }
}
