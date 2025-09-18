//! Core types and configurations for graph embeddings

use crate::base::Node;
use std::collections::HashMap;

/// Configuration for Node2Vec embedding algorithm
#[derive(Debug, Clone)]
pub struct Node2VecConfig {
    /// Dimensions of the embedding vectors
    pub dimensions: usize,
    /// Length of each random walk
    pub walk_length: usize,
    /// Number of random walks per node
    pub num_walks: usize,
    /// Window size for skip-gram model
    pub window_size: usize,
    /// Return parameter p (likelihood of immediate revisiting)
    pub p: f64,
    /// In-out parameter q (exploration vs exploitation)
    pub q: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Number of negative samples for training
    pub negative_samples: usize,
}

impl Default for Node2VecConfig {
    fn default() -> Self {
        Node2VecConfig {
            dimensions: 128,
            walk_length: 80,
            num_walks: 10,
            window_size: 10,
            p: 1.0,
            q: 1.0,
            epochs: 1,
            learning_rate: 0.025,
            negative_samples: 5,
        }
    }
}

/// Configuration for DeepWalk embedding algorithm
#[derive(Debug, Clone)]
pub struct DeepWalkConfig {
    /// Dimensions of the embedding vectors
    pub dimensions: usize,
    /// Length of each random walk
    pub walk_length: usize,
    /// Number of random walks per node
    pub num_walks: usize,
    /// Window size for skip-gram model
    pub window_size: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of negative samples
    pub negative_samples: usize,
}

impl Default for DeepWalkConfig {
    fn default() -> Self {
        DeepWalkConfig {
            dimensions: 128,
            walk_length: 40,
            num_walks: 80,
            window_size: 5,
            epochs: 1,
            learning_rate: 0.025,
            negative_samples: 5,
        }
    }
}

/// A random walk on a graph
#[derive(Debug, Clone)]
pub struct RandomWalk<N: Node> {
    /// The sequence of nodes in the walk
    pub nodes: Vec<N>,
}

/// Advanced optimization techniques for embeddings
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Learning rate schedule type
    pub lr_schedule: LearningRateSchedule,
    /// Initial learning rate
    pub initial_lr: f64,
    /// Final learning rate
    pub final_lr: f64,
    /// Use momentum optimization
    pub use_momentum: bool,
    /// Momentum factor (0.9 is typical)
    pub momentum: f64,
    /// Use Adam optimizer
    pub use_adam: bool,
    /// Adam beta1 parameter
    pub adam_beta1: f64,
    /// Adam beta2 parameter
    pub adam_beta2: f64,
    /// Adam epsilon parameter
    pub adam_epsilon: f64,
    /// L2 regularization strength
    pub l2_regularization: f64,
    /// Gradient clipping threshold
    pub gradient_clip: Option<f64>,
    /// Use hierarchical softmax instead of negative sampling
    pub use_hierarchical_softmax: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        OptimizationConfig {
            lr_schedule: LearningRateSchedule::Linear,
            initial_lr: 0.025,
            final_lr: 0.0001,
            use_momentum: false,
            momentum: 0.9,
            use_adam: false,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
            l2_regularization: 0.0,
            gradient_clip: Some(1.0),
            use_hierarchical_softmax: false,
        }
    }
}

/// Learning rate scheduling strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant,
    /// Linear decay from initial to final
    Linear,
    /// Exponential decay
    Exponential,
    /// Cosine annealing
    Cosine,
    /// Step decay (reduce by factor at specific epochs)
    Step,
}

/// Enhanced training metrics and monitoring
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Current epoch
    pub epoch: usize,
    /// Total training steps
    pub steps: usize,
    /// Current learning rate
    pub learning_rate: f64,
    /// Training loss (negative log likelihood)
    pub loss: f64,
    /// Loss moving average
    pub loss_avg: f64,
    /// Gradient norm
    pub gradient_norm: f64,
    /// Processing speed (steps per second)
    pub steps_per_second: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Convergence indicator (rate of loss change)
    pub convergence_rate: f64,
    /// Training accuracy on positive samples
    pub positive_accuracy: f64,
    /// Training accuracy on negative samples
    pub negative_accuracy: f64,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        TrainingMetrics {
            epoch: 0,
            steps: 0,
            learning_rate: 0.025,
            loss: 0.0,
            loss_avg: 0.0,
            gradient_norm: 0.0,
            steps_per_second: 0.0,
            memory_usage: 0,
            convergence_rate: 0.0,
            positive_accuracy: 0.0,
            negative_accuracy: 0.0,
        }
    }
}

/// Adaptive negative sampling strategies
#[derive(Debug, Clone)]
pub enum NegativeSamplingStrategy {
    /// Uniform random sampling
    Uniform,
    /// Frequency-based sampling (more frequent nodes sampled more often)
    Frequency,
    /// Degree-based sampling (higher degree nodes sampled more often)
    Degree,
    /// Adaptive sampling based on embedding quality
    Adaptive,
    /// Hierarchical sampling using word2vec-style tree
    Hierarchical,
}

/// Advanced optimizer state for Adam/momentum
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// Momentum buffers for each parameter
    pub momentum_buffers: HashMap<String, Vec<f64>>,
    /// Adam first moment estimates
    pub adam_m: HashMap<String, Vec<f64>>,
    /// Adam second moment estimates
    pub adam_v: HashMap<String, Vec<f64>>,
    /// Time step for bias correction
    pub time_step: usize,
}

impl Default for OptimizerState {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizerState {
    pub fn new() -> Self {
        OptimizerState {
            momentum_buffers: HashMap::new(),
            adam_m: HashMap::new(),
            adam_v: HashMap::new(),
            time_step: 0,
        }
    }
}

/// Skip-gram training context pair
#[derive(Debug, Clone)]
pub struct ContextPair<N: Node> {
    /// Target node
    pub target: N,
    /// Context node
    pub context: N,
}
