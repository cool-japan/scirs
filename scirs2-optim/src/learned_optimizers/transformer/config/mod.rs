//! Configuration structures for transformer-based optimization
//!
//! This module provides comprehensive configuration options for:
//! - Transformer architecture settings
//! - Attention mechanism configurations
//! - Meta-learning parameters
//! - Performance optimization settings

use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for transformer-based optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerBasedOptimizerConfig<T: Float> {
    /// Transformer architecture configuration
    pub architecture: TransformerArchConfig,
    /// Attention mechanism configuration
    pub attention: AttentionConfig<T>,
    /// Meta-learning configuration
    pub meta_learning: MetaLearningConfig<T>,
    /// Sequence processing configuration
    pub sequence: SequenceConfig,
    /// Memory management configuration
    pub memory: MemoryConfig,
    /// Performance tracking configuration
    pub performance: PerformanceConfig,
    /// Training configuration
    pub training: TrainingConfig<T>,
}

impl<T: Float> Default for TransformerBasedOptimizerConfig<T> {
    fn default() -> Self {
        Self {
            architecture: TransformerArchConfig::default(),
            attention: AttentionConfig::default(),
            meta_learning: MetaLearningConfig::default(),
            sequence: SequenceConfig::default(),
            memory: MemoryConfig::default(),
            performance: PerformanceConfig::default(),
            training: TrainingConfig::default(),
        }
    }
}

/// Transformer architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerArchConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Feed-forward dimension
    pub ff_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Use pre-layer normalization
    pub pre_norm: bool,
    /// Activation function type
    pub activation: ActivationType,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Use learnable positional encoding
    pub learnable_pe: bool,
    /// Embedding dimension
    pub embedding_dim: usize,
}

impl Default for TransformerArchConfig {
    fn default() -> Self {
        Self {
            num_layers: 6,
            hidden_dim: 512,
            ff_dim: 2048,
            num_heads: 8,
            dropout_rate: 0.1,
            pre_norm: true,
            activation: ActivationType::Gelu,
            max_seq_length: 1024,
            learnable_pe: false,
            embedding_dim: 512,
        }
    }
}

/// Activation function types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActivationType {
    /// ReLU activation
    Relu,
    /// GELU activation
    Gelu,
    /// Swish activation
    Swish,
    /// Mish activation
    Mish,
    /// Tanh activation
    Tanh,
    /// Sigmoid activation
    Sigmoid,
    /// LeakyReLU activation
    LeakyRelu { negative_slope: f64 },
    /// ELU activation
    Elu { alpha: f64 },
}

/// Attention mechanism configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig<T: Float> {
    /// Attention head dimension
    pub head_dim: usize,
    /// Use scaled dot-product attention
    pub scaled: bool,
    /// Attention dropout rate
    pub attention_dropout: f64,
    /// Use relative positional encoding
    pub relative_pe: bool,
    /// Maximum relative position
    pub max_relative_position: usize,
    /// Attention bias type
    pub bias_type: AttentionBiasType,
    /// Temperature for attention softmax
    pub temperature: T,
    /// Use sparse attention
    pub sparse_attention: bool,
    /// Sparsity pattern configuration
    pub sparsity_config: Option<SparsityConfig>,
}

impl<T: Float> Default for AttentionConfig<T> {
    fn default() -> Self {
        Self {
            head_dim: 64,
            scaled: true,
            attention_dropout: 0.1,
            relative_pe: false,
            max_relative_position: 32,
            bias_type: AttentionBiasType::None,
            temperature: T::one(),
            sparse_attention: false,
            sparsity_config: None,
        }
    }
}

/// Attention bias types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionBiasType {
    /// No attention bias
    None,
    /// Causal (lower triangular) mask
    Causal,
    /// Custom bias matrix
    Custom,
    /// Relative position bias
    Relative,
    /// Learnable bias
    Learnable,
}

/// Sparsity configuration for sparse attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsityConfig {
    /// Sparsity pattern type
    pub pattern: SparsityPattern,
    /// Block size for block-sparse attention
    pub block_size: usize,
    /// Stride for strided attention
    pub stride: usize,
    /// Number of local attention positions
    pub local_attention: usize,
    /// Number of global attention positions
    pub global_attention: usize,
}

/// Sparsity patterns for attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SparsityPattern {
    /// Block-sparse attention
    BlockSparse,
    /// Strided attention
    Strided,
    /// Local + global attention
    LocalGlobal,
    /// Random sparse attention
    Random { sparsity_ratio: f64 },
    /// Custom sparsity pattern
    Custom,
}

/// Meta-learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningConfig<T: Float> {
    /// Meta-learning strategy
    pub strategy: MetaLearningStrategy,
    /// Learning rate for meta-learning
    pub meta_learning_rate: T,
    /// Number of inner optimization steps
    pub inner_steps: usize,
    /// Meta-batch size
    pub meta_batch_size: usize,
    /// Gradient clipping threshold
    pub gradient_clip: Option<T>,
    /// Use second-order gradients
    pub second_order: bool,
    /// Adaptation configuration
    pub adaptation: AdaptationConfig<T>,
    /// Task sampling configuration
    pub task_sampling: TaskSamplingConfig,
}

impl<T: Float> Default for MetaLearningConfig<T> {
    fn default() -> Self {
        Self {
            strategy: MetaLearningStrategy::Maml,
            meta_learning_rate: num_traits::cast::cast(0.001).unwrap_or_else(|| T::zero()),
            inner_steps: 5,
            meta_batch_size: 16,
            gradient_clip: Some(num_traits::cast::cast(1.0).unwrap_or_else(|| T::zero())),
            second_order: false,
            adaptation: AdaptationConfig::default(),
            task_sampling: TaskSamplingConfig::default(),
        }
    }
}

/// Meta-learning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaLearningStrategy {
    /// Model-Agnostic Meta-Learning
    Maml,
    /// Reptile algorithm
    Reptile,
    /// Prototypical networks
    Prototypical,
    /// Matching networks
    Matching,
    /// Meta-SGD
    MetaSgd,
    /// Learned optimizer
    LearnedOptimizer,
    /// Gradient-based meta-learning
    GradientBased,
    /// Memory-augmented meta-learning
    MemoryAugmented,
}

/// Adaptation configuration for meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationConfig<T: Float> {
    /// Number of adaptation steps
    pub num_steps: usize,
    /// Adaptation learning rate
    pub learning_rate: T,
    /// Use adaptive learning rate
    pub adaptive_lr: bool,
    /// Learning rate schedule
    pub lr_schedule: LearningRateSchedule<T>,
    /// Stop early if converged
    pub early_stopping: bool,
    /// Convergence tolerance
    pub tolerance: T,
}

impl<T: Float> Default for AdaptationConfig<T> {
    fn default() -> Self {
        Self {
            num_steps: 10,
            learning_rate: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            adaptive_lr: false,
            lr_schedule: LearningRateSchedule::Constant,
            early_stopping: false,
            tolerance: num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero()),
        }
    }
}

/// Learning rate schedules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateSchedule<T: Float> {
    /// Constant learning rate
    Constant,
    /// Exponential decay
    Exponential { decay_rate: T },
    /// Cosine annealing
    Cosine { min_lr: T },
    /// Step decay
    Step { 
        step_size: usize, 
        gamma: T 
    },
    /// Polynomial decay
    Polynomial { power: T },
    /// Warm restart
    WarmRestart { 
        t_0: usize, 
        t_mult: usize 
    },
}

/// Task sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSamplingConfig {
    /// Sampling strategy
    pub strategy: TaskSamplingStrategy,
    /// Number of support examples per task
    pub num_support: usize,
    /// Number of query examples per task
    pub num_query: usize,
    /// Task difficulty curriculum
    pub curriculum: Option<CurriculumConfig>,
    /// Task augmentation
    pub augmentation: TaskAugmentationConfig,
}

impl Default for TaskSamplingConfig {
    fn default() -> Self {
        Self {
            strategy: TaskSamplingStrategy::Uniform,
            num_support: 10,
            num_query: 15,
            curriculum: None,
            augmentation: TaskAugmentationConfig::default(),
        }
    }
}

/// Task sampling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskSamplingStrategy {
    /// Uniform random sampling
    Uniform,
    /// Weighted sampling based on difficulty
    Weighted,
    /// Curriculum learning
    Curriculum,
    /// Anti-curriculum learning
    AntiCurriculum,
    /// Adaptive sampling
    Adaptive,
}

/// Curriculum learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumConfig {
    /// Starting difficulty level
    pub start_difficulty: f64,
    /// Ending difficulty level
    pub end_difficulty: f64,
    /// Curriculum schedule
    pub schedule: CurriculumSchedule,
    /// Difficulty metric
    pub difficulty_metric: DifficultyMetric,
}

/// Curriculum schedules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CurriculumSchedule {
    /// Linear progression
    Linear,
    /// Exponential progression
    Exponential { rate: f64 },
    /// Step-wise progression
    Step { step_size: usize },
    /// Performance-based progression
    PerformanceBased { threshold: f64 },
}

/// Difficulty metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifficultyMetric {
    /// Loss-based difficulty
    Loss,
    /// Gradient norm
    GradientNorm,
    /// Convergence rate
    ConvergenceRate,
    /// Number of optimization steps
    OptimizationSteps,
    /// Custom metric
    Custom(String),
}

/// Task augmentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAugmentationConfig {
    /// Enable noise injection
    pub noise_injection: bool,
    /// Noise level
    pub noise_level: f64,
    /// Enable parameter perturbation
    pub parameter_perturbation: bool,
    /// Perturbation magnitude
    pub perturbation_magnitude: f64,
    /// Enable gradient augmentation
    pub gradient_augmentation: bool,
    /// Augmentation methods
    pub augmentation_methods: Vec<AugmentationMethod>,
}

impl Default for TaskAugmentationConfig {
    fn default() -> Self {
        Self {
            noise_injection: false,
            noise_level: 0.01,
            parameter_perturbation: false,
            perturbation_magnitude: 0.1,
            gradient_augmentation: false,
            augmentation_methods: vec![],
        }
    }
}

/// Augmentation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AugmentationMethod {
    /// Gaussian noise
    GaussianNoise { std: f64 },
    /// Dropout
    Dropout { rate: f64 },
    /// Gradient clipping
    GradientClipping { threshold: f64 },
    /// Parameter scaling
    ParameterScaling { factor: f64 },
    /// Custom augmentation
    Custom(String),
}

/// Sequence processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceConfig {
    /// Maximum sequence length
    pub max_length: usize,
    /// Sequence padding strategy
    pub padding_strategy: PaddingStrategy,
    /// Sequence truncation strategy
    pub truncation_strategy: TruncationStrategy,
    /// Use dynamic batching
    pub dynamic_batching: bool,
    /// Batch size
    pub batch_size: usize,
    /// Sequence augmentation
    pub augmentation: SequenceAugmentationConfig,
}

impl Default for SequenceConfig {
    fn default() -> Self {
        Self {
            max_length: 1024,
            padding_strategy: PaddingStrategy::Zero,
            truncation_strategy: TruncationStrategy::Right,
            dynamic_batching: true,
            batch_size: 32,
            augmentation: SequenceAugmentationConfig::default(),
        }
    }
}

/// Padding strategies for sequences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaddingStrategy {
    /// Zero padding
    Zero,
    /// Mean padding
    Mean,
    /// Constant padding
    Constant(f64),
    /// Reflection padding
    Reflection,
    /// Replication padding
    Replication,
}

/// Truncation strategies for sequences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TruncationStrategy {
    /// Truncate from the right
    Right,
    /// Truncate from the left
    Left,
    /// Truncate from the center
    Center,
    /// Random truncation
    Random,
    /// No truncation (error if too long)
    None,
}

/// Sequence augmentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceAugmentationConfig {
    /// Enable temporal jittering
    pub temporal_jittering: bool,
    /// Jittering strength
    pub jitter_strength: f64,
    /// Enable sequence dropout
    pub sequence_dropout: bool,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Enable sequence permutation
    pub permutation: bool,
    /// Permutation probability
    pub permutation_prob: f64,
}

impl Default for SequenceAugmentationConfig {
    fn default() -> Self {
        Self {
            temporal_jittering: false,
            jitter_strength: 0.1,
            sequence_dropout: false,
            dropout_rate: 0.1,
            permutation: false,
            permutation_prob: 0.1,
        }
    }
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Memory type
    pub memory_type: MemoryType,
    /// Memory size (number of entries)
    pub memory_size: usize,
    /// Memory update strategy
    pub update_strategy: MemoryUpdateStrategy,
    /// Memory retrieval strategy
    pub retrieval_strategy: MemoryRetrievalStrategy,
    /// Enable memory compression
    pub compression: bool,
    /// Compression algorithm
    pub compression_algorithm: CompressionAlgorithm,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            memory_type: MemoryType::Fifo,
            memory_size: 10000,
            update_strategy: MemoryUpdateStrategy::Replace,
            retrieval_strategy: MemoryRetrievalStrategy::MostRecent,
            compression: false,
            compression_algorithm: CompressionAlgorithm::None,
        }
    }
}

/// Memory types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    /// First-in-first-out memory
    Fifo,
    /// Least-recently-used memory
    Lru,
    /// Priority-based memory
    Priority,
    /// Associative memory
    Associative,
    /// Episodic memory
    Episodic,
    /// Working memory
    Working,
}

/// Memory update strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryUpdateStrategy {
    /// Replace existing entries
    Replace,
    /// Average with existing entries
    Average,
    /// Weighted update
    Weighted { weight: f64 },
    /// Append new entries
    Append,
    /// Selective update based on importance
    Selective,
}

/// Memory retrieval strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryRetrievalStrategy {
    /// Most recent entries
    MostRecent,
    /// Most relevant entries
    MostRelevant,
    /// Random sampling
    Random,
    /// Similarity-based retrieval
    Similarity,
    /// Attention-based retrieval
    Attention,
}

/// Compression algorithms for memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Principal Component Analysis
    Pca { components: usize },
    /// Singular Value Decomposition
    Svd { rank: usize },
    /// Autoencoder compression
    Autoencoder { latent_dim: usize },
    /// Quantization
    Quantization { bits: usize },
}

/// Performance tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable performance tracking
    pub enabled: bool,
    /// Tracking interval
    pub tracking_interval: Duration,
    /// Metrics to track
    pub metrics: Vec<PerformanceMetric>,
    /// Enable profiling
    pub profiling: bool,
    /// Memory usage tracking
    pub memory_tracking: bool,
    /// Computation time tracking
    pub time_tracking: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tracking_interval: Duration::from_secs(1),
            metrics: vec![
                PerformanceMetric::Loss,
                PerformanceMetric::Accuracy,
                PerformanceMetric::ConvergenceRate,
            ],
            profiling: false,
            memory_tracking: true,
            time_tracking: true,
        }
    }
}

/// Performance metrics to track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetric {
    /// Loss value
    Loss,
    /// Accuracy
    Accuracy,
    /// Convergence rate
    ConvergenceRate,
    /// Gradient norm
    GradientNorm,
    /// Learning rate
    LearningRate,
    /// Memory usage
    MemoryUsage,
    /// Computation time
    ComputationTime,
    /// Attention weights
    AttentionWeights,
    /// Custom metric
    Custom(String),
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig<T: Float> {
    /// Learning rate
    pub learning_rate: T,
    /// Optimizer type
    pub optimizer: OptimizerType,
    /// Weight decay
    pub weight_decay: T,
    /// Gradient clipping
    pub gradient_clipping: Option<GradientClippingConfig<T>>,
    /// Learning rate schedule
    pub lr_schedule: LearningRateSchedule<T>,
    /// Warmup configuration
    pub warmup: Option<WarmupConfig>,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig<T>>,
}

impl<T: Float> Default for TrainingConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate: num_traits::cast::cast(0.001).unwrap_or_else(|| T::zero()),
            optimizer: OptimizerType::Adam,
            weight_decay: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            gradient_clipping: Some(GradientClippingConfig::default()),
            lr_schedule: LearningRateSchedule::Constant,
            warmup: None,
            early_stopping: None,
        }
    }
}

/// Optimizer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Adam optimizer
    Adam,
    /// AdamW optimizer
    AdamW,
    /// SGD optimizer
    Sgd,
    /// RMSprop optimizer
    RmsProp,
    /// Adagrad optimizer
    Adagrad,
    /// Custom optimizer
    Custom(String),
}

/// Gradient clipping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientClippingConfig<T: Float> {
    /// Clipping method
    pub method: ClippingMethod<T>,
    /// Clipping threshold
    pub threshold: T,
    /// Adaptive clipping
    pub adaptive: bool,
}

impl<T: Float> Default for GradientClippingConfig<T> {
    fn default() -> Self {
        Self {
            method: ClippingMethod::Norm,
            threshold: num_traits::cast::cast(1.0).unwrap_or_else(|| T::zero()),
            adaptive: false,
        }
    }
}

/// Gradient clipping methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClippingMethod<T: Float> {
    /// Clip by norm
    Norm,
    /// Clip by value
    Value,
    /// Clip by global norm
    GlobalNorm,
    /// Adaptive clipping
    Adaptive { percentile: T },
}

/// Warmup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupConfig {
    /// Number of warmup steps
    pub steps: usize,
    /// Warmup type
    pub warmup_type: WarmupType,
}

/// Warmup types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarmupType {
    /// Linear warmup
    Linear,
    /// Cosine warmup
    Cosine,
    /// Exponential warmup
    Exponential,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig<T: Float> {
    /// Patience (number of epochs to wait)
    pub patience: usize,
    /// Minimum delta for improvement
    pub min_delta: T,
    /// Metric to monitor
    pub monitor: String,
    /// Mode (minimize or maximize)
    pub mode: MonitorMode,
}

/// Monitoring modes for early stopping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitorMode {
    /// Minimize the metric
    Min,
    /// Maximize the metric
    Max,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TransformerBasedOptimizerConfig::<f64>::default();
        assert_eq!(config.architecture.num_layers, 6);
        assert_eq!(config.architecture.hidden_dim, 512);
        assert_eq!(config.attention.head_dim, 64);
    }

    #[test]
    fn test_activation_types() {
        let relu = ActivationType::Relu;
        let gelu = ActivationType::Gelu;
        let leaky_relu = ActivationType::LeakyRelu { negative_slope: 0.01 };
        
        assert_eq!(relu, ActivationType::Relu);
        assert_eq!(gelu, ActivationType::Gelu);
        
        if let ActivationType::LeakyRelu { negative_slope } = leaky_relu {
            assert_eq!(negative_slope, 0.01);
        }
    }

    #[test]
    fn test_meta_learning_config() {
        let config = MetaLearningConfig::<f32>::default();
        assert_eq!(config.inner_steps, 5);
        assert_eq!(config.meta_batch_size, 16);
        assert!(matches!(config.strategy, MetaLearningStrategy::Maml));
    }

    #[test]
    fn test_memory_config() {
        let config = MemoryConfig::default();
        assert_eq!(config.memory_size, 10000);
        assert!(matches!(config.memory_type, MemoryType::Fifo));
        assert!(!config.compression);
    }

    #[test]
    fn test_sequence_config() {
        let config = SequenceConfig::default();
        assert_eq!(config.max_length, 1024);
        assert_eq!(config.batch_size, 32);
        assert!(config.dynamic_batching);
    }

    #[test]
    fn test_performance_config() {
        let config = PerformanceConfig::default();
        assert!(config.enabled);
        assert!(config.memory_tracking);
        assert!(config.time_tracking);
        assert_eq!(config.metrics.len(), 3);
    }
}