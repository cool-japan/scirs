//! LSTM-based Neural Optimizer
//!
//! This module implements a learned optimizer using LSTM networks to adaptively
//! update optimization parameters. The LSTM learns optimization strategies through
//! meta-learning, enabling automatic discovery of effective optimization patterns.

#![allow(dead_code)]

use ndarray::{s, Array, Array1, Array2, ArrayBase, Data, Dimension};
use num_traits::Float;
use rand::Rng;
use std::collections::{HashMap, VecDeque};

use super::{LearnedOptimizerConfig, MetaOptimizationStrategy};
use crate::error::{OptimError, Result};

/// LSTM-based neural optimizer with meta-learning capabilities
#[derive(Debug, Clone)]
pub struct LSTMOptimizer<T: Float> {
    /// Configuration for the LSTM optimizer
    config: LearnedOptimizerConfig,

    /// LSTM network architecture
    lstm_network: LSTMNetwork<T>,

    /// Gradient and parameter history for context
    history_buffer: HistoryBuffer<T>,

    /// Meta-learning components
    meta_learner: MetaLearner<T>,

    /// Adaptive learning rate controller
    lr_controller: AdaptiveLearningRateController<T>,

    /// Optimization state tracker
    state_tracker: OptimizationStateTracker<T>,

    /// Performance metrics
    metrics: LSTMOptimizerMetrics,

    /// Current optimization step
    step_count: usize,

    /// Random number generator for noise and initialization
    rng: scirs2_core::random::Random,
}

/// LSTM network architecture for optimization
#[derive(Debug, Clone)]
pub struct LSTMNetwork<T: Float> {
    /// LSTM layers
    layers: Vec<LSTMLayer<T>>,

    /// Output projection layer
    output_projection: OutputProjection<T>,

    /// Attention mechanism (optional)
    attention: Option<AttentionMechanism<T>>,

    /// Normalization layers
    layer_norms: Vec<LayerNormalization<T>>,

    /// Dropout for regularization
    dropout_rate: f64,
}

/// Individual LSTM layer
#[derive(Debug, Clone)]
pub struct LSTMLayer<T: Float> {
    /// Input-to-hidden weights (for i, f, g, o gates)
    weight_ih: Array2<T>,

    /// Hidden-to-hidden weights (for i, f, g, o gates)
    weight_hh: Array2<T>,

    /// Input biases
    bias_ih: Array1<T>,

    /// Hidden biases
    bias_hh: Array1<T>,

    /// Hidden state
    hidden_state: Array1<T>,

    /// Cell state
    cell_state: Array1<T>,

    /// Hidden size
    hiddensize: usize,
}

/// Output projection for generating parameter updates
#[derive(Debug, Clone)]
pub struct OutputProjection<T: Float> {
    /// Projection weights
    weights: Array2<T>,

    /// Projection biases
    bias: Array1<T>,

    /// Output transformation
    output_transform: OutputTransform,
}

impl<T: Float + Default + Clone> OutputProjection<T> {
    /// Create a new output projection
    pub fn new(
        input_size: usize,
        output_size: usize,
        output_transform: OutputTransform,
    ) -> Result<Self> {
        let weights = Array2::zeros((output_size, input_size));
        let bias = Array1::zeros(output_size);

        Ok(Self {
            weights,
            bias,
            output_transform,
        })
    }

    /// Forward pass through output projection
    pub fn forward(&self, input: &Array1<T>) -> Result<Array1<T>> {
        // Simplified implementation - just return the input for now
        Ok(input.clone())
    }
}

/// Output transformation types
#[derive(Debug, Clone, Copy)]
pub enum OutputTransform {
    /// Direct output (no transformation)
    Identity,

    /// Tanh activation
    Tanh,

    /// Scaled tanh for bounded updates
    ScaledTanh { scale: f64 },

    /// Adaptive scaling based on gradient norms
    AdaptiveScale,

    /// Learned nonlinear transformation
    LearnedNonlinear,
}

/// Attention mechanism for focusing on relevant history
#[derive(Debug, Clone)]
pub struct AttentionMechanism<T: Float> {
    /// Query projection
    query_proj: Array2<T>,

    /// Key projection
    key_proj: Array2<T>,

    /// Value projection
    value_proj: Array2<T>,

    /// Output projection
    output_proj: Array2<T>,

    /// Number of attention heads
    num_heads: usize,

    /// Attention head size
    head_size: usize,

    /// Attention weights from last forward pass
    attentionweights: Option<Array2<T>>,
}

impl<T: Float + Default + Clone> AttentionMechanism<T> {
    /// Create a new attention mechanism
    pub fn new(config: &LearnedOptimizerConfig) -> Result<Self> {
        let hiddensize = config.hidden_size;
        let num_heads = config.attention_heads;
        let head_size = hiddensize / num_heads;

        Ok(Self {
            query_proj: Array2::zeros((hiddensize, hiddensize)),
            key_proj: Array2::zeros((hiddensize, hiddensize)),
            value_proj: Array2::zeros((hiddensize, hiddensize)),
            output_proj: Array2::zeros((hiddensize, hiddensize)),
            num_heads,
            head_size,
            attentionweights: None,
        })
    }

    /// Forward pass through attention mechanism
    pub fn forward(&mut self, input: &Array1<T>) -> Result<Array1<T>> {
        // Simplified implementation - just return the input for now
        Ok(input.clone())
    }
}

/// Layer normalization for stable training
#[derive(Debug, Clone)]
pub struct LayerNormalization<T: Float> {
    /// Scale parameters
    gamma: Array1<T>,

    /// Shift parameters
    beta: Array1<T>,

    /// Epsilon for numerical stability
    epsilon: T,
}

impl<T: Float + Default + Clone> LayerNormalization<T> {
    /// Create a new layer normalization
    pub fn new(features: usize) -> Result<Self> {
        Ok(Self {
            gamma: Array1::ones(features),
            beta: Array1::zeros(features),
            epsilon: T::from(1e-5).unwrap(),
        })
    }

    /// Forward pass through layer normalization
    pub fn forward(&self, input: &Array1<T>) -> Result<Array1<T>> {
        // Simplified implementation - just return the input for now
        Ok(input.clone())
    }
}

/// History buffer for maintaining context
#[derive(Debug, Clone)]
pub struct HistoryBuffer<T: Float> {
    /// Gradient history
    gradients: VecDeque<Array1<T>>,

    /// Parameter history
    parameters: VecDeque<Array1<T>>,

    /// Loss history
    losses: VecDeque<T>,

    /// Learning rate history
    learning_rates: VecDeque<T>,

    /// Update magnitude history
    update_magnitudes: VecDeque<T>,

    /// Maximum history length
    _maxlength: usize,

    /// Preprocessed features cache
    feature_cache: Option<Array2<T>>,
}

/// Meta-learning component for optimizer adaptation
#[derive(Debug, Clone)]
pub struct MetaLearner<T: Float> {
    /// Meta-optimization strategy
    strategy: MetaOptimizationStrategy,

    /// Meta-parameters (optimizer parameters)
    meta_parameters: HashMap<String, Array1<T>>,

    /// Meta-gradients accumulator
    meta_gradients: HashMap<String, Array1<T>>,

    /// Task history for meta-learning
    task_history: VecDeque<MetaTask<T>>,

    /// Meta-learning state
    meta_state: MetaLearningState<T>,

    /// Transfer learning capabilities
    transfer_learner: TransferLearner<T>,
}

/// Meta-learning task
#[derive(Debug, Clone)]
pub struct MetaTask<T: Float> {
    /// Task identifier
    pub id: String,

    /// Task type
    pub task_type: TaskType,

    /// Training trajectory
    pub training_trajectory: Vec<TrajectoryPoint<T>>,

    /// Final performance
    pub final_performance: T,

    /// Task characteristics
    pub characteristics: TaskCharacteristics<T>,

    /// Task weight for meta-learning
    pub weight: T,
}

/// Types of optimization tasks
#[derive(Debug, Clone, Copy)]
pub enum TaskType {
    /// Standard supervised learning
    SupervisedLearning,

    /// Reinforcement learning
    ReinforcementLearning,

    /// Unsupervised learning
    UnsupervisedLearning,

    /// Few-shot learning
    FewShotLearning,

    /// Online learning
    OnlineLearning,

    /// Adversarial training
    AdversarialTraining,
}

/// Point in optimization trajectory
#[derive(Debug, Clone)]
pub struct TrajectoryPoint<T: Float> {
    /// Step number
    pub step: usize,

    /// Gradient at this step
    pub gradient: Array1<T>,

    /// Parameters at this step
    pub parameters: Array1<T>,

    /// Loss at this step
    pub loss: T,

    /// Learning rate used
    pub learning_rate: T,

    /// Update direction
    pub update: Array1<T>,
}

/// Task characteristics for meta-learning
#[derive(Debug, Clone)]
pub struct TaskCharacteristics<T: Float> {
    /// Problem dimensionality
    pub dimensionality: usize,

    /// Loss landscape curvature estimate
    pub curvature: T,

    /// Noise level estimate
    pub noise_level: T,

    /// Conditioning number estimate
    pub conditioning: T,

    /// Convergence difficulty
    pub difficulty: T,

    /// Task domain features
    pub domain_features: Array1<T>,
}

/// Meta-learning state
#[derive(Debug, Clone)]
pub struct MetaLearningState<T: Float> {
    /// Current meta-learning step
    pub meta_step: usize,

    /// Meta-learning rate
    pub meta_lr: T,

    /// Adaptation rate
    pub adaptation_rate: T,

    /// Meta-validation performance
    pub meta_validation_performance: T,

    /// Task adaptation history
    pub adaptation_history: VecDeque<AdaptationEvent<T>>,

    /// Inner loop state
    pub inner_loop_state: InnerLoopState<T>,
}

/// Adaptation event tracking
#[derive(Debug, Clone)]
pub struct AdaptationEvent<T: Float> {
    /// Source task
    pub source_task: String,

    /// Target task
    pub target_task: String,

    /// Adaptation steps required
    pub adaptation_steps: usize,

    /// Transfer efficiency
    pub transfer_efficiency: T,

    /// Final performance improvement
    pub performance_improvement: T,
}

/// Inner loop optimization state
#[derive(Debug, Clone)]
pub struct InnerLoopState<T: Float> {
    /// Current inner step
    pub inner_step: usize,

    /// Inner loop parameters
    pub inner_parameters: Array1<T>,

    /// Inner loop optimizer state
    pub inner_optimizer_state: HashMap<String, Array1<T>>,

    /// Inner loop performance
    pub inner_performance: T,
}

/// Transfer learning component
#[derive(Debug, Clone)]
pub struct TransferLearner<T: Float> {
    /// Source domain knowledge
    pub source_knowledge: HashMap<String, Array1<T>>,

    /// Domain adaptation parameters
    pub adaptation_parameters: Array1<T>,

    /// Transfer efficiency metrics
    pub transfer_metrics: TransferMetrics<T>,

    /// Domain similarity estimator
    pub similarity_estimator: DomainSimilarityEstimator<T>,
}

/// Transfer learning metrics
#[derive(Debug, Clone)]
pub struct TransferMetrics<T: Float> {
    /// Transfer efficiency
    pub efficiency: T,

    /// Adaptation speed
    pub adaptation_speed: T,

    /// Knowledge retention
    pub knowledge_retention: T,

    /// Negative transfer detection
    pub negative_transfer_score: T,
}

/// Domain similarity estimator
#[derive(Debug, Clone)]
pub struct DomainSimilarityEstimator<T: Float> {
    /// Domain embeddings
    pub domain_embeddings: HashMap<String, Array1<T>>,

    /// Similarity metric parameters
    pub similarity_params: Array1<T>,

    /// Learned similarity function
    pub similarity_function: SimilarityFunction,
}

/// Similarity function types
#[derive(Debug, Clone, Copy)]
pub enum SimilarityFunction {
    /// Cosine similarity
    Cosine,

    /// Euclidean distance
    Euclidean,

    /// Learned metric
    LearnedMetric,

    /// Task-specific similarity
    TaskSpecific,
}

/// Adaptive learning rate controller
#[derive(Debug, Clone)]
pub struct AdaptiveLearningRateController<T: Float> {
    /// Base learning rate
    base_lr: T,

    /// Current learning rate
    current_lr: T,

    /// Learning rate adaptation parameters
    adaptation_params: LRAdaptationParams<T>,

    /// Learning rate history
    lr_history: VecDeque<T>,

    /// Performance-based adaptation
    performance_tracker: PerformanceTracker<T>,

    /// Learned LR schedule parameters
    schedule_params: Option<Array1<T>>,
}

/// Learning rate adaptation parameters
#[derive(Debug, Clone)]
pub struct LRAdaptationParams<T: Float> {
    /// Momentum for LR adaptation
    pub momentum: T,

    /// Sensitivity to gradient changes
    pub gradient_sensitivity: T,

    /// Sensitivity to loss changes
    pub loss_sensitivity: T,

    /// Minimum learning rate
    pub min_lr: T,

    /// Maximum learning rate
    pub max_lr: T,

    /// Adaptation rate
    pub adaptation_rate: T,
}

/// Performance tracker for adaptive learning rate
#[derive(Debug, Clone)]
pub struct PerformanceTracker<T: Float> {
    /// Recent loss values
    recent_losses: VecDeque<T>,

    /// Performance trend
    trend: PerformanceTrend,

    /// Stagnation detection
    stagnation_counter: usize,

    /// Best performance seen
    best_performance: T,

    /// Performance improvement rate
    improvement_rate: T,
}

/// Performance trend indicators
#[derive(Debug, Clone, Copy)]
pub enum PerformanceTrend {
    /// Performance is improving
    Improving,

    /// Performance is stagnating
    Stagnating,

    /// Performance is degrading
    Degrading,

    /// Performance is oscillating
    Oscillating,

    /// Insufficient data
    Unknown,
}

/// Optimization state tracker
#[derive(Debug, Clone)]
pub struct OptimizationStateTracker<T: Float> {
    /// Current optimization phase
    phase: OptimizationPhase,

    /// Convergence indicators
    convergence_indicators: ConvergenceIndicators<T>,

    /// Gradient analysis
    gradient_analyzer: GradientAnalyzer<T>,

    /// Loss landscape analysis
    landscape_analyzer: LossLandscapeAnalyzer<T>,

    /// Stability metrics
    stability_metrics: StabilityMetrics<T>,
}

/// Optimization phases
#[derive(Debug, Clone, Copy)]
pub enum OptimizationPhase {
    /// Initial rapid descent
    InitialDescent,

    /// Steady progress
    SteadyProgress,

    /// Fine-tuning
    FineTuning,

    /// Converged
    Converged,

    /// Stuck/Plateau
    Plateau,

    /// Diverging
    Diverging,
}

/// Convergence indicators
#[derive(Debug, Clone)]
pub struct ConvergenceIndicators<T: Float> {
    /// Gradient norm trend
    pub gradient_norm_trend: Vec<T>,

    /// Loss change trend
    pub loss_change_trend: Vec<T>,

    /// Parameter change magnitude
    pub parameter_change_magnitude: T,

    /// Convergence probability
    pub convergence_probability: T,

    /// Estimated steps to convergence
    pub estimated_steps_to_convergence: Option<usize>,
}

/// Gradient analysis component
#[derive(Debug, Clone)]
pub struct GradientAnalyzer<T: Float> {
    /// Gradient statistics
    pub gradient_stats: GradientStatistics<T>,

    /// Gradient correlation tracking
    pub correlation_tracker: GradientCorrelationTracker<T>,

    /// Gradient noise estimation
    pub noise_estimator: GradientNoiseEstimator<T>,

    /// Gradient flow analysis
    pub flow_analyzer: GradientFlowAnalyzer<T>,
}

/// Gradient statistics
#[derive(Debug, Clone)]
pub struct GradientStatistics<T: Float> {
    /// Mean gradient norm
    pub mean_norm: T,

    /// Gradient norm variance
    pub norm_variance: T,

    /// Gradient direction consistency
    pub direction_consistency: T,

    /// Gradient magnitude distribution
    pub magnitude_distribution: Vec<T>,

    /// Component-wise statistics
    pub component_stats: Array1<T>,
}

/// Gradient correlation tracker
#[derive(Debug, Clone)]
pub struct GradientCorrelationTracker<T: Float> {
    /// Correlation matrix
    pub correlation_matrix: Array2<T>,

    /// Temporal correlations
    pub temporal_correlations: VecDeque<T>,

    /// Cross-parameter correlations
    pub cross_correlations: HashMap<String, T>,
}

/// Gradient noise estimator
#[derive(Debug, Clone)]
pub struct GradientNoiseEstimator<T: Float> {
    /// Estimated noise level
    pub noise_level: T,

    /// Signal-to-noise ratio
    pub signal_to_noise_ratio: T,

    /// Noise characteristics
    pub noise_characteristics: NoiseCharacteristics<T>,
}

/// Noise characteristics
#[derive(Debug, Clone)]
pub struct NoiseCharacteristics<T: Float> {
    /// Noise type
    pub noise_type: NoiseType,

    /// Noise scale
    pub scale: T,

    /// Temporal correlation
    pub temporal_correlation: T,

    /// Spatial correlation
    pub spatial_correlation: T,
}

/// Types of gradient noise
#[derive(Debug, Clone, Copy)]
pub enum NoiseType {
    /// White noise (uncorrelated)
    White,

    /// Colored noise (correlated)
    Colored,

    /// Structured noise
    Structured,

    /// Adaptive noise
    Adaptive,
}

/// Gradient flow analyzer
#[derive(Debug, Clone)]
pub struct GradientFlowAnalyzer<T: Float> {
    /// Flow field estimation
    pub flow_field: Array2<T>,

    /// Critical points
    pub critical_points: Vec<Array1<T>>,

    /// Flow stability
    pub stability: FlowStability,

    /// Attractors and repellers
    pub attractors: Vec<Array1<T>>,
    pub repellers: Vec<Array1<T>>,
}

/// Flow stability indicators
#[derive(Debug, Clone, Copy)]
pub enum FlowStability {
    /// Stable flow
    Stable,

    /// Unstable flow
    Unstable,

    /// Chaotic flow
    Chaotic,

    /// Unknown stability
    Unknown,
}

/// Loss landscape analyzer
#[derive(Debug, Clone)]
pub struct LossLandscapeAnalyzer<T: Float> {
    /// Local curvature estimation
    pub local_curvature: T,

    /// Hessian eigenvalue estimates
    pub hessian_eigenvalues: Option<Array1<T>>,

    /// Landscape roughness
    pub roughness: T,

    /// Basin of attraction size
    pub basin_size: T,

    /// Barrier heights
    pub barrier_heights: Vec<T>,
}

/// Stability metrics
#[derive(Debug, Clone)]
pub struct StabilityMetrics<T: Float> {
    /// Lyapunov exponents
    pub lyapunov_exponents: Array1<T>,

    /// Stability margin
    pub stability_margin: T,

    /// Perturbation sensitivity
    pub perturbation_sensitivity: T,

    /// Robustness score
    pub robustness_score: T,
}

/// Performance metrics for LSTM optimizer
#[derive(Debug, Clone)]
pub struct LSTMOptimizerMetrics {
    /// Meta-learning performance
    pub meta_learning_loss: f64,

    /// Average convergence speed
    pub avg_convergence_speed: f64,

    /// Generalization performance
    pub generalization_performance: f64,

    /// Adaptation efficiency
    pub adaptation_efficiency: f64,

    /// Transfer learning success rate
    pub transfer_success_rate: f64,

    /// Memory usage
    pub memory_usage_mb: f64,

    /// Computational overhead
    pub computational_overhead: f64,

    /// LSTM network statistics
    pub lstm_stats: LSTMNetworkStats,

    /// Attention statistics (if using attention)
    pub attention_stats: Option<AttentionStats>,
}

/// LSTM network statistics
#[derive(Debug, Clone)]
pub struct LSTMNetworkStats {
    /// Gate activation statistics
    pub gate_activations: GateActivationStats,

    /// Hidden state statistics
    pub hidden_state_stats: StateStatistics,

    /// Cell state statistics
    pub cell_state_stats: StateStatistics,

    /// Gradient flow statistics
    pub gradient_flow_stats: GradientFlowStats,
}

/// Gate activation statistics
#[derive(Debug, Clone)]
pub struct GateActivationStats {
    /// Input gate activations
    pub input_gate: StateStatistics,

    /// Forget gate activations
    pub forget_gate: StateStatistics,

    /// Output gate activations
    pub output_gate: StateStatistics,

    /// Cell gate activations
    pub cell_gate: StateStatistics,
}

/// State statistics
#[derive(Debug, Clone)]
pub struct StateStatistics {
    /// Mean activation
    pub mean: f64,

    /// Standard deviation
    pub std: f64,

    /// Minimum value
    pub min: f64,

    /// Maximum value
    pub max: f64,

    /// Saturation percentage
    pub saturation_percent: f64,
}

/// Gradient flow statistics
#[derive(Debug, Clone)]
pub struct GradientFlowStats {
    /// Gradient norm through layers
    pub layer_gradient_norms: Vec<f64>,

    /// Gradient correlation between layers
    pub layer_correlations: Vec<f64>,

    /// Vanishing gradient indicator
    pub vanishing_gradient_score: f64,

    /// Exploding gradient indicator
    pub exploding_gradient_score: f64,
}

/// Attention mechanism statistics
#[derive(Debug, Clone)]
pub struct AttentionStats {
    /// Attention entropy
    pub attention_entropy: f64,

    /// Attention concentration
    pub attention_concentration: f64,

    /// Head diversity
    pub head_diversity: f64,

    /// Temporal attention patterns
    pub temporal_patterns: Vec<f64>,
}

impl<
        T: Float
            + Default
            + Clone
            + Send
            + Sync
            + std::iter::Sum
            + for<'a> std::iter::Sum<&'a T>
            + ndarray::ScalarOperand
            + std::fmt::Debug,
    > LSTMOptimizer<T>
{
    /// Create a new LSTM optimizer
    pub fn new(config: LearnedOptimizerConfig) -> Result<Self> {
        // Validate configuration
        Self::validate_config(&config)?;

        // Initialize LSTM network
        let lstm_network = LSTMNetwork::new(&config)?;

        // Initialize history buffer
        let history_buffer = HistoryBuffer::new(config.gradient_history_size);

        // Initialize meta-learner
        let meta_learner = MetaLearner::new(&config)?;

        // Initialize learning rate controller
        let lr_controller = AdaptiveLearningRateController::new(&config)?;

        // Initialize state tracker
        let state_tracker = OptimizationStateTracker::new();

        // Initialize metrics
        let metrics = LSTMOptimizerMetrics::new();

        // Initialize RNG
        let rng = scirs2_core::random::rng();

        Ok(Self {
            config,
            lstm_network,
            history_buffer,
            meta_learner,
            lr_controller,
            state_tracker,
            metrics,
            step_count: 0,
            rng,
        })
    }

    /// Perform LSTM-based optimization step
    pub fn lstm_step<S, D>(
        &mut self,
        parameters: &ArrayBase<S, D>,
        gradients: &ArrayBase<S, D>,
        loss: Option<T>,
    ) -> Result<Array<T, D>>
    where
        S: Data<Elem = T>,
        D: Dimension + Clone,
    {
        // Convert to flat arrays for processing
        let flat_params = self.flatten_to_1d(parameters)?;
        let flat_gradients = self.flatten_to_1d(gradients)?;

        // Update history buffer
        self.history_buffer
            .update(&flat_params, &flat_gradients, loss);

        // Prepare LSTM input features
        let lstm_input = self.prepare_lstm_input(&flat_gradients)?;

        // Forward pass through LSTM
        let lstm_output = self.lstm_network.forward(&lstm_input)?;

        // Compute adaptive learning rate
        let learning_rate =
            self.lr_controller
                .compute_lr(&flat_gradients, loss, &self.history_buffer)?;

        // Generate parameter updates
        let updates = self.generate_updates(&lstm_output, &flat_gradients, learning_rate)?;

        // Apply updates to parameters
        let updated_flat = &flat_params - &updates;

        // Update state tracking
        self.state_tracker.update(&flat_gradients, &updates, loss);

        // Update metrics
        self.update_metrics(&flat_gradients, &updates, learning_rate);

        // Reshape back to original dimensions
        let updated_params = self.reshape_from_1d(&updated_flat, parameters.raw_dim())?;

        self.step_count += 1;

        Ok(updated_params)
    }

    /// Meta-learning step for optimizer adaptation
    pub fn meta_learning_step(&mut self, tasks: &[MetaTask<T>]) -> Result<T> {
        // Perform meta-learning update
        let meta_loss = self.meta_learner.step(tasks, &mut self.lstm_network)?;

        // Update meta-learning metrics
        self.metrics.meta_learning_loss = meta_loss.to_f64().unwrap_or(0.0);

        Ok(meta_loss)
    }

    /// Transfer learning to new optimization domain
    pub fn transfer_to_domain(
        &mut self,
        target_tasks: &[MetaTask<T>],
    ) -> Result<TransferResults<T>> {
        self.meta_learner
            .transfer_learner
            .transfer_to_domain(target_tasks, &mut self.lstm_network)
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> &LSTMOptimizerMetrics {
        &self.metrics
    }

    /// Get optimization state analysis
    pub fn get_state_analysis(&self) -> OptimizationStateAnalysis<T> {
        OptimizationStateAnalysis {
            current_phase: self.state_tracker.phase,
            convergence_indicators: self.state_tracker.convergence_indicators.clone(),
            gradient_analysis: self.state_tracker.gradient_analyzer.clone(),
            landscape_analysis: self.state_tracker.landscape_analyzer.clone(),
            stability_metrics: self.state_tracker.stability_metrics.clone(),
        }
    }

    /// Prepare input features for LSTM
    fn prepare_lstm_input(&self, gradients: &Array1<T>) -> Result<Array1<T>> {
        let mut features = Vec::new();

        // Current gradient features
        features.extend_from_slice(gradients.as_slice().unwrap());

        // Historical gradient features
        if let Some(prev_gradients) = self.history_buffer.get_recent_gradients(5) {
            for prev_grad in prev_gradients {
                // Gradient differences
                let grad_diff: Vec<T> = gradients
                    .iter()
                    .zip(prev_grad.iter())
                    .map(|(&g1, &g2)| g1 - g2)
                    .collect();
                features.extend(grad_diff);
            }
        }

        // Statistical features
        let grad_norm = gradients.iter().map(|&g| g * g).sum::<T>().sqrt();
        let grad_mean = gradients.iter().cloned().sum::<T>() / T::from(gradients.len()).unwrap();
        let grad_std = {
            let variance = gradients
                .iter()
                .map(|&g| (g - grad_mean) * (g - grad_mean))
                .sum::<T>()
                / T::from(gradients.len()).unwrap();
            variance.sqrt()
        };

        features.extend([grad_norm, grad_mean, grad_std]);

        // Loss-based features
        if let Some(loss_features) = self.history_buffer.get_loss_features() {
            features.extend(loss_features);
        }

        // Pad or truncate to expected input size
        features.resize(self.config.input_features, T::zero());

        Ok(Array1::from_vec(features))
    }

    /// Generate parameter updates from LSTM output
    fn generate_updates(
        &self,
        lstm_output: &Array1<T>,
        gradients: &Array1<T>,
        learning_rate: T,
    ) -> Result<Array1<T>> {
        // Apply _output transformation
        let transformed_output = match self.lstm_network.output_projection.output_transform {
            OutputTransform::Identity => lstm_output.clone(),
            OutputTransform::Tanh => lstm_output.mapv(|x| x.tanh()),
            OutputTransform::ScaledTanh { scale } => {
                let scale_t = T::from(scale).unwrap();
                lstm_output.mapv(|x| x.tanh() * scale_t)
            }
            OutputTransform::AdaptiveScale => {
                let grad_norm = gradients.iter().map(|&g| g * g).sum::<T>().sqrt();
                let adaptive_scale = T::one() / (T::one() + grad_norm);
                lstm_output.mapv(|x| x * adaptive_scale)
            }
            OutputTransform::LearnedNonlinear => {
                // Apply learned nonlinear transformation
                lstm_output.mapv(|x| {
                    let exp_x = x.exp();
                    (exp_x - (-x).exp()) / (exp_x + (-x).exp()) // tanh via exp
                })
            }
        };

        // Combine with gradient information
        let updates = &transformed_output * learning_rate;

        Ok(updates)
    }

    /// Update performance metrics
    fn update_metrics(&mut self, gradients: &Array1<T>, updates: &Array1<T>, lr: T) {
        // Compute gradient statistics
        let grad_norm = gradients.iter().map(|&g| g * g).sum::<T>().sqrt();
        let update_norm = updates.iter().map(|&u| u * u).sum::<T>().sqrt();

        // Update LSTM statistics
        self.update_lstm_stats();

        // Update efficiency metrics
        self.metrics.adaptation_efficiency = (update_norm / grad_norm).to_f64().unwrap_or(1.0);

        // Update computational overhead
        self.metrics.computational_overhead = self.estimate_computational_overhead();

        // Update memory usage
        self.metrics.memory_usage_mb = self.estimate_memory_usage();
    }

    /// Update LSTM network statistics
    fn update_lstm_stats(&mut self) {
        // Update gate activation statistics
        for (_i, layer) in self.lstm_network.layers.iter().enumerate() {
            let hidden_stats = self.compute_state_stats(&layer.hidden_state);
            let cell_stats = self.compute_state_stats(&layer.cell_state);

            // Update statistics (simplified)
            self.metrics.lstm_stats.hidden_state_stats = hidden_stats;
            self.metrics.lstm_stats.cell_state_stats = cell_stats;
        }

        // Update attention statistics if available
        if let Some(ref attention) = self.lstm_network.attention {
            if let Some(ref attentionweights) = attention.attentionweights {
                self.metrics.attention_stats = Some(self.compute_attention_stats(attentionweights));
            }
        }
    }

    /// Compute state statistics
    fn compute_state_stats(&self, state: &Array1<T>) -> StateStatistics {
        let values: Vec<f64> = state.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect();

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let saturation_count = values.iter().filter(|&&x| x.abs() > 0.95).count();
        let saturation_percent = saturation_count as f64 / values.len() as f64 * 100.0;

        StateStatistics {
            mean,
            std,
            min,
            max,
            saturation_percent,
        }
    }

    /// Compute attention statistics
    fn compute_attention_stats(&self, attentionweights: &Array2<T>) -> AttentionStats {
        let weights: Vec<f64> = attentionweights
            .iter()
            .map(|&w| w.to_f64().unwrap_or(0.0))
            .collect();

        // Compute entropy
        let entropy = weights
            .iter()
            .filter(|&&w| w > 0.0)
            .map(|&w| -w * w.ln())
            .sum::<f64>();

        // Compute concentration (inverse of entropy)
        let concentration = 1.0 / (1.0 + entropy);

        // Simplified diversity measure
        let head_diversity = weights.iter().map(|&w| w.abs()).sum::<f64>() / weights.len() as f64;

        AttentionStats {
            attention_entropy: entropy,
            attention_concentration: concentration,
            head_diversity,
            temporal_patterns: vec![0.0; 10], // Placeholder
        }
    }

    /// Estimate computational overhead
    fn estimate_computational_overhead(&self) -> f64 {
        // Simplified overhead estimation
        let lstm_overhead = self.config.num_layers as f64 * 0.1;
        let attention_overhead = if self.config.use_attention { 0.2 } else { 0.0 };
        let meta_learning_overhead = 0.1;

        1.0 + lstm_overhead + attention_overhead + meta_learning_overhead
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self) -> f64 {
        // Simplified memory estimation in MB
        let parameter_memory =
            self.config.hidden_size as f64 * self.config.num_layers as f64 * 8.0 / 1024.0 / 1024.0;
        let history_memory =
            self.config.gradient_history_size as f64 * self.config.input_features as f64 * 8.0
                / 1024.0
                / 1024.0;
        let lstm_state_memory =
            self.config.hidden_size as f64 * self.config.num_layers as f64 * 2.0 * 8.0
                / 1024.0
                / 1024.0;

        parameter_memory + history_memory + lstm_state_memory
    }

    /// Validate configuration
    fn validate_config(config: &LearnedOptimizerConfig) -> Result<()> {
        if config.hidden_size == 0 {
            return Err(OptimError::InvalidConfig(
                "Hidden size must be positive".to_string(),
            ));
        }

        if config.num_layers == 0 {
            return Err(OptimError::InvalidConfig(
                "Number of layers must be positive".to_string(),
            ));
        }

        if config.input_features == 0 {
            return Err(OptimError::InvalidConfig(
                "Input features must be positive".to_string(),
            ));
        }

        if config.meta_learning_rate <= 0.0 {
            return Err(OptimError::InvalidConfig(
                "Meta learning rate must be positive".to_string(),
            ));
        }

        Ok(())
    }

    /// Utility functions for array manipulation
    fn flatten_to_1d<S, D>(&self, array: &ArrayBase<S, D>) -> Result<Array1<T>>
    where
        S: Data<Elem = T>,
        D: Dimension,
    {
        Ok(Array1::from_iter(array.iter().cloned()))
    }

    fn reshape_from_1d<D>(&self, flat: &Array1<T>, shape: D) -> Result<Array<T, D>>
    where
        D: Dimension + Clone,
    {
        Array::from_shape_vec(shape, flat.to_vec())
            .map_err(|e| OptimError::InvalidConfig(format!("Reshape error: {}", e)))
    }
}

// Implementation of major components

impl<T: Float + Default + Clone + 'static> LSTMNetwork<T> {
    /// Create new LSTM network
    fn new(config: &LearnedOptimizerConfig) -> Result<Self> {
        let mut layers = Vec::new();

        // Create LSTM layers
        for i in 0..config.num_layers {
            let input_size = if i == 0 {
                config.input_features
            } else {
                config.hidden_size
            };
            let layer = LSTMLayer::new(input_size, config.hidden_size)?;
            layers.push(layer);
        }

        // Create output projection
        let output_projection = OutputProjection::new(
            config.hidden_size,
            config.output_features,
            OutputTransform::ScaledTanh { scale: 0.1 },
        )?;

        // Create attention mechanism if enabled
        let attention = if config.use_attention {
            Some(AttentionMechanism::new(config)?)
        } else {
            None
        };

        // Create layer normalization
        let layer_norms = (0..config.num_layers)
            .map(|_| LayerNormalization::new(config.hidden_size))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            layers,
            output_projection,
            attention,
            layer_norms,
            dropout_rate: config.dropout_rate,
        })
    }

    /// Forward pass through LSTM network
    fn forward(&mut self, input: &Array1<T>) -> Result<Array1<T>> {
        let mut current_input = input.clone();

        // Forward through LSTM layers
        for i in 0..self.layers.len() {
            current_input = self.layers[i].forward(&current_input)?;

            // Apply layer normalization
            current_input = self.layer_norms[i].forward(&current_input)?;

            // Apply dropout during training
            if self.dropout_rate > 0.0 {
                current_input = self.apply_dropout(&current_input)?;
            }
        }

        // Apply attention if enabled
        if let Some(ref mut attention) = self.attention {
            current_input = attention.forward(&current_input)?;
        }

        // Final output projection
        let output = self.output_projection.forward(&current_input)?;

        Ok(output)
    }

    /// Apply dropout for regularization
    fn apply_dropout(&self, input: &Array1<T>) -> Result<Array1<T>> {
        // Simplified dropout implementation
        Ok(input.mapv(|x| {
            if T::from(scirs2_core::random::rng().gen_range(0.0..1.0)).unwrap()
                < T::from(self.dropout_rate).unwrap()
            {
                T::zero()
            } else {
                x / T::from(1.0 - self.dropout_rate).unwrap()
            }
        }))
    }
}

impl<T: Float + Default + Clone + 'static> LSTMLayer<T> {
    /// Create new LSTM layer
    fn new(_input_size: usize, hiddensize: usize) -> Result<Self> {
        // Xavier initialization
        let scale = (2.0 / (_input_size + hiddensize) as f64).sqrt();

        Ok(Self {
            weight_ih: Self::xavier_init(4 * hiddensize, _input_size, scale),
            weight_hh: Self::xavier_init(4 * hiddensize, hiddensize, scale),
            bias_ih: Array1::zeros(4 * hiddensize),
            bias_hh: Array1::zeros(4 * hiddensize),
            hidden_state: Array1::zeros(hiddensize),
            cell_state: Array1::zeros(hiddensize),
            hiddensize,
        })
    }

    /// Forward pass through LSTM layer
    fn forward(&mut self, input: &Array1<T>) -> Result<Array1<T>> {
        // LSTM computation: i, f, g, o = σ(W_ih @ x + W_hh @ h + b)
        let ih_linear = self.weight_ih.dot(input) + &self.bias_ih;
        let hh_linear = self.weight_hh.dot(&self.hidden_state) + &self.bias_hh;
        let gates = ih_linear + hh_linear;

        // Split into gates
        let input_gate = Self::sigmoid(&gates.slice(s![0..self.hiddensize]).to_owned());
        let forget_gate = Self::sigmoid(
            &gates
                .slice(s![self.hiddensize..2 * self.hiddensize])
                .to_owned(),
        );
        let cell_gate = Self::tanh(
            &gates
                .slice(s![2 * self.hiddensize..3 * self.hiddensize])
                .to_owned(),
        );
        let output_gate = Self::sigmoid(
            &gates
                .slice(s![3 * self.hiddensize..4 * self.hiddensize])
                .to_owned(),
        );

        // Update cell state
        self.cell_state = &forget_gate * &self.cell_state + &input_gate * &cell_gate;

        // Update hidden state
        self.hidden_state = &output_gate * &Self::tanh(&self.cell_state);

        Ok(self.hidden_state.clone())
    }

    /// Xavier initialization
    fn xavier_init(rows: usize, cols: usize, scale: f64) -> Array2<T> {
        Array2::from_shape_fn((rows, cols), |_| {
            let val = (scirs2_core::random::rng().gen_range(0.0..1.0) - 0.5) * 2.0 * scale;
            T::from(val).unwrap()
        })
    }

    /// Sigmoid activation
    fn sigmoid(x: &Array1<T>) -> Array1<T> {
        x.mapv(|xi| T::one() / (T::one() + (-xi).exp()))
    }

    /// Tanh activation
    fn tanh(x: &Array1<T>) -> Array1<T> {
        x.mapv(|xi| xi.tanh())
    }
}

impl<T: Float + Default + Clone> HistoryBuffer<T> {
    /// Create new history buffer
    fn new(_maxlength: usize) -> Self {
        Self {
            gradients: VecDeque::with_capacity(_maxlength),
            parameters: VecDeque::with_capacity(_maxlength),
            losses: VecDeque::with_capacity(_maxlength),
            learning_rates: VecDeque::with_capacity(_maxlength),
            update_magnitudes: VecDeque::with_capacity(_maxlength),
            _maxlength,
            feature_cache: None,
        }
    }

    /// Update history with new data
    fn update(&mut self, params: &Array1<T>, grads: &Array1<T>, loss: Option<T>) {
        // Add new entries
        self.parameters.push_back(params.clone());
        self.gradients.push_back(grads.clone());

        if let Some(l) = loss {
            self.losses.push_back(l);
        }

        // Maintain size limits
        while self.parameters.len() > self._maxlength {
            self.parameters.pop_front();
        }
        while self.gradients.len() > self._maxlength {
            self.gradients.pop_front();
        }
        while self.losses.len() > self._maxlength {
            self.losses.pop_front();
        }

        // Invalidate cache
        self.feature_cache = None;
    }

    /// Get recent gradients
    fn get_recent_gradients(&self, count: usize) -> Option<Vec<&Array1<T>>> {
        if self.gradients.len() < count {
            return None;
        }

        Some(self.gradients.iter().rev().take(count).collect())
    }

    /// Get loss-based features
    fn get_loss_features(&self) -> Option<Vec<T>> {
        if self.losses.len() < 2 {
            return None;
        }

        let current_loss = *self.losses.back().unwrap();
        let prev_loss = self.losses[self.losses.len() - 2];

        let loss_change = current_loss - prev_loss;
        let loss_ratio = if prev_loss.abs() > T::from(1e-8).unwrap() {
            current_loss / prev_loss
        } else {
            T::one()
        };

        Some(vec![loss_change, loss_ratio])
    }
}

/// Additional implementations for other components...
/// Results from optimization state analysis
#[derive(Debug, Clone)]
pub struct OptimizationStateAnalysis<T: Float> {
    pub current_phase: OptimizationPhase,
    pub convergence_indicators: ConvergenceIndicators<T>,
    pub gradient_analysis: GradientAnalyzer<T>,
    pub landscape_analysis: LossLandscapeAnalyzer<T>,
    pub stability_metrics: StabilityMetrics<T>,
}

/// Transfer learning results
#[derive(Debug, Clone)]
pub struct TransferResults<T: Float> {
    pub initial_performance: T,
    pub final_performance: T,
    pub adaptation_steps: usize,
    pub transfer_efficiency: T,
}

// Additional default implementations and stubs for remaining components...

impl Default for LSTMOptimizerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl LSTMOptimizerMetrics {
    fn new() -> Self {
        Self {
            meta_learning_loss: 0.0,
            avg_convergence_speed: 0.0,
            generalization_performance: 0.0,
            adaptation_efficiency: 0.0,
            transfer_success_rate: 0.0,
            memory_usage_mb: 0.0,
            computational_overhead: 1.0,
            lstm_stats: LSTMNetworkStats {
                gate_activations: GateActivationStats {
                    input_gate: StateStatistics::default(),
                    forget_gate: StateStatistics::default(),
                    output_gate: StateStatistics::default(),
                    cell_gate: StateStatistics::default(),
                },
                hidden_state_stats: StateStatistics::default(),
                cell_state_stats: StateStatistics::default(),
                gradient_flow_stats: GradientFlowStats {
                    layer_gradient_norms: Vec::new(),
                    layer_correlations: Vec::new(),
                    vanishing_gradient_score: 0.0,
                    exploding_gradient_score: 0.0,
                },
            },
            attention_stats: None,
        }
    }
}

impl Default for StateStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            saturation_percent: 0.0,
        }
    }
}

// Placeholder implementations for remaining complex components
// These would be fully implemented in a production system

impl<T: Float + Default + Clone> MetaLearner<T> {
    fn new(config: &LearnedOptimizerConfig) -> Result<Self> {
        // Placeholder implementation
        Ok(Self {
            strategy: MetaOptimizationStrategy::MAML,
            meta_parameters: HashMap::new(),
            meta_gradients: HashMap::new(),
            task_history: VecDeque::new(),
            meta_state: MetaLearningState {
                meta_step: 0,
                meta_lr: T::from(0.001).unwrap(),
                adaptation_rate: T::from(0.1).unwrap(),
                meta_validation_performance: T::zero(),
                adaptation_history: VecDeque::new(),
                inner_loop_state: InnerLoopState {
                    inner_step: 0,
                    inner_parameters: Array1::zeros(1),
                    inner_optimizer_state: HashMap::new(),
                    inner_performance: T::zero(),
                },
            },
            transfer_learner: TransferLearner {
                source_knowledge: HashMap::new(),
                adaptation_parameters: Array1::zeros(1),
                transfer_metrics: TransferMetrics {
                    efficiency: T::zero(),
                    adaptation_speed: T::zero(),
                    knowledge_retention: T::zero(),
                    negative_transfer_score: T::zero(),
                },
                similarity_estimator: DomainSimilarityEstimator {
                    domain_embeddings: HashMap::new(),
                    similarity_params: Array1::zeros(1),
                    similarity_function: SimilarityFunction::Cosine,
                },
            },
        })
    }

    fn step(&mut self, tasks: &[MetaTask<T>], network: &mut LSTMNetwork<T>) -> Result<T> {
        // Placeholder meta-learning step
        Ok(T::zero())
    }
}

impl<T: Float + Default + Clone> TransferLearner<T> {
    fn transfer_to_domain(
        &mut self,
        _target_tasks: &[MetaTask<T>],
        _network: &mut LSTMNetwork<T>,
    ) -> Result<TransferResults<T>> {
        // Placeholder transfer learning
        Ok(TransferResults {
            initial_performance: T::zero(),
            final_performance: T::zero(),
            adaptation_steps: 0,
            transfer_efficiency: T::zero(),
        })
    }
}

impl<T: Float + Default + Clone> AdaptiveLearningRateController<T> {
    fn new(config: &LearnedOptimizerConfig) -> Result<Self> {
        // Placeholder implementation
        Ok(Self {
            base_lr: T::from(0.001).unwrap(),
            current_lr: T::from(0.001).unwrap(),
            adaptation_params: LRAdaptationParams {
                momentum: T::from(0.9).unwrap(),
                gradient_sensitivity: T::from(0.1).unwrap(),
                loss_sensitivity: T::from(0.1).unwrap(),
                min_lr: T::from(1e-6).unwrap(),
                max_lr: T::from(0.1).unwrap(),
                adaptation_rate: T::from(0.01).unwrap(),
            },
            lr_history: VecDeque::new(),
            performance_tracker: PerformanceTracker {
                recent_losses: VecDeque::new(),
                trend: PerformanceTrend::Unknown,
                stagnation_counter: 0,
                best_performance: T::zero(),
                improvement_rate: T::zero(),
            },
            schedule_params: None,
        })
    }

    fn compute_lr(
        &mut self,
        gradients: &Array1<T>,
        _loss: Option<T>,
        _history: &HistoryBuffer<T>,
    ) -> Result<T> {
        // Placeholder adaptive LR computation
        Ok(self.current_lr)
    }
}

impl<T: Float + Default + Clone> OptimizationStateTracker<T> {
    fn new() -> Self {
        Self {
            phase: OptimizationPhase::InitialDescent,
            convergence_indicators: ConvergenceIndicators {
                gradient_norm_trend: Vec::new(),
                loss_change_trend: Vec::new(),
                parameter_change_magnitude: T::zero(),
                convergence_probability: T::zero(),
                estimated_steps_to_convergence: None,
            },
            gradient_analyzer: GradientAnalyzer {
                gradient_stats: GradientStatistics {
                    mean_norm: T::zero(),
                    norm_variance: T::zero(),
                    direction_consistency: T::zero(),
                    magnitude_distribution: Vec::new(),
                    component_stats: Array1::zeros(1),
                },
                correlation_tracker: GradientCorrelationTracker {
                    correlation_matrix: Array2::zeros((1, 1)),
                    temporal_correlations: VecDeque::new(),
                    cross_correlations: HashMap::new(),
                },
                noise_estimator: GradientNoiseEstimator {
                    noise_level: T::zero(),
                    signal_to_noise_ratio: T::zero(),
                    noise_characteristics: NoiseCharacteristics {
                        noise_type: NoiseType::White,
                        scale: T::zero(),
                        temporal_correlation: T::zero(),
                        spatial_correlation: T::zero(),
                    },
                },
                flow_analyzer: GradientFlowAnalyzer {
                    flow_field: Array2::zeros((1, 1)),
                    critical_points: Vec::new(),
                    stability: FlowStability::Unknown,
                    attractors: Vec::new(),
                    repellers: Vec::new(),
                },
            },
            landscape_analyzer: LossLandscapeAnalyzer {
                local_curvature: T::zero(),
                hessian_eigenvalues: None,
                roughness: T::zero(),
                basin_size: T::zero(),
                barrier_heights: Vec::new(),
            },
            stability_metrics: StabilityMetrics {
                lyapunov_exponents: Array1::zeros(1),
                stability_margin: T::zero(),
                perturbation_sensitivity: T::zero(),
                robustness_score: T::zero(),
            },
        }
    }

    fn update(&mut self, gradients: &Array1<T>, _updates: &Array1<T>, loss: Option<T>) {
        // Placeholder state update
    }
}

// Additional implementations would continue for all remaining components...

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_optimizer_creation() {
        let config = LearnedOptimizerConfig::default();
        let optimizer = LSTMOptimizer::<f64>::new(config);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_lstm_layer_creation() {
        let layer = LSTMLayer::<f64>::new(10, 20);
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        assert_eq!(layer.hiddensize, 20);
        assert_eq!(layer.weight_ih.shape(), &[80, 10]); // 4 * hiddensize, input_size
        assert_eq!(layer.weight_hh.shape(), &[80, 20]); // 4 * hiddensize, hiddensize
    }

    #[test]
    fn test_history_buffer() {
        let mut buffer = HistoryBuffer::<f64>::new(5);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let grads = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        buffer.update(&params, &grads, Some(0.5));

        assert_eq!(buffer.gradients.len(), 1);
        assert_eq!(buffer.parameters.len(), 1);
        assert_eq!(buffer.losses.len(), 1);
    }

    #[test]
    fn test_config_validation() {
        let mut config = LearnedOptimizerConfig::default();
        assert!(LSTMOptimizer::<f64>::validate_config(&config).is_ok());

        config.hidden_size = 0;
        assert!(LSTMOptimizer::<f64>::validate_config(&config).is_err());
    }

    #[test]
    fn test_lstm_network_creation() {
        let config = LearnedOptimizerConfig::default();
        let network = LSTMNetwork::<f64>::new(&config);
        assert!(network.is_ok());

        let network = network.unwrap();
        assert_eq!(network.layers.len(), config.num_layers);
        assert!(network.attention.is_some()); // attention enabled by default
    }

    #[test]
    fn test_metrics_initialization() {
        let metrics = LSTMOptimizerMetrics::new();
        assert_eq!(metrics.meta_learning_loss, 0.0);
        assert_eq!(metrics.computational_overhead, 1.0);
        assert!(metrics.attention_stats.is_none());
    }
}
