//! Adaptive Transformer Enhancement for Optimization
//!
//! This module implements transformer-based neural architectures that adaptively
//! enhance optimization algorithms. The transformers learn to attend to different
//! aspects of the optimization landscape and adapt their strategies accordingly.

use super::{
    ActivationType, LearnedOptimizationConfig, LearnedOptimizer, MetaOptimizerState,
    OptimizationProblem, TrainingTask,
};
use crate::error::{OptimizeError, OptimizeResult};
use crate::result::OptimizeResults;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::Rng;
use statrs::statistics::Statistics;
use std::collections::{HashMap, VecDeque};

/// Adaptive Transformer-Enhanced Optimizer
#[derive(Debug, Clone)]
pub struct AdaptiveTransformerOptimizer {
    /// Configuration
    config: LearnedOptimizationConfig,
    /// Multi-head transformer
    transformer: OptimizationTransformer,
    /// Problem encoder
    problem_encoder: TransformerProblemEncoder,
    /// Optimization history buffer
    history_buffer: OptimizationHistory,
    /// Meta-optimizer state
    meta_state: MetaOptimizerState,
    /// Adaptive components
    adaptive_components: AdaptiveComponents,
    /// Performance metrics
    performance_metrics: TransformerMetrics,
}

/// Transformer architecture for optimization
#[derive(Debug, Clone)]
pub struct OptimizationTransformer {
    /// Number of transformer layers
    num_layers: usize,
    /// Transformer blocks
    transformer_blocks: Vec<TransformerBlock>,
    /// Position encoding
    position_encoding: Array2<f64>,
    /// Input embedding layer
    input_embedding: Array2<f64>,
    /// Output projection layer
    output_projection: Array2<f64>,
    /// Model dimension
    model_dim: usize,
}

/// Single transformer block
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    /// Multi-head attention
    attention: MultiHeadAttention,
    /// Feed-forward network
    feed_forward: FeedForwardNetwork,
    /// Layer normalization 1
    layer_norm1: LayerNormalization,
    /// Layer normalization 2
    layer_norm2: LayerNormalization,
    /// Dropout rate
    dropout_rate: f64,
}

/// Multi-head attention mechanism
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Query weights
    w_query: Array2<f64>,
    /// Key weights
    w_key: Array2<f64>,
    /// Value weights
    w_value: Array2<f64>,
    /// Output projection
    w_output: Array2<f64>,
    /// Attention scores history (for analysis)
    attention_scores: Vec<Array2<f64>>,
}

/// Feed-forward network
#[derive(Debug, Clone)]
pub struct FeedForwardNetwork {
    /// First linear layer
    linear1: Array2<f64>,
    /// Second linear layer
    linear2: Array2<f64>,
    /// Bias terms
    bias1: Array1<f64>,
    /// Bias terms
    bias2: Array1<f64>,
    /// Activation function
    activation: ActivationType,
    /// Hidden dimension
    hidden_dim: usize,
}

/// Layer normalization
#[derive(Debug, Clone)]
pub struct LayerNormalization {
    /// Scale parameters
    gamma: Array1<f64>,
    /// Shift parameters
    beta: Array1<f64>,
    /// Epsilon for numerical stability
    epsilon: f64,
}

/// Problem encoder for transformers
#[derive(Debug, Clone)]
pub struct TransformerProblemEncoder {
    /// Gradient encoding layer
    gradient_encoder: Array2<f64>,
    /// Hessian encoding layer
    hessian_encoder: Array2<f64>,
    /// Parameter encoding layer
    parameter_encoder: Array2<f64>,
    /// Temporal encoding layer
    temporal_encoder: Array2<f64>,
    /// Context encoding layer
    context_encoder: Array2<f64>,
    /// Embedding dimension
    embedding_dim: usize,
}

/// Optimization history for transformer context
#[derive(Debug, Clone)]
pub struct OptimizationHistory {
    /// Parameter history
    parameter_history: VecDeque<Array1<f64>>,
    /// Objective value history
    objective_history: VecDeque<f64>,
    /// Gradient history
    gradient_history: VecDeque<Array1<f64>>,
    /// Step size history
    step_size_history: VecDeque<f64>,
    /// Success/failure history
    success_history: VecDeque<bool>,
    /// Maximum history length
    max_length: usize,
    /// Current step
    current_step: usize,
}

/// Adaptive components for transformer optimization
#[derive(Debug, Clone)]
pub struct AdaptiveComponents {
    /// Adaptive attention weights
    attention_adaptation: AttentionAdaptation,
    /// Dynamic learning rate scheduler
    learning_rate_adapter: LearningRateAdapter,
    /// Gradient scaling mechanism
    gradient_scaler: GradientScaler,
    /// Step size predictor
    step_size_predictor: StepSizePredictor,
    /// Convergence detector
    convergence_detector: ConvergenceDetector,
}

/// Attention adaptation mechanism
#[derive(Debug, Clone)]
pub struct AttentionAdaptation {
    /// Adaptation rate
    adaptation_rate: f64,
    /// Current attention focus
    attention_focus: Array1<f64>,
    /// Focus history
    focus_history: VecDeque<Array1<f64>>,
    /// Problem-specific attention patterns
    problem_patterns: HashMap<String, Array1<f64>>,
}

/// Learning rate adapter
#[derive(Debug, Clone)]
pub struct LearningRateAdapter {
    /// Base learning rate
    base_lr: f64,
    /// Current learning rate
    current_lr: f64,
    /// Adaptation parameters
    adaptation_params: Array1<f64>,
    /// Performance window
    performance_window: VecDeque<f64>,
    /// Adaptation history
    lr_history: Vec<f64>,
}

/// Gradient scaling mechanism
#[derive(Debug, Clone)]
pub struct GradientScaler {
    /// Scaling factors
    scale_factors: Array1<f64>,
    /// Gradient statistics
    gradient_stats: GradientStatistics,
    /// Adaptive scaling parameters
    scaling_params: Array1<f64>,
}

/// Gradient statistics
#[derive(Debug, Clone)]
pub struct GradientStatistics {
    /// Running mean
    mean: Array1<f64>,
    /// Running variance
    variance: Array1<f64>,
    /// Update count
    count: usize,
    /// Momentum parameter
    momentum: f64,
}

/// Step size predictor
#[derive(Debug, Clone)]
pub struct StepSizePredictor {
    /// Prediction network
    predictor_network: Array2<f64>,
    /// Input features
    feature_dim: usize,
    /// Prediction history
    prediction_history: Vec<f64>,
    /// Actual step sizes
    actual_steps: Vec<f64>,
}

/// Convergence detector
#[derive(Debug, Clone)]
pub struct ConvergenceDetector {
    /// Detection threshold
    threshold: f64,
    /// Window size for analysis
    window_size: usize,
    /// Recent improvements
    recent_improvements: VecDeque<f64>,
    /// Convergence probability
    convergence_prob: f64,
}

/// Performance metrics for transformer
#[derive(Debug, Clone)]
pub struct TransformerMetrics {
    /// Attention entropy
    attention_entropy: f64,
    /// Learning rate adaptation efficiency
    lr_adaptation_efficiency: f64,
    /// Gradient prediction accuracy
    gradient_prediction_accuracy: f64,
    /// Step size prediction accuracy
    step_size_prediction_accuracy: f64,
    /// Convergence detection accuracy
    convergence_detection_accuracy: f64,
}

impl AdaptiveTransformerOptimizer {
    /// Create new adaptive transformer optimizer
    pub fn new(config: LearnedOptimizationConfig) -> Self {
        let model_dim = config.hidden_size;
        let transformer = OptimizationTransformer::new(
            config.num_heads,
            model_dim,
            config.max_parameters,
            6, // num_layers
        );

        let problem_encoder = TransformerProblemEncoder::new(model_dim);
        let history_buffer = OptimizationHistory::new(100);

        Self {
            config,
            transformer,
            problem_encoder,
            history_buffer,
            meta_state: MetaOptimizerState {
                meta_params: Array1::zeros(model_dim),
                network_weights: Array2::zeros((model_dim, model_dim)),
                performance_history: Vec::new(),
                adaptation_stats: super::AdaptationStatistics::default(),
                episode: 0,
            },
            adaptive_components: AdaptiveComponents::new(model_dim),
            performance_metrics: TransformerMetrics::default(),
        }
    }

    /// Process optimization step with transformer
    pub fn process_optimization_step<F>(
        &mut self,
        objective: &F,
        current_params: &ArrayView1<f64>,
        problem: &OptimizationProblem,
    ) -> OptimizeResult<OptimizationStep>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        // Encode current state
        let state_encoding = self.encode_optimization_state(objective, current_params, problem)?;

        // Process through transformer
        let transformer_output = self.transformer.forward(&state_encoding.view())?;

        // Extract optimization decisions
        let optimization_step = self.decode_optimization_step(&transformer_output.view())?;

        // Update adaptive components
        self.update_adaptive_components(&optimization_step)?;

        // Record in history
        self.history_buffer.add_step(
            current_params.to_owned(),
            objective(current_params),
            optimization_step.clone(),
        );

        Ok(optimization_step)
    }

    /// Encode current optimization state
    fn encode_optimization_state<F>(
        &self,
        objective: &F,
        current_params: &ArrayView1<f64>,
        problem: &OptimizationProblem,
    ) -> OptimizeResult<Array2<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let seq_len = self.history_buffer.current_step.min(50) + 1; // Include current state
        let model_dim = self.transformer.model_dim;
        let mut sequence = Array2::zeros((seq_len, model_dim));

        // Encode historical states
        for i in 0..seq_len - 1 {
            if let Some(historical_encoding) = self.encode_historical_state(i) {
                for j in 0..model_dim.min(historical_encoding.len()) {
                    sequence[[i, j]] = historical_encoding[j];
                }
            }
        }

        // Encode current state
        let current_encoding =
            self.problem_encoder
                .encode_current_state(objective, current_params, problem)?;

        let last_idx = seq_len - 1;
        for j in 0..model_dim.min(current_encoding.len()) {
            sequence[[last_idx, j]] = current_encoding[j];
        }

        Ok(sequence)
    }

    /// Encode historical state
    fn encode_historical_state(&self, history_index: usize) -> Option<Array1<f64>> {
        if history_index >= self.history_buffer.parameter_history.len() {
            return None;
        }

        let params = &self.history_buffer.parameter_history[history_index];
        let obj_val = self.history_buffer.objective_history[history_index];

        // Create encoding from historical data
        let mut encoding = Array1::zeros(self.transformer.model_dim);

        // Parameter features
        for (i, &param) in params.iter().enumerate() {
            if i < encoding.len() / 4 {
                encoding[i] = param.tanh();
            }
        }

        // Objective value features
        let obj_idx = encoding.len() / 4;
        if obj_idx < encoding.len() {
            encoding[obj_idx] = obj_val.ln().abs().tanh();
        }

        // Gradient features (if available)
        if let Some(gradient) = self.history_buffer.gradient_history.get(history_index) {
            let grad_start = encoding.len() / 2;
            for (i, &grad) in gradient.iter().enumerate() {
                if grad_start + i < encoding.len() {
                    encoding[grad_start + i] = grad.tanh();
                }
            }
        }

        Some(encoding)
    }

    /// Decode optimization step from transformer output
    fn decode_optimization_step(
        &self,
        transformer_output: &ArrayView2<f64>,
    ) -> OptimizeResult<OptimizationStep> {
        if transformer_output.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "Empty transformer _output".to_string(),
            ));
        }

        // Extract last timestep _output
        let last_output = transformer_output.row(transformer_output.nrows() - 1);

        // Decode step size
        let step_size_raw = last_output.get(0).copied().unwrap_or(0.0);
        let step_size = (step_size_raw.tanh() + 1.0) * 0.01; // Map to [0, 0.02]

        // Decode direction (simplified)
        let direction_dim = self.meta_state.meta_params.len().min(last_output.len() - 1);
        let mut direction = Array1::zeros(direction_dim);
        for i in 0..direction_dim {
            direction[i] = last_output.get(i + 1).copied().unwrap_or(0.0).tanh();
        }

        // Decode learning rate adaptation
        let lr_factor_raw = last_output
            .get(last_output.len() / 2)
            .copied()
            .unwrap_or(0.0);
        let lr_adaptation_factor = (lr_factor_raw.tanh() + 1.0) * 0.5 + 0.5; // Map to [0.5, 1.5]

        // Decode convergence confidence
        let conv_raw = last_output
            .get(last_output.len() - 1)
            .copied()
            .unwrap_or(0.0);
        let convergence_confidence = (conv_raw.tanh() + 1.0) * 0.5; // Map to [0, 1]

        Ok(OptimizationStep {
            step_size,
            direction,
            lr_adaptation_factor,
            convergence_confidence,
            attention_weights: self.get_attention_weights(),
        })
    }

    /// Get current attention weights for analysis
    fn get_attention_weights(&self) -> Array2<f64> {
        if let Some(first_block) = self.transformer.transformer_blocks.first() {
            if let Some(last_attention) = first_block.attention.attention_scores.last() {
                return last_attention.clone();
            }
        }
        Array2::zeros((1, 1))
    }

    /// Update adaptive components
    fn update_adaptive_components(&mut self, step: &OptimizationStep) -> OptimizeResult<()> {
        // Update attention adaptation
        self.adaptive_components
            .attention_adaptation
            .update(&step.attention_weights)?;

        // Update learning rate adapter
        self.adaptive_components
            .learning_rate_adapter
            .update(step.lr_adaptation_factor)?;

        // Update convergence detector
        self.adaptive_components
            .convergence_detector
            .update(step.convergence_confidence)?;

        Ok(())
    }

    /// Adapt transformer to specific problem characteristics
    pub fn adapt_to_problem_class(&mut self, problem_class: &str) -> OptimizeResult<()> {
        // Adjust attention patterns based on problem type
        match problem_class {
            "quadratic" => {
                // Focus more on recent gradients
                self.adaptive_components
                    .attention_adaptation
                    .set_focus_pattern(
                        Array1::from(vec![0.1, 0.2, 0.7]), // Recent bias
                    );
            }
            "neural_network" => {
                // Balance between recent and historical information
                self.adaptive_components
                    .attention_adaptation
                    .set_focus_pattern(
                        Array1::from(vec![0.3, 0.4, 0.3]), // Balanced
                    );
            }
            "sparse" => {
                // Focus on gradient magnitude patterns
                self.adaptive_components
                    .attention_adaptation
                    .set_focus_pattern(
                        Array1::from(vec![0.5, 0.3, 0.2]), // Historical bias
                    );
            }
            _ => {
                // Default balanced pattern
                self.adaptive_components
                    .attention_adaptation
                    .set_focus_pattern(Array1::from(vec![0.3, 0.4, 0.3]));
            }
        }

        Ok(())
    }

    /// Fine-tune transformer on specific optimization trajectories
    pub fn fine_tune_on_trajectories(
        &mut self,
        trajectories: &[OptimizationTrajectory],
    ) -> OptimizeResult<()> {
        for trajectory in trajectories {
            // Process each step in trajectory
            for step in &trajectory.steps {
                // Update transformer weights based on successful steps
                if step.improvement > 0.0 {
                    self.update_transformer_weights(&step.state_encoding, &step.action_encoding)?;
                }
            }
        }

        Ok(())
    }

    fn update_transformer_weights(
        &mut self,
        state_encoding: &Array2<f64>,
        action_encoding: &Array1<f64>,
    ) -> OptimizeResult<()> {
        // Simplified weight update (in practice would use proper backpropagation)
        let learning_rate = self.config.meta_learning_rate;

        // Update output projection to better predict successful actions
        for i in 0..self
            .transformer
            .output_projection
            .nrows()
            .min(action_encoding.len())
        {
            for j in 0..self.transformer.output_projection.ncols() {
                if let Some(&state_val) = state_encoding.get((state_encoding.nrows() - 1, j)) {
                    self.transformer.output_projection[[i, j]] +=
                        learning_rate * action_encoding[i] * state_val;
                }
            }
        }

        Ok(())
    }

    /// Get transformer performance metrics
    pub fn get_performance_metrics(&self) -> &TransformerMetrics {
        &self.performance_metrics
    }

    /// Update performance metrics
    fn update_performance_metrics(&mut self) {
        // Compute attention entropy
        if let Some(attention_scores) = self.get_latest_attention_scores() {
            self.performance_metrics.attention_entropy =
                compute_attention_entropy(&attention_scores);
        }

        // Update other metrics based on adaptive components
        self.performance_metrics.lr_adaptation_efficiency = self
            .adaptive_components
            .learning_rate_adapter
            .get_efficiency();

        self.performance_metrics.convergence_detection_accuracy =
            self.adaptive_components.convergence_detector.get_accuracy();
    }

    fn get_latest_attention_scores(&self) -> Option<Array2<f64>> {
        self.transformer
            .transformer_blocks
            .first()?
            .attention
            .attention_scores
            .last()
            .cloned()
    }
}

/// Optimization step output from transformer
#[derive(Debug, Clone)]
pub struct OptimizationStep {
    /// Suggested step size
    pub step_size: f64,
    /// Search direction
    pub direction: Array1<f64>,
    /// Learning rate adaptation factor
    pub lr_adaptation_factor: f64,
    /// Confidence in convergence
    pub convergence_confidence: f64,
    /// Attention weights for interpretability
    pub attention_weights: Array2<f64>,
}

/// Optimization trajectory for training
#[derive(Debug, Clone)]
pub struct OptimizationTrajectory {
    /// Steps in the trajectory
    pub steps: Vec<TrajectoryStep>,
    /// Final objective value
    pub final_objective: f64,
    /// Success indicator
    pub success: bool,
}

/// Single step in optimization trajectory
#[derive(Debug, Clone)]
pub struct TrajectoryStep {
    /// State encoding at this step
    pub state_encoding: Array2<f64>,
    /// Action taken
    pub action_encoding: Array1<f64>,
    /// Improvement achieved
    pub improvement: f64,
    /// Step number
    pub step_number: usize,
}

impl OptimizationTransformer {
    /// Create new optimization transformer
    pub fn new(num_heads: usize, model_dim: usize, max_seq_len: usize, num_layers: usize) -> Self {
        let mut transformer_blocks = Vec::new();

        for _ in 0..num_layers {
            transformer_blocks.push(TransformerBlock::new(num_heads, model_dim));
        }

        // Position encoding
        let position_encoding = Self::create_position_encoding(max_seq_len, model_dim);

        // Input embedding
        let input_embedding = Array2::from_shape_fn((model_dim, model_dim), |_| {
            (rand::rng().gen::<f64>() - 0.5) * (2.0 / model_dim as f64).sqrt()
        });

        // Output projection
        let output_projection = Array2::from_shape_fn((model_dim, model_dim), |_| {
            (rand::rng().gen::<f64>() - 0.5) * (2.0 / model_dim as f64).sqrt()
        });

        Self {
            num_layers,
            transformer_blocks,
            position_encoding,
            input_embedding,
            output_projection,
            model_dim,
        }
    }

    /// Forward pass through transformer
    pub fn forward(&mut self, input_sequence: &ArrayView2<f64>) -> OptimizeResult<Array2<f64>> {
        let seq_len = input_sequence.nrows();
        let input_dim = input_sequence.ncols();

        // Input embedding
        let mut embedded = Array2::zeros((seq_len, self.model_dim));
        for i in 0..seq_len {
            for j in 0..self.model_dim {
                for k in 0..input_dim.min(self.input_embedding.ncols()) {
                    embedded[[i, j]] += self.input_embedding[[j, k]] * input_sequence[[i, k]];
                }
            }
        }

        // Add positional encoding
        for i in 0..seq_len.min(self.position_encoding.nrows()) {
            for j in 0..self.model_dim.min(self.position_encoding.ncols()) {
                embedded[[i, j]] += self.position_encoding[[i, j]];
            }
        }

        // Pass through transformer blocks
        let mut current = embedded;
        for block in &mut self.transformer_blocks {
            current = block.forward(&current.view())?;
        }

        // Output projection
        let mut output = Array2::zeros((seq_len, self.model_dim));
        for i in 0..seq_len {
            for j in 0..self.model_dim {
                for k in 0..self.model_dim.min(self.output_projection.ncols()) {
                    output[[i, j]] += self.output_projection[[j, k]] * current[[i, k]];
                }
            }
        }

        Ok(output)
    }

    /// Create sinusoidal position encoding
    fn create_position_encoding(_max_len: usize, model_dim: usize) -> Array2<f64> {
        let mut pos_encoding = Array2::zeros((_max_len, model_dim));

        for pos in 0.._max_len {
            for i in 0..model_dim {
                let angle = pos as f64 / 10000_f64.powf(2.0 * i as f64 / model_dim as f64);
                if i % 2 == 0 {
                    pos_encoding[[pos, i]] = angle.sin();
                } else {
                    pos_encoding[[pos, i]] = angle.cos();
                }
            }
        }

        pos_encoding
    }
}

impl TransformerBlock {
    /// Create new transformer block
    pub fn new(num_heads: usize, model_dim: usize) -> Self {
        let attention = MultiHeadAttention::new(num_heads, model_dim);
        let feed_forward = FeedForwardNetwork::new(model_dim, model_dim * 4);
        let layer_norm1 = LayerNormalization::new(model_dim);
        let layer_norm2 = LayerNormalization::new(model_dim);

        Self {
            attention,
            feed_forward,
            layer_norm1,
            layer_norm2,
            dropout_rate: 0.1,
        }
    }

    /// Forward pass through transformer block
    pub fn forward(&mut self, input: &ArrayView2<f64>) -> OptimizeResult<Array2<f64>> {
        // Multi-head attention with residual connection
        let attention_output = self.attention.forward(input, input, input)?;
        let residual1 = input + &attention_output.view();
        let after_attention = self.layer_norm1.forward(&residual1.view())?;

        // Feed-forward with residual connection
        let ff_output = self.feed_forward.forward(&after_attention.view())?;
        let residual2 = &after_attention + &ff_output.view();
        let output = self.layer_norm2.forward(&residual2.view())?;

        Ok(output)
    }
}

impl MultiHeadAttention {
    /// Create new multi-head attention
    pub fn new(num_heads: usize, model_dim: usize) -> Self {
        assert_eq!(model_dim % num_heads, 0);
        let head_dim = model_dim / num_heads;

        let w_query = Array2::from_shape_fn((model_dim, model_dim), |_| {
            (rand::rng().gen::<f64>() - 0.5) * (2.0 / model_dim as f64).sqrt()
        });
        let w_key = Array2::from_shape_fn((model_dim, model_dim), |_| {
            (rand::rng().gen::<f64>() - 0.5) * (2.0 / model_dim as f64).sqrt()
        });
        let w_value = Array2::from_shape_fn((model_dim, model_dim), |_| {
            (rand::rng().gen::<f64>() - 0.5) * (2.0 / model_dim as f64).sqrt()
        });
        let w_output = Array2::from_shape_fn((model_dim, model_dim), |_| {
            (rand::rng().gen::<f64>() - 0.5) * (2.0 / model_dim as f64).sqrt()
        });

        Self {
            num_heads,
            head_dim,
            w_query,
            w_key,
            w_value,
            w_output,
            attention_scores: Vec::new(),
        }
    }

    /// Forward pass through multi-head attention
    pub fn forward(
        &mut self,
        query: &ArrayView2<f64>,
        key: &ArrayView2<f64>,
        value: &ArrayView2<f64>,
    ) -> OptimizeResult<Array2<f64>> {
        let seq_len = query.nrows();
        let model_dim = query.ncols();

        // Compute Q, K, V
        let q = self.linear_transform(query, &self.w_query)?;
        let k = self.linear_transform(key, &self.w_key)?;
        let v = self.linear_transform(value, &self.w_value)?;

        // Reshape for multi-head attention
        let mut attention_output = Array2::zeros((seq_len, model_dim));

        for head in 0..self.num_heads {
            let head_start = head * self.head_dim;
            let head_end = head_start + self.head_dim;

            // Extract head-specific Q, K, V
            let q_head = q.slice(ndarray::s![.., head_start..head_end]);
            let k_head = k.slice(ndarray::s![.., head_start..head_end]);
            let v_head = v.slice(ndarray::s![.., head_start..head_end]);

            // Compute attention scores
            let scores = self.compute_attention_scores(&q_head, &k_head)?;

            // Apply attention to values
            let head_output = self.apply_attention(&scores, &v_head)?;

            // Add to output
            for i in 0..seq_len {
                for j in 0..self.head_dim.min(model_dim - head_start) {
                    attention_output[[i, head_start + j]] = head_output[[i, j]];
                }
            }
        }

        // Output projection
        let output = self.linear_transform(&attention_output.view(), &self.w_output)?;

        Ok(output)
    }

    fn linear_transform(
        &self,
        input: &ArrayView2<f64>,
        weight: &Array2<f64>,
    ) -> OptimizeResult<Array2<f64>> {
        let seq_len = input.nrows();
        let input_dim = input.ncols();
        let output_dim = weight.nrows();

        let mut output = Array2::zeros((seq_len, output_dim));

        for i in 0..seq_len {
            for j in 0..output_dim {
                for k in 0..input_dim.min(weight.ncols()) {
                    output[[i, j]] += weight[[j, k]] * input[[i, k]];
                }
            }
        }

        Ok(output)
    }

    fn compute_attention_scores(
        &mut self,
        query: &ArrayView2<f64>,
        key: &ArrayView2<f64>,
    ) -> OptimizeResult<Array2<f64>> {
        let seq_len = query.nrows();
        let head_dim = query.ncols();

        let mut scores = Array2::zeros((seq_len, seq_len));
        let scale = 1.0 / (head_dim as f64).sqrt();

        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot_product = 0.0;
                for k in 0..head_dim {
                    dot_product += query[[i, k]] * key[[j, k]];
                }
                scores[[i, j]] = dot_product * scale;
            }
        }

        // Apply softmax
        for i in 0..seq_len {
            let mut row_sum = 0.0;
            let max_val = scores.row(i).fold(-f64::INFINITY, |a, &b| a.max(b));

            for j in 0..seq_len {
                scores[[i, j]] = (scores[[i, j]] - max_val).exp();
                row_sum += scores[[i, j]];
            }

            if row_sum > 0.0 {
                for j in 0..seq_len {
                    scores[[i, j]] /= row_sum;
                }
            }
        }

        // Store for analysis
        self.attention_scores.push(scores.clone());
        if self.attention_scores.len() > 10 {
            self.attention_scores.remove(0);
        }

        Ok(scores)
    }

    fn apply_attention(
        &self,
        scores: &Array2<f64>,
        values: &ArrayView2<f64>,
    ) -> OptimizeResult<Array2<f64>> {
        let seq_len = scores.nrows();
        let head_dim = values.ncols();

        let mut output = Array2::zeros((seq_len, head_dim));

        for i in 0..seq_len {
            for j in 0..head_dim {
                for k in 0..seq_len {
                    output[[i, j]] += scores[[i, k]] * values[[k, j]];
                }
            }
        }

        Ok(output)
    }
}

impl FeedForwardNetwork {
    /// Create new feed-forward network
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let linear1 = Array2::from_shape_fn((hidden_dim, input_dim), |_| {
            (rand::rng().gen::<f64>() - 0.5) * (2.0 / input_dim as f64).sqrt()
        });
        let linear2 = Array2::from_shape_fn((input_dim, hidden_dim), |_| {
            (rand::rng().gen::<f64>() - 0.5) * (2.0 / hidden_dim as f64).sqrt()
        });

        Self {
            linear1,
            linear2,
            bias1: Array1::zeros(hidden_dim),
            bias2: Array1::zeros(input_dim),
            activation: ActivationType::GELU,
            hidden_dim,
        }
    }

    /// Forward pass through feed-forward network
    pub fn forward(&self, input: &ArrayView2<f64>) -> OptimizeResult<Array2<f64>> {
        let seq_len = input.nrows();
        let input_dim = input.ncols();

        // First linear layer
        let mut hidden = Array2::zeros((seq_len, self.hidden_dim));
        for i in 0..seq_len {
            for j in 0..self.hidden_dim {
                for k in 0..input_dim.min(self.linear1.ncols()) {
                    hidden[[i, j]] += self.linear1[[j, k]] * input[[i, k]];
                }
                hidden[[i, j]] += self.bias1[j];
                hidden[[i, j]] = self.activation.apply(hidden[[i, j]]);
            }
        }

        // Second linear layer
        let mut output = Array2::zeros((seq_len, input_dim));
        for i in 0..seq_len {
            for j in 0..input_dim {
                for k in 0..self.hidden_dim.min(self.linear2.ncols()) {
                    output[[i, j]] += self.linear2[[j, k]] * hidden[[i, k]];
                }
                output[[i, j]] += self.bias2[j];
            }
        }

        Ok(output)
    }
}

impl LayerNormalization {
    /// Create new layer normalization
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: Array1::ones(dim),
            beta: Array1::zeros(dim),
            epsilon: 1e-6,
        }
    }

    /// Forward pass through layer normalization
    pub fn forward(&self, input: &ArrayView2<f64>) -> OptimizeResult<Array2<f64>> {
        let seq_len = input.nrows();
        let dim = input.ncols();
        let mut output = Array2::zeros((seq_len, dim));

        for i in 0..seq_len {
            // Compute mean and variance for this sequence position
            let row = input.row(i);
            let mean = row.mean();
            let var = input.row(i).variance();
            let std = (var + self.epsilon).sqrt();

            // Normalize
            for j in 0..dim.min(self.gamma.len()) {
                output[[i, j]] = self.gamma[j] * (input[[i, j]] - mean) / std + self.beta[j];
            }
        }

        Ok(output)
    }
}

impl TransformerProblemEncoder {
    /// Create new transformer problem encoder
    pub fn new(embedding_dim: usize) -> Self {
        let feature_dim = 20; // Fixed feature dimension

        Self {
            gradient_encoder: Array2::from_shape_fn((embedding_dim, feature_dim), |_| {
                (rand::rng().gen::<f64>() - 0.5) * 0.1
            }),
            hessian_encoder: Array2::from_shape_fn((embedding_dim, feature_dim), |_| {
                (rand::rng().gen::<f64>() - 0.5) * 0.1
            }),
            parameter_encoder: Array2::from_shape_fn((embedding_dim, feature_dim), |_| {
                (rand::rng().gen::<f64>() - 0.5) * 0.1
            }),
            temporal_encoder: Array2::from_shape_fn((embedding_dim, feature_dim), |_| {
                (rand::rng().gen::<f64>() - 0.5) * 0.1
            }),
            context_encoder: Array2::from_shape_fn((embedding_dim, feature_dim), |_| {
                (rand::rng().gen::<f64>() - 0.5) * 0.1
            }),
            embedding_dim,
        }
    }

    /// Encode current state for transformer
    pub fn encode_current_state<F>(
        &self,
        objective: &F,
        current_params: &ArrayView1<f64>,
        problem: &OptimizationProblem,
    ) -> OptimizeResult<Array1<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let mut encoding = Array1::zeros(self.embedding_dim);

        // Encode different aspects
        let param_features = self.encode_parameter_features(current_params);
        let grad_features = self.encode_gradient_features(objective, current_params);
        let context_features = self.encode_context_features(problem);

        // Combine features
        self.combine_features(&mut encoding, &param_features, &self.parameter_encoder);
        self.combine_features(&mut encoding, &grad_features, &self.gradient_encoder);
        self.combine_features(&mut encoding, &context_features, &self.context_encoder);

        Ok(encoding)
    }

    fn encode_parameter_features(&self, params: &ArrayView1<f64>) -> Array1<f64> {
        let mut features = Array1::zeros(20);

        if !params.is_empty() {
            features[0] = params.view().mean().tanh();
            features[1] = params.view().variance().sqrt().tanh();
            features[2] = params.fold(-f64::INFINITY, |a, &b| a.max(b)).tanh();
            features[3] = params.fold(f64::INFINITY, |a, &b| a.min(b)).tanh();
            features[4] = (params.len() as f64).ln().tanh();

            // L-norms
            features[5] =
                (params.iter().map(|&x| x.abs()).sum::<f64>() / params.len() as f64).tanh(); // L1
            features[6] = (params.iter().map(|&x| x * x).sum::<f64>()).sqrt().tanh(); // L2

            // Statistical moments
            let mean = features[0];
            let skewness = params
                .iter()
                .map(|&x| ((x - mean) / (features[1] + 1e-8)).powi(3))
                .sum::<f64>()
                / params.len() as f64;
            features[7] = skewness.tanh();

            // Sparsity
            let zero_count = params.iter().filter(|&&x| x.abs() < 1e-8).count();
            features[8] = (zero_count as f64 / params.len() as f64).tanh();
        }

        features
    }

    fn encode_gradient_features<F>(&self, objective: &F, params: &ArrayView1<f64>) -> Array1<f64>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let mut features = Array1::zeros(20);

        let h = 1e-6;
        let f0 = objective(params);
        let mut gradient = Array1::zeros(params.len());

        // Compute finite difference gradient
        for i in 0..params.len().min(20) {
            // Limit for efficiency
            let mut params_plus = params.to_owned();
            params_plus[i] += h;
            let f_plus = objective(&params_plus.view());
            gradient[i] = (f_plus - f0) / h;
        }

        if !gradient.is_empty() {
            features[0] = (gradient.iter().map(|&g| g * g).sum::<f64>())
                .sqrt()
                .ln()
                .tanh(); // Gradient norm
            features[1] = f0.abs().ln().tanh(); // Objective magnitude
            features[2] = gradient.view().mean().tanh(); // Gradient mean
            features[3] = gradient.view().variance().sqrt().tanh(); // Gradient std

            // Gradient direction consistency
            let grad_consistency = gradient
                .iter()
                .zip(params.iter())
                .map(|(&g, &p)| if p * g < 0.0 { 1.0 } else { 0.0 })
                .sum::<f64>()
                / gradient.len() as f64;
            features[4] = grad_consistency.tanh();
        }

        features
    }

    fn encode_context_features(&self, problem: &OptimizationProblem) -> Array1<f64> {
        let mut features = Array1::zeros(20);

        features[0] = (problem.dimension as f64).ln().tanh();
        features[1] = (problem.max_evaluations as f64).ln().tanh();
        features[2] = problem.target_accuracy.ln().abs().tanh();

        // Problem class encoding (simplified)
        match problem.problem_class.as_str() {
            "quadratic" => features[3] = 1.0,
            "neural_network" => features[4] = 1.0,
            "sparse" => {
                features[5] = 1.0;
                features[6] = 1.0;
            }
            _ => {} // Default case for unknown problem classes
        }

        features
    }

    fn combine_features(
        &self,
        encoding: &mut Array1<f64>,
        features: &Array1<f64>,
        encoder: &Array2<f64>,
    ) {
        for i in 0..encoding.len() {
            for j in 0..features.len().min(encoder.ncols()) {
                encoding[i] += encoder[[i, j]] * features[j];
            }
        }
    }
}

impl OptimizationHistory {
    /// Create new optimization history
    pub fn new(max_length: usize) -> Self {
        Self {
            parameter_history: VecDeque::with_capacity(max_length),
            objective_history: VecDeque::with_capacity(max_length),
            gradient_history: VecDeque::with_capacity(max_length),
            step_size_history: VecDeque::with_capacity(max_length),
            success_history: VecDeque::with_capacity(max_length),
            max_length,
            current_step: 0,
        }
    }

    /// Add optimization step to history
    pub fn add_step(&mut self, params: Array1<f64>, objective: f64, step: OptimizationStep) {
        if self.parameter_history.len() >= self.max_length {
            self.parameter_history.pop_front();
            self.objective_history.pop_front();
            self.gradient_history.pop_front();
            self.step_size_history.pop_front();
            self.success_history.pop_front();
        }

        self.parameter_history.push_back(params);
        self.objective_history.push_back(objective);
        self.gradient_history.push_back(step.direction);
        self.step_size_history.push_back(step.step_size);
        self.success_history
            .push_back(step.convergence_confidence > 0.5);

        self.current_step += 1;
    }
}

impl AdaptiveComponents {
    /// Create new adaptive components
    pub fn new(model_dim: usize) -> Self {
        Self {
            attention_adaptation: AttentionAdaptation::new(model_dim),
            learning_rate_adapter: LearningRateAdapter::new(),
            gradient_scaler: GradientScaler::new(model_dim),
            step_size_predictor: StepSizePredictor::new(model_dim),
            convergence_detector: ConvergenceDetector::new(),
        }
    }
}

impl AttentionAdaptation {
    /// Create new attention adaptation
    pub fn new(model_dim: usize) -> Self {
        Self {
            adaptation_rate: 0.01,
            attention_focus: Array1::from_elem(model_dim, 1.0 / model_dim as f64),
            focus_history: VecDeque::with_capacity(100),
            problem_patterns: HashMap::new(),
        }
    }

    /// Update attention adaptation
    pub fn update(&mut self, attention_weights: &Array2<f64>) -> OptimizeResult<()> {
        if attention_weights.is_empty() {
            return Ok(());
        }

        // Compute attention focus from _weights
        let mut new_focus = Array1::zeros(self.attention_focus.len());
        for i in 0..attention_weights.nrows().min(new_focus.len()) {
            new_focus[i] = attention_weights.row(i).mean();
        }

        // Update focus with momentum
        for i in 0..self.attention_focus.len() {
            self.attention_focus[i] = (1.0 - self.adaptation_rate) * self.attention_focus[i]
                + self.adaptation_rate * new_focus.get(i).copied().unwrap_or(0.0);
        }

        // Record in history
        self.focus_history.push_back(self.attention_focus.clone());
        if self.focus_history.len() > 100 {
            self.focus_history.pop_front();
        }

        Ok(())
    }

    /// Set specific focus pattern
    pub fn set_focus_pattern(&mut self, pattern: Array1<f64>) {
        if pattern.len() <= self.attention_focus.len() {
            for (i, &val) in pattern.iter().enumerate() {
                self.attention_focus[i] = val;
            }
        }
    }
}

impl LearningRateAdapter {
    /// Create new learning rate adapter
    pub fn new() -> Self {
        Self {
            base_lr: 0.01,
            current_lr: 0.01,
            adaptation_params: Array1::from(vec![0.9, 0.1, 0.001]),
            performance_window: VecDeque::with_capacity(10),
            lr_history: Vec::new(),
        }
    }

    /// Update learning rate
    pub fn update(&mut self, lr_factor: f64) -> OptimizeResult<()> {
        self.current_lr = self.base_lr * lr_factor;
        self.lr_history.push(self.current_lr);

        Ok(())
    }

    /// Get adaptation efficiency
    pub fn get_efficiency(&self) -> f64 {
        if self.lr_history.len() < 2 {
            return 0.5;
        }

        // Simple efficiency metric based on LR stability
        let recent_changes: Vec<f64> = self
            .lr_history
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();

        let avg_change = recent_changes.iter().sum::<f64>() / recent_changes.len() as f64;
        (1.0 / (1.0 + avg_change)).min(1.0)
    }
}

impl GradientScaler {
    /// Create new gradient scaler
    pub fn new(model_dim: usize) -> Self {
        Self {
            scale_factors: Array1::ones(model_dim),
            gradient_stats: GradientStatistics {
                mean: Array1::zeros(model_dim),
                variance: Array1::ones(model_dim),
                count: 0,
                momentum: 0.9,
            },
            scaling_params: Array1::from_elem(model_dim, 1.0),
        }
    }
}

impl StepSizePredictor {
    /// Create new step size predictor
    pub fn new(feature_dim: usize) -> Self {
        Self {
            predictor_network: Array2::from_shape_fn((1, feature_dim), |_| {
                (rand::rng().gen::<f64>() - 0.5) * 0.1
            }),
            feature_dim,
            prediction_history: Vec::new(),
            actual_steps: Vec::new(),
        }
    }
}

impl ConvergenceDetector {
    /// Create new convergence detector
    pub fn new() -> Self {
        Self {
            threshold: 1e-6,
            window_size: 10,
            recent_improvements: VecDeque::with_capacity(10),
            convergence_prob: 0.0,
        }
    }

    /// Update convergence detection
    pub fn update(&mut self, confidence: f64) -> OptimizeResult<()> {
        self.convergence_prob = 0.9 * self.convergence_prob + 0.1 * confidence;
        Ok(())
    }

    /// Get detection accuracy
    pub fn get_accuracy(&self) -> f64 {
        self.convergence_prob
    }
}

impl Default for TransformerMetrics {
    fn default() -> Self {
        Self {
            attention_entropy: 0.0,
            lr_adaptation_efficiency: 0.5,
            gradient_prediction_accuracy: 0.5,
            step_size_prediction_accuracy: 0.5,
            convergence_detection_accuracy: 0.5,
        }
    }
}

impl LearnedOptimizer for AdaptiveTransformerOptimizer {
    fn meta_train(&mut self, training_tasks: &[TrainingTask]) -> OptimizeResult<()> {
        for task in training_tasks {
            self.adapt_to_problem_class(&task.problem.problem_class)?;

            // Simulate optimization on task
            let initial_params = match &task.initial_distribution {
                super::ParameterDistribution::Uniform { low, high } => {
                    Array1::from_shape_fn(task.problem.dimension, |_| {
                        low + rand::rng().gen::<f64>() * (high - low)
                    })
                }
                super::ParameterDistribution::Normal { mean, std } => {
                    Array1::from_shape_fn(task.problem.dimension, |_| {
                        mean + std * (rand::rng().gen::<f64>() - 0.5) * 2.0
                    })
                }
                super::ParameterDistribution::Custom { samples } => {
                    if !samples.is_empty() {
                        samples[rand::rng().random_range(0..samples.len())].clone()
                    } else {
                        Array1::zeros(task.problem.dimension)
                    }
                }
            };

            // Simple training objective for meta-learning
            let training_objective = |x: &ArrayView1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();

            // Process a few optimization steps
            for _ in 0..10 {
                let step = self.process_optimization_step(
                    &training_objective,
                    &initial_params.view(),
                    &task.problem,
                )?;
                self.update_performance_metrics();
            }
        }

        Ok(())
    }

    fn adapt_to_problem(
        &mut self,
        problem: &OptimizationProblem,
        initial_params: &ArrayView1<f64>,
    ) -> OptimizeResult<()> {
        self.adapt_to_problem_class(&problem.problem_class)
    }

    fn optimize<F>(
        &mut self,
        objective: F,
        initial_params: &ArrayView1<f64>,
    ) -> OptimizeResult<OptimizeResults<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let mut current_params = initial_params.to_owned();
        let mut best_value = objective(initial_params);
        let mut iterations = 0;

        // Create default problem for encoding
        let default_problem = OptimizationProblem {
            name: "unknown".to_string(),
            dimension: initial_params.len(),
            problem_class: "general".to_string(),
            metadata: HashMap::new(),
            max_evaluations: 1000,
            target_accuracy: 1e-6,
        };

        for iter in 0..1000 {
            iterations = iter;

            // Get optimization step from transformer
            let step = self.process_optimization_step(
                &objective,
                &current_params.view(),
                &default_problem,
            )?;

            // Apply optimization step
            for i in 0..current_params.len().min(step.direction.len()) {
                current_params[i] -= step.step_size * step.direction[i];
            }

            let current_value = objective(&current_params.view());

            if current_value < best_value {
                best_value = current_value;
            }

            // Check convergence
            if step.convergence_confidence > 0.95 || step.step_size < 1e-8 {
                break;
            }
        }

        Ok(OptimizeResults::<f64> {
            x: current_params,
            fun: best_value,
            success: true,
            nit: iterations,
            message: "Transformer optimization completed".to_string(),
            ..OptimizeResults::default()
        })
    }

    fn get_state(&self) -> &MetaOptimizerState {
        &self.meta_state
    }

    fn reset(&mut self) {
        self.history_buffer = OptimizationHistory::new(100);
        self.performance_metrics = TransformerMetrics::default();
        self.meta_state.episode = 0;
    }
}

/// Compute attention entropy for analysis
#[allow(dead_code)]
fn compute_attention_entropy(attention_scores: &Array2<f64>) -> f64 {
    let mut total_entropy = 0.0;
    let num_heads = attention_scores.nrows();

    for i in 0..num_heads {
        let row = attention_scores.row(i);
        let entropy = -row
            .iter()
            .filter(|&&p| p > 1e-8)
            .map(|&p| p * p.ln())
            .sum::<f64>();
        total_entropy += entropy;
    }

    total_entropy / num_heads as f64
}

/// Convenience function for transformer-based optimization
#[allow(dead_code)]
pub fn transformer_optimize<F>(
    objective: F,
    initial_params: &ArrayView1<f64>,
    config: Option<LearnedOptimizationConfig>,
) -> super::OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let config = config.unwrap_or_default();
    let mut optimizer = AdaptiveTransformerOptimizer::new(config);
    optimizer.optimize(objective, initial_params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_optimizer_creation() {
        let config = LearnedOptimizationConfig::default();
        let optimizer = AdaptiveTransformerOptimizer::new(config);

        assert_eq!(optimizer.transformer.num_layers, 6);
        assert!(!optimizer.transformer.transformer_blocks.is_empty());
    }

    #[test]
    fn test_optimization_transformer() {
        let transformer = OptimizationTransformer::new(4, 64, 100, 2);

        assert_eq!(transformer.num_layers, 2);
        assert_eq!(transformer.model_dim, 64);
        assert_eq!(transformer.transformer_blocks.len(), 2);
    }

    #[test]
    fn test_multi_head_attention() {
        let attention = MultiHeadAttention::new(4, 64);

        assert_eq!(attention.num_heads, 4);
        assert_eq!(attention.head_dim, 16);
    }

    #[test]
    fn test_transformer_forward_pass() {
        let mut transformer = OptimizationTransformer::new(2, 32, 10, 1);
        let input = Array2::from_shape_fn((5, 32), |_| rand::rng().gen::<f64>());

        let output = transformer.forward(&input.view()).unwrap();

        assert_eq!(output.nrows(), 5);
        assert_eq!(output.ncols(), 32);
    }

    #[test]
    fn test_problem_encoder() {
        let encoder = TransformerProblemEncoder::new(64);
        let params = Array1::from(vec![1.0, 2.0]);
        let objective = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);

        let problem = OptimizationProblem {
            name: "test".to_string(),
            dimension: 2,
            problem_class: "quadratic".to_string(),
            metadata: HashMap::new(),
            max_evaluations: 1000,
            target_accuracy: 1e-6,
        };

        let encoding = encoder
            .encode_current_state(&objective, &params.view(), &problem)
            .unwrap();

        assert_eq!(encoding.len(), 64);
        assert!(encoding.iter().all(|&x| x.is_finite()));
    }

    #[test]
    #[ignore = "timeout"]
    fn test_transformer_optimization() {
        let objective = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
        let initial = Array1::from(vec![2.0, 2.0]);

        let config = LearnedOptimizationConfig {
            meta_training_episodes: 5,
            hidden_size: 32,
            num_heads: 2,
            ..Default::default()
        };

        let result = transformer_optimize(objective, &initial.view(), Some(config)).unwrap();

        assert!(result.fun >= 0.0);
        assert_eq!(result.x.len(), 2);
        assert!(result.success);
    }
}

#[allow(dead_code)]
pub fn placeholder() {
    // Placeholder function to prevent unused module warnings
}
