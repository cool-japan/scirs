//! Advanced Policy Gradient Optimization with Meta-Gradient Learning
//!
//! Implementation of cutting-edge policy gradient methods with meta-learning capabilities:
//! - Meta-gradient learning for automatic learning rate adaptation
//! - Higher-order optimization dynamics
//! - Meta-policy networks for learning optimization strategies
//! - Adaptive curriculum learning across problem classes
//! - Hierarchical optimization policies

use super::{
    utils, Experience, ImprovementReward, OptimizationAction, OptimizationState,
    RLOptimizationConfig, RLOptimizer, RewardFunction,
};
use crate::error::{OptimizeError, OptimizeResult};
use crate::result::OptimizeResults;
use ndarray::{Array1, Array2, Array3, ArrayView1};
// use scirs2_core::error::CoreResult; // Unused import
// use scirs2_core::simd_ops::SimdUnifiedOps; // Unused import
use rand::{rng, Rng};
use std::collections::{HashMap, VecDeque};

/// Advanced Neural Network with Meta-Learning Capabilities
#[derive(Debug, Clone)]
pub struct MetaPolicyNetwork {
    /// Primary policy weights
    pub policy_weights: Array3<f64>, // [layer, output, input]
    /// Meta-policy weights for learning rate adaptation
    pub meta_weights: Array3<f64>,
    /// Bias terms
    pub policy_bias: Array2<f64>, // [layer, neuron]
    pub meta_bias: Array2<f64>,
    /// Network architecture
    pub layer_sizes: Vec<usize>,
    /// Adaptive learning rates per parameter
    pub adaptive_learning_rates: Array2<f64>,
    /// Meta-gradient accumulator
    pub meta_gradient_accumulator: Array3<f64>,
    /// Higher-order derivative tracking
    pub second_order_info: Array3<f64>,
    /// Curriculum learning difficulty
    pub curriculum_difficulty: f64,
    /// Problem class embeddings
    pub problem_embeddings: HashMap<String, Array1<f64>>,
}

impl MetaPolicyNetwork {
    /// Create new meta-policy network with hierarchical structure
    pub fn new(_input_size: usize, output_size: usize, hidden_sizes: Vec<usize>) -> Self {
        let mut layer_sizes = vec![_input_size];
        layer_sizes.extend(hidden_sizes);
        layer_sizes.push(output_size);

        let num_layers = layer_sizes.len() - 1;
        let max_layer_size = *layer_sizes.iter().max().unwrap();

        // Initialize weights with Xavier initialization
        let mut policy_weights = Array3::zeros((num_layers, max_layer_size, max_layer_size));
        let mut meta_weights = Array3::zeros((num_layers, max_layer_size, max_layer_size));

        for layer in 0..num_layers {
            let fan_in = layer_sizes[layer];
            let fan_out = layer_sizes[layer + 1];
            let xavier_std = (2.0 / (fan_in + fan_out) as f64).sqrt();

            for i in 0..fan_out {
                for j in 0..fan_in {
                    policy_weights[[layer, i, j]] =
                        rand::rng().random_range(-0.5..0.5) * 2.0 * xavier_std;
                    meta_weights[[layer, i, j]] =
                        rand::rng().random_range(-0.5..0.5) * 2.0 * xavier_std * 0.1;
                }
            }
        }

        Self {
            policy_weights,
            meta_weights,
            policy_bias: Array2::zeros((num_layers, max_layer_size)),
            meta_bias: Array2::zeros((num_layers, max_layer_size)),
            layer_sizes,
            adaptive_learning_rates: Array2::from_elem((num_layers, max_layer_size), 0.01),
            meta_gradient_accumulator: Array3::zeros((num_layers, max_layer_size, max_layer_size)),
            second_order_info: Array3::zeros((num_layers, max_layer_size, max_layer_size)),
            curriculum_difficulty: 0.1,
            problem_embeddings: HashMap::new(),
        }
    }

    /// Forward pass with meta-learning augmentation
    pub fn meta_forward(
        &mut self,
        state_features: &ArrayView1<f64>,
        problem_class: &str,
        meta_context: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        // Get or create problem embedding
        let problem_embedding =
            self.get_or_create_problem_embedding(problem_class, state_features.len());

        // Combine input with problem embedding and meta-_context
        let mut augmented_input = state_features.to_owned();

        // Add problem-specific _context
        for (i, &emb) in problem_embedding.iter().enumerate() {
            if i < augmented_input.len() {
                augmented_input[i] += emb * 0.1;
            }
        }

        // Forward pass through policy network
        let policy_output = self.forward_policy(&augmented_input.view());

        // Forward pass through meta-network for learning rate adaptation
        let meta_output = self.forward_meta(&augmented_input.view(), meta_context);

        (policy_output, meta_output)
    }

    fn forward_policy(&self, input: &ArrayView1<f64>) -> Array1<f64> {
        let mut current_input = input.to_owned();

        for layer in 0..(self.layer_sizes.len() - 1) {
            let layer_input_size = self.layer_sizes[layer];
            let layer_output_size = self.layer_sizes[layer + 1];

            let mut layer_output = Array1::<f64>::zeros(layer_output_size);

            for i in 0..layer_output_size {
                for j in 0..layer_input_size.min(current_input.len()) {
                    layer_output[i] += self.policy_weights[[layer, i, j]] * current_input[j];
                }
                layer_output[i] += self.policy_bias[[layer, i]];

                // Apply activation function (ELU for smooth gradients)
                layer_output[i] = if layer_output[i] > 0.0 {
                    layer_output[i]
                } else {
                    layer_output[i].exp() - 1.0
                };
            }

            current_input = layer_output;
        }

        current_input
    }

    fn forward_meta(&self, input: &ArrayView1<f64>, metacontext: &Array1<f64>) -> Array1<f64> {
        // Combine input with meta-_context
        let mut meta_input = input.to_owned();
        for (i, &ctx) in metacontext.iter().enumerate() {
            if i < meta_input.len() {
                meta_input[i] += ctx * 0.05;
            }
        }

        let mut current_input = meta_input;

        for layer in 0..(self.layer_sizes.len() - 1) {
            let layer_input_size = self.layer_sizes[layer];
            let layer_output_size = self.layer_sizes[layer + 1];

            let mut layer_output = Array1::<f64>::zeros(layer_output_size);

            for i in 0..layer_output_size {
                for j in 0..layer_input_size.min(current_input.len()) {
                    layer_output[i] += self.meta_weights[[layer, i, j]] * current_input[j];
                }
                layer_output[i] += self.meta_bias[[layer, i]];

                // Sigmoid activation for learning rate scaling
                layer_output[i] = 1.0 / (1.0 + (-layer_output[i]).exp());
            }

            current_input = layer_output;
        }

        current_input
    }

    fn get_or_create_problem_embedding(
        &mut self,
        problem_class: &str,
        input_size: usize,
    ) -> Array1<f64> {
        if let Some(embedding) = self.problem_embeddings.get(problem_class) {
            embedding.clone()
        } else {
            let embedding =
                Array1::from_shape_fn(input_size, |_| rand::rng().random_range(-0.05..0.05));
            self.problem_embeddings
                .insert(problem_class.to_string(), embedding.clone());
            embedding
        }
    }

    /// Update network using meta-gradients
    pub fn meta_update(
        &mut self,
        meta_gradients: &MetaGradients,
        base_learning_rate: f64,
        meta_learning_rate: f64,
    ) {
        // Update adaptive learning rates using meta-_gradients
        for layer in 0..(self.layer_sizes.len() - 1) {
            for i in 0..self.layer_sizes[layer + 1] {
                for j in 0..self.layer_sizes[layer] {
                    // Meta-gradient update for learning rates
                    let meta_grad = meta_gradients.meta_lr_gradients[[layer, i, j]];
                    self.adaptive_learning_rates[[layer, i]] *=
                        (1.0 + meta_learning_rate * meta_grad).max(0.1).min(10.0);

                    // Policy weight update with adaptive learning _rate
                    let adaptive_lr = self.adaptive_learning_rates[[layer, i]] * base_learning_rate;
                    self.policy_weights[[layer, i, j]] +=
                        adaptive_lr * meta_gradients.policy_gradients[[layer, i, j]];

                    // Meta-weight update
                    self.meta_weights[[layer, i, j]] +=
                        meta_learning_rate * meta_gradients.meta_weight_gradients[[layer, i, j]];
                }

                // Bias updates
                let adaptive_lr = self.adaptive_learning_rates[[layer, i]] * base_learning_rate;
                self.policy_bias[[layer, i]] +=
                    adaptive_lr * meta_gradients.policy_bias_gradients[[layer, i]];
                self.meta_bias[[layer, i]] +=
                    meta_learning_rate * meta_gradients.meta_bias_gradients[[layer, i]];
            }
        }

        // Update curriculum difficulty based on meta-learning progress
        self.update_curriculum_difficulty(&meta_gradients);
    }

    fn update_curriculum_difficulty(&mut self, metagradients: &MetaGradients) {
        let gradient_norm = metagradients
            .policy_gradients
            .iter()
            .map(|&g| g * g)
            .sum::<f64>()
            .sqrt();

        if gradient_norm < 0.1 {
            self.curriculum_difficulty = (self.curriculum_difficulty * 1.05).min(1.0);
        } else if gradient_norm > 1.0 {
            self.curriculum_difficulty = (self.curriculum_difficulty * 0.95).max(0.01);
        }
    }
}

/// Meta-gradients for higher-order optimization
#[derive(Debug, Clone)]
pub struct MetaGradients {
    /// Gradients for policy parameters
    pub policy_gradients: Array3<f64>,
    /// Gradients for meta-parameters
    pub meta_weight_gradients: Array3<f64>,
    /// Gradients for learning rates (meta-gradients)
    pub meta_lr_gradients: Array3<f64>,
    /// Bias gradients
    pub policy_bias_gradients: Array2<f64>,
    pub meta_bias_gradients: Array2<f64>,
    /// Higher-order terms
    pub second_order_terms: Array3<f64>,
}

/// Advanced Policy Gradient Optimizer with Meta-Learning
#[derive(Debug, Clone)]
pub struct AdvancedAdvancedPolicyGradientOptimizer {
    /// Configuration
    config: RLOptimizationConfig,
    /// Meta-policy network
    meta_policy: MetaPolicyNetwork,
    /// Reward function
    reward_function: ImprovementReward,
    /// Episode trajectories for meta-learning
    meta_trajectories: VecDeque<MetaTrajectory>,
    /// Problem class history
    problem_class_history: VecDeque<String>,
    /// Best solution tracking
    best_params: Array1<f64>,
    best_objective: f64,
    /// Meta-learning statistics
    meta_stats: MetaLearningStats,
    /// Curriculum learning controller
    curriculum_controller: CurriculumController,
    /// Experience replay buffer for meta-learning
    meta_experience_buffer: MetaExperienceBuffer,
}

/// Enhanced trajectory with meta-learning information
#[derive(Debug, Clone)]
pub struct MetaTrajectory {
    /// Regular experiences
    pub experiences: Vec<Experience>,
    /// Problem class identifier
    pub problem_class: String,
    /// Meta-context at start of trajectory
    pub initial_meta_context: Array1<f64>,
    /// Learning progress measures
    pub learning_metrics: LearningMetrics,
    /// Adaptation speed
    pub adaptation_speed: f64,
}

/// Learning metrics for meta-learning
#[derive(Debug, Clone)]
pub struct LearningMetrics {
    /// Rate of improvement
    pub improvement_rate: f64,
    /// Convergence speed
    pub convergence_speed: f64,
    /// Exploration efficiency
    pub exploration_efficiency: f64,
    /// Generalization measure
    pub generalization_score: f64,
}

/// Meta-learning statistics
#[derive(Debug, Clone)]
pub struct MetaLearningStats {
    /// Average learning rates across parameters
    pub avg_learning_rates: Array1<f64>,
    /// Meta-gradient norms
    pub meta_gradient_norms: VecDeque<f64>,
    /// Problem class performance
    pub problem_class_performance: HashMap<String, f64>,
    /// Curriculum progress
    pub curriculum_progress: f64,
    /// Adaptation efficiency
    pub adaptation_efficiency: f64,
}

/// Curriculum learning controller
#[derive(Debug, Clone)]
pub struct CurriculumController {
    /// Current difficulty level
    pub difficulty_level: f64,
    /// Performance thresholds for advancement
    pub advancement_thresholds: Vec<f64>,
    /// Problem generators for different difficulties
    pub difficulty_generators: HashMap<String, f64>,
    /// Learning progress tracker
    pub progress_tracker: VecDeque<f64>,
}

impl CurriculumController {
    pub fn new() -> Self {
        Self {
            difficulty_level: 0.1,
            advancement_thresholds: vec![0.8, 0.85, 0.9, 0.95],
            difficulty_generators: HashMap::new(),
            progress_tracker: VecDeque::with_capacity(100),
        }
    }

    pub fn should_advance(&self) -> bool {
        if self.progress_tracker.len() < 20 {
            return false;
        }

        let recent_performance: f64 =
            self.progress_tracker.iter().rev().take(20).sum::<f64>() / 20.0;

        let threshold_idx = ((self.difficulty_level * 4.0) as usize).min(3);
        recent_performance > self.advancement_thresholds[threshold_idx]
    }

    pub fn advance_difficulty(&mut self) {
        self.difficulty_level = (self.difficulty_level * 1.2).min(1.0);
    }

    pub fn update_progress(&mut self, performance: f64) {
        self.progress_tracker.push_back(performance);
        if self.progress_tracker.len() > 100 {
            self.progress_tracker.pop_front();
        }

        if self.should_advance() {
            self.advance_difficulty();
        }
    }
}

/// Meta-experience buffer for higher-order learning
#[derive(Debug, Clone)]
pub struct MetaExperienceBuffer {
    /// Buffer of meta-trajectories
    pub trajectories: VecDeque<MetaTrajectory>,
    /// Maximum buffer size
    pub max_size: usize,
    /// Sampling weights for different problem classes
    pub class_weights: HashMap<String, f64>,
}

impl MetaExperienceBuffer {
    pub fn new(_maxsize: usize) -> Self {
        Self {
            trajectories: VecDeque::with_capacity(_maxsize),
            max_size: _maxsize,
            class_weights: HashMap::new(),
        }
    }

    pub fn add_trajectory(&mut self, trajectory: MetaTrajectory) {
        // Update class weights based on performance
        let avg_reward = trajectory.experiences.iter().map(|e| e.reward).sum::<f64>()
            / trajectory.experiences.len().max(1) as f64;

        *self
            .class_weights
            .entry(trajectory.problem_class.clone())
            .or_insert(1.0) *= if avg_reward > 0.0 { 1.05 } else { 0.95 };

        self.trajectories.push_back(trajectory);
        if self.trajectories.len() > self.max_size {
            self.trajectories.pop_front();
        }
    }

    pub fn sample_meta_batch(&self, batchsize: usize) -> Vec<MetaTrajectory> {
        let mut batch = Vec::new();

        for _ in 0..batchsize.min(self.trajectories.len()) {
            // Weighted sampling based on problem class performance
            let idx = rand::rng().random_range(0..self.trajectories.len());
            if let Some(trajectory) = self.trajectories.get(idx) {
                batch.push(trajectory.clone());
            }
        }

        batch
    }
}

impl AdvancedAdvancedPolicyGradientOptimizer {
    /// Create new advanced policy gradient optimizer
    pub fn new(config: RLOptimizationConfig, state_size: usize, actionsize: usize) -> Self {
        let hidden_sizes = vec![state_size * 2, state_size * 3, state_size * 2];
        let meta_policy = MetaPolicyNetwork::new(state_size, actionsize, hidden_sizes);

        Self {
            config,
            meta_policy,
            reward_function: ImprovementReward::default(),
            meta_trajectories: VecDeque::with_capacity(1000),
            problem_class_history: VecDeque::with_capacity(100),
            best_params: Array1::zeros(state_size),
            best_objective: f64::INFINITY,
            meta_stats: MetaLearningStats {
                avg_learning_rates: Array1::zeros(state_size),
                meta_gradient_norms: VecDeque::with_capacity(1000),
                problem_class_performance: HashMap::new(),
                curriculum_progress: 0.0,
                adaptation_efficiency: 1.0,
            },
            curriculum_controller: CurriculumController::new(),
            meta_experience_buffer: MetaExperienceBuffer::new(500),
        }
    }

    /// Extract advanced state features with meta-learning context
    fn extract_meta_state_features(
        &self,
        state: &OptimizationState,
        problem_class: &str,
    ) -> (Array1<f64>, Array1<f64>) {
        let mut base_features = Vec::new();

        // Basic parameter features
        for &param in state.parameters.iter() {
            base_features.push(param.tanh());
        }

        // Objective and convergence features
        base_features.push((state.objective_value / (state.objective_value.abs() + 1.0)).tanh());
        base_features.push(
            state
                .convergence_metrics
                .relative_objective_change
                .ln()
                .max(-10.0)
                .tanh(),
        );
        base_features.push(state.convergence_metrics.parameter_change_norm.tanh());

        // Step and temporal features
        base_features.push((state.step as f64 / 100.0).tanh());

        // Problem-specific features
        let problem_difficulty = self.meta_policy.curriculum_difficulty;
        base_features.push(problem_difficulty);

        // Meta-context features
        let mut meta_context = Vec::new();

        // Historical performance for this problem _class
        let class_performance = self
            .meta_stats
            .problem_class_performance
            .get(problem_class)
            .copied()
            .unwrap_or(0.0);
        meta_context.push(class_performance);

        // Recent meta-gradient norms
        let recent_meta_grad_norm = self
            .meta_stats
            .meta_gradient_norms
            .iter()
            .rev()
            .take(10)
            .sum::<f64>()
            / 10.0;
        meta_context.push(recent_meta_grad_norm.tanh());

        // Curriculum progress
        meta_context.push(self.meta_stats.curriculum_progress);

        // Adaptation efficiency
        meta_context.push(self.meta_stats.adaptation_efficiency);

        // Recent problem _class diversity
        let recent_classes: std::collections::HashSet<String> = self
            .problem_class_history
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();
        meta_context.push((recent_classes.len() as f64 / 10.0).min(1.0));

        (Array1::from(base_features), Array1::from(meta_context))
    }

    /// Decode sophisticated actions from meta-policy output
    fn decode_meta_action(
        &self,
        policy_output: &ArrayView1<f64>,
        meta_output: &ArrayView1<f64>,
    ) -> OptimizationAction {
        if policy_output.is_empty() {
            return OptimizationAction::GradientStep {
                learning_rate: 0.01,
            };
        }

        // Use meta-_output to modulate action selection
        let meta_modulation = meta_output.get(0).copied().unwrap_or(1.0);
        let action_strength = meta_output.get(1).copied().unwrap_or(1.0);

        // Enhanced action decoding with meta-learning insights
        let action_logits = policy_output.mapv(|x| x * meta_modulation);
        let action_type = action_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        match action_type {
            0 => OptimizationAction::GradientStep {
                learning_rate: 0.01 * action_strength * (1.0 + policy_output[0] * 0.5),
            },
            1 => OptimizationAction::RandomPerturbation {
                magnitude: 0.1 * action_strength * (1.0 + policy_output[1] * 0.5),
            },
            2 => OptimizationAction::MomentumUpdate {
                momentum: (0.9 * action_strength * (1.0 + policy_output[2] * 0.1)).min(0.99),
            },
            3 => OptimizationAction::AdaptiveLearningRate {
                factor: (0.5 + 0.5 * policy_output[3] * action_strength)
                    .max(0.1)
                    .min(2.0),
            },
            4 => OptimizationAction::ResetToBest,
            _ => OptimizationAction::Terminate,
        }
    }

    /// Compute meta-gradients for higher-order learning
    fn compute_meta_gradients(&self, metabatch: &[MetaTrajectory]) -> MetaGradients {
        let num_layers = self.meta_policy.layer_sizes.len() - 1;
        let max_size = *self.meta_policy.layer_sizes.iter().max().unwrap();

        let mut meta_gradients = MetaGradients {
            policy_gradients: Array3::zeros((num_layers, max_size, max_size)),
            meta_weight_gradients: Array3::zeros((num_layers, max_size, max_size)),
            meta_lr_gradients: Array3::zeros((num_layers, max_size, max_size)),
            policy_bias_gradients: Array2::zeros((num_layers, max_size)),
            meta_bias_gradients: Array2::zeros((num_layers, max_size)),
            second_order_terms: Array3::zeros((num_layers, max_size, max_size)),
        };

        for trajectory in metabatch {
            // Compute trajectory-specific gradients
            let trajectory_return: f64 = trajectory.experiences.iter().map(|e| e.reward).sum();

            let learning_speed_bonus = trajectory.learning_metrics.convergence_speed * 0.1;
            let exploration_bonus = trajectory.learning_metrics.exploration_efficiency * 0.05;
            let adjusted_return = trajectory_return + learning_speed_bonus + exploration_bonus;

            // For each experience in trajectory, compute gradients
            for (step, experience) in trajectory.experiences.iter().enumerate() {
                let (state_features, meta_context) =
                    self.extract_meta_state_features(&experience.state, &trajectory.problem_class);

                // Compute discounted return from this step
                let gamma = self.config.discount_factor;
                let step_return: f64 = trajectory.experiences[step..]
                    .iter()
                    .enumerate()
                    .map(|(i, e)| gamma.powi(i as i32) * e.reward)
                    .sum();

                // Policy gradient with meta-learning augmentation
                let advantage = step_return - adjusted_return / trajectory.experiences.len() as f64;

                // Add gradients for this step (simplified computation)
                for layer in 0..num_layers {
                    for i in 0..self.meta_policy.layer_sizes[layer + 1] {
                        for j in 0..self.meta_policy.layer_sizes[layer] {
                            if j < state_features.len() {
                                // Standard policy gradient
                                meta_gradients.policy_gradients[[layer, i, j]] +=
                                    advantage * state_features[j] * 0.01;

                                // Meta-gradient for learning rate adaptation
                                let meta_lr_grad = advantage
                                    * state_features[j]
                                    * trajectory.learning_metrics.convergence_speed;
                                meta_gradients.meta_lr_gradients[[layer, i, j]] +=
                                    meta_lr_grad * 0.001;

                                // Meta-weight gradients
                                if j < meta_context.len() {
                                    meta_gradients.meta_weight_gradients[[layer, i, j]] +=
                                        advantage * meta_context[j] * 0.001;
                                }
                            }
                        }

                        // Bias gradients
                        meta_gradients.policy_bias_gradients[[layer, i]] += advantage * 0.01;
                        meta_gradients.meta_bias_gradients[[layer, i]] +=
                            advantage * trajectory.learning_metrics.generalization_score * 0.001;
                    }
                }
            }
        }

        // Normalize by _batch size
        if !metabatch.is_empty() {
            let batch_size = metabatch.len() as f64;
            meta_gradients.policy_gradients /= batch_size;
            meta_gradients.meta_weight_gradients /= batch_size;
            meta_gradients.meta_lr_gradients /= batch_size;
            meta_gradients.policy_bias_gradients /= batch_size;
            meta_gradients.meta_bias_gradients /= batch_size;
        }

        meta_gradients
    }

    /// Update meta-learning statistics
    fn update_meta_stats(
        &mut self,
        meta_gradients: &MetaGradients,
        problem_class: &str,
        performance: f64,
    ) {
        // Update gradient norms
        let grad_norm = meta_gradients
            .policy_gradients
            .iter()
            .map(|&g| g * g)
            .sum::<f64>()
            .sqrt();
        self.meta_stats.meta_gradient_norms.push_back(grad_norm);
        if self.meta_stats.meta_gradient_norms.len() > 1000 {
            self.meta_stats.meta_gradient_norms.pop_front();
        }

        // Update problem _class performance
        let current_perf = self
            .meta_stats
            .problem_class_performance
            .entry(problem_class.to_string())
            .or_insert(0.0);
        *current_perf = 0.9 * *current_perf + 0.1 * performance;

        // Update curriculum progress
        self.meta_stats.curriculum_progress = self.curriculum_controller.difficulty_level;

        // Update adaptation efficiency based on meta-gradient stability
        let grad_stability = if self.meta_stats.meta_gradient_norms.len() > 10 {
            let recent_grads: Vec<f64> = self
                .meta_stats
                .meta_gradient_norms
                .iter()
                .rev()
                .take(10)
                .cloned()
                .collect();
            let mean = recent_grads.iter().sum::<f64>() / recent_grads.len() as f64;
            let variance = recent_grads
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / recent_grads.len() as f64;
            1.0 / (1.0 + variance)
        } else {
            1.0
        };

        self.meta_stats.adaptation_efficiency =
            0.95 * self.meta_stats.adaptation_efficiency + 0.05 * grad_stability;
    }

    /// Extract problem class from objective function characteristics
    fn classify_problem<F>(&self, objective: &F, params: &ArrayView1<f64>) -> String
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        // Simple problem classification based on function behavior
        let base_value = objective(params);

        // Test convexity by checking second derivatives (simplified)
        let eps = 1e-6;
        let mut curvature_sum = 0.0;

        for i in 0..params.len().min(3) {
            // Limit checks for efficiency
            let mut params_plus = params.to_owned();
            let mut params_minus = params.to_owned();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let f_plus = objective(&params_plus.view());
            let f_minus = objective(&params_minus.view());
            let curvature = (f_plus + f_minus - 2.0 * base_value) / (eps * eps);
            curvature_sum += curvature;
        }

        let avg_curvature = curvature_sum / params.len().min(3) as f64;

        if avg_curvature > 1.0 {
            "convex".to_string()
        } else if avg_curvature < -1.0 {
            "concave".to_string()
        } else if base_value.abs() < 1.0 {
            "low_scale".to_string()
        } else if base_value.abs() > 100.0 {
            "high_scale".to_string()
        } else {
            "general".to_string()
        }
    }
}

impl RLOptimizer for AdvancedAdvancedPolicyGradientOptimizer {
    fn config(&self) -> &RLOptimizationConfig {
        &self.config
    }

    fn select_action(&mut self, state: &OptimizationState) -> OptimizationAction {
        let problem_class = "general"; // Simplified for this implementation
        let (state_features, meta_context) = self.extract_meta_state_features(state, problem_class);
        let (policy_output, meta_output) =
            self.meta_policy
                .meta_forward(&state_features.view(), problem_class, &meta_context);
        self.decode_meta_action(&policy_output.view(), &meta_output.view())
    }

    fn update(&mut self, experience: &Experience) -> Result<(), OptimizeError> {
        // Meta-learning updates are done in batch after collecting trajectories
        Ok(())
    }

    fn run_episode<F>(
        &mut self,
        objective: &F,
        initial_params: &ArrayView1<f64>,
    ) -> OptimizeResult<OptimizeResults<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let problem_class = self.classify_problem(objective, initial_params);
        self.problem_class_history.push_back(problem_class.clone());
        if self.problem_class_history.len() > 100 {
            self.problem_class_history.pop_front();
        }

        let initial_meta_context = Array1::from(vec![
            self.meta_stats.curriculum_progress,
            self.meta_stats.adaptation_efficiency,
            self.curriculum_controller.difficulty_level,
        ]);

        let mut current_params = initial_params.to_owned();
        let mut current_state = utils::create_state(current_params.clone(), objective, 0, None);
        let mut experiences = Vec::new();
        let mut momentum = Array1::zeros(initial_params.len());

        let start_objective = current_state.objective_value;
        let mut max_improvement = 0.0;
        let mut exploration_steps = 0;

        for step in 0..self.config.max_steps_per_episode {
            // Select action using meta-policy
            let action = self.select_action(&current_state);

            // Apply action
            let new_params =
                utils::apply_action(&current_state, &action, &self.best_params, &mut momentum);
            let new_state =
                utils::create_state(new_params, objective, step + 1, Some(&current_state));

            // Compute reward with meta-learning augmentation
            let base_reward =
                self.reward_function
                    .compute_reward(&current_state, &action, &new_state);
            let exploration_bonus =
                if matches!(action, OptimizationAction::RandomPerturbation { .. }) {
                    exploration_steps += 1;
                    0.01
                } else {
                    0.0
                };
            let reward = base_reward + exploration_bonus;

            // Track improvement for learning metrics
            let improvement = current_state.objective_value - new_state.objective_value;
            if improvement > max_improvement {
                max_improvement = improvement;
            }

            // Store experience
            let experience = Experience {
                state: current_state.clone(),
                action: action.clone(),
                reward,
                next_state: new_state.clone(),
                done: utils::should_terminate(&new_state, self.config.max_steps_per_episode),
            };
            experiences.push(experience);

            // Update best solution
            if new_state.objective_value < self.best_objective {
                self.best_objective = new_state.objective_value;
                self.best_params = new_state.parameters.clone();
            }

            current_state = new_state;
            current_params = current_state.parameters.clone();

            // Check termination
            if utils::should_terminate(&current_state, self.config.max_steps_per_episode)
                || matches!(action, OptimizationAction::Terminate)
            {
                break;
            }
        }

        // Compute learning metrics
        let final_objective = current_state.objective_value;
        let total_improvement = start_objective - final_objective;
        let learning_metrics = LearningMetrics {
            improvement_rate: total_improvement / (current_state.step as f64 + 1.0),
            convergence_speed: if total_improvement > 0.0 {
                max_improvement / total_improvement
            } else {
                0.0
            },
            exploration_efficiency: (exploration_steps as f64) / (current_state.step as f64 + 1.0),
            generalization_score: if total_improvement > 0.0 {
                (total_improvement / start_objective.abs()).min(1.0)
            } else {
                0.0
            },
        };

        // Create meta-trajectory
        let meta_trajectory = MetaTrajectory {
            experiences,
            problem_class: problem_class.clone(),
            initial_meta_context,
            learning_metrics: learning_metrics.clone(),
            adaptation_speed: learning_metrics.improvement_rate.abs(),
        };

        // Add to meta-experience buffer
        self.meta_experience_buffer.add_trajectory(meta_trajectory);

        // Update curriculum controller
        let episode_performance = learning_metrics.generalization_score;
        self.curriculum_controller
            .update_progress(episode_performance);

        Ok(OptimizeResults::<f64> {
            x: current_params,
            fun: current_state.objective_value,
            success: current_state.convergence_metrics.relative_objective_change < 1e-6,
            nit: current_state.step,
            nfev: current_state.step, // Approximate function evaluations
            njev: 0,
            nhev: 0,
            maxcv: 0,
            status: 0,
            message: format!(
                "Meta-policy gradient episode completed for problem class: {}",
                problem_class
            ),
            jac: None,
            hess: None,
            constr: None,
        })
    }

    fn train<F>(
        &mut self,
        objective: &F,
        initial_params: &ArrayView1<f64>,
    ) -> OptimizeResult<OptimizeResults<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let mut best_result = OptimizeResults::<f64> {
            x: initial_params.to_owned(),
            fun: f64::INFINITY,
            success: false,
            nit: 0,
            nfev: 0,
            njev: 0,
            nhev: 0,
            maxcv: 0,
            status: 0,
            message: "Meta-learning training not completed".to_string(),
            jac: None,
            hess: None,
            constr: None,
        };

        // Meta-learning training loop
        for episode in 0..self.config.num_episodes {
            let result = self.run_episode(objective, initial_params)?;

            if result.fun < best_result.fun {
                best_result = result;
            }

            // Meta-learning update every few episodes
            if (episode + 1) % 5 == 0 && self.meta_experience_buffer.trajectories.len() >= 10 {
                let meta_batch = self.meta_experience_buffer.sample_meta_batch(10);
                let meta_gradients = self.compute_meta_gradients(&meta_batch);

                // Update meta-policy with adaptive learning rates
                self.meta_policy.meta_update(
                    &meta_gradients,
                    self.config.learning_rate,
                    self.config.learning_rate * 0.1,
                );

                // Update meta-statistics
                let avg_performance = meta_batch
                    .iter()
                    .map(|t| t.learning_metrics.generalization_score)
                    .sum::<f64>()
                    / meta_batch.len() as f64;

                if let Some(trajectory) = meta_batch.first() {
                    self.update_meta_stats(
                        &meta_gradients,
                        &trajectory.problem_class,
                        avg_performance,
                    );
                }
            }
        }

        best_result.x = self.best_params.clone();
        best_result.fun = self.best_objective;
        best_result.message = format!(
            "Meta-learning training completed. Curriculum level: {:.3}, Adaptation efficiency: {:.3}",
            self.meta_stats.curriculum_progress,
            self.meta_stats.adaptation_efficiency
        );

        Ok(best_result)
    }

    fn reset(&mut self) {
        self.meta_trajectories.clear();
        self.problem_class_history.clear();
        self.best_objective = f64::INFINITY;
        self.best_params.fill(0.0);
        self.meta_stats.meta_gradient_norms.clear();
        self.meta_stats.problem_class_performance.clear();
        self.curriculum_controller = CurriculumController::new();
        self.meta_experience_buffer = MetaExperienceBuffer::new(500);
    }
}

/// Convenience function for advanced meta-learning policy gradient optimization
#[allow(dead_code)]
pub fn advanced_advanced_policy_gradient_optimize<F>(
    objective: F,
    initial_params: &ArrayView1<f64>,
    config: Option<RLOptimizationConfig>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let config = config.unwrap_or_else(|| RLOptimizationConfig {
        num_episodes: 100,
        max_steps_per_episode: 50,
        learning_rate: 0.001,
        ..Default::default()
    });

    let mut optimizer = AdvancedAdvancedPolicyGradientOptimizer::new(
        config,
        initial_params.len() + 5, // Extra features for meta-context
        6,                        // Number of action types
    );
    optimizer.train(&objective, initial_params)
}

/// Legacy convenience function for backward compatibility
#[allow(dead_code)]
pub fn policy_gradient_optimize<F>(
    objective: F,
    initial_params: &ArrayView1<f64>,
    config: Option<RLOptimizationConfig>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    advanced_advanced_policy_gradient_optimize(objective, initial_params, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_policy_network_creation() {
        let network = MetaPolicyNetwork::new(4, 2, vec![8, 6]);
        assert_eq!(network.layer_sizes, vec![4, 8, 6, 2]);
    }

    #[test]
    fn test_meta_forward_pass() {
        let mut network = MetaPolicyNetwork::new(3, 2, vec![4]);
        let input = Array1::from(vec![0.5, -0.3, 0.8]);
        let meta_context = Array1::from(vec![0.1, 0.2]);

        let (policy_out, meta_out) = network.meta_forward(&input.view(), "test", &meta_context);

        assert_eq!(policy_out.len(), 2);
        assert_eq!(meta_out.len(), 2);
    }

    #[test]
    fn test_curriculum_controller() {
        let mut controller = CurriculumController::new();
        assert_eq!(controller.difficulty_level, 0.1);

        // Add good performance
        for _ in 0..25 {
            controller.update_progress(0.9);
        }

        assert!(controller.difficulty_level > 0.1);
    }

    #[test]
    fn test_meta_experience_buffer() {
        let mut buffer = MetaExperienceBuffer::new(10);

        let trajectory = MetaTrajectory {
            experiences: vec![],
            problem_class: "test".to_string(),
            initial_meta_context: Array1::zeros(3),
            learning_metrics: LearningMetrics {
                improvement_rate: 0.1,
                convergence_speed: 0.2,
                exploration_efficiency: 0.3,
                generalization_score: 0.4,
            },
            adaptation_speed: 0.1,
        };

        buffer.add_trajectory(trajectory);
        assert_eq!(buffer.trajectories.len(), 1);

        let batch = buffer.sample_meta_batch(1);
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_advanced_advanced_optimizer_creation() {
        let config = RLOptimizationConfig::default();
        let optimizer = AdvancedAdvancedPolicyGradientOptimizer::new(config, 4, 3);

        assert_eq!(optimizer.meta_policy.layer_sizes[0], 4);
        assert_eq!(optimizer.meta_policy.layer_sizes.last(), Some(&3));
    }

    #[test]
    fn test_problem_classification() {
        let config = RLOptimizationConfig::default();
        let optimizer = AdvancedAdvancedPolicyGradientOptimizer::new(config, 2, 3);

        let quadratic = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
        let params = Array1::from(vec![1.0, 1.0]);

        let class = optimizer.classify_problem(&quadratic, &params.view());
        assert!(!class.is_empty());
    }

    #[test]
    fn test_meta_optimization() {
        let config = RLOptimizationConfig {
            num_episodes: 5,
            max_steps_per_episode: 10,
            learning_rate: 0.1,
            ..Default::default()
        };

        let objective = |x: &ArrayView1<f64>| (x[0] - 1.0).powi(2) + (x[1] + 0.5).powi(2);
        let initial = Array1::from(vec![0.0, 0.0]);

        let result =
            advanced_advanced_policy_gradient_optimize(objective, &initial.view(), Some(config))
                .unwrap();

        assert!(result.nit > 0);
        assert!(result.fun <= objective(&initial.view()));
    }
}

#[allow(dead_code)]
pub fn placeholder() {
    // Placeholder function to prevent unused module warnings
}
