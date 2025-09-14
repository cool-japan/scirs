//! Actor-Critic Methods for Optimization
//!
//! Advanced implementation of actor-critic algorithms that combine policy learning (actor)
//! with value function estimation (critic) for sophisticated optimization strategies.

use super::{
    utils, Experience, ExperienceBuffer, ImprovementReward, OptimizationAction, OptimizationState,
    RLOptimizationConfig, RLOptimizer, RewardFunction,
};
use crate::error::{OptimizeError, OptimizeResult};
use crate::result::OptimizeResults;
use ndarray::{Array1, Array2, ArrayView1};
use rand::{rng, Rng};
// use std::collections::VecDeque; // Unused import

/// Actor network for policy learning
#[derive(Debug, Clone)]
pub struct ActorNetwork {
    /// Hidden layer weights
    pub hidden_weights: Array2<f64>,
    /// Hidden layer biases
    pub hidden_bias: Array1<f64>,
    /// Output layer weights
    pub output_weights: Array2<f64>,
    /// Output layer biases
    pub output_bias: Array1<f64>,
    /// Network architecture
    pub _input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    /// Activation function type
    pub activation: ActivationType,
}

/// Critic network for value function estimation
#[derive(Debug, Clone)]
pub struct CriticNetwork {
    /// Hidden layer weights
    pub hidden_weights: Array2<f64>,
    /// Hidden layer biases
    pub hidden_bias: Array1<f64>,
    /// Output layer weights (single value output)
    pub output_weights: Array1<f64>,
    /// Output bias
    pub output_bias: f64,
    /// Network architecture
    pub _input_size: usize,
    pub hidden_size: usize,
    /// Activation function type
    pub activation: ActivationType,
}

/// Types of activation functions
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    Tanh,
    ReLU,
    Sigmoid,
    LeakyReLU,
    ELU,
}

impl ActivationType {
    fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationType::Tanh => x.tanh(),
            ActivationType::ReLU => x.max(0.0),
            ActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationType::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            ActivationType::ELU => {
                if x > 0.0 {
                    x
                } else {
                    x.exp() - 1.0
                }
            }
        }
    }

    fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationType::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            ActivationType::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationType::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            ActivationType::LeakyReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            ActivationType::ELU => {
                if x > 0.0 {
                    1.0
                } else {
                    x.exp()
                }
            }
        }
    }
}

impl ActorNetwork {
    /// Create new actor network
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        activation: ActivationType,
    ) -> Self {
        let xavier_scale = (2.0 / (input_size + hidden_size) as f64).sqrt();

        Self {
            hidden_weights: Array2::from_shape_fn((hidden_size, input_size), |_| {
                (rand::rng().random::<f64>() - 0.5) * 2.0 * xavier_scale
            }),
            hidden_bias: Array1::zeros(hidden_size),
            output_weights: Array2::from_shape_fn((output_size, hidden_size), |_| {
                (rand::rng().random::<f64>() - 0.5) * 2.0 * xavier_scale
            }),
            output_bias: Array1::zeros(output_size),
            _input_size: input_size,
            hidden_size,
            output_size,
            activation,
        }
    }

    /// Forward pass through actor network
    pub fn forward(&self, input: &ArrayView1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        // Hidden layer
        let mut hidden_raw = Array1::zeros(self.hidden_size);
        for i in 0..self.hidden_size {
            for j in 0..self._input_size.min(input.len()) {
                hidden_raw[i] += self.hidden_weights[[i, j]] * input[j];
            }
            hidden_raw[i] += self.hidden_bias[i];
        }

        let hidden_activated = hidden_raw.mapv(|x| self.activation.apply(x));

        // Output layer
        let mut output_raw = Array1::zeros(self.output_size);
        for i in 0..self.output_size {
            for j in 0..self.hidden_size {
                output_raw[i] += self.output_weights[[i, j]] * hidden_activated[j];
            }
            output_raw[i] += self.output_bias[i];
        }

        let output_activated = output_raw.mapv(|x| self.activation.apply(x));

        (hidden_raw, hidden_activated, output_activated)
    }

    /// Compute action probabilities with temperature scaling
    pub fn action_probabilities(
        &self,
        policy_output: &ArrayView1<f64>,
        temperature: f64,
    ) -> Array1<f64> {
        let scaled_output = policy_output.mapv(|x| x / temperature);
        let max_val = scaled_output.fold(-f64::INFINITY, |a, &b| a.max(b));
        let exp_output = scaled_output.mapv(|x| (x - max_val).exp());
        let sum_exp = exp_output.sum();

        if sum_exp > 0.0 {
            exp_output / sum_exp
        } else {
            Array1::from_elem(policy_output.len(), 1.0 / policy_output.len() as f64)
        }
    }

    /// Backward pass and update weights
    pub fn backward_and_update(
        &mut self,
        input: &ArrayView1<f64>,
        hidden_raw: &Array1<f64>,
        hidden_activated: &Array1<f64>,
        output_gradient: &ArrayView1<f64>,
        learning_rate: f64,
    ) {
        // Output layer gradients
        let output_raw_gradient = output_gradient.mapv(|g| g); // Assuming linear output

        // Hidden layer gradients
        let mut hidden_gradient: Array1<f64> = Array1::zeros(self.hidden_size);
        for j in 0..self.hidden_size {
            for i in 0..self.output_size {
                hidden_gradient[j] += output_raw_gradient[i] * self.output_weights[[i, j]];
            }
            hidden_gradient[j] *= self.activation.derivative(hidden_raw[j]);
        }

        // Update output layer weights and biases
        for i in 0..self.output_size {
            for j in 0..self.hidden_size {
                self.output_weights[[i, j]] -=
                    learning_rate * output_raw_gradient[i] * hidden_activated[j];
            }
            self.output_bias[i] -= learning_rate * output_raw_gradient[i];
        }

        // Update hidden layer weights and biases
        for i in 0..self.hidden_size {
            for j in 0..self._input_size.min(input.len()) {
                self.hidden_weights[[i, j]] -= learning_rate * hidden_gradient[i] * input[j];
            }
            self.hidden_bias[i] -= learning_rate * hidden_gradient[i];
        }
    }
}

impl CriticNetwork {
    /// Create new critic network
    pub fn new(input_size: usize, hidden_size: usize, activation: ActivationType) -> Self {
        let xavier_scale = (2.0 / (input_size + hidden_size) as f64).sqrt();

        Self {
            hidden_weights: Array2::from_shape_fn((hidden_size, input_size), |_| {
                (rand::rng().random::<f64>() - 0.5) * 2.0 * xavier_scale
            }),
            hidden_bias: Array1::zeros(hidden_size),
            output_weights: Array1::from_shape_fn(hidden_size, |_| {
                (rand::rng().random::<f64>() - 0.5) * 2.0 * xavier_scale
            }),
            output_bias: 0.0,
            _input_size: input_size,
            hidden_size,
            activation,
        }
    }

    /// Forward pass through critic network
    pub fn forward(&self, input: &ArrayView1<f64>) -> (Array1<f64>, Array1<f64>, f64) {
        // Hidden layer
        let mut hidden_raw = Array1::zeros(self.hidden_size);
        for i in 0..self.hidden_size {
            for j in 0..self._input_size.min(input.len()) {
                hidden_raw[i] += self.hidden_weights[[i, j]] * input[j];
            }
            hidden_raw[i] += self.hidden_bias[i];
        }

        let hidden_activated = hidden_raw.mapv(|x| self.activation.apply(x));

        // Output layer (single value)
        let mut output = 0.0;
        for j in 0..self.hidden_size {
            output += self.output_weights[j] * hidden_activated[j];
        }
        output += self.output_bias;

        (hidden_raw, hidden_activated, output)
    }

    /// Backward pass and update weights
    pub fn backward_and_update(
        &mut self,
        input: &ArrayView1<f64>,
        hidden_raw: &Array1<f64>,
        hidden_activated: &Array1<f64>,
        target_value: f64,
        predicted_value: f64,
        learning_rate: f64,
    ) {
        let value_error = target_value - predicted_value;

        // Hidden layer gradients
        let mut hidden_gradient: Array1<f64> = Array1::zeros(self.hidden_size);
        for j in 0..self.hidden_size {
            hidden_gradient[j] =
                value_error * self.output_weights[j] * self.activation.derivative(hidden_raw[j]);
        }

        // Update output layer weights and bias
        for j in 0..self.hidden_size {
            self.output_weights[j] += learning_rate * value_error * hidden_activated[j];
        }
        self.output_bias += learning_rate * value_error;

        // Update hidden layer weights and biases
        for i in 0..self.hidden_size {
            for j in 0..self._input_size.min(input.len()) {
                self.hidden_weights[[i, j]] += learning_rate * hidden_gradient[i] * input[j];
            }
            self.hidden_bias[i] += learning_rate * hidden_gradient[i];
        }
    }
}

/// Advantage Actor-Critic (A2C) optimizer
#[derive(Debug, Clone)]
pub struct AdvantageActorCriticOptimizer {
    /// Configuration
    config: RLOptimizationConfig,
    /// Actor network
    actor: ActorNetwork,
    /// Critic network
    critic: CriticNetwork,
    /// Experience buffer
    experience_buffer: ExperienceBuffer,
    /// Reward function
    reward_function: ImprovementReward,
    /// Best solution found
    best_params: Array1<f64>,
    /// Best objective value
    best_objective: f64,
    /// Temperature for exploration
    temperature: f64,
    /// Baseline for variance reduction
    baseline: f64,
    /// Training statistics
    training_stats: A2CTrainingStats,
    /// Entropy coefficient for exploration
    entropy_coeff: f64,
    /// Value function coefficient
    value_coeff: f64,
}

/// Training statistics for A2C
#[derive(Debug, Clone)]
pub struct A2CTrainingStats {
    /// Average actor loss
    pub avg_actor_loss: f64,
    /// Average critic loss
    pub avg_critic_loss: f64,
    /// Average advantage
    pub avg_advantage: f64,
    /// Average entropy
    pub avg_entropy: f64,
    /// Episodes completed
    pub episodes_completed: usize,
    /// Total steps
    pub total_steps: usize,
}

impl Default for A2CTrainingStats {
    fn default() -> Self {
        Self {
            avg_actor_loss: 0.0,
            avg_critic_loss: 0.0,
            avg_advantage: 0.0,
            avg_entropy: 0.0,
            episodes_completed: 0,
            total_steps: 0,
        }
    }
}

impl AdvantageActorCriticOptimizer {
    /// Create new A2C optimizer
    pub fn new(
        config: RLOptimizationConfig,
        state_size: usize,
        action_size: usize,
        hidden_size: usize,
    ) -> Self {
        let memory_size = config.memory_size;
        Self {
            config,
            actor: ActorNetwork::new(state_size, hidden_size, action_size, ActivationType::Tanh),
            critic: CriticNetwork::new(state_size, hidden_size, ActivationType::ReLU),
            experience_buffer: ExperienceBuffer::new(memory_size),
            reward_function: ImprovementReward::default(),
            best_params: Array1::zeros(state_size),
            best_objective: f64::INFINITY,
            temperature: 1.0,
            baseline: 0.0,
            training_stats: A2CTrainingStats::default(),
            entropy_coeff: 0.01,
            value_coeff: 0.5,
        }
    }

    /// Extract state features for networks
    fn extract_state_features(&self, state: &OptimizationState) -> Array1<f64> {
        let mut features = Vec::new();

        // Parameter values (normalized)
        for &param in state.parameters.iter() {
            features.push(param.tanh());
        }

        // Objective value (log-normalized)
        let log_obj = (state.objective_value.abs() + 1e-8).ln();
        features.push(log_obj.tanh());

        // Convergence metrics
        features.push(
            state
                .convergence_metrics
                .relative_objective_change
                .ln()
                .tanh(),
        );
        features.push(state.convergence_metrics.parameter_change_norm.tanh());
        features.push((state.convergence_metrics.steps_since_improvement as f64 / 10.0).tanh());

        // History features
        if state.objective_history.len() >= 2 {
            let recent_change = state.objective_history[state.objective_history.len() - 1]
                - state.objective_history[state.objective_history.len() - 2];
            features.push(recent_change.tanh());

            let trend = if state.objective_history.len() >= 3 {
                let slope = (state.objective_history[state.objective_history.len() - 1]
                    - state.objective_history[0])
                    / state.objective_history.len() as f64;
                slope.tanh()
            } else {
                0.0
            };
            features.push(trend);
        } else {
            features.push(0.0);
            features.push(0.0);
        }

        // Step information
        features.push((state.step as f64 / self.config.max_steps_per_episode as f64).tanh());

        Array1::from(features)
    }

    /// Select action using actor network with exploration
    fn select_action_with_exploration(
        &mut self,
        state: &OptimizationState,
    ) -> (OptimizationAction, Array1<f64>) {
        let state_features = self.extract_state_features(state);
        let (_, _, policy_output) = self.actor.forward(&state_features.view());

        // Add exploration noise
        let exploration_noise = if self.training_stats.episodes_completed
            < self.config.num_episodes / 2
        {
            0.1 * (1.0
                - self.training_stats.episodes_completed as f64 / self.config.num_episodes as f64)
        } else {
            0.01
        };

        let noisy_output =
            policy_output.mapv(|x| x + (rand::rng().random::<f64>() - 0.5) * exploration_noise);
        let action_probs = self
            .actor
            .action_probabilities(&noisy_output.view(), self.temperature);

        // Sample action based on probabilities
        let cumulative_probs: Vec<f64> = action_probs
            .iter()
            .scan(0.0, |acc, &p| {
                *acc += p;
                Some(*acc)
            })
            .collect();

        let rand_val = rand::rng().random::<f64>();
        let action_idx = cumulative_probs
            .iter()
            .position(|&cp| rand_val <= cp)
            .unwrap_or(action_probs.len() - 1);

        let action = self.decode_action_from_index(action_idx, &noisy_output);

        (action, action_probs)
    }

    /// Decode action from action index and policy output
    fn decode_action_from_index(
        &self,
        action_idx: usize,
        policy_output: &Array1<f64>,
    ) -> OptimizationAction {
        let magnitude_factor = 1.0 + policy_output.get(1).unwrap_or(&0.0).abs();

        match action_idx {
            0 => OptimizationAction::GradientStep {
                learning_rate: 0.001 * magnitude_factor,
            },
            1 => OptimizationAction::RandomPerturbation {
                magnitude: 0.01 * magnitude_factor,
            },
            2 => OptimizationAction::MomentumUpdate {
                momentum: 0.9 * (1.0 + policy_output.get(2).unwrap_or(&0.0) * 0.1),
            },
            3 => OptimizationAction::AdaptiveLearningRate {
                factor: 0.5 + 0.5 * policy_output.get(3).unwrap_or(&0.0).tanh(),
            },
            4 => OptimizationAction::ResetToBest,
            _ => OptimizationAction::Terminate,
        }
    }

    /// Compute advantage function
    fn compute_advantage(
        &self,
        reward: f64,
        current_value: f64,
        next_value: f64,
        done: bool,
    ) -> f64 {
        let target = if done {
            reward
        } else {
            reward + self.config.discount_factor * next_value
        };
        target - current_value
    }

    /// Update actor and critic networks using A2C algorithm
    fn update_networks(&mut self, experiences: &[Experience]) -> Result<(), OptimizeError> {
        let mut total_actor_loss = 0.0;
        let mut total_critic_loss = 0.0;
        let mut total_advantage = 0.0;
        let mut total_entropy = 0.0;

        for experience in experiences {
            let state_features = self.extract_state_features(&experience.state);
            let next_state_features = self.extract_state_features(&experience.next_state);

            // Forward pass through critic for current and next state
            let (hidden_raw, hidden_activated, current_value) =
                self.critic.forward(&state_features.view());
            let (_, _, next_value) = self.critic.forward(&next_state_features.view());

            // Compute advantage
            let advantage = self.compute_advantage(
                experience.reward,
                current_value,
                next_value,
                experience.done,
            );

            // Compute target value for critic
            let target_value = if experience.done {
                experience.reward
            } else {
                experience.reward + self.config.discount_factor * next_value
            };

            // Update critic network
            self.critic.backward_and_update(
                &state_features.view(),
                &hidden_raw,
                &hidden_activated,
                target_value,
                current_value,
                self.config.learning_rate * self.value_coeff,
            );

            // Forward pass through actor
            let (actor_hidden_raw, actor_hidden_activated, policy_output) =
                self.actor.forward(&state_features.view());

            let action_probs = self
                .actor
                .action_probabilities(&policy_output.view(), self.temperature);

            // Compute entropy for exploration
            let entropy = -action_probs
                .iter()
                .filter(|&&p| p > 1e-8)
                .map(|&p| p * p.ln())
                .sum::<f64>();

            // Compute policy gradient (simplified REINFORCE with baseline)
            let action_idx = self.get_action_index(&experience.action);
            let log_prob = action_probs.get(action_idx).unwrap_or(&1e-8).ln();
            let policy_loss = -log_prob * (advantage - self.baseline);

            // Actor gradient (simplified)
            let mut actor_gradient = Array1::zeros(policy_output.len());
            if action_idx < actor_gradient.len() {
                actor_gradient[action_idx] =
                    -(advantage - self.baseline) / (action_probs[action_idx] + 1e-8);
                // Add entropy bonus
                actor_gradient[action_idx] += self.entropy_coeff * (1.0 + log_prob);
            }

            // Update actor network
            self.actor.backward_and_update(
                &state_features.view(),
                &actor_hidden_raw,
                &actor_hidden_activated,
                &actor_gradient.view(),
                self.config.learning_rate,
            );

            // Update statistics
            total_actor_loss += policy_loss;
            total_critic_loss += (target_value - current_value).powi(2);
            total_advantage += advantage;
            total_entropy += entropy;
        }

        // Update baseline
        if !experiences.is_empty() {
            self.baseline =
                0.9 * self.baseline + 0.1 * (total_advantage / experiences.len() as f64);

            // Update training statistics
            let num_exp = experiences.len() as f64;
            self.training_stats.avg_actor_loss =
                0.9 * self.training_stats.avg_actor_loss + 0.1 * (total_actor_loss / num_exp);
            self.training_stats.avg_critic_loss =
                0.9 * self.training_stats.avg_critic_loss + 0.1 * (total_critic_loss / num_exp);
            self.training_stats.avg_advantage =
                0.9 * self.training_stats.avg_advantage + 0.1 * (total_advantage / num_exp);
            self.training_stats.avg_entropy =
                0.9 * self.training_stats.avg_entropy + 0.1 * (total_entropy / num_exp);
        }

        Ok(())
    }

    /// Get action index for gradient computation
    fn get_action_index(&self, action: &OptimizationAction) -> usize {
        match action {
            OptimizationAction::GradientStep { .. } => 0,
            OptimizationAction::RandomPerturbation { .. } => 1,
            OptimizationAction::MomentumUpdate { .. } => 2,
            OptimizationAction::AdaptiveLearningRate { .. } => 3,
            OptimizationAction::ResetToBest => 4,
            OptimizationAction::Terminate => 5,
        }
    }

    /// Get training statistics
    pub fn get_training_stats(&self) -> &A2CTrainingStats {
        &self.training_stats
    }

    /// Adjust exploration parameters
    fn adjust_exploration(&mut self) {
        // Decay temperature for exploration
        self.temperature = (self.temperature * 0.999).max(0.1);

        // Adjust entropy coefficient
        self.entropy_coeff = (self.entropy_coeff * 0.9995).max(0.001);
    }
}

impl RLOptimizer for AdvantageActorCriticOptimizer {
    fn config(&self) -> &RLOptimizationConfig {
        &self.config
    }

    fn select_action(&mut self, state: &OptimizationState) -> OptimizationAction {
        let (action, _) = self.select_action_with_exploration(state);
        action
    }

    fn update(&mut self, experience: &Experience) -> Result<(), OptimizeError> {
        self.experience_buffer.add(experience.clone());

        // Update networks when we have enough experiences
        if self.experience_buffer.size() >= self.config.batch_size {
            let batch = self.experience_buffer.sample_batch(self.config.batch_size);
            self.update_networks(&batch)?;
        }

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
        let mut current_params = initial_params.to_owned();
        let mut current_state = utils::create_state(current_params.clone(), objective, 0, None);
        let mut momentum = Array1::zeros(initial_params.len());
        let mut total_reward = 0.0;

        for step in 0..self.config.max_steps_per_episode {
            // Select action
            let (action, _) = self.select_action_with_exploration(&current_state);

            // Apply action
            let new_params =
                utils::apply_action(&current_state, &action, &self.best_params, &mut momentum);
            let new_state =
                utils::create_state(new_params, objective, step + 1, Some(&current_state));

            // Compute reward
            let reward = self
                .reward_function
                .compute_reward(&current_state, &action, &new_state);
            total_reward += reward;

            // Create experience
            let experience = Experience {
                state: current_state.clone(),
                action: action.clone(),
                reward,
                next_state: new_state.clone(),
                done: utils::should_terminate(&new_state, self.config.max_steps_per_episode),
            };

            // Update networks
            self.update(&experience)?;

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

        self.training_stats.episodes_completed += 1;
        self.training_stats.total_steps += current_state.step;

        // Adjust exploration parameters
        self.adjust_exploration();

        Ok(OptimizeResults::<f64> {
            x: current_params,
            fun: current_state.objective_value,
            success: current_state.convergence_metrics.relative_objective_change < 1e-6,
            nit: current_state.step,
            message: format!("A2C episode completed, total reward: {:.4}", total_reward),
            jac: None,
            hess: None,
            constr: None,
            nfev: current_state.step,
            njev: 0,
            nhev: 0,
            maxcv: 0,
            status: if current_state.convergence_metrics.relative_objective_change < 1e-6 {
                0
            } else {
                1
            },
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
            message: "Training not completed".to_string(),
            jac: None,
            hess: None,
            constr: None,
            nfev: 0,
            njev: 0,
            nhev: 0,
            maxcv: 0,
            status: 1, // Failure status by default
        };

        for episode in 0..self.config.num_episodes {
            let result = self.run_episode(objective, initial_params)?;

            if result.fun < best_result.fun {
                best_result = result;
            }

            // Periodic logging (every 100 episodes)
            if (episode + 1) % 100 == 0 {
                println!("Episode {}: Best objective = {:.6}, Avg advantage = {:.4}, Temperature = {:.4}",
                    episode + 1, best_result.fun, self.training_stats.avg_advantage, self.temperature);
            }
        }

        best_result.x = self.best_params.clone();
        best_result.fun = self.best_objective;
        best_result.message = format!(
            "A2C training completed: {} episodes, {} total steps, final best = {:.6}",
            self.training_stats.episodes_completed,
            self.training_stats.total_steps,
            self.best_objective
        );

        Ok(best_result)
    }

    fn reset(&mut self) {
        self.best_objective = f64::INFINITY;
        self.best_params.fill(0.0);
        self.training_stats = A2CTrainingStats::default();
        self.temperature = 1.0;
        self.baseline = 0.0;
        self.entropy_coeff = 0.01;
    }
}

/// Convenience function for Actor-Critic optimization
#[allow(dead_code)]
pub fn actor_critic_optimize<F>(
    objective: F,
    initial_params: &ArrayView1<f64>,
    config: Option<RLOptimizationConfig>,
    hidden_size: Option<usize>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let config = config.unwrap_or_default();
    let hidden_size = hidden_size.unwrap_or(64);
    let state_size = initial_params.len() + 8; // Additional features
    let action_size = 6; // Number of different action types

    let mut optimizer =
        AdvantageActorCriticOptimizer::new(config, state_size, action_size, hidden_size);
    optimizer.train(&objective, initial_params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_network_creation() {
        let actor = ActorNetwork::new(10, 20, 6, ActivationType::Tanh);
        assert_eq!(actor._input_size, 10);
        assert_eq!(actor.hidden_size, 20);
        assert_eq!(actor.output_size, 6);
    }

    #[test]
    fn test_critic_network_creation() {
        let critic = CriticNetwork::new(10, 20, ActivationType::ReLU);
        assert_eq!(critic._input_size, 10);
        assert_eq!(critic.hidden_size, 20);
    }

    #[test]
    fn test_actor_forward_pass() {
        let actor = ActorNetwork::new(5, 10, 3, ActivationType::Tanh);
        let input = Array1::from(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let (hidden_raw, hidden_activated, output) = actor.forward(&input.view());

        assert_eq!(hidden_raw.len(), 10);
        assert_eq!(hidden_activated.len(), 10);
        assert_eq!(output.len(), 3);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_critic_forward_pass() {
        let critic = CriticNetwork::new(5, 10, ActivationType::ReLU);
        let input = Array1::from(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let (hidden_raw, hidden_activated, value) = critic.forward(&input.view());

        assert_eq!(hidden_raw.len(), 10);
        assert_eq!(hidden_activated.len(), 10);
        assert!(value.is_finite());
    }

    #[test]
    fn test_activation_functions() {
        assert!((ActivationType::Tanh.apply(0.0) - 0.0).abs() < 1e-10);
        assert!((ActivationType::ReLU.apply(-1.0) - 0.0).abs() < 1e-10);
        assert!(ActivationType::ReLU.apply(1.0) == 1.0);
        assert!((ActivationType::Sigmoid.apply(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_action_probabilities() {
        let actor = ActorNetwork::new(3, 5, 4, ActivationType::Tanh);
        let output = Array1::from(vec![1.0, 2.0, 0.5, -1.0]);

        let probs = actor.action_probabilities(&output.view(), 1.0);

        assert_eq!(probs.len(), 4);
        assert!((probs.sum() - 1.0).abs() < 1e-6);
        assert!(probs.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_a2c_optimizer_creation() {
        let config = RLOptimizationConfig::default();
        let optimizer = AdvantageActorCriticOptimizer::new(config, 10, 6, 20);

        assert_eq!(optimizer.actor._input_size, 10);
        assert_eq!(optimizer.actor.output_size, 6);
        assert_eq!(optimizer.critic._input_size, 10);
    }

    #[test]
    fn test_advantage_computation() {
        let config = RLOptimizationConfig::default();
        let optimizer = AdvantageActorCriticOptimizer::new(config, 5, 6, 10);

        let advantage = optimizer.compute_advantage(1.0, 2.0, 3.0, false);
        let expected = 1.0 + 0.99 * 3.0 - 2.0; // reward + gamma * next_value - current_value

        assert!((advantage - expected).abs() < 1e-6);
    }

    #[test]
    fn test_action_index_mapping() {
        let config = RLOptimizationConfig::default();
        let optimizer = AdvantageActorCriticOptimizer::new(config, 5, 6, 10);

        let actions = vec![
            OptimizationAction::GradientStep {
                learning_rate: 0.01,
            },
            OptimizationAction::RandomPerturbation { magnitude: 0.1 },
            OptimizationAction::MomentumUpdate { momentum: 0.9 },
            OptimizationAction::AdaptiveLearningRate { factor: 0.5 },
            OptimizationAction::ResetToBest,
            OptimizationAction::Terminate,
        ];

        for (expected_idx, action) in actions.iter().enumerate() {
            assert_eq!(optimizer.get_action_index(action), expected_idx);
        }
    }

    #[test]
    fn test_basic_a2c_optimization() {
        let config = RLOptimizationConfig {
            num_episodes: 10,
            max_steps_per_episode: 20,
            learning_rate: 0.01,
            ..Default::default()
        };

        let objective = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
        let initial = Array1::from(vec![2.0, 2.0]);

        let result =
            actor_critic_optimize(objective, &initial.view(), Some(config), Some(16)).unwrap();

        // Should make some progress
        let initial_obj = objective(&initial.view());
        assert!(result.fun <= initial_obj);
        assert!(result.nit > 0);
    }
}

#[allow(dead_code)]
pub fn placeholder() {
    // Placeholder function to prevent unused module warnings
}
