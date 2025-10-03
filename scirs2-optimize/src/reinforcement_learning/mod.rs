//! Reinforcement Learning Optimization Module
//!
//! This module implements optimization algorithms based on reinforcement learning
//! principles, where optimization strategies are learned through interaction
//! with the objective function environment.
//!
//! # Key Features
//!
//! - **Policy Gradient Optimization**: Learn optimization policies using policy gradients
//! - **Q-Learning for Optimization**: Value-based approach to optimization strategy learning
//! - **Actor-Critic Methods**: Combined policy and value learning for optimization
//! - **Bandit-based Optimization**: Multi-armed bandit approaches for hyperparameter tuning
//! - **Evolutionary Strategies**: Population-based RL optimization
//! - **Meta-Learning**: Learning to optimize across different problem classes
//!
//! # Applications
//!
//! - Automatic hyperparameter tuning
//! - Adaptive optimization algorithms
//! - Black-box optimization
//! - Neural architecture search
//! - AutoML optimization pipelines

use crate::error::OptimizeResult;
use crate::result::OptimizeResults;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::random::{rng, Rng};

pub mod actor_critic;
pub mod bandit_optimization;
pub mod evolutionary_strategies;
pub mod meta_learning;
pub mod policy_gradient;
pub mod q_learning_optimization;

#[allow(ambiguous_glob_reexports)]
pub use actor_critic::*;
#[allow(ambiguous_glob_reexports)]
pub use bandit_optimization::*;
#[allow(ambiguous_glob_reexports)]
pub use evolutionary_strategies::*;
#[allow(ambiguous_glob_reexports)]
pub use meta_learning::*;
#[allow(ambiguous_glob_reexports)]
pub use policy_gradient::*;
#[allow(ambiguous_glob_reexports)]
pub use q_learning_optimization::*;

/// Configuration for reinforcement learning optimization
#[derive(Debug, Clone)]
pub struct RLOptimizationConfig {
    /// Number of episodes for training
    pub num_episodes: usize,
    /// Maximum steps per episode
    pub max_steps_per_episode: usize,
    /// Learning rate for policy/value updates
    pub learning_rate: f64,
    /// Discount factor for future rewards
    pub discount_factor: f64,
    /// Exploration parameter (epsilon for epsilon-greedy)
    pub exploration_rate: f64,
    /// Decay rate for exploration
    pub exploration_decay: f64,
    /// Minimum exploration rate
    pub min_exploration_rate: f64,
    /// Batch size for experience replay
    pub batch_size: usize,
    /// Memory buffer size
    pub memory_size: usize,
    /// Whether to use experience replay
    pub use_experience_replay: bool,
}

impl Default for RLOptimizationConfig {
    fn default() -> Self {
        Self {
            num_episodes: 1000,
            max_steps_per_episode: 100,
            learning_rate: 0.001,
            discount_factor: 0.99,
            exploration_rate: 0.1,
            exploration_decay: 0.995,
            min_exploration_rate: 0.01,
            batch_size: 32,
            memory_size: 10000,
            use_experience_replay: true,
        }
    }
}

/// State representation for optimization RL
#[derive(Debug, Clone)]
pub struct OptimizationState {
    /// Current parameter values
    pub parameters: Array1<f64>,
    /// Current objective value
    pub objective_value: f64,
    /// Gradient information (if available)
    pub gradient: Option<Array1<f64>>,
    /// Step number in episode
    pub step: usize,
    /// History of recent objective values
    pub objective_history: Vec<f64>,
    /// Convergence indicators
    pub convergence_metrics: ConvergenceMetrics,
}

/// Convergence metrics for RL state
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    /// Relative change in objective
    pub relative_objective_change: f64,
    /// Gradient norm (if available)
    pub gradient_norm: Option<f64>,
    /// Parameter change norm
    pub parameter_change_norm: f64,
    /// Number of steps since last improvement
    pub steps_since_improvement: usize,
}

/// Action space for optimization RL
#[derive(Debug, Clone)]
pub enum OptimizationAction {
    /// Gradient-based step with learning rate
    GradientStep { learning_rate: f64 },
    /// Random perturbation with magnitude
    RandomPerturbation { magnitude: f64 },
    /// Momentum update with coefficient
    MomentumUpdate { momentum: f64 },
    /// Adaptive learning rate adjustment
    AdaptiveLearningRate { factor: f64 },
    /// Reset to best known solution
    ResetToBest,
    /// Early termination
    Terminate,
}

/// Experience tuple for RL
#[derive(Debug, Clone)]
pub struct Experience {
    /// State before action
    pub state: OptimizationState,
    /// Action taken
    pub action: OptimizationAction,
    /// Reward received
    pub reward: f64,
    /// Next state
    pub next_state: OptimizationState,
    /// Whether episode terminated
    pub done: bool,
}

/// Trait for RL-based optimizers
pub trait RLOptimizer {
    /// Configuration
    fn config(&self) -> &RLOptimizationConfig;

    /// Select action given current state
    fn select_action(&mut self, state: &OptimizationState) -> OptimizationAction;

    /// Update policy/value function based on experience
    fn update(&mut self, experience: &Experience) -> OptimizeResult<()>;

    /// Run optimization episode
    fn run_episode<F>(
        &mut self,
        objective: &F,
        initial_params: &ArrayView1<f64>,
    ) -> OptimizeResult<OptimizeResults<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64;

    /// Train the RL optimizer
    fn train<F>(
        &mut self,
        objective: &F,
        initial_params: &ArrayView1<f64>,
    ) -> OptimizeResult<OptimizeResults<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64;

    /// Reset optimizer state
    fn reset(&mut self);
}

/// Reward function for optimization RL
pub trait RewardFunction {
    /// Compute reward based on state transition
    fn compute_reward(
        &self,
        prev_state: &OptimizationState,
        action: &OptimizationAction,
        new_state: &OptimizationState,
    ) -> f64;
}

/// Simple improvement-based reward function
#[derive(Debug, Clone)]
pub struct ImprovementReward {
    /// Scaling factor for objective improvement
    pub improvement_scale: f64,
    /// Penalty for taking steps
    pub step_penalty: f64,
    /// Bonus for convergence
    pub convergence_bonus: f64,
}

impl Default for ImprovementReward {
    fn default() -> Self {
        Self {
            improvement_scale: 10.0,
            step_penalty: 0.01,
            convergence_bonus: 1.0,
        }
    }
}

impl RewardFunction for ImprovementReward {
    fn compute_reward(
        &self,
        prev_state: &OptimizationState,
        _action: &OptimizationAction,
        new_state: &OptimizationState,
    ) -> f64 {
        // Reward for objective improvement
        let improvement = prev_state.objective_value - new_state.objective_value;
        let improvement_reward = self.improvement_scale * improvement;

        // Penalty for taking steps (encourages efficiency)
        let step_penalty = -self.step_penalty;

        // Bonus for convergence
        let convergence_bonus = if new_state.convergence_metrics.relative_objective_change < 1e-6 {
            self.convergence_bonus
        } else {
            0.0
        };

        improvement_reward + step_penalty + convergence_bonus
    }
}

/// Experience replay buffer
#[derive(Debug, Clone)]
pub struct ExperienceBuffer {
    /// Buffer for experiences
    pub buffer: Vec<Experience>,
    /// Maximum buffer size
    pub max_size: usize,
    /// Current position (for circular buffer)
    pub position: usize,
}

impl ExperienceBuffer {
    /// Create new experience buffer
    pub fn new(max_size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(max_size),
            max_size,
            position: 0,
        }
    }

    /// Add experience to buffer
    pub fn add(&mut self, experience: Experience) {
        if self.buffer.len() < self.max_size {
            self.buffer.push(experience);
        } else {
            self.buffer[self.position] = experience;
            self.position = (self.position + 1) % self.max_size;
        }
    }

    /// Sample batch of experiences
    pub fn sample_batch(&self, batchsize: usize) -> Vec<Experience> {
        let mut batch = Vec::with_capacity(batchsize);
        for _ in 0..batchsize.min(self.buffer.len()) {
            let idx = scirs2_core::random::rng().random_range(0..self.buffer.len());
            batch.push(self.buffer[idx].clone());
        }
        batch
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.buffer.len()
    }
}

/// Utility functions for RL optimization
pub mod utils {
    use super::*;

    /// Create optimization state from parameters and objective
    pub fn create_state<F>(
        parameters: Array1<f64>,
        objective: &F,
        step: usize,
        prev_state: Option<&OptimizationState>,
    ) -> OptimizationState
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let objective_value = objective(&parameters.view());

        // Compute convergence metrics
        let convergence_metrics = if let Some(prev) = prev_state {
            let relative_change = (prev.objective_value - objective_value).abs()
                / (prev.objective_value.abs() + 1e-12);

            // Ensure parameter arrays have the same shape before computing difference
            let param_change = if parameters.len() == prev.parameters.len() {
                (&parameters - &prev.parameters)
                    .mapv(|x| x * x)
                    .sum()
                    .sqrt()
            } else {
                // If shapes don't match, use parameter norm as fallback
                parameters.mapv(|x| x * x).sum().sqrt()
            };
            let steps_since_improvement = if objective_value < prev.objective_value {
                0
            } else {
                prev.convergence_metrics.steps_since_improvement + 1
            };

            ConvergenceMetrics {
                relative_objective_change: relative_change,
                gradient_norm: None,
                parameter_change_norm: param_change,
                steps_since_improvement,
            }
        } else {
            ConvergenceMetrics {
                relative_objective_change: f64::INFINITY,
                gradient_norm: None,
                parameter_change_norm: 0.0,
                steps_since_improvement: 0,
            }
        };

        // Update objective history
        let mut objective_history = prev_state
            .map(|s| s.objective_history.clone())
            .unwrap_or_default();
        objective_history.push(objective_value);
        if objective_history.len() > 10 {
            objective_history.remove(0);
        }

        OptimizationState {
            parameters,
            objective_value,
            gradient: None, // Would be computed if needed
            step,
            objective_history,
            convergence_metrics,
        }
    }

    /// Apply action to current state
    pub fn apply_action(
        state: &OptimizationState,
        action: &OptimizationAction,
        best_params: &Array1<f64>,
        momentum: &mut Array1<f64>,
    ) -> Array1<f64> {
        match action {
            OptimizationAction::GradientStep { learning_rate } => {
                // Simplified: use finite difference gradient
                let mut new_params = state.parameters.clone();

                // Random direction as proxy for gradient
                for i in 0..new_params.len() {
                    let step = (scirs2_core::random::rng().random::<f64>() - 0.5) * learning_rate;
                    new_params[i] += step;
                }

                new_params
            }
            OptimizationAction::RandomPerturbation { magnitude } => {
                let mut new_params = state.parameters.clone();
                for i in 0..new_params.len() {
                    let perturbation =
                        (scirs2_core::random::rng().random::<f64>() - 0.5) * 2.0 * magnitude;
                    new_params[i] += perturbation;
                }
                new_params
            }
            OptimizationAction::MomentumUpdate {
                momentum: momentum_coeff,
            } => {
                // Ensure momentum has the same shape as parameters
                if momentum.len() != state.parameters.len() {
                    *momentum = Array1::zeros(state.parameters.len());
                }

                // Update momentum (simplified)
                for i in 0..momentum.len().min(state.parameters.len()) {
                    let gradient_estimate =
                        (scirs2_core::random::rng().random::<f64>() - 0.5) * 0.1;
                    momentum[i] =
                        momentum_coeff * momentum[i] + (1.0 - momentum_coeff) * gradient_estimate;
                }

                &state.parameters - &*momentum
            }
            OptimizationAction::AdaptiveLearningRate { factor: _factor } => {
                // Adaptive step (simplified)
                let step_size = 0.01 / (1.0 + state.step as f64 * 0.01);
                let direction = Array1::from(vec![step_size; state.parameters.len()]);
                &state.parameters - &direction
            }
            OptimizationAction::ResetToBest => best_params.clone(),
            OptimizationAction::Terminate => state.parameters.clone(),
        }
    }

    /// Check if optimization should terminate
    pub fn should_terminate(state: &OptimizationState, max_steps: usize) -> bool {
        state.step >= max_steps
            || state.convergence_metrics.relative_objective_change < 1e-8
            || state.convergence_metrics.steps_since_improvement > 50
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_state_creation() {
        let params = Array1::from(vec![1.0, 2.0]);
        let objective = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);

        let state = utils::create_state(params, &objective, 0, None);

        assert_eq!(state.parameters.len(), 2);
        assert_eq!(state.objective_value, 5.0);
        assert_eq!(state.step, 0);
    }

    #[test]
    fn test_experience_buffer() {
        let mut buffer = ExperienceBuffer::new(5);

        let params = Array1::from(vec![1.0]);
        let objective = |x: &ArrayView1<f64>| x[0].powi(2);
        let state = utils::create_state(params.clone(), &objective, 0, None);

        let experience = Experience {
            state: state.clone(),
            action: OptimizationAction::GradientStep {
                learning_rate: 0.01,
            },
            reward: 1.0,
            next_state: state,
            done: false,
        };

        buffer.add(experience);
        assert_eq!(buffer.size(), 1);

        let batch = buffer.sample_batch(1);
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_improvement_reward() {
        let reward_fn = ImprovementReward::default();

        let params1 = Array1::from(vec![2.0]);
        let params2 = Array1::from(vec![1.0]);
        let objective = |x: &ArrayView1<f64>| x[0].powi(2);

        let state1 = utils::create_state(params1, &objective, 0, None);
        let state2 = utils::create_state(params2, &objective, 1, Some(&state1));

        let action = OptimizationAction::GradientStep { learning_rate: 0.1 };
        let reward = reward_fn.compute_reward(&state1, &action, &state2);

        // Should get positive reward for improvement (4.0 -> 1.0)
        assert!(reward > 0.0);
    }

    #[test]
    fn test_action_application() {
        let params = Array1::from(vec![1.0, 2.0]);
        let objective = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
        let state = utils::create_state(params.clone(), &objective, 0, None);
        let mut momentum = Array1::zeros(2);

        let action = OptimizationAction::RandomPerturbation { magnitude: 0.1 };
        let new_params = utils::apply_action(&state, &action, &params, &mut momentum);

        assert_eq!(new_params.len(), 2);
        // Parameters should have changed due to perturbation
        assert!(new_params != state.parameters);
    }

    #[test]
    fn test_termination_condition() {
        let params = Array1::from(vec![1.0]);
        let objective = |x: &ArrayView1<f64>| x[0].powi(2);
        let state = utils::create_state(params, &objective, 100, None);

        // Should terminate due to max steps
        assert!(utils::should_terminate(&state, 50));
    }

    #[test]
    fn test_convergence_metrics() {
        let params1 = Array1::from(vec![2.0]);
        let params2 = Array1::from(vec![1.9]);
        let objective = |x: &ArrayView1<f64>| x[0].powi(2);

        let state1 = utils::create_state(params1, &objective, 0, None);
        let state2 = utils::create_state(params2, &objective, 1, Some(&state1));

        assert!(state2.convergence_metrics.relative_objective_change > 0.0);
        assert!(state2.convergence_metrics.parameter_change_norm > 0.0);
        assert_eq!(state2.convergence_metrics.steps_since_improvement, 0); // Improvement occurred
    }
}
