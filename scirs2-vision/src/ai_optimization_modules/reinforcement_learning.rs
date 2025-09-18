//! Reinforcement learning optimization for computer vision pipelines
//!
//! This module implements reinforcement learning-based optimization techniques
//! for automatically tuning streaming processing parameters using Q-learning
//! and experience replay.

use crate::error::Result;
use scirs2_core::rng;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Reinforcement learning agent for parameter optimization
#[derive(Debug)]
pub struct RLParameterOptimizer {
    /// Q-table for state-action values
    q_table: HashMap<StateDiscrete, HashMap<ActionDiscrete, f64>>,
    /// Current state
    current_state: StateDiscrete,
    /// Learning parameters
    learning_params: RLLearningParams,
    /// Action space
    pub action_space: Vec<ActionDiscrete>,
    /// State space
    state_space: Vec<StateDiscrete>,
    /// Experience replay buffer
    experience_buffer: VecDeque<Experience>,
    /// Performance history
    performance_history: VecDeque<PerformanceMetric>,
}

/// Discrete state representation for Q-learning
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct StateDiscrete {
    /// Processing latency bucket (0-4: very low to very high)
    pub latency_bucket: usize,
    /// CPU usage bucket (0-4)
    pub cpu_bucket: usize,
    /// Memory usage bucket (0-4)
    pub memory_bucket: usize,
    /// Quality score bucket (0-4)
    pub quality_bucket: usize,
    /// Input complexity bucket (0-4)
    pub complexity_bucket: usize,
}

/// Discrete action representation
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ActionDiscrete {
    /// Parameter adjustment type
    pub param_type: ParameterType,
    /// Adjustment direction and magnitude
    pub adjustment: AdjustmentAction,
}

/// Types of parameters to optimize
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ParameterType {
    /// Gaussian blur sigma
    BlurSigma,
    /// Edge detection threshold
    EdgeThreshold,
    /// Thread count
    ThreadCount,
    /// Buffer size
    BufferSize,
    /// SIMD mode selection
    SimdMode,
    /// Processing quality level
    QualityLevel,
}

/// Parameter adjustment actions
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum AdjustmentAction {
    /// Decrease parameter significantly
    DecreaseLarge,
    /// Decrease parameter slightly
    DecreaseSmall,
    /// Keep parameter unchanged
    NoChange,
    /// Increase parameter slightly
    IncreaseSmall,
    /// Increase parameter significantly
    IncreaseLarge,
}

/// RL learning parameters
#[derive(Debug, Clone)]
pub struct RLLearningParams {
    /// Learning rate (alpha)
    pub learning_rate: f64,
    /// Discount factor (gamma)
    pub discount_factor: f64,
    /// Exploration rate (epsilon)
    pub epsilon: f64,
    /// Epsilon decay rate
    pub epsilon_decay: f64,
    /// Minimum epsilon
    pub epsilon_min: f64,
}

impl Default for RLLearningParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            discount_factor: 0.95,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
        }
    }
}

/// Experience tuple for replay learning
#[derive(Debug, Clone)]
pub struct Experience {
    /// State before action
    pub state: StateDiscrete,
    /// Action taken
    pub action: ActionDiscrete,
    /// Reward received
    pub reward: f64,
    /// Next state
    pub next_state: StateDiscrete,
    /// Episode finished flag
    pub done: bool,
}

/// Performance metric for reward calculation
#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    /// Processing latency in milliseconds
    pub latency: f64,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in MB
    pub memory_usage: f64,
    /// Quality score (0-1)
    pub quality_score: f64,
    /// Energy consumption estimate
    pub energy_consumption: f64,
    /// Timestamp
    pub timestamp: Instant,
}

impl Default for RLParameterOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl RLParameterOptimizer {
    /// Create a new RL parameter optimizer
    pub fn new() -> Self {
        let learning_params = RLLearningParams::default();
        let action_space = Self::create_action_space();
        let state_space = Self::create_state_space();

        Self {
            q_table: HashMap::new(),
            current_state: StateDiscrete::default(),
            learning_params,
            action_space,
            state_space,
            experience_buffer: VecDeque::with_capacity(10000),
            performance_history: VecDeque::with_capacity(1000),
        }
    }

    /// Create the action space
    fn create_action_space() -> Vec<ActionDiscrete> {
        let mut actions = Vec::new();

        let param_types = [
            ParameterType::BlurSigma,
            ParameterType::EdgeThreshold,
            ParameterType::ThreadCount,
            ParameterType::BufferSize,
            ParameterType::SimdMode,
            ParameterType::QualityLevel,
        ];

        let adjustments = [
            AdjustmentAction::DecreaseLarge,
            AdjustmentAction::DecreaseSmall,
            AdjustmentAction::NoChange,
            AdjustmentAction::IncreaseSmall,
            AdjustmentAction::IncreaseLarge,
        ];

        for param_type in &param_types {
            for adjustment in &adjustments {
                actions.push(ActionDiscrete {
                    param_type: param_type.clone(),
                    adjustment: adjustment.clone(),
                });
            }
        }

        actions
    }

    /// Create the state space
    fn create_state_space() -> Vec<StateDiscrete> {
        let mut states = Vec::new();

        // Create all combinations of state buckets
        for latency in 0..5 {
            for cpu in 0..5 {
                for memory in 0..5 {
                    for quality in 0..5 {
                        for complexity in 0..5 {
                            states.push(StateDiscrete {
                                latency_bucket: latency,
                                cpu_bucket: cpu,
                                memory_bucket: memory,
                                quality_bucket: quality,
                                complexity_bucket: complexity,
                            });
                        }
                    }
                }
            }
        }

        states
    }

    /// Select action using epsilon-greedy policy
    pub fn select_action(&mut self, state: &StateDiscrete) -> ActionDiscrete {
        let mut rng = rng();

        if rng.random_f64() < self.learning_params.epsilon {
            // Explore: random action
            let idx = rng.gen_range(0..self.action_space.len());
            self.action_space[idx].clone()
        } else {
            // Exploit: best known action
            self.get_best_action(state)
        }
    }

    /// Get the best action for a state
    fn get_best_action(&self, state: &StateDiscrete) -> ActionDiscrete {
        if let Some(action_values) = self.q_table.get(state) {
            action_values
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(action, _)| action.clone())
                .unwrap_or_else(|| self.action_space[0].clone())
        } else {
            self.action_space[0].clone()
        }
    }

    /// Update Q-values using Bellman equation
    pub fn update_q_values(&mut self, experience: Experience) {
        let alpha = self.learning_params.learning_rate;
        let gamma = self.learning_params.discount_factor;

        // Calculate max Q-value for next state first
        let max_next_q = if experience.done {
            0.0
        } else {
            self.q_table
                .get(&experience.next_state)
                .map(|action_values| {
                    *action_values
                        .values()
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(&0.0)
                })
                .unwrap_or(0.0)
        };

        // Get current Q-value and update it
        let current_q = self
            .q_table
            .entry(experience.state.clone())
            .or_default()
            .entry(experience.action.clone())
            .or_insert(0.0);

        // Update Q-value using Bellman equation
        *current_q += alpha * (experience.reward + gamma * max_next_q - *current_q);

        // Store experience in replay buffer
        self.experience_buffer.push_back(experience);
        if self.experience_buffer.len() > 10000 {
            self.experience_buffer.pop_front();
        }

        // Decay epsilon
        self.learning_params.epsilon = (self.learning_params.epsilon
            * self.learning_params.epsilon_decay)
            .max(self.learning_params.epsilon_min);
    }

    /// Convert continuous metrics to discrete state
    pub fn metrics_to_state(&self, metrics: &PerformanceMetric) -> StateDiscrete {
        StateDiscrete {
            latency_bucket: Self::bucket_value(metrics.latency, 0.0, 100.0, 5),
            cpu_bucket: Self::bucket_value(metrics.cpu_usage, 0.0, 100.0, 5),
            memory_bucket: Self::bucket_value(metrics.memory_usage, 0.0, 2000.0, 5),
            quality_bucket: Self::bucket_value(metrics.quality_score, 0.0, 1.0, 5),
            complexity_bucket: 2, // Simplified - would analyze frame complexity
        }
    }

    /// Bucket continuous value into discrete categories
    fn bucket_value(value: f64, min_val: f64, max_val: f64, numbuckets: usize) -> usize {
        let normalized = (value - min_val) / (max_val - min_val);
        let bucket = (normalized * numbuckets as f64).floor() as usize;
        bucket.min(numbuckets - 1)
    }

    /// Calculate reward from performance metrics
    pub fn calculate_reward(&self, metrics: &PerformanceMetric) -> f64 {
        // Multi-objective reward function
        let latency_reward = 1.0 - (metrics.latency / 100.0).min(1.0);
        let cpu_reward = 1.0 - (metrics.cpu_usage / 100.0);
        let memory_reward = 1.0 - (metrics.memory_usage / 2000.0).min(1.0);
        let quality_reward = metrics.quality_score;
        let energy_reward = 1.0 - (metrics.energy_consumption / 10.0).min(1.0);

        // Weighted combination
        0.3 * latency_reward
            + 0.2 * cpu_reward
            + 0.2 * memory_reward
            + 0.2 * quality_reward
            + 0.1 * energy_reward
    }

    /// Perform experience replay learning
    pub fn experience_replay(&mut self, batchsize: usize) {
        if self.experience_buffer.len() < batchsize {
            return;
        }

        let mut rng = rng();
        let sample_indices: Vec<usize> = (0..batchsize)
            .map(|_| rng.gen_range(0..self.experience_buffer.len()))
            .collect();

        for &idx in &sample_indices {
            if let Some(experience) = self.experience_buffer.get(idx) {
                self.update_q_values(experience.clone());
            }
        }
    }

    /// Initialize RL optimizer
    pub async fn initialize_rl_optimizer(&mut self) -> Result<()> {
        // Reset experience buffer
        self.experience_buffer.clear();

        // Reset Q-values to initial state
        self.q_table = HashMap::new();

        // Reset learning parameters to defaults
        self.learning_params = RLLearningParams::default();

        Ok(())
    }
}

impl Default for StateDiscrete {
    fn default() -> Self {
        Self {
            latency_bucket: 2,
            cpu_bucket: 2,
            memory_bucket: 2,
            quality_bucket: 2,
            complexity_bucket: 2,
        }
    }
}
