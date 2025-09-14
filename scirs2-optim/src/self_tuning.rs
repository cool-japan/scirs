//! Self-tuning optimization strategies
//!
//! This module provides adaptive optimization strategies that automatically
//! tune hyperparameters, select optimizers, and adjust configurations based
//! on training dynamics and problem characteristics.

#![allow(dead_code)]

use crate::error::Result;
use crate::optimizers::*;
use crate::schedulers::*;
use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use rand::Rng;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::{Duration, Instant};

/// Configuration for self-tuning optimization
#[derive(Debug, Clone)]
pub struct SelfTuningConfig {
    /// Window size for performance evaluation
    pub evaluation_window: usize,

    /// Minimum improvement threshold for parameter updates
    pub improvement_threshold: f64,

    /// Maximum number of optimizer switches per epoch
    pub max_switches_per_epoch: usize,

    /// Enable automatic learning rate adjustment
    pub auto_lr_adjustment: bool,

    /// Enable automatic optimizer selection
    pub auto_optimizer_selection: bool,

    /// Enable automatic batch size tuning
    pub auto_batch_size_tuning: bool,

    /// Warmup period before starting adaptations
    pub warmup_steps: usize,

    /// Exploration probability for optimizer selection
    pub exploration_rate: f64,

    /// Decay rate for exploration
    pub exploration_decay: f64,

    /// Performance metric to optimize
    pub target_metric: TargetMetric,
}

impl Default for SelfTuningConfig {
    fn default() -> Self {
        Self {
            evaluation_window: 100,
            improvement_threshold: 0.01,
            max_switches_per_epoch: 3,
            auto_lr_adjustment: true,
            auto_optimizer_selection: true,
            auto_batch_size_tuning: false,
            warmup_steps: 1000,
            exploration_rate: 0.1,
            exploration_decay: 0.99,
            target_metric: TargetMetric::Loss,
        }
    }
}

/// Target optimization metric
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TargetMetric {
    /// Minimize loss
    Loss,
    /// Maximize accuracy
    Accuracy,
    /// Minimize convergence time
    ConvergenceTime,
    /// Maximize training throughput
    Throughput,
    /// Custom metric (user-defined)
    Custom,
}

/// Performance statistics for tracking optimization progress
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Current loss value
    pub loss: f64,

    /// Current accuracy (if available)
    pub accuracy: Option<f64>,

    /// Gradient norm
    pub gradient_norm: f64,

    /// Training throughput (samples/second)
    pub throughput: f64,

    /// Memory usage (MB)
    pub memory_usage: f64,

    /// Wall clock time for this step
    pub step_time: Duration,

    /// Learning rate used
    pub learning_rate: f64,

    /// Optimizer type used
    pub optimizer_type: String,

    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Adaptive optimizer that automatically tunes hyperparameters
pub struct SelfTuningOptimizer<A: Float, D: Dimension> {
    /// Configuration
    config: SelfTuningConfig,

    /// Current active optimizer
    current_optimizer: Box<dyn OptimizerTrait<A, D>>,

    /// Available optimizer candidates
    optimizer_candidates: Vec<OptimizerCandidate<A, D>>,

    /// Performance history
    performance_history: VecDeque<PerformanceStats>,

    /// Hyperparameter search state
    search_state: HyperparameterSearchState,

    /// Learning rate scheduler
    lr_scheduler: Option<Box<dyn LearningRateScheduler<A>>>,

    /// Optimizer selection strategy
    selection_strategy: OptimizerSelectionStrategy,

    /// Current step count
    step_count: usize,

    /// Number of optimizer switches in current epoch
    switches_this_epoch: usize,

    /// Best performance seen so far
    best_performance: Option<f64>,

    /// Time of last adaptation
    last_adaptation_time: Instant,

    /// Exploration state for multi-armed bandit
    bandit_state: BanditState,
}

/// Optimizer candidate with its configuration
struct OptimizerCandidate<A: Float, D: Dimension> {
    /// Name/identifier
    name: String,

    /// Factory function to create the optimizer
    factory: Box<dyn Fn() -> Box<dyn OptimizerTrait<A, D>>>,

    /// Performance history for this optimizer
    performance_history: Vec<f64>,

    /// Usage count
    usage_count: usize,

    /// Average performance
    average_performance: f64,

    /// Confidence interval
    confidence_interval: (f64, f64),
}

/// Hyperparameter search state
#[derive(Debug)]
struct HyperparameterSearchState {
    /// Current learning rate
    learning_rate: f64,

    /// Learning rate search bounds
    lr_bounds: (f64, f64),

    /// Current batch size
    batch_size: usize,

    /// Batch size search bounds
    batch_size_bounds: (usize, usize),

    /// Number of search iterations performed
    search_iterations: usize,

    /// Best hyperparameters found
    best_hyperparameters: HashMap<String, f64>,

    /// Search algorithm state
    search_algorithm: SearchAlgorithm,
}

/// Hyperparameter search algorithms
#[derive(Debug)]
enum SearchAlgorithm {
    /// Random search
    Random {
        /// Random number generator seed
        seed: u64,
    },

    /// Bayesian optimization
    Bayesian {
        /// Gaussian process state
        gp_state: GaussianProcessState,
    },

    /// Grid search
    Grid {
        /// Current grid position
        position: Vec<usize>,
        /// Grid dimensions
        dimensions: Vec<usize>,
    },

    /// Successive halving (Hyperband)
    SuccessiveHalving {
        /// Current bracket
        bracket: usize,
        /// Configurations in current round
        configurations: Vec<HashMap<String, f64>>,
    },
}

/// Gaussian process state for Bayesian optimization
#[derive(Debug)]
struct GaussianProcessState {
    /// Observed points
    observed_points: Vec<Vec<f64>>,

    /// Observed values
    observed_values: Vec<f64>,

    /// Kernel hyperparameters
    kernel_params: Vec<f64>,

    /// Acquisition function type
    acquisition_function: AcquisitionFunction,
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug, Clone, Copy)]
enum AcquisitionFunction {
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
}

/// Optimizer selection strategies
#[derive(Debug, Clone)]
enum OptimizerSelectionStrategy {
    /// Multi-armed bandit approach
    MultiArmedBandit {
        /// Bandit algorithm type
        algorithm: BanditAlgorithm,
    },

    /// Performance-based selection
    PerformanceBased {
        /// Minimum performance difference for switching
        min_difference: f64,
    },

    /// Round-robin testing
    RoundRobin {
        /// Current optimizer index
        current_index: usize,
    },

    /// Meta-learning based selection
    MetaLearning {
        /// Problem characteristics
        problem_features: Vec<f64>,
        /// Learned optimizer mappings
        optimizer_mappings: HashMap<String, f64>,
    },
}

/// Multi-armed bandit algorithms
#[derive(Debug, Clone, Copy)]
enum BanditAlgorithm {
    EpsilonGreedy,
    UCB1,
    ThompsonSampling,
    LinUCB,
}

/// Multi-armed bandit state
#[derive(Debug)]
struct BanditState {
    /// Reward estimates for each optimizer
    reward_estimates: Vec<f64>,

    /// Confidence bounds
    confidence_bounds: Vec<f64>,

    /// Selection counts
    selection_counts: Vec<usize>,

    /// Total selections
    total_selections: usize,

    /// Exploration parameter
    exploration_param: f64,
}

/// Trait for optimizer implementations that can be used with self-tuning
pub trait OptimizerTrait<A: Float + ScalarOperand + Debug, D: Dimension>: Send + Sync {
    /// Get optimizer name
    fn name(&self) -> &str;

    /// Perform optimization step
    fn step(&mut self, params: &mut [Array<A, D>], grads: &[Array<A, D>]) -> Result<()>;

    /// Get current learning rate
    fn learning_rate(&self) -> A;

    /// Set learning rate
    fn set_learning_rate(&mut self, lr: A);

    /// Get optimizer state for serialization
    fn get_state(&self) -> HashMap<String, Vec<u8>>;

    /// Set optimizer state from serialization
    fn set_state(&mut self, state: HashMap<String, Vec<u8>>) -> Result<()>;

    /// Clone the optimizer
    fn clone_optimizer(&self) -> Box<dyn OptimizerTrait<A, D>>;
}

impl<
        A: Float + ScalarOperand + Debug + Send + Sync + 'static + num_traits::FromPrimitive,
        D: Dimension + 'static,
    > SelfTuningOptimizer<A, D>
{
    /// Create new self-tuning optimizer
    pub fn new(config: SelfTuningConfig) -> Result<Self> {
        let mut optimizer_candidates = Vec::new();

        // Add default optimizer candidates
        optimizer_candidates.push(OptimizerCandidate {
            name: "Adam".to_string(),
            factory: Box::new(|| Box::new(AdamOptimizerWrapper::new(0.001, 0.9, 0.999, 1e-8, 0.0))),
            performance_history: Vec::new(),
            usage_count: 0,
            average_performance: 0.0,
            confidence_interval: (0.0, 0.0),
        });

        optimizer_candidates.push(OptimizerCandidate {
            name: "SGD".to_string(),
            factory: Box::new(|| Box::new(SGDOptimizerWrapper::new(0.01, 0.9, 0.0, false))),
            performance_history: Vec::new(),
            usage_count: 0,
            average_performance: 0.0,
            confidence_interval: (0.0, 0.0),
        });

        optimizer_candidates.push(OptimizerCandidate {
            name: "AdamW".to_string(),
            factory: Box::new(|| {
                Box::new(AdamWOptimizerWrapper::new(0.001, 0.9, 0.999, 1e-8, 0.01))
            }),
            performance_history: Vec::new(),
            usage_count: 0,
            average_performance: 0.0,
            confidence_interval: (0.0, 0.0),
        });

        // Start with Adam as default
        let current_optimizer = (optimizer_candidates[0].factory)();

        let search_state = HyperparameterSearchState {
            learning_rate: 0.001,
            lr_bounds: (1e-6, 1.0),
            batch_size: 32,
            batch_size_bounds: (8, 512),
            search_iterations: 0,
            best_hyperparameters: HashMap::new(),
            search_algorithm: SearchAlgorithm::Random { seed: 42 },
        };

        let selection_strategy = OptimizerSelectionStrategy::MultiArmedBandit {
            algorithm: BanditAlgorithm::UCB1,
        };

        let bandit_state = BanditState {
            reward_estimates: vec![0.0; optimizer_candidates.len()],
            confidence_bounds: vec![1.0; optimizer_candidates.len()],
            selection_counts: vec![0; optimizer_candidates.len()],
            total_selections: 0,
            exploration_param: 2.0,
        };

        Ok(Self {
            config,
            current_optimizer,
            optimizer_candidates,
            performance_history: VecDeque::new(),
            search_state,
            lr_scheduler: None,
            selection_strategy,
            step_count: 0,
            switches_this_epoch: 0,
            best_performance: None,
            last_adaptation_time: Instant::now(),
            bandit_state,
        })
    }

    /// Add a custom optimizer candidate
    pub fn add_optimizer_candidate<F>(&mut self, name: String, factory: F)
    where
        F: Fn() -> Box<dyn OptimizerTrait<A, D>> + 'static,
    {
        self.optimizer_candidates.push(OptimizerCandidate {
            name,
            factory: Box::new(factory),
            performance_history: Vec::new(),
            usage_count: 0,
            average_performance: 0.0,
            confidence_interval: (0.0, 0.0),
        });

        // Update bandit state
        self.bandit_state.reward_estimates.push(0.0);
        self.bandit_state.confidence_bounds.push(1.0);
        self.bandit_state.selection_counts.push(0);
    }

    /// Perform optimization step with automatic tuning
    pub fn step(
        &mut self,
        params: &mut [Array<A, D>],
        grads: &[Array<A, D>],
        stats: PerformanceStats,
    ) -> Result<()> {
        self.step_count += 1;

        // Record performance
        self.performance_history.push_back(stats.clone());
        if self.performance_history.len() > self.config.evaluation_window {
            self.performance_history.pop_front();
        }

        // Perform optimization step
        self.current_optimizer.step(params, grads)?;

        // Self-tuning adaptations
        if self.step_count > self.config.warmup_steps {
            self.maybe_adapt_optimizer(&stats)?;
            self.maybe_adapt_learning_rate(&stats)?;
            self.maybe_adapt_hyperparameters(&stats)?;
        }

        // Update best performance
        let current_performance = self.extract_performance_metric(&stats);
        if let Some(performance) = current_performance {
            if self.best_performance.is_none()
                || self.is_better_performance(performance, self.best_performance.unwrap())
            {
                self.best_performance = Some(performance);
            }
        }

        Ok(())
    }

    /// Check if we should adapt the optimizer
    fn maybe_adapt_optimizer(&mut self, stats: &PerformanceStats) -> Result<()> {
        if !self.config.auto_optimizer_selection {
            return Ok(());
        }

        if self.switches_this_epoch >= self.config.max_switches_per_epoch {
            return Ok(());
        }

        let should_adapt = self.should_adapt_optimizer(stats);

        if should_adapt {
            self.adapt_optimizer(stats)?;
            self.switches_this_epoch += 1;
        }

        Ok(())
    }

    /// Determine if optimizer should be adapted
    fn should_adapt_optimizer(&self, stats: &PerformanceStats) -> bool {
        if self.performance_history.len() < self.config.evaluation_window / 2 {
            return false;
        }

        // Check for performance degradation or stagnation
        let recent_performance: Vec<f64> = self
            .performance_history
            .iter()
            .rev()
            .take(self.config.evaluation_window / 4)
            .filter_map(|s| self.extract_performance_metric(s))
            .collect();

        let older_performance: Vec<f64> = self
            .performance_history
            .iter()
            .rev()
            .skip(self.config.evaluation_window / 4)
            .take(self.config.evaluation_window / 4)
            .filter_map(|s| self.extract_performance_metric(s))
            .collect();

        if recent_performance.is_empty() || older_performance.is_empty() {
            return false;
        }

        let recent_avg = recent_performance.iter().sum::<f64>() / recent_performance.len() as f64;
        let older_avg = older_performance.iter().sum::<f64>() / older_performance.len() as f64;

        // Check for stagnation or degradation
        match self.config.target_metric {
            TargetMetric::Loss => {
                (recent_avg - older_avg).abs() < self.config.improvement_threshold
                    || recent_avg > older_avg
            }
            TargetMetric::Accuracy | TargetMetric::Throughput => {
                (recent_avg - older_avg).abs() < self.config.improvement_threshold
                    || recent_avg < older_avg
            }
            _ => false,
        }
    }

    /// Adapt the optimizer based on performance
    fn adapt_optimizer(&mut self, stats: &PerformanceStats) -> Result<()> {
        let new_optimizer_idx = match &self.selection_strategy {
            OptimizerSelectionStrategy::MultiArmedBandit { algorithm } => {
                self.select_optimizer_bandit(*algorithm)
            }
            OptimizerSelectionStrategy::PerformanceBased { .. } => {
                self.select_optimizer_performance_based()
            }
            OptimizerSelectionStrategy::RoundRobin { current_index } => {
                (*current_index + 1) % self.optimizer_candidates.len()
            }
            OptimizerSelectionStrategy::MetaLearning { .. } => {
                self.select_optimizer_meta_learning(stats)
            }
        };

        // Switch to new optimizer
        if new_optimizer_idx < self.optimizer_candidates.len() {
            let current_lr = self.current_optimizer.learning_rate();
            let current_state = self.current_optimizer.get_state();

            self.current_optimizer = (self.optimizer_candidates[new_optimizer_idx].factory)();
            self.current_optimizer.set_learning_rate(current_lr);

            // Try to transfer compatible state
            if let Err(_) = self.current_optimizer.set_state(current_state) {
                // State transfer failed, continue with fresh state
            }

            // Update usage statistics
            self.optimizer_candidates[new_optimizer_idx].usage_count += 1;
        }

        Ok(())
    }

    /// Select optimizer using multi-armed bandit
    fn select_optimizer_bandit(&mut self, algorithm: BanditAlgorithm) -> usize {
        match algorithm {
            BanditAlgorithm::UCB1 => self.select_ucb1(),
            BanditAlgorithm::EpsilonGreedy => self.select_epsilon_greedy(),
            BanditAlgorithm::ThompsonSampling => self.select_thompson_sampling(),
            BanditAlgorithm::LinUCB => self.select_linucb(),
        }
    }

    /// UCB1 optimizer selection
    fn select_ucb1(&self) -> usize {
        if self.bandit_state.total_selections == 0 {
            return 0;
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0;

        for (i, candidate) in self.optimizer_candidates.iter().enumerate() {
            let ucb_score = if self.bandit_state.selection_counts[i] == 0 {
                f64::INFINITY
            } else {
                let mean_reward = self.bandit_state.reward_estimates[i];
                let confidence = (self.bandit_state.exploration_param
                    * (self.bandit_state.total_selections as f64).ln()
                    / self.bandit_state.selection_counts[i] as f64)
                    .sqrt();
                mean_reward + confidence
            };

            if ucb_score > best_score {
                best_score = ucb_score;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Epsilon-greedy optimizer selection
    fn select_epsilon_greedy(&self) -> usize {
        let mut rng = scirs2_core::random::rng();

        if A::from(rng.random_f64()).unwrap() < A::from(self.config.exploration_rate).unwrap() {
            // Explore: random selection
            rng.gen_range(0..self.optimizer_candidates.len())
        } else {
            // Exploit: best performing optimizer
            self.bandit_state
                .reward_estimates
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }
    }

    /// Thompson sampling optimizer selection
    fn select_thompson_sampling(&self) -> usize {
        // Simplified Thompson sampling - in practice would use Beta distributions
        let mut rng = scirs2_core::random::rng();

        let mut best_sample = f64::NEG_INFINITY;
        let mut best_idx = 0;

        for (i, _) in self.optimizer_candidates.iter().enumerate() {
            let mean = self.bandit_state.reward_estimates[i];
            let std = self.bandit_state.confidence_bounds[i];
            let sample = rng.gen_range(mean - std..mean + std);

            if sample > best_sample {
                best_sample = sample;
                best_idx = i;
            }
        }

        best_idx
    }

    /// LinUCB optimizer selection (contextual bandit)
    fn select_linucb(&self) -> usize {
        // Simplified LinUCB - would use contextual features in practice
        self.select_ucb1()
    }

    /// Performance-based optimizer selection
    fn select_optimizer_performance_based(&self) -> usize {
        self.optimizer_candidates
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.average_performance
                    .partial_cmp(&b.1.average_performance)
                    .unwrap()
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Meta-learning based optimizer selection
    fn select_optimizer_meta_learning(&self, stats: &PerformanceStats) -> usize {
        // Simplified meta-learning - would use problem features in practice
        0
    }

    /// Adapt learning rate based on performance
    fn maybe_adapt_learning_rate(&mut self, stats: &PerformanceStats) -> Result<()> {
        if !self.config.auto_lr_adjustment {
            return Ok(());
        }

        // Simple adaptive learning rate based on gradient norm
        let current_lr = self.current_optimizer.learning_rate().to_f64().unwrap();
        let gradient_norm = stats.gradient_norm;

        let new_lr = if gradient_norm > 10.0 {
            // Large gradients - reduce learning rate
            current_lr * 0.9
        } else if gradient_norm < 0.1 {
            // Small gradients - increase learning rate
            current_lr * 1.1
        } else {
            current_lr
        };

        let clamped_lr = new_lr
            .max(self.search_state.lr_bounds.0)
            .min(self.search_state.lr_bounds.1);

        if (clamped_lr - current_lr).abs() > current_lr * 0.01 {
            self.current_optimizer
                .set_learning_rate(A::from(clamped_lr).unwrap());
            self.search_state.learning_rate = clamped_lr;
        }

        Ok(())
    }

    /// Adapt other hyperparameters
    fn maybe_adapt_hyperparameters(&mut self, stats: &PerformanceStats) -> Result<()> {
        // Placeholder for hyperparameter adaptation
        // Would implement Bayesian optimization, random search, etc.
        Ok(())
    }

    /// Extract performance metric from stats
    fn extract_performance_metric(&self, stats: &PerformanceStats) -> Option<f64> {
        match self.config.target_metric {
            TargetMetric::Loss => Some(stats.loss),
            TargetMetric::Accuracy => stats.accuracy,
            TargetMetric::Throughput => Some(stats.throughput),
            TargetMetric::ConvergenceTime => Some(stats.step_time.as_secs_f64()),
            TargetMetric::Custom => stats.custom_metrics.values().next().copied(),
        }
    }

    /// Check if performance is better
    fn is_better_performance(&self, new_perf: f64, oldperf: f64) -> bool {
        match self.config.target_metric {
            TargetMetric::Loss | TargetMetric::ConvergenceTime => new_perf < oldperf,
            TargetMetric::Accuracy | TargetMetric::Throughput => new_perf > oldperf,
            TargetMetric::Custom => new_perf > oldperf, // Assume higher is better for custom
        }
    }

    /// Reset epoch counters
    pub fn reset_epoch(&mut self) {
        self.switches_this_epoch = 0;
    }

    /// Get current optimizer information
    pub fn get_optimizer_info(&self) -> OptimizerInfo {
        OptimizerInfo {
            name: self.current_optimizer.name().to_string(),
            learning_rate: self.current_optimizer.learning_rate().to_f64().unwrap(),
            step_count: self.step_count,
            switches_this_epoch: self.switches_this_epoch,
            performance_window_size: self.performance_history.len(),
            best_performance: self.best_performance,
        }
    }

    /// Get optimization statistics
    pub fn get_statistics(&self) -> SelfTuningStatistics {
        let optimizer_usage: HashMap<String, usize> = self
            .optimizer_candidates
            .iter()
            .map(|c| (c.name.clone(), c.usage_count))
            .collect();

        SelfTuningStatistics {
            total_steps: self.step_count,
            total_optimizer_switches: self
                .optimizer_candidates
                .iter()
                .map(|c| c.usage_count)
                .sum(),
            optimizer_usage,
            current_learning_rate: self.search_state.learning_rate,
            average_step_time: self
                .performance_history
                .iter()
                .map(|s| s.step_time.as_secs_f64())
                .sum::<f64>()
                / self.performance_history.len().max(1) as f64,
            exploration_rate: self.config.exploration_rate,
        }
    }
}

/// Information about current optimizer state
#[derive(Debug, Clone)]
pub struct OptimizerInfo {
    pub name: String,
    pub learning_rate: f64,
    pub step_count: usize,
    pub switches_this_epoch: usize,
    pub performance_window_size: usize,
    pub best_performance: Option<f64>,
}

/// Statistics about self-tuning optimization
#[derive(Debug, Clone)]
pub struct SelfTuningStatistics {
    pub total_steps: usize,
    pub total_optimizer_switches: usize,
    pub optimizer_usage: HashMap<String, usize>,
    pub current_learning_rate: f64,
    pub average_step_time: f64,
    pub exploration_rate: f64,
}

// Wrapper implementations for existing optimizers
struct AdamOptimizerWrapper<A: Float + ScalarOperand + Debug, D: Dimension> {
    inner: crate::optimizers::Adam<A>,
    _phantom: std::marker::PhantomData<D>,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> AdamOptimizerWrapper<A, D> {
    fn new(_lr: f64, beta1: f64, beta2: f64, eps: f64, weightdecay: f64) -> Self {
        Self {
            inner: crate::optimizers::Adam::new_with_config(
                A::from(_lr).unwrap(),
                A::from(beta1).unwrap(),
                A::from(beta2).unwrap(),
                A::from(eps).unwrap(),
                A::from(weightdecay).unwrap(),
            ),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A: Float + ScalarOperand + Debug + Send + Sync + 'static, D: Dimension + 'static>
    OptimizerTrait<A, D> for AdamOptimizerWrapper<A, D>
{
    fn name(&self) -> &str {
        "Adam"
    }

    fn step(&mut self, params: &mut [Array<A, D>], grads: &[Array<A, D>]) -> Result<()> {
        if params.len() != grads.len() {
            return Err(crate::error::OptimError::InvalidParameter(
                "Mismatched number of parameters and gradients".to_string(),
            ));
        }

        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            let updated = self.inner.step(param, grad)?;
            *param = updated;
        }
        Ok(())
    }

    fn learning_rate(&self) -> A {
        self.inner.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: A) {
        <crate::optimizers::Adam<A> as crate::optimizers::Optimizer<A, D>>::set_learning_rate(
            &mut self.inner,
            lr,
        );
    }

    fn get_state(&self) -> HashMap<String, Vec<u8>> {
        HashMap::new()
    }

    fn set_state(&mut self, state: HashMap<String, Vec<u8>>) -> Result<()> {
        Ok(())
    }

    fn clone_optimizer(&self) -> Box<dyn OptimizerTrait<A, D>> {
        Box::new(AdamOptimizerWrapper {
            inner: self.inner.clone(),
            _phantom: std::marker::PhantomData,
        })
    }
}

struct SGDOptimizerWrapper<A: Float + ScalarOperand + Debug, D: Dimension> {
    inner: crate::optimizers::SGD<A>,
    _phantom: std::marker::PhantomData<D>,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> SGDOptimizerWrapper<A, D> {
    fn new(_lr: f64, momentum: f64, weightdecay: f64, nesterov: bool) -> Self {
        Self {
            inner: crate::optimizers::SGD::new_with_config(
                A::from(_lr).unwrap(),
                A::from(momentum).unwrap(),
                A::from(weightdecay).unwrap(),
            ),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A: Float + ScalarOperand + Debug + Send + Sync + 'static, D: Dimension + 'static>
    OptimizerTrait<A, D> for SGDOptimizerWrapper<A, D>
{
    fn name(&self) -> &str {
        "SGD"
    }

    fn step(&mut self, params: &mut [Array<A, D>], grads: &[Array<A, D>]) -> Result<()> {
        if params.len() != grads.len() {
            return Err(crate::error::OptimError::InvalidParameter(
                "Mismatched number of parameters and gradients".to_string(),
            ));
        }

        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            let updated = self.inner.step(param, grad)?;
            *param = updated;
        }
        Ok(())
    }

    fn learning_rate(&self) -> A {
        self.inner.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: A) {
        <crate::optimizers::SGD<A> as crate::optimizers::Optimizer<A, D>>::set_learning_rate(
            &mut self.inner,
            lr,
        );
    }

    fn get_state(&self) -> HashMap<String, Vec<u8>> {
        HashMap::new()
    }

    fn set_state(&mut self, state: HashMap<String, Vec<u8>>) -> Result<()> {
        Ok(())
    }

    fn clone_optimizer(&self) -> Box<dyn OptimizerTrait<A, D>> {
        Box::new(SGDOptimizerWrapper {
            inner: self.inner.clone(),
            _phantom: std::marker::PhantomData,
        })
    }
}

struct AdamWOptimizerWrapper<A: Float + ScalarOperand + Debug, D: Dimension> {
    inner: crate::optimizers::AdamW<A>,
    _phantom: std::marker::PhantomData<D>,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> AdamWOptimizerWrapper<A, D> {
    fn new(_lr: f64, beta1: f64, beta2: f64, eps: f64, weightdecay: f64) -> Self {
        Self {
            inner: crate::optimizers::AdamW::new_with_config(
                A::from(_lr).unwrap(),
                A::from(beta1).unwrap(),
                A::from(beta2).unwrap(),
                A::from(eps).unwrap(),
                A::from(weightdecay).unwrap(),
            ),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A: Float + ScalarOperand + Debug + Send + Sync + 'static, D: Dimension + 'static>
    OptimizerTrait<A, D> for AdamWOptimizerWrapper<A, D>
{
    fn name(&self) -> &str {
        "AdamW"
    }

    fn step(&mut self, params: &mut [Array<A, D>], grads: &[Array<A, D>]) -> Result<()> {
        if params.len() != grads.len() {
            return Err(crate::error::OptimError::InvalidParameter(
                "Mismatched number of parameters and gradients".to_string(),
            ));
        }

        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            let updated = self.inner.step(param, grad)?;
            *param = updated;
        }
        Ok(())
    }

    fn learning_rate(&self) -> A {
        self.inner.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: A) {
        <crate::optimizers::AdamW<A> as crate::optimizers::Optimizer<A, D>>::set_learning_rate(
            &mut self.inner,
            lr,
        );
    }

    fn get_state(&self) -> HashMap<String, Vec<u8>> {
        HashMap::new()
    }

    fn set_state(&mut self, state: HashMap<String, Vec<u8>>) -> Result<()> {
        Ok(())
    }

    fn clone_optimizer(&self) -> Box<dyn OptimizerTrait<A, D>> {
        Box::new(AdamWOptimizerWrapper {
            inner: self.inner.clone(),
            _phantom: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use std::time::Duration;

    #[test]
    fn test_self_tuning_config_default() {
        let config = SelfTuningConfig::default();
        assert_eq!(config.evaluation_window, 100);
        assert!(config.auto_lr_adjustment);
        assert!(config.auto_optimizer_selection);
    }

    #[test]
    fn test_self_tuning_optimizer_creation() {
        let config = SelfTuningConfig::default();
        let optimizer: Result<SelfTuningOptimizer<f64, ndarray::Ix1>> =
            SelfTuningOptimizer::new(config);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_performance_stats() {
        let stats = PerformanceStats {
            loss: 0.5,
            accuracy: Some(0.9),
            gradient_norm: 1.2,
            throughput: 100.0,
            memory_usage: 1024.0,
            step_time: Duration::from_millis(50),
            learning_rate: 0.001,
            optimizer_type: "Adam".to_string(),
            custom_metrics: HashMap::new(),
        };

        assert_eq!(stats.loss, 0.5);
        assert_eq!(stats.accuracy, Some(0.9));
    }

    #[test]
    fn test_optimizer_step() {
        let config = SelfTuningConfig::default();
        let mut optimizer: SelfTuningOptimizer<f64, ndarray::Ix1> =
            SelfTuningOptimizer::new(config).unwrap();

        let mut params = vec![Array1::zeros(10)];
        let grads = vec![Array1::ones(10)];

        let stats = PerformanceStats {
            loss: 1.0,
            accuracy: None,
            gradient_norm: 1.0,
            throughput: 50.0,
            memory_usage: 512.0,
            step_time: Duration::from_millis(10),
            learning_rate: 0.001,
            optimizer_type: "Adam".to_string(),
            custom_metrics: HashMap::new(),
        };

        let result = optimizer.step(&mut params, &grads, stats);
        assert!(result.is_ok());

        let info = optimizer.get_optimizer_info();
        assert_eq!(info.name, "Adam");
        assert_eq!(info.step_count, 1);
    }

    #[test]
    fn test_bandit_selection() {
        let config = SelfTuningConfig::default();
        let optimizer: SelfTuningOptimizer<f64, ndarray::Ix1> =
            SelfTuningOptimizer::new(config).unwrap();

        let selection = optimizer.select_ucb1();
        assert!(selection < optimizer.optimizer_candidates.len());
    }

    #[test]
    fn test_performance_metric_extraction() {
        let config = SelfTuningConfig {
            target_metric: TargetMetric::Loss,
            ..Default::default()
        };
        let optimizer: SelfTuningOptimizer<f64, ndarray::Ix1> =
            SelfTuningOptimizer::new(config).unwrap();

        let stats = PerformanceStats {
            loss: 0.8,
            accuracy: Some(0.85),
            gradient_norm: 1.1,
            throughput: 75.0,
            memory_usage: 800.0,
            step_time: Duration::from_millis(20),
            learning_rate: 0.001,
            optimizer_type: "Adam".to_string(),
            custom_metrics: HashMap::new(),
        };

        let metric = optimizer.extract_performance_metric(&stats);
        assert_eq!(metric, Some(0.8));
    }

    #[test]
    fn test_statistics() {
        let config = SelfTuningConfig::default();
        let optimizer: SelfTuningOptimizer<f64, ndarray::Ix1> =
            SelfTuningOptimizer::new(config).unwrap();

        let stats = optimizer.get_statistics();
        assert_eq!(stats.total_steps, 0);
        assert!(stats.optimizer_usage.contains_key("Adam"));
    }
}
