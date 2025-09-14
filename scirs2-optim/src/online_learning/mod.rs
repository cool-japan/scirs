//! Online learning and lifelong optimization
//!
//! This module provides optimization strategies for continuous learning scenarios,
//! including online learning, continual learning, and lifelong optimization systems.

use crate::error::{OptimError, Result};
use ndarray::{Array, Array1, Dimension, ScalarOperand};
use num_traits::Float;
use rand::Rng;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

/// Online learning strategy
#[derive(Debug, Clone)]
pub enum OnlineLearningStrategy {
    /// Stochastic Gradient Descent with adaptive learning rate
    AdaptiveSGD {
        /// Initial learning rate
        initial_lr: f64,
        /// Learning rate adaptation method
        adaptation_method: LearningRateAdaptation,
    },
    /// Online Newton's method (second-order)
    OnlineNewton {
        /// Damping parameter for stability
        damping: f64,
        /// Window size for Hessian estimation
        hessian_window: usize,
    },
    /// Follow The Regularized Leader (FTRL)
    FTRL {
        /// L1 regularization strength
        l1_regularization: f64,
        /// L2 regularization strength
        l2_regularization: f64,
        /// Learning rate power
        learning_rate_power: f64,
    },
    /// Online Mirror Descent
    MirrorDescent {
        /// Mirror function type
        mirror_function: MirrorFunction,
        /// Regularization strength
        regularization: f64,
    },
    /// Adaptive Multi-Task Learning
    AdaptiveMultiTask {
        /// Task similarity threshold
        similarity_threshold: f64,
        /// Task-specific learning rates
        task_lr_adaptation: bool,
    },
}

/// Learning rate adaptation methods for online learning
#[derive(Debug, Clone)]
pub enum LearningRateAdaptation {
    /// AdaGrad-style adaptation
    AdaGrad {
        /// Small constant for numerical stability
        epsilon: f64,
    },
    /// RMSprop-style adaptation
    RMSprop {
        /// Decay rate
        decay: f64,
        /// Small constant for numerical stability
        epsilon: f64,
    },
    /// Adam-style adaptation
    Adam {
        /// Exponential decay rate for first moment
        beta1: f64,
        /// Exponential decay rate for second moment
        beta2: f64,
        /// Small constant for numerical stability
        epsilon: f64,
    },
    /// Exponential decay
    ExponentialDecay {
        /// Decay rate
        decay_rate: f64,
    },
    /// Inverse scaling
    InverseScaling {
        /// Scaling power
        power: f64,
    },
}

/// Mirror functions for mirror descent
#[derive(Debug, Clone)]
pub enum MirrorFunction {
    /// Euclidean (L2) regularization
    Euclidean,
    /// Entropy regularization (for probability simplex)
    Entropy,
    /// L1 regularization
    L1,
    /// Nuclear norm (for matrices)
    Nuclear,
}

/// Lifelong learning strategy
#[derive(Debug, Clone)]
pub enum LifelongStrategy {
    /// Elastic Weight Consolidation (EWC)
    ElasticWeightConsolidation {
        /// Importance weight for previous tasks
        importance_weight: f64,
        /// Fisher information estimation samples
        fisher_samples: usize,
    },
    /// Progressive Neural Networks
    ProgressiveNetworks {
        /// Lateral connection strength
        lateral_strength: f64,
        /// Column growth strategy
        growth_strategy: ColumnGrowthStrategy,
    },
    /// Memory-Augmented Networks
    MemoryAugmented {
        /// Memory size
        memory_size: usize,
        /// Memory update strategy
        update_strategy: MemoryUpdateStrategy,
    },
    /// Meta-Learning Based Continual Learning
    MetaLearning {
        /// Meta-learning rate
        meta_lr: f64,
        /// Inner loop steps
        inner_steps: usize,
        /// Task embedding size
        task_embedding_size: usize,
    },
    /// Gradient Episodic Memory (GEM)
    GradientEpisodicMemory {
        /// Memory buffer size per task
        memory_per_task: usize,
        /// Constraint violation tolerance
        violation_tolerance: f64,
    },
}

/// Column growth strategies for progressive networks
#[derive(Debug, Clone)]
pub enum ColumnGrowthStrategy {
    /// Add new column for each task
    PerTask,
    /// Add new column when performance drops
    PerformanceBased {
        /// Performance threshold
        threshold: f64,
    },
    /// Add new column after fixed intervals
    FixedInterval {
        /// Fixed interval
        interval: usize,
    },
}

/// Memory update strategies
#[derive(Debug, Clone)]
pub enum MemoryUpdateStrategy {
    /// First In First Out
    FIFO,
    /// Random replacement
    Random,
    /// Importance-based replacement
    ImportanceBased,
    /// Gradient diversity
    GradientDiversity,
}

/// Online optimizer that adapts to streaming data
#[derive(Debug)]
pub struct OnlineOptimizer<A: Float, D: Dimension> {
    /// Online learning strategy
    strategy: OnlineLearningStrategy,
    /// Current parameters
    parameters: Array<A, D>,
    /// Accumulated gradients for adaptation
    gradient_accumulator: Array<A, D>,
    /// Second moment accumulator (for Adam-like methods)
    second_moment_accumulator: Option<Array<A, D>>,
    /// Current learning rate
    current_lr: A,
    /// Step counter
    step_count: usize,
    /// Performance history
    performance_history: VecDeque<A>,
    /// Regret bounds tracking
    regret_bound: A,
}

/// Lifelong optimizer that learns continuously across tasks
#[derive(Debug)]
pub struct LifelongOptimizer<A: Float, D: Dimension> {
    /// Lifelong learning strategy
    strategy: LifelongStrategy,
    /// Task-specific optimizers
    task_optimizers: HashMap<String, OnlineOptimizer<A, D>>,
    /// Shared knowledge across tasks
    #[allow(dead_code)]
    shared_knowledge: SharedKnowledge<A, D>,
    /// Task sequence and relationships
    task_graph: TaskGraph,
    /// Memory buffer for important examples
    memory_buffer: MemoryBuffer<A, D>,
    /// Current active task
    current_task: Option<String>,
    /// Performance tracking across tasks
    task_performance: HashMap<String, Vec<A>>,
}

/// Shared knowledge representation for lifelong learning
#[derive(Debug)]
pub struct SharedKnowledge<A: Float, D: Dimension> {
    /// Fisher Information Matrix (for EWC)
    #[allow(dead_code)]
    fisher_information: Option<Array<A, D>>,
    /// Important parameters (for EWC)
    #[allow(dead_code)]
    important_parameters: Option<Array<A, D>>,
    /// Task embeddings
    #[allow(dead_code)]
    task_embeddings: HashMap<String, Array1<A>>,
    /// Cross-task transfer weights
    #[allow(dead_code)]
    transfer_weights: HashMap<(String, String), A>,
    /// Meta-parameters learned across tasks
    #[allow(dead_code)]
    meta_parameters: Option<Array1<A>>,
}

/// Task relationship graph
#[derive(Debug)]
pub struct TaskGraph {
    /// Task relationships (similarity scores)
    task_similarities: HashMap<(String, String), f64>,
    /// Task dependencies
    #[allow(dead_code)]
    task_dependencies: HashMap<String, Vec<String>>,
    /// Task categories/clusters
    #[allow(dead_code)]
    task_clusters: HashMap<String, String>,
}

/// Memory buffer for important examples
#[derive(Debug)]
pub struct MemoryBuffer<A: Float, D: Dimension> {
    /// Stored examples
    examples: VecDeque<MemoryExample<A, D>>,
    /// Maximum buffer size
    max_size: usize,
    /// Update strategy
    update_strategy: MemoryUpdateStrategy,
    /// Importance scores
    importance_scores: VecDeque<A>,
}

/// Single memory example
#[derive(Debug, Clone)]
pub struct MemoryExample<A: Float, D: Dimension> {
    /// Input data
    pub input: Array<A, D>,
    /// Target output
    pub target: Array<A, D>,
    /// Task identifier
    pub task_id: String,
    /// Importance score
    pub importance: A,
    /// Gradient information
    pub gradient: Option<Array<A, D>>,
}

/// Online learning performance metrics
#[derive(Debug, Clone)]
pub struct OnlinePerformanceMetrics<A: Float> {
    /// Cumulative regret
    pub cumulative_regret: A,
    /// Average loss over window
    pub average_loss: A,
    /// Learning rate stability
    pub lr_stability: A,
    /// Adaptation speed
    pub adaptation_speed: A,
    /// Memory efficiency
    pub memory_efficiency: A,
}

impl<A: Float + ScalarOperand + Debug + std::iter::Sum, D: Dimension> OnlineOptimizer<A, D> {
    /// Create a new online optimizer
    pub fn new(strategy: OnlineLearningStrategy, initial_parameters: Array<A, D>) -> Self {
        let paramshape = initial_parameters.raw_dim();
        let gradient_accumulator = Array::zeros(paramshape.clone());
        let second_moment_accumulator = match &strategy {
            OnlineLearningStrategy::AdaptiveSGD {
                adaptation_method: LearningRateAdaptation::Adam { .. },
                ..
            } => Some(Array::zeros(paramshape)),
            _ => None,
        };

        let current_lr = match &strategy {
            OnlineLearningStrategy::AdaptiveSGD { initial_lr, .. } => A::from(*initial_lr).unwrap(),
            OnlineLearningStrategy::OnlineNewton { .. } => A::from(0.01).unwrap(),
            OnlineLearningStrategy::FTRL { .. } => A::from(0.1).unwrap(),
            OnlineLearningStrategy::MirrorDescent { .. } => A::from(0.01).unwrap(),
            OnlineLearningStrategy::AdaptiveMultiTask { .. } => A::from(0.001).unwrap(),
        };

        Self {
            strategy,
            parameters: initial_parameters,
            gradient_accumulator,
            second_moment_accumulator,
            current_lr,
            step_count: 0,
            performance_history: VecDeque::new(),
            regret_bound: A::zero(),
        }
    }

    /// Perform online update with new gradient
    pub fn online_update(&mut self, gradient: &Array<A, D>, loss: A) -> Result<()> {
        self.step_count += 1;
        self.performance_history.push_back(loss);

        // Keep performance history bounded
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        match self.strategy.clone() {
            OnlineLearningStrategy::AdaptiveSGD {
                adaptation_method, ..
            } => {
                self.adaptive_sgd_update(gradient, &adaptation_method)?;
            }
            OnlineLearningStrategy::OnlineNewton { damping, .. } => {
                self.online_newton_update(gradient, damping)?;
            }
            OnlineLearningStrategy::FTRL {
                l1_regularization,
                l2_regularization,
                learning_rate_power,
            } => {
                self.ftrl_update(
                    gradient,
                    l1_regularization,
                    l2_regularization,
                    learning_rate_power,
                )?;
            }
            OnlineLearningStrategy::MirrorDescent {
                mirror_function,
                regularization,
            } => {
                self.mirror_descent_update(gradient, &mirror_function, regularization)?;
            }
            OnlineLearningStrategy::AdaptiveMultiTask { .. } => {
                self.adaptive_multitask_update(gradient)?;
            }
        }

        // Update regret bound
        self.update_regret_bound(loss);

        Ok(())
    }

    /// Adaptive SGD update
    fn adaptive_sgd_update(
        &mut self,
        gradient: &Array<A, D>,
        adaptation: &LearningRateAdaptation,
    ) -> Result<()> {
        match adaptation {
            LearningRateAdaptation::AdaGrad { epsilon } => {
                // Accumulate squared gradients
                self.gradient_accumulator = &self.gradient_accumulator + &gradient.mapv(|g| g * g);

                // Compute adaptive learning rate
                let adaptive_lr = self
                    .gradient_accumulator
                    .mapv(|acc| A::from(*epsilon).unwrap() + A::sqrt(acc));

                // Update parameters
                self.parameters = &self.parameters - &(gradient / &adaptive_lr * self.current_lr);
            }
            LearningRateAdaptation::RMSprop { decay, epsilon } => {
                let decay_factor = A::from(*decay).unwrap();
                let one_minus_decay = A::one() - decay_factor;

                // Update moving average of squared gradients
                self.gradient_accumulator = &self.gradient_accumulator * decay_factor
                    + &gradient.mapv(|g| g * g * one_minus_decay);

                // Compute adaptive learning rate
                let adaptive_lr = self
                    .gradient_accumulator
                    .mapv(|acc| A::sqrt(acc + A::from(*epsilon).unwrap()));

                // Update parameters
                self.parameters = &self.parameters - &(gradient / &adaptive_lr * self.current_lr);
            }
            LearningRateAdaptation::Adam {
                beta1,
                beta2,
                epsilon,
            } => {
                let beta1_val = A::from(*beta1).unwrap();
                let beta2_val = A::from(*beta2).unwrap();
                let one_minus_beta1 = A::one() - beta1_val;
                let one_minus_beta2 = A::one() - beta2_val;

                // Update first moment (gradient accumulator)
                self.gradient_accumulator =
                    &self.gradient_accumulator * beta1_val + gradient * one_minus_beta1;

                // Update second moment
                if let Some(ref mut second_moment) = self.second_moment_accumulator {
                    *second_moment =
                        &*second_moment * beta2_val + &gradient.mapv(|g| g * g * one_minus_beta2);

                    // Bias correction
                    let step_count_float = A::from(self.step_count).unwrap();
                    let bias_correction1 = A::one() - A::powf(beta1_val, step_count_float);
                    let bias_correction2 = A::one() - A::powf(beta2_val, step_count_float);

                    let corrected_first = &self.gradient_accumulator / bias_correction1;
                    let corrected_second = &*second_moment / bias_correction2;

                    // Update parameters
                    let adaptive_lr =
                        corrected_second.mapv(|v| A::sqrt(v) + A::from(*epsilon).unwrap());
                    self.parameters =
                        &self.parameters - &(corrected_first / adaptive_lr * self.current_lr);
                }
            }
            LearningRateAdaptation::ExponentialDecay { decay_rate } => {
                // Simple exponential decay
                self.current_lr = self.current_lr * A::from(*decay_rate).unwrap();
                self.parameters = &self.parameters - gradient * self.current_lr;
            }
            LearningRateAdaptation::InverseScaling { power } => {
                // Inverse scaling: lr = initial_lr / (step^power)
                let step_power =
                    A::powf(A::from(self.step_count).unwrap(), A::from(*power).unwrap());
                let decayed_lr = self.current_lr / step_power;
                self.parameters = &self.parameters - gradient * decayed_lr;
            }
        }

        Ok(())
    }

    /// Online Newton's method update
    fn online_newton_update(&mut self, gradient: &Array<A, D>, damping: f64) -> Result<()> {
        // Simplified online Newton update with damping
        let damping_val = A::from(damping).unwrap();

        // Approximate Hessian diagonal with gradient squares (simplified)
        let hessian_approx = gradient.mapv(|g| g * g + damping_val);

        // Newton step
        let newton_step = gradient / hessian_approx;
        self.parameters = &self.parameters - &newton_step * self.current_lr;

        Ok(())
    }

    /// FTRL update
    fn ftrl_update(
        &mut self,
        gradient: &Array<A, D>,
        l1_reg: f64,
        l2_reg: f64,
        lr_power: f64,
    ) -> Result<()> {
        // Accumulate gradients
        self.gradient_accumulator = &self.gradient_accumulator + gradient;

        // FTRL update rule (simplified)
        let step_factor = A::powf(
            A::from(self.step_count).unwrap(),
            A::from(lr_power).unwrap(),
        );
        let learning_rate = self.current_lr / step_factor;

        // Apply L1 and L2 regularization
        let l1_weight = A::from(l1_reg).unwrap();
        let l2_weight = A::from(l2_reg).unwrap();

        self.parameters = self.gradient_accumulator.mapv(|g| {
            let abs_g = A::abs(g);
            if abs_g <= l1_weight {
                A::zero()
            } else {
                let sign = if g > A::zero() { A::one() } else { -A::one() };
                -sign * (abs_g - l1_weight) / (l2_weight + A::sqrt(abs_g))
            }
        }) * learning_rate;

        Ok(())
    }

    /// Mirror descent update
    fn mirror_descent_update(
        &mut self,
        gradient: &Array<A, D>,
        mirror_fn: &MirrorFunction,
        regularization: f64,
    ) -> Result<()> {
        match mirror_fn {
            MirrorFunction::Euclidean => {
                // Standard gradient descent
                self.parameters = &self.parameters - gradient * self.current_lr;
            }
            MirrorFunction::Entropy => {
                // Entropy regularized update (for probability simplex)
                let reg_val = A::from(regularization).unwrap();
                let updated = self
                    .parameters
                    .mapv(|p| A::exp(A::ln(p) - self.current_lr * reg_val));
                let sum = updated.sum();
                self.parameters = updated / sum; // Normalize to probability simplex
            }
            MirrorFunction::L1 => {
                // L1 regularized update with soft thresholding
                let threshold = self.current_lr * A::from(regularization).unwrap();
                self.parameters = (&self.parameters - gradient * self.current_lr).mapv(|p| {
                    if A::abs(p) <= threshold {
                        A::zero()
                    } else {
                        p - A::signum(p) * threshold
                    }
                });
            }
            MirrorFunction::Nuclear => {
                // Simplified nuclear norm update (requires matrix structure)
                self.parameters = &self.parameters - gradient * self.current_lr;
            }
        }

        Ok(())
    }

    /// Adaptive multi-task update
    fn adaptive_multitask_update(&mut self, gradient: &Array<A, D>) -> Result<()> {
        // Simplified multi-task update
        self.parameters = &self.parameters - gradient * self.current_lr;
        Ok(())
    }

    /// Update regret bound estimation
    fn update_regret_bound(&mut self, loss: A) {
        if let Some(&best_loss) = self
            .performance_history
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
        {
            let regret = loss - best_loss;
            self.regret_bound = self.regret_bound + regret.max(A::zero());
        }
    }

    /// Get current parameters
    pub fn parameters(&self) -> &Array<A, D> {
        &self.parameters
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> OnlinePerformanceMetrics<A> {
        let average_loss = if self.performance_history.is_empty() {
            A::zero()
        } else {
            self.performance_history.iter().copied().sum::<A>()
                / A::from(self.performance_history.len()).unwrap()
        };

        let lr_stability = A::from(1.0).unwrap(); // Simplified
        let adaptation_speed = A::from(self.step_count as f64).unwrap(); // Simplified
        let memory_efficiency = A::from(0.8).unwrap(); // Simplified

        OnlinePerformanceMetrics {
            cumulative_regret: self.regret_bound,
            average_loss,
            lr_stability,
            adaptation_speed,
            memory_efficiency,
        }
    }
}

impl<A: Float + ScalarOperand + Debug + std::iter::Sum, D: Dimension> LifelongOptimizer<A, D> {
    /// Create a new lifelong optimizer
    pub fn new(strategy: LifelongStrategy) -> Self {
        Self {
            strategy,
            task_optimizers: HashMap::new(),
            shared_knowledge: SharedKnowledge {
                fisher_information: None,
                important_parameters: None,
                task_embeddings: HashMap::new(),
                transfer_weights: HashMap::new(),
                meta_parameters: None,
            },
            task_graph: TaskGraph {
                task_similarities: HashMap::new(),
                task_dependencies: HashMap::new(),
                task_clusters: HashMap::new(),
            },
            memory_buffer: MemoryBuffer {
                examples: VecDeque::new(),
                max_size: 1000,
                update_strategy: MemoryUpdateStrategy::FIFO,
                importance_scores: VecDeque::new(),
            },
            current_task: None,
            task_performance: HashMap::new(),
        }
    }

    /// Start learning a new task
    pub fn start_task(&mut self, task_id: String, initial_parameters: Array<A, D>) -> Result<()> {
        self.current_task = Some(task_id.clone());

        // Create task-specific optimizer
        let online_strategy = OnlineLearningStrategy::AdaptiveSGD {
            initial_lr: 0.001,
            adaptation_method: LearningRateAdaptation::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
        };

        let task_optimizer = OnlineOptimizer::new(online_strategy, initial_parameters);
        self.task_optimizers.insert(task_id.clone(), task_optimizer);

        // Initialize task performance tracking
        self.task_performance.insert(task_id, Vec::new());

        Ok(())
    }

    /// Update current task with new data
    pub fn update_current_task(&mut self, gradient: &Array<A, D>, loss: A) -> Result<()> {
        let task_id = self
            .current_task
            .as_ref()
            .ok_or_else(|| OptimError::InvalidConfig("No current task set".to_string()))?
            .clone();

        // Update task-specific optimizer
        if let Some(optimizer) = self.task_optimizers.get_mut(&task_id) {
            optimizer.online_update(gradient, loss)?;
        }

        // Track performance
        if let Some(performance) = self.task_performance.get_mut(&task_id) {
            performance.push(loss);
        }

        // Apply lifelong learning strategy
        match &self.strategy {
            LifelongStrategy::ElasticWeightConsolidation {
                importance_weight, ..
            } => {
                self.apply_ewc_regularization(gradient, *importance_weight)?;
            }
            LifelongStrategy::ProgressiveNetworks { .. } => {
                self.apply_progressive_networks(gradient)?;
            }
            LifelongStrategy::MemoryAugmented { .. } => {
                self.update_memory_buffer(gradient, loss)?;
            }
            LifelongStrategy::MetaLearning { .. } => {
                self.apply_meta_learning(gradient)?;
            }
            LifelongStrategy::GradientEpisodicMemory { .. } => {
                self.apply_gem_constraints(gradient)?;
            }
        }

        Ok(())
    }

    /// Apply Elastic Weight Consolidation regularization
    fn apply_ewc_regularization(
        &mut self,
        gradient: &Array<A, D>,
        _importance_weight: f64,
    ) -> Result<()> {
        // Simplified EWC implementation
        // In practice, this would compute Fisher Information Matrix and apply regularization
        Ok(())
    }

    /// Apply Progressive Networks strategy
    fn apply_progressive_networks(&mut self, gradient: &Array<A, D>) -> Result<()> {
        // Simplified Progressive Networks implementation
        // In practice, this would manage lateral connections between task columns
        Ok(())
    }

    /// Update memory buffer with important examples
    fn update_memory_buffer(&mut self, gradient: &Array<A, D>, loss: A) -> Result<()> {
        if let Some(task_id) = &self.current_task {
            let example = MemoryExample {
                input: Array::zeros(gradient.raw_dim()),  // Placeholder
                target: Array::zeros(gradient.raw_dim()), // Placeholder
                task_id: task_id.clone(),
                importance: loss,
                gradient: Some(gradient.clone()),
            };

            // Add to buffer
            if self.memory_buffer.examples.len() >= self.memory_buffer.max_size {
                match self.memory_buffer.update_strategy {
                    MemoryUpdateStrategy::FIFO => {
                        self.memory_buffer.examples.pop_front();
                        self.memory_buffer.importance_scores.pop_front();
                    }
                    MemoryUpdateStrategy::Random => {
                        let idx = scirs2_core::random::rng()
                            .gen_range(0..self.memory_buffer.examples.len());
                        self.memory_buffer.examples.remove(idx);
                        self.memory_buffer.importance_scores.remove(idx);
                    }
                    MemoryUpdateStrategy::ImportanceBased => {
                        // Remove least important example
                        if let Some(min_idx) = self
                            .memory_buffer
                            .importance_scores
                            .iter()
                            .enumerate()
                            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .map(|(idx, _)| idx)
                        {
                            self.memory_buffer.examples.remove(min_idx);
                            self.memory_buffer.importance_scores.remove(min_idx);
                        }
                    }
                    MemoryUpdateStrategy::GradientDiversity => {
                        // Remove most similar gradient (simplified)
                        self.memory_buffer.examples.pop_front();
                        self.memory_buffer.importance_scores.pop_front();
                    }
                }
            }

            self.memory_buffer.examples.push_back(example);
            self.memory_buffer.importance_scores.push_back(loss);
        }

        Ok(())
    }

    /// Apply meta-learning strategy
    fn apply_meta_learning(&mut self, gradient: &Array<A, D>) -> Result<()> {
        // Simplified meta-learning implementation
        // In practice, this would update meta-parameters based on task performance
        Ok(())
    }

    /// Apply Gradient Episodic Memory constraints
    fn apply_gem_constraints(&mut self, gradient: &Array<A, D>) -> Result<()> {
        // Simplified GEM implementation
        // In practice, this would project gradients to satisfy memory constraints
        Ok(())
    }

    /// Compute task similarity
    pub fn compute_task_similarity(&self, task1: &str, task2: &str) -> f64 {
        self.task_graph
            .task_similarities
            .get(&(task1.to_string(), task2.to_string()))
            .or_else(|| {
                self.task_graph
                    .task_similarities
                    .get(&(task2.to_string(), task1.to_string()))
            })
            .copied()
            .unwrap_or(0.0)
    }

    /// Get lifelong learning statistics
    pub fn get_lifelong_stats(&self) -> LifelongStats<A> {
        let num_tasks = self.task_optimizers.len();
        let avg_performance = if self.task_performance.is_empty() {
            A::zero()
        } else {
            let total_performance: A = self.task_performance.values().flatten().copied().sum();
            let total_samples = self
                .task_performance
                .values()
                .map(|v| v.len())
                .sum::<usize>();
            if total_samples > 0 {
                total_performance / A::from(total_samples).unwrap()
            } else {
                A::zero()
            }
        };

        LifelongStats {
            num_tasks,
            average_performance: avg_performance,
            memory_usage: self.memory_buffer.examples.len(),
            transfer_efficiency: A::from(0.8).unwrap(), // Placeholder
            catastrophic_forgetting: A::from(0.1).unwrap(), // Placeholder
        }
    }
}

/// Lifelong learning statistics
#[derive(Debug, Clone)]
pub struct LifelongStats<A: Float> {
    /// Number of tasks learned
    pub num_tasks: usize,
    /// Average performance across all tasks
    pub average_performance: A,
    /// Current memory usage
    pub memory_usage: usize,
    /// Transfer learning efficiency
    pub transfer_efficiency: A,
    /// Catastrophic forgetting measure
    pub catastrophic_forgetting: A,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_online_optimizer_creation() {
        let strategy = OnlineLearningStrategy::AdaptiveSGD {
            initial_lr: 0.01,
            adaptation_method: LearningRateAdaptation::AdaGrad { epsilon: 1e-8 },
        };

        let initial_params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let optimizer = OnlineOptimizer::new(strategy, initial_params);

        assert_eq!(optimizer.step_count, 0);
        assert_relative_eq!(optimizer.current_lr, 0.01, epsilon = 1e-6);
    }

    #[test]
    fn test_online_update() {
        let strategy = OnlineLearningStrategy::AdaptiveSGD {
            initial_lr: 0.1,
            adaptation_method: LearningRateAdaptation::ExponentialDecay { decay_rate: 0.99 },
        };

        let initial_params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut optimizer = OnlineOptimizer::new(strategy, initial_params);

        let gradient = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let loss = 0.5;

        optimizer.online_update(&gradient, loss).unwrap();

        assert_eq!(optimizer.step_count, 1);
        assert_eq!(optimizer.performance_history.len(), 1);
        assert_relative_eq!(optimizer.performance_history[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_lifelong_optimizer_creation() {
        let strategy = LifelongStrategy::ElasticWeightConsolidation {
            importance_weight: 1000.0,
            fisher_samples: 100,
        };

        let optimizer = LifelongOptimizer::<f64, ndarray::Ix1>::new(strategy);

        assert_eq!(optimizer.task_optimizers.len(), 0);
        assert!(optimizer.current_task.is_none());
    }

    #[test]
    fn test_task_management() {
        let strategy = LifelongStrategy::MemoryAugmented {
            memory_size: 100,
            update_strategy: MemoryUpdateStrategy::FIFO,
        };

        let mut optimizer = LifelongOptimizer::<f64, ndarray::Ix1>::new(strategy);
        let initial_params = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        optimizer
            .start_task("task1".to_string(), initial_params)
            .unwrap();

        assert_eq!(optimizer.current_task, Some("task1".to_string()));
        assert!(optimizer.task_optimizers.contains_key("task1"));
        assert!(optimizer.task_performance.contains_key("task1"));
    }

    #[test]
    fn test_memory_buffer_update() {
        let strategy = LifelongStrategy::MemoryAugmented {
            memory_size: 2,
            update_strategy: MemoryUpdateStrategy::FIFO,
        };

        let mut optimizer = LifelongOptimizer::<f64, ndarray::Ix1>::new(strategy);
        optimizer.memory_buffer.max_size = 2;

        let initial_params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        optimizer
            .start_task("task1".to_string(), initial_params)
            .unwrap();

        let gradient = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // Add first example
        optimizer.update_current_task(&gradient, 0.5).unwrap();
        assert_eq!(optimizer.memory_buffer.examples.len(), 1);

        // Add second example
        optimizer.update_current_task(&gradient, 0.6).unwrap();
        assert_eq!(optimizer.memory_buffer.examples.len(), 2);

        // Add third example (should remove first due to FIFO)
        optimizer.update_current_task(&gradient, 0.7).unwrap();
        assert_eq!(optimizer.memory_buffer.examples.len(), 2);
    }

    #[test]
    fn test_performance_metrics() {
        let strategy = OnlineLearningStrategy::AdaptiveSGD {
            initial_lr: 0.01,
            adaptation_method: LearningRateAdaptation::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
        };

        let initial_params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut optimizer = OnlineOptimizer::new(strategy, initial_params);

        // Add some performance data
        optimizer.performance_history.push_back(0.8);
        optimizer.performance_history.push_back(0.6);
        optimizer.performance_history.push_back(0.4);
        optimizer.regret_bound = 0.5;

        let metrics = optimizer.get_performance_metrics();

        assert_relative_eq!(metrics.cumulative_regret, 0.5, epsilon = 1e-6);
        assert_relative_eq!(metrics.average_loss, 0.6, epsilon = 1e-6);
    }

    #[test]
    fn test_lifelong_stats() {
        let strategy = LifelongStrategy::MetaLearning {
            meta_lr: 0.001,
            inner_steps: 5,
            task_embedding_size: 64,
        };

        let mut optimizer = LifelongOptimizer::<f64, ndarray::Ix1>::new(strategy);

        // Add some tasks
        let initial_params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        optimizer
            .start_task("task1".to_string(), initial_params.clone())
            .unwrap();
        optimizer
            .start_task("task2".to_string(), initial_params)
            .unwrap();

        // Add some performance data
        optimizer
            .task_performance
            .get_mut("task1")
            .unwrap()
            .extend(vec![0.8, 0.7]);
        optimizer
            .task_performance
            .get_mut("task2")
            .unwrap()
            .extend(vec![0.9, 0.8]);

        let stats = optimizer.get_lifelong_stats();

        assert_eq!(stats.num_tasks, 2);
        assert_relative_eq!(stats.average_performance, 0.8, epsilon = 1e-6);
    }

    #[test]
    fn test_learning_rate_adaptations() {
        let strategies = vec![
            LearningRateAdaptation::AdaGrad { epsilon: 1e-8 },
            LearningRateAdaptation::RMSprop {
                decay: 0.9,
                epsilon: 1e-8,
            },
            LearningRateAdaptation::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            LearningRateAdaptation::ExponentialDecay { decay_rate: 0.99 },
            LearningRateAdaptation::InverseScaling { power: 0.5 },
        ];

        for adaptation in strategies {
            let strategy = OnlineLearningStrategy::AdaptiveSGD {
                initial_lr: 0.01,
                adaptation_method: adaptation,
            };

            let initial_params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
            let mut optimizer = OnlineOptimizer::new(strategy, initial_params);

            let gradient = Array1::from_vec(vec![0.1, 0.2, 0.3]);
            let result = optimizer.online_update(&gradient, 0.5);

            assert!(result.is_ok());
            assert_eq!(optimizer.step_count, 1);
        }
    }
}
