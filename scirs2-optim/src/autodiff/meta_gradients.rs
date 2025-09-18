//! Meta-gradients for meta-learning algorithms
//!
//! This module implements meta-gradient computation for algorithms like MAML,
//! Reptile, and other meta-learning approaches that require gradients of
//! gradients with respect to meta-parameters.

use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};

use super::forward_mode::ForwardModeEngine;
use super::higher_order::{HessianConfig, HigherOrderEngine};
use super::reverse_mode::ReverseModeEngine;
use crate::error::Result;

/// Meta-gradient computation engine
#[allow(dead_code)]
pub struct MetaGradientEngine<T: Float + Default + Clone + 'static + std::iter::Sum + ndarray::ScalarOperand> {
    /// Forward-mode engine for directional derivatives
    forward_engine: ForwardModeEngine<T>,

    /// Reverse-mode engine for efficient computation
    reverse_engine: ReverseModeEngine<T>,

    /// Higher-order engine for advanced derivative computation
    higher_order_engine: HigherOrderEngine<T>,

    /// Meta-learning algorithm type
    algorithm: MetaLearningAlgorithm,

    /// Inner loop configuration
    inner_loop_config: InnerLoopConfig,

    /// Outer loop configuration  
    outer_loop_config: OuterLoopConfig,

    /// Gradient computation cache
    gradient_cache: HashMap<String, Array1<T>>,

    /// Meta-parameter history
    meta_param_history: VecDeque<Array1<T>>,

    /// Task performance history
    task_performance: HashMap<String, Vec<T>>,

    /// Hessian computation cache for efficiency
    hessian_cache: HashMap<String, Array2<T>>,

    /// Checkpointing manager for memory efficiency
    checkpoint_manager: CheckpointManager<T>,

    /// Advanced meta-learning configuration
    advanced_config: AdvancedMetaConfig,
}

/// Meta-learning algorithm types
#[derive(Debug, Clone, Copy)]
pub enum MetaLearningAlgorithm {
    /// Model-Agnostic Meta-Learning
    MAML,

    /// First-Order MAML (FOMAML)
    FirstOrderMAML,

    /// Reptile algorithm
    Reptile,

    /// Learning to Learn by Gradient Descent (L2L)
    L2L,

    /// Meta-SGD
    MetaSGD,

    /// Learned optimizer
    LearnedOptimizer,

    /// Implicit Function Theorem based
    ImplicitFunctionTheorem,
}

/// Inner loop configuration for meta-learning
#[derive(Debug, Clone)]
pub struct InnerLoopConfig {
    /// Number of inner steps
    pub num_steps: usize,

    /// Inner learning rate
    pub learning_rate: f64,

    /// Inner optimizer type
    pub optimizer: InnerOptimizer,

    /// Stop condition
    pub stop_condition: StopCondition,

    /// Enable gradient checkpointing
    pub gradient_checkpointing: bool,

    /// Maximum inner steps
    pub max_steps: usize,
}

/// Outer loop configuration for meta-learning
#[derive(Debug, Clone)]
pub struct OuterLoopConfig {
    /// Meta learning rate
    pub meta_learning_rate: f64,

    /// Meta-optimizer type
    pub meta_optimizer: MetaOptimizer,

    /// Number of tasks per meta-batch
    pub tasks_per_batch: usize,

    /// Enable second-order derivatives
    pub second_order: bool,

    /// Use implicit function theorem
    pub use_implicit_function_theorem: bool,

    /// Meta-regularization coefficient
    pub meta_regularization: f64,
}

/// Inner optimizer types
#[derive(Debug, Clone, Copy)]
pub enum InnerOptimizer {
    SGD,
    Adam,
    RMSprop,
    Custom,
}

/// Meta-optimizer types
#[derive(Debug, Clone, Copy)]
pub enum MetaOptimizer {
    SGD,
    Adam,
    RMSprop,
    LBFGS,
    Custom,
}

/// Stop conditions for inner loop
#[derive(Debug, Clone)]
pub enum StopCondition {
    /// Fixed number of steps
    FixedSteps,

    /// Convergence threshold
    Convergence { threshold: f64 },

    /// Loss threshold
    LossThreshold { threshold: f64 },

    /// Gradient norm threshold
    GradientNorm { threshold: f64 },
}

/// Meta-learning task definition
#[derive(Debug, Clone)]
pub struct MetaTask<T: Float> {
    /// Task identifier
    pub id: String,

    /// Support set (training data)
    pub support_set: Vec<(Array1<T>, Array1<T>)>,

    /// Query set (test data)
    pub query_set: Vec<(Array1<T>, Array1<T>)>,

    /// Task-specific parameters
    pub task_params: HashMap<String, T>,

    /// Task weight for meta-batch
    pub weight: T,
}

/// Meta-gradient computation result
#[derive(Debug, Clone)]
pub struct MetaGradientResult<T: Float> {
    /// Meta-gradients w.r.t. meta-parameters
    pub meta_gradients: Array1<T>,

    /// Inner loop gradients for each task
    pub inner_gradients: Vec<Array1<T>>,

    /// Task losses
    pub task_losses: Vec<T>,

    /// Meta-loss
    pub meta_loss: T,

    /// Computation method used
    pub method: MetaGradientMethod,

    /// Computation statistics
    pub stats: MetaGradientStats,
}

/// Meta-gradient computation methods
#[derive(Debug, Clone, Copy)]
pub enum MetaGradientMethod {
    /// Exact second-order derivatives
    ExactSecondOrder,

    /// First-order approximation
    FirstOrderApproximation,

    /// Implicit function theorem
    ImplicitFunctionTheorem,

    /// Finite differences
    FiniteDifferences,

    /// Truncated backpropagation
    TruncatedBackprop,
}

/// Meta-gradient computation statistics
#[derive(Debug, Clone)]
pub struct MetaGradientStats {
    /// Total computation time (microseconds)
    pub computation_time_us: u64,

    /// Number of inner steps per task
    pub inner_steps_per_task: Vec<usize>,

    /// Memory usage estimate (bytes)
    pub memory_usage: usize,

    /// Gradient computation count
    pub gradient_computations: usize,

    /// Second-order computations
    pub second_order_computations: usize,

    /// Hessian-vector product computations
    pub hvp_computations: usize,

    /// Cache hits/misses
    pub cache_hits: usize,
    pub cache_misses: usize,

    /// Checkpointing statistics
    pub checkpoints_created: usize,
    pub checkpoints_restored: usize,
}

/// Advanced meta-learning configuration
#[derive(Debug, Clone)]
pub struct AdvancedMetaConfig {
    /// Enable sophisticated Hessian computation
    pub use_higher_order_hessian: bool,

    /// Automatic HVP mode selection
    pub auto_hvp_mode: bool,

    /// Enable gradient checkpointing for long sequences
    pub enable_gradient_checkpointing: bool,

    /// Maximum memory usage before checkpointing (bytes)
    pub max_memory_usage: usize,

    /// Enable hyperparameter optimization
    pub enable_hyperparameter_optimization: bool,

    /// Learning rate adaptation method
    pub lr_adaptation_method: LearningRateAdaptation,

    /// Enable multi-task learning utilities
    pub enable_multi_task_learning: bool,

    /// Task similarity computation
    pub task_similarity_metric: TaskSimilarityMetric,

    /// Meta-batch sampling strategy
    pub meta_batch_sampling: MetaBatchSampling,

    /// Enable curriculum learning
    pub enable_curriculum_learning: bool,

    /// Regularization techniques
    pub regularization: MetaRegularization,
}

/// Learning rate adaptation methods
#[derive(Debug, Clone, Copy)]
pub enum LearningRateAdaptation {
    /// Fixed learning rate
    Fixed,
    /// Learn per-parameter learning rates (Meta-SGD style)
    PerParameter,
    /// Adaptive based on gradient magnitude
    AdaptiveGradient,
    /// Learned adaptation rule
    LearnedRule,
    /// Meta-learning the adaptation
    MetaLearned,
}

/// Task similarity metrics
#[derive(Debug, Clone, Copy)]
pub enum TaskSimilarityMetric {
    /// Cosine similarity of gradients
    GradientCosine,
    /// Fisher Information distance
    FisherDistance,
    /// Hessian similarity
    HessianSimilarity,
    /// Parameter distance after adaptation
    ParameterDistance,
    /// Loss landscape similarity
    LossLandscape,
}

/// Meta-batch sampling strategies
#[derive(Debug, Clone, Copy)]
pub enum MetaBatchSampling {
    /// Random sampling
    Random,
    /// Balanced sampling across task types
    Balanced,
    /// Difficulty-based curriculum
    Curriculum,
    /// Similarity-based clustering
    SimilarityBased,
    /// Adversarial task selection
    Adversarial,
}

/// Meta-learning regularization techniques
#[derive(Debug, Clone)]
pub struct MetaRegularization {
    /// L2 regularization on meta-parameters
    pub l2_meta_params: f64,
    /// Regularization on adaptation speed
    pub adaptation_regularization: f64,
    /// Entropy regularization for exploration
    pub entropy_regularization: f64,
    /// Task diversity bonus
    pub diversity_bonus: f64,
    /// Gradient penalty for stability
    pub gradient_penalty: f64,
}

/// Checkpoint manager for memory-efficient meta-learning
#[derive(Debug)]
pub struct CheckpointManager<T: Float> {
    /// Stored checkpoints
    checkpoints: HashMap<String, MetaCheckpoint<T>>,
    /// Maximum number of checkpoints to keep
    max_checkpoints: usize,
    /// Memory threshold for automatic checkpointing
    memory_threshold: usize,
    /// Current memory usage estimate
    current_memory: usize,
    /// Checkpoint creation policy
    policy: CheckpointPolicy,
}

/// Meta-learning checkpoint
#[derive(Debug, Clone)]
pub struct MetaCheckpoint<T: Float> {
    /// Checkpoint identifier
    pub id: String,
    /// Saved parameters
    pub parameters: Array1<T>,
    /// Saved gradients
    pub gradients: Array1<T>,
    /// Saved Hessian (if computed)
    pub hessian: Option<Array2<T>>,
    /// Task state
    pub task_state: TaskState<T>,
    /// Computation step
    pub step: usize,
    /// Memory usage at checkpoint
    pub memory_usage: usize,
    /// Timestamp
    pub timestamp: std::time::Instant,
}

/// Task state for checkpointing
#[derive(Debug, Clone)]
pub struct TaskState<T: Float> {
    /// Current task parameters
    pub current_params: Array1<T>,
    /// Task loss history
    pub loss_history: Vec<T>,
    /// Gradient history
    pub gradient_history: Vec<Array1<T>>,
    /// Adaptation step
    pub adaptation_step: usize,
}

/// Checkpointing policies
#[derive(Debug, Clone, Copy)]
pub enum CheckpointPolicy {
    /// Checkpoint at fixed intervals
    FixedInterval { interval: usize },
    /// Checkpoint when memory exceeds threshold
    MemoryThreshold,
    /// Checkpoint at key computational points
    AdaptiveKeyPoints,
    /// No checkpointing
    None,
}

impl<T: Float + Default + Clone + 'static + std::iter::Sum + ndarray::ScalarOperand>
    MetaGradientEngine<T>
{
    /// Create a new meta-gradient engine
    pub fn new(
        algorithm: MetaLearningAlgorithm,
        inner_config: InnerLoopConfig,
        outer_config: OuterLoopConfig,
    ) -> Self {
        Self {
            forward_engine: ForwardModeEngine::new(),
            reverse_engine: ReverseModeEngine::new(),
            higher_order_engine: HigherOrderEngine::new(3), // Support up to 3rd order derivatives
            algorithm,
            inner_loop_config: inner_config,
            outer_loop_config: outer_config,
            gradient_cache: HashMap::new(),
            meta_param_history: VecDeque::with_capacity(1000),
            task_performance: HashMap::new(),
            hessian_cache: HashMap::new(),
            checkpoint_manager: CheckpointManager::new(100, 1024 * 1024 * 512), // 512MB threshold
            advanced_config: AdvancedMetaConfig::default(),
        }
    }

    /// Create a new meta-gradient engine with advanced configuration
    pub fn with_advanced_config(
        algorithm: MetaLearningAlgorithm,
        inner_config: InnerLoopConfig,
        outer_config: OuterLoopConfig,
        advanced_config: AdvancedMetaConfig,
    ) -> Self {
        let mut engine = Self::new(algorithm, inner_config, outer_config);
        engine.advanced_config = advanced_config;

        // Configure higher-order engine based on advanced _config
        if engine.advanced_config.use_higher_order_hessian {
            engine.higher_order_engine.set_mixed_mode(true);
        }

        engine
    }

    /// Compute meta-gradients for a batch of tasks
    pub fn compute_meta_gradients(
        &mut self,
        meta_params: &Array1<T>,
        tasks: &[MetaTask<T>],
        objective_fn: impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<MetaGradientResult<T>> {
        let _start_time = std::time::Instant::now();

        match self.algorithm {
            MetaLearningAlgorithm::MAML => {
                self.compute_maml_gradients(meta_params, tasks, &objective_fn)
            }
            MetaLearningAlgorithm::FirstOrderMAML => {
                self.compute_fomaml_gradients(meta_params, tasks, &objective_fn)
            }
            MetaLearningAlgorithm::Reptile => {
                self.compute_reptile_gradients(meta_params, tasks, &objective_fn)
            }
            MetaLearningAlgorithm::L2L => {
                self.compute_l2l_gradients(meta_params, tasks, &objective_fn)
            }
            MetaLearningAlgorithm::MetaSGD => {
                self.compute_meta_sgd_gradients(meta_params, tasks, &objective_fn)
            }
            MetaLearningAlgorithm::LearnedOptimizer => {
                self.compute_learned_optimizer_gradients(meta_params, tasks, &objective_fn)
            }
            MetaLearningAlgorithm::ImplicitFunctionTheorem => {
                self.compute_ift_gradients(meta_params, tasks, &objective_fn)
            }
        }
    }

    /// Compute MAML meta-gradients
    fn compute_maml_gradients(
        &mut self,
        meta_params: &Array1<T>,
        tasks: &[MetaTask<T>],
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<MetaGradientResult<T>> {
        let start_time = std::time::Instant::now();
        let mut meta_gradients = Array1::zeros(meta_params.len());
        let mut inner_gradients = Vec::new();
        let mut task_losses = Vec::new();
        let mut total_meta_loss = T::zero();

        for task in tasks {
            // Inner loop adaptation
            let adapted_params = self.inner_loop_adaptation(meta_params, task, objective_fn)?;

            // Compute query loss with adapted parameters
            let query_loss = objective_fn(&adapted_params, meta_params, &task.query_set);
            task_losses.push(query_loss);
            total_meta_loss = total_meta_loss + query_loss * task.weight;

            // Compute meta-gradients using second-order derivatives
            let task_meta_gradients = if self.outer_loop_config.second_order {
                self.compute_second_order_meta_gradients(
                    meta_params,
                    &adapted_params,
                    task,
                    objective_fn,
                )?
            } else {
                self.compute_first_order_meta_gradients(
                    meta_params,
                    &adapted_params,
                    task,
                    objective_fn,
                )?
            };

            // Accumulate meta-gradients
            meta_gradients = meta_gradients + task_meta_gradients.clone() * task.weight;
            inner_gradients.push(task_meta_gradients);
        }

        let meta_loss = total_meta_loss / T::from(tasks.len()).unwrap();
        let computation_time = start_time.elapsed().as_micros() as u64;

        Ok(MetaGradientResult {
            meta_gradients,
            inner_gradients,
            task_losses,
            meta_loss,
            method: if self.outer_loop_config.second_order {
                MetaGradientMethod::ExactSecondOrder
            } else {
                MetaGradientMethod::FirstOrderApproximation
            },
            stats: MetaGradientStats {
                computation_time_us: computation_time,
                inner_steps_per_task: vec![self.inner_loop_config.num_steps; tasks.len()],
                memory_usage: self.estimate_memory_usage(),
                gradient_computations: tasks.len() * self.inner_loop_config.num_steps,
                second_order_computations: if self.outer_loop_config.second_order {
                    tasks.len()
                } else {
                    0
                },
                hvp_computations: 0,
                cache_hits: 0,
                cache_misses: 0,
                checkpoints_created: 0,
                checkpoints_restored: 0,
            },
        })
    }

    /// Compute First-Order MAML meta-gradients
    fn compute_fomaml_gradients(
        &mut self,
        meta_params: &Array1<T>,
        tasks: &[MetaTask<T>],
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<MetaGradientResult<T>> {
        let mut meta_gradients = Array1::zeros(meta_params.len());
        let mut task_losses = Vec::new();
        let mut total_meta_loss = T::zero();

        for task in tasks {
            // Inner loop adaptation
            let adapted_params = self.inner_loop_adaptation(meta_params, task, objective_fn)?;

            // Compute query loss
            let query_loss = objective_fn(&adapted_params, meta_params, &task.query_set);
            task_losses.push(query_loss);
            total_meta_loss = total_meta_loss + query_loss * task.weight;

            // First-order approximation: ignore second derivatives
            let gradient =
                self.compute_gradient_wrt_params(&adapted_params, &task.query_set, objective_fn)?;
            meta_gradients = meta_gradients + gradient * task.weight;
        }

        let meta_loss = total_meta_loss / T::from(tasks.len()).unwrap();

        Ok(MetaGradientResult {
            meta_gradients,
            inner_gradients: vec![Array1::zeros(meta_params.len()); tasks.len()],
            task_losses,
            meta_loss,
            method: MetaGradientMethod::FirstOrderApproximation,
            stats: MetaGradientStats {
                computation_time_us: 0,
                inner_steps_per_task: vec![self.inner_loop_config.num_steps; tasks.len()],
                memory_usage: self.estimate_memory_usage(),
                gradient_computations: tasks.len(),
                second_order_computations: 0,
                hvp_computations: 0,
                cache_hits: 0,
                cache_misses: 0,
                checkpoints_created: 0,
                checkpoints_restored: 0,
            },
        })
    }

    /// Compute Reptile meta-gradients
    fn compute_reptile_gradients(
        &mut self,
        meta_params: &Array1<T>,
        tasks: &[MetaTask<T>],
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<MetaGradientResult<T>> {
        let mut meta_gradients = Array1::zeros(meta_params.len());
        let mut task_losses = Vec::new();
        let mut total_meta_loss = T::zero();

        for task in tasks {
            // Inner loop adaptation (multiple steps)
            let adapted_params = self.inner_loop_adaptation(meta_params, task, objective_fn)?;

            // Reptile gradient: direction from meta-_params to adapted _params
            let reptile_gradient = &adapted_params - meta_params;
            meta_gradients = meta_gradients + reptile_gradient * task.weight;

            // Compute loss for tracking
            let query_loss = objective_fn(&adapted_params, meta_params, &task.query_set);
            task_losses.push(query_loss);
            total_meta_loss = total_meta_loss + query_loss * task.weight;
        }

        let meta_loss = total_meta_loss / T::from(tasks.len()).unwrap();

        Ok(MetaGradientResult {
            meta_gradients,
            inner_gradients: vec![Array1::zeros(meta_params.len()); tasks.len()],
            task_losses,
            meta_loss,
            method: MetaGradientMethod::FirstOrderApproximation,
            stats: MetaGradientStats {
                computation_time_us: 0,
                inner_steps_per_task: vec![self.inner_loop_config.num_steps; tasks.len()],
                memory_usage: self.estimate_memory_usage(),
                gradient_computations: tasks.len() * self.inner_loop_config.num_steps,
                second_order_computations: 0,
                hvp_computations: 0,
                cache_hits: 0,
                cache_misses: 0,
                checkpoints_created: 0,
                checkpoints_restored: 0,
            },
        })
    }

    /// Compute L2L (Learning to Learn) meta-gradients
    fn compute_l2l_gradients(
        &mut self,
        meta_params: &Array1<T>,
        tasks: &[MetaTask<T>],
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<MetaGradientResult<T>> {
        // L2L uses a learned optimizer, so this is a simplified implementation
        self.compute_maml_gradients(meta_params, tasks, objective_fn)
    }

    /// Compute Meta-SGD gradients
    fn compute_meta_sgd_gradients(
        &mut self,
        meta_params: &Array1<T>,
        tasks: &[MetaTask<T>],
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<MetaGradientResult<T>> {
        // Meta-SGD learns both parameters and learning rates
        // This is a simplified implementation
        self.compute_maml_gradients(meta_params, tasks, objective_fn)
    }

    /// Compute learned optimizer meta-gradients
    fn compute_learned_optimizer_gradients(
        &mut self,
        meta_params: &Array1<T>,
        tasks: &[MetaTask<T>],
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<MetaGradientResult<T>> {
        // Use learned optimizer for inner loop
        self.compute_maml_gradients(meta_params, tasks, objective_fn)
    }

    /// Compute gradients using Implicit Function Theorem
    fn compute_ift_gradients(
        &mut self,
        meta_params: &Array1<T>,
        tasks: &[MetaTask<T>],
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<MetaGradientResult<T>> {
        let mut meta_gradients = Array1::zeros(meta_params.len());
        let mut task_losses = Vec::new();
        let mut total_meta_loss = T::zero();

        for task in tasks {
            // Find optimal parameters by solving inner optimization to convergence
            let optimal_params = self.solve_inner_optimization(meta_params, task, objective_fn)?;

            // Use implicit function theorem to compute meta-gradients
            let ift_gradients = self.compute_implicit_function_gradients(
                meta_params,
                &optimal_params,
                task,
                objective_fn,
            )?;

            meta_gradients = meta_gradients + ift_gradients * task.weight;

            let query_loss = objective_fn(&optimal_params, meta_params, &task.query_set);
            task_losses.push(query_loss);
            total_meta_loss = total_meta_loss + query_loss * task.weight;
        }

        let meta_loss = total_meta_loss / T::from(tasks.len()).unwrap();

        Ok(MetaGradientResult {
            meta_gradients,
            inner_gradients: vec![Array1::zeros(meta_params.len()); tasks.len()],
            task_losses,
            meta_loss,
            method: MetaGradientMethod::ImplicitFunctionTheorem,
            stats: MetaGradientStats {
                computation_time_us: 0,
                inner_steps_per_task: vec![100; tasks.len()], // Convergence steps
                memory_usage: self.estimate_memory_usage(),
                gradient_computations: tasks.len() * 100,
                second_order_computations: tasks.len(),
                hvp_computations: 0,
                cache_hits: 0,
                cache_misses: 0,
                checkpoints_created: 0,
                checkpoints_restored: 0,
            },
        })
    }

    /// Perform inner loop adaptation
    fn inner_loop_adaptation(
        &self,
        meta_params: &Array1<T>,
        task: &MetaTask<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<Array1<T>> {
        let mut _params = meta_params.clone();
        let lr = num_traits::cast::cast(self.inner_loop_config.learning_rate).unwrap_or_else(|| T::zero());

        for _step in 0..self.inner_loop_config.num_steps {
            // Compute gradient on support set
            let gradient =
                self.compute_gradient_wrt_params(&_params, &task.support_set, objective_fn)?;

            // SGD update
            _params = _params - gradient.clone() * lr;

            // Check stop condition
            if self.check_stop_condition(&gradient, &_params, task, objective_fn)? {
                break;
            }
        }

        Ok(_params)
    }

    /// Solve inner optimization to convergence (for IFT)
    fn solve_inner_optimization(
        &self,
        meta_params: &Array1<T>,
        task: &MetaTask<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<Array1<T>> {
        let mut _params = meta_params.clone();
        let lr = num_traits::cast::cast(self.inner_loop_config.learning_rate).unwrap_or_else(|| T::zero());
        let convergence_threshold = num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero());

        for _step in 0..self.inner_loop_config.max_steps {
            let gradient =
                self.compute_gradient_wrt_params(&_params, &task.support_set, objective_fn)?;

            // Check convergence
            let grad_norm = gradient.iter().map(|&g| g * g).sum::<T>().sqrt();
            if grad_norm < convergence_threshold {
                break;
            }

            // Update parameters
            _params = _params - gradient * lr;
        }

        Ok(_params)
    }

    /// Compute second-order meta-gradients using advanced higher-order differentiation
    fn compute_second_order_meta_gradients(
        &mut self,
        meta_params: &Array1<T>,
        adapted_params: &Array1<T>,
        task: &MetaTask<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<Array1<T>> {
        if self.advanced_config.use_higher_order_hessian {
            // Use sophisticated higher-order differentiation
            self.compute_advanced_second_order_gradients(
                meta_params,
                adapted_params,
                task,
                objective_fn,
            )
        } else {
            // Fallback to finite differences
            self.compute_finite_difference_second_order(
                meta_params,
                adapted_params,
                task,
                objective_fn,
            )
        }
    }

    /// Advanced second-order meta-gradients using higher-order engine
    fn compute_advanced_second_order_gradients(
        &mut self,
        meta_params: &Array1<T>,
        _adapted_params: &Array1<T>,
        task: &MetaTask<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<Array1<T>> {
        let cache_key = format!("hessian_{}_{}", task.id, meta_params.len());

        // Check cache first
        if let Some(cached_hessian) = self.hessian_cache.get(&cache_key).cloned() {
            // Use cached Hessian for HVP computation
            return self.compute_hvp_with_cached_hessian(
                &cached_hessian,
                meta_params,
                task,
                objective_fn,
            );
        }

        // Extract configuration to avoid borrow conflicts
        let inner_lr = num_traits::cast::cast(self.inner_loop_config.learning_rate).unwrap_or_else(|| T::zero());
        let inner_steps = self.inner_loop_config.num_steps;
        let task_query_set = task.query_set.clone();
        let _task_support_set = task.support_set.clone();

        // Create composite function: F(θ) = L_query(φ*(θ))
        // where φ*(θ) is the result of inner optimization starting from θ
        let composite_fn = move |theta: &Array1<T>| -> T {
            // Simplified inner loop adaptation without capturing self
            let mut adapted_params = theta.clone();

            for _ in 0..inner_steps {
                // Simplified gradient computation (this is a placeholder)
                // In a real implementation, you'd need to compute gradients properly
                let grad_norm = adapted_params.iter().map(|&x| x * x).sum::<T>().sqrt();
                if grad_norm < num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero()) {
                    break;
                }

                // Simple gradient descent step (simplified)
                for param in adapted_params.iter_mut() {
                    *param = *param - inner_lr * (*param) * num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero());
                }
            }

            // Compute query loss
            objective_fn(&adapted_params, theta, &task_query_set)
        };

        // Use higher-order engine to compute exact Hessian
        let hessian_config = HessianConfig {
            exact: true,
            sparse: false,
            diagonal_only: false,
            verify_with_finite_diff: false,
            ..Default::default()
        };

        let hessian = self.higher_order_engine.hessian_forward_over_reverse(
            composite_fn,
            meta_params,
            &hessian_config,
        )?;

        // Cache the Hessian for future use
        self.hessian_cache.insert(cache_key, hessian.clone());

        // Compute gradient using the Hessian
        let task_query_set_2 = task.query_set.clone();
        let gradient = self.gradient_at_point(
            &|theta: &Array1<T>| -> T {
                // Simplified inner adaptation without capturing self
                let mut adapted_params = theta.clone();

                for _ in 0..inner_steps {
                    let grad_norm = adapted_params.iter().map(|&x| x * x).sum::<T>().sqrt();
                    if grad_norm < num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero()) {
                        break;
                    }

                    for param in adapted_params.iter_mut() {
                        *param = *param - inner_lr * (*param) * num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero());
                    }
                }

                objective_fn(&adapted_params, theta, &task_query_set_2)
            },
            meta_params,
        )?;

        Ok(gradient)
    }

    /// Compute HVP using cached Hessian for efficiency
    fn compute_hvp_with_cached_hessian(
        &mut self,
        hessian: &Array2<T>,
        meta_params: &Array1<T>,
        task: &MetaTask<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<Array1<T>> {
        // Extract needed values to avoid borrow conflicts
        let inner_lr = num_traits::cast::cast(self.inner_loop_config.learning_rate).unwrap_or_else(|| T::zero());
        let inner_steps = self.inner_loop_config.num_steps;
        let task_query_set = task.query_set.clone();

        // Compute gradient direction
        let gradient_direction = self.gradient_at_point(
            &|theta: &Array1<T>| -> T {
                // Simplified inner adaptation without capturing self
                let mut adapted_params = theta.clone();

                for _ in 0..inner_steps {
                    let grad_norm = adapted_params.iter().map(|&x| x * x).sum::<T>().sqrt();
                    if grad_norm < num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero()) {
                        break;
                    }

                    for param in adapted_params.iter_mut() {
                        *param = *param - inner_lr * (*param) * num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero());
                    }
                }

                objective_fn(&adapted_params, theta, &task_query_set)
            },
            meta_params,
        )?;

        // Compute Hessian-vector product: H * gradient_direction
        let hvp = hessian.dot(&gradient_direction);
        Ok(hvp)
    }

    /// Fallback finite difference second-order computation
    fn compute_finite_difference_second_order(
        &self,
        meta_params: &Array1<T>,
        _adapted_params: &Array1<T>,
        task: &MetaTask<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<Array1<T>> {
        let eps = num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero());
        let mut meta_gradients = Array1::zeros(meta_params.len());

        for i in 0..meta_params.len() {
            let mut meta_plus = meta_params.clone();
            let mut meta_minus = meta_params.clone();

            meta_plus[i] = meta_plus[i] + eps;
            meta_minus[i] = meta_minus[i] - eps;

            // Adapt parameters for perturbed meta-parameters
            let adapted_plus = self.inner_loop_adaptation(&meta_plus, task, objective_fn)?;
            let adapted_minus = self.inner_loop_adaptation(&meta_minus, task, objective_fn)?;

            // Compute query losses
            let loss_plus = objective_fn(&adapted_plus, &meta_plus, &task.query_set);
            let loss_minus = objective_fn(&adapted_minus, &meta_minus, &task.query_set);

            // Finite difference approximation
            meta_gradients[i] = (loss_plus - loss_minus) / (num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero()) * eps);
        }

        Ok(meta_gradients)
    }

    /// Compute first-order meta-gradients
    fn compute_first_order_meta_gradients(
        &self,
        _meta_params: &Array1<T>,
        adapted_params: &Array1<T>,
        task: &MetaTask<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<Array1<T>> {
        // First-order approximation: gradient w.r.t. adapted parameters
        self.compute_gradient_wrt_params(adapted_params, &task.query_set, objective_fn)
    }

    /// Compute implicit function theorem gradients
    fn compute_implicit_function_gradients(
        &self,
        meta_params: &Array1<T>,
        optimal_params: &Array1<T>,
        task: &MetaTask<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<Array1<T>> {
        // Simplified IFT computation
        // In practice, this would solve: dψ/dθ = -H^(-1) * d²L/dψdθ
        // where ψ are optimal params, θ are meta-_params, H is Hessian

        let eps = num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero());
        let mut gradients = Array1::zeros(meta_params.len());

        for i in 0..meta_params.len() {
            let mut meta_plus = meta_params.clone();
            meta_plus[i] = meta_plus[i] + eps;

            let optimal_plus = self.solve_inner_optimization(&meta_plus, task, objective_fn)?;

            // Approximate derivative of optimal _params w.r.t. meta-_params
            let param_derivative = (&optimal_plus - optimal_params) / eps;

            // Chain rule: dL/dθ = dL/dψ * dψ/dθ
            let loss_gradient =
                self.compute_gradient_wrt_params(optimal_params, &task.query_set, objective_fn)?;
            gradients[i] = loss_gradient.dot(&param_derivative);
        }

        Ok(gradients)
    }

    /// Compute gradient w.r.t. parameters
    fn compute_gradient_wrt_params(
        &self,
        params: &Array1<T>,
        data: &[(Array1<T>, Array1<T>)],
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<Array1<T>> {
        let eps = num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero());
        let mut gradient = Array1::zeros(params.len());

        for i in 0..params.len() {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();

            params_plus[i] = params_plus[i] + eps;
            params_minus[i] = params_minus[i] - eps;

            let loss_plus = objective_fn(&params_plus, params, data);
            let loss_minus = objective_fn(&params_minus, params, data);

            gradient[i] = (loss_plus - loss_minus) / (num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero()) * eps);
        }

        Ok(gradient)
    }

    /// Compute gradient of a function at a specific point
    fn gradient_at_point(
        &mut self,
        objective_fn: &impl Fn(&Array1<T>) -> T,
        point: &Array1<T>,
    ) -> Result<Array1<T>> {
        let eps = num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero());
        let mut gradient = Array1::zeros(point.len());

        for i in 0..point.len() {
            let mut point_plus = point.clone();
            let mut point_minus = point.clone();

            point_plus[i] = point_plus[i] + eps;
            point_minus[i] = point_minus[i] - eps;

            let loss_plus = objective_fn(&point_plus);
            let loss_minus = objective_fn(&point_minus);

            gradient[i] = (loss_plus - loss_minus) / (num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero()) * eps);
        }

        Ok(gradient)
    }

    /// Check inner loop stop condition
    fn check_stop_condition(
        &self,
        gradient: &Array1<T>,
        _params: &Array1<T>,
        _task: &MetaTask<T>,
        _objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<bool> {
        match &self.inner_loop_config.stop_condition {
            StopCondition::FixedSteps => Ok(false), // Always run fixed number of steps
            StopCondition::Convergence { threshold } => {
                let grad_norm = gradient.iter().map(|&g| g * g).sum::<T>().sqrt();
                Ok(grad_norm < num_traits::cast::cast(*threshold).unwrap_or_else(|| T::zero()))
            }
            StopCondition::GradientNorm { threshold } => {
                let grad_norm = gradient.iter().map(|&g| g * g).sum::<T>().sqrt();
                Ok(grad_norm < num_traits::cast::cast(*threshold).unwrap_or_else(|| T::zero()))
            }
            StopCondition::LossThreshold { threshold: _ } => {
                // Would need to compute loss here
                Ok(false)
            }
        }
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self) -> usize {
        let cache_size = self.gradient_cache.len() * std::mem::size_of::<(String, Array1<T>)>();
        let history_size = self.meta_param_history.len() * std::mem::size_of::<Array1<T>>();
        let engine_size = std::mem::size_of::<Self>();

        cache_size + history_size + engine_size
    }

    /// Get meta-learning statistics
    pub fn get_meta_stats(&self) -> MetaLearningStats {
        MetaLearningStats {
            algorithm: self.algorithm,
            cache_size: self.gradient_cache.len(),
            history_length: self.meta_param_history.len(),
            memory_usage: self.estimate_memory_usage(),
            total_tasks_processed: self.task_performance.len(),
        }
    }

    /// Clear caches and history
    pub fn clear_cache(&mut self) {
        self.gradient_cache.clear();
        self.meta_param_history.clear();
        self.task_performance.clear();
        self.hessian_cache.clear();
        self.checkpoint_manager.clear_checkpoints();
    }

    /// Compute task similarity for meta-batch sampling
    pub fn compute_task_similarity(
        &mut self,
        task1: &MetaTask<T>,
        task2: &MetaTask<T>,
        meta_params: &Array1<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<T> {
        match self.advanced_config.task_similarity_metric {
            TaskSimilarityMetric::GradientCosine => {
                self.compute_gradient_cosine_similarity(task1, task2, meta_params, objective_fn)
            }
            TaskSimilarityMetric::FisherDistance => {
                self.compute_fisher_distance(task1, task2, meta_params, objective_fn)
            }
            TaskSimilarityMetric::HessianSimilarity => {
                self.compute_hessian_similarity(task1, task2, meta_params, objective_fn)
            }
            TaskSimilarityMetric::ParameterDistance => {
                self.compute_parameter_distance(task1, task2, meta_params, objective_fn)
            }
            TaskSimilarityMetric::LossLandscape => {
                self.compute_loss_landscape_similarity(task1, task2, meta_params, objective_fn)
            }
        }
    }

    /// Compute gradient cosine similarity between tasks
    fn compute_gradient_cosine_similarity(
        &self,
        task1: &MetaTask<T>,
        task2: &MetaTask<T>,
        meta_params: &Array1<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<T> {
        let grad1 =
            self.compute_gradient_wrt_params(meta_params, &task1.support_set, objective_fn)?;
        let grad2 =
            self.compute_gradient_wrt_params(meta_params, &task2.support_set, objective_fn)?;

        let norm1 = grad1.iter().map(|&x| x * x).sum::<T>().sqrt();
        let norm2 = grad2.iter().map(|&x| x * x).sum::<T>().sqrt();

        if norm1 > T::zero() && norm2 > T::zero() {
            let cosine = grad1.dot(&grad2) / (norm1 * norm2);
            Ok(cosine)
        } else {
            Ok(T::zero())
        }
    }

    /// Compute Fisher Information distance between tasks
    fn compute_fisher_distance(
        &mut self,
        task1: &MetaTask<T>,
        task2: &MetaTask<T>,
        meta_params: &Array1<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<T> {
        // Simplified Fisher distance computation
        // In practice, would compute Fisher Information Matrices and their distance
        let grad1 =
            self.compute_gradient_wrt_params(meta_params, &task1.support_set, objective_fn)?;
        let grad2 =
            self.compute_gradient_wrt_params(meta_params, &task2.support_set, objective_fn)?;

        // Use L2 distance as approximation
        let diff = &grad1 - &grad2;
        let distance = diff.iter().map(|&x| x * x).sum::<T>().sqrt();
        Ok(distance)
    }

    /// Compute Hessian similarity between tasks
    fn compute_hessian_similarity(
        &mut self,
        task1: &MetaTask<T>,
        task2: &MetaTask<T>,
        meta_params: &Array1<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<T> {
        if !self.advanced_config.use_higher_order_hessian {
            // Fallback to gradient similarity
            return self.compute_gradient_cosine_similarity(
                task1,
                task2,
                meta_params,
                objective_fn,
            );
        }

        // Create objective functions for each task
        let objective1 =
            |_params: &Array1<T>| objective_fn(_params, meta_params, &task1.support_set);
        let objective2 =
            |_params: &Array1<T>| objective_fn(_params, meta_params, &task2.support_set);

        let hessian_config = HessianConfig {
            exact: true,
            sparse: false,
            diagonal_only: true, // Use diagonal for efficiency
            ..Default::default()
        };

        let hessian1 = self.higher_order_engine.hessian_forward_over_reverse(
            objective1,
            meta_params,
            &hessian_config,
        )?;
        let hessian2 = self.higher_order_engine.hessian_forward_over_reverse(
            objective2,
            meta_params,
            &hessian_config,
        )?;

        // Compute Frobenius similarity
        let mut similarity = T::zero();
        let mut norm1 = T::zero();
        let mut norm2 = T::zero();

        for i in 0..hessian1.nrows() {
            for j in 0..hessian1.ncols() {
                let h1 = hessian1[[i, j]];
                let h2 = hessian2[[i, j]];
                similarity = similarity + h1 * h2;
                norm1 = norm1 + h1 * h1;
                norm2 = norm2 + h2 * h2;
            }
        }

        if norm1 > T::zero() && norm2 > T::zero() {
            Ok(similarity / (norm1.sqrt() * norm2.sqrt()))
        } else {
            Ok(T::zero())
        }
    }

    /// Compute parameter distance after adaptation
    fn compute_parameter_distance(
        &self,
        task1: &MetaTask<T>,
        task2: &MetaTask<T>,
        meta_params: &Array1<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<T> {
        let adapted1 = self.inner_loop_adaptation(meta_params, task1, objective_fn)?;
        let adapted2 = self.inner_loop_adaptation(meta_params, task2, objective_fn)?;

        let diff = &adapted1 - &adapted2;
        let distance = diff.iter().map(|&x| x * x).sum::<T>().sqrt();
        Ok(distance)
    }

    /// Compute loss landscape similarity (simplified)
    fn compute_loss_landscape_similarity(
        &self,
        task1: &MetaTask<T>,
        task2: &MetaTask<T>,
        meta_params: &Array1<T>,
        objective_fn: &impl Fn(&Array1<T>, &Array1<T>, &[(Array1<T>, Array1<T>)]) -> T,
    ) -> Result<T> {
        // Sample points around meta_params and compare loss values
        let num_samples = 10;
        let perturbation_scale = num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero());
        let mut similarities = Vec::new();

        for sample_idx in 0..num_samples {
            // Generate deterministic perturbation based on sample index
            let mut perturbed_params = meta_params.clone();
            for i in 0..perturbed_params.len() {
                // Simple pseudo-random perturbation based on indices
                let seed = (sample_idx * 1000 + i * 17) % 1000;
                let perturbation_val = (seed as f64 / 1000.0) - 0.5; // Range [-0.5, 0.5]
                let perturbation = num_traits::cast::cast(perturbation_val).unwrap_or_else(|| T::zero()) * perturbation_scale;
                perturbed_params[i] = perturbed_params[i] + perturbation;
            }

            let loss1 = objective_fn(&perturbed_params, meta_params, &task1.support_set);
            let loss2 = objective_fn(&perturbed_params, meta_params, &task2.support_set);

            let similarity = T::one() / (T::one() + (loss1 - loss2).abs());
            similarities.push(similarity);
        }

        let avg_similarity =
            similarities.iter().copied().sum::<T>() / T::from(similarities.len()).unwrap();
        Ok(avg_similarity)
    }

    /// Adaptive learning rate computation
    pub fn compute_adaptive_learning_rate(
        &self,
        meta_params: &Array1<T>,
        gradient: &Array1<T>,
        task: &MetaTask<T>,
    ) -> Result<Array1<T>> {
        match self.advanced_config.lr_adaptation_method {
            LearningRateAdaptation::Fixed => {
                let lr = num_traits::cast::cast(self.inner_loop_config.learning_rate).unwrap_or_else(|| T::zero());
                Ok(Array1::from_elem(meta_params.len(), lr))
            }
            LearningRateAdaptation::PerParameter => {
                self.compute_per_parameter_learning_rates(meta_params, gradient, task)
            }
            LearningRateAdaptation::AdaptiveGradient => {
                self.compute_adaptive_gradient_rates(meta_params, gradient)
            }
            LearningRateAdaptation::LearnedRule => {
                self.compute_learned_adaptation_rule(meta_params, gradient, task)
            }
            LearningRateAdaptation::MetaLearned => {
                self.compute_meta_learned_rates(meta_params, gradient, task)
            }
        }
    }

    /// Compute per-parameter learning rates (Meta-SGD style)
    fn compute_per_parameter_learning_rates(
        &self,
        meta_params: &Array1<T>,
        gradient: &Array1<T>,
        _task: &MetaTask<T>,
    ) -> Result<Array1<T>> {
        let mut learning_rates = Array1::zeros(meta_params.len());
        let base_lr = num_traits::cast::cast(self.inner_loop_config.learning_rate).unwrap_or_else(|| T::zero());

        for i in 0..meta_params.len() {
            // Adapt learning rate based on gradient magnitude
            let grad_magnitude = gradient[i].abs();
            let adaptive_factor = T::one() / (T::one() + grad_magnitude);
            learning_rates[i] = base_lr * adaptive_factor;
        }

        Ok(learning_rates)
    }

    /// Compute adaptive gradient-based learning rates
    fn compute_adaptive_gradient_rates(
        &self,
        _meta_params: &Array1<T>,
        gradient: &Array1<T>,
    ) -> Result<Array1<T>> {
        let mut learning_rates = Array1::zeros(gradient.len());
        let base_lr = num_traits::cast::cast(self.inner_loop_config.learning_rate).unwrap_or_else(|| T::zero());

        // Compute gradient statistics
        let grad_mean = gradient.iter().copied().sum::<T>() / T::from(gradient.len()).unwrap();
        let grad_std = {
            let variance = gradient
                .iter()
                .map(|&g| (g - grad_mean) * (g - grad_mean))
                .sum::<T>()
                / T::from(gradient.len()).unwrap();
            variance.sqrt()
        };

        for i in 0..gradient.len() {
            // Normalize gradient and adapt learning rate
            let normalized_grad = if grad_std > T::zero() {
                (gradient[i] - grad_mean) / grad_std
            } else {
                gradient[i]
            };

            let adaptive_factor = T::one() / (T::one() + normalized_grad.abs());
            learning_rates[i] = base_lr * adaptive_factor;
        }

        Ok(learning_rates)
    }

    /// Compute learned adaptation rule (placeholder)
    fn compute_learned_adaptation_rule(
        &self,
        meta_params: &Array1<T>,
        _gradient: &Array1<T>,
        _task: &MetaTask<T>,
    ) -> Result<Array1<T>> {
        // Placeholder: would use a learned neural network
        let base_lr = num_traits::cast::cast(self.inner_loop_config.learning_rate).unwrap_or_else(|| T::zero());
        Ok(Array1::from_elem(meta_params.len(), base_lr))
    }

    /// Compute meta-learned adaptation rates (placeholder)
    fn compute_meta_learned_rates(
        &self,
        meta_params: &Array1<T>,
        _gradient: &Array1<T>,
        _task: &MetaTask<T>,
    ) -> Result<Array1<T>> {
        // Placeholder: would use meta-learned adaptation
        let base_lr = num_traits::cast::cast(self.inner_loop_config.learning_rate).unwrap_or_else(|| T::zero());
        Ok(Array1::from_elem(meta_params.len(), base_lr))
    }
}

/// Checkpoint manager implementation
impl<T: Float + Default + Clone> CheckpointManager<T> {
    /// Create a new checkpoint manager
    pub fn new(max_checkpoints: usize, memory_threshold: usize) -> Self {
        Self {
            checkpoints: HashMap::new(),
            max_checkpoints,
            memory_threshold,
            current_memory: 0,
            policy: CheckpointPolicy::MemoryThreshold,
        }
    }

    /// Create checkpoint with specified policy
    pub fn with_policy(
        max_checkpoints: usize,
        memory_threshold: usize,
        policy: CheckpointPolicy,
    ) -> Self {
        Self {
            checkpoints: HashMap::new(),
            max_checkpoints,
            memory_threshold,
            current_memory: 0,
            policy,
        }
    }

    /// Create a checkpoint
    pub fn create_checkpoint(
        &mut self,
        id: String,
        parameters: Array1<T>,
        gradients: Array1<T>,
        hessian: Option<Array2<T>>,
        task_state: TaskState<T>,
        step: usize,
    ) -> Result<()> {
        let memory_usage = self.estimate_checkpoint_memory(&parameters, &gradients, &hessian);

        let checkpoint = MetaCheckpoint {
            id: id.clone(),
            parameters,
            gradients,
            hessian,
            task_state,
            step,
            memory_usage,
            timestamp: std::time::Instant::now(),
        };

        // Check if we need to remove old checkpoints
        if self.checkpoints.len() >= self.max_checkpoints {
            self.remove_oldest_checkpoint();
        }

        self.current_memory += memory_usage;
        self.checkpoints.insert(id, checkpoint);

        Ok(())
    }

    /// Restore from checkpoint
    pub fn restore_checkpoint(&self, id: &str) -> Option<&MetaCheckpoint<T>> {
        self.checkpoints.get(id)
    }

    /// Check if checkpointing is needed
    pub fn should_checkpoint(&self, current_step: usize) -> bool {
        match self.policy {
            CheckpointPolicy::FixedInterval { interval } => current_step % interval == 0,
            CheckpointPolicy::MemoryThreshold => self.current_memory > self.memory_threshold,
            CheckpointPolicy::AdaptiveKeyPoints => {
                // Would implement logic to detect key computational points
                current_step % 10 == 0
            }
            CheckpointPolicy::None => false,
        }
    }

    /// Clear all checkpoints
    pub fn clear_checkpoints(&mut self) {
        self.checkpoints.clear();
        self.current_memory = 0;
    }

    /// Get checkpoint statistics
    pub fn get_stats(&self) -> CheckpointStats {
        CheckpointStats {
            num_checkpoints: self.checkpoints.len(),
            total_memory_usage: self.current_memory,
            average_checkpoint_size: if self.checkpoints.is_empty() {
                0
            } else {
                self.current_memory / self.checkpoints.len()
            },
            oldest_checkpoint_age: self.get_oldest_checkpoint_age(),
        }
    }

    /// Remove oldest checkpoint
    fn remove_oldest_checkpoint(&mut self) {
        if let Some((oldest_id, oldest_checkpoint)) = self
            .checkpoints
            .iter()
            .min_by_key(|(_, checkpoint)| checkpoint.timestamp)
            .map(|(id, checkpoint)| (id.clone(), checkpoint.clone()))
        {
            self.current_memory -= oldest_checkpoint.memory_usage;
            self.checkpoints.remove(&oldest_id);
        }
    }

    /// Estimate memory usage of a checkpoint
    fn estimate_checkpoint_memory(
        &self,
        parameters: &Array1<T>,
        gradients: &Array1<T>,
        hessian: &Option<Array2<T>>,
    ) -> usize {
        let param_size = parameters.len() * std::mem::size_of::<T>();
        let grad_size = gradients.len() * std::mem::size_of::<T>();
        let hessian_size = if let Some(h) = hessian {
            h.len() * std::mem::size_of::<T>()
        } else {
            0
        };

        param_size + grad_size + hessian_size + std::mem::size_of::<MetaCheckpoint<T>>()
    }

    /// Get age of oldest checkpoint
    fn get_oldest_checkpoint_age(&self) -> std::time::Duration {
        self.checkpoints
            .values()
            .map(|checkpoint| checkpoint.timestamp.elapsed())
            .max()
            .unwrap_or(std::time::Duration::from_secs(0))
    }
}

/// Checkpoint statistics
#[derive(Debug, Clone)]
pub struct CheckpointStats {
    pub num_checkpoints: usize,
    pub total_memory_usage: usize,
    pub average_checkpoint_size: usize,
    pub oldest_checkpoint_age: std::time::Duration,
}

/// Meta-learning statistics
#[derive(Debug, Clone)]
pub struct MetaLearningStats {
    pub algorithm: MetaLearningAlgorithm,
    pub cache_size: usize,
    pub history_length: usize,
    pub memory_usage: usize,
    pub total_tasks_processed: usize,
}

/// Default configurations
impl Default for InnerLoopConfig {
    fn default() -> Self {
        Self {
            num_steps: 5,
            learning_rate: 0.01,
            optimizer: InnerOptimizer::SGD,
            stop_condition: StopCondition::FixedSteps,
            gradient_checkpointing: false,
            max_steps: 100,
        }
    }
}

impl Default for OuterLoopConfig {
    fn default() -> Self {
        Self {
            meta_learning_rate: 0.001,
            meta_optimizer: MetaOptimizer::Adam,
            tasks_per_batch: 16,
            second_order: true,
            use_implicit_function_theorem: false,
            meta_regularization: 0.0,
        }
    }
}

impl Default for AdvancedMetaConfig {
    fn default() -> Self {
        Self {
            use_higher_order_hessian: true,
            auto_hvp_mode: true,
            enable_gradient_checkpointing: true,
            max_memory_usage: 1024 * 1024 * 1024, // 1GB
            enable_hyperparameter_optimization: false,
            lr_adaptation_method: LearningRateAdaptation::Fixed,
            enable_multi_task_learning: true,
            task_similarity_metric: TaskSimilarityMetric::GradientCosine,
            meta_batch_sampling: MetaBatchSampling::Random,
            enable_curriculum_learning: false,
            regularization: MetaRegularization::default(),
        }
    }
}

impl Default for MetaRegularization {
    fn default() -> Self {
        Self {
            l2_meta_params: 0.0,
            adaptation_regularization: 0.0,
            entropy_regularization: 0.0,
            diversity_bonus: 0.0,
            gradient_penalty: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_gradient_engine_creation() {
        let inner_config = InnerLoopConfig::default();
        let outer_config = OuterLoopConfig::default();
        let engine =
            MetaGradientEngine::<f64>::new(MetaLearningAlgorithm::MAML, inner_config, outer_config);

        assert!(matches!(engine.algorithm, MetaLearningAlgorithm::MAML));
    }

    #[test]
    fn test_meta_task_creation() {
        let support_set = vec![
            (
                Array1::from_vec(vec![1.0, 2.0]),
                Array1::from_vec(vec![3.0]),
            ),
            (
                Array1::from_vec(vec![2.0, 3.0]),
                Array1::from_vec(vec![5.0]),
            ),
        ];

        let query_set = vec![(
            Array1::from_vec(vec![3.0, 4.0]),
            Array1::from_vec(vec![7.0]),
        )];

        let task = MetaTask {
            id: "test_task".to_string(),
            support_set,
            query_set,
            task_params: HashMap::new(),
            weight: 1.0,
        };

        assert_eq!(task.id, "test_task");
        assert_eq!(task.support_set.len(), 2);
        assert_eq!(task.query_set.len(), 1);
    }

    #[test]
    fn test_inner_loop_config_default() {
        let config = InnerLoopConfig::default();
        assert_eq!(config.num_steps, 5);
        assert_eq!(config.learning_rate, 0.01);
        assert!(matches!(config.optimizer, InnerOptimizer::SGD));
    }

    #[test]
    fn test_outer_loop_config_default() {
        let config = OuterLoopConfig::default();
        assert_eq!(config.meta_learning_rate, 0.001);
        assert!(matches!(config.meta_optimizer, MetaOptimizer::Adam));
        assert_eq!(config.tasks_per_batch, 16);
        assert!(config.second_order);
    }

    #[test]
    fn test_stop_condition_convergence() {
        let condition = StopCondition::Convergence { threshold: 1e-6 };

        match condition {
            StopCondition::Convergence { threshold } => {
                assert_eq!(threshold, 1e-6);
            }
            _ => panic!("Wrong stop condition type"),
        }
    }

    #[test]
    fn test_meta_learning_algorithm_types() {
        let algorithms = [
            MetaLearningAlgorithm::MAML,
            MetaLearningAlgorithm::FirstOrderMAML,
            MetaLearningAlgorithm::Reptile,
            MetaLearningAlgorithm::L2L,
            MetaLearningAlgorithm::MetaSGD,
            MetaLearningAlgorithm::LearnedOptimizer,
            MetaLearningAlgorithm::ImplicitFunctionTheorem,
        ];

        assert_eq!(algorithms.len(), 7);
    }
}
