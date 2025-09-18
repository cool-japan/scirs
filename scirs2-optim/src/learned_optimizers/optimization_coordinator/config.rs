//! Configuration structures for optimization coordinator

use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use serde::{Serialize, Deserialize};
use super::adaptation::AdaptationConfig;

/// Advanced configuration for optimization coordinator
#[derive(Debug, Clone)]
pub struct AdvancedConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Enable neural architecture search
    pub enable_nas: bool,

    /// Enable adaptive transformer enhancement
    pub enable_transformer_enhancement: bool,

    /// Enable few-shot learning
    pub enable_few_shot_learning: bool,

    /// Enable meta-learning orchestration
    pub enable_meta_learning: bool,

    /// Maximum parallel optimizers
    pub max_parallel_optimizers: usize,

    /// Performance prediction horizon
    pub prediction_horizon: usize,

    /// Adaptation threshold
    pub adaptation_threshold: T,

    /// Resource allocation strategy
    pub resource_allocation: ResourceAllocationStrategy,

    /// Optimization objective weights
    pub objective_weights: HashMap<OptimizationObjective, T>,

    /// Enable dynamic reconfiguration
    pub enable_dynamic_reconfiguration: bool,

    /// Enable advanced analytics
    pub enable_advanced_analytics: bool,

    /// Cache size limit
    pub cache_size_limit: usize,

    /// Adaptation configuration
    pub adaptation_config: AdaptationConfig<T>,
}

impl<T: Float> Default for AdvancedConfig<T> {
    fn default() -> Self {
        let mut objective_weights = HashMap::new();
        objective_weights.insert(
            OptimizationObjective::ConvergenceSpeed,
            num_traits::cast::cast(0.3).unwrap_or_else(|| T::zero()),
        );
        objective_weights.insert(
            OptimizationObjective::SolutionQuality,
            num_traits::cast::cast(0.4).unwrap_or_else(|| T::zero()),
        );
        objective_weights.insert(
            OptimizationObjective::ResourceEfficiency,
            num_traits::cast::cast(0.2).unwrap_or_else(|| T::zero()),
        );
        objective_weights.insert(OptimizationObjective::Robustness, num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()));

        Self {
            enable_nas: true,
            enable_transformer_enhancement: true,
            enable_few_shot_learning: true,
            enable_meta_learning: true,
            max_parallel_optimizers: 8,
            prediction_horizon: 100,
            adaptation_threshold: num_traits::cast::cast(0.05).unwrap_or_else(|| T::zero()),
            resource_allocation: ResourceAllocationStrategy::Adaptive,
            objective_weights,
            enable_dynamic_reconfiguration: true,
            enable_advanced_analytics: true,
            cache_size_limit: 10000,
            adaptation_config: AdaptationConfig::default(),
        }
    }
}

/// Resource allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceAllocationStrategy {
    /// Uniform allocation across all optimizers
    Uniform,
    /// Performance-weighted allocation
    PerformanceWeighted,
    /// Adaptive allocation based on current context
    Adaptive,
    /// Priority-based allocation
    PriorityBased,
    /// Dynamic load balancing
    DynamicLoadBalancing,
}

/// Optimization objectives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationObjective {
    /// Convergence speed
    ConvergenceSpeed,
    /// Solution quality
    SolutionQuality,
    /// Resource efficiency
    ResourceEfficiency,
    /// Robustness to noise
    Robustness,
    /// Generalization capability
    Generalization,
    /// Memory efficiency
    MemoryEfficiency,
}

/// Ensemble strategy types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnsembleStrategy {
    /// Weighted voting based on performance
    WeightedVoting,
    /// Stacking with meta-learner
    Stacking,
    /// Boosting approach
    Boosting,
    /// Bagging approach
    Bagging,
    /// Dynamic selection
    DynamicSelection,
    /// Mixture of experts
    MixtureOfExperts,
}

/// Optimizer selection algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerSelectionAlgorithm {
    /// Multi-armed bandit approach
    MultiArmedBandit,
    /// Upper confidence bound
    UpperConfidenceBound,
    /// Thompson sampling
    ThompsonSampling,
    /// Epsilon-greedy
    EpsilonGreedy,
    /// Portfolio approach
    Portfolio,
    /// Contextual bandits
    ContextualBandits,
}

/// Problem types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemType {
    /// Convex optimization
    Convex,
    /// Non-convex optimization
    NonConvex,
    /// Multi-objective optimization
    MultiObjective,
    /// Constrained optimization
    Constrained,
    /// Stochastic optimization
    Stochastic,
    /// Black-box optimization
    BlackBox,
    /// Large-scale optimization
    LargeScale,
    /// Online optimization
    Online,
}

/// Scalability characteristics
#[derive(Debug, Clone)]
pub struct ScalabilityInfo {
    /// Maximum parameter count supported
    pub max_parameters: usize,
    /// Memory complexity
    pub memory_complexity: ComplexityClass,
    /// Time complexity
    pub time_complexity: ComplexityClass,
    /// Parallel scalability
    pub parallel_scalability: ParallelScalability,
}

/// Complexity classes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexityClass {
    /// Constant O(1)
    Constant,
    /// Logarithmic O(log n)
    Logarithmic,
    /// Linear O(n)
    Linear,
    /// Linearithmic O(n log n)
    Linearithmic,
    /// Quadratic O(n²)
    Quadratic,
    /// Cubic O(n³)
    Cubic,
    /// Exponential O(2^n)
    Exponential,
}

/// Parallel scalability characteristics
#[derive(Debug, Clone)]
pub struct ParallelScalability {
    /// Maximum effective threads
    pub max_threads: usize,
    /// Parallel efficiency
    pub efficiency: f64,
    /// Communication overhead
    pub communication_overhead: f64,
}

/// Optimization context
#[derive(Debug, Clone)]
pub struct OptimizationContext<T: Float> {
    /// Problem characteristics
    pub problem_type: ProblemType,
    /// Dimensionality
    pub dimensionality: usize,
    /// Current iteration
    pub iteration: usize,
    /// Available computational budget
    pub computational_budget: ComputationalBudget,
    /// Convergence criteria
    pub convergence_criteria: ConvergenceCriteria<T>,
    /// Historical performance
    pub historical_performance: Vec<T>,
    /// Problem-specific metadata
    pub metadata: HashMap<String, String>,
}


/// Computational budget
#[derive(Debug, Clone)]
pub struct ComputationalBudget {
    /// Maximum function evaluations
    pub max_evaluations: usize,
    /// Maximum wall-clock time
    pub max_time: std::time::Duration,
    /// Maximum memory usage (in bytes)
    pub max_memory: usize,
    /// Available CPU cores
    pub available_cores: usize,
}


/// Convergence criteria
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria<T: Float> {
    /// Tolerance for function value changes
    pub function_tolerance: T,
    /// Tolerance for parameter changes
    pub parameter_tolerance: T,
    /// Tolerance for gradient norm
    pub gradient_tolerance: T,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Consecutive improvement threshold
    pub stagnation_threshold: usize,
}


/// Optimization phases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationPhase {
    /// Initialization phase
    Initialization,
    /// Exploration phase
    Exploration,
    /// Exploitation phase
    Exploitation,
    /// Refinement phase
    Refinement,
    /// Convergence phase
    Convergence,
    /// Finalization phase
    Finalization,
}

/// Adaptation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdaptationType {
    /// Learning rate adaptation
    LearningRate,
    /// Optimizer selection adaptation
    OptimizerSelection,
    /// Ensemble weight adaptation
    EnsembleWeights,
    /// Resource allocation adaptation
    ResourceAllocation,
    /// Strategy adaptation
    Strategy,
    /// Architecture adaptation
    Architecture,
}

/// Meta-learning schedule types
#[derive(Debug, Clone)]
pub enum MetaLearningSchedule {
    /// Fixed schedule
    Fixed {
        /// Episodes per meta-update
        episodes_per_update: usize,
    },
    /// Adaptive schedule
    Adaptive {
        /// Performance threshold for updates
        performance_threshold: f64,
        /// Minimum episodes between updates
        min_episodes: usize,
        /// Maximum episodes between updates
        max_episodes: usize,
    },
    /// Event-driven schedule
    EventDriven {
        /// Trigger events
        trigger_events: Vec<String>,
    },
}

/// Configuration validation
impl<T: Float> AdvancedConfig<T> {
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.max_parallel_optimizers == 0 {
            return Err("max_parallel_optimizers must be greater than 0".to_string());
        }

        if self.prediction_horizon == 0 {
            return Err("prediction_horizon must be greater than 0".to_string());
        }

        if self.adaptation_threshold < T::zero() {
            return Err("adaptation_threshold must be non-negative".to_string());
        }

        if self.cache_size_limit == 0 {
            return Err("cache_size_limit must be greater than 0".to_string());
        }

        // Validate objective weights sum to 1
        let weight_sum: T = self.objective_weights.values().fold(T::zero(), |acc, &w| acc + w);
        let expected_sum = T::one();
        let tolerance = num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero());

        if (weight_sum - expected_sum).abs() > tolerance {
            return Err("objective_weights must sum to 1.0".to_string());
        }

        // Validate individual weights are non-negative
        for (&objective, &weight) in &self.objective_weights {
            if weight < T::zero() {
                return Err(format!("Weight for {:?} must be non-negative", objective));
            }
        }

        Ok(())
    }

    /// Create configuration for different use cases
    pub fn for_research() -> Self {
        let mut config = Self::default();
        config.enable_advanced_analytics = true;
        config.enable_dynamic_reconfiguration = true;
        config.max_parallel_optimizers = 16;
        config.prediction_horizon = 200;
        config
    }

    pub fn for_production() -> Self {
        let mut config = Self::default();
        config.enable_advanced_analytics = false;
        config.enable_dynamic_reconfiguration = false;
        config.max_parallel_optimizers = 4;
        config.prediction_horizon = 50;
        config.cache_size_limit = 5000;
        config
    }

    pub fn for_resource_constrained() -> Self {
        let mut config = Self::default();
        config.enable_nas = false;
        config.enable_transformer_enhancement = false;
        config.enable_few_shot_learning = false;
        config.max_parallel_optimizers = 2;
        config.prediction_horizon = 20;
        config.cache_size_limit = 1000;
        config
    }
}

impl Default for ComputationalBudget {
    fn default() -> Self {
        Self {
            max_evaluations: 10000,
            max_time: std::time::Duration::from_secs(3600), // 1 hour
            max_memory: 1024 * 1024 * 1024, // 1 GB
            available_cores: num_cpus::get(),
        }
    }
}

impl<T: Float> Default for ConvergenceCriteria<T> {
    fn default() -> Self {
        Self {
            function_tolerance: num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero()),
            parameter_tolerance: num_traits::cast::cast(1e-8).unwrap_or_else(|| T::zero()),
            gradient_tolerance: num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero()),
            max_iterations: 1000,
            stagnation_threshold: 50,
        }
    }
}

impl<T: Float> Default for OptimizationContext<T> {
    fn default() -> Self {
        Self {
            problem_type: ProblemType::NonConvex,
            dimensionality: 100,
            iteration: 0,
            computational_budget: ComputationalBudget::default(),
            convergence_criteria: ConvergenceCriteria::default(),
            historical_performance: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

impl Default for ScalabilityInfo {
    fn default() -> Self {
        Self {
            max_parameters: 1_000_000,
            memory_complexity: ComplexityClass::Linear,
            time_complexity: ComplexityClass::Linear,
            parallel_scalability: ParallelScalability {
                max_threads: num_cpus::get(),
                efficiency: 0.8,
                communication_overhead: 0.1,
            },
        }
    }
}

impl Default for MetaLearningSchedule {
    fn default() -> Self {
        Self::Adaptive {
            performance_threshold: 0.01,
            min_episodes: 5,
            max_episodes: 50,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = AdvancedConfig::<f32>::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let mut config = AdvancedConfig::<f32>::default();
        config.max_parallel_optimizers = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_preset_configs() {
        let research_config = AdvancedConfig::<f32>::for_research();
        assert!(research_config.validate().is_ok());
        assert_eq!(research_config.max_parallel_optimizers, 16);

        let production_config = AdvancedConfig::<f32>::for_production();
        assert!(production_config.validate().is_ok());
        assert_eq!(production_config.max_parallel_optimizers, 4);

        let constrained_config = AdvancedConfig::<f32>::for_resource_constrained();
        assert!(constrained_config.validate().is_ok());
        assert!(!constrained_config.enable_nas);
    }

    #[test]
    fn test_objective_weights_sum() {
        let config = AdvancedConfig::<f64>::default();
        let weight_sum: f64 = config.objective_weights.values().sum();
        assert!((weight_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_computational_budget() {
        let budget = ComputationalBudget::default();
        assert!(budget.max_evaluations > 0);
        assert!(budget.max_time.as_secs() > 0);
        assert!(budget.available_cores > 0);
    }
}