//! Configuration types for the Adaptive Neural Architecture Search System
//!
//! This module defines configuration structures and enums that control
//! the behavior of the adaptive NAS system.

use num_traits::Float;
use crate::neural_architecture_search::NASConfig;

/// Configuration for the Adaptive NAS System
#[derive(Debug, Clone)]
pub struct AdaptiveNASConfig<T: Float> {
    /// Base NAS configuration
    pub base_config: NASConfig<T>,

    /// Performance tracking window
    pub performance_window: usize,

    /// Minimum performance improvement threshold
    pub improvement_threshold: T,

    /// Adaptation learning rate
    pub adaptation_lr: T,

    /// Architecture complexity penalty
    pub complexity_penalty: T,

    /// Enable online learning
    pub online_learning: bool,

    /// Enable architecture transfer
    pub architecture_transfer: bool,

    /// Enable curriculum search
    pub curriculum_search: bool,

    /// Search diversity weight
    pub _diversityweight: T,

    /// Exploration vs exploitation balance
    pub exploration_weight: T,

    /// Performance prediction confidence threshold
    pub prediction_confidence_threshold: T,

    /// Maximum architecture complexity
    pub max_complexity: usize,

    /// Minimum architecture performance
    pub _minperformance: T,

    /// Enable meta-learning for architecture search
    pub enable_meta_learning: bool,

    /// Meta-learning update frequency
    pub meta_learning_frequency: usize,

    /// Architecture novelty weight
    pub novelty_weight: T,

    /// Enable progressive search
    pub progressive_search: bool,

    /// Search budget allocation strategy
    pub budget_allocation: BudgetAllocationStrategy,

    /// Quality assessment criteria
    pub quality_criteria: Vec<QualityCriterion>,
}

impl<T: Float> Default for AdaptiveNASConfig<T> {
    fn default() -> Self {
        Self {
            base_config: NASConfig::default(),
            performance_window: 100,
            improvement_threshold: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            adaptation_lr: num_traits::cast::cast(0.001).unwrap_or_else(|| T::zero()),
            complexity_penalty: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
            online_learning: true,
            architecture_transfer: true,
            curriculum_search: false,
            _diversityweight: num_traits::cast::cast(0.2).unwrap_or_else(|| T::zero()),
            exploration_weight: num_traits::cast::cast(0.3).unwrap_or_else(|| T::zero()),
            prediction_confidence_threshold: num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()),
            max_complexity: 1_000_000,
            _minperformance: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
            enable_meta_learning: true,
            meta_learning_frequency: 50,
            novelty_weight: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
            progressive_search: true,
            budget_allocation: BudgetAllocationStrategy::ExpectedImprovement,
            quality_criteria: vec![
                QualityCriterion::ValidationPerformance,
                QualityCriterion::ConvergenceSpeed,
                QualityCriterion::ComputationalEfficiency,
            ],
        }
    }
}

/// Budget allocation strategies for search resource management
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetAllocationStrategy {
    /// Equal allocation
    Uniform,

    /// Performance-based allocation
    PerformanceBased,

    /// Uncertainty-based allocation
    UncertaintyBased,

    /// Expected improvement
    ExpectedImprovement,

    /// Upper confidence bound
    UpperConfidenceBound,

    /// Thompson sampling
    ThompsonSampling,
}

/// Quality criteria for architecture assessment
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum QualityCriterion {
    /// Performance on validation set
    ValidationPerformance,

    /// Convergence speed
    ConvergenceSpeed,

    /// Computational efficiency
    ComputationalEfficiency,

    /// Memory efficiency
    MemoryEfficiency,

    /// Generalization ability
    GeneralizationAbility,

    /// Robustness to hyperparameters
    RobustnessToHyperparams,

    /// Transfer learning capability
    TransferLearningCapability,
}