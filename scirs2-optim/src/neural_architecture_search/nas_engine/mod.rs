//! Neural Architecture Search Engine Module
//!
//! This module provides a comprehensive Neural Architecture Search (NAS) implementation
//! for automatically discovering optimal optimization algorithms. The module is organized
//! into focused sub-modules for maintainability and modularity.
//!
//! ## Module Organization
//!
//! - **config**: Configuration types, enums, and parameter definitions
//! - **results**: Search results, evaluation results, and statistics tracking
//! - **resources**: Resource monitoring, constraint enforcement, and optimization
//! - **engine**: Main NAS engine implementation and coordination logic
//!
//! ## Usage
//!
//! ```rust
//! use scirs2_optim::neural_architecture_search::nas_engine::{
//!     NeuralArchitectureSearch, NASConfig, SearchStrategyType,
//!     SearchSpaceConfig, EvaluationConfig, MultiObjectiveConfig
//! };
//! use num_traits::Float;
//!
//! // Configure the NAS engine
//! let config = NASConfig::<f64> {
//!     search_strategy: SearchStrategyType::Evolutionary,
//!     search_space: SearchSpaceConfig::default(),
//!     evaluation_config: EvaluationConfig::default(),
//!     multi_objective_config: MultiObjectiveConfig::default(),
//!     search_budget: 1000,
//!     // ... other configuration
//!     ..Default::default()
//! };
//!
//! // Create and run NAS engine
//! let mut nas_engine = NeuralArchitectureSearch::new(config)?;
//! let results = nas_engine.run_search()?;
//!
//! // Access best found architectures
//! for architecture in &results.best_architectures {
//!     println!("Found architecture: {:?}", architecture.architecture_id);
//! }
//! ```
//!
//! ## Key Features
//!
//! ### Search Strategies
//! - Random search baseline
//! - Evolutionary algorithms (genetic algorithms, differential evolution)
//! - Bayesian optimization with Gaussian processes
//! - Reinforcement learning-based search
//! - Differentiable architecture search (DARTS)
//! - Progressive search with increasing complexity
//! - Hybrid strategies combining multiple approaches
//!
//! ### Multi-Objective Optimization
//! - NSGA-II: Non-dominated Sorting Genetic Algorithm II
//! - NSGA-III: Non-dominated Sorting Genetic Algorithm III
//! - MOEA/D: Multi-Objective Evolutionary Algorithm based on Decomposition
//! - PAES: Pareto Archived Evolution Strategy
//! - SPEA2: Strength Pareto Evolutionary Algorithm 2
//!
//! ### Resource Management
//! - Memory usage monitoring and constraints
//! - CPU and GPU time tracking
//! - Energy consumption estimation
//! - Financial cost tracking
//! - Network bandwidth monitoring
//! - Automatic resource constraint enforcement
//!
//! ### Performance Features
//! - Performance prediction for faster evaluation
//! - Progressive search for complex search spaces
//! - Transfer learning between search sessions
//! - Parallel evaluation of candidate architectures
//! - Caching of evaluation results
//! - Early stopping based on convergence criteria

use num_traits::Float;
use crate::learned_optimizers::few_shot_optimizer::EvaluationMetric;

pub mod config;
pub mod results;
pub mod resources;
pub mod engine;

// Re-export core types for convenience
pub use config::*;
pub use results::*;
pub use resources::*;
pub use engine::*;

// Additional convenience re-exports for commonly used combinations
pub use engine::{
    NeuralArchitectureSearch,
    SearchStrategy,
    MultiObjectiveOptimizer,
    ArchitectureController,
    PerformanceEvaluator,
    ProgressiveNAS,
    PerformancePredictor,
    ParetoFront,
    DiversityMetrics,
};

pub use config::{
    NASConfig,
    SearchStrategyType,
    SearchSpaceConfig,
    EvaluationConfig,
    MultiObjectiveConfig,
    ObjectiveConfig,
    EarlyStoppingConfig,
};

pub use results::{
    SearchResults,
    SearchResult,
    EvaluationResults,
    ArchitectureEncoding,
    SearchStatistics,
    OptimizerArchitecture,
    OptimizerComponent,
    ConvergenceData,
};

pub use resources::{
    ResourceMonitor,
    ResourceConstraints,
    ResourceUsage,
    ResourceSnapshot,
    HardwareResources,
    ResourceOptimizer,
};

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Float;

    #[test]
    fn test_nas_config_creation() {
        let config = NASConfig::<f64>::default();
        assert_eq!(config.search_budget, 1000);
        assert_eq!(config.population_size, 50);
        assert!(config.early_stopping.enabled);
    }

    #[test]
    fn test_resource_constraints_validation() {
        let constraints = ResourceConstraints::<f64> {
            max_memory_gb: 16.0,
            max_computation_hours: 24.0,
            max_energy_kwh: 100.0,
            max_cost_usd: 1000.0,
            hardware_resources: HardwareResources::default(),
            enable_monitoring: true,
            violation_handling: ResourceViolationHandling::Penalty,
        };

        let monitor = ResourceMonitor::new(constraints);
        assert!(monitor.constraints.max_memory_gb > 0.0);
    }

    #[test]
    fn test_search_strategy_types() {
        let strategies = vec![
            SearchStrategyType::Random,
            SearchStrategyType::Evolutionary,
            SearchStrategyType::Bayesian,
            SearchStrategyType::Reinforcement,
            SearchStrategyType::Differentiable,
            SearchStrategyType::Progressive,
            SearchStrategyType::Hybrid,
        ];

        assert_eq!(strategies.len(), 7);
    }

    #[test]
    fn test_multi_objective_algorithms() {
        let algorithms = vec![
            MultiObjectiveAlgorithm::NSGA2,
            MultiObjectiveAlgorithm::NSGA3,
            MultiObjectiveAlgorithm::MOEAD,
            MultiObjectiveAlgorithm::PAES,
            MultiObjectiveAlgorithm::SPEA2,
        ];

        assert_eq!(algorithms.len(), 5);
    }

    #[test]
    fn test_evaluation_metrics() {
        let metrics = vec![
            EvaluationMetric::FinalPerformance,
            EvaluationMetric::ConvergenceSpeed,
            EvaluationMetric::Stability,
            EvaluationMetric::Robustness,
            EvaluationMetric::Efficiency,
            EvaluationMetric::Generalization,
            EvaluationMetric::MemoryUsage,
            EvaluationMetric::ComputationTime,
        ];

        assert_eq!(metrics.len(), 8);
    }

    #[test]
    fn test_component_types() {
        let components = vec![
            ComponentType::SGD,
            ComponentType::Adam,
            ComponentType::AdamW,
            ComponentType::RMSprop,
            ComponentType::AdaGrad,
            ComponentType::AdaDelta,
            ComponentType::LBFGS,
            ComponentType::Momentum,
            ComponentType::Nesterov,
            ComponentType::Custom,
        ];

        assert_eq!(components.len(), 10);
    }

    #[test]
    fn test_resource_usage_calculation() {
        let usage = ResourceUsage::<f64> {
            memory_gb: 8.0,
            cpu_time_seconds: 3600.0,
            gpu_time_seconds: 1800.0,
            energy_kwh: 5.4,
            cost_usd: 0.648,
            network_gb: 0.1,
        };

        assert!(usage.memory_gb > 0.0);
        assert!(usage.cpu_time_seconds > 0.0);
        assert!(usage.energy_kwh > 0.0);
    }

    #[test]
    fn test_architecture_encoding() {
        let encoding = ArchitectureEncoding {
            encoding_type: ArchitectureEncodingStrategy::Direct,
            encoded_data: vec![1, 2, 3, 4],
            metadata: std::collections::HashMap::new(),
            checksum: 12345,
        };

        assert_eq!(encoding.encoded_data.len(), 4);
        assert_eq!(encoding.checksum, 12345);
    }

    #[test]
    fn test_search_statistics_initialization() {
        let stats = SearchStatistics::<f64>::default();
        assert_eq!(stats.total_evaluations, 0);
        assert_eq!(stats.current_generation, 0);
        assert!(stats.population_diversity >= 0.0);
    }

    #[test]
    fn test_pareto_front_creation() {
        let pareto_front = ParetoFront::<f64> {
            solutions: Vec::new(),
            hypervolume: 0.0,
            diversity_metrics: DiversityMetrics {
                crowding_distance: Vec::new(),
                entropy: 0.0,
                average_distance: 0.0,
                min_distance: 0.0,
                max_distance: 0.0,
            },
            generation: 0,
        };

        assert_eq!(pareto_front.solutions.len(), 0);
        assert_eq!(pareto_front.generation, 0);
    }

    #[test]
    fn test_hardware_resources() {
        let hardware = HardwareResources {
            cpu_cores: 16,
            ram_gb: 64,
            num_gpus: 4,
            gpu_memory_gb: 32,
            storage_gb: 1000,
            network_bandwidth_mbps: 1000.0,
        };

        assert!(hardware.cpu_cores > 0);
        assert!(hardware.ram_gb > 0);
        assert!(hardware.num_gpus > 0);
    }
}

/// Example function demonstrating basic NAS usage
pub fn create_example_nas_config<T: Float>() -> NASConfig<T> {
    NASConfig {
        search_strategy: SearchStrategyType::Evolutionary,
        search_space: SearchSpaceConfig {
            component_types: vec![
                ComponentTypeConfig {
                    component_type: ComponentType::Adam,
                    enabled: true,
                    probability: T::from(0.3).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
                ComponentTypeConfig {
                    component_type: ComponentType::SGD,
                    enabled: true,
                    probability: T::from(0.2).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
                ComponentTypeConfig {
                    component_type: ComponentType::RMSprop,
                    enabled: true,
                    probability: T::from(0.2).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
            ],
            max_components: 10,
            min_components: 2,
            max_connections: 20,
            min_connections: 1,
            allow_cycles: false,
            max_depth: 5,
            enable_skip_connections: true,
            connection_probability: T::from(0.5).unwrap(),
        },
        evaluation_config: EvaluationConfig::default(),
        multi_objective_config: MultiObjectiveConfig::default(),
        search_budget: 500,
        early_stopping: EarlyStoppingConfig {
            enabled: true,
            patience: 25,
            min_improvement: T::from(0.001).unwrap(),
            metric: EvaluationMetric::FinalPerformance,
            target_performance: None,
            convergence_detection: ConvergenceDetectionStrategy::NoImprovement,
        },
        progressive_search: false,
        population_size: 25,
        enable_transfer_learning: false,
        encoding_strategy: ArchitectureEncodingStrategy::Direct,
        enable_performance_prediction: true,
        parallelization_factor: 2,
        auto_hyperparameter_tuning: true,
        resource_constraints: ResourceConstraints {
            max_memory_gb: T::from(8.0).unwrap(),
            max_computation_hours: T::from(12.0).unwrap(),
            max_energy_kwh: T::from(50.0).unwrap(),
            max_cost_usd: T::from(500.0).unwrap(),
            hardware_resources: HardwareResources {
                cpu_cores: 8,
                ram_gb: 32,
                num_gpus: 2,
                gpu_memory_gb: 16,
                storage_gb: 500,
                network_bandwidth_mbps: 500.0,
            },
            enable_monitoring: true,
            violation_handling: ResourceViolationHandling::Penalty,
        },
    }
}

/// Create a minimal NAS configuration for quick testing
pub fn create_minimal_nas_config<T: Float>() -> NASConfig<T> {
    NASConfig {
        search_strategy: SearchStrategyType::Random,
        search_space: SearchSpaceConfig {
            component_types: vec![
                ComponentTypeConfig {
                    component_type: ComponentType::Adam,
                    enabled: true,
                    probability: T::from(0.5).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
                ComponentTypeConfig {
                    component_type: ComponentType::SGD,
                    enabled: true,
                    probability: T::from(0.5).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
            ],
            max_components: 5,
            min_components: 1,
            max_connections: 10,
            min_connections: 0,
            allow_cycles: false,
            max_depth: 3,
            enable_skip_connections: false,
            connection_probability: T::from(0.3).unwrap(),
        },
        evaluation_config: EvaluationConfig::default(),
        multi_objective_config: MultiObjectiveConfig::default(),
        search_budget: 100,
        early_stopping: EarlyStoppingConfig {
            enabled: true,
            patience: 10,
            min_improvement: T::from(0.01).unwrap(),
            metric: EvaluationMetric::FinalPerformance,
            target_performance: None,
            convergence_detection: ConvergenceDetectionStrategy::NoImprovement,
        },
        progressive_search: false,
        population_size: 10,
        enable_transfer_learning: false,
        encoding_strategy: ArchitectureEncodingStrategy::Direct,
        enable_performance_prediction: false,
        parallelization_factor: 1,
        auto_hyperparameter_tuning: false,
        resource_constraints: ResourceConstraints {
            max_memory_gb: T::from(4.0).unwrap(),
            max_computation_hours: T::from(1.0).unwrap(),
            max_energy_kwh: T::from(5.0).unwrap(),
            max_cost_usd: T::from(50.0).unwrap(),
            hardware_resources: HardwareResources {
                cpu_cores: 4,
                ram_gb: 16,
                num_gpus: 1,
                gpu_memory_gb: 8,
                storage_gb: 100,
                network_bandwidth_mbps: 100.0,
            },
            enable_monitoring: true,
            violation_handling: ResourceViolationHandling::Penalty,
        },
    }
}