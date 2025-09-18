//! Neural Architecture Search for Optimization Algorithms
//!
//! This module implements Neural Architecture Search (NAS) for automatically discovering
//! optimal optimization algorithms. The NAS system can search through various optimizer
//! architectures to find the best-performing configurations for specific problem domains.
//!
//! ## Overview
//!
//! Neural Architecture Search is a powerful technique that automates the discovery of
//! neural network architectures. In this implementation, we extend NAS to the domain
//! of optimization algorithms, allowing automatic discovery of novel optimizer designs
//! that can outperform traditional hand-crafted optimizers.
//!
//! ## Features
//!
//! - **Multiple Search Strategies**: Random, evolutionary, Bayesian optimization,
//!   reinforcement learning, differentiable, progressive, and hybrid approaches
//! - **Multi-Objective Optimization**: Support for optimizing multiple objectives
//!   simultaneously using algorithms like NSGA-II, NSGA-III, MOEA/D, PAES, and SPEA2
//! - **Resource Management**: Comprehensive monitoring and constraint enforcement
//!   for memory, computation time, energy, and financial costs
//! - **Performance Prediction**: Optional performance prediction to accelerate search
//! - **Progressive Search**: Gradual increase in search space complexity
//! - **Transfer Learning**: Knowledge transfer between search sessions
//!
//! ## Module Structure
//!
//! The implementation is organized into focused modules:
//!
//! - **nas_engine**: Core NAS engine implementation
//!   - `config`: Configuration types and parameter definitions
//!   - `results`: Search results and statistics tracking
//!   - `resources`: Resource monitoring and management
//!   - `engine`: Main coordination logic and algorithms
//!
//! ## Usage Example
//!
//! ```rust
//! use scirs2_optim::neural_architecture_search::{
//!     NeuralArchitectureSearch, NASConfig, SearchStrategyType,
//!     create_example_nas_config
//! };
//!
//! // Create a configuration for the NAS engine
//! let config = create_example_nas_config::<f64>();
//!
//! // Initialize the NAS engine
//! let mut nas_engine = NeuralArchitectureSearch::new(config)?;
//!
//! // Run the architecture search
//! let results = nas_engine.run_search()?;
//!
//! // Examine the best found architectures
//! println!("Found {} best architectures", results.best_architectures.len());
//! for (i, architecture) in results.best_architectures.iter().enumerate() {
//!     println!("Architecture {}: {}", i + 1, architecture.architecture_id);
//!     println!("  Components: {}", architecture.components.len());
//!     println!("  Connections: {}", architecture.connections.len());
//! }
//!
//! // Check search statistics
//! println!("Total evaluations: {}", results.search_statistics.total_evaluations);
//! println!("Final diversity: {:.6}", results.search_statistics.population_diversity);
//! println!("Best score: {:.6}", results.search_statistics.best_score);
//! ```
//!
//! ## Configuration
//!
//! The NAS system is highly configurable through the `NASConfig` struct:
//!
//! ```rust
//! use scirs2_optim::neural_architecture_search::*;
//!
//! let config = NASConfig::<f64> {
//!     // Search strategy selection
//!     search_strategy: SearchStrategyType::Evolutionary,
//!
//!     // Search space definition
//!     search_space: SearchSpaceConfig {
//!         component_types: vec![
//!             ComponentTypeConfig {
//!                 component_type: ComponentType::Adam,
//!                 enabled: true,
//!                 probability: 0.3,
//!                 // ... other parameters
//!             },
//!             // ... more component types
//!         ],
//!         max_components: 10,
//!         min_components: 2,
//!         // ... other search space parameters
//!     },
//!
//!     // Evaluation configuration
//!     evaluation_config: EvaluationConfig {
//!         metrics: vec![EvaluationMetric::FinalPerformance, EvaluationMetric::Efficiency],
//!         num_trials: 5,
//!         max_iterations: 1000,
//!         // ... other evaluation parameters
//!     },
//!
//!     // Resource constraints
//!     resource_constraints: ResourceConstraints {
//!         max_memory_gb: 16.0,
//!         max_computation_hours: 24.0,
//!         max_energy_kwh: 100.0,
//!         max_cost_usd: 1000.0,
//!         // ... other resource limits
//!     },
//!
//!     // Search budget and stopping criteria
//!     search_budget: 1000,
//!     early_stopping: EarlyStoppingConfig {
//!         enabled: true,
//!         patience: 50,
//!         min_improvement: 0.01,
//!         // ... other stopping criteria
//!     },
//!
//!     // ... other configuration options
//!     ..Default::default()
//! };
//! ```
//!
//! ## Search Strategies
//!
//! ### Random Search
//! A baseline approach that randomly samples architectures from the search space.
//! Useful for establishing performance baselines and exploring diverse solutions.
//!
//! ### Evolutionary Search
//! Uses genetic algorithms to evolve populations of architectures over generations.
//! Supports mutation, crossover, and selection operators optimized for optimizer architectures.
//!
//! ### Bayesian Optimization
//! Employs Gaussian processes to model the relationship between architecture
//! parameters and performance, enabling efficient search through informed sampling.
//!
//! ### Reinforcement Learning
//! Trains an agent to generate architectures, using performance feedback to
//! improve the generation policy over time.
//!
//! ### Differentiable Search (DARTS)
//! Uses gradient-based optimization to search through a continuous relaxation
//! of the architecture search space.
//!
//! ### Progressive Search
//! Gradually increases search space complexity, starting with simple architectures
//! and progressively adding more sophisticated components.
//!
//! ### Hybrid Strategies
//! Combines multiple search approaches to leverage their complementary strengths.
//!
//! ## Multi-Objective Optimization
//!
//! The NAS system supports optimizing multiple objectives simultaneously:
//!
//! - **Performance**: Final optimization performance on target problems
//! - **Efficiency**: Computational and memory efficiency
//! - **Robustness**: Stability across different problem instances
//! - **Convergence Speed**: Rate of convergence to optimal solutions
//! - **Generalization**: Performance across diverse problem domains
//!
//! ## Resource Management
//!
//! Comprehensive resource monitoring and constraint enforcement:
//!
//! - **Memory Monitoring**: Track RAM and GPU memory usage
//! - **Computation Time**: Monitor CPU and GPU computation time
//! - **Energy Consumption**: Estimate and track energy usage
//! - **Financial Costs**: Track cloud computing and electricity costs
//! - **Network Usage**: Monitor data transfer and bandwidth usage
//!
//! ## Performance Optimization
//!
//! Several features help accelerate the search process:
//!
//! - **Performance Prediction**: Train models to predict architecture performance
//! - **Early Termination**: Stop poor-performing evaluations early
//! - **Parallel Evaluation**: Evaluate multiple architectures simultaneously
//! - **Result Caching**: Cache evaluation results to avoid re-computation
//! - **Progressive Complexity**: Start simple and gradually increase complexity
//!
//! ## Architecture Representation
//!
//! Optimizer architectures are represented as directed acyclic graphs (DAGs) where:
//!
//! - **Nodes** represent optimizer components (e.g., Adam, SGD, momentum)
//! - **Edges** represent data flow and component connections
//! - **Hyperparameters** configure individual components
//! - **Constraints** ensure valid and efficient architectures
//!
//! ## Extensibility
//!
//! The system is designed for extensibility:
//!
//! - **Custom Components**: Add new optimizer component types
//! - **Custom Strategies**: Implement new search strategies
//! - **Custom Metrics**: Define domain-specific evaluation metrics
//! - **Custom Constraints**: Add problem-specific architecture constraints
//! - **Custom Predictors**: Implement specialized performance prediction models

use crate::learned_optimizers::few_shot_optimizer::EvaluationMetric;

pub mod nas_engine;

// Re-export everything from the nas_engine module for convenience
pub use nas_engine::*;

// Additional utility functions and examples
pub use nas_engine::{
    create_example_nas_config,
    create_minimal_nas_config,
};

/// Create a comprehensive NAS configuration for production use
pub fn create_production_nas_config<T: num_traits::Float>() -> NASConfig<T> {
    NASConfig {
        search_strategy: SearchStrategyType::Hybrid,
        search_space: SearchSpaceConfig {
            component_types: vec![
                ComponentTypeConfig {
                    component_type: ComponentType::Adam,
                    enabled: true,
                    probability: T::from(0.25).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
                ComponentTypeConfig {
                    component_type: ComponentType::AdamW,
                    enabled: true,
                    probability: T::from(0.25).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
                ComponentTypeConfig {
                    component_type: ComponentType::SGD,
                    enabled: true,
                    probability: T::from(0.15).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
                ComponentTypeConfig {
                    component_type: ComponentType::RMSprop,
                    enabled: true,
                    probability: T::from(0.15).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
                ComponentTypeConfig {
                    component_type: ComponentType::Momentum,
                    enabled: true,
                    probability: T::from(0.1).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
                ComponentTypeConfig {
                    component_type: ComponentType::Nesterov,
                    enabled: true,
                    probability: T::from(0.1).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
            ],
            max_components: 15,
            min_components: 3,
            max_connections: 30,
            min_connections: 2,
            allow_cycles: false,
            max_depth: 7,
            enable_skip_connections: true,
            connection_probability: T::from(0.6).unwrap(),
        },
        evaluation_config: EvaluationConfig {
            metrics: vec![
                EvaluationMetric::FinalPerformance,
                EvaluationMetric::ConvergenceSpeed,
                EvaluationMetric::Stability,
                EvaluationMetric::Efficiency,
                EvaluationMetric::Robustness,
            ],
            num_trials: 10,
            max_iterations: 2000,
            convergence_threshold: T::from(1e-6).unwrap(),
            stability_window: 100,
            timeout_seconds: 3600.0,
            parallel_trials: true,
            cache_results: true,
            early_termination: true,
            confidence_level: T::from(0.95).unwrap(),
            statistical_significance: true,
        },
        multi_objective_config: MultiObjectiveConfig {
            objectives: vec![
                ObjectiveConfig {
                    name: "performance".to_string(),
                    objective_type: ObjectiveType::Performance,
                    direction: OptimizationDirection::Maximize,
                    weight: T::from(0.4).unwrap(),
                    priority: ObjectivePriority::High,
                    tolerance: Some(T::from(0.01).unwrap()),
                },
                ObjectiveConfig {
                    name: "efficiency".to_string(),
                    objective_type: ObjectiveType::Efficiency,
                    direction: OptimizationDirection::Maximize,
                    weight: T::from(0.3).unwrap(),
                    priority: ObjectivePriority::Medium,
                    tolerance: Some(T::from(0.05).unwrap()),
                },
                ObjectiveConfig {
                    name: "robustness".to_string(),
                    objective_type: ObjectiveType::Robustness,
                    direction: OptimizationDirection::Maximize,
                    weight: T::from(0.3).unwrap(),
                    priority: ObjectivePriority::Medium,
                    tolerance: Some(T::from(0.05).unwrap()),
                },
            ],
            algorithm: MultiObjectiveAlgorithm::NSGA3,
            pareto_front_size: 100,
            enable_preferences: true,
            user_preferences: Some(UserPreferences::default()),
            diversity_strategy: DiversityStrategy::HyperVolume,
            constraint_handling: ConstraintHandlingMethod::PenaltyFunction,
        },
        search_budget: 2000,
        early_stopping: EarlyStoppingConfig {
            enabled: true,
            patience: 100,
            min_improvement: T::from(0.001).unwrap(),
            metric: EvaluationMetric::FinalPerformance,
            target_performance: None,
            convergence_detection: ConvergenceDetectionStrategy::RelativeImprovement,
        },
        progressive_search: true,
        population_size: 100,
        enable_transfer_learning: true,
        encoding_strategy: ArchitectureEncodingStrategy::Direct,
        enable_performance_prediction: true,
        parallelization_factor: 8,
        auto_hyperparameter_tuning: true,
        resource_constraints: ResourceConstraints {
            max_memory_gb: T::from(32.0).unwrap(),
            max_computation_hours: T::from(48.0).unwrap(),
            max_energy_kwh: T::from(200.0).unwrap(),
            max_cost_usd: T::from(2000.0).unwrap(),
            hardware_resources: HardwareResources {
                cpu_cores: 32,
                ram_gb: 128,
                num_gpus: 8,
                gpu_memory_gb: 80,
                storage_gb: 2000,
                network_bandwidth_mbps: 2000.0,
            },
            enable_monitoring: true,
            violation_handling: ResourceViolationHandling::Penalty,
        },
    }
}

/// Create a research-focused NAS configuration
pub fn create_research_nas_config<T: num_traits::Float>() -> NASConfig<T> {
    NASConfig {
        search_strategy: SearchStrategyType::Bayesian,
        search_space: SearchSpaceConfig {
            component_types: vec![
                ComponentTypeConfig {
                    component_type: ComponentType::Adam,
                    enabled: true,
                    probability: T::from(0.2).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
                ComponentTypeConfig {
                    component_type: ComponentType::AdamW,
                    enabled: true,
                    probability: T::from(0.2).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
                ComponentTypeConfig {
                    component_type: ComponentType::SGD,
                    enabled: true,
                    probability: T::from(0.15).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
                ComponentTypeConfig {
                    component_type: ComponentType::RMSprop,
                    enabled: true,
                    probability: T::from(0.15).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
                ComponentTypeConfig {
                    component_type: ComponentType::AdaGrad,
                    enabled: true,
                    probability: T::from(0.1).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
                ComponentTypeConfig {
                    component_type: ComponentType::AdaDelta,
                    enabled: true,
                    probability: T::from(0.1).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
                ComponentTypeConfig {
                    component_type: ComponentType::Custom,
                    enabled: true,
                    probability: T::from(0.1).unwrap(),
                    hyperparameter_ranges: std::collections::HashMap::new(),
                    constraints: ArchitectureConstraints::default(),
                    dependencies: Vec::new(),
                },
            ],
            max_components: 20,
            min_components: 1,
            max_connections: 50,
            min_connections: 0,
            allow_cycles: true,
            max_depth: 10,
            enable_skip_connections: true,
            connection_probability: T::from(0.7).unwrap(),
        },
        evaluation_config: EvaluationConfig {
            metrics: vec![
                EvaluationMetric::FinalPerformance,
                EvaluationMetric::ConvergenceSpeed,
                EvaluationMetric::Stability,
                EvaluationMetric::Efficiency,
                EvaluationMetric::Robustness,
                EvaluationMetric::Generalization,
                EvaluationMetric::MemoryUsage,
                EvaluationMetric::ComputationTime,
            ],
            num_trials: 20,
            max_iterations: 5000,
            convergence_threshold: T::from(1e-8).unwrap(),
            stability_window: 200,
            timeout_seconds: 7200.0,
            parallel_trials: true,
            cache_results: true,
            early_termination: false, // Don't terminate early for research
            confidence_level: T::from(0.99).unwrap(),
            statistical_significance: true,
        },
        multi_objective_config: MultiObjectiveConfig {
            objectives: vec![
                ObjectiveConfig {
                    name: "performance".to_string(),
                    objective_type: ObjectiveType::Performance,
                    direction: OptimizationDirection::Maximize,
                    weight: T::from(0.3).unwrap(),
                    priority: ObjectivePriority::High,
                    tolerance: None, // No tolerance for research
                },
                ObjectiveConfig {
                    name: "convergence".to_string(),
                    objective_type: ObjectiveType::ConvergenceSpeed,
                    direction: OptimizationDirection::Maximize,
                    weight: T::from(0.2).unwrap(),
                    priority: ObjectivePriority::Medium,
                    tolerance: None,
                },
                ObjectiveConfig {
                    name: "efficiency".to_string(),
                    objective_type: ObjectiveType::Efficiency,
                    direction: OptimizationDirection::Maximize,
                    weight: T::from(0.2).unwrap(),
                    priority: ObjectivePriority::Medium,
                    tolerance: None,
                },
                ObjectiveConfig {
                    name: "robustness".to_string(),
                    objective_type: ObjectiveType::Robustness,
                    direction: OptimizationDirection::Maximize,
                    weight: T::from(0.15).unwrap(),
                    priority: ObjectivePriority::Medium,
                    tolerance: None,
                },
                ObjectiveConfig {
                    name: "generalization".to_string(),
                    objective_type: ObjectiveType::Generalization,
                    direction: OptimizationDirection::Maximize,
                    weight: T::from(0.15).unwrap(),
                    priority: ObjectivePriority::Low,
                    tolerance: None,
                },
            ],
            algorithm: MultiObjectiveAlgorithm::NSGA3,
            pareto_front_size: 200,
            enable_preferences: false, // Explore full space for research
            user_preferences: None,
            diversity_strategy: DiversityStrategy::HyperVolume,
            constraint_handling: ConstraintHandlingMethod::PenaltyFunction,
        },
        search_budget: 5000,
        early_stopping: EarlyStoppingConfig {
            enabled: false, // Don't stop early for research
            patience: 200,
            min_improvement: T::from(0.0001).unwrap(),
            metric: EvaluationMetric::FinalPerformance,
            target_performance: None,
            convergence_detection: ConvergenceDetectionStrategy::RelativeImprovement,
        },
        progressive_search: true,
        population_size: 200,
        enable_transfer_learning: true,
        encoding_strategy: ArchitectureEncodingStrategy::Direct,
        enable_performance_prediction: true,
        parallelization_factor: 16,
        auto_hyperparameter_tuning: true,
        resource_constraints: ResourceConstraints {
            max_memory_gb: T::from(64.0).unwrap(),
            max_computation_hours: T::from(168.0).unwrap(), // 1 week
            max_energy_kwh: T::from(1000.0).unwrap(),
            max_cost_usd: T::from(10000.0).unwrap(),
            hardware_resources: HardwareResources {
                cpu_cores: 64,
                ram_gb: 256,
                num_gpus: 16,
                gpu_memory_gb: 160,
                storage_gb: 10000,
                network_bandwidth_mbps: 10000.0,
            },
            enable_monitoring: true,
            violation_handling: ResourceViolationHandling::Warning, // Don't stop for research
        },
    }
}