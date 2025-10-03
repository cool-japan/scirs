//! Multi-objective optimization algorithms and utilities
//!
//! This module provides comprehensive multi-objective optimization capabilities:
//! - Solution representations and population management
//! - Various multi-objective algorithms (NSGA-II, NSGA-III, etc.)
//! - Genetic operators (crossover, mutation, selection)
//! - Performance indicators and metrics
//!
//! The module is organized into focused submodules:
//! - `solutions`: Core data structures for solutions and populations
//! - `algorithms`: Multi-objective optimization algorithms
//! - `crossover`: Crossover operators for genetic algorithms
//! - `mutation`: Mutation operators for genetic algorithms
//! - `selection`: Selection operators for genetic algorithms
//! - `indicators`: Performance indicators and quality metrics

pub mod algorithms;
pub mod crossover;
pub mod indicators;
pub mod mutation;
pub mod selection;
pub mod solutions;

// Re-export main types for easier access
pub use algorithms::{
    MultiObjectiveConfig, MultiObjectiveOptimizer, MultiObjectiveOptimizerWrapper,
    OptimizerFactory, NSGAII, NSGAIII,
};
pub use solutions::{
    MultiObjectiveResult, MultiObjectiveSolution, OptimizationMetrics, Population,
};

use crate::error::OptimizeError;
use scirs2_core::ndarray::{s, Array1, ArrayView1};

/// Convenience function to optimize using NSGA-II
pub fn nsga2<F>(
    objective_function: F,
    n_objectives: usize,
    n_variables: usize,
    config: Option<MultiObjectiveConfig>,
) -> Result<MultiObjectiveResult, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    let config = config.unwrap_or_default();
    let mut optimizer =
        algorithms::OptimizerFactory::create_nsga2(config, n_objectives, n_variables)?;
    optimizer.optimize(objective_function)
}

/// Convenience function to optimize using NSGA-III
pub fn nsga3<F>(
    objective_function: F,
    n_objectives: usize,
    n_variables: usize,
    config: Option<MultiObjectiveConfig>,
    reference_points: Option<Vec<Array1<f64>>>,
) -> Result<MultiObjectiveResult, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    let config = config.unwrap_or_default();
    let mut optimizer = algorithms::OptimizerFactory::create_nsga3(
        config,
        n_objectives,
        n_variables,
        reference_points,
    )?;
    optimizer.optimize(objective_function)
}

/// Convenience function to optimize using any algorithm by name
pub fn optimize<F>(
    algorithm: &str,
    objective_function: F,
    n_objectives: usize,
    n_variables: usize,
    config: Option<MultiObjectiveConfig>,
) -> Result<MultiObjectiveResult, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    let config = config.unwrap_or_default();
    let mut optimizer =
        algorithms::OptimizerFactory::create_by_name(algorithm, config, n_objectives, n_variables)?;

    // Adapt the objective function signature
    let adapted_fn = |x: &ArrayView1<f64>| objective_function(x);

    optimizer.optimize(adapted_fn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // Simple test problem (ZDT1)
    fn zdt1(x: &ArrayView1<f64>) -> Array1<f64> {
        let f1 = x[0];
        let g = 1.0 + 9.0 * x.slice(s![1..]).sum() / (x.len() - 1) as f64;
        let f2 = g * (1.0 - (f1 / g).sqrt());
        array![f1, f2]
    }

    #[test]
    fn test_nsga2_convenience_function() {
        let mut config = MultiObjectiveConfig::default();
        config.max_generations = 5;
        config.population_size = 10;
        config.bounds = Some((Array1::zeros(2), Array1::ones(2)));

        let result = nsga2(zdt1, 2, 2, Some(config));
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.success);
        assert!(!result.pareto_front.is_empty());
    }

    #[test]
    fn test_optimize_by_name() {
        let mut config = MultiObjectiveConfig::default();
        config.max_generations = 5;
        config.population_size = 10;
        config.bounds = Some((Array1::zeros(2), Array1::ones(2)));

        let result = optimize("nsga2", zdt1, 2, 2, Some(config.clone()));
        assert!(result.is_ok());

        let result = optimize("unknown", zdt1, 2, 2, Some(config));
        assert!(result.is_err());
    }

    #[test]
    #[ignore] // Skip this slow test (>30s) during normal test runs
    fn test_default_config() {
        let result = nsga2(zdt1, 2, 2, None);
        // Should work with default config, though might take longer
        assert!(result.is_ok() || matches!(result, Err(OptimizeError::MaxEvaluationsReached)));
    }
}
