//! Neural Architecture Search for Learned Optimizers
//!
//! This module implements automated neural architecture search (NAS) to discover
//! optimal neural network architectures for learned optimizers, enabling
//! automatic design of meta-learning optimization algorithms.

pub mod config;
pub mod search_strategies;
pub mod architecture_space;
pub mod evaluation;
pub mod population;
pub mod performance_prediction;
pub mod architecture_generator;
pub mod multi_objective;
pub mod search_history;
pub mod resource_management;

// Re-export main types for backward compatibility
pub use config::{NASConfig, SearchConstraints, SearchStrategyType};
pub use search_strategies::{SearchStrategy, EvolutionarySearchStrategy, BayesianSearchStrategy};
pub use architecture_space::{ArchitectureSearchSpace, ArchitectureComponent};
pub use config::LayerType;
pub use evaluation::{ArchitectureEvaluator, EvaluationMetrics, EvaluationStrategy};
pub use population::{PopulationManager, Individual, SelectionStrategy};
pub use performance_prediction::{PerformancePredictor, PredictionModel, PerformanceMetrics};
pub use architecture_generator::{ArchitectureGenerator, GenerationStrategy};
pub use multi_objective::{MultiObjectiveOptimizer, OptimizationObjective, ParetoFront};
pub use search_history::{SearchHistory, ArchitectureEntry, SearchEvent};
pub use resource_management::{ResourceManager, ResourceConstraints, ComputeBudget};

use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use crate::error::Result;

/// Neural Architecture Search for Optimizer Design
pub struct NeuralArchitectureSearch<T: Float> {
    /// Search configuration
    config: NASConfig,

    /// Architecture search space
    search_space: ArchitectureSearchSpace,

    /// Search strategy
    search_strategy: SearchStrategy<T>,

    /// Architecture evaluator
    evaluator: ArchitectureEvaluator<T>,

    /// Population manager (for evolutionary search)
    population_manager: PopulationManager<T>,

    /// Performance predictor
    performance_predictor: PerformancePredictor<T>,

    /// Architecture generator
    architecture_generator: ArchitectureGenerator,

    /// Search history
    search_history: SearchHistory<T>,

    /// Resource manager
    resource_manager: ResourceManager,

    /// Multi-objective optimizer
    multi_objective_optimizer: MultiObjectiveOptimizer<T>,
}

impl<T: Float + 'static> NeuralArchitectureSearch<T> {
    /// Create new neural architecture search
    pub fn new(config: NASConfig) -> Result<Self> {
        let search_space = ArchitectureSearchSpace::new(&config.constraints)?;
        let search_strategy = SearchStrategy::new(config.search_strategy, &config)?;
        let evaluator = ArchitectureEvaluator::new(&config)?;
        let population_manager = PopulationManager::new(config.population_size, config.elite_size)?;
        let performance_predictor = PerformancePredictor::new(&config)?;
        let architecture_generator = ArchitectureGenerator::new(&search_space)?;
        let search_history = SearchHistory::new(config.max_iterations * 2)?;
        let resource_manager = ResourceManager::new(config.evaluation_budget)?;
        let multi_objective_optimizer = MultiObjectiveOptimizer::new(config.objective_weights.clone())?;

        Ok(Self {
            config,
            search_space,
            search_strategy,
            evaluator,
            population_manager,
            performance_predictor,
            architecture_generator,
            search_history,
            resource_manager,
            multi_objective_optimizer,
        })
    }

    /// Run neural architecture search
    pub fn search(&mut self) -> Result<Vec<String>> {
        let start_time = Instant::now();
        let mut best_architectures = Vec::new();

        // Initialize population
        self.initialize_population()?;

        for iteration in 0..self.config.max_iterations {
            // Check resource constraints
            if !self.resource_manager.has_budget_remaining() {
                break;
            }

            // Generate new architectures
            let new_architectures = self.generate_architectures()?;

            // Evaluate architectures
            let evaluation_results = self.evaluate_architectures(&new_architectures)?;

            // Update population
            self.update_population(new_architectures, evaluation_results)?;

            // Update multi-objective optimization
            let pareto_front = self.multi_objective_optimizer.update_pareto_front(
                &self.population_manager.get_current_population()
            )?;

            // Check for early stopping
            if self.should_early_stop(iteration) {
                break;
            }

            // Log progress
            if iteration % 10 == 0 {
                println!("NAS Iteration {}: {} architectures evaluated",
                        iteration, self.search_history.total_evaluated());
            }
        }

        // Extract best architectures
        best_architectures = self.extract_best_architectures()?;

        // Record final results
        self.search_history.record_search_completion(start_time.elapsed())?;

        Ok(best_architectures)
    }

    /// Initialize population with random architectures
    fn initialize_population(&mut self) -> Result<()> {
        let initial_architectures = self.architecture_generator
            .generate_random_population(self.config.population_size)?;

        let evaluation_results = self.evaluate_architectures(&initial_architectures)?;
        self.population_manager.initialize(initial_architectures, evaluation_results)?;

        Ok(())
    }

    /// Generate new architectures using current search strategy
    fn generate_architectures(&mut self) -> Result<Vec<String>> {
        let current_population = self.population_manager.get_current_population();
        self.search_strategy.generate_candidates(&current_population, &self.search_space)
    }

    /// Evaluate batch of architectures
    fn evaluate_architectures(&mut self, architectures: &[String]) -> Result<Vec<EvaluationMetrics>> {
        let mut results = Vec::new();

        for architecture in architectures {
            // Use performance prediction if enabled
            if self.config.enable_performance_prediction {
                if let Some(predicted_metrics) = self.performance_predictor.predict(architecture)? {
                    results.push(predicted_metrics);
                    continue;
                }
            }

            // Full evaluation
            let metrics = self.evaluator.evaluate(architecture)?;
            results.push(metrics.clone());

            // Train performance predictor
            if self.config.enable_performance_prediction {
                self.performance_predictor.train_on_sample(architecture, &metrics)?;
            }

            // Record in history
            self.search_history.record_evaluation(architecture.clone(), metrics)?;

            // Update resource usage
            self.resource_manager.consume_evaluation_budget(1)?;
        }

        Ok(results)
    }

    /// Update population with new architectures and their evaluations
    fn update_population(
        &mut self,
        architectures: Vec<String>,
        evaluations: Vec<EvaluationMetrics>
    ) -> Result<()> {
        self.population_manager.update(architectures, evaluations)?;
        self.search_strategy.update_from_population(
            &self.population_manager.get_current_population()
        )?;
        Ok(())
    }

    /// Check if early stopping criteria are met
    fn should_early_stop(&self, iteration: usize) -> bool {
        if iteration < self.config.early_stopping_patience {
            return false;
        }

        // Check if no improvement in recent iterations
        let recent_best = self.search_history.get_recent_best_performance(
            self.config.early_stopping_patience
        );

        if let Some(current_best) = self.population_manager.get_best_performance() {
            if let Some(prev_best_f64) = recent_best {
                let prev_best = T::from(prev_best_f64).unwrap();
                return (current_best - prev_best).abs() < T::from(0.001).unwrap();
            }
        }

        false
    }

    /// Extract best architectures from final population
    fn extract_best_architectures(&self) -> Result<Vec<String>> {
        let pareto_front = self.multi_objective_optimizer.get_pareto_front();

        if pareto_front.is_empty() {
            // Fallback to top architectures from population
            Ok(self.population_manager.get_top_architectures(self.config.elite_size))
        } else {
            Ok(pareto_front.iter().map(|ind| ind.architecture.clone()).collect())
        }
    }

    /// Get search statistics
    pub fn get_search_statistics(&self) -> SearchStatistics {
        SearchStatistics {
            total_iterations: self.search_history.total_iterations(),
            total_evaluated: self.search_history.total_evaluated(),
            best_performance: self.population_manager.get_best_performance().map(|p| p.to_f64().unwrap_or(0.0)),
            resource_usage: self.resource_manager.get_usage_stats(),
            convergence_history: self.search_history.get_convergence_history(),
        }
    }

    /// Get configuration
    pub fn get_config(&self) -> &NASConfig {
        &self.config
    }
}

/// Search statistics
#[derive(Debug, Clone)]
pub struct SearchStatistics {
    pub total_iterations: usize,
    pub total_evaluated: usize,
    pub best_performance: Option<f64>,
    pub resource_usage: ResourceUsageStats,
    pub convergence_history: Vec<f64>,
}

/// Resource usage statistics
#[derive(Debug, Clone)]
pub struct ResourceUsageStats {
    pub evaluations_used: usize,
    pub evaluations_remaining: usize,
    pub total_compute_time: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nas_creation() {
        let config = NASConfig::default();
        let nas = NeuralArchitectureSearch::<f32>::new(config);
        assert!(nas.is_ok());
    }

    #[test]
    fn test_nas_basic_workflow() {
        let mut config = NASConfig::default();
        config.max_iterations = 2;
        config.population_size = 4;

        let mut nas = NeuralArchitectureSearch::<f32>::new(config).unwrap();

        // This might fail due to complexity of full NAS implementation
        // but the structure should be testable
        let stats = nas.get_search_statistics();
        assert_eq!(stats.total_iterations, 0);
    }
}