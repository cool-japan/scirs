//! NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation
//!
//! This module provides a complete implementation of the NSGA-II algorithm
//! for multi-objective optimization problems.

use super::{utils, MultiObjectiveConfig, MultiObjectiveOptimizer};
use crate::error::OptimizeError;
use crate::multi_objective::crossover::{CrossoverOperator, SimulatedBinaryCrossover};
use crate::multi_objective::mutation::{MutationOperator, PolynomialMutation};
use crate::multi_objective::selection::{SelectionOperator, TournamentSelection};
use crate::multi_objective::solutions::{MultiObjectiveResult, MultiObjectiveSolution, Population};
use scirs2_core::ndarray::{s, Array1, ArrayView1};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{prelude::*, SeedableRng};

/// NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation
pub struct NSGAII {
    config: MultiObjectiveConfig,
    n_objectives: usize,
    n_variables: usize,
    population: Population,
    generation: usize,
    n_evaluations: usize,
    rng: StdRng,
    crossover: SimulatedBinaryCrossover,
    mutation: PolynomialMutation,
    selection: TournamentSelection,
    convergence_history: Vec<f64>,
}

impl NSGAII {
    /// Create a new NSGA-II optimizer
    pub fn new(
        config: MultiObjectiveConfig,
        n_objectives: usize,
        n_variables: usize,
    ) -> Result<Self, OptimizeError> {
        if n_objectives == 0 {
            return Err(OptimizeError::InvalidInput(
                "Number of objectives must be > 0".to_string(),
            ));
        }
        if n_variables == 0 {
            return Err(OptimizeError::InvalidInput(
                "Number of variables must be > 0".to_string(),
            ));
        }
        if config.population_size == 0 {
            return Err(OptimizeError::InvalidInput(
                "Population size must be > 0".to_string(),
            ));
        }

        let seed = config.random_seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        });

        let rng = StdRng::seed_from_u64(seed);

        let crossover =
            SimulatedBinaryCrossover::new(config.crossover_eta, config.crossover_probability);

        let mutation = PolynomialMutation::new(config.mutation_probability, config.mutation_eta);

        let selection = TournamentSelection::new(2);

        Ok(Self {
            config,
            n_objectives,
            n_variables,
            population: Population::new(),
            generation: 0,
            n_evaluations: 0,
            rng,
            crossover,
            mutation,
            selection,
            convergence_history: Vec::new(),
        })
    }

    /// Evaluate a single individual
    fn evaluate_individual<F>(
        &mut self,
        variables: &Array1<f64>,
        objective_function: &F,
    ) -> Result<Array1<f64>, OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    {
        self.n_evaluations += 1;

        // Check evaluation limit
        if let Some(max_evals) = self.config.max_evaluations {
            if self.n_evaluations > max_evals {
                return Err(OptimizeError::MaxEvaluationsReached);
            }
        }

        let objectives = objective_function(&variables.view());

        if objectives.len() != self.n_objectives {
            return Err(OptimizeError::InvalidInput(format!(
                "Expected {} objectives, got {}",
                self.n_objectives,
                objectives.len()
            )));
        }

        Ok(objectives)
    }

    /// Create offspring through crossover and mutation
    fn create_offspring<F>(
        &mut self,
        objective_function: &F,
    ) -> Result<Vec<MultiObjectiveSolution>, OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    {
        let mut offspring = Vec::new();

        while offspring.len() < self.config.population_size {
            // Select parents using tournament selection (clone to avoid borrowing issues)
            let population_solutions = self.population.solutions().to_vec();
            let selected_parents = self.selection.select(&population_solutions, 2);
            if selected_parents.len() < 2 {
                break;
            }

            let parent1 = &selected_parents[0];
            let parent2 = &selected_parents[1];

            // Perform crossover
            let (mut child1_vars, mut child2_vars) = self.crossover.crossover(
                parent1.variables.as_slice().unwrap(),
                parent2.variables.as_slice().unwrap(),
            );

            // Apply mutation (need bounds for mutation)
            let bounds: Vec<(f64, f64)> = if let Some((lower, upper)) = &self.config.bounds {
                lower
                    .iter()
                    .zip(upper.iter())
                    .map(|(&l, &u)| (l, u))
                    .collect()
            } else {
                vec![(-1.0, 1.0); child1_vars.len()]
            };

            self.mutation.mutate(&mut child1_vars, &bounds);
            self.mutation.mutate(&mut child2_vars, &bounds);

            // Convert to Array1 for evaluation
            let child1_array = Array1::from_vec(child1_vars);
            let child2_array = Array1::from_vec(child2_vars);

            // Evaluate offspring
            let child1_objs = self.evaluate_individual(&child1_array, objective_function)?;
            let child2_objs = self.evaluate_individual(&child2_array, objective_function)?;

            offspring.push(MultiObjectiveSolution::new(child1_array, child1_objs));

            if offspring.len() < self.config.population_size {
                offspring.push(MultiObjectiveSolution::new(child2_array, child2_objs));
            }
        }

        Ok(offspring)
    }

    /// Environmental selection using NSGA-II procedure
    fn environmental_selection(
        &mut self,
        combined_population: Vec<MultiObjectiveSolution>,
    ) -> Vec<MultiObjectiveSolution> {
        let mut temp_population = Population::from_solutions(combined_population);
        temp_population.select_best(self.config.population_size)
    }

    /// Calculate generation metrics
    fn calculate_metrics(&mut self) {
        if let Some(ref_point) = &self.config.reference_point {
            let pareto_front = self.population.extract_pareto_front();
            let hypervolume = utils::calculate_hypervolume(&pareto_front, ref_point);
            self.convergence_history.push(hypervolume);
        }
    }
}

impl MultiObjectiveOptimizer for NSGAII {
    fn optimize<F>(&mut self, objective_function: F) -> Result<MultiObjectiveResult, OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    {
        // Initialize population
        self.initialize_population()?;

        // Evaluate initial population
        let initial_variables = utils::generate_random_population(
            self.config.population_size,
            self.n_variables,
            &self.config.bounds,
        );

        let mut initial_solutions = Vec::new();
        for variables in initial_variables {
            let objectives = self.evaluate_individual(&variables, &objective_function)?;
            initial_solutions.push(MultiObjectiveSolution::new(variables, objectives));
        }

        self.population = Population::from_solutions(initial_solutions);

        // Main evolution loop
        while self.generation < self.config.max_generations {
            if self.check_convergence() {
                break;
            }

            self.evolve_generation(&objective_function)?;
        }

        // Extract final results
        let pareto_front = self.population.extract_pareto_front();
        let hypervolume = self
            .config
            .reference_point
            .as_ref()
            .map(|ref_point| utils::calculate_hypervolume(&pareto_front, ref_point));

        let mut result = MultiObjectiveResult::new(
            pareto_front,
            self.population.solutions().to_vec(),
            self.n_evaluations,
            self.generation,
        );

        result.hypervolume = hypervolume;
        result.metrics.convergence_history = self.convergence_history.clone();
        result.metrics.population_stats = self.population.calculate_statistics();

        Ok(result)
    }

    fn initialize_population(&mut self) -> Result<(), OptimizeError> {
        self.population.clear();
        self.generation = 0;
        self.n_evaluations = 0;
        self.convergence_history.clear();
        Ok(())
    }

    fn evolve_generation<F>(&mut self, objective_function: &F) -> Result<(), OptimizeError>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync,
    {
        // Create offspring
        let offspring = self.create_offspring(objective_function)?;

        // Combine parent and offspring populations
        let mut combined = self.population.solutions().to_vec();
        combined.extend(offspring);

        // Environmental selection
        let next_population = self.environmental_selection(combined);
        self.population = Population::from_solutions(next_population);

        // Update generation counter
        self.generation += 1;

        // Calculate metrics
        self.calculate_metrics();

        Ok(())
    }

    fn check_convergence(&self) -> bool {
        // Check maximum evaluations
        if let Some(max_evals) = self.config.max_evaluations {
            if self.n_evaluations >= max_evals {
                return true;
            }
        }

        // Check hypervolume convergence
        if self.convergence_history.len() >= 10 {
            let recent_history = &self.convergence_history[self.convergence_history.len() - 10..];
            let max_hv = recent_history
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_hv = recent_history.iter().fold(f64::INFINITY, |a, &b| a.min(b));

            if (max_hv - min_hv) < self.config.tolerance {
                return true;
            }
        }

        false
    }

    fn get_population(&self) -> &Population {
        &self.population
    }

    fn get_generation(&self) -> usize {
        self.generation
    }

    fn get_evaluations(&self) -> usize {
        self.n_evaluations
    }

    fn name(&self) -> &str {
        "NSGA-II"
    }
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
    fn test_nsga2_creation() {
        let config = MultiObjectiveConfig::default();
        let nsga2 = NSGAII::new(config, 2, 3);
        assert!(nsga2.is_ok());

        let nsga2 = nsga2.unwrap();
        assert_eq!(nsga2.n_objectives, 2);
        assert_eq!(nsga2.n_variables, 3);
        assert_eq!(nsga2.generation, 0);
    }

    #[test]
    fn test_nsga2_invalid_parameters() {
        let config = MultiObjectiveConfig::default();

        // Zero objectives
        let nsga2 = NSGAII::new(config.clone(), 0, 3);
        assert!(nsga2.is_err());

        // Zero variables
        let nsga2 = NSGAII::new(config.clone(), 2, 0);
        assert!(nsga2.is_err());

        // Zero population size
        let mut config_zero_pop = config;
        config_zero_pop.population_size = 0;
        let nsga2 = NSGAII::new(config_zero_pop, 2, 3);
        assert!(nsga2.is_err());
    }

    #[test]
    fn test_nsga2_optimization() {
        let mut config = MultiObjectiveConfig::default();
        config.max_generations = 10;
        config.population_size = 20;
        config.bounds = Some((Array1::zeros(2), Array1::ones(2)));
        config.random_seed = Some(42);

        let mut nsga2 = NSGAII::new(config, 2, 2).unwrap();
        let result = nsga2.optimize(zdt1);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert!(!result.pareto_front.is_empty());
        assert_eq!(result.n_generations, 10);
        assert!(result.n_evaluations > 0);
    }

    #[test]
    fn test_nsga2_with_max_evaluations() {
        let mut config = MultiObjectiveConfig::default();
        config.max_generations = 1000; // High value
        config.max_evaluations = Some(50); // Low evaluation limit
        config.population_size = 10;
        config.bounds = Some((Array1::zeros(2), Array1::ones(2)));

        let mut nsga2 = NSGAII::new(config, 2, 2).unwrap();
        let result = nsga2.optimize(zdt1);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.n_evaluations <= 50);
    }

    #[test]
    fn test_nsga2_hypervolume_calculation() {
        let mut config = MultiObjectiveConfig::default();
        config.max_generations = 5;
        config.population_size = 10;
        config.bounds = Some((Array1::zeros(2), Array1::ones(2)));
        config.reference_point = Some(array![2.0, 2.0]);

        let mut nsga2 = NSGAII::new(config, 2, 2).unwrap();
        let result = nsga2.optimize(zdt1).unwrap();

        assert!(result.hypervolume.is_some());
        assert!(result.hypervolume.unwrap() >= 0.0);
        assert!(!result.metrics.convergence_history.is_empty());
    }

    #[test]
    fn test_nsga2_convergence_check() {
        let mut config = MultiObjectiveConfig::default();
        config.tolerance = 1e-10; // Very tight tolerance
        config.max_generations = 2; // Few generations
        config.population_size = 10;

        let nsga2 = NSGAII::new(config, 2, 2).unwrap();

        // With empty convergence history, should not converge
        assert!(!nsga2.check_convergence());
    }

    #[test]
    fn test_nsga2_name() {
        let config = MultiObjectiveConfig::default();
        let nsga2 = NSGAII::new(config, 2, 2).unwrap();
        assert_eq!(nsga2.name(), "NSGA-II");
    }
}
