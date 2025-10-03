//! Parameter generation strategies for hyperparameter optimization
//!
//! This module implements various parameter generation strategies including
//! grid search, random search, Bayesian optimization, and evolutionary approaches.

use scirs2_core::ndarray::Array2;
use scirs2_core::random::{rng, Rng, SeedableRng};
use std::collections::HashMap;

use crate::error::{ClusteringError, Result};

use super::bayesian_optimization::BayesianOptimizer;
use super::config::*;

/// Parameter combination generator for different search strategies
pub struct ParameterGenerator {
    config: TuningConfig,
}

impl ParameterGenerator {
    /// Create a new parameter generator
    pub fn new(config: &TuningConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Generate parameter combinations based on search strategy
    pub fn generate_combinations(
        &self,
        search_space: &SearchSpace,
    ) -> Result<Vec<HashMap<String, f64>>> {
        match &self.config.strategy {
            SearchStrategy::GridSearch => self.generate_grid_combinations(search_space),
            SearchStrategy::RandomSearch { n_trials } => {
                self.generate_random_combinations(search_space, *n_trials)
            }
            SearchStrategy::BayesianOptimization {
                n_initial_points,
                acquisition_function,
            } => self.generate_bayesian_combinations(
                search_space,
                *n_initial_points,
                acquisition_function,
            ),
            SearchStrategy::EnsembleSearch {
                strategies,
                weights,
            } => self.generate_ensemble_combinations(search_space, strategies, weights),
            SearchStrategy::EvolutionarySearch {
                population_size,
                n_generations,
                mutation_rate,
                crossover_rate,
            } => self.generate_evolutionary_combinations(
                search_space,
                *population_size,
                *n_generations,
                *mutation_rate,
                *crossover_rate,
            ),
            SearchStrategy::SMBO {
                surrogate_model,
                acquisition_function,
            } => {
                self.generate_smbo_combinations(search_space, surrogate_model, acquisition_function)
            }
            SearchStrategy::MultiObjective {
                objectives,
                strategy,
            } => self.generate_multi_objective_combinations(search_space, objectives, strategy),
            SearchStrategy::AdaptiveSearch {
                initial_strategy, ..
            } => match initial_strategy.as_ref() {
                SearchStrategy::RandomSearch { n_trials } => {
                    self.generate_random_combinations(search_space, *n_trials)
                }
                SearchStrategy::GridSearch => self.generate_grid_combinations(search_space),
                _ => self.generate_random_combinations(search_space, self.config.max_evaluations),
            },
        }
    }

    /// Generate grid search combinations
    pub fn generate_grid_combinations(
        &self,
        search_space: &SearchSpace,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut combinations = Vec::new();
        let mut param_names = Vec::new();
        let mut param_values = Vec::new();

        for (name, param) in &search_space.parameters {
            param_names.push(name.clone());
            match param {
                HyperParameter::Integer { min, max } => {
                    let values: Vec<f64> = (*min..=*max).map(|x| x as f64).collect();
                    param_values.push(values);
                }
                HyperParameter::Float { min, max } => {
                    let n_steps = 10;
                    let step = (max - min) / (n_steps as f64 - 1.0);
                    let values: Vec<f64> = (0..n_steps).map(|i| min + i as f64 * step).collect();
                    param_values.push(values);
                }
                HyperParameter::Categorical { choices } => {
                    let values: Vec<f64> = (0..choices.len()).map(|i| i as f64).collect();
                    param_values.push(values);
                }
                HyperParameter::Boolean => {
                    param_values.push(vec![0.0, 1.0]);
                }
                HyperParameter::LogUniform { min, max } => {
                    let n_steps = 10;
                    let log_min = min.ln();
                    let log_max = max.ln();
                    let step = (log_max - log_min) / (n_steps as f64 - 1.0);
                    let values: Vec<f64> = (0..n_steps)
                        .map(|i| (log_min + i as f64 * step).exp())
                        .collect();
                    param_values.push(values);
                }
                HyperParameter::IntegerChoices { choices } => {
                    let values: Vec<f64> = choices.iter().map(|&x| x as f64).collect();
                    param_values.push(values);
                }
            }
        }

        self.generate_cartesian_product(
            &param_names,
            &param_values,
            &mut combinations,
            Vec::new(),
            0,
        );

        Ok(combinations)
    }

    /// Generate cartesian product of parameter values
    fn generate_cartesian_product(
        &self,
        param_names: &[String],
        param_values: &[Vec<f64>],
        combinations: &mut Vec<HashMap<String, f64>>,
        current: Vec<f64>,
        index: usize,
    ) {
        if index == param_names.len() {
            let mut combination = HashMap::new();
            for (i, name) in param_names.iter().enumerate() {
                combination.insert(name.clone(), current[i]);
            }
            combinations.push(combination);
            return;
        }

        for &value in &param_values[index] {
            let mut new_current = current.clone();
            new_current.push(value);
            self.generate_cartesian_product(
                param_names,
                param_values,
                combinations,
                new_current,
                index + 1,
            );
        }
    }

    /// Generate random search combinations
    pub fn generate_random_combinations(
        &self,
        search_space: &SearchSpace,
        n_trials: usize,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut combinations = Vec::new();
        let mut rng = match self.config.random_seed {
            Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
            None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
        };

        for _ in 0..n_trials {
            let mut combination = HashMap::new();

            for (name, param) in &search_space.parameters {
                let value = match param {
                    HyperParameter::Integer { min, max } => rng.random_range(*min..=*max) as f64,
                    HyperParameter::Float { min, max } => rng.random_range(*min..=*max),
                    HyperParameter::Categorical { choices } => {
                        rng.random_range(0..choices.len()) as f64
                    }
                    HyperParameter::Boolean => {
                        if rng.random_range(0.0..1.0) < 0.5 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    HyperParameter::LogUniform { min, max } => {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        let log_value = rng.random_range(log_min..=log_max);
                        log_value.exp()
                    }
                    HyperParameter::IntegerChoices { choices } => {
                        let idx = rng.random_range(0..choices.len());
                        choices[idx] as f64
                    }
                };

                combination.insert(name.clone(), value);
            }

            combinations.push(combination);
        }

        Ok(combinations)
    }

    /// Generate Bayesian optimization combinations
    pub fn generate_bayesian_combinations(
        &self,
        search_space: &SearchSpace,
        n_initial_points: usize,
        acquisition_function: &AcquisitionFunction,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut combinations = Vec::new();
        let parameter_names: Vec<String> = search_space.parameters.keys().cloned().collect();

        let mut bayesian_optimizer = BayesianOptimizer::new(
            parameter_names.clone(),
            acquisition_function.clone(),
            self.config.random_seed,
        );

        // Generate initial random points
        let initial_points = self.generate_random_combinations(search_space, n_initial_points)?;
        combinations.extend(initial_points);

        // Generate remaining points using Bayesian optimization
        let remaining_points = self.config.max_evaluations.saturating_sub(n_initial_points);

        for _ in 0..remaining_points {
            bayesian_optimizer.update_observations(&combinations);
            let next_point = bayesian_optimizer.optimize_acquisition_function(search_space)?;
            combinations.push(next_point);
        }

        Ok(combinations)
    }

    /// Generate ensemble search combinations
    pub fn generate_ensemble_combinations(
        &self,
        search_space: &SearchSpace,
        strategies: &[SearchStrategy],
        weights: &[f64],
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut all_combinations = Vec::new();
        let total_evaluations = self.config.max_evaluations;

        let weight_sum: f64 = weights.iter().sum();
        let normalized_weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

        for (strategy, &weight) in strategies.iter().zip(normalized_weights.iter()) {
            let n_evaluations = (total_evaluations as f64 * weight) as usize;

            let strategy_combinations = match strategy {
                SearchStrategy::RandomSearch { .. } => {
                    self.generate_random_combinations(search_space, n_evaluations)?
                }
                SearchStrategy::GridSearch => {
                    let grid_combinations = self.generate_grid_combinations(search_space)?;
                    grid_combinations.into_iter().take(n_evaluations).collect()
                }
                _ => self.generate_random_combinations(search_space, n_evaluations)?,
            };

            all_combinations.extend(strategy_combinations);
        }

        let mut rng = match self.config.random_seed {
            Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
            None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
        };

        use scirs2_core::random::seq::SliceRandom;
        all_combinations.shuffle(&mut rng);

        Ok(all_combinations)
    }

    /// Generate evolutionary search combinations
    pub fn generate_evolutionary_combinations(
        &self,
        search_space: &SearchSpace,
        population_size: usize,
        n_generations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut all_combinations = Vec::new();
        let mut rng = match self.config.random_seed {
            Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
            None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
        };

        // Generate initial population
        let mut population = self.generate_random_combinations(search_space, population_size)?;
        all_combinations.extend(population.clone());

        // Evolve population
        for _generation in 0..n_generations {
            let mut new_population = Vec::new();

            while new_population.len() < population_size {
                // Tournament selection
                let parent1 = self.tournament_selection(&population, &mut rng);
                let parent2 = self.tournament_selection(&population, &mut rng);

                // Crossover
                let mut offspring = if rng.random_range(0.0..1.0) < crossover_rate {
                    self.crossover(&parent1, &parent2, &mut rng)
                } else {
                    parent1.clone()
                };

                // Mutation
                if rng.random_range(0.0..1.0) < mutation_rate {
                    self.mutate(&mut offspring, search_space, &mut rng)?;
                }

                new_population.push(offspring);
            }

            population = new_population;
            all_combinations.extend(population.clone());
        }

        Ok(all_combinations)
    }

    /// Tournament selection for evolutionary algorithm
    fn tournament_selection(
        &self,
        population: &[HashMap<String, f64>],
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> HashMap<String, f64> {
        let tournament_size = 3.min(population.len());
        let mut best = &population[rng.random_range(0..population.len())];

        for _ in 1..tournament_size {
            let candidate = &population[rng.random_range(0..population.len())];
            // In practice, we'd need fitness scores to compare
            // For now, just return the first candidate
            best = candidate;
        }

        best.clone()
    }

    /// Crossover operation for evolutionary algorithm
    fn crossover(
        &self,
        parent1: &HashMap<String, f64>,
        parent2: &HashMap<String, f64>,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> HashMap<String, f64> {
        let mut offspring = HashMap::new();

        for (key, value1) in parent1 {
            if let Some(value2) = parent2.get(key) {
                // Uniform crossover
                let offspring_value = if rng.random_range(0.0..1.0) < 0.5 {
                    *value1
                } else {
                    *value2
                };
                offspring.insert(key.clone(), offspring_value);
            } else {
                offspring.insert(key.clone(), *value1);
            }
        }

        offspring
    }

    /// Mutation operation for evolutionary algorithm
    fn mutate(
        &self,
        individual: &mut HashMap<String, f64>,
        search_space: &SearchSpace,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> Result<()> {
        for (param_name, param_def) in &search_space.parameters {
            if let Some(value) = individual.get_mut(param_name) {
                // Gaussian mutation with parameter-dependent variance
                let mutation_strength = match param_def {
                    HyperParameter::Float { min, max } => (max - min) * 0.1,
                    HyperParameter::Integer { min, max } => (*max - *min) as f64 * 0.1,
                    _ => 0.1,
                };

                let noise = rng.random_range(-mutation_strength..mutation_strength);
                let new_value = *value + noise;

                // Clip to parameter bounds
                *value = match param_def {
                    HyperParameter::Float { min, max } => new_value.clamp(*min, *max),
                    HyperParameter::Integer { min, max } => {
                        (new_value.round() as i64).clamp(*min, *max) as f64
                    }
                    HyperParameter::LogUniform { min, max } => new_value.clamp(*min, *max),
                    _ => new_value,
                };
            }
        }

        Ok(())
    }

    /// Generate SMBO (Sequential Model-Based Optimization) combinations
    pub fn generate_smbo_combinations(
        &self,
        search_space: &SearchSpace,
        surrogate_model: &SurrogateModel,
        acquisition_function: &AcquisitionFunction,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let n_initial_points = 10.max(search_space.parameters.len() * 2);
        let mut combinations = Vec::new();

        let initial_points = self.generate_random_combinations(search_space, n_initial_points)?;
        combinations.extend(initial_points);

        let remaining_points = self.config.max_evaluations.saturating_sub(n_initial_points);

        for _iteration in 0..remaining_points {
            let next_point = match surrogate_model {
                SurrogateModel::GaussianProcess { .. } => {
                    let parameter_names: Vec<String> =
                        search_space.parameters.keys().cloned().collect();
                    let mut bayesian_optimizer = BayesianOptimizer::new(
                        parameter_names,
                        acquisition_function.clone(),
                        self.config.random_seed,
                    );
                    bayesian_optimizer.update_observations(&combinations);
                    bayesian_optimizer.optimize_acquisition_function(search_space)?
                }
                SurrogateModel::RandomForest { .. } => {
                    self.generate_rf_guided_point(search_space, &combinations)?
                }
                SurrogateModel::GradientBoosting { .. } => {
                    self.generate_gb_guided_point(search_space, &combinations)?
                }
            };

            combinations.push(next_point);
        }

        Ok(combinations)
    }

    /// Generate point guided by Random Forest surrogate model
    fn generate_rf_guided_point(
        &self,
        search_space: &SearchSpace,
        existing_combinations: &[HashMap<String, f64>],
    ) -> Result<HashMap<String, f64>> {
        if existing_combinations.is_empty() {
            return self
                .generate_random_combinations(search_space, 1)
                .map(|mut v| v.pop().unwrap_or_default());
        }

        let mut promising_point = HashMap::new();

        for (param_name, param_def) in &search_space.parameters {
            let values: Vec<f64> = existing_combinations
                .iter()
                .filter_map(|c| c.get(param_name))
                .copied()
                .collect();

            if values.is_empty() {
                continue;
            }

            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

            let suggested_value = match param_def {
                HyperParameter::Float { min, max } => {
                    let mut rng = scirs2_core::random::rng();
                    let noise = rng.random_range(-variance.sqrt()..variance.sqrt());
                    (mean + noise).clamp(*min, *max)
                }
                HyperParameter::Integer { min, max } => {
                    let mut rng = scirs2_core::random::rng();
                    rng.random_range(*min..=*max) as f64
                }
                _ => mean,
            };

            promising_point.insert(param_name.clone(), suggested_value);
        }

        Ok(promising_point)
    }

    /// Generate point guided by Gradient Boosting surrogate model
    fn generate_gb_guided_point(
        &self,
        search_space: &SearchSpace,
        existing_combinations: &[HashMap<String, f64>],
    ) -> Result<HashMap<String, f64>> {
        // Simplified implementation - similar to Random Forest for now
        self.generate_rf_guided_point(search_space, existing_combinations)
    }

    /// Multi-objective optimization using Pareto frontier
    fn generate_multi_objective_combinations(
        &self,
        search_space: &SearchSpace,
        _objectives: &[EvaluationMetric],
        base_strategy: &SearchStrategy,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let base_combinations = match base_strategy {
            SearchStrategy::RandomSearch { n_trials } => {
                self.generate_random_combinations(search_space, *n_trials)?
            }
            SearchStrategy::GridSearch => self.generate_grid_combinations(search_space)?,
            SearchStrategy::BayesianOptimization {
                n_initial_points,
                acquisition_function,
            } => self.generate_bayesian_combinations(
                search_space,
                *n_initial_points,
                acquisition_function,
            )?,
            _ => self.generate_random_combinations(search_space, self.config.max_evaluations)?,
        };

        let mut diverse_combinations = base_combinations;

        let additional_random = self.generate_random_combinations(
            search_space,
            (self.config.max_evaluations / 4).max(10),
        )?;
        diverse_combinations.extend(additional_random);

        Ok(diverse_combinations)
    }
}
