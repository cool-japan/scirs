//! Search strategy implementations for hyperparameter optimization
//!
//! This module contains implementations of various search strategies including
//! grid search, random search, Bayesian optimization, and evolutionary algorithms.

use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::random::{Rng, SeedableRng};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

use super::config::{
    AcquisitionFunction, BayesianState, GpHyperparameters, HyperParameter, KernelType, SearchSpace,
    SearchStrategy, TuningConfig,
};

/// Search strategy generator for hyperparameter optimization
pub struct StrategyGenerator<F: Float> {
    config: TuningConfig,
    phantom: std::marker::PhantomData<F>,
}

impl<F: Float + FromPrimitive + Debug + std::fmt::Display + Send + Sync + PartialOrd>
    StrategyGenerator<F>
where
    f64: From<F>,
{
    /// Create new strategy generator
    pub fn new(config: TuningConfig) -> Self {
        Self {
            config,
            phantom: std::marker::PhantomData,
        }
    }

    /// Generate parameter combinations based on search strategy
    pub fn generate_parameter_combinations(
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
            } => {
                // For multi-objective, we need special handling
                self.generate_multi_objective_combinations(search_space, objectives, strategy)
            }
            SearchStrategy::AdaptiveSearch {
                initial_strategy, ..
            } => {
                // Start with initial strategy
                match initial_strategy.as_ref() {
                    SearchStrategy::RandomSearch { n_trials } => {
                        self.generate_random_combinations(search_space, *n_trials)
                    }
                    SearchStrategy::GridSearch => self.generate_grid_combinations(search_space),
                    _ => {
                        // Fallback to random search
                        self.generate_random_combinations(search_space, self.config.max_evaluations)
                    }
                }
            }
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

        // Extract parameter ranges
        for (name, param) in &search_space.parameters {
            param_names.push(name.clone());
            match param {
                HyperParameter::Integer { min, max } => {
                    let values: Vec<f64> = (*min..=*max).map(|x| x as f64).collect();
                    param_values.push(values);
                }
                HyperParameter::Float { min, max } => {
                    // Create a reasonable grid for float parameters
                    let n_steps = 10; // Could be configurable
                    let step = (max - min) / (n_steps as f64 - 1.0);
                    let values: Vec<f64> = (0..n_steps).map(|i| min + i as f64 * step).collect();
                    param_values.push(values);
                }
                HyperParameter::Categorical { choices } => {
                    // Map categorical choices to numeric values
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

        // Generate all combinations
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
                    HyperParameter::Integer { min, max } => rng.gen_range(*min..=*max) as f64,
                    HyperParameter::Float { min, max } => rng.gen_range(*min..=*max),
                    HyperParameter::Categorical { choices } => {
                        rng.gen_range(0..choices.len()) as f64
                    }
                    HyperParameter::Boolean => {
                        if rng.gen_range(0.0..1.0) < 0.5 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    HyperParameter::LogUniform { min, max } => {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        let log_value = rng.gen_range(log_min..=log_max);
                        log_value.exp()
                    }
                    HyperParameter::IntegerChoices { choices } => {
                        let idx = rng.gen_range(0..choices.len());
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
        // Extract parameter names for consistent ordering
        let parameter_names: Vec<String> = search_space.parameters.keys().cloned().collect();

        let mut bayesian_state = BayesianState {
            observations: Vec::new(),
            gp_mean: None,
            gp_covariance: None,
            acquisition_values: Vec::new(),
            parameter_names: parameter_names.clone(),
            gp_hyperparameters: GpHyperparameters {
                length_scales: vec![1.0; parameter_names.len()],
                signal_variance: 1.0,
                noise_variance: 0.1,
                kernel_type: KernelType::RBF { length_scale: 1.0 },
            },
            noise_level: 0.1,
            currentbest: f64::NEG_INFINITY,
        };

        // Generate initial random points
        let initial_points = self.generate_random_combinations(search_space, n_initial_points)?;
        combinations.extend(initial_points);

        // Generate remaining points using Bayesian optimization
        let remaining_points = self.config.max_evaluations.saturating_sub(n_initial_points);

        for _ in 0..remaining_points {
            // Update Gaussian process with current observations
            self.update_gaussian_process(&mut bayesian_state, &combinations);

            // Find next point with highest acquisition function value
            let next_point = self.optimize_acquisition_function(
                search_space,
                &bayesian_state,
                acquisition_function,
            )?;

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

        // Normalize weights
        let weight_sum: f64 = weights.iter().sum();
        let normalized_weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

        // Allocate evaluations based on weights
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
                _ => {
                    // Fallback to random search for complex strategies
                    self.generate_random_combinations(search_space, n_evaluations)?
                }
            };

            all_combinations.extend(strategy_combinations);
        }

        // Shuffle to mix different strategies
        let mut rng = match self.config.random_seed {
            Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
            None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
        };

        use scirs2_core::random::seq::SliceRandom;
        all_combinations.shuffle(&mut rng);

        Ok(all_combinations)
    }

    /// Generate evolutionary search combinations using genetic algorithm
    pub fn generate_evolutionary_combinations(
        &self,
        search_space: &SearchSpace,
        population_size: usize,
        n_generations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut rng = match self.config.random_seed {
            Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
            None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
        };

        // Initialize population
        let mut population = self.generate_random_combinations(search_space, population_size)?;
        let mut all_combinations = population.clone();

        // Evolution loop
        for _generation in 0..n_generations {
            let mut new_population = Vec::new();

            // Elitism: keep best individual from previous generation
            if !population.is_empty() {
                new_population.push(population[0].clone());
            }

            // Generate new offspring
            while new_population.len() < population_size {
                // Selection: tournament selection
                let parent1 = self.tournament_selection(&population, &mut rng)?;
                let parent2 = self.tournament_selection(&population, &mut rng)?;

                // Crossover
                let (mut child1, mut child2) = if rng.gen_range(0.0..1.0) < crossover_rate {
                    self.crossover(&parent1, &parent2, search_space, &mut rng)?
                } else {
                    (parent1.clone(), parent2.clone())
                };

                // Mutation
                if rng.gen_range(0.0..1.0) < mutation_rate {
                    self.mutate(&mut child1, search_space, &mut rng)?;
                }
                if rng.gen_range(0.0..1.0) < mutation_rate {
                    self.mutate(&mut child2, search_space, &mut rng)?;
                }

                new_population.push(child1);
                if new_population.len() < population_size {
                    new_population.push(child2);
                }
            }

            population = new_population;
            all_combinations.extend(population.clone());

            // Early termination if we have enough evaluations
            if all_combinations.len() >= self.config.max_evaluations {
                break;
            }
        }

        // Trim to max evaluations
        all_combinations.truncate(self.config.max_evaluations);
        Ok(all_combinations)
    }

    /// Tournament selection for evolutionary algorithm
    fn tournament_selection(
        &self,
        population: &[HashMap<String, f64>],
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> Result<HashMap<String, f64>> {
        let tournament_size = 3.min(population.len());
        let mut best_individual = None;

        for _ in 0..tournament_size {
            let idx = rng.gen_range(0..population.len());
            let individual = &population[idx];

            // In a real implementation, we would evaluate fitness here
            // For now, just return the first selected individual
            if best_individual.is_none() {
                best_individual = Some(individual.clone());
            }
        }

        best_individual.ok_or_else(|| ClusteringError::InvalidInput("Empty population".to_string()))
    }

    /// Crossover operation for evolutionary algorithm
    fn crossover(
        &self,
        parent1: &HashMap<String, f64>,
        parent2: &HashMap<String, f64>,
        search_space: &SearchSpace,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> Result<(HashMap<String, f64>, HashMap<String, f64>)> {
        let mut child1 = HashMap::new();
        let mut child2 = HashMap::new();

        for (param_name, param_spec) in &search_space.parameters {
            let val1 = parent1.get(param_name).copied().unwrap_or(0.0);
            let val2 = parent2.get(param_name).copied().unwrap_or(0.0);

            // Uniform crossover
            if rng.gen_range(0.0..1.0) < 0.5 {
                child1.insert(param_name.clone(), val1);
                child2.insert(param_name.clone(), val2);
            } else {
                child1.insert(param_name.clone(), val2);
                child2.insert(param_name.clone(), val1);
            }
        }

        Ok((child1, child2))
    }

    /// Mutation operation for evolutionary algorithm
    fn mutate(
        &self,
        individual: &mut HashMap<String, f64>,
        search_space: &SearchSpace,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> Result<()> {
        for (param_name, param_spec) in &search_space.parameters {
            if rng.gen_range(0.0..1.0) < 0.1 {
                // 10% chance to mutate each parameter
                let new_value = self.sample_parameter(param_spec, rng);
                individual.insert(param_name.clone(), new_value);
            }
        }
        Ok(())
    }

    /// Sample a value from a parameter specification
    fn sample_parameter(&self, param_spec: &HyperParameter, rng: &mut scirs2_core::random::rngs::StdRng) -> f64 {
        match param_spec {
            HyperParameter::Integer { min, max } => rng.gen_range(*min..=*max) as f64,
            HyperParameter::Float { min, max } => rng.gen_range(*min..=*max),
            HyperParameter::Categorical { choices } => rng.gen_range(0..choices.len()) as f64,
            HyperParameter::Boolean => {
                if rng.gen_range(0.0..1.0) < 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            HyperParameter::LogUniform { min, max } => {
                let log_min = min.ln();
                let log_max = max.ln();
                let log_value = rng.gen_range(log_min..=log_max);
                log_value.exp()
            }
            HyperParameter::IntegerChoices { choices } => {
                let idx = rng.gen_range(0..choices.len());
                choices[idx] as f64
            }
        }
    }

    /// Generate SMBO combinations (simplified implementation)
    fn generate_smbo_combinations(
        &self,
        search_space: &SearchSpace,
        _surrogate_model: &super::config::SurrogateModel,
        acquisition_function: &AcquisitionFunction,
    ) -> Result<Vec<HashMap<String, f64>>> {
        // For now, fallback to Bayesian optimization
        self.generate_bayesian_combinations(search_space, 10, acquisition_function)
    }

    /// Generate multi-objective combinations
    fn generate_multi_objective_combinations(
        &self,
        search_space: &SearchSpace,
        _objectives: &[super::config::EvaluationMetric],
        base_strategy: &SearchStrategy,
    ) -> Result<Vec<HashMap<String, f64>>> {
        // For multi-objective optimization, we need to maintain a Pareto frontier
        // This is a simplified implementation

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
            _ => {
                // Fallback to random search
                self.generate_random_combinations(search_space, self.config.max_evaluations)?
            }
        };

        Ok(base_combinations)
    }

    /// Update Gaussian Process with current observations
    fn update_gaussian_process(
        &self,
        bayesian_state: &mut BayesianState,
        combinations: &[HashMap<String, f64>],
    ) {
        // For demonstration, we'll implement a simplified GP update
        // In practice, this would involve matrix operations and hyperparameter optimization

        if combinations.is_empty() {
            return;
        }

        // Convert parameter combinations to feature matrix
        let n_samples = combinations.len();
        let _n_features = bayesian_state.parameter_names.len();

        if n_samples < 2 {
            return;
        }

        // Update GP hyperparameters using maximum likelihood estimation
        self.optimize_gp_hyperparameters(bayesian_state, combinations);

        // Build covariance matrix
        let mut covariance = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                let x_i =
                    self.extract_feature_vector(&combinations[i], &bayesian_state.parameter_names);
                let x_j =
                    self.extract_feature_vector(&combinations[j], &bayesian_state.parameter_names);
                covariance[[i, j]] =
                    self.compute_kernel(&x_i, &x_j, &bayesian_state.gp_hyperparameters);
            }
        }

        // Add noise to diagonal
        for i in 0..n_samples {
            covariance[[i, i]] += bayesian_state.gp_hyperparameters.noise_variance;
        }

        bayesian_state.gp_covariance = Some(covariance);
    }

    /// Optimize acquisition function to find next evaluation point
    fn optimize_acquisition_function(
        &self,
        search_space: &SearchSpace,
        bayesian_state: &BayesianState,
        acquisition_function: &AcquisitionFunction,
    ) -> Result<HashMap<String, f64>> {
        let mut best_acquisition = f64::NEG_INFINITY;
        let mut best_point = HashMap::new();

        // Generate candidate points for acquisition optimization
        let n_candidates = 1000;
        let candidates = self.generate_random_combinations(search_space, n_candidates)?;

        for candidate in candidates {
            let acquisition_value = self.evaluate_acquisition_function(
                &candidate,
                bayesian_state,
                acquisition_function,
            );

            if acquisition_value > best_acquisition {
                best_acquisition = acquisition_value;
                best_point = candidate;
            }
        }

        Ok(best_point)
    }

    /// Evaluate acquisition function at a point
    fn evaluate_acquisition_function(
        &self,
        point: &HashMap<String, f64>,
        bayesian_state: &BayesianState,
        acquisition_function: &AcquisitionFunction,
    ) -> f64 {
        let x = self.extract_feature_vector(point, &bayesian_state.parameter_names);
        let (mean, variance) = self.predict_gp(&x, bayesian_state);
        let std_dev = variance.sqrt();

        match acquisition_function {
            AcquisitionFunction::ExpectedImprovement => {
                self.expected_improvement(mean, std_dev, bayesian_state.currentbest)
            }
            AcquisitionFunction::UpperConfidenceBound { beta } => mean + beta * std_dev,
            AcquisitionFunction::ProbabilityOfImprovement => {
                self.probability_of_improvement(mean, std_dev, bayesian_state.currentbest)
            }
            AcquisitionFunction::EntropySearch => {
                // Simplified entropy search implementation
                -variance * (variance.ln())
            }
            AcquisitionFunction::KnowledgeGradient => {
                // Simplified knowledge gradient
                std_dev * (1.0 / (1.0 + variance))
            }
            AcquisitionFunction::ThompsonSampling => {
                // Sample from posterior
                let mut rng = scirs2_core::random::thread_rng();
                let sample: f64 = rng.gen_range(0.0..1.0);
                mean + std_dev * self.inverse_normal_cdf(sample)
            }
        }
    }

    /// Expected Improvement acquisition function
    fn expected_improvement(&self, mean: f64, std_dev: f64, currentbest: f64) -> f64 {
        if std_dev <= 1e-10 {
            return 0.0;
        }

        let improvement = mean - currentbest;
        let z = improvement / std_dev;

        improvement * self.normal_cdf(z) + std_dev * self.normal_pdf(z)
    }

    /// Probability of Improvement acquisition function
    fn probability_of_improvement(&self, mean: f64, std_dev: f64, currentbest: f64) -> f64 {
        if std_dev <= 1e-10 {
            return if mean > currentbest { 1.0 } else { 0.0 };
        }

        let z = (mean - currentbest) / std_dev;
        self.normal_cdf(z)
    }

    /// Gaussian Process prediction
    fn predict_gp(&self, x: &[f64], bayesian_state: &BayesianState) -> (f64, f64) {
        if bayesian_state.observations.is_empty() {
            return (0.0, 1.0); // Prior mean and variance
        }

        // Simplified GP prediction - in practice would use proper matrix operations
        let mut mean = 0.0;
        let mut variance = 1.0;

        // Compute similarity-weighted average (simplified)
        let mut total_weight = 0.0;
        for (params, score) in &bayesian_state.observations {
            let x_obs = self.extract_feature_vector(params, &bayesian_state.parameter_names);
            let similarity = self.compute_kernel(x, &x_obs, &bayesian_state.gp_hyperparameters);
            mean += similarity * score;
            total_weight += similarity;
        }

        if total_weight > 1e-10 {
            mean /= total_weight;
            variance = 1.0 - total_weight.min(1.0); // Simplified variance calculation
        }

        (mean, variance.max(1e-6))
    }

    /// Compute kernel function
    fn compute_kernel(&self, x1: &[f64], x2: &[f64], hyperparams: &GpHyperparameters) -> f64 {
        match &hyperparams.kernel_type {
            KernelType::RBF { length_scale } => {
                let squared_distance: f64 =
                    x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                hyperparams.signal_variance
                    * (-squared_distance / (2.0 * length_scale.powi(2))).exp()
            }
            KernelType::Matern { length_scale, nu } => {
                let distance: f64 = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if distance == 0.0 {
                    hyperparams.signal_variance
                } else {
                    let scaled_distance = (2.0 * nu).sqrt() * distance / length_scale;
                    let bessel_term = if nu == &0.5 {
                        (-scaled_distance).exp()
                    } else if nu == &1.5 {
                        (1.0 + scaled_distance) * (-scaled_distance).exp()
                    } else {
                        // Simplified for other nu values
                        (-scaled_distance).exp()
                    };
                    hyperparams.signal_variance * bessel_term
                }
            }
            KernelType::Linear => {
                let dot_product: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum();
                hyperparams.signal_variance * dot_product
            }
            KernelType::Polynomial { degree } => {
                let dot_product: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum();
                hyperparams.signal_variance * (1.0 + dot_product).powf(*degree as f64)
            }
        }
    }

    /// Optimize GP hyperparameters using maximum likelihood
    fn optimize_gp_hyperparameters(
        &self,
        bayesian_state: &mut BayesianState,
        combinations: &[HashMap<String, f64>],
    ) {
        // Simplified hyperparameter optimization
        // In practice, this would use gradient-based optimization

        if combinations.len() < 3 {
            return;
        }

        // Estimate length scales based on data variance
        for (i, param_name) in bayesian_state.parameter_names.iter().enumerate() {
            let values: Vec<f64> = combinations
                .iter()
                .filter_map(|c| c.get(param_name))
                .copied()
                .collect();

            if !values.is_empty() {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

                if i < bayesian_state.gp_hyperparameters.length_scales.len() {
                    bayesian_state.gp_hyperparameters.length_scales[i] = variance.sqrt().max(0.1);
                }
            }
        }

        // Update signal and noise variance based on observations
        if !bayesian_state.observations.is_empty() {
            let scores: Vec<f64> = bayesian_state
                .observations
                .iter()
                .map(|(_, s)| *s)
                .collect();
            let score_mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let score_variance =
                scores.iter().map(|s| (s - score_mean).powi(2)).sum::<f64>() / scores.len() as f64;

            bayesian_state.gp_hyperparameters.signal_variance = score_variance.max(0.1);
            bayesian_state.gp_hyperparameters.noise_variance = (score_variance * 0.1).max(0.01);
        }
    }

    /// Extract feature vector from parameter map
    fn extract_feature_vector(
        &self,
        params: &HashMap<String, f64>,
        param_names: &[String],
    ) -> Vec<f64> {
        param_names
            .iter()
            .map(|name| params.get(name).copied().unwrap_or(0.0))
            .collect()
    }

    /// Standard normal CDF approximation
    fn normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + self.erf(x / 2.0_f64.sqrt()))
    }

    /// Standard normal PDF
    fn normal_pdf(&self, x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }

    /// Error function approximation
    fn erf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Inverse normal CDF approximation
    fn inverse_normal_cdf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        if (p - 0.5).abs() < 1e-10 {
            return 0.0;
        }

        // Beasley-Springer-Moro algorithm
        let a0 = -3.969683028665376e+01;
        let a1 = 2.209460984245205e+02;
        let a2 = -2.759285104469687e+02;
        let a3 = 1.383577518672690e+02;
        let a4 = -3.066479806614716e+01;
        let a5 = 2.506628277459239e+00;

        let b1 = -5.447609879822406e+01;
        let b2 = 1.615858368580409e+02;
        let b3 = -1.556989798598866e+02;
        let b4 = 6.680131188771972e+01;
        let b5 = -1.328068155288572e+01;

        let c0 = -7.784894002430293e-03;
        let c1 = -3.223964580411365e-01;
        let c2 = -2.400758277161838e+00;
        let c3 = -2.549732539343734e+00;
        let c4 = 4.374664141464968e+00;
        let c5 = 2.938163982698783e+00;

        let d1 = 7.784695709041462e-03;
        let d2 = 3.224671290700398e-01;
        let d3 = 2.445134137142996e+00;
        let d4 = 3.754408661907416e+00;

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            return (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5)
                / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }

        if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            return (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q
                / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
        }

        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    }
}
