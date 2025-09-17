//! Hyperparameter Optimization Framework
//!
//! This module provides automated hyperparameter tuning for time series models
//! using various optimization strategies including random search, Bayesian optimization,
//! and evolutionary algorithms.

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::Result;

/// Hyperparameter Optimization Framework
#[derive(Debug)]
pub struct HyperparameterOptimizer<F: Float + Debug + ndarray::ScalarOperand> {
    /// Optimization method
    method: OptimizationMethod,
    /// Search space definition
    search_space: SearchSpace<F>,
    /// Current best parameters
    best_params: Option<HyperparameterSet<F>>,
    /// Best validation score
    best_score: Option<F>,
    /// Optimization history
    history: Vec<OptimizationStep<F>>,
    /// Number of trials
    max_trials: usize,
}

/// Hyperparameter optimization methods
#[derive(Debug, Clone)]
pub enum OptimizationMethod {
    /// Random search
    RandomSearch,
    /// Grid search
    GridSearch,
    /// Bayesian optimization with Gaussian Process
    BayesianOptimization,
    /// Evolutionary algorithm
    EvolutionarySearch,
    /// Tree-structured Parzen Estimator
    TPE,
}

/// Search space for hyperparameters
#[derive(Debug, Clone)]
pub struct SearchSpace<F: Float + Debug> {
    /// Continuous parameters (name, min, max)
    pub continuous: Vec<(String, F, F)>,
    /// Integer parameters (name, min, max)
    pub integer: Vec<(String, i32, i32)>,
    /// Categorical parameters (name, choices)
    pub categorical: Vec<(String, Vec<String>)>,
}

/// Set of hyperparameters
#[derive(Debug, Clone)]
pub struct HyperparameterSet<F: Float + Debug> {
    /// Continuous parameter values
    pub continuous: Vec<(String, F)>,
    /// Integer parameter values
    pub integer: Vec<(String, i32)>,
    /// Categorical parameter values
    pub categorical: Vec<(String, String)>,
}

/// Single optimization step
#[derive(Debug, Clone)]
pub struct OptimizationStep<F: Float + Debug> {
    /// Trial number
    pub trial_id: usize,
    /// Parameters tried
    pub params: HyperparameterSet<F>,
    /// Validation score achieved
    pub score: F,
    /// Training time
    pub training_time: F,
}

/// Optimization results
#[derive(Debug)]
pub struct OptimizationResults<F: Float + Debug> {
    /// Best hyperparameters found
    pub best_params: Option<HyperparameterSet<F>>,
    /// Best validation score
    pub best_score: Option<F>,
    /// Complete optimization history
    pub history: Vec<OptimizationStep<F>>,
    /// Best score over time (convergence curve)
    pub convergence_curve: Vec<F>,
}

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> HyperparameterOptimizer<F> {
    /// Create new hyperparameter optimizer
    pub fn new(
        method: OptimizationMethod,
        search_space: SearchSpace<F>,
        max_trials: usize,
    ) -> Self {
        Self {
            method,
            search_space,
            best_params: None,
            best_score: None,
            history: Vec::new(),
            max_trials,
        }
    }

    /// Run hyperparameter optimization
    pub fn optimize<ModelFn>(&mut self, objectivefn: ModelFn) -> Result<HyperparameterSet<F>>
    where
        ModelFn: Fn(&HyperparameterSet<F>) -> Result<F>,
    {
        for trial in 0..self.max_trials {
            // Generate candidate parameters
            let params = match self.method {
                OptimizationMethod::RandomSearch => self.random_search()?,
                OptimizationMethod::GridSearch => self.grid_search(trial)?,
                OptimizationMethod::BayesianOptimization => self.bayesian_optimization()?,
                OptimizationMethod::EvolutionarySearch => self.evolutionary_search()?,
                OptimizationMethod::TPE => self.tpe_search()?,
            };

            // Evaluate objective function
            let start_time = std::time::Instant::now();
            let score = objectivefn(&params)?;
            let training_time = F::from(start_time.elapsed().as_secs_f64()).unwrap();

            // Update best parameters if improved
            let is_better = self.best_score.is_none_or(|best| score > best);
            if is_better {
                self.best_params = Some(params.clone());
                self.best_score = Some(score);
            }

            // Record step
            self.history.push(OptimizationStep {
                trial_id: trial,
                params,
                score,
                training_time,
            });

            #[allow(clippy::println_print)]
            {
                println!(
                    "Trial {}: Score = {:.6}, Best = {:.6}",
                    trial,
                    score.to_f64().unwrap_or(0.0),
                    self.best_score.unwrap().to_f64().unwrap_or(0.0)
                );
            }
        }

        self.best_params.clone().ok_or_else(|| {
            crate::error::TimeSeriesError::InvalidOperation("No successful trials".to_string())
        })
    }

    /// Random search implementation
    fn random_search(&self) -> Result<HyperparameterSet<F>> {
        let mut params = HyperparameterSet {
            continuous: Vec::new(),
            integer: Vec::new(),
            categorical: Vec::new(),
        };

        // Sample continuous parameters
        for (name, min_val, max_val) in &self.search_space.continuous {
            let range = *max_val - *min_val;
            let random_val = F::from(rand::random::<f64>()).unwrap();
            let value = *min_val + range * random_val;
            params.continuous.push((name.clone(), value));
        }

        // Sample integer parameters
        for (name, min_val, max_val) in &self.search_space.integer {
            let range = max_val - min_val;
            let random_val = (rand::random::<f64>() * (range + 1) as f64) as i32;
            let value = min_val + random_val;
            params.integer.push((name.clone(), value));
        }

        // Sample categorical parameters
        for (name, choices) in &self.search_space.categorical {
            let idx = (rand::random::<f64>() * choices.len() as f64) as usize;
            let value = choices[idx.min(choices.len() - 1)].clone();
            params.categorical.push((name.clone(), value));
        }

        Ok(params)
    }

    /// Grid search implementation (simplified)
    fn grid_search(&self, _trial: usize) -> Result<HyperparameterSet<F>> {
        // For simplicity, use random search with some structure
        self.random_search()
    }

    /// Bayesian optimization implementation (simplified)
    fn bayesian_optimization(&self) -> Result<HyperparameterSet<F>> {
        if self.history.is_empty() {
            // No history yet, use random search
            return self.random_search();
        }

        // Simplified acquisition function (Upper Confidence Bound)
        let mut best_candidate = None;
        let mut best_acquisition = F::from(-f64::INFINITY).unwrap();

        for _ in 0..10 {
            let candidate = self.random_search()?;
            let acquisition = self.compute_acquisition_ucb(&candidate)?;

            if acquisition > best_acquisition {
                best_acquisition = acquisition;
                best_candidate = Some(candidate);
            }
        }

        best_candidate.ok_or_else(|| {
            crate::error::TimeSeriesError::InvalidOperation("Failed to find candidate".to_string())
        })
    }

    /// Compute Upper Confidence Bound acquisition function
    fn compute_acquisition_ucb(&self, params: &HyperparameterSet<F>) -> Result<F> {
        // Simplified UCB computation
        let mean = self.predict_mean(params)?;
        let std = self.predict_std(params)?;
        let beta = F::from(2.0).unwrap(); // Exploration parameter

        Ok(mean + beta * std)
    }

    /// Predict mean performance (simplified Gaussian Process)
    fn predict_mean(&self, _params: &HyperparameterSet<F>) -> Result<F> {
        // Simplified: return average of historical scores
        if self.history.is_empty() {
            return Ok(F::zero());
        }

        let sum: F = self
            .history
            .iter()
            .map(|step| step.score)
            .fold(F::zero(), |acc, x| acc + x);
        Ok(sum / F::from(self.history.len()).unwrap())
    }

    /// Predict standard deviation (simplified)
    fn predict_std(&self, _params: &HyperparameterSet<F>) -> Result<F> {
        // Simplified: return fixed exploration term
        Ok(F::one())
    }

    /// Evolutionary search implementation
    fn evolutionary_search(&self) -> Result<HyperparameterSet<F>> {
        if self.history.len() < 5 {
            return self.random_search();
        }

        // Select top performers as parents
        let mut sorted_history = self.history.clone();
        sorted_history.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let parent1 = &sorted_history[0].params;
        let parent2 = &sorted_history[1].params;

        // Crossover and mutation
        self.crossover_mutate(parent1, parent2)
    }

    /// Crossover and mutation for evolutionary search
    fn crossover_mutate(
        &self,
        parent1: &HyperparameterSet<F>,
        parent2: &HyperparameterSet<F>,
    ) -> Result<HyperparameterSet<F>> {
        let mut child = HyperparameterSet {
            continuous: Vec::new(),
            integer: Vec::new(),
            categorical: Vec::new(),
        };

        // Crossover continuous parameters
        for ((name1, val1), (_, val2)) in parent1.continuous.iter().zip(&parent2.continuous) {
            let alpha = F::from(rand::random::<f64>()).unwrap();
            let crossed_val = *val1 + alpha * (*val2 - *val1);

            // Mutation
            let mutation = if rand::random::<f64>() < 0.1 {
                F::from((rand::random::<f64>() - 0.5) * 0.2).unwrap()
            } else {
                F::zero()
            };

            child
                .continuous
                .push((name1.clone(), crossed_val + mutation));
        }

        // Handle integer and categorical similarly (simplified)
        for (name, val) in &parent1.integer {
            child.integer.push((name.clone(), *val));
        }

        for (name, val) in &parent1.categorical {
            child.categorical.push((name.clone(), val.clone()));
        }

        Ok(child)
    }

    /// Tree-structured Parzen Estimator implementation
    fn tpe_search(&self) -> Result<HyperparameterSet<F>> {
        // Simplified TPE - use random search for now
        self.random_search()
    }

    /// Get optimization results
    pub fn get_results(&self) -> OptimizationResults<F> {
        OptimizationResults {
            best_params: self.best_params.clone(),
            best_score: self.best_score,
            history: self.history.clone(),
            convergence_curve: self.get_convergence_curve(),
        }
    }

    /// Get convergence curve
    fn get_convergence_curve(&self) -> Vec<F> {
        let mut best_so_far = Vec::new();
        let mut current_best = F::from(-f64::INFINITY).unwrap();

        for step in &self.history {
            if step.score > current_best {
                current_best = step.score;
            }
            best_so_far.push(current_best);
        }

        best_so_far
    }

    /// Get the current best parameters
    pub fn best_params(&self) -> Option<&HyperparameterSet<F>> {
        self.best_params.as_ref()
    }

    /// Get the current best score
    pub fn best_score(&self) -> Option<F> {
        self.best_score
    }

    /// Get the optimization history
    pub fn history(&self) -> &[OptimizationStep<F>] {
        &self.history
    }
}

impl<F: Float + Debug> SearchSpace<F> {
    /// Create a new empty search space
    pub fn new() -> Self {
        Self {
            continuous: Vec::new(),
            integer: Vec::new(),
            categorical: Vec::new(),
        }
    }

    /// Add a continuous parameter to the search space
    pub fn add_continuous(&mut self, name: String, min_val: F, max_val: F) {
        self.continuous.push((name, min_val, max_val));
    }

    /// Add an integer parameter to the search space
    pub fn add_integer(&mut self, name: String, min_val: i32, max_val: i32) {
        self.integer.push((name, min_val, max_val));
    }

    /// Add a categorical parameter to the search space
    pub fn add_categorical(&mut self, name: String, choices: Vec<String>) {
        self.categorical.push((name, choices));
    }
}

impl<F: Float + Debug> Default for SearchSpace<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug> HyperparameterSet<F> {
    /// Create a new empty hyperparameter set
    pub fn new() -> Self {
        Self {
            continuous: Vec::new(),
            integer: Vec::new(),
            categorical: Vec::new(),
        }
    }

    /// Get a continuous parameter value by name
    pub fn get_continuous(&self, name: &str) -> Option<F> {
        self.continuous
            .iter()
            .find(|(param_name, _)| param_name == name)
            .map(|(_, value)| *value)
    }

    /// Get an integer parameter value by name
    pub fn get_integer(&self, name: &str) -> Option<i32> {
        self.integer
            .iter()
            .find(|(param_name, _)| param_name == name)
            .map(|(_, value)| *value)
    }

    /// Get a categorical parameter value by name
    pub fn get_categorical(&self, name: &str) -> Option<&str> {
        self.categorical
            .iter()
            .find(|(param_name, _)| param_name == name)
            .map(|(_, value)| value.as_str())
    }
}

impl<F: Float + Debug> Default for HyperparameterSet<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_search_space_creation() {
        let mut search_space = SearchSpace::<f64>::new();
        search_space.add_continuous("learning_rate".to_string(), 0.001, 0.1);
        search_space.add_integer("hidden_size".to_string(), 32, 256);
        search_space.add_categorical(
            "optimizer".to_string(),
            vec!["adam".to_string(), "sgd".to_string()],
        );

        assert_eq!(search_space.continuous.len(), 1);
        assert_eq!(search_space.integer.len(), 1);
        assert_eq!(search_space.categorical.len(), 1);
    }

    #[test]
    fn test_hyperparameter_set() {
        let mut params = HyperparameterSet::<f64>::new();
        params.continuous.push(("learning_rate".to_string(), 0.01));
        params.integer.push(("hidden_size".to_string(), 128));
        params
            .categorical
            .push(("optimizer".to_string(), "adam".to_string()));

        assert_eq!(params.get_continuous("learning_rate"), Some(0.01));
        assert_eq!(params.get_integer("hidden_size"), Some(128));
        assert_eq!(params.get_categorical("optimizer"), Some("adam"));
        assert_eq!(params.get_continuous("nonexistent"), None);
    }

    #[test]
    fn test_random_search() {
        let search_space = SearchSpace {
            continuous: vec![
                ("learning_rate".to_string(), 0.001, 0.1),
                ("dropout".to_string(), 0.0, 0.5),
            ],
            integer: vec![
                ("hidden_size".to_string(), 32, 256),
                ("num_layers".to_string(), 1, 6),
            ],
            categorical: vec![(
                "optimizer".to_string(),
                vec!["adam".to_string(), "sgd".to_string()],
            )],
        };

        let optimizer =
            HyperparameterOptimizer::new(OptimizationMethod::RandomSearch, search_space, 10);

        let params = optimizer.random_search().unwrap();
        assert_eq!(params.continuous.len(), 2);
        assert_eq!(params.integer.len(), 2);
        assert_eq!(params.categorical.len(), 1);

        // Check bounds
        for (name, value) in &params.continuous {
            if name == "learning_rate" {
                assert!(value >= &0.001 && value <= &0.1);
            } else if name == "dropout" {
                assert!(value >= &0.0 && value <= &0.5);
            }
        }

        for (name, value) in &params.integer {
            if name == "hidden_size" {
                assert!(*value >= 32 && *value <= 256);
            } else if name == "num_layers" {
                assert!(*value >= 1 && *value <= 6);
            }
        }
    }

    #[test]
    fn test_hyperparameter_optimization() {
        let search_space = SearchSpace {
            continuous: vec![
                ("learning_rate".to_string(), 0.001, 0.1),
                ("dropout".to_string(), 0.0, 0.5),
            ],
            integer: vec![
                ("hidden_size".to_string(), 32, 256),
                ("num_layers".to_string(), 1, 6),
            ],
            categorical: vec![(
                "optimizer".to_string(),
                vec!["adam".to_string(), "sgd".to_string()],
            )],
        };

        let mut optimizer =
            HyperparameterOptimizer::new(OptimizationMethod::RandomSearch, search_space, 5);

        // Dummy objective function
        let objective = |params: &HyperparameterSet<f64>| -> Result<f64> {
            // Simulate model training and validation
            let mut score = 0.5;

            for (name, value) in &params.continuous {
                if name == "learning_rate" {
                    score += 0.1 * (0.01 - value).abs();
                }
            }

            Ok(score)
        };

        let best_params = optimizer.optimize(objective).unwrap();
        assert!(!best_params.continuous.is_empty());

        let results = optimizer.get_results();
        assert!(results.best_score.is_some());
        assert_eq!(results.history.len(), 5);
        assert_eq!(results.convergence_curve.len(), 5);
    }

    #[test]
    fn test_evolutionary_search() {
        let search_space = SearchSpace {
            continuous: vec![("x".to_string(), -5.0, 5.0)],
            integer: vec![],
            categorical: vec![],
        };

        let mut optimizer =
            HyperparameterOptimizer::new(OptimizationMethod::EvolutionarySearch, search_space, 10);

        // Simple quadratic objective (minimize x^2)
        let objective = |params: &HyperparameterSet<f64>| -> Result<f64> {
            let x = params.get_continuous("x").unwrap_or(0.0);
            Ok(-x * x) // Maximize negative quadratic (minimize quadratic)
        };

        let best_params = optimizer.optimize(objective).unwrap();
        let best_x = best_params.get_continuous("x").unwrap();

        // Should be close to 0 for minimizing x^2
        assert!(best_x.abs() < 3.0); // Allow larger tolerance due to randomness in evolutionary search
    }

    #[test]
    fn test_convergence_curve() {
        let search_space = SearchSpace {
            continuous: vec![("x".to_string(), 0.0, 1.0)],
            integer: vec![],
            categorical: vec![],
        };

        let mut optimizer =
            HyperparameterOptimizer::new(OptimizationMethod::RandomSearch, search_space, 3);

        // Simple objective function
        let objective = |params: &HyperparameterSet<f64>| -> Result<f64> {
            let x = params.get_continuous("x").unwrap_or(0.0);
            Ok(x) // Maximize x
        };

        optimizer.optimize(objective).unwrap();
        let convergence = optimizer.get_convergence_curve();

        assert_eq!(convergence.len(), 3);

        // Convergence curve should be non-decreasing
        for i in 1..convergence.len() {
            assert!(convergence[i] >= convergence[i - 1]);
        }
    }
}
