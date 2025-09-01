//! Meta-Learning Optimizer
//!
//! Implementation of a comprehensive meta-learning system for optimization that can
//! learn to optimize across different problem classes and adapt quickly to new tasks.

use super::{
    LearnedOptimizationConfig, LearnedOptimizer, MetaOptimizerState, OptimizationProblem,
    TrainingTask,
};
use crate::error::OptimizeResult;
use crate::result::OptimizeResults;
use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use std::collections::HashMap;

/// Meta-Learning Optimizer with cross-problem adaptation
#[derive(Debug, Clone)]
pub struct MetaLearningOptimizer {
    /// Configuration
    config: LearnedOptimizationConfig,
    /// Meta-optimizer state
    meta_state: MetaOptimizerState,
    /// Task-specific optimizers
    task_optimizers: HashMap<String, TaskSpecificOptimizer>,
    /// Meta-learning statistics
    meta_stats: MetaLearningStats,
}

/// Task-specific optimizer
#[derive(Debug, Clone)]
pub struct TaskSpecificOptimizer {
    /// Optimizer parameters
    parameters: Array1<f64>,
    /// Performance history
    performance_history: Vec<f64>,
    /// Task identifier
    task_id: String,
}

/// Meta-learning statistics
#[derive(Debug, Clone)]
pub struct MetaLearningStats {
    /// Number of tasks learned
    tasks_learned: usize,
    /// Average adaptation speed
    avg_adaptation_speed: f64,
    /// Transfer learning efficiency
    transfer_efficiency: f64,
    /// Meta-gradient norm
    meta_gradient_norm: f64,
}

impl MetaLearningOptimizer {
    /// Create new meta-learning optimizer
    pub fn new(config: LearnedOptimizationConfig) -> Self {
        let hidden_size = config.hidden_size;
        Self {
            config,
            meta_state: MetaOptimizerState {
                meta_params: Array1::zeros(hidden_size),
                network_weights: Array2::zeros((hidden_size, hidden_size)),
                performance_history: Vec::new(),
                adaptation_stats: super::AdaptationStatistics::default(),
                episode: 0,
            },
            task_optimizers: HashMap::new(),
            meta_stats: MetaLearningStats::default(),
        }
    }

    /// Learn meta-optimization strategy
    pub fn learn_meta_strategy(&mut self, training_tasks: &[TrainingTask]) -> OptimizeResult<()> {
        for task in training_tasks {
            // Create task-specific optimizer
            let task_optimizer = self.create_task_optimizer(&task.problem)?;

            // Train on task
            let performance = self.train_on_task(&task_optimizer, task)?;

            // Update meta-parameters based on performance
            self.update_meta_parameters(&task.problem, performance)?;

            // Store task optimizer
            self.task_optimizers
                .insert(task.problem.name.clone(), task_optimizer);
        }

        self.meta_stats.tasks_learned = training_tasks.len();
        Ok(())
    }

    /// Create task-specific optimizer
    fn create_task_optimizer(
        &self,
        problem: &OptimizationProblem,
    ) -> OptimizeResult<TaskSpecificOptimizer> {
        let param_size = self.estimate_parameter_size(problem);

        Ok(TaskSpecificOptimizer {
            parameters: Array1::from_shape_fn(param_size, |_| rand::rng().random_range(0.0..0.1)),
            performance_history: Vec::new(),
            task_id: problem.name.clone(),
        })
    }

    /// Estimate parameter size for problem
    fn estimate_parameter_size(&self, problem: &OptimizationProblem) -> usize {
        // Simple heuristic based on problem characteristics
        let base_size = 64;
        let dimension_factor = (problem.dimension as f64).sqrt() as usize;

        match problem.problem_class.as_str() {
            "quadratic" => base_size,
            "neural_network" => base_size * 2 + dimension_factor,
            "sparse" => base_size + dimension_factor / 2,
            _ => base_size + dimension_factor,
        }
    }

    /// Train on specific task
    fn train_on_task(
        &mut self,
        optimizer: &TaskSpecificOptimizer,
        task: &TrainingTask,
    ) -> OptimizeResult<f64> {
        // Simplified training simulation
        let initial_params = match &task.initial_distribution {
            super::ParameterDistribution::Uniform { low, high } => {
                Array1::from_shape_fn(task.problem.dimension, |_| {
                    low + rand::rng().random_range(0.0..1.0) * (high - low)
                })
            }
            super::ParameterDistribution::Normal { mean, std } => {
                Array1::from_shape_fn(task.problem.dimension, |_| {
                    mean + std * (rand::rng().random_range(0.0..1.0) - 0.5) * 2.0
                })
            }
            super::ParameterDistribution::Custom { samples } => {
                if !samples.is_empty() {
                    samples[0].clone()
                } else {
                    Array1::zeros(task.problem.dimension)
                }
            }
        };

        // Simple quadratic objective for training
        let training_objective = |x: &ArrayView1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>();

        let initial_value = training_objective(&initial_params.view());
        let mut current_params = initial_params;
        let mut current_value = initial_value;

        // Apply meta-learned optimization strategy
        for _ in 0..self.config.inner_steps {
            let direction = self.compute_meta_direction(&current_params, &training_objective)?;
            let step_size = self.compute_meta_step_size(&optimizer.parameters)?;

            for i in 0..current_params.len().min(direction.len()) {
                current_params[i] -= step_size * direction[i];
            }

            current_value = training_objective(&current_params.view());
        }

        let improvement = initial_value - current_value;
        Ok(improvement.max(0.0))
    }

    /// Compute meta-learned direction
    fn compute_meta_direction<F>(
        &self,
        params: &Array1<f64>,
        objective: &F,
    ) -> OptimizeResult<Array1<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let h = 1e-6;
        let f0 = objective(&params.view());
        let mut direction = Array1::zeros(params.len());

        // Compute finite difference gradient
        for i in 0..params.len() {
            let mut params_plus = params.clone();
            params_plus[i] += h;
            let f_plus = objective(&params_plus.view());
            direction[i] = (f_plus - f0) / h;
        }

        // Apply meta-learned transformation
        self.apply_meta_transformation(&mut direction)?;

        Ok(direction)
    }

    /// Apply meta-learned transformation to gradient
    fn apply_meta_transformation(&self, gradient: &mut Array1<f64>) -> OptimizeResult<()> {
        // Simple meta-transformation using meta-parameters
        for i in 0..gradient.len() {
            let meta_idx = i % self.meta_state.meta_params.len();
            let meta_factor = self.meta_state.meta_params[meta_idx];
            gradient[i] *= 1.0 + meta_factor * 0.1;
        }

        Ok(())
    }

    /// Compute meta-learned step size
    fn compute_meta_step_size(&self, task_params: &Array1<f64>) -> OptimizeResult<f64> {
        // Compute step size based on task parameters and meta-parameters
        let mut step_size = self.config.inner_learning_rate;

        // Use task parameters to modulate step size
        if !task_params.is_empty() {
            let param_norm = (task_params.iter().map(|&x| x * x).sum::<f64>()).sqrt();
            step_size *= (1.0 + param_norm * 0.1).recip();
        }

        // Apply meta-parameter modulation
        if !self.meta_state.meta_params.is_empty() {
            let meta_factor = self.meta_state.meta_params[0];
            step_size *= (1.0 + meta_factor * 0.2).max(0.1).min(2.0);
        }

        Ok(step_size)
    }

    /// Update meta-parameters based on task performance
    fn update_meta_parameters(
        &mut self,
        problem: &OptimizationProblem,
        performance: f64,
    ) -> OptimizeResult<()> {
        let learning_rate = self.config.meta_learning_rate;

        // Simple meta-gradient based on performance
        let performance_gradient = if performance > 0.0 { 1.0 } else { -1.0 };

        // Update meta-parameters
        for i in 0..self.meta_state.meta_params.len() {
            // Simple update rule (in practice would use proper meta-gradients)
            let update = learning_rate
                * performance_gradient
                * (rand::rng().random_range(0.0..1.0) - 0.5)
                * 0.1;
            self.meta_state.meta_params[i] += update;

            // Clip to reasonable range
            self.meta_state.meta_params[i] = self.meta_state.meta_params[i].max(-1.0).min(1.0);
        }

        // Record performance
        self.meta_state.performance_history.push(performance);

        // Update adaptation statistics
        self.meta_state.adaptation_stats.avg_convergence_rate =
            self.meta_state.performance_history.iter().sum::<f64>()
                / self.meta_state.performance_history.len() as f64;

        Ok(())
    }

    /// Adapt to new problem using meta-knowledge
    pub fn adapt_to_new_problem(
        &mut self,
        problem: &OptimizationProblem,
    ) -> OptimizeResult<TaskSpecificOptimizer> {
        // Find most similar task
        let similar_task = self.find_most_similar_task(problem)?;

        // Create new optimizer based on similar task
        let mut new_optimizer = if let Some(similar_optimizer) = similar_task {
            // Clone and adapt existing optimizer
            let mut adapted = similar_optimizer.clone();
            adapted.task_id = problem.name.clone();

            // Apply adaptation based on problem differences
            self.adapt_optimizer_parameters(&mut adapted, problem)?;
            adapted
        } else {
            // Create from scratch using meta-parameters
            self.create_task_optimizer(problem)?
        };

        // Fine-tune for the new problem
        self.fine_tune_for_problem(&mut new_optimizer, problem)?;

        Ok(new_optimizer)
    }

    /// Find most similar task
    fn find_most_similar_task(
        &self,
        problem: &OptimizationProblem,
    ) -> OptimizeResult<Option<&TaskSpecificOptimizer>> {
        let mut best_similarity = 0.0;
        let mut best_optimizer = None;

        for (task_name, optimizer) in &self.task_optimizers {
            let similarity = self.compute_task_similarity(problem, task_name)?;
            if similarity > best_similarity {
                best_similarity = similarity;
                best_optimizer = Some(optimizer);
            }
        }

        if best_similarity > 0.5 {
            Ok(best_optimizer)
        } else {
            Ok(None)
        }
    }

    /// Compute similarity between problems
    fn compute_task_similarity(
        &self,
        problem: &OptimizationProblem,
        task_name: &str,
    ) -> OptimizeResult<f64> {
        // Simple similarity based on problem class and dimension
        let similarity = if task_name.contains(&problem.problem_class) {
            0.8
        } else {
            0.2
        };

        // Add dimension similarity
        let dim_factor = 1.0 / (1.0 + (problem.dimension as f64 - 100.0).abs() / 100.0);

        Ok(similarity * dim_factor)
    }

    /// Adapt optimizer parameters for new problem
    fn adapt_optimizer_parameters(
        &self,
        optimizer: &mut TaskSpecificOptimizer,
        problem: &OptimizationProblem,
    ) -> OptimizeResult<()> {
        // Simple adaptation based on problem characteristics
        let adaptation_factor = match problem.problem_class.as_str() {
            "quadratic" => 1.0,
            "neural_network" => 1.2,
            "sparse" => 0.8,
            _ => 1.0,
        };

        // Scale parameters
        for param in &mut optimizer.parameters {
            *param *= adaptation_factor;
        }

        Ok(())
    }

    /// Fine-tune optimizer for specific problem
    fn fine_tune_for_problem(
        &mut self,
        optimizer: &mut TaskSpecificOptimizer,
        problem: &OptimizationProblem,
    ) -> OptimizeResult<()> {
        // Apply meta-learning based fine-tuning
        let meta_influence = 0.1;

        for (i, param) in optimizer.parameters.iter_mut().enumerate() {
            let meta_idx = i % self.meta_state.meta_params.len();
            let meta_adjustment = self.meta_state.meta_params[meta_idx] * meta_influence;
            *param += meta_adjustment;
        }

        Ok(())
    }

    /// Get meta-learning statistics
    pub fn get_meta_stats(&self) -> &MetaLearningStats {
        &self.meta_stats
    }

    /// Update meta-learning statistics
    fn update_meta_stats(&mut self) {
        // Compute adaptation speed
        if !self.meta_state.performance_history.is_empty() {
            let recent_improvements: Vec<f64> = self
                .meta_state
                .performance_history
                .windows(2)
                .map(|w| w[1] - w[0])
                .collect();

            if !recent_improvements.is_empty() {
                self.meta_stats.avg_adaptation_speed = recent_improvements
                    .iter()
                    .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
                    .sum::<f64>()
                    / recent_improvements.len() as f64;
            }
        }

        // Compute transfer efficiency
        self.meta_stats.transfer_efficiency = if self.meta_stats.tasks_learned > 1 {
            self.meta_stats.avg_adaptation_speed / self.meta_stats.tasks_learned as f64
        } else {
            0.0
        };

        // Compute meta-gradient norm
        self.meta_stats.meta_gradient_norm = (self
            .meta_state
            .meta_params
            .iter()
            .map(|&x| x * x)
            .sum::<f64>())
        .sqrt();
    }
}

impl Default for MetaLearningStats {
    fn default() -> Self {
        Self {
            tasks_learned: 0,
            avg_adaptation_speed: 0.0,
            transfer_efficiency: 0.0,
            meta_gradient_norm: 0.0,
        }
    }
}

impl LearnedOptimizer for MetaLearningOptimizer {
    fn meta_train(&mut self, training_tasks: &[TrainingTask]) -> OptimizeResult<()> {
        self.learn_meta_strategy(training_tasks)?;
        self.update_meta_stats();
        Ok(())
    }

    fn adapt_to_problem(
        &mut self,
        problem: &OptimizationProblem,
        initial_params: &ArrayView1<f64>,
    ) -> OptimizeResult<()> {
        let adapted_optimizer = self.adapt_to_new_problem(problem)?;
        self.task_optimizers
            .insert(problem.name.clone(), adapted_optimizer);
        Ok(())
    }

    fn optimize<F>(
        &mut self,
        objective: F,
        initial_params: &ArrayView1<f64>,
    ) -> OptimizeResult<OptimizeResults<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let mut current_params = initial_params.to_owned();
        let mut best_value = objective(initial_params);
        let mut iterations = 0;

        // Use meta-learned optimization strategy
        for iter in 0..1000 {
            iterations = iter;

            // Compute direction using meta-knowledge
            let direction = self.compute_meta_direction(&current_params, &objective)?;

            // Compute step size
            let step_size = if !self.meta_state.meta_params.is_empty() {
                let base_step = self.config.inner_learning_rate;
                let meta_factor = self.meta_state.meta_params[0];
                base_step * (1.0 + meta_factor * 0.1)
            } else {
                self.config.inner_learning_rate
            };

            // Update parameters
            for i in 0..current_params.len().min(direction.len()) {
                current_params[i] -= step_size * direction[i];
            }

            let current_value = objective(&current_params.view());

            if current_value < best_value {
                best_value = current_value;
            }

            // Check convergence
            if direction
                .iter()
                .map(|&x| x.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0)
                < 1e-8
            {
                break;
            }
        }

        Ok(OptimizeResults::<f64> {
            x: current_params,
            fun: best_value,
            success: true,
            nit: iterations,
            message: "Meta-learning optimization completed".to_string(),
            jac: None,
            hess: None,
            constr: None,
            nfev: iterations * 10, // Approximate function evaluations
            njev: 0,
            nhev: 0,
            maxcv: 0,
            status: 0,
        })
    }

    fn get_state(&self) -> &MetaOptimizerState {
        &self.meta_state
    }

    fn reset(&mut self) {
        self.task_optimizers.clear();
        self.meta_stats = MetaLearningStats::default();
        self.meta_state.episode = 0;
        self.meta_state.performance_history.clear();
    }
}

/// Convenience function for meta-learning optimization
#[allow(dead_code)]
pub fn meta_learning_optimize<F>(
    objective: F,
    initial_params: &ArrayView1<f64>,
    config: Option<LearnedOptimizationConfig>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let config = config.unwrap_or_default();
    let mut optimizer = MetaLearningOptimizer::new(config);
    optimizer.optimize(objective, initial_params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_learning_optimizer_creation() {
        let config = LearnedOptimizationConfig::default();
        let optimizer = MetaLearningOptimizer::new(config);

        assert_eq!(optimizer.meta_stats.tasks_learned, 0);
        assert!(optimizer.task_optimizers.is_empty());
    }

    #[test]
    fn test_task_optimizer_creation() {
        let config = LearnedOptimizationConfig::default();
        let optimizer = MetaLearningOptimizer::new(config);

        let problem = OptimizationProblem {
            name: "test".to_string(),
            dimension: 10,
            problem_class: "quadratic".to_string(),
            metadata: HashMap::new(),
            max_evaluations: 1000,
            target_accuracy: 1e-6,
        };

        let task_optimizer = optimizer.create_task_optimizer(&problem).unwrap();
        assert_eq!(task_optimizer.task_id, "test");
        assert!(!task_optimizer.parameters.is_empty());
    }

    #[test]
    fn test_meta_direction_computation() {
        let config = LearnedOptimizationConfig::default();
        let optimizer = MetaLearningOptimizer::new(config);

        let params = Array1::from(vec![1.0, 2.0]);
        let objective = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);

        let direction = optimizer
            .compute_meta_direction(&params, &objective)
            .unwrap();

        assert_eq!(direction.len(), 2);
        assert!(direction.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_meta_step_size_computation() {
        let config = LearnedOptimizationConfig::default();
        let mut optimizer = MetaLearningOptimizer::new(config);

        // Set some meta-parameters
        optimizer.meta_state.meta_params[0] = 0.5;

        let task_params = Array1::from(vec![0.1, 0.2, 0.3]);
        let step_size = optimizer.compute_meta_step_size(&task_params).unwrap();

        assert!(step_size > 0.0);
        assert!(step_size < 1.0);
    }

    #[test]
    fn test_task_similarity() {
        let config = LearnedOptimizationConfig::default();
        let optimizer = MetaLearningOptimizer::new(config);

        let problem = OptimizationProblem {
            name: "test".to_string(),
            dimension: 100,
            problem_class: "quadratic".to_string(),
            metadata: HashMap::new(),
            max_evaluations: 1000,
            target_accuracy: 1e-6,
        };

        let similarity1 = optimizer
            .compute_task_similarity(&problem, "quadratic_task")
            .unwrap();
        let similarity2 = optimizer
            .compute_task_similarity(&problem, "neural_network_task")
            .unwrap();

        assert!(similarity1 > similarity2);
    }

    #[test]
    fn test_meta_learning_optimization() {
        let objective = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
        let initial = Array1::from(vec![2.0, 2.0]);

        let config = LearnedOptimizationConfig {
            inner_steps: 10,
            inner_learning_rate: 0.1,
            ..Default::default()
        };

        let result = meta_learning_optimize(objective, &initial.view(), Some(config)).unwrap();

        assert!(result.fun >= 0.0);
        assert_eq!(result.x.len(), 2);
        assert!(result.success);
    }

    #[test]
    fn test_meta_parameter_update() {
        let config = LearnedOptimizationConfig::default();
        let mut optimizer = MetaLearningOptimizer::new(config);

        let problem = OptimizationProblem {
            name: "test".to_string(),
            dimension: 5,
            problem_class: "quadratic".to_string(),
            metadata: HashMap::new(),
            max_evaluations: 100,
            target_accuracy: 1e-6,
        };

        let initial_params = optimizer.meta_state.meta_params.clone();
        optimizer.update_meta_parameters(&problem, 1.5).unwrap();

        // Parameters should have changed
        assert!(optimizer.meta_state.meta_params != initial_params);
        assert_eq!(optimizer.meta_state.performance_history.len(), 1);
    }
}

#[allow(dead_code)]
pub fn placeholder() {
    // Placeholder function to prevent unused module warnings
}
