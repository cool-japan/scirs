//! Bayesian optimization for neural architecture search
//!
//! This module implements Bayesian optimization using Gaussian processes
//! and other surrogate models for efficient architecture search.

use std::collections::HashMap;
use num_traits::Float;
use ndarray::{Array1, Array2};

use super::super::architecture::{ArchitectureSpec, ArchitectureCandidate};

/// Bayesian optimization state
#[derive(Debug)]
pub struct BayesianOptimizationState<T: Float> {
    /// Gaussian process surrogate model
    pub surrogate_model: SurrogateModel<T>,

    /// Acquisition function
    pub acquisition_function: AcquisitionFunction,

    /// Observed data points
    pub observations: Vec<(ArchitectureSpec, f64)>,

    /// Hyperparameters
    pub hyperparameters: BayesianHyperparameters,

    /// Acquisition function parameters
    pub acquisition_params: AcquisitionParameters,

    /// Model training history
    pub training_history: Vec<ModelTrainingRecord>,
}

/// Surrogate model for Bayesian optimization
#[derive(Debug)]
pub struct SurrogateModel<T: Float> {
    /// Model type
    pub model_type: SurrogateModelType,

    /// Model parameters
    pub parameters: HashMap<String, T>,

    /// Training data
    pub training_data: Vec<(Vec<T>, T)>,

    /// Model uncertainty estimates
    pub uncertainty_estimates: Vec<T>,

    /// Kernel function
    pub kernel: KernelFunction<T>,

    /// Model is trained flag
    pub is_trained: bool,
}

/// Types of surrogate models
#[derive(Debug, Clone, Copy)]
pub enum SurrogateModelType {
    GaussianProcess,
    RandomForest,
    NeuralNetwork,
    BayesianNeuralNetwork,
    TreeParzenEstimator,
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug, Clone, Copy)]
pub enum AcquisitionFunction {
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    EntropySearch,
    KnowledgeGradient,
    ThompsonSampling,
}

/// Bayesian optimization hyperparameters
#[derive(Debug, Clone)]
pub struct BayesianHyperparameters {
    /// Length scale
    pub lengthscale: f64,

    /// Noise variance
    pub noise_variance: f64,

    /// Signal variance
    pub signal_variance: f64,

    /// Kernel parameters
    pub kernel_parameters: HashMap<String, f64>,

    /// Learning rate for hyperparameter optimization
    pub learning_rate: f64,

    /// Number of optimization steps
    pub optimization_steps: usize,
}

impl Default for BayesianHyperparameters {
    fn default() -> Self {
        let mut kernel_parameters = HashMap::new();
        kernel_parameters.insert("gamma".to_string(), 1.0);
        kernel_parameters.insert("alpha".to_string(), 1e-6);

        Self {
            lengthscale: 1.0,
            noise_variance: 0.1,
            signal_variance: 1.0,
            kernel_parameters,
            learning_rate: 0.01,
            optimization_steps: 100,
        }
    }
}

/// Acquisition function parameters
#[derive(Debug, Clone)]
pub struct AcquisitionParameters {
    /// Exploration-exploitation trade-off
    pub kappa: f64,

    /// Current best observed value
    pub best_observed: f64,

    /// Number of fantasy points for Thompson sampling
    pub num_fantasy_points: usize,

    /// Epsilon for probability of improvement
    pub epsilon: f64,
}

impl Default for AcquisitionParameters {
    fn default() -> Self {
        Self {
            kappa: 2.576, // 99% confidence interval
            best_observed: f64::NEG_INFINITY,
            num_fantasy_points: 100,
            epsilon: 0.01,
        }
    }
}

/// Kernel function for Gaussian processes
#[derive(Debug)]
pub struct KernelFunction<T: Float> {
    /// Kernel type
    pub kernel_type: KernelType,

    /// Kernel parameters
    pub parameters: HashMap<String, T>,
}

/// Types of kernel functions
#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    RBF,        // Radial basis function
    Matern32,   // Matérn 3/2
    Matern52,   // Matérn 5/2
    Linear,     // Linear kernel
    Polynomial, // Polynomial kernel
    Composite,  // Composite kernel
}

/// Model training record
#[derive(Debug, Clone)]
pub struct ModelTrainingRecord {
    /// Training iteration
    pub iteration: usize,

    /// Log likelihood
    pub log_likelihood: f64,

    /// Hyperparameters at this iteration
    pub hyperparameters: BayesianHyperparameters,

    /// Training time
    pub training_time: std::time::Duration,

    /// Cross-validation score
    pub cv_score: Option<f64>,
}

impl<T: Float + Default + std::fmt::Debug + From<f64> + Into<f64>> BayesianOptimizationState<T> {
    /// Create new Bayesian optimization state
    pub fn new() -> Self {
        Self {
            surrogate_model: SurrogateModel::new(SurrogateModelType::GaussianProcess),
            acquisition_function: AcquisitionFunction::ExpectedImprovement,
            observations: Vec::new(),
            hyperparameters: BayesianHyperparameters::default(),
            acquisition_params: AcquisitionParameters::default(),
            training_history: Vec::new(),
        }
    }

    /// Add observation to the Bayesian optimization
    pub fn add_observation(&mut self, architecture: ArchitectureSpec, performance: f64) {
        self.observations.push((architecture, performance));
        
        // Update best observed value
        if performance > self.acquisition_params.best_observed {
            self.acquisition_params.best_observed = performance;
        }

        // Retrain the surrogate model
        if self.observations.len() >= 3 {
            let _ = self.retrain_surrogate_model();
        }
    }

    /// Generate next candidate using acquisition function
    pub fn generate_candidate(
        &self,
        search_space: &super::super::space::ArchitectureSearchSpace,
    ) -> Result<ArchitectureCandidate, super::SearchError> {
        if !self.surrogate_model.is_trained {
            // If model not trained, return random sample
            let arch = search_space.sample_random();
            return Ok(ArchitectureCandidate::new("bo_random".to_string(), arch));
        }

        // Optimize acquisition function to find next candidate
        let candidate_arch = self.optimize_acquisition_function(search_space)?;
        
        Ok(ArchitectureCandidate::new(
            format!("bo_iter_{}", self.observations.len()),
            candidate_arch,
        ))
    }

    /// Optimize acquisition function to find next candidate
    fn optimize_acquisition_function(
        &self,
        search_space: &super::super::space::ArchitectureSearchSpace,
    ) -> Result<ArchitectureSpec, super::SearchError> {
        let num_candidates = 1000; // Number of random candidates to evaluate
        let mut best_candidate = None;
        let mut best_acquisition_value = f64::NEG_INFINITY;

        for _ in 0..num_candidates {
            let candidate = search_space.sample_random();
            let acquisition_value = self.evaluate_acquisition_function(&candidate)?;

            if acquisition_value > best_acquisition_value {
                best_acquisition_value = acquisition_value;
                best_candidate = Some(candidate);
            }
        }

        best_candidate.ok_or_else(|| {
            super::SearchError::GenerationFailed("No candidate found".to_string())
        })
    }

    /// Evaluate acquisition function for a given architecture
    fn evaluate_acquisition_function(&self, architecture: &ArchitectureSpec) -> Result<f64, super::SearchError> {
        if !self.surrogate_model.is_trained {
            return Ok(0.0);
        }

        // Encode architecture to feature vector
        let features = self.encode_architecture(architecture);

        // Get prediction from surrogate model
        let (mean, std) = self.surrogate_model.predict(&features)?;

        // Calculate acquisition function value
        let acquisition_value = match self.acquisition_function {
            AcquisitionFunction::ExpectedImprovement => {
                self.expected_improvement(mean, std)
            }
            AcquisitionFunction::ProbabilityOfImprovement => {
                self.probability_of_improvement(mean, std)
            }
            AcquisitionFunction::UpperConfidenceBound => {
                self.upper_confidence_bound(mean, std)
            }
            _ => {
                // Default to expected improvement
                self.expected_improvement(mean, std)
            }
        };

        Ok(acquisition_value)
    }

    /// Expected Improvement acquisition function
    fn expected_improvement(&self, mean: f64, std: f64) -> f64 {
        if std <= 0.0 {
            return 0.0;
        }

        let improvement = mean - self.acquisition_params.best_observed - self.acquisition_params.epsilon;
        let z = improvement / std;
        
        let phi = normal_pdf(z);
        let capital_phi = normal_cdf(z);
        
        improvement * capital_phi + std * phi
    }

    /// Probability of Improvement acquisition function
    fn probability_of_improvement(&self, mean: f64, std: f64) -> f64 {
        if std <= 0.0 {
            return if mean > self.acquisition_params.best_observed { 1.0 } else { 0.0 };
        }

        let improvement = mean - self.acquisition_params.best_observed - self.acquisition_params.epsilon;
        let z = improvement / std;
        
        normal_cdf(z)
    }

    /// Upper Confidence Bound acquisition function
    fn upper_confidence_bound(&self, mean: f64, std: f64) -> f64 {
        mean + self.acquisition_params.kappa * std
    }

    /// Retrain the surrogate model with current observations
    fn retrain_surrogate_model(&mut self) -> Result<(), super::SearchError> {
        let start_time = std::time::Instant::now();
        
        // Prepare training data
        let mut training_inputs = Vec::new();
        let mut training_outputs = Vec::new();

        for (architecture, performance) in &self.observations {
            let features = self.encode_architecture(architecture);
            training_inputs.push(features);
            training_outputs.push(*performance);
        }

        // Update surrogate model training data
        self.surrogate_model.training_data = training_inputs
            .into_iter()
            .zip(training_outputs.into_iter())
            .map(|(input, output)| (input.into_iter().map(T::from).collect(), T::from(output)))
            .collect();

        // Train the model (simplified)
        self.surrogate_model.is_trained = true;

        // Record training
        let training_record = ModelTrainingRecord {
            iteration: self.training_history.len(),
            log_likelihood: 0.0, // Would calculate actual log likelihood
            hyperparameters: self.hyperparameters.clone(),
            training_time: start_time.elapsed(),
            cv_score: None,
        };
        self.training_history.push(training_record);

        Ok(())
    }

    /// Encode architecture to feature vector
    fn encode_architecture(&self, architecture: &ArchitectureSpec) -> Vec<f64> {
        let mut features = Vec::new();

        // Basic architecture features
        features.push(architecture.layers.len() as f64);
        features.push(architecture.parameter_count() as f64);
        features.push(architecture.memory_usage_estimate() as f64);

        // Layer type distribution
        let mut layer_type_counts = HashMap::new();
        for layer in &architecture.layers {
            *layer_type_counts.entry(layer.layer_type).or_insert(0) += 1;
        }

        // Add layer type features (normalized by total layers)
        let total_layers = architecture.layers.len() as f64;
        features.push(layer_type_counts.get(&super::super::architecture::LayerType::Linear).copied().unwrap_or(0) as f64 / total_layers);
        features.push(layer_type_counts.get(&super::super::architecture::LayerType::LSTM).copied().unwrap_or(0) as f64 / total_layers);
        features.push(layer_type_counts.get(&super::super::architecture::LayerType::Attention).copied().unwrap_or(0) as f64 / total_layers);

        // Average layer dimensions
        if !architecture.layers.is_empty() {
            let avg_input_dim = architecture.layers.iter().map(|l| l.dimensions.input_dim).sum::<usize>() as f64 / total_layers;
            let avg_output_dim = architecture.layers.iter().map(|l| l.dimensions.output_dim).sum::<usize>() as f64 / total_layers;
            features.push(avg_input_dim);
            features.push(avg_output_dim);
        } else {
            features.push(0.0);
            features.push(0.0);
        }

        features
    }
}

impl<T: Float + Default + std::fmt::Debug> SurrogateModel<T> {
    /// Create new surrogate model
    pub fn new(model_type: SurrogateModelType) -> Self {
        Self {
            model_type,
            parameters: HashMap::new(),
            training_data: Vec::new(),
            uncertainty_estimates: Vec::new(),
            kernel: KernelFunction {
                kernel_type: KernelType::RBF,
                parameters: HashMap::new(),
            },
            is_trained: false,
        }
    }

    /// Predict mean and standard deviation for input
    pub fn predict(&self, input: &[f64]) -> Result<(f64, f64), super::SearchError> {
        if !self.is_trained {
            return Err(super::SearchError::GenerationFailed("Model not trained".to_string()));
        }

        // Simplified prediction - in practice would use actual GP inference
        let mean = 0.5; // Placeholder
        let std = 0.1;  // Placeholder

        Ok((mean, std))
    }

    /// Calculate kernel value between two inputs
    pub fn kernel_value(&self, x1: &[T], x2: &[T]) -> T {
        match self.kernel.kernel_type {
            KernelType::RBF => {
                let squared_distance = x1.iter()
                    .zip(x2.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .fold(T::zero(), |acc, x| acc + x);
                
                // exp(-γ * ||x1 - x2||²)
                let gamma = num_traits::cast::cast(1.0).unwrap_or_else(|| T::zero());
                (-gamma * squared_distance).exp()
            }
            _ => T::one(), // Placeholder for other kernels
        }
    }
}

/// Normal probability density function
fn normal_pdf(x: f64) -> f64 {
    (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * x * x).exp()
}

/// Normal cumulative distribution function (approximation)
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Multi-objective Bayesian optimization state
pub struct MultiObjectiveBayesianState<T: Float> {
    /// Surrogate models for each objective
    pub surrogate_models: Vec<SurrogateModel<T>>,

    /// Pareto front
    pub pareto_front: Vec<ArchitectureCandidate>,

    /// Acquisition function for multi-objective
    pub mo_acquisition: MultiObjectiveAcquisition,

    /// Objective names
    pub objectives: Vec<String>,
}

/// Multi-objective acquisition functions
#[derive(Debug, Clone, Copy)]
pub enum MultiObjectiveAcquisition {
    ExpectedHypervolume,
    ParEGO,
    NSGA2Expected,
    WeightedSum,
}

impl<T: Float + Default + std::fmt::Debug> MultiObjectiveBayesianState<T> {
    pub fn new(objectives: Vec<String>) -> Self {
        let surrogate_models = objectives
            .iter()
            .map(|_| SurrogateModel::new(SurrogateModelType::GaussianProcess))
            .collect();

        Self {
            surrogate_models,
            pareto_front: Vec::new(),
            mo_acquisition: MultiObjectiveAcquisition::ExpectedHypervolume,
            objectives,
        }
    }

    /// Update Pareto front with new candidate
    pub fn update_pareto_front(&mut self, candidate: ArchitectureCandidate) {
        // Add candidate to front
        self.pareto_front.push(candidate);

        // Remove dominated solutions
        let mut non_dominated = Vec::new();
        
        for i in 0..self.pareto_front.len() {
            let mut is_dominated = false;
            
            for j in 0..self.pareto_front.len() {
                if i != j && self.dominates(&self.pareto_front[j], &self.pareto_front[i]) {
                    is_dominated = true;
                    break;
                }
            }
            
            if !is_dominated {
                non_dominated.push(self.pareto_front[i].clone());
            }
        }
        
        self.pareto_front = non_dominated;
    }

    /// Check if candidate a dominates candidate b
    fn dominates(&self, a: &ArchitectureCandidate, b: &ArchitectureCandidate) -> bool {
        let perf_a = &a.performance;
        let perf_b = &b.performance;

        let better_in_any = perf_a.optimization_performance > perf_b.optimization_performance
            || perf_a.convergence_speed > perf_b.convergence_speed;

        let worse_in_any = perf_a.optimization_performance < perf_b.optimization_performance
            || perf_a.convergence_speed < perf_b.convergence_speed;

        better_in_any && !worse_in_any
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_optimization_state_creation() {
        let state: BayesianOptimizationState<f64> = BayesianOptimizationState::new();
        assert_eq!(state.observations.len(), 0);
        assert!(!state.surrogate_model.is_trained);
    }

    #[test]
    fn test_surrogate_model_creation() {
        let model: SurrogateModel<f64> = SurrogateModel::new(SurrogateModelType::GaussianProcess);
        assert!(matches!(model.model_type, SurrogateModelType::GaussianProcess));
        assert!(!model.is_trained);
    }

    #[test]
    fn test_normal_pdf() {
        let result = normal_pdf(0.0);
        assert!((result - 0.3989422804).abs() < 1e-6);
    }

    #[test]
    fn test_normal_cdf() {
        let result = normal_cdf(0.0);
        assert!((result - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_acquisition_functions() {
        let state: BayesianOptimizationState<f64> = BayesianOptimizationState::new();
        
        // Test expected improvement
        let ei = state.expected_improvement(1.0, 0.5);
        assert!(ei > 0.0);
        
        // Test probability of improvement
        let pi = state.probability_of_improvement(1.0, 0.5);
        assert!(pi >= 0.0 && pi <= 1.0);
        
        // Test upper confidence bound
        let ucb = state.upper_confidence_bound(1.0, 0.5);
        assert!(ucb > 1.0);
    }

    #[test]
    fn test_multi_objective_state() {
        let objectives = vec!["performance".to_string(), "efficiency".to_string()];
        let state: MultiObjectiveBayesianState<f64> = MultiObjectiveBayesianState::new(objectives);
        assert_eq!(state.objectives.len(), 2);
        assert_eq!(state.surrogate_models.len(), 2);
    }
}