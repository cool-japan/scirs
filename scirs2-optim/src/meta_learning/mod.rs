//! Meta-learning optimization support
//!
//! This module provides meta-learning capabilities for optimization, including learnable optimizers,
//! hyperparameter optimization, and neural optimization approaches.

use crate::error::{OptimError, Result};
use ndarray::{Array, Array1, Array2, Dimension, ScalarOperand, Zip};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Hyperparameter optimization strategy
#[derive(Debug, Clone)]
pub enum HyperparameterStrategy {
    /// Grid search over predefined values
    GridSearch {
        /// Parameter grids
        grids: HashMap<String, Vec<f64>>,
    },
    /// Random search
    RandomSearch {
        /// Number of trials
        num_trials: usize,
        /// Parameter bounds
        bounds: HashMap<String, (f64, f64)>,
    },
    /// Bayesian optimization
    BayesianOptimization {
        /// Number of trials
        num_trials: usize,
        /// Parameter bounds
        bounds: HashMap<String, (f64, f64)>,
        /// Acquisition function
        acquisition: AcquisitionFunction,
    },
    /// Population-based training
    PopulationBased {
        /// Population size
        population_size: usize,
        /// Perturbation factor
        perturbation_factor: f64,
    },
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug, Clone, Copy)]
pub enum AcquisitionFunction {
    /// Expected improvement
    ExpectedImprovement,
    /// Upper confidence bound
    UpperConfidenceBound {
        /// Beta parameter for upper confidence bound
        beta: f64,
    },
    /// Probability of improvement
    ProbabilityOfImprovement,
}

/// Meta-learning optimizer that learns to optimize
#[derive(Debug)]
pub struct MetaOptimizer<A: Float, D: Dimension> {
    /// Base optimizer type
    #[allow(dead_code)]
    base_optimizer: String,
    /// Learnable parameters for the optimizer
    meta_parameters: Array1<A>,
    /// Meta-learning rate
    meta_learning_rate: A,
    /// History of optimization trajectories
    trajectories: Vec<OptimizationTrajectory<A, D>>,
    /// Neural network for predicting optimal hyperparameters
    predictor_network: Option<HyperparameterPredictor<A>>,
    /// Performance history
    performance_history: Vec<A>,
    /// Current step count
    step_count: usize,
}

/// Optimization trajectory for meta-learning
#[derive(Debug, Clone)]
pub struct OptimizationTrajectory<A: Float, D: Dimension> {
    /// Sequence of parameter states
    #[allow(dead_code)]
    parameter_history: Vec<Array<A, D>>,
    /// Sequence of gradient states
    #[allow(dead_code)]
    gradient_history: Vec<Array<A, D>>,
    /// Sequence of loss values
    loss_history: Vec<A>,
    /// Final performance metric
    final_performance: A,
    /// Optimizer hyperparameters used
    hyperparameters: HashMap<String, A>,
}

/// Neural network for predicting optimal hyperparameters
#[derive(Debug)]
pub struct HyperparameterPredictor<A: Float> {
    /// Input layer weights (problem features -> hidden)
    input_weights: Array2<A>,
    /// Hidden layer weights (hidden -> output)
    output_weights: Array2<A>,
    /// Input layer biases
    input_bias: Array1<A>,
    /// Output layer biases
    output_bias: Array1<A>,
    /// Hidden layer size
    hidden_size: usize,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> MetaOptimizer<A, D> {
    /// Create a new meta-optimizer
    pub fn new(
        base_optimizer: String,
        meta_learning_rate: A,
        initial_meta_params: Array1<A>,
    ) -> Self {
        Self {
            base_optimizer,
            meta_parameters: initial_meta_params,
            meta_learning_rate,
            trajectories: Vec::new(),
            predictor_network: None,
            performance_history: Vec::new(),
            step_count: 0,
        }
    }

    /// Initialize the hyperparameter predictor network
    pub fn init_predictor_network(
        &mut self,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) {
        let predictor = HyperparameterPredictor::new(input_size, hidden_size, output_size);
        self.predictor_network = Some(predictor);
    }

    /// Record an optimization trajectory
    pub fn record_trajectory(&mut self, trajectory: OptimizationTrajectory<A, D>) {
        self.performance_history.push(trajectory.final_performance);
        self.trajectories.push(trajectory);

        // Keep only recent trajectories to manage memory
        if self.trajectories.len() > 1000 {
            self.trajectories.remove(0);
            self.performance_history.remove(0);
        }
    }

    /// Predict optimal hyperparameters for a given problem
    pub fn predict_hyperparameters(
        &self,
        problem_features: &Array1<A>,
    ) -> Result<HashMap<String, A>> {
        if let Some(predictor) = &self.predictor_network {
            predictor.predict(problem_features)
        } else {
            // Fall back to meta-parameter based prediction
            self.meta_parameter_prediction(problem_features)
        }
    }

    /// Meta-parameter based hyperparameter prediction
    fn meta_parameter_prediction(
        &self,
        _problem_features: &Array1<A>,
    ) -> Result<HashMap<String, A>> {
        let mut hyperparams = HashMap::new();

        // Use meta-parameters to generate hyperparameters
        if self.meta_parameters.len() >= 3 {
            hyperparams.insert("learning_rate".to_string(), self.meta_parameters[0]);
            hyperparams.insert("weight_decay".to_string(), self.meta_parameters[1]);
            hyperparams.insert("momentum".to_string(), self.meta_parameters[2]);
        }

        Ok(hyperparams)
    }

    /// Update meta-parameters based on recent performance
    pub fn update_meta_parameters(&mut self) -> Result<()> {
        if self.performance_history.len() < 2 {
            return Ok(()); // Need at least 2 data points
        }

        // Simple gradient-based update using finite differences
        let recent_perf = self.performance_history[self.performance_history.len() - 1];
        let prev_perf = self.performance_history[self.performance_history.len() - 2];

        let performance_gradient = recent_perf - prev_perf;

        // Update meta-parameters in direction of improvement
        let update_factor = self.meta_learning_rate * performance_gradient;
        for param in self.meta_parameters.iter_mut() {
            *param = *param + update_factor;
        }

        self.step_count += 1;
        Ok(())
    }

    /// Get current meta-parameters
    pub fn get_meta_parameters(&self) -> &Array1<A> {
        &self.meta_parameters
    }

    /// Get performance history
    pub fn get_performance_history(&self) -> &[A] {
        &self.performance_history
    }

    /// Get number of recorded trajectories
    pub fn trajectory_count(&self) -> usize {
        self.trajectories.len()
    }

    /// Train the predictor network on recorded trajectories
    pub fn train_predictor_network(&mut self, epochs: usize) -> Result<()> {
        if let Some(predictor) = &mut self.predictor_network {
            predictor.train(&self.trajectories, epochs)?;
        }
        Ok(())
    }
}

impl<A: Float + ScalarOperand + Debug> HyperparameterPredictor<A> {
    /// Create a new hyperparameter predictor
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        // Initialize with small random weights
        let input_weights = Array2::from_shape_fn((hidden_size, input_size), |_| {
            A::from(0.01).unwrap()
                * (A::from(rand::random::<f64>()).unwrap() - A::from(0.5).unwrap())
        });

        let output_weights = Array2::from_shape_fn((output_size, hidden_size), |_| {
            A::from(0.01).unwrap()
                * (A::from(rand::random::<f64>()).unwrap() - A::from(0.5).unwrap())
        });

        let input_bias = Array1::zeros(hidden_size);
        let output_bias = Array1::zeros(output_size);

        Self {
            input_weights,
            output_weights,
            input_bias,
            output_bias,
            hidden_size,
        }
    }

    /// Predict hyperparameters given problem features
    pub fn predict(&self, features: &Array1<A>) -> Result<HashMap<String, A>> {
        // Forward pass through the network
        let hidden = self.forward_hidden(features)?;
        let output = self.forward_output(&hidden)?;

        // Convert output to hyperparameters
        let mut hyperparams = HashMap::new();
        if output.len() >= 3 {
            hyperparams.insert("learning_rate".to_string(), A::exp(output[0])); // exp for positive LR
            hyperparams.insert("weight_decay".to_string(), A::max(A::zero(), output[1])); // ReLU for WD
            hyperparams.insert("momentum".to_string(), A::tanh(output[2])); // tanh for momentum in [-1,1]
        }

        Ok(hyperparams)
    }

    /// Forward pass through hidden layer
    fn forward_hidden(&self, input: &Array1<A>) -> Result<Array1<A>> {
        if input.len() != self.input_weights.ncols() {
            return Err(OptimError::DimensionMismatch(format!(
                "Expected input size {}, got {}",
                self.input_weights.ncols(),
                input.len()
            )));
        }

        let mut hidden = Array1::zeros(self.hidden_size);

        // Linear transformation: hidden = W * input + b
        for (i, h) in hidden.iter_mut().enumerate() {
            *h = self.input_bias[i];
            for (j, &inp) in input.iter().enumerate() {
                *h = *h + self.input_weights[(i, j)] * inp;
            }
            // Apply ReLU activation
            *h = A::max(A::zero(), *h);
        }

        Ok(hidden)
    }

    /// Forward pass through output layer
    fn forward_output(&self, hidden: &Array1<A>) -> Result<Array1<A>> {
        let output_size = self.output_weights.nrows();
        let mut output = Array1::zeros(output_size);

        // Linear transformation: output = W * hidden + b
        for (i, out) in output.iter_mut().enumerate() {
            *out = self.output_bias[i];
            for (j, &h) in hidden.iter().enumerate() {
                *out = *out + self.output_weights[(i, j)] * h;
            }
        }

        Ok(output)
    }

    /// Train the network on optimization trajectories
    pub fn train<D: Dimension>(
        &mut self,
        trajectories: &[OptimizationTrajectory<A, D>],
        epochs: usize,
    ) -> Result<()> {
        let learning_rate = A::from(0.001).unwrap();

        for epoch in 0..epochs {
            let mut total_loss = A::zero();
            let mut count = 0;

            for trajectory in trajectories {
                // Extract features from trajectory
                let features = self.extract_features(trajectory)?;

                // Target hyperparameters (the ones that were actually used)
                let targets = self.hyperparams_to_array(&trajectory.hyperparameters)?;

                // Forward pass
                let hidden = self.forward_hidden(&features)?;
                let predicted = self.forward_output(&hidden)?;

                // Compute loss (MSE)
                let loss = self.compute_loss(&predicted, &targets);
                total_loss = total_loss + loss;
                count += 1;

                // Backward pass and update weights
                self.backward_pass(&features, &hidden, &predicted, &targets, learning_rate)?;
            }

            if epoch % 100 == 0 {
                let avg_loss = total_loss / A::from(count).unwrap();
                println!("Epoch {}: Average loss = {:?}", epoch, avg_loss);
            }
        }

        Ok(())
    }

    /// Extract features from an optimization trajectory
    fn extract_features<D: Dimension>(
        &self,
        trajectory: &OptimizationTrajectory<A, D>,
    ) -> Result<Array1<A>> {
        // Extract simple features from the trajectory
        let mut features = Vec::new();

        // Final loss
        if let Some(&final_loss) = trajectory.loss_history.last() {
            features.push(final_loss);
        } else {
            features.push(A::zero());
        }

        // Loss improvement (first - last)
        if trajectory.loss_history.len() >= 2 {
            let improvement = trajectory.loss_history[0]
                - trajectory.loss_history[trajectory.loss_history.len() - 1];
            features.push(improvement);
        } else {
            features.push(A::zero());
        }

        // Trajectory length
        features.push(A::from(trajectory.loss_history.len()).unwrap());

        // Pad or truncate to match expected input size
        let expected_size = self.input_weights.ncols();
        features.resize(expected_size, A::zero());

        Ok(Array1::from_vec(features))
    }

    /// Convert hyperparameters HashMap to Array for training
    fn hyperparams_to_array(&self, hyperparams: &HashMap<String, A>) -> Result<Array1<A>> {
        // Extract in consistent order
        let targets = vec![
            hyperparams
                .get("learning_rate")
                .copied()
                .unwrap_or_else(|| A::from(0.001).unwrap())
                .ln(),
            hyperparams
                .get("weight_decay")
                .copied()
                .unwrap_or_else(|| A::zero()),
            hyperparams
                .get("momentum")
                .copied()
                .unwrap_or_else(|| A::zero())
                .atanh(),
        ];

        Ok(Array1::from_vec(targets))
    }

    /// Compute MSE loss
    fn compute_loss(&self, predicted: &Array1<A>, targets: &Array1<A>) -> A {
        let mut loss = A::zero();
        for (pred, &target) in predicted.iter().zip(targets.iter()) {
            let diff = *pred - target;
            loss = loss + diff * diff;
        }
        loss / A::from(predicted.len()).unwrap()
    }

    /// Backward pass and weight update
    fn backward_pass(
        &mut self,
        input: &Array1<A>,
        hidden: &Array1<A>,
        predicted: &Array1<A>,
        targets: &Array1<A>,
        learning_rate: A,
    ) -> Result<()> {
        // Compute output layer gradients
        let mut output_grad = Array1::zeros(predicted.len());
        for (i, (&pred, &target)) in predicted.iter().zip(targets.iter()).enumerate() {
            output_grad[i] =
                A::from(2.0).unwrap() * (pred - target) / A::from(predicted.len()).unwrap();
        }

        // Update output weights and biases
        for (i, &grad) in output_grad.iter().enumerate() {
            self.output_bias[i] = self.output_bias[i] - learning_rate * grad;
            for (j, &h) in hidden.iter().enumerate() {
                self.output_weights[(i, j)] =
                    self.output_weights[(i, j)] - learning_rate * grad * h;
            }
        }

        // Compute hidden layer gradients
        let mut hidden_grad = Array1::zeros(hidden.len());
        for (j, &h) in hidden.iter().enumerate() {
            let mut grad = A::zero();
            for (i, &output_g) in output_grad.iter().enumerate() {
                grad = grad + output_g * self.output_weights[(i, j)];
            }
            // Apply ReLU derivative
            hidden_grad[j] = if h > A::zero() { grad } else { A::zero() };
        }

        // Update input weights and biases
        for (i, &grad) in hidden_grad.iter().enumerate() {
            self.input_bias[i] = self.input_bias[i] - learning_rate * grad;
            for (j, &inp) in input.iter().enumerate() {
                self.input_weights[(i, j)] =
                    self.input_weights[(i, j)] - learning_rate * grad * inp;
            }
        }

        Ok(())
    }
}

/// Hyperparameter optimization manager
#[derive(Debug)]
pub struct HyperparameterOptimizer<A: Float> {
    /// Optimization strategy
    strategy: HyperparameterStrategy,
    /// Best hyperparameters found so far
    best_hyperparameters: Option<HashMap<String, A>>,
    /// Best performance achieved
    best_performance: Option<A>,
    /// Trial history
    trial_history: Vec<(HashMap<String, A>, A)>,
    /// Current trial count
    trial_count: usize,
}

impl<A: Float + ScalarOperand + Debug> HyperparameterOptimizer<A> {
    /// Create a new hyperparameter optimizer
    pub fn new(strategy: HyperparameterStrategy) -> Self {
        Self {
            strategy,
            best_hyperparameters: None,
            best_performance: None,
            trial_history: Vec::new(),
            trial_count: 0,
        }
    }

    /// Suggest next hyperparameters to try
    pub fn suggest_hyperparameters(&mut self) -> Result<HashMap<String, A>> {
        match &self.strategy {
            HyperparameterStrategy::GridSearch { grids } => self.suggest_grid_search(grids),
            HyperparameterStrategy::RandomSearch { bounds, .. } => {
                self.suggest_random_search(bounds)
            }
            HyperparameterStrategy::BayesianOptimization {
                bounds,
                acquisition,
                ..
            } => self.suggest_bayesian_optimization(bounds, *acquisition),
            HyperparameterStrategy::PopulationBased {
                population_size,
                perturbation_factor,
            } => self.suggest_population_based(*population_size, *perturbation_factor),
        }
    }

    /// Report performance for the last suggested hyperparameters
    pub fn report_performance(&mut self, hyperparameters: HashMap<String, A>, performance: A) {
        self.trial_history
            .push((hyperparameters.clone(), performance));
        self.trial_count += 1;

        // Update best if this is better
        if self.best_performance.is_none_or(|best| performance > best) {
            self.best_performance = Some(performance);
            self.best_hyperparameters = Some(hyperparameters);
        }
    }

    /// Get the best hyperparameters found so far
    pub fn get_best_hyperparameters(&self) -> Option<&HashMap<String, A>> {
        self.best_hyperparameters.as_ref()
    }

    /// Get the best performance achieved
    pub fn get_best_performance(&self) -> Option<A> {
        self.best_performance
    }

    /// Grid search suggestion
    fn suggest_grid_search(&self, grids: &HashMap<String, Vec<f64>>) -> Result<HashMap<String, A>> {
        // Simple grid traversal based on trial count
        let mut hyperparams = HashMap::new();
        let mut index = self.trial_count;

        for (param_name, grid) in grids {
            let grid_size = grid.len();
            if grid_size == 0 {
                continue;
            }
            let grid_index = index % grid_size;
            hyperparams.insert(param_name.clone(), A::from(grid[grid_index]).unwrap());
            index /= grid_size;
        }

        Ok(hyperparams)
    }

    /// Random search suggestion
    fn suggest_random_search(
        &self,
        bounds: &HashMap<String, (f64, f64)>,
    ) -> Result<HashMap<String, A>> {
        let mut hyperparams = HashMap::new();

        for (param_name, &(min_val, max_val)) in bounds {
            let random_val = min_val + rand::random::<f64>() * (max_val - min_val);
            hyperparams.insert(param_name.clone(), A::from(random_val).unwrap());
        }

        Ok(hyperparams)
    }

    /// Bayesian optimization suggestion (simplified)
    fn suggest_bayesian_optimization(
        &self,
        bounds: &HashMap<String, (f64, f64)>,
        _acquisition: AcquisitionFunction,
    ) -> Result<HashMap<String, A>> {
        // Simplified Bayesian optimization - in practice, this would use a Gaussian process
        if self.trial_history.is_empty() {
            // First trial - random sample
            self.suggest_random_search(bounds)
        } else {
            // Use best performing hyperparameters as basis with small perturbation
            let best_params = self.best_hyperparameters.as_ref().unwrap();
            let mut perturbed_params = HashMap::new();

            for (param_name, &(min_val, max_val)) in bounds {
                let current_val = best_params
                    .get(param_name)
                    .map(|v| v.to_f64().unwrap())
                    .unwrap_or((min_val + max_val) / 2.0);

                let perturbation = 0.1 * (max_val - min_val) * (rand::random::<f64>() - 0.5);
                let new_val = (current_val + perturbation).max(min_val).min(max_val);

                perturbed_params.insert(param_name.clone(), A::from(new_val).unwrap());
            }

            Ok(perturbed_params)
        }
    }

    /// Population-based training suggestion
    fn suggest_population_based(
        &self,
        population_size: usize,
        perturbation_factor: f64,
    ) -> Result<HashMap<String, A>> {
        if self.trial_history.len() < population_size {
            // Initialize population randomly
            let mut hyperparams = HashMap::new();
            hyperparams.insert(
                "learning_rate".to_string(),
                A::from(0.001 * (1.0 + rand::random::<f64>())).unwrap(),
            );
            hyperparams.insert(
                "weight_decay".to_string(),
                A::from(0.0001 * rand::random::<f64>()).unwrap(),
            );
            Ok(hyperparams)
        } else {
            // Select from top performers and perturb
            let mut sorted_trials = self.trial_history.clone();
            sorted_trials.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_performer = &sorted_trials[0].0;
            let mut perturbed = HashMap::new();

            for (param_name, &value) in top_performer {
                let perturbation =
                    A::from(perturbation_factor * (rand::random::<f64>() - 0.5)).unwrap();
                let new_value = value * (A::one() + perturbation);
                perturbed.insert(param_name.clone(), new_value);
            }

            Ok(perturbed)
        }
    }

    /// Get trial history
    pub fn get_trial_history(&self) -> &[(HashMap<String, A>, A)] {
        &self.trial_history
    }

    /// Check if optimization should continue
    pub fn should_continue(&self) -> bool {
        match &self.strategy {
            HyperparameterStrategy::GridSearch { grids } => {
                let total_combinations: usize = grids.values().map(|g| g.len()).product();
                self.trial_count < total_combinations
            }
            HyperparameterStrategy::RandomSearch { num_trials, .. } => {
                self.trial_count < *num_trials
            }
            HyperparameterStrategy::BayesianOptimization { num_trials, .. } => {
                self.trial_count < *num_trials
            }
            HyperparameterStrategy::PopulationBased { .. } => {
                true // PBT continues indefinitely
            }
        }
    }
}

/// Neural optimizer that learns update rules
pub struct NeuralOptimizer<A: Float, D: Dimension> {
    /// Neural network for computing updates
    update_network: UpdateNetwork<A>,
    /// Optimizer for the neural network itself
    meta_optimizer: Box<dyn MetaOptimizerTrait<A>>,
    /// History of parameter updates
    update_history: Vec<Array<A, D>>,
    /// History of gradients
    gradient_history: Vec<Array<A, D>>,
    /// Current step
    step_count: usize,
}

/// Trait for meta-optimizers
pub trait MetaOptimizerTrait<A: Float> {
    /// Update meta-parameters
    fn meta_step(&mut self, meta_gradients: &Array1<A>) -> Result<()>;

    /// Get current meta-parameters
    fn get_meta_parameters(&self) -> &Array1<A>;
}

/// Simple SGD meta-optimizer
#[derive(Debug)]
pub struct SGDMetaOptimizer<A: Float> {
    /// Meta-parameters
    meta_params: Array1<A>,
    /// Meta learning rate
    meta_lr: A,
}

impl<A: Float> SGDMetaOptimizer<A> {
    /// Create a new SGD meta-optimizer
    pub fn new(meta_params: Array1<A>, meta_lr: A) -> Self {
        Self {
            meta_params,
            meta_lr,
        }
    }
}

impl<A: Float + ScalarOperand> MetaOptimizerTrait<A> for SGDMetaOptimizer<A> {
    fn meta_step(&mut self, meta_gradients: &Array1<A>) -> Result<()> {
        if meta_gradients.len() != self.meta_params.len() {
            return Err(OptimError::DimensionMismatch(
                "Meta-gradient dimension mismatch".to_string(),
            ));
        }

        Zip::from(&mut self.meta_params)
            .and(meta_gradients)
            .for_each(|param, &grad| {
                *param = *param - self.meta_lr * grad;
            });

        Ok(())
    }

    fn get_meta_parameters(&self) -> &Array1<A> {
        &self.meta_params
    }
}

/// Neural network for computing parameter updates
#[derive(Debug)]
pub struct UpdateNetwork<A: Float> {
    /// Network weights
    weights: Array2<A>,
    /// Network biases
    biases: Array1<A>,
    /// Input size (gradient features)
    input_size: usize,
    /// Output size (update features)
    #[allow(dead_code)]
    output_size: usize,
}

impl<A: Float + ScalarOperand + Debug> UpdateNetwork<A> {
    /// Create a new update network
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            A::from(0.01).unwrap()
                * (A::from(rand::random::<f64>()).unwrap() - A::from(0.5).unwrap())
        });
        let biases = Array1::zeros(output_size);

        Self {
            weights,
            biases,
            input_size,
            output_size,
        }
    }

    /// Compute parameter update given gradient features
    pub fn compute_update(&self, gradient_features: &Array1<A>) -> Result<Array1<A>> {
        if gradient_features.len() != self.input_size {
            return Err(OptimError::DimensionMismatch(format!(
                "Expected input size {}, got {}",
                self.input_size,
                gradient_features.len()
            )));
        }

        let mut update = self.biases.clone();

        // Linear transformation
        for (i, u) in update.iter_mut().enumerate() {
            for (j, &feat) in gradient_features.iter().enumerate() {
                *u = *u + self.weights[(i, j)] * feat;
            }
            // Apply tanh activation for bounded updates
            *u = A::tanh(*u);
        }

        Ok(update)
    }

    /// Get network parameters for meta-optimization
    pub fn get_parameters(&self) -> Array1<A> {
        let mut params = Vec::new();

        // Flatten weights
        for weight in self.weights.iter() {
            params.push(*weight);
        }

        // Add biases
        for bias in self.biases.iter() {
            params.push(*bias);
        }

        Array1::from_vec(params)
    }

    /// Set network parameters from meta-optimization
    pub fn set_parameters(&mut self, params: &Array1<A>) -> Result<()> {
        let expected_size = self.weights.len() + self.biases.len();
        if params.len() != expected_size {
            return Err(OptimError::DimensionMismatch(format!(
                "Expected {} parameters, got {}",
                expected_size,
                params.len()
            )));
        }

        let mut idx = 0;

        // Set weights
        for weight in self.weights.iter_mut() {
            *weight = params[idx];
            idx += 1;
        }

        // Set biases
        for bias in self.biases.iter_mut() {
            *bias = params[idx];
            idx += 1;
        }

        Ok(())
    }
}

impl<A: Float + ScalarOperand + Debug + 'static, D: Dimension> NeuralOptimizer<A, D> {
    /// Create a new neural optimizer
    pub fn new(
        input_size: usize,
        output_size: usize,
        meta_optimizer: Box<dyn MetaOptimizerTrait<A>>,
    ) -> Self {
        Self {
            update_network: UpdateNetwork::new(input_size, output_size),
            meta_optimizer,
            update_history: Vec::new(),
            gradient_history: Vec::new(),
            step_count: 0,
        }
    }

    /// Compute parameter update using the neural network
    pub fn compute_update(&mut self, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // Extract features from gradients
        let gradient_features = self.extract_gradient_features(gradients)?;

        // Compute update using the neural network
        let update_features = self.update_network.compute_update(&gradient_features)?;

        // Convert update features back to parameter space
        let update = self.features_to_update(gradients, &update_features)?;

        // Store history
        self.gradient_history.push(gradients.clone());
        self.update_history.push(update.clone());

        // Keep limited history
        if self.gradient_history.len() > 100 {
            self.gradient_history.remove(0);
            self.update_history.remove(0);
        }

        self.step_count += 1;
        Ok(update)
    }

    /// Extract features from gradients
    fn extract_gradient_features(&self, gradients: &Array<A, D>) -> Result<Array1<A>> {
        let mut features = Vec::new();

        // Gradient magnitude
        let magnitude = gradients.mapv(|x| x * x).sum().sqrt();
        features.push(magnitude);

        // Gradient mean
        let mean = gradients.sum() / A::from(gradients.len()).unwrap();
        features.push(mean);

        // Gradient standard deviation (approximated)
        let variance =
            gradients.mapv(|x| (x - mean) * (x - mean)).sum() / A::from(gradients.len()).unwrap();
        let std_dev = variance.sqrt();
        features.push(std_dev);

        // Step count (normalized)
        features.push(A::from(self.step_count as f64 / 1000.0).unwrap());

        Ok(Array1::from_vec(features))
    }

    /// Convert update features back to parameter space
    fn features_to_update(
        &self,
        gradients: &Array<A, D>,
        update_features: &Array1<A>,
    ) -> Result<Array<A, D>> {
        // Simple approach: scale gradients by the update features
        let mut update = gradients.clone();

        if !update_features.is_empty() {
            let scale_factor = update_features[0];
            update.mapv_inplace(|x| x * scale_factor);
        }

        Ok(update)
    }

    /// Meta-learning step to update the neural network
    pub fn meta_step(&mut self, meta_loss: A) -> Result<()> {
        // Compute meta-gradients using finite differences
        let current_params = self.update_network.get_parameters();
        let epsilon = A::from(1e-6).unwrap();
        let mut meta_gradients = Array1::zeros(current_params.len());

        // This is a simplified meta-gradient computation
        // In practice, you'd use proper automatic differentiation
        for i in 0..current_params.len() {
            let mut perturbed_params = current_params.clone();
            perturbed_params[i] = perturbed_params[i] + epsilon;

            // The meta-gradient would be computed based on how this perturbation
            // affects the final performance. For simplicity, we'll use the meta_loss
            meta_gradients[i] = meta_loss / epsilon;
        }

        // Update the neural network parameters
        self.meta_optimizer.meta_step(&meta_gradients)?;

        // Set the updated parameters
        let updated_params = self.meta_optimizer.get_meta_parameters();
        if updated_params.len() == current_params.len() {
            self.update_network.set_parameters(updated_params)?;
        }

        Ok(())
    }

    /// Get step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get update history
    pub fn get_update_history(&self) -> &[Array<A, D>] {
        &self.update_history
    }

    /// Get gradient history
    pub fn get_gradient_history(&self) -> &[Array<A, D>] {
        &self.gradient_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_meta_optimizer_creation() {
        let meta_params = Array1::from_vec(vec![0.01, 0.001, 0.9]);
        let meta_optimizer =
            MetaOptimizer::<f64, ndarray::Ix1>::new("adam".to_string(), 0.001, meta_params);

        assert_eq!(meta_optimizer.trajectory_count(), 0);
        assert_eq!(meta_optimizer.get_meta_parameters().len(), 3);
    }

    #[test]
    fn test_hyperparameter_predictor() {
        let predictor = HyperparameterPredictor::<f64>::new(3, 5, 3);
        let features = Array1::from_vec(vec![1.0, 0.5, 10.0]);

        let hyperparams = predictor.predict(&features).unwrap();

        assert!(hyperparams.contains_key("learning_rate"));
        assert!(hyperparams.contains_key("weight_decay"));
        assert!(hyperparams.contains_key("momentum"));
    }

    #[test]
    fn test_hyperparameter_optimizer_random_search() {
        let mut bounds = HashMap::new();
        bounds.insert("learning_rate".to_string(), (0.0001, 0.1));
        bounds.insert("weight_decay".to_string(), (0.0, 0.01));

        let strategy = HyperparameterStrategy::RandomSearch {
            num_trials: 10,
            bounds,
        };

        let mut optimizer = HyperparameterOptimizer::new(strategy);

        let hyperparams = optimizer.suggest_hyperparameters().unwrap();
        assert!(hyperparams.contains_key("learning_rate"));
        assert!(hyperparams.contains_key("weight_decay"));

        // Test bounds
        let lr: f64 = hyperparams["learning_rate"];
        let wd: f64 = hyperparams["weight_decay"];
        assert!(lr >= 0.0001 && lr <= 0.1);
        assert!(wd >= 0.0 && wd <= 0.01);
    }

    #[test]
    fn test_hyperparameter_optimizer_performance_tracking() {
        let mut bounds = HashMap::new();
        bounds.insert("learning_rate".to_string(), (0.001, 0.1));

        let strategy = HyperparameterStrategy::RandomSearch {
            num_trials: 5,
            bounds,
        };

        let mut optimizer = HyperparameterOptimizer::new(strategy);

        // Report some performance results
        let mut hyperparams1 = HashMap::new();
        hyperparams1.insert("learning_rate".to_string(), 0.01);
        optimizer.report_performance(hyperparams1, 0.8);

        let mut hyperparams2 = HashMap::new();
        hyperparams2.insert("learning_rate".to_string(), 0.05);
        optimizer.report_performance(hyperparams2, 0.9);

        // Best should be the second one
        assert_relative_eq!(
            optimizer.get_best_performance().unwrap(),
            0.9,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            optimizer.get_best_hyperparameters().unwrap()["learning_rate"],
            0.05,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_sgd_meta_optimizer() {
        let meta_params = Array1::from_vec(vec![0.01, 0.001]);
        let mut meta_opt = SGDMetaOptimizer::new(meta_params, 0.1);

        let meta_gradients = Array1::from_vec(vec![0.1, -0.05]);
        meta_opt.meta_step(&meta_gradients).unwrap();

        // Check parameter updates
        assert_relative_eq!(meta_opt.get_meta_parameters()[0], 0.0, epsilon = 1e-6); // 0.01 - 0.1 * 0.1
        assert_relative_eq!(meta_opt.get_meta_parameters()[1], 0.006, epsilon = 1e-6);
        // 0.001 - 0.1 * (-0.05)
    }

    #[test]
    fn test_update_network() {
        let network = UpdateNetwork::<f64>::new(3, 2);
        let gradient_features = Array1::from_vec(vec![1.0, -0.5, 0.2]);

        let update = network.compute_update(&gradient_features).unwrap();
        assert_eq!(update.len(), 2);

        // Check that outputs are bounded (due to tanh activation)
        for &val in update.iter() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_neural_optimizer() {
        let meta_params = Array1::from_vec(vec![0.1; 10]); // 10 parameters for small network
        let meta_opt = Box::new(SGDMetaOptimizer::new(meta_params, 0.01));

        let mut neural_opt = NeuralOptimizer::<f64, ndarray::Ix1>::new(4, 1, meta_opt);

        let gradients = Array1::from_vec(vec![0.1, -0.2, 0.05]);
        let update = neural_opt.compute_update(&gradients).unwrap();

        assert_eq!(update.len(), gradients.len());
        assert_eq!(neural_opt.step_count(), 1);
    }

    #[test]
    fn test_trajectory_recording() {
        let meta_params = Array1::from_vec(vec![0.01, 0.001, 0.9]);
        let mut meta_optimizer =
            MetaOptimizer::<f64, ndarray::Ix1>::new("adam".to_string(), 0.001, meta_params);

        let mut hyperparams = HashMap::new();
        hyperparams.insert("learning_rate".to_string(), 0.01);

        let trajectory = OptimizationTrajectory {
            parameter_history: vec![Array1::from_vec(vec![1.0, 2.0])],
            gradient_history: vec![Array1::from_vec(vec![0.1, -0.1])],
            loss_history: vec![1.0, 0.5],
            final_performance: 0.85,
            hyperparameters: hyperparams,
        };

        meta_optimizer.record_trajectory(trajectory);
        assert_eq!(meta_optimizer.trajectory_count(), 1);
        assert_eq!(meta_optimizer.get_performance_history().len(), 1);
    }
}
