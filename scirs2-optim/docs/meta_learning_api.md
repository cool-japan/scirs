# Meta-Learning API Reference

The `meta_learning` module provides comprehensive meta-learning capabilities for optimization, including learnable optimizers, hyperparameter optimization, and neural optimization approaches.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Hyperparameter Optimization](#hyperparameter-optimization)
3. [Neural Optimizers](#neural-optimizers)
4. [Meta-Learning Framework](#meta-learning-framework)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)

## Core Concepts

Meta-learning in optimization involves:
- **Learning to optimize**: Using machine learning to improve optimization algorithms
- **Hyperparameter optimization**: Automatically tuning optimizer hyperparameters
- **Neural optimizers**: Networks that learn parameter update rules
- **Adaptation**: Quickly adapting to new optimization problems

## Hyperparameter Optimization

### HyperparameterStrategy

Different strategies for optimizing hyperparameters.

```rust
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
```

### AcquisitionFunction

Acquisition functions for Bayesian optimization.

```rust
pub enum AcquisitionFunction {
    /// Expected improvement
    ExpectedImprovement,
    /// Upper confidence bound
    UpperConfidenceBound { 
        /// Beta parameter for upper confidence bound
        beta: f64 
    },
    /// Probability of improvement
    ProbabilityOfImprovement,
}
```

### HyperparameterOptimizer

Main component for hyperparameter optimization.

```rust
pub struct HyperparameterOptimizer<A: Float> {
    // Internal fields...
}

impl<A: Float + ScalarOperand + Debug> HyperparameterOptimizer<A> {
    /// Create a new hyperparameter optimizer
    pub fn new(strategy: HyperparameterStrategy) -> Self;

    /// Suggest next hyperparameters to try
    pub fn suggest(&mut self) -> Result<HashMap<String, A>>;

    /// Update with trial results
    pub fn update(&mut self, hyperparameters: HashMap<String, A>, performance: A);

    /// Get best hyperparameters found so far
    pub fn get_best_hyperparameters(&self) -> Option<&HashMap<String, A>>;

    /// Get best performance achieved
    pub fn get_best_performance(&self) -> Option<A>;

    /// Get number of trials completed
    pub fn num_trials(&self) -> usize;

    /// Check if optimization is complete
    pub fn is_complete(&self) -> bool;
}
```

## Neural Optimizers

### HyperparameterPredictor

Neural network for predicting optimal hyperparameters.

```rust
pub struct HyperparameterPredictor<A: Float> {
    /// Input layer weights (problem features -> hidden)
    input_weights: Array2<A>,
    /// Hidden layer weights (hidden -> output)
    output_weights: Array2<A>,
    /// Input layer biases
    input_bias: Array1<A>,
    /// Output layer biases
    output_bias: Array1<A>,
}

impl<A: Float + ScalarOperand + Debug> HyperparameterPredictor<A> {
    /// Create a new hyperparameter predictor
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self;

    /// Predict hyperparameters for given problem features
    pub fn predict(&self, features: &Array1<A>) -> Result<HashMap<String, A>>;

    /// Train the predictor on historical data
    pub fn train(
        &mut self,
        features: &[Array1<A>],
        hyperparameters: &[HashMap<String, A>],
        learning_rate: A,
        epochs: usize,
    ) -> Result<()>;

    /// Forward pass through the network
    pub fn forward(&self, input: &Array1<A>) -> Result<Array1<A>>;
}
```

### UpdateNetwork

Neural network for computing parameter updates.

```rust
pub struct UpdateNetwork<A: Float> {
    /// Network weights
    weights: Array2<A>,
    /// Network biases
    biases: Array1<A>,
    /// Input size (gradient features)
    input_size: usize,
    /// Output size (update features)
    output_size: usize,
}

impl<A: Float + ScalarOperand + Debug> UpdateNetwork<A> {
    /// Create a new update network
    pub fn new(input_size: usize, output_size: usize) -> Self;

    /// Compute parameter update given gradient features
    pub fn compute_update(&self, gradient_features: &Array1<A>) -> Result<Array1<A>>;

    /// Train the network on optimization trajectories
    pub fn train_on_trajectory(
        &mut self,
        trajectory: &OptimizationTrajectory<A, ndarray::Ix1>,
        learning_rate: A,
    ) -> Result<()>;
}
```

## Meta-Learning Framework

### MetaOptimizer

Core meta-learning optimizer that learns to optimize.

```rust
pub struct MetaOptimizer<A: Float, D: Dimension> {
    /// Base optimizer type
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

impl<A: Float + ScalarOperand + Debug, D: Dimension> MetaOptimizer<A, D> {
    /// Create a new meta-optimizer
    pub fn new(
        base_optimizer: String,
        meta_parameters: Array1<A>,
        meta_learning_rate: A,
    ) -> Self;

    /// Add an optimization trajectory for learning
    pub fn add_trajectory(&mut self, trajectory: OptimizationTrajectory<A, D>) -> Result<()>;

    /// Update meta-parameters based on accumulated trajectories
    pub fn meta_update(&mut self) -> Result<()>;

    /// Get current meta-parameters
    pub fn get_meta_parameters(&self) -> &Array1<A>;

    /// Set meta-learning rate
    pub fn set_meta_learning_rate(&mut self, lr: A);

    /// Get number of trajectories
    pub fn trajectory_count(&self) -> usize;

    /// Clear trajectory history
    pub fn clear_trajectories(&mut self);

    /// Enable neural hyperparameter prediction
    pub fn enable_neural_prediction(&mut self, predictor: HyperparameterPredictor<A>);

    /// Predict hyperparameters for new problem
    pub fn predict_hyperparameters(&self, features: &Array1<A>) -> Result<HashMap<String, A>>;
}
```

### OptimizationTrajectory

Represents a complete optimization trajectory for meta-learning.

```rust
pub struct OptimizationTrajectory<A: Float, D: Dimension> {
    /// Sequence of parameter states
    parameter_history: Vec<Array<A, D>>,
    /// Sequence of gradient states
    gradient_history: Vec<Array<A, D>>,
    /// Sequence of loss values
    loss_history: Vec<A>,
    /// Final performance metric
    final_performance: A,
    /// Optimizer hyperparameters used
    hyperparameters: HashMap<String, A>,
}

impl<A: Float, D: Dimension> OptimizationTrajectory<A, D> {
    /// Create a new optimization trajectory
    pub fn new(
        parameter_history: Vec<Array<A, D>>,
        gradient_history: Vec<Array<A, D>>,
        loss_history: Vec<A>,
        final_performance: A,
        hyperparameters: HashMap<String, A>,
    ) -> Self;

    /// Get final performance
    pub fn final_performance(&self) -> A;

    /// Get trajectory length
    pub fn length(&self) -> usize;

    /// Get hyperparameters
    pub fn hyperparameters(&self) -> &HashMap<String, A>;

    /// Get loss at specific step
    pub fn loss_at_step(&self, step: usize) -> Option<A>;

    /// Extract features for meta-learning
    pub fn extract_features(&self) -> Array1<A>;
}
```

### NeuralOptimizer

A neural network that learns optimization update rules.

```rust
pub struct NeuralOptimizer<A: Float> {
    /// Update networks for different parameter types
    update_networks: HashMap<String, UpdateNetwork<A>>,
    /// Meta-parameters for the optimizer
    meta_params: Array1<A>,
    /// Learning rate for meta-learning
    meta_learning_rate: A,
    /// Training history
    training_history: Vec<(Array1<A>, A)>,
}

impl<A: Float + ScalarOperand + Debug> NeuralOptimizer<A> {
    /// Create a new neural optimizer
    pub fn new(network_configs: HashMap<String, (usize, usize)>, meta_learning_rate: A) -> Self;

    /// Compute parameter update using neural networks
    pub fn compute_update(
        &self,
        param_type: &str,
        gradient: &Array1<A>,
        context: &Array1<A>,
    ) -> Result<Array1<A>>;

    /// Train the optimizer on optimization trajectories
    pub fn train_on_trajectories(
        &mut self,
        trajectories: &[OptimizationTrajectory<A, ndarray::Ix1>],
        epochs: usize,
    ) -> Result<()>;

    /// Update meta-parameters
    pub fn meta_update(&mut self, performance_feedback: A) -> Result<()>;

    /// Get meta-parameters
    pub fn get_meta_parameters(&self) -> &Array1<A>;
}
```

## Usage Examples

### Hyperparameter Optimization

```rust
use scirs2_optim::meta_learning::*;
use std::collections::HashMap;

// Grid search example
let grids = HashMap::from([
    ("learning_rate".to_string(), vec![0.001, 0.01, 0.1]),
    ("weight_decay".to_string(), vec![0.0, 0.001, 0.01]),
    ("momentum".to_string(), vec![0.0, 0.9, 0.99]),
]);

let strategy = HyperparameterStrategy::GridSearch { grids };
let mut optimizer = HyperparameterOptimizer::<f64>::new(strategy);

// Optimization loop
while !optimizer.is_complete() {
    // Get next hyperparameters to try
    let hyperparams = optimizer.suggest()?;
    
    // Train model with these hyperparameters
    let performance = train_model_with_hyperparams(&hyperparams)?;
    
    // Update optimizer with results
    optimizer.update(hyperparams, performance);
}

// Get best results
if let Some(best_params) = optimizer.get_best_hyperparameters() {
    println!("Best hyperparameters: {:?}", best_params);
    println!("Best performance: {:?}", optimizer.get_best_performance());
}
```

### Bayesian Optimization

```rust
// Bayesian optimization with expected improvement
let bounds = HashMap::from([
    ("learning_rate".to_string(), (1e-5, 1e-1)),
    ("weight_decay".to_string(), (0.0, 0.1)),
    ("batch_size".to_string(), (16.0, 512.0)),
]);

let strategy = HyperparameterStrategy::BayesianOptimization {
    num_trials: 50,
    bounds,
    acquisition: AcquisitionFunction::ExpectedImprovement,
};

let mut optimizer = HyperparameterOptimizer::<f64>::new(strategy);

// More sophisticated optimization loop
for trial in 0..50 {
    let hyperparams = optimizer.suggest()?;
    
    // Extract hyperparameters
    let lr = hyperparams["learning_rate"];
    let wd = hyperparams["weight_decay"];
    let bs = hyperparams["batch_size"] as usize;
    
    // Train and evaluate
    let performance = train_and_evaluate(lr, wd, bs)?;
    
    optimizer.update(hyperparams, performance);
    
    println!("Trial {}: performance = {:.4}", trial, performance);
}
```

### Neural Hyperparameter Prediction

```rust
use ndarray::Array1;

// Create and train hyperparameter predictor
let mut predictor = HyperparameterPredictor::<f64>::new(10, 50, 3);

// Prepare training data
let mut features = Vec::new();
let mut hyperparams = Vec::new();

for _ in 0..1000 {
    // Extract problem features (dataset size, dimensions, etc.)
    let problem_features = extract_problem_features(&dataset);
    features.push(problem_features);
    
    // Get optimal hyperparameters for this problem
    let optimal_hyperparams = find_optimal_hyperparams(&dataset);
    hyperparams.push(optimal_hyperparams);
}

// Train the predictor
predictor.train(&features, &hyperparams, 0.001, 100)?;

// Use predictor for new problems
let new_problem_features = Array1::from_vec(vec![
    1000.0, 784.0, 10.0, // dataset_size, input_dim, output_dim
    0.8, 0.1, 0.1,       // train_ratio, val_ratio, test_ratio
    2.5, 1.2, 0.3, 0.05  // mean, std, sparsity, noise_level
]);

let predicted_hyperparams = predictor.predict(&new_problem_features)?;
println!("Predicted hyperparameters: {:?}", predicted_hyperparams);
```

### Meta-Learning Optimizer

```rust
use scirs2_optim::meta_learning::*;
use ndarray::{Array1, Array2};

// Create meta-optimizer
let meta_params = Array1::from_vec(vec![0.001, 0.9, 0.999]); // lr, beta1, beta2
let mut meta_optimizer = MetaOptimizer::new(
    "Adam".to_string(),
    meta_params,
    0.01, // meta-learning rate
);

// Collect optimization trajectories from different tasks
for task in tasks {
    let mut trajectory_params = Vec::new();
    let mut trajectory_grads = Vec::new();
    let mut trajectory_losses = Vec::new();
    
    // Run optimization on this task
    let mut params = initialize_parameters(&task);
    for step in 0..1000 {
        let (loss, gradients) = compute_loss_and_gradients(&params, &task);
        
        trajectory_params.push(params.clone());
        trajectory_grads.push(gradients.clone());
        trajectory_losses.push(loss);
        
        // Update parameters
        params = update_parameters(params, gradients, &meta_optimizer);
    }
    
    let final_performance = evaluate_final_performance(&params, &task);
    let hyperparams = HashMap::from([
        ("learning_rate".to_string(), 0.001),
        ("beta1".to_string(), 0.9),
        ("beta2".to_string(), 0.999),
    ]);
    
    let trajectory = OptimizationTrajectory::new(
        trajectory_params,
        trajectory_grads,
        trajectory_losses,
        final_performance,
        hyperparams,
    );
    
    meta_optimizer.add_trajectory(trajectory)?;
}

// Update meta-parameters based on all trajectories
meta_optimizer.meta_update()?;

// Use learned meta-parameters for new tasks
let learned_params = meta_optimizer.get_meta_parameters();
println!("Learned meta-parameters: {:?}", learned_params);
```

### Neural Optimizer

```rust
// Configure neural optimizer with different networks for different parameter types
let network_configs = HashMap::from([
    ("weights".to_string(), (20, 10)),  // 20 input features, 10 output features
    ("biases".to_string(), (15, 5)),    // 15 input features, 5 output features
]);

let mut neural_optimizer = NeuralOptimizer::<f64>::new(network_configs, 0.001);

// Training loop
for epoch in 0..100 {
    // Collect trajectories from current neural optimizer
    let trajectories = collect_optimization_trajectories(&neural_optimizer)?;
    
    // Train the neural optimizer
    neural_optimizer.train_on_trajectories(&trajectories, 10)?;
    
    // Evaluate performance and provide feedback
    let performance = evaluate_neural_optimizer(&neural_optimizer)?;
    neural_optimizer.meta_update(performance)?;
    
    println!("Epoch {}: meta-performance = {:.4}", epoch, performance);
}

// Use trained neural optimizer
let gradient = Array1::from_vec(vec![0.1, -0.05, 0.2, 0.15]);
let context = Array1::from_vec(vec![1.0, 0.5, 2.0, 0.8, 1.2]);

let update = neural_optimizer.compute_update("weights", &gradient, &context)?;
println!("Neural optimizer update: {:?}", update);
```

### Population-Based Training

```rust
// Population-based hyperparameter optimization
let strategy = HyperparameterStrategy::PopulationBased {
    population_size: 20,
    perturbation_factor: 0.2,
};

let mut optimizer = HyperparameterOptimizer::<f64>::new(strategy);

// Initialize population
let mut population = Vec::new();
for _ in 0..20 {
    let hyperparams = optimizer.suggest()?;
    let performance = train_model_with_hyperparams(&hyperparams)?;
    population.push((hyperparams, performance));
}

// Evolution loop
for generation in 0..50 {
    // Sort population by performance
    population.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    // Keep top performers, evolve bottom half
    for i in 10..20 {
        // Copy from top performer
        let parent_idx = i % 10;
        let mut new_hyperparams = population[parent_idx].0.clone();
        
        // Perturb hyperparameters
        for (_, value) in new_hyperparams.iter_mut() {
            *value *= 1.0 + (random::<f64>() - 0.5) * 0.2;
        }
        
        // Evaluate new individual
        let performance = train_model_with_hyperparams(&new_hyperparams)?;
        population[i] = (new_hyperparams, performance);
        
        // Update optimizer
        optimizer.update(population[i].0.clone(), population[i].1);
    }
    
    let best_performance = population[0].1;
    println!("Generation {}: best performance = {:.4}", generation, best_performance);
}
```

## Best Practices

### Hyperparameter Optimization

1. **Start Simple**: Begin with grid search or random search
2. **Use Priors**: Leverage domain knowledge for bounds and distributions
3. **Early Stopping**: Implement early stopping for expensive evaluations
4. **Multi-Fidelity**: Use lower-fidelity evaluations for initial screening
5. **Parallel Evaluation**: Run multiple evaluations in parallel when possible

### Neural Optimizers

1. **Feature Engineering**: Design good gradient and context features
2. **Architecture**: Start with simple networks, increase complexity gradually
3. **Training Data**: Collect diverse optimization trajectories
4. **Regularization**: Prevent overfitting to specific problem types
5. **Evaluation**: Test on held-out problem distributions

### Meta-Learning

1. **Task Diversity**: Ensure meta-training covers diverse optimization problems
2. **Adaptation Speed**: Balance between quick adaptation and stability
3. **Transfer Learning**: Leverage similarities between problem domains
4. **Computational Budget**: Consider the cost of meta-learning vs. benefits
5. **Validation**: Use proper meta-validation splits

### Implementation Tips

```rust
// Use proper error handling
match optimizer.suggest() {
    Ok(hyperparams) => {
        // Use hyperparams
    }
    Err(e) => {
        eprintln!("Failed to suggest hyperparams: {}", e);
        // Handle error appropriately
    }
}

// Monitor convergence
let mut convergence_history = Vec::new();
for trial in 0..max_trials {
    let performance = run_trial(&optimizer)?;
    convergence_history.push(performance);
    
    // Check for convergence
    if is_converged(&convergence_history) {
        break;
    }
}

// Save and load meta-learning state
let state = meta_optimizer.save_state()?;
save_to_file("meta_optimizer_state.json", &state)?;

// Later...
let loaded_state = load_from_file("meta_optimizer_state.json")?;
meta_optimizer.load_state(loaded_state)?;
```

The meta-learning API provides powerful tools for automatic optimization algorithm improvement, from simple hyperparameter tuning to sophisticated neural optimizers that learn to adapt to new problems.