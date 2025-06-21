# Comprehensive Examples

This document provides detailed, real-world examples demonstrating how to use `scirs2-optimize` for various optimization problems.

## Table of Contents

1. [Basic Optimization Examples](#basic-optimization-examples)
2. [Machine Learning Applications](#machine-learning-applications)
3. [Engineering Applications](#engineering-applications)
4. [Financial Applications](#financial-applications)
5. [Scientific Computing](#scientific-computing)
6. [Advanced Use Cases](#advanced-use-cases)

## Basic Optimization Examples

### 1. Simple Quadratic Function

The most basic example - minimizing a simple quadratic function.

```rust
use scirs2_optimize::prelude::*;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Minimize f(x, y) = (x-1)² + (y-2)²
    // Analytical minimum: (1, 2) with value 0
    let objective = |x: &ArrayView1<f64>| -> f64 {
        (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
    };
    
    // Analytical gradient: [2(x-1), 2(y-2)]
    let gradient = |x: &ArrayView1<f64>| -> Array1<f64> {
        Array1::from_vec(vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)])
    };
    
    let x0 = Array1::from_vec(vec![0.0, 0.0]);
    
    let mut options = Options::default();
    options.jac = Some(Box::new(gradient));
    
    let result = minimize(objective, &x0, Method::BFGS, Some(options))?;
    
    println!("Solution: [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("Function value: {:.2e}", result.fun);
    println!("Iterations: {}", result.nit);
    println!("Success: {}", result.success);
    
    // Verify solution
    assert!((result.x[0] - 1.0).abs() < 1e-6);
    assert!((result.x[1] - 2.0).abs() < 1e-6);
    assert!(result.fun < 1e-10);
    
    Ok(())
}
```

### 2. Rosenbrock Function

The classic Rosenbrock function - a challenging test case for optimization algorithms.

```rust
use scirs2_optimize::prelude::*;
use ndarray::Array1;

// Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
// Global minimum at (a,a²) with value 0
// Standard parameters: a=1, b=100
fn rosenbrock(x: &ArrayView1<f64>) -> f64 {
    let (a, b) = (1.0, 100.0);
    (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
}

// Analytical gradient
fn rosenbrock_gradient(x: &ArrayView1<f64>) -> Array1<f64> {
    let (a, b) = (1.0, 100.0);
    let df_dx = -2.0 * (a - x[0]) - 4.0 * b * x[0] * (x[1] - x[0].powi(2));
    let df_dy = 2.0 * b * (x[1] - x[0].powi(2));
    Array1::from_vec(vec![df_dx, df_dy])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x0 = Array1::from_vec(vec![-1.2, 1.0]); // Standard starting point
    
    // Compare different algorithms
    let algorithms = vec![
        ("BFGS", Method::BFGS),
        ("L-BFGS", Method::LBFGS),
        ("CG", Method::CG),
        ("Powell", Method::Powell),
    ];
    
    for (name, method) in algorithms {
        let mut options = Options::default();
        options.max_iter = 1000;
        
        // Provide gradient for gradient-based methods
        if matches!(method, Method::BFGS | Method::LBFGS | Method::CG) {
            options.jac = Some(Box::new(rosenbrock_gradient));
        }
        
        let start = std::time::Instant::now();
        let result = minimize(rosenbrock, &x0, method, Some(options))?;
        let duration = start.elapsed();
        
        println!("\n{} Results:", name);
        println!("  Solution: [{:.6}, {:.6}]", result.x[0], result.x[1]);
        println!("  Function value: {:.2e}", result.fun);
        println!("  Iterations: {}", result.nit);
        println!("  Function evaluations: {}", result.nfev);
        println!("  Time: {:.2?}", duration);
        println!("  Success: {}", result.success);
    }
    
    Ok(())
}
```

### 3. Constrained Optimization Example

Minimize a function subject to constraints.

```rust
use scirs2_optimize::{constrained::*, prelude::*};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Minimize f(x,y) = (x-2)² + (y-1)²
    // Subject to: x + y ≤ 3 (inequality)
    //            x² + y² = 5 (equality)
    
    let objective = |x: &ArrayView1<f64>| -> f64 {
        (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2)
    };
    
    // Inequality constraint: 3 - x - y ≥ 0
    let ineq_constraint = |x: &ArrayView1<f64>| -> f64 {
        3.0 - x[0] - x[1]
    };
    
    // Equality constraint: x² + y² - 5 = 0
    let eq_constraint = |x: &ArrayView1<f64>| -> f64 {
        x[0].powi(2) + x[1].powi(2) - 5.0
    };
    
    let constraints = vec![
        Constraint {
            fun: Box::new(ineq_constraint),
            constraint_type: ConstraintType::Inequality,
            jac: None, // Auto-computed
        },
        Constraint {
            fun: Box::new(eq_constraint),
            constraint_type: ConstraintType::Equality,
            jac: None,
        },
    ];
    
    // Starting point on the circle x² + y² = 5
    let x0 = Array1::from_vec(vec![2.0, 1.0]);
    
    let result = minimize_constrained(
        objective,
        &x0,
        &constraints,
        Method::SLSQP,
        None
    )?;
    
    println!("Constrained minimum:");
    println!("  Solution: [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Objective value: {:.6}", result.fun);
    
    // Verify constraints
    let ineq_val = ineq_constraint(&result.x.view());
    let eq_val = eq_constraint(&result.x.view());
    println!("  Inequality constraint (≥0): {:.6}", ineq_val);
    println!("  Equality constraint (=0): {:.6}", eq_val);
    
    assert!(ineq_val >= -1e-6, "Inequality constraint violated");
    assert!(eq_val.abs() < 1e-6, "Equality constraint violated");
    
    Ok(())
}
```

## Machine Learning Applications

### 1. Logistic Regression from Scratch

Implement logistic regression using stochastic optimization.

```rust
use scirs2_optimize::stochastic::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::prelude::*;

struct LogisticRegressionData {
    features: Array2<f64>,
    labels: Array1<f64>,
    n_samples: usize,
    n_features: usize,
}

impl LogisticRegressionData {
    fn new(features: Array2<f64>, labels: Array1<f64>) -> Self {
        let (n_samples, n_features) = features.dim();
        Self { features, labels, n_samples, n_features }
    }
    
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z.max(-709.0).min(709.0)).exp()) // Clamp to prevent overflow
    }
    
    fn predict(&self, params: &ArrayView1<f64>, features: &ArrayView1<f64>) -> f64 {
        let z = features.dot(params);
        Self::sigmoid(z)
    }
    
    fn log_likelihood(&self, params: &ArrayView1<f64>, batch_indices: &[f64]) -> f64 {
        let mut ll = 0.0;
        
        for &idx in batch_indices {
            let i = (idx as usize) % self.n_samples;
            let features = self.features.row(i);
            let y = self.labels[i];
            
            let p = self.predict(params, &features);
            let p_clamped = p.max(1e-15).min(1.0 - 1e-15); // Numerical stability
            
            ll += y * p_clamped.ln() + (1.0 - y) * (1.0 - p_clamped).ln();
        }
        
        -ll / batch_indices.len() as f64 // Negative log-likelihood
    }
}

impl StochasticGradientFunction for LogisticRegressionData {
    fn compute_gradient(&mut self, params: &ArrayView1<f64>, batch_indices: &[f64]) -> Array1<f64> {
        let mut grad = Array1::zeros(params.len());
        
        for &idx in batch_indices {
            let i = (idx as usize) % self.n_samples;
            let features = self.features.row(i);
            let y = self.labels[i];
            
            let p = self.predict(params, &features);
            let error = p - y;
            
            for j in 0..params.len() {
                grad[j] += error * features[j];
            }
        }
        
        grad / batch_indices.len() as f64
    }
    
    fn compute_value(&mut self, params: &ArrayView1<f64>, batch_indices: &[f64]) -> f64 {
        self.log_likelihood(params, batch_indices)
    }
}

fn generate_classification_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = thread_rng();
    
    // Generate features
    let mut features = Array2::zeros((n_samples, n_features + 1)); // +1 for bias term
    for i in 0..n_samples {
        features[[i, 0]] = 1.0; // Bias term
        for j in 1..n_features + 1 {
            features[[i, j]] = rng.gen_range(-2.0..2.0);
        }
    }
    
    // Generate true parameters
    let true_params: Array1<f64> = (0..n_features + 1)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    
    // Generate labels
    let labels: Array1<f64> = (0..n_samples)
        .map(|i| {
            let logit = features.row(i).dot(&true_params);
            let p = 1.0 / (1.0 + (-logit).exp());
            if rng.gen::<f64>() < p { 1.0 } else { 0.0 }
        })
        .collect();
    
    (features, labels)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate synthetic classification data
    let (features, labels) = generate_classification_data(1000, 5);
    println!("Generated {} samples with {} features", features.nrows(), features.ncols());
    
    // Split into train/test
    let train_size = 800;
    let train_features = features.slice(s![0..train_size, ..]).to_owned();
    let train_labels = labels.slice(s![0..train_size]).to_owned();
    let test_features = features.slice(s![train_size.., ..]).to_owned();
    let test_labels = labels.slice(s![train_size..]).to_owned();
    
    // Create logistic regression model
    let mut logreg = LogisticRegressionData::new(train_features, train_labels);
    
    // Initial parameters (random initialization)
    let mut rng = thread_rng();
    let x0: Array1<f64> = (0..features.ncols())
        .map(|_| rng.gen_range(-0.1..0.1))
        .collect();
    
    // Create data provider
    let data_provider = Box::new(InMemoryDataProvider::new(
        (0..train_size).map(|i| i as f64).collect()
    ));
    
    // Training with different optimizers
    let optimizers = vec![
        ("SGD", OptimizerType::SGD),
        ("Adam", OptimizerType::Adam),
        ("AdamW", OptimizerType::AdamW),
    ];
    
    for (name, opt_type) in optimizers {
        println!("\nTraining with {}:", name);
        
        let mut model = LogisticRegressionData::new(
            logreg.features.clone(),
            logreg.labels.clone()
        );
        
        let data_provider = Box::new(InMemoryDataProvider::new(
            (0..train_size).map(|i| i as f64).collect()
        ));
        
        let result = match opt_type {
            OptimizerType::SGD => {
                let options = SGDOptions {
                    learning_rate: 0.01,
                    max_iter: 1000,
                    batch_size: Some(32),
                    tol: 1e-6,
                    ..Default::default()
                };
                minimize_sgd(model, x0.clone(), data_provider, options)?
            }
            OptimizerType::Adam => {
                let options = AdamOptions {
                    learning_rate: 0.001,
                    max_iter: 1000,
                    batch_size: Some(32),
                    tol: 1e-6,
                    ..Default::default()
                };
                minimize_adam(model, x0.clone(), data_provider, options)?
            }
            OptimizerType::AdamW => {
                let options = AdamWOptions {
                    learning_rate: 0.001,
                    max_iter: 1000,
                    batch_size: Some(32),
                    weight_decay: 0.01,
                    tol: 1e-6,
                    ..Default::default()
                };
                minimize_adamw(model, x0.clone(), data_provider, options)?
            }
        };
        
        // Evaluate on test set
        let mut correct = 0;
        for i in 0..test_features.nrows() {
            let features = test_features.row(i);
            let true_label = test_labels[i];
            
            let z = features.dot(&result.x);
            let predicted_prob = 1.0 / (1.0 + (-z).exp());
            let predicted_label = if predicted_prob > 0.5 { 1.0 } else { 0.0 };
            
            if predicted_label == true_label {
                correct += 1;
            }
        }
        
        let accuracy = correct as f64 / test_features.nrows() as f64;
        
        println!("  Final loss: {:.6}", result.fun);
        println!("  Iterations: {}", result.nit);
        println!("  Test accuracy: {:.2}%", accuracy * 100.0);
    }
    
    Ok(())
}

#[derive(Clone)]
enum OptimizerType {
    SGD,
    Adam,
    AdamW,
}
```

### 2. Neural Network Training

Train a simple neural network for regression.

```rust
use scirs2_optimize::stochastic::*;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use rand::prelude::*;

struct NeuralNetwork {
    features: Array2<f64>,
    targets: Array1<f64>,
    architecture: Vec<usize>, // [input_size, hidden1, hidden2, ..., output_size]
    n_samples: usize,
}

impl NeuralNetwork {
    fn new(features: Array2<f64>, targets: Array1<f64>, hidden_sizes: Vec<usize>) -> Self {
        let n_samples = features.nrows();
        let input_size = features.ncols();
        let output_size = 1; // Regression
        
        let mut architecture = vec![input_size];
        architecture.extend(hidden_sizes);
        architecture.push(output_size);
        
        Self { features, targets, architecture, n_samples }
    }
    
    fn get_weight_ranges(&self) -> Vec<(usize, usize)> {
        let mut ranges = Vec::new();
        let mut start = 0;
        
        for i in 0..self.architecture.len() - 1 {
            let weight_count = self.architecture[i] * self.architecture[i + 1];
            let bias_count = self.architecture[i + 1];
            ranges.push((start, start + weight_count));
            ranges.push((start + weight_count, start + weight_count + bias_count));
            start += weight_count + bias_count;
        }
        
        ranges
    }
    
    fn forward(&self, params: &ArrayView1<f64>, input: &ArrayView1<f64>) -> f64 {
        let mut activation = input.to_owned();
        let ranges = self.get_weight_ranges();
        
        for layer in 0..self.architecture.len() - 1 {
            let weight_range = &ranges[layer * 2];
            let bias_range = &ranges[layer * 2 + 1];
            
            let weights = params.slice(s![weight_range.0..weight_range.1]);
            let biases = params.slice(s![bias_range.0..bias_range.1]);
            
            // Reshape weights to matrix
            let weight_matrix = Array2::from_shape_vec(
                (self.architecture[layer], self.architecture[layer + 1]),
                weights.to_vec(),
            ).unwrap();
            
            // Linear transformation
            let mut new_activation = Array1::zeros(self.architecture[layer + 1]);
            for j in 0..self.architecture[layer + 1] {
                for i in 0..self.architecture[layer] {
                    new_activation[j] += activation[i] * weight_matrix[[i, j]];
                }
                new_activation[j] += biases[j];
            }
            
            // Apply activation function (tanh for hidden layers, linear for output)
            if layer < self.architecture.len() - 2 {
                new_activation.mapv_inplace(|x| x.tanh());
            }
            
            activation = new_activation;
        }
        
        activation[0] // Single output for regression
    }
    
    fn total_parameters(&self) -> usize {
        let mut total = 0;
        for i in 0..self.architecture.len() - 1 {
            total += self.architecture[i] * self.architecture[i + 1]; // Weights
            total += self.architecture[i + 1]; // Biases
        }
        total
    }
}

impl StochasticGradientFunction for NeuralNetwork {
    fn compute_gradient(&mut self, params: &ArrayView1<f64>, batch_indices: &[f64]) -> Array1<f64> {
        let mut grad = Array1::zeros(params.len());
        let h = 1e-7; // Finite difference step
        
        // Numerical gradient (for simplicity)
        for i in 0..params.len() {
            let mut params_plus = params.to_owned();
            let mut params_minus = params.to_owned();
            params_plus[i] += h;
            params_minus[i] -= h;
            
            let loss_plus = self.compute_value(&params_plus.view(), batch_indices);
            let loss_minus = self.compute_value(&params_minus.view(), batch_indices);
            
            grad[i] = (loss_plus - loss_minus) / (2.0 * h);
        }
        
        grad
    }
    
    fn compute_value(&mut self, params: &ArrayView1<f64>, batch_indices: &[f64]) -> f64 {
        let mut loss = 0.0;
        
        for &idx in batch_indices {
            let i = (idx as usize) % self.n_samples;
            let features = self.features.row(i);
            let target = self.targets[i];
            
            let prediction = self.forward(params, &features);
            loss += 0.5 * (prediction - target).powi(2); // MSE
        }
        
        loss / batch_indices.len() as f64
    }
}

// Generate synthetic regression data
fn generate_regression_data(n_samples: usize, noise_level: f64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = thread_rng();
    
    // Features: x1, x2
    let features = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
        rng.gen_range(-2.0..2.0)
    });
    
    // Target: y = sin(x1) + cos(x2) + noise
    let targets: Array1<f64> = features.outer_iter()
        .map(|row| {
            let x1 = row[0];
            let x2 = row[1];
            x1.sin() + x2.cos() + rng.gen_range(-noise_level..noise_level)
        })
        .collect();
    
    (features, targets)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate synthetic data
    let (features, targets) = generate_regression_data(1000, 0.1);
    println!("Generated {} samples for regression", features.nrows());
    
    // Create neural network: 2 -> 10 -> 5 -> 1
    let hidden_sizes = vec![10, 5];
    let mut nn = NeuralNetwork::new(features, targets, hidden_sizes);
    
    println!("Neural network architecture: {:?}", nn.architecture);
    println!("Total parameters: {}", nn.total_parameters());
    
    // Random initialization
    let mut rng = thread_rng();
    let x0: Array1<f64> = (0..nn.total_parameters())
        .map(|_| rng.gen_range(-0.1..0.1))
        .collect();
    
    // Create data provider
    let data_provider = Box::new(InMemoryDataProvider::new(
        (0..nn.n_samples).map(|i| i as f64).collect()
    ));
    
    // Train with Adam
    let options = AdamOptions {
        learning_rate: 0.001,
        max_iter: 2000,
        batch_size: Some(32),
        tol: 1e-6,
        ..Default::default()
    };
    
    let start_time = std::time::Instant::now();
    let result = minimize_adam(nn, x0, data_provider, options)?;
    let training_time = start_time.elapsed();
    
    println!("\nTraining Results:");
    println!("  Final loss: {:.6}", result.fun);
    println!("  Iterations: {}", result.nit);
    println!("  Training time: {:.2?}", training_time);
    println!("  Success: {}", result.success);
    
    // Test the trained network
    let test_features = Array2::from_shape_vec((5, 2), vec![
        0.0, 0.0,    // sin(0) + cos(0) = 1
        1.0, 0.0,    // sin(1) + cos(0) ≈ 1.84
        0.0, 1.0,    // sin(0) + cos(1) ≈ 0.54
        1.5, 1.5,    // sin(1.5) + cos(1.5) ≈ 1.07
        -1.0, -1.0,  // sin(-1) + cos(-1) ≈ -0.30
    ])?;
    
    println!("\nTest Predictions:");
    for (i, row) in test_features.outer_iter().enumerate() {
        let x1 = row[0];
        let x2 = row[1];
        let true_value = x1.sin() + x2.cos();
        
        // Create a dummy NN for prediction (we need the architecture)
        let dummy_features = Array2::zeros((1, 2));
        let dummy_targets = Array1::zeros(1);
        let dummy_nn = NeuralNetwork::new(dummy_features, dummy_targets, vec![10, 5]);
        let predicted = dummy_nn.forward(&result.x.view(), &row);
        
        println!("  x=[{:.1}, {:.1}]: true={:.3}, predicted={:.3}, error={:.3}",
                 x1, x2, true_value, predicted, (true_value - predicted).abs());
    }
    
    Ok(())
}
```

## Engineering Applications

### 1. Parameter Identification in Dynamical Systems

Identify parameters of a damped oscillator from experimental data.

```rust
use scirs2_optimize::least_squares::*;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

// Model: x(t) = A * exp(-ζωt) * cos(ωd*t + φ)
// where ωd = ω * sqrt(1 - ζ²) for underdamped case
struct DampedOscillator {
    time_data: Vec<f64>,
    position_data: Vec<f64>,
}

impl DampedOscillator {
    fn new(time_data: Vec<f64>, position_data: Vec<f64>) -> Self {
        assert_eq!(time_data.len(), position_data.len());
        Self { time_data, position_data }
    }
    
    // Parameters: [A, ω, ζ, φ]
    fn model(&self, params: &[f64], t: f64) -> f64 {
        let (a, omega, zeta, phi) = (params[0], params[1], params[2], params[3]);
        
        if zeta >= 1.0 {
            // Overdamped case
            let r1 = -omega * (zeta + (zeta.powi(2) - 1.0).sqrt());
            let r2 = -omega * (zeta - (zeta.powi(2) - 1.0).sqrt());
            a * ((r1 * t).exp() + (r2 * t).exp()) // Simplified for demonstration
        } else {
            // Underdamped case
            let omega_d = omega * (1.0 - zeta.powi(2)).sqrt();
            a * (-zeta * omega * t).exp() * (omega_d * t + phi).cos()
        }
    }
    
    fn residuals(&self, params: &[f64]) -> Array1<f64> {
        let mut residuals = Array1::zeros(self.time_data.len());
        
        for (i, (&t, &x_measured)) in self.time_data.iter()
            .zip(self.position_data.iter())
            .enumerate() {
            let x_predicted = self.model(params, t);
            residuals[i] = x_measured - x_predicted;
        }
        
        residuals
    }
}

// Generate synthetic experimental data
fn generate_oscillator_data(
    true_params: &[f64], 
    t_max: f64, 
    n_points: usize, 
    noise_level: f64
) -> (Vec<f64>, Vec<f64>) {
    use rand::prelude::*;
    let mut rng = thread_rng();
    
    let dt = t_max / (n_points - 1) as f64;
    let time_data: Vec<f64> = (0..n_points).map(|i| i as f64 * dt).collect();
    
    let oscillator = DampedOscillator::new(vec![], vec![]);
    
    let position_data: Vec<f64> = time_data.iter()
        .map(|&t| {
            let true_value = oscillator.model(true_params, t);
            true_value + rng.gen_range(-noise_level..noise_level)
        })
        .collect();
    
    (time_data, position_data)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // True parameters: A=2.0, ω=5.0, ζ=0.1, φ=π/4
    let true_params = vec![2.0, 5.0, 0.1, PI / 4.0];
    
    // Generate synthetic experimental data
    let (time_data, position_data) = generate_oscillator_data(&true_params, 2.0, 100, 0.05);
    
    println!("Generated {} data points for parameter identification", time_data.len());
    println!("True parameters: A={:.2}, ω={:.2}, ζ={:.3}, φ={:.3}", 
             true_params[0], true_params[1], true_params[2], true_params[3]);
    
    let oscillator = DampedOscillator::new(time_data, position_data);
    
    // Residual function for least squares
    let residual_fn = |params: &[f64], _data: &[f64]| -> Array1<f64> {
        oscillator.residuals(params)
    };
    
    // Initial guess (perturbed true values)
    let x0 = Array1::from_vec(vec![1.5, 4.0, 0.2, 0.5]);
    let dummy_data = Array1::zeros(0);
    
    println!("\nInitial guess: A={:.2}, ω={:.2}, ζ={:.3}, φ={:.3}", 
             x0[0], x0[1], x0[2], x0[3]);
    
    // Parameter identification using different methods
    let methods = vec![
        ("Levenberg-Marquardt", Method::LevenbergMarquardt),
        ("Trust Region Reflective", Method::TrustRegionReflective),
    ];
    
    for (name, method) in methods {
        println!("\n{} Results:", name);
        
        let result = least_squares(
            residual_fn,
            &x0,
            method,
            None, // Auto-compute Jacobian
            &dummy_data,
            None
        )?;
        
        println!("  Identified parameters:");
        println!("    A = {:.4} (true: {:.4}, error: {:.2}%)", 
                 result.x[0], true_params[0], 
                 100.0 * (result.x[0] - true_params[0]).abs() / true_params[0]);
        println!("    ω = {:.4} (true: {:.4}, error: {:.2}%)", 
                 result.x[1], true_params[1], 
                 100.0 * (result.x[1] - true_params[1]).abs() / true_params[1]);
        println!("    ζ = {:.4} (true: {:.4}, error: {:.2}%)", 
                 result.x[2], true_params[2], 
                 100.0 * (result.x[2] - true_params[2]).abs() / true_params[2]);
        println!("    φ = {:.4} (true: {:.4}, error: {:.2}%)", 
                 result.x[3], true_params[3], 
                 100.0 * (result.x[3] - true_params[3]).abs() / true_params[3]);
        
        println!("  Final residual norm: {:.2e}", result.fun);
        println!("  Iterations: {}", result.nit);
        println!("  Success: {}", result.success);
        
        // Compute R² coefficient of determination
        let residuals = oscillator.residuals(&result.x.to_vec());
        let ss_res: f64 = residuals.iter().map(|r| r.powi(2)).sum();
        let y_mean: f64 = oscillator.position_data.iter().sum::<f64>() / oscillator.position_data.len() as f64;
        let ss_tot: f64 = oscillator.position_data.iter()
            .map(|y| (y - y_mean).powi(2))
            .sum();
        let r_squared = 1.0 - ss_res / ss_tot;
        println!("  R² = {:.6}", r_squared);
    }
    
    Ok(())
}
```

### 2. Optimal Control Problem

Solve a trajectory optimization problem for a robotic arm.

```rust
use scirs2_optimize::prelude::*;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

struct RobotArmOptimization {
    n_timesteps: usize,
    dt: f64,
    target_position: [f64; 2],
    arm_lengths: [f64; 2],
    q_weight: f64,    // State cost weight
    r_weight: f64,    // Control cost weight
    qf_weight: f64,   // Terminal cost weight
}

impl RobotArmOptimization {
    fn new(n_timesteps: usize, dt: f64) -> Self {
        Self {
            n_timesteps,
            dt,
            target_position: [1.0, 1.0],
            arm_lengths: [1.0, 1.0],
            q_weight: 1.0,
            r_weight: 0.1,
            qf_weight: 100.0,
        }
    }
    
    // Forward kinematics: compute end-effector position
    fn forward_kinematics(&self, angles: &[f64]) -> [f64; 2] {
        let (q1, q2) = (angles[0], angles[1]);
        let (l1, l2) = (self.arm_lengths[0], self.arm_lengths[1]);
        
        let x = l1 * q1.cos() + l2 * (q1 + q2).cos();
        let y = l1 * q1.sin() + l2 * (q1 + q2).sin();
        
        [x, y]
    }
    
    // Dynamics: simple double integrator model
    fn dynamics(&self, state: &[f64], control: &[f64]) -> [f64; 4] {
        // State: [q1, q2, q1_dot, q2_dot]
        // Control: [tau1, tau2] (torques)
        let q1_dot = state[2];
        let q2_dot = state[3];
        let q1_ddot = control[0]; // Simplified dynamics
        let q2_ddot = control[1];
        
        [q1_dot, q2_dot, q1_ddot, q2_ddot]
    }
    
    // Integrate dynamics using Euler method
    fn integrate_step(&self, state: &[f64], control: &[f64]) -> [f64; 4] {
        let state_dot = self.dynamics(state, control);
        [
            state[0] + self.dt * state_dot[0],
            state[1] + self.dt * state_dot[1],
            state[2] + self.dt * state_dot[2],
            state[3] + self.dt * state_dot[3],
        ]
    }
    
    // Cost function for trajectory optimization
    fn trajectory_cost(&self, decision_vars: &ArrayView1<f64>) -> f64 {
        // Decision variables: [u0, u1, ..., u_{N-1}] where each u_i = [tau1_i, tau2_i]
        // Initial state is fixed: [0, 0, 0, 0]
        let mut state = [0.0, 0.0, 0.0, 0.0]; // [q1, q2, q1_dot, q2_dot]
        let mut total_cost = 0.0;
        
        // Simulate trajectory and accumulate cost
        for k in 0..self.n_timesteps {
            let control = [decision_vars[2 * k], decision_vars[2 * k + 1]];
            
            // Running cost: state deviation + control effort
            let end_effector = self.forward_kinematics(&state[0..2]);
            let position_error = [
                end_effector[0] - self.target_position[0],
                end_effector[1] - self.target_position[1],
            ];
            
            let state_cost = self.q_weight * (
                position_error[0].powi(2) + position_error[1].powi(2) +
                0.1 * (state[2].powi(2) + state[3].powi(2)) // Velocity penalty
            );
            let control_cost = self.r_weight * (control[0].powi(2) + control[1].powi(2));
            
            total_cost += self.dt * (state_cost + control_cost);
            
            // Integrate dynamics
            state = self.integrate_step(&state, &control);
        }
        
        // Terminal cost
        let final_end_effector = self.forward_kinematics(&state[0..2]);
        let final_error = [
            final_end_effector[0] - self.target_position[0],
            final_end_effector[1] - self.target_position[1],
        ];
        let terminal_cost = self.qf_weight * (
            final_error[0].powi(2) + final_error[1].powi(2)
        );
        
        total_cost + terminal_cost
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_timesteps = 50;
    let dt = 0.02; // 20ms timesteps
    let robot = RobotArmOptimization::new(n_timesteps, dt);
    
    println!("Robot Arm Trajectory Optimization");
    println!("  Target position: [{:.2}, {:.2}]", 
             robot.target_position[0], robot.target_position[1]);
    println!("  Time horizon: {:.1}s ({} timesteps)", n_timesteps as f64 * dt, n_timesteps);
    
    // Decision variables: torques for each timestep [tau1_0, tau2_0, tau1_1, tau2_1, ...]
    let n_vars = 2 * n_timesteps;
    
    // Initial guess: zero torques
    let x0 = Array1::zeros(n_vars);
    
    let objective = |x: &ArrayView1<f64>| robot.trajectory_cost(x);
    
    // Add bounds on torques: -10 to 10 Nm
    let bounds = Bounds::new(&vec![(Some(-10.0), Some(10.0)); n_vars]);
    
    let mut options = Options::default();
    options.bounds = Some(bounds);
    options.max_iter = 1000;
    options.gtol = 1e-6;
    
    println!("\nOptimizing trajectory...");
    let start_time = std::time::Instant::now();
    
    let result = minimize(objective, &x0, Method::LBFGS, Some(options))?;
    
    let optimization_time = start_time.elapsed();
    
    println!("\nOptimization Results:");
    println!("  Final cost: {:.6}", result.fun);
    println!("  Iterations: {}", result.nit);
    println!("  Function evaluations: {}", result.nfev);
    println!("  Optimization time: {:.2?}", optimization_time);
    println!("  Success: {}", result.success);
    
    // Simulate the optimal trajectory
    println!("\nSimulating optimal trajectory:");
    let mut state = [0.0, 0.0, 0.0, 0.0];
    
    println!("Time   q1      q2      x_ee    y_ee    Distance to target");
    for k in 0..n_timesteps.min(10) { // Show first 10 timesteps
        let control = [result.x[2 * k], result.x[2 * k + 1]];
        let end_effector = robot.forward_kinematics(&state[0..2]);
        let distance = ((end_effector[0] - robot.target_position[0]).powi(2) +
                       (end_effector[1] - robot.target_position[1]).powi(2)).sqrt();
        
        println!("{:.2}   {:.3}   {:.3}   {:.3}   {:.3}   {:.3}",
                 k as f64 * dt, state[0], state[1], 
                 end_effector[0], end_effector[1], distance);
        
        state = robot.integrate_step(&state, &control);
    }
    
    // Final position
    let final_end_effector = robot.forward_kinematics(&state[0..2]);
    let final_distance = ((final_end_effector[0] - robot.target_position[0]).powi(2) +
                         (final_end_effector[1] - robot.target_position[1]).powi(2)).sqrt();
    
    println!("...");
    println!("Final end-effector position: [{:.4}, {:.4}]", 
             final_end_effector[0], final_end_effector[1]);
    println!("Final distance to target: {:.4}", final_distance);
    
    // Analyze control effort
    let max_torque = result.x.iter().map(|&u| u.abs()).fold(0.0, f64::max);
    let avg_torque = result.x.iter().map(|&u| u.abs()).sum::<f64>() / result.x.len() as f64;
    
    println!("\nControl Analysis:");
    println!("  Maximum torque: {:.2} Nm", max_torque);
    println!("  Average torque magnitude: {:.2} Nm", avg_torque);
    
    Ok(())
}
```

## Financial Applications

### 1. Portfolio Optimization with Risk Management

Modern portfolio theory with additional constraints.

```rust
use scirs2_optimize::{constrained::*, prelude::*};
use ndarray::{Array1, Array2};

struct PortfolioOptimizer {
    expected_returns: Array1<f64>,
    covariance_matrix: Array2<f64>,
    n_assets: usize,
}

impl PortfolioOptimizer {
    fn new(expected_returns: Array1<f64>, covariance_matrix: Array2<f64>) -> Self {
        let n_assets = expected_returns.len();
        assert_eq!(covariance_matrix.dim(), (n_assets, n_assets));
        
        Self { expected_returns, covariance_matrix, n_assets }
    }
    
    // Mean-variance utility: μ'w - (λ/2)w'Σw
    fn utility(&self, weights: &ArrayView1<f64>, risk_aversion: f64) -> f64 {
        let expected_return = self.expected_returns.dot(weights);
        let portfolio_variance = weights.dot(&self.covariance_matrix.dot(weights));
        expected_return - 0.5 * risk_aversion * portfolio_variance
    }
    
    // Portfolio variance
    fn portfolio_variance(&self, weights: &ArrayView1<f64>) -> f64 {
        weights.dot(&self.covariance_matrix.dot(weights))
    }
    
    // Portfolio return
    fn portfolio_return(&self, weights: &ArrayView1<f64>) -> f64 {
        self.expected_returns.dot(weights)
    }
    
    // Maximum drawdown constraint (simplified)
    fn max_drawdown_constraint(&self, weights: &ArrayView1<f64>, max_dd: f64) -> f64 {
        // Simplified: assume drawdown is proportional to volatility
        let volatility = self.portfolio_variance(weights).sqrt();
        max_dd - 2.0 * volatility // Rough approximation
    }
    
    // Sector exposure constraint
    fn sector_constraint(&self, weights: &ArrayView1<f64>, sector_indices: &[usize], max_exposure: f64) -> f64 {
        let sector_weight: f64 = sector_indices.iter().map(|&i| weights[i]).sum();
        max_exposure - sector_weight
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example data: 5 assets with different characteristics
    let asset_names = vec!["Stocks", "Bonds", "REITs", "Commodities", "Cash"];
    let n_assets = asset_names.len();
    
    // Annual expected returns (%)
    let expected_returns = Array1::from_vec(vec![0.10, 0.04, 0.08, 0.06, 0.02]);
    
    // Covariance matrix (annual)
    let covariance_data = vec![
        0.0400, 0.0012, 0.0080, 0.0100, 0.0000, // Stocks
        0.0012, 0.0025, 0.0010, 0.0020, 0.0000, // Bonds
        0.0080, 0.0010, 0.0225, 0.0050, 0.0000, // REITs
        0.0100, 0.0020, 0.0050, 0.0625, 0.0000, // Commodities
        0.0000, 0.0000, 0.0000, 0.0000, 0.0001, // Cash
    ];
    let covariance_matrix = Array2::from_shape_vec((n_assets, n_assets), covariance_data)?;
    
    let portfolio = PortfolioOptimizer::new(expected_returns, covariance_matrix);
    
    println!("Portfolio Optimization with Risk Management");
    println!("Assets: {:?}", asset_names);
    println!("Expected returns: {:?}", portfolio.expected_returns);
    
    // Risk aversion parameter
    let risk_aversion = 5.0;
    
    // Objective: maximize utility = minimize negative utility
    let objective = |w: &ArrayView1<f64>| -> f64 {
        -portfolio.utility(w, risk_aversion)
    };
    
    // Constraints
    let mut constraints = Vec::new();
    
    // 1. Weights sum to 1
    constraints.push(Constraint {
        fun: Box::new(|w: &ArrayView1<f64>| w.sum() - 1.0),
        constraint_type: ConstraintType::Equality,
        jac: None,
    });
    
    // 2. Minimum return constraint (6% annually)
    let min_return = 0.06;
    constraints.push(Constraint {
        fun: Box::new(move |w: &ArrayView1<f64>| {
            portfolio.portfolio_return(w) - min_return
        }),
        constraint_type: ConstraintType::Inequality,
        jac: None,
    });
    
    // 3. Maximum volatility constraint (15% annually)
    let max_volatility = 0.15;
    constraints.push(Constraint {
        fun: Box::new(move |w: &ArrayView1<f64>| {
            max_volatility.powi(2) - portfolio.portfolio_variance(w)
        }),
        constraint_type: ConstraintType::Inequality,
        jac: None,
    });
    
    // 4. Sector constraints: max 60% in risky assets (stocks + REITs + commodities)
    let risky_indices = vec![0, 2, 3]; // Stocks, REITs, Commodities
    let max_risky_exposure = 0.60;
    constraints.push(Constraint {
        fun: Box::new(move |w: &ArrayView1<f64>| {
            portfolio.sector_constraint(w, &risky_indices, max_risky_exposure)
        }),
        constraint_type: ConstraintType::Inequality,
        jac: None,
    });
    
    // Bounds: 0 <= w_i <= 0.4 (max 40% in any single asset)
    let bounds = Bounds::new(&vec![(Some(0.0), Some(0.4)); n_assets]);
    
    // Initial guess: equal weights
    let x0 = Array1::from_elem(n_assets, 1.0 / n_assets as f64);
    
    let mut options = Options::default();
    options.bounds = Some(bounds);
    options.max_iter = 1000;
    
    println!("\nOptimizing portfolio...");
    let start_time = std::time::Instant::now();
    
    let result = minimize_constrained(objective, &x0, &constraints, Method::SLSQP, Some(options))?;
    
    let optimization_time = start_time.elapsed();
    
    println!("\nOptimization Results:");
    println!("  Success: {}", result.success);
    println!("  Iterations: {}", result.nit);
    println!("  Optimization time: {:.2?}", optimization_time);
    
    // Analyze optimal portfolio
    println!("\nOptimal Portfolio Allocation:");
    for (i, &weight) in result.x.iter().enumerate() {
        println!("  {}: {:.1}%", asset_names[i], weight * 100.0);
    }
    
    let portfolio_return = portfolio.portfolio_return(&result.x.view());
    let portfolio_vol = portfolio.portfolio_variance(&result.x.view()).sqrt();
    let sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol; // Assume 2% risk-free rate
    
    println!("\nPortfolio Characteristics:");
    println!("  Expected return: {:.2}%", portfolio_return * 100.0);
    println!("  Volatility: {:.2}%", portfolio_vol * 100.0);
    println!("  Sharpe ratio: {:.3}", sharpe_ratio);
    
    // Verify constraints
    println!("\nConstraint Verification:");
    println!("  Sum of weights: {:.6}", result.x.sum());
    println!("  Return constraint: {:.2}% >= {:.2}%", 
             portfolio_return * 100.0, min_return * 100.0);
    println!("  Volatility constraint: {:.2}% <= {:.2}%", 
             portfolio_vol * 100.0, max_volatility * 100.0);
    
    let risky_allocation: f64 = risky_indices.iter().map(|&i| result.x[i]).sum();
    println!("  Risky assets allocation: {:.1}% <= {:.1}%", 
             risky_allocation * 100.0, max_risky_exposure * 100.0);
    
    // Risk analysis
    println!("\nRisk Analysis:");
    
    // Value at Risk (simplified normal approximation)
    let confidence_levels = vec![0.95, 0.99];
    for &confidence in &confidence_levels {
        let z_score = match confidence {
            0.95 => 1.645,
            0.99 => 2.326,
            _ => 1.960,
        };
        let var = z_score * portfolio_vol;
        println!("  VaR ({}%): {:.2}%", (confidence * 100.0) as i32, var * 100.0);
    }
    
    // Component contributions to risk
    println!("\nRisk Contributions:");
    let total_risk = portfolio_vol.powi(2);
    for (i, &weight) in result.x.iter().enumerate() {
        let marginal_risk = portfolio.covariance_matrix.row(i).dot(&result.x);
        let contribution = weight * marginal_risk / total_risk;
        println!("  {}: {:.1}%", asset_names[i], contribution * 100.0);
    }
    
    Ok(())
}
```

## Scientific Computing

### 1. Protein Folding Energy Minimization

Simplified protein folding using optimization.

```rust
use scirs2_optimize::{global::*, prelude::*};
use ndarray::Array1;
use std::f64::consts::PI;

struct ProteinFolder {
    n_residues: usize,
    bond_length: f64,
    amino_acid_types: Vec<u8>, // 0-19 for 20 amino acids
}

impl ProteinFolder {
    fn new(sequence: &str) -> Self {
        let amino_acids = "ACDEFGHIKLMNPQRSTVWY";
        let amino_acid_types: Vec<u8> = sequence.chars()
            .map(|c| amino_acids.find(c).unwrap_or(0) as u8)
            .collect();
        
        Self {
            n_residues: sequence.len(),
            bond_length: 3.8, // Angstroms
            amino_acid_types,
        }
    }
    
    // Convert angles to 3D coordinates
    fn angles_to_coordinates(&self, angles: &ArrayView1<f64>) -> Vec<[f64; 3]> {
        let mut coords = vec![[0.0; 3]; self.n_residues];
        
        // First residue at origin
        coords[0] = [0.0, 0.0, 0.0];
        
        if self.n_residues > 1 {
            // Second residue along x-axis
            coords[1] = [self.bond_length, 0.0, 0.0];
        }
        
        // Build the rest using spherical coordinates
        for i in 2..self.n_residues {
            let phi = angles[2 * (i - 2)];     // Dihedral angle
            let theta = angles[2 * (i - 2) + 1]; // Bond angle
            
            // Previous two residues
            let p1 = coords[i - 1];
            let p2 = coords[i - 2];
            
            // Direction vector from p2 to p1
            let v = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]];
            let v_len = (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt();
            let v_norm = [v[0] / v_len, v[1] / v_len, v[2] / v_len];
            
            // New position using spherical coordinates
            let x = p1[0] + self.bond_length * (theta.cos() * v_norm[0] - 
                                               theta.sin() * phi.cos() * v_norm[1] - 
                                               theta.sin() * phi.sin() * v_norm[2]);
            let y = p1[1] + self.bond_length * (theta.cos() * v_norm[1] + 
                                               theta.sin() * phi.cos() * v_norm[0]);
            let z = p1[2] + self.bond_length * (theta.cos() * v_norm[2] + 
                                               theta.sin() * phi.sin() * v_norm[0]);
            
            coords[i] = [x, y, z];
        }
        
        coords
    }
    
    // Simplified energy function
    fn energy(&self, angles: &ArrayView1<f64>) -> f64 {
        let coords = self.angles_to_coordinates(angles);
        let mut total_energy = 0.0;
        
        // 1. Bond length energy (harmonic potential)
        for i in 1..self.n_residues {
            let dist = self.distance(&coords[i - 1], &coords[i]);
            let bond_energy = 100.0 * (dist - self.bond_length).powi(2);
            total_energy += bond_energy;
        }
        
        // 2. Lennard-Jones potential for non-bonded interactions
        for i in 0..self.n_residues {
            for j in i + 2..self.n_residues { // Skip adjacent residues
                let dist = self.distance(&coords[i], &coords[j]);
                let sigma = self.get_interaction_distance(i, j);
                let epsilon = self.get_interaction_strength(i, j);
                
                if dist > 0.1 { // Avoid division by zero
                    let r6 = (sigma / dist).powi(6);
                    let lj_energy = 4.0 * epsilon * (r6.powi(2) - r6);
                    total_energy += lj_energy;
                }
            }
        }
        
        // 3. Electrostatic interactions (simplified)
        for i in 0..self.n_residues {
            for j in i + 1..self.n_residues {
                let dist = self.distance(&coords[i], &coords[j]);
                let charge_i = self.get_charge(i);
                let charge_j = self.get_charge(j);
                
                if dist > 0.1 {
                    let electrostatic = 332.0 * charge_i * charge_j / dist; // kcal/mol
                    total_energy += electrostatic;
                }
            }
        }
        
        // 4. Secondary structure preferences
        for i in 0..self.n_residues.saturating_sub(3) {
            let alpha_helix_penalty = self.alpha_helix_penalty(&angles, i);
            let beta_sheet_penalty = self.beta_sheet_penalty(&coords, i);
            total_energy += alpha_helix_penalty + beta_sheet_penalty;
        }
        
        total_energy
    }
    
    fn distance(&self, p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt()
    }
    
    fn get_interaction_distance(&self, i: usize, j: usize) -> f64 {
        // Simplified: based on amino acid types
        let type_i = self.amino_acid_types[i];
        let type_j = self.amino_acid_types[j];
        3.5 + 0.1 * (type_i as f64 + type_j as f64) / 40.0 // 3.5-4.0 Angstroms
    }
    
    fn get_interaction_strength(&self, i: usize, j: usize) -> f64 {
        // Simplified: hydrophobic interactions
        let hydrophobic = [0, 5, 8, 11, 15, 17, 18, 19]; // Indices of hydrophobic amino acids
        let type_i = self.amino_acid_types[i];
        let type_j = self.amino_acid_types[j];
        
        let is_hydrophobic_i = hydrophobic.contains(&type_i);
        let is_hydrophobic_j = hydrophobic.contains(&type_j);
        
        match (is_hydrophobic_i, is_hydrophobic_j) {
            (true, true) => 1.0,   // Strong hydrophobic interaction
            (true, false) | (false, true) => 0.5, // Weak interaction
            (false, false) => 0.2, // Hydrophilic interaction
        }
    }
    
    fn get_charge(&self, i: usize) -> f64 {
        // Simplified charges for amino acids
        match self.amino_acid_types[i] {
            3 => -1.0,  // D (Asp)
            4 => -1.0,  // E (Glu)
            9 => 1.0,   // K (Lys)
            14 => 1.0,  // R (Arg)
            6 => 0.5,   // H (His)
            _ => 0.0,   // Neutral
        }
    }
    
    fn alpha_helix_penalty(&self, angles: &ArrayView1<f64>, start_idx: usize) -> f64 {
        // Ideal alpha helix: phi ≈ -60°, psi ≈ -45°
        let ideal_phi = -60.0 * PI / 180.0;
        let ideal_psi = -45.0 * PI / 180.0;
        
        let mut penalty = 0.0;
        for i in 0..4.min(self.n_residues - start_idx - 2) {
            let idx = start_idx + i;
            if 2 * idx + 1 < angles.len() {
                let phi = angles[2 * idx];
                let psi = angles[2 * idx + 1];
                
                let phi_diff = (phi - ideal_phi).abs();
                let psi_diff = (psi - ideal_psi).abs();
                
                penalty += 0.1 * (phi_diff.powi(2) + psi_diff.powi(2));
            }
        }
        penalty
    }
    
    fn beta_sheet_penalty(&self, coords: &[[f64; 3]], start_idx: usize) -> f64 {
        // Simplified: penalize extended conformations
        if start_idx + 2 >= coords.len() {
            return 0.0;
        }
        
        let d1 = self.distance(&coords[start_idx], &coords[start_idx + 1]);
        let d2 = self.distance(&coords[start_idx + 1], &coords[start_idx + 2]);
        let d3 = self.distance(&coords[start_idx], &coords[start_idx + 2]);
        
        // Beta sheet preference: extended conformation
        let extension = d3 / (d1 + d2);
        if extension > 0.8 {
            -0.5 // Reward extended conformations
        } else {
            0.0
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example protein sequence (simplified)
    let sequence = "MKKLKKLLKKLLKK"; // Alternating hydrophobic/hydrophilic
    let folder = ProteinFolder::new(sequence);
    
    println!("Protein Folding Optimization");
    println!("Sequence: {}", sequence);
    println!("Length: {} residues", folder.n_residues);
    
    // Two angles per residue (except first two): phi and psi
    let n_angles = 2 * (folder.n_residues - 2);
    println!("Degrees of freedom: {} angles", n_angles);
    
    // Energy function
    let energy_func = |angles: &ArrayView1<f64>| folder.energy(angles);
    
    // Bounds: angles in [-π, π]
    let bounds: Vec<(f64, f64)> = (0..n_angles).map(|_| (-PI, PI)).collect();
    
    println!("\nOptimizing protein conformation...");
    
    // Try different global optimization methods
    let methods = vec![
        ("Differential Evolution", "DE"),
        ("Simulated Annealing", "SA"),
        ("Basin-hopping", "BH"),
    ];
    
    let mut best_energy = f64::INFINITY;
    let mut best_conformation = Array1::zeros(n_angles);
    
    for (method_name, method_code) in methods {
        println!("\n{} optimization:", method_name);
        let start_time = std::time::Instant::now();
        
        let result = match method_code {
            "DE" => {
                let options = DifferentialEvolutionOptions {
                    population_size: 50,
                    max_iter: 1000,
                    f: 0.8,
                    cr: 0.9,
                    tol: 1e-6,
                    ..Default::default()
                };
                differential_evolution(energy_func, &bounds, Some(options))?
            }
            "SA" => {
                let options = SimulatedAnnealingOptions {
                    max_iter: 10000,
                    initial_temp: 1000.0,
                    cooling_rate: 0.95,
                    min_temp: 1e-3,
                    ..Default::default()
                };
                simulated_annealing(energy_func, &bounds, Some(options))?
            }
            "BH" => {
                // Start from random point
                use rand::prelude::*;
                let mut rng = thread_rng();
                let x0: Array1<f64> = (0..n_angles)
                    .map(|_| rng.gen_range(-PI..PI))
                    .collect();
                
                let options = BasinHoppingOptions {
                    n_iter: 100,
                    temperature: 10.0,
                    step_size: 0.5,
                    ..Default::default()
                };
                basinhopping(energy_func, &x0, Some(options))?
            }
            _ => unreachable!(),
        };
        
        let optimization_time = start_time.elapsed();
        
        println!("  Final energy: {:.2} kcal/mol", result.fun);
        println!("  Iterations: {}", result.nit);
        println!("  Time: {:.2?}", optimization_time);
        println!("  Success: {}", result.success);
        
        if result.fun < best_energy {
            best_energy = result.fun;
            best_conformation = result.x.clone();
            println!("  *** New best conformation! ***");
        }
    }
    
    println!("\nBest conformation analysis:");
    println!("  Final energy: {:.2} kcal/mol", best_energy);
    
    // Analyze the best structure
    let coords = folder.angles_to_coordinates(&best_conformation.view());
    
    // Radius of gyration
    let center = coords.iter().fold([0.0, 0.0, 0.0], |acc, coord| {
        [acc[0] + coord[0], acc[1] + coord[1], acc[2] + coord[2]]
    });
    let center = [
        center[0] / coords.len() as f64,
        center[1] / coords.len() as f64,
        center[2] / coords.len() as f64,
    ];
    
    let rg_squared = coords.iter()
        .map(|coord| {
            (coord[0] - center[0]).powi(2) + 
            (coord[1] - center[1]).powi(2) + 
            (coord[2] - center[2]).powi(2)
        })
        .sum::<f64>() / coords.len() as f64;
    
    let radius_of_gyration = rg_squared.sqrt();
    
    println!("  Radius of gyration: {:.2} Å", radius_of_gyration);
    
    // End-to-end distance
    let end_to_end = folder.distance(&coords[0], &coords[coords.len() - 1]);
    println!("  End-to-end distance: {:.2} Å", end_to_end);
    
    // Compactness
    let max_extent = coords.iter()
        .flat_map(|coord| coord.iter())
        .fold(0.0, |acc, &x| acc.max(x.abs()));
    println!("  Maximum extent: {:.2} Å", max_extent);
    
    // Secondary structure analysis (simplified)
    println!("\nSecondary structure angles (degrees):");
    for i in 0..5.min(n_angles / 2) {
        let phi = best_conformation[2 * i] * 180.0 / PI;
        let psi = best_conformation[2 * i + 1] * 180.0 / PI;
        println!("  Residue {}: φ = {:.1}°, ψ = {:.1}°", i + 3, phi, psi);
    }
    
    Ok(())
}
```

This comprehensive examples document demonstrates the versatility and power of the `scirs2-optimize` library across various domains. Each example includes complete, runnable code with detailed explanations of the problem setup, optimization approach, and result analysis.