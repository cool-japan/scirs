//! Example demonstrating the optimizer composition framework
//!
//! This example shows how to use the three types of optimizer compositions:
//! 1. Sequential: Apply multiple optimizers in sequence
//! 2. Parallel: Apply different optimizers to different parameter groups
//! 3. Chained: Wrap one optimizer with another (like Lookahead)

use ndarray::{s, Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use scirs2_optim::optimizer_composition::{
    ChainedOptimizer, ParallelOptimizer, ParameterGroup, SequentialOptimizer,
};
use scirs2_optim::optimizers::{Adam, Optimizer, RMSprop, SGD};
use std::time::Instant;

fn main() {
    println!("Optimizer Composition Framework Example");
    println!("======================================\n");

    // Generate synthetic regression data
    let n_samples = 100;
    let n_features = 10;

    println!(
        "Generating synthetic linear regression data with {} samples and {} features",
        n_samples, n_features
    );

    let (x_train, y_train, true_weights, _true_bias) = generate_data(n_samples, n_features);

    // Define our optimization problem
    println!("\nRunning regression with different optimizer compositions...\n");

    // 1. Sequential Optimizer Example
    sequential_optimizer_example(&x_train, &y_train, true_weights.len());

    // 2. Parallel Optimizer Example
    parallel_optimizer_example(&x_train, &y_train, true_weights.len());

    // 3. Chained Optimizer Example
    chained_optimizer_example(&x_train, &y_train, true_weights.len());
}

/// Example demonstrating a sequential optimizer composition
fn sequential_optimizer_example(x_train: &Array2<f64>, y_train: &Array1<f64>, n_features: usize) {
    println!("Sequential Optimizer Example");
    println!("----------------------------");
    println!(
        "Applying SGD followed by Adam - this can be useful for coarse-to-fine optimization\n"
    );

    // Initialize parameters
    let mut weights = Array1::<f64>::zeros(n_features);
    let mut bias = 0.0;

    // Create a sequential optimizer with SGD followed by Adam
    let sgd = SGD::new_with_config(0.1, 0.0, 0.0); // No momentum or weight decay
    let adam = Adam::new(0.01);

    let mut sequential_optimizer = SequentialOptimizer::new(vec![Box::new(sgd), Box::new(adam)]);

    // Train the model with the sequential optimizer
    let n_iterations = 100;
    let start_time = Instant::now();

    for i in 0..n_iterations {
        // Forward pass (compute predictions)
        let predictions = compute_predictions(x_train, &weights, bias);

        // Compute loss and gradients
        let (loss, weight_grad, bias_grad) =
            compute_gradients(x_train, y_train, &predictions, &weights, bias);

        // Update weights using the sequential optimizer
        weights = sequential_optimizer.step(&weights, &weight_grad).unwrap();

        // Update bias using simple SGD
        bias -= 0.01 * bias_grad;

        // Print progress
        if i == 0 || i == n_iterations - 1 || (i + 1) % 25 == 0 {
            println!("  Iteration {}: loss = {:.6}", i + 1, loss);
        }
    }

    let elapsed = start_time.elapsed();

    // Compute final predictions and loss
    let predictions = compute_predictions(x_train, &weights, bias);
    let (final_loss, _, _) = compute_gradients(x_train, y_train, &predictions, &weights, bias);

    println!("\nResults:");
    println!("  Training time: {:?}", elapsed);
    println!("  Final loss: {:.6}", final_loss);
    println!("  Weight norm: {:.6}", weights.mapv(|w| w * w).sum().sqrt());
    println!();
}

/// Example demonstrating a parallel optimizer composition
fn parallel_optimizer_example(x_train: &Array2<f64>, y_train: &Array1<f64>, n_features: usize) {
    println!("Parallel Optimizer Example");
    println!("---------------------------");
    println!("Using different optimizers for different parameter groups\n");

    // Split the weights into two groups for demonstration
    let split_point = n_features / 2;

    // Initialize parameters
    let weights_group1 = Array1::<f64>::zeros(split_point);
    let weights_group2 = Array1::<f64>::zeros(n_features - split_point);
    let mut bias = 0.0;

    // Create a parallel optimizer with SGD for group 1 and Adam for group 2
    let sgd = SGD::new(0.1);
    let adam = Adam::new(0.01);

    let group1 = ParameterGroup::new(weights_group1, 0); // Use SGD
    let group2 = ParameterGroup::new(weights_group2, 1); // Use Adam

    let mut parallel_optimizer =
        ParallelOptimizer::new(vec![Box::new(sgd), Box::new(adam)], vec![group1, group2]);

    // Train the model with the parallel optimizer
    let n_iterations = 100;
    let start_time = Instant::now();

    for i in 0..n_iterations {
        // Get current weights from the optimizer
        let current_weights = parallel_optimizer.get_all_parameters().unwrap();

        // Combine the weights for predictions
        let mut combined_weights = Array1::<f64>::zeros(n_features);
        for j in 0..split_point {
            combined_weights[j] = current_weights[0][j];
        }
        for j in 0..(n_features - split_point) {
            combined_weights[j + split_point] = current_weights[1][j];
        }

        // Forward pass (compute predictions)
        let predictions = compute_predictions(x_train, &combined_weights, bias);

        // Compute loss and gradients
        let (loss, weight_grad, bias_grad) =
            compute_gradients(x_train, y_train, &predictions, &combined_weights, bias);

        // Split the gradients for the separate parameter groups
        let weight_grad_group1 = weight_grad.slice(s![0..split_point]).to_owned();
        let weight_grad_group2 = weight_grad.slice(s![split_point..]).to_owned();

        // Update weights using the parallel optimizer
        parallel_optimizer
            .update_all_parameters(&[weight_grad_group1, weight_grad_group2])
            .unwrap();

        // Update bias using simple SGD
        bias -= 0.01 * bias_grad;

        // Print progress
        if i == 0 || i == n_iterations - 1 || (i + 1) % 25 == 0 {
            println!("  Iteration {}: loss = {:.6}", i + 1, loss);
        }
    }

    let elapsed = start_time.elapsed();

    // Get final weights
    let final_weights = parallel_optimizer.get_all_parameters().unwrap();

    // Combine the weights for final predictions
    let mut combined_weights = Array1::<f64>::zeros(n_features);
    for j in 0..split_point {
        combined_weights[j] = final_weights[0][j];
    }
    for j in 0..(n_features - split_point) {
        combined_weights[j + split_point] = final_weights[1][j];
    }

    // Compute final predictions and loss
    let predictions = compute_predictions(x_train, &combined_weights, bias);
    let (final_loss, _, _) =
        compute_gradients(x_train, y_train, &predictions, &combined_weights, bias);

    println!("\nResults:");
    println!("  Training time: {:?}", elapsed);
    println!("  Final loss: {:.6}", final_loss);
    println!(
        "  Weight norm (group 1): {:.6}",
        final_weights[0].mapv(|w| w * w).sum().sqrt()
    );
    println!(
        "  Weight norm (group 2): {:.6}",
        final_weights[1].mapv(|w| w * w).sum().sqrt()
    );
    println!();
}

/// Example demonstrating a chained optimizer composition
fn chained_optimizer_example(x_train: &Array2<f64>, y_train: &Array1<f64>, n_features: usize) {
    println!("Chained Optimizer Example");
    println!("-------------------------");
    println!("Using RMSprop wrapped with Adam\n");

    // Initialize parameters
    let mut weights = Array1::<f64>::zeros(n_features);
    let mut bias = 0.0;

    // Create a chained optimizer with RMSprop as inner and Adam as outer
    let inner = RMSprop::new(0.01);
    let outer = Adam::new(0.001);

    let mut chained_optimizer = ChainedOptimizer::new(Box::new(inner), Box::new(outer));

    // Train the model with the chained optimizer
    let n_iterations = 100;
    let start_time = Instant::now();

    for i in 0..n_iterations {
        // Forward pass (compute predictions)
        let predictions = compute_predictions(x_train, &weights, bias);

        // Compute loss and gradients
        let (loss, weight_grad, bias_grad) =
            compute_gradients(x_train, y_train, &predictions, &weights, bias);

        // Update weights using the chained optimizer
        weights = chained_optimizer.step(&weights, &weight_grad).unwrap();

        // Update bias using simple SGD
        bias -= 0.01 * bias_grad;

        // Print progress
        if i == 0 || i == n_iterations - 1 || (i + 1) % 25 == 0 {
            println!("  Iteration {}: loss = {:.6}", i + 1, loss);
        }
    }

    let elapsed = start_time.elapsed();

    // Compute final predictions and loss
    let predictions = compute_predictions(x_train, &weights, bias);
    let (final_loss, _, _) = compute_gradients(x_train, y_train, &predictions, &weights, bias);

    println!("\nResults:");
    println!("  Training time: {:?}", elapsed);
    println!("  Final loss: {:.6}", final_loss);
    println!("  Weight norm: {:.6}", weights.mapv(|w| w * w).sum().sqrt());
    println!();
}

/// Generate synthetic regression data
fn generate_data(
    n_samples: usize,
    n_features: usize,
) -> (Array2<f64>, Array1<f64>, Array1<f64>, f64) {
    // Create random true weights and bias
    let true_weights = Array1::random(n_features, Normal::new(0.0, 1.0).unwrap());
    let true_bias = 1.0;

    // Generate random features
    let x = Array2::random((n_samples, n_features), Normal::new(0.0, 1.0).unwrap());

    // Generate target values with noise
    let y_without_noise = x.dot(&true_weights) + true_bias;
    let noise = Array1::random(n_samples, Normal::new(0.0, 0.1).unwrap());
    let y = &y_without_noise + &noise;

    (x, y, true_weights, true_bias)
}

/// Compute predictions for linear regression
fn compute_predictions(x: &Array2<f64>, weights: &Array1<f64>, bias: f64) -> Array1<f64> {
    &x.dot(weights) + bias
}

/// Compute loss and gradients for linear regression
fn compute_gradients(
    x: &Array2<f64>,
    y: &Array1<f64>,
    predictions: &Array1<f64>,
    _weights: &Array1<f64>,
    _bias: f64,
) -> (f64, Array1<f64>, f64) {
    // Compute the error
    let error = predictions - y;

    // Mean squared error loss
    let loss = (&error * &error).sum() / (2.0 * y.len() as f64);

    // Gradients with respect to weights and bias
    let weight_grad = x.t().dot(&error) / (y.len() as f64);
    let bias_grad = error.sum() / (y.len() as f64);

    (loss, weight_grad, bias_grad)
}
