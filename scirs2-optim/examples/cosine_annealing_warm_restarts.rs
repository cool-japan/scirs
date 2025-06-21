//! Example demonstrating the Cosine Annealing with Warm Restarts scheduler
//!
//! This example shows how to use the SGDR (Stochastic Gradient Descent with Warm Restarts)
//! scheduler and compares it with standard cosine annealing.

use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use scirs2_optim::optimizers::{Optimizer, SGD};
use scirs2_optim::schedulers::{
    CosineAnnealing, CosineAnnealingWarmRestarts, LearningRateScheduler,
};
use std::time::Instant;

fn main() {
    println!("Cosine Annealing with Warm Restarts Example");
    println!("===========================================\n");

    // Generate synthetic regression data
    let n_samples = 100;
    let n_features = 5;

    println!(
        "Generating synthetic linear regression data with {} samples and {} features",
        n_samples, n_features
    );

    let (x_train, y_train, true_weights, _true_bias) = generate_data(n_samples, n_features);

    // Parameters for optimization
    let initial_lr = 0.1;
    let min_lr = 0.001;
    let n_iterations = 300;

    println!("\nComparing learning rate schedules:");
    println!("  - Standard Cosine Annealing");
    println!("  - Cosine Annealing with Warm Restarts (SGDR)");
    println!();

    // Train with standard cosine annealing
    println!("Training with standard Cosine Annealing:");
    let mut standard_cosine_scheduler =
        CosineAnnealing::new(initial_lr, min_lr, n_iterations, false);

    let start_time = Instant::now();
    let standard_result = train_linear_regression(
        &x_train,
        &y_train,
        &true_weights,
        n_features,
        n_iterations,
        &mut standard_cosine_scheduler,
    );
    let standard_time = start_time.elapsed();

    // Train with cosine annealing with warm restarts
    println!("\nTraining with Cosine Annealing with Warm Restarts (SGDR):");
    // Use 3 cycles with increasing length
    let t_0 = 50; // Initial cycle length
    let t_mult = 2.0; // Multiplicative factor
    let mut sgdr_scheduler = CosineAnnealingWarmRestarts::new(initial_lr, min_lr, t_0, t_mult);

    let start_time = Instant::now();
    let sgdr_result = train_linear_regression(
        &x_train,
        &y_train,
        &true_weights,
        n_features,
        n_iterations,
        &mut sgdr_scheduler,
    );
    let sgdr_time = start_time.elapsed();

    // Print results
    println!("\nResults:");
    println!("  Standard Cosine Annealing:");
    println!("    - Training time: {:?}", standard_time);
    println!("    - Final loss: {:.6}", standard_result.0);
    println!("    - Weight error: {:.6}", standard_result.1);

    println!("  Cosine Annealing with Warm Restarts (SGDR):");
    println!("    - Training time: {:?}", sgdr_time);
    println!("    - Final loss: {:.6}", sgdr_result.0);
    println!("    - Weight error: {:.6}", sgdr_result.1);

    // Visualization information
    println!("\nVisualization of learning rate schedules:");
    println!("Standard Cosine Annealing gradually decreases the learning rate over the entire");
    println!("training period, which can lead to getting stuck in local minima.");
    println!();
    println!("Cosine Annealing with Warm Restarts (SGDR) periodically resets the learning rate");
    println!("to allow for exploration of different regions of the loss landscape, potentially");
    println!("finding better solutions. It also increases the cycle length after each restart,");
    println!("spending more time in promising regions as training progresses.");
    println!();
    println!("For this run, SGDR used:");
    println!("  - Initial cycle length: {}", t_0);
    println!("  - Multiplicative factor: {}", t_mult);
    println!(
        "  - Resulting in cycle lengths: {}, {}, {}, ...",
        t_0,
        (t_0 as f64 * t_mult).round() as usize,
        (t_0 as f64 * t_mult * t_mult).round() as usize
    );
}

/// Train linear regression with a given learning rate scheduler
fn train_linear_regression<S: LearningRateScheduler<f64>>(
    x_train: &Array2<f64>,
    y_train: &Array1<f64>,
    true_weights: &Array1<f64>,
    n_features: usize,
    n_iterations: usize,
    scheduler: &mut S,
) -> (f64, f64) {
    // Initialize SGD optimizer
    let mut optimizer = SGD::<f64>::new(scheduler.get_learning_rate());

    // Initialize parameters
    let mut weights = Array1::<f64>::zeros(n_features);
    let mut bias = 0.0;

    // Training loop
    for i in 0..n_iterations {
        // Update learning rate from scheduler
        let lr = scheduler.step();
        <SGD<f64> as Optimizer<f64, ndarray::Ix1>>::set_learning_rate(&mut optimizer, lr);

        // Forward pass
        let predictions = &x_train.dot(&weights) + bias;

        // Compute error
        let error = predictions - y_train;

        // Compute loss (mean squared error)
        let loss = (&error * &error).sum() / (2.0 * error.len() as f64);

        // Compute gradients
        let weight_grad = x_train.t().dot(&error) / (y_train.len() as f64);
        let bias_grad = error.sum() / (y_train.len() as f64);

        // Update parameters
        weights = optimizer.step(&weights, &weight_grad).unwrap();
        bias -= lr * bias_grad;

        // Print progress at intervals or warm restart points
        if i == 0
            || i == n_iterations - 1
            || (i + 1) % 50 == 0
            || (scheduler.get_learning_rate() - lr).abs() > 0.01
        {
            // Detect warm restart
            println!("  Iteration {}: loss = {:.6}, lr = {:.6}", i + 1, loss, lr);
        }
    }

    // Compute final predictions
    let predictions = &x_train.dot(&weights) + bias;

    // Compute final loss
    let error = predictions - y_train;
    let final_loss = (&error * &error).sum() / (2.0 * error.len() as f64);

    // Compute weight error (L2 distance from true weights)
    let weight_diff = &weights - true_weights;
    let weight_error = (&weight_diff * &weight_diff).sum().sqrt();

    (final_loss, weight_error)
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
