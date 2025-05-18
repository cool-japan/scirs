//! Example demonstrating the SparseAdam optimizer
//!
//! This example compares the SparseAdam optimizer with the standard Adam
//! optimizer on a sparse linear regression task, showing how SparseAdam
//! can be more efficient when dealing with sparse gradients.

use ndarray::Array1;
use rand;
use scirs2_optim::optimizers::{Adam, Optimizer, SparseAdam, SparseGradient};
use std::time::Instant;

fn main() {
    println!("SparseAdam Optimizer Example");
    println!("===========================\n");

    // Configuration parameters
    let n_features = 10000; // Large feature space (e.g., vocabulary size)
    let n_samples = 100; // Number of training samples
    let n_active_features = 20; // Number of non-zero features per sample (sparse)
    let lr = 0.01; // Learning rate
    let n_iterations = 50; // Number of training iterations

    println!("Sparse Linear Regression Task:");
    println!("  - {} features total", n_features);
    println!("  - {} training samples", n_samples);
    println!(
        "  - Only {}/{} features are non-zero in each sample ({:.2}% sparsity)",
        n_active_features,
        n_features,
        100.0 * (1.0 - n_active_features as f64 / n_features as f64)
    );
    println!();

    // Generate sparse data for testing
    let (x_train, y_train, true_weights) =
        generate_sparse_data(n_samples, n_features, n_active_features);

    // 1. Train with standard Adam
    println!("Training with standard Adam optimizer:");
    let mut adam = Adam::new(lr);
    let start_time = Instant::now();
    let adam_result = train_with_adam(&x_train, &y_train, &true_weights, n_iterations, &mut adam);
    let adam_time = start_time.elapsed();

    // 2. Train with SparseAdam
    println!("\nTraining with SparseAdam optimizer:");
    let mut sparse_adam = SparseAdam::new(lr);
    let start_time = Instant::now();
    let sparse_adam_result = train_with_sparse_adam(
        &x_train,
        &y_train,
        &true_weights,
        n_iterations,
        &mut sparse_adam,
    );
    let sparse_adam_time = start_time.elapsed();

    // Results
    println!("\nResults Comparison:");
    println!("  Standard Adam:");
    println!("    - Time: {:?}", adam_time);
    println!("    - Final Loss: {:.6}", adam_result.0);
    println!("    - Parameter Error: {:.6}", adam_result.1);

    println!("  SparseAdam:");
    println!("    - Time: {:?}", sparse_adam_time);
    println!("    - Final Loss: {:.6}", sparse_adam_result.0);
    println!("    - Parameter Error: {:.6}", sparse_adam_result.1);

    // Performance comparison
    let speedup = adam_time.as_secs_f64() / sparse_adam_time.as_secs_f64();
    println!("\nPerformance Comparison:");
    println!(
        "  - SparseAdam is {:.2}x faster than standard Adam",
        speedup
    );
    println!(
        "  - Loss difference: {:.6}",
        (adam_result.0 - sparse_adam_result.0).abs()
    );

    println!("\nConclusion:");
    println!("  SparseAdam provides similar optimization quality to standard Adam,");
    println!("  but can be significantly faster when dealing with sparse gradients.");
    println!("  The speedup is especially noticeable with very large, sparse feature spaces.");
}

/// Generate sparse data for testing
///
/// Returns:
/// - x_train: Training inputs with sparse features
/// - y_train: Training targets
/// - true_weights: The true weights used to generate the data
fn generate_sparse_data(
    n_samples: usize,
    n_features: usize,
    n_active_features: usize,
) -> (Vec<SparseInput>, Array1<f64>, Array1<f64>) {
    // Generate true weights (only a small fraction are non-zero)
    let mut true_weights = Array1::zeros(n_features);
    let active_indices: Vec<usize> = (0..n_features)
        .step_by(n_features / n_active_features)  // Evenly spaced active features
        .take(n_active_features)
        .collect();

    for &idx in &active_indices {
        true_weights[idx] = rand_value(-1.0, 1.0);
    }

    // Generate sparse inputs and compute targets
    let mut x_train = Vec::with_capacity(n_samples);
    let mut y_train = Array1::zeros(n_samples);

    for i in 0..n_samples {
        // For each sample, randomly select n_active_features
        let mut indices = Vec::with_capacity(n_active_features);
        let mut values = Vec::with_capacity(n_active_features);

        // Choose random features (including some that are in the true model)
        let mut target_sum = 0.0;

        // Always include some of the truly active features
        let n_true_active = n_active_features / 2;
        for j in 0..n_true_active {
            let idx = active_indices[j % active_indices.len()];
            indices.push(idx);
            let value = rand_value(-1.0, 1.0);
            values.push(value);
            target_sum += value * true_weights[idx];
        }

        // Add some random other features
        for _ in n_true_active..n_active_features {
            let idx = (rand_value(0.0, 1.0) * n_features as f64) as usize % n_features;
            if !indices.contains(&idx) {
                indices.push(idx);
                let value = rand_value(-1.0, 1.0);
                values.push(value);
                target_sum += value * true_weights[idx];
            }
        }

        // Add noise to the target
        y_train[i] = target_sum + rand_value(-0.1, 0.1);

        // Store the sparse input
        x_train.push(SparseInput {
            indices,
            values,
            dim: n_features,
        });
    }

    (x_train, y_train, true_weights)
}

/// A struct representing a sparse input with non-zero features
#[allow(dead_code)]
struct SparseInput {
    indices: Vec<usize>,
    values: Vec<f64>,
    #[allow(dead_code)]
    dim: usize,
}

impl SparseInput {
    /// Compute dot product with weights
    fn dot(&self, weights: &Array1<f64>) -> f64 {
        self.indices
            .iter()
            .zip(&self.values)
            .map(|(&idx, &val)| val * weights[idx])
            .sum()
    }

    /// Compute gradient for the given error and add to existing gradients
    fn accumulate_gradient(&self, error: f64, gradients: &mut Array1<f64>) {
        for (&idx, &val) in self.indices.iter().zip(&self.values) {
            gradients[idx] += error * val;
        }
    }

    /// Compute gradient as a SparseGradient
    #[allow(dead_code)]
    fn compute_sparse_gradient(&self, error: f64) -> SparseGradient<f64> {
        let values: Vec<f64> = self.values.iter().map(|&val| error * val).collect();

        SparseGradient::new(self.indices.clone(), values, self.dim)
    }
}

/// Train model using standard Adam optimizer
fn train_with_adam(
    x_train: &[SparseInput],
    y_train: &Array1<f64>,
    true_weights: &Array1<f64>,
    n_iterations: usize,
    optimizer: &mut Adam<f64>,
) -> (f64, f64) {
    let n_samples = x_train.len();
    let n_features = true_weights.len();

    // Initialize weights to zero
    let mut weights = Array1::zeros(n_features);

    for iter in 0..n_iterations {
        // Forward pass - compute predictions and loss
        let mut loss = 0.0;
        for (i, x) in x_train.iter().enumerate() {
            let pred = x.dot(&weights);
            let error = pred - y_train[i];
            loss += error * error;
        }
        loss /= n_samples as f64;

        // Compute gradients (accumulated over all samples)
        let mut gradients = Array1::zeros(n_features);
        for (i, x) in x_train.iter().enumerate() {
            let pred = x.dot(&weights);
            let error = 2.0 * (pred - y_train[i]) / n_samples as f64;
            x.accumulate_gradient(error, &mut gradients);
        }

        // Update weights using Adam
        weights = optimizer.step(&weights, &gradients).unwrap();

        // Calculate parameter error (L2 norm of difference)
        let param_error = (&weights - true_weights).map(|x| x * x).sum().sqrt();

        // Print progress
        if iter == 0 || iter == n_iterations - 1 || (iter + 1) % 10 == 0 {
            println!(
                "  Iteration {}/{}: loss = {:.6}, param error = {:.6}",
                iter + 1,
                n_iterations,
                loss,
                param_error
            );
        }
    }

    // Final evaluation
    let mut final_loss = 0.0;
    for (i, x) in x_train.iter().enumerate() {
        let pred = x.dot(&weights);
        let error = pred - y_train[i];
        final_loss += error * error;
    }
    final_loss /= n_samples as f64;

    let param_error = (&weights - true_weights).map(|x| x * x).sum().sqrt();

    (final_loss, param_error)
}

/// Train model using SparseAdam optimizer
fn train_with_sparse_adam(
    x_train: &[SparseInput],
    y_train: &Array1<f64>,
    true_weights: &Array1<f64>,
    n_iterations: usize,
    optimizer: &mut SparseAdam<f64>,
) -> (f64, f64) {
    let n_samples = x_train.len();
    let n_features = true_weights.len();

    // Initialize weights to zero
    let mut weights = Array1::zeros(n_features);

    for iter in 0..n_iterations {
        // Forward pass - compute predictions and loss
        let mut loss = 0.0;
        for (i, x) in x_train.iter().enumerate() {
            let pred = x.dot(&weights);
            let error = pred - y_train[i];
            loss += error * error;
        }
        loss /= n_samples as f64;

        // For SparseAdam, we need to compute a single sparse gradient
        // We'll combine all sample gradients into one sparse gradient
        let mut all_indices = Vec::new();
        let mut all_values = Vec::new();
        let mut index_map = std::collections::HashMap::new();

        // Gather gradients from all samples
        for (i, x) in x_train.iter().enumerate() {
            let pred = x.dot(&weights);
            let error = 2.0 * (pred - y_train[i]) / n_samples as f64;

            // Add each feature's gradient contribution
            for (&idx, &val) in x.indices.iter().zip(&x.values) {
                let grad_val = error * val;

                if let Some(pos) = index_map.get(&idx) {
                    // Add to existing gradient value
                    all_values[*pos] += grad_val;
                } else {
                    // Add new gradient value
                    index_map.insert(idx, all_values.len());
                    all_indices.push(idx);
                    all_values.push(grad_val);
                }
            }
        }

        // Create sparse gradient
        let sparse_grad = SparseGradient::new(all_indices, all_values, n_features);

        // Update weights using SparseAdam
        weights = optimizer.step_sparse(&weights, &sparse_grad).unwrap();

        // Calculate parameter error (L2 norm of difference)
        let param_error = (&weights - true_weights).map(|x| x * x).sum().sqrt();

        // Print progress
        if iter == 0 || iter == n_iterations - 1 || (iter + 1) % 10 == 0 {
            println!(
                "  Iteration {}/{}: loss = {:.6}, param error = {:.6}",
                iter + 1,
                n_iterations,
                loss,
                param_error
            );
        }
    }

    // Final evaluation
    let mut final_loss = 0.0;
    for (i, x) in x_train.iter().enumerate() {
        let pred = x.dot(&weights);
        let error = pred - y_train[i];
        final_loss += error * error;
    }
    final_loss /= n_samples as f64;

    let param_error = (&weights - true_weights).map(|x| x * x).sum().sqrt();

    (final_loss, param_error)
}

/// Generate a random value in the given range
fn rand_value(min: f64, max: f64) -> f64 {
    min + (max - min) * rand::random::<f64>()
}
