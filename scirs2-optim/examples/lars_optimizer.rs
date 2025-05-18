// Example demonstrating the LARS optimizer
use ndarray::{s, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scirs2_optim::optimizers::{Adam, LARS, SGD};
use scirs2_optim::Optimizer;
use std::error::Error;
use std::time::Instant;

// Simple loss function for demonstration: linear regression with L2 loss
fn compute_loss_and_gradient(
    weights: &Array1<f64>,
    bias: &f64,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> (f64, Array1<f64>, f64) {
    // Predictions: X * w + b
    let preds = x.dot(weights) + *bias;

    // Compute difference
    let diff = &preds - y;

    // L2 loss
    let loss = diff.mapv(|v| v * v).mean().unwrap();

    // Gradient for weights: 2/n * X^T * (X*w + b - y)
    let n = x.nrows() as f64;
    let grad_w = x.t().dot(&diff) * (2.0 / n);

    // Gradient for bias: 2/n * sum(X*w + b - y)
    let grad_b = diff.sum() * (2.0 / n);

    (loss, grad_w, grad_b)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Create a synthetic dataset for linear regression
    let n_samples = 10000;
    let n_features = 100;

    // Create random data
    let x = Array2::random((n_samples, n_features), Uniform::new(-1.0, 1.0));
    let true_weights = Array1::random(n_features, Uniform::new(-1.0, 1.0));
    let true_bias = 0.5;

    // Create target values with some noise
    let noise = Array1::random(n_samples, Uniform::new(-0.1, 0.1));
    let y = x.dot(&true_weights) + true_bias + noise;

    // Split data into batches
    let batch_size = 1000; // Large batch for LARS
    let n_batches = n_samples / batch_size;
    let batches: Vec<_> = (0..n_batches)
        .map(|i| {
            let start = i * batch_size;
            let end = start + batch_size;
            let x_batch = x.slice(s![start..end, ..]).to_owned();
            let y_batch = y.slice(s![start..end]).to_owned();
            (x_batch, y_batch)
        })
        .collect();

    println!("Comparing optimizers on linear regression task");
    println!("==============================================");
    println!("Dataset: {} samples, {} features", n_samples, n_features);
    println!("Batch size: {} (Large batch training)", batch_size);
    println!();

    // Initialize optimizers
    let mut optimizers: Vec<(_, &str, f64)> = vec![
        (
            Box::new(SGD::new(0.01)) as Box<dyn Optimizer<f64, _>>,
            "SGD",
            0.01,
        ),
        (
            Box::new(Adam::new(0.01)) as Box<dyn Optimizer<f64, _>>,
            "Adam",
            0.01,
        ),
        (
            Box::new(LARS::new(0.01).with_trust_coefficient(0.001)) as Box<dyn Optimizer<f64, _>>,
            "LARS",
            0.01,
        ),
    ];

    // Train with each optimizer
    for (optimizer, name, lr) in optimizers.iter_mut() {
        println!("Training with {}, lr={}", name, lr);

        // Initialize parameters
        let mut weights = Array1::zeros(n_features);
        let mut bias = 0.0;

        // Start timing
        let start_time = Instant::now();

        // Train for some epochs
        let epochs = 10;
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for (batch_x, batch_y) in &batches {
                // Compute loss and gradients
                let (loss, grad_w, grad_b) =
                    compute_loss_and_gradient(&weights, &bias, batch_x, batch_y);
                epoch_loss += loss;

                // Update weights with optimizer
                let new_weights = optimizer.step(&weights, &grad_w)?;
                weights = new_weights;

                // Update bias (simple update, not using LARS for scalar)
                bias -= *lr * grad_b;
            }

            epoch_loss /= n_batches as f64;

            // Print progress
            if epoch % 2 == 0 || epoch == epochs - 1 {
                println!("  Epoch {}: Loss = {:.6}", epoch, epoch_loss);
            }
        }

        let duration = start_time.elapsed();

        // Final evaluation
        let (final_loss, _, _) = compute_loss_and_gradient(&weights, &bias, &x, &y);

        println!("  Training completed in {:?}", duration);
        println!("  Final loss: {:.6}", final_loss);

        // Calculate error compared to true weights
        let weight_error = (&weights - &true_weights).mapv(|v| v.abs()).mean().unwrap();
        let bias_error = (bias - true_bias).abs();

        println!("  Weight error: {:.6}", weight_error);
        println!("  Bias error: {:.6}", bias_error);
        println!();
    }

    println!("Conclusion:");
    println!("==========");
    println!("For large batch training, LARS is designed to maintain convergence by");
    println!("adapting the learning rate per layer based on the weight and gradient norms.");
    println!("This makes LARS particularly effective for training with very large batch sizes");
    println!("where traditional optimizers like SGD and Adam often struggle.");

    Ok(())
}
