// Example demonstrating the Lookahead optimizer
use ndarray::{s, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scirs2_optim::optimizers::{Adam, Lookahead, SGD};
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

// Train a model using the provided optimizer
fn train_model(
    optimizer: &mut dyn Optimizer<f64, ndarray::Ix1>,
    x: &Array2<f64>,
    y: &Array1<f64>,
    train_batches: &Vec<(Array2<f64>, Array1<f64>)>,
    lr: f64,
    epochs: usize,
) -> Result<(Array1<f64>, f64, f64), Box<dyn Error>> {
    // Initialize parameters
    let n_features = x.ncols();
    let mut weights = Array1::zeros(n_features);
    let mut bias = 0.0;

    // Start timing
    let start_time = Instant::now();

    // Train for given number of epochs
    let n_batches = train_batches.len();
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        for (batch_x, batch_y) in train_batches {
            // Compute loss and gradients
            let (loss, grad_w, grad_b) =
                compute_loss_and_gradient(&weights, &bias, batch_x, batch_y);
            epoch_loss += loss;

            // Update weights with optimizer
            weights = optimizer.step(&weights, &grad_w)?;

            // Update bias with simple SGD
            bias -= lr * grad_b;
        }

        epoch_loss /= n_batches as f64;

        // Print progress every few epochs
        if epoch % 5 == 0 || epoch == epochs - 1 {
            println!("  Epoch {}: Loss = {:.6}", epoch, epoch_loss);
        }
    }

    let _duration = start_time.elapsed();

    // Final evaluation
    let (final_loss, _, _) = compute_loss_and_gradient(&weights, &bias, x, y);

    Ok((weights, bias, final_loss))
}

fn main() -> Result<(), Box<dyn Error>> {
    // Create a synthetic dataset for linear regression
    let n_samples = 1000;
    let n_features = 20;

    println!("Comparing optimizers on noisy linear regression task");
    println!("====================================================");
    println!("Dataset: {} samples, {} features", n_samples, n_features);
    println!();

    // Create random data with noise
    let x = Array2::random((n_samples, n_features), Uniform::new(-1.0, 1.0));
    let true_weights = Array1::random(n_features, Uniform::new(-1.0, 1.0));
    let true_bias = 0.5;

    // Add noise to make optimization more challenging
    let noise = Array1::random(n_samples, Uniform::new(-0.3, 0.3)); // More noise
    let y = x.dot(&true_weights) + true_bias + noise;

    // Split data into batches
    let batch_size = 50;
    let n_batches = n_samples / batch_size;
    let train_batches: Vec<_> = (0..n_batches)
        .map(|i| {
            let start = i * batch_size;
            let end = start + batch_size;
            let x_batch = x.slice(s![start..end, ..]).to_owned();
            let y_batch = y.slice(s![start..end]).to_owned();
            (x_batch, y_batch)
        })
        .collect();

    // Learning rate for all optimizers
    let lr = 0.01;
    let epochs = 30;

    // 1. Train with SGD
    println!("Training with SGD:");
    let mut sgd = SGD::new(lr).with_momentum(0.9).with_weight_decay(0.0001);
    let (sgd_weights, sgd_bias, sgd_loss) =
        train_model(&mut sgd, &x, &y, &train_batches, lr, epochs)?;

    // 2. Train with Adam
    println!("\nTraining with Adam:");
    let mut adam = Adam::new(lr).with_weight_decay(0.0001);
    let (adam_weights, adam_bias, adam_loss) =
        train_model(&mut adam, &x, &y, &train_batches, lr, epochs)?;

    // 3. Train with Lookahead(SGD)
    println!("\nTraining with Lookahead(SGD):");
    let sgd = SGD::new(lr).with_momentum(0.9).with_weight_decay(0.0001);
    let mut lookahead_sgd = Lookahead::with_config(sgd, 0.5.into(), 5);
    let (lookahead_sgd_weights, lookahead_sgd_bias, lookahead_sgd_loss) =
        train_model(&mut lookahead_sgd, &x, &y, &train_batches, lr, epochs)?;

    // 4. Train with Lookahead(Adam)
    println!("\nTraining with Lookahead(Adam):");
    let adam = Adam::new(lr).with_weight_decay(0.0001);
    let mut lookahead_adam = Lookahead::with_config(adam, 0.5.into(), 5);
    let (lookahead_adam_weights, lookahead_adam_bias, lookahead_adam_loss) =
        train_model(&mut lookahead_adam, &x, &y, &train_batches, lr, epochs)?;

    // Show results comparison
    println!("\nResults Comparison:");
    println!("===================");

    // Calculate errors
    let sgd_weight_error = (&sgd_weights - &true_weights)
        .mapv(|v| v.abs())
        .mean()
        .unwrap();
    let sgd_bias_error = (sgd_bias - true_bias).abs();

    let adam_weight_error = (&adam_weights - &true_weights)
        .mapv(|v| v.abs())
        .mean()
        .unwrap();
    let adam_bias_error = (adam_bias - true_bias).abs();

    let lookahead_sgd_weight_error = (&lookahead_sgd_weights - &true_weights)
        .mapv(|v| v.abs())
        .mean()
        .unwrap();
    let lookahead_sgd_bias_error = (lookahead_sgd_bias - true_bias).abs();

    let lookahead_adam_weight_error = (&lookahead_adam_weights - &true_weights)
        .mapv(|v| v.abs())
        .mean()
        .unwrap();
    let lookahead_adam_bias_error = (lookahead_adam_bias - true_bias).abs();

    println!(
        "{:<17} {:<10} {:<12} {:<10}",
        "Optimizer", "Loss", "Weight Error", "Bias Error"
    );
    println!("{:-<17} {:-<10} {:-<12} {:-<10}", "", "", "", "");
    println!(
        "{:<17} {:<10.6} {:<12.6} {:<10.6}",
        "SGD", sgd_loss, sgd_weight_error, sgd_bias_error
    );
    println!(
        "{:<17} {:<10.6} {:<12.6} {:<10.6}",
        "Adam", adam_loss, adam_weight_error, adam_bias_error
    );
    println!(
        "{:<17} {:<10.6} {:<12.6} {:<10.6}",
        "Lookahead(SGD)", lookahead_sgd_loss, lookahead_sgd_weight_error, lookahead_sgd_bias_error
    );
    println!(
        "{:<17} {:<10.6} {:<12.6} {:<10.6}",
        "Lookahead(Adam)",
        lookahead_adam_loss,
        lookahead_adam_weight_error,
        lookahead_adam_bias_error
    );

    println!("\nConclusions:");
    println!("============");
    println!("Lookahead typically provides more stable optimization by taking");
    println!("\"k steps forward, 1 step back,\" which can lead to better generalization");
    println!("performance. It is particularly effective when used with SGD, often");
    println!("matching or exceeding the performance of Adam while maintaining SGD's");
    println!("generalization benefits. This comes at minimal computational overhead.");

    Ok(())
}
