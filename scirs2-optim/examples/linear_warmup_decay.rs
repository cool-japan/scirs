use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::Rng;
use scirs2_optim::optimizers::SGD;
use scirs2_optim::schedulers::{DecayStrategy, LearningRateScheduler, LinearWarmupDecay};
use scirs2_optim::Optimizer;

/// Generate synthetic data for linear regression
fn generate_data<A: Float>(n_samples: usize, n_features: usize) -> (Array2<A>, Array1<A>) {
    let mut rng = rand::rng();
    let mut x = Array2::<A>::zeros((n_samples, n_features));
    let mut y = Array1::<A>::zeros(n_samples);

    // Generate random weights
    let true_weights: Vec<A> = (0..n_features)
        .map(|_| A::from(rng.random_range(-1.0..1.0)).unwrap())
        .collect();

    // Generate random features and compute targets
    for i in 0..n_samples {
        for j in 0..n_features {
            let x_val = A::from(rng.random_range(-5.0..5.0)).unwrap();
            x[[i, j]] = x_val;
        }

        // Compute target = X * w + noise
        let mut target = A::zero();
        for j in 0..n_features {
            target = target + x[[i, j]] * true_weights[j];
        }
        // Add some noise
        target = target + A::from(rng.random_range(-0.1..0.1)).unwrap();
        y[i] = target;
    }

    (x, y)
}

/// Calculate mean squared error
fn mean_squared_error<A: Float>(y_true: &Array1<A>, y_pred: &Array1<A>) -> A {
    let diff = y_pred - y_true;
    let squared = diff.mapv(|x| x * x);
    let sum = squared.sum();
    sum / A::from(y_true.len()).unwrap()
}

/// Predict values using linear model
fn predict<A: Float + 'static>(x: &Array2<A>, weights: &Array1<A>) -> Array1<A> {
    x.dot(weights)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate synthetic data
    let n_samples = 100;
    let n_features = 5;
    let (x, y) = generate_data::<f64>(n_samples, n_features);
    println!(
        "Generated data with {} samples and {} features",
        n_samples, n_features
    );

    // Training parameters
    let n_epochs = 50;
    let batch_size = 10;
    let n_batches = n_samples / batch_size;

    // Calculate total steps for warmup and decay
    let warmup_epochs = 5;
    let total_steps = n_epochs * n_batches;
    let warmup_steps = warmup_epochs * n_batches;
    let decay_steps = total_steps - warmup_steps;

    // Initialize parameters
    let mut weights = Array1::<f64>::zeros(n_features);

    // Create optimizers with different schedulers to compare
    let mut sgd_constant = SGD::new(0.1);

    // Create a linear warmup with linear decay scheduler
    let mut linear_warmup_linear_decay = LinearWarmupDecay::new(
        0.1,                                       // initial_lr (peak learning rate)
        0.01,                                      // min_lr (starting learning rate)
        warmup_steps,                              // warmup_steps
        decay_steps,                               // total_decay_steps
        DecayStrategy::Linear { final_lr: 0.001 }, // decay strategy
    );

    // Create a linear warmup with cosine decay scheduler
    let mut linear_warmup_cosine_decay = LinearWarmupDecay::new(
        0.1,                                     // initial_lr (peak learning rate)
        0.01,                                    // min_lr (starting learning rate)
        warmup_steps,                            // warmup_steps
        decay_steps,                             // total_decay_steps
        DecayStrategy::Cosine { min_lr: 0.001 }, // decay strategy
    );

    // Track learning rates and losses for visualization
    let mut linear_warmup_linear_decay_lrs = Vec::with_capacity(total_steps);
    let mut linear_warmup_cosine_decay_lrs = Vec::with_capacity(total_steps);

    // Training loop
    println!(
        "Training for {} epochs with {} batches per epoch",
        n_epochs, n_batches
    );
    println!(
        "Warmup for first {} epochs, then decay for {} epochs",
        warmup_epochs,
        n_epochs - warmup_epochs
    );

    let mut weights_linear_decay = weights.clone();
    let mut weights_cosine_decay = weights.clone();

    let mut losses_constant = Vec::with_capacity(n_epochs);
    let mut losses_linear_decay = Vec::with_capacity(n_epochs);
    let mut losses_cosine_decay = Vec::with_capacity(n_epochs);

    for epoch in 0..n_epochs {
        // Shuffle data - just using a basic shuffling approach since we had rand version issues
        let mut indices: Vec<usize> = (0..n_samples).collect();
        // Simple Fisher-Yates shuffle
        let mut rng = rand::rng();
        for i in (1..indices.len()).rev() {
            let j = rng.random_range(0..=i);
            indices.swap(i, j);
        }

        let mut epoch_loss_constant = 0.0;
        let mut epoch_loss_linear = 0.0;
        let mut epoch_loss_cosine = 0.0;

        for batch in 0..n_batches {
            // Create batch
            let batch_indices = &indices[batch * batch_size..(batch + 1) * batch_size];
            let x_batch =
                Array2::from_shape_fn((batch_size, n_features), |(i, j)| x[[batch_indices[i], j]]);
            let y_batch = Array1::from_shape_fn(batch_size, |i| y[batch_indices[i]]);

            // Constant learning rate SGD
            let y_pred_constant = predict(&x_batch, &weights);
            let loss_constant = mean_squared_error(&y_batch, &y_pred_constant);
            epoch_loss_constant += loss_constant;

            // Compute gradients
            let error_constant = &y_pred_constant - &y_batch;
            let gradient_constant = x_batch.t().dot(&error_constant) / f64::from(batch_size as u32);

            // Update parameters with constant learning rate
            weights = sgd_constant.step(&weights, &gradient_constant)?;

            // Linear warmup with linear decay
            let lr_linear = linear_warmup_linear_decay.step();
            linear_warmup_linear_decay_lrs.push(lr_linear);

            // Prediction and loss
            let y_pred_linear = predict(&x_batch, &weights_linear_decay);
            let loss_linear = mean_squared_error(&y_batch, &y_pred_linear);
            epoch_loss_linear += loss_linear;

            // Compute gradients
            let error_linear = &y_pred_linear - &y_batch;
            let gradient_linear = x_batch.t().dot(&error_linear) / f64::from(batch_size as u32);

            // Manual update with scheduled learning rate
            weights_linear_decay = &weights_linear_decay - &(&gradient_linear * lr_linear);

            // Linear warmup with cosine decay
            let lr_cosine = linear_warmup_cosine_decay.step();
            linear_warmup_cosine_decay_lrs.push(lr_cosine);

            // Prediction and loss
            let y_pred_cosine = predict(&x_batch, &weights_cosine_decay);
            let loss_cosine = mean_squared_error(&y_batch, &y_pred_cosine);
            epoch_loss_cosine += loss_cosine;

            // Compute gradients
            let error_cosine = &y_pred_cosine - &y_batch;
            let gradient_cosine = x_batch.t().dot(&error_cosine) / f64::from(batch_size as u32);

            // Manual update with scheduled learning rate
            weights_cosine_decay = &weights_cosine_decay - &(&gradient_cosine * lr_cosine);
        }

        // Record average epoch loss
        losses_constant.push(epoch_loss_constant / n_batches as f64);
        losses_linear_decay.push(epoch_loss_linear / n_batches as f64);
        losses_cosine_decay.push(epoch_loss_cosine / n_batches as f64);

        // Print progress
        if epoch % 5 == 0 || epoch == n_epochs - 1 {
            println!(
                "Epoch {}/{} - Loss (Constant): {:.6}, Loss (Linear Warmup+Decay): {:.6}, Loss (Linear Warmup+Cosine Decay): {:.6}",
                epoch + 1,
                n_epochs,
                losses_constant[epoch],
                losses_linear_decay[epoch],
                losses_cosine_decay[epoch]
            );
        }
    }

    // Show learning rate trajectory
    println!("\nLearning Rate Trajectories:");
    println!("Constant LR: 0.1 (all steps)");

    println!("\nScheduler Performance Comparison:");
    println!(
        "Final loss (Constant LR): {:.6}",
        losses_constant.last().unwrap()
    );
    println!(
        "Final loss (Linear Warmup+Linear Decay): {:.6}",
        losses_linear_decay.last().unwrap()
    );
    println!(
        "Final loss (Linear Warmup+Cosine Decay): {:.6}",
        losses_cosine_decay.last().unwrap()
    );

    println!("\nLearning Rates at Key Points:");
    let warmup_midpoint = warmup_steps / 2;
    let decay_midpoint = warmup_steps + decay_steps / 2;

    println!(
        "Start: Linear Warmup+Linear Decay={:.4}, Linear Warmup+Cosine Decay={:.4}",
        linear_warmup_linear_decay_lrs[0], linear_warmup_cosine_decay_lrs[0]
    );

    println!(
        "Mid-warmup: Linear Warmup+Linear Decay={:.4}, Linear Warmup+Cosine Decay={:.4}",
        linear_warmup_linear_decay_lrs[warmup_midpoint],
        linear_warmup_cosine_decay_lrs[warmup_midpoint]
    );

    println!(
        "End of warmup: Linear Warmup+Linear Decay={:.4}, Linear Warmup+Cosine Decay={:.4}",
        linear_warmup_linear_decay_lrs[warmup_steps - 1],
        linear_warmup_cosine_decay_lrs[warmup_steps - 1]
    );

    println!(
        "Mid-decay: Linear Warmup+Linear Decay={:.4}, Linear Warmup+Cosine Decay={:.4}",
        linear_warmup_linear_decay_lrs[decay_midpoint],
        linear_warmup_cosine_decay_lrs[decay_midpoint]
    );

    println!(
        "End: Linear Warmup+Linear Decay={:.4}, Linear Warmup+Cosine Decay={:.4}",
        linear_warmup_linear_decay_lrs.last().unwrap(),
        linear_warmup_cosine_decay_lrs.last().unwrap()
    );

    println!("\nTraining complete!");

    Ok(())
}
