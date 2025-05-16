use ndarray::Array2;
use rand::prelude::*;
use rand::rngs::SmallRng;
use scirs2_neural::callbacks::{CosineAnnealingLR, ScheduleMethod};
use scirs2_neural::error::Result;
use scirs2_neural::layers::Dense;
use scirs2_neural::losses::MeanSquaredError;
use scirs2_neural::models::{sequential::Sequential, Model};
use scirs2_neural::optimizers::{with_cosine_annealing, with_step_decay, Adam, Optimizer};
use std::time::Instant;

// Create XOR dataset
fn create_xor_dataset() -> (Array2<f32>, Array2<f32>) {
    // XOR truth table inputs
    let x = Array2::from_shape_vec(
        (4, 2),
        vec![
            0.0, 0.0, // 0 XOR 0 = 0
            0.0, 1.0, // 0 XOR 1 = 1
            1.0, 0.0, // 1 XOR 0 = 1
            1.0, 1.0, // 1 XOR 1 = 0
        ],
    )
    .unwrap();

    // XOR truth table outputs
    let y = Array2::from_shape_vec(
        (4, 1),
        vec![
            0.0, // 0 XOR 0 = 0
            1.0, // 0 XOR 1 = 1
            1.0, // 1 XOR 0 = 1
            0.0, // 1 XOR 1 = 0
        ],
    )
    .unwrap();

    (x, y)
}

// Create a simple neural network model for the XOR problem
fn create_xor_model(rng: &mut SmallRng) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();

    // Input layer with 2 neurons (XOR has 2 inputs)
    let dense1 = Dense::new(2, 8, Some("relu"), rng)?;
    model.add_layer(dense1);

    // Hidden layer
    let dense2 = Dense::new(8, 4, Some("relu"), rng)?;
    model.add_layer(dense2);

    // Output layer with 1 neuron (XOR has 1 output)
    let dense3 = Dense::new(4, 1, Some("sigmoid"), rng)?;
    model.add_layer(dense3);

    Ok(model)
}

// Evaluate model by printing predictions for the XOR problem
fn evaluate_model(model: &Sequential<f32>, x: &Array2<f32>, y: &Array2<f32>) -> Result<f32> {
    let predictions = model.forward(&x.clone().into_dyn())?;
    let binary_thresh = 0.5;

    println!("\nModel predictions:");
    println!("-----------------");
    println!("   X₁   |   X₂   | Target | Prediction | Binary");
    println!("----------------------------------------------");

    let mut correct = 0;
    for i in 0..x.shape()[0] {
        let pred = predictions[[i, 0]];
        let binary_pred = pred > binary_thresh;
        let target = y[[i, 0]];
        let is_correct = (binary_pred as i32 as f32 - target).abs() < 1e-6;

        if is_correct {
            correct += 1;
        }

        println!(
            " {:.4}  | {:.4}  | {:.4}  |   {:.4}   |  {}  {}",
            x[[i, 0]],
            x[[i, 1]],
            target,
            pred,
            binary_pred as i32,
            if is_correct { "✓" } else { "✗" }
        );
    }

    let accuracy = correct as f32 / x.shape()[0] as f32;
    println!(
        "\nAccuracy: {:.2}% ({}/{})",
        accuracy * 100.0,
        correct,
        x.shape()[0]
    );

    Ok(accuracy)
}

fn main() -> Result<()> {
    println!("Learning Rate Scheduler Integration Example");
    println!("===========================================\n");

    // Initialize random number generator with a fixed seed for reproducibility
    let mut rng = SmallRng::seed_from_u64(42);

    // Create XOR dataset
    let (x, y) = create_xor_dataset();
    println!("Dataset created (XOR problem)");

    // Train with different scheduler-optimizer integrations
    train_with_step_decay(&mut rng, &x, &y)?;
    train_with_cosine_annealing(&mut rng, &x, &y)?;
    train_with_manual_scheduler_integration(&mut rng, &x, &y)?;

    println!("\nAll training examples completed successfully!");
    Ok(())
}

// Train with step decay learning rate scheduling
fn train_with_step_decay(rng: &mut SmallRng, x: &Array2<f32>, y: &Array2<f32>) -> Result<()> {
    println!("\n1. Training with Step Decay Learning Rate Scheduling");
    println!("--------------------------------------------------");

    let mut model = create_xor_model(rng)?;
    println!("Created model with {} layers", model.num_layers());

    // Setup loss function and optimizer with step decay scheduling
    let loss_fn = MeanSquaredError::new();

    // Option 1: Using the helper function
    let epochs = 300;
    let mut optimizer = with_step_decay(
        Adam::new(0.1, 0.9, 0.999, 1e-8),
        0.1.into(),   // Initial LR
        0.5.into(),   // Factor (reduce by half)
        50,           // Step size (every 50 epochs)
        0.001.into(), // Min LR
        epochs,       // Total steps
    );

    println!("Starting training with step decay LR scheduling...");
    println!("Initial LR: 0.1, Factor: 0.5, Step size: 50 epochs");
    let start_time = Instant::now();

    // Convert to dynamic arrays
    let x_dyn = x.clone().into_dyn();
    let y_dyn = y.clone().into_dyn();

    // Training loop with learning rate tracking
    let mut lr_history = Vec::<(usize, f32)>::new();

    for epoch in 0..epochs {
        // Train one batch
        let loss = model.train_batch(&x_dyn, &y_dyn, &loss_fn, &mut optimizer)?;

        // Record current learning rate
        let current_lr = optimizer.get_learning_rate();

        // Track learning rate changes
        if epoch == 0 || lr_history.is_empty() || lr_history.last().unwrap().1 != current_lr {
            lr_history.push((epoch, current_lr));
        }

        // Print progress
        if epoch % 50 == 0 || epoch == epochs - 1 {
            println!(
                "Epoch {}/{}: loss = {:.6}, lr = {:.6}",
                epoch + 1,
                epochs,
                loss,
                current_lr
            );
        }
    }

    let elapsed = start_time.elapsed();
    println!("Training completed in {:.2}s", elapsed.as_secs_f32());

    // Print learning rate history
    println!("\nLearning rate changes:");
    for (epoch, lr) in lr_history {
        println!("Epoch {}: lr = {:.6}", epoch + 1, lr);
    }

    // Evaluate the model
    evaluate_model(&model, x, y)?;

    Ok(())
}

// Train with cosine annealing learning rate scheduling
fn train_with_cosine_annealing(rng: &mut SmallRng, x: &Array2<f32>, y: &Array2<f32>) -> Result<()> {
    println!("\n2. Training with Cosine Annealing Learning Rate Scheduling");
    println!("--------------------------------------------------------");

    let mut model = create_xor_model(rng)?;
    println!("Created model with {} layers", model.num_layers());

    // Setup loss function and optimizer with cosine annealing scheduling
    let loss_fn = MeanSquaredError::new();

    // Using the helper function for cosine annealing
    let epochs = 300;
    let cycle_length = 50;
    let mut optimizer = with_cosine_annealing(
        Adam::new(0.01, 0.9, 0.999, 1e-8),
        0.01.into(),   // Max LR
        0.0001.into(), // Min LR
        cycle_length,  // Cycle length
        epochs,        // Total steps
    );

    println!("Starting training with cosine annealing LR scheduling...");
    println!(
        "Max LR: 0.01, Min LR: 0.0001, Cycle length: {} epochs",
        cycle_length
    );
    let start_time = Instant::now();

    // Convert to dynamic arrays
    let x_dyn = x.clone().into_dyn();
    let y_dyn = y.clone().into_dyn();

    // Training loop with learning rate tracking
    let mut lr_samples = Vec::<(usize, f32)>::new();

    for epoch in 0..epochs {
        // Train one batch
        let loss = model.train_batch(&x_dyn, &y_dyn, &loss_fn, &mut optimizer)?;

        // Get current learning rate
        let current_lr = optimizer.get_learning_rate();

        // Record learning rate at specific points to show the cycle
        if epoch % 10 == 0 || epoch == epochs - 1 {
            lr_samples.push((epoch, current_lr));
        }

        // Print progress
        if epoch % 50 == 0 || epoch == epochs - 1 {
            println!(
                "Epoch {}/{}: loss = {:.6}, lr = {:.6}",
                epoch + 1,
                epochs,
                loss,
                current_lr
            );
        }
    }

    let elapsed = start_time.elapsed();
    println!("Training completed in {:.2}s", elapsed.as_secs_f32());

    // Print learning rate samples to demonstrate the cosine curve
    println!("\nLearning rate samples (showing cosine curve):");
    for (epoch, lr) in lr_samples {
        println!("Epoch {}: lr = {:.6}", epoch + 1, lr);
    }

    // Evaluate the model
    evaluate_model(&model, x, y)?;

    Ok(())
}

// Train with manual scheduler integration
fn train_with_manual_scheduler_integration(
    rng: &mut SmallRng,
    x: &Array2<f32>,
    y: &Array2<f32>,
) -> Result<()> {
    println!("\n3. Training with Manual Scheduler Integration");
    println!("-------------------------------------------");

    let mut model = create_xor_model(rng)?;
    println!("Created model with {} layers", model.num_layers());

    // Setup loss function and optimizer
    let loss_fn = MeanSquaredError::new();
    let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);

    // Create scheduler manually
    let epochs = 300;
    let scheduler = CosineAnnealingLR::new(
        0.01.into(),   // Max LR
        0.0001.into(), // Min LR
        100,           // Cycle length
        ScheduleMethod::Epoch,
        epochs, // Total steps
    );

    println!("Starting training with manual scheduler integration...");
    println!("Max LR: 0.01, Min LR: 0.0001, Cycle length: 100 epochs");
    let start_time = Instant::now();

    // Convert to dynamic arrays
    let x_dyn = x.clone().into_dyn();
    let y_dyn = y.clone().into_dyn();

    // Training loop with manual scheduler updates
    for epoch in 0..epochs {
        // Update learning rate using scheduler
        let current_lr = scheduler.calculate_lr(epoch);
        optimizer.set_learning_rate(current_lr);

        // Train one batch
        let loss = model.train_batch(&x_dyn, &y_dyn, &loss_fn, &mut optimizer)?;

        // Print progress
        if epoch % 50 == 0 || epoch == epochs - 1 {
            println!(
                "Epoch {}/{}: loss = {:.6}, lr = {:.6}",
                epoch + 1,
                epochs,
                loss,
                current_lr
            );
        }
    }

    let elapsed = start_time.elapsed();
    println!("Training completed in {:.2}s", elapsed.as_secs_f32());

    // Evaluate the model
    evaluate_model(&model, x, y)?;

    Ok(())
}
