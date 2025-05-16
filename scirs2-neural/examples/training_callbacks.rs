use ndarray::Array2;
use rand::prelude::*;
use rand::rngs::SmallRng;
use scirs2_neural::callbacks::{CallbackManager, EarlyStopping};
use scirs2_neural::error::Result;
use scirs2_neural::layers::Dense;
use scirs2_neural::losses::MeanSquaredError;
use scirs2_neural::models::{sequential::Sequential, Model};
use scirs2_neural::optimizers::Adam;
use std::collections::HashMap;
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
    println!("Early Stopping and Learning Rate Scheduling Example");
    println!("==================================================\n");

    // Initialize random number generator with a fixed seed for reproducibility
    let mut rng = SmallRng::seed_from_u64(42);

    // Create XOR dataset
    let (x, y) = create_xor_dataset();
    println!("Dataset created (XOR problem)");

    // Train with different callback configurations
    train_with_early_stopping(&mut rng, &x, &y)?;

    // For now, we'll skip the other examples due to integration challenges
    // train_with_step_decay(&mut rng, &x, &y)?;
    // train_with_reduce_on_plateau(&mut rng, &x, &y)?;

    println!("\nTraining example completed successfully!");
    Ok(())
}

// Train a model with early stopping
fn train_with_early_stopping(rng: &mut SmallRng, x: &Array2<f32>, y: &Array2<f32>) -> Result<()> {
    println!("\n1. Training with Early Stopping");
    println!("------------------------------");

    let mut model = create_xor_model(rng)?;
    println!("Created model with {} layers", model.num_layers());

    // Setup loss function and optimizer
    let loss_fn = MeanSquaredError::new();
    let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);

    // Setup early stopping callback
    // Stop training if loss doesn't improve for 20 epochs
    // with a minimum improvement of 0.0001
    let early_stopping = EarlyStopping::new(20, 0.0001, false);
    let mut callback_manager = CallbackManager::<f32>::new();
    callback_manager.add_callback(Box::new(early_stopping));

    println!("Starting training with early stopping (patience = 20 epochs)...");
    let start_time = Instant::now();

    // Set up batch training parameters
    let x_dyn = x.clone().into_dyn();
    let y_dyn = y.clone().into_dyn();
    let max_epochs = 200;

    // Training loop
    let mut epoch_metrics = HashMap::new();
    let mut stop_training = false;

    for epoch in 0..max_epochs {
        // Call callbacks before epoch
        callback_manager.on_epoch_begin(epoch)?;

        // Train one batch
        let loss = model.train_batch(&x_dyn, &y_dyn, &loss_fn, &mut optimizer)?;

        // Update metrics
        epoch_metrics.insert("loss".to_string(), loss);

        // Call callbacks after epoch
        let should_stop = callback_manager.on_epoch_end(epoch, &epoch_metrics)?;

        if should_stop {
            println!("Early stopping triggered after {} epochs", epoch + 1);
            stop_training = true;
        }

        // Print progress
        if epoch % 20 == 0 || epoch == max_epochs - 1 || stop_training {
            println!("Epoch {}/{}: loss = {:.6}", epoch + 1, max_epochs, loss);
        }

        if stop_training {
            break;
        }
    }

    let elapsed = start_time.elapsed();
    println!(
        "Training completed in {:.2}s{}",
        elapsed.as_secs_f32(),
        if stop_training {
            " (early stopped)"
        } else {
            ""
        }
    );

    // Evaluate the model
    evaluate_model(&model, x, y)?;

    Ok(())
}
