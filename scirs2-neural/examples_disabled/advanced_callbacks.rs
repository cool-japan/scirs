use ndarray::{Array2, ScalarOperand};
use num_traits::Float;
use rand::prelude::*;
use rand::rngs::SmallRng;
use scirs2_neural::callbacks::{CallbackManager, EarlyStopping};
use scirs2_neural::error::Result;
use scirs2_neural::layers::Dense;
use scirs2_neural::losses::MeanSquaredError;
use scirs2_neural::models::{sequential::Sequential, Model};
use scirs2_neural::optimizers::Adam;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::Instant;

// Create a synthetic regression dataset (noisy sine wave)
#[allow(dead_code)]
fn create_sine_dataset(
    n_samples: usize,
    noise_level: f32,
    rng: &mut SmallRng,
) -> (Array2<f32>, Array2<f32>) {
    let mut x = Array2::<f32>::zeros((n_samples, 1));
    let mut y = Array2::<f32>::zeros((n_samples, 1));
    for i in 0..n_samples {
        let x_val = (i as f32) / (n_samples as f32) * 4.0 * std::f32::consts::PI;
        let y_val = x_val.sin();
        // Add some noise
        let noise = rng.gen_range(-noise_level..noise_level);
        x[[i..0]] = x_val;
        y[[i, 0]] = y_val + noise;
    }
    (x, y)
}
// Create a neural network model for regression
#[allow(dead_code)]
fn create_regression_model(_inputdim: usize, rng: &mut SmallRng) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    // Input layer
    let dense1 = Dense::new(_input_dim, 16, Some("relu"), rng)?;
    model.add_layer(dense1);
    // Hidden layers
    let dense2 = Dense::new(16, 8, Some("relu"), rng)?;
    model.add_layer(dense2);
    // Output layer (linear activation for regression)
    let dense3 = Dense::new(8, 1, None, rng)?;
    model.add_layer(dense3);
    Ok(model)
// Calculate mean squared error
#[allow(dead_code)]
fn calculate_mse<F: Float + Debug + ScalarOperand>(
    model: &Sequential<F>,
    x: &Array2<F>,
    y: &Array2<F>,
) -> Result<F> {
    let predictions = model.forward(&x.clone().into_dyn())?;
    let mut sum_squared_error = F::zero();
    for i in 0..x.nrows() {
        let diff = predictions[[i, 0]] - y[[i, 0]];
        sum_squared_error = sum_squared_error + diff * diff;
    Ok(sum_squared_error / F::from(x.nrows()).unwrap())
#[allow(dead_code)]
fn main() -> Result<()> {
    println!("Advanced Learning Rate Scheduling and Early Stopping Example");
    println!("==========================================================\n");
    // Initialize random number generator
    let mut rng = SmallRng::from_seed([42; 32]);
    // Create synthetic regression dataset
    let n_samples = 100;
    let (x, y) = create_sine_dataset(n_samples, 0.1, &mut rng);
    println!(
        "Created synthetic sine wave regression dataset with {} samples",
        n_samples
    );
    // Generate 80% training data, 20% validation data
    let train_size = (n_samples as f32 * 0.8) as usize;
    let (x_train, y_train) = (
        x.slice(ndarray::s![0..train_size, ..]).to_owned(),
        y.slice(ndarray::s![0..train_size, ..]).to_owned(),
    let (x_val, y_val) = (
        x.slice(ndarray::s![train_size.., ..]).to_owned(),
        y.slice(ndarray::s![train_size.., ..]).to_owned(),
        "Split into {} training and {} validation samples",
        x_train.nrows(),
        x_val.nrows()
    // Train with early stopping
    println!("\nTraining with early stopping...");
    let model = train_with_early_stopping(&mut rng, &x_train, &y_train, &x_val, &y_val)?;
    // Evaluate final validation loss
    let val_mse = calculate_mse(&model, &x_val, &y_val)?;
    println!("\nFinal validation MSE: {:.6}", val_mse);
    println!("\nAdvanced callbacks example completed successfully!");
    Ok(())
// Train with early stopping and validation
#[allow(dead_code)]
fn train_with_early_stopping(
    x_train: &Array2<f32>,
    y_train: &Array2<f32>,
    x_val: &Array2<f32>,
    y_val: &Array2<f32>,
) -> Result<Sequential<f32>> {
    let mut model = create_regression_model(x_train.ncols(), rng)?;
    println!("Created model with {} layers", model.num_layers());
    // Setup loss function and optimizer
    let loss_fn = MeanSquaredError::new();
    let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
    // Setup early stopping callback
    // Stop training if validation loss doesn't improve for 30 epochs
    let early_stopping = EarlyStopping::new(30, 0.0001, true);
    let mut callback_manager = CallbackManager::<f32>::new();
    callback_manager.add_callback(Box::new(early_stopping));
    println!("Starting training with early stopping (patience = 30 epochs)...");
    let start_time = Instant::now();
    // Convert to dynamic arrays
    let x_train_dyn = x_train.clone().into_dyn();
    let y_train_dyn = y_train.clone().into_dyn();
    // Set up training parameters
    let max_epochs = 500;
    // Training loop with validation
    let mut epoch_metrics = HashMap::new();
    let mut best_val_loss = f32::MAX;
    let mut stop_training = false;
    for epoch in 0..max_epochs {
        // Call callbacks before epoch
        callback_manager.on_epoch_begin(epoch)?;
        // Train one epoch
        let train_loss = model.train_batch(&x_train_dyn, &y_train_dyn, &loss_fn, &mut optimizer)?;
        // Validate
        let val_loss = calculate_mse(&model, x_val, y_val)?;
        // Update metrics
        epoch_metrics.insert("loss".to_string(), train_loss);
        epoch_metrics.insert("val_loss".to_string(), val_loss);
        // Call callbacks after epoch
        let should_stop = callback_manager.on_epoch_end(epoch, &epoch_metrics)?;
        if should_stop {
            println!("Early stopping triggered after {} epochs", epoch + 1);
            stop_training = true;
        }
        // Track best validation loss
        if val_loss < best_val_loss {
            best_val_loss = val_loss;
        // Print progress
        if epoch % 50 == 0 || epoch == max_epochs - 1 || stop_training {
            println!(
                "Epoch {}/{}: train_loss = {:.6}, val_loss = {:.6}",
                epoch + 1,
                max_epochs,
                train_loss,
                val_loss
            );
        if stop_training {
            break;
    let elapsed = start_time.elapsed();
    println!("Training completed in {:.2}s", elapsed.as_secs_f32());
    println!("Best validation MSE: {:.6}", best_val_loss);
