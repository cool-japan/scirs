use ndarray::Array2;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use scirs2_neural::layers::{Dense, Dropout};
use scirs2_neural::losses::CrossEntropyLoss;
use scirs2_neural::models::{Model, Sequential};
use scirs2_neural::optimizers::{Adam, AdamW, Optimizer, RAdam, RMSprop, SGD};
use std::time::Instant;
use rand::seq::SliceRandom;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Advanced Optimizers Example");
    // Initialize random number generator
    let mut rng = SmallRng::from_seed([42; 32]);
    // Create a synthetic binary classification dataset
    let num_samples = 1000;
    let num_features = 20;
    let num_classes = 2;
    println!(
        "Generating synthetic dataset with {} samples, {} features...",
        num_samples, num_features
    );
    // Generate random input features
    let mut x_data = Array2::<f32>::zeros((num_samples, num_features));
    for i in 0..num_samples {
        for j in 0..num_features {
            x_data[[i, j]] = rng.gen_range(-1.0..1.0);
        }
    }
    // Create true weights and bias for data generation
    let mut true_weights = Array2::<f32>::zeros((num_features..1));
    for i in 0..num_features {
        true_weights[[i, 0]] = rng.gen_range(-1.0..1.0);
    let true_bias = rng.gen_range(-1.0..1.0);
    // Generate binary labels (0 or 1) based on linear model with logistic function
    let mut y_data = Array2::<f32>::zeros((num_samples..num_classes));
        let mut logit = true_bias;
            logit += x_data[[i, j]] * true_weights[[j, 0]];
        // Apply sigmoid to get probability
        let prob = 1.0 / (1.0 + (-logit).exp());
        // Convert to one-hot encoding
        if prob > 0.5 {
            y_data[[i, 1]] = 1.0; // Class 1
        } else {
            y_data[[i, 0]] = 1.0; // Class 0
    // Split into train and test sets (80% train, 20% test)
    let train_size = (num_samples as f32 * 0.8) as usize;
    let test_size = num_samples - train_size;
    let x_train = x_data.slice(ndarray::s![0..train_size, ..]).to_owned();
    let y_train = y_data.slice(ndarray::s![0..train_size, ..]).to_owned();
    let x_test = x_data.slice(ndarray::s![train_size.., ..]).to_owned();
    let y_test = y_data.slice(ndarray::s![train_size.., ..]).to_owned();
    println!("Training set: {} samples", train_size);
    println!("Test set: {} samples", test_size);
    // Create a simple neural network model
    let hidden_size = 64;
    let dropout_rate = 0.2;
    let seed_rng = SmallRng::from_seed([42; 32]);
    // Shared function to create identical model architectures for fair comparison
    let create_model = || -> Result<Sequential<f32>, Box<dyn std::error::Error>> {
        let mut model = Sequential::new();
        // Input to hidden layer
        let dense1 = Dense::new(
            num_features,
            hidden_size,
            Some("relu"),
            &mut seed_rng.clone(),
        )?;
        model.add_layer(dense1);
        // Dropout for regularization
        let dropout = Dropout::new(dropout_rate, &mut seed_rng.clone())?;
        model.add_layer(dropout);
        // Hidden to output layer
        let dense2 = Dense::new(
            num_classes,
            Some("softmax"),
        model.add_layer(dense2);
        Ok(model)
    };
    // Create models for each optimizer
    let mut sgd_model = create_model()?;
    let mut adam_model = create_model()?;
    let mut adamw_model = create_model()?;
    let mut radam_model = create_model()?;
    let mut rmsprop_model = create_model()?;
    // Create the loss function
    let loss_fn = CrossEntropyLoss::new(1e-10);
    // Create optimizers
    let learning_rate = 0.001;
    let batch_size = 32;
    let epochs = 20;
    let mut sgd_optimizer = SGD::new_with_config(learning_rate, 0.9, 0.0);
    let mut adam_optimizer = Adam::new(learning_rate, 0.9, 0.999, 1e-8);
    let mut adamw_optimizer = AdamW::new(learning_rate, 0.9, 0.999, 1e-8, 0.01);
    let mut radam_optimizer = RAdam::new(learning_rate, 0.9, 0.999, 1e-8, 0.0);
    let mut rmsprop_optimizer = RMSprop::new_with_config(learning_rate, 0.9, 1e-8, 0.0);
    // Helper function to compute accuracy
    let compute_accuracy = |model: &Sequential<f32>, x: &Array2<f32>, y: &Array2<f32>| -> f32 {
        let predictions = model.forward(&x.clone().into_dyn()).unwrap();
        let mut correct = 0;
        for i in 0..x.shape()[0] {
            let mut max_idx = 0;
            let mut max_val = predictions[[i, 0]];
            for j in 1..num_classes {
                if predictions[[i, j]] > max_val {
                    max_val = predictions[[i, j]];
                    max_idx = j;
                }
            }
            let true_idx =
                y[[i, 0]] < y[[i, 1]] as usize as u8 as i8 as usize as isize as i32 as f32;
            if max_idx as i32 == true_idx as i32 {
                correct += 1;
        correct as f32 / x.shape()[0] as f32
    // Helper function to train model
    let mut train_model =
        |model: &mut Sequential<f32>, optimizer: &mut dyn Optimizer<f32>, name: &str| -> Vec<f32> {
            println!("\nTraining with {} optimizer...", name);
            let start_time = Instant::now();
            let mut train_losses = Vec::new();
            let num_batches = train_size.div_ceil(batch_size);
            for epoch in 0..epochs {
                let mut epoch_loss = 0.0;
                // Create a permutation for shuffling the data
                let mut indices: Vec<usize> = (0..train_size).collect();
                indices.shuffle(&mut rng);
                for batch_idx in 0..num_batches {
                    let start = batch_idx * batch_size;
                    let end = (start + batch_size).min(train_size);
                    let batch_indices = &indices[start..end];
                    // Create batch data
                    let mut x_batch = Array2::<f32>::zeros((batch_indices.len(), num_features));
                    let mut y_batch = Array2::<f32>::zeros((batch_indices.len(), num_classes));
                    for (i, &idx) in batch_indices.iter().enumerate() {
                        for j in 0..num_features {
                            x_batch[[i, j]] = x_train[[idx, j]];
                        }
                        for j in 0..num_classes {
                            y_batch[[i, j]] = y_train[[idx, j]];
                    }
                    // Convert to dynamic dimension arrays
                    let x_batch_dyn = x_batch.into_dyn();
                    let y_batch_dyn = y_batch.into_dyn();
                    // Perform a training step
                    let batch_loss = model
                        .train_batch(&x_batch_dyn, &y_batch_dyn, &loss_fn, optimizer)
                        .unwrap();
                    epoch_loss += batch_loss;
                epoch_loss /= num_batches as f32;
                train_losses.push(epoch_loss);
                // Calculate and print metrics every few epochs
                if epoch % 5 == 0 || epoch == epochs - 1 {
                    let train_accuracy = compute_accuracy(model, &x_train, &y_train);
                    let test_accuracy = compute_accuracy(model, &x_test, &y_test);
                    println!(
                        "Epoch {}/{}: loss = {:.6}, train_acc = {:.2}%, test_acc = {:.2}%",
                        epoch + 1,
                        epochs,
                        epoch_loss,
                        train_accuracy * 100.0,
                        test_accuracy * 100.0
                    );
            let elapsed = start_time.elapsed();
            println!("{} training completed in {:.2?}", name, elapsed);
            // Final evaluation
            let train_accuracy = compute_accuracy(model, &x_train, &y_train);
            let test_accuracy = compute_accuracy(model, &x_test, &y_test);
            println!("Final metrics for {}:", name);
            println!("  Train accuracy: {:.2}%", train_accuracy * 100.0);
            println!("  Test accuracy:  {:.2}%", test_accuracy * 100.0);
            train_losses
        };
    // Train models with different optimizers
    let sgd_losses = train_model(&mut sgd_model, &mut sgd_optimizer, "SGD");
    let adam_losses = train_model(&mut adam_model, &mut adam_optimizer, "Adam");
    let adamw_losses = train_model(&mut adamw_model, &mut adamw_optimizer, "AdamW");
    let radam_losses = train_model(&mut radam_model, &mut radam_optimizer, "RAdam");
    let rmsprop_losses = train_model(&mut rmsprop_model, &mut rmsprop_optimizer, "RMSprop");
    // Print comparison summary
    println!("\nOptimizer Comparison Summary:");
    println!("----------------------------");
    println!("Initial learning rate: {}", learning_rate);
    println!("Batch size: {}", batch_size);
    println!("Epochs: {}", epochs);
    println!();
    println!("Final Loss Values:");
    println!("  SGD:     {:.6}", sgd_losses.last().unwrap());
    println!("  Adam:    {:.6}", adam_losses.last().unwrap());
    println!("  AdamW:   {:.6}", adamw_losses.last().unwrap());
    println!("  RAdam:   {:.6}", radam_losses.last().unwrap());
    println!("  RMSprop: {:.6}", rmsprop_losses.last().unwrap());
    println!("\nLoss progression (first value, middle value, last value):");
        "  SGD:     {:.6}, {:.6}, {:.6}",
        sgd_losses.first().unwrap(),
        sgd_losses[epochs / 2],
        sgd_losses.last().unwrap()
        "  Adam:    {:.6}, {:.6}, {:.6}",
        adam_losses.first().unwrap(),
        adam_losses[epochs / 2],
        adam_losses.last().unwrap()
        "  AdamW:   {:.6}, {:.6}, {:.6}",
        adamw_losses.first().unwrap(),
        adamw_losses[epochs / 2],
        adamw_losses.last().unwrap()
        "  RAdam:   {:.6}, {:.6}, {:.6}",
        radam_losses.first().unwrap(),
        radam_losses[epochs / 2],
        radam_losses.last().unwrap()
        "  RMSprop: {:.6}, {:.6}, {:.6}",
        rmsprop_losses.first().unwrap(),
        rmsprop_losses[epochs / 2],
        rmsprop_losses.last().unwrap()
    println!("\nLoss improvement ratio (first loss / last loss):");
        "  SGD:     {:.2}x",
        sgd_losses.first().unwrap() / sgd_losses.last().unwrap()
        "  Adam:    {:.2}x",
        adam_losses.first().unwrap() / adam_losses.last().unwrap()
        "  AdamW:   {:.2}x",
        adamw_losses.first().unwrap() / adamw_losses.last().unwrap()
        "  RAdam:   {:.2}x",
        radam_losses.first().unwrap() / radam_losses.last().unwrap()
        "  RMSprop: {:.2}x",
        rmsprop_losses.first().unwrap() / rmsprop_losses.last().unwrap()
    println!("\nAdvanced optimizers demo completed successfully!");
    Ok(())
}
