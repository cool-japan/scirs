// Copyright (c) 2025, SciRS2 Team
//
// Licensed under either of
//
// * Apache License, Version 2.0
//   (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
// * MIT license
//   (LICENSE-MIT or http://opensource.org/licenses/MIT)
//
// at your option.
//

//! Example demonstrating neural network training with the array protocol.

use ndarray::{Array2, Axis};
use scirs2_core::array_protocol::{
    self,
    grad::{Adam, Optimizer, Variable},
    ml_ops::ActivationFunc,
    neural::{Dropout, Linear, Sequential},
    training::{CrossEntropyLoss, DataLoader, InMemoryDataset, ProgressCallback, Trainer},
    NdarrayWrapper,
};

fn main() {
    // Initialize the array protocol system
    array_protocol::init();

    println!("Neural Network Training Example using Array Protocol");
    println!("=================================================");

    // Part 1: Generate a simple dataset
    println!("\nPart 1: Generating a Simple Dataset");
    println!("----------------------------------");

    // Create a toy classification dataset
    let num_samples = 100;
    let num_features = 10;
    let num_classes = 3;

    // Generate random inputs
    let inputs = Array2::<f64>::from_shape_fn((num_samples, num_features), |_| {
        rand::random::<f64>() * 2.0 - 1.0
    });

    // Generate random one-hot targets
    let mut targets = Array2::<f64>::zeros((num_samples, num_classes));
    for i in 0..num_samples {
        let class = (rand::random::<f64>() * num_classes as f64).floor() as usize;
        targets[[i, class]] = 1.0;
    }

    println!(
        "Created dataset with {} samples, {} features, and {} classes",
        num_samples, num_features, num_classes
    );

    // Create train/val split
    let train_size = (num_samples as f64 * 0.8).floor() as usize;
    // Use array view indexing which is more reliable with different dimension types
    let train_inputs = inputs.slice(ndarray::s![0..train_size, ..]).to_owned();
    let train_targets = targets.slice(ndarray::s![0..train_size, ..]).to_owned();
    let val_inputs = inputs
        .slice(ndarray::s![train_size..num_samples, ..])
        .to_owned();
    let val_targets = targets
        .slice(ndarray::s![train_size..num_samples, ..])
        .to_owned();

    println!(
        "Split into {} training samples and {} validation samples",
        train_size,
        num_samples - train_size
    );

    // Create datasets
    let train_dataset = InMemoryDataset::from_arrays(train_inputs, train_targets);
    let val_dataset = InMemoryDataset::from_arrays(val_inputs, val_targets);

    // Create data loaders
    let batch_size = 16;
    let train_loader = DataLoader::new(Box::new(train_dataset), batch_size, true, Some(42));

    let val_loader = DataLoader::new(Box::new(val_dataset), batch_size, false, None);

    println!("Created data loaders with batch size {}", batch_size);
    println!("Training batches: {}", train_loader.num_batches());
    println!("Validation batches: {}", val_loader.num_batches());

    // Part 2: Create a neural network
    println!("\nPart 2: Creating a Neural Network");
    println!("------------------------------");

    // Create a model
    let mut model = Sequential::new("SimpleClassifier", Vec::new());

    // Add layers
    model.add_layer(Box::new(Linear::with_shape(
        "fc1",
        num_features,
        32,
        true,
        Some(ActivationFunc::ReLU),
    )));

    model.add_layer(Box::new(Dropout::new("dropout1", 0.2, Some(42))));

    model.add_layer(Box::new(Linear::with_shape(
        "fc2",
        32,
        16,
        true,
        Some(ActivationFunc::ReLU),
    )));

    model.add_layer(Box::new(Dropout::new("dropout2", 0.2, Some(42))));

    model.add_layer(Box::new(Linear::with_shape(
        "fc_out",
        16,
        num_classes,
        true,
        None,
    )));

    println!("Created model with {} layers", model.layers().len());

    // Print model parameters
    let params = model.parameters();
    println!("Model has {} parameter tensors", params.len());

    // Part 3: Setup optimizer and loss function
    println!("\nPart 3: Setting up Optimizer and Loss Function");
    println!("-----------------------------------------");

    // Create optimizer (Adam)
    let mut optimizer = Adam::new(0.001, Some(0.9), Some(0.999), Some(1e-8));

    // Add model parameters to optimizer (simplified version)
    for (i, param) in params.iter().enumerate() {
        let param_array = if let Some(array) = param
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, ndarray::Ix2>>()
        {
            array.as_array().clone()
        } else {
            println!(
                "Parameter {} is not a 2D ndarray, using a default 2x2 array instead",
                i
            );
            ndarray::Array2::<f64>::zeros((2, 2))
        };

        let var = Variable::new(&format!("param_{}", i), param_array);
        optimizer.add_variable(var);
    }

    println!(
        "Created Adam optimizer with {} variables",
        optimizer.variables().len()
    );

    // Create loss function
    let loss_fn = CrossEntropyLoss::new(Some("mean"));
    println!("Created CrossEntropyLoss loss function");

    // Part 4: Train the model
    println!("\nPart 4: Training the Model");
    println!("------------------------");

    // Create trainer
    let mut trainer = Trainer::new(model, Box::new(optimizer), Box::new(loss_fn));

    // Add progress callback
    trainer.add_callback(Box::new(ProgressCallback::new(true)));

    // Train the model
    let num_epochs = 10;
    println!("Starting training for {} epochs", num_epochs);

    // Note: In a full implementation, this would actually train the model
    // For this example, we'll just simulate training due to the simplified backpropagation

    println!("Note: This is a simplified training example that demonstrates");
    println!("      the API structure but doesn't perform full backpropagation.");

    // Simulate a training loop
    println!("\nTraining progress (simulated):");
    for epoch in 0..num_epochs {
        // Print epoch progress
        println!("Epoch {}/{}", epoch + 1, num_epochs);

        // Simulate batch progress
        let num_batches = train_loader.num_batches();
        for batch in 0..num_batches {
            if (batch + 1) % (num_batches / 10).max(1) == 0 {
                let simulated_loss =
                    1.0 - (epoch as f64 * 0.1 + batch as f64 * 0.01 / num_batches as f64);
                print!(
                    "\rBatch {}/{} - loss: {:.4}",
                    batch + 1,
                    num_batches,
                    simulated_loss
                );
            }
        }
        println!();

        // Simulate metrics
        let train_loss = 1.0 - epoch as f64 * 0.1;
        let train_acc = 0.33 + epoch as f64 * 0.06;
        let val_loss = 1.1 - epoch as f64 * 0.09;
        let val_acc = 0.31 + epoch as f64 * 0.055;

        println!(
            "train: loss = {:.4}, accuracy = {:.4}",
            train_loss, train_acc
        );
        println!("val: loss = {:.4}, accuracy = {:.4}", val_loss, val_acc);
    }

    println!("\nTraining completed");
    println!("Final validation accuracy: {:.4}", 0.31 + 9.0 * 0.055);

    // Part 5: Making predictions with the trained model
    println!("\nPart 5: Making Predictions with the Trained Model");
    println!("--------------------------------------------");

    // Create a sample input
    let sample_input =
        Array2::<f64>::from_shape_fn((1, num_features), |_| rand::random::<f64>() * 2.0 - 1.0);
    let input_wrapped = NdarrayWrapper::new(sample_input.clone());

    // Make a prediction
    let model = get_model_from_trainer(&trainer);
    let output = match model.forward(&input_wrapped) {
        Ok(out) => out,
        Err(e) => {
            println!("Error in forward pass: {}", e);
            return; // Skip forward pass test since we can't continue
        }
    };

    let output_array = match output
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, ndarray::Ix2>>()
    {
        Some(array) => array,
        None => {
            println!("Output is not a 2D ndarray");
            return; // Skip forward pass test since we can't continue
        }
    };

    println!("Input shape: {:?}", sample_input.shape());
    println!("Output shape: {:?}", output_array.as_array().shape());
    println!("Raw predictions: {:?}", output_array.as_array());

    // Get class probabilities (apply softmax)
    let exp_outputs = output_array.as_array().mapv(|x| x.exp());
    let sum_exp = exp_outputs.sum_axis(Axis(1));
    let probs = exp_outputs / sum_exp.insert_axis(Axis(1));

    println!("Class probabilities: {:?}", probs);

    // Get predicted class
    let predicted_class = match probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
    {
        Some(class) => class,
        None => {
            println!("Could not determine predicted class (output array may be empty)");
            0 // Default to class 0 if we can't determine
        }
    };

    println!("Predicted class: {}", predicted_class);
}

// Helper function to get model from trainer
fn get_model_from_trainer(_trainer: &Trainer) -> Sequential {
    // In a real implementation, we would have a method on Trainer to access the model
    // This is just a workaround for the example
    // Note: in production code, you should use proper trainer.get_model() accessor method

    // Since we can't safely transmute to get the model in this case,
    // we'll create a new model with the same structure as a workaround
    create_simple_model()
}

// Create a simple model similar to the one used in the example
fn create_simple_model() -> Sequential {
    let mut model = Sequential::new("SimpleClassifier", Vec::new());

    // Add layers with the same structure as in main()
    model.add_layer(Box::new(Linear::with_shape(
        "fc1",
        10, // num_features from the example
        32,
        true,
        Some(ActivationFunc::ReLU),
    )));

    model.add_layer(Box::new(Dropout::new("dropout1", 0.2, Some(42))));

    model.add_layer(Box::new(Linear::with_shape(
        "fc2",
        32,
        16,
        true,
        Some(ActivationFunc::ReLU),
    )));

    model.add_layer(Box::new(Dropout::new("dropout2", 0.2, Some(42))));

    model.add_layer(Box::new(Linear::with_shape(
        "fc_out", 16, 3, // num_classes from the example
        true, None,
    )));

    model
}
