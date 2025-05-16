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

//! Example demonstrating distributed training and model serialization with the array protocol.

use std::collections::HashMap;
use tempfile::tempdir;

use ndarray::Array2;
use scirs2_core::array_protocol::{
    self,
    distributed_training::{
        DistributedStrategy, DistributedTrainingConfig, DistributedTrainingFactory,
    },
    grad::Adam,
    ml_ops::ActivationFunc,
    neural::{Conv2D, Linear, MaxPool2D, Sequential},
    serialization::{load_checkpoint, save_checkpoint, ModelSerializer, OnnxExporter},
    training::Dataset,
    training::{CrossEntropyLoss, DataLoader, InMemoryDataset, Trainer},
};

fn main() {
    // Initialize the array protocol system
    array_protocol::init();

    println!("Distributed Training and Model Serialization Example");
    println!("==================================================");

    // Part 1: Create a Model and Dataset
    println!("\nPart 1: Creating a Model and Dataset");
    println!("----------------------------------");

    // Create a model
    let model = create_model();
    println!("Created model with {} layers", model.layers().len());

    // Create a dataset
    let (train_dataset, val_dataset) = create_dataset();
    println!(
        "Created dataset with {} training samples and {} validation samples",
        Dataset::len(&train_dataset),
        Dataset::len(&val_dataset)
    );

    // Part 2: Distributed Training Setup
    println!("\nPart 2: Distributed Training Setup");
    println!("-------------------------------");

    // Create distributed training configuration
    let dist_config = DistributedTrainingConfig {
        strategy: DistributedStrategy::DataParallel,
        num_workers: 2,
        rank: 0,         // This would be different for each worker
        is_master: true, // This would be different for each worker
        sync_interval: 1,
        backend: "threaded".to_string(),
        mixed_precision: false,
        gradient_accumulation_steps: 1,
    };

    println!(
        "Created distributed training config with {} workers using {} strategy",
        dist_config.num_workers, dist_config.strategy
    );

    // Create distributed dataset
    let dist_train_dataset =
        DistributedTrainingFactory::create_dataset(Box::new(train_dataset), &dist_config);

    let dist_val_dataset =
        DistributedTrainingFactory::create_dataset(Box::new(val_dataset), &dist_config);

    println!(
        "Created distributed datasets with {} training samples and {} validation samples",
        dist_train_dataset.len(),
        dist_val_dataset.len()
    );

    // Create data loaders
    let batch_size = 32;
    let _train_loader = DataLoader::new(dist_train_dataset, batch_size, true, Some(42));

    let _val_loader = DataLoader::new(dist_val_dataset, batch_size, false, None);

    println!("Created data loaders with batch size {}", batch_size);

    // Part 3: Training Setup
    println!("\nPart 3: Training Setup");
    println!("--------------------");

    // Create optimizer
    let optimizer = Box::new(Adam::new(0.001, Some(0.9), Some(0.999), Some(1e-8)));

    // Create loss function
    let loss_fn = Box::new(CrossEntropyLoss::new(Some("mean")));

    // Create a new model instance for the trainer
    // Note: In production, you would implement proper cloning for Sequential
    let new_model = create_model();

    // Create trainer with the new model and optimizer
    // We don't clone the optimizer but create a new one with the same parameters
    let trainer = Trainer::new(
        new_model,
        Box::new(Adam::new(0.001, Some(0.9), Some(0.999), Some(1e-8))),
        loss_fn,
    );

    println!("Created trainer with Adam optimizer and CrossEntropyLoss");

    // Create distributed trainer
    let _dist_trainer = DistributedTrainingFactory::create_trainer(trainer, dist_config.clone());

    println!("Created distributed trainer");

    // Part 4: Model Serialization
    println!("\nPart 4: Model Serialization");
    println!("-------------------------");

    // Create a temporary directory for saving models
    let temp_dir = tempdir().unwrap();
    let model_dir = temp_dir.path().join("models");

    println!("Created model directory at: {}", model_dir.display());

    // Create model serializer
    let serializer = ModelSerializer::new(&model_dir);

    // Save model
    let model_path =
        serializer.save_model(&model, "example_model", "v1.0", Some(optimizer.as_ref()));

    match model_path {
        Ok(path) => println!("Saved model to: {}", path.display()),
        Err(e) => println!("Error saving model: {}", e),
    }

    // Load model
    let loaded_model = serializer.load_model("example_model", "v1.0");

    match loaded_model {
        Ok((model, optimizer)) => {
            println!("Loaded model with {} layers", model.layers().len());
            println!(
                "Loaded optimizer: {}",
                if optimizer.is_some() { "yes" } else { "no" }
            );
        }
        Err(e) => println!("Error loading model: {}", e),
    }

    // Part 5: Checkpoint Management
    println!("\nPart 5: Checkpoint Management");
    println!("---------------------------");

    // Create metrics
    let mut metrics = HashMap::new();
    metrics.insert("loss".to_string(), 0.5);
    metrics.insert("accuracy".to_string(), 0.85);

    // Save checkpoint
    let checkpoint_path = model_dir.join("checkpoint");
    let result = save_checkpoint(
        &model,
        optimizer.as_ref(),
        &checkpoint_path,
        10,
        metrics.clone(),
    );

    match result {
        Ok(()) => println!("Saved checkpoint at epoch 10"),
        Err(e) => println!("Error saving checkpoint: {}", e),
    }

    // Load checkpoint
    let result = load_checkpoint(&checkpoint_path);

    match result {
        Ok((model, _optimizer, epoch, metrics)) => {
            println!("Loaded checkpoint from epoch {}", epoch);
            println!("Loaded model with {} layers", model.layers().len());
            println!(
                "Metrics: loss = {}, accuracy = {}",
                metrics.get("loss").unwrap_or(&0.0),
                metrics.get("accuracy").unwrap_or(&0.0)
            );
        }
        Err(e) => println!("Error loading checkpoint: {}", e),
    }

    // Part 6: ONNX Export
    println!("\nPart 6: ONNX Export");
    println!("-----------------");

    // Export model to ONNX
    let onnx_path = model_dir.join("model.onnx");
    let result = OnnxExporter::export_model(&model, &onnx_path, &[1, 3, 224, 224]);

    match result {
        Ok(()) => println!("Exported model to ONNX format at: {}", onnx_path.display()),
        Err(e) => println!("Error exporting model to ONNX: {}", e),
    }

    println!("\nDistributed Training and Model Serialization Example completed successfully!");
}

/// Create a simple model for demonstration purposes.
fn create_model() -> Sequential {
    let mut model = Sequential::new("SimpleModel", Vec::new());

    // Add layers
    model.add_layer(Box::new(Conv2D::with_shape(
        "conv1",
        3,
        3, // Filter size
        3,
        16,     // In/out channels
        (1, 1), // Stride
        (1, 1), // Padding
        true,   // With bias
        Some(ActivationFunc::ReLU),
    )));

    model.add_layer(Box::new(MaxPool2D::new(
        "pool1",
        (2, 2), // Kernel size
        None,   // Stride (default to kernel size)
        (0, 0), // Padding
    )));

    model.add_layer(Box::new(Conv2D::with_shape(
        "conv2",
        3,
        3, // Filter size
        16,
        32,     // In/out channels
        (1, 1), // Stride
        (1, 1), // Padding
        true,   // With bias
        Some(ActivationFunc::ReLU),
    )));

    model.add_layer(Box::new(MaxPool2D::new(
        "pool2",
        (2, 2), // Kernel size
        None,   // Stride (default to kernel size)
        (0, 0), // Padding
    )));

    model.add_layer(Box::new(Linear::with_shape(
        "fc1",
        32 * 6 * 6, // Input features (assuming input size is 28x28)
        128,        // Output features
        true,       // With bias
        Some(ActivationFunc::ReLU),
    )));

    model.add_layer(Box::new(Linear::with_shape(
        "fc_out", 128,  // Input features
        10,   // Output features
        true, // With bias
        None, // No activation (will be applied in loss function)
    )));

    model
}

/// Create a simple dataset for demonstration purposes.
fn create_dataset() -> (InMemoryDataset, InMemoryDataset) {
    // Generate random data
    let num_samples = 1000;
    let num_features = 3 * 28 * 28; // 3 channels, 28x28 image
    let num_classes = 10;

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

    // Split into train/val
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

    // Create datasets
    let train_dataset = InMemoryDataset::from_arrays(train_inputs, train_targets);
    let val_dataset = InMemoryDataset::from_arrays(val_inputs, val_targets);

    (train_dataset, val_dataset)
}
