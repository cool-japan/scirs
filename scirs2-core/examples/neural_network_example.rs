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

//! Example demonstrating neural network implementation using the array protocol.

use ndarray::{Array, Array1, Array2, Ix2, Ix4};
use scirs2_core::array_protocol::{
    self,
    ml_ops::ActivationFunc,
    neural::{create_simple_cnn, Conv2D, Dropout, Layer, Linear, MaxPool2D, Sequential},
    GPUBackend, GPUConfig, GPUNdarray, NdarrayWrapper,
};

fn main() {
    // Initialize the array protocol system
    array_protocol::init();

    println!("Neural Network Example using Array Protocol");
    println!("==========================================");

    // Part 1: Creating and Using Layers
    println!("\nPart 1: Creating and Using Layers");
    println!("--------------------------------");

    // Create a linear layer
    let weights = Array2::<f64>::eye(3);
    println!("Weights: {:?}", weights);

    let bias = Array1::<f64>::ones(3);
    println!("Bias: {:?}", bias);

    let linear = Linear::new(
        "linear1",
        Box::new(NdarrayWrapper::new(weights.clone())),
        Some(Box::new(NdarrayWrapper::new(bias.clone()))),
        Some(ActivationFunc::ReLU),
    );

    println!("Created linear layer: {}", linear.name());

    // Create a random input
    let input = Array2::<f64>::ones((1, 3));
    println!("Input: {:?}", input);

    // Forward pass through linear layer
    let input_wrapped = NdarrayWrapper::new(input.clone());
    match linear.forward(&input_wrapped) {
        Ok(output) => {
            if let Some(output_array) = output.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
                println!("Output from linear layer: {:?}", output_array.as_array());
            } else {
                println!("Output is not of expected type");
            }
        }
        Err(e) => println!("Error in forward pass: {}", e),
    }

    // Create a convolutional layer
    println!("\nCreating and using a convolutional layer:");

    let filters = Array::<f64, _>::ones((3, 3, 1, 6));
    println!("Filters shape: {:?}", filters.shape());

    let conv = Conv2D::new(
        "conv1",
        Box::new(NdarrayWrapper::new(filters)),
        None,
        (1, 1),
        (1, 1),
        Some(ActivationFunc::ReLU),
    );

    println!("Created convolutional layer: {}", conv.name());

    // Create a 4D input (batch_size, height, width, channels)
    let input_4d = Array::<f64, _>::ones((1, 28, 28, 1));
    println!("Input shape: {:?}", input_4d.shape());

    // Forward pass through convolutional layer
    let input_wrapped = NdarrayWrapper::new(input_4d.clone());
    match conv.forward(&input_wrapped) {
        Ok(output) => {
            if let Some(output_array) = output.as_any().downcast_ref::<NdarrayWrapper<f64, Ix4>>() {
                println!(
                    "Output shape from conv layer: {:?}",
                    output_array.as_array().shape()
                );
            } else {
                println!("Output is not of expected type");
            }
        }
        Err(e) => println!("Error in convolutional forward pass: {}", e),
    }

    // Part 2: Creating and Using a Sequential Model
    println!("\nPart 2: Creating and Using a Sequential Model");
    println!("-------------------------------------------");

    // Create a simple CNN model
    let mut model = Sequential::new("SimpleCNN", Vec::new());

    // Add layers to the model
    model.add_layer(Box::new(Conv2D::with_shape(
        "conv1",
        3,
        3, // Filter size
        1,
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

    // Add fully connected layers
    let feature_size = 32 * 6 * 6; // 32 channels, 6x6 spatial dimensions

    model.add_layer(Box::new(Linear::with_shape(
        "fc1",
        feature_size, // Input features
        120,          // Output features
        true,         // With bias
        Some(ActivationFunc::ReLU),
    )));

    model.add_layer(Box::new(Dropout::new(
        "dropout",
        0.5,      // Dropout rate
        Some(42), // Fixed seed for reproducibility
    )));

    model.add_layer(Box::new(Linear::with_shape(
        "fc2",
        120,  // Input features
        84,   // Output features
        true, // With bias
        Some(ActivationFunc::ReLU),
    )));

    model.add_layer(Box::new(Linear::with_shape(
        "fc3", 84,   // Input features
        10,   // Output features (10 classes)
        true, // With bias
        None, // No activation for output layer
    )));

    println!(
        "Created a sequential model with {} layers",
        model.layers().len()
    );

    // Forward pass through the model
    let input_wrapped = NdarrayWrapper::new(input_4d.clone());
    match model.forward(&input_wrapped) {
        Ok(output) => {
            if let Some(output_array) = output.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
                println!(
                    "Output shape from model: {:?}",
                    output_array.as_array().shape()
                );
                println!("Output values: {:?}", output_array.as_array());
            } else {
                println!("Model output is not of expected type");
            }
        }
        Err(e) => println!("Error in model forward pass: {}", e),
    }

    // Part 3: Using the model builder function
    println!("\nPart 3: Using the Model Builder Function");
    println!("--------------------------------------");

    // Create a model using the builder function
    let model = create_simple_cnn((28, 28, 1), 10);

    println!("Created CNN model with {} layers", model.layers().len());

    // Get model parameters
    let params = model.parameters();
    println!("Model has {} parameter tensors", params.len());

    // Check if the model is in training mode
    println!(
        "Model is in training mode: {}",
        model.layers()[0].is_training()
    );

    // Part 4: Using different array types
    println!("\nPart 4: Using Different Array Types");
    println!("---------------------------------");

    // Create a GPU array
    let gpu_config = GPUConfig {
        backend: GPUBackend::CUDA,
        device_id: 0,
        async_ops: false,
        mixed_precision: false,
        memory_fraction: 0.9,
    };

    let gpu_input = GPUNdarray::new(input_4d.clone(), gpu_config);
    println!("Created GPU input array");

    // Forward pass with GPU array
    match model.forward(&gpu_input) {
        Ok(_) => println!("Successfully ran forward pass with GPU array"),
        Err(e) => println!("Error in GPU forward pass: {}", e),
    }

    // Part 5: Evaluation mode
    println!("\nPart 5: Switching to Evaluation Mode");
    println!("----------------------------------");

    // Create a model and set to evaluation mode
    let mut model = create_simple_cnn((28, 28, 1), 10);
    model.eval();

    println!("Switched model to evaluation mode");
    println!(
        "Model is in training mode: {}",
        model.layers()[0].is_training()
    );

    // Forward pass in evaluation mode
    let input_wrapped = NdarrayWrapper::new(input_4d.clone());
    match model.forward(&input_wrapped) {
        Ok(_) => println!("Successfully ran forward pass in evaluation mode"),
        Err(e) => println!("Error in evaluation mode forward pass: {}", e),
    }
}
