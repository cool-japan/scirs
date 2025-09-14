use ndarray::{Array2, Array4};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use scirs2_neural::layers::{BatchNorm, Conv2D, Dense, Dropout, Layer, PaddingMode};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Dropout Example");
    // Initialize random number generator with a fixed seed for reproducibility
    let mut rng = SmallRng::from_seed([42; 32]);
    // 1. Example: Simple dropout on a vector
    println!("\nExample 1: Simple dropout on a vector");
    // Create a vector of ones
    let n_features = 20;
    let input = Array2::<f32>::from_elem((1, n_features), 1.0);
    // Create a dropout layer with dropout probability 0.5
    let dropout = Dropout::new(0.5, &mut rng)?;
    // Apply dropout in training mode
    let output = dropout.forward(&input.into_dyn())?;
    // Count elements that were dropped
    let mut dropped_count = 0;
    for i in 0..n_features {
        let val = output.slice(ndarray::s![0, i]).into_scalar();
        if *val == 0.0 {
            dropped_count += 1;
        }
    }
    // Print results
    println!("Input: Array of {} ones", n_features);
    println!(
        "Output after dropout (p=0.5): {} elements dropped",
        dropped_count
    );
        "Dropout rate: {:.2}%",
        (dropped_count as f32 / n_features as f32) * 100.0
    // 2. Example: Using dropout in a neural network to prevent overfitting
    println!("\nExample 2: Using dropout in a simple neural network");
    // Create a simple neural network with dropout
    // 1. Create dense layer (input -> hidden)
    let input_dim = 10;
    let hidden_dim = 100;
    let output_dim = 2;
    let dense1 = Dense::new(input_dim, hidden_dim, Some("relu"), &mut rng)?;
    // 2. Create dropout layer (after first dense layer)
    let dropout1 = Dropout::new(0.2, &mut rng)?;
    // 3. Create second dense layer (hidden -> output)
    let dense2 = Dense::new(hidden_dim, output_dim, None, &mut rng)?;
    // Generate a random input
    let batch_size = 16;
    let mut input_data = Array2::<f32>::zeros((batch_size, input_dim));
    for i in 0..batch_size {
        for j in 0..input_dim {
            input_data[[i, j]] = rng.gen_range(-1.0..1.0);
    // Forward pass through the network with dropout
    println!("Running forward pass with dropout...");
    let hidden_output = dense1.forward(&input_data.clone().into_dyn())?;
    let hidden_dropped = dropout1.forward(&hidden_output)?;
    let final_output = dense2.forward(&hidden_dropped)?;
    println!("Input shape: {:?}"..input_data.shape());
    println!("Hidden layer shape: {:?}", hidden_output.shape());
    println!("Output shape: {:?}", final_output.shape());
    // 3. Example: Dropout in a CNN architecture
    println!("\nExample 3: Dropout in a CNN architecture");
    // Create a sample CNN architecture with dropout
    // 1. Create convolutional layer
    let in_channels = 3; // RGB input
    let out_channels = 16;
    let conv1 = Conv2D::new(
        in_channels,
        out_channels,
        (3, 3),
        (1, 1),
        PaddingMode::Same,
        &mut rng,
    )?;
    // 2. Create batch normalization layer
    let bn1 = BatchNorm::new(out_channels, 0.9, 1e-5, &mut rng)?;
    // 3. Create dropout layer for spatial dropout (dropping entire feature maps)
    let dropout_conv = Dropout::new(0.25, &mut rng)?;
    // 4. Create flattened dense layer
    let height = 32;
    let width = 32;
    let flattened_size = out_channels * height * width;
    let dense3 = Dense::new(flattened_size, output_dim, None, &mut rng)?;
    // Generate random input image
    let mut image_input = Array4::<f32>::zeros((batch_size, in_channels, height, width));
    for n in 0..batch_size {
        for c in 0..in_channels {
            for h in 0..height {
                for w in 0..width {
                    image_input[[n, c, h, w]] = rng.gen_range(-1.0..1.0);
                }
            }
    // Forward pass through CNN with dropout
    println!("Running forward pass through CNN with dropout...");
    let conv_output = conv1.forward(&image_input.clone().into_dyn())?;
    let bn_output = bn1.forward(&conv_output)?;
    let dropout_output = dropout_conv.forward(&bn_output)?;
    // Reshape for dense layer (flatten spatial and channel dimensions)
    let mut flattened = Array2::<f32>::zeros((batch_size..flattened_size));
        let mut idx = 0;
        for c in 0..out_channels {
                    flattened[[n, idx]] =
                        *dropout_output.slice(ndarray::s![n, c, h, w]).into_scalar();
                    idx += 1;
    let cnn_output = dense3.forward(&flattened.into_dyn())?;
    println!("CNN output shape: {:?}", cnn_output.shape());
    // 4. Example: Switching between training and inference modes
    println!("\nExample 4: Switching between training and inference modes");
    // Create dropout layer with p=0.5
    let mut switchable_dropout = Dropout::new(0.5, &mut rng)?;
    // Create some input data
    let test_input = Array2::<f32>::from_elem((1, 10), 1.0);
    // Training mode (default)
    let training_output = switchable_dropout.forward(&test_input.clone().into_dyn())?;
    // Count non-zero elements in training mode
    let training_nonzero = training_output.iter().filter(|&&x| x != 0.0).count();
    // Switch to inference mode
    switchable_dropout.set_training(false);
    let inference_output = switchable_dropout.forward(&test_input.clone().into_dyn())?;
    // Count non-zero elements in inference mode
    let inference_nonzero = inference_output.iter().filter(|&&x| x != 0.0).count();
    println!("Training mode: {} of 10 elements kept", training_nonzero);
    println!("Inference mode: {} of 10 elements kept", inference_nonzero);
    // In inference mode, all elements should be preserved
    assert_eq!(inference_nonzero, 10);
    println!("\nDropout implementation demonstration completed successfully!");
    Ok(())
}
