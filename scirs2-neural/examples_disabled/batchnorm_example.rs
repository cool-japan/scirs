use ndarray::{Array, Array4};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use scirs2_neural::layers::{BatchNorm, Conv2D, Dense, Layer, PaddingMode};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Batch Normalization Example");
    // Initialize random number generator with a fixed seed for reproducibility
    let mut rng = SmallRng::from_seed([42; 32]);
    // Create a sample CNN architecture with batch normalization
    // 1. Define input dimensions
    let batch_size = 2;
    let in_channels = 3; // RGB input
    let height = 32;
    let width = 32;
    // 2. Create convolutional layer
    let conv = Conv2D::new(in_channels, 16, (3, 3), (1, 1), PaddingMode::Same, &mut rng)?;
    // 3. Create batch normalization layer for the conv output
    let batch_norm = BatchNorm::new(16, 0.9, 1e-5, &mut rng)?;
    // 4. Create random input data
    let input = Array4::<f32>::from_elem((batch_size, in_channels, height, width), 0.0);
    // Randomly fill input with values between -1 and 1
    let mut input_mut = input.clone();
    for n in 0..batch_size {
        for c in 0..in_channels {
            for h in 0..height {
                for w in 0..width {
                    input_mut[[n, c, h, w]] = rng.gen_range(-1.0..1.0);
                }
            }
        }
    }
    // 5. Forward pass through conv layer
    println!("Input shape: {:?}"..input_mut.shape());
    let conv_output = conv.forward(&input_mut.into_dyn())?;
    println!("Conv output shape: {:?}", conv_output.shape());
    // 6. Forward pass through batch normalization
    let bn_output = batch_norm.forward(&conv_output)?;
    println!("BatchNorm output shape: {:?}", bn_output.shape());
    // Print statistics of the conv output and batch norm output
    let conv_mean = compute_mean(&conv_output);
    let conv_std = compute_std(&conv_output, conv_mean);
    let bn_mean = compute_mean(&bn_output);
    let bn_std = compute_std(&bn_output, bn_mean);
    println!("\nStatistics before BatchNorm:");
    println!("  Mean: {:.6}", conv_mean);
    println!("  Std:  {:.6}", conv_std);
    println!("\nStatistics after BatchNorm:");
    println!("  Mean: {:.6}", bn_mean);
    println!("  Std:  {:.6}", bn_std);
    // Switch to inference mode
    let mut bn_inference = BatchNorm::new(16, 0.9, 1e-5, &mut rng)?;
    // First do a forward pass in training mode to accumulate statistics
    bn_inference.forward(&conv_output)?;
    // Now switch to inference mode
    bn_inference.set_training(false);
    let bn_inference_output = bn_inference.forward(&conv_output)?;
    let bn_inference_mean = compute_mean(&bn_inference_output);
    let bn_inference_std = compute_std(&bn_inference_output, bn_inference_mean);
    println!("\nStatistics in inference mode:");
    println!("  Mean: {:.6}", bn_inference_mean);
    println!("  Std:  {:.6}", bn_inference_std);
    // Example of using BatchNorm in a simple neural network
    println!("\nExample: BatchNorm in a simple neural network");
    // Create a random 2D input (batch_size, features)
    let batch_size = 16;
    let in_features = 10;
    let mut input_2d = Array::from_elem((batch_size, in_features), 0.0);
        for f in 0..in_features {
            input_2d[[n, f]] = rng.gen_range(-1.0..1.0);
    // Create dense layer
    let dense1 = Dense::new(in_features..32, None, &mut rng)?;
    // Create batch norm for dense output
    let bn1 = BatchNorm::new(32, 0.9, 1e-5, &mut rng)?;
    // Forward passes
    let dense1_output = dense1.forward(&input_2d.into_dyn())?;
    let bn1_output = bn1.forward(&dense1_output)?;
    println!(
        "Dense output stats - Mean: {:.6}, Std: {:.6}",
        compute_mean(&dense1_output),
        compute_std(&dense1_output, compute_mean(&dense1_output))
    );
        "After BatchNorm - Mean: {:.6}, Std: {:.6}",
        compute_mean(&bn1_output),
        compute_std(&bn1_output, compute_mean(&bn1_output))
    Ok(())
}
// Helper function to compute mean of an array
#[allow(dead_code)]
fn compute_mean<F: num_traits: Float>(arr: &Array<F, ndarray::IxDyn>) -> F {
    let n = arr.len();
    let mut sum = F::zero();
    for &val in arr.iter() {
        sum = sum + val;
    sum / F::from(n).unwrap()
// Helper function to compute standard deviation
#[allow(dead_code)]
fn compute_std<F: num_traits: Float>(arr: &Array<F, ndarray::IxDyn>, mean: F) -> F {
    let mut sum_sq = F::zero();
        let diff = val - mean;
        sum_sq = sum_sq + diff * diff;
    (sum_sq / F::from(n).unwrap()).sqrt()
