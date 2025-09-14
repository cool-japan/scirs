use ndarray::Array2;
use rand::prelude::*;
use rand::rngs::SmallRng;
use scirs2_neural::error::Result;
use scirs2_neural::layers::{Dense, Dropout};
use scirs2_neural::losses::CrossEntropyLoss;
use scirs2_neural::models::{Model, Sequential};
use scirs2_neural::optimizers::Adam;
use scirs2_neural::serialization::{self, SerializationFormat};
use std::path::Path;

// Create a simple neural network model for the XOR problem
#[allow(dead_code)]
fn create_xor_model(rng: &mut SmallRng) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    // XOR problem requires a hidden layer
    let input_dim = 2;
    let hidden_dim = 4;
    let output_dim = 1;
    // Input to hidden layer with ReLU activation
    let dense1 = Dense::new(input_dim, hidden_dim, Some("relu"), rng)?;
    model.add_layer(dense1);
    // Optional dropout for regularization (low rate as XOR is small)
    let dropout = Dropout::new(0.1, rng)?;
    model.add_layer(dropout);
    // Hidden to output layer with sigmoid activation (binary output)
    let dense2 = Dense::new(hidden_dim, output_dim, Some("sigmoid"), rng)?;
    model.add_layer(dense2);
    Ok(model)
}
// Create XOR dataset
#[allow(dead_code)]
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
            0.0, // 0 XOR 0 = 0
            1.0, // 0 XOR 1 = 1
            1.0, // 1 XOR 0 = 1
            0.0, // 1 XOR 1 = 0
    (x, y)
// Train the model on XOR problem
#[allow(dead_code)]
fn train_model(
    model: &mut Sequential<f32>,
    x: &Array2<f32>,
    y: &Array2<f32>,
    epochs: usize,
) -> Result<()> {
    println!("Training XOR model...");
    // Setup loss function and optimizer
    let loss_fn = CrossEntropyLoss::new(1e-10);
    let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
    // Train for specified number of epochs
    for epoch in 0..epochs {
        // Convert to dynamic dimension arrays
        let x_dyn = x.clone().into_dyn();
        let y_dyn = y.clone().into_dyn();
        // Perform a training step
        let loss = model.train_batch(&x_dyn, &y_dyn, &loss_fn, &mut optimizer)?;
        // Print progress every 100 epochs
        if epoch % 100 == 0 || epoch == epochs - 1 {
            println!("Epoch {}/{}: loss = {:.6}", epoch + 1, epochs, loss);
        }
    }
    println!("Training completed.");
    Ok(())
// Evaluate model performance on XOR problem
#[allow(dead_code)]
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
        println!(
            " {:.4}  | {:.4}  | {:.4}  |   {:.4}   |  {}  {}",
            x[[i, 0]],
            x[[i, 1]],
            target,
            pred,
            binary_pred as i32,
            if is_correct { "✓" } else { "✗" }
        );
    let accuracy = correct as f32 / x.shape()[0] as f32;
    println!(
        "\nAccuracy: {:.2}% ({}/{})",
        accuracy * 100.0,
        correct,
        x.shape()[0]
    );
    Ok(accuracy)
// A more realistic dataset with noise to better test serialization
#[allow(dead_code)]
fn create_noisy_xor_dataset(
    size: usize,
    noise_level: f32,
    rng: &mut SmallRng,
) -> (Array2<f32>, Array2<f32>) {
    let mut x = Array2::<f32>::zeros((size, 2));
    let mut y = Array2::<f32>::zeros((size, 1));
    for i in 0..size {
        // Generate binary inputs with some randomness
        let x1 = (rng.gen_range(0.0..1.0) > 0.5) as i32 as f32;
        let x2 = (rng.gen_range(0.0..1.0) > 0.5) as i32 as f32;
        // Add some noise to inputs
        x[[i..0]] = x1 + rng.gen_range(-noise_level / 2.0..noise_level / 2.0);
        x[[i, 1]] = x2 + rng.gen_range(-noise_level / 2.0..noise_level / 2.0);
        // Standard XOR calculation for target
        y[[i..0]] = (x1 as i32 ^ x2 as i32) as f32;
#[allow(dead_code)]
fn main() -> Result<()> {
    println!("Improved Model Serialization and Loading Example");
    println!("===============================================\n");
    // Initialize random number generator
    let mut rng = SmallRng::from_seed([42; 32]);
    // 1. Create XOR datasets
    let (x_train, y_train) = create_xor_dataset();
    println!("XOR dataset created");
    // 2. Create and train the model
    let mut model = create_xor_model(&mut rng)?;
    println!("Model created with {} layers", model.num_layers());
    // Train the model
    train_model(&mut model, &x_train, &y_train, 2000)?;
    // 3. Evaluate the model before saving
    println!("\nEvaluating model before saving:");
    evaluate_model(&model, &x_train, &y_train)?;
    // 4. Save the model in different formats
    println!("\nSaving model in different formats...");
    // Save in JSON format (human-readable)
    let json_path = Path::new("xor_model.json");
    serialization::save_model(&model, json_path, SerializationFormat::JSON)?;
    println!("Model saved to {} in JSON format", json_path.display());
    // Save in CBOR format (compact binary)
    let cbor_path = Path::new("xor_model.cbor");
    serialization::save_model(&model, cbor_path, SerializationFormat::CBOR)?;
    println!("Model saved to {} in CBOR format", cbor_path.display());
    // Save in MessagePack format (efficient binary)
    let msgpack_path = Path::new("xor_model.msgpack");
    serialization::save_model(&model, msgpack_path, SerializationFormat::MessagePack)?;
        "Model saved to {} in MessagePack format",
        msgpack_path.display()
    // 5. Load models from each format and evaluate
    println!("\nLoading and evaluating models from each format:");
    // Load and evaluate JSON model
    println!("\n--- JSON Format ---");
    let json_model = serialization::load_model::<f32>(json_path, SerializationFormat::JSON)?;
    println!("JSON model loaded with {} layers", json_model.num_layers());
    evaluate_model(&json_model, &x_train, &y_train)?;
    // Load and evaluate CBOR model
    println!("\n--- CBOR Format ---");
    let cbor_model = serialization::load_model::<f32>(cbor_path, SerializationFormat::CBOR)?;
    println!("CBOR model loaded with {} layers", cbor_model.num_layers());
    evaluate_model(&cbor_model, &x_train, &y_train)?;
    // Load and evaluate MessagePack model
    println!("\n--- MessagePack Format ---");
    let msgpack_model =
        serialization::load_model::<f32>(msgpack_path, SerializationFormat::MessagePack)?;
        "MessagePack model loaded with {} layers",
        msgpack_model.num_layers()
    evaluate_model(&msgpack_model, &x_train, &y_train)?;
    // 6. Test with a larger, noisy dataset to verify model works with unseen data
    println!("\nTesting with larger, noisy dataset:");
    let (x_test, y_test) = create_noisy_xor_dataset(100, 0.2, &mut rng);
    evaluate_model(&model, &x_test, &y_test)?;
    // File sizes for comparison
    let json_size = std::fs::metadata(json_path)?.len();
    let cbor_size = std::fs::metadata(cbor_path)?.len();
    let msgpack_size = std::fs::metadata(msgpack_path)?.len();
    println!("\nSerialization Format Comparison:");
    println!("  JSON:       {} bytes", json_size);
        "  CBOR:       {} bytes ({:.1}% of JSON)",
        cbor_size,
        (cbor_size as f64 / json_size as f64) * 100.0
        "  MessagePack: {} bytes ({:.1}% of JSON)",
        msgpack_size,
        (msgpack_size as f64 / json_size as f64) * 100.0
    println!("\nModel serialization and loading example completed successfully!");
