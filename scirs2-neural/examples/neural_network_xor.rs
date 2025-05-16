use ndarray::{Array, IxDyn};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use scirs2_neural::error::Result;
use scirs2_neural::layers::Dense;
use scirs2_neural::losses::MeanSquaredError;
use scirs2_neural::models::{Model, Sequential};
use scirs2_neural::optimizers::SGD;

fn main() -> Result<()> {
    println!("Simple Neural Network Example");

    // Create a simple dataset for XOR problem
    //   Input: (x1, x2)
    //   Output: x1 XOR x2
    let inputs = Array::from_shape_vec(
        IxDyn(&[4, 2]),
        vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    )?;
    let targets = Array::from_shape_vec(IxDyn(&[4, 1]), vec![0.0f32, 1.0, 1.0, 0.0])?;

    println!("XOR problem dataset:");
    println!("Inputs:\n{:?}", inputs);
    println!("Targets:\n{:?}", targets);

    // Create a neural network model
    // XOR is not linearly separable, so we need at least one hidden layer
    let mut rng = SmallRng::seed_from_u64(42);
    let mut model = create_model(&mut rng)?;

    // Create loss function and optimizer
    let loss_fn = MeanSquaredError::new();
    let mut optimizer = SGD::new(0.05f32); // Learning rate of 0.05

    // Training loop
    let num_epochs = 5000;
    println!("\nTraining for {} epochs", num_epochs);

    for epoch in 0..num_epochs {
        // Perform a single training step
        let loss = model.train_batch(&inputs, &targets, &loss_fn, &mut optimizer)?;

        // Print progress every 500 epochs
        if epoch % 500 == 0 || epoch == num_epochs - 1 {
            println!("Epoch {}/{}: loss = {:.6}", epoch + 1, num_epochs, loss);
        }
    }

    // Evaluate the model
    println!("\nEvaluation:");
    let predictions = model.predict(&inputs)?;
    println!("Predictions:\n{:.3?}", predictions);

    // Test with specific inputs
    println!("\nTesting with specific inputs:");

    let test_cases = vec![
        (0.0f32, 0.0f32),
        (0.0f32, 1.0f32),
        (1.0f32, 0.0f32),
        (1.0f32, 1.0f32),
    ];

    for (x1, x2) in test_cases {
        let test_input = Array::from_shape_vec(IxDyn(&[1, 2]), vec![x1, x2])?;
        let prediction = model.predict(&test_input)?;
        println!(
            "Input: [{:.1}, {:.1}], Predicted: {:.3}, Expected: {:.1}",
            x1,
            x2,
            prediction[[0, 0]],
            if (x1 == 1.0 && x2 == 0.0) || (x1 == 0.0 && x2 == 1.0) {
                1.0
            } else {
                0.0
            }
        );
    }

    Ok(())
}

// Create a simple feed-forward neural network for the XOR problem
fn create_model<R: Rng>(rng: &mut R) -> Result<impl Model<f32>> {
    // Create a sequential model
    let mut model = Sequential::new();

    // First layer: 2 inputs -> 4 hidden neurons with ReLU activation
    // 2 inputs (x1, x2)
    let layer1 = Dense::new(2, 4, Some("relu"), rng)?;
    model.add_layer(layer1);

    // Output layer: 4 hidden neurons -> 1 output (no activation for regression)
    // No activation for simple regression
    let layer2 = Dense::new(4, 1, None, rng)?;
    model.add_layer(layer2);

    Ok(model)
}
