use ndarray::{Array, IxDyn};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use scirs2_neural::error::Result;
use scirs2_neural::layers::{Dense, Layer};
use scirs2_neural::losses::{Loss, MeanSquaredError};

fn main() -> Result<()> {
    println!("Manual XOR Neural Network Example");

    // Create a simple dataset for XOR problem
    let inputs = Array::from_shape_vec(
        IxDyn(&[4, 2]),
        vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    )?;
    let targets = Array::from_shape_vec(IxDyn(&[4, 1]), vec![0.0f32, 1.0, 1.0, 0.0])?;

    println!("XOR problem dataset:");
    println!("Inputs:\n{:?}", inputs);
    println!("Targets:\n{:?}", targets);

    // Create neural network layers
    let mut rng = SmallRng::seed_from_u64(42);
    let mut hidden_layer = Dense::new(2, 4, Some("relu"), &mut rng)?;
    let mut output_layer = Dense::new(4, 1, None, &mut rng)?;

    // Create loss function
    let loss_fn = MeanSquaredError::new();

    // Training parameters
    let learning_rate = 0.5f32;
    let num_epochs = 10000;

    println!("\nTraining for {} epochs", num_epochs);

    for epoch in 0..num_epochs {
        // Forward pass through the network
        let hidden_output = hidden_layer.forward(&inputs)?;
        let final_output = output_layer.forward(&hidden_output)?;

        // Compute loss
        let loss = loss_fn.forward(&final_output, &targets)?;

        if epoch % 500 == 0 || epoch == num_epochs - 1 {
            println!("Epoch {}/{}: loss = {:.6}", epoch + 1, num_epochs, loss);
        }

        // Backward pass
        let output_grad = loss_fn.backward(&final_output, &targets)?;
        let hidden_grad = output_layer.backward(&hidden_output, &output_grad)?;
        let _input_grad = hidden_layer.backward(&inputs, &hidden_grad)?;

        // Update parameters
        hidden_layer.update(learning_rate)?;
        output_layer.update(learning_rate)?;
    }

    // Evaluate the model
    println!("\nEvaluation:");
    let hidden_output = hidden_layer.forward(&inputs)?;
    let final_output = output_layer.forward(&hidden_output)?;
    println!("Predictions:\n{:.3?}", final_output);

    // Test with individual inputs
    println!("\nTesting with specific inputs:");

    let test_cases = vec![
        (0.0f32, 0.0f32),
        (0.0f32, 1.0f32),
        (1.0f32, 0.0f32),
        (1.0f32, 1.0f32),
    ];

    for (x1, x2) in test_cases {
        let test_input = Array::from_shape_vec(IxDyn(&[1, 2]), vec![x1, x2])?;
        let hidden_output = hidden_layer.forward(&test_input)?;
        let prediction = output_layer.forward(&hidden_output)?;
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
