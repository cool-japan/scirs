use ndarray::{Array, IxDyn};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use scirs2_neural::error::Result;
use scirs2_neural::layers::{Dense, Layer, ParamLayer};
use scirs2_neural::losses::{Loss, MeanSquaredError};

// Custom implementation of a neural network for the XOR problem
struct XORNetwork {
    hidden_layer: Dense<f32>,
    output_layer: Dense<f32>,
}

impl XORNetwork {
    fn new() -> Result<Self> {
        let mut rng = SmallRng::seed_from_u64(42);

        // Create two layers: 2 inputs -> 5 hidden -> 1 output
        let hidden_layer = Dense::new(2, 5, Some("relu"), &mut rng)?;
        let output_layer = Dense::new(5, 1, None, &mut rng)?;

        Ok(Self {
            hidden_layer,
            output_layer,
        })
    }

    fn forward(&self, x: &Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>> {
        let hidden = self.hidden_layer.forward(x)?;
        self.output_layer.forward(&hidden)
    }

    fn train(
        &mut self,
        inputs: &Array<f32, IxDyn>,
        targets: &Array<f32, IxDyn>,
        learning_rate: f32,
        epochs: usize,
    ) -> Result<()> {
        let loss_fn = MeanSquaredError::new();

        println!("Training for {} epochs", epochs);

        for epoch in 0..epochs {
            // Forward pass
            let hidden_output = self.hidden_layer.forward(inputs)?;
            let final_output = self.output_layer.forward(&hidden_output)?;

            // Compute loss
            let loss = loss_fn.forward(&final_output, targets)?;

            if epoch % 500 == 0 || epoch == epochs - 1 {
                println!("Epoch {}/{}: loss = {:.6}", epoch + 1, epochs, loss);
            }

            // Backward pass
            let output_grad = loss_fn.backward(&final_output, targets)?;
            let hidden_grad = self.output_layer.backward(&hidden_output, &output_grad)?;
            let _input_grad = self.hidden_layer.backward(inputs, &hidden_grad)?;

            // Custom update using direct access to the layers' fields
            // Note: This is non-idiomatic as it breaks layer encapsulation
            // but necessary due to implementation limitations
            custom_update_layer(&mut self.hidden_layer, learning_rate)?;
            custom_update_layer(&mut self.output_layer, learning_rate)?;
        }

        Ok(())
    }
}

fn custom_update_layer(layer: &mut Dense<f32>, learning_rate: f32) -> Result<()> {
    // Use the ParamLayer trait methods to access weights and gradients
    let params = layer.get_parameters();
    let gradients = layer.get_gradients();

    let weights = params[0];
    let biases = params[1];
    let dweights = gradients[0];
    let dbiases = gradients[1];

    // Create new arrays with the updated parameters
    let new_weights = weights - &(dweights * learning_rate);
    let new_biases = biases - &(dbiases * learning_rate);

    // Set the updated parameters
    layer.set_parameters(vec![new_weights, new_biases])?;

    Ok(())
}

fn main() -> Result<()> {
    println!("Improved XOR Neural Network Example");

    // Create XOR dataset
    let inputs = Array::from_shape_vec(
        IxDyn(&[4, 2]),
        vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    )?;
    let targets = Array::from_shape_vec(IxDyn(&[4, 1]), vec![0.0f32, 1.0, 1.0, 0.0])?;

    println!("XOR problem dataset:");
    println!("Inputs:\n{:?}", inputs);
    println!("Targets:\n{:?}", targets);

    // Create and train the network
    let mut network = XORNetwork::new()?;
    network.train(&inputs, &targets, 0.1, 10000)?;

    // Test the trained network
    println!("\nEvaluation:");
    let predictions = network.forward(&inputs)?;
    println!("Predictions:\n{:.3?}", predictions);

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
        let prediction = network.forward(&test_input)?;
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
