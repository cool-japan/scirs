use ndarray::{Array2, Axis};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use scirs2_neural::error::Result;

/// Simple neural network implementation for XOR problem
struct XORNetwork {
    // First layer: 2 inputs -> 4 hidden
    w1: Array2<f32>,
    b1: Array2<f32>,
    // Second layer: 4 hidden -> 1 output
    w2: Array2<f32>,
    b2: Array2<f32>,
    // For storing intermediate values during forward/backward passes
    z1: Option<Array2<f32>>,
    a1: Option<Array2<f32>>,
    z2: Option<Array2<f32>>,
    a2: Option<Array2<f32>>,
}

impl XORNetwork {
    fn new() -> Self {
        let mut rng = SmallRng::seed_from_u64(42);

        // Initialize weights with small random values
        // First layer: 2 inputs -> 4 hidden
        let w1 = Array2::from_shape_fn((2, 4), |_| rng.random_range(-0.5..0.5));
        let b1 = Array2::from_shape_fn((1, 4), |_| rng.random_range(-0.5..0.5));

        // Second layer: 4 hidden -> 1 output
        let w2 = Array2::from_shape_fn((4, 1), |_| rng.random_range(-0.5..0.5));
        let b2 = Array2::from_shape_fn((1, 1), |_| rng.random_range(-0.5..0.5));

        Self {
            w1,
            b1,
            w2,
            b2,
            z1: None,
            a1: None,
            z2: None,
            a2: None,
        }
    }

    /// ReLU activation function
    fn relu(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v.max(0.0))
    }

    /// Derivative of ReLU
    fn relu_derivative(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }

    /// Forward pass through the network
    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        // First layer: z1 = x @ w1 + b1
        let z1 = x.dot(&self.w1) + &self.b1;
        // Apply ReLU activation: a1 = relu(z1)
        let a1 = Self::relu(&z1);

        // Second layer: z2 = a1 @ w2 + b2
        let z2 = a1.dot(&self.w2) + &self.b2;
        // Linear activation for output (for regression): a2 = z2
        let a2 = z2.clone();

        // Store activations for backward pass
        self.z1 = Some(z1);
        self.a1 = Some(a1);
        self.z2 = Some(z2);
        self.a2 = Some(a2.clone());

        a2
    }

    /// Mean squared error loss
    fn mse_loss(&self, y_pred: &Array2<f32>, y_true: &Array2<f32>) -> f32 {
        let diff = y_pred - y_true;
        let squared = diff.mapv(|v| v * v);
        squared.sum() / (y_pred.len() as f32)
    }

    /// Backward pass and parameter update
    fn backward(&mut self, x: &Array2<f32>, y: &Array2<f32>, learning_rate: f32) {
        // Make sure we have the activations from the forward pass
        let a1 = self.a1.as_ref().expect("Forward pass must be called first");
        let a2 = self.a2.as_ref().expect("Forward pass must be called first");
        let z1 = self.z1.as_ref().expect("Forward pass must be called first");

        // Output layer error: dL/dz2 = 2 * (a2 - y) / N
        let dz2 = (a2 - y) * (2.0 / (y.len() as f32));

        // Gradient for W2: dL/dW2 = a1.T @ dz2
        let dw2 = a1.t().dot(&dz2);

        // Gradient for b2: dL/db2 = sum(dz2, axis=0)
        let db2 = dz2.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Hidden layer error: dL/dz1 = dz2 @ W2.T * relu'(z1)
        let dz1 = dz2.dot(&self.w2.t()) * Self::relu_derivative(z1);

        // Gradient for W1: dL/dW1 = X.T @ dz1
        let dw1 = x.t().dot(&dz1);

        // Gradient for b1: dL/db1 = sum(dz1, axis=0)
        let db1 = dz1.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Update parameters
        self.w1 = &self.w1 - learning_rate * dw1;
        self.b1 = &self.b1 - learning_rate * db1;
        self.w2 = &self.w2 - learning_rate * dw2;
        self.b2 = &self.b2 - learning_rate * db2;
    }

    /// Train the network on the given inputs and targets
    fn train(&mut self, x: &Array2<f32>, y: &Array2<f32>, learning_rate: f32, epochs: usize) {
        for epoch in 0..epochs {
            // Forward pass
            let y_pred = self.forward(x);

            // Calculate loss
            let loss = self.mse_loss(&y_pred, y);

            // Print loss occasionally
            if epoch % 1000 == 0 || epoch == epochs - 1 {
                println!("Epoch {}/{}: loss = {:.6}", epoch + 1, epochs, loss);
            }

            // Backward pass and parameter update
            self.backward(x, y, learning_rate);
        }
    }
}

fn main() -> Result<()> {
    println!("Simple XOR Neural Network Example");

    // Create XOR dataset
    let x = Array2::from_shape_vec((4, 2), vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])?;
    let y = Array2::from_shape_vec((4, 1), vec![0.0f32, 1.0, 1.0, 0.0])?;

    println!("XOR problem dataset:");
    println!("Inputs:\n{:?}", x);
    println!("Targets:\n{:?}", y);

    // Create and train network
    let mut network = XORNetwork::new();
    println!("\nTraining for 10000 epochs with learning rate 0.1");
    network.train(&x, &y, 0.1, 10000);

    // Evaluate on training data
    let predictions = network.forward(&x);
    println!("\nEvaluation:");
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
        let input = Array2::from_shape_vec((1, 2), vec![x1, x2])?;
        let prediction = network.forward(&input);
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
