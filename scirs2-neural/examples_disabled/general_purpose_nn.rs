//! General-purpose neural network implementation in Rust
//!
//! This example implements a flexible neural network architecture
//! that supports multiple layers, different activation functions,
//! and various loss functions. It's designed to be educational
//! and demonstrate neural network concepts in Rust.

use ndarray::{Array2, Axis};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use scirs2_neural::error::Result;
use std::f32;
/// Activation function type
#[derive(Debug, Clone, Copy)]
enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}
impl ActivationFunction {
    /// Apply the activation function to an array
    fn apply(&self, x: &Array2<f32>) -> Array2<f32> {
        match self {
            ActivationFunction::ReLU => x.mapv(|v| v.max(0.0)),
            ActivationFunction::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            ActivationFunction::Tanh => x.mapv(|v| v.tanh()),
            ActivationFunction::Linear => x.clone(),
        }
    }
    /// Compute the derivative of the activation function
    fn derivative(&self, x: &Array2<f32>) -> Array2<f32> {
            ActivationFunction::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            ActivationFunction::Sigmoid => {
                let sigmoid = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
                sigmoid.mapv(|s| s * (1.0 - s))
            }
            ActivationFunction::Tanh => {
                let tanh = x.mapv(|v| v.tanh());
                tanh.mapv(|t| 1.0 - t * t)
            ActivationFunction::Linear => Array2::ones(x.dim()),
    /// Get a string representation of the activation function
    fn as_str(&self) -> &str {
            ActivationFunction::ReLU => "ReLU",
            ActivationFunction::Sigmoid => "Sigmoid",
            ActivationFunction::Tanh => "Tanh",
            ActivationFunction::Linear => "Linear",
/// Loss function type
#[allow(clippy::upper_case_acronyms)]
enum LossFunction {
    MSE,
    BinaryCrossEntropy,
impl LossFunction {
    /// Compute the loss between predictions and targets
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
            LossFunction::MSE => {
                let diff = predictions - targets;
                let squared = diff.mapv(|v| v * v);
                squared.sum() / (predictions.len() as f32)
            LossFunction::BinaryCrossEntropy => {
                let epsilon = 1e-15; // To avoid log(0)
                let mut sum = 0.0;
                for (y_pred, y_true) in predictions.iter().zip(targets.iter()) {
                    let y_pred_safe = y_pred.max(epsilon).min(1.0 - epsilon);
                    sum += y_true * y_pred_safe.ln() + (1.0 - y_true) * (1.0 - y_pred_safe).ln();
                }
                -sum / (predictions.len() as f32)
    /// Compute the derivative of the loss function with respect to predictions
    fn derivative(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32> {
                // d(MSE)/dŷ = 2(ŷ - y)/n
                let n = predictions.len() as f32;
                (predictions - targets) * (2.0 / n)
                // d(BCE)/dŷ = ((1-y)/(1-ŷ) - y/ŷ)/n
                let epsilon = 1e-15;
                Array2::from_shape_fn(predictions.dim(), |(i, j)| {
                    let y_pred = predictions[(i, j)].max(epsilon).min(1.0 - epsilon);
                    let y_true = targets[(i, j)];
                    ((1.0 - y_true) / (1.0 - y_pred) - y_true / y_pred) / n
                })
    /// Get a string representation of the loss function
            LossFunction::MSE => "Mean Squared Error",
            LossFunction::BinaryCrossEntropy => "Binary Cross Entropy",
/// A layer in the neural network
struct Layer {
    weights: Array2<f32>,
    biases: Array2<f32>,
    activation: ActivationFunction,
    // Cached values for backpropagation
    z: Option<Array2<f32>>,
    a: Option<Array2<f32>>,
impl Layer {
    /// Create a new layer with random weights and biases
    fn new(
        input_size: usize,
        output_size: usize,
        activation: ActivationFunction,
        rng: &mut SmallRng,
    ) -> Self {
        // Xavier/Glorot initialization
        let scale = (1.0 / input_size as f32).sqrt();
        // Initialize weights and biases
        let weights = Array2::from_shape_fn((input_size, output_size), |_| {
            rng.gen_range(-scale..scale)
        });
        let biases = Array2::from_shape_fn((1..output_size), |_| rng.gen_range(-scale..scale));
        Self {
            weights..biases,
            activation,
            z: None,
            a: None,
    /// Forward pass through the layer
    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        // z = x @ w + b
        let z = x.dot(&self.weights) + &self.biases;
        // a = activation(z)
        let a = self.activation.apply(&z);
        // Store for backpropagation
        self.z = Some(z);
        self.a = Some(a.clone());
        a
    /// Backward pass through the layer
    fn backward(
        &mut self,
        input: &Array2<f32>,
        grad_output: &Array2<f32>,
        learning_rate: f32,
    ) -> Array2<f32> {
        let z = self.z.as_ref().expect("Forward pass must be called first");
        // Gradient through activation: dL/dz = dL/da * da/dz
        let dz = grad_output * &self.activation.derivative(z);
        // Gradient for weights: dL/dW = X.T @ dz
        let dw = input.t().dot(&dz);
        // Gradient for biases: dL/db = sum(dz, axis=0)
        let db = dz.sum_axis(Axis(0)).insert_axis(Axis(0));
        // Gradient for previous layer: dL/dX = dz @ W.T
        let dx = dz.dot(&self.weights.t());
        // Update parameters
        self.weights = &self.weights - dw * learning_rate;
        self.biases = &self.biases - db * learning_rate;
        dx
/// A neural network composed of multiple layers
struct NeuralNetwork {
    layers: Vec<Layer>,
    loss_fn: LossFunction,
impl NeuralNetwork {
    /// Create a new neural network with the given layer sizes and activations
        layer_sizes: &[usize],
        activations: &[ActivationFunction],
        loss_fn: LossFunction,
        seed: u64,
        assert!(
            layer_sizes.len() >= 2,
            "Network must have at least input and output layers"
        );
        assert_eq!(
            layer_sizes.len() - 1,
            activations.len(),
            "Number of activations must match number of layers - 1"
        let mut rng = SmallRng::from_seed(seed);
        let mut layers = Vec::with_capacity(layer_sizes.len() - 1);
        // Create layers
        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];
            let activation = activations[i];
            layers.push(Layer::new(input_size, output_size, activation, &mut rng));
        Self { layers, loss_fn }
    /// Forward pass through the network
        let mut output = x.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        output
    /// Compute the loss for given predictions and targets
    fn loss(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        self.loss_fn.compute(predictions, targets)
    /// Backward pass and update parameters
    fn backward(&mut self, x: &Array2<f32>, y: &Array2<f32>, learningrate: f32) {
        // Get the output from the last forward pass
        let output = self.layers.last().unwrap().a.as_ref().unwrap();
        // Compute loss derivative
        let mut grad = self.loss_fn.derivative(output, y);
        // Store inputs for layers
        let mut inputs = Vec::with_capacity(self.layers.len());
        inputs.push(x.clone());
        for i in 0..self.layers.len() - 1 {
            inputs.push(self.layers[i].a.as_ref().unwrap().clone());
        // Backward pass through all layers
        for i in (0..self.layers.len()).rev() {
            grad = self.layers[i].backward(&inputs[i], &grad, learning_rate);
    /// Train the network for a number of epochs
    fn train(
        x: &Array2<f32>,
        y: &Array2<f32>,
        epochs: usize,
    ) -> Vec<f32> {
        let mut losses = Vec::with_capacity(epochs);
        for epoch in 0..epochs {
            // Forward pass
            let predictions = self.forward(x);
            // Compute loss
            let loss = self.loss(&predictions, y);
            losses.push(loss);
            // Print progress
            if epoch % 1000 == 0 || epoch == epochs - 1 {
                println!("Epoch {}/{}: loss = {:.6}", epoch + 1, epochs, loss);
            // Backward pass
            self.backward(x, y, learning_rate);
        losses
    /// Make predictions on new data
    fn predict(&mut self, x: &Array2<f32>) -> Array2<f32> {
        self.forward(x)
    /// Print a summary of the network architecture
    fn summary(&self) {
        println!("Neural Network Summary:");
        println!("------------------------");
        println!("Loss function: {}", self.loss_fn.as_str());
        println!("Number of layers: {}", self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            let input_size = layer.weights.shape()[0];
            let output_size = layer.weights.shape()[1];
            let num_params = input_size * output_size + output_size;
            println!(
                "Layer {}: Input={}, Output={}, Activation={}, Parameters={}",
                i + 1,
                input_size,
                output_size,
                layer.activation.as_str(),
                num_params
            );
        // Total parameters
        let total_params: usize = self
            .layers
            .iter()
            .map(|l| {
                let shape = l.weights.shape();
                shape[0] * shape[1] + shape[1]
            })
            .sum();
        println!("Total parameters: {}", total_params);
/// Train a neural network for the XOR problem
#[allow(dead_code)]
fn train_xor_network() -> Result<()> {
    // XOR dataset
    let x = Array2::from_shape_vec((4, 2), vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])?;
    let y = Array2::from_shape_vec((4, 1), vec![0.0f32, 1.0, 1.0, 0.0])?;
    println!("XOR problem dataset:");
    println!("Inputs:\n{:?}", x);
    println!("Targets:\n{:?}", y);
    // Create network: 2 inputs -> 4 hidden (ReLU) -> 1 output (Sigmoid)
    let mut network = NeuralNetwork::new(
        &[2, 4, 1],
        &[ActivationFunction::ReLU, ActivationFunction::Sigmoid],
        LossFunction::MSE,
        42, // Seed
    );
    network.summary();
    // Train the network
    println!("\nTraining the network...");
    let losses = network.train(&x, &y, 0.1, 10000);
    // Plot the loss curve (simple ASCII art)
    println!("\nLoss Curve:");
    print_loss_curve(&losses, 50);
    // Evaluate on training data
    let predictions = network.predict(&x);
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
        let prediction = network.predict(&input);
        println!(
            "Input: [{:.1}, {:.1}], Predicted: {:.3}, Expected: {:.1}",
            x1,
            x2,
            prediction[[0, 0]],
            if (x1 == 1.0 && x2 == 0.0) || (x1 == 0.0 && x2 == 1.0) {
                1.0
            } else {
                0.0
    // Try different activation functions
    println!("\nTrying different activation functions for hidden layer:");
    let activations = [
        ActivationFunction::ReLU,
        ActivationFunction::Sigmoid,
        ActivationFunction::Tanh,
    for activation in &activations {
            "\nTraining with hidden layer activation: {}",
            activation.as_str()
        let mut network = NeuralNetwork::new(
            &[2, 4, 1],
            &[*activation, ActivationFunction::Sigmoid],
            LossFunction::MSE,
            42, // Same seed for comparison
        network.train(&x, &y, 0.1, 5000);
        let predictions = network.predict(&x);
            "Final predictions with {}:\n{:.3?}",
            activation.as_str(),
            predictions
    // Try different loss functions
    println!("\nTrying different loss functions:");
    let loss_functions = [LossFunction::MSE, LossFunction::BinaryCrossEntropy];
    for loss_fn in &loss_functions {
        println!("\nTraining with loss function: {}", loss_fn.as_str());
            &[ActivationFunction::ReLU, ActivationFunction::Sigmoid],
            *loss_fn,
            loss_fn.as_str(),
    Ok(())
/// Train a neural network for a simple regression problem
#[allow(dead_code)]
fn train_regression_network() -> Result<()> {
    // Create a simple regression dataset: y = sin(x)
    let n_samples = 100;
    let x_data: Vec<f32> = (0..n_samples)
        .map(|i| 2.0 * std::f32::consts::PI * (i as f32) / (n_samples as f32))
        .collect();
    let y_data: Vec<f32> = x_data.iter().map(|&x| x.sin()).collect();
    // Reshape data for the network
    let x = Array2::from_shape_fn((n_samples, 1), |(i_)| x_data[i]);
    let y = Array2::from_shape_fn((n_samples, 1), |(i_)| y_data[i]);
    println!("\nRegression Problem: y = sin(x)");
    println!("Number of samples: {}", n_samples);
    // Create a network with 3 hidden layers
        &[1, 10, 10, 5, 1],
        &[
            ActivationFunction::Tanh,
            ActivationFunction::Linear,
        ],
        42,
    println!("\nTraining the regression network...");
    let losses = network.train(&x, &y, 0.01, 5000);
    // Plot the loss curve
    // Evaluate on a few samples
    println!("\nEvaluation on selected samples:");
    let test_points = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0];
    for &x_val in &test_points {
        let input = Array2::from_shape_vec((1, 1), vec![x_val])?;
            "x = {:.2}, Predicted: {:.4}, Expected: {:.4}",
            x_val,
            x_val.sin()
/// Print a simple ASCII loss curve
#[allow(dead_code)]
fn print_loss_curve(losses: &[f32], width: usize) {
    // Skip the first few values which might be very high
    let start_idx = losses.len().min(10);
    let relevant_losses = &_losses[start_idx..];
    if relevant_losses.is_empty() {
        println!("Not enough data points for loss curve");
        return;
    // Find min and max for scaling
    let min_loss = relevant_losses.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_loss = relevant_losses.iter().fold(0.0f32, |a, &b| a.max(b));
    // Number of points to display (downsample if too many)
    let n_display = width.min(relevant_losses.len());
    let step = relevant_losses.len() / n_display;
    // Create the curve
    for i in 0..n_display {
        let idx = i * step;
        let loss = relevant_losses[idx];
        let normalized = if max_loss > min_loss {
            (loss - min_loss) / (max_loss - min_loss)
        } else {
            0.5
        };
        let bar_len = (normalized * 40.0) as usize;
        print!("{:5}: ", idx + start_idx);
        print!("{:.6} ", loss);
        println!("{}", "#".repeat(bar_len));
#[allow(dead_code)]
fn main() -> Result<()> {
    println!("General-Purpose Neural Network Example");
    println!("======================================\n");
    // Train a network for XOR problem
    train_xor_network()?;
    // Train a network for regression
    train_regression_network()?;
