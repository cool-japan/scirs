//! Mini-batch training example for neural networks
//!
//! This example demonstrates how to implement mini-batch training
//! for neural networks in Rust, which is useful for large datasets
//! and can improve training stability and performance.

use ndarray::{s, Array2, Axis};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use scirs2_neural::error::Result;
use std::f32;
/// Activation function type
#[derive(Debug, Clone, Copy)]
enum ActivationFunction {
    ReLU,
    Sigmoid,
    #[allow(dead_code)]
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
    fn to_string(&self) -> &str {
            ActivationFunction::ReLU => "ReLU",
            ActivationFunction::Sigmoid => "Sigmoid",
            ActivationFunction::Tanh => "Tanh",
            ActivationFunction::Linear => "Linear",
/// Loss function type
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
    // Gradients
    dw: Option<Array2<f32>>,
    db: Option<Array2<f32>>,
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
            dw: None,
            db: None,
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
    /// Compute gradients during backward pass
    fn compute_gradients(&mut self, input: &Array2<f32>, gradoutput: &Array2<f32>) -> Array2<f32> {
        let z = self.z.as_ref().expect("Forward pass must be called first");
        // Gradient through activation: dL/dz = dL/da * da/dz
        let dz = grad_output * &self.activation.derivative(z);
        // Gradient for weights: dL/dW = X.T @ dz
        let dw = input.t().dot(&dz);
        // Gradient for biases: dL/db = sum(dz, axis=0)
        let db = dz.sum_axis(Axis(0)).insert_axis(Axis(0));
        // Gradient for previous layer: dL/dX = dz @ W.T
        let dx = dz.dot(&self.weights.t());
        // Store gradients
        self.dw = Some(dw);
        self.db = Some(db);
        dx
    /// Update parameters with computed gradients
    fn update_parameters(&mut self, learningrate: f32) {
        if let (Some(dw), Some(db)) = (&self.dw, &self.db) {
            // Update parameters
            self.weights = &self.weights - dw * learning_rate;
            self.biases = &self.biases - db * learning_rate;
            // Clear gradients
            self.dw = None;
            self.db = None;
/// A neural network composed of multiple layers with mini-batch support
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
    /// Compute gradients for a mini-batch
    fn compute_gradients(&mut self, x: &Array2<f32>, y: &Array2<f32>) {
        // Forward pass
        let predictions = self.forward(x);
        // Compute loss derivative
        let mut grad = self.loss_fn.derivative(&predictions, y);
        // Store inputs for layers
        let mut inputs = Vec::with_capacity(self.layers.len());
        inputs.push(x.clone());
        for i in 0..self.layers.len() - 1 {
            inputs.push(self.layers[i].a.as_ref().unwrap().clone());
        // Backward pass to compute gradients
        for i in (0..self.layers.len()).rev() {
            grad = self.layers[i].compute_gradients(&inputs[i], &grad);
    /// Update parameters after computing gradients
            layer.update_parameters(learning_rate);
    /// Train the network for a number of epochs using mini-batches
    fn train_mini_batch(
        &mut self,
        x: &Array2<f32>,
        y: &Array2<f32>,
        batch_size: usize,
        learning_rate: f32,
        epochs: usize,
    ) -> Vec<f32> {
        let n_samples = x.shape()[0];
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = SmallRng::from_seed([42; 32]);
        let mut losses = Vec::with_capacity(epochs);
        for epoch in 0..epochs {
            // Shuffle indices for random batches
            indices.shuffle(&mut rng);
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;
            // Process mini-batches
            for batch_start in (0..n_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_samples);
                let batch_indices = &indices[batch_start..batch_end];
                // Create mini-batch
                let batch_x = create_batch(x, batch_indices);
                let batch_y = create_batch(y, batch_indices);
                // Forward pass
                let predictions = self.forward(&batch_x);
                // Compute loss
                let batch_loss = self.loss(&predictions, &batch_y);
                epoch_loss += batch_loss;
                batch_count += 1;
                // Backward pass and update
                let mut grad = self.loss_fn.derivative(&predictions, &batch_y);
                // Store inputs for layers
                let mut inputs = Vec::with_capacity(self.layers.len());
                inputs.push(batch_x);
                for i in 0..self.layers.len() - 1 {
                    inputs.push(self.layers[i].a.as_ref().unwrap().clone());
                // Backward pass to compute gradients
                for i in (0..self.layers.len()).rev() {
                    grad = self.layers[i].compute_gradients(&inputs[i], &grad);
                // Update parameters
                self.update_parameters(learning_rate);
            // Compute average loss for the epoch
            epoch_loss /= batch_count as f32;
            losses.push(epoch_loss);
            // Print progress
            if epoch % 100 == 0 || epoch == epochs - 1 {
                println!("Epoch {}/{}: loss = {:.6}", epoch + 1, epochs, epoch_loss);
        losses
    /// Make predictions on new data
    fn predict(&mut self, x: &Array2<f32>) -> Array2<f32> {
        self.forward(x)
    /// Print a summary of the network architecture
    fn summary(&self) {
        println!("Neural Network Summary:");
        println!("------------------------");
        println!("Loss function: {}", self.loss_fn.to_string());
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
                layer.activation.to_string(),
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
/// Helper function to create a mini-batch from indices
#[allow(dead_code)]
fn create_batch(data: &Array2<f32>, indices: &[usize]) -> Array2<f32> {
    let batch_size = indices.len();
    let feature_dim = data.shape()[1];
    let mut batch = Array2::zeros((batch_size, feature_dim));
    for (batch_idx, &data_idx) in indices.iter().enumerate() {
        let row = data.slice(s![data_idx, ..]);
        batch.slice_mut(s![batch_idx, ..]).assign(&row);
    batch
/// Simple print function for a loss curve
#[allow(dead_code)]
fn print_loss_curve(losses: &[f32], width: usize) {
    // Skip the first few values which might be very high
    let start_idx = losses.len().min(10);
    let relevant_losses = &losses[start_idx..];
    if relevant_losses.is_empty() {
        println!("Not enough _data points for loss curve");
        return;
    // Find min and max for scaling
    let min_loss = relevant_losses.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_loss = relevant_losses.iter().fold(0.0f32, |a, &b| a.max(b));
    // Number of points to display (downsample if too many)
    let n_display = width.min(relevant_losses.len());
    let step = (relevant_losses.len() / n_display).max(1);
    // Create the curve
    println!("Loss range: {:.6} - {:.6}", min_loss, max_loss);
    for i in 0..n_display {
        let idx = i * step;
        if idx >= relevant_losses.len() {
            break;
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
/// Generate a synthetic dataset for binary classification
#[allow(dead_code)]
fn generate_classification_dataset(_nsamples: usize, seed: u64) -> (Array2<f32>, Array2<f32>) {
    let mut rng = SmallRng::from_seed(seed);
    // Generate two clusters of points
    let mut features = Array2::zeros((_n_samples, 2));
    let mut labels = Array2::zeros((_n_samples, 1));
    for i in 0.._n_samples {
        let cluster = i < _n_samples / 2;
        if cluster {
            // Cluster 1: centered at (1, 1)
            features[[i, 0]] = 1.0 + rng.gen_range(-0.5..0.5);
            features[[i..1]] = 1.0 + rng.gen_range(-0.5..0.5);
            labels[[i, 0]] = 1.0;
            // Cluster 2: centered at (0, 0)
            features[[i, 0]] = rng.gen_range(-0.5..0.5);
            features[[i..1]] = rng.gen_range(-0.5..0.5);
            labels[[i, 0]] = 0.0;
    (features, labels)
/// Generate a sine wave dataset for regression
#[allow(dead_code)]
fn generate_regression_dataset(_nsamples: usize) -> (Array2<f32>, Array2<f32>) {
    // Generate x values between 0 and 2π
    let mut x = Array2::zeros((_n_samples, 1));
    let mut y = Array2::zeros((_n_samples, 1));
        let x_val = 2.0 * std::f32::consts::PI * (i as f32) / (_n_samples as f32);
        x[[i, 0]] = x_val;
        y[[i, 0]] = x_val.sin();
    (x, y)
/// Demonstrate mini-batch training on the XOR problem
#[allow(dead_code)]
fn train_xor_minibatch() -> Result<()> {
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
    // Train the network with batch size of 2
    println!("\nTraining with mini-batch size 2...");
    let losses = network.train_mini_batch(&x, &y, 2, 0.1, 2000);
    // Plot the loss curve
    println!("\nLoss Curve:");
    print_loss_curve(&losses, 30);
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
    Ok(())
/// Demonstrate mini-batch training on a larger dataset
#[allow(dead_code)]
fn train_classification_minibatch() -> Result<()> {
    let n_samples = 500;
    // Generate synthetic dataset
    let (features, labels) = generate_classification_dataset(n_samples, 42);
    println!("\nBinary Classification Problem");
    println!("Number of samples: {}", n_samples);
    // Create different batch sizes for comparison
    let batch_sizes = [8, 32, 128];
    for &batch_size in &batch_sizes {
        println!("\n--- Training with batch size {} ---", batch_size);
        // Create network
        let mut network = NeuralNetwork::new(
            &[2, 8, 1],
            &[ActivationFunction::ReLU, ActivationFunction::Sigmoid],
            LossFunction::BinaryCrossEntropy,
            42, // Seed for reproducibility
        // Train the network
        let losses = network.train_mini_batch(&features, &labels, batch_size, 0.05, 500);
        // Plot the loss curve
        println!("\nLoss Curve (batch size {}):", batch_size);
        print_loss_curve(&losses, 20);
        // Evaluate accuracy
        let predictions = network.predict(&features);
        let mut correct = 0;
        for i in 0..n_samples {
            let predicted = predictions[[i, 0]] > 0.5;
            let expected = labels[[i, 0]] > 0.5;
            if predicted == expected {
                correct += 1;
        let accuracy = correct as f32 / n_samples as f32;
            "Accuracy with batch size {}: {:.2}%",
            batch_size,
            accuracy * 100.0
#[allow(dead_code)]
fn main() -> Result<()> {
    println!("Mini-Batch Training Example");
    println!("==========================\n");
    // Train a network for XOR problem using mini-batches
    train_xor_minibatch()?;
    // Train a network on a larger classification dataset with different batch sizes
    train_classification_minibatch()?;
