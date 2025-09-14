//! Neural network model serialization example
//!
//! This example demonstrates how to save and load neural network
//! models in Rust, which is essential for preserving trained models
//! and using them later for inference.

use ndarray::{Array2, Axis};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use scirs2_neural::error::Result;
use serde::{Deserialize, Serialize};
use std::f32;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
/// Activation function type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
#[derive(Debug, Serialize, Deserialize)]
struct Layer {
    weights: Array2<f32>,
    biases: Array2<f32>,
    activation: ActivationFunction,
    // Cached values for backpropagation - not serialized
    #[serde(skip)]
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
    /// Get layer input size
    fn input_size(&self) -> usize {
        self.weights.shape()[0]
    /// Get layer output size
    fn output_size(&self) -> usize {
        self.weights.shape()[1]
    /// Get number of parameters in the layer
    fn num_parameters(&self) -> usize {
        let shape = self.weights.shape();
        shape[0] * shape[1] + shape[1]
/// A neural network composed of multiple layers
struct NeuralNetwork {
    layers: Vec<Layer>,
    loss_fn: LossFunction,
    input_size: usize,
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
            layers,
            loss_fn,
            input_size: layer_sizes[0],
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
        // Forward pass
        let predictions = self.forward(x);
        // Compute loss derivative
        let mut grad = self.loss_fn.derivative(&predictions, y);
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
            if epoch % 100 == 0 || epoch == epochs - 1 {
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
        println!("Loss function: {}", self.loss_fn.to_string());
        println!("Number of layers: {}", self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            let input_size = layer.input_size();
            let output_size = layer.output_size();
            let num_params = layer.num_parameters();
            println!(
                "Layer {}: Input={}, Output={}, Activation={}, Parameters={}",
                i + 1,
                input_size,
                output_size,
                layer.activation.to_string(),
                num_params
            );
        // Total parameters
        let total_params: usize = self.layers.iter().map(|l| l.num_parameters()).sum();
        println!("Total parameters: {}", total_params);
    /// Save the neural network model to a file
    fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        // Serialize the model
        serde_json::to_writer(writer, self)?;
        Ok(())
    /// Load a neural network model from a file
    fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(_path)?;
        let reader = BufReader::new(file);
        // Deserialize the model
        let mut model: Self = serde_json::from_reader(reader)?;
        // Update input size
        if !model.layers.is_empty() {
            model.input_size = model.layers[0].input_size();
        Ok(model)
/// Train and save a model for the XOR problem
#[allow(dead_code)]
fn train_and_save_xor_model() -> Result<()> {
    // XOR dataset
    let x = Array2::from_shape_vec((4, 2), vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])?;
    let y = Array2::from_shape_vec((4, 1), vec![0.0f32, 1.0, 1.0, 0.0])?;
    println!("XOR problem dataset:");
    println!("Inputs:\n{:?}", x);
    println!("Targets:\n{:?}", y);
    // Create network with ReLU hidden layer and Sigmoid output
    let mut network = NeuralNetwork::new(
        &[2, 4, 1],
        &[ActivationFunction::ReLU, ActivationFunction::Sigmoid],
        LossFunction::MSE,
        42, // Seed
    );
    network.summary();
    // Train the network
    println!("\nTraining the network...");
    network.train(&x, &y, 0.1, 2000);
    // Evaluate before saving
    let predictions = network.predict(&x);
    println!("\nPredictions before saving:");
    println!("{:?}", predictions);
    // Save the model
    let model_path = "xor_model.json";
    println!("\nSaving model to {}", model_path);
    network.save(model_path)?;
    println!("Model saved successfully!");
    Ok(())
/// Load and use a previously trained XOR model
#[allow(dead_code)]
fn load_and_evaluate_xor_model() -> Result<()> {
    println!("\nLoading model from {}", model_path);
    // Load the model
    let mut network = match NeuralNetwork::load(model_path) {
        Ok(model) => {
            println!("Model loaded successfully!");
            model
        Err(e) => {
            println!("Error loading model: {}", e);
            return Ok(());
    };
    // Print model summary
    // Create XOR input data
    // Make predictions with the loaded model
    println!("\nPredictions with loaded model:");
    // Calculate accuracy
    let mut correct = 0;
    for i in 0..4 {
        let predicted = predictions[[i, 0]] > 0.5;
        let expected = y[[i, 0]] > 0.5;
        if predicted == expected {
            correct += 1;
        println!(
            "Input: [{:.1}, {:.1}], Predicted: {:.3} ({}), Expected: {:.1} ({})",
            x[[i, 0]],
            x[[i, 1]],
            predictions[[i, 0]],
            predicted,
            y[[i, 0]],
            expected
    let accuracy = (correct as f32) / 4.0;
    println!("\nAccuracy: {:.2}% ({}/{})", accuracy * 100.0, correct, 4);
#[allow(dead_code)]
fn main() -> Result<()> {
    println!("Neural Network Model Serialization Example");
    println!("=========================================\n");
    // Train and save a model
    train_and_save_xor_model()?;
    // Load and use the saved model
    load_and_evaluate_xor_model()?;
