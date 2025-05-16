//! Regularization Techniques for Neural Networks
//!
//! This example demonstrates various regularization techniques to prevent overfitting
//! in neural networks, including:
//! - L1 regularization (Lasso)
//! - L2 regularization (Ridge)
//! - Dropout
//! - Early stopping

use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use scirs2_neural::error::Result;
use std::f32;

/// Activation function type
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}

impl ActivationFunction {
    /// Apply the activation function
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
        match self {
            ActivationFunction::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            ActivationFunction::Sigmoid => {
                let sigmoid = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
                sigmoid.mapv(|s| s * (1.0 - s))
            }
            ActivationFunction::Tanh => {
                let tanh = x.mapv(|v| v.tanh());
                tanh.mapv(|t| 1.0 - t * t)
            }
            ActivationFunction::Linear => Array2::ones(x.dim()),
        }
    }

    /// Get a string representation of the activation function
    fn to_string(&self) -> &str {
        match self {
            ActivationFunction::ReLU => "ReLU",
            ActivationFunction::Sigmoid => "Sigmoid",
            ActivationFunction::Tanh => "Tanh",
            ActivationFunction::Linear => "Linear",
        }
    }
}

/// Loss function type
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum LossFunction {
    MSE,
    BinaryCrossEntropy,
}

impl LossFunction {
    /// Compute the loss between predictions and targets
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        match self {
            LossFunction::MSE => {
                let diff = predictions - targets;
                let squared = diff.mapv(|v| v * v);
                squared.sum() / (predictions.len() as f32)
            }
            LossFunction::BinaryCrossEntropy => {
                let epsilon = 1e-15; // To avoid log(0)
                let mut sum = 0.0;

                for (y_pred, y_true) in predictions.iter().zip(targets.iter()) {
                    let y_pred_safe = y_pred.max(epsilon).min(1.0 - epsilon);
                    sum += y_true * y_pred_safe.ln() + (1.0 - y_true) * (1.0 - y_pred_safe).ln();
                }

                -sum / (predictions.len() as f32)
            }
        }
    }

    /// Compute the derivative of the loss function with respect to predictions
    fn derivative(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32> {
        match self {
            LossFunction::MSE => {
                // d(MSE)/dŷ = 2(ŷ - y)/n
                let n = predictions.len() as f32;
                (predictions - targets) * (2.0 / n)
            }
            LossFunction::BinaryCrossEntropy => {
                // d(BCE)/dŷ = ((1-y)/(1-ŷ) - y/ŷ)/n
                let epsilon = 1e-15;
                let n = predictions.len() as f32;

                Array2::from_shape_fn(predictions.dim(), |(i, j)| {
                    let y_pred = predictions[(i, j)].max(epsilon).min(1.0 - epsilon);
                    let y_true = targets[(i, j)];
                    ((1.0 - y_true) / (1.0 - y_pred) - y_true / y_pred) / n
                })
            }
        }
    }

    /// Get a string representation of the loss function
    fn to_string(&self) -> &str {
        match self {
            LossFunction::MSE => "Mean Squared Error",
            LossFunction::BinaryCrossEntropy => "Binary Cross Entropy",
        }
    }
}

/// Regularization type
#[derive(Debug, Clone, Copy)]
enum RegularizationType {
    None,
    L1(f32), // L1 with strength parameter
    L2(f32), // L2 with strength parameter
}

/// Dropout layer for regularization
struct Dropout {
    drop_prob: f32,
    training: bool,
    mask: Option<Array2<f32>>,
}

impl Dropout {
    /// Create a new dropout layer
    fn new(drop_prob: f32) -> Self {
        assert!(
            drop_prob >= 0.0 && drop_prob < 1.0,
            "Dropout probability must be between 0 and 1"
        );

        Self {
            drop_prob,
            training: true,
            mask: None,
        }
    }

    /// Set training mode
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Forward pass
    fn forward(&mut self, x: &Array2<f32>, rng: &mut SmallRng) -> Array2<f32> {
        if !self.training || self.drop_prob == 0.0 {
            // During inference or if dropout is disabled, just pass through
            return x.clone();
        }

        // Generate binary mask (1 = keep, 0 = drop)
        let mask = Array2::from_shape_fn(x.dim(), |_| {
            if rng.random::<f32>() > self.drop_prob {
                1.0
            } else {
                0.0
            }
        });

        // Store mask for backward pass
        self.mask = Some(mask.clone());

        // Apply mask and scale (inverted dropout)
        let scale = 1.0 / (1.0 - self.drop_prob);
        x * &mask * scale
    }

    /// Backward pass
    fn backward(&self, grad_output: &Array2<f32>) -> Array2<f32> {
        if !self.training || self.drop_prob == 0.0 {
            // During inference or if dropout is disabled, just pass through
            return grad_output.clone();
        }

        let mask = self
            .mask
            .as_ref()
            .expect("Forward pass must be called first");
        let scale = 1.0 / (1.0 - self.drop_prob);
        grad_output * mask * scale
    }
}

/// Dense (fully connected) layer with regularization
struct Dense {
    input_size: usize,
    output_size: usize,
    activation: ActivationFunction,
    regularization: RegularizationType,

    // Parameters
    weights: Array2<f32>,
    biases: Array1<f32>,

    // Cached values for backward pass
    input: Option<Array2<f32>>,
    z: Option<Array2<f32>>,
}

impl Dense {
    /// Create a new dense layer
    fn new(
        input_size: usize,
        output_size: usize,
        activation: ActivationFunction,
        regularization: RegularizationType,
        rng: &mut SmallRng,
    ) -> Self {
        // He/Kaiming initialization for weights
        let std_dev = (2.0 / input_size as f32).sqrt();

        // Initialize weights with random values
        let mut weights = Array2::zeros((input_size, output_size));
        for elem in weights.iter_mut() {
            *elem = rng.random_range(-std_dev..std_dev);
        }
        let biases = Array1::zeros(output_size);

        Self {
            input_size,
            output_size,
            activation,
            regularization,
            weights,
            biases,
            input: None,
            z: None,
        }
    }

    /// Forward pass
    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        // Store input for backward pass
        self.input = Some(x.clone());

        // Linear transformation: z = x @ W + b
        let mut z = Array2::zeros((x.shape()[0], self.output_size));

        for i in 0..x.shape()[0] {
            for j in 0..self.output_size {
                let mut sum = 0.0;
                for k in 0..self.input_size {
                    sum += x[[i, k]] * self.weights[[k, j]];
                }
                z[[i, j]] = sum + self.biases[j];
            }
        }

        // Store pre-activation output
        self.z = Some(z.clone());

        // Apply activation
        self.activation.apply(&z)
    }

    /// Compute regularization loss
    fn regularization_loss(&self) -> f32 {
        match self.regularization {
            RegularizationType::None => 0.0,
            RegularizationType::L1(lambda) => {
                let l1_norm = self.weights.iter().map(|w| w.abs()).sum::<f32>();
                lambda * l1_norm
            }
            RegularizationType::L2(lambda) => {
                let l2_norm = self.weights.iter().map(|w| w * w).sum::<f32>();
                0.5 * lambda * l2_norm
            }
        }
    }

    /// Backward pass
    fn backward(&mut self, grad_output: &Array2<f32>, learning_rate: f32) -> Array2<f32> {
        let input = self
            .input
            .as_ref()
            .expect("Forward pass must be called first");
        let z = self.z.as_ref().expect("Forward pass must be called first");

        // Compute gradient through activation
        let dactivation = self.activation.derivative(z);
        let delta = grad_output * &dactivation;

        // Compute gradients with respect to weights and biases
        let dweights = input.t().dot(&delta);
        let dbiases = delta.sum_axis(Axis(0));

        // Add regularization gradients
        let reg_grad = match self.regularization {
            RegularizationType::None => Array2::zeros(self.weights.dim()),
            RegularizationType::L1(lambda) => {
                // Sign of weights (L1 gradient)
                let sign = self.weights.mapv(|w| {
                    if w > 0.0 {
                        1.0
                    } else if w < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                });
                sign * lambda
            }
            RegularizationType::L2(lambda) => {
                // Weights (L2 gradient)
                &self.weights * lambda
            }
        };

        // Update weights and biases
        self.weights = &self.weights - &((&dweights + &reg_grad) * learning_rate);
        self.biases = &self.biases - &(dbiases * learning_rate);

        // Compute gradient with respect to input
        delta.dot(&self.weights.t())
    }
}

/// Simple neural network with regularization
struct NeuralNetwork {
    layers: Vec<Dense>,
    dropout_layers: Vec<Dropout>,
    loss_fn: LossFunction,
    rng: SmallRng,
}

impl NeuralNetwork {
    /// Create a new neural network
    fn new(loss_fn: LossFunction, seed: u64) -> Self {
        Self {
            layers: Vec::new(),
            dropout_layers: Vec::new(),
            loss_fn,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Add a dense layer
    fn add_dense(
        &mut self,
        input_size: usize,
        output_size: usize,
        activation: ActivationFunction,
        regularization: RegularizationType,
    ) -> &mut Self {
        let layer = Dense::new(
            input_size,
            output_size,
            activation,
            regularization,
            &mut self.rng,
        );

        self.layers.push(layer);
        self
    }

    /// Add a dropout layer
    fn add_dropout(&mut self, drop_prob: f32) -> &mut Self {
        let dropout = Dropout::new(drop_prob);
        self.dropout_layers.push(dropout);
        self
    }

    /// Set training mode for all dropout layers
    fn set_training(&mut self, training: bool) {
        for dropout in &mut self.dropout_layers {
            dropout.set_training(training);
        }
    }

    /// Forward pass
    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let mut output = x.clone();
        let mut dropout_idx = 0;

        for layer in &mut self.layers {
            // Apply dense layer
            output = layer.forward(&output);

            // Apply dropout if available
            if dropout_idx < self.dropout_layers.len() {
                output = self.dropout_layers[dropout_idx].forward(&output, &mut self.rng);
                dropout_idx += 1;
            }
        }

        output
    }

    /// Compute total loss including regularization
    fn loss(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        // Data loss
        let data_loss = self.loss_fn.compute(predictions, targets);

        // Regularization loss
        let reg_loss = self
            .layers
            .iter()
            .map(|layer| layer.regularization_loss())
            .sum::<f32>();

        data_loss + reg_loss
    }

    /// Backward pass and parameter update
    fn backward(&mut self, x: &Array2<f32>, y: &Array2<f32>, learning_rate: f32) -> f32 {
        // Forward pass
        let predictions = self.forward(x);

        // Compute loss
        let loss = self.loss(&predictions, y);

        // Compute initial gradient from loss function
        let mut grad = self.loss_fn.derivative(&predictions, y);

        // Backward pass through dropout and dense layers in reverse
        let mut dropout_idx = self.dropout_layers.len();

        for layer_idx in (0..self.layers.len()).rev() {
            // Apply dropout backward if applicable
            if dropout_idx > 0 && dropout_idx > layer_idx {
                dropout_idx -= 1;
                grad = self.dropout_layers[dropout_idx].backward(&grad);
            }

            // Backward through dense layer
            grad = self.layers[layer_idx].backward(&grad, learning_rate);
        }

        loss
    }

    /// Train the network
    fn train(
        &mut self,
        x_train: &Array2<f32>,
        y_train: &Array2<f32>,
        x_val: &Array2<f32>,
        y_val: &Array2<f32>,
        learning_rate: f32,
        epochs: usize,
        batch_size: usize,
        patience: Option<usize>,
    ) -> Vec<(f32, f32)> {
        // Returns (train_loss, val_loss) for each epoch
        let n_samples = x_train.shape()[0];
        let mut losses = Vec::with_capacity(epochs);

        // For early stopping
        let patience = patience.unwrap_or(usize::MAX);
        let mut best_val_loss = f32::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..epochs {
            // Create random indices for shuffling
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut self.rng);

            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            // Enable training mode
            self.set_training(true);

            // Process mini-batches
            for batch_start in (0..n_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_samples);
                let batch_indices = &indices[batch_start..batch_end];

                // Create mini-batch
                let mut batch_x = Array2::zeros([batch_indices.len(), x_train.shape()[1]]);
                let mut batch_y = Array2::zeros([batch_indices.len(), y_train.shape()[1]]);

                for (i, &idx) in batch_indices.iter().enumerate() {
                    for j in 0..x_train.shape()[1] {
                        batch_x[[i, j]] = x_train[[idx, j]];
                    }

                    for j in 0..y_train.shape()[1] {
                        batch_y[[i, j]] = y_train[[idx, j]];
                    }
                }

                // Train on mini-batch
                let batch_loss = self.backward(&batch_x, &batch_y, learning_rate);
                epoch_loss += batch_loss;
                batch_count += 1;
            }

            // Compute average training loss
            let train_loss = epoch_loss / batch_count as f32;

            // Evaluate on validation set
            self.set_training(false); // Disable dropout for validation
            let predictions = self.forward(x_val);
            let val_loss = self.loss(&predictions, y_val);

            // Store losses
            losses.push((train_loss, val_loss));

            // Print progress
            if epoch % 10 == 0 || epoch == epochs - 1 {
                println!(
                    "Epoch {}/{}: train_loss = {:.6}, val_loss = {:.6}",
                    epoch + 1,
                    epochs,
                    train_loss,
                    val_loss
                );
            }

            // Early stopping
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= patience {
                    println!("Early stopping at epoch {}", epoch + 1);
                    break;
                }
            }
        }

        losses
    }

    /// Make predictions
    fn predict(&mut self, x: &Array2<f32>) -> Array2<f32> {
        self.set_training(false); // Disable dropout for prediction
        self.forward(x)
    }

    /// Print a summary of the network
    fn summary(&self) {
        println!("Neural Network Summary:");
        println!("------------------------");
        println!("Loss function: {}", self.loss_fn.to_string());

        let mut total_params = 0;
        let mut layer_idx = 0;
        let mut dropout_idx = 0;

        for (i, layer) in self.layers.iter().enumerate() {
            // Print dense layer
            let params = layer.input_size * layer.output_size + layer.output_size;
            total_params += params;

            println!(
                "Layer {}: Dense ({} -> {}), Activation: {}, Regularization: {}, Parameters: {}",
                layer_idx + 1,
                layer.input_size,
                layer.output_size,
                layer.activation.to_string(),
                match layer.regularization {
                    RegularizationType::None => "None".to_string(),
                    RegularizationType::L1(lambda) => format!("L1 (lambda={})", lambda),
                    RegularizationType::L2(lambda) => format!("L2 (lambda={})", lambda),
                },
                params
            );

            layer_idx += 1;

            // Print dropout layer if it follows this dense layer
            if dropout_idx < self.dropout_layers.len() && i == dropout_idx {
                println!(
                    "Layer {}: Dropout (p={})",
                    layer_idx + 1,
                    self.dropout_layers[dropout_idx].drop_prob
                );

                layer_idx += 1;
                dropout_idx += 1;
            }
        }

        println!("------------------------");
        println!("Total parameters: {}", total_params);
    }
}

/// Generate a noisy non-linear dataset with irrelevant features (for demonstrating regularization)
fn generate_dataset(
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    noise_level: f32,
    rng: &mut SmallRng,
) -> (Array2<f32>, Array2<f32>) {
    assert!(
        n_informative <= n_features,
        "n_informative must be <= n_features"
    );

    // Generate random coefficients for informative features
    let mut coefficients = vec![0.0; n_features];
    for i in 0..n_informative {
        coefficients[i] = rng.random::<f32>() * 2.0 - 1.0; // Random between -1 and 1
    }

    // Generate features (including irrelevant ones)
    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array2::zeros((n_samples, 1));

    for i in 0..n_samples {
        // Generate feature values
        for j in 0..n_features {
            x[[i, j]] = rng.random::<f32>() * 2.0 - 1.0; // Random between -1 and 1
        }

        // Compute target with non-linear transformation and noise
        let mut target = 0.0;
        for j in 0..n_features {
            // Add non-linear interactions between first few features
            if j < n_informative {
                target += coefficients[j] * x[[i, j]];

                // Add some non-linearity
                if j < n_informative / 2 {
                    target += 0.5 * x[[i, j]].powi(2);
                }
            }
        }

        // Add sine wave non-linearity for first feature
        if n_informative > 0 {
            target += (x[[i, 0]] * std::f32::consts::PI).sin();
        }

        // Add noise
        target += rng.random::<f32>() * noise_level;

        // Scale to 0-1 range (sigmoid for binary classification)
        y[[i, 0]] = 1.0 / (1.0 + (-target).exp());
    }

    (x, y)
}

/// Evaluate binary classification accuracy
fn evaluate_accuracy(predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
    let n_samples = predictions.shape()[0];
    let mut correct = 0;

    for i in 0..n_samples {
        let pred = predictions[[i, 0]] > 0.5;
        let target = targets[[i, 0]] > 0.5;

        if pred == target {
            correct += 1;
        }
    }

    correct as f32 / n_samples as f32
}

/// Training configuration for a regularization experiment
struct ExperimentConfig {
    regularization: RegularizationType,
    dropout_prob: Option<f32>,
    early_stopping: Option<usize>,
}

impl ExperimentConfig {
    fn new(
        regularization: RegularizationType,
        dropout_prob: Option<f32>,
        early_stopping: Option<usize>,
    ) -> Self {
        Self {
            regularization,
            dropout_prob,
            early_stopping,
        }
    }

    fn get_description(&self) -> String {
        let reg_desc = match self.regularization {
            RegularizationType::None => "No regularization".to_string(),
            RegularizationType::L1(lambda) => format!("L1 regularization (lambda={})", lambda),
            RegularizationType::L2(lambda) => format!("L2 regularization (lambda={})", lambda),
        };

        let dropout_desc = match self.dropout_prob {
            Some(p) => format!("Dropout (p={})", p),
            None => "No dropout".to_string(),
        };

        let early_stopping_desc = match self.early_stopping {
            Some(patience) => format!("Early stopping (patience={})", patience),
            None => "No early stopping".to_string(),
        };

        format!("{}, {}, {}", reg_desc, dropout_desc, early_stopping_desc)
    }
}

/// Run an experiment with different regularization methods
fn run_experiment(config: &ExperimentConfig) -> Result<()> {
    // Set up RNG
    let mut rng = SmallRng::seed_from_u64(42);

    // Generate dataset with irrelevant features to demonstrate regularization
    let n_samples = 1000;
    let n_features = 20;
    let n_informative = 5;
    let noise_level = 0.3;

    println!(
        "Generating dataset with {} samples, {} features ({} informative), noise level {}",
        n_samples, n_features, n_informative, noise_level
    );

    let (x, y) = generate_dataset(n_samples, n_features, n_informative, noise_level, &mut rng);

    // Split into training, validation, and test sets
    let train_size = (n_samples as f32 * 0.6) as usize;
    let val_size = (n_samples as f32 * 0.2) as usize;
    let test_size = n_samples - train_size - val_size;

    // Create indices
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);

    // Split data
    let mut x_train = Array2::zeros((train_size, n_features));
    let mut y_train = Array2::zeros((train_size, 1));
    let mut x_val = Array2::zeros((val_size, n_features));
    let mut y_val = Array2::zeros((val_size, 1));
    let mut x_test = Array2::zeros((test_size, n_features));
    let mut y_test = Array2::zeros((test_size, 1));

    for (i, &idx) in indices.iter().take(train_size).enumerate() {
        for j in 0..n_features {
            x_train[[i, j]] = x[[idx, j]];
        }
        y_train[[i, 0]] = y[[idx, 0]];
    }

    for (i, &idx) in indices.iter().skip(train_size).take(val_size).enumerate() {
        for j in 0..n_features {
            x_val[[i, j]] = x[[idx, j]];
        }
        y_val[[i, 0]] = y[[idx, 0]];
    }

    for (i, &idx) in indices.iter().skip(train_size + val_size).enumerate() {
        for j in 0..n_features {
            x_test[[i, j]] = x[[idx, j]];
        }
        y_test[[i, 0]] = y[[idx, 0]];
    }

    println!(
        "Split data into {} training, {} validation, and {} test samples",
        train_size, val_size, test_size
    );

    // Create neural network
    let mut model = NeuralNetwork::new(LossFunction::BinaryCrossEntropy, 42);

    // Add layers
    model.add_dense(
        n_features,
        64,
        ActivationFunction::ReLU,
        config.regularization,
    );

    if let Some(dropout_prob) = config.dropout_prob {
        model.add_dropout(dropout_prob);
    }

    model.add_dense(64, 32, ActivationFunction::ReLU, config.regularization);

    if let Some(dropout_prob) = config.dropout_prob {
        model.add_dropout(dropout_prob);
    }

    model.add_dense(32, 1, ActivationFunction::Sigmoid, RegularizationType::None);

    // Print model summary
    println!("\nExperiment: {}", config.get_description());
    model.summary();

    // Train the model
    println!("\nTraining model...");
    let learning_rate = 0.01;
    let batch_size = 32;
    let epochs = 100;

    let losses = model.train(
        &x_train,
        &y_train,
        &x_val,
        &y_val,
        learning_rate,
        epochs,
        batch_size,
        config.early_stopping,
    );

    // Evaluate on test set
    let predictions = model.predict(&x_test);
    let test_loss = model.loss(&predictions, &y_test);
    let accuracy = evaluate_accuracy(&predictions, &y_test);

    println!("\nTest results:");
    println!("Loss: {:.6}", test_loss);
    println!("Accuracy: {:.2}%", accuracy * 100.0);

    // Print final loss curve
    println!("\nLoss curve:");
    println!("Epoch | Train Loss |  Val Loss");
    println!("----------------------------");

    for (i, (train_loss, val_loss)) in losses
        .iter()
        .enumerate()
        .skip(losses.len().saturating_sub(10))
    {
        println!("{:5} | {:10.6} | {:10.6}", i + 1, train_loss, val_loss);
    }

    Ok(())
}

fn main() -> Result<()> {
    println!("Regularization Techniques Example");
    println!("================================\n");

    // Run experiments with different regularization methods
    let experiments = vec![
        // No regularization
        ExperimentConfig::new(RegularizationType::None, None, None),
        // L1 regularization
        ExperimentConfig::new(RegularizationType::L1(0.001), None, None),
        // L2 regularization
        ExperimentConfig::new(RegularizationType::L2(0.001), None, None),
        // Dropout
        ExperimentConfig::new(RegularizationType::None, Some(0.2), None),
        // Early stopping
        ExperimentConfig::new(RegularizationType::None, None, Some(5)),
        // Combined: L2 + Dropout + Early stopping
        ExperimentConfig::new(RegularizationType::L2(0.001), Some(0.2), Some(5)),
    ];

    for config in &experiments {
        run_experiment(config)?;
        println!("\n-----------------------------------\n");
    }

    Ok(())
}
