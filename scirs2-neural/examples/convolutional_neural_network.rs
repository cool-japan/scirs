//! Convolutional Neural Network Implementation Example
//!
//! This example demonstrates how to implement convolutional neural networks
//! (CNNs) in Rust, including Conv2D layers, pooling, and fully connected
//! layers for image classification tasks.

#![allow(dead_code)]

use ndarray::{s, Array, Array1, Array2, Array4, ArrayView1, Axis};
use ndarray_rand::rand::distributions::{Distribution, Uniform};
use ndarray_rand::rand::prelude::SliceRandom;
use ndarray_rand::rand::rngs::SmallRng;
use ndarray_rand::rand::{Rng, SeedableRng};
use scirs2_neural::error::Result;
// use serde::{Deserialize, Serialize};
use std::f32;
/// Padding mode for convolutional layers
#[derive(Debug, Clone, Copy)]
enum PaddingMode {
    Valid, // No padding
    Same,  // Pad to keep same dimensions
}
/// Activation function type
#[derive(Debug)]
enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}

impl ActivationFunction {
    /// Apply the activation function to an array
    fn apply<D>(&self, x: &Array<f32, D>) -> Array<f32, D>
    where
        D: ndarray::Dimension,
    {
        match self {
            ActivationFunction::ReLU => x.mapv(|v| v.max(0.0)),
            ActivationFunction::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            ActivationFunction::Tanh => x.mapv(|v| v.tanh()),
            ActivationFunction::Linear => x.clone(),
        }
    }
    /// Compute the derivative of the activation function
    fn derivative<D>(&self, x: &Array<f32, D>) -> Array<f32, D>
    where
        D: ndarray::Dimension,
    {
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
            ActivationFunction::Linear => Array::ones(x.dim()),
        }
    }
    /// Get a string representation of the activation function
    fn as_str(&self) -> &str {
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
#[allow(clippy::upper_case_acronyms)]
enum LossFunction {
    MSE,
    CategoricalCrossEntropy,
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
            LossFunction::CategoricalCrossEntropy => {
                let epsilon = 1e-15; // To avoid log(0)
                let mut sum = 0.0;
                for (i, target_row) in targets.rows().into_iter().enumerate() {
                    let mut row_sum = 0.0;
                    for (j, &t) in target_row.iter().enumerate() {
                        if t > 0.0 {
                            // Only consider non-zero targets (one-hot encoded)
                            let pred = predictions[[i, j]].max(epsilon).min(1.0 - epsilon);
                            row_sum += t * pred.ln();
                        }
                    }
                    sum += row_sum;
                }
                -sum / (predictions.shape()[0] as f32)
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
            LossFunction::CategoricalCrossEntropy => {
                // d(CCE)/dŷ = -y/ŷ / n
                let epsilon = 1e-15;
                let n = predictions.shape()[0] as f32;
                Array2::from_shape_fn(predictions.dim(), |(i, j)| {
                    let y_pred = predictions[(i, j)].max(epsilon).min(1.0 - epsilon);
                    let y_true = targets[(i, j)];
                    if y_true > 0.0 {
                        -y_true / y_pred / n
                    } else {
                        0.0
                    }
                })
            }
        }
    }
    /// Get a string representation of the loss function
    fn as_str(&self) -> &str {
        match self {
            LossFunction::MSE => "Mean Squared Error",
            LossFunction::CategoricalCrossEntropy => "Categorical Cross Entropy",
        }
    }
}
/// Base trait for network layers
trait Layer {
    /// Forward pass through the layer
    fn forward(&mut self, x: &Array4<f32>) -> Array4<f32>;
    /// Backward pass to compute gradients
    fn backward(&mut self, gradoutput: &Array4<f32>) -> Array4<f32>;
    /// Update parameters with gradients
    fn update_parameters(&mut self, learningrate: f32);
    /// Get a description of the layer
    fn get_description(&self) -> String;
    /// Get number of trainable parameters
    fn num_parameters(&self) -> usize;
}
/// Convolutional layer (Conv2D)
#[derive(Debug)]
struct Conv2D {
    filters: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: PaddingMode,
    activation: ActivationFunction,
    // Parameters
    weights: Array4<f32>, // Shape: [filters, channels, kernel_h, kernel_w]
    biases: Array1<f32>,  // Shape: [filters]
    // Gradients
    dweights: Option<Array4<f32>>,
    dbiases: Option<Array1<f32>>,
    // Cache for backward pass
    // #[serde(skip)]
    input: Option<Array4<f32>>,
    z: Option<Array4<f32>>, // Pre-activation
    _output: Option<Array4<f32>>,
}

impl Conv2D {
    /// Create a new Conv2D layer
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_channels: usize,
        filters: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: PaddingMode,
        activation: ActivationFunction,
        rng: &mut SmallRng,
    ) -> Self {
        // He/Kaiming initialization for weights
        let fan_in = input_channels * kernel_size.0 * kernel_size.1;
        let std_dev = (2.0 / fan_in as f32).sqrt();
        let dist = Uniform::new_inclusive(-std_dev, std_dev);
        // Initialize weights: [filters, input_channels, kernel_h, kernel_w]
        let weights = Array4::from_shape_fn(
            [filters, input_channels, kernel_size.0, kernel_size.1],
            |_| dist.sample(rng),
        );
        // Initialize biases to zero
        let biases = Array1::zeros(filters);
        Self {
            filters,
            kernel_size,
            stride,
            padding,
            activation,
            weights,
            biases,
            dweights: None,
            dbiases: None,
            input: None,
            z: None,
            _output: None,
        }
    }
    /// Calculate output dimensions based on input and layer parameters
    fn calculate_outputshape(&self, inputshape: &[usize]) -> [usize; 4] {
        let batch_size = inputshape[0];
        let input_height = inputshape[2];
        let input_width = inputshape[3];
        let (output_height, output_width) = match self.padding {
            PaddingMode::Valid => {
                let h = (input_height - self.kernel_size.0) / self.stride.0 + 1;
                let w = (input_width - self.kernel_size.1) / self.stride.1 + 1;
                (h, w)
            }
            PaddingMode::Same => (input_height / self.stride.0, input_width / self.stride.1),
        };
        [batch_size, self.filters, output_height, output_width]
    }
    /// Calculate padding based on padding mode
    fn calculate_padding(&self, inputshape: &[usize]) -> (usize, usize, usize, usize) {
        match self.padding {
            PaddingMode::Valid => (0, 0, 0, 0),
            PaddingMode::Same => {
                let input_height = inputshape[2];
                let input_width = inputshape[3];
                let output_height = (input_height as f32 / self.stride.0 as f32).ceil() as usize;
                let output_width = (input_width as f32 / self.stride.1 as f32).ceil() as usize;
                let pad_height = ((output_height - 1) * self.stride.0 + self.kernel_size.0)
                    .saturating_sub(input_height);
                let pad_width = ((output_width - 1) * self.stride.1 + self.kernel_size.1)
                    .saturating_sub(input_width);
                let pad_top = pad_height / 2;
                let pad_bottom = pad_height - pad_top;
                let pad_left = pad_width / 2;
                let pad_right = pad_width - pad_left;
                (pad_top, pad_bottom, pad_left, pad_right)
            }
        }
    }
    /// Apply convolution operation
    fn convolve(&self, input: &Array4<f32>) -> Array4<f32> {
        let inputshape = input.shape();
        let batch_size = inputshape[0];
        let input_channels = inputshape[1];
        let input_height = inputshape[2];
        let input_width = inputshape[3];
        // Calculate padding
        let (pad_top, pad_bottom, pad_left, pad_right) = self.calculate_padding(inputshape);
        // Apply padding if needed
        let padded_input = if pad_top > 0 || pad_bottom > 0 || pad_left > 0 || pad_right > 0 {
            let padded_height = input_height + pad_top + pad_bottom;
            let padded_width = input_width + pad_left + pad_right;
            let mut padded =
                Array4::zeros([batch_size, input_channels, padded_height, padded_width]);
            for b in 0..batch_size {
                for c in 0..input_channels {
                    for h in 0..input_height {
                        for w in 0..input_width {
                            padded[[b, c, h + pad_top, w + pad_left]] = input[[b, c, h, w]];
                        }
                    }
                }
            }
            padded
        } else {
            input.clone()
        };
        // Calculate output dimensions
        let outputshape = self.calculate_outputshape(inputshape);
        let output_height = outputshape[2];
        let output_width = outputshape[3];
        let mut output = Array4::zeros(outputshape);
        // Perform convolution for each batch, filter, and spatial position
        for b in 0..batch_size {
            for f in 0..self.filters {
                for h_out in 0..output_height {
                    let h_in = h_out * self.stride.0;
                    for w_out in 0..output_width {
                        let w_in = w_out * self.stride.1;
                        // Compute convolution at this position
                        let mut val = 0.0;
                        for c in 0..input_channels {
                            for kh in 0..self.kernel_size.0 {
                                for kw in 0..self.kernel_size.1 {
                                    let h = h_in + kh;
                                    let w = w_in + kw;
                                    if h < padded_input.shape()[2] && w < padded_input.shape()[3] {
                                        val += padded_input[[b, c, h, w]]
                                            * self.weights[[f, c, kh, kw]];
                                    }
                                }
                            }
                        }
                        // Add bias
                        val += self.biases[f];
                        // Store result
                        output[[b, f, h_out, w_out]] = val;
                    }
                }
            }
        }
        output
    }
    /// Compute gradients for the convolution operation
    fn convolve_backward(&mut self, gradoutput: &Array4<f32>) -> Array4<f32> {
        let input = self
            .input
            .as_ref()
            .expect("Forward pass must be called first");
        let outputshape = gradoutput.shape();
        let batch_size = outputshape[0];
        let output_height = outputshape[2];
        let output_width = outputshape[3];
        let input_channels = input.shape()[1];
        let input_height = input.shape()[2];
        let input_width = input.shape()[3];

        // Calculate padding
        let (pad_top, pad_bottom, pad_left, pad_right) = self.calculate_padding(input.shape());

        // Apply padding to input if needed
        let padded_input = if pad_top > 0 || pad_bottom > 0 || pad_left > 0 || pad_right > 0 {
            let padded_height = input_height + pad_top + pad_bottom;
            let padded_width = input_width + pad_left + pad_right;
            let mut padded =
                Array4::zeros([batch_size, input_channels, padded_height, padded_width]);
            for b in 0..batch_size {
                for c in 0..input_channels {
                    for h in 0..input_height {
                        for w in 0..input_width {
                            padded[[b, c, h + pad_top, w + pad_left]] = input[[b, c, h, w]];
                        }
                    }
                }
            }
            padded
        } else {
            input.clone()
        };

        // Initialize gradients for weights and biases
        let mut dweights = Array4::zeros(self.weights.dim());
        let mut dbiases = Array1::zeros(self.filters);
        // Initialize input gradients
        let mut dinput = Array4::zeros(input.dim());
        // Compute gradients for biases (sum over batch, height, and width dimensions)
        for f in 0..self.filters {
            let mut sum = 0.0;
            for b in 0..batch_size {
                for h in 0..output_height {
                    for w in 0..output_width {
                        sum += gradoutput[[b, f, h, w]];
                    }
                }
            }
            dbiases[f] = sum;
        }
        // Compute gradients for weights
        for f in 0..self.filters {
            for c in 0..input_channels {
                for kh in 0..self.kernel_size.0 {
                    for kw in 0..self.kernel_size.1 {
                        let mut sum = 0.0;
                        for b in 0..batch_size {
                            for h_out in 0..output_height {
                                let h_in = h_out * self.stride.0;
                                for w_out in 0..output_width {
                                    let w_in = w_out * self.stride.1;
                                    let h = h_in + kh;
                                    let w = w_in + kw;
                                    if h < padded_input.shape()[2] && w < padded_input.shape()[3] {
                                        sum += padded_input[[b, c, h, w]]
                                            * gradoutput[[b, f, h_out, w_out]];
                                    }
                                }
                            }
                        }
                        dweights[[f, c, kh, kw]] = sum;
                    }
                }
            }
        }
        // Compute gradients for inputs (full convolution with transposed weights)
        let padded_height = padded_input.shape()[2];
        let padded_width = padded_input.shape()[3];
        // Rotate weights by 180 degrees (flip horizontally and vertically)
        let mut rotated_weights = Array4::zeros(self.weights.dim());
        for f in 0..self.filters {
            for c in 0..input_channels {
                for kh in 0..self.kernel_size.0 {
                    for kw in 0..self.kernel_size.1 {
                        rotated_weights[[f, c, kh, kw]] = self.weights[[
                            f,
                            c,
                            self.kernel_size.0 - 1 - kh,
                            self.kernel_size.1 - 1 - kw,
                        ]];
                    }
                }
            }
        }
        let mut padded_grad =
            Array4::zeros([batch_size, input_channels, padded_height, padded_width]);
        // Compute input gradients
        for b in 0..batch_size {
            for c in 0..input_channels {
                for h in 0..padded_height {
                    for w in 0..padded_width {
                        let mut sum = 0.0;
                        for f in 0..self.filters {
                            // Calculate bounds for the convolution window
                            let h_start = h.saturating_sub(self.kernel_size.0 - 1);
                            let h_end = (h / self.stride.0 + 1).min(output_height);
                            let w_start = w.saturating_sub(self.kernel_size.1 - 1);
                            let w_end = (w / self.stride.1 + 1).min(output_width);
                            for h_out in (h_start..h_end).step_by(self.stride.0) {
                                let kh = h - h_out * self.stride.0;
                                if kh >= self.kernel_size.0 {
                                    continue;
                                }
                                for w_out in (w_start..w_end).step_by(self.stride.1) {
                                    let kw = w - w_out * self.stride.1;
                                    if kw >= self.kernel_size.1 {
                                        continue;
                                    }
                                    sum += rotated_weights[[f, c, kh, kw]]
                                        * gradoutput[[b, f, h_out, w_out]];
                                }
                            }
                        }
                        padded_grad[[b, c, h, w]] = sum;
                    }
                }
            }
        }
        // Extract the gradient without padding
        for b in 0..batch_size {
            for c in 0..input_channels {
                for h in 0..input_height {
                    for w in 0..input_width {
                        dinput[[b, c, h, w]] = padded_grad[[b, c, h + pad_top, w + pad_left]];
                    }
                }
            }
        }

        // Store gradients
        self.dweights = Some(dweights);
        self.dbiases = Some(dbiases);
        dinput
    }
}

impl Layer for Conv2D {
    fn forward(&mut self, x: &Array4<f32>) -> Array4<f32> {
        // Store input for backward pass
        self.input = Some(x.clone());
        // Apply convolution
        let z = self.convolve(x);
        self.z = Some(z.clone());
        // Apply activation function
        let output = self.activation.apply(&z);
        self._output = Some(output.clone());
        output
    }
    fn backward(&mut self, gradoutput: &Array4<f32>) -> Array4<f32> {
        // Derivative of activation function
        let z = self.z.as_ref().expect("Forward pass must be called first");
        let activation_grad = self.activation.derivative(z);
        // Apply chain rule
        let grad_z = gradoutput * &activation_grad;
        // Compute gradients for weights, biases, and inputs
        self.convolve_backward(&grad_z)
    }
    fn update_parameters(&mut self, learningrate: f32) {
        if let (Some(dweights), Some(dbiases)) = (&self.dweights, &self.dbiases) {
            // Update weights
            self.weights = &self.weights - &(dweights * learningrate);
            // Update biases
            self.biases = &self.biases - &(dbiases * learningrate);
            // Clear gradients
            self.dweights = None;
            self.dbiases = None;
        }
    }
    fn get_description(&self) -> String {
        format!(
            "Conv2D: {}x{} kernels, {} filters, stride {:?}, padding {:?}, activation {}",
            self.kernel_size.0,
            self.kernel_size.1,
            self.filters,
            self.stride,
            match self.padding {
                PaddingMode::Valid => "valid",
                PaddingMode::Same => "same",
            },
            self.activation.as_str()
        )
    }

    fn num_parameters(&self) -> usize {
        self.weights.len() + self.biases.len()
    }
}
/// Max pooling layer
struct MaxPool2D {
    pool_size: (usize, usize),
    stride: (usize, usize),
    max_indices: Option<Array4<(usize, usize)>>, // Indices of max values
}

impl MaxPool2D {
    /// Create a new MaxPool2D layer
    fn new(pool_size: (usize, usize), stride: Option<(usize, usize)>) -> Self {
        // Default stride is same as pool size
        let stride = stride.unwrap_or(pool_size);
        Self {
            pool_size,
            stride,
            max_indices: None,
        }
    }
    /// Calculate output dimensions
    fn calculate_outputshape(&self, inputshape: &[usize]) -> [usize; 4] {
        let batch_size = inputshape[0];
        let channels = inputshape[1];
        let input_height = inputshape[2];
        let input_width = inputshape[3];
        let output_height = (input_height - self.pool_size.0) / self.stride.0 + 1;
        let output_width = (input_width - self.pool_size.1) / self.stride.1 + 1;
        [batch_size, channels, output_height, output_width]
    }
}
impl Layer for MaxPool2D {
    fn forward(&mut self, x: &Array4<f32>) -> Array4<f32> {
        let inputshape = x.shape();
        let batch_size = inputshape[0];
        let channels = inputshape[1];
        let _input_height = inputshape[2];
        let _input_width = inputshape[3];
        // Calculate output shape
        let outputshape = self.calculate_outputshape(inputshape);
        let output_height = outputshape[2];
        let output_width = outputshape[3];
        let mut output = Array4::zeros(outputshape);
        // Initialize output array and max indices
        let mut max_indices = Array4::from_elem(outputshape, (0, 0));
        // Perform max pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for h_out in 0..output_height {
                    let h_start = h_out * self.stride.0;
                    let h_end = h_start + self.pool_size.0;
                    for w_out in 0..output_width {
                        let w_start = w_out * self.stride.1;
                        let w_end = w_start + self.pool_size.1;
                        // Find maximum value in the pooling window
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_h = 0;
                        let mut max_w = 0;
                        for h in h_start..h_end {
                            for w in w_start..w_end {
                                let val = x[[b, c, h, w]];
                                if val > max_val {
                                    max_val = val;
                                    max_h = h;
                                    max_w = w;
                                }
                            }
                        }
                        output[[b, c, h_out, w_out]] = max_val;
                        max_indices[[b, c, h_out, w_out]] = (max_h, max_w);
                    }
                }
            }
        }
        // Store for backward pass
        self.max_indices = Some(max_indices);
        output
    }
    fn backward(&mut self, gradoutput: &Array4<f32>) -> Array4<f32> {
        let max_indices = self
            .max_indices
            .as_ref()
            .expect("Forward pass must be called first");

        let outputshape = gradoutput.shape();
        let batch_size = outputshape[0];
        let channels = outputshape[1];
        let output_height = outputshape[2];
        let output_width = outputshape[3];

        // Calculate input dimensions from _output and pooling parameters
        let input_height = (output_height - 1) * self.stride.0 + self.pool_size.0;
        let input_width = (output_width - 1) * self.stride.1 + self.pool_size.1;

        // Initialize gradient with respect to input
        let mut dinput = Array4::zeros([batch_size, channels, input_height, input_width]);

        // Distribute gradients to max locations
        for b in 0..batch_size {
            for c in 0..channels {
                for h_out in 0..output_height {
                    for w_out in 0..output_width {
                        let (max_h, max_w) = max_indices[[b, c, h_out, w_out]];
                        dinput[[b, c, max_h, max_w]] += gradoutput[[b, c, h_out, w_out]];
                    }
                }
            }
        }
        dinput
    }
    fn update_parameters(&mut self, _learningrate: f32) {
        // MaxPool has no parameters to update
    }

    fn get_description(&self) -> String {
        format!(
            "MaxPool2D: {}x{} pool size, stride {:?}",
            self.pool_size.0, self.pool_size.1, self.stride
        )
    }

    fn num_parameters(&self) -> usize {
        0 // No trainable parameters
    }
}
/// Flatten layer to convert from 4D to 2D (for fully connected layers)
struct Flatten {
    inputshape: Option<Vec<usize>>,
}

impl Flatten {
    /// Create a new Flatten layer
    fn new() -> Self {
        Self { inputshape: None }
    }
}
impl Layer for Flatten {
    fn forward(&mut self, x: &Array4<f32>) -> Array4<f32> {
        let inputshape = x.shape().to_vec();
        let batch_size = inputshape[0];
        // Calculate flattened size (excluding batch dimension)
        let flat_size: usize = inputshape[1..].iter().product();
        // Reshape to (batch_size, flat_size, 1, 1) - a 4D tensor with last two dims = 1
        let flattened = x
            .clone()
            .into_shape_with_order((batch_size, flat_size, 1, 1))
            .unwrap();
        // Store original shape for backward pass
        self.inputshape = Some(inputshape);
        flattened
    }
    fn backward(&mut self, gradoutput: &Array4<f32>) -> Array4<f32> {
        let inputshape = self
            .inputshape
            .as_ref()
            .expect("Forward pass must be called first");
        // Reshape back to original shape
        let reshaped = gradoutput
            .clone()
            .into_shape_with_order(inputshape.clone())
            .unwrap();
        let reshaped_4d: Array4<f32> = reshaped.into_dimensionality().unwrap();
        reshaped_4d
    }
    fn update_parameters(&mut self, _learningrate: f32) {
        // Flatten has no parameters to update
    }

    fn get_description(&self) -> String {
        "Flatten".to_string()
    }

    fn num_parameters(&self) -> usize {
        0 // No trainable parameters
    }
}
/// Dense (fully connected) layer
struct Dense {
    input_size: usize,
    output_size: usize,
    activation: ActivationFunction,
    weights: Array2<f32>,
    biases: Array1<f32>,
    dweights: Option<Array2<f32>>,
    dbiases: Option<Array1<f32>>,
    z: Option<Array4<f32>>,
    input: Option<Array4<f32>>,
}

impl Dense {
    /// Create a new Dense layer
    fn new(
        input_size: usize,
        output_size: usize,
        activation: ActivationFunction,
        rng: &mut SmallRng,
    ) -> Self {
        let std_dev = (2.0 / input_size as f32).sqrt();
        let dist = Uniform::new_inclusive(-std_dev, std_dev);
        // Initialize weights and biases
        let weights = Array2::from_shape_fn((input_size, output_size), |_| dist.sample(rng));
        let biases = Array1::zeros(output_size);
        Self {
            input_size,
            output_size,
            activation,
            weights,
            biases,
            dweights: None,
            dbiases: None,
            z: None,
            input: None,
        }
    }
}
impl Layer for Dense {
    fn forward(&mut self, x: &Array4<f32>) -> Array4<f32> {
        // Store input for backward pass
        self.input = Some(x.clone());
        let batch_size = x.shape()[0];
        // Reshape input from [batch_size, features, 1, 1] to [batch_size, features]
        let x_2d = x
            .clone()
            .into_shape_with_order((batch_size, self.input_size))
            .unwrap();
        // Compute linear transformation: z = x @ W + b
        let mut z_2d = Array2::<f32>::zeros((batch_size, self.output_size));
        for i in 0..batch_size {
            for j in 0..self.output_size {
                let mut sum = 0.0;
                for k in 0..self.input_size {
                    sum += x_2d[[i, k]] * self.weights[[k, j]];
                }
                z_2d[[i, j]] = sum + self.biases[j];
            }
        }
        let output_2d = self.activation.apply(&z_2d);
        // Reshape back to 4D: [batch_size, output_size, 1, 1]
        let z = z_2d
            .clone()
            .into_shape_with_order((batch_size, self.output_size, 1, 1))
            .unwrap();
        let output = output_2d
            .into_shape_with_order((batch_size, self.output_size, 1, 1))
            .unwrap();
        // Store for backward pass
        self.z = Some(z);
        output
    }
    fn backward(&mut self, gradoutput: &Array4<f32>) -> Array4<f32> {
        let input = self
            .input
            .as_ref()
            .expect("Forward pass must be called first");
        let z = self.z.as_ref().expect("Forward pass must be called first");
        let batch_size = input.shape()[0];
        // Reshape arrays to 2D
        let input_2d = input
            .clone()
            .into_shape_with_order((batch_size, self.input_size))
            .unwrap();
        let z_2d = z
            .clone()
            .into_shape_with_order((batch_size, self.output_size))
            .unwrap();
        let grad_output_2d = gradoutput
            .clone()
            .into_shape_with_order((batch_size, self.output_size))
            .unwrap();
        // Compute gradient of activation function
        let dactivation = self.activation.derivative(&z_2d);
        let delta = &grad_output_2d * &dactivation;
        // Compute gradients for weights and biases
        let dweights = input_2d.t().dot(&delta);
        let dbiases = delta.sum_axis(Axis(0));
        // Compute gradient for input
        let dinput_2d = delta.dot(&self.weights.t());
        // Reshape to 4D
        let dinput = dinput_2d.into_shape_with_order(input.dim()).unwrap();
        // Store gradients
        self.dweights = Some(dweights);
        self.dbiases = Some(dbiases);
        dinput
    }
    fn update_parameters(&mut self, learningrate: f32) {
        if let (Some(dweights), Some(dbiases)) = (&self.dweights, &self.dbiases) {
            // Update weights
            self.weights = &self.weights - &(dweights * learningrate);
            // Update biases
            self.biases = &self.biases - &(dbiases * learningrate);
            // Clear gradients
            self.dweights = None;
            self.dbiases = None;
        }
    }

    fn get_description(&self) -> String {
        format!(
            "Dense: {} -> {}, activation {}",
            self.input_size,
            self.output_size,
            self.activation.as_str()
        )
    }

    fn num_parameters(&self) -> usize {
        self.weights.len() + self.biases.len()
    }
}
/// Sequential model that chains multiple layers
struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    loss_fn: LossFunction,
}

impl Sequential {
    /// Create a new sequential model
    fn new(loss_fn: LossFunction) -> Self {
        Self {
            layers: Vec::new(),
            loss_fn,
        }
    }
    /// Add a layer to the model
    fn add<L: Layer + 'static>(&mut self, layer: L) -> &mut Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Forward pass through all layers
    fn forward(&mut self, x: &Array4<f32>) -> Array4<f32> {
        let mut output = x.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }
    /// Compute loss (for 2D output)
    fn compute_loss(&self, predictions: &Array4<f32>, targets: &Array2<f32>) -> f32 {
        let batch_size = predictions.shape()[0];
        let output_size = predictions.shape()[1];
        // Reshape predictions to 2D for loss computation
        let predictions_2d = predictions
            .clone()
            .into_shape_with_order((batch_size, output_size))
            .unwrap();
        self.loss_fn.compute(&predictions_2d, targets)
    }
    /// Backward pass and update parameters
    fn backward(&mut self, x: &Array4<f32>, y: &Array2<f32>, learningrate: f32) -> f32 {
        // Forward pass
        let predictions = self.forward(x);
        // Compute loss
        let loss = self.compute_loss(&predictions, y);
        // Reshape predictions to 2D for loss derivative
        let batch_size = predictions.shape()[0];
        let output_size = predictions.shape()[1];
        let predictions_2d = predictions
            .clone()
            .into_shape_with_order((batch_size, output_size))
            .unwrap();
        // Compute gradient of loss with respect to predictions
        let dloss = self.loss_fn.derivative(&predictions_2d, y);
        // Reshape back to 4D for backward pass
        let dloss_4d = dloss.into_shape_with_order(predictions.dim()).unwrap();
        // Backward pass through all layers
        let mut grad = dloss_4d;
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
        // Update parameters
        for layer in &mut self.layers {
            layer.update_parameters(learningrate);
        }
        loss
    }
    /// Train the model for a number of epochs
    #[allow(clippy::too_many_arguments)]
    fn train(
        &mut self,
        x: &Array4<f32>,
        y: &Array2<f32>,
        learning_rate: f32,
        epochs: usize,
        batch_size: usize,
    ) -> Vec<f32> {
        let n_samples = x.shape()[0];
        let mut losses = Vec::with_capacity(epochs);
        let mut rng = SmallRng::from_seed([42; 32]);
        for epoch in 0..epochs {
            // Shuffle the indices
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;
            // Process batches
            for batch_start in (0..n_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_samples);
                let batch_indices = &indices[batch_start..batch_end];
                // Create mini-batch
                let mut batch_x = Array4::<f32>::zeros([
                    batch_indices.len(),
                    x.shape()[1],
                    x.shape()[2],
                    x.shape()[3],
                ]);
                let mut batch_y = Array2::<f32>::zeros([batch_indices.len(), y.shape()[1]]);
                for (i, &idx) in batch_indices.iter().enumerate() {
                    for c in 0..x.shape()[1] {
                        for h in 0..x.shape()[2] {
                            for w in 0..x.shape()[3] {
                                batch_x[[i, c, h, w]] = x[[idx, c, h, w]];
                            }
                        }
                    }
                    for j in 0..y.shape()[1] {
                        batch_y[[i, j]] = y[[idx, j]];
                    }
                }
                // Train on mini-batch
                let batch_loss = self.backward(&batch_x, &batch_y, learning_rate);
                epoch_loss += batch_loss;
                batch_count += 1;
            }
            // Compute average loss
            epoch_loss /= batch_count as f32;
            losses.push(epoch_loss);
            // Print progress
            if epoch % 10 == 0 || epoch == epochs - 1 {
                println!("Epoch {}/{}: loss = {epoch_loss:.6}", epoch + 1, epochs);
            }
        }
        losses
    }
    /// Make predictions
    fn predict(&mut self, x: &Array4<f32>) -> Array2<f32> {
        let predictions = self.forward(x);
        let batch_size = predictions.shape()[0];
        let output_size = predictions.shape()[1];
        // Reshape to 2D for easier handling
        predictions
            .into_shape_with_order((batch_size, output_size))
            .unwrap()
    }
    /// Print a summary of the model
    fn summary(&self) {
        println!("Model Summary:");
        println!("==============");
        println!("Loss function: {}", self.loss_fn.as_str());
        println!("Number of layers: {}", self.layers.len());
        println!();
        println!("Layers:");
        println!("-------");
        let mut total_params = 0;
        for (i, layer) in self.layers.iter().enumerate() {
            let params = layer.num_parameters();
            total_params += params;
            println!(
                "Layer {}: {} - {} parameters",
                i + 1,
                layer.get_description(),
                params
            );
        }
        println!("Total parameters: {total_params}");
        println!();
    }
}
/// Create an MNIST-like synthetic dataset (small 4x4 images, 10 classes)
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn create_synthetic_dataset(
    num_samples: usize,
    num_classes: usize,
    image_size: (usize, usize),
    rng: &mut SmallRng,
) -> (Array4<f32>, Array2<f32>) {
    let channels = 1;
    let mut images = Array4::zeros([num_samples, channels, image_size.0, image_size.1]);
    let mut labels = Array2::zeros([num_samples, num_classes]);
    // Create synthetic patterns for each class
    let mut class_patterns = Vec::with_capacity(num_classes);
    for _ in 0..num_classes {
        let pattern =
            Array2::from_shape_fn(
                image_size,
                |_| {
                    if rng.gen::<f32>() > 0.7 {
                        1.0
                    } else {
                        0.0
                    }
                },
            );
        class_patterns.push(pattern);
    }
    // Generate _samples with noise
    for i in 0..num_samples {
        // Assign a random class
        let class = rng.gen_range(0..num_classes);
        // Add the class pattern with noise
        for h in 0..image_size.0 {
            for w in 0..image_size.1 {
                let noise = rng.gen::<f32>() * 0.3;
                let pixel = (class_patterns[class][[h, w]] + noise).min(1.0);
                images[[i, 0, h, w]] = pixel;
            }
        }
        // One-hot encode the label
        labels[[i, class]] = 1.0;
    }
    (images, labels)
}
// Helper function to find index of maximum value in array
#[allow(dead_code)]
fn argmax(arr: ArrayView1<f32>) -> usize {
    let mut max_idx = 0;
    let mut max_val = arr[0];
    for (idx, &val) in arr.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }
    max_idx
}
/// Train and evaluate a simple CNN model
#[allow(dead_code)]
fn train_cnn_example() -> Result<()> {
    // Set up RNG
    let mut rng = SmallRng::from_seed([42; 32]);
    // Create a synthetic dataset
    let num_samples = 1000;
    let num_classes = 10;
    let image_size = (8, 8);
    println!(
        "Creating synthetic dataset with {} samples, {} classes, {}x{} images",
        num_samples, num_classes, image_size.0, image_size.1
    );
    let (images, labels) = create_synthetic_dataset(num_samples, num_classes, image_size, &mut rng);
    // Split into training and test sets
    let train_size = (num_samples as f32 * 0.8) as usize;
    let test_size = num_samples - train_size;
    let train_images = images.slice(s![0..train_size, .., .., ..]).to_owned();
    let train_labels = labels.slice(s![0..train_size, ..]).to_owned();
    let test_images = images.slice(s![train_size.., .., .., ..]).to_owned();
    let test_labels = labels.slice(s![train_size.., ..]).to_owned();
    println!("Training set: {train_size} samples");
    println!("Test set: {test_size} samples");
    // Create a CNN model
    let mut model = Sequential::new(LossFunction::CategoricalCrossEntropy);
    // Add layers
    model.add(Conv2D::new(
        1,
        16,
        (3, 3),
        (1, 1),
        PaddingMode::Same,
        ActivationFunction::ReLU,
        &mut rng,
    ));
    model.add(MaxPool2D::new((2, 2), None));
    model.add(Conv2D::new(
        16,
        32,
        (3, 3),
        (1, 1),
        PaddingMode::Same,
        ActivationFunction::ReLU,
        &mut rng,
    ));
    model.add(MaxPool2D::new((2, 2), None));
    model.add(Flatten::new());
    // Calculate the size after flattening
    // After 2 max pooling operations with size (2,2), the image dimension is reduced to 1/4
    let flattened_size = 32 * (image_size.0 / 4) * (image_size.1 / 4);
    model.add(Dense::new(
        flattened_size,
        64,
        ActivationFunction::ReLU,
        &mut rng,
    ));
    model.add(Dense::new(
        64,
        num_classes,
        ActivationFunction::Sigmoid,
        &mut rng,
    ));
    // Print model summary
    model.summary();
    // Train the model
    println!("\nTraining model...");
    let learning_rate = 0.01;
    let batch_size = 32;
    let epochs = 50;
    let _losses = model.train(
        &train_images,
        &train_labels,
        learning_rate,
        epochs,
        batch_size,
    );
    // Evaluate on test set
    println!("\nEvaluating model on test set...");
    let predictions = model.predict(&test_images);
    // Calculate accuracy
    let mut correct = 0;
    for i in 0..test_size {
        let true_class = argmax(test_labels.row(i));
        let predicted_class = argmax(predictions.row(i));
        if true_class == predicted_class {
            correct += 1;
        }
    }
    let accuracy = correct as f32 / test_size as f32;
    println!(
        "Test accuracy: {:.2}% ({}/{})",
        accuracy * 100.0,
        correct,
        test_size
    );
    // Make some example predictions
    println!("\nExample predictions:");
    for i in 0..5 {
        let idx = rng.gen_range(0..test_size);
        let true_class = argmax(test_labels.row(idx));
        let predicted_class = argmax(predictions.row(idx));
        println!(
            "Example {}: True class = {}..Predicted class = {}",
            i + 1,
            true_class,
            predicted_class
        );
    }
    Ok(())
}
/// Main function
#[allow(dead_code)]
fn main() -> Result<()> {
    println!("Convolutional Neural Network Implementation Example\n");
    train_cnn_example()?;
    Ok(())
}
