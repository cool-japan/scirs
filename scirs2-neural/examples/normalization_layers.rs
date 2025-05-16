//! Normalization Layers for Neural Networks
//!
//! This example demonstrates normalization techniques that improve training
//! stability and performance:
//! - Batch Normalization
//! - Layer Normalization
//! - Group Normalization
//! - Instance Normalization

use autograd::rand::{
    distributions::{Distribution, Uniform},
    prelude::*,
    rngs::SmallRng,
    Rng, SeedableRng,
};
use ndarray::{s, Array, Array1, Array2, Array4, Axis};
use scirs2_neural::error::Result;
use std::f32;
use std::fmt::Debug;

/// Activation function type
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU(f32),
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
            ActivationFunction::LeakyReLU(alpha) => x.mapv(|v| if v > 0.0 { v } else { v * alpha }),
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
            ActivationFunction::LeakyReLU(alpha) => x.mapv(|v| if v > 0.0 { 1.0 } else { *alpha }),
            ActivationFunction::Linear => Array::ones(x.dim()),
        }
    }

    /// Get a string representation of the activation function
    fn to_string(&self) -> String {
        match self {
            ActivationFunction::ReLU => "ReLU".to_string(),
            ActivationFunction::Sigmoid => "Sigmoid".to_string(),
            ActivationFunction::Tanh => "Tanh".to_string(),
            ActivationFunction::LeakyReLU(alpha) => format!("LeakyReLU(alpha={})", alpha),
            ActivationFunction::Linear => "Linear".to_string(),
        }
    }
}

/// Base trait for all layers
trait Layer<T>
where
    T: std::fmt::Debug + Clone,
{
    /// Forward pass through the layer
    fn forward(&mut self, x: &T, is_training: bool) -> T;

    /// Backward pass to compute gradients
    fn backward(&mut self, grad_output: &T) -> T;

    /// Update parameters with gradients
    fn update_parameters(&mut self, learning_rate: f32);

    /// Get a description of the layer
    fn get_description(&self) -> String;

    /// Get number of trainable parameters
    fn num_parameters(&self) -> usize;
}

/// Batch Normalization layer
struct BatchNorm2D {
    num_features: usize,
    epsilon: f32,
    momentum: f32,

    // Learnable parameters
    gamma: Array1<f32>, // Scale parameter
    beta: Array1<f32>,  // Shift parameter

    // Gradients
    dgamma: Option<Array1<f32>>,
    dbeta: Option<Array1<f32>>,

    // Running statistics for inference
    running_mean: Array1<f32>,
    running_var: Array1<f32>,

    // Cache for backward pass
    input: Option<Array4<f32>>,
    batch_mean: Option<Array1<f32>>,
    batch_var: Option<Array1<f32>>,
    normalized: Option<Array4<f32>>,
    std_dev: Option<Array1<f32>>,
}

impl BatchNorm2D {
    /// Create a new BatchNorm2D layer
    fn new(num_features: usize, epsilon: f32, momentum: f32) -> Self {
        Self {
            num_features,
            epsilon,
            momentum,

            // Initialize gamma to ones and beta to zeros
            gamma: Array1::ones(num_features),
            beta: Array1::zeros(num_features),

            // Gradients
            dgamma: None,
            dbeta: None,

            // Running statistics
            running_mean: Array1::zeros(num_features),
            running_var: Array1::ones(num_features),

            // Cache
            input: None,
            batch_mean: None,
            batch_var: None,
            normalized: None,
            std_dev: None,
        }
    }
}

impl Layer<Array4<f32>> for BatchNorm2D {
    fn forward(&mut self, x: &Array4<f32>, is_training: bool) -> Array4<f32> {
        let batch_size = x.shape()[0];
        let channels = x.shape()[1];
        let height = x.shape()[2];
        let width = x.shape()[3];

        assert_eq!(
            channels, self.num_features,
            "Input channel dimension mismatch"
        );

        // Store input for backward pass
        self.input = Some(x.clone());

        // Create output array
        let mut output = Array4::zeros(x.dim());

        if is_training {
            // Compute batch statistics (mean and variance) along batch, height, width dimensions
            let mut batch_mean = Array1::zeros(channels);
            let mut batch_var = Array1::zeros(channels);

            // Compute mean
            for c in 0..channels {
                let mut sum = 0.0;
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            sum += x[[b, c, h, w]];
                        }
                    }
                }
                batch_mean[c] = sum / (batch_size * height * width) as f32;
            }

            // Compute variance
            for c in 0..channels {
                let mut sum_squared_diff = 0.0;
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let diff = x[[b, c, h, w]] - batch_mean[c];
                            sum_squared_diff += diff * diff;
                        }
                    }
                }
                batch_var[c] = sum_squared_diff / (batch_size * height * width) as f32;
            }

            // Compute standard deviation
            let std_dev = batch_var.mapv(|v| (v + self.epsilon).sqrt());

            // Normalize the input
            let mut normalized = Array4::zeros(x.dim());
            for b in 0..batch_size {
                for c in 0..channels {
                    for h in 0..height {
                        for w in 0..width {
                            normalized[[b, c, h, w]] =
                                (x[[b, c, h, w]] - batch_mean[c]) / std_dev[c];
                        }
                    }
                }
            }

            // Scale and shift
            for b in 0..batch_size {
                for c in 0..channels {
                    for h in 0..height {
                        for w in 0..width {
                            output[[b, c, h, w]] =
                                self.gamma[c] * normalized[[b, c, h, w]] + self.beta[c];
                        }
                    }
                }
            }

            // Update running statistics
            self.running_mean =
                &self.running_mean * self.momentum + &batch_mean * (1.0 - self.momentum);
            self.running_var =
                &self.running_var * self.momentum + &batch_var * (1.0 - self.momentum);

            // Store for backward pass
            self.batch_mean = Some(batch_mean);
            self.batch_var = Some(batch_var);
            self.normalized = Some(normalized);
            self.std_dev = Some(std_dev);
        } else {
            // During inference, use running statistics
            for b in 0..batch_size {
                for c in 0..channels {
                    for h in 0..height {
                        for w in 0..width {
                            let normalized = (x[[b, c, h, w]] - self.running_mean[c])
                                / (self.running_var[c] + self.epsilon).sqrt();
                            output[[b, c, h, w]] = self.gamma[c] * normalized + self.beta[c];
                        }
                    }
                }
            }
        }

        output
    }

    fn backward(&mut self, grad_output: &Array4<f32>) -> Array4<f32> {
        let input = self
            .input
            .as_ref()
            .expect("Forward pass must be called first");
        let batch_mean = self
            .batch_mean
            .as_ref()
            .expect("Forward pass must be called first");
        let std_dev = self
            .std_dev
            .as_ref()
            .expect("Forward pass must be called first");
        let normalized = self
            .normalized
            .as_ref()
            .expect("Forward pass must be called first");

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let height = input.shape()[2];
        let width = input.shape()[3];
        let n_pixels = (batch_size * height * width) as f32;

        // Initialize gradient arrays
        let mut dgamma = Array1::zeros(channels);
        let mut dbeta = Array1::zeros(channels);
        let mut dinput = Array4::zeros(input.dim());

        // Compute gradients for gamma and beta
        for c in 0..channels {
            let mut sum_dgamma = 0.0;
            let mut sum_dbeta = 0.0;

            for b in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        sum_dgamma += grad_output[[b, c, h, w]] * normalized[[b, c, h, w]];
                        sum_dbeta += grad_output[[b, c, h, w]];
                    }
                }
            }

            dgamma[c] = sum_dgamma;
            dbeta[c] = sum_dbeta;
        }

        // Compute gradient with respect to normalized inputs
        let mut dxhat = Array4::zeros(input.dim());
        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        dxhat[[b, c, h, w]] = grad_output[[b, c, h, w]] * self.gamma[c];
                    }
                }
            }
        }

        // Compute gradient with respect to variance
        let mut dvar = Array1::zeros(channels);
        for c in 0..channels {
            let mut sum = 0.0;
            for b in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        sum += dxhat[[b, c, h, w]]
                            * (input[[b, c, h, w]] - batch_mean[c])
                            * (-0.5)
                            * std_dev[c].powi(-3);
                    }
                }
            }
            dvar[c] = sum;
        }

        // Compute gradient with respect to mean
        let mut dmean = Array1::zeros(channels);
        for c in 0..channels {
            let mut sum1 = 0.0;
            let mut sum2 = 0.0;

            for b in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        sum1 += dxhat[[b, c, h, w]] * (-1.0 / std_dev[c]);
                        sum2 += -2.0 * (input[[b, c, h, w]] - batch_mean[c]);
                    }
                }
            }

            dmean[c] = sum1 + dvar[c] * sum2 / n_pixels;
        }

        // Compute gradient with respect to input
        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        dinput[[b, c, h, w]] = dxhat[[b, c, h, w]] / std_dev[c]
                            + dvar[c] * 2.0 * (input[[b, c, h, w]] - batch_mean[c]) / n_pixels
                            + dmean[c] / n_pixels;
                    }
                }
            }
        }

        // Store gradients
        self.dgamma = Some(dgamma);
        self.dbeta = Some(dbeta);

        dinput
    }

    fn update_parameters(&mut self, learning_rate: f32) {
        if let (Some(dgamma), Some(dbeta)) = (&self.dgamma, &self.dbeta) {
            // Update gamma and beta
            self.gamma = &self.gamma - learning_rate * dgamma;
            self.beta = &self.beta - learning_rate * dbeta;

            // Clear gradients
            self.dgamma = None;
            self.dbeta = None;
        }
    }

    fn get_description(&self) -> String {
        format!(
            "BatchNorm2D: {} features, epsilon={}, momentum={}",
            self.num_features, self.epsilon, self.momentum
        )
    }

    fn num_parameters(&self) -> usize {
        // Gamma and beta
        2 * self.num_features
    }
}

/// Layer Normalization (normalizes over features)
struct LayerNorm {
    normalized_shape: Vec<usize>,
    epsilon: f32,

    // Learnable parameters
    gamma: Array1<f32>, // Scale parameter
    beta: Array1<f32>,  // Shift parameter

    // Gradients
    dgamma: Option<Array1<f32>>,
    dbeta: Option<Array1<f32>>,

    // Cache for backward pass
    input: Option<Array2<f32>>,
    normalized: Option<Array2<f32>>,
    std_dev: Option<Array1<f32>>,
    mean: Option<Array1<f32>>,
}

impl LayerNorm {
    /// Create a new LayerNorm layer
    fn new(normalized_shape: Vec<usize>, epsilon: f32) -> Self {
        // Calculate number of features
        let num_features: usize = normalized_shape.iter().product();

        Self {
            normalized_shape,
            epsilon,

            // Initialize gamma to ones and beta to zeros
            gamma: Array1::ones(num_features),
            beta: Array1::zeros(num_features),

            // Gradients
            dgamma: None,
            dbeta: None,

            // Cache
            input: None,
            normalized: None,
            std_dev: None,
            mean: None,
        }
    }
}

impl Layer<Array2<f32>> for LayerNorm {
    fn forward(&mut self, x: &Array2<f32>, _is_training: bool) -> Array2<f32> {
        let batch_size = x.shape()[0];
        let feature_size = x.shape()[1];

        // Store input for backward pass
        self.input = Some(x.clone());

        // Create output array
        let mut output = Array2::zeros(x.dim());
        let mut normalized = Array2::zeros(x.dim());
        let mut mean = Array1::zeros(batch_size);
        let mut std_dev = Array1::zeros(batch_size);

        // Compute mean for each sample (along feature dimension)
        for b in 0..batch_size {
            let mut sum = 0.0;
            for f in 0..feature_size {
                sum += x[[b, f]];
            }
            mean[b] = sum / feature_size as f32;
        }

        // Compute variance for each sample
        for b in 0..batch_size {
            let mut sum_squared_diff = 0.0;
            for f in 0..feature_size {
                let diff = x[[b, f]] - mean[b];
                sum_squared_diff += diff * diff;
            }
            let variance = sum_squared_diff / feature_size as f32;
            std_dev[b] = (variance + self.epsilon).sqrt();
        }

        // Normalize, scale and shift
        for b in 0..batch_size {
            for f in 0..feature_size {
                normalized[[b, f]] = (x[[b, f]] - mean[b]) / std_dev[b];
                output[[b, f]] = self.gamma[f] * normalized[[b, f]] + self.beta[f];
            }
        }

        // Store for backward pass
        self.normalized = Some(normalized);
        self.std_dev = Some(std_dev);
        self.mean = Some(mean);

        output
    }

    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        let input = self
            .input
            .as_ref()
            .expect("Forward pass must be called first");
        let normalized = self
            .normalized
            .as_ref()
            .expect("Forward pass must be called first");
        let std_dev = self
            .std_dev
            .as_ref()
            .expect("Forward pass must be called first");
        let mean = self
            .mean
            .as_ref()
            .expect("Forward pass must be called first");

        let batch_size = input.shape()[0];
        let feature_size = input.shape()[1];
        let n_features = feature_size as f32;

        // Initialize gradient arrays
        let mut dgamma = Array1::zeros(feature_size);
        let mut dbeta = Array1::zeros(feature_size);
        let mut dinput = Array2::zeros(input.dim());

        // Compute gradients for gamma and beta
        for f in 0..feature_size {
            let mut sum_dgamma = 0.0;
            let mut sum_dbeta = 0.0;

            for b in 0..batch_size {
                sum_dgamma += grad_output[[b, f]] * normalized[[b, f]];
                sum_dbeta += grad_output[[b, f]];
            }

            dgamma[f] = sum_dgamma;
            dbeta[f] = sum_dbeta;
        }

        // Compute gradient with respect to normalized inputs
        let mut dxhat = Array2::zeros(input.dim());
        for b in 0..batch_size {
            for f in 0..feature_size {
                dxhat[[b, f]] = grad_output[[b, f]] * self.gamma[f];
            }
        }

        // Compute gradients with respect to mean and variance
        let mut dmean = Array1::zeros(batch_size);
        let mut dvar = Array1::zeros(batch_size);

        for b in 0..batch_size {
            let mut sum_dmean = 0.0;
            let mut sum_dvar = 0.0;

            for f in 0..feature_size {
                sum_dmean += dxhat[[b, f]] * (-1.0 / std_dev[b]);
                sum_dvar +=
                    dxhat[[b, f]] * (input[[b, f]] - mean[b]) * (-0.5) * std_dev[b].powi(-3);
            }

            dmean[b] = sum_dmean;
            dvar[b] = sum_dvar;
        }

        // Compute gradient with respect to input
        for b in 0..batch_size {
            for f in 0..feature_size {
                let dvar_term = 2.0 * (input[[b, f]] - mean[b]) / n_features;

                dinput[[b, f]] =
                    dxhat[[b, f]] / std_dev[b] + dvar[b] * dvar_term + dmean[b] / n_features;
            }
        }

        // Store gradients
        self.dgamma = Some(dgamma);
        self.dbeta = Some(dbeta);

        dinput
    }

    fn update_parameters(&mut self, learning_rate: f32) {
        if let (Some(dgamma), Some(dbeta)) = (&self.dgamma, &self.dbeta) {
            // Update gamma and beta
            self.gamma = &self.gamma - learning_rate * dgamma;
            self.beta = &self.beta - learning_rate * dbeta;

            // Clear gradients
            self.dgamma = None;
            self.dbeta = None;
        }
    }

    fn get_description(&self) -> String {
        format!(
            "LayerNorm: normalized_shape={:?}, epsilon={}",
            self.normalized_shape, self.epsilon
        )
    }

    fn num_parameters(&self) -> usize {
        // Gamma and beta
        2 * self.gamma.len()
    }
}

/// Group Normalization layer
struct GroupNorm {
    num_groups: usize,
    num_channels: usize,
    epsilon: f32,

    // Learnable parameters
    gamma: Array1<f32>, // Scale parameter
    beta: Array1<f32>,  // Shift parameter

    // Gradients
    dgamma: Option<Array1<f32>>,
    dbeta: Option<Array1<f32>>,

    // Cache for backward pass
    input: Option<Array4<f32>>,
    normalized: Option<Array4<f32>>,
    group_std: Option<Array2<f32>>,
    group_mean: Option<Array2<f32>>,
}

impl GroupNorm {
    /// Create a new GroupNorm layer
    fn new(num_groups: usize, num_channels: usize, epsilon: f32) -> Self {
        assert!(
            num_channels % num_groups == 0,
            "Number of channels must be divisible by number of groups"
        );

        Self {
            num_groups,
            num_channels,
            epsilon,

            // Initialize gamma to ones and beta to zeros
            gamma: Array1::ones(num_channels),
            beta: Array1::zeros(num_channels),

            // Gradients
            dgamma: None,
            dbeta: None,

            // Cache
            input: None,
            normalized: None,
            group_std: None,
            group_mean: None,
        }
    }
}

impl Layer<Array4<f32>> for GroupNorm {
    fn forward(&mut self, x: &Array4<f32>, _is_training: bool) -> Array4<f32> {
        let batch_size = x.shape()[0];
        let channels = x.shape()[1];
        let height = x.shape()[2];
        let width = x.shape()[3];

        assert_eq!(channels, self.num_channels, "Channel dimension mismatch");

        // Store input for backward pass
        self.input = Some(x.clone());

        let channels_per_group = channels / self.num_groups;
        let pixels_per_group = height * width * channels_per_group;

        // Create output array
        let mut output = Array4::zeros(x.dim());
        let mut normalized = Array4::zeros(x.dim());
        let mut group_mean = Array2::zeros((batch_size, self.num_groups));
        let mut group_std = Array2::zeros((batch_size, self.num_groups));

        // Compute mean and std for each group
        for b in 0..batch_size {
            for g in 0..self.num_groups {
                let start_c = g * channels_per_group;
                let end_c = (g + 1) * channels_per_group;

                // Compute mean
                let mut sum = 0.0;
                for c in start_c..end_c {
                    for h in 0..height {
                        for w in 0..width {
                            sum += x[[b, c, h, w]];
                        }
                    }
                }
                group_mean[[b, g]] = sum / pixels_per_group as f32;

                // Compute variance
                let mut sum_squared_diff = 0.0;
                for c in start_c..end_c {
                    for h in 0..height {
                        for w in 0..width {
                            let diff = x[[b, c, h, w]] - group_mean[[b, g]];
                            sum_squared_diff += diff * diff;
                        }
                    }
                }
                let variance = sum_squared_diff / pixels_per_group as f32;
                group_std[[b, g]] = (variance + self.epsilon).sqrt();
            }
        }

        // Normalize, scale and shift
        for b in 0..batch_size {
            for c in 0..channels {
                let g = c / channels_per_group;

                for h in 0..height {
                    for w in 0..width {
                        normalized[[b, c, h, w]] =
                            (x[[b, c, h, w]] - group_mean[[b, g]]) / group_std[[b, g]];
                        output[[b, c, h, w]] =
                            self.gamma[c] * normalized[[b, c, h, w]] + self.beta[c];
                    }
                }
            }
        }

        // Store for backward pass
        self.normalized = Some(normalized);
        self.group_std = Some(group_std);
        self.group_mean = Some(group_mean);

        output
    }

    fn backward(&mut self, grad_output: &Array4<f32>) -> Array4<f32> {
        let input = self
            .input
            .as_ref()
            .expect("Forward pass must be called first");
        let normalized = self
            .normalized
            .as_ref()
            .expect("Forward pass must be called first");
        let group_std = self
            .group_std
            .as_ref()
            .expect("Forward pass must be called first");
        let group_mean = self
            .group_mean
            .as_ref()
            .expect("Forward pass must be called first");

        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let height = input.shape()[2];
        let width = input.shape()[3];

        let channels_per_group = channels / self.num_groups;
        let pixels_per_group = height * width * channels_per_group;
        let n_pixels_per_group = pixels_per_group as f32;

        // Initialize gradient arrays
        let mut dgamma = Array1::zeros(channels);
        let mut dbeta = Array1::zeros(channels);
        let mut dinput = Array4::zeros(input.dim());

        // Compute gradients for gamma and beta
        for c in 0..channels {
            let mut sum_dgamma = 0.0;
            let mut sum_dbeta = 0.0;

            for b in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        sum_dgamma += grad_output[[b, c, h, w]] * normalized[[b, c, h, w]];
                        sum_dbeta += grad_output[[b, c, h, w]];
                    }
                }
            }

            dgamma[c] = sum_dgamma;
            dbeta[c] = sum_dbeta;
        }

        // Compute gradient with respect to normalized inputs
        let mut dxhat = Array4::zeros(input.dim());
        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        dxhat[[b, c, h, w]] = grad_output[[b, c, h, w]] * self.gamma[c];
                    }
                }
            }
        }

        // Compute gradients with respect to mean and variance for each group
        let mut dgroup_mean = Array2::zeros((batch_size, self.num_groups));
        let mut dgroup_var = Array2::zeros((batch_size, self.num_groups));

        for b in 0..batch_size {
            for g in 0..self.num_groups {
                let start_c = g * channels_per_group;
                let end_c = (g + 1) * channels_per_group;

                let mut sum_dmean = 0.0;
                let mut sum_dvar = 0.0;

                for c in start_c..end_c {
                    for h in 0..height {
                        for w in 0..width {
                            sum_dmean += dxhat[[b, c, h, w]] * (-1.0 / group_std[[b, g]]);
                            sum_dvar += dxhat[[b, c, h, w]]
                                * (input[[b, c, h, w]] - group_mean[[b, g]])
                                * (-0.5)
                                * group_std[[b, g]].powi(-3);
                        }
                    }
                }

                dgroup_mean[[b, g]] = sum_dmean;
                dgroup_var[[b, g]] = sum_dvar;
            }
        }

        // Compute gradient with respect to input
        for b in 0..batch_size {
            for c in 0..channels {
                let g = c / channels_per_group;

                for h in 0..height {
                    for w in 0..width {
                        dinput[[b, c, h, w]] = dxhat[[b, c, h, w]] / group_std[[b, g]]
                            + dgroup_var[[b, g]] * 2.0 * (input[[b, c, h, w]] - group_mean[[b, g]])
                                / n_pixels_per_group
                            + dgroup_mean[[b, g]] / n_pixels_per_group;
                    }
                }
            }
        }

        // Store gradients
        self.dgamma = Some(dgamma);
        self.dbeta = Some(dbeta);

        dinput
    }

    fn update_parameters(&mut self, learning_rate: f32) {
        if let (Some(dgamma), Some(dbeta)) = (&self.dgamma, &self.dbeta) {
            // Update gamma and beta
            self.gamma = &self.gamma - learning_rate * dgamma;
            self.beta = &self.beta - learning_rate * dbeta;

            // Clear gradients
            self.dgamma = None;
            self.dbeta = None;
        }
    }

    fn get_description(&self) -> String {
        format!(
            "GroupNorm: {} groups, {} channels, epsilon={}",
            self.num_groups, self.num_channels, self.epsilon
        )
    }

    fn num_parameters(&self) -> usize {
        // Gamma and beta
        2 * self.num_channels
    }
}

/// Simple dense layer for examples
struct Dense {
    input_size: usize,
    output_size: usize,
    activation: ActivationFunction,

    // Parameters
    weights: Array2<f32>,
    biases: Array1<f32>,

    // Gradients
    dweights: Option<Array2<f32>>,
    dbiases: Option<Array1<f32>>,

    // Cache for backward pass
    input: Option<Array2<f32>>,
    z: Option<Array2<f32>>,
}

impl Dense {
    /// Create a new dense layer
    fn new(
        input_size: usize,
        output_size: usize,
        activation: ActivationFunction,
        rng: &mut SmallRng,
    ) -> Self {
        // He/Kaiming initialization for weights
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
            input: None,
            z: None,
        }
    }
}

impl Layer<Array2<f32>> for Dense {
    fn forward(&mut self, x: &Array2<f32>, _is_training: bool) -> Array2<f32> {
        // Store input for backward pass
        self.input = Some(x.clone());

        // Linear transformation: z = x @ W + b
        let mut z = x.dot(&self.weights);
        for i in 0..z.shape()[0] {
            for j in 0..z.shape()[1] {
                z[[i, j]] += self.biases[j];
            }
        }

        // Store pre-activation output
        self.z = Some(z.clone());

        // Apply activation
        self.activation.apply(&z)
    }

    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        let input = self
            .input
            .as_ref()
            .expect("Forward pass must be called first");
        let z = self.z.as_ref().expect("Forward pass must be called first");

        // Compute gradient through activation
        let dactivation = self.activation.derivative(z);
        let delta = grad_output * &dactivation;

        // Compute gradients for weights and biases
        let dweights = input.t().dot(&delta);
        let dbiases = delta.sum_axis(Axis(0));

        // Store gradients
        self.dweights = Some(dweights);
        self.dbiases = Some(dbiases);

        // Compute gradient with respect to input
        delta.dot(&self.weights.t())
    }

    fn update_parameters(&mut self, learning_rate: f32) {
        if let (Some(dweights), Some(dbiases)) = (&self.dweights, &self.dbiases) {
            // Update weights and biases
            self.weights = &self.weights - learning_rate * dweights;
            self.biases = &self.biases - learning_rate * dbiases;

            // Clear gradients
            self.dweights = None;
            self.dbiases = None;
        }
    }

    fn get_description(&self) -> String {
        format!(
            "Dense: {} -> {}, Activation: {}",
            self.input_size,
            self.output_size,
            self.activation.to_string()
        )
    }

    fn num_parameters(&self) -> usize {
        self.input_size * self.output_size + self.output_size
    }
}

/// Loss function type
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum LossFunction {
    MSE,
    CrossEntropy,
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
            LossFunction::CrossEntropy => {
                let epsilon = 1e-15;
                let mut sum = 0.0;

                for i in 0..predictions.shape()[0] {
                    let mut row_sum = 0.0;
                    for j in 0..predictions.shape()[1] {
                        let pred = predictions[[i, j]].max(epsilon).min(1.0 - epsilon);
                        let target = targets[[i, j]];

                        if target > 0.0 {
                            row_sum += target * pred.ln();
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
            LossFunction::CrossEntropy => {
                // Simplified gradient for cross-entropy: -y/ŷ
                let epsilon = 1e-15;
                let n = predictions.shape()[0] as f32;

                Array2::from_shape_fn(predictions.dim(), |(i, j)| {
                    let y_pred = predictions[[i, j]].max(epsilon).min(1.0 - epsilon);
                    let y_true = targets[[i, j]];

                    if y_true > 0.0 {
                        -y_true / y_pred / n
                    } else {
                        0.0
                    }
                })
            }
        }
    }
}

/// Example of a network with BatchNorm
fn example_batchnorm() -> Result<()> {
    println!("Batch Normalization Example\n");

    // Create a simple dataset
    let mut rng = SmallRng::seed_from_u64(42);
    let batch_size = 32;
    let channels = 3;
    let height = 16;
    let width = 16;

    // Create random input tensor
    let x = Array4::from_shape_fn((batch_size, channels, height, width), |_| {
        rng.gen::<f32>() * 2.0 - 1.0 // Random values between -1 and 1
    });

    // Create a BatchNorm2D layer
    let mut bn = BatchNorm2D::new(channels, 1e-5, 0.1);

    // Print layer description
    println!("{}", bn.get_description());
    println!("Number of parameters: {}", bn.num_parameters());

    // Forward pass
    println!("\nForward pass:");
    let output = bn.forward(&x, true);

    // Print statistics of input and output
    println!("Input mean: {:.6}", mean_channels(&x));
    println!("Input std: {:.6}", std_channels(&x));
    println!("Output mean: {:.6}", mean_channels(&output));
    println!("Output std: {:.6}", std_channels(&output));

    // Simple backward pass example (gradient of ones)
    let grad_output = Array4::ones(output.dim());
    let _grad_input = bn.backward(&grad_output);

    // Update parameters
    bn.update_parameters(0.01);

    // Forward pass in evaluation mode
    println!("\nForward pass (evaluation mode):");
    let eval_output = bn.forward(&x, false);
    println!("Output mean: {:.6}", mean_channels(&eval_output));
    println!("Output std: {:.6}", std_channels(&eval_output));

    Ok(())
}

/// Example of layer normalization
fn example_layernorm() -> Result<()> {
    println!("\nLayer Normalization Example\n");

    // Create a simple dataset
    let mut rng = SmallRng::seed_from_u64(42);
    let batch_size = 32;
    let features = 64;

    // Create random input tensor
    let x = Array2::from_shape_fn((batch_size, features), |_| {
        rng.gen::<f32>() * 2.0 - 1.0 // Random values between -1 and 1
    });

    // Create a LayerNorm layer
    let mut ln = LayerNorm::new(vec![features], 1e-5);

    // Print layer description
    println!("{}", ln.get_description());
    println!("Number of parameters: {}", ln.num_parameters());

    // Forward pass
    println!("\nForward pass:");
    let output = ln.forward(&x, true);

    // Print statistics of input and output
    println!("Input mean: {:.6}", mean_batch(&x));
    println!("Input std: {:.6}", std_batch(&x));
    println!("Output mean: {:.6}", mean_batch(&output));
    println!("Output std: {:.6}", std_batch(&output));

    // Simple backward pass example (gradient of ones)
    let grad_output = Array2::ones(output.dim());
    let _grad_input = ln.backward(&grad_output);

    // Update parameters
    ln.update_parameters(0.01);

    Ok(())
}

/// Example of group normalization
fn example_groupnorm() -> Result<()> {
    println!("\nGroup Normalization Example\n");

    // Create a simple dataset
    let mut rng = SmallRng::seed_from_u64(42);
    let batch_size = 32;
    let channels = 16; // Must be divisible by num_groups
    let num_groups = 4;
    let height = 16;
    let width = 16;

    // Create random input tensor
    let x = Array4::from_shape_fn((batch_size, channels, height, width), |_| {
        rng.gen::<f32>() * 2.0 - 1.0 // Random values between -1 and 1
    });

    // Create a GroupNorm layer
    let mut gn = GroupNorm::new(num_groups, channels, 1e-5);

    // Print layer description
    println!("{}", gn.get_description());
    println!("Number of parameters: {}", gn.num_parameters());

    // Forward pass
    println!("\nForward pass:");
    let output = gn.forward(&x, true);

    // Print statistics of input and output
    println!("Input mean: {:.6}", mean_channels(&x));
    println!("Input std: {:.6}", std_channels(&x));
    println!("Output mean: {:.6}", mean_channels(&output));
    println!("Output std: {:.6}", std_channels(&output));

    // Simple backward pass example (gradient of ones)
    let grad_output = Array4::ones(output.dim());
    let _grad_input = gn.backward(&grad_output);

    // Update parameters
    gn.update_parameters(0.01);

    Ok(())
}

/// Example of normalization in a simple network for MNIST-like dataset
fn compare_normalization_methods() -> Result<()> {
    println!("\nComparing Normalization Methods for MNIST-like Dataset\n");

    // Create a synthetic dataset
    let mut rng = SmallRng::seed_from_u64(42);
    let n_samples = 1000;
    let n_features = 784; // 28x28
    let n_classes = 10;

    // Generate synthetic data
    let x_train = Array2::from_shape_fn((n_samples, n_features), |_| {
        rng.gen::<f32>() // Values between 0 and 1
    });

    // Generate one-hot encoded labels
    let mut y_train = Array2::zeros((n_samples, n_classes));
    for i in 0..n_samples {
        let class = (rng.gen::<f32>() * n_classes as f32).floor() as usize;
        y_train[[i, class]] = 1.0;
    }

    // Split into training and validation sets
    let train_size = (n_samples as f32 * 0.8) as usize;
    let val_size = n_samples - train_size;

    let x_train_split = x_train.slice(s![0..train_size, ..]).to_owned();
    let y_train_split = y_train.slice(s![0..train_size, ..]).to_owned();
    let x_val = x_train.slice(s![train_size.., ..]).to_owned();
    let y_val = y_train.slice(s![train_size.., ..]).to_owned();

    println!(
        "Generated dataset: {} training samples, {} validation samples",
        train_size, val_size
    );

    // Define network configurations
    let configurations = vec![
        ("No Normalization", false, false, false),
        ("With BatchNorm", true, false, false),
        ("With LayerNorm", false, true, false),
        ("With Both", true, true, false),
    ];

    // Compare different configurations
    for (name, use_batchnorm, use_layernorm, _use_residual) in configurations {
        println!("\n--- {} ---", name);

        // Create a network
        let mut network = create_network(
            n_features,
            n_classes,
            use_batchnorm,
            use_layernorm,
            &mut rng,
        );

        // Train the network
        let train_losses = train_network(
            &mut network,
            &x_train_split,
            &y_train_split,
            &x_val,
            &y_val,
            20,
            32,
            0.01,
        )?;

        // Print final losses
        println!("Final training loss: {:.6}", train_losses.last().unwrap().0);
        println!(
            "Final validation loss: {:.6}",
            train_losses.last().unwrap().1
        );

        // Print final metrics
        let train_acc = evaluate(&mut network, &x_train_split, &y_train_split)?;
        let val_acc = evaluate(&mut network, &x_val, &y_val)?;
        println!("Training accuracy: {:.2}%", train_acc * 100.0);
        println!("Validation accuracy: {:.2}%", val_acc * 100.0);
    }

    Ok(())
}

/// Create a simple network with optional normalization layers
fn create_network(
    input_size: usize,
    output_size: usize,
    use_batchnorm: bool,
    use_layernorm: bool,
    rng: &mut SmallRng,
) -> Vec<Box<dyn Layer<Array2<f32>>>> {
    let mut layers: Vec<Box<dyn Layer<Array2<f32>>>> = Vec::new();

    // Hidden layer 1
    layers.push(Box::new(Dense::new(
        input_size,
        128,
        ActivationFunction::ReLU,
        rng,
    )));

    // Add BatchNorm after first dense layer
    if use_batchnorm {
        layers.push(Box::new(LayerNorm::new(vec![128], 1e-5)));
    }

    // Hidden layer 2
    layers.push(Box::new(Dense::new(128, 64, ActivationFunction::ReLU, rng)));

    // Add LayerNorm after second dense layer
    if use_layernorm {
        layers.push(Box::new(LayerNorm::new(vec![64], 1e-5)));
    }

    // Output layer
    layers.push(Box::new(Dense::new(
        64,
        output_size,
        ActivationFunction::Sigmoid,
        rng,
    )));

    layers
}

/// Train a network for a number of epochs
fn train_network(
    network: &mut Vec<Box<dyn Layer<Array2<f32>>>>,
    x_train: &Array2<f32>,
    y_train: &Array2<f32>,
    x_val: &Array2<f32>,
    y_val: &Array2<f32>,
    epochs: usize,
    batch_size: usize,
    learning_rate: f32,
) -> Result<Vec<(f32, f32)>> {
    let loss_fn = LossFunction::CrossEntropy;
    let n_samples = x_train.shape()[0];
    let mut losses = Vec::new();

    let mut rng = SmallRng::seed_from_u64(42);

    for epoch in 0..epochs {
        // Shuffle data
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        // Train on batches
        for batch_start in (0..n_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_samples);
            let batch_size = batch_end - batch_start;

            // Create batch
            let mut batch_x = Array2::zeros((batch_size, x_train.shape()[1]));
            let mut batch_y = Array2::zeros((batch_size, y_train.shape()[1]));

            for (i, &idx) in indices[batch_start..batch_end].iter().enumerate() {
                batch_x.row_mut(i).assign(&x_train.row(idx));
                batch_y.row_mut(i).assign(&y_train.row(idx));
            }

            // Forward pass
            let mut output = batch_x.clone();
            for layer in network.iter_mut() {
                output = layer.forward(&output, true);
            }

            // Compute loss
            let batch_loss = loss_fn.compute(&output, &batch_y);
            epoch_loss += batch_loss;
            batch_count += 1;

            // Backward pass
            let mut grad = loss_fn.derivative(&output, &batch_y);
            for layer in network.iter_mut().rev() {
                grad = layer.backward(&grad);
            }

            // Update parameters
            for layer in network.iter_mut() {
                layer.update_parameters(learning_rate);
            }
        }

        // Compute average loss for epoch
        epoch_loss /= batch_count as f32;

        // Evaluate on validation set
        let val_loss = evaluate_loss(network, x_val, y_val, &loss_fn)?;

        if epoch % 5 == 0 || epoch == epochs - 1 {
            println!(
                "Epoch {}/{}: train_loss={:.6}, val_loss={:.6}",
                epoch + 1,
                epochs,
                epoch_loss,
                val_loss
            );
        }

        losses.push((epoch_loss, val_loss));
    }

    Ok(losses)
}

/// Evaluate loss on a dataset
fn evaluate_loss(
    network: &mut Vec<Box<dyn Layer<Array2<f32>>>>,
    x: &Array2<f32>,
    y: &Array2<f32>,
    loss_fn: &LossFunction,
) -> Result<f32> {
    // Forward pass
    let mut output = x.clone();
    for layer in network.iter_mut() {
        output = layer.forward(&output, false); // Evaluation mode
    }

    // Compute loss
    let loss = loss_fn.compute(&output, y);

    Ok(loss)
}

/// Evaluate accuracy on a dataset
fn evaluate(
    network: &mut Vec<Box<dyn Layer<Array2<f32>>>>,
    x: &Array2<f32>,
    y: &Array2<f32>,
) -> Result<f32> {
    // Forward pass
    let mut output = x.clone();
    for layer in network.iter_mut() {
        output = layer.forward(&output, false); // Evaluation mode
    }

    // Compute accuracy
    let mut correct = 0;
    let n_samples = x.shape()[0];

    for i in 0..n_samples {
        // Find predicted class (argmax)
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;

        for j in 0..output.shape()[1] {
            if output[[i, j]] > max_val {
                max_val = output[[i, j]];
                max_idx = j;
            }
        }

        // Find true class (argmax)
        let mut true_max_val = f32::NEG_INFINITY;
        let mut true_max_idx = 0;

        for j in 0..y.shape()[1] {
            if y[[i, j]] > true_max_val {
                true_max_val = y[[i, j]];
                true_max_idx = j;
            }
        }

        if max_idx == true_max_idx {
            correct += 1;
        }
    }

    Ok(correct as f32 / n_samples as f32)
}

/// Calculate mean across channels for a 4D array
fn mean_channels(x: &Array4<f32>) -> f32 {
    let batch_size = x.shape()[0];
    let channels = x.shape()[1];
    let height = x.shape()[2];
    let width = x.shape()[3];

    let mut sum = 0.0;
    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    sum += x[[b, c, h, w]];
                }
            }
        }
    }

    sum / (batch_size * channels * height * width) as f32
}

/// Calculate standard deviation across channels for a 4D array
fn std_channels(x: &Array4<f32>) -> f32 {
    let batch_size = x.shape()[0];
    let channels = x.shape()[1];
    let height = x.shape()[2];
    let width = x.shape()[3];

    let mean = mean_channels(x);

    let mut sum_squared_diff = 0.0;
    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let diff = x[[b, c, h, w]] - mean;
                    sum_squared_diff += diff * diff;
                }
            }
        }
    }

    let variance = sum_squared_diff / (batch_size * channels * height * width) as f32;
    variance.sqrt()
}

/// Calculate mean across batch dimension for a 2D array
fn mean_batch(x: &Array2<f32>) -> f32 {
    x.sum() / x.len() as f32
}

/// Calculate standard deviation across batch dimension for a 2D array
fn std_batch(x: &Array2<f32>) -> f32 {
    let mean = mean_batch(x);

    let mut sum_squared_diff = 0.0;
    for v in x.iter() {
        let diff = *v - mean;
        sum_squared_diff += diff * diff;
    }

    let variance = sum_squared_diff / x.len() as f32;
    variance.sqrt()
}

fn main() -> Result<()> {
    println!("Normalization Layers Example");
    println!("============================\n");

    // Example 1: Batch Normalization
    example_batchnorm()?;

    // Example 2: Layer Normalization
    example_layernorm()?;

    // Example 3: Group Normalization
    example_groupnorm()?;

    // Example 4: Compare normalization methods
    compare_normalization_methods()?;

    Ok(())
}
