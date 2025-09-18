//! Neural Network Components for Bifurcation Prediction
//!
//! This module contains the core neural network structures and activation functions
//! used in bifurcation prediction and classification.

use crate::analysis::types::*;
use crate::error::{IntegrateError, IntegrateResult};
use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;

/// Neural network for bifurcation classification and prediction
#[derive(Debug, Clone)]
pub struct BifurcationPredictionNetwork {
    /// Network architecture specification
    pub architecture: NetworkArchitecture,
    /// Trained model weights and biases
    pub model_parameters: ModelParameters,
    /// Training configuration
    pub training_config: super::training::TrainingConfiguration,
    /// Feature extraction settings
    pub feature_extraction: super::features::FeatureExtraction,
    /// Model performance metrics
    pub performance_metrics: super::uncertainty::PerformanceMetrics,
    /// Uncertainty quantification
    pub uncertainty_quantification: super::uncertainty::UncertaintyQuantification,
}

/// Neural network architecture configuration
#[derive(Debug, Clone)]
pub struct NetworkArchitecture {
    /// Input layer size (feature dimension)
    pub input_size: usize,
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Output layer size (number of bifurcation types)
    pub output_size: usize,
    /// Activation functions for each layer
    pub activation_functions: Vec<ActivationFunction>,
    /// Dropout rates for regularization
    pub dropoutrates: Vec<f64>,
    /// Batch normalization layers
    pub batch_normalization: Vec<bool>,
    /// Skip connections (ResNet-style)
    pub skip_connections: Vec<SkipConnection>,
}

/// Activation function types
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    /// Rectified Linear Unit
    ReLU,
    /// Leaky ReLU with negative slope
    LeakyReLU(f64),
    /// Hyperbolic tangent
    Tanh,
    /// Sigmoid function
    Sigmoid,
    /// Softmax (for output layer)
    Softmax,
    /// Swish activation (x * sigmoid(x))
    Swish,
    /// GELU (Gaussian Error Linear Unit)
    GELU,
    /// ELU (Exponential Linear Unit)
    ELU(f64),
}

/// Skip connection configuration
#[derive(Debug, Clone)]
pub struct SkipConnection {
    /// Source layer index
    pub from_layer: usize,
    /// Destination layer index
    pub to_layer: usize,
    /// Connection type
    pub connection_type: ConnectionType,
}

/// Types of skip connections
#[derive(Debug, Clone, Copy)]
pub enum ConnectionType {
    /// Direct addition (ResNet-style)
    Addition,
    /// Concatenation (DenseNet-style)
    Concatenation,
    /// Gated connection
    Gated,
}

/// Model parameters (weights and biases)
#[derive(Debug, Clone)]
pub struct ModelParameters {
    /// Weight matrices for each layer
    pub weights: Vec<Array2<f64>>,
    /// Bias vectors for each layer
    pub biases: Vec<Array1<f64>>,
    /// Batch normalization parameters
    pub batch_norm_params: Vec<BatchNormParams>,
    /// Dropout masks (if applicable)
    pub dropout_masks: Vec<Array1<bool>>,
}

/// Batch normalization parameters
#[derive(Debug, Clone)]
pub struct BatchNormParams {
    /// Scale parameters (gamma)
    pub scale: Array1<f64>,
    /// Shift parameters (beta)
    pub shift: Array1<f64>,
    /// Running mean (for inference)
    pub running_mean: Array1<f64>,
    /// Running variance (for inference)
    pub running_var: Array1<f64>,
}

/// Bifurcation prediction result
#[derive(Debug, Clone)]
pub struct BifurcationPrediction {
    /// Predicted bifurcation type
    pub bifurcation_type: BifurcationType,
    /// Predicted parameter value
    pub predicted_parameter: f64,
    /// Prediction confidence
    pub confidence: f64,
    /// Raw network output
    pub raw_output: Array1<f64>,
    /// Uncertainty estimate
    pub uncertainty_estimate: Option<UncertaintyEstimate>,
}

/// Uncertainty estimate for predictions
#[derive(Debug, Clone)]
pub struct UncertaintyEstimate {
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic_uncertainty: f64,
    /// Aleatoric uncertainty (data uncertainty)
    pub aleatoric_uncertainty: f64,
    /// Total uncertainty
    pub total_uncertainty: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
}

impl BifurcationPredictionNetwork {
    /// Create a new bifurcation prediction network
    pub fn new(input_size: usize, hidden_layers: Vec<usize>, output_size: usize) -> Self {
        let architecture = NetworkArchitecture {
            input_size,
            hidden_layers: hidden_layers.clone(),
            output_size,
            activation_functions: vec![ActivationFunction::ReLU; hidden_layers.len() + 1],
            dropoutrates: vec![0.0; hidden_layers.len() + 1],
            batch_normalization: vec![false; hidden_layers.len() + 1],
            skip_connections: Vec::new(),
        };

        let model_parameters = Self::initialize_parameters(&architecture);

        Self {
            architecture,
            model_parameters,
            training_config: super::training::TrainingConfiguration::default(),
            feature_extraction: super::features::FeatureExtraction::default(),
            performance_metrics: super::uncertainty::PerformanceMetrics::default(),
            uncertainty_quantification: super::uncertainty::UncertaintyQuantification::default(),
        }
    }

    /// Initialize network parameters
    fn initialize_parameters(arch: &NetworkArchitecture) -> ModelParameters {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        let mut prev_size = arch.input_size;
        for &layer_size in &arch.hidden_layers {
            weights.push(Array2::zeros((prev_size, layer_size)));
            biases.push(Array1::zeros(layer_size));
            prev_size = layer_size;
        }

        // Output layer
        weights.push(Array2::zeros((prev_size, arch.output_size)));
        biases.push(Array1::zeros(arch.output_size));

        ModelParameters {
            weights,
            biases,
            batch_norm_params: Vec::new(),
            dropout_masks: Vec::new(),
        }
    }

    /// Forward pass through the network
    pub fn forward(&self, input: &Array1<f64>) -> IntegrateResult<Array1<f64>> {
        let mut activation = input.clone();

        for (i, (weights, bias)) in self
            .model_parameters
            .weights
            .iter()
            .zip(&self.model_parameters.biases)
            .enumerate()
        {
            // Linear transformation
            activation = weights.t().dot(&activation) + bias;

            // Apply activation function
            activation = self.apply_activation_function(
                &activation,
                self.architecture.activation_functions[i],
            )?;

            // Apply dropout if training
            if self.architecture.dropoutrates[i] > 0.0 {
                activation = Self::apply_dropout(&activation, self.architecture.dropoutrates[i])?;
            }
        }

        Ok(activation)
    }

    /// Apply activation function
    fn apply_activation_function(
        &self,
        x: &Array1<f64>,
        func: ActivationFunction,
    ) -> IntegrateResult<Array1<f64>> {
        let result = match func {
            ActivationFunction::ReLU => x.mapv(|v| v.max(0.0)),
            ActivationFunction::LeakyReLU(alpha) => x.mapv(|v| if v > 0.0 { v } else { alpha * v }),
            ActivationFunction::Tanh => x.mapv(|v| v.tanh()),
            ActivationFunction::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            ActivationFunction::Softmax => {
                let exp_x = x.mapv(|v| v.exp());
                let sum = exp_x.sum();
                exp_x / sum
            }
            ActivationFunction::Swish => x.mapv(|v| v / (1.0 + (-v).exp())),
            ActivationFunction::GELU => x.mapv(|v| 0.5 * v * (1.0 + (v / (2.0_f64).sqrt()).tanh())),
            ActivationFunction::ELU(alpha) => {
                x.mapv(|v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) })
            }
        };

        Ok(result)
    }

    /// Apply dropout during training
    fn apply_dropout(x: &Array1<f64>, dropout_rate: f64) -> IntegrateResult<Array1<f64>> {
        if dropout_rate == 0.0 {
            return Ok(x.clone());
        }

        let mut rng = rand::rng();
        let mask: Array1<f64> = Array1::from_shape_fn(x.len(), |_| {
            if rng.random::<f64>() < dropout_rate {
                0.0
            } else {
                1.0 / (1.0 - dropout_rate)
            }
        });

        Ok(x * &mask)
    }

    /// Train the network on bifurcation data
    pub fn train(
        &mut self,
        training_data: &[(Array1<f64>, Array1<f64>)],
        validation_data: Option<&[(Array1<f64>, Array1<f64>)]>,
    ) -> IntegrateResult<()> {
        let mut training_metrics = Vec::new();
        let mut validation_metrics = Vec::new();

        for epoch in 0..self.training_config.epochs {
            let epoch_loss = self.train_epoch(training_data)?;

            let epoch_metric = super::uncertainty::EpochMetrics {
                epoch,
                loss: epoch_loss,
                accuracy: None,
                precision: None,
                recall: None,
                f1_score: None,
                learning_rate: self.get_current_learning_rate(epoch),
            };

            training_metrics.push(epoch_metric.clone());

            if let Some(val_data) = validation_data {
                let val_loss = self.evaluate(val_data)?;
                let val_metric = super::uncertainty::EpochMetrics {
                    epoch,
                    loss: val_loss,
                    accuracy: None,
                    precision: None,
                    recall: None,
                    f1_score: None,
                    learning_rate: epoch_metric.learning_rate,
                };
                validation_metrics.push(val_metric);
            }

            // Early stopping check
            if self.should_early_stop(&training_metrics, &validation_metrics) {
                break;
            }
        }

        self.performance_metrics.training_metrics = training_metrics;
        self.performance_metrics.validation_metrics = validation_metrics;

        Ok(())
    }

    /// Train for one epoch
    fn train_epoch(
        &mut self,
        training_data: &[(Array1<f64>, Array1<f64>)],
    ) -> IntegrateResult<f64> {
        let mut total_loss = 0.0;
        let batch_size = self.training_config.batch_size;

        for batch_start in (0..training_data.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(training_data.len());
            let batch = &training_data[batch_start..batch_end];

            let batch_loss = self.train_batch(batch)?;
            total_loss += batch_loss;
        }

        Ok(total_loss / (training_data.len() as f64 / batch_size as f64))
    }

    /// Train on a single batch
    fn train_batch(&mut self, batch: &[(Array1<f64>, Array1<f64>)]) -> IntegrateResult<f64> {
        let mut total_loss = 0.0;

        for (input, target) in batch {
            let prediction = self.forward(input)?;
            let loss = self.calculate_loss(&prediction, target)?;
            total_loss += loss;

            // Backpropagation would be implemented here
            self.backward(&prediction, target, input)?;
        }

        Ok(total_loss / batch.len() as f64)
    }

    /// Calculate loss
    fn calculate_loss(
        &self,
        prediction: &Array1<f64>,
        target: &Array1<f64>,
    ) -> IntegrateResult<f64> {
        match self.training_config.loss_function {
            super::training::LossFunction::MSE => {
                let diff = prediction - target;
                Ok(diff.dot(&diff) / prediction.len() as f64)
            }
            super::training::LossFunction::CrossEntropy => {
                let epsilon = 1e-15;
                let pred_clipped = prediction.mapv(|p| p.max(epsilon).min(1.0 - epsilon));
                let loss = -target
                    .iter()
                    .zip(pred_clipped.iter())
                    .map(|(&t, &p)| t * p.ln())
                    .sum::<f64>();
                Ok(loss)
            }
            super::training::LossFunction::FocalLoss(alpha, gamma) => {
                let epsilon = 1e-15;
                let pred_clipped = prediction.mapv(|p| p.max(epsilon).min(1.0 - epsilon));
                let loss = -alpha
                    * target
                        .iter()
                        .zip(pred_clipped.iter())
                        .map(|(&t, &p)| t * (1.0 - p).powf(gamma) * p.ln())
                        .sum::<f64>();
                Ok(loss)
            }
            super::training::LossFunction::HuberLoss(delta) => {
                let diff = prediction - target;
                let abs_diff = diff.mapv(|d| d.abs());
                let loss = abs_diff
                    .iter()
                    .map(|&d| {
                        if d <= delta {
                            0.5 * d * d
                        } else {
                            delta * d - 0.5 * delta * delta
                        }
                    })
                    .sum::<f64>();
                Ok(loss / prediction.len() as f64)
            }
            super::training::LossFunction::WeightedMSE => {
                // Placeholder implementation
                let diff = prediction - target;
                Ok(diff.dot(&diff) / prediction.len() as f64)
            }
        }
    }

    /// Backward pass (gradient computation)
    fn backward(
        &mut self,
        _prediction: &Array1<f64>,
        _target: &Array1<f64>,
        _input: &Array1<f64>,
    ) -> IntegrateResult<()> {
        // Placeholder for backpropagation implementation
        // In a real implementation, this would compute gradients and update weights
        Ok(())
    }

    /// Evaluate model performance
    pub fn evaluate(&self, test_data: &[(Array1<f64>, Array1<f64>)]) -> IntegrateResult<f64> {
        let mut total_loss = 0.0;

        for (input, target) in test_data {
            let prediction = self.forward(input)?;
            let loss = self.calculate_loss(&prediction, target)?;
            total_loss += loss;
        }

        Ok(total_loss / test_data.len() as f64)
    }

    /// Get current learning rate
    fn get_current_learning_rate(&self, epoch: usize) -> f64 {
        match &self.training_config.learning_rate {
            super::training::LearningRateSchedule::Constant(lr) => *lr,
            super::training::LearningRateSchedule::ExponentialDecay {
                initial_lr,
                decay_rate,
                decay_steps,
            } => initial_lr * decay_rate.powf(epoch as f64 / *decay_steps as f64),
            super::training::LearningRateSchedule::CosineAnnealing {
                initial_lr,
                min_lr,
                cycle_length,
            } => {
                let cycle_pos = (epoch % cycle_length) as f64 / *cycle_length as f64;
                min_lr
                    + (initial_lr - min_lr) * (1.0 + (cycle_pos * std::f64::consts::PI).cos()) / 2.0
            }
            super::training::LearningRateSchedule::StepDecay {
                initial_lr,
                drop_rate,
                epochs_drop,
            } => initial_lr * drop_rate.powf((epoch / epochs_drop) as f64),
            super::training::LearningRateSchedule::Adaptive { initial_lr, .. } => {
                // Placeholder for adaptive learning rate
                *initial_lr
            }
        }
    }

    /// Check if early stopping should be triggered
    fn should_early_stop(
        &self,
        _training_metrics: &[super::uncertainty::EpochMetrics],
        _validation_metrics: &[super::uncertainty::EpochMetrics],
    ) -> bool {
        if !self.training_config.early_stopping.enabled {
            return false;
        }

        // Placeholder for early stopping logic
        false
    }

    /// Predict bifurcation type and location
    pub fn predict_bifurcation(
        &self,
        features: &Array1<f64>,
    ) -> IntegrateResult<BifurcationPrediction> {
        let raw_output = self.forward(features)?;

        // Convert network output to bifurcation prediction
        let bifurcation_type = self.classify_bifurcation_type(&raw_output)?;
        let confidence = self.calculate_confidence(&raw_output)?;
        let predicted_parameter = raw_output[0]; // Assuming first output is parameter

        Ok(BifurcationPrediction {
            bifurcation_type,
            predicted_parameter,
            confidence,
            raw_output,
            uncertainty_estimate: None,
        })
    }

    /// Classify bifurcation type from network output
    fn classify_bifurcation_type(&self, output: &Array1<f64>) -> IntegrateResult<BifurcationType> {
        // Find the class with highest probability
        let max_idx = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        // Map index to bifurcation type
        let bifurcation_type = match max_idx {
            0 => BifurcationType::Fold,
            1 => BifurcationType::Transcritical,
            2 => BifurcationType::Pitchfork,
            3 => BifurcationType::Hopf,
            4 => BifurcationType::PeriodDoubling,
            5 => BifurcationType::Homoclinic,
            _ => BifurcationType::Unknown,
        };

        Ok(bifurcation_type)
    }

    /// Calculate prediction confidence
    fn calculate_confidence(&self, output: &Array1<f64>) -> IntegrateResult<f64> {
        // Use max probability as confidence
        let max_prob = output.iter().cloned().fold(0.0, f64::max);
        Ok(max_prob)
    }
}
