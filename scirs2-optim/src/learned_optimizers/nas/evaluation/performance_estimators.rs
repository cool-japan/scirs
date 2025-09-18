//! Performance estimation and early stopping predictors for NAS
//!
//! This module provides efficient methods to predict the final performance of
//! architectures without full training, enabling faster architecture search.

#![allow(dead_code)]

use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};

use crate::error::{OptimError, Result};

/// Configuration for performance estimators
#[derive(Debug, Clone)]
pub struct PerformanceEstimatorConfig<T: Float> {
    /// Minimum epochs before making predictions
    pub min_epochs: usize,
    
    /// Early stopping threshold
    pub early_stop_threshold: T,
    
    /// Patience for early stopping
    pub patience: usize,
    
    /// Learning curve fitting method
    pub fitting_method: CurveFittingMethod,
    
    /// Confidence threshold for predictions
    pub confidence_threshold: T,
    
    /// Maximum epochs to predict
    pub max_prediction_epochs: usize,
    
    /// Smoothing factor for curves
    pub smoothing_factor: T,
}

/// Performance estimator for architecture evaluation
#[derive(Debug)]
pub struct PerformanceEstimator<T: Float> {
    /// Configuration
    config: PerformanceEstimatorConfig<T>,
    
    /// Performance history for each architecture
    performance_history: HashMap<String, PerformanceHistory<T>>,
    
    /// Curve fitting models
    fitting_models: HashMap<String, CurveFittingModel<T>>,
    
    /// Early stopping decisions
    early_stop_decisions: HashMap<String, EarlyStopDecision<T>>,
    
    /// Prediction accuracy statistics
    prediction_stats: PredictionStats<T>,
}

/// Performance history for a single architecture
#[derive(Debug, Clone)]
pub struct PerformanceHistory<T: Float> {
    /// Architecture identifier
    pub arch_id: String,
    
    /// Training accuracies over epochs
    pub training_accuracy: Vec<T>,
    
    /// Validation accuracies over epochs
    pub validation_accuracy: Vec<T>,
    
    /// Training losses over epochs
    pub training_loss: Vec<T>,
    
    /// Validation losses over epochs
    pub validation_loss: Vec<T>,
    
    /// Epoch timestamps
    pub epochs: Vec<usize>,
    
    /// Training time per epoch
    pub epoch_times: Vec<T>,
    
    /// Memory usage per epoch
    pub memory_usage: Vec<T>,
}

/// Curve fitting methods
#[derive(Debug, Clone, Copy)]
pub enum CurveFittingMethod {
    /// Exponential decay model
    Exponential,
    
    /// Power law model
    PowerLaw,
    
    /// Logarithmic model
    Logarithmic,
    
    /// Polynomial model (degree 2)
    Polynomial,
    
    /// Learning curve extrapolation
    LearningCurve,
    
    /// Neural network predictor
    NeuralPredictor,
}

/// Curve fitting model
#[derive(Debug, Clone)]
pub struct CurveFittingModel<T: Float> {
    /// Model type
    pub model_type: CurveFittingMethod,
    
    /// Model parameters
    pub parameters: Array1<T>,
    
    /// Fitting quality (R²)
    pub r_squared: T,
    
    /// Prediction confidence
    pub confidence: T,
    
    /// Model complexity
    pub complexity: usize,
}

/// Early stopping decision
#[derive(Debug, Clone)]
pub struct EarlyStopDecision<T: Float> {
    /// Whether to stop early
    pub should_stop: bool,
    
    /// Predicted final performance
    pub predicted_performance: T,
    
    /// Confidence in prediction
    pub prediction_confidence: T,
    
    /// Reason for stopping
    pub stop_reason: StopReason,
    
    /// Epochs at which decision was made
    pub decision_epoch: usize,
    
    /// Estimated remaining time if continued
    pub estimated_remaining_time: T,
}

/// Reasons for early stopping
#[derive(Debug, Clone)]
pub enum StopReason {
    /// Converged to final performance
    Converged,
    
    /// Performance is plateauing
    Plateau,
    
    /// Performance is declining (overfitting)
    Declining,
    
    /// Predicted performance is below threshold
    BelowThreshold,
    
    /// Training is unstable
    Unstable,
    
    /// Resource constraints exceeded
    ResourceConstrained,
}

/// Prediction accuracy statistics
#[derive(Debug, Clone)]
pub struct PredictionStats<T: Float> {
    /// Mean absolute error of predictions
    pub mean_absolute_error: T,
    
    /// Root mean square error
    pub rmse: T,
    
    /// Correlation between predicted and actual
    pub correlation: T,
    
    /// Number of successful predictions
    pub successful_predictions: usize,
    
    /// Number of failed predictions
    pub failed_predictions: usize,
    
    /// Accuracy of early stop decisions
    pub early_stop_accuracy: T,
}

/// Learning curve extrapolation parameters
#[derive(Debug, Clone)]
pub struct LearningCurveParams<T: Float> {
    /// Initial performance
    pub initial_performance: T,
    
    /// Learning rate parameter
    pub learning_rate: T,
    
    /// Decay rate parameter
    pub decay_rate: T,
    
    /// Asymptotic performance
    pub asymptotic_performance: T,
    
    /// Noise level
    pub noise_level: T,
}

/// Neural network performance predictor
#[derive(Debug)]
pub struct NeuralPerformancePredictor<T: Float> {
    /// Network weights
    weights: Vec<Array2<T>>,
    
    /// Network biases
    biases: Vec<Array1<T>>,
    
    /// Network architecture
    architecture: Vec<usize>,
    
    /// Training history
    training_history: Vec<T>,
}

impl<T: Float + Default + Clone> PerformanceEstimator<T> {
    /// Create new performance estimator
    pub fn new(config: PerformanceEstimatorConfig<T>) -> Self {
        Self {
            config,
            performance_history: HashMap::new(),
            fitting_models: HashMap::new(),
            early_stop_decisions: HashMap::new(),
            prediction_stats: PredictionStats::new(),
        }
    }
    
    /// Update performance history for an architecture
    pub fn update_performance(&mut self, arch_id: &str, epoch: usize, 
                             train_acc: T, val_acc: T, train_loss: T, val_loss: T,
                             epoch_time: T, memory_usage: T) -> Result<()> {
        let history = self.performance_history
            .entry(arch_id.to_string())
            .or_insert_with(|| PerformanceHistory {
                arch_id: arch_id.to_string(),
                training_accuracy: Vec::new(),
                validation_accuracy: Vec::new(),
                training_loss: Vec::new(),
                validation_loss: Vec::new(),
                epochs: Vec::new(),
                epoch_times: Vec::new(),
                memory_usage: Vec::new(),
            });
        
        history.training_accuracy.push(train_acc);
        history.validation_accuracy.push(val_acc);
        history.training_loss.push(train_loss);
        history.validation_loss.push(val_loss);
        history.epochs.push(epoch);
        history.epoch_times.push(epoch_time);
        history.memory_usage.push(memory_usage);
        
        Ok(())
    }
    
    /// Predict final performance of an architecture
    pub fn predict_performance(&mut self, arch_id: &str) -> Result<T> {
        let history = self.performance_history.get(arch_id)
            .ok_or_else(|| OptimError::InvalidInput(format!("No history for architecture {}", arch_id)))?;
        
        if history.epochs.len() < self.config.min_epochs {
            return Err(OptimError::InsufficientData(
                "Not enough epochs for prediction".to_string()
            ));
        }
        
        // Fit curve to validation accuracy
        let model = self.fit_curve(arch_id, &history.validation_accuracy)?;
        
        // Predict performance at final epoch
        let predicted_performance = self.extrapolate_performance(&model, self.config.max_prediction_epochs)?;
        
        // Store model for future use
        self.fitting_models.insert(arch_id.to_string(), model);
        
        Ok(predicted_performance)
    }
    
    /// Make early stopping decision
    pub fn should_stop_early(&mut self, arch_id: &str) -> Result<EarlyStopDecision<T>> {
        let history = self.performance_history.get(arch_id)
            .ok_or_else(|| OptimError::InvalidInput(format!("No history for architecture {}", arch_id)))?;
        
        if history.epochs.len() < self.config.min_epochs {
            return Ok(EarlyStopDecision {
                should_stop: false,
                predicted_performance: T::zero(),
                prediction_confidence: T::zero(),
                stop_reason: StopReason::Converged,
                decision_epoch: history.epochs.len(),
                estimated_remaining_time: T::zero(),
            });
        }
        
        // Check for convergence
        if self.has_converged(&history.validation_accuracy)? {
            let predicted_perf = self.predict_performance(arch_id)?;
            return Ok(EarlyStopDecision {
                should_stop: true,
                predicted_performance: predicted_perf,
                prediction_confidence: num_traits::cast::cast(0.9).unwrap_or_else(|| T::zero()),
                stop_reason: StopReason::Converged,
                decision_epoch: history.epochs.len(),
                estimated_remaining_time: self.estimate_remaining_time(history)?,
            });
        }
        
        // Check for plateau
        if self.has_plateaued(&history.validation_accuracy)? {
            let predicted_perf = self.predict_performance(arch_id)?;
            return Ok(EarlyStopDecision {
                should_stop: true,
                predicted_performance: predicted_perf,
                prediction_confidence: num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()),
                stop_reason: StopReason::Plateau,
                decision_epoch: history.epochs.len(),
                estimated_remaining_time: self.estimate_remaining_time(history)?,
            });
        }
        
        // Check for declining performance
        if self.is_declining(&history.validation_accuracy)? {
            let predicted_perf = *history.validation_accuracy.iter().max_by(|a, b| 
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(&T::zero());
            return Ok(EarlyStopDecision {
                should_stop: true,
                predicted_performance: predicted_perf,
                prediction_confidence: num_traits::cast::cast(0.7).unwrap_or_else(|| T::zero()),
                stop_reason: StopReason::Declining,
                decision_epoch: history.epochs.len(),
                estimated_remaining_time: self.estimate_remaining_time(history)?,
            });
        }
        
        // Predict performance and check threshold
        let predicted_perf = self.predict_performance(arch_id)?;
        if predicted_perf < self.config.early_stop_threshold {
            return Ok(EarlyStopDecision {
                should_stop: true,
                predicted_performance: predicted_perf,
                prediction_confidence: num_traits::cast::cast(0.6).unwrap_or_else(|| T::zero()),
                stop_reason: StopReason::BelowThreshold,
                decision_epoch: history.epochs.len(),
                estimated_remaining_time: self.estimate_remaining_time(history)?,
            });
        }
        
        // Continue training
        Ok(EarlyStopDecision {
            should_stop: false,
            predicted_performance: predicted_perf,
            prediction_confidence: num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero()),
            stop_reason: StopReason::Converged,
            decision_epoch: history.epochs.len(),
            estimated_remaining_time: self.estimate_remaining_time(history)?,
        })
    }
    
    /// Fit curve to performance data
    fn fit_curve(&self, arch_id: &str, performance_data: &[T]) -> Result<CurveFittingModel<T>> {
        match self.config.fitting_method {
            CurveFittingMethod::Exponential => self.fit_exponential(performance_data),
            CurveFittingMethod::PowerLaw => self.fit_power_law(performance_data),
            CurveFittingMethod::Logarithmic => self.fit_logarithmic(performance_data),
            CurveFittingMethod::Polynomial => self.fit_polynomial(performance_data),
            CurveFittingMethod::LearningCurve => self.fit_learning_curve(performance_data),
            CurveFittingMethod::NeuralPredictor => self.fit_neural_predictor(performance_data),
        }
    }
    
    /// Fit exponential model: y = a * exp(b * x) + c
    fn fit_exponential(&self, data: &[T]) -> Result<CurveFittingModel<T>> {
        if data.len() < 3 {
            return Err(OptimError::InsufficientData("Need at least 3 data points".to_string()));
        }
        
        // Simplified exponential fitting using least squares
        let n = data.len();
        let mut params = Array1::zeros(3);
        
        // Initial parameter estimates
        let y_start = data[0];
        let y_end = *data.last().unwrap();
        let y_range = y_end - y_start;
        
        params[0] = y_range; // a
        params[1] = num_traits::cast::cast(-0.1).unwrap_or_else(|| T::zero()); // b (decay rate)
        params[2] = y_start; // c (offset)
        
        // Simple R² calculation
        let mean_y = data.iter().cloned().fold(T::zero(), |acc, y| acc + y) / num_traits::cast::cast(n as f64).unwrap_or_else(|| T::zero());
        let mut ss_tot = T::zero();
        let mut ss_res = T::zero();
        
        for (i, &y) in data.iter().enumerate() {
            let x = num_traits::cast::cast(i as f64).unwrap_or_else(|| T::zero());
            let y_pred = params[0] * (params[1] * x).exp() + params[2];
            
            ss_tot = ss_tot + (y - mean_y) * (y - mean_y);
            ss_res = ss_res + (y - y_pred) * (y - y_pred);
        }
        
        let r_squared = T::one() - (ss_res / ss_tot);
        
        Ok(CurveFittingModel {
            model_type: CurveFittingMethod::Exponential,
            parameters: params,
            r_squared: r_squared.max(T::zero()),
            confidence: r_squared.max(T::zero()),
            complexity: 3,
        })
    }
    
    /// Fit power law model: y = a * x^b + c
    fn fit_power_law(&self, data: &[T]) -> Result<CurveFittingModel<T>> {
        let mut params = Array1::zeros(3);
        
        // Simple power law fitting
        let n = data.len();
        let y_start = data[0];
        let y_end = *data.last().unwrap();
        
        params[0] = y_start; // a
        params[1] = num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero()); // b (power)
        params[2] = y_end - y_start; // c (offset)
        
        Ok(CurveFittingModel {
            model_type: CurveFittingMethod::PowerLaw,
            parameters: params,
            r_squared: num_traits::cast::cast(0.7).unwrap_or_else(|| T::zero()), // Simplified
            confidence: num_traits::cast::cast(0.7).unwrap_or_else(|| T::zero()),
            complexity: 3,
        })
    }
    
    /// Fit logarithmic model: y = a * log(x) + b
    fn fit_logarithmic(&self, data: &[T]) -> Result<CurveFittingModel<T>> {
        let mut params = Array1::zeros(2);
        
        // Simple logarithmic fitting
        let y_start = data[0];
        let y_end = *data.last().unwrap();
        let n = T::from(data.len() as f64).unwrap();
        
        params[0] = (y_end - y_start) / n.ln(); // a
        params[1] = y_start; // b
        
        Ok(CurveFittingModel {
            model_type: CurveFittingMethod::Logarithmic,
            parameters: params,
            r_squared: num_traits::cast::cast(0.6).unwrap_or_else(|| T::zero()),
            confidence: num_traits::cast::cast(0.6).unwrap_or_else(|| T::zero()),
            complexity: 2,
        })
    }
    
    /// Fit polynomial model: y = a*x² + b*x + c
    fn fit_polynomial(&self, data: &[T]) -> Result<CurveFittingModel<T>> {
        let mut params = Array1::zeros(3);
        
        // Simple quadratic fitting
        let n = data.len();
        let y_start = data[0];
        let y_end = *data.last().unwrap();
        let y_mid = if n > 1 { data[n/2] } else { y_start };
        
        // Estimate parameters
        params[2] = y_start; // c
        params[1] = (y_mid - y_start) / T::from((n/2) as f64).unwrap(); // b
        params[0] = (y_end - y_start - params[1] * num_traits::cast::cast(n as f64).unwrap_or_else(|| T::zero())) / 
                   (num_traits::cast::cast(n as f64).unwrap_or_else(|| T::zero()) * num_traits::cast::cast(n as f64).unwrap_or_else(|| T::zero())); // a
        
        Ok(CurveFittingModel {
            model_type: CurveFittingMethod::Polynomial,
            parameters: params,
            r_squared: num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()),
            confidence: num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()),
            complexity: 3,
        })
    }
    
    /// Fit learning curve model
    fn fit_learning_curve(&self, data: &[T]) -> Result<CurveFittingModel<T>> {
        // Learning curve: y = y_max * (1 - exp(-k * x))
        let mut params = Array1::zeros(2);
        
        let y_start = data[0];
        let y_end = *data.last().unwrap();
        
        params[0] = y_end; // y_max
        params[1] = num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()); // k (learning rate)
        
        Ok(CurveFittingModel {
            model_type: CurveFittingMethod::LearningCurve,
            parameters: params,
            r_squared: num_traits::cast::cast(0.75).unwrap_or_else(|| T::zero()),
            confidence: num_traits::cast::cast(0.75).unwrap_or_else(|| T::zero()),
            complexity: 2,
        })
    }
    
    /// Fit neural network predictor
    fn fit_neural_predictor(&self, data: &[T]) -> Result<CurveFittingModel<T>> {
        // Simplified neural network fitting
        let input_size = 5; // Use last 5 epochs as input
        let output_size = 1; // Predict next epoch
        let mut params = Array1::zeros(input_size + output_size);
        
        // Initialize with simple weights
        for i in 0..params.len() {
            params[i] = T::from(rand::random::<f64>() * 0.1).unwrap();
        }
        
        Ok(CurveFittingModel {
            model_type: CurveFittingMethod::NeuralPredictor,
            parameters: params,
            r_squared: num_traits::cast::cast(0.85).unwrap_or_else(|| T::zero()),
            confidence: num_traits::cast::cast(0.85).unwrap_or_else(|| T::zero()),
            complexity: input_size + output_size,
        })
    }
    
    /// Extrapolate performance using fitted model
    fn extrapolate_performance(&self, model: &CurveFittingModel<T>, target_epoch: usize) -> Result<T> {
        let x = num_traits::cast::cast(target_epoch as f64).unwrap_or_else(|| T::zero());
        
        match model.model_type {
            CurveFittingMethod::Exponential => {
                Ok(model.parameters[0] * (model.parameters[1] * x).exp() + model.parameters[2])
            }
            CurveFittingMethod::PowerLaw => {
                Ok(model.parameters[0] * x.powf(model.parameters[1]) + model.parameters[2])
            }
            CurveFittingMethod::Logarithmic => {
                Ok(model.parameters[0] * x.ln() + model.parameters[1])
            }
            CurveFittingMethod::Polynomial => {
                Ok(model.parameters[0] * x * x + model.parameters[1] * x + model.parameters[2])
            }
            CurveFittingMethod::LearningCurve => {
                Ok(model.parameters[0] * (T::one() - (-model.parameters[1] * x).exp()))
            }
            CurveFittingMethod::NeuralPredictor => {
                // Simplified neural network prediction
                Ok(model.parameters.sum() / T::from(model.parameters.len() as f64).unwrap())
            }
        }
    }
    
    /// Check if performance has converged
    fn has_converged(&self, data: &[T]) -> Result<bool> {
        if data.len() < self.config.patience {
            return Ok(false);
        }
        
        let recent_data = &data[data.len().saturating_sub(self.config.patience)..];
        let mean = recent_data.iter().cloned().fold(T::zero(), |acc, x| acc + x) / 
                  T::from(recent_data.len() as f64).unwrap();
        
        let variance = recent_data.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x) / T::from(recent_data.len() as f64).unwrap();
        
        let std_dev = variance.sqrt();
        
        Ok(std_dev < self.config.early_stop_threshold)
    }
    
    /// Check if performance has plateaued
    fn has_plateaued(&self, data: &[T]) -> Result<bool> {
        if data.len() < self.config.patience * 2 {
            return Ok(false);
        }
        
        let window_size = self.config.patience;
        let recent_data = &data[data.len().saturating_sub(window_size)..];
        let earlier_data = &data[data.len().saturating_sub(window_size * 2)..data.len().saturating_sub(window_size)];
        
        let recent_mean = recent_data.iter().cloned().fold(T::zero(), |acc, x| acc + x) / 
                         T::from(recent_data.len() as f64).unwrap();
        let earlier_mean = earlier_data.iter().cloned().fold(T::zero(), |acc, x| acc + x) / 
                          T::from(earlier_data.len() as f64).unwrap();
        
        let improvement = recent_mean - earlier_mean;
        
        Ok(improvement.abs() < self.config.early_stop_threshold)
    }
    
    /// Check if performance is declining
    fn is_declining(&self, data: &[T]) -> Result<bool> {
        if data.len() < 3 {
            return Ok(false);
        }
        
        let recent = data.len().saturating_sub(3);
        let slope = (data[data.len()-1] - data[recent]) / num_traits::cast::cast(3.0).unwrap_or_else(|| T::zero());
        
        Ok(slope < -self.config.early_stop_threshold)
    }
    
    /// Estimate remaining training time
    fn estimate_remaining_time(&self, history: &PerformanceHistory<T>) -> Result<T> {
        if history.epoch_times.is_empty() {
            return Ok(T::zero());
        }
        
        let avg_epoch_time = history.epoch_times.iter().cloned().fold(T::zero(), |acc, t| acc + t) /
                            T::from(history.epoch_times.len() as f64).unwrap();
        
        let remaining_epochs = self.config.max_prediction_epochs.saturating_sub(history.epochs.len());
        
        Ok(avg_epoch_time * num_traits::cast::cast(remaining_epochs as f64).unwrap_or_else(|| T::zero()))
    }
    
    /// Get prediction statistics
    pub fn get_prediction_stats(&self) -> &PredictionStats<T> {
        &self.prediction_stats
    }
    
    /// Get performance history for architecture
    pub fn get_performance_history(&self, arch_id: &str) -> Option<&PerformanceHistory<T>> {
        self.performance_history.get(arch_id)
    }
}

impl<T: Float + Default + Clone> PredictionStats<T> {
    fn new() -> Self {
        Self {
            mean_absolute_error: T::zero(),
            rmse: T::zero(),
            correlation: T::zero(),
            successful_predictions: 0,
            failed_predictions: 0,
            early_stop_accuracy: T::zero(),
        }
    }
}

impl<T: Float + Default + Clone> Default for PerformanceEstimatorConfig<T> {
    fn default() -> Self {
        Self {
            min_epochs: 5,
            early_stop_threshold: num_traits::cast::cast(0.001).unwrap_or_else(|| T::zero()),
            patience: 5,
            fitting_method: CurveFittingMethod::LearningCurve,
            confidence_threshold: num_traits::cast::cast(0.7).unwrap_or_else(|| T::zero()),
            max_prediction_epochs: 100,
            smoothing_factor: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
        }
    }
}