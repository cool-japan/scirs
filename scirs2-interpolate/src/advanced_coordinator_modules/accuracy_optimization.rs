//! Accuracy optimization engine for interpolation methods
//!
//! This module provides sophisticated accuracy optimization capabilities,
//! including error prediction, adaptive parameter tuning, and performance-accuracy
//! trade-off management.

use super::types::*;
use crate::error::InterpolateResult;
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::time::Instant;

/// Accuracy optimization engine
#[derive(Debug)]
pub struct AccuracyOptimizationEngine<F: Float + Debug> {
    /// Current optimization strategy
    strategy: AccuracyOptimizationStrategy,
    /// Accuracy targets
    targets: AccuracyTargets<F>,
    /// Error prediction model
    error_predictor: ErrorPredictionModel<F>,
    /// Optimization history
    optimization_history: VecDeque<AccuracyOptimizationResult>,
}

/// Accuracy optimization strategy
#[derive(Debug, Clone)]
pub enum AccuracyOptimizationStrategy {
    /// Maximize accuracy regardless of cost
    MaximizeAccuracy,
    /// Balance accuracy and performance
    BalancedAccuracy,
    /// Meet minimum accuracy with best performance
    MinimumAccuracy,
    /// Adaptive based on data characteristics
    Adaptive,
    /// Custom weighted strategy
    Custom {
        accuracy_weight: f64,
        performance_weight: f64,
    },
}

/// Accuracy targets for optimization
#[derive(Debug, Clone)]
pub struct AccuracyTargets<F: Float> {
    /// Target absolute error
    pub target_absolute_error: Option<F>,
    /// Target relative error
    pub target_relative_error: Option<F>,
    /// Maximum acceptable error
    pub max_acceptable_error: F,
    /// Confidence level for error bounds
    pub confidence_level: F,
}

/// Error prediction model
#[derive(Debug)]
pub struct ErrorPredictionModel<F: Float> {
    /// Prediction parameters
    prediction_params: HashMap<String, F>,
    /// Historical error data
    error_history: VecDeque<ErrorRecord<F>>,
    /// Model accuracy
    model_accuracy: F,
}

/// Error record for prediction training
#[derive(Debug, Clone)]
pub struct ErrorRecord<F: Float> {
    /// Predicted error
    pub predicted_error: F,
    /// Actual error
    pub actual_error: F,
    /// Data characteristics
    pub data_characteristics: String,
    /// Method used
    pub method: InterpolationMethodType,
    /// Timestamp
    pub timestamp: Instant,
}

/// Result of accuracy optimization
#[derive(Debug, Clone)]
pub struct AccuracyOptimizationResult {
    /// Method that was optimized
    pub method: InterpolationMethodType,
    /// Parameters that were adjusted
    pub adjusted_parameters: HashMap<String, f64>,
    /// Accuracy improvement achieved
    pub accuracy_improvement: f64,
    /// Performance impact
    pub performance_impact: f64,
    /// Success/failure status
    pub success: bool,
    /// Timestamp
    pub timestamp: Instant,
}

impl<F: Float + Debug + std::ops::AddAssign> AccuracyOptimizationEngine<F> {
    /// Create a new accuracy optimization engine
    pub fn new() -> InterpolateResult<Self> {
        Ok(Self {
            strategy: AccuracyOptimizationStrategy::BalancedAccuracy,
            targets: AccuracyTargets::default(),
            error_predictor: ErrorPredictionModel::new()?,
            optimization_history: VecDeque::new(),
        })
    }

    /// Set optimization strategy
    pub fn set_strategy(&mut self, strategy: AccuracyOptimizationStrategy) {
        self.strategy = strategy;
    }

    /// Set accuracy targets
    pub fn set_targets(&mut self, targets: AccuracyTargets<F>) {
        self.targets = targets;
    }

    /// Optimize interpolation accuracy for given method and data
    pub fn optimize_accuracy(
        &mut self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        current_parameters: &HashMap<String, f64>,
    ) -> InterpolateResult<AccuracyOptimizationResult> {
        let start_time = Instant::now();

        // Predict current accuracy
        let predicted_accuracy = self.predict_accuracy(method, data_profile, current_parameters)?;

        // Check if optimization is needed
        if self.meets_accuracy_targets(&predicted_accuracy)? {
            return Ok(AccuracyOptimizationResult {
                method,
                adjusted_parameters: current_parameters.clone(),
                accuracy_improvement: 0.0,
                performance_impact: 0.0,
                success: true,
                timestamp: start_time,
            });
        }

        // Apply optimization strategy
        let optimized_params = match &self.strategy {
            AccuracyOptimizationStrategy::MaximizeAccuracy => {
                self.maximize_accuracy_optimization(method, data_profile, current_parameters)?
            }
            AccuracyOptimizationStrategy::BalancedAccuracy => {
                self.balanced_optimization(method, data_profile, current_parameters)?
            }
            AccuracyOptimizationStrategy::MinimumAccuracy => {
                self.minimum_accuracy_optimization(method, data_profile, current_parameters)?
            }
            AccuracyOptimizationStrategy::Adaptive => {
                self.adaptive_optimization(method, data_profile, current_parameters)?
            }
            AccuracyOptimizationStrategy::Custom {
                accuracy_weight,
                performance_weight,
            } => self.custom_weighted_optimization(
                method,
                data_profile,
                current_parameters,
                *accuracy_weight,
                *performance_weight,
            )?,
        };

        // Calculate improvement and performance impact
        let optimized_accuracy = self.predict_accuracy(method, data_profile, &optimized_params)?;
        let accuracy_improvement = optimized_accuracy
            .predicted_accuracy
            .to_f64()
            .unwrap_or(0.0)
            - predicted_accuracy
                .predicted_accuracy
                .to_f64()
                .unwrap_or(0.0);

        let performance_impact =
            self.estimate_performance_impact(&optimized_params, current_parameters);

        let result = AccuracyOptimizationResult {
            method,
            adjusted_parameters: optimized_params,
            accuracy_improvement,
            performance_impact,
            success: accuracy_improvement > 0.0,
            timestamp: start_time,
        };

        // Store optimization result
        self.optimization_history.push_back(result.clone());
        if self.optimization_history.len() > 100 {
            self.optimization_history.pop_front();
        }

        Ok(result)
    }

    /// Predict accuracy for given method and parameters
    pub fn predict_accuracy(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        parameters: &HashMap<String, f64>,
    ) -> InterpolateResult<AccuracyPrediction<F>> {
        self.error_predictor
            .predict_accuracy(method, data_profile, parameters)
    }

    /// Update error prediction model with actual results
    pub fn update_error_model(
        &mut self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        predicted_error: F,
        actual_error: F,
    ) -> InterpolateResult<()> {
        let error_record = ErrorRecord {
            predicted_error,
            actual_error,
            data_characteristics: format!(
                "size:{},dim:{}",
                data_profile.size, data_profile.dimensionality
            ),
            method,
            timestamp: Instant::now(),
        };

        self.error_predictor.add_error_record(error_record)?;
        self.error_predictor.update_model()?;

        Ok(())
    }

    /// Get optimization history
    pub fn get_optimization_history(&self) -> &VecDeque<AccuracyOptimizationResult> {
        &self.optimization_history
    }

    /// Get current accuracy targets
    pub fn get_targets(&self) -> &AccuracyTargets<F> {
        &self.targets
    }

    /// Check if predicted accuracy meets targets
    fn meets_accuracy_targets(
        &self,
        prediction: &AccuracyPrediction<F>,
    ) -> InterpolateResult<bool> {
        let predicted_error = prediction.predicted_accuracy;

        // Check maximum acceptable error
        if predicted_error > self.targets.max_acceptable_error {
            return Ok(false);
        }

        // Check target absolute error if specified
        if let Some(target_abs) = self.targets.target_absolute_error {
            if predicted_error > target_abs {
                return Ok(false);
            }
        }

        // Check target relative error if specified
        if let Some(target_rel) = self.targets.target_relative_error {
            // This would require knowledge of the expected value range
            // Simplified check for now
            if predicted_error > target_rel {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Maximize accuracy optimization strategy
    fn maximize_accuracy_optimization(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        current_parameters: &HashMap<String, f64>,
    ) -> InterpolateResult<HashMap<String, f64>> {
        let mut optimized = current_parameters.clone();

        match method {
            InterpolationMethodType::CubicSpline => {
                // Reduce smoothing factor for higher accuracy
                if let Some(smoothing) = optimized.get_mut("smoothing") {
                    *smoothing *= 0.1;
                }
            }
            InterpolationMethodType::BSpline => {
                // Increase degree for higher accuracy (within limits)
                if let Some(degree) = optimized.get_mut("degree") {
                    *degree = (*degree + 1.0).min(5.0);
                }
            }
            InterpolationMethodType::RadialBasisFunction => {
                // Optimize shape parameter
                if let Some(shape) = optimized.get_mut("shape_parameter") {
                    *shape = self.optimize_rbf_shape_parameter(data_profile);
                }
            }
            _ => {
                // Generic optimization: reduce tolerance, increase precision
                optimized.insert("tolerance".to_string(), 1e-12);
                optimized.insert("max_iterations".to_string(), 1000.0);
            }
        }

        Ok(optimized)
    }

    /// Balanced accuracy optimization strategy
    fn balanced_optimization(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        current_parameters: &HashMap<String, f64>,
    ) -> InterpolateResult<HashMap<String, f64>> {
        let mut optimized = current_parameters.clone();

        // Apply moderate optimizations based on data characteristics
        let noise_level = data_profile.noise_level.to_f64().unwrap_or(0.1);
        let smoothness = data_profile.smoothness.to_f64().unwrap_or(0.5);

        match method {
            InterpolationMethodType::CubicSpline => {
                // Balance smoothing based on noise level
                let smoothing_factor = if noise_level > 0.1 {
                    noise_level * 0.5
                } else {
                    0.01
                };
                optimized.insert("smoothing".to_string(), smoothing_factor);
            }
            InterpolationMethodType::BSpline => {
                // Choose degree based on smoothness
                let degree = if smoothness > 0.8 { 3.0 } else { 2.0 };
                optimized.insert("degree".to_string(), degree);
            }
            _ => {
                // Moderate tolerance and iterations
                optimized.insert("tolerance".to_string(), 1e-8);
                optimized.insert("max_iterations".to_string(), 100.0);
            }
        }

        Ok(optimized)
    }

    /// Minimum accuracy optimization strategy
    fn minimum_accuracy_optimization(
        &self,
        _method: InterpolationMethodType,
        _data_profile: &DataProfile<F>,
        current_parameters: &HashMap<String, f64>,
    ) -> InterpolateResult<HashMap<String, f64>> {
        let mut optimized = current_parameters.clone();

        // Use relaxed tolerances for faster computation
        optimized.insert("tolerance".to_string(), 1e-4);
        optimized.insert("max_iterations".to_string(), 50.0);

        Ok(optimized)
    }

    /// Adaptive optimization based on data characteristics
    fn adaptive_optimization(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        current_parameters: &HashMap<String, f64>,
    ) -> InterpolateResult<HashMap<String, f64>> {
        let noise_level = data_profile.noise_level.to_f64().unwrap_or(0.1);
        let data_size = data_profile.size;

        // Choose strategy based on data characteristics
        if noise_level > 0.2 {
            // High noise: prioritize robustness
            self.balanced_optimization(method, data_profile, current_parameters)
        } else if data_size > 10000 {
            // Large data: prioritize performance
            self.minimum_accuracy_optimization(method, data_profile, current_parameters)
        } else {
            // Small, clean data: prioritize accuracy
            self.maximize_accuracy_optimization(method, data_profile, current_parameters)
        }
    }

    /// Custom weighted optimization strategy
    fn custom_weighted_optimization(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        current_parameters: &HashMap<String, f64>,
        accuracy_weight: f64,
        performance_weight: f64,
    ) -> InterpolateResult<HashMap<String, f64>> {
        // Blend between accuracy and performance optimization
        let accuracy_params =
            self.maximize_accuracy_optimization(method, data_profile, current_parameters)?;
        let performance_params =
            self.minimum_accuracy_optimization(method, data_profile, current_parameters)?;

        let mut optimized = HashMap::new();

        // Weighted average of parameters
        for (key, &acc_val) in &accuracy_params {
            let perf_val = performance_params.get(key).copied().unwrap_or(acc_val);
            let weighted_val = accuracy_weight * acc_val + performance_weight * perf_val;
            optimized.insert(key.clone(), weighted_val);
        }

        Ok(optimized)
    }

    /// Optimize RBF shape parameter based on data characteristics
    fn optimize_rbf_shape_parameter(&self, data_profile: &DataProfile<F>) -> f64 {
        let typical_distance = (data_profile.value_range.1 - data_profile.value_range.0)
            .to_f64()
            .unwrap_or(1.0)
            / (data_profile.size as f64).sqrt();

        // Shape parameter should be inversely related to typical distance
        1.0 / typical_distance
    }

    /// Estimate performance impact of parameter changes
    fn estimate_performance_impact(
        &self,
        optimized_params: &HashMap<String, f64>,
        current_params: &HashMap<String, f64>,
    ) -> f64 {
        let mut impact = 0.0;

        // Tolerance changes affect performance significantly
        if let (Some(&opt_tol), Some(&cur_tol)) = (
            optimized_params.get("tolerance"),
            current_params.get("tolerance"),
        ) {
            if opt_tol < cur_tol {
                impact += (cur_tol / opt_tol).log10() * 0.1; // Tighter tolerance = more computation
            }
        }

        // Iteration limit changes
        if let (Some(&opt_iter), Some(&cur_iter)) = (
            optimized_params.get("max_iterations"),
            current_params.get("max_iterations"),
        ) {
            impact += (opt_iter / cur_iter - 1.0) * 0.05; // More iterations = more time
        }

        // Degree changes for splines
        if let (Some(&opt_deg), Some(&cur_deg)) =
            (optimized_params.get("degree"), current_params.get("degree"))
        {
            impact += (opt_deg - cur_deg) * 0.15; // Higher degree = more computation
        }

        impact.max(-0.5).min(2.0) // Cap impact between -50% and +200%
    }
}

impl<F: Float> Default for AccuracyTargets<F> {
    fn default() -> Self {
        Self {
            target_absolute_error: None,
            target_relative_error: None,
            max_acceptable_error: F::from(1e-6).unwrap(),
            confidence_level: F::from(0.95).unwrap(),
        }
    }
}

impl<F: Float + std::ops::AddAssign> ErrorPredictionModel<F> {
    /// Create a new error prediction model
    pub fn new() -> InterpolateResult<Self> {
        Ok(Self {
            prediction_params: HashMap::new(),
            error_history: VecDeque::new(),
            model_accuracy: F::from(0.8).unwrap(),
        })
    }

    /// Predict accuracy for given method and parameters
    pub fn predict_accuracy(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        _parameters: &HashMap<String, f64>,
    ) -> InterpolateResult<AccuracyPrediction<F>> {
        // Simplified accuracy prediction based on method and data characteristics
        let base_accuracy = self.get_base_accuracy(method);
        let noise_penalty = data_profile.noise_level.to_f64().unwrap_or(0.1) * 0.5;
        let size_bonus = if data_profile.size > 1000 { 0.05 } else { 0.0 };

        let predicted_error = F::from(1.0 - base_accuracy + noise_penalty - size_bonus).unwrap();
        let confidence = self.model_accuracy;

        Ok(AccuracyPrediction {
            predicted_accuracy: predicted_error.max(F::from(1e-12).unwrap()),
            confidence_interval: (
                predicted_error * F::from(0.8).unwrap(),
                predicted_error * F::from(1.2).unwrap(),
            ),
            prediction_confidence: confidence,
            accuracy_factors: vec![
                AccuracyFactor {
                    name: "Method capability".to_string(),
                    impact: F::from(base_accuracy - 0.5).unwrap(),
                    confidence: F::from(0.9).unwrap(),
                    mitigations: vec!["Consider higher-order methods".to_string()],
                },
                AccuracyFactor {
                    name: "Data noise level".to_string(),
                    impact: F::from(-noise_penalty).unwrap(),
                    confidence: F::from(0.8).unwrap(),
                    mitigations: vec![
                        "Apply data smoothing".to_string(),
                        "Use robust methods".to_string(),
                    ],
                },
            ],
        })
    }

    /// Add error record for model training
    pub fn add_error_record(&mut self, record: ErrorRecord<F>) -> InterpolateResult<()> {
        self.error_history.push_back(record);

        // Limit history size
        if self.error_history.len() > 1000 {
            self.error_history.pop_front();
        }

        Ok(())
    }

    /// Update prediction model based on historical data
    pub fn update_model(&mut self) -> InterpolateResult<()> {
        if self.error_history.len() < 10 {
            return Ok(()); // Need sufficient data for training
        }

        // Calculate model accuracy based on recent predictions
        let recent_records: Vec<_> = self.error_history.iter().rev().take(50).collect();
        let mut total_error = F::zero();
        let mut count = 0;

        for record in recent_records {
            let relative_error =
                (record.predicted_error - record.actual_error).abs() / record.actual_error;
            total_error += relative_error;
            count += 1;
        }

        if count > 0 {
            let avg_relative_error = total_error / F::from(count).unwrap();
            self.model_accuracy = (F::one() - avg_relative_error).max(F::from(0.1).unwrap());
        }

        Ok(())
    }

    /// Get base accuracy for different methods
    fn get_base_accuracy(&self, method: InterpolationMethodType) -> f64 {
        match method {
            InterpolationMethodType::Linear => 0.7,
            InterpolationMethodType::CubicSpline => 0.9,
            InterpolationMethodType::BSpline => 0.92,
            InterpolationMethodType::RadialBasisFunction => 0.95,
            InterpolationMethodType::Kriging => 0.98,
            InterpolationMethodType::Polynomial => 0.85,
            InterpolationMethodType::PchipInterpolation => 0.88,
            InterpolationMethodType::AkimaSpline => 0.87,
            InterpolationMethodType::ThinPlateSpline => 0.93,
            InterpolationMethodType::NaturalNeighbor => 0.86,
            InterpolationMethodType::ShepardsMethod => 0.75,
            InterpolationMethodType::QuantumInspired => 0.99,
        }
    }

    /// Get model accuracy
    pub fn get_model_accuracy(&self) -> F {
        self.model_accuracy
    }

    /// Get prediction history statistics
    pub fn get_prediction_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        if !self.error_history.is_empty() {
            let mut total_abs_error = F::zero();
            let mut total_rel_error = F::zero();
            let count = self.error_history.len();

            for record in &self.error_history {
                let abs_error = (record.predicted_error - record.actual_error).abs();
                let rel_error = abs_error / record.actual_error;

                total_abs_error += abs_error;
                total_rel_error += rel_error;
            }

            stats.insert(
                "mean_absolute_error".to_string(),
                (total_abs_error / F::from(count).unwrap())
                    .to_f64()
                    .unwrap_or(0.0),
            );
            stats.insert(
                "mean_relative_error".to_string(),
                (total_rel_error / F::from(count).unwrap())
                    .to_f64()
                    .unwrap_or(0.0),
            );
            stats.insert(
                "model_accuracy".to_string(),
                self.model_accuracy.to_f64().unwrap_or(0.0),
            );
            stats.insert("sample_count".to_string(), count as f64);
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_targets_default() {
        let targets: AccuracyTargets<f64> = AccuracyTargets::default();
        assert_eq!(targets.max_acceptable_error, 1e-6);
        assert_eq!(targets.confidence_level, 0.95);
        assert!(targets.target_absolute_error.is_none());
    }

    #[test]
    fn test_error_prediction_model_creation() {
        let model: ErrorPredictionModel<f64> = ErrorPredictionModel::new().unwrap();
        assert_eq!(model.model_accuracy, 0.8);
        assert!(model.error_history.is_empty());
    }

    #[test]
    fn test_accuracy_optimization_engine_creation() {
        let engine: AccuracyOptimizationEngine<f64> = AccuracyOptimizationEngine::new().unwrap();
        assert!(matches!(
            engine.strategy,
            AccuracyOptimizationStrategy::BalancedAccuracy
        ));
        assert!(engine.optimization_history.is_empty());
    }
}
