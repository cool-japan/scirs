//! Streaming Optimization Module
//!
//! This module provides optimization algorithms specifically designed for streaming data
//! and real-time optimization scenarios. Unlike traditional batch optimization methods,
//! these algorithms can process data incrementally and adapt to changing conditions.
//!
//! # Key Features
//!
//! - **Online Gradient Descent**: Incremental parameter updates for streaming data
//! - **Streaming Trust Region**: Adaptive trust region methods for non-stationary problems
//! - **Incremental Newton Methods**: Memory-efficient quasi-Newton updates
//! - **Rolling Window Optimization**: Optimization over sliding windows of data
//! - **Real-time Parameter Estimation**: Continuous parameter adaptation
//!
//! # Applications
//!
//! - Real-time signal processing optimization
//! - Online system identification
//! - Adaptive control systems
//! - Streaming least squares problems
//! - Non-stationary optimization problems

use crate::error::OptimizeError;
// Unused import
// use crate::result::OptimizeResults;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
// Unused import: ArrayView2

type Result<T> = std::result::Result<T, OptimizeError>;
// Unused import
// use scirs2_core::error::CoreResult;

pub mod advanced_adaptive_streaming;
pub mod incremental_newton;
pub mod online_gradient_descent;
pub mod real_time_estimation;
pub mod rolling_window;
pub mod streaming_trust_region;

pub use advanced_adaptive_streaming::*;
pub use incremental_newton::*;
pub use online_gradient_descent::*;
pub use real_time_estimation::*;
pub use rolling_window::*;
pub use streaming_trust_region::*;

/// Configuration for streaming optimization algorithms
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum number of iterations per update
    pub max_nit: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Learning rate for online methods
    pub learning_rate: f64,
    /// Forgetting factor for exponential weighting
    pub forgetting_factor: f64,
    /// Window size for rolling methods
    pub window_size: usize,
    /// Whether to use adaptive learning rates
    pub adaptive_lr: bool,
    /// Regularization parameter
    pub regularization: f64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_nit: 100,
            tolerance: 1e-6,
            learning_rate: 0.01,
            forgetting_factor: 0.9,
            window_size: 100,
            adaptive_lr: true,
            regularization: 1e-8,
        }
    }
}

/// Statistics for streaming optimization
#[derive(Debug, Clone)]
pub struct StreamingStats {
    /// Total number of data points processed
    pub points_processed: usize,
    /// Number of parameter updates performed
    pub updates_performed: usize,
    /// Current loss value
    pub current_loss: f64,
    /// Average loss over recent window
    pub average_loss: f64,
    /// Convergence indicator
    pub converged: bool,
    /// Processing time statistics
    pub processing_time_ms: f64,
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self {
            points_processed: 0,
            updates_performed: 0,
            current_loss: f64::INFINITY,
            average_loss: f64::INFINITY,
            converged: false,
            processing_time_ms: 0.0,
        }
    }
}

/// A data point for streaming optimization
#[derive(Debug, Clone)]
pub struct StreamingDataPoint {
    /// Input features
    pub features: Array1<f64>,
    /// Target or measurement
    pub target: f64,
    /// Optional weight for this data point
    pub weight: Option<f64>,
    /// Timestamp for temporal ordering
    pub timestamp: Option<f64>,
}

impl StreamingDataPoint {
    /// Create a new streaming data point
    pub fn new(features: Array1<f64>, target: f64) -> Self {
        Self {
            features,
            target,
            weight: None,
            timestamp: None,
        }
    }

    /// Create a weighted streaming data point
    pub fn with_weight(features: Array1<f64>, target: f64, weight: f64) -> Self {
        Self {
            features,
            target,
            weight: Some(weight),
            timestamp: None,
        }
    }

    /// Create a timestamped streaming data point
    pub fn with_timestamp(features: Array1<f64>, target: f64, timestamp: f64) -> Self {
        Self {
            features,
            target,
            weight: None,
            timestamp: Some(timestamp),
        }
    }
}

/// Trait for streaming optimization algorithms
pub trait StreamingOptimizer {
    /// Process a single data point and update parameters
    fn update(&mut self, datapoint: &StreamingDataPoint) -> Result<()>;

    /// Process a batch of data points
    fn update_batch(&mut self, datapoints: &[StreamingDataPoint]) -> Result<()> {
        for _point in datapoints {
            self.update(_point)?;
        }
        Ok(())
    }

    /// Get current parameter estimates
    fn parameters(&self) -> &Array1<f64>;

    /// Get current optimization statistics
    fn stats(&self) -> &StreamingStats;

    /// Reset the optimizer state
    fn reset(&mut self);

    /// Check if the optimizer has converged
    fn converged(&self) -> bool {
        self.stats().converged
    }
}

/// Trait for objective functions in streaming optimization
pub trait StreamingObjective {
    /// Evaluate the objective function for a single data point
    fn evaluate(&self, parameters: &ArrayView1<f64>, datapoint: &StreamingDataPoint) -> f64;

    /// Compute the gradient for a single data point
    fn gradient(&self, parameters: &ArrayView1<f64>, datapoint: &StreamingDataPoint)
        -> Array1<f64>;

    /// Compute the Hessian for a single data point (optional)
    fn hessian(
        self_parameters: &ArrayView1<f64>,
        _datapoint: &StreamingDataPoint,
    ) -> Option<Array2<f64>> {
        None
    }
}

/// Linear regression objective for streaming optimization
#[derive(Debug, Clone)]
pub struct LinearRegressionObjective;

impl StreamingObjective for LinearRegressionObjective {
    fn evaluate(&self, parameters: &ArrayView1<f64>, datapoint: &StreamingDataPoint) -> f64 {
        let prediction = parameters.dot(&datapoint.features);
        let residual = prediction - datapoint.target;
        let weight = datapoint.weight.unwrap_or(1.0);
        0.5 * weight * residual * residual
    }

    fn gradient(
        &self,
        parameters: &ArrayView1<f64>,
        datapoint: &StreamingDataPoint,
    ) -> Array1<f64> {
        let prediction = parameters.dot(&datapoint.features);
        let residual = prediction - datapoint.target;
        let weight = datapoint.weight.unwrap_or(1.0);
        weight * residual * &datapoint.features
    }

    fn hessian(
        self_parameters: &ArrayView1<f64>,
        datapoint: &StreamingDataPoint,
    ) -> Option<Array2<f64>> {
        let weight = datapoint.weight.unwrap_or(1.0);
        let n = datapoint.features.len();
        let mut hessian = Array2::zeros((n, n));

        // H = weight * X^T * X for linear regression
        for i in 0..n {
            for j in 0..n {
                hessian[[i, j]] = weight * datapoint.features[i] * datapoint.features[j];
            }
        }

        Some(hessian)
    }
}

/// Logistic regression objective for streaming optimization
#[derive(Debug, Clone)]
pub struct LogisticRegressionObjective;

impl StreamingObjective for LogisticRegressionObjective {
    fn evaluate(&self, parameters: &ArrayView1<f64>, datapoint: &StreamingDataPoint) -> f64 {
        let z = parameters.dot(&datapoint.features);
        let weight = datapoint.weight.unwrap_or(1.0);

        // Numerical stability for sigmoid
        let loss = if z > 0.0 {
            z + (1.0 + (-z).exp()).ln() - datapoint.target * z
        } else {
            (1.0 + z.exp()).ln() - datapoint.target * z
        };

        weight * loss
    }

    fn gradient(
        &self,
        parameters: &ArrayView1<f64>,
        datapoint: &StreamingDataPoint,
    ) -> Array1<f64> {
        let z = parameters.dot(&datapoint.features);
        let sigmoid = 1.0 / (1.0 + (-z).exp());
        let weight = datapoint.weight.unwrap_or(1.0);

        weight * (sigmoid - datapoint.target) * &datapoint.features
    }

    fn hessian(
        parameters: &ArrayView1<f64>,
        datapoint: &StreamingDataPoint,
    ) -> Option<Array2<f64>> {
        let z = parameters.dot(&datapoint.features);
        let sigmoid = 1.0 / (1.0 + (-z).exp());
        let weight = datapoint.weight.unwrap_or(1.0);
        let scale = weight * sigmoid * (1.0 - sigmoid);

        let n = datapoint.features.len();
        let mut hessian = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                hessian[[i, j]] = scale * datapoint.features[i] * datapoint.features[j];
            }
        }

        Some(hessian)
    }
}

/// Utility functions for streaming optimization
pub mod utils {
    use super::*;

    /// Compute exponentially weighted moving average
    pub fn ewma_update(_current: f64, newvalue: f64, alpha: f64) -> f64 {
        alpha * newvalue + (1.0 - alpha) * _current
    }

    /// Adaptive learning rate based on gradient history
    pub fn adaptive_learning_rate(
        base_lr: f64,
        gradient_norm: f64,
        avg_gradient_norm: f64,
        min_lr: f64,
        max_lr: f64,
    ) -> f64 {
        if avg_gradient_norm > 0.0 {
            let scale = (avg_gradient_norm / gradient_norm).sqrt();
            (base_lr * scale).max(min_lr).min(max_lr)
        } else {
            base_lr
        }
    }

    /// Check convergence based on parameter change
    pub fn check_convergence(
        old_params: &ArrayView1<f64>,
        new_params: &ArrayView1<f64>,
        tolerance: f64,
    ) -> bool {
        let change = (new_params - old_params).mapv(|x| x.abs()).sum();
        let scale = new_params.mapv(|x| x.abs()).sum().max(1.0);
        change / scale < tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_datapoint_creation() {
        let features = Array1::from(vec![1.0, 2.0, 3.0]);
        let target = 5.0;

        let point = StreamingDataPoint::new(features.clone(), target);
        assert_eq!(point.features, features);
        assert_eq!(point.target, target);
        assert!(point.weight.is_none());
        assert!(point.timestamp.is_none());
    }

    #[test]
    fn test_linear_regression_objective() {
        let objective = LinearRegressionObjective;
        let params = Array1::from(vec![1.0, 2.0]);
        let features = Array1::from(vec![3.0, 4.0]);
        let target = 10.0;
        let point = StreamingDataPoint::new(features, target);

        let loss = objective.evaluate(&params.view(), &point);
        let gradient = objective.gradient(&params.view(), &point);

        // prediction = 1*3 + 2*4 = 11, residual = 11 - 10 = 1
        // loss = 0.5 * 1^2 = 0.5
        assert!((loss - 0.5).abs() < 1e-10);

        // gradient = residual * features = 1 * [3, 4] = [3, 4]
        assert!((gradient[0] - 3.0).abs() < 1e-10);
        assert!((gradient[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_utils_ewma() {
        let current = 10.0;
        let newvalue = 20.0;
        let alpha = 0.1;

        let result = utils::ewma_update(current, newvalue, alpha);
        let expected = 0.1 * 20.0 + 0.9 * 10.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_utils_convergence() {
        let old_params = Array1::from(vec![1.0, 2.0]);
        let new_params = Array1::from(vec![1.001, 2.001]);

        // Should converge with loose tolerance
        assert!(utils::check_convergence(
            &old_params.view(),
            &new_params.view(),
            1e-2
        ));

        // Should not converge with tight tolerance
        assert!(!utils::check_convergence(
            &old_params.view(),
            &new_params.view(),
            1e-6
        ));
    }
}
