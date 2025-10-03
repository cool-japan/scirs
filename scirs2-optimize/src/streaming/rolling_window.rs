//! Rolling Window Optimization
//!
//! This module implements optimization algorithms that operate over sliding windows
//! of streaming data. These methods are useful for non-stationary optimization
//! problems where recent data should have more influence than older data.

use super::{
    utils, StreamingConfig, StreamingDataPoint, StreamingObjective, StreamingOptimizer,
    StreamingStats,
};
use crate::error::OptimizeError;
use scirs2_core::ndarray::{Array1, Array2}; // Unused import: ArrayView1
                                            // Unused import
                                            // use scirs2_core::error::CoreResult;
use std::collections::VecDeque;

type Result<T> = std::result::Result<T, OptimizeError>;

/// Rolling window optimizer that maintains a sliding window of recent data
#[derive(Debug, Clone)]
pub struct RollingWindowOptimizer<T: StreamingObjective> {
    /// Current parameter estimates
    parameters: Array1<f64>,
    /// Objective function
    objective: T,
    /// Configuration
    config: StreamingConfig,
    /// Statistics
    stats: StreamingStats,
    /// Sliding window of data points
    data_window: VecDeque<StreamingDataPoint>,
    /// Window-based optimizer (could be any StreamingOptimizer)
    window_optimizer: WindowOptimizerType,
    /// Whether to refit on every update or use incremental updates
    refit_every_update: bool,
    /// How often to refit (if not every update)
    refit_frequency: usize,
    /// Counter for refit frequency
    update_counter: usize,
}

/// Types of optimizers that can be used within the rolling window
#[derive(Debug, Clone)]
enum WindowOptimizerType {
    /// Gradient descent with accumulated gradients
    GradientDescent {
        gradient_accumulator: Array1<f64>,
        learning_rate: f64,
    },
    /// Least squares solution (for linear problems)
    LeastSquares {
        /// X^T X matrix
        xtx: Array2<f64>,
        /// X^T y vector  
        xty: Array1<f64>,
        /// Regularization parameter
        regularization: f64,
    },
    /// Weighted least squares with exponential decay
    WeightedLeastSquares {
        /// Weighted X^T X matrix
        weighted_xtx: Array2<f64>,
        /// Weighted X^T y vector
        weighted_xty: Array1<f64>,
        /// Regularization parameter
        regularization: f64,
        /// Decay factor for weights
        decay_factor: f64,
    },
}

impl<T: StreamingObjective> RollingWindowOptimizer<T> {
    /// Create a new rolling window optimizer
    pub fn new(
        initial_parameters: Array1<f64>,
        objective: T,
        config: StreamingConfig,
        window_optimizer_type: WindowOptimizerType,
        refit_every_update: bool,
    ) -> Self {
        let window_size = config.window_size;
        Self {
            parameters: initial_parameters,
            objective,
            config,
            stats: StreamingStats::default(),
            data_window: VecDeque::with_capacity(window_size),
            window_optimizer: window_optimizer_type,
            refit_every_update,
            refit_frequency: window_size / 4, // Default: refit every quarter window
            update_counter: 0,
        }
    }

    /// Add a data point to the window and remove old ones if necessary
    fn update_window(&mut self, datapoint: StreamingDataPoint) {
        if self.data_window.len() >= self.config.window_size {
            self.data_window.pop_front();
        }
        self.data_window.push_back(datapoint);
    }

    /// Optimize parameters based on current window contents
    fn optimize_window(&mut self) -> Result<()> {
        if self.data_window.is_empty() {
            return Ok(());
        }

        // Extract optimizer type and data temporarily to avoid borrowing conflicts
        let mut temp_optimizer = std::mem::replace(
            &mut self.window_optimizer,
            WindowOptimizerType::GradientDescent {
                gradient_accumulator: Array1::zeros(0),
                learning_rate: 0.01,
            },
        );

        let result = match &mut temp_optimizer {
            WindowOptimizerType::GradientDescent {
                gradient_accumulator,
                learning_rate,
            } => {
                let learning_rate = *learning_rate;
                self.optimize_gradient_descent(gradient_accumulator, learning_rate)
            }
            WindowOptimizerType::LeastSquares {
                xtx,
                xty,
                regularization,
            } => {
                let regularization = *regularization;
                self.optimize_least_squares(xtx, xty, regularization)
            }
            WindowOptimizerType::WeightedLeastSquares {
                weighted_xtx,
                weighted_xty,
                regularization,
                decay_factor,
            } => {
                let regularization = *regularization;
                let decay_factor = *decay_factor;
                self.optimize_weighted_least_squares(
                    weighted_xtx,
                    weighted_xty,
                    regularization,
                    decay_factor,
                )
            }
        };

        // Restore the optimizer
        self.window_optimizer = temp_optimizer;
        result
    }

    /// Gradient descent optimization over the window
    fn optimize_gradient_descent(
        &mut self,
        gradient_accumulator: &mut Array1<f64>,
        learning_rate: f64,
    ) -> Result<()> {
        gradient_accumulator.fill(0.0);
        let mut total_weight = 0.0;

        // Accumulate gradients from all points in window
        for (i, data_point) in self.data_window.iter().enumerate() {
            let gradient = self.objective.gradient(&self.parameters.view(), data_point);
            let weight = data_point.weight.unwrap_or(1.0);

            // Apply temporal weighting (more recent data gets higher weight)
            let temporal_weight = self
                .config
                .forgetting_factor
                .powi((self.data_window.len() - 1 - i) as i32);
            let effective_weight = weight * temporal_weight;

            *gradient_accumulator = &*gradient_accumulator + &(effective_weight * &gradient);
            total_weight += effective_weight;
        }

        if total_weight > 0.0 {
            *gradient_accumulator /= total_weight;

            // Apply gradient descent update
            self.parameters = &self.parameters - &(&*gradient_accumulator * learning_rate);
        }

        Ok(())
    }

    /// Least squares optimization over the window (for linear objectives)
    fn optimize_least_squares(
        &mut self,
        xtx: &mut Array2<f64>,
        xty: &mut Array1<f64>,
        regularization: f64,
    ) -> Result<()> {
        let n_features = self.parameters.len();
        xtx.fill(0.0);
        xty.fill(0.0);

        // Build normal equations from window data
        for data_point in &self.data_window {
            let x = &data_point.features;
            let y = data_point.target;
            let weight = data_point.weight.unwrap_or(1.0);

            // X^T X accumulation
            for i in 0..n_features {
                for j in 0..n_features {
                    xtx[[i, j]] += weight * x[i] * x[j];
                }
                // X^T y accumulation
                xty[i] += weight * x[i] * y;
            }
        }

        // Add regularization
        for i in 0..n_features {
            xtx[[i, i]] += regularization;
        }

        // Solve normal equations
        match scirs2_linalg::solve(&xtx.view(), &xty.view(), None) {
            Ok(solution) => {
                self.parameters = solution;
                Ok(())
            }
            Err(_) => {
                // Fall back to gradient descent if linear system fails
                let mut dummy_grad = Array1::zeros(n_features);
                self.optimize_gradient_descent(&mut dummy_grad, self.config.learning_rate)
            }
        }
    }

    /// Weighted least squares with exponential decay
    fn optimize_weighted_least_squares(
        &mut self,
        weighted_xtx: &mut Array2<f64>,
        weighted_xty: &mut Array1<f64>,
        regularization: f64,
        decay_factor: f64,
    ) -> Result<()> {
        let n_features = self.parameters.len();
        weighted_xtx.fill(0.0);
        weighted_xty.fill(0.0);

        // Build weighted normal equations
        for (i, data_point) in self.data_window.iter().enumerate() {
            let x = &data_point.features;
            let y = data_point.target;
            let base_weight = data_point.weight.unwrap_or(1.0);

            // Exponential decay weighting (more recent data gets higher weight)
            let age = self.data_window.len() - 1 - i;
            let temporal_weight = decay_factor.powi(age as i32);
            let total_weight = base_weight * temporal_weight;

            // Weighted X^T X accumulation
            for j in 0..n_features {
                for k in 0..n_features {
                    weighted_xtx[[j, k]] += total_weight * x[j] * x[k];
                }
                // Weighted X^T y accumulation
                weighted_xty[j] += total_weight * x[j] * y;
            }
        }

        // Add regularization
        for i in 0..n_features {
            weighted_xtx[[i, i]] += regularization;
        }

        // Solve weighted normal equations
        match scirs2_linalg::solve(&weighted_xtx.view(), &weighted_xty.view(), None) {
            Ok(solution) => {
                self.parameters = solution;
                Ok(())
            }
            Err(_) => {
                // Fall back to gradient descent
                let mut dummy_grad = Array1::zeros(n_features);
                self.optimize_gradient_descent(&mut dummy_grad, self.config.learning_rate)
            }
        }
    }

    /// Compute average loss over the current window
    fn compute_window_loss(&self) -> f64 {
        if self.data_window.is_empty() {
            return f64::INFINITY;
        }

        let mut total_loss = 0.0;
        let mut total_weight = 0.0;

        for data_point in &self.data_window {
            let loss = self.objective.evaluate(&self.parameters.view(), data_point);
            let weight = data_point.weight.unwrap_or(1.0);
            total_loss += weight * loss;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            total_loss / total_weight
        } else {
            f64::INFINITY
        }
    }

    /// Check convergence based on window statistics
    fn check_window_convergence(&self) -> bool {
        if self.data_window.len() < 2 {
            return false;
        }

        // Check if parameters are stable across recent window updates
        // This is a simplified convergence check - in practice, we'd track
        // parameter history across multiple window updates
        self.stats.average_loss.is_finite() && self.stats.average_loss < self.config.tolerance
    }
}

impl<T: StreamingObjective + Clone> StreamingOptimizer for RollingWindowOptimizer<T> {
    fn update(&mut self, datapoint: &StreamingDataPoint) -> Result<()> {
        let start_time = std::time::Instant::now();
        let old_parameters = self.parameters.clone();

        // Add data _point to window
        self.update_window(datapoint.clone());
        self.update_counter += 1;

        // Decide whether to reoptimize
        let should_reoptimize =
            self.refit_every_update || self.update_counter.is_multiple_of(self.refit_frequency);

        if should_reoptimize {
            // Reoptimize based on current window
            self.optimize_window()?;
            self.stats.updates_performed += 1;
        }

        // Update statistics
        self.stats.points_processed += 1;
        self.stats.current_loss = self.compute_window_loss();
        self.stats.average_loss = utils::ewma_update(
            self.stats.average_loss,
            self.stats.current_loss,
            0.1, // Use higher smoothing for window-based methods
        );

        // Check convergence
        self.stats.converged = utils::check_convergence(
            &old_parameters.view(),
            &self.parameters.view(),
            self.config.tolerance,
        ) || self.check_window_convergence();

        self.stats.processing_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(())
    }

    fn parameters(&self) -> &Array1<f64> {
        &self.parameters
    }

    fn stats(&self) -> &StreamingStats {
        &self.stats
    }

    fn reset(&mut self) {
        self.data_window.clear();
        self.update_counter = 0;
        self.stats = StreamingStats::default();

        // Reset window optimizer state
        match &mut self.window_optimizer {
            WindowOptimizerType::GradientDescent {
                gradient_accumulator,
                ..
            } => {
                gradient_accumulator.fill(0.0);
            }
            WindowOptimizerType::LeastSquares { xtx, xty, .. } => {
                xtx.fill(0.0);
                xty.fill(0.0);
            }
            WindowOptimizerType::WeightedLeastSquares {
                weighted_xtx,
                weighted_xty,
                ..
            } => {
                weighted_xtx.fill(0.0);
                weighted_xty.fill(0.0);
            }
        }
    }
}

/// Create a rolling window optimizer with gradient descent
#[allow(dead_code)]
pub fn rolling_window_gradient_descent<T: StreamingObjective>(
    initial_parameters: Array1<f64>,
    objective: T,
    config: StreamingConfig,
    learning_rate: Option<f64>,
) -> RollingWindowOptimizer<T> {
    let lr = learning_rate.unwrap_or(config.learning_rate);
    let n_params = initial_parameters.len();
    let optimizer_type = WindowOptimizerType::GradientDescent {
        gradient_accumulator: Array1::zeros(n_params),
        learning_rate: lr,
    };

    RollingWindowOptimizer::new(initial_parameters, objective, config, optimizer_type, false)
}

/// Create a rolling window optimizer with least squares (for linear problems)
#[allow(dead_code)]
pub fn rolling_window_least_squares<T: StreamingObjective>(
    initial_parameters: Array1<f64>,
    objective: T,
    config: StreamingConfig,
    regularization: Option<f64>,
) -> RollingWindowOptimizer<T> {
    let reg = regularization.unwrap_or(config.regularization);
    let n_params = initial_parameters.len();
    let optimizer_type = WindowOptimizerType::LeastSquares {
        xtx: Array2::zeros((n_params, n_params)),
        xty: Array1::zeros(n_params),
        regularization: reg,
    };

    RollingWindowOptimizer::new(initial_parameters, objective, config, optimizer_type, true)
}

/// Create a rolling window optimizer with weighted least squares
#[allow(dead_code)]
pub fn rolling_window_weighted_least_squares<T: StreamingObjective>(
    initial_parameters: Array1<f64>,
    objective: T,
    config: StreamingConfig,
    regularization: Option<f64>,
    decay_factor: Option<f64>,
) -> RollingWindowOptimizer<T> {
    let reg = regularization.unwrap_or(config.regularization);
    let decay = decay_factor.unwrap_or(config.forgetting_factor);
    let n_params = initial_parameters.len();
    let optimizer_type = WindowOptimizerType::WeightedLeastSquares {
        weighted_xtx: Array2::zeros((n_params, n_params)),
        weighted_xty: Array1::zeros(n_params),
        regularization: reg,
        decay_factor: decay,
    };

    RollingWindowOptimizer::new(initial_parameters, objective, config, optimizer_type, true)
}

/// Convenience function for rolling window linear regression
#[allow(dead_code)]
pub fn rolling_window_linear_regression(
    n_features: usize,
    window_size: usize,
    use_weighted: bool,
    config: Option<StreamingConfig>,
) -> RollingWindowOptimizer<super::LinearRegressionObjective> {
    let mut config = config.unwrap_or_default();
    config.window_size = window_size;

    let initial_params = Array1::zeros(n_features);
    let objective = super::LinearRegressionObjective;

    if use_weighted {
        rolling_window_weighted_least_squares(initial_params, objective, config, None, None)
    } else {
        rolling_window_least_squares(initial_params, objective, config, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::{LinearRegressionObjective, StreamingDataPoint};

    #[test]
    fn test_rolling_window_creation() {
        let optimizer = rolling_window_linear_regression(2, 10, false, None);
        assert_eq!(optimizer.data_window.capacity(), 10);
        assert_eq!(optimizer.parameters().len(), 2);
    }

    #[test]
    fn test_window_update() {
        let mut optimizer = rolling_window_linear_regression(2, 3, false, None);

        // Add data points to fill window
        for i in 0..5 {
            let features = Array1::from(vec![i as f64, (i + 1) as f64]);
            let target = (2 * i + 1) as f64;
            let point = StreamingDataPoint::new(features, target);

            optimizer.update(&point).unwrap();
        }

        // Window should be at capacity
        assert_eq!(optimizer.data_window.len(), 3);
        assert_eq!(optimizer.stats().points_processed, 5);
    }

    #[test]
    fn test_gradient_descent_window() {
        let config = StreamingConfig {
            window_size: 5,
            learning_rate: 0.1,
            ..Default::default()
        };

        let mut optimizer = rolling_window_gradient_descent(
            Array1::zeros(2),
            LinearRegressionObjective,
            config,
            None,
        );

        // Add some data points
        let data_points = vec![
            StreamingDataPoint::new(Array1::from(vec![1.0, 0.0]), 2.0),
            StreamingDataPoint::new(Array1::from(vec![0.0, 1.0]), 3.0),
            StreamingDataPoint::new(Array1::from(vec![1.0, 1.0]), 5.0),
        ];

        for point in &data_points {
            optimizer.update(point).unwrap();
        }

        assert_eq!(optimizer.stats().points_processed, 3);
        assert!(optimizer.stats().updates_performed > 0);
    }

    #[test]
    fn test_least_squares_window() {
        let mut optimizer = rolling_window_linear_regression(2, 10, false, None);

        // Generate data for y = 2*x1 + 3*x2
        let data_points = vec![
            StreamingDataPoint::new(Array1::from(vec![1.0, 0.0]), 2.0),
            StreamingDataPoint::new(Array1::from(vec![0.0, 1.0]), 3.0),
            StreamingDataPoint::new(Array1::from(vec![1.0, 1.0]), 5.0),
            StreamingDataPoint::new(Array1::from(vec![2.0, 1.0]), 7.0),
        ];

        for point in &data_points {
            optimizer.update(point).unwrap();
        }

        // Parameters should be close to [2, 3] for exact linear data
        let params = optimizer.parameters();
        assert!(
            (params[0] - 2.0).abs() < 0.1,
            "First parameter: {}",
            params[0]
        );
        assert!(
            (params[1] - 3.0).abs() < 0.1,
            "Second parameter: {}",
            params[1]
        );
    }

    #[test]
    fn test_weighted_least_squares_window() {
        let mut optimizer = rolling_window_linear_regression(2, 10, true, None);

        // Add data points with some having higher weights implicitly through recency
        let data_points = vec![
            StreamingDataPoint::new(Array1::from(vec![1.0, 0.0]), 2.0),
            StreamingDataPoint::new(Array1::from(vec![0.0, 1.0]), 3.0),
            StreamingDataPoint::new(Array1::from(vec![1.0, 1.0]), 5.0),
        ];

        for point in &data_points {
            optimizer.update(point).unwrap();
        }

        assert_eq!(optimizer.stats().points_processed, 3);
        assert!(optimizer.stats().current_loss.is_finite());
    }

    #[test]
    fn test_window_overflow() {
        let mut optimizer = rolling_window_linear_regression(2, 2, false, None);

        // Add more points than window size
        for i in 0..5 {
            let features = Array1::from(vec![i as f64, 1.0]);
            let target = i as f64;
            let point = StreamingDataPoint::new(features, target);

            optimizer.update(&point).unwrap();
        }

        // Window should be at capacity, not larger
        assert_eq!(optimizer.data_window.len(), 2);
        assert_eq!(optimizer.stats().points_processed, 5);
    }

    #[test]
    fn test_window_reset() {
        let mut optimizer = rolling_window_linear_regression(2, 5, false, None);

        // Add some data
        let point = StreamingDataPoint::new(Array1::from(vec![1.0, 2.0]), 3.0);
        optimizer.update(&point).unwrap();

        assert_eq!(optimizer.data_window.len(), 1);
        assert_eq!(optimizer.stats().points_processed, 1);

        // Reset should clear everything
        optimizer.reset();
        assert_eq!(optimizer.data_window.len(), 0);
        assert_eq!(optimizer.stats().points_processed, 0);
    }
}
